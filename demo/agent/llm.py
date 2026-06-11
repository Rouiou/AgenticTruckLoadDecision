"""稳健的模型调用：默认关闭 thinking（实测 0.5s/44tok vs 3.1s/371tok），
解析失败/网络错误时按 3-strike 逐步降级重试，彻底失败返回 None 交由上层兜底。
"""

from __future__ import annotations

import json
import logging
from typing import Any

_logger = logging.getLogger("agent.llm")


def chat_json(api: Any, messages: list[dict[str, Any]], *, max_tokens: int = 700) -> dict[str, Any] | None:
    """请求一个 JSON 对象动作，返回解析后的 dict；全部尝试失败返回 None。"""
    variants = [
        # 首选：temperature=0(同输入同输出,降方差) + 关 thinking + 限长 + 强制 JSON
        {"messages": messages, "response_format": {"type": "json_object"}, "enable_thinking": False, "max_tokens": max_tokens, "temperature": 0},
        # 去掉 temperature(以防评测网关拒收该参数)
        {"messages": messages, "response_format": {"type": "json_object"}, "enable_thinking": False, "max_tokens": max_tokens},
        # 去掉 max_tokens（以防限长截断 JSON）
        {"messages": messages, "response_format": {"type": "json_object"}, "enable_thinking": False},
        # 退路：去掉 enable_thinking（若评测模型不接受该参数则改走这里，避免彻底失败）
        {"messages": messages, "response_format": {"type": "json_object"}, "max_tokens": max_tokens},
        {"messages": messages, "response_format": {"type": "json_object"}},
        # 最后退路：连 response_format 也去掉（宁可慢/松也别返回 None）
        {"messages": messages, "enable_thinking": False},
        {"messages": messages},
    ]
    last_err: str | None = None
    for i, payload in enumerate(variants, 1):
        try:
            resp = api.model_chat_completion(dict(payload))
            content = _extract_content(resp)
            if not content:
                last_err = "empty content"
                continue
            obj = json.loads(content)
            if isinstance(obj, dict):
                return obj
            last_err = "not a json object"
        except json.JSONDecodeError as e:
            last_err = f"json: {e}"
            _logger.warning("chat_json 第%d次解析失败: %s", i, last_err)
        except Exception as e:  # 网络/HTTP/超时等
            last_err = f"{type(e).__name__}: {e}"
            _logger.warning("chat_json 第%d次调用失败: %s", i, last_err)
    _logger.error("chat_json 全部尝试失败: %s", last_err)
    return None


def _extract_content(resp: Any) -> str | None:
    if not isinstance(resp, dict):
        return None
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = msg.get("content")
    return content.strip() if isinstance(content, str) and content.strip() else None
