"""动作校验 + 失败兜底。不含任何偏好逻辑。

护栏只做客观合法性检查：动作类型、参数格式/范围、take_order 必须取自候选(防幻觉)。
"""

from __future__ import annotations

from typing import Any

MAX_WAIT_MINUTES = 24 * 60


def parse_and_validate(obj: Any, candidate_ids: set[str]) -> dict[str, Any] | None:
    """把 LLM 返回的 dict 规范化为合法动作；非法返回 None（交由上层兜底）。"""
    if not isinstance(obj, dict):
        return None
    name = str(obj.get("action", "")).strip().lower()
    params = obj.get("params")
    if not isinstance(params, dict):
        return None

    if name == "take_order":
        cid = str(params.get("cargo_id", "")).strip()
        if not cid:
            return None
        # 防止模型编造不存在/已被预筛掉的货：候选为空时也必须拒绝（→ 兜底 wait）
        if cid not in candidate_ids:
            return None
        return {"action": "take_order", "params": {"cargo_id": cid}}

    if name == "reposition":
        try:
            lat = float(params["latitude"])
            lng = float(params["longitude"])
        except (KeyError, TypeError, ValueError):
            return None
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
            return None
        return {"action": "reposition", "params": {"latitude": lat, "longitude": lng}}

    if name == "wait":
        try:
            dur = int(params["duration_minutes"])
        except (KeyError, TypeError, ValueError):
            return None
        if dur <= 0:
            return None
        return {"action": "wait", "params": {"duration_minutes": min(dur, MAX_WAIT_MINUTES)}}

    return None


def fallback_action() -> dict[str, Any]:
    """模型彻底失败时的中性安全动作：短暂休息（不产生里程与成本）。"""
    return {"action": "wait", "params": {"duration_minutes": 60}}
