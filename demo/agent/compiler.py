"""偏好编译器：把 get_driver_status 的 preferences 自然语言原文，用 LLM 一次性编译成
【通用结构化约束】（开放谓词），供 checker 做确定性的"标注 + 否决"。

合规要点（关键）：
- 这是唯一"理解偏好"的地方，完全由 LLM 完成；
- 代码只实现"按字段取值 + 按算子比较 + 时段静止判定"这类【与具体规则无关】的通用求值，
  约束里的一切具体值（地名/品类/时段/阈值）都来自 LLM 运行时输出，**代码内零偏好常量**；
- 按 preferences 原文哈希缓存：原文不变就不再调用 LLM（省 token、消除随机抖动），
  突发偏好出现（哈希变了）立即重编译。
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from . import llm

_logger = logging.getLogger("agent.compiler")

COMPILER_SYSTEM = (
    "你是偏好编译器。读取一位卡车司机的【偏好原文】，把其中可以【客观逐条核对】的硬性约束，"
    "翻译成下面这种通用 JSON（只输出 JSON、无解释）。你负责理解偏好语义，之后由通用代码按你给的"
    "字段/算子/值机械执行。\n"
    "输出格式：\n"
    "{\n"
    '  "order_filters": [ {"field":"<候选字段>","op":"<算子>","value":<值>,"raw_text":"<对应偏好原文>"} ],\n'
    '  "rest_windows":  [ {"kind":"scheduled 或 continuous","window":"HH:MM-HH:MM" 或 null,"min_continuous_minutes":整数 或 null,"raw_text":"<原文>"} ],\n'
    '  "off_days_required": 整数,\n'
    '  "category_quotas": [ {"category":"<品类关键词>","min_orders":整数,"month":整数1-12,"active_from":"M-D","active_to":"M-D","penalty_per_unit":整数,"unrecoverable":true 或 false,"raw_text":"<原文>"} ],\n'
    '  "monthly_count_caps": [ {"field":"cost_time_minutes","op":"gt","threshold":整数分钟,"max_per_month":整数,"penalty_per_unit":整数,"raw_text":"<原文>"} ]\n'
    "}\n"
    "field 仅可用：cargo_name(品类) / start_city(起点城市) / end_city(终点城市) / deadhead_km(赴装货空驶km) / truck_length(需求车长)。\n"
    "op 仅可用：contains(字段文本包含 value 任一关键词即命中) / equals(相等) / gt(字段数值 > value)。\n"
    "order_filters 表示【接这单就违反偏好】的情形（如禁某品类、禁起点或终点在某地、赴装货空驶超过某 km）；"
    "value 可为字符串、字符串数组或数字。"
    "若该规避【只在某些具体日期生效】(偏好写明仅某月某日)，给该条加 \"active_from\":\"M-D\",\"active_to\":\"M-D\"(含两端)；常驻生效则不加这两字段。\n"
    "rest_windows 表示【作息类】，每条必须标 kind，分两类：\n"
    "- kind=\"scheduled\"：偏好指定了【具体钟点区间】(如'晚上11点到早上6点''零点到六点'必须休息)。window 填该钟点区间(可跨午夜，如 23:00-06:00)；min_continuous_minutes 填 null。\n"
    "- kind=\"continuous\"：偏好只说【每天连续休息≥N 小时】、没指定具体几点(如'每天至少连续休息7个小时')。min_continuous_minutes 填 N×60；window 填 null(由代码自行在单日内安排连续块)。\n"
    "判断关键：偏好出现了具体钟点数字 → scheduled；只说时长不说钟点 → continuous。没有作息类偏好则 rest_windows 为空数组。\n"
    "off_days_required：把'每月至少留 N 个完全不出车的整天''留 N 个整天进厂检修'这类，读成整数 N(没有这类偏好则 0)。\n"
    "【重要·逐条别漏】category_quotas 表示【某自然月必须【主动接满】N 单某品类货】的硬性配额(与 order_filters 的'禁接'相反——这是'凑够')。"
    "触发词：偏好出现'考核指标 / 必须接满 / 接够 / 把欠的单补上 / 某类货这个月接 N 单'等，一律编进 category_quotas，每条配额单独一项、别遗漏。"
    "字段：category=该品类关键词(对应候选 cargo_name，取'货源类型是XX的货'里的 XX)；min_orders=N(中文数字也要转，如'十二'=12)；month=所属自然月1-12('四月'→4、'五月'→5)；active_from/active_to=该月起止(如 4-1 / 4-30)；penalty_per_unit=少一单罚款额；"
    "unrecoverable：偏好写明【本月欠额结转下月继续扣 / 过期不可补】填 true，否则 false。\n"
    "区分：'不接/禁接某品类'→order_filters；'必须接满 N 单某品类'→category_quotas。\n"
    "monthly_count_caps 表示【每月对某类单的【数量上限】】：如'每月超过 N 小时的长途/远活最多接 M 单，多一单扣一次'。"
    "field 填 cost_time_minutes(订单干线时长，分钟)；op 填 gt；threshold 填该时长阈值的【分钟数】(如'八小时/8小时'→480、'十小时'→600)；max_per_month 填每月最多单数 M；penalty_per_unit 填超一单罚款额。没有这类上限则空数组。\n"
    "示例：偏好'五月玩具类的货必须接满八单，少一单扣300' → category_quotas 含 {\"category\":\"玩具\",\"min_orders\":8,\"month\":5,\"active_from\":\"5-1\",\"active_to\":\"5-31\",\"penalty_per_unit\":300,\"unrecoverable\":false}。\n"
    "其余需跨天规划的(某地累计去够 N 个不同日、某具体日期到某地办事)仍【不要】放进来，由别的模块处理。"
)


def _digest(prefs: Any) -> str:
    return hashlib.sha256(
        json.dumps(prefs, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


class PreferenceCompiler:
    """实例随决策服务全程存活；按偏好原文哈希缓存编译结果。"""

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, Any]] = {}
        self._one_cache: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {"order_filters": [], "rest_windows": [], "off_days_required": 0, "category_quotas": [], "monthly_count_caps": []}

    @staticmethod
    def _nonempty(ir: dict[str, Any]) -> bool:
        return bool(
            ir.get("order_filters")
            or ir.get("rest_windows")
            or ir.get("category_quotas")
            or ir.get("monthly_count_caps")
            or int(ir.get("off_days_required") or 0) > 0
        )

    @staticmethod
    def _merge(dst: dict[str, Any], src: dict[str, Any]) -> None:
        for k in ("order_filters", "rest_windows", "category_quotas", "monthly_count_caps"):
            dst[k].extend(src.get(k) or [])
        dst["off_days_required"] = max(int(dst.get("off_days_required") or 0), int(src.get("off_days_required") or 0))

    def compile(self, api: Any, preferences: list[Any]) -> dict[str, Any]:
        """【逐条编译再合并】：单条偏好的抽取任务更简单，弱模型(flash)更稳、更不漏。
        双层哈希缓存(整组 + 单条)；单条偏好跨月复用(如作息条在三个月都一样)。"""
        try:
            if not preferences:
                return self._empty()
            key = _digest(preferences)
            if key in self._cache:
                return self._cache[key]
            merged = self._empty()
            for pref in preferences:
                self._merge(merged, self._compile_one(api, pref))
            self._cache[key] = merged
            _logger.info("compiled IR: %s", json.dumps(merged, ensure_ascii=False))
            return merged
        except Exception as e:  # 编译失败不能拖垮决策
            _logger.warning("compile 失败，返回空约束: %s", e)
            return self._empty()

    def _compile_one(self, api: Any, pref: Any) -> dict[str, Any]:
        """编译单条偏好(按单条哈希缓存)。弱模型偶发漏抽 → 结果全空时重试至多 3 次。"""
        pkey = _digest(pref)
        if pkey in self._one_cache:
            return self._one_cache[pkey]
        content = pref.get("content") if isinstance(pref, dict) else str(pref)
        _logger.info("compile 单偏好: %s", str(content)[:70])
        messages = [
            {"role": "system", "content": COMPILER_SYSTEM},
            {"role": "user", "content": json.dumps({"偏好原文": [pref]}, ensure_ascii=False)},
        ]
        ir = self._empty()
        for _ in range(3):
            ir = self._normalize(llm.chat_json(api, messages, max_tokens=700))
            if self._nonempty(ir):
                break
        self._one_cache[pkey] = ir
        return ir

    @staticmethod
    def _normalize(obj: Any) -> dict[str, Any]:
        out: dict[str, Any] = {"order_filters": [], "rest_windows": [], "off_days_required": 0, "category_quotas": [], "monthly_count_caps": []}
        if not isinstance(obj, dict):
            return out
        for f in obj.get("order_filters") or []:
            if not isinstance(f, dict):
                continue
            field = str(f.get("field", "")).strip()
            op = str(f.get("op", "")).strip()
            if field and op:
                out["order_filters"].append(
                    {
                        "field": field,
                        "op": op,
                        "value": f.get("value"),
                        "raw_text": str(f.get("raw_text", "") or ""),
                        "active_from": (str(f["active_from"]).strip() if f.get("active_from") else None),
                        "active_to": (str(f["active_to"]).strip() if f.get("active_to") else None),
                    }
                )
        for w in obj.get("rest_windows") or []:
            if not isinstance(w, dict):
                continue
            mcm = w.get("min_continuous_minutes")
            try:
                mcm = int(mcm) if mcm is not None else None
            except (TypeError, ValueError):
                mcm = None
            out["rest_windows"].append(
                {
                    "kind": str(w.get("kind", "") or "").strip().lower(),
                    "window": (str(w["window"]).strip() if w.get("window") else None),
                    "min_continuous_minutes": mcm,
                    "raw_text": str(w.get("raw_text", "") or ""),
                }
            )
        try:
            out["off_days_required"] = max(0, int(obj.get("off_days_required") or 0))
        except (TypeError, ValueError):
            out["off_days_required"] = 0
        for q in (obj.get("category_quotas") or obj.get("category_quota") or []):
            if not isinstance(q, dict):
                continue
            cat = str(q.get("category", "") or "").strip()
            try:
                mo = int(q.get("min_orders") or 0)
            except (TypeError, ValueError):
                mo = 0
            try:
                mon = int(q.get("month") or 0)
            except (TypeError, ValueError):
                mon = 0
            if not cat or mo <= 0 or not (1 <= mon <= 12):
                continue
            try:
                ppu = float(q.get("penalty_per_unit") or q.get("penalty_per_miss") or q.get("penalty") or 0)
            except (TypeError, ValueError):
                ppu = 0.0
            out["category_quotas"].append(
                {
                    "category": cat,
                    "min_orders": mo,
                    "month": mon,
                    "active_from": (str(q["active_from"]).strip() if q.get("active_from") else None),
                    "active_to": (str(q["active_to"]).strip() if q.get("active_to") else None),
                    "penalty_per_unit": ppu,
                    "unrecoverable": bool(q.get("unrecoverable", False)),
                    "raw_text": str(q.get("raw_text", "") or ""),
                }
            )
        for cap in (obj.get("monthly_count_caps") or obj.get("monthly_count_cap") or []):
            if not isinstance(cap, dict):
                continue
            field = str(cap.get("field", "") or "").strip()
            try:
                thr = float(cap.get("threshold"))
                maxm = int(cap.get("max_per_month"))
            except (TypeError, ValueError):
                continue
            if not field or maxm < 0:
                continue
            try:
                ppu = float(cap.get("penalty_per_unit") or cap.get("penalty_per_miss") or cap.get("penalty") or 0)
            except (TypeError, ValueError):
                ppu = 0.0
            out["monthly_count_caps"].append(
                {
                    "field": field,
                    "op": str(cap.get("op", "gt") or "gt").strip(),
                    "threshold": thr,
                    "max_per_month": maxm,
                    "penalty_per_unit": ppu,
                    "raw_text": str(cap.get("raw_text", "") or ""),
                }
            )
        return out
