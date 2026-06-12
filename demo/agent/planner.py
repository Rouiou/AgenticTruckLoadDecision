"""日度规划层（planner）：低频 LLM 调用，把整月义务 + 每日作息拆成「今日计划」，
注入执行器每步决策，弥补逐步贪心缺乏的前瞻性。

合规：偏好的理解与取舍【完全由 LLM 完成】（planner 也是一次 LLM 调用）；
代码只负责"每天调一次"(客观的天数判断)与把计划透传给执行器，不写死任何偏好规则。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from . import llm

_logger = logging.getLogger("agent.planner")


def _norm_rest_block(rb: Any) -> dict[str, str] | None:
    if isinstance(rb, dict) and rb.get("start") and rb.get("end"):
        return {"start": str(rb["start"]).strip(), "end": str(rb["end"]).strip()}
    return None


def _num(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _norm_int(x: Any) -> int:
    try:
        return max(0, int(x or 0))
    except (TypeError, ValueError):
        return 0


def _norm_dated_tasks(v: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(v, list):
        for t in v:
            if not isinstance(t, dict):
                continue
            lat, lng = _num(t.get("lat")), _num(t.get("lng"))
            date = str(t.get("date", "")).strip()
            if lat is None or lng is None or not date:
                continue
            try:
                wm = int(t.get("wait_minutes") or 0)
            except (TypeError, ValueError):
                wm = 0
            out.append({"date": date, "lat": lat, "lng": lng, "wait_minutes": wm})
    return out


def _norm_avoid_filters(v: Any) -> list[dict[str, Any]]:
    """planner 的当日规避过滤(与编译器 order_filters 同构)：只接受白名单字段与算子。"""
    out: list[dict[str, Any]] = []
    if isinstance(v, list):
        for f in v:
            if not isinstance(f, dict):
                continue
            field = str(f.get("field", "")).strip()
            op = str(f.get("op", "")).strip()
            if field in ("cargo_name", "start_city", "end_city", "deadhead_km") and op in ("contains", "gt"):
                out.append(
                    {
                        "field": field,
                        "op": op,
                        "value": f.get("value"),
                        "raw_text": str(f.get("raw_text", "") or ""),
                    }
                )
    return out[:6]


def _norm_region_target(v: Any) -> dict[str, Any] | None:
    if not isinstance(v, dict):
        return None
    kw = str(v.get("keyword", "")).strip()
    lat, lng = _num(v.get("lat")), _num(v.get("lng"))
    if not kw or lat is None or lng is None:
        return None
    try:
        nd = int(v.get("need_days") or 0)
    except (TypeError, ValueError):
        nd = 0
    return {"keyword": kw, "lat": lat, "lng": lng, "need_days": nd}

PLANNER_SYSTEM = (
    "你是卡车司机的【日度规划助手】。依据【司机偏好原文】+【历史活动台账】+【今天日期/本月剩余天数】，"
    "为今天制定一份简短计划，帮助执行器在守住所有偏好（尤其每日作息类、整月义务类）的同时尽量赚钱。"
    "偏好的理解与取舍完全由你来做，执行器会照你的计划执行。"
    "只输出一个严格 JSON 对象（无解释、无 markdown），字段如下："
    '{"today_plan":"今天总体安排，一句话",'
    '"rest_windows_today":"今天作息类偏好要求的休息时段(从偏好原文推断出具体起止)，并给出执行建议：大约几点开始用一次长 wait、睡到几点(略提前开始、略推迟结束以完整覆盖整个时段)；傍晚后不要接会延伸进该时段的单。无作息类偏好就写 无",'
    '"today_is_off_day":true 或 false（依据"整月需留 N 个完全不出车整天"类偏好 + 台账已休几个整天 + 剩余天数，'
    '判断今天是否应安排为完全不出车的整休日；为 true 时执行器全天只休息）,'
    '"off_days_required":整数 表示偏好要求的【整月完全不出车的整天数】(从"每月至少 N 个整天休息/留 N 个整天检修"等读出；没有这类偏好则 0),'
    '"rest_block_today":{"start":"HH:MM","end":"HH:MM"} 表示今天应当用一次长 wait 完整覆盖的【具体连续休息块】（从作息类偏好推断：若偏好是固定时段就用该时段；若是"每天连续休息≥N 小时"这类，请你为今天选一个 N 小时的连续块）。【重要】该休息块必须完全落在【同一个自然日内、不要跨过午夜】，否则按单日计算连续休息会被从午夜切成两段、两头都不够；没有作息类偏好则填 null。'
    '"avoid_today":"今天要规避的货：品类/地域等（无就写 无）",'
    '"today_special_task":"若偏好里有【针对今天这个具体日期】的硬性日程(如某日须到某地停留/赴约)，写清：去哪个坐标(lat,lng)、几点前必须到、到了停留多少分钟、若多个站点按什么顺序；没有就写 无",'
    '"dated_tasks":[{"date":"M-D","lat":数值,"lng":数值,"wait_minutes":整数}] 把偏好里【某具体日期要到某坐标停留办事】的任务结构化(从原文读出日期/坐标/停留分钟)，可多条。'
    '【多站点行程】(如先到 A 再到 B)请按到访顺序拆成多条(同一 date、先到的排前面)；只路过不久留的站点 wait_minutes 填 1；有"几点前到"的截止就尽早安排。没有则空数组,'
    '"region_target":{"keyword":"地名关键词","lat":数值,"lng":数值,"need_days":整数} 【仅限真实地理地名】针对【在某地名累计接单需≥N个不同日】类偏好：地名关键词 + 该地代表坐标 + 需要的不同天数(均从原文读)；'
    '【严禁】把"某品类货必须接满 N 单"(品类配额，如某类货物指标)当成 region_target——品类不是地名、由别的模块处理，这里只放真实地名；没有地名累计类偏好则 null,'
    '"monthly_todo":"对照台账，本月还没完成的整月类义务及缺口 + 建议何时做（如：某累计型地域偏好还差 N 个不同日；某具体日期的行程在 X 天后、今明别接会拖到那时的单；还差 N 个整休日）",'
    '"avoid_filters":[{"field":"cargo_name 或 start_city 或 end_city 或 deadhead_km","op":"contains 或 gt","value":值,"raw_text":"依据的偏好原文"}] '
    "【查漏补缺】逐条对照偏好原文：今天接单时必须规避、且无法用上述其他字段表达的事项（尤其是措辞特殊/结构化不了的规避型偏好），"
    "按候选字段写成过滤条件(contains 填关键词或数组,gt 填数值)。只写【规避型】(不接什么)，不写必做型；已被常规理解覆盖的不必重复；没有则空数组。}"
)


def plan_day(
    api: Any,
    status: dict[str, Any],
    memory_summary: str,
    now_wall: str,
    month_end_wall: str,
) -> dict[str, Any]:
    """为今天产出一份计划字典；失败返回空字典（执行器照常工作）。绝不抛异常。"""
    try:
        ctx = {
            "今天": now_wall,
            "本月截止": month_end_wall,
            "司机偏好原文": status.get("preferences") or [],
            "历史活动台账": memory_summary,
        }
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user", "content": json.dumps(ctx, ensure_ascii=False)},
        ]
        obj = llm.chat_json(api, messages, max_tokens=700)
        if not isinstance(obj, dict):
            return {}
        directive = {
            "today_plan": str(obj.get("today_plan", "") or ""),
            "rest_windows_today": str(obj.get("rest_windows_today", "") or ""),
            "today_is_off_day": bool(obj.get("today_is_off_day", False)),
            "off_days_required": _norm_int(obj.get("off_days_required")),
            "avoid_today": str(obj.get("avoid_today", "") or ""),
            "today_special_task": str(obj.get("today_special_task", "") or ""),
            "monthly_todo": str(obj.get("monthly_todo", "") or ""),
            "rest_block_today": _norm_rest_block(obj.get("rest_block_today")),
            "dated_tasks": _norm_dated_tasks(obj.get("dated_tasks")),
            "region_target": _norm_region_target(obj.get("region_target")),
            "avoid_filters": _norm_avoid_filters(obj.get("avoid_filters")),
        }
        _logger.info("planner directive: %s", json.dumps(directive, ensure_ascii=False))
        return directive
    except Exception as e:
        _logger.warning("planner.plan_day 失败，返回空计划: %s", e)
        return {}
