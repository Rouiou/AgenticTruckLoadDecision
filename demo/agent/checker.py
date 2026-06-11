"""确定性合规检查器：用编译出的【通用约束】对候选/动作做"标注 + 否决"。

合规要点（关键）：
- 代码只做"取字段 + 按算子比较 + 时段静止判定"，**不含任何偏好常量**（地名/品类/时段/阈值全来自 IR）；
- **绝不删货、绝不替偏好决定休息**：annotate 只给候选打违规标签；veto 只对违规动作返回原因、退回 LLM 重决策。
  "按偏好取舍"的主体始终是大模型。
"""

from __future__ import annotations

from typing import Any


def _field_value(cand: dict[str, Any], field: str) -> Any:
    if field == "cargo_name":
        return cand.get("cargo_name") or ""
    if field == "start_city":
        return (cand.get("start") or {}).get("city") or ""
    if field == "end_city":
        return (cand.get("end") or {}).get("city") or ""
    if field == "deadhead_km":
        try:
            return float(cand.get("deadhead_km") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    if field == "truck_length":
        v = cand.get("truck_length")
        return " ".join(str(x) for x in v) if isinstance(v, list) else str(v or "")
    return ""


def _as_keywords(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if value is None:
        return []
    return [str(value)]


def _violates_filter(cand: dict[str, Any], f: dict[str, Any]) -> bool:
    fv = _field_value(cand, str(f.get("field", "")))
    op = str(f.get("op", ""))
    val = f.get("value")
    if op in ("gt", ">"):
        try:
            return float(fv) > float(val)
        except (TypeError, ValueError):
            return False
    # 其余一律按【文本包含任一关键词】判定：字段是完整名称(如"某省某市某区")，
    # 偏好值是关键词(如某地名/某品类)，故无论 LLM 给 contains 还是 equals 都用包含，最稳。
    kws = _as_keywords(val)
    s = str(fv)
    return any(k and k in s for k in kws)


def _md_tuple(s: Any) -> tuple[int, int] | None:
    try:
        p = str(s).strip().split("-")
        return (int(p[-2]), int(p[-1]))
    except (TypeError, ValueError):
        return None


def _filter_active_today(f: dict[str, Any], today_md: tuple[int, int] | None) -> bool:
    """该规避是否在今天生效。带 active_from/to 的只在日期区间内生效；否则常驻。"""
    frm = _md_tuple(f.get("active_from")) if f.get("active_from") else None
    to = _md_tuple(f.get("active_to")) if f.get("active_to") else None
    if frm is None and to is None:
        return True
    if today_md is None:
        return True
    lo, hi = (frm or to), (to or frm)
    return lo <= today_md <= hi


def has_dated_restriction_today(ir: dict[str, Any], today_md: tuple[int, int] | None) -> bool:
    """今天是否落在某条【日期条件型规避】的生效窗口内（如某偏好仅某几日生效）。
    用于让"反闲置强制接单"在这种日子让位给 LLM（它更懂这类需整体规避的规则）。"""
    for f in (ir or {}).get("order_filters") or []:
        if (f.get("active_from") or f.get("active_to")) and _filter_active_today(f, today_md):
            return True
    return False


def annotate(candidates: list[dict[str, Any]], ir: dict[str, Any], today_md: tuple[int, int] | None = None) -> None:
    """给每个候选打 violates 标签（命中且【当日生效】的偏好原文）。只标注，不删货。"""
    filters = (ir or {}).get("order_filters") or []
    for c in candidates:
        c["violates"] = [
            (f.get("raw_text") or f.get("field"))
            for f in filters
            if _filter_active_today(f, today_md) and _violates_filter(c, f)
        ]


def _parse_window(win: str) -> tuple[int, int] | None:
    """'HH:MM-HH:MM' → (起分, 止分)（一天内分钟）。"""
    try:
        a, b = win.split("-")

        def m(x: str) -> int:
            h, mm = x.strip().split(":")
            return int(h) * 60 + int(mm)

        return m(a), m(b)
    except Exception:
        return None


def window_tuples(ir: dict[str, Any], directive: dict[str, Any] | None = None) -> list[tuple[int, int]]:
    """汇总每天需保持静止的【休息时段】(一天内分钟 (起,止))，全部来自编译器对偏好原文的解析。
    用 compiler 的稳定 IR(按偏好哈希缓存)而非每日 planner，保证跨天/跨次一致、低方差。"""
    out: list[tuple[int, int]] = []
    for w in (ir or {}).get("rest_windows") or []:
        kind = str(w.get("kind") or "").strip().lower()
        win = w.get("window")
        pw = _parse_window(str(win)) if win else None
        mc = w.get("min_continuous_minutes")
        if kind == "continuous" or (kind != "scheduled" and mc and pw is None):
            # 【连续 N 小时型】评分按单日内最长连续休息算 → 强制单日内 0:00~N 连续块(不跨午夜)。
            out.append((0, max(1, min(int(mc or 420), 1439))))
        elif pw:
            # 【固定钟点型】照搬该钟点(可跨午夜，按覆盖判定，满足 0-6/23-6 这类)。
            out.append(pw)
        elif mc:
            out.append((0, max(1, min(int(mc), 1439))))
    return out


def in_any_window(now_min: int, windows: list[tuple[int, int]]) -> tuple[bool, int | None]:
    """当前是否处于任一休息时段；返回 (是否, 该时段结束的绝对分钟)。"""
    day = now_min // 1440
    tod = now_min - day * 1440
    for s, e in windows:
        if e <= s:  # 跨午夜
            if tod >= s or tod < e:
                return True, day * 1440 + (e + 1440 if tod >= s else e)
        elif s <= tod < e:
            return True, day * 1440 + e
    return False, None


# 接单防撞缓冲(分钟)：预计完成时刻的估算偏乐观(空驶/装货/路况)，给个余量，
# 不接"刚好压着休息时段开头结束"的单——否则尾巴会溢进休息时段被判违规。非偏好常量、与具体作息无关。
REST_ACCEPT_BUFFER_MIN = 60


def overlaps_any_window(
    now_min: int, finish_min: int, windows: list[tuple[int, int]], buffer_min: int = REST_ACCEPT_BUFFER_MIN
) -> bool:
    """区间 [now, finish(+缓冲)] 是否与任一休息时段(按天展开)相交——用于拦"会跨进休息块"的单。
    finish 加 buffer_min 余量：估算偏乐观时也不至于压线接单导致尾巴溢入休息时段。"""
    if not windows:
        return False
    finish_min = finish_min + max(0, int(buffer_min))
    if finish_min <= now_min:
        return False
    for d in range(now_min // 1440, finish_min // 1440 + 1):
        for s, e in windows:
            ws = d * 1440 + s
            we = d * 1440 + (e if e > s else e + 1440)
            if now_min < we and finish_min > ws:
                return True
    return False


def veto(
    action: dict[str, Any],
    cand_by_id: dict[str, Any],
    windows: list[tuple[int, int]],
    now_min: int,
) -> str | None:
    """返回违规原因（字符串）或 None。只否决、不替偏好做决定。"""
    name = action.get("action")
    if name == "take_order":
        cid = str((action.get("params") or {}).get("cargo_id", ""))
        c = cand_by_id.get(cid)
        if c and c.get("violates"):
            return "该单违反偏好: " + "; ".join(str(x) for x in c["violates"])
        if c:
            try:
                finish = int(c.get("est_finish_min") or (now_min + int(c.get("cost_time_minutes") or 0)))
            except (TypeError, ValueError):
                finish = now_min
            if overlaps_any_window(now_min, finish, windows):
                return "接这单会让车在休息时段仍在行驶(空驶/等装货/干线跨入休息块)，应改休息或选更短的单"
    if name in ("take_order", "reposition"):
        inside, _ = in_any_window(now_min, windows)
        if inside:
            return f"当前处于休息时段，不应{name}，应改为休息"
    return None
