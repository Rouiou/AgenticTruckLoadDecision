"""确定性计算工具（与司机偏好无关）：地理、时间、可行性预筛、ROI 排序。

合规：本模块只对 query_cargo 返回的候选做【客观计算】，不读原始数据、不含任何偏好规则
（无地名 / driver_id / 固定时间窗 / 扣钱金额）。偏好判断一律交给 LLM。
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

EPOCH = datetime(2026, 3, 1, 0, 0, 0)
_WALL_FMTS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M")

# 仅用于候选排序的经济假设（非偏好；同一司机对所有候选一致，不影响相对排序）。
COST_PER_KM_ASSUMED = 1.5


def parse_wall(s: Any) -> datetime | None:
    s = str(s).strip()
    for fmt in _WALL_FMTS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def wall_to_min(s: Any) -> int | None:
    dt = parse_wall(s)
    if dt is None:
        return None
    return int((dt - EPOCH).total_seconds() // 60)


def min_to_hhmm(m: int) -> str:
    """仿真分钟 → 当天墙钟 HH:MM（客观时间换算）。"""
    return (EPOCH + timedelta(minutes=int(m))).strftime("%H:%M")


def min_to_wall(m: int) -> str:
    """仿真分钟 → 墙钟「月-日 时:分」（客观换算）。"""
    return (EPOCH + timedelta(minutes=int(m))).strftime("%m-%d %H:%M")


def month_of_min(m: Any) -> int | None:
    """仿真分钟 → 自然月(1-12)。纯客观时间换算，与偏好无关。"""
    if m is None:
        return None
    try:
        return (EPOCH + timedelta(minutes=int(m))).month
    except (TypeError, ValueError, OverflowError):
        return None


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    h = min(1.0, max(0.0, h))
    return 2 * r * math.asin(math.sqrt(h))


def month_end(now_wall: str) -> tuple[int, str]:
    """当前墙钟所在自然月的下月 1 日零点（≈ 仿真月末上界）。返回 (分钟, 墙钟字符串)。
    纯客观时间推算，无偏好。"""
    dt = parse_wall(now_wall) or EPOCH
    y, m = dt.year, dt.month
    nxt = datetime(y + 1, 1, 1) if m == 12 else datetime(y, m + 1, 1)
    return int((nxt - EPOCH).total_seconds() // 60), nxt.strftime("%Y-%m-%d %H:%M:%S")


def _coord(d: Any) -> tuple[float, float] | None:
    if not isinstance(d, dict):
        return None
    try:
        return float(d["lat"]), float(d["lng"])
    except (KeyError, TypeError, ValueError):
        return None


# 赛题默认空驶速度(km/h)；仅用于客观时间估算，非偏好常量。
SPEED_ASSUMED_KMH = 60.0


def _dead_minutes(distance_km: float) -> int:
    if distance_km <= 1e-6:
        return 0
    return max(1, math.ceil(distance_km / SPEED_ASSUMED_KMH * 60))


def score_candidate(c: dict[str, Any], now_min: int, shadow_map: dict[str, float] | None = None) -> float:
    """确定性选单打分 = 有效时薪(元/分钟) + 配额影子价格。
    - 有效时薪 = 净收益 / 占用分钟。占用 = 预计完成 − 现在(含空驶+等装货窗+干线)，
      下限取干线时长(防压线短单/等窗被低估时时薪发散)。
    - 影子价格：候选品类若命中【落后且按配速来不及】的配额，加一单的边际罚款
      (接这单额外避免一单欠额罚)——"凑配额 vs 接高时薪货"由同一把尺自动权衡，
      替代人工履约档位。全部数值来自 LLM 编译的 IR 与客观计数，零偏好常量。"""
    try:
        net = float(c.get("price") or 0.0) - COST_PER_KM_ASSUMED * (
            float(c.get("deadhead_km") or 0.0) + float(c.get("haul_km") or 0.0)
        )
        busy = max(
            int(c.get("est_finish_min") or now_min) - now_min,
            int(c.get("cost_time_minutes") or 0),
            1,
        )
        shadow = float(c.get("_pred_bonus") or 0.0)  # 月度数量下限的影子价格(service 注入)
        if shadow_map:
            shadow += float(shadow_map.get(str(c.get("cargo_name") or ""), 0.0))
        return (net + shadow) / busy
    except (TypeError, ValueError):
        return -1.0


def snap_category(kw: str, known: set[str]) -> tuple[str, str]:
    """把 LLM 编译出的品类关键词对齐到【运行时见过的真实品类名全集】(query 返回累积,零硬编码)。
    评分按 cargo_name 与品类【完全相等】计数——关键词若与真实品类名不完全一致,
    我方会"自以为凑满、官方照罚"。返回 (对齐后词, 状态):
    exact=本身就是完整品类名; snapped=唯一子串/超串对齐; unmapped=无法唯一对齐(调用方降级)。"""
    kw = str(kw or "").strip()
    if not kw:
        return kw, "unmapped"
    if kw in known:
        return kw, "exact"
    sup = [n for n in known if kw in n]
    if len(sup) == 1:
        return sup[0], "snapped"
    sub = [n for n in known if n in kw]
    if len(sub) == 1:
        return sub[0], "snapped"
    return kw, "unmapped"


def estimate_finish_min(now_min: int, deadhead_km: float, load_time: Any, cost_time: int) -> int:
    """预计真实完成时刻(分钟)，镜像 simkit：max(到达装货点, 装货窗开始) + 干线时长。
    纯客观物理时间(空驶+等装货窗+干线)，与任何偏好无关。"""
    arrival = now_min + _dead_minutes(deadhead_km)
    ready = arrival
    if isinstance(load_time, list) and len(load_time) == 2:
        ls = wall_to_min(load_time[0])
        if ls is not None and arrival < ls:
            ready = ls  # 到早了要等装货窗开门(可能很长)
    return ready + max(0, int(cost_time))


def prepare_candidates(
    items: list[dict[str, Any]],
    now_min: int,
    month_end_min: int,
    *,
    top_n: int = 10,
    quota_categories: list[str] | None = None,
) -> list[dict[str, Any]]:
    """对 query_cargo 候选做【偏好无关】预筛 + ROI 排序，返回 Top-N 紧凑候选。

    预筛仅剔除【环境物理上不可行】的货：
      1) 装货窗已确定性过期（当前已晚于窗结束）；
      2) 干线时长导致本月内不可能完成（保守：忽略空驶，只会更超）。
    绝不按偏好剔货。
    """
    out: list[dict[str, Any]] = []
    for it in items:
        cargo = it.get("cargo", {}) if isinstance(it, dict) else {}
        cid = str(cargo.get("cargo_id", "")).strip()
        if not cid:
            continue
        s_raw = cargo.get("start") or {}
        e_raw = cargo.get("end") or {}
        start = _coord(s_raw)
        end = _coord(e_raw)
        if start is None or end is None:
            continue
        try:
            price = float(cargo.get("price", 0.0))
        except (TypeError, ValueError):
            price = 0.0
        try:
            cost_time = int(cargo.get("cost_time_minutes") or 0)
        except (TypeError, ValueError):
            cost_time = 0
        deadhead = float(it.get("distance_km") or 0.0)
        haul = haversine_km(start[0], start[1], end[0], end[1])
        total_km = deadhead + haul

        # —— 偏好无关的环境可行性预筛 ——
        lw = cargo.get("load_time")
        if isinstance(lw, list) and len(lw) == 2:
            end_m = wall_to_min(lw[1])
            if end_m is not None and now_min + _dead_minutes(deadhead) > end_m:
                continue  # 到达装货点时窗已关(含当前已过期)→接单必败、白扣空驶
        est_finish = estimate_finish_min(now_min, deadhead, lw, cost_time)
        if cost_time and est_finish > month_end_min:
            continue  # 本月内完不成(用真实完成时刻：空驶+等窗+干线)

        roi = price - COST_PER_KM_ASSUMED * total_km
        ppm = price / cost_time if cost_time > 0 else price
        out.append(
            {
                "cargo_id": cid,
                "cargo_name": cargo.get("cargo_name"),
                "price": round(price, 2),
                "cost_time_minutes": cost_time,
                "load_time": lw,
                "truck_length": cargo.get("truck_length"),
                "start": {"city": s_raw.get("city"), "lat": round(start[0], 5), "lng": round(start[1], 5)},
                "end": {"city": e_raw.get("city"), "lat": round(end[0], 5), "lng": round(end[1], 5)},
                "deadhead_km": round(deadhead, 2),
                "haul_km": round(haul, 2),
                "预计完成时刻": min_to_wall(est_finish),
                "est_finish_min": est_finish,
                "roi_score": round(roi, 2),
                "price_per_min": round(ppm, 3),
            }
        )
    out.sort(key=lambda c: c["roi_score"], reverse=True)
    top = out[:top_n]
    # —— 品类配额：把"必须接满"的品类货 surface 进候选(否则稀有品类会被 ROI Top-N 丢掉) ——
    cats = [str(k).strip() for k in (quota_categories or []) if str(k).strip()]
    if cats:
        chosen = {c["cargo_id"] for c in top}
        extra: list[dict[str, Any]] = []
        for c in out:
            if not any(k in str(c.get("cargo_name") or "") for k in cats):
                continue
            c["matches_quota"] = True  # 打标，供履约识别
            if c["cargo_id"] not in chosen:
                extra.append(c)
        top = top + extra[:8]  # 已按 ROI 排序，补最多 8 条配额品类货
    return top
