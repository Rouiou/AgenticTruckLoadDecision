"""跨步记忆：把官方 query_decision_history 的已执行记录整理成【客观事实台账】，
供决策时喂给 LLM。

合规：只记录事实与计数（动作、时间、位置、里程、价格），**不对任何具体偏好下判断**、
不含偏好常量（无地名 / 固定时间窗 / 阈值）。是否满足偏好由 LLM 看台账自行判断。
"""

from __future__ import annotations

import logging
from typing import Any

from . import tools

_logger = logging.getLogger("agent.memory")


def _result_pos(rec: dict[str, Any]) -> dict[str, Any] | None:
    p = rec.get("position_after")
    if isinstance(p, dict) and "lat" in p and "lng" in p:
        try:
            return {"lat": float(p["lat"]), "lng": float(p["lng"])}
        except (TypeError, ValueError):
            return None
    return None


class DriverMemory:
    """每个 driver_id 一本台账。实例随 ModelDecisionService 全程存活，跨步累积。"""

    def __init__(self) -> None:
        self._mem: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _fresh() -> dict[str, Any]:
        return {
            "ingested": 0, "events": [], "cargo": {}, "last_progress": None,
            "planned_day": None, "directive": {}, "dated_done": set(),
        }

    def region_days_done(self, driver_id: str, keyword: str) -> set[int]:
        """已在含 keyword 地名(起或终点)成功接单的【不同日期序号集合】。keyword 来自 LLM 编译。"""
        d = self._mem.get(driver_id)
        if not d or not keyword:
            return set()
        days: set[int] = set()
        for ev in d["events"]:
            if ev.get("action") == "take_order" and ev.get("accepted") and ev.get("end_min") is not None:
                if keyword in str(ev.get("start_city") or "") or keyword in str(ev.get("dest_city") or ""):
                    days.add(ev["end_min"] // 1440)
        return days

    def category_orders_done(self, driver_id: str, keyword: str, month: int) -> int:
        """本月(自然月 month)已【成功接单】、品类(cargo_name)含 keyword 的【单数】。
        供品类配额履约用。keyword/month 均来自 LLM 编译，代码只做文本包含 + 计数(零偏好常量)。"""
        d = self._mem.get(driver_id)
        if not d or not keyword:
            return 0
        cnt = 0
        for ev in d["events"]:
            if ev.get("action") != "take_order" or not ev.get("accepted"):
                continue
            if ev.get("month") != month:
                continue
            if keyword in str(ev.get("cargo_name") or ""):
                cnt += 1
        return cnt

    def monthly_predicate_count(self, driver_id: str, field: str, threshold: float, month: int) -> int:
        """本月已成功接单中，数值字段 field > threshold 的【单数】(供'月度长途上限'用)。
        field/threshold/month 均来自 LLM 编译；代码只做数值比较 + 计数。"""
        d = self._mem.get(driver_id)
        if not d:
            return 0
        cnt = 0
        for ev in d["events"]:
            if ev.get("action") != "take_order" or not ev.get("accepted") or ev.get("month") != month:
                continue
            try:
                if float(ev.get(field) or 0) > float(threshold):
                    cnt += 1
            except (TypeError, ValueError):
                continue
        return cnt

    def off_days_taken(self, driver_id: str, current_day: int) -> int:
        """今天之前【完全没出车的整天】数。客观计数，供整休配额刹车用。
        关键：按动作区间[start,end]【覆盖】到的天算活跃(与评分器一致)——长途会跨天，
        某天没有新接单事件 ≠ 那天空闲(前一天的长途可能整天在路上)。"""
        d = self._mem.get(driver_id)
        if not d:
            return 0
        active: set[int] = set()
        for ev in d["events"]:
            if ev.get("action") not in ("take_order", "reposition"):
                continue
            if ev.get("action") == "take_order" and not ev.get("accepted"):
                continue
            s, e = ev.get("start_min"), ev.get("end_min")
            if s is None or e is None:
                continue
            for dd in range(int(s) // 1440, int(e) // 1440 + 1):
                active.add(dd)
        return sum(1 for dd in range(max(0, current_day)) if dd not in active)

    def mark_dated_done(self, driver_id: str, key: str) -> None:
        self._mem.setdefault(driver_id, self._fresh()).setdefault("dated_done", set()).add(key)

    def is_dated_done(self, driver_id: str, key: str) -> bool:
        return key in (self._mem.get(driver_id, {}).get("dated_done") or set())

    def planned_day(self, driver_id: str) -> int | None:
        return self._mem.get(driver_id, {}).get("planned_day")

    def set_directive(self, driver_id: str, day: int, directive: dict[str, Any]) -> None:
        d = self._mem.setdefault(driver_id, self._fresh())
        d["planned_day"] = day
        d["directive"] = directive or {}

    def get_directive(self, driver_id: str) -> dict[str, Any]:
        return self._mem.get(driver_id, {}).get("directive") or {}

    def note_candidate(self, driver_id: str, cargo_id: str, cand: dict[str, Any]) -> None:
        """缓存本步展示过的候选信息，便于之后从历史回查价格/坐标。"""
        d = self._mem.setdefault(driver_id, self._fresh())
        d["cargo"][cargo_id] = {
            "price": cand.get("price"),
            "cargo_name": cand.get("cargo_name"),
            "start": cand.get("start"),
            "end": cand.get("end"),
            "cost_time_minutes": cand.get("cost_time_minutes"),
        }

    def observe(self, driver_id: str, status: dict[str, Any], api: Any) -> None:
        """每步起始调用：检测新月份重置 + 增量摄取已执行历史。**绝不抛异常**。"""
        try:
            d = self._mem.setdefault(driver_id, self._fresh())
            progress = int(status.get("simulation_progress_minutes") or 0)
            if progress == 0 and d["events"]:
                # 同一实例复用、时间归零 → 新一轮仿真，重置台账
                self._mem[driver_id] = d = self._fresh()
            d["last_progress"] = progress
            hist = api.query_decision_history(driver_id, -1)
            records = hist.get("records") or []
            if len(records) <= d["ingested"]:
                return
            for rec in records[d["ingested"]:]:
                self._ingest(d, rec)
            d["ingested"] = len(records)
        except Exception as e:  # 记忆失败绝不能拖垮决策
            _logger.warning("memory.observe 跳过(%s): %s", driver_id, e)

    def _ingest(self, d: dict[str, Any], rec: dict[str, Any]) -> None:
        action = rec.get("action", {}) or {}
        name = str(action.get("action", "")).strip().lower()
        result = rec.get("result", {}) or {}
        end_min = tools.wall_to_min(rec.get("simulation_end_time"))
        elapsed = int(rec.get("step_elapsed_minutes") or 0)
        start_min = end_min - elapsed if end_min is not None else None
        ev: dict[str, Any] = {"action": name, "end_min": end_min, "start_min": start_min, "elapsed": elapsed}
        if name == "wait":
            ev["rest_min"] = elapsed
        elif name == "take_order":
            cid = str((action.get("params") or {}).get("cargo_id", "")).strip()
            info = d["cargo"].get(cid, {})
            end_info = info.get("end") or {}
            start_info = info.get("start") or {}
            ev.update(
                {
                    "cargo_id": cid,
                    "accepted": bool(result.get("accepted")),
                    "eligible": bool(result.get("income_eligible", result.get("accepted"))),
                    "month": tools.month_of_min(start_min),  # 该单计入的自然月(对齐评分器 action_start 月)
                    "cost_time_minutes": info.get("cost_time_minutes"),
                    "price": info.get("price"),
                    "cargo_name": info.get("cargo_name"),
                    "dest": _result_pos(rec) or end_info,
                    "dest_city": end_info.get("city"),
                    "start_city": start_info.get("city"),
                    "deadhead_km": result.get("pickup_deadhead_km"),
                    "haul_km": result.get("haul_distance_km"),
                }
            )
        elif name == "reposition":
            ev["dest"] = _result_pos(rec)
        d["events"].append(ev)

    def summary_text(self, driver_id: str, status: dict[str, Any]) -> str:
        d = self._mem.get(driver_id)
        if not d or not d["events"]:
            return "（暂无历史动作，这是本月首个决策）"
        events = d["events"]
        gross = dist = 0.0
        n_orders = 0
        for ev in events:
            if ev["action"] == "take_order" and ev.get("accepted"):
                n_orders += 1
                if ev.get("eligible") and ev.get("price"):
                    gross += float(ev["price"])
                dist += float(ev.get("deadhead_km") or 0) + float(ev.get("haul_km") or 0)

        days: dict[int, dict[str, Any]] = {}
        for ev in events:
            if ev.get("end_min") is None:
                continue
            day = ev["end_min"] // 1440
            g = days.setdefault(day, {"orders": [], "rest": 0, "moved": False, "rests": []})
            if ev["action"] == "wait":
                g["rest"] += int(ev.get("rest_min") or 0)
                if ev.get("start_min") is not None and ev.get("end_min") is not None:
                    g["rests"].append((ev["start_min"], ev["end_min"]))
            elif ev["action"] == "take_order" and ev.get("accepted"):
                g["moved"] = True
                g["orders"].append(ev)
            elif ev["action"] == "reposition":
                g["moved"] = True

        active_days = sorted(dd for dd, g in days.items() if g["moved"])
        cur_day = int(status.get("simulation_progress_minutes") or 0) // 1440
        lines = [
            f"累计: 完成订单 {n_orders} 笔, 计入收入约 ¥{gross:.0f}, 行驶约 {dist:.0f} km",
            f"今天是第 {cur_day + 1} 天(第1天=3月1日); 已出车的日期序号: {active_days or '无'}",
            "(逐日明细, 用于你判断作息/回家/区域/整天休息等偏好是否满足)",
        ]
        for day in sorted(days.keys())[-7:]:
            g = days[day]
            dest = "; ".join(
                f"{o.get('start_city') or '?'}→{o.get('dest_city') or '?'}[{o.get('cargo_name') or '?'}]¥{o.get('price') if o.get('price') is not None else '?'}"
                for o in g["orders"]
            ) or "无接单"
            rest_str = ", ".join(
                f"{tools.min_to_hhmm(s)}→{tools.min_to_hhmm(e)}" for s, e in g["rests"][:6]
            ) or "无"
            lines.append(
                f"  第{day + 1}天: 休息{g['rest']}min(时段:{rest_str}), 出车={'是' if g['moved'] else '否'}, {dest}"
            )
        return "\n".join(lines)
