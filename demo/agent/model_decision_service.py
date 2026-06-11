"""Budgeted DriverOps-MultiAgent Council decision service.

LLM owns strategy, preference interpretation, debate, and final decisions. The
code only calls allowed environment APIs, computes factual metrics, filters hard
illegal orders, and validates the final action shape.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from simkit.ports import SimulationApiPort

_SIMULATION_EPOCH = datetime(2026, 3, 1, 0, 0, 0)
_WALL_FMT = "%Y-%m-%d %H:%M:%S"
_DEFAULT_SPEED_KMPH = 60.0
_DEFAULT_COST_PER_KM = 1.5
_HORIZON_MINUTES = 92 * 24 * 60
_TOKEN_SOFT_LIMIT = 4_600_000
_MAX_WAIT_MINUTES = _HORIZON_MINUTES


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_km = 6371.0
    p1, l1 = math.radians(lat1), math.radians(lng1)
    p2, l2 = math.radians(lat2), math.radians(lng2)
    dp, dl = p2 - p1, l2 - l1
    h = math.sin(dp * 0.5) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl * 0.5) ** 2
    return 2.0 * radius_km * math.asin(math.sqrt(max(0.0, min(1.0, h))))


def _distance_minutes(distance_km: float, speed_kmph: float = _DEFAULT_SPEED_KMPH) -> int:
    if distance_km <= 0:
        return 1
    return max(1, math.ceil(distance_km / speed_kmph * 60.0))


def _parse_wall_minutes(value: str | None) -> int | None:
    if not value:
        return None
    try:
        dt = datetime.strptime(str(value).strip(), _WALL_FMT)
    except ValueError:
        return None
    return int((dt - _SIMULATION_EPOCH).total_seconds() // 60)


def _wall_text(sim_minutes: int) -> str:
    return (_SIMULATION_EPOCH + timedelta(minutes=int(sim_minutes))).strftime(_WALL_FMT)


def _month_key(sim_minutes: int) -> str:
    return (_SIMULATION_EPOCH + timedelta(minutes=int(sim_minutes))).strftime("%Y-%m")


def _weekday_text(sim_minutes: int) -> str:
    return (_SIMULATION_EPOCH + timedelta(minutes=int(sim_minutes))).strftime("%A")


def _covered_clock_hours(start_min: int, end_min: int) -> list[int]:
    if end_min <= start_min:
        return []
    hours: set[int] = set()
    cursor = start_min
    while cursor < end_min:
        hours.add((cursor // 60) % 24)
        cursor = ((cursor // 60) + 1) * 60
    return sorted(hours)


def _interval_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return max(a_start, b_start) < min(a_end, b_end)


def _cargo_point(cargo: dict[str, Any], key: str) -> tuple[float, float]:
    point = cargo.get(key) or {}
    return float(point["lat"]), float(point["lng"])


def _cargo_price_yuan(cargo: dict[str, Any]) -> float:
    price = float(cargo.get("price", 0.0) or 0.0)
    # query_cargo in the provided simulator already normalizes raw cents to yuan.
    # Keep a guard for unnormalized API payloads without dividing normal yuan prices twice.
    return price / 100.0 if price > 10000 else price


def _optional_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class CandidateFact:
    cargo_id: str
    cargo: dict[str, Any]
    source: str
    query_km: float
    pickup_km: float
    haul_km: float
    pickup_min: int
    wait_min: int
    transport_min: int
    finish_min: int
    price_yuan: float
    cost_yuan: float
    net_yuan_before_pref: float
    net_per_hour_before_pref: float
    near_end_cargo_seen: int
    legal: bool
    veto_reasons: list[str] = field(default_factory=list)


class ModelDecisionService:
    """Batched multi-agent council.

    One LLM call per normal step:
    1. Full Multi-Agent Council: State Observer + Commander + Scout + Cargo Analysts
       + Preference + Risk + Future + Critic + Arbiter

    In budget guard mode, candidate count shrinks, but the LLM still makes the
    final decision. Invalid LLM actions fail fast instead of being replaced by a
    deterministic backup action.
    """

    ENABLE_THINKING: bool = False

    def __init__(self, api: SimulationApiPort, *, enable_thinking: bool | None = None) -> None:
        self._api = api
        self._enable_thinking = self.ENABLE_THINKING if enable_thinking is None else enable_thinking
        self._logger = logging.getLogger("agent.decision_service")
        self._observed_points_by_driver: dict[str, list[dict[str, float]]] = {}
        self._chosen_orders_by_driver: dict[str, list[dict[str, Any]]] = {}
        self._preference_policy_by_driver: dict[str, dict[str, Any]] = {}
        self._target_memory_by_driver: dict[str, list[dict[str, Any]]] = {}

    def decide(self, driver_id: str) -> dict[str, Any]:
        status = self._api.get_driver_status(driver_id)
        all_history = self._api.query_decision_history(driver_id, -1)
        records = all_history.get("records") if isinstance(all_history.get("records"), list) else []
        recent = records[-8:]
        cumulative_tokens = self._history_tokens(records)
        budget_guard = cumulative_tokens >= _TOKEN_SOFT_LIMIT
        pref_policy = self._compiled_preference_policy(status, recent)

        rest_wait = self._current_rest_wait_minutes(status, pref_policy)
        if rest_wait is not None:
            action = {"action": "wait", "params": {"duration_minutes": rest_wait}}
            self._logger.info(
                "deterministic rest decision driver=%s sim_min=%s tokens_so_far=%s action=%s",
                driver_id,
                status.get("simulation_progress_minutes"),
                cumulative_tokens,
                action,
            )
            return action

        items = self._observe_current_market(driver_id, status, budget_guard)
        status_after_query = self._api.get_driver_status(driver_id)
        self._remember_observed_points(driver_id, items)
        first_pass = self._build_candidate_facts(status_after_query, items, source="local")
        expansion_plan = None
        if self._should_expand_market(first_pass, budget_guard):
            extra_items = self._run_deterministic_market_scout_queries(driver_id, status_after_query, first_pass, budget_guard)
            if extra_items:
                items.extend(extra_items)
                status_after_query = self._api.get_driver_status(driver_id)
                self._remember_observed_points(driver_id, extra_items)
        else:
            self._logger.info("market scout skipped: local portfolio is sufficiently broad")
        all_candidates = self._build_candidate_facts(status_after_query, items, source="portfolio")
        candidates = self._select_prompt_candidates(all_candidates, status_after_query, budget_guard)
        self._logger.info(
            "candidate_facts_json data=%s",
            json.dumps([self._candidate_prompt(c) for c in candidates], ensure_ascii=False, separators=(",", ":")),
        )
        if self._unknown_constraints(pref_policy) and candidates:
            guardian = self._preference_guardian_council(
                status_after_query,
                recent,
                candidates,
                cumulative_tokens,
                pref_policy,
            )
            all_candidates = self._guardian_downgrade_candidates(all_candidates, guardian)
        action = self._deterministic_execution_decision(driver_id, status_after_query, all_candidates, pref_policy)
        self._remember_chosen_action(driver_id, action, all_candidates)

        self._logger.info(
            "deterministic decision driver=%s sim_min=%s tokens_so_far=%s items=%s candidates=%s action=%s params=%s",
            driver_id,
            status_after_query.get("simulation_progress_minutes"),
            cumulative_tokens,
            len(items),
            len(all_candidates),
            action.get("action"),
            action.get("params"),
        )
        return action

    def _compiled_preference_policy(self, status: dict[str, Any], recent: list[dict[str, Any]]) -> dict[str, Any]:
        driver_id = str(status.get("driver_id", ""))
        prefs = status.get("preferences") or []
        signature = json.dumps(prefs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        cached = self._preference_policy_by_driver.get(driver_id)
        if cached and cached.get("_signature") == signature:
            return cached
        policy = self._agent_json(
            "Preference_Compiler_Agent",
            (
                "你是一次性 Preference Multi-Agent Compiler。请在同一次回答中完成三个子 Agent 的工作："
                "1) Preference Parser Agent 抽取休息、限额、最低指标；"
                "2) Target Continuity Auditor Agent 专门检查跨月欠额、补上月、延续指标、累计/重置口径；"
                "3) Risk Compiler Agent 输出可执行 machine_ir。"
                "最终只输出稳定、简短的经营 policy，供后续 Duty、Guardian、Arbiter 多 Agent 共用。"
                "不要做最终动作，不要评价具体货源。"
                "必须泛化到隐藏司机：只根据文本抽取约束，不要写死司机、月份、货类或公开集。"
                "若文本有跨午夜时段，必须说明凌晨属于前一日夜间窗口；例如 23:00-06:00 中的周日00:30"
                "属于周六23:00-周日06:00 这一夜。"
                "若文本说晚两个小时再休息，表示休息开始时间顺延，结束时间不自动顺延。"
                "强制自检：21:00-06:00 + 晚两个小时再休息 = 23:00-06:00；"
                "不得输出 23:00-08:00，除非文本明确说晚两个小时起床、晚两个小时结束或休息结束顺延。"
                "若有至少/最多/不得/必须/罚/扣/指标，输出它们的优先级和计数口径。"
                "若文本提到“上月没完成、欠额、本月补、接着补”，必须结合 preference_ledger 和历史目标语义"
                "判断欠额来自哪一个旧指标；如果当前文本没有明说旧货类，但历史 policy/ledger 能确定旧货类，"
                "machine_ir.cargo_targets 必须额外输出本月补欠目标，并设置 makeup_from_previous=true。"
                "若文本同时有“本月新指标”和“补旧欠额”，不要把旧欠额混入本月新指标的 min_count；"
                "本月明说的货类目标只填明说数量，旧欠额应作为单独 cargo_target 输出或交给 target_memory 补齐。"
                "若目标是按月考核，month 必须是计分归属月，min_count 必须是该 target 自身需要完成的数量。"
                "输出必须包含 machine_ir，字段尽量用数字和枚举；后续代码执行层会直接读取它。"
            ),
            {
                "state": self._state_summary(status),
                "preferences_visible_now": prefs,
                "preference_ledger": self._preference_ledger(driver_id),
                "target_memory_from_previous_policies": self._target_memory_by_driver.get(driver_id, []),
                "recent_actions": [self._compact_history(r) for r in recent[-4:]],
                "output_schema": {
                    "compiler_notes": {
                        "preference_parser_agent": "brief",
                        "target_continuity_auditor_agent": "brief",
                        "risk_compiler_agent": "brief",
                    },
                    "policy_summary": "brief",
                    "duty_windows": [
                        {
                            "label": "brief",
                            "window": "clock/date text",
                            "forbidden": ["take_order", "reposition", "other if text says so"],
                            "overnight_note": "brief when crosses midnight",
                        }
                    ],
                    "quota_limits": [
                        {"scope": "month/date", "condition": "brief", "limit": "brief", "counting_basis": "brief"}
                    ],
                    "minimum_targets": [
                        {"scope": "month/date", "condition": "brief", "target": "brief", "makeup_rule": "brief"}
                    ],
                    "soft_preferences": ["brief"],
                    "critical_checks": ["short checklist for later agents"],
                    "machine_ir": {
                        "rest_windows": [
                            {
                                "label": "string",
                                "days": "all|weekday|weekend",
                                "start_hour": "0..23 integer",
                                "end_hour": "0..23 integer",
                                "forbid_take_order": "boolean",
                                "forbid_reposition": "boolean",
                            }
                        ],
                        "long_haul_limits": [
                            {
                                "scope": "monthly",
                                "threshold_minutes": "integer",
                                "max_count": "integer",
                            }
                        ],
                        "cargo_targets": [
                            {
                                "month": "1..12 integer",
                                "cargo_name": "string",
                                "min_count": "integer",
                                "penalty_amount": "number if visible preference has per-violation penalty",
                                "makeup_from_previous": "boolean",
                            }
                        ],
                        "cargo_max_limits": [
                            {
                                "month": "1..12 integer or null for all months",
                                "cargo_name": "string",
                                "max_count": "integer",
                                "penalty_amount": "number if visible",
                            }
                        ],
                        "region_avoid": [{"field": "origin|destination|either", "keyword": "city/province/region text"}],
                        "origin_avoid": [{"keyword": "city/province/region text"}],
                        "destination_prefer": [{"keyword": "city/province/region text", "bonus": "number or null"}],
                        "region_min_targets": [],
                        "date_or_weekday_rules": [],
                        "makeup_targets": [],
                        "soft_preferences": [],
                        "unknown_constraints": [
                            {
                                "preference_text": "text not covered by executable IR",
                                "why_unsupported": "brief",
                                "risk_level": "low|medium|high",
                            }
                        ],
                        "penalty_model": [],
                    },
                },
            },
        )
        policy = self._merge_preference_target_memory(driver_id, policy, prefs)
        policy["_signature"] = signature
        self._preference_policy_by_driver[driver_id] = policy
        self._logger.info(
            "compiled_preference_policy driver=%s data=%s",
            driver_id,
            json.dumps(policy, ensure_ascii=False, separators=(",", ":")),
        )
        return policy

    def _pre_market_duty_council(
        self,
        status: dict[str, Any],
        recent: list[dict[str, Any]],
        cumulative_tokens: int,
        pref_policy: dict[str, Any],
    ) -> dict[str, Any]:
        return self._agent_json(
            "PreMarket_Duty_Preference_Agent",
            (
                "你是查货前的 Duty / Preference Agent。你只判断一件事："
                "在调用 query_cargo 之前，司机是否必须先等待，以免查询耗时、接单或空驶破坏可见偏好。"
                "优先使用 compiled_preference_policy 中已编译的窗口、计数和指标解释。"
                "如果当前就在这类窗口内，或距离窗口开始很近、继续查货很可能让非等待时间切入窗口，"
                "输出 must_wait=true，并给出 wait 分钟数；否则输出 must_wait=false。"
                "跨午夜窗口必须向前归属：例如 23:00-06:00 中的 00:30 属于前一日夜间窗口。"
                "不要根据公开集写死司机；只解释可见偏好文本。"
            ),
            {
                "state": self._state_summary(status),
                "preferences_visible_now": status.get("preferences") or [],
                "compiled_preference_policy": self._compact_policy(pref_policy),
                "preference_ledger": self._preference_ledger(str(status.get("driver_id", ""))),
                "recent_actions": [self._compact_history(r) for r in recent[-4:]],
                "output_schema": {
                    "pref_interpretation": "brief",
                    "current_window_status": "inside_rest|near_rest_start|free_to_query|unclear",
                    "must_wait": "boolean",
                    "duration_minutes": "integer 1..1440 only when must_wait=true",
                    "reason": "brief",
                },
            },
        )

    @staticmethod
    def _must_wait_before_market(duty: dict[str, Any]) -> bool:
        if not bool(duty.get("must_wait")):
            return False
        duration = int(duty.get("duration_minutes", 0) or 0)
        if not 1 <= duration <= _MAX_WAIT_MINUTES:
            raise ValueError(f"PreMarket_Duty_Preference_Agent invalid wait duration: {duration}")
        return True

    def _deterministic_execution_decision(
        self, driver_id: str, status: dict[str, Any], candidates: list[CandidateFact], pref_policy: dict[str, Any]
    ) -> dict[str, Any]:
        sim_min = int(status.get("simulation_progress_minutes", 0) or 0)
        ledger = self._preference_ledger(driver_id)
        scored: list[tuple[float, CandidateFact, list[str]]] = []
        for cand in candidates:
            if not cand.legal:
                continue
            vetoes = self._deterministic_vetoes(cand, pref_policy, ledger)
            if vetoes:
                continue
            score = self._candidate_score(cand, pref_policy, ledger, sim_min)
            scored.append((score, cand, []))

        scored.sort(key=lambda item: item[0], reverse=True)
        if scored and scored[0][0] > 0:
            best = scored[0][1]
            self._logger.info(
                "deterministic_score_choice cargo=%s score=%.2f facts=%s",
                best.cargo_id,
                scored[0][0],
                json.dumps(self._candidate_prompt(best), ensure_ascii=False, separators=(",", ":")),
            )
            return {"action": "take_order", "params": {"cargo_id": best.cargo_id}}

        wait = self._next_useful_wait_minutes(sim_min, pref_policy)
        self._logger.info("deterministic_wait no_positive_candidate wait=%s seen=%s", wait, len(candidates))
        return {"action": "wait", "params": {"duration_minutes": wait}}

    def _deterministic_vetoes(
        self, cand: CandidateFact, pref_policy: dict[str, Any], ledger: dict[str, Any]
    ) -> list[str]:
        active_start = cand.finish_min - cand.pickup_min - cand.wait_min - cand.transport_min
        vetoes: list[str] = []
        if self._interval_overlaps_forbidden_window(active_start, cand.finish_min, pref_policy):
            vetoes.append("rest_window_overlap")
        for limit in self._long_haul_limits(pref_policy):
            threshold = int(limit.get("threshold_minutes", 480) or 480)
            max_count = int(limit.get("max_count", 999999) or 999999)
            if cand.transport_min <= threshold:
                continue
            month = _month_key(cand.finish_min)
            used = int((ledger.get("transport_duration_bins_by_month") or {}).get(month, {}).get("over_8h", 0) or 0)
            if used >= max_count:
                vetoes.append("long_haul_quota_full")
        for limit in self._cargo_max_limits(pref_policy):
            name = self._cargo_name(cand.cargo)
            target_name = str(limit.get("cargo_name") or "").strip()
            if not name or not target_name or name != target_name:
                continue
            max_count = _optional_int(limit.get("max_count"), 999999)
            month_num = self._target_month(limit)
            month = f"2026-{month_num:02d}" if month_num is not None else _month_key(cand.finish_min)
            used = int((ledger.get("cargo_name_counts_by_month") or {}).get(month, {}).get(name, 0) or 0)
            if used >= max_count:
                vetoes.append("cargo_max_limit_full")
        if self._matches_region_avoid(cand.cargo, pref_policy):
            vetoes.append("region_avoid")
        return vetoes

    def _candidate_score(
        self, cand: CandidateFact, pref_policy: dict[str, Any], ledger: dict[str, Any], sim_min: int
    ) -> float:
        active_min = max(1, cand.pickup_min + cand.wait_min + cand.transport_min)
        net = cand.net_yuan_before_pref
        nph = net / (active_min / 60.0)
        score = net + 5.0 * nph + 12.0 * cand.near_end_cargo_seen - 0.35 * cand.pickup_km
        score += self._target_bonus(cand, pref_policy, ledger, sim_min)
        for limit in self._long_haul_limits(pref_policy):
            threshold = int(limit.get("threshold_minutes", 480) or 480)
            if cand.transport_min > threshold:
                score -= 150.0
        if cand.finish_min - sim_min > 20 * 60:
            score -= 300.0
        return score

    def _target_bonus(
        self, cand: CandidateFact, pref_policy: dict[str, Any], ledger: dict[str, Any], sim_min: int
    ) -> float:
        name = self._cargo_name(cand.cargo)
        if not name:
            return 0.0
        bonus = 0.0
        counts = ledger.get("cargo_name_counts_by_month") or {}
        now_month = (_SIMULATION_EPOCH + timedelta(minutes=sim_min)).month
        for target in self._cargo_targets(pref_policy):
            target_name = str(target.get("cargo_name", "") or "").strip()
            if not target_name or target_name != name:
                continue
            try:
                month = int(target.get("month"))
                min_count = int(target.get("min_count"))
            except (TypeError, ValueError):
                continue
            month_key = f"2026-{month:02d}"
            used = int((counts.get(month_key) or {}).get(name, 0) or 0)
            shortfall = max(0, min_count - used)
            if shortfall <= 0:
                continue
            penalty = self._target_penalty_amount(target, default=700.0)
            days_left_factor = 1.0 + max(0, now_month - month + 1) * 0.35
            bonus += (penalty + 0.12 * penalty * shortfall) * days_left_factor
        return bonus

    def _current_rest_wait_minutes(self, status: dict[str, Any], pref_policy: dict[str, Any]) -> int | None:
        sim_min = int(status.get("simulation_progress_minutes", 0) or 0)
        for start, end in self._rest_intervals_around(sim_min, sim_min + 1, pref_policy):
            if start <= sim_min < end:
                return max(1, min(_MAX_WAIT_MINUTES, end - sim_min))
        return None

    def _next_useful_wait_minutes(self, sim_min: int, pref_policy: dict[str, Any]) -> int:
        next_rest = None
        for start, end in self._rest_intervals_around(sim_min, sim_min + 24 * 60, pref_policy):
            if sim_min < start:
                next_rest = (start, end)
                break
            if start <= sim_min < end:
                return max(1, end - sim_min)
        if next_rest and next_rest[0] - sim_min <= 240:
            return max(1, min(_MAX_WAIT_MINUTES, next_rest[1] - sim_min))
        return 120

    def _interval_overlaps_forbidden_window(self, start_min: int, end_min: int, pref_policy: dict[str, Any]) -> bool:
        return any(_interval_overlap(start_min, end_min, s, e) for s, e in self._rest_intervals_around(start_min, end_min, pref_policy))

    def _rest_intervals_around(self, start_min: int, end_min: int, pref_policy: dict[str, Any]) -> list[tuple[int, int]]:
        windows = self._rest_windows(pref_policy)
        if not windows:
            return []
        first_day = start_min // 1440 - 1
        last_day = max(first_day, end_min // 1440 + 1)
        intervals: list[tuple[int, int]] = []
        for day in range(first_day, last_day + 1):
            for window in windows:
                if not self._window_applies_to_day(window, day):
                    continue
                sh = _optional_int(window.get("start_hour"), 21)
                eh = _optional_int(window.get("end_hour"), 6)
                s = day * 1440 + sh * 60
                e = day * 1440 + eh * 60
                if e <= s:
                    e += 1440
                intervals.append((s, e))
        intervals.sort()
        return intervals

    def _rest_windows(self, pref_policy: dict[str, Any]) -> list[dict[str, Any]]:
        ir = pref_policy.get("machine_ir") if isinstance(pref_policy.get("machine_ir"), dict) else {}
        raw = ir.get("rest_windows") if isinstance(ir.get("rest_windows"), list) else []
        windows = [w for w in raw if isinstance(w, dict)]
        if windows:
            return windows
        parsed: list[dict[str, Any]] = []
        for item in pref_policy.get("duty_windows") or []:
            if not isinstance(item, dict):
                continue
            text = " ".join(str(item.get(k, "")) for k in ("label", "window", "overnight_note"))
            match = __import__("re").search(r"(\d{1,2}):\d{2}\s*[-至到]\s*(?:次日)?(\d{1,2}):\d{2}", text)
            if not match:
                continue
            days = "all"
            if any(token in text.lower() for token in ("weekend", "周末", "sat", "sun")):
                days = "weekend"
            elif any(token in text.lower() for token in ("weekday", "工作日", "平日", "mon", "fri")):
                days = "weekday"
            parsed.append(
                {
                    "label": str(item.get("label", "")),
                    "days": days,
                    "start_hour": int(match.group(1)),
                    "end_hour": int(match.group(2)),
                    "forbid_take_order": True,
                    "forbid_reposition": True,
                }
            )
        return parsed

    @staticmethod
    def _window_applies_to_day(window: dict[str, Any], day: int) -> bool:
        days = str(window.get("days", "all") or "all").lower()
        weekday = (_SIMULATION_EPOCH + timedelta(days=day)).weekday()
        if "weekend" in days or "周末" in days:
            return weekday >= 5
        if "weekday" in days or "工作日" in days or "平日" in days:
            return weekday < 5
        return True

    @staticmethod
    def _long_haul_limits(pref_policy: dict[str, Any]) -> list[dict[str, Any]]:
        ir = pref_policy.get("machine_ir") if isinstance(pref_policy.get("machine_ir"), dict) else {}
        raw = ir.get("long_haul_limits") if isinstance(ir.get("long_haul_limits"), list) else []
        return [x for x in raw if isinstance(x, dict)]

    @staticmethod
    def _cargo_max_limits(pref_policy: dict[str, Any]) -> list[dict[str, Any]]:
        ir = pref_policy.get("machine_ir") if isinstance(pref_policy.get("machine_ir"), dict) else {}
        raw = ir.get("cargo_max_limits") if isinstance(ir.get("cargo_max_limits"), list) else []
        return [x for x in raw if isinstance(x, dict)]

    @staticmethod
    def _cargo_targets(pref_policy: dict[str, Any]) -> list[dict[str, Any]]:
        ir = pref_policy.get("machine_ir") if isinstance(pref_policy.get("machine_ir"), dict) else {}
        raw = ir.get("cargo_targets") if isinstance(ir.get("cargo_targets"), list) else []
        return [x for x in raw if isinstance(x, dict)]

    @staticmethod
    def _unknown_constraints(pref_policy: dict[str, Any]) -> list[dict[str, Any]]:
        ir = pref_policy.get("machine_ir") if isinstance(pref_policy.get("machine_ir"), dict) else {}
        raw = ir.get("unknown_constraints") if isinstance(ir.get("unknown_constraints"), list) else []
        return [x for x in raw if isinstance(x, dict)]

    @staticmethod
    def _guardian_downgrade_candidates(candidates: list[CandidateFact], guardian: dict[str, Any]) -> list[CandidateFact]:
        blocked: set[str] = set()
        for item in guardian.get("candidate_judgments") or []:
            if not isinstance(item, dict):
                continue
            verdict = str(item.get("verdict") or "").lower()
            cargo_id = str(item.get("id") or "").strip()
            if cargo_id and verdict == "hard_avoid":
                blocked.add(cargo_id)
        if not blocked:
            return candidates
        out: list[CandidateFact] = []
        for cand in candidates:
            if cand.cargo_id in blocked:
                cand.legal = False
                cand.veto_reasons.append("guardian_hard_avoid")
            out.append(cand)
        return out

    @staticmethod
    def _matches_region_avoid(cargo: dict[str, Any], pref_policy: dict[str, Any]) -> bool:
        ir = pref_policy.get("machine_ir") if isinstance(pref_policy.get("machine_ir"), dict) else {}
        rules: list[dict[str, Any]] = []
        for key in ("region_avoid", "origin_avoid"):
            raw = ir.get(key) if isinstance(ir.get(key), list) else []
            rules.extend(x for x in raw if isinstance(x, dict))
        if not rules:
            return False
        start = cargo.get("start") if isinstance(cargo.get("start"), dict) else {}
        end = cargo.get("end") if isinstance(cargo.get("end"), dict) else {}
        start_text = " ".join(str(start.get(k, "") or "") for k in ("province", "city", "district", "address"))
        end_text = " ".join(str(end.get(k, "") or "") for k in ("province", "city", "district", "address"))
        for rule in rules:
            keyword = str(rule.get("keyword") or rule.get("region") or "").strip()
            if not keyword:
                continue
            field = str(rule.get("field") or "origin").lower()
            if field in ("origin", "start") and keyword in start_text:
                return True
            if field in ("destination", "end") and keyword in end_text:
                return True
            if field in ("either", "any", "all") and (keyword in start_text or keyword in end_text):
                return True
        return False

    def _merge_preference_target_memory(
        self, driver_id: str, policy: dict[str, Any], prefs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        ir = policy.get("machine_ir") if isinstance(policy.get("machine_ir"), dict) else {}
        raw_targets = ir.get("cargo_targets") if isinstance(ir.get("cargo_targets"), list) else []
        targets = [dict(t) for t in raw_targets if isinstance(t, dict)]
        for target in targets:
            if "penalty_amount" not in target:
                penalty = self._infer_target_penalty_amount(target, prefs)
                if penalty is not None:
                    target["penalty_amount"] = penalty
        ledger = self._preference_ledger(driver_id)
        memory = [dict(t) for t in self._target_memory_by_driver.get(driver_id, [])]

        current_months = [self._target_month(t) for t in targets]
        current_months = [m for m in current_months if m is not None]
        current_month = max(current_months, default=None)
        if current_month is not None and any(bool(t.get("makeup_from_previous")) for t in targets):
            existing_keys = {(self._target_month(t), str(t.get("cargo_name") or "")) for t in targets}
            counts = ledger.get("cargo_name_counts_by_month") or {}
            for old in memory:
                old_month = self._target_month(old)
                old_name = str(old.get("cargo_name") or "").strip()
                old_min = self._target_min_count(old)
                if old_month is None or not old_name or old_min <= 0 or old_month >= current_month:
                    continue
                month_key = f"2026-{old_month:02d}"
                used = int((counts.get(month_key) or {}).get(old_name, 0) or 0)
                shortfall = max(0, old_min - used)
                key = (current_month, old_name)
                if shortfall > 0 and key not in existing_keys:
                    targets.append(
                        {
                            "month": current_month,
                            "cargo_name": old_name,
                            "min_count": shortfall,
                            "penalty_amount": self._target_penalty_amount(old, default=700.0),
                            "makeup_from_previous": True,
                            "makeup_for_month": old_month,
                        }
                    )
                    existing_keys.add(key)

        merged_memory = memory[:]
        seen_memory = {(self._target_month(t), str(t.get("cargo_name") or "")) for t in merged_memory}
        for target in targets:
            month = self._target_month(target)
            name = str(target.get("cargo_name") or "").strip()
            min_count = self._target_min_count(target)
            if month is None or not name or min_count <= 0:
                continue
            key = (month, name)
            if key not in seen_memory:
                merged_memory.append(
                    {
                        "month": month,
                        "cargo_name": name,
                        "min_count": min_count,
                        "penalty_amount": self._target_penalty_amount(target, default=700.0),
                    }
                )
                seen_memory.add(key)
        self._target_memory_by_driver[driver_id] = merged_memory

        if not isinstance(policy.get("machine_ir"), dict):
            policy["machine_ir"] = {}
        policy["machine_ir"]["cargo_targets"] = targets
        self._logger.info(
            "target_memory driver=%s targets=%s memory=%s",
            driver_id,
            json.dumps(targets, ensure_ascii=False, separators=(",", ":")),
            json.dumps(merged_memory, ensure_ascii=False, separators=(",", ":")),
        )
        return policy

    @staticmethod
    def _target_month(target: dict[str, Any]) -> int | None:
        value = target.get("month")
        if value is None:
            scope = str(target.get("scope", "") or "")
            match = __import__("re").search(r"2026[-年](\d{1,2})|(\d{1,2})月", scope)
            value = (match.group(1) or match.group(2)) if match else None
        try:
            month = int(value)
        except (TypeError, ValueError):
            return None
        return month if 1 <= month <= 12 else None

    @staticmethod
    def _target_min_count(target: dict[str, Any]) -> int:
        for key in ("min_count", "target", "count"):
            try:
                value = int(target.get(key))
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return 0

    @staticmethod
    def _target_penalty_amount(target: dict[str, Any], *, default: float) -> float:
        try:
            value = float(target.get("penalty_amount"))
        except (TypeError, ValueError):
            return default
        if value <= 0:
            return default
        return min(10000.0, value)

    @staticmethod
    def _infer_target_penalty_amount(target: dict[str, Any], prefs: list[dict[str, Any]]) -> float | None:
        name = str(target.get("cargo_name") or "").strip()
        month = ModelDecisionService._target_month(target)
        for pref in prefs:
            if not isinstance(pref, dict):
                continue
            text = str(pref.get("content") or "")
            if name and name not in text:
                continue
            if month is not None and f"{month}月" not in text and f"{month:02d}月" not in text:
                if not any(token in text for token in ("欠额", "补", "指标", "必须", "至少")):
                    continue
            try:
                penalty = float(pref.get("penalty_amount"))
            except (TypeError, ValueError):
                continue
            if penalty > 0:
                return penalty
        return None

    @staticmethod
    def _should_expand_market(first_pass: list[CandidateFact], budget_guard: bool) -> bool:
        if budget_guard:
            return False
        legal = [c for c in first_pass if c.legal]
        if len(first_pass) < 60 or len(legal) < 10:
            return True
        best_net = max((c.net_yuan_before_pref for c in legal), default=0.0)
        best_nph = max((c.net_per_hour_before_pref for c in legal), default=0.0)
        return best_net < 400.0 and best_nph < 45.0

    def _market_scout_council(
        self,
        status: dict[str, Any],
        recent: list[dict[str, Any]],
        first_pass: list[CandidateFact],
        cumulative_tokens: int,
    ) -> dict[str, Any]:
        legal = [c for c in first_pass if c.legal]
        top = self._select_prompt_candidates(first_pass, status, budget_guard=True)
        market_brief = {
            "observed_count": len(first_pass),
            "legal_count": len(legal),
            "best_net": round(max((c.net_yuan_before_pref for c in legal), default=0.0), 1),
            "best_nph": round(max((c.net_per_hour_before_pref for c in legal), default=0.0), 1),
            "top_local_candidates": [self._candidate_prompt(c) for c in top],
            "destination_seeds": [
                {
                    "from_cargo": c.cargo_id,
                    "lat": self._compact_point(c.cargo.get("end"))["lat"] if self._compact_point(c.cargo.get("end")) else None,
                    "lng": self._compact_point(c.cargo.get("end"))["lng"] if self._compact_point(c.cargo.get("end")) else None,
                    "net0": round(c.net_yuan_before_pref, 1),
                    "near_end": c.near_end_cargo_seen,
                    "finish": _wall_text(c.finish_min),
                }
                for c in sorted(legal, key=lambda x: (x.net_yuan_before_pref + 20.0 * x.near_end_cargo_seen), reverse=True)[:8]
            ],
        }
        return self._agent_json(
            "Scout_Planner_Agent",
            (
                "你是 Scout Planner Agent。你不做最终动作，只决定本轮是否需要追加查货。"
                "目标是扩大候选空间、寻找更高净收益和更好后续位置，同时尊重可见 preferences。"
                "只能基于接口已返回的局部市场摘要、候选目的地和历史观测点规划查询；不要写死公开集。"
                "如果当前已经处于必须休息或本地候选足够好，可以输出空 queries。"
                "追加查询会消耗仿真时间，所以每个查询必须有明确经营理由。"
                "若 preferences 里说周末可以晚两个小时再休息，语义是休息开始时间顺延两个小时，"
                "不是把次日结束时间提前或延后；例如 21:00-06:00 加上晚两小时休息，应理解为 23:00-06:00。"
                "反例：周日 06:12 已经不在周六夜间 23:00-06:00 窗口内，不能继续等到 08:00；"
                "周末白天 06:00 之后不因为“晚两个小时”而禁止接单。"
                "若当前距离你解释出的每日休息开始不足约 90 分钟，通常不要追加查询，"
                "因为查询会推迟最终动作并可能把动作起点推进休息窗口。"
            ),
            {
                "state": self._state_summary(status),
                "preferences_visible_now": status.get("preferences") or [],
                "preference_ledger": self._preference_ledger(str(status.get("driver_id", ""))),
                "recent_actions": [self._compact_history(r) for r in recent[-5:]],
                "observed_hot_points": self._observed_points_by_driver.get(str(status.get("driver_id", "")), [])[:12],
                "local_market": market_brief,
                "budget": {
                    "cumulative_tokens": cumulative_tokens,
                    "max_additional_queries": 2,
                    "k_range": "80..160",
                },
                "output_schema": {
                    "state_observer": "brief",
                    "commander_intent": "brief",
                    "queries": [
                        {"lat": "number", "lng": "number", "k": "80..220 integer", "purpose": "brief"}
                    ],
                    "stop_reason": "brief",
                },
            },
        )

    def _executive_council(
        self,
        status: dict[str, Any],
        recent: list[dict[str, Any]],
        cumulative_tokens: int,
        budget_guard: bool,
    ) -> dict[str, Any]:
        return self._agent_json(
            "State_Commander_Executive_Council",
            (
                "你同时扮演 State Observer Agent 和 Commander Agent。"
                "先观察司机状态、偏好、历史节奏，再制定本轮经营意图和风险偏好。"
                "不要规划具体查询点，不要输出最终动作。"
            ),
            {
                "state": self._state_summary(status),
                "preferences_visible_now": status.get("preferences") or [],
                "recent_actions": [self._compact_history(r) for r in recent],
                "observed_points": self._observed_points_by_driver.get(str(status.get("driver_id", "")), [])[:6],
                "budget": {
                    "cumulative_tokens": cumulative_tokens,
                    "driver_limit": 5_000_000,
                    "budget_guard": budget_guard,
                },
                "output_schema": {
                    "state_observer": {"operating_state": "brief", "warnings": []},
                    "commander": {"round_intent": "brief", "risk_level": "low|medium|high", "priority": []},
                },
            },
        )

    def _scout_planner_council(
        self,
        status: dict[str, Any],
        recent: list[dict[str, Any]],
        cumulative_tokens: int,
        budget_guard: bool,
    ) -> dict[str, Any]:
        lat = float(status["current_lat"])
        lng = float(status["current_lng"])
        max_queries = 1 if budget_guard else 2
        max_k = 160 if budget_guard else 360
        return self._agent_json(
            "State_Commander_Scout_Agent",
            (
                "你同时扮演 State Observer Agent、Commander Agent、Scout Planner Agent。"
                "先观察司机状态、偏好、历史节奏，再制定本轮经营意图，并规划 query_cargo 查询点。"
                "查询会消耗仿真时间；不要读取原始数据文件；不要输出最终动作。"
            ),
            {
                "state": self._state_summary(status),
                "preferences_visible_now": status.get("preferences") or [],
                "recent_actions": [self._compact_history(r) for r in recent[-5:]],
                "observed_points": self._observed_points_by_driver.get(str(status.get("driver_id", "")), [])[:10],
                "budget": {
                    "cumulative_tokens": cumulative_tokens,
                    "driver_limit": 5_000_000,
                    "budget_guard": budget_guard,
                    "max_queries": max_queries,
                    "max_k": max_k,
                },
                "rules": [
                    "queries length must be 0..max_queries",
                    "each k must be 20..max_k",
                    "usually query current position when active operation is reasonable",
                    "return zero queries when waiting is clearly preferred from visible state/preferences",
                ],
                "output_schema": {
                    "state_observer": {"operating_state": "brief", "warnings": []},
                    "commander": {"round_intent": "brief", "risk_level": "low|medium|high", "priority": []},
                    "scout_planner": {
                        "queries": [{"lat": lat, "lng": lng, "k": min(220, max_k), "purpose": "brief"}],
                        "reason": "brief",
                    }
                },
            },
        )

    def _full_decision_council(
        self,
        status: dict[str, Any],
        recent: list[dict[str, Any]],
        candidates: list[CandidateFact],
        observed_items: list[dict[str, Any]],
        cumulative_tokens: int,
        budget_guard: bool,
        guardian: dict[str, Any],
        pref_policy: dict[str, Any],
    ) -> dict[str, Any]:
        actionable_ids = self._guardian_actionable_ids(guardian)
        legal_ids = list(actionable_ids) if actionable_ids is not None else [c.cargo_id for c in candidates if c.legal]
        return self._agent_json(
            "Full_MultiAgent_Decision_Council",
            (
                "你同时扮演 State Observer Agent、Commander Agent、Scout Planner Agent、"
                "Cargo Analyst Agents、Preference Interpreter Agent、Legality & Risk Judge Agent、"
                "Future Planner Agent、Debate/Critic Agent、Decision Arbiter Agent。"
                "各角色用极短短语给出判断，再由 Arbiter 最终只输出 take_order、wait、reposition 之一。"
                "没有 fallback、没有 retry；任何非法 JSON、非法 cargo_id、越界 wait 都会让整轮仿真失败。"
                "take_order 只能选择 legal_cargo_ids；wait.duration_minutes 必须是 1..1440 的整数，"
                "reposition 必须给 latitude 和 longitude 数字。不要因为隐藏经验写死司机或公开集偏好。"
                "Preference Interpreter Agent 必须阅读 preferences_visible_now 和 preference_ledger，"
                "把偏好文本解释成本轮应避免或应优先满足的经营约束；含“必须/不得/最多/至少/扣/罚”的偏好优先级高于单笔毛收益。"
                "若偏好文本要求某个每日时段停车、休息、不得接单或不得空驶，Preference Interpreter 必须按整段窗口严格解释："
                "中文“晚两个小时再休息”表示休息开始时间顺延两小时，不表示次日结束时间提前；"
                "例如平日 21:00-06:00，周末晚两小时再休息就是 23:00-06:00，而不是 21:00-01:00 或 23:00-08:00。"
                "周末清晨 06:00 之后已经离开上一夜休息窗口；若任何 Agent 声称周末必须等到 08:00，"
                "Debate/Critic Agent 必须指出这是错误解释并改正。"
                "take_order 或 reposition 的 active_interval 只要与该时段有任意重叠就是高风险；"
                "不能把早上窗口结束后完成、或晚上窗口开始后几分钟完成误判为安全；"
                "take_order 是一个从 active_interval.start 到 active_interval.finish 的原子动作，"
                "其中 pickup、等待装货、运输都不能被拆开插入停车休息；"
                "因此若 active_interval.clock_hours 命中偏好文本禁止的休息时钟段，就必须当作偏好风险。"
                "Critic 必须反驳“跨过休息窗口但完成后再休息即可”的理由；这是错误解释。"
                "如果当前 state.wall_time 本身处在、或可能仍处在偏好文本解释出的休息窗口内，"
                "Arbiter 只能选择 wait 到窗口结束之后再评估；不能因为候选在当天晚间之前完成就提前接单。"
                "跨日期或 days_spanned 大于 1 的 take_order 必须由 Critic 逐日检查所有途经日期的休息窗口，"
                "不能只看最终 finish 时间。"
                "在休息窗口内补足休息必须输出 wait 覆盖到窗口结束。"
                "若偏好文本限制超过某个运输时长的订单数量，优先用 tmin/transport_min 判断，"
                "不要把等待装货时间或去提货时间混同为运输时长，除非偏好文本明确说总占用时间。"
                "若偏好文本说“最多 N 单/不超过 N 单”，第 N 单仍在上限内，只有第 N+1 单及之后才触发超额风险。"
                "候选动作若可能触发偏好风险，必须在 pref_risks 中点名；Arbiter 不应选择明显触发高额偏好罚分的动作。"
                "必须把单司机 92 天、1 小时内运行作为效率约束；Critic 要反对没有明确收益的小等待和重复查询。"
                "Decision Arbiter 输出前必须自检 action 与 params 满足上述动作 API。"
            ),
            {
                "state": self._state_summary(status),
                "preferences_visible_now": status.get("preferences") or [],
                "compiled_preference_policy": self._compact_policy(pref_policy),
                "preference_ledger": self._preference_ledger(str(status.get("driver_id", ""))),
                "preference_guardian_report": guardian,
                "recent_actions": [self._compact_history(r) for r in recent[-6:]],
                "market_observation": {
                    "query_scope": "current_position_api_observation",
                    "observed_items": len(observed_items),
                    "note": "cargo observations are factual API data; strategy and final action are decided by this council",
                },
                "candidate_facts": [self._candidate_prompt(c) for c in candidates],
                "legal_cargo_ids": legal_ids,
                "budget": {
                    "cumulative_tokens": cumulative_tokens,
                    "driver_limit": 5_000_000,
                    "budget_guard": budget_guard,
                    "advice": (
                        "when waiting, usually choose a meaningful duration such as the next operating window "
                        "or 120..1440 minutes; avoid waits under 120 minutes unless there is a precise reason"
                    ),
                },
                "api_constraints": [
                    "legal=false candidates cannot be selected by take_order",
                    "take_order can only select ids listed in preference_guardian_report.actionable_cargo_ids when that list is non-empty",
                    "when preference_guardian_report.must_wait=true, choose wait unless there is an explicitly safer reason in guardian_report",
                    "preference interpretation must come from visible text only",
                    "do not hardcode driver_id or public-set-specific behavior",
                ],
                "allowed_output": {
                    "take_order": {"action": "take_order", "params": {"cargo_id": "from legal_cargo_ids"}},
                    "wait": {"action": "wait", "params": {"duration_minutes": "1..1440"}},
                    "reposition": {"action": "reposition", "params": {"latitude": "number", "longitude": "number"}},
                },
                "output_schema": {
                    "obs": "brief",
                    "cmd": "brief",
                    "scout": "brief",
                    "analyst": [{"id": "...", "r": "brief"}],
                    "pref": "brief",
                    "pref_risks": [],
                    "risk": "brief",
                    "future": "brief",
                    "critic": "brief; must explicitly check current rest-window status and selected active_interval",
                    "api_check": "brief confirmation that action params are valid",
                    "action": "take_order|wait|reposition",
                    "params": {},
                    "reason": "brief",
                },
            },
        )

    def _risk_gated_decision_council(
        self,
        status: dict[str, Any],
        recent: list[dict[str, Any]],
        candidates: list[CandidateFact],
        observed_items: list[dict[str, Any]],
        cumulative_tokens: int,
        budget_guard: bool,
        pref_policy: dict[str, Any],
    ) -> dict[str, Any]:
        legal_ids = [c.cargo_id for c in candidates if c.legal]
        return self._agent_json(
            "Risk_Gated_MultiAgent_Decision_Council",
            (
                "你是合并后的多 Agent 决策委员会，必须在一次 JSON 输出里完成："
                "State Observer、Commander、Cargo Analyst Agents、Preference Guardian、"
                "Legality & Risk Judge、Future Planner、Debate/Critic、Decision Arbiter。"
                "这不是单 Agent；各角色必须先给极短结论，再由 Arbiter 最终输出动作。"
                "最终动作只允许 take_order、wait、reposition。没有 fallback，没有 retry。"
                "Preference Guardian 必须使用 compiled_preference_policy + preference_ledger + candidate_facts "
                "逐个审计候选，先生成 prefer/allow/soft_avoid/hard_avoid。"
                "Arbiter 只能选择 Preference Guardian 标记为 prefer 或 allow 且 legal=true 的 cargo_id。"
                "若所有候选都是 hard_avoid/soft_avoid，优先 wait 到下一可运营窗口，不要硬接。"
                "active_interval 是原子占用区间，从去提货到完成运输，不能拆开插入休息。"
                "任何 take_order/reposition 与 policy 中禁止窗口重叠都必须 hard_avoid。"
                "跨午夜窗口按 policy 的 overnight_note 处理；凌晨属于前一日夜间窗口。"
                "若 policy 限制超过某运输时长的月度数量，用 transport_min 判断，最多 N 单表示第 N 单允许。"
                "若 policy 有月度/日期货类最低目标，Preference Guardian 要把对应货类设为 prefer 或 allow，"
                "但仍不得违反休息窗口和硬性禁止。"
                "收益目标：在偏好安全集合中优先高 net0、高 nph0、低空驶、好后续位置；"
                "不要为微小收益牺牲高额偏好罚分。"
                "输出必须是紧凑 JSON，不要长篇解释。"
            ),
            {
                "state": self._state_summary(status),
                "preferences_visible_now": status.get("preferences") or [],
                "compiled_preference_policy": self._compact_policy(pref_policy),
                "preference_ledger": self._preference_ledger(str(status.get("driver_id", ""))),
                "recent_actions": [self._compact_history(r) for r in recent[-5:]],
                "market_observation": {"observed_items": len(observed_items)},
                "candidate_facts": [self._candidate_prompt(c) for c in candidates],
                "legal_cargo_ids": legal_ids,
                "budget": {
                    "cumulative_tokens": cumulative_tokens,
                    "driver_limit": 5_000_000,
                    "budget_guard": budget_guard,
                    "speed_requirement": "single driver 92 days must finish within 1 hour",
                },
                "allowed_output": {
                    "take_order": {"action": "take_order", "params": {"cargo_id": "legal id judged prefer/allow"}},
                    "wait": {"action": "wait", "params": {"duration_minutes": "1..1440"}},
                    "reposition": {"action": "reposition", "params": {"latitude": "number", "longitude": "number"}},
                },
                "output_schema": {
                    "obs": "short",
                    "cmd": "short",
                    "guardian": {
                        "prefer": ["ids"],
                        "allow": ["ids"],
                        "avoid": [{"id": "id", "why": "short"}],
                        "must_wait": "boolean",
                    },
                    "analyst": [{"id": "id", "r": "short"}],
                    "risk": "short",
                    "future": "short",
                    "critic": "short",
                    "api_check": "short",
                    "action": "take_order|wait|reposition",
                    "params": {},
                    "reason": "short",
                },
            },
        )

    def _observe_current_market(
        self, driver_id: str, status: dict[str, Any], budget_guard: bool
    ) -> list[dict[str, Any]]:
        lat = float(status["current_lat"])
        lng = float(status["current_lng"])
        k = 120 if budget_guard else 220
        resp = self._api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng, k=k)
        batch = resp.get("items") if isinstance(resp.get("items"), list) else []
        self._logger.info("market observation lat=%.4f lng=%.4f k=%s returned=%s", lat, lng, k, len(batch))
        return [item for item in batch if isinstance(item, dict) and isinstance(item.get("cargo"), dict)]

    def _preference_guardian_council(
        self,
        status: dict[str, Any],
        recent: list[dict[str, Any]],
        candidates: list[CandidateFact],
        cumulative_tokens: int,
        pref_policy: dict[str, Any],
    ) -> dict[str, Any]:
        return self._agent_json(
            "Preference_Interpreter_Risk_Guardian_Agent",
            (
                "你是 Preference Interpreter Agent + Legality & Risk Judge Agent。"
                "你不做最终动作，只把候选货源按可见 preferences 的偏好风险分组，供最终 Arbiter 使用。"
                "必须优先使用 compiled_preference_policy；只有 policy 信息不足时才回读原始 preferences。"
                "候选事实里的 active_interval 是从开始去提货到完成运输的原子占用区间，不能拆开插入休息。"
                "如果偏好要求某个时段停车/熄火/休息/不得接单/不得空驶，任何 take_order/reposition "
                "占用区间与该窗口重叠都应 hard_avoid。"
                "跨午夜窗口必须向前归属：例如 23:00-06:00 中的 00:30 属于前一日夜间窗口。"
                "若偏好限制运输时长超过 N 小时的月度数量，用 transport_min 判断，最多 N 单表示第 N 单允许。"
                "若偏好要求某月某货类至少接满 N 单，结合 ledger 的货类计数，给相关货类更高优先级。"
                "不要写死公开集、driver_id 或原始数据；只能依据可见偏好、ledger 和候选事实。"
            ),
            {
                "state": self._state_summary(status),
                "preferences_visible_now": status.get("preferences") or [],
                "compiled_preference_policy": self._compact_policy(pref_policy),
                "preference_ledger": self._preference_ledger(str(status.get("driver_id", ""))),
                "recent_actions": [self._compact_history(r) for r in recent[-6:]],
                "candidate_facts": [self._candidate_prompt(c) for c in candidates],
                "output_schema": {
                    "preference_interpretation": "brief",
                    "monthly_targets": [{"month": "YYYY-MM or unknown", "cargo_name": "string", "progress": "brief"}],
                    "current_window_status": "inside_rest|near_rest_start|free_to_act|unclear",
                    "must_wait": "boolean",
                    "recommended_wait_minutes": "integer 1..1440 or null",
                    "candidate_judgments": [
                        {
                            "id": "cargo id",
                            "verdict": "prefer|allow|soft_avoid|hard_avoid",
                            "risk": "brief",
                            "preference_value": "brief",
                        }
                    ],
                    "actionable_cargo_ids": ["ids with prefer or allow verdict"],
                    "hard_avoid_cargo_ids": ["ids with hard_avoid verdict"],
                    "critic_note": "brief",
                },
            },
        )

    @staticmethod
    def _guardian_actionable_ids(guardian: dict[str, Any]) -> set[str] | None:
        raw = guardian.get("actionable_cargo_ids")
        if not isinstance(raw, list):
            return None
        return {str(item).strip() for item in raw if str(item).strip()}

    @staticmethod
    def _compact_policy(policy: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in policy.items() if k != "_signature"}

    def _run_market_scout_queries(
        self, driver_id: str, status: dict[str, Any], planning: dict[str, Any]
    ) -> list[dict[str, Any]]:
        queries = planning.get("queries") if isinstance(planning.get("queries"), list) else []
        items: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        for query in queries[:2]:
            if not isinstance(query, dict):
                continue
            try:
                lat = float(query.get("lat", query.get("latitude")))
                lng = float(query.get("lng", query.get("longitude")))
                k = max(80, min(160, int(query.get("k", 120) or 120)))
            except (TypeError, ValueError):
                continue
            key = (round(lat * 1000), round(lng * 1000))
            if key in seen:
                continue
            seen.add(key)
            resp = self._api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng, k=k)
            batch = resp.get("items") if isinstance(resp.get("items"), list) else []
            self._logger.info(
                "market scout query lat=%.4f lng=%.4f k=%s returned=%s purpose=%s",
                lat,
                lng,
                k,
                len(batch),
                query.get("purpose"),
            )
            items.extend(item for item in batch if isinstance(item, dict) and isinstance(item.get("cargo"), dict))
        return items

    def _run_deterministic_market_scout_queries(
        self, driver_id: str, status: dict[str, Any], first_pass: list[CandidateFact], budget_guard: bool
    ) -> list[dict[str, Any]]:
        if budget_guard:
            return []
        seeds: list[tuple[float, float, str]] = []
        for cand in sorted(first_pass, key=lambda c: (c.legal, c.net_yuan_before_pref, c.near_end_cargo_seen), reverse=True)[:4]:
            for key in ("start", "end"):
                point = self._compact_point(cand.cargo.get(key))
                if point:
                    seeds.append((float(point["lat"]), float(point["lng"]), f"{cand.cargo_id}:{key}"))
        for point in self._observed_points_by_driver.get(str(status.get("driver_id", "")), [])[:4]:
            seeds.append((float(point["lat"]), float(point["lng"]), "memory_hotspot"))

        items: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        for lat, lng, purpose in seeds[:2]:
            key = (round(lat * 1000), round(lng * 1000))
            if key in seen:
                continue
            seen.add(key)
            resp = self._api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng, k=120)
            batch = resp.get("items") if isinstance(resp.get("items"), list) else []
            self._logger.info(
                "deterministic scout query lat=%.4f lng=%.4f k=120 returned=%s purpose=%s",
                lat,
                lng,
                len(batch),
                purpose,
            )
            items.extend(item for item in batch if isinstance(item, dict) and isinstance(item.get("cargo"), dict))
        return items

    def _agent_json(self, agent_name: str, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = {
            "enable_thinking": self._enable_thinking,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        system
                        + " 只输出一个 JSON 对象；不要 Markdown；不要 JSON 外解释；"
                        + "不得建议读取或解析任何原始数据文件。"
                    ),
                },
                {"role": "user", "content": json.dumps({"agent_name": agent_name, **payload}, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
        }
        resp = self._api.model_chat_completion(request)
        choices = resp.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"{agent_name} missing choices")
        content = choices[0].get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{agent_name} empty content")
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError(f"{agent_name} did not return object")
        self._logger.info(
            "agent_council_json agent=%s data=%s",
            agent_name,
            json.dumps(data, ensure_ascii=False, separators=(",", ":")),
        )
        return data

    def _run_scout_queries(
        self,
        driver_id: str,
        status: dict[str, Any],
        planning: dict[str, Any],
        budget_guard: bool,
    ) -> list[dict[str, Any]]:
        current_lat = float(status["current_lat"])
        current_lng = float(status["current_lng"])
        scout = planning.get("scout_planner") if isinstance(planning.get("scout_planner"), dict) else {}
        queries = scout.get("queries") if isinstance(scout.get("queries"), list) else []
        max_queries = 1 if budget_guard else 2
        max_k = 120 if budget_guard else 260
        items: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        for query in queries[:max_queries]:
            if not isinstance(query, dict):
                continue
            try:
                lat = float(query.get("lat", query.get("latitude", current_lat)))
                lng = float(query.get("lng", query.get("longitude", current_lng)))
                k = max(20, min(max_k, int(query.get("k", 140) or 140)))
            except (TypeError, ValueError):
                continue
            key = (round(lat * 1000), round(lng * 1000))
            if key in seen:
                continue
            seen.add(key)
            resp = self._api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng, k=k)
            batch = resp.get("items") if isinstance(resp.get("items"), list) else []
            self._logger.info("scout query lat=%.4f lng=%.4f k=%s returned=%s", lat, lng, k, len(batch))
            items.extend(item for item in batch if isinstance(item, dict) and isinstance(item.get("cargo"), dict))
        return items

    def _build_candidate_facts(
        self, status: dict[str, Any], items: list[dict[str, Any]], *, source: str
    ) -> list[CandidateFact]:
        sim_min = int(status.get("simulation_progress_minutes", 0) or 0)
        current_lat = float(status["current_lat"])
        current_lng = float(status["current_lng"])
        cost_per_km = float(status.get("cost_per_km", _DEFAULT_COST_PER_KM) or _DEFAULT_COST_PER_KM)
        speed_kmph = float(status.get("reposition_speed_km_per_hour", _DEFAULT_SPEED_KMPH) or _DEFAULT_SPEED_KMPH)
        seen: set[str] = set()
        facts: list[CandidateFact] = []
        for item in items:
            cargo = item.get("cargo") or {}
            cargo_id = str(cargo.get("cargo_id", "") or "").strip()
            if not cargo_id or cargo_id in seen:
                continue
            seen.add(cargo_id)
            try:
                start_lat, start_lng = _cargo_point(cargo, "start")
                end_lat, end_lng = _cargo_point(cargo, "end")
                pickup_km = _haversine_km(current_lat, current_lng, start_lat, start_lng)
                haul_km = _haversine_km(start_lat, start_lng, end_lat, end_lng)
                pickup_min = _distance_minutes(pickup_km, speed_kmph) if pickup_km > 1e-6 else 0
                arrival_min = sim_min + pickup_min
                wait_min = self._load_wait_minutes(cargo, arrival_min)
                transport_min = int(cargo.get("cost_time_minutes", 0) or 0)
                finish_min = arrival_min + wait_min + transport_min
                legal, vetoes = self._legality(cargo, arrival_min, finish_min)
                price = _cargo_price_yuan(cargo)
                cost = (pickup_km + haul_km) * cost_per_km
                active_min = max(1, pickup_min + wait_min + transport_min)
                net = price - cost
                facts.append(
                    CandidateFact(
                        cargo_id=cargo_id,
                        cargo=cargo,
                        source=source,
                        query_km=float(item.get("distance_km", pickup_km) or pickup_km),
                        pickup_km=pickup_km,
                        haul_km=haul_km,
                        pickup_min=pickup_min,
                        wait_min=wait_min,
                        transport_min=transport_min,
                        finish_min=finish_min,
                        price_yuan=price,
                        cost_yuan=cost,
                        net_yuan_before_pref=net,
                        net_per_hour_before_pref=net / (active_min / 60.0),
                        near_end_cargo_seen=self._observed_next_count(cargo, items),
                        legal=legal,
                        veto_reasons=vetoes,
                    )
                )
            except Exception as exc:
                self._logger.debug("candidate skipped cargo_id=%s error=%s", cargo_id, exc)
        return facts

    def _select_prompt_candidates(
        self, facts: list[CandidateFact], status: dict[str, Any], budget_guard: bool
    ) -> list[CandidateFact]:
        limit = 7 if budget_guard else 10
        by_id: dict[str, CandidateFact] = {}
        selected: list[CandidateFact] = []
        pref_text = " ".join(
            str(p.get("content", p)) if isinstance(p, dict) else str(p)
            for p in (status.get("preferences") or [])
        )
        legal_facts = [c for c in facts if c.legal]
        pref_hits = [
            c for c in legal_facts if self._cargo_name(c.cargo) and self._cargo_name(c.cargo) in pref_text
        ]

        def add_bucket(bucket: list[CandidateFact], quota: int) -> None:
            added = 0
            for cand in bucket:
                if cand.cargo_id in by_id:
                    continue
                by_id[cand.cargo_id] = cand
                selected.append(cand)
                added += 1
                if added >= quota or len(selected) >= limit:
                    return

        add_bucket(sorted(pref_hits, key=lambda c: c.net_yuan_before_pref, reverse=True), 4 if not budget_guard else 2)
        add_bucket(sorted(legal_facts, key=lambda c: c.net_yuan_before_pref, reverse=True), 3)
        add_bucket(sorted(legal_facts, key=lambda c: c.net_per_hour_before_pref, reverse=True), 2)
        add_bucket(sorted(legal_facts, key=lambda c: c.near_end_cargo_seen, reverse=True), 1)
        add_bucket(sorted(legal_facts, key=lambda c: c.pickup_min + c.wait_min + c.transport_min), 1)
        add_bucket(sorted(legal_facts, key=lambda c: c.query_km), 1)

        if len(selected) < limit:
            add_bucket(
                sorted(facts, key=lambda c: (not c.legal, -c.net_yuan_before_pref, -c.net_per_hour_before_pref, c.query_km)),
                limit - len(selected),
            )
        return selected[:limit]

    def _candidate_prompt(self, cand: CandidateFact) -> dict[str, Any]:
        cargo_name = self._cargo_name(cand.cargo)
        active_start = cand.finish_min - cand.pickup_min - cand.wait_min - cand.transport_min
        active_dates = sorted(
            {
                (_SIMULATION_EPOCH + timedelta(minutes=m)).strftime("%Y-%m-%d")
                for m in (active_start, max(active_start, cand.finish_min - 1))
            }
        )
        if cand.finish_min - active_start > 24 * 60:
            start_day = active_start // 1440
            finish_day = max(active_start, cand.finish_min - 1) // 1440
            active_dates = [
                (_SIMULATION_EPOCH + timedelta(days=day)).strftime("%Y-%m-%d")
                for day in range(start_day, finish_day + 1)
            ]
        return {
            "id": cand.cargo_id,
            "name": cargo_name,
            "legal": cand.legal,
            "veto": cand.veto_reasons,
            "price": round(cand.price_yuan, 1),
            "net0": round(cand.net_yuan_before_pref, 1),
            "nph0": round(cand.net_per_hour_before_pref, 1),
            "qkm": round(cand.query_km, 1),
            "pkm": round(cand.pickup_km, 1),
            "hkm": round(cand.haul_km, 1),
            "pmin": cand.pickup_min,
            "wmin": cand.wait_min,
            "tmin": cand.transport_min,
            "transport_min": cand.transport_min,
            "transport_gt_8h": cand.transport_min > 480,
            "active_min": cand.pickup_min + cand.wait_min + cand.transport_min,
            "finish": _wall_text(cand.finish_min),
            "active_interval": {
                "start": _wall_text(active_start),
                "finish": _wall_text(cand.finish_min),
                "minutes": cand.pickup_min + cand.wait_min + cand.transport_min,
                "days_spanned": max(1, (max(active_start, cand.finish_min - 1) // 1440) - (active_start // 1440) + 1),
                "active_dates": active_dates,
                "transport_minutes": cand.transport_min,
                "pickup_minutes": cand.pickup_min,
                "load_wait_minutes": cand.wait_min,
                "month": _month_key(cand.finish_min),
                "finish_hour": (cand.finish_min // 60) % 24,
                "clock_hours": _covered_clock_hours(active_start, cand.finish_min),
            },
            "near_end": cand.near_end_cargo_seen,
            "load_time": cand.cargo.get("load_time"),
            "start": self._compact_point(cand.cargo.get("start")),
            "end": self._compact_point(cand.cargo.get("end")),
        }

    @staticmethod
    def _cargo_name(cargo: dict[str, Any]) -> str:
        for key in ("cargo_name", "category", "cargo_type", "goods_type", "name"):
            value = str(cargo.get(key, "") or "").strip()
            if value:
                return value
        return ""

    @staticmethod
    def _compact_point(point: Any) -> dict[str, float] | None:
        if not isinstance(point, dict):
            return None
        try:
            return {"lat": round(float(point["lat"]), 3), "lng": round(float(point["lng"]), 3)}
        except (KeyError, TypeError, ValueError):
            return None

    def _load_wait_minutes(self, cargo: dict[str, Any], arrival_min: int) -> int:
        raw = cargo.get("load_time")
        if not isinstance(raw, list) or len(raw) != 2:
            return 0
        start = _parse_wall_minutes(str(raw[0]))
        return 0 if start is None else max(0, start - arrival_min)

    def _legality(self, cargo: dict[str, Any], arrival_min: int, finish_min: int) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        raw = cargo.get("load_time")
        if isinstance(raw, list) and len(raw) == 2:
            end = _parse_wall_minutes(str(raw[1]))
            if end is not None and arrival_min > end:
                reasons.append("late_load")
        remove_min = _parse_wall_minutes(str(cargo.get("remove_time", "") or ""))
        if remove_min is not None and arrival_min > remove_min:
            reasons.append("removed")
        if finish_min > _HORIZON_MINUTES:
            reasons.append("after_horizon")
        if int(cargo.get("cost_time_minutes", 0) or 0) <= 0:
            reasons.append("bad_duration")
        return not reasons, reasons

    def _observed_next_count(self, cargo: dict[str, Any], items: list[dict[str, Any]]) -> int:
        try:
            end_lat, end_lng = _cargo_point(cargo, "end")
        except Exception:
            return 0
        count = 0
        for item in items:
            other = item.get("cargo") or {}
            if other.get("cargo_id") == cargo.get("cargo_id"):
                continue
            try:
                start_lat, start_lng = _cargo_point(other, "start")
            except Exception:
                continue
            if _haversine_km(end_lat, end_lng, start_lat, start_lng) <= 120:
                count += 1
        return count

    def _validate_action_or_raise(
        self, action: dict[str, Any], candidates: list[CandidateFact], actionable_ids: set[str] | None = None
    ) -> dict[str, Any]:
        if not isinstance(action, dict):
            raise ValueError("LLM action must be a JSON object")
        name = str(action.get("action", "")).strip().lower()
        params = action.get("params") if isinstance(action.get("params"), dict) else {}
        legal_ids = {c.cargo_id for c in candidates if c.legal}
        if name == "take_order":
            cargo_id = str(params.get("cargo_id", "")).strip()
            if actionable_ids is not None and cargo_id not in actionable_ids:
                raise ValueError(f"LLM selected cargo_id outside Preference Guardian actionable set: {cargo_id!r}")
            if cargo_id in legal_ids:
                return {"action": "take_order", "params": {"cargo_id": cargo_id}}
            raise ValueError(f"LLM selected illegal or unobserved cargo_id={cargo_id!r}")
        if name == "wait":
            try:
                duration = int(params["duration_minutes"])
            except (TypeError, ValueError):
                raise ValueError(f"LLM wait duration must be an integer: {params!r}") from None
            except KeyError:
                raise ValueError("LLM wait action missing duration_minutes") from None
            if not 1 <= duration <= _MAX_WAIT_MINUTES:
                raise ValueError(f"LLM wait duration out of range: {duration}")
            return {"action": "wait", "params": {"duration_minutes": duration}}
        if name == "reposition":
            try:
                return {
                    "action": "reposition",
                    "params": {"latitude": float(params["latitude"]), "longitude": float(params["longitude"])},
                }
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"LLM reposition action has invalid params: {params!r}") from None
        raise ValueError(f"LLM returned unsupported action={name!r}")

    def _remember_chosen_action(self, driver_id: str, action: dict[str, Any], candidates: list[CandidateFact]) -> None:
        if action.get("action") != "take_order":
            return
        cargo_id = str((action.get("params") or {}).get("cargo_id", "")).strip()
        cand = next((item for item in candidates if item.cargo_id == cargo_id), None)
        if cand is None:
            return
        active_start = cand.finish_min - cand.pickup_min - cand.wait_min - cand.transport_min
        self._chosen_orders_by_driver.setdefault(driver_id, []).append(
            {
                "cargo_id": cargo_id,
                "name": self._cargo_name(cand.cargo) or "unknown",
                "month": _month_key(cand.finish_min),
                "active_start": active_start,
                "finish": cand.finish_min,
                "active_duration_min": cand.pickup_min + cand.wait_min + cand.transport_min,
                "transport_min": cand.transport_min,
                "pickup_min": cand.pickup_min,
                "load_wait_min": cand.wait_min,
            }
        )

    def _preference_ledger(self, driver_id: str) -> dict[str, Any]:
        orders = self._chosen_orders_by_driver.get(driver_id, [])
        name_counts: dict[str, dict[str, int]] = {}
        active_duration_bins: dict[str, dict[str, int]] = {}
        transport_duration_bins: dict[str, dict[str, int]] = {}
        active_clock_hours: dict[str, int] = {}
        for order in orders:
            month = str(order.get("month") or "")
            name = str(order.get("name") or "unknown")
            name_counts.setdefault(month, {})
            name_counts[month][name] = name_counts[month].get(name, 0) + 1

            active_duration = int(order.get("active_duration_min", 0) or 0)
            active_bins = active_duration_bins.setdefault(month, {"over_4h": 0, "over_8h": 0, "over_12h": 0})
            if active_duration > 240:
                active_bins["over_4h"] += 1
            if active_duration > 480:
                active_bins["over_8h"] += 1
            if active_duration > 720:
                active_bins["over_12h"] += 1

            transport_duration = int(order.get("transport_min", 0) or 0)
            transport_bins = transport_duration_bins.setdefault(month, {"over_4h": 0, "over_8h": 0, "over_12h": 0})
            if transport_duration > 240:
                transport_bins["over_4h"] += 1
            if transport_duration > 480:
                transport_bins["over_8h"] += 1
            if transport_duration > 720:
                transport_bins["over_12h"] += 1

            for hour in _covered_clock_hours(int(order["active_start"]), int(order["finish"])):
                key = f"{hour:02d}:00"
                active_clock_hours[key] = active_clock_hours.get(key, 0) + 1
        return {
            "note": "factual memory of previous LLM-selected orders in this run; use only to interpret visible text preferences",
            "orders_total": len(orders),
            "cargo_name_counts_by_month": name_counts,
            "transport_duration_bins_by_month": transport_duration_bins,
            "active_duration_bins_by_month": active_duration_bins,
            "active_clock_hour_counts": active_clock_hours,
        }

    def _state_summary(self, status: dict[str, Any]) -> dict[str, Any]:
        sim_min = int(status.get("simulation_progress_minutes", 0) or 0)
        return {
            "driver_id": status.get("driver_id"),
            "sim_min": sim_min,
            "wall_time": status.get("simulation_wall_time") or _wall_text(sim_min),
            "month": _month_key(sim_min),
            "hour": (sim_min // 60) % 24,
            "weekday": _weekday_text(sim_min),
            "remaining_min": max(0, _HORIZON_MINUTES - sim_min),
            "lat": round(float(status.get("current_lat", 0.0)), 4),
            "lng": round(float(status.get("current_lng", 0.0)), 4),
            "truck_length": status.get("truck_length"),
            "completed": status.get("completed_order_count"),
        }

    @staticmethod
    def _compact_history(record: dict[str, Any]) -> dict[str, Any]:
        action = record.get("action") if isinstance(record.get("action"), dict) else {}
        result = record.get("result") if isinstance(record.get("result"), dict) else {}
        return {
            "s": record.get("step"),
            "a": action.get("action"),
            "p": action.get("params"),
            "elapsed": record.get("step_elapsed_minutes"),
            "accepted": result.get("accepted"),
            "end": record.get("simulation_end_time"),
        }

    @staticmethod
    def _history_tokens(records: list[dict[str, Any]]) -> int:
        total = 0
        for record in records:
            usage = record.get("token_usage")
            if isinstance(usage, dict):
                total += int(usage.get("total_tokens", 0) or 0)
        return total

    def _remember_observed_points(self, driver_id: str, items: list[dict[str, Any]]) -> None:
        points = self._observed_points_by_driver.setdefault(driver_id, [])
        for item in items:
            cargo = item.get("cargo") or {}
            try:
                start_lat, start_lng = _cargo_point(cargo, "start")
                price = _cargo_price_yuan(cargo)
            except Exception:
                continue
            if any(_haversine_km(start_lat, start_lng, p["lat"], p["lng"]) < 25 for p in points):
                continue
            points.append({"lat": round(start_lat, 4), "lng": round(start_lng, 4), "price": round(price, 1)})
        points.sort(key=lambda p: p["price"], reverse=True)
        del points[20:]
