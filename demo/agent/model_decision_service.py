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
_PRE_QUERY_SAFETY_MIN = 5  # 通用安全余量: 查货扫描预估耗时之外再留几分钟,防扫描尾巴擦进禁动窗头
_VISIT_DEFAULT_RADIUS_KM = 3.0   # 通用默认: 文本未给半径时,坐标版"进点才算到达"的小半径
_VISIT_SCOUT_KM = 120.0          # 通用外圈: 距目标此距离内仅弱引导/定向scout,绝不当达标(红线)
_VISIT_WEAK_BONUS = 150.0        # 通用弱引导分值(远小于真达标bonus,且不计入visit计数)
_VISIT_MONTHEND_BUFFER_DAYS = 2  # 通用缓冲: 月末强化真达标候选的剩余天数阈值

# ===== K底Z甲 feature flags(形状参数,零偏好常量;全部默认=v19原行为,逐项验证后开启) =====
FEATURE_FLAGS = {
    "compile_temperature0": True,   # Day1: 编译temperature=0+单次重试
    "rest_window_strict": True,     # Day1: 删默认兜底,五条件验证不满足→unknown
    "compile_audit": True,          # Day1: 编译后本地audit九项+冲突剥离+family去重
    "field_voting": False,           # Day1: 字段级3票(period/数量5票),不一致→降级unknown
    "dynamic_longhaul": True,       # Day1: ledger存transport_min列表,按limit阈值现算
    "snap_vocab": True,             # Day1: 品类等值snap词表(三处匹配点)
    "council_v2": True,             # Day2: guardian→Council(top10对齐/±800限幅/must_wait/need_scout)
    "monthend_scout": True,         # Day3: 月末安全广查闸门
    "subgrad_shadow": True,         # Day3: 次梯度影子价格(关=0.12常数)
    "pre_query_rest_guard": True,   # P0-3: 查货前若扫描会跨进禁动窗头→提前wait睡穿(防作息禁动型漏罚)
    "location_visit": True,         # 目标点打卡执行器(getter+visit_days按天去重+真达标bonus);死字段补全
    "recompile_stable_merge": True, # 跨月重编译保护: 稳定约束字段(作息/整歇/区域/长途/打卡)只增不减,防重编译非确定丢窗
    "ltd_reposition": False,         # Day4: 受限reposition(三重闸门)
    "dest_value": False,             # Day4: V落点价值表
    "aggressive_fulfill": False,     # 实验: 激进履约(配速驱动)——落后线性配速即λ→1抢配额+提早定向广查;默认关=保守K底Z甲
}



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
    # query_cargo 接口返回的 price 已是元(官方文档+全量数据实测确认)。
    # 旧守卫(>10000才/100)会把全数据集仅有的49条万元级真高价单误除100倍——直接信任接口。
    return float(cargo.get("price", 0.0) or 0.0)


_PUNCT = "，。；：、！？「」『』“”‘’（）()【】《》—-…·,.;:!?\"'"


def _norm_text(s: str) -> str:
    # 去空格+去中英文标点: LLM 摘录 source_quote/preference_text 时常省略或改写标点,
    # 子串匹配必须标点无关(否则"没完成，五月"vs"没完成五月"误判为不匹配)。
    out = "".join(str(s).split())
    for ch in _PUNCT:
        out = out.replace(ch, "")
    return out


def _snap_category(kw: str, known: set[str]) -> tuple[str, str]:
    """编译出的品类关键词→运行时观测词表唯一对齐(评分按cargo_name完全相等计数)。
    exact=完整品类名; snapped=唯一子串/超串; unmapped=无法唯一对齐(调用方降级)。零硬编码。"""
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
    llm_adjustment: float = 0.0  # Council调分(限幅±800): LLM影响有界,代码裁决


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
        # 运行时见过的真实品类名全集(query返回累积,零硬编码)——供编译audit的品类词表校验/snap对齐
        self._seen_cargo_names_by_driver: dict[str, set[str]] = {}
        self._last_reposition_day: dict[str, int] = {}

    def decide(self, driver_id: str) -> dict[str, Any]:
        self._cur_driver_id = driver_id  # 供打分层回查该司机的观测热点(影子价格机会数估计)
        status = self._api.get_driver_status(driver_id)
        all_history = self._api.query_decision_history(driver_id, -1)
        records = all_history.get("records") if isinstance(all_history.get("records"), list) else []
        recent = records[-8:]
        cumulative_tokens = self._history_tokens(records)
        budget_guard = cumulative_tokens >= _TOKEN_SOFT_LIMIT
        pref_policy = self._compiled_preference_policy(status, recent)

        off_wait = self._forced_off_day_wait(driver_id, status, pref_policy)
        if off_wait is not None:
            self._logger.info(
                "deterministic off-day driver=%s sim_min=%s wait=%s",
                driver_id, status.get("simulation_progress_minutes"), off_wait,
            )
            return {"action": "wait", "params": {"duration_minutes": off_wait}}

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

        pre_q_wait = self._pre_query_rest_guard(status, pref_policy, budget_guard)
        if pre_q_wait is not None:
            self._logger.info(
                "deterministic pre-query rest guard driver=%s sim_min=%s wait=%s (睡穿,防扫描跨进禁动窗头)",
                driver_id, status.get("simulation_progress_minutes"), pre_q_wait,
            )
            return {"action": "wait", "params": {"duration_minutes": pre_q_wait}}

        items = self._observe_current_market(driver_id, status, budget_guard)
        status_after_query = self._api.get_driver_status(driver_id)
        self._remember_observed_points(driver_id, items)
        first_pass = self._build_candidate_facts(status_after_query, items, source="local")
        expansion_plan = None
        sim_min_q = int(status_after_query.get("simulation_progress_minutes", 0) or 0)
        shortfall_cats = self._monthend_shortfall_targets(pref_policy, driver_id, sim_min_q)
        if self._should_expand_market(first_pass, budget_guard) or shortfall_cats:
            extra_items = self._run_deterministic_market_scout_queries(
                driver_id, status_after_query, first_pass, budget_guard, shortfall_cats
            )
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
        # 仅【high risk 未知约束】触发 Council——high=需要 agent 额外行动才能满足(每日量/回家/
        # 在线时长/区域累计);low/medium=对已编译约束的描述补充(罚款金额/计数口径),不必每步调
        # Council(治墙钟+token:全描述性 unknown 的司机走零LLM快车道)。比文本子串匹配鲁棒。
        high_unknowns = [
            u for u in self._unknown_constraints(pref_policy)
            if str(u.get("risk_level", "")).lower() == "high"
        ]
        if high_unknowns and candidates:
            if FEATURE_FLAGS.get("council_v2"):
                # Council v2: 审【按确定性score排序的top-10】(与argmax对齐,堵"argmax选中
                # guardian没见过的候选"缺口);未审候选在守护激活时不参与argmax(宁wait不接未审单)
                sim_min_now = int(status_after_query.get("simulation_progress_minutes", 0) or 0)
                ledger_now = self._preference_ledger(driver_id)
                ranked = sorted(
                    (c for c in all_candidates if c.legal and not self._deterministic_vetoes(c, pref_policy, ledger_now)),
                    key=lambda c: self._candidate_score(c, pref_policy, ledger_now, sim_min_now),
                    reverse=True,
                )
                review_set = ranked[:10]
                guardian = self._preference_guardian_council(
                    status_after_query, recent, review_set, cumulative_tokens, pref_policy
                )
                all_candidates = self._apply_council_judgments(
                    all_candidates, review_set, guardian, driver_id=driver_id, sim_min=sim_min_now
                )
                mw = guardian.get("must_wait")
                if mw is True:
                    rw = _optional_int(guardian.get("recommended_wait_minutes"), 120)
                    dur = max(30, min(240, rw))  # 钳位; 确定性rest在decide入口已优先,不会被本分支覆盖
                    self._logger.info("council must_wait driver=%s wait=%s", driver_id, dur)
                    return {"action": "wait", "params": {"duration_minutes": dur}}
            else:
                guardian = self._preference_guardian_council(
                    status_after_query, recent, candidates, cumulative_tokens, pref_policy
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
        policy = self._compile_once_or_vote(
            "Preference_Compiler_Agent",
            (
                "你是一次性 Preference Multi-Agent Compiler。请在同一次回答中完成三个子 Agent 的工作："
                "1) Preference Parser Agent 抽取休息、限额、最低指标；"
                "2) Target Continuity Auditor Agent 专门检查跨月欠额、补上月、延续指标、累计/重置口径；"
                "3) Risk Compiler Agent 输出可执行 machine_ir。"
                "最终只输出稳定、简短的经营 policy，供后续 Duty、Guardian、Arbiter 多 Agent 共用。"
                "不要做最终动作，不要评价具体货源。"
                "必须泛化到隐藏司机：只根据文本抽取约束，不要写死司机、月份、货类或公开集。"
                "若文本有跨午夜时段，凌晨部分归属前一日夜间窗口：窗口 A点至次日B点 中位于午夜后的时刻，"
                "属于【前一日A点起】的那一夜，不是当日的。"
                "若文本说【晚 N 小时再休息】，表示休息开始时间顺延 N 小时，结束时间不自动顺延："
                "原窗 A点至B点 → 新窗 (A+N)点至B点；严禁输出 (A+N)点至(B+N)点，"
                "除非文本明确说晚 N 小时起床/晚 N 小时结束/休息结束顺延。"
                "数字示例(虚构,仅示意规则): 19:00-05:00 + 晚一个小时再休息 = 20:00-05:00。"
                "【时长型作息‼️】若文本是'每天至少连续休息N小时/连续停车熄火满N小时'(只给时长、不给具体钟点),"
                "【严禁】输出 start_hour==end_hour 的占位窗(会被执行成整天休息=毛利归零灾难)。官方按【单个日历日内"
                "(00:00-24:00)最长连续休息≥N小时】判定——跨午夜的窗会被午夜切成两段、两天都不足N→必须落成一个"
                "【不跨午夜、同一日内的N小时窗】: start_hour=0, end_hour=N, days=all (00:00至N点,清晨低价值时段)。"
                "例: 每天连续休息8小时→start_hour=0,end_hour=8。"
                "每条约束必须输出 source_quote 字段=依据的偏好原文片段(逐字摘录,供审计回查)。"
                "若原文暗示但未明说某约束(隐式约束),也要列出并标 low confidence,放入 unknown_constraints。"
                "unknown_constraints 的 risk_level 严格区分:【需要 agent 采取额外行动才能满足】"
                "(如每天至少 N 单、每隔 X 天回家、每天在线 X 小时)填 high;"
                "只是对【已编译约束】的补充说明(罚款金额/周末定义/比上月罚得重/计数口径)填 low。"
                "【绝对禁止】把已经编进 rest_windows/cargo_targets/long_haul_limits/cargo_max_limits/"
                "daily_order_caps/off_day_requirements/location_visit_targets/region_avoid 任一字段的偏好,再重复放进 unknown_constraints——那会导致同一约束被"
                "结构化执行器和守护层重复处理。unknown_constraints 只放【上述字段都无法表达】的偏好。"
                "【周期严格匹配】cargo_targets/cargo_max_limits/long_haul_limits 都是【按月】计数的槽位。"
                "【每日接单数上限】(每天/一天/每日 最多接 N 单)→ 编进 machine_ir.daily_order_caps(max_per_day=N),"
                "由确定性执行器按日计数拦截,不要放进月度槽位、也不要放 unknown_constraints。"
                "【整天歇车】(每月至少N整天不出车/歇车/完全不出工/那天别排活的整天) → 编进 "
                "machine_ir.off_day_requirements(min_off_days=N),由确定性执行器月末预留整天,不要放 unknown_constraints。"
                "【每月至少N天到达目标点】(到某地累计/打卡/每月至少N天去某固定地点或坐标) → 编进 "
                "machine_ir.location_visit_targets: 文本给经纬度→填 target_lat/target_lng(radius_km 未给则留空,执行层补通用默认);"
                "文本只给地名→填 keyword(留空 target_lat/target_lng); min_days=该月需到达的不同【天】数(按天去重,非次数);"
                "month=计分归属月。由确定性执行器逐月按天打卡+月末兜底导向,不要放 unknown_constraints、也不要塞进 cargo_targets(那是按品类计数)。"
                "其余非月度周期(每天【至少】N单、每周、每隔 X 天回家 等)schema 无对应槽位——"
                "严禁硬塞进月度槽位(每日≠每月,塞错会灾难性执行),必须整条放入 unknown_constraints 并注明真实周期。"
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
                        "daily_order_caps": [
                            {
                                "max_per_day": "integer (每天/一天/每日 最多接 N 单 → N)",
                                "penalty_amount": "number if visible",
                                "source_quote": "exact text",
                            }
                        ],
                        "off_day_requirements": [
                            {
                                "min_off_days": "integer (每月至少N整天不出车/歇车/完全不出工 → N)",
                                "penalty_amount": "number if visible",
                                "source_quote": "exact text",
                            }
                        ],
                        "location_visit_targets": [
                            {
                                "month": "1..12 integer (计分归属月)",
                                "min_days": "integer (该月需到达目标点的不同天数, 按天去重非次数)",
                                "target_lat": "latitude number if text gives coordinates, else null",
                                "target_lng": "longitude number if text gives coordinates, else null",
                                "radius_km": "number if visible, else null (执行层补通用默认小半径)",
                                "keyword": "location/region text if only place name given, else null",
                                "penalty_amount": "number if visible (整月不满足一次性罚额)",
                                "source_quote": "exact text",
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
        policy = self._audit_policy(driver_id, policy, prefs)
        policy = self._merge_stable_fields_from_prior(driver_id, policy)
        policy = self._merge_preference_target_memory(driver_id, policy, prefs)
        policy["_signature"] = signature
        self._preference_policy_by_driver[driver_id] = policy
        self._logger.info(
            "compiled_preference_policy driver=%s data=%s",
            driver_id,
            json.dumps(policy, ensure_ascii=False, separators=(",", ":")),
        )
        return policy

    def _merge_stable_fields_from_prior(self, driver_id: str, policy: dict[str, Any]) -> dict[str, Any]:
        """跨月重编译保护(治非确定丢窗): 偏好按月增长(夜休→+某月配额→…)会触发整月重编译,而重编译是
        LLM 调用、非确定——实测某次重编译丢了 weekend 作息窗→周末夜无窗→跨夜单漏罚。作息/整歇/区域/长途/
        打卡这些【稳定约束】month-to-month 文本不变,不该被重编译丢失。故把上次已验证 policy 里这些字段中、
        本次缺失的条目【并回】(按签名去重,只增不减),绝不让重编译丢掉已确认的硬约束。配额型(cargo_targets/
        cargo_max_limits)月月变,不在此列,由 target_memory 另行结转。"""
        if not FEATURE_FLAGS.get("recompile_stable_merge"):
            return policy
        prior = self._preference_policy_by_driver.get(driver_id)
        nir = policy.get("machine_ir") if isinstance(policy.get("machine_ir"), dict) else None
        pir = prior.get("machine_ir") if isinstance(prior, dict) and isinstance(prior.get("machine_ir"), dict) else None
        if pir is None or nir is None:
            return policy
        stable = ("rest_windows", "off_day_requirements", "region_avoid", "origin_avoid",
                  "long_haul_limits", "daily_order_caps", "location_visit_targets")
        cosmetic = {"label", "source_quote", "penalty_amount", "penalty_cap"}

        def fsig(e: dict[str, Any]) -> str:  # 功能签名(去掉 label/罚额/quote 等易抖动的装饰字段)
            return json.dumps({k: v for k, v in e.items() if k not in cosmetic}, ensure_ascii=False, sort_keys=True)

        for fld in stable:
            new_list = [e for e in (nir.get(fld) or []) if isinstance(e, dict)]
            old_list = [e for e in (pir.get(fld) or []) if isinstance(e, dict)]
            if not old_list:
                continue
            seen = {fsig(e) for e in new_list}
            restored = list(new_list)
            for e in old_list:
                if fsig(e) not in seen:
                    restored.append(e)
                    self._logger.warning(
                        "recompile_stable_merge 并回 %s(本次重编译丢失,从上次policy恢复): %s", fld, fsig(e)[:140]
                    )
            nir[fld] = restored
        return policy

    def _compile_once_or_vote(self, agent_name: str, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        """编译调用。field_voting 开启时同 prompt 独立跑 3 次,对 machine_ir 各列表字段做
        【条目级结构签名多数票】——只出现 1 票的条目(疑似不稳定抽取/幻觉)整条降级进
        unknown_constraints 交守护层,绝不硬写进执行器(错编比丢弃毒,DG8/73.5k 双实锤)。
        unknown_constraints 自身取三票并集(多看无害,丢了才危险)。"""
        if not FEATURE_FLAGS.get("field_voting"):
            return self._agent_json(agent_name, system, payload)
        votes: list[dict[str, Any]] = []
        for _ in range(3):
            try:
                votes.append(self._agent_json(agent_name, system, payload))
            except Exception as e:
                self._logger.warning("compile vote attempt failed: %s", e)
        if not votes:
            raise ValueError("compile failed: all vote attempts errored")
        base = votes[0]
        if len(votes) == 1:
            return base
        skip_keys = {"label", "source_quote", "reason", "note", "why_unsupported", "preference_text", "overnight_note"}

        def sig(entry: dict[str, Any]) -> str:
            return json.dumps(
                {k: v for k, v in sorted(entry.items()) if k not in skip_keys},
                ensure_ascii=False, sort_keys=True,
            )

        # 【硬执行字段】(有确定性执行器,投票不稳定→降级unknown交守护:错编会真违规):
        hard_fields = ["rest_windows", "long_haul_limits", "cargo_targets", "cargo_max_limits",
                       "location_visit_targets", "region_avoid", "origin_avoid"]
        # 【软/无执行字段】(date_or_weekday_rules等在v19无执行器,destination_prefer/makeup另路处理):
        # 投票不稳定时取多数票即可、绝不降级unknown——否则冗余字段抖动会把【全已知司机】踢出
        # 零LLM快车道、每步触发Council(公开司机实测token 9k→742k/79倍)。
        soft_fields = ["destination_prefer", "region_min_targets", "date_or_weekday_rules", "makeup_targets"]
        base_ir = base.get("machine_ir") if isinstance(base.get("machine_ir"), dict) else {}
        demoted: list[dict[str, Any]] = []
        for fld in hard_fields + soft_fields:
            counts: dict[str, int] = {}
            first: dict[str, dict[str, Any]] = {}
            for v in votes:
                ir = v.get("machine_ir") if isinstance(v.get("machine_ir"), dict) else {}
                seen = set()
                for e in ir.get(fld) or []:
                    if not isinstance(e, dict):
                        continue
                    s = sig(e)
                    if s in seen:
                        continue
                    seen.add(s)
                    counts[s] = counts.get(s, 0) + 1
                    first.setdefault(s, e)
            base_ir[fld] = [first[s] for s, c in counts.items() if c >= 2]
            if fld in soft_fields:
                continue  # 软字段不降级(无执行器,抖动无害,降级只会无谓触发Council)
            for s, c in counts.items():
                if c < 2:
                    e = first[s]
                    demoted.append({
                        "preference_text": str(e.get("source_quote") or json.dumps(e, ensure_ascii=False)[:120]),
                        "why_unsupported": f"compile_vote_unstable:{fld}",
                        "risk_level": "medium",
                    })
                    self._logger.warning("compile vote demoted %s: %s", fld, json.dumps(e, ensure_ascii=False)[:150])
        # unknown_constraints: 三票并集按文本去重
        merged_unknown: list[dict[str, Any]] = []
        seen_txt: set[str] = set()
        for v in votes:
            ir = v.get("machine_ir") if isinstance(v.get("machine_ir"), dict) else {}
            for u in ir.get("unknown_constraints") or []:
                if not isinstance(u, dict):
                    continue
                t = _norm_text(str(u.get("preference_text", "")))
                if t and t not in seen_txt:
                    seen_txt.add(t)
                    merged_unknown.append(u)
        base_ir["unknown_constraints"] = merged_unknown + demoted
        base["machine_ir"] = base_ir
        return base

    def _audit_policy(self, driver_id: str, policy: dict[str, Any], prefs: list[Any]) -> dict[str, Any]:
        """编译后本地 audit(确定性,零偏好常量): 九项检查 + rest五条件 + avoid×target冲突剥离
        + semantic-family unknown去重。任何不合格约束【降级进 unknown_constraints 交守护层】,
        绝不静默删除、绝不带病执行。"""
        if not FEATURE_FLAGS.get("compile_audit"):
            return policy
        try:
            ir = policy.get("machine_ir") if isinstance(policy.get("machine_ir"), dict) else None
            if ir is None:
                return policy
            raw_all = _norm_text("".join(str(p.get("content", "")) for p in prefs if isinstance(p, dict)))
            vocab = self._seen_cargo_names_by_driver.get(driver_id) or set()
            unknowns = [u for u in (ir.get("unknown_constraints") or []) if isinstance(u, dict)]

            def demote(entry: dict[str, Any], why: str, risk: str = "medium") -> None:
                unknowns.append({
                    "preference_text": str(entry.get("source_quote") or json.dumps(entry, ensure_ascii=False)[:120]),
                    "why_unsupported": why, "risk_level": risk,
                })
                self._logger.warning("audit demoted: %s | %s", why, json.dumps(entry, ensure_ascii=False)[:150])

            def quote_ok(entry: dict[str, Any]) -> bool:
                q = _norm_text(str(entry.get("source_quote", "")))
                return (not q) or (q in raw_all)  # 缺失宽容(兼容旧schema),给了但回找失败=幻觉

            # 1) rest_windows: 时间锚点完整 + 钟点合法 + days枚举 + source_quote回找
            kept = []
            for w in ir.get("rest_windows") or []:
                if not isinstance(w, dict):
                    continue
                try:
                    sh, eh = int(w.get("start_hour")), int(w.get("end_hour"))
                except (TypeError, ValueError):
                    demote(w, "rest_window缺时间锚点(绝不按猜测小时执行)", "high")
                    continue
                if not (0 <= sh <= 23 and 0 <= eh <= 23):
                    demote(w, "rest_window钟点越界", "high")
                    continue
                if not quote_ok(w):
                    demote(w, "rest_window source_quote回找失败(疑似幻觉)", "high")
                    continue
                d = str(w.get("days", "all") or "all").lower()
                w["days"] = d if d in ("all", "weekday", "weekend") else "all"
                w["start_hour"], w["end_hour"] = sh, eh
                kept.append(w)
            ir["rest_windows"] = kept

            # 2) cargo_targets / cargo_max_limits: month/数量/品类词表(snap)
            for fld, cnt_key, allow_null_month in (
                ("cargo_targets", "min_count", False),
                ("cargo_max_limits", "max_count", True),
            ):
                kept = []
                for t in ir.get(fld) or []:
                    if not isinstance(t, dict):
                        continue
                    mon = t.get("month")
                    if mon is None and not allow_null_month:
                        demote(t, f"{fld}缺month")
                        continue
                    if mon is not None:
                        try:
                            mon = int(mon)
                        except (TypeError, ValueError):
                            demote(t, f"{fld} month非整数")
                            continue
                        if not (1 <= mon <= 12):
                            demote(t, f"{fld} month越界")
                            continue
                        t["month"] = mon
                    try:
                        cnt = int(t.get(cnt_key))
                    except (TypeError, ValueError):
                        demote(t, f"{fld} {cnt_key}非整数")
                        continue
                    if cnt <= 0:
                        demote(t, f"{fld} {cnt_key}非正数")
                        continue
                    t[cnt_key] = cnt
                    name = str(t.get("cargo_name", "") or "").strip()
                    if not name:
                        demote(t, f"{fld} cargo_name为空")
                        continue
                    if not quote_ok(t):
                        demote(t, f"{fld} source_quote回找失败(疑似幻觉)", "high")
                        continue
                    # 周期一致性: 月度槽位的依据原文若是日/周周期('每天N单'≠'每月N单'),
                    # 属于语义错配硬塞——三票一致也防不了,这里是确定性最后防线
                    q = str(t.get("source_quote", ""))
                    if q and any(w in q for w in ("每天", "一天", "每日", "每周", "每隔")) and not any(
                        w in q for w in ("每月", "个月", "月度", "当月", "本月")
                    ):
                        demote(t, f"{fld} 周期错配(原文为日/周周期,槽位按月计数)", "high")
                        continue
                    if FEATURE_FLAGS.get("snap_vocab") and vocab:
                        snapped, st = _snap_category(name, vocab)
                        if st == "unmapped":
                            demote(t, f"品类[{name}]未唯一命中观测词表(等值计数会失配)")
                            continue
                        t["cargo_name"] = snapped
                    kept.append(t)
                ir[fld] = kept

            # 3) long_haul_limits: threshold/max_count 正整数
            kept = []
            for l in ir.get("long_haul_limits") or []:
                if not isinstance(l, dict):
                    continue
                try:
                    thr, mc = int(l.get("threshold_minutes")), int(l.get("max_count"))
                except (TypeError, ValueError):
                    demote(l, "long_haul_limit字段非整数")
                    continue
                if thr <= 0 or mc < 0:
                    demote(l, "long_haul_limit数值非法")
                    continue
                l["threshold_minutes"], l["max_count"] = thr, mc
                kept.append(l)
            ir["long_haul_limits"] = kept

            # 4) region_avoid/origin_avoid: keyword非空 + 与配额目标品类互含→剥离交守护
            target_names = {str(t.get("cargo_name", "")) for t in ir.get("cargo_targets") or [] if isinstance(t, dict)}
            target_names.discard("")
            for fld in ("region_avoid", "origin_avoid"):
                kept = []
                for r in ir.get(fld) or []:
                    if not isinstance(r, dict):
                        continue
                    kw = str(r.get("keyword", "") or "").strip()
                    if not kw:
                        demote(r, f"{fld} keyword为空")
                        continue
                    if any(kw in tn or tn in kw for tn in target_names):
                        demote(r, f"avoid[{kw}]与必做配额品类冲突,剥离交守护层(防误编封杀)", "high")
                        continue
                    kept.append(r)
                ir[fld] = kept

            # 4b) location_visit_targets(目标点打卡): month/min_days正整数 + 坐标对成套或keyword
            #     二选一锚点 + radius缺省补 + quote回找。无任何锚点(坐标地名都没有)→降级 high unknown 交守护(§9.7)。
            kept = []
            for t in ir.get("location_visit_targets") or []:
                if not isinstance(t, dict):
                    continue
                try:
                    mon, md = int(t.get("month")), int(t.get("min_days"))
                except (TypeError, ValueError):
                    demote(t, "location_visit_targets month/min_days非整数")
                    continue
                if not (1 <= mon <= 12):
                    demote(t, "location_visit_targets month越界")
                    continue
                if md <= 0:
                    demote(t, "location_visit_targets min_days非正数")
                    continue
                # 单日 date-task 防线: '某号/当天/那天到某地停一趟'是单日打卡(官方按 route_stops/某天停留判,
                # 非月度N天),易被误编进本月度槽位。若 source_quote 含具体单日标记且无月度累计语义→降级交守护。
                q = str(t.get("source_quote", ""))
                if q and any(w in q for w in ("号", "当天", "那天", "这天", "当日")) and not any(
                    w in q for w in ("每月", "个月", "月度", "累计", "每个月")
                ):
                    demote(t, "location_visit_targets 疑似单日date-task(含具体日期、无月度累计语义)", "high")
                    continue
                lat, lng = t.get("target_lat"), t.get("target_lng")
                has_coord = lat is not None and lng is not None
                if has_coord:
                    try:
                        lat, lng = float(lat), float(lng)
                    except (TypeError, ValueError):
                        demote(t, "location_visit_targets 坐标非数值", "high")
                        continue
                    if not (-90.0 <= lat <= 90.0 and -180.0 <= lng <= 180.0):
                        demote(t, "location_visit_targets 坐标越界", "high")
                        continue
                    t["target_lat"], t["target_lng"] = lat, lng
                kw = str(t.get("keyword", "") or "").strip()
                if not has_coord and not kw:
                    demote(t, "location_visit_targets 缺坐标且缺keyword(无可执行锚点)", "high")
                    continue
                t["keyword"] = kw
                try:
                    r = float(t.get("radius_km"))
                    t["radius_km"] = r if r > 0 else _VISIT_DEFAULT_RADIUS_KM
                except (TypeError, ValueError):
                    t["radius_km"] = _VISIT_DEFAULT_RADIUS_KM
                if not quote_ok(t):
                    demote(t, "location_visit_targets source_quote回找失败(疑似幻觉)", "high")
                    continue
                t["month"], t["min_days"] = mon, md
                kept.append(t)
            ir["location_visit_targets"] = kept

            # 5) semantic-family unknown 去重: 与已编译约束 source_quote 互为子串的 unknown=重复,删
            cov_quotes = []
            for fld in ("rest_windows", "long_haul_limits", "cargo_targets", "cargo_max_limits",
                        "daily_order_caps", "off_day_requirements", "location_visit_targets",
                        "region_avoid", "origin_avoid", "destination_prefer"):
                for e in ir.get(fld) or []:
                    if isinstance(e, dict):
                        q = _norm_text(str(e.get("source_quote", "")))
                        if q:
                            cov_quotes.append(q)
            pref_norms = [_norm_text(str(p.get("content", "")) if isinstance(p, dict) else str(p)) for p in prefs]
            # 【配额型偏好原文】(产出了 cargo_target/cargo_max 的那条 pref)：其配额已由确定性
            # 影子价格+makeup结转履约,该 pref 的所有 unknown 子句(含"补欠额/罚得比上月重"等
            # 结转描述,即使LLM标high)都是配额的附属说明,全部剔除——否则结转期每步触发Council、
            # 墙钟逼近上限(实测公开司机5月单跑25分钟未完)。判据=pref含某配额品类名(鲁棒,不依赖
            # source_quote);作息pref不含配额品类名,其混入的真未知(如"每隔X天回家")不受影响。
            quota_names = [
                str(t.get("cargo_name", "") or "").strip()
                for fld in ("cargo_targets", "cargo_max_limits")
                for t in (ir.get(fld) or []) if isinstance(t, dict)
            ]
            quota_prefs = [
                pc for pc in pref_norms
                if any(nm and nm in pc for nm in quota_names)
            ]
            deduped, seen_txt = [], set()
            for u in unknowns:
                txt = _norm_text(str(u.get("preference_text", "")))
                if not txt or txt in seen_txt:
                    continue
                seen_txt.add(txt)
                why = str(u.get("why_unsupported", ""))
                why_l = why.lower()
                risk = str(u.get("risk_level", "")).lower()
                # 投票降级条目若回原文找不到依据=单票幻觉(如把长途误编进cargo_max_limits的
                # long_haul_generic),直接丢弃——不值得为幻觉把全已知司机踢出零LLM快车道
                if "vote_unstable" in why and txt not in raw_all:
                    self._logger.info("audit丢弃vote幻觉(原文无依据): %s", txt[:80])
                    continue
                # 配额型pref的unknown子句(含结转high)全剔除——配额已确定性履约,不必每步Council
                if "冲突" not in why and "vote_unstable" not in why and any(
                    txt in qp for qp in quota_prefs
                ):
                    self._logger.info("audit剔除配额附属子句(配额已确定性履约): %s", txt[:60])
                    continue
                # 结转语义剔除(概念词,比文本子串鲁棒): 跨月补欠额已由 makeup_targets+影子价格
                # 确定性处理,该司机有配额目标时,任何"补欠/没完成/接着补/makeup/deficit"类 unknown
                # 都是结转描述,剔除。无配额目标的司机(DG8/回家)不含这些词,不受影响。
                _ct = txt + why_l if (txt or why_l) else ""
                if quota_names and "冲突" not in why and "vote_unstable" not in why and any(
                    w in _ct for w in ("接着补", "欠的单", "没完成", "补上月", "补欠",
                                       "makeup", "deficit", "carryover", "carry over", "previousmonth")
                ):
                    self._logger.info("audit剔除结转语义(makeup已确定性处理): %s", (txt or why)[:50])
                    continue
                # LLM自认冗余: why 表明该约束【已被结构化字段覆盖】(如"covered by rest_windows
                # logic"/"recurring"),却仍放进unknown——是重复,剔除(治作息整条原文被重复放unknown)。
                # 真未知(每日上限/回家)的why是"schema无槽位/daily≠monthly/no slot",不含已覆盖语义。
                if "vote_unstable" not in why and "冲突" not in why and any(
                    w in why_l for w in ("covered", "handled by", "structurally", "recurring",
                                         "already captured", "already represented", "已覆盖", "已编")
                ) and any(txt in pc for pc in pref_norms):
                    self._logger.info("audit剔除LLM自认冗余(why表明已覆盖): %s", txt[:60])
                    continue
                # 描述性子句剔除: non-high 且是某条偏好原文的【真子串】=对已编译约束的补充说明
                # (罚款细节/周末定义/比上月重等),不必每步触发Council(治结转期墙钟+token)。
                # 真未知类型(每日上限/回家/在线时长)是独立完整偏好且标 high,受保护不被剔除。
                if risk != "high" and "冲突" not in why and "vote_unstable" not in why and any(
                    txt in pc and txt != pc for pc in pref_norms
                ):
                    self._logger.info("audit剔除描述性子句(non-high且是偏好子串): %s", txt[:60])
                    continue
                covered = any(txt in q or q in txt for q in cov_quotes)
                # 已被硬执行字段覆盖的=重复条目,一律跳过(含vote降级的重复,如long_haul重复编进
                # cargo_max_limits);仅"冲突剥离"和"other兜底"必须保留交Council。
                if covered and "other" not in why and "冲突" not in why:
                    continue
                deduped.append(u)
            ir["unknown_constraints"] = deduped
            policy["machine_ir"] = ir
            self._logger.info(
                "audit_policy done: rest=%s targets=%s caps=%s lh=%s avoid=%s unknown=%s",
                len(ir.get("rest_windows") or []), len(ir.get("cargo_targets") or []),
                len(ir.get("cargo_max_limits") or []), len(ir.get("long_haul_limits") or []),
                len(ir.get("region_avoid") or []) + len(ir.get("origin_avoid") or []),
                len(deduped),
            )
        except Exception as e:
            self._logger.warning("audit_policy failed(保留原policy): %s", e)
        return policy

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

        repo = self._limited_reposition(driver_id, status, sim_min, pref_policy)
        if repo is not None:
            return repo
        wait = self._next_useful_wait_minutes(sim_min, pref_policy)
        self._logger.info("deterministic_wait no_positive_candidate wait=%s seen=%s", wait, len(candidates))
        return {"action": "wait", "params": {"duration_minutes": wait}}

    def _limited_reposition(
        self, driver_id: str, status: dict[str, Any], sim_min: int, pref_policy: dict[str, Any]
    ) -> dict[str, Any] | None:
        """受限主动迁移(三重闸门): 仅当①无正分候选(调用处保证) ②距下一休息窗>240分钟
        ③最优观测热点的期望净值−迁移成本>200元。每日≤1次,迁移段不得与任何作息窗重叠。
        治"冷区只能干等"的毛利黑洞;闸门保证不重蹈大空驶伤害。绝不抛异常。"""
        if not FEATURE_FLAGS.get("ltd_reposition"):
            return None
        try:
            day = sim_min // 1440
            if self._last_reposition_day.get(driver_id) == day:
                return None
            nxt = [s for s, e in self._rest_intervals_around(sim_min, sim_min + 24 * 60, pref_policy) if s > sim_min]
            if nxt and nxt[0] - sim_min <= 240:
                return None  # 临近休息窗不折腾
            lat = float(status.get("current_lat", 0) or 0)
            lng = float(status.get("current_lng", 0) or 0)
            best = None
            for p in self._observed_points_by_driver.get(driver_id, [])[:8]:
                dist = _haversine_km(lat, lng, float(p["lat"]), float(p["lng"]))
                if dist < 30:
                    continue  # 本来就在附近,迁移无意义
                move_min = _distance_minutes(dist)
                if nxt and sim_min + move_min + 120 > nxt[0]:
                    continue  # 迁移+起码2小时作业必须全部落在休息窗前
                ev = float(p.get("price", 0)) * 0.6 - dist * _DEFAULT_COST_PER_KM
                if ev > 200 and (best is None or ev > best[0]):
                    best = (ev, p, dist)
            if best is None:
                return None
            self._last_reposition_day[driver_id] = day
            _ev, p, dist = best
            self._logger.info(
                "limited_reposition driver=%s -> (%.4f,%.4f) dist=%.0fkm ev=%.0f", driver_id, p["lat"], p["lng"], dist, _ev
            )
            return {"action": "reposition", "params": {"latitude": float(p["lat"]), "longitude": float(p["lng"])}}
        except Exception as e:
            self._logger.warning("limited_reposition failed: %s", e)
            return None

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
            if FEATURE_FLAGS.get("dynamic_longhaul"):
                # 按该limit自己的阈值现算(隐藏司机阈值可能是任意小时数;固定over_8h分箱会全口径错配)
                mins = (ledger.get("transport_minutes_by_month") or {}).get(month, [])
                used = sum(1 for m in mins if m > threshold)
            else:
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
        for cap in self._daily_order_caps(pref_policy):
            # 每天最多 N 单: 确定性按日计数拦截(治 Council 每步触发=超时源,且 Council 漏拦~10%)。
            # day = 订单起始日(与官方按 action_start//1440 计数同口径); 当日已接>=N 则否决第 N+1 单。
            max_per_day = _optional_int(cap.get("max_per_day"), 999999)
            if max_per_day <= 0:
                continue
            day = active_start // 1440
            used = int((ledger.get("order_count_by_day") or {}).get(f"day{day}", 0) or 0)
            if used >= max_per_day:
                vetoes.append("daily_order_cap_full")
                break
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
        score += self._visit_target_bonus(cand, pref_policy, ledger, sim_min)
        score += cand.llm_adjustment  # Council调分(已限幅±800),LLM影响有界、代码裁决
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
            if FEATURE_FLAGS.get("subgrad_shadow"):
                # 次梯度影子价格(拉格朗日松弛): 紧迫度λ=欠额/剩余机会数估计。
                # 机会数=该品类近期日均可见单数(观测热点统计)×当月剩余天数;无观测→保守取欠额本身(λ=1)。
                day_in_month = ((_SIMULATION_EPOCH + timedelta(minutes=sim_min)).day) if month == now_month else 1
                days_left = max(1, 30 - day_in_month) if month == now_month else 30
                seen_pts = sum(
                    1 for p in self._observed_points_by_driver.get(getattr(self, "_cur_driver_id", ""), [])
                    if str(p.get("cargo_name", "")) == name
                )
                est_chances = max(1.0, float(seen_pts) * days_left / 4.0)
                lam = min(1.0, shortfall / est_chances)
                if FEATURE_FLAGS.get("aggressive_fulfill"):
                    # 激进履约: 落后线性配速即视为紧迫(λ→1),提早抢配额而非赌后续机会。
                    # bonus仍按penalty比例(不超配额经济价值)、且只加分不veto——作息/上限veto在上游,激进抢单越不过作息防线。
                    nowdt = _SIMULATION_EPOCH + timedelta(minutes=sim_min)
                    if month != now_month or used * self._days_in_month(nowdt) < min_count * nowdt.day:
                        lam = 1.0
                bonus += penalty * (1.0 + lam * shortfall) * days_left_factor
            else:
                bonus += (penalty + 0.12 * penalty * shortfall) * days_left_factor
        return bonus

    def _visit_target_bonus(
        self, cand: CandidateFact, pref_policy: dict[str, Any], ledger: dict[str, Any], sim_min: int
    ) -> float:
        """目标点打卡引导(死字段补全): 候选终点【真达标】(进 radius / 地名命中)→ +min(penalty,1200),
        月末欠额时×2 强化压过普通净收益; 仅【120km 内但不达标】→ +通用弱引导(elif 物理隔离,绝不计 visit、
        绝不叠真达标——120km 当达标会重演'自以为凑满官方照罚')。不veto(作息/上限veto在上游)。"""
        if not FEATURE_FLAGS.get("location_visit"):
            return 0.0
        targets = self._location_visit_targets(pref_policy)
        if not targets:
            return 0.0
        driver_id = getattr(self, "_cur_driver_id", "")
        try:
            end_lat, end_lng = _cargo_point(cand.cargo, "end")
        except Exception:
            end_lat = end_lng = None
        region_text = self._cargo_region_text(cand.cargo)
        nowdt = _SIMULATION_EPOCH + timedelta(minutes=sim_min)
        bonus = 0.0
        for target in targets:
            try:
                month, min_days = int(target.get("month")), int(target.get("min_days"))
            except (TypeError, ValueError):
                continue
            shortfall = max(0, min_days - len(self._visit_days_by_month(driver_id, target)))
            if shortfall <= 0:
                continue
            penalty = self._target_penalty_amount(target, default=3000.0)
            if self._hits_visit_target(end_lat, end_lng, region_text, target):
                base = min(penalty, 1200.0)
                if month == nowdt.month:
                    days_left = self._days_in_month(nowdt) - nowdt.day + 1
                    if days_left <= shortfall + _VISIT_MONTHEND_BUFFER_DAYS:
                        base *= 2.0  # 月末强化真达标候选(仍不veto)
                bonus += base
            elif end_lat is not None:
                lat, lng = target.get("target_lat"), target.get("target_lng")
                if lat is not None and lng is not None:
                    try:
                        if _haversine_km(end_lat, end_lng, float(lat), float(lng)) <= _VISIT_SCOUT_KM:
                            bonus += _VISIT_WEAK_BONUS  # 弱引导,绝不计 visit 达标
                    except (TypeError, ValueError):
                        pass
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
                if FEATURE_FLAGS.get("rest_window_strict"):
                    # 缺时间锚点的窗口绝不按猜测小时数执行(隐藏司机窗口未知,猜=错配风险)；
                    # 该窗已在编译audit阶段降级进 unknown_constraints 交守护层。
                    if window.get("start_hour") is None or window.get("end_hour") is None:
                        continue
                sh = _optional_int(window.get("start_hour"), 21)
                eh = _optional_int(window.get("end_hour"), 6)
                if sh == eh:
                    # start==end 是"时长约束"(如"连续休息N小时")被误塞进钟点schema的占位artifact
                    # (0/0→整天)——绝不执行成全天休息(实测致重作息司机 net=0 毛利灾难)。时长类应由
                    # 编译期改写成具体过夜窗(end=6/start=(6-N));此处兜底跳过占位窗。
                    continue
                s = day * 1440 + sh * 60
                e = day * 1440 + eh * 60
                if e <= s:
                    e += 1440
                intervals.append((s, e))
        intervals.sort()
        return intervals

    def _merge_no_go_segments(self, start_min: int, end_min: int, pref_policy: dict[str, Any]) -> list[tuple[int, int]]:
        """§9.4: 当日"禁动区间"并集→合并相邻/重叠窗成连续禁动段,睡穿时睡到段的【真正末端】
        (而非单个窗的末端),防"睡到A窗尾紧接着违反相邻B窗"。当前并集源=作息禁动窗;
        回家门禁型(home-quiet)落地后在此并入(同一并集口径)。"""
        raw = self._rest_intervals_around(start_min, end_min, pref_policy)
        if not raw:
            return []
        raw.sort()
        merged: list[tuple[int, int]] = []
        for s, e in raw:
            if merged and s <= merged[-1][1]:  # 重叠或相接→合并
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged

    def _pre_query_rest_guard(self, status: dict[str, Any], pref_policy: dict[str, Any], budget_guard: bool) -> int | None:
        """P0-3: query_cargo 会推进 ceil(返回条数/10)≤ceil(k/10) 仿真分钟。若此刻距下一禁动段段头
        ≤ 预估扫描(按 k 取保守上界)+ 安全余量,扫描尾巴会擦进窗头→wait 覆盖不到窗头→作息禁动型
        漏罚(尤其"时段不接单不空驶"型要求窗内全 wait 覆盖)。故查货前提前 wait 睡穿到禁动段真正末端(§9.4),
        不让查货跨进窗。仅当①此刻不在窗内(在窗内由 _current_rest_wait_minutes 处理)且②扫描会跨窗头
        时触发;无作息窗司机 no-op(_rest_intervals_around 返回空)。"""
        if not FEATURE_FLAGS.get("pre_query_rest_guard"):
            return None
        sim_min = int(status.get("simulation_progress_minutes", 0) or 0)
        estimated_scan = math.ceil((120 if budget_guard else 220) / 10)
        reach = sim_min + estimated_scan + _PRE_QUERY_SAFETY_MIN
        for start, end in self._merge_no_go_segments(sim_min, sim_min + 24 * 60, pref_policy):
            if sim_min < start <= reach:
                return max(1, min(_MAX_WAIT_MINUTES, end - sim_min))  # 睡穿到禁动段末端
        return None

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
    def _daily_order_caps(pref_policy: dict[str, Any]) -> list[dict[str, Any]]:
        ir = pref_policy.get("machine_ir") if isinstance(pref_policy.get("machine_ir"), dict) else {}
        raw = ir.get("daily_order_caps") if isinstance(ir.get("daily_order_caps"), list) else []
        return [x for x in raw if isinstance(x, dict)]

    @staticmethod
    def _off_day_requirements(pref_policy: dict[str, Any]) -> list[dict[str, Any]]:
        ir = pref_policy.get("machine_ir") if isinstance(pref_policy.get("machine_ir"), dict) else {}
        raw = ir.get("off_day_requirements") if isinstance(ir.get("off_day_requirements"), list) else []
        return [x for x in raw if isinstance(x, dict)]

    # ===== 目标点打卡(location_visit_targets)消费链: getter→统一达标判据→visit_days去重 =====
    @staticmethod
    def _location_visit_targets(pref_policy: dict[str, Any]) -> list[dict[str, Any]]:
        ir = pref_policy.get("machine_ir") if isinstance(pref_policy.get("machine_ir"), dict) else {}
        raw = ir.get("location_visit_targets") if isinstance(ir.get("location_visit_targets"), list) else []
        return [x for x in raw if isinstance(x, dict)]

    @staticmethod
    def _cargo_region_text(cargo: dict[str, Any]) -> str:
        start = cargo.get("start") if isinstance(cargo.get("start"), dict) else {}
        end = cargo.get("end") if isinstance(cargo.get("end"), dict) else {}
        keys = ("province", "city", "district", "address")
        return (" ".join(str(start.get(k, "") or "") for k in keys) + " "
                + " ".join(str(end.get(k, "") or "") for k in keys))

    @staticmethod
    def _hits_visit_target(end_lat: Any, end_lng: Any, region_text: str, target: dict[str, Any]) -> bool:
        """统一达标判据(打分侧与计数侧共用,防口径分裂)。**地名优先(对齐官方唯一存在的 _eval_
        required_region_cargo_days=按 city 文本含 keyword)**: 只要 target 有 keyword,就以【货源起终点
        文本含 keyword】判达标,坐标此时仅作 scout/弱引导(不在此判)。仅当【纯坐标、无 keyword】时
        才回退到终点进 radius_km。否则【地名+坐标都给】的司机会被小半径泡误判、自以为凑满官方照罚。"""
        kw = str(target.get("keyword") or "").strip()
        if kw:
            return kw in (region_text or "")
        lat, lng = target.get("target_lat"), target.get("target_lng")
        if lat is not None and lng is not None and end_lat is not None and end_lng is not None:
            try:
                r = float(target.get("radius_km") or _VISIT_DEFAULT_RADIUS_KM)
                return _haversine_km(float(end_lat), float(end_lng), float(lat), float(lng)) <= r
            except (TypeError, ValueError):
                return False
        return False

    def _visit_days_by_month(self, driver_id: str, target: dict[str, Any]) -> set[int]:
        """该 target 计分月已打卡【天集合】(§9.6 按 day_index 去重)。坐标版按到货日(finish//1440=
        司机抵达终点之日),地名版按订单起始日(active_start//1440,对齐官方 _eval_required_region_cargo_days)。
        仅从 chosen_orders(take_order)派生(纯 reposition 打卡留月末强制导向增量处理)。"""
        try:
            tgt_month = int(target.get("month"))
        except (TypeError, ValueError):
            return set()
        # 与 _hits_visit_target 同模式: 有 keyword=地名版(按订单起始日,对齐官方); 纯坐标=按到货日
        is_coord = (not str(target.get("keyword") or "").strip()) and \
            target.get("target_lat") is not None and target.get("target_lng") is not None
        days: set[int] = set()
        for o in self._chosen_orders_by_driver.get(driver_id, []):
            if not self._hits_visit_target(o.get("end_lat"), o.get("end_lng"), o.get("region_text"), target):
                continue
            try:
                d = (int(o.get("finish", 0)) // 1440) if is_coord else (int(o.get("active_start", 0)) // 1440)
            except (TypeError, ValueError):
                continue
            if (_SIMULATION_EPOCH + timedelta(days=d)).month == tgt_month:
                days.add(d)
        return days

    def _active_day_set(self, driver_id: str) -> set[int]:
        """活动天集合(activity-interval 口径,对齐官方 _active_minutes_by_day): 任一已选单的
        [active_start, finish] 与某日历日有交集 → 该日有活动。这修正了 order_count_by_day 仅按
        【起始日】计数的口径错配——跨夜单的运行时间溢入次日(乃至跨月)本是次日活动,旧口径却把
        次日误判为整歇(official 按分钟逐日累计,会算到次日)。day-index 自带月份,跨月自动归位。
        (reposition 关→活动仅 take_order;若日后开 ltd_reposition,须在此并入 reposition 区间。)"""
        days: set[int] = set()
        for order in self._chosen_orders_by_driver.get(driver_id, []):
            try:
                start = int(order.get("active_start", 0))
                end = int(order.get("finish", 0))
            except (TypeError, ValueError):
                continue
            if end <= start:
                end = start + 1
            d = start // 1440
            while d * 1440 < end:
                days.add(d)
                d += 1
        return days

    def _off_days_taken(self, driver_id: str, sim_min: int) -> int:
        """本月已取得的整天歇车数 = 过去日(不含今天)中【无任何 activity_interval 交集】的天。"""
        active = self._active_day_set(driver_id)
        now = _SIMULATION_EPOCH + timedelta(minutes=sim_min)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        start_day = int((month_start - _SIMULATION_EPOCH).total_seconds() // 60) // 1440
        today = sim_min // 1440
        return sum(1 for d in range(start_day, today) if d not in active)

    def _forced_off_day_wait(self, driver_id: str, status: dict[str, Any], pref_policy: dict[str, Any]) -> int | None:
        """每月至少N整天不出车: 月末逼近(剩余天数 <= 还需off天数+1缓冲)且今天还没接单时,
        强制整天歇(歇到今日结束)。对齐官方 _eval_off_days(当日活动分钟==0)。reposition关→不歇车的天必有接单。"""
        reqs = self._off_day_requirements(pref_policy)
        if not reqs:
            return None
        sim_min = int(status.get("simulation_progress_minutes", 0) or 0)
        now = _SIMULATION_EPOCH + timedelta(minutes=sim_min)
        # 含今天的本月剩余天数,但夹到【仿真实际剩余天数】: 官方 off-day 统计窗按 simulation_duration_days
        # 取 days,末月/短horizon时日历整月剩余会高估→按日历推迟到月末才强制→horizon内永不触发=漏歇。
        # 复赛92天=3个完整日历月,二者恒等(本夹钳为no-op);仅对短horizon收紧,且只会更早强制,绝不漏罚。
        cal_days_left = self._days_in_month(now) - now.day + 1
        remaining_sim_days = max(1, -(-(_HORIZON_MINUTES - sim_min) // 1440))  # ceil 含今天
        days_left = min(cal_days_left, remaining_sim_days)
        taken = self._off_days_taken(driver_id, sim_min)
        today = sim_min // 1440
        # 今天是否仍"干净"(可被强制为整歇)= 今日无 activity_interval 交集(含昨日跨夜单溢入)。
        today_clean = today not in self._active_day_set(driver_id)
        for req in reqs:
            try:
                need = int(req.get("min_off_days"))
            except (TypeError, ValueError):
                continue
            still = need - taken
            if still <= 0:
                continue
            if days_left <= still + 1 and today_clean:
                return max(1, 1440 - (sim_min % 1440))  # 歇到今日结束=整天off
        return None

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

    def _apply_council_judgments(
        self, all_candidates: list[CandidateFact], review_set: list[CandidateFact], guardian: dict[str, Any],
        driver_id: str = "", sim_min: int = 0,
    ) -> list[CandidateFact]:
        """Council v2 判定应用(权限有界): hard_avoid须带依据(source_quote/preference_index/
        risk_reason),缺依据自动降级soft_avoid; soft_avoid按风险分级调分(low-80/medium-200/
        high-500); score_adjustment限幅±800; 未审候选在守护激活时不参与argmax。LLM调分,代码裁决。
        P0-1 埋点: 每条判定纯日志记录(不写外部文件),四步执行器做完后若扣分仍卡,据此看
        Council 还在挡什么规则=还剩什么类型该确定性化(零行为改动)。"""
        judg: dict[str, dict[str, Any]] = {}
        for item in guardian.get("candidate_judgments") or []:
            if isinstance(item, dict) and str(item.get("id") or "").strip():
                judg[str(item["id"]).strip()] = item
        reviewed_ids = {c.cargo_id for c in review_set}
        soft_by_risk = {"low": -80.0, "medium": -200.0, "high": -500.0}
        for cand in all_candidates:
            if cand.cargo_id not in reviewed_ids:
                if cand.legal:
                    cand.legal = False
                    cand.veto_reasons.append("council_unreviewed")
                continue
            item = judg.get(cand.cargo_id)
            if not item:
                continue
            verdict = str(item.get("verdict") or "").lower()
            has_basis = bool(
                str(item.get("source_quote") or "").strip()
                or item.get("preference_index") is not None
                or str(item.get("risk_reason") or item.get("risk") or "").strip()
            )
            if verdict == "hard_avoid" and not has_basis:
                verdict = "soft_avoid"  # 无依据的封杀降级(kiki第5条权限规则)
                self._logger.warning("council hard_avoid缺依据,降级soft_avoid: %s", cand.cargo_id)
            adj = 0.0
            try:
                adj = float(item.get("score_adjustment") or 0.0)
            except (TypeError, ValueError):
                adj = 0.0
            adj = max(-800.0, min(800.0, adj))
            self._logger.info(
                "council_judgment data=%s",
                json.dumps({
                    "event": "council_judgment",
                    "driver_id": driver_id,
                    "sim_min": sim_min,
                    "cargo_id": cand.cargo_id,
                    "verdict": verdict,
                    "source_quote": str(item.get("source_quote") or ""),
                    "preference_index": item.get("preference_index"),
                    "risk_reason": str(item.get("risk_reason") or item.get("risk") or ""),
                    "risk_level": str(item.get("risk_level") or ""),
                    "score_adjustment": adj,
                }, ensure_ascii=False, separators=(",", ":")),
            )
            if verdict == "hard_avoid":
                cand.legal = False
                cand.veto_reasons.append("guardian_hard_avoid")
            elif verdict == "soft_avoid":
                risk = str(item.get("risk_level") or "medium").lower()
                cand.llm_adjustment = min(adj, soft_by_risk.get(risk, -200.0))
            else:  # prefer / allow
                cand.llm_adjustment = adj
        return all_candidates

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

    @staticmethod
    def _days_in_month(dt: datetime) -> int:
        return ((dt.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)).day

    def _monthend_shortfall_targets(self, pref_policy: dict[str, Any], driver_id: str, sim_min: int) -> list[str]:
        """配额欠额定向广查闸门。保守(默认): 仅月末7天内仍欠额的配额品类→触发强制扩查+定向种子。
        激进(aggressive_fulfill): 落后线性配速即提早触发,不再死等月末。纯日历算术+客观计数,零偏好常量。"""
        if not FEATURE_FLAGS.get("monthend_scout"):
            return []
        try:
            now = _SIMULATION_EPOCH + timedelta(minutes=sim_min)
            month_days = self._days_in_month(now)
            days_left = month_days - now.day
            aggressive = bool(FEATURE_FLAGS.get("aggressive_fulfill"))
            if days_left > 7 and not aggressive:
                return []  # 保守: 只在月末窗口动用大查询(全月广查曾实测伤隐藏司机)
            ledger = self._preference_ledger(driver_id)
            counts = ledger.get("cargo_name_counts_by_month") or {}
            out = []
            for t in self._cargo_targets(pref_policy):
                try:
                    if int(t.get("month")) != now.month:
                        continue
                    need = int(t.get("min_count"))
                except (TypeError, ValueError):
                    continue
                name = str(t.get("cargo_name", "") or "").strip()
                if not name:
                    continue
                used = int((counts.get(f"2026-{now.month:02d}") or {}).get(name, 0) or 0)
                if need - used <= 0:
                    continue
                # 月末7天=安全网无条件; 激进额外: 落后线性配速(used*月天数 < need*当前日)即提早定向找货
                if days_left <= 7 or (aggressive and used * month_days < need * now.day):
                    out.append(name)
            return out
        except Exception as e:
            self._logger.warning("monthend_shortfall failed: %s", e)
            return []






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
                            "score_adjustment": "integer -800..800, bounded bonus/malus to the deterministic score",
                            "source_quote": "exact preference text snippet this judgment is based on (required for hard_avoid)",
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
    def _compact_policy(policy: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in policy.items() if k != "_signature"}


    def _run_deterministic_market_scout_queries(
        self,
        driver_id: str,
        status: dict[str, Any],
        first_pass: list[CandidateFact],
        budget_guard: bool,
        shortfall_cats: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if budget_guard:
            return []
        seeds: list[tuple[float, float, str]] = []
        # 月末配额欠额时: 种子优先指向该品类历史出现过的观测热点(定向找货,削自然扣分主成分)
        if shortfall_cats:
            for point in self._observed_points_by_driver.get(str(status.get("driver_id", "")), []):
                if str(point.get("cargo_name", "")) in shortfall_cats:
                    seeds.append((float(point["lat"]), float(point["lng"]), f"shortfall:{point.get('cargo_name')}"))
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
        if FEATURE_FLAGS.get("compile_temperature0"):
            request["temperature"] = 0  # 同输入同输出,降编译/守护方差(网关拒收时重试分支会去掉)
        last_err: Exception | None = None
        for attempt in range(2):  # 单次重试: 容忍偶发坏JSON/网络抖动,仍保留fail-fast精神
            try:
                resp = self._api.model_chat_completion(dict(request))
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
            except Exception as e:
                last_err = e
                request.pop("temperature", None)  # 重试时去掉可选参数,防网关拒收
                self._logger.warning("agent_json retry agent=%s err=%s", agent_name, e)
        raise last_err if last_err else ValueError(f"{agent_name} failed")


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


    def _remember_chosen_action(self, driver_id: str, action: dict[str, Any], candidates: list[CandidateFact]) -> None:
        if action.get("action") != "take_order":
            return
        cargo_id = str((action.get("params") or {}).get("cargo_id", "")).strip()
        cand = next((item for item in candidates if item.cargo_id == cargo_id), None)
        if cand is None:
            return
        active_start = cand.finish_min - cand.pickup_min - cand.wait_min - cand.transport_min
        try:
            end_lat, end_lng = _cargo_point(cand.cargo, "end")
        except Exception:
            end_lat = end_lng = None
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
                "end_lat": end_lat,             # 坐标版打卡判定(终点进 radius)
                "end_lng": end_lng,
                "region_text": self._cargo_region_text(cand.cargo),  # 地名版打卡判定(含 keyword)
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
        # 每单原始干线分钟按月列表(供动态阈值长途计数——隐藏司机的阈值可能是任意小时数,
        # 固定 over_8h 分箱会全口径错配)
        transport_minutes_by_month: dict[str, list[int]] = {}
        for order in orders:
            month = str(order.get("month") or "")
            transport_minutes_by_month.setdefault(month, []).append(int(order.get("transport_min", 0) or 0))
        # 按日接单计数(近8天)——守护层判断"每天最多/至少N单/接单间隔"类台账依赖型偏好的事实基础
        day_counts: dict[str, int] = {}
        last_take_min: int | None = None
        for order in orders:
            try:
                d = int(order.get("active_start", 0)) // 1440
                day_counts[f"day{d}"] = day_counts.get(f"day{d}", 0) + 1
                last_take_min = max(last_take_min or 0, int(order.get("active_start", 0)))
            except (TypeError, ValueError):
                pass
        recent_days = dict(sorted(day_counts.items(), key=lambda kv: kv[0])[-8:])
        return {
            "note": "factual memory of previous LLM-selected orders in this run; use only to interpret visible text preferences",
            "orders_total": len(orders),
            "cargo_name_counts_by_month": name_counts,
            "transport_duration_bins_by_month": transport_duration_bins,
            "transport_minutes_by_month": transport_minutes_by_month,
            "active_duration_bins_by_month": active_duration_bins,
            "active_clock_hour_counts": active_clock_hours,
            "orders_per_day_recent": recent_days,
            "order_count_by_day": day_counts,
            "last_order_start_min": last_take_min,
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
        names = self._seen_cargo_names_by_driver.setdefault(driver_id, set())
        for item in items:
            cargo = item.get("cargo") or {}
            nm = str(cargo.get("cargo_name", "") or "").strip()
            if nm:
                names.add(nm)  # 词表累积(audit品类校验/snap对齐用,客观观测零硬编码)
            try:
                start_lat, start_lng = _cargo_point(cargo, "start")
                price = _cargo_price_yuan(cargo)
            except Exception:
                continue
            if any(_haversine_km(start_lat, start_lng, p["lat"], p["lng"]) < 25 for p in points):
                continue
            entry = {"lat": round(start_lat, 4), "lng": round(start_lng, 4), "price": round(price, 1)}
            if nm:
                entry["cargo_name"] = nm  # 供月末定向广查种子(欠额品类历史热点)
            points.append(entry)
        points.sort(key=lambda p: p["price"], reverse=True)
        del points[20:]
