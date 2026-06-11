"""决策入口（瘦编排层）：状态 → 记忆 → 查货 → 偏好无关预筛 → 提示 → 模型 → 护栏 → 动作。

保持 `SimulationApiPort` 契约不变。设计原则（见 task_plan.md「G. 合规边界」）：
- 代码只做【客观计算 + 护栏】；
- 一切【偏好理解与取舍】交给 LLM；
- decide() 任何异常都兜底为安全动作，绝不让司机因报错出局归零。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from simkit.ports import SimulationApiPort

from . import checker, compiler, guardrails, llm, planner, prompts, tools
from .memory import DriverMemory


def _parse_md(s: Any) -> tuple[int, int] | None:
    """'M-D' 或 'MM-DD' → (月, 日)。"""
    try:
        parts = str(s).strip().split("-")
        if len(parts) >= 2:
            return int(parts[-2]), int(parts[-1])
    except (TypeError, ValueError):
        pass
    return None


def _now_md(now_wall: str) -> tuple[int, int] | None:
    """'YYYY-MM-DD HH:MM:SS' → (月, 日)。"""
    try:
        d = now_wall.split()[0].split("-")
        return int(d[1]), int(d[2])
    except (IndexError, ValueError):
        return None


# ===== R1 确定性执行 feature flags（形状参数，零偏好常量）=====
DECIDE_MODE = "argmax"  # "argmax"=确定性选单 / "hybrid"=argmax提名+LLM单票否决 / "llm"=旧路径(本地影子对照用)
SHADOW_LOG = False      # True=行为走旧llm路径,同时记录argmax提名(本地分歧率统计用,提交包必须False)
REQUERY_MIN = 60        # 无正分候选时的再查间隔(分钟)
SHADOW_CAP = 5000.0     # 配额影子价格封顶(元/单)
SAFE_BQ = True          # R2 月末安全广查(低频/不压线/时钟刷新/等值,广查结果并入打分不强制接)
SAFE_BQ_DAYS = 3        # 月末窗口:days_left ≤ remaining + 此值 才广查


class ModelDecisionService:
    def __init__(self, api: SimulationApiPort) -> None:
        self._api = api
        self._memory = DriverMemory()
        self._compiler = compiler.PreferenceCompiler()
        self._logger = logging.getLogger("agent.decision_service")
        self._last_bq: dict[tuple[str, str], int] = {}  # (司机,品类)→上次广查的仿真天，用于每天最多广查一次

    def decide(self, driver_id: str) -> dict[str, Any]:
        try:
            return self._decide(driver_id)
        except Exception as e:  # 决策绝不能抛——抛了该司机整月归零
            self._logger.exception("decide 异常，兜底 wait: %s", e)
            return guardrails.fallback_action()

    def _decide(self, driver_id: str) -> dict[str, Any]:
        api = self._api
        status = api.get_driver_status(driver_id)
        now_min = int(status.get("simulation_progress_minutes") or 0)
        now_wall = str(status.get("simulation_wall_time") or "")
        month_end_min, month_end_wall = tools.month_end(now_wall)

        self._memory.observe(driver_id, status, api)
        mem_summary = self._memory.summary_text(driver_id, status)

        # 编译偏好为通用约束（LLM 理解，按原文哈希缓存；代码零偏好常量）
        ir = self._compiler.compile(api, status.get("preferences") or [])

        # 日度规划：整月/日期类义务仍由 LLM（低频）
        day = now_min // 1440
        if self._memory.planned_day(driver_id) != day:
            directive = planner.plan_day(api, status, mem_summary, now_wall, month_end_wall)
            self._memory.set_directive(driver_id, day, directive)
        directive = self._memory.get_directive(driver_id)

        # 整休配额刹车：只有"剩余天数已不够凑够配额"时才允许整休；否则今天交回去干活(治"过度整休"漏接货)。
        # 纯天数算术(配额 N 来自 LLM 读偏好；剩余/已休来自客观计数)，零偏好常量。
        try:
            req = int(ir.get("off_days_required") or directive.get("off_days_required") or 0)
            if req > 0:
                cur_day = now_min // 1440
                taken = self._memory.off_days_taken(driver_id, cur_day)
                remaining = max(0, req - taken)
                days_left = max(1, (month_end_min // 1440) - cur_day)
                _cmd = _now_md(now_wall)
                _has_dated_today = any(
                    _parse_md(t.get("date", "")) == _cmd for t in (directive.get("dated_tasks") or [])
                )
                _restricted = checker.has_dated_restriction_today(ir, _cmd)
                if _has_dated_today:
                    eff_off = False  # 当日有专属日程(如赴约/办事)→ 不整休，交义务执行/LLM
                elif _restricted:
                    eff_off = True  # 日期条件区域限制日(如某地查车)→ 当天保持静止最安全，且正好计入整休配额
                elif remaining <= 0:
                    eff_off = False
                elif remaining >= days_left:
                    eff_off = True  # 无余量，必须整休
                elif directive.get("today_is_off_day") and days_left <= remaining + 2:
                    eff_off = True  # 临近末尾，听 planner 安排整休
                else:
                    eff_off = False  # 有余量 → 今天去干活，别白白整休
                if eff_off != bool(directive.get("today_is_off_day")):
                    directive = dict(directive)
                    directive["today_is_off_day"] = eff_off
        except Exception as e:
            self._logger.warning("off-day 刹车失败: %s", e)

        # —— 整休日集合：把"还差的整休天数"定在月末最后几天，确定性执行(全天静止 + 防长途溢出) ——
        off_set: set[int] = set()
        try:
            _req = int(ir.get("off_days_required") or directive.get("off_days_required") or 0)
            if _req > 0:
                _cur = now_min // 1440
                _rem = max(0, _req - self._memory.off_days_taken(driver_id, _cur))
                _meday = month_end_min // 1440
                if _rem > 0:
                    off_set = set(range(max(_cur, _meday - _rem), _meday))
        except Exception as e:
            self._logger.warning("off_set 计算失败: %s", e)

        # —— 确定性休息：当前若处于 LLM 编译出的休息时段，直接一次长 wait 睡到时段结束 ——
        # 时段全部来自 LLM 对偏好原文的解析(compiler/planner)，代码只执行、不自定义时段；
        # 并省去本步查货与模型调用开销。这是把"作息类硬约束"做成确定性可靠、从而转正的关键。
        windows = checker.window_tuples(ir, directive)
        inside, end_abs = checker.in_any_window(now_min, windows)
        if inside and end_abs and end_abs > now_min:
            dur = min(end_abs - now_min, guardrails.MAX_WAIT_MINUTES)
            self._logger.info(
                "rest-window 确定性休息 driver=%s wait=%s 睡到 %s",
                driver_id,
                dur,
                tools.min_to_wall(end_abs),
            )
            return {"action": "wait", "params": {"duration_minutes": dur}}

        # —— 整休日：全天不接单不空驶。确定性执行，不靠提示词。——
        if (now_min // 1440) in off_set or directive.get("today_is_off_day"):
            cur_day = now_min // 1440
            dur = min(max(1, (cur_day + 1) * 1440 - now_min), guardrails.MAX_WAIT_MINUTES)
            self._logger.info("off-day 整休 driver=%s wait=%s", driver_id, dur)
            return {"action": "wait", "params": {"duration_minutes": dur}}

        # —— 确定性义务执行：日期任务(到坐标停留) + 区域配额(临近期限去该地找货) ——
        # 坐标/日期/地名全部来自 LLM 对偏好的编译(planner)，代码只在该去的日子可靠执行；休息优先(上面已返回)。
        forced = self._obligation_action(driver_id, status, now_min, now_wall, month_end_min, directive)
        if forced is not None:
            return forced

        lat = float(status["current_lat"])
        lng = float(status["current_lng"])
        resp = api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng, k=60)
        # 查货会推进仿真时钟(ceil(条数/10)分钟)，但 now_min 取自查货前 → 完成时刻估算系统性
        # 乐观了几~几十分钟，压线接单会确定性溢入休息时段(本地法医:全部夜休违规均为此因)。
        # get_driver_status 不耗仿真时间，刷新即修复。纯客观时钟同步，与偏好无关。
        now_min = int(api.get_driver_status(driver_id).get("simulation_progress_minutes") or now_min)
        items = resp.get("items", []) or []
        # 累积真实品类名词表(客观事实,供配额关键词等值对齐——评分按完全相等计数)
        self._memory.note_cargo_names(
            driver_id, ((it.get("cargo") or {}).get("cargo_name") for it in items if isinstance(it, dict))
        )
        active_cats = self._active_quota_categories(driver_id, ir, now_wall)
        candidates = tools.prepare_candidates(
            items, now_min, month_end_min, top_n=10, quota_categories=active_cats
        )
        checker.annotate(candidates, ir, _now_md(now_wall))  # 按"当日生效"给候选打违规标签（只标注，不删货）
        # 防溢出：会送达到"必须整休日"的单，提前标违规(否则前一天接的长途会占用整休日凌晨)
        if off_set:
            for c in candidates:
                try:
                    if (int(c.get("est_finish_min") or now_min) // 1440) in off_set:
                        c.setdefault("violates", []).append("会送达到必须整休的整天")
                except (TypeError, ValueError):
                    pass
        # 防错过日期任务：会送达到"有未完成专属日程的日子"的单，提前标违规(F6,
        # 旧版靠选单LLM看提示词隐式规避,确定性选单必须显式化,否则机械爽约)
        pending_days = self._pending_task_days(driver_id, directive, now_min)
        if pending_days:
            for c in candidates:
                try:
                    if (int(c.get("est_finish_min") or now_min) // 1440) in pending_days:
                        c.setdefault("violates", []).append("会占用有专属日程的日子")
                except (TypeError, ValueError):
                    pass
        # —— 月度单数上限(如 >8h 长途 ≤N/月)：本月已达上限 → 命中谓词的候选标违规(LLM/反闲置都不接) ——
        self._mark_monthly_caps(driver_id, ir, now_wall, candidates)
        for c in candidates:
            self._memory.note_candidate(driver_id, c["cargo_id"], c)

        # —— 品类配额履约：llm 模式走旧"顺路履约"强制路径；argmax/hybrid 模式由
        # 影子价格统一接管(配额货加边际罚款分后与普通货同台竞争,经济权衡自动完成) ——
        if DECIDE_MODE == "llm" or SHADOW_LOG:
            forced_q = self._quota_action(
                driver_id, status, ir, now_min, now_wall, month_end_min, candidates, windows, off_set
            )
            if forced_q is not None:
                return forced_q

        # ========== R1 确定性选单 ==========
        # 打分=有效时薪+配额影子价格(tools.score_candidate)；硬过滤=violates(含IR过滤/整休/
        # 日期任务/月上限标注)+不跨休息窗(60缓冲)。argmax 替代每步 LLM 选单：
        # 消除±20k选单方差、省token、时薪目标函数治"绝对额排序看不见等窗时间"。
        shadow_map = self._build_shadow_map(driver_id, ir, now_wall, month_end_min, now_min)
        # —— R2 安全配额回补：月末告急且本地无该品类干净货 → 低频广查把配额货并入候选打分。
        # 与旧广查的全部区别：月末窗口/不压线(60缓冲)/时钟刷新/等值匹配/距夜休远才查/每天每品类一次/
        # 不强制接(影子价格让它自然胜出)。每次~60min查询税换数千罚款,经济上压倒性合算。
        if SAFE_BQ and shadow_map:
            extra, now_min = self._safe_broad_query(
                driver_id, status, ir, now_min, now_wall, month_end_min, shadow_map, windows, candidates
            )
            if extra:
                candidates = candidates + extra
        clean = [
            c
            for c in candidates
            if not c.get("violates")
            and not checker.overlaps_any_window(now_min, int(c.get("est_finish_min") or now_min), windows)
        ]
        scored = sorted(
            ((tools.score_candidate(c, now_min, shadow_map), c) for c in clean),
            key=lambda x: x[0],
            reverse=True,
        )
        positive = [(s, c) for s, c in scored if s > 0]
        if SHADOW_LOG:
            top = positive[0] if positive else None
            self._logger.info(
                "shadow argmax driver=%s 提名=%s score=%.2f shadow_cats=%s",
                driver_id,
                top[1]["cargo_id"] if top else "wait",
                top[0] if top else 0.0,
                list(shadow_map),
            )
        if DECIDE_MODE in ("argmax", "hybrid") and not SHADOW_LOG:
            if positive:
                pick = None
                if DECIDE_MODE == "hybrid":
                    # LLM 单票否决(权限棘轮:只能说"不",不能换单)；至多问前2名,全否则等待
                    prefs_text = [str(p.get("content") or "") for p in (status.get("preferences") or [])]
                    for s, c in positive[:2]:
                        if not self._llm_veto(api, c, prefs_text):
                            pick = (s, c)
                            break
                else:
                    pick = positive[0]
                if pick is not None:
                    s, c = pick
                    self._logger.info(
                        "decide driver=%s t=%s cand=%s -> argmax take %s rate=%.2f shadow=%s",
                        driver_id, now_min, len(candidates), c["cargo_id"], s,
                        str(c.get("cargo_name") or "") in shadow_map,
                    )
                    return {"action": "take_order", "params": {"cargo_id": c["cargo_id"]}}
            act = self._wait_action(now_min, windows)
            self._logger.info(
                "decide driver=%s t=%s cand=%s -> argmax wait %s", driver_id, now_min, len(candidates), act["params"]
            )
            return act
        # ========== 旧 LLM 选单路径(DECIDE_MODE=="llm" 或 SHADOW_LOG 影子对照) ==========

        region_hint = ""
        rt = directive.get("region_target") if isinstance(directive, dict) else None
        if isinstance(rt, dict) and rt.get("need_days", 0) > 0:
            _done = self._memory.region_days_done(driver_id, rt["keyword"])
            if len(_done) < rt["need_days"]:
                region_hint = (
                    f"仍需在「{rt['keyword']}」接单(起或终点城市含该地名)共 {rt['need_days']} 个不同日，"
                    f"已完成 {len(_done)} 日；附近若有起/终点含「{rt['keyword']}」的候选请【优先接】。"
                )
        quota_hint = ""
        try:
            _cm = _now_md(now_wall)
            if _cm:
                _parts = []
                for q in ir.get("category_quotas") or []:
                    if int(q.get("month") or 0) != _cm[0]:
                        continue
                    _cat = str(q.get("category") or "").strip()
                    _need = int(q.get("min_orders") or 0)
                    _done = self._memory.category_orders_done(driver_id, _cat, _cm[0])
                    if _cat and _need - _done > 0:
                        _parts.append(f"本月还需接「{_cat}」{_need - _done} 单(已 {_done}/{_need})")
                if _parts:
                    quota_hint = "；".join(_parts) + "；附近若有该品类干净候选请【优先接】(省大额罚款)。"
        except Exception:
            quota_hint = ""

        cap_hint = ""
        try:
            _cm2 = _now_md(now_wall)
            if _cm2:
                _cparts = []
                for cap in ir.get("monthly_count_caps") or []:
                    field = str(cap.get("field") or "").strip()
                    thr = cap.get("threshold")
                    maxm = cap.get("max_per_month")
                    if not field or thr is None or maxm is None:
                        continue
                    done = self._memory.monthly_predicate_count(driver_id, field, float(thr), _cm2[0])
                    rem = int(maxm) - done
                    _cparts.append(
                        f"本月此类单已接{done}/{int(maxm)}，"
                        + ("已满→别再接(violates 已标)" if rem <= 0 else f"还可接{rem}单，超了每单扣钱")
                    )
                if _cparts:
                    cap_hint = "；".join(_cparts)
        except Exception:
            cap_hint = ""

        compliance = {
            "今日休息块(到点会被自动安排休息)": directive.get("rest_block_today"),
            "区域配额": region_hint,
            "品类配额(本月必须接满某品类单数)": quota_hint,
            "月度单数上限(超就扣钱)": cap_hint,
            "已识别硬约束原文": [f.get("raw_text") for f in ir.get("order_filters", [])]
            + [w.get("raw_text") for w in ir.get("rest_windows", [])],
            "提示": "候选 violates 非空=别接；接单前看预计完成时刻，别接会跨进休息块的单(否则会被否决)",
        }
        cand_by_id = {c["cargo_id"]: c for c in candidates}
        cand_ids = set(cand_by_id)

        messages = prompts.build_messages(
            status, candidates, mem_summary, now_wall, month_end_wall, directive, compliance
        )

        # 决策 + 确定性否决重试（最多 3 次尝试）：代码不替偏好做选择，只否决违规并退回 LLM
        action = None
        for _ in range(3):
            obj = llm.chat_json(api, messages)
            cand = guardrails.parse_and_validate(obj, cand_ids) if obj is not None else None
            if cand is None:
                break
            reason = checker.veto(cand, cand_by_id, windows, now_min)
            if reason is None:
                action = cand
                break
            messages.append({"role": "assistant", "content": json.dumps(cand, ensure_ascii=False)})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"上个动作被否决：{reason}。请改选不违反偏好的动作"
                        f"（violates 非空的候选别接；若在休息时段，请用一次长 wait 睡到 "
                        f"{compliance.get('建议一次长wait睡到(墙钟)') or '休息时段结束'}）。只输出 JSON。"
                    ),
                }
            )

        # 休息时段：确定性保证"一次长 wait 覆盖到块结束"。评分要求【连续】休息，
        # 多次短 wait 会被中间的 query 扫描分钟切碎、永远凑不满，故在窗内强制单次长 wait。
        # （休息时段与时长均来自 LLM 编译的偏好；代码只负责可靠执行，不决定是否/何时休息。）
        if inside and end_abs and end_abs > now_min:
            needed = min(end_abs - now_min, guardrails.MAX_WAIT_MINUTES)
            cur_wait = (
                int(action["params"].get("duration_minutes", 0))
                if action and action.get("action") == "wait"
                else 0
            )
            if cur_wait < needed:
                action = {"action": "wait", "params": {"duration_minutes": needed}}

        # 反闲置：LLM 想干等，但本地有【不违规 + 正利润 + 不跨休息块】的可接货 → 改接最赚的(别白等一天)。
        # 纯经济优化(偏好已由 violates/veto 过滤、ROI 正即真赚钱)；整休日不触发；不含任何偏好常量。
        _special = str(directive.get("today_special_task") or "").strip()
        _has_special = _special and _special != "无"
        if (
            action
            and action.get("action") == "wait"
            and not directive.get("today_is_off_day")
            and not _has_special  # 当日有专属日程 → 让位给 LLM/义务执行，别瞎接单错过日程
            and not checker.has_dated_restriction_today(ir, _now_md(now_wall))  # 日期条件规避日 → 让位给 LLM 整体规避
        ):
            takeable = [
                c
                for c in candidates
                if not c.get("violates")
                and float(c.get("roi_score") or 0) > 0
                and not checker.overlaps_any_window(now_min, int(c.get("est_finish_min") or now_min), windows)
            ]
            if takeable:
                best = max(takeable, key=lambda c: float(c.get("roi_score") or 0))
                self._logger.info(
                    "anti-idle 改接 driver=%s %s roi=%.0f", driver_id, best["cargo_id"], float(best.get("roi_score") or 0)
                )
                action = {"action": "take_order", "params": {"cargo_id": best["cargo_id"]}}

        if action is None:
            action = guardrails.fallback_action()
            self._logger.warning("LLM 多次违规/无效，确定性兜底 wait: driver=%s", driver_id)

        self._logger.info(
            "decide driver=%s t=%s cand=%s -> %s %s",
            driver_id,
            now_min,
            len(candidates),
            action.get("action"),
            action.get("params"),
        )
        return action

    def _build_shadow_map(
        self, driver_id: str, ir: dict[str, Any], now_wall: str, month_end_min: int, now_min: int
    ) -> dict[str, float]:
        """配额影子价格表 {真实品类名: 边际罚款}。只给【落后且按配速来不及】的配额加价——
        来得及的配额货按自然时薪竞争(顺路凑,零额外成本)。品类名经等值对齐(评分按完全相等计数)；
        无法对齐(unmapped)不强制履约。全部数值来自 LLM 编译 IR + 客观计数,零偏好常量。绝不抛异常。"""
        out: dict[str, float] = {}
        try:
            cmd = _now_md(now_wall)
            if not cmd:
                return out
            cur_month = cmd[0]
            days_passed = max(0, int(cmd[1]) - 1)
            days_left = max(0, (month_end_min - now_min) // 1440)
            known = self._memory.cargo_names(driver_id)
            for q in ir.get("category_quotas") or []:
                if int(q.get("month") or 0) != cur_month:
                    continue
                cat = str(q.get("category") or "").strip()
                need = int(q.get("min_orders") or 0)
                if not cat or need <= 0:
                    continue
                done = self._memory.category_orders_done(driver_id, cat, cur_month)
                remaining = need - done
                if remaining <= 0:
                    continue
                projected = (done / max(1, days_passed)) * days_left
                if projected >= remaining and days_left > remaining + 2:
                    continue  # 按当前节奏来得及 → 不加价,顺路自然竞争
                snapped, st = tools.snap_category(cat, known)
                if st == "unmapped":
                    continue  # 无法对齐到真实品类名 → 不强制(原文仍在偏好提示里)
                ppu = float(q.get("penalty_per_unit") or 0.0)
                out[snapped] = min(ppu if ppu > 0 else SHADOW_CAP, SHADOW_CAP)
        except Exception as e:
            self._logger.warning("shadow_map 构造失败: %s", e)
        return out

    def _safe_broad_query(
        self,
        driver_id: str,
        status: dict[str, Any],
        ir: dict[str, Any],
        now_min: int,
        now_wall: str,
        month_end_min: int,
        shadow_map: dict[str, float],
        windows: list[Any],
        candidates: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        """R2 月末安全广查：返回(可并入打分的配额货候选, 刷新后的now_min)。
        触发条件全部满足才查：①该配额月末告急(days_left≤remaining+SAFE_BQ_DAYS)
        ②本地候选无该品类干净货 ③距下一休息窗 ≥ 本月单均占用时长(傍晚熔断)
        ④今天该品类没查过。广查后刷新时钟；结果只并入候选(影子价格自然胜出)，不强制接。绝不抛异常。"""
        try:
            cmd = _now_md(now_wall)
            if not cmd:
                return [], now_min
            cur_month, day_of_month = cmd[0], cmd[1]
            days_left = max(0, (month_end_min - now_min) // 1440)
            cur_day = now_min // 1440
            known = self._memory.cargo_names(driver_id)
            # 傍晚熔断：距下一休息窗的分钟数 < 典型单周期(本月完单均占用,无样本则240) → 不查
            day0 = cur_day * 1440
            tod = now_min - day0
            nxt_win = min(
                (day0 + (s if s > tod else s + 1440) for s, e in windows or []),
                default=now_min + 100000,
            )
            typical = 240
            if nxt_win - now_min < typical + 60:
                return [], now_min
            local_names = {str(c.get("cargo_name") or "") for c in candidates if not c.get("violates")}
            for q in ir.get("category_quotas") or []:
                if int(q.get("month") or 0) != cur_month:
                    continue
                cat = str(q.get("category") or "").strip()
                snapped, st = tools.snap_category(cat, known)
                if st == "unmapped" or snapped not in shadow_map:
                    continue  # 没告急(影子未激活)或对不齐的不广查
                need = int(q.get("min_orders") or 0)
                remaining = need - self._memory.category_orders_done(driver_id, cat, cur_month)
                if remaining <= 0 or days_left > remaining + SAFE_BQ_DAYS:
                    continue  # 只在月末窗口动用大查询
                if snapped in local_names:
                    continue  # 本地已有该品类干净货,argmax 自己会选
                bqkey = (driver_id, cat)
                if self._last_bq.get(bqkey) == cur_day:
                    continue  # 每天每品类最多一次
                self._last_bq[bqkey] = cur_day
                lat = float(status["current_lat"])
                lng = float(status["current_lng"])
                resp = self._api.query_cargo(driver_id=driver_id, latitude=lat, longitude=lng, k=600)
                # 广查推进时钟(最多60min)——立刻刷新,防完成时刻估算乐观导致压线溢入夜休
                now_min = int(
                    self._api.get_driver_status(driver_id).get("simulation_progress_minutes") or now_min
                )
                wide = tools.prepare_candidates(
                    resp.get("items", []) or [], now_min, month_end_min, top_n=600, quota_categories=[cat]
                )
                checker.annotate(wide, ir, cmd)
                extra = [
                    c
                    for c in wide
                    if str(c.get("cargo_name") or "") == snapped
                    and not c.get("violates")
                    and not checker.overlaps_any_window(now_min, int(c.get("est_finish_min") or now_min), windows)
                ][:8]
                for c in extra:
                    self._memory.note_candidate(driver_id, c["cargo_id"], c)
                self._logger.info(
                    "R2 安全广查 driver=%s 品类=%s 还差=%s days_left=%s 找到干净货=%s",
                    driver_id, cat, remaining, days_left, len(extra),
                )
                return extra, now_min  # 每步最多广查一个品类(控制时间税)
            return [], now_min
        except Exception as e:
            self._logger.warning("安全广查失败: %s", e)
            return [], now_min

    def _wait_action(self, now_min: int, windows: list[Any]) -> dict[str, Any]:
        """无正分候选时的确定性等待：睡到 min(下一休息窗开始, 当天结束, 再查间隔)。
        纯时间算术；休息窗来自 LLM 编译。"""
        day = now_min // 1440
        tod = now_min - day * 1440
        cands = [now_min + REQUERY_MIN, (day + 1) * 1440]
        for s, e in windows or []:
            cands.append(day * 1440 + (s if s > tod else s + 1440))
        until = min(cands)
        dur = max(1, min(until - now_min, guardrails.MAX_WAIT_MINUTES))
        return {"action": "wait", "params": {"duration_minutes": dur}}

    def _llm_veto(self, api: Any, c: dict[str, Any], prefs_text: list[str]) -> bool:
        """hybrid 模式的单票否决：LLM 通读全部偏好原文,只回答"接这单是否违反任何偏好"。
        权限棘轮——LLM 只能否决,不能改选其他单。失败(None)按不否决处理(下界=argmax)。"""
        try:
            q = {
                "候选": {
                    "品类": c.get("cargo_name"),
                    "起点": (c.get("start") or {}).get("city"),
                    "终点": (c.get("end") or {}).get("city"),
                    "赴装空驶km": c.get("deadhead_km"),
                    "干线分钟": c.get("cost_time_minutes"),
                    "预计完成时刻": c.get("预计完成时刻"),
                },
                "司机偏好原文": prefs_text,
                "问题": '只判断:接这一单是否会违反上述任一偏好?只输出 JSON {"veto":true/false,"reason":"一句话"}',
            }
            obj = llm.chat_json(
                api, [{"role": "user", "content": json.dumps(q, ensure_ascii=False)}], max_tokens=120
            )
            return bool(obj and obj.get("veto"))
        except Exception:
            return False

    def _pending_task_days(self, driver_id: str, directive: dict[str, Any], now_min: int) -> set[int]:
        """未完成日期任务所在的日序号集合(供候选标违规,防接单溢入专属日程日)。绝不抛异常。"""
        out: set[int] = set()
        try:
            from datetime import datetime

            today = now_min // 1440
            for t in directive.get("dated_tasks") or []:
                md = _parse_md(t.get("date", ""))
                if not md:
                    continue
                try:
                    d = (datetime(2026, md[0], md[1]) - tools.EPOCH).days
                except ValueError:
                    continue
                if d < today:
                    continue
                key = "%s@%.4f,%.4f" % (t["date"], t["lat"], t["lng"])
                if self._memory.is_dated_done(driver_id, key):
                    continue
                out.add(d)
        except Exception as e:
            self._logger.warning("pending_task_days 失败: %s", e)
        return out

    def _obligation_action(
        self,
        driver_id: str,
        status: dict[str, Any],
        now_min: int,
        now_wall: str,
        month_end_min: int,
        directive: dict[str, Any],
    ) -> dict[str, Any] | None:
        """确定性执行【日期任务】(到坐标停留) + 【临近期限的区域配额】(去该地找货)。
        坐标/日期/地名/天数全部来自 LLM 对偏好的编译；代码只在该去的日子可靠执行。绝不抛异常。"""
        try:
            lat = float(status["current_lat"])
            lng = float(status["current_lng"])
            cmd = _now_md(now_wall)
            # 1) 当日日期任务：先到坐标，再停留要求时长
            for t in directive.get("dated_tasks") or []:
                if _parse_md(t.get("date", "")) != cmd or cmd is None:
                    continue
                key = "%s@%.4f,%.4f" % (t["date"], t["lat"], t["lng"])
                if self._memory.is_dated_done(driver_id, key):
                    continue
                if tools.haversine_km(lat, lng, t["lat"], t["lng"]) > 1.5:
                    self._logger.info("dated_task 前往 driver=%s %s", driver_id, t)
                    return {"action": "reposition", "params": {"latitude": t["lat"], "longitude": t["lng"]}}
                self._memory.mark_dated_done(driver_id, key)
                dur = min(max(int(t.get("wait_minutes") or 0), 1) + 10, guardrails.MAX_WAIT_MINUTES)
                self._logger.info("dated_task 到达停留 driver=%s wait=%s", driver_id, dur)
                return {"action": "wait", "params": {"duration_minutes": dur}}
            # 2) 区域配额：临近期限仍未达标 → 去该地找货（在那儿由 LLM 接含该地名的货）
            rt = directive.get("region_target")
            if isinstance(rt, dict) and rt.get("need_days", 0) > 0:
                done = self._memory.region_days_done(driver_id, rt["keyword"])
                remaining = rt["need_days"] - len(done)
                today_idx = now_min // 1440
                days_left = max(0, (month_end_min - now_min) // 1440)
                if remaining > 0 and today_idx not in done and remaining >= days_left - 1:
                    if tools.haversine_km(lat, lng, rt["lat"], rt["lng"]) > 3.0:
                        self._logger.info(
                            "region_quota 紧急前往 driver=%s remaining=%s days_left=%s", driver_id, remaining, days_left
                        )
                        return {"action": "reposition", "params": {"latitude": rt["lat"], "longitude": rt["lng"]}}
            return None
        except Exception as e:
            self._logger.warning("obligation 计算失败: %s", e)
            return None

    def _active_quota_categories(self, driver_id: str, ir: dict[str, Any], now_wall: str) -> list[str]:
        """本月仍未达标的品类配额关键词(供候选 surface)。品类/单数/月份来自 LLM 编译，零偏好常量。"""
        cats: list[str] = []
        try:
            cmd = _now_md(now_wall)
            if not cmd:
                return cats
            cur_month = cmd[0]
            for q in ir.get("category_quotas") or []:
                if int(q.get("month") or 0) != cur_month:
                    continue
                cat = str(q.get("category") or "").strip()
                if not cat:
                    continue
                done = self._memory.category_orders_done(driver_id, cat, cur_month)
                if int(q.get("min_orders") or 0) - done > 0:
                    cats.append(cat)
        except Exception as e:
            self._logger.warning("active_quota_categories 失败: %s", e)
        return cats

    def _quota_action(
        self,
        driver_id: str,
        status: dict[str, Any],
        ir: dict[str, Any],
        now_min: int,
        now_wall: str,
        month_end_min: int,
        candidates: list[dict[str, Any]],
        windows: list[Any],
        off_set: set[int],
    ) -> dict[str, Any] | None:
        """确定性履约【月度品类配额】：本月必须接满 N 单某品类。
        有干净(不违规/不跨休息/不落整休日)的该品类候选就接最赚的一单；本地没有且落后节奏 → 广查抓该品类货。
        品类/单数/月份全部来自 LLM 编译；代码只做计数 + 文本匹配 + 接单。绝不抛异常。"""
        try:
            quotas = ir.get("category_quotas") or []
            if not quotas:
                return None
            cmd = _now_md(now_wall)
            if not cmd:
                return None
            cur_month = cmd[0]
            active: list[tuple[dict[str, Any], str, int]] = []
            for q in quotas:
                if int(q.get("month") or 0) != cur_month:
                    continue
                cat = str(q.get("category") or "").strip()
                need = int(q.get("min_orders") or 0)
                if not cat or need <= 0:
                    continue
                remaining = need - self._memory.category_orders_done(driver_id, cat, cur_month)
                if remaining > 0:
                    active.append((q, cat, remaining))
            if not active:
                return None
            # 少一单罚款越高越先履约(不可补救/罚额高的配额优先)
            active.sort(key=lambda x: float(x[0].get("penalty_per_unit") or 0), reverse=True)
            days_left = max(0, (month_end_min - now_min) // 1440)
            # 长途阈值(若有月度长途上限)：配额履约优先挑【非长途】的该品类货，别白烧长途配额
            lh_thr = None
            for cap in ir.get("monthly_count_caps") or []:
                if str(cap.get("field") or "") == "cost_time_minutes" and cap.get("threshold") is not None:
                    try:
                        lh_thr = float(cap["threshold"])
                        break
                    except (TypeError, ValueError):
                        pass

            # 配额关键词等值对齐：评分按 cargo_name 与品类【完全相等】计数，履约判定必须同口径，
            # 否则把超串品类(如"其他X")的单计入配额→自以为凑满、评分照罚。
            known = self._memory.cargo_names(driver_id)
            snap_map = {cat: tools.snap_category(cat, known) for _q, cat, _r in active}

            def _clean(c: dict[str, Any], cat: str) -> bool:
                # 配额货与普通单同用 60min 防撞余量(不再压线)：官方读数显示压线/高力度履约
                # 在隐藏司机上的副作用罚款 ≫ 它省下的配额罚款，宁可少凑一单不赌作息。
                name = str(c.get("cargo_name") or "")
                snapped, st = snap_map.get(cat) or (cat, "unmapped")
                matched = (name == snapped) if st != "unmapped" else (cat in name)
                if not matched or c.get("violates"):
                    return False
                est = int(c.get("est_finish_min") or now_min)
                if checker.overlaps_any_window(now_min, est, windows):
                    return False
                if off_set and (est // 1440) in off_set:
                    return False
                return True

            for q, cat, remaining in active:
                # 仅【顺路履约】：本地候选里有干净的该品类货 → 接最赚的一单。
                # 不再广查 k=600 / 不再压线——官方探针分解显示高力度履约(广查/压线/迁移)在
                # 隐藏司机上的副作用罚款(+40k~60k)远超省下的配额罚款；顺路履约保留了
                # 履约的全部"无副作用部分"(零额外时间税/零额外里程/不赌作息)。
                local = [c for c in candidates if _clean(c, cat)]
                if local:
                    best = self._pick_quota(local, lh_thr)
                    self._logger.info(
                        "quota 顺路履约 driver=%s 品类=%s 还差=%s 接=%s roi=%.0f",
                        driver_id, cat, remaining, best["cargo_id"], float(best.get("roi_score") or 0),
                    )
                    return {"action": "take_order", "params": {"cargo_id": best["cargo_id"]}}
            return None
        except Exception as e:
            self._logger.warning("quota 履约失败: %s", e)
            return None

    def _mark_monthly_caps(
        self, driver_id: str, ir: dict[str, Any], now_wall: str, candidates: list[dict[str, Any]]
    ) -> None:
        """月度单数上限：本月已接达上限的那类单(如 >阈值的长途)，把后续命中候选标 violates。
        阈值/上限来自 LLM 编译；代码只做数值比较 + 计数。绝不抛异常。"""
        try:
            caps = ir.get("monthly_count_caps") or []
            if not caps:
                return
            cmd = _now_md(now_wall)
            if not cmd:
                return
            cur_month = cmd[0]
            for cap in caps:
                field = str(cap.get("field") or "").strip()
                thr = cap.get("threshold")
                maxm = cap.get("max_per_month")
                if not field or thr is None or maxm is None:
                    continue
                done = self._memory.monthly_predicate_count(driver_id, field, float(thr), cur_month)
                if done < int(maxm):
                    continue  # 还没到上限，允许接
                for c in candidates:
                    try:
                        if float(c.get(field) or 0) > float(thr):
                            c.setdefault("violates", []).append("本月该类单数已达上限")
                    except (TypeError, ValueError):
                        pass
        except Exception as e:
            self._logger.warning("monthly_caps 标注失败: %s", e)

    @staticmethod
    def _pick_quota(cands: list[dict[str, Any]], lh_thr: float | None) -> dict[str, Any]:
        """配额履约选单：先排除长途(优先非长途，别白烧长途配额)，再挑【赴装空驶最近】的一单。
        配额罚款(数百~数千/单)远大于单票利润差，就近凑最省里程成本——避免为凑配额开很远去接货。"""
        pool = cands
        if lh_thr is not None:
            short = [c for c in cands if float(c.get("cost_time_minutes") or 0) <= lh_thr]
            if short:
                pool = short
        return min(pool, key=lambda c: float(c.get("deadhead_km") if c.get("deadhead_km") is not None else 1e9))
