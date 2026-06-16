from __future__ import annotations

import sys
import unittest
from pathlib import Path


DEMO_ROOT = Path(__file__).resolve().parents[1]
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

from agent.model_decision_service import CandidateFact, ModelDecisionService


def candidate(
    *,
    cargo_name: str = "测试货",
    pickup_km: float = 20.0,
    transport_min: int = 180,
    finish_min: int = 600,
    net: float = 2000.0,
) -> CandidateFact:
    return CandidateFact(
        cargo_id="C1",
        cargo={
            "cargo_id": "C1",
            "cargo_name": cargo_name,
            "cost_time_minutes": transport_min,
            "start": {"lat": 22.5, "lng": 113.5, "city": "起点城"},
            "end": {"lat": 23.0, "lng": 114.0, "city": "终点城"},
        },
        source="test",
        query_km=pickup_km,
        pickup_km=pickup_km,
        haul_km=100.0,
        pickup_min=20,
        wait_min=0,
        transport_min=transport_min,
        finish_min=finish_min,
        price_yuan=3000.0,
        cost_yuan=1000.0,
        net_yuan_before_pref=net,
        net_per_hour_before_pref=net / ((20 + transport_min) / 60),
        near_end_cargo_seen=0,
        legal=True,
    )


def policy_with_order_rules(*rules: dict) -> dict:
    return {"machine_ir": {"order_rules": list(rules)}}


class GenericOrderRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ModelDecisionService(api=None)
        self.ledger: dict = {}

    def test_matching_penalty_rule_reduces_score_by_exact_penalty(self) -> None:
        cand = candidate(pickup_km=80)
        base = self.service._candidate_score(cand, {}, self.ledger, 0)
        policy = policy_with_order_rules(
            {
                "field": "deadhead_km",
                "op": "gt",
                "value": 55,
                "effect": "penalty",
                "penalty_amount": 120,
            }
        )

        adjusted = self.service._candidate_score(cand, policy, self.ledger, 0)
        active_hours = (cand.pickup_min + cand.wait_min + cand.transport_min) / 60

        self.assertEqual(base - adjusted, 120 + 5 * 120 / active_hours)

    def test_matching_hard_rule_is_a_deterministic_veto(self) -> None:
        cand = candidate(cargo_name="禁接货")
        policy = policy_with_order_rules(
            {
                "field": "cargo_name",
                "op": "contains",
                "value": "禁接",
                "effect": "hard_veto",
            }
        )

        vetoes = self.service._deterministic_vetoes(cand, policy, self.ledger)

        self.assertIn("order_rule_hard_veto", vetoes)

    def test_non_matching_rule_does_not_change_candidate(self) -> None:
        cand = candidate(pickup_km=20)
        base = self.service._candidate_score(cand, {}, self.ledger, 0)
        policy = policy_with_order_rules(
            {
                "field": "deadhead_km",
                "op": "gt",
                "value": 55,
                "effect": "penalty",
                "penalty_amount": 120,
            }
        )

        adjusted = self.service._candidate_score(cand, policy, self.ledger, 0)

        self.assertEqual(base, adjusted)

    def test_audit_removes_duplicate_region_hard_ban_when_penalty_rule_exists(self) -> None:
        quote = "卸货地在目标城的货每接一次扣钱"
        policy = {
            "machine_ir": {
                "region_avoid": [{"field": "destination", "keyword": "目标城", "source_quote": quote}],
                "order_rules": [
                    {
                        "field": "end_region",
                        "op": "contains",
                        "value": "目标城",
                        "effect": "penalty",
                        "penalty_amount": 300,
                        "source_quote": quote,
                    }
                ],
            }
        }

        audited = self.service._audit_policy("DTEST", policy, [{"content": quote}])

        self.assertEqual(audited["machine_ir"]["region_avoid"], [])
        self.assertEqual(len(audited["machine_ir"]["order_rules"]), 1)

    def test_audit_demotes_order_rule_without_source_quote(self) -> None:
        policy = {
            "machine_ir": {
                "order_rules": [
                    {
                        "field": "deadhead_km",
                        "op": "gt",
                        "value": 50,
                        "effect": "hard_veto",
                    }
                ]
            }
        }

        audited = self.service._audit_policy("DTEST", policy, [{"content": "赴装货空驶太远的不接"}])

        self.assertEqual(audited["machine_ir"]["order_rules"], [])
        self.assertEqual(audited["machine_ir"]["unknown_constraints"][0]["risk_level"], "high")

    def test_audit_removes_per_order_penalty_duplicate_of_monthly_count_limit(self) -> None:
        quote = "每月超过八小时的长途最多五单，多一单扣一次"
        policy = {
            "machine_ir": {
                "long_haul_limits": [
                    {
                        "threshold_minutes": 480,
                        "max_count": 5,
                    }
                ],
                "order_rules": [
                    {
                        "field": "transport_minutes",
                        "op": "gt",
                        "value": 480,
                        "effect": "penalty",
                        "penalty_amount": 1000,
                        "source_quote": quote,
                    }
                ],
            }
        }

        audited = self.service._audit_policy("DTEST", policy, [{"content": quote}])

        self.assertEqual(audited["machine_ir"]["order_rules"], [])

    def test_audit_removes_order_rule_that_conflicts_with_minimum_target(self) -> None:
        quote = "目标货必须接满十二单，少一单扣一次"
        policy = {
            "machine_ir": {
                "cargo_targets": [
                    {"month": 4, "cargo_name": "目标货", "min_count": 12, "source_quote": quote}
                ],
                "order_rules": [
                    {
                        "field": "cargo_name",
                        "op": "equals",
                        "value": "目标货",
                        "effect": "penalty",
                        "penalty_amount": 500,
                        "source_quote": quote,
                    }
                ],
            }
        }

        audited = self.service._audit_policy("DTEST", policy, [{"content": quote}])

        self.assertEqual(audited["machine_ir"]["order_rules"], [])


class RestWindowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ModelDecisionService(api=None)
        self.policy = {
            "machine_ir": {
                "rest_windows": [
                    {"days": "all", "start_hour": 21, "end_hour": 6},
                    {"days": "weekend", "start_hour": 23, "end_hour": 6},
                ]
            }
        }

    def test_weekend_specific_window_overrides_all_days_window(self) -> None:
        intervals = self.service._rest_intervals_around(0, 24 * 60, self.policy)

        self.assertIn((23 * 60, 30 * 60), intervals)
        self.assertNotIn((21 * 60, 30 * 60), intervals)

    def test_weekday_uses_all_days_window(self) -> None:
        intervals = self.service._rest_intervals_around(24 * 60, 48 * 60, self.policy)

        self.assertIn((24 * 60 + 21 * 60, 48 * 60 + 6 * 60), intervals)


class HomeCurfewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ModelDecisionService(api=None)
        self.service._initial_position_by_driver["DTEST"] = (22.0, 113.0)
        self.policy = {
            "machine_ir": {
                "home_curfews": [
                    {
                        "home_lat": 22.0,
                        "home_lng": 113.0,
                        "radius_km": 1.0,
                        "deadline_hour": 21,
                        "quiet_until_hour": 6,
                        "forbid_take_order": True,
                        "forbid_reposition": True,
                    }
                ]
            }
        }

    def test_candidate_that_cannot_return_before_deadline_is_vetoed(self) -> None:
        cand = candidate(finish_min=20 * 60 + 30)

        vetoes = self.service._deterministic_vetoes(cand, self.policy, {})

        self.assertIn("home_curfew_no_return", vetoes)

    def test_executor_repositions_home_at_latest_safe_departure(self) -> None:
        status = {
            "driver_id": "DTEST",
            "simulation_progress_minutes": 19 * 60,
            "current_lat": 23.0,
            "current_lng": 113.0,
        }

        action = self.service._home_curfew_decision("DTEST", status, self.policy)

        self.assertEqual(action, {"action": "reposition", "params": {"latitude": 22.0, "longitude": 113.0}})

    def test_executor_waits_through_quiet_interval_when_home(self) -> None:
        status = {
            "driver_id": "DTEST",
            "simulation_progress_minutes": 22 * 60,
            "current_lat": 22.0,
            "current_lng": 113.0,
        }

        action = self.service._home_curfew_decision("DTEST", status, self.policy)

        self.assertEqual(action, {"action": "wait", "params": {"duration_minutes": 8 * 60}})


class TargetUrgencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ModelDecisionService(api=None)
        self.policy = {
            "machine_ir": {
                "cargo_targets": [
                    {"month": 4, "cargo_name": "测试货", "min_count": 12, "penalty_amount": 500}
                ]
            }
        }
        self.ledger = {"cargo_name_counts_by_month": {"2026-04": {"测试货": 11}}}

    def test_monthend_shortfall_has_more_value_than_early_shortfall(self) -> None:
        early_april = 31 * 1440
        late_april = (31 + 27) * 1440

        early = self.service._target_bonus(candidate(), self.policy, self.ledger, early_april)
        late = self.service._target_bonus(candidate(), self.policy, self.ledger, late_april)

        self.assertGreater(late, early)

    def test_target_bonus_returns_penalty_avoided_tuple(self) -> None:
        # §7 配额熔断接口: return_penalty_avoided=True 返回 (bonus, pen_avoided);
        # pen_avoided = 匹配欠额目标的单位罚额(接1单省的罚金)。默认仍返回 float(向后兼容)。
        result = self.service._target_bonus(
            candidate(cargo_name="测试货"), self.policy, self.ledger, 40 * 1440,
            return_penalty_avoided=True,
        )
        self.assertIsInstance(result, tuple)
        bonus, pen_avoided = result
        self.assertGreater(bonus, 0.0)
        self.assertEqual(pen_avoided, 500.0)
        self.assertIsInstance(
            self.service._target_bonus(candidate(cargo_name="测试货"), self.policy, self.ledger, 40 * 1440),
            float,
        )

    def test_quota_fuse_refuses_net_negative_keeps_net_positive(self) -> None:
        # §7 决策算术: 靠配额bonus才被选(tb>0)且接它净亏(net+min(bonus,pen_avoided)<0)→认罚不接;
        # 货源充足品类(净正)恒 net+eff>=0→不触发→不伤已履约配额。
        tb, pen_avoided = self.service._target_bonus(
            candidate(cargo_name="测试货"), self.policy, self.ledger, 40 * 1440,
            return_penalty_avoided=True,
        )
        self.assertGreater(tb, 0.0)
        eff = min(tb, pen_avoided)
        self.assertLess(-3000.0 + eff, 0.0)      # 净亏单 → 认罚不接
        self.assertGreaterEqual(2000.0 + eff, 0.0)  # 净正单 → 保留

    def test_quota_fuse_does_not_fire_without_shortfall(self) -> None:
        # 配额已满(无欠额)→ tb=0、pen_avoided=0 → §7 闸门不触发(tb>0 为前置条件)。
        ledger_full = {"cargo_name_counts_by_month": {"2026-04": {"测试货": 12}}}
        tb, pen_avoided = self.service._target_bonus(
            candidate(cargo_name="测试货"), self.policy, ledger_full, 40 * 1440,
            return_penalty_avoided=True,
        )
        self.assertEqual(tb, 0.0)
        self.assertEqual(pen_avoided, 0.0)


class GuardedRepositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ModelDecisionService(api=None)
        self.status = {"current_lat": 22.5, "current_lng": 113.5}

    def _run(self, machine_ir: dict):
        from agent.model_decision_service import FEATURE_FLAGS
        orig = FEATURE_FLAGS["ltd_reposition"]
        FEATURE_FLAGS["ltd_reposition"] = True
        try:
            return self.service._limited_reposition("D", self.status, 5000, {"machine_ir": machine_ir})
        finally:
            FEATURE_FLAGS["ltd_reposition"] = orig

    def test_guard_blocks_driver_with_off_day(self) -> None:
        # 有整歇日约束 → 即便 flag 开也恒 no-op(防迁移撞整歇日, §9.4)。
        self.assertIsNone(self._run({"off_day_requirements": [{"min_off_days": 2}]}))

    def test_guard_blocks_driver_with_home_curfew(self) -> None:
        # 有 home 门禁约束 → 即便 flag 开也恒 no-op(防迁移撞门禁, §9.4)。
        self.assertIsNone(self._run({"home_curfews": [{"home_lat": 22.5, "home_lng": 113.5, "deadline_hour": 22}]}))

    def test_guard2_blocks_inside_rest_window_but_fires_in_daytime(self) -> None:
        # gate-fix: 查货扫描推进 sim 进作息窗后, ltd 绝不迁移(防窗内空驶=夜休违规); 安全时段才迁移。
        from agent.model_decision_service import FEATURE_FLAGS
        svc = self.service
        svc._observed_points_by_driver["D"] = [{"lat": 23.0, "lng": 113.9, "price": 8000.0, "cargo_name": "x"}]
        pol = {"machine_ir": {"rest_windows": [
            {"label": "Night", "days": "weekday", "start_hour": 21, "end_hour": 6,
             "forbid_take_order": True, "forbid_reposition": True}]}}
        orig = FEATURE_FLAGS["ltd_reposition"]
        FEATURE_FLAGS["ltd_reposition"] = True
        try:
            in_win = 64 * 1440 + 22 * 60    # 2026-05-04(周一) 22:00, 在 21-6 窗内
            daytime = 64 * 1440 + 12 * 60   # 同日 12:00, 距 21:00 窗 540min>240
            self.assertIsNone(svc._limited_reposition("D", self.status, in_win, pol))   # guard② 拦截窗内
            res = svc._limited_reposition("D", self.status, daytime, pol)
            self.assertIsNotNone(res)                          # 安全时段应能迁移(证明拦截来自 guard②)
            self.assertEqual(res.get("action"), "reposition")
        finally:
            FEATURE_FLAGS["ltd_reposition"] = orig


if __name__ == "__main__":
    unittest.main()
