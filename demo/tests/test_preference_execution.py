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


if __name__ == "__main__":
    unittest.main()
