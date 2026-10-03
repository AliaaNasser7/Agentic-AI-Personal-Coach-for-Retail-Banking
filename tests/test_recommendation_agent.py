"""
Recommendation Agent — Unit Tests

Fixtures below mirror the RAW output shapes of the real agents exactly
(spending_agent_tools.analyze_customer(), GoalAgent.recommend(),
build_simulation_output()) -- no enrichment, no extra keys invented.
If a teammate changes a field name on their branch, a fixture here
should be the first thing to update, followed by contracts.py.

Covers:
    1. Normal recommendation (spending + goals + simulation all present, overspending flagged)
    2. Missing Spending result
    3. Missing Goals result
    4. Missing Simulation result
    5. All three results available (healthy customer, no overspending)
    6. Conflicting information (simulation says "ahead" but Goals Agent flags a gap this month)
    7. Insufficient information (all three missing)
    8. The additive ML priority layer

Run with (from the folder that CONTAINS recommendation_agent/):
    python -m unittest recommendation_agent.tests.test_recommendation_agent -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from recommendation_agent import run, run_recommendation_agent


SPENDING_WITH_OVERSPEND = {
    "customer_id": "CUSTX",
    "avg_monthly_income": 27700.0,
    "avg_monthly_spending": 22557.0,
    "monthly_surplus": 5143.0,
    "months_analyzed": 6,
    "overspending_categories": [
        {
            "month": "2026-04",
            "category": "Shopping",
            "month_amount": 3927.21,
            "avg_other_months": 1273.03,
            "vs_avg_pct": 208.5,
        }
    ],
    "analyzed_by": "Spending Agent",
}

SPENDING_NO_OVERSPEND = {
    "customer_id": "CUSTX",
    "avg_monthly_income": 27700.0,
    "avg_monthly_spending": 22557.0,
    "monthly_surplus": 5143.0,
    "months_analyzed": 6,
    "overspending_categories": [],
    "analyzed_by": "Spending Agent",
}

# -- Goals Agent fixtures: exact raw GoalAgent.recommend() shape --

_GOAL_NEEDS_ACTION = {
    "goal_id": "GOAL1",
    "goal_name": "Emergency Fund",
    "target_amount": 73200.0,
    "monthly_required": 12200.0,
    "disposable_income": 8524.6,
    "monthly_gap": 3675.4,
    "status": "needs_action",
    "suggestions": [
        {
            "transaction_id": "TX1",
            "description": "CINEMA TICKET PURCHASE",
            "category": "Entertainment",
            "original_amount": 1250.28,
            "tenor_months": 12,
            "fee_rate": 0.015,
            "monthly_installment": 122.94,
            "monthly_cash_freed": 1127.34,
            "contribution_to_goal_pct": 9.2,
        }
    ],
    "remaining_gap_after_suggestions": 2548.06,
    "customer_message": None,
}

_GOAL_ON_TRACK = {
    "goal_id": "GOAL1",
    "goal_name": "Emergency Fund",
    "target_amount": 73200.0,
    "monthly_required": 12200.0,
    "disposable_income": 13000.0,
    "monthly_gap": -800.0,
    "status": "on_track",
    "suggestions": [],
    "customer_message": None,
}

GOALS_RESULT = {
    "customer_id": "CUSTX",
    "customer_name": "Test Customer",
    "monthly_income": 27700.0,
    "analyzed_month": "2026-06",
    "goals": [_GOAL_NEEDS_ACTION],
}

GOALS_RESULT_ON_TRACK = {
    "customer_id": "CUSTX",
    "customer_name": "Test Customer",
    "monthly_income": 27700.0,
    "analyzed_month": "2026-06",
    "goals": [_GOAL_ON_TRACK],
}

# -- Simulation Agent fixtures: exact raw build_simulation_output() shape --

SIMULATION_NOT_ON_TRACK = {
    "agent": "simulation",
    "schema_version": "1.0",
    "customer_id": "CUSTX",
    "months_ahead": 6,
    "cash_flow": {"avg_monthly_income": 27700.0, "avg_monthly_spend": 22557.0, "avg_monthly_net": 5143.0},
    "baseline_projection": {
        "last_known_balance": 30000.0, "last_known_date": "2026-06-28",
        "avg_monthly_net": 5143.0, "projection": [{"date": "2026-07-28", "projected_balance": 35143.0}],
    },
    "goals": {
        "per_goal": [{
            "goal_name": "Emergency Fund",
            "target_amount": 73200.0,
            "planned_months": 6,
            "required_monthly": 12200.0,
            "allocated_monthly_savings": 546.09,
            "actual_months_needed": 134.0,
            "status": "behind",
        }],
        "overall": {
            "avg_monthly_saved": 546.09, "total_required_monthly": 12200.0,
            "status": "behind", "monthly_shortfall": 11653.91,
        },
    },
    "top_scenarios": [
        {"scenario": {"Shopping": -0.2}, "adjusted_avg_monthly_net": 7797.18,
         "projection": [], "monthly_gain": 2654.18}
    ],
    "top_products": {"flexible": [], "locked": []},
    "insight_text": "You're earning about 27,700 per month...",
    "analyzed_by": "Simulation Agent",
}

SIMULATION_ON_TRACK = {
    "agent": "simulation",
    "schema_version": "1.0",
    "customer_id": "CUSTX",
    "months_ahead": 6,
    "cash_flow": {"avg_monthly_income": 27700.0, "avg_monthly_spend": 14700.0, "avg_monthly_net": 13000.0},
    "baseline_projection": {
        "last_known_balance": 30000.0, "last_known_date": "2026-06-28",
        "avg_monthly_net": 13000.0, "projection": [{"date": "2026-07-28", "projected_balance": 43000.0}],
    },
    "goals": {
        "per_goal": [{
            "goal_name": "Emergency Fund",
            "target_amount": 73200.0,
            "planned_months": 6,
            "required_monthly": 12200.0,
            "allocated_monthly_savings": 13000.0,
            "actual_months_needed": 5.6,
            "status": "ahead",
        }],
        "overall": {
            "avg_monthly_saved": 13000.0, "total_required_monthly": 12200.0,
            "status": "on track", "monthly_shortfall": 0.0,
        },
    },
    # No meaningful cut opportunity for an already-healthy saver -- the
    # real rank_scenarios filters out non-improving results, so an empty
    # list here is realistic, not a stub gap.
    "top_scenarios": [],
    "top_products": {"flexible": [], "locked": []},
    "insight_text": "You're earning about 27,700 per month...",
    "analyzed_by": "Simulation Agent",
}


class TestNormalRecommendation(unittest.TestCase):
    """1. Normal case: all three inputs present, overspending exists,
    goal not on track -> Recommendation should tie them together,
    preferring the Goals Agent's own installment suggestion as the action."""

    def test_combines_spending_and_goal(self):
        result = run_recommendation_agent(SPENDING_WITH_OVERSPEND, GOALS_RESULT, SIMULATION_NOT_ON_TRACK)
        titles = [r["title"] for r in result["recommendations"]]
        self.assertTrue(any("Emergency Fund" in t for t in titles))
        goal_rec = result["recommendations"][0]
        self.assertIn("Emergency Fund", goal_rec["title"])
        self.assertIn("Entertainment", goal_rec["action"])
        self.assertFalse(result["goal_impact"]["on_track"])
        self.assertTrue(any("Shopping" in t for t in titles))
        self.assertEqual(result["customer_id"], "CUSTX")  # inferred from the input dicts

    def test_uses_real_numbers_not_invented(self):
        result = run_recommendation_agent(SPENDING_WITH_OVERSPEND, GOALS_RESULT, SIMULATION_NOT_ON_TRACK)
        rec = result["recommendations"][0]
        self.assertIn("12,200", rec["reason"])
        self.assertIn("73,200", rec["reason"])
        self.assertIn("1,127.34", rec["estimated_impact"])


class TestMissingSpending(unittest.TestCase):
    """2. Spending Agent failed/unavailable -> should still give a
    goal-based recommendation without inventing spending numbers."""

    def test_falls_back_to_goal_only(self):
        result = run_recommendation_agent(None, GOALS_RESULT, None)
        self.assertEqual(len(result["recommendations"]), 1)
        self.assertIn("Emergency Fund", result["recommendations"][0]["title"])
        self.assertFalse(result["used_inputs"]["spending"])


class TestMissingGoals(unittest.TestCase):
    """3. Goals Agent failed/unavailable -> should fall back to a
    spending-only or general recommendation, no goal mentioned."""

    def test_falls_back_to_spending_only(self):
        result = run_recommendation_agent(SPENDING_WITH_OVERSPEND, None, None)
        self.assertEqual(len(result["recommendations"]), 1)
        rec = result["recommendations"][0]
        self.assertIn("Shopping", rec["title"])
        self.assertIsNone(result["goal_impact"])
        self.assertFalse(result["used_inputs"]["goals"])

    def test_no_overspend_and_no_goals_gives_general_advice(self):
        result = run_recommendation_agent(SPENDING_NO_OVERSPEND, None, None)
        self.assertEqual(len(result["recommendations"]), 1)
        self.assertIn("Surplus", result["recommendations"][0]["title"])


class TestMissingSimulation(unittest.TestCase):
    """4. Simulation Agent failed/unavailable -> the Goals Agent's own
    disposable-income gap and installment suggestion mean a goal
    recommendation still works, just without a saving-history projection."""

    def test_goal_recommendation_without_projection(self):
        result = run_recommendation_agent(SPENDING_WITH_OVERSPEND, GOALS_RESULT, None)
        titles = [r["title"] for r in result["recommendations"]]
        self.assertTrue(any("Emergency Fund" in t for t in titles))
        self.assertIsNone(result["goal_impact"]["projected_months_to_reach"])
        self.assertIn("Entertainment", result["recommendations"][0]["action"])

    def test_no_overspending_gives_single_goal_recommendation(self):
        result = run_recommendation_agent(SPENDING_NO_OVERSPEND, GOALS_RESULT, None)
        self.assertEqual(len(result["recommendations"]), 1)
        self.assertIn("Emergency Fund", result["recommendations"][0]["title"])


class TestAllThreeAvailable(unittest.TestCase):
    """5. All three present, healthy customer (no overspending, goal
    already on track, no installment suggestions, no cut-scenario
    opportunity) -> should not invent a spending concern."""

    def test_healthy_customer_no_invented_concern(self):
        result = run_recommendation_agent(SPENDING_NO_OVERSPEND, GOALS_RESULT_ON_TRACK, SIMULATION_ON_TRACK)
        rec = result["recommendations"][0]
        self.assertNotIn("Shopping", str(rec))
        self.assertTrue(result["goal_impact"]["on_track"])


class TestConflictingInformation(unittest.TestCase):
    """6. Simulation says the goal is "ahead" based on saving history,
    but the Goals Agent still flags a gap for THIS specific month --
    Recommendation should surface both signals rather than let one
    silently override the other."""

    def test_conflicting_signals_both_surfaced(self):
        result = run_recommendation_agent(SPENDING_WITH_OVERSPEND, GOALS_RESULT, SIMULATION_ON_TRACK)
        self.assertTrue(result["goal_impact"]["on_track"])
        rec = result["recommendations"][0]
        self.assertIn("gap", rec["reason"].lower())
        self.assertIn("Shopping", str(result["recommendations"]))


class TestInsufficientInformation(unittest.TestCase):
    """7. All three missing -> clear 'not enough information' response,
    no crash, no fabricated recommendation."""

    def test_all_missing_returns_no_recommendation(self):
        result = run_recommendation_agent(None, None, None)
        self.assertEqual(result["recommendations"], [])
        self.assertIn("Not enough information", result["summary"])
        self.assertIsNone(result["goal_impact"])
        self.assertIsNone(result["priority"])


class TestMLPriorityLayer(unittest.TestCase):
    """8. The ML priority score/label is additive: it must never change
    which recommendations are produced or their order (that stays
    fully rule-based), only annotate the result with a data-driven
    urgency read."""

    def test_priority_present_and_well_formed_when_data_available(self):
        result = run_recommendation_agent(SPENDING_WITH_OVERSPEND, GOALS_RESULT, SIMULATION_NOT_ON_TRACK)
        priority = result["priority"]
        self.assertIsNotNone(priority)
        self.assertGreaterEqual(priority["score"], 0)
        self.assertLessEqual(priority["score"], 100)
        self.assertIn(priority["label"], ("High", "Medium", "Low"))

    def test_priority_does_not_change_recommendation_content(self):
        with_ml = run_recommendation_agent(SPENDING_WITH_OVERSPEND, GOALS_RESULT, SIMULATION_NOT_ON_TRACK)

        import recommendation_agent.core as core
        original_predict = core.predict_priority
        core.predict_priority = lambda features: None
        try:
            without_ml = run_recommendation_agent(SPENDING_WITH_OVERSPEND, GOALS_RESULT, SIMULATION_NOT_ON_TRACK)
        finally:
            core.predict_priority = original_predict

        self.assertIsNone(without_ml["priority"])
        self.assertEqual(with_ml["recommendations"], without_ml["recommendations"])
        self.assertEqual(with_ml["goal_impact"], without_ml["goal_impact"])

    def test_healthy_customer_scores_lower_than_urgent_customer(self):
        urgent = run_recommendation_agent(SPENDING_WITH_OVERSPEND, GOALS_RESULT, SIMULATION_NOT_ON_TRACK)
        healthy = run_recommendation_agent(SPENDING_NO_OVERSPEND, GOALS_RESULT_ON_TRACK, SIMULATION_ON_TRACK)
        self.assertLess(healthy["priority"]["score"], urgent["priority"]["score"])


class TestCustomerIdInference(unittest.TestCase):
    """run() without an explicit customer_id should pull it from
    whichever input dict carries one -- run_recommendation_agent()
    relies on this since the Coordinator's stub signature doesn't pass
    customer_id separately."""

    def test_infers_from_spending_when_goals_and_simulation_absent(self):
        result = run(spending=SPENDING_WITH_OVERSPEND, goals=None, simulation=None)
        self.assertEqual(result["customer_id"], "CUSTX")

    def test_explicit_customer_id_still_works(self):
        result = run("CUSTOM_ID", SPENDING_WITH_OVERSPEND, GOALS_RESULT, SIMULATION_NOT_ON_TRACK)
        self.assertEqual(result["customer_id"], "CUSTOM_ID")


if __name__ == "__main__":
    unittest.main()
