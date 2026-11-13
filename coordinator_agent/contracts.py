"""
This file defines the exact shape of data each specialist agent must return.
Every real agent (once built) must match these shapes exactly, so the
Coordinator can combine them without needing to know how each agent works internally.
"""

# Spending Agent → matches Alia's real analyze_customer() output
EXAMPLE_SPENDING_ANALYSIS = {
    "customer_id": "CUST59072A",
    "avg_monthly_income": 27700.0,
    "avg_monthly_spending": 21956.4,
    "monthly_surplus": 5743.6,
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

# Spending Agent → matches Alia's real generate_coach_message() output
EXAMPLE_COACH_MESSAGE = {
    "customer_id": "CUST59072A",
    "coach_message": "You're spending a bit more than usual on Shopping this month...",
    "guardrail_warning": None,
}

# Goals Agent → real output shape (from GoalAgent.recommend()), no longer a stub:
# {
#   "customer_id": ..., "customer_name": ..., "monthly_income": ...,
#   "analyzed_month": "2026-06",
#   "goals": [
#     {"goal_id": ..., "goal_name": ..., "target_amount": ..., "monthly_required": ...,
#      "disposable_income": ..., "monthly_gap": ..., "status": "on_track"|"needs_action",
#      "suggestions": [...], "remaining_gap_after_suggestions": ..., "customer_message": ... or None}
#   ]
# }

# Simulation Agent → real output shape (from build_simulation_output()), no longer a stub:
# {
#   "agent": "simulation", "schema_version": "1.0", "customer_id": ...,
#   "cash_flow": {"avg_monthly_income":..., "avg_monthly_spend":..., "avg_monthly_net":...},
#   "baseline_projection": {"last_known_balance":..., "projection": [{"date":..., "projected_balance":...}]},
#   "goals": {"per_goal": [...], "overall": {...}},
#   "top_scenarios": [{"scenario": {...}, "monthly_gain":..., "projection": [...]}],
#   "top_products": {"flexible": [...], "locked": [...]},
#   "insight_text": "..."
# }

# Simulation Agent → placeholder until Ahlam delivers her real handoff
EXAMPLE_SIMULATION_OUTPUT = {
    "summary_text": "At this pace your balance reaches about 48,200 by December.",
    "current_balance": 26600.0,
    "avg_monthly_income": 15000.0,
    "avg_monthly_spend": 11400.0,
    "avg_monthly_net": 3600.0,
    "projected_6_months": 48200.0,
    "best_scenario_monthly_add": 360.0,
    "best_scenario_6_months": 50360.0,
    "balance_projection": {
        "months": ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov"],
        "current_pace": [26600, 30200, 33800, 37400, 41000, 44600],
        "best_scenario": [26600, 30560, 34520, 38480, 42440, 46400],
    },
    "ways_to_free_up_cash": [
        {"action": "Cut Dining, Entertainment, Shopping, Subscriptions 15% each",
            "monthly_gain": 360},
        {"action": "Cut Dining/Restaurants 30%", "monthly_gain": 270},
    ],
}

# Recommendation Agent → placeholder until Alaa delivers her real handoff
EXAMPLE_RECOMMENDATION_OUTPUT = {
    "summary_text": "You're saving consistently, but dining spend is creeping up — small cuts there would meaningfully speed up your Vacation goal.",
}
