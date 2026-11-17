"""
Feature engineering for the Recommendation Agent's ML priority model.

Turns a customer's Spending / Goals / Simulation Agent output (the same
three dicts recommendation_agent/core.py's run() already receives —
nothing re-read from CSVs here) into a fixed-size numeric feature vector
describing their overall financial situation.

Ratios are used instead of raw amounts wherever possible so the model
generalizes across customers with very different income levels, rather
than memorizing absolute amounts from the ~50-customer training set.
Every feature is clipped to a sane range so a single outlier customer
can't dominate the model.

Any of spending/goals/simulation can be None (an upstream agent failed
or wasn't run) — every feature has a neutral default (0.0) for that
case, and priority_model.py always has a full vector to score.
"""

FEATURE_NAMES = [
    "surplus_ratio",           # monthly_surplus / avg_monthly_income -- how much room the customer has
    "overspend_severity",      # top overspending category's vs_avg_pct / 100
    "overspend_excess_ratio",  # top overspend excess amount / avg_monthly_income
    "goal_gap_ratio",          # this month's monthly_gap / monthly_required (0 if on_track)
    "has_installment_option",  # 1.0 if the Goals Agent found a usable installment suggestion
    "installment_coverage",    # best suggestion's contribution_to_goal_pct / 100
    "months_behind_ratio",     # how much longer than planned, at the current saving pace
    "saving_rate_ratio",       # allocated_monthly_savings / required_monthly, capped at 1
    "best_scenario_gain_ratio",  # best spending-cut scenario's monthly_gain / avg_monthly_income
    "has_product_option",      # 1.0 if Simulation found a ranked savings product to suggest
]


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def extract_features(spending: dict, goals: dict, simulation: dict) -> dict:
    """Returns a dict keyed by FEATURE_NAMES, safe to call with any of
    the three inputs missing (None)."""
    f = {name: 0.0 for name in FEATURE_NAMES}

    income = None
    if spending:
        income = spending.get("avg_monthly_income") or None
        surplus = spending.get("monthly_surplus")
        if income and surplus is not None:
            f["surplus_ratio"] = _clip(surplus / income, -1.0, 1.0)
        overspend = spending.get("overspending_categories") or []
        if overspend:
            top = overspend[0]
            f["overspend_severity"] = _clip((top.get("vs_avg_pct") or 0.0) / 100.0, 0.0, 3.0)
            excess = (top.get("month_amount", 0) or 0) - (top.get("avg_other_months", 0) or 0)
            if income:
                f["overspend_excess_ratio"] = _clip(excess / income, 0.0, 1.0)

    primary_goal = goals.get("primary_goal") if goals else None
    if primary_goal:
        required = primary_goal.get("monthly_required") or 0
        gap = primary_goal.get("monthly_gap")
        if required and gap is not None and gap > 0:
            f["goal_gap_ratio"] = _clip(gap / required, 0.0, 1.0)
        suggestions = primary_goal.get("suggestions") or []
        if suggestions:
            f["has_installment_option"] = 1.0
            f["installment_coverage"] = _clip((suggestions[0].get("contribution_to_goal_pct") or 0.0) / 100.0, 0.0, 1.0)

        if simulation:
            for g in simulation.get("goals", {}).get("per_goal", []):
                if g.get("goal_name") == primary_goal.get("goal_name"):
                    planned = g.get("planned_months") or primary_goal.get("duration_months")
                    actual = g.get("actual_months_needed")
                    if planned and actual is not None:
                        f["months_behind_ratio"] = _clip((actual - planned) / planned, 0.0, 5.0)
                    req_m = g.get("required_monthly")
                    alloc = g.get("allocated_monthly_savings")
                    if req_m:
                        f["saving_rate_ratio"] = _clip((alloc or 0.0) / req_m, 0.0, 1.0)
                    break

    if simulation:
        scenarios = simulation.get("top_scenarios") or []
        if scenarios and income:
            f["best_scenario_gain_ratio"] = _clip((scenarios[0].get("monthly_gain") or 0.0) / income, 0.0, 1.0)
        top_products = simulation.get("top_products") or {}
        if (top_products.get("flexible") or top_products.get("locked")):
            f["has_product_option"] = 1.0

    return f


def features_to_vector(features: dict) -> list:
    """Fixed feature order for the model, matching how it was trained."""
    return [features.get(name, 0.0) for name in FEATURE_NAMES]
