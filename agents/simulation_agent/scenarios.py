"""
simulation_agent/scenarios.py

Step 4: simulate a single "what if I cut category X by Y%" scenario.
Step 4b: try several candidate scenarios, rank by monthly cash-flow gain.
"""

import pandas as pd
from shared.finance_utils import monthly_cash_flow

# Categories a customer can realistically choose to cut back on.
# Excludes fixed/necessary categories like Rent, Utilities, Healthcare.
DISCRETIONARY_CATEGORIES = ["Dining/Restaurants", "Entertainment", "Shopping", "Subscriptions"]


def simulate_scenario(customer_id, transactions_df, category_changes=None, months_ahead=6):
    category_changes = category_changes or {}
    cust_txns = transactions_df[transactions_df["customer_id"] == customer_id].copy()

    for category, pct_change in category_changes.items():
        mask = (cust_txns["category"] == category) & (cust_txns["direction"] == "debit")
        cust_txns.loc[mask, "amount"] = cust_txns.loc[mask, "amount"] * (1 + pct_change)

    cust_txns["month"] = cust_txns["date"].dt.to_period("M")
    monthly = cust_txns.groupby(["month", "direction"])["amount"].sum().unstack(fill_value=0)
    monthly["net"] = monthly.get("credit", 0) - monthly.get("debit", 0)
    adjusted_avg_net = monthly["net"].mean()

    real_cust_txns = transactions_df[transactions_df["customer_id"] == customer_id].sort_values("date")
    last_balance = real_cust_txns.iloc[-1]["balance_after"]
    last_date = real_cust_txns.iloc[-1]["date"]

    projection = []
    for n in range(1, months_ahead + 1):
        future_date = last_date + pd.DateOffset(months=n)
        projected = last_balance + (adjusted_avg_net * n)
        projection.append({"date": future_date.strftime("%Y-%m-%d"), "projected_balance": round(projected, 2)})

    return {
        "scenario": category_changes,
        "adjusted_avg_monthly_net": round(adjusted_avg_net, 2),
        "projection": projection,
    }


def generate_candidate_scenarios(categories=None, pct_options=None):
    """
    Builds a list of {category: pct_change} dicts to try - one category cut
    at a time across a few percentages, plus one combined "cut everything a
    bit" scenario. This is a starting set; tune it as you learn what customers
    respond to.
    """
    categories = categories or DISCRETIONARY_CATEGORIES
    pct_options = pct_options or [-0.10, -0.20, -0.30]

    candidates = [{cat: pct} for cat in categories for pct in pct_options]
    candidates.append({cat: -0.15 for cat in categories})  # combined moderate cut
    return candidates


def rank_scenarios(customer_id, transactions_df, months_ahead=6, top_n=3, candidate_scenarios=None):
    baseline_net = monthly_cash_flow(customer_id, transactions_df)["avg_monthly_net"]
    candidate_scenarios = candidate_scenarios or generate_candidate_scenarios()

    results = []
    for scenario in candidate_scenarios:
        r = simulate_scenario(customer_id, transactions_df, scenario, months_ahead)
        r["monthly_gain"] = round(r["adjusted_avg_monthly_net"] - baseline_net, 2)
        results.append(r)

    results.sort(key=lambda r: r["monthly_gain"], reverse=True)
    return results[:top_n]
