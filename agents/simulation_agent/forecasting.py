"""
simulation_agent/forecasting.py

Step 2 (balance projection) and Step 3 (goal feasibility). Both build on
shared.finance_utils rather than recomputing cash flow / savings rate.
"""

import pandas as pd
from shared.finance_utils import monthly_cash_flow, savings_rate


def project_balance(customer_id, transactions_df, months_ahead=6):
    cust_txns = transactions_df[transactions_df["customer_id"] == customer_id].sort_values("date")
    last_balance = cust_txns.iloc[-1]["balance_after"]
    last_date = cust_txns.iloc[-1]["date"]

    avg_net = monthly_cash_flow(customer_id, transactions_df)["avg_monthly_net"]

    projection = []
    for n in range(1, months_ahead + 1):
        future_date = last_date + pd.DateOffset(months=n)
        projected = last_balance + (avg_net * n)
        projection.append({"date": future_date.strftime("%Y-%m-%d"), "projected_balance": round(projected, 2)})

    return {
        "last_known_balance": round(last_balance, 2),
        "last_known_date": last_date.strftime("%Y-%m-%d"),
        "avg_monthly_net": round(avg_net, 2),
        "projection": projection,
    }


def goal_feasibility(customer_id, transactions_df, goals_df):
    """
    Per-goal status is proportional to overall savings, not checked
    independently - avg_saved is one shared pool across all the customer's
    goals, so each goal is credited its proportional share of it. This keeps
    per-goal badges consistent with the 'overall' status (an earlier version
    checked each goal against the FULL avg_saved independently, which could
    show every goal "ahead" while the combined total was actually short).
    """
    cust_goals = goals_df[goals_df["customer_id"] == customer_id]
    avg_saved = savings_rate(customer_id, transactions_df)
    goal_rows = list(cust_goals.iterrows())

    if not goal_rows:
        # No goals set yet - "on track" would be a false claim about
        # something that doesn't exist. Use a distinct status so the
        # narration layer can pivot to "here's what you could grow"
        # instead of implying a goal is being met.
        return {
            "per_goal": [],
            "overall": {
                "avg_monthly_saved": round(avg_saved, 2),
                "total_required_monthly": 0,
                "status": "no_goals_set",
                "monthly_shortfall": 0,
            },
        }

    total_required_monthly = sum(g["monthly_required"] for _, g in goal_rows)

    results = []
    for _, goal in goal_rows:
        target = goal["target_amount"]
        planned_months = goal["duration_months"]
        required_monthly = goal["monthly_required"]

        if total_required_monthly > 0:
            allocated_monthly = avg_saved * (required_monthly / total_required_monthly)
        else:
            allocated_monthly = 0

        actual_months_needed = (target / allocated_monthly) if allocated_monthly > 0 else float("inf")

        if allocated_monthly >= required_monthly * 1.05:
            status = "ahead"
        elif allocated_monthly >= required_monthly * 0.95:
            status = "on track"
        else:
            status = "behind"

        results.append({
            "goal_name": goal["goal_name"],
            "target_amount": target,
            "planned_months": planned_months,
            "required_monthly": round(required_monthly, 2),
            "allocated_monthly_savings": round(allocated_monthly, 2),
            "actual_months_needed": round(actual_months_needed, 1) if actual_months_needed != float("inf") else None,
            "status": status,
        })

    overall_status = "on track" if avg_saved >= total_required_monthly else "behind"
    shortfall = max(0, total_required_monthly - avg_saved)

    return {
        "per_goal": results,
        "overall": {
            "avg_monthly_saved": round(avg_saved, 2),
            "total_required_monthly": round(total_required_monthly, 2),
            "status": overall_status,
            "monthly_shortfall": round(shortfall, 2),
        },
    }
