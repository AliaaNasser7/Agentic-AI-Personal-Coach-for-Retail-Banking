# Spending Agent → what it actually returns (from analyze_customer())
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

# Spending Agent → what generate_coach_message() returns
EXAMPLE_COACH_MESSAGE = {
    "customer_id": "CUST59072A",
    "coach_message": "You're spending a bit more than usual on Shopping this month...",
    "guardrail_warning": None,  # or a string describing what looked wrong
}
