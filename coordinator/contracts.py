"""
Shared contracts — what each agent actually returns.

Reference documentation only; nothing imports these constants at
runtime. Kept up to date with what each agent's adapter (agents/*.py)
actually returns, for the team to check against.

Goals and Simulation below are now the REAL agents from the team's
`goal_agent` and `Simulation_agent` GitHub branches (vendored unmodified
under agents/vendor/), wired in through thin adapters — not the earlier
placeholders. Spending was already the real `spending-agent` branch code.
"""

# Spending Agent -> agents.spending_agent.analyze_customer()
# (real spending-agent branch code, vendored at agents/vendor/spending_agent_tools.py)
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

# Goals Agent -> agents.goals_agent.run()
# Wraps the real GoalAgent ("Instant Installment Advisor") from the
# goal_agent branch (agents/vendor/goal_agent.py). Each goal now carries
# a real this-month disposable-income gap and priced installment
# suggestions, not just the target/duration/monthly_required read
# straight from goals.csv. duration_months/start_date are added by the
# adapter (agents/goals_agent.py) from goals.csv, since GoalAgent.recommend()
# doesn't return them on its own.
EXAMPLE_GOALS_RESULT = {
    "customer_id": "CUSTE2483D",
    "customer_name": "Allison Hill",
    "monthly_income": 27700.0,
    "analyzed_month": "2026-06",
    "goals": [
        {
            "goal_id": "GOAL9DB818",
            "goal_name": "Emergency Fund",
            "target_amount": 73200.0,
            "monthly_required": 12200.0,
            "disposable_income": 8524.6,
            "monthly_gap": 3675.4,
            "status": "needs_action",  # "on_track" | "needs_action"
            "suggestions": [
                {
                    "transaction_id": "ed3068c4-1",
                    "description": "CINEMA TICKET PURCHASE",
                    "category": "Entertainment",
                    "original_amount": 1250.28,
                    "tenor_months": 12,
                    "fee_rate": 0.015,
                    "monthly_installment": 122.94,
                    "monthly_cash_freed": 1127.34,
                    "contribution_to_goal_pct": 9.2,
                }
                # ... up to one entry per eligible purchase, sorted by fit to the gap
            ],
            "remaining_gap_after_suggestions": 1297.85,  # only present when status == "needs_action"
            "customer_message": None,  # filled in if GOALS_AGENT_USE_LLM=1 and Ollama is running
            "duration_months": 6,       # added by the adapter, from goals.csv
            "start_date": "2026-01-01",  # added by the adapter, from goals.csv
        }
        # ... one entry per goal the customer has
    ],
    "primary_goal": {"...": "same shape as above -- the goal with the soonest deadline"},
    "analyzed_by": "Goals Agent",
}

# Simulation Agent -> agents.simulation_agent.run(customer_id, spending, goals)
# Wraps the real build_simulation_output() from the Simulation_agent
# branch (agents/vendor/simulation_agent/, agents/vendor/shared/).
# spending/goals are accepted for signature compatibility but not
# actually required -- this agent recomputes cash flow and goal
# feasibility itself, directly from transactions.csv/goals.csv.
EXAMPLE_SIMULATION_RESULT = {
    "agent": "simulation",
    "schema_version": "1.0",
    "customer_id": "CUSTE2483D",
    "months_ahead": 6,
    "cash_flow": {
        "avg_monthly_income": 27700.0,
        "avg_monthly_spend": 22557.92,
        "avg_monthly_net": 5142.08,
    },
    "baseline_projection": {
        "last_known_balance": 34903.17,
        "last_known_date": "2026-06-28",
        "avg_monthly_net": 5142.08,
        "projection": [
            {"date": "2026-07-28", "projected_balance": 40045.25}
            # ... one entry per month, up to months_ahead
        ],
    },
    "goals": {
        "per_goal": [
            {
                "goal_name": "Emergency Fund",
                "target_amount": 73200.0,
                "planned_months": 6,
                "required_monthly": 12200.0,
                "allocated_monthly_savings": 546.09,
                "actual_months_needed": 134.0,
                "status": "behind",  # "ahead" | "on track" | "behind"
            }
        ],
        "overall": {
            "avg_monthly_saved": 546.09,
            "total_required_monthly": 12200.0,
            "status": "behind",  # "on track" | "behind" | "no_goals_set"
            "monthly_shortfall": 11653.91,
        },
    },
    "top_scenarios": [
        {
            "scenario": {"Entertainment": -0.3},
            "adjusted_avg_monthly_net": 5837.18,
            "projection": [{"date": "2026-07-28", "projected_balance": 40740.35}],
            "monthly_gain": 695.1,
        }
        # ... up to top_n, sorted best-first by monthly_gain
    ],
    "top_products": {
        "flexible": [
            {
                "product": "Regular Savings Account",
                "annual_rate": 0.09,
                "contribution_type": "lump_sum",  # "lump_sum" | "monthly"
                "contribution_amount": 34903.17,
                "months_simulated": 6,
                "total_contributed": 34903.17,
                "final_balance": 36503.71,
                "interest_earned": 1600.54,
                "warning": None,
            }
        ],
        "locked": [
            {
                "product": "5-Year Fixed Deposit Certificate",
                "annual_rate": 0.185,
                "contribution_type": "lump_sum",
                "contribution_amount": 34903.17,
                "months_simulated": 6,
                "total_contributed": 34903.17,
                "final_balance": 38258.73,
                "interest_earned": 3355.56,
                "warning": "Note: 5-Year Fixed Deposit Certificate locks funds for 60 months. "
                           "You simulated only 6 months - early withdrawal isn't modeled here.",
            }
        ],
    },
    "insight_text": "You're earning about 27,700 per month and spending about 22,558, "
                     "leaving a net of 5,142 per month. ...",
    "analyzed_by": "Simulation Agent",
}

# Recommendation Agent -> agents.recommendation_agent.run(customer_id, spending, goals, simulation)
# Unchanged in shape from the original version -- still a deterministic,
# no-ML fan-in step -- but now reads the real Goals/Simulation shapes
# above, and prefers the Goals Agent's own priced installment suggestion
# as the recommended action when there is one.
EXAMPLE_RECOMMENDATION_RESULT = {
    "customer_id": "CUSTE2483D",
    "summary": "You currently have a monthly surplus of 5,142.08. Your top goal is \"Emergency Fund\". "
               "Overall, you're saving 11,653.91 less per month than your goals need.",
    "recommendations": [
        {
            "title": "Accelerate Your \"Emergency Fund\" Goal",
            "reason": "Your \"Emergency Fund\" goal needs about 12,200 per month to reach 73,200 in 6 months. "
                      "After essential expenses this month, you have about 8,524.60 available toward it -- "
                      "a gap of 3,675.40 per month. Based on your actual saving history, it would take about "
                      "134 months -- longer than the 6-month plan.",
            "estimated_impact": "Frees up about 1,127.34 per month (9.20% of what this goal requires).",
            "action": "Convert \"CINEMA TICKET PURCHASE\" (1,250.28, Entertainment) into a 12-month "
                      "installment plan, and redirect the freed-up cash toward your \"Emergency Fund\" goal.",
        }
    ],
    "goal_impact": {
        "goal_name": "Emergency Fund",
        "target_amount": 73200.0,
        "duration_months": 6,
        "monthly_required": 12200.0,
        "current_monthly_pace": 546.09,
        "projected_months_to_reach": 134.0,
        "on_track": False,
    },
    "used_inputs": {"spending": True, "goals": True, "simulation": True},
    "analyzed_by": "Recommendation Agent",
}
