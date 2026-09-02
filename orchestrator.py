from data_access import get_customer
from agent_stubs import (
    SpendingAgentError,
    analyze_customer,
    generate_coach_message,
    run_spending_agent,
    run_goals_agent,
    run_simulation_agent,
    run_recommendation_agent,
)


def load_dashboard(customer_id):
    customer = get_customer(customer_id)
    if customer is None:
        return {"error": f"No customer found with id {customer_id}"}

    spending = run_spending_agent(customer_id)
    goals = run_goals_agent(customer_id)
    simulation = run_simulation_agent(customer_id, spending, goals)
    recommendation = run_recommendation_agent(spending, goals, simulation)

    return {
        "customer": customer,
        "tabs": {
            "spending": spending,
            "goals": goals,
            "simulation": simulation,
            "overview": recommendation,
        },
    }


def fallback_spending_message(analysis):
    """A safe, non-LLM sentence built directly from the numbers.
    Used whenever the LLM message is missing or flagged by a guardrail."""
    msg = (
        f"Your average monthly income is {analysis['avg_monthly_income']:.0f} "
        f"and spending is {analysis['avg_monthly_spending']:.0f}, "
        f"leaving a surplus of {analysis['monthly_surplus']:.0f}."
    )
    if analysis["overspending_categories"]:
        cat = analysis["overspending_categories"][0]
        msg += f" In {cat['month']}, your {cat['category']} spending was notably higher than usual."
    return msg


def log_guardrail_warning(agent_name, customer_id, warning):
    # For now: just print. Later: write to a log file your team can review.
    print(f"[GUARDRAIL] {agent_name} — customer {customer_id}: {warning}")


def run_spending_agent(customer_id):
    try:
        analysis = analyze_customer(customer_id)
    except SpendingAgentError as e:
        return {"error": str(e), "analyzed_by": "Spending Agent"}

    message_result = generate_coach_message(customer_id)

    if message_result.get("guardrail_warning"):
        log_guardrail_warning("Spending", customer_id,
                              message_result["guardrail_warning"])
        safe_message = fallback_spending_message(analysis)
    else:
        safe_message = message_result["coach_message"]

    return {**analysis, "coach_message": safe_message}
if __name__ == "__main__":
    import json
    dashboard = load_dashboard("CUSTE2483D")
    print(json.dumps(dashboard, indent=2))
