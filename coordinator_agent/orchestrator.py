from data_access import get_customer
from agent_stubs import (
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


if __name__ == "__main__":
    import json
    dashboard = load_dashboard("CUSTE2483D")
    print(json.dumps(dashboard, indent=2))
