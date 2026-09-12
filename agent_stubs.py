from contracts import EXAMPLE_RECOMMENDATION_OUTPUT
from shared.finance_utils import load_data, load_products
from simulation_agent import build_simulation_output
from goal_agent import GoalAgent, GoalAgentConfig
from spending_agent_local import generate_coach_message
from spending_agent_tools import analyze_customer, SpendingAgentError
import spending_agent_tools
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = BASE_DIR / "output"


spending_agent_tools.TRANSACTIONS_PATH = str(DATA_DIR / "transactions.csv")
spending_agent_tools.CUSTOMERS_PATH = str(DATA_DIR / "customers.csv")
print(
    f"[DEBUG] Spending Agent will read customers from: {spending_agent_tools.CUSTOMERS_PATH}")


def fallback_spending_message(analysis):
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
    print(f"[GUARDRAIL] {agent_name} — customer {customer_id}: {warning}")


def run_spending_agent(customer_id):
    try:
        analysis = analyze_customer(customer_id)
    except SpendingAgentError as e:
        return {"error": str(e), "analyzed_by": "Spending Agent"}

    try:
        message_result = generate_coach_message(customer_id)
    except (ConnectionError, requests.exceptions.RequestException) as e:
        log_guardrail_warning("Spending", customer_id,
                              f"LLM call failed, using fallback: {e}")
        return {**analysis, "coach_message": fallback_spending_message(analysis)}

    if message_result.get("guardrail_warning"):
        log_guardrail_warning("Spending", customer_id,
                              message_result["guardrail_warning"])
        safe_message = fallback_spending_message(analysis)
    else:
        safe_message = message_result["coach_message"]

    return {**analysis, "coach_message": safe_message}


try:
    _goal_agent = GoalAgent(
        customers_path=str(DATA_DIR / "customers.csv"),
        goals_path=str(DATA_DIR / "goals.csv"),
        transactions_path=str(DATA_DIR / "transactions.csv"),
        products_path=str(DATA_DIR / "products_catalog.json"),
        config=GoalAgentConfig(use_llm_message=True),
    )
    _goal_agent_init_error = None
except (ValueError, FileNotFoundError) as e:
    _goal_agent = None
    _goal_agent_init_error = str(e)


def run_goals_agent(customer_id):
    if _goal_agent is None:
        return {"error": f"Goal Agent failed to initialize: {_goal_agent_init_error}",
                "analyzed_by": "Goal Agent"}

    result = _goal_agent.recommend(customer_id)

    if "error" in result:
        return {**result, "analyzed_by": "Goal Agent"}

    return result


try:
    _sim_transactions, _sim_goals, _ = load_data(str(DATA_DIR))
    _sim_products = load_products(str(DATA_DIR / "products_catalog.json"))
    _sim_init_error = None
except Exception as e:
    _sim_transactions = _sim_goals = _sim_products = None
    _sim_init_error = str(e)


def run_simulation_agent(customer_id, spending_output=None, goals_output=None):
    if _sim_init_error is not None:
        return {"error": f"Simulation Agent failed to initialize: {_sim_init_error}",
                "analyzed_by": "Simulation Agent"}

    try:
        return build_simulation_output(
            customer_id, _sim_transactions, _sim_goals, _sim_products,
            months_ahead=6, top_n=3, model="llama3", use_llm_polish=True,
        )
    except Exception as e:
        return {"error": f"Simulation failed for this customer: {e}", "analyzed_by": "Simulation Agent"}


def run_recommendation_agent(spending_output, goals_output, simulation_output):
    return EXAMPLE_RECOMMENDATION_OUTPUT
