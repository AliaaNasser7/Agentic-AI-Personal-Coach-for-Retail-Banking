"""
Coordinator / Orchestrator
==========================
Fixes from the original coordinator-agent branch:
  - agent_stubs.py imported run_goals_agent / run_simulation_agent /
    run_recommendation_agent, none of which existed anywhere → replaced
    with real calls into agents/goals_agent.py, agents/simulation_agent.py,
    agents/recommendation_agent.py.
  - agent_stubs.py imported `from coordinator_agent.contarcts import ...`,
    a package that didn't exist → removed; contracts.py is now just
    reference documentation, not imported at runtime.
  - data_access.py had a hardcoded Windows path → fixed in the shared
    data_access.py at the project root.

agents/goals_agent.py and agents/simulation_agent.py now wrap the real
Goals Agent (`goal_agent` branch) and Simulation Agent (`Simulation_agent`
branch) instead of the placeholders that used to live here — see those
two files' docstrings, and coordinator/contracts.py, for what changed.
This file didn't need to change for that: it only ever passed whatever
dict each agent returned along to the next one, and both adapters keep
the same run(...) signatures / *AgentError exception types the old
placeholders had.

Architecture (matches the brief):
    Spending Agent  ---\\
                         >---> Simulation Agent ---> Recommendation Agent
    Goals Agent     ---/

Spending and Goals don't depend on each other, so they run in parallel.
Simulation depends on both of their outputs (this dependency already
existed in the real, non-stub orchestrator.py code, so it's kept as-is
rather than redesigned). Recommendation is the final fan-in step.
"""
import sys
import os
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_access import get_customer
from agents.spending_agent import analyze_customer, SpendingAgentError
from agents.goals_agent import run as run_goals_agent, GoalsAgentError
from agents.simulation_agent import run as run_simulation_agent, SimulationAgentError
from agents.recommendation_agent import run as run_recommendation_agent


def fallback_spending_message(analysis):
    """A safe, non-LLM sentence built directly from the numbers. Used
    whenever the LLM coach message is missing, flagged, or Ollama isn't
    running (kept from the original orchestrator.py)."""
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
    print(f"[GUARDRAIL] {agent_name} - customer {customer_id}: {warning}")


def run_spending_agent(customer_id):
    """Runs the numeric analysis, and tries the LLM coach message on top
    (optional — falls back to a plain sentence if Ollama isn't running,
    so the whole system still works without it installed)."""
    try:
        analysis = analyze_customer(customer_id)
    except SpendingAgentError as e:
        return {"error": str(e), "analyzed_by": "Spending Agent"}

    try:
        from agents.spending_agent_llm import generate_coach_message
        message_result = generate_coach_message(customer_id)
        if message_result.get("guardrail_warning"):
            log_guardrail_warning("Spending", customer_id, message_result["guardrail_warning"])
            safe_message = fallback_spending_message(analysis)
        else:
            safe_message = message_result["coach_message"]
    except ConnectionError:
        safe_message = fallback_spending_message(analysis)
    except Exception:
        safe_message = fallback_spending_message(analysis)

    return {**analysis, "coach_message": safe_message}


def run_goals_agent_safe(customer_id):
    try:
        return run_goals_agent(customer_id)
    except GoalsAgentError as e:
        return {"error": str(e), "analyzed_by": "Goals Agent"}


def run_simulation_agent_safe(customer_id, spending, goals):
    spending_ok = spending if not (spending and spending.get("error")) else None
    goals_ok = goals if not (goals and goals.get("error")) else None
    try:
        return run_simulation_agent(customer_id, spending_ok, goals_ok)
    except SimulationAgentError as e:
        return {"error": str(e), "analyzed_by": "Simulation Agent"}


def load_dashboard(customer_id):
    """Full dashboard view: runs every agent and returns a tab-ready
    structure. Spending and Goals run in parallel since neither depends
    on the other; Simulation and Recommendation run after, in order."""
    customer = get_customer(customer_id)
    if customer is None:
        return {"error": f"No customer found with id {customer_id}"}

    with ThreadPoolExecutor(max_workers=2) as pool:
        spending_future = pool.submit(run_spending_agent, customer_id)
        goals_future = pool.submit(run_goals_agent_safe, customer_id)
        spending = spending_future.result()
        goals = goals_future.result()

    simulation = run_simulation_agent_safe(customer_id, spending, goals)

    spending_for_rec = spending if not spending.get("error") else None
    goals_for_rec = goals if not goals.get("error") else None
    simulation_for_rec = simulation if not simulation.get("error") else None

    recommendation = run_recommendation_agent(
        customer_id, spending_for_rec, goals_for_rec, simulation_for_rec
    )

    return {
        "customer": customer,
        "tabs": {
            "spending": spending,
            "goals": goals,
            "simulation": simulation,
            "overview": recommendation,
        },
    }


# ---------------------------------------------------------------------
# Lightweight question router for the chat/demo entry point.
# Not NLP, just keyword matching — deliberately simple, per the brief.
# Only calls the agents actually needed for the question, and only calls
# the Recommendation Agent when the question actually needs a
# recommendation (not for a plain lookup like "how much did I spend on X").
# ---------------------------------------------------------------------
_RECOMMENDATION_TRIGGERS = [
    "save more", "how can i save", "recommend", "recommendation",
    "reach my goal", "reach goal", "faster", "afford", "should i",
    "what should i do", "advice", "help me save",
]
_GOAL_KEYWORDS = ["goal", "afford", "emergency fund", "wedding", "vacation", "down payment", "new car"]
_SIMULATION_KEYWORDS = ["faster", "simulate", "projection", "future", "afford", "reach my goal"]


def handle_question(customer_id: str, question: str) -> dict:
    """Routes a free-text question to the minimum set of agents needed,
    and only runs Recommendation when the question actually calls for one."""
    customer = get_customer(customer_id)
    if customer is None:
        return {"error": f"No customer found with id {customer_id}"}

    q = question.lower()
    needs_recommendation = any(trigger in q for trigger in _RECOMMENDATION_TRIGGERS)
    needs_goals = needs_recommendation or any(k in q for k in _GOAL_KEYWORDS)
    needs_simulation = needs_recommendation or any(k in q for k in _SIMULATION_KEYWORDS)

    spending = run_spending_agent(customer_id)
    goals = run_goals_agent_safe(customer_id) if needs_goals else None
    simulation = (
        run_simulation_agent_safe(customer_id, spending, goals)
        if needs_simulation else None
    )

    if not needs_recommendation:
        # A plain lookup question — the Spending Agent's own numbers/message
        # already answer it, no need to fan into Recommendation.
        return {
            "customer_id": customer_id,
            "question": question,
            "agents_used": ["Spending Agent"],
            "answer": spending,
        }

    spending_for_rec = spending if not spending.get("error") else None
    goals_for_rec = goals if goals and not goals.get("error") else None
    simulation_for_rec = simulation if simulation and not simulation.get("error") else None

    recommendation = run_recommendation_agent(
        customer_id, spending_for_rec, goals_for_rec, simulation_for_rec
    )

    agents_used = ["Spending Agent"]
    if needs_goals:
        agents_used.append("Goals Agent")
    if needs_simulation:
        agents_used.append("Simulation Agent")
    agents_used.append("Recommendation Agent")

    return {
        "customer_id": customer_id,
        "question": question,
        "agents_used": agents_used,
        "answer": recommendation,
    }


if __name__ == "__main__":
    import json
    dashboard = load_dashboard("CUSTE2483D")
    print(json.dumps(dashboard, indent=2, default=str))
