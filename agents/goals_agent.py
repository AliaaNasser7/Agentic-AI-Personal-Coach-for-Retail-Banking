"""
Goals Agent — thin adapter
============================
Wraps the real Goal Agent ("Instant Installment Advisor") built on the
team's `goal_agent` branch, instead of the placeholder that used to live
here (which only read goals.csv and echoed target/duration/monthly_required
straight back). The teammate's file is vendored, byte-for-byte unmodified,
at agents/vendor/goal_agent.py.

What the real Goal Agent adds over the old placeholder:
  - For each goal, it looks at the customer's actual disposable income
    THIS month (income minus essential spending) and compares it to what
    the goal requires, producing a real monthly_gap instead of just
    restating monthly_required.
  - When there's a gap, it scans that month's non-essential purchases
    (Shopping, Electronics, Travel, Dining, Entertainment, Furniture) for
    ones worth converting into an installment plan, prices each option
    against the products catalog (or a default fee), and picks the
    tenor that best closes the gap — see agents/vendor/goal_agent.py's
    module docstring for the full logic.

Adapter responsibilities (kept intentionally small):
  - Point GoalAgent at this project's shared data_access.py paths
    instead of hardcoded branch-local ones.
  - GoalAgent.recommend() doesn't return duration_months or start_date
    per goal (it was never built to need them) — this adapter enriches
    each goal with those two fields from goals.csv, purely so downstream
    code (the Simulation Agent, the frontend, the existing "primary
    goal" pick) keeps working the way it did before.
  - Pick a "primary_goal" the same way the old file did: the goal with
    the soonest deadline (shortest duration_months) — a simple,
    explainable rule, not a scoring model.

Public contract (unchanged from the old file):
    get_customer_goals(customer_id) -> list[dict]
    run(customer_id)                -> dict   (single entry point for the Coordinator)
    GoalsAgentError                 -> the one exception type this module raises
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_VENDOR_DIR = os.path.join(_THIS_DIR, "vendor")
for _p in (_PROJECT_ROOT, _VENDOR_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_access import CUSTOMERS_PATH, GOALS_PATH, TRANSACTIONS_PATH, PRODUCTS_PATH, load_goals
from goal_agent import GoalAgent, GoalAgentConfig  # vendored, unmodified


class GoalsAgentError(Exception):
    """Raised for any Goals Agent data/lookup problem, mirroring the style
    of SpendingAgentError so the Coordinator can catch one clear type."""
    pass


# use_llm_message stays off by default (the vendored GoalAgentConfig's
# own default) so a normal run of this project doesn't require Ollama
# and doesn't print a connection-refused notice per goal when it isn't
# running. Flip it to True (or set it via env var below) once Ollama is
# installed and `ollama serve` is running, to get a per-goal
# customer_message the same way the Spending Agent's coach message works
# — GoalAgent._call_ollama already falls back to None cleanly either way.
import os as _os
_config = GoalAgentConfig(use_llm_message=_os.environ.get("GOALS_AGENT_USE_LLM", "").lower() in ("1", "true", "yes"))

try:
    _agent = GoalAgent(
        customers_path=CUSTOMERS_PATH,
        goals_path=GOALS_PATH,
        transactions_path=TRANSACTIONS_PATH,
        products_path=PRODUCTS_PATH,
        config=_config,
    )
    _init_error = None
except (ValueError, FileNotFoundError) as e:
    _agent = None
    _init_error = str(e)


def _run_goal_agent(customer_id: str) -> dict:
    if not customer_id or not isinstance(customer_id, str):
        raise GoalsAgentError(f"customer_id must be a non-empty string, got: {customer_id!r}")
    if _agent is None:
        raise GoalsAgentError(f"Goal Agent failed to initialize: {_init_error}")

    result = _agent.recommend(customer_id)
    if "error" in result:
        raise GoalsAgentError(result["error"])
    return result


def _enrich_with_schedule(goal_result: dict, goals_df) -> dict:
    """Adds duration_months + start_date from goals.csv, which
    GoalAgent.recommend() doesn't return on its own."""
    row = goals_df[goals_df["goal_id"] == goal_result["goal_id"]]
    if row.empty:
        goal_result["duration_months"] = None
        goal_result["start_date"] = None
        return goal_result
    row = row.iloc[0]
    goal_result["duration_months"] = int(row["duration_months"])
    goal_result["start_date"] = row["start_date"].strftime("%Y-%m-%d")
    return goal_result


def get_customer_goals(customer_id: str) -> list:
    """Returns every goal's full analysis (target, disposable income,
    gap, installment suggestions, etc.) for a customer, as a list of
    plain dicts."""
    result = _run_goal_agent(customer_id)
    goals_df = load_goals()
    return [_enrich_with_schedule(dict(g), goals_df) for g in result["goals"]]


def _pick_primary_goal(goals: list) -> dict:
    """Picks the goal to lead with when a customer has more than one:
    the one with the soonest deadline (shortest duration_months). Same
    rule as before this integration — simple and explainable, not a
    scoring model."""
    with_duration = [g for g in goals if g.get("duration_months") is not None]
    pool = with_duration or goals
    return min(pool, key=lambda g: g.get("duration_months") if g.get("duration_months") is not None else float("inf"))


def run(customer_id: str) -> dict:
    """Single entry point the Coordinator calls to get a customer's full
    goal picture (now including the real Goal Agent's disposable-income
    gap analysis and installment suggestions) in one call."""
    result = _run_goal_agent(customer_id)
    goals_df = load_goals()
    goals = [_enrich_with_schedule(dict(g), goals_df) for g in result["goals"]]

    return {
        "customer_id": customer_id,
        "customer_name": result.get("customer_name"),
        "monthly_income": result.get("monthly_income"),
        "analyzed_month": result.get("analyzed_month"),
        "goals": goals,
        "primary_goal": _pick_primary_goal(goals),
        "analyzed_by": "Goals Agent",
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run("CUSTE2483D"), indent=2, default=str))
