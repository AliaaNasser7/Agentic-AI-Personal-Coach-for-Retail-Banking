"""
Simulation Agent — thin adapter
==================================
Wraps the real Simulation Agent built on the team's `Simulation_agent`
branch (forecasting.py, scenarios.py, products.py, narration.py,
report.py — the only branch of the five with a full, tested
implementation and its own docs/README.md) instead of the placeholder
arithmetic that used to live here. The teammate's package is vendored,
byte-for-byte unmodified, at agents/vendor/shared/ and
agents/vendor/simulation_agent/.

What the real Simulation Agent adds over the old placeholder:
  - A linear balance projection N months ahead (project_balance), not
    just "months to reach one goal at the current surplus".
  - Per-goal AND combined feasibility (goal_feasibility): each goal's
    proportional share of actual average monthly savings vs. what it
    requires, status "ahead" / "on track" / "behind" (or "no_goals_set"
    if the customer has none — never a false "on track").
  - Ranked "what-if" spending-cut scenarios (rank_scenarios) across
    discretionary categories, not just a single overspending flag.
  - Ranked savings/investment product options from the real products
    catalog, split into flexible (no lock-in) vs. locked (fixed-term)
    (rank_product_options).
  - A fact-grounded narrative paragraph (narrate_report) — every number
    is computed in Python first; an optional local LLM is only ever
    allowed to rephrase already-correct sentences, never invent one.

Adapter responsibilities (kept intentionally small): load this
project's shared CSVs/catalog through data_access.py, call the real
`build_simulation_output`, and translate its error convention
(`{"error": ...}` dict) into the SimulationAgentError exception the
Coordinator already knows how to catch — mirroring the SpendingAgentError
/ GoalsAgentError pattern used by the other two agents.

Public contract (unchanged from the old file):
    run(customer_id, spending=None, goals=None) -> dict
    SimulationAgentError                        -> the one exception type this module raises
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_VENDOR_DIR = os.path.join(_THIS_DIR, "vendor")
for _p in (_PROJECT_ROOT, _VENDOR_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data_access import load_transactions, load_goals, load_products
from simulation_agent import build_simulation_output  # vendored package, unmodified
from simulation_agent.errors import SimulationAgentError as _CoreSimulationAgentError


class SimulationAgentError(Exception):
    """Raised when the Simulation Agent can't compute a result for the
    given input (matches the style of SpendingAgentError / GoalsAgentError
    so the Coordinator can catch one clear type)."""
    pass


def run(customer_id: str, spending: dict = None, goals: dict = None,
        months_ahead: int = 6, top_n: int = 3) -> dict:
    """
    spending, goals: accepted for signature compatibility with the
    Coordinator (which already fans Spending and Goals results into this
    call) but not required — the real Simulation Agent recomputes cash
    flow and goal feasibility directly from transactions.csv/goals.csv
    itself, the same way Spending and Goals do, rather than trusting
    numbers a sibling agent already rounded (see finance_utils.py's
    note on avoiding cross-agent rounding drift).
    """
    transactions = load_transactions()
    goals_df = load_goals()
    products = load_products()

    try:
        result = build_simulation_output(
            customer_id, transactions, goals_df, products,
            months_ahead=months_ahead, top_n=top_n, use_llm_polish=True,
        )
    except _CoreSimulationAgentError as e:
        raise SimulationAgentError(str(e)) from e

    if "error" in result:
        raise SimulationAgentError(result["error"])

    result["analyzed_by"] = "Simulation Agent"
    return result


if __name__ == "__main__":
    import json
    print(json.dumps(run("CUSTE2483D"), indent=2, default=str))
