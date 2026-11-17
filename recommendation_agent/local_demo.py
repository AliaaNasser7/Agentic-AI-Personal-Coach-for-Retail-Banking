"""
Optional, local-only sanity check -- NOT meant to be merged into the
coordinator repo. Runs the REAL Spending, Goals, and Simulation agents
(no mocks, no fixtures) and feeds their real output into
run_recommendation_agent(), so you can see one full, real result before
merging anything.

SETUP
------
1. Unzip the three branches you already have as SIBLING folders next to
   this project's root (same level as the `recommendation_agent/`
   folder), using exactly these names:

       <project root>/
         recommendation_agent/   <- this package
         scripts/                 <- this script
         goal_agent/               <- unzipped goal_agent branch
         spending-agent/            <- unzipped spending-agent branch
         Simulation_agent/          <- unzipped Simulation_agent branch

   (If you downloaded each branch as a GitHub zip, its contents are one
   level deeper, e.g. goal_agent/Agentic-AI-Personal-Coach-for-Retail-
   Banking-goal_agent/goal_agent.py -- move that inner folder's
   contents up one level so goal_agent.py sits directly inside
   goal_agent/.)

2. pip install pandas requests scikit-learn joblib --break-system-packages

RUN (from the project root)
-----------------------------
    python scripts/local_demo.py                 # first customer in the data
    python scripts/local_demo.py CUSTE2483D       # a specific customer
"""
import json
import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS_DIR)

_GOAL_DIR = os.path.join(_ROOT, "goal_agent")
_SPEND_DIR = os.path.join(_ROOT, "spending-agent")
_SIM_ROOT = os.path.join(_ROOT, "Simulation_agent")
_SIM_AGENTS_DIR = os.path.join(_SIM_ROOT, "agents")
_SIM_OUTPUT_DIR = os.path.join(_SIM_ROOT, "output")

for label, path in [
    ("goal_agent/ (goal_agent branch)", _GOAL_DIR),
    ("spending-agent/ (spending-agent branch)", _SPEND_DIR),
    ("Simulation_agent/agents/ (Simulation_agent branch)", _SIM_AGENTS_DIR),
]:
    if not os.path.isdir(path):
        sys.exit(
            f"Missing expected folder for {label}: {path}\n"
            "See this script's docstring (`python scripts/local_demo.py --help` "
            "or just open the file) for the exact folder layout needed."
        )

sys.path.insert(0, _ROOT)           # so `import recommendation_agent` works
sys.path.insert(0, _GOAL_DIR)       # goal_agent.py
sys.path.insert(0, _SPEND_DIR)      # spending_agent_tools.py
sys.path.insert(0, _SIM_AGENTS_DIR)  # shared/, simulation_agent/

import spending_agent_tools
spending_agent_tools.CUSTOMERS_PATH = os.path.join(_SPEND_DIR, "output", "customers.csv")
spending_agent_tools.TRANSACTIONS_PATH = os.path.join(_SPEND_DIR, "output", "transactions.csv")
from spending_agent_tools import analyze_customer, SpendingAgentError

from goal_agent import GoalAgent, GoalAgentConfig

from shared.finance_utils import load_data, load_products
from simulation_agent import build_simulation_output

from recommendation_agent import run_recommendation_agent


def main():
    customer_id = sys.argv[1] if len(sys.argv) > 1 else None

    customers_df, transactions, goals_df, products = load_data(_SIM_OUTPUT_DIR)
    if customer_id is None:
        customer_id = customers_df["customer_id"].iloc[0]

    print(f"Running the real pipeline for customer: {customer_id}\n")

    # --- Spending Agent (real) ---
    try:
        spending = analyze_customer(customer_id)
        print("Spending Agent: OK")
    except SpendingAgentError as e:
        print(f"Spending Agent error (continuing without it): {e}")
        spending = None

    # --- Goals Agent (real) ---
    goal_agent = GoalAgent(
        customers_path=os.path.join(_GOAL_DIR, "output", "customers.csv"),
        goals_path=os.path.join(_GOAL_DIR, "output", "goals.csv"),
        transactions_path=os.path.join(_GOAL_DIR, "output", "transactions.csv"),
        products_path=os.path.join(_GOAL_DIR, "output", "products_catalog.json"),
        config=GoalAgentConfig(use_llm_message=False),  # no Ollama needed for this smoke test
    )
    goals_result = goal_agent.recommend(customer_id)
    if "error" in goals_result:
        print(f"Goals Agent error (continuing without it): {goals_result['error']}")
        goals = None
    else:
        print("Goals Agent: OK")
        goals = goals_result

    # --- Simulation Agent (real) ---
    try:
        simulation = build_simulation_output(
            customer_id, transactions, goals_df, products,
            months_ahead=6, top_n=3, use_llm_polish=False,  # no Ollama needed for this smoke test
        )
        if "error" in simulation:
            print(f"Simulation Agent error (continuing without it): {simulation['error']}")
            simulation = None
        else:
            print("Simulation Agent: OK")
    except Exception as e:
        print(f"Simulation Agent error (continuing without it): {e}")
        simulation = None

    # --- Recommendation Agent (this package, unmodified call) ---
    print("\nRunning Recommendation Agent...\n")
    recommendation = run_recommendation_agent(spending, goals, simulation)
    print(json.dumps(recommendation, indent=2, default=str))


if __name__ == "__main__":
    main()
