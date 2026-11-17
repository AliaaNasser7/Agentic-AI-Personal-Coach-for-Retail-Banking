"""
Recommendation Agent : DEMO
A walkthrough of the Recommendation Agent (the fan-in step):
    Spending + Goals + Simulation output  ->  one clear, explainable recommendation

Unlike the other agents, this one needs no LLM and no Ollama: the content is
rule-based and every number in it is read straight from the three inputs.
This demo also runs standalone, so it does NOT need the other three agents'
code. It feeds the agent fixtures shaped exactly like their real output
(see contracts.py). For a run against the REAL agents, see local_demo.py.

Shows the agent working end-to-end on 3 different situations,
chosen on purpose to demonstrate it handles different real cases:
  - A customer who is BEHIND on a goal and has an overspending flag
  - A HEALTHY customer (goal on track, no overspending)
  - A DEGRADED case (Goals + Simulation agents failed, only Spending available)

Run from the project root:
    python recommendation_agent/demo_recommendation_agent.py
"""

import copy
import json
import os
import sys

# The agent is a package (relative imports), so the project root must be on sys.path.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from recommendation_agent import run_recommendation_agent
from recommendation_agent.contracts import (
    EXAMPLE_GOALS_RESULT,
    EXAMPLE_SIMULATION_RESULT,
    EXAMPLE_SPENDING_ANALYSIS,
)


def _print_section(text):
    print(f"\n{text}")


def build_behind_customer():
    """Behind on the goal + a real overspending flag (the contracts.py example)."""
    spending = copy.deepcopy(EXAMPLE_SPENDING_ANALYSIS)
    goals = copy.deepcopy(EXAMPLE_GOALS_RESULT)
    simulation = copy.deepcopy(EXAMPLE_SIMULATION_RESULT)
    return spending, goals, simulation


def build_healthy_customer():
    """Healthy: no overspending, goal on track, simulation says 'ahead'."""
    spending = copy.deepcopy(EXAMPLE_SPENDING_ANALYSIS)
    spending["customer_id"] = "DEMO_HEALTHY"
    spending["overspending_categories"] = []

    goals = copy.deepcopy(EXAMPLE_GOALS_RESULT)
    goals["customer_id"] = "DEMO_HEALTHY"
    for g in goals["goals"]:
        g["status"] = "on_track"
        g["monthly_gap"] = 0.0
        g["suggestions"] = []
        g.pop("remaining_gap_after_suggestions", None)

    simulation = copy.deepcopy(EXAMPLE_SIMULATION_RESULT)
    simulation["customer_id"] = "DEMO_HEALTHY"
    for g in simulation["goals"]["per_goal"]:
        g["status"] = "ahead"
        g["allocated_monthly_savings"] = g["required_monthly"]
        g["actual_months_needed"] = float(g["planned_months"])
    simulation["goals"]["overall"].update(
        {"status": "on track", "monthly_shortfall": 0.0,
         "avg_monthly_saved": simulation["goals"]["overall"]["total_required_monthly"]}
    )
    return spending, goals, simulation


def build_degraded_customer():
    """Goals + Simulation agents failed upstream: only Spending is available."""
    spending = copy.deepcopy(EXAMPLE_SPENDING_ANALYSIS)
    spending["customer_id"] = "DEMO_PARTIAL"
    return spending, None, None


SCENARIOS = {
    "Customer BEHIND on a goal, with an overspending flag": build_behind_customer,
    "HEALTHY customer (on track, no overspending)": build_healthy_customer,
    "DEGRADED case (Goals + Simulation missing, Spending only)": build_degraded_customer,
}


def demo_customer(label, builder):
    _print_section("=" * 70)
    _print_section(f"SCENARIO: {label}")
    _print_section("=" * 70)

    spending, goals, simulation = builder()

    _print_section("--- INPUTS RECEIVED ---")
    print(f"  Spending:   {'available' if spending else 'MISSING'}")
    print(f"  Goals:      {'available' if goals else 'MISSING'}")
    print(f"  Simulation: {'available' if simulation else 'MISSING'}")

    result = run_recommendation_agent(spending, goals, simulation)

    _print_section("--- SUMMARY ---")
    print(f"  {result['summary']}")

    _print_section("--- RECOMMENDATIONS ---")
    if not result["recommendations"]:
        print("  (none)")
    for i, rec in enumerate(result["recommendations"], start=1):
        print(f"  {i}. {rec['title']}")
        print(f"     Why:    {rec['reason']}")
        print(f"     Impact: {rec['estimated_impact']}")
        print(f"     Action: {rec['action']}")

    _print_section("--- GOAL IMPACT ---")
    gi = result.get("goal_impact")
    if gi:
        print(f"  Goal: {gi['goal_name']} (target {gi['target_amount']:,.0f} in {gi['duration_months']} months)")
        print(f"  Needs {gi['monthly_required']:,.2f}/month; current pace {gi['current_monthly_pace']}")
        print(f"  On track: {gi['on_track']}")
    else:
        print("  (not available)")

    _print_section("--- PRIORITY (additive ML layer) ---")
    pr = result.get("priority")
    if pr:
        print(f"  {pr['label']} ({pr['score']}/100), model: {pr['model']}")
    else:
        print("  (model unavailable: scikit-learn/joblib missing, so priority is None; recommendations unaffected)")

    _print_section("--- WHY AM I SEEING THIS? (used_inputs) ---")
    print(f"  {json.dumps(result['used_inputs'])}")


def main():
    _print_section("RECOMMENDATION AGENT DEMO")
    for label, builder in SCENARIOS.items():
        demo_customer(label, builder)

    _print_section("=" * 70)
    _print_section("Demo complete.")


if __name__ == "__main__":
    main()
