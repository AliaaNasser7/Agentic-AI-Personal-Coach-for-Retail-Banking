"""
simulation_agent/report.py

The entry point for this agent. Everything else in this package is a
building block - this file is what actually gets imported and called.

Two public functions:
  - build_simulation_output   -> use this one. Pure, JSON-serializable,
                                  no prints. This is what the Coordinator /
                                  API layer should call.
  - run_full_simulation_report -> same computation, but also prints a
                                  human-readable report to the console.
                                  Useful for local debugging and demos,
                                  not meant to be called by other agents.
"""

from shared.finance_utils import monthly_cash_flow
from .forecasting import project_balance, goal_feasibility
from .scenarios import rank_scenarios
from .products import rank_product_options
from .narration import narrate_report, DEFAULT_MODEL
from .errors import SimulationAgentError


def _compute(customer_id, transactions_df, goals_df, products, months_ahead, top_n):
    if months_ahead < 1:
        raise SimulationAgentError(f"months_ahead must be at least 1, got {months_ahead}")

    cash_flow = monthly_cash_flow(customer_id, transactions_df)
    baseline_projection = project_balance(customer_id, transactions_df, months_ahead)
    goals_result = goal_feasibility(customer_id, transactions_df, goals_df)
    top_scenarios = rank_scenarios(customer_id, transactions_df, months_ahead, top_n=top_n)
    top_products = rank_product_options(customer_id, transactions_df, products, months_ahead, top_n=top_n)

    return {
        "cash_flow": cash_flow,
        "baseline_projection": baseline_projection,
        "goals": goals_result,
        "top_scenarios": top_scenarios,
        "top_products": top_products,
    }


def _to_native(obj):
    """
    Recursively converts numpy scalar types (int64, float64, bool_) to
    native Python types. pandas operations (.mean(), .iloc[-1], etc.)
    return numpy scalars even after round() - round() on a numpy float64
    stays a numpy float64, it doesn't downcast. json.dumps() happens to
    handle numpy.float64 (it's a float subclass) but raises TypeError on
    numpy.int64 (it isn't an int subclass) - this was caught by testing a
    customer whose balance_after column loaded as all-integer values,
    where the very first field in the whole output crashed a plain
    json.dumps() call. Centralizing the fix here (once, on the final
    output) is safer than hunting every round()/aggregation call site
    across forecasting.py, scenarios.py, and products.py.
    """
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalar (int64, float64, bool_, ...)
        return obj.item()
    return obj


def build_simulation_output(customer_id, transactions_df, goals_df, products,
                             months_ahead=6, top_n=3, model=DEFAULT_MODEL, use_llm_polish=True):
    """
    THIS is the function other agents / the Coordinator should call.
    Returns a single JSON-serializable dict: no prints, no side effects,
    no leftover numpy types (see _to_native).

    If customer_id has no transaction history, the underlying calls raise
    SimulationAgentError - caught here and converted into a small error
    dict instead, consistent with how simulate_product_investment reports
    invalid input, and safer for an API-style caller that expects a JSON
    response either way rather than a crash.
    """
    try:
        output = _compute(customer_id, transactions_df, goals_df, products, months_ahead, top_n)
    except SimulationAgentError as e:
        return {
            "agent": "simulation",
            "schema_version": "1.0",
            "customer_id": customer_id,
            "error": str(e),
        }

    output["agent"] = "simulation"
    output["schema_version"] = "1.0"
    output["customer_id"] = customer_id
    output["months_ahead"] = months_ahead
    output["insight_text"] = narrate_report(output, model=model, use_llm_polish=use_llm_polish)
    return _to_native(output)


def run_full_simulation_report(customer_id, transactions_df, goals_df, products,
                                months_ahead=6, top_n=3):
    """Console-printing version, for local debugging/demos only."""
    try:
        r = _compute(customer_id, transactions_df, goals_df, products, months_ahead, top_n)
    except SimulationAgentError as e:
        print(f"ERROR: {e}")
        return None

    cash_flow, baseline_projection = r["cash_flow"], r["baseline_projection"]
    goals_result, top_scenarios, top_products = r["goals"], r["top_scenarios"], r["top_products"]

    print("=" * 60)
    print(f"SIMULATION REPORT - customer {customer_id}")
    print("=" * 60)

    print("\n-- Cash flow (avg per month) --")
    print(f"Income: {cash_flow['avg_monthly_income']:.2f}")
    print(f"Spend:  {cash_flow['avg_monthly_spend']:.2f}")
    print(f"Net:    {cash_flow['avg_monthly_net']:.2f}")

    print(f"\n-- Baseline balance projection ({months_ahead} months) --")
    print(f"Last known balance: {baseline_projection['last_known_balance']} on {baseline_projection['last_known_date']}")
    for row in baseline_projection["projection"]:
        print(f"  {row['date']}: {row['projected_balance']}")

    print("\n-- Goal feasibility --")
    for g in goals_result["per_goal"]:
        print(f"  {g['goal_name']}: status={g['status']}, "
              f"needs {g['actual_months_needed']} months at current pace "
              f"(planned {g['planned_months']})")
    overall = goals_result["overall"]
    print(f"  OVERALL: saving {overall['avg_monthly_saved']}/mo, "
          f"need {overall['total_required_monthly']}/mo, status={overall['status']}, "
          f"shortfall={overall['monthly_shortfall']}")

    print(f"\n-- Top {top_n} spending-cut scenarios (ranked by monthly gain) --")
    for i, s in enumerate(top_scenarios, 1):
        print(f"  #{i}: {s['scenario']} -> +{s['monthly_gain']}/mo, "
              f"balance in {months_ahead}mo: {s['projection'][-1]['projected_balance']}")

    print(f"\n-- Top {top_n} flexible product options (no lock-in, ranked by interest) --")
    for i, p in enumerate(top_products["flexible"], 1):
        print(f"  #{i}: {p['contribution_type']} {p['contribution_amount']} into {p['product']} "
              f"-> +{p['interest_earned']} interest, final balance {p['final_balance']}")

    print(f"\n-- Top {top_n} locked product options (fixed term, ranked by interest) --")
    for i, p in enumerate(top_products["locked"], 1):
        print(f"  #{i}: {p['contribution_type']} {p['contribution_amount']} into {p['product']} "
              f"-> +{p['interest_earned']} interest, final balance {p['final_balance']}"
              f"{' | ' + p['warning'] if p['warning'] else ''}")

    print("=" * 60)
    return r