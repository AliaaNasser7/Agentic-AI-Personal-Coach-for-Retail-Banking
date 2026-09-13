"""
Spending Agent : DEMO
A walkthrough of the full Spending Agent pipeline:
    Raw transaction data, Numeric analysis, Guardrail-checked, message via a local LLM

Shows the agent working end-to-end on 3 different customer profiles,
chosen on purpose to demonstrate it handles different real situations:
  - A NORMAL customer (no flags)
  - A customer with a real detected overspending anomaly
  - The lowest income customer in the dataset (edge case)
"""

import spending_agent_tools as sat

try:
    from spending_agent_local import generate_coach_message
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


def _print_section(text):
    print(f"\n{text}")


def pick_demo_customers():
    #Selects 3 customers that showcase different real scenarios
    customers = sat._load_customers()
    txns = sat._load_transactions()

    anomaly_ids = txns[txns["is_anomaly"] == True]["customer_id"].unique()
    non_anomaly = customers[~customers["customer_id"].isin(anomaly_ids)]

    normal_customer = non_anomaly.iloc[0]["customer_id"]
    anomaly_customer = anomaly_ids[0] if len(anomaly_ids) > 0 else None
    lowest_income_customer = customers.sort_values("monthly_income").iloc[0]["customer_id"]

    return {
        "Normal customer (healthy spending) ": normal_customer,
        "Customer with a real overspending anomaly ": anomaly_customer,
        "Lowest income customer in the dataset ": lowest_income_customer,
    }


def run_demo_for_customer(label: str, customer_id: str):
    print(f"{label}  ->  {customer_id}")

    customers = sat._load_customers()
    profile = customers[customers["customer_id"] == customer_id].iloc[0]
    print(f"Name: {profile['name']} | Age: {profile['age']} | Job: {profile['job']} "
        f"| Income: {profile['monthly_income']} EGP")

    _print_section("Numeric analysis (no LLM and pure data)")
    analysis = sat.analyze_customer(customer_id)
    print(f"  Avg monthly income:    {analysis['avg_monthly_income']} EGP")
    print(f"  Avg monthly spending:  {analysis['avg_monthly_spending']} EGP")
    print(f"  Monthly surplus:       {analysis['monthly_surplus']} EGP")
    print(f"  Months analyzed:       {analysis['months_analyzed']}")

    if analysis["overspending_categories"]:
        print(f"  Overspending flags:")
        for flag in analysis["overspending_categories"][:3]:
            print(f"    - {flag['month']} | {flag['category']}: "
                f"{flag['month_amount']} EGP ({flag['vs_avg_pct']:+.1f}% vs. their own baseline)")
    else:
        print("  Overspending flags:    None")

    _print_section("Coach message (LLM + guardrail check)")
    if not LLM_AVAILABLE:
        print("Skipped, spending_agent_local.py not found.")
        return

    try:
        result = generate_coach_message(customer_id)
        print(f"  \"{result['coach_message']}\"")
        if result["guardrail_warning"]:
            print(f"\n  [GUARDRAIL WARNING] {result['guardrail_warning']}")
        else:
            print("\n  [Guardrail check: passed, no invented categories/numbers detected]")
    except ConnectionError:
        print("Skipped, Ollama isn't running locally. Start it with `ollama serve` "
            "(or open the Ollama app) to see the generated coach message.")


def run_error_handling_demo():
    print("Bonus: Error handling in action")
    try:
        sat.get_customer_transactions("THIS_CUSTOMER_DOES_NOT_EXIST")
    except sat.SpendingAgentError as e:
        print(f"  Invalid customer_id correctly caught as SpendingAgentError:")
        print(f"  -> {e}")


def run_categorization_demo():
    print("Bonus: Automatic transaction categorization from raw text")
    examples = [
        "MCDONALDS #482",
        "ACH DEBIT - RENT PAYMENT",
        "UBER *TRIP 213",
        "NETFLIX.COM SUBSCRIPTION",
        "SOME UNKNOWN MERCHANT XYZ",
    ]
    for desc in examples:
        category = sat.categorize_transaction_description(desc)
        print(f"  '{desc}'  ->  {category}")


if __name__ == "__main__":
    print("NBE AGENTIC AI PERSONAL COACH , SPENDING AGENT DEMO")
    demo_customers = pick_demo_customers()
    for label, customer_id in demo_customers.items():
        if customer_id is not None:
            run_demo_for_customer(label, customer_id)

    run_categorization_demo()
    run_error_handling_demo()