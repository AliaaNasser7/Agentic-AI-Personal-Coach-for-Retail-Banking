"""
run_simulation.py

Local entry point - run this to generate the console report and the
JSON contract file. Other agents should import
simulation_agent.build_simulation_output directly instead of running this.
"""

import json
from shared.finance_utils import load_data
from simulation_agent import build_simulation_output, run_full_simulation_report

if __name__ == "__main__":
    customers, transactions, goals, products = load_data("output")
    sample_customer = customers["customer_id"].iloc[0]

    # Human-readable console report (debugging / local demo)
    run_full_simulation_report(sample_customer, transactions, goals, products, months_ahead=6, top_n=3)

    # Clean JSON contract (what the Coordinator actually consumes)
    output = build_simulation_output(sample_customer, transactions, goals, products,
                                      months_ahead=6, top_n=3, use_llm_polish=False)

    with open("output/simulation_output_sample.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print("\nJSON contract saved to output/simulation_output_sample.json")
