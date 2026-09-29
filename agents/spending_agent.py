"""
Spending Agent — thin adapter
==============================
Wraps the real Spending Agent from the team's `spending-agent` branch
(`spending_agent_tools.py` — 25 passing unit tests on that branch)
instead of reimplementing its logic. The teammate's file is vendored,
byte-for-byte unmodified, at agents/vendor/spending_agent_tools.py.

The only integration work needed here is pointing its two hardcoded
path constants (TRANSACTIONS_PATH, CUSTOMERS_PATH — originally
`<branch folder>/output/*.csv`) at this project's shared data_access.py
paths, the same way the team's own `coordinator-agent` branch did it in
`agent_stubs.py`:

    spending_agent_tools.TRANSACTIONS_PATH = str(DATA_DIR / "transactions.csv")
    spending_agent_tools.CUSTOMERS_PATH = str(DATA_DIR / "customers.csv")

Everything else — get_customer_transactions, calculate_monthly_surplus,
detect_overspending_categories, categorize_transaction_description,
analyze_customer, SpendingAgentError — is re-exported as-is, so nothing
else in this project (spending_agent_llm.py, the Coordinator, the tests)
has to change.
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_VENDOR_DIR = os.path.join(_THIS_DIR, "vendor")
for _p in (_PROJECT_ROOT, _VENDOR_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import spending_agent_tools as _tools  # vendored, unmodified
from data_access import TRANSACTIONS_PATH, CUSTOMERS_PATH

# Point the vendored module's file paths at this project's shared data
# folder instead of its original branch-local `output/` folder.
_tools.TRANSACTIONS_PATH = TRANSACTIONS_PATH
_tools.CUSTOMERS_PATH = CUSTOMERS_PATH

# Re-export the real agent's public contract, unchanged.
SpendingAgentError = _tools.SpendingAgentError
get_customer_transactions = _tools.get_customer_transactions
calculate_monthly_surplus = _tools.calculate_monthly_surplus
detect_overspending_categories = _tools.detect_overspending_categories
categorize_transaction_description = _tools.categorize_transaction_description
analyze_customer = _tools.analyze_customer


if __name__ == "__main__":
    from data_access import load_customers
    customers = load_customers()
    sample_id = customers.iloc[0]["customer_id"]
    print(f"Testing Spending Agent on customer: {sample_id}\n")
    print(analyze_customer(sample_id))
