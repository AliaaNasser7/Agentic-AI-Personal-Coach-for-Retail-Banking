"""
Shared data access for all agents.

Fixes the bug in the original coordinator-agent branch, where
data_access.py had a hardcoded Windows path (E:\\AI Track\\...) that only
worked on one teammate's machine.

Every agent should import from here instead of reading CSVs directly,
so there is exactly one place that knows where the data folder is.
"""
import json
import os
import pandas as pd

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE_DIR, "data")

CUSTOMERS_PATH = os.path.join(DATA_DIR, "customers.csv")
TRANSACTIONS_PATH = os.path.join(DATA_DIR, "transactions.csv")
GOALS_PATH = os.path.join(DATA_DIR, "goals.csv")
PRODUCTS_PATH = os.path.join(DATA_DIR, "products_catalog.json")


def load_customers() -> pd.DataFrame:
    return pd.read_csv(CUSTOMERS_PATH)


def get_customer(customer_id: str):
    df = load_customers()
    row = df[df["customer_id"] == customer_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def load_transactions() -> pd.DataFrame:
    return pd.read_csv(TRANSACTIONS_PATH, parse_dates=["date"])


def load_goals() -> pd.DataFrame:
    return pd.read_csv(GOALS_PATH, parse_dates=["start_date"])


def load_products() -> list:
    """The products catalog, as a plain list of dicts (not a DataFrame) —
    both the Goals Agent (installment products) and the Simulation Agent
    (savings/investment products) read it this way."""
    with open(PRODUCTS_PATH, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    customer = get_customer("CUSTE2483D")
    print(customer)
