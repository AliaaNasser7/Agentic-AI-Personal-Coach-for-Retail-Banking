import pandas as pd
from pathlib import Path

# project root, not coordinator_agent
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "output"


def load_customers(path=None):
    path = path or DATA_DIR / "customers.csv"
    return pd.read_csv(path)


def get_customer(customer_id, path=None):
    df = load_customers(path)
    row = df[df["customer_id"] == customer_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


if __name__ == "__main__":
    customer = get_customer("CUSTE2483D")
    print(customer)
