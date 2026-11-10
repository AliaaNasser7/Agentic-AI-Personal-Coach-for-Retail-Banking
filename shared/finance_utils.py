"""
shared/finance_utils.py

Finance calculations that more than one agent needs. Both the Simulation
Agent and the Goals Agent need "how much does this customer earn/spend/save
per month" - if each agent computed that separately, the dashboard could
show two different numbers for the same thing. Import from here instead of
recomputing it locally.
"""

import json
import pandas as pd


def load_data(output_dir="output"):
    """
    Loads all four datasets. Returns (customers, transactions, goals, products).
    """
    customers = pd.read_csv(f"{output_dir}/customers.csv")
    transactions = pd.read_csv(f"{output_dir}/transactions.csv", parse_dates=["date"])
    goals = pd.read_csv(f"{output_dir}/goals.csv")
    with open(f"{output_dir}/products_catalog.json", encoding="utf-8") as f:
        products = json.load(f)
    return customers, transactions, goals, products


def load_products(path="output/products_catalog.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_product(product_code, products):
    return next((p for p in products if p["product_code"] == product_code), None)


def customer_exists(customer_id, transactions_df):
    """True if this customer has at least one transaction on record."""
    return (transactions_df["customer_id"] == customer_id).any()


def monthly_cash_flow(customer_id, transactions_df):
    """
    Average income / spend / net per month for this customer.

    Rounded to 2 decimals HERE, at the source - not by each caller. This is
    the fix for cross-agent rounding drift (e.g. Spending Agent computing
    22557.92 for the same figure Simulation computed as 22557.923333...
    from an unrounded mean): every agent importing this function now gets
    the identical rounded number instead of rounding independently.
    """
    cust_txns = transactions_df[transactions_df["customer_id"] == customer_id].copy()
    cust_txns["month"] = cust_txns["date"].dt.to_period("M")

    monthly = cust_txns.groupby(["month", "direction"])["amount"].sum().unstack(fill_value=0)
    monthly["net"] = monthly.get("credit", 0) - monthly.get("debit", 0)

    return {
        "avg_monthly_income": round(monthly.get("credit", pd.Series(dtype=float)).mean(), 2),
        "avg_monthly_spend": round(monthly.get("debit", pd.Series(dtype=float)).mean(), 2),
        "avg_monthly_net": round(monthly["net"].mean(), 2),
    }


def savings_rate(customer_id, transactions_df, use_category="Savings Transfer"):
    """Average amount per month this customer actually moves into savings. Rounded at the source (see monthly_cash_flow)."""
    cust_txns = transactions_df[transactions_df["customer_id"] == customer_id]
    savings_txns = cust_txns[
        (cust_txns["category"] == use_category) & (cust_txns["direction"] == "debit")
    ].copy()
    savings_txns["month"] = savings_txns["date"].dt.to_period("M")
    monthly_savings = savings_txns.groupby("month")["amount"].sum()
    return round(monthly_savings.mean(), 2) if len(monthly_savings) > 0 else 0.0
