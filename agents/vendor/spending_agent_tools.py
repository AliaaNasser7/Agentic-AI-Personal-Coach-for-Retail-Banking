import os
import pandas as pd

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSACTIONS_PATH = os.path.join(_BASE_DIR, "output", "transactions.csv")
CUSTOMERS_PATH = os.path.join(_BASE_DIR, "output", "customers.csv")


class SpendingAgentError(Exception):
    """Raised for any Spending Agent data/lookup problem (missing customer,
    missing files, etc.) so the Coordinator can catch one clear exception
    type instead of guessing which built-in error might come out."""
    pass

def _load_transactions():
    if not os.path.exists(TRANSACTIONS_PATH):
        raise SpendingAgentError(
            f"transactions.csv not found at {TRANSACTIONS_PATH}. "
            f"Make sure it's in the same folder as this script, or re-run generate_all_data.py."
        )
    try:
        df = pd.read_csv(TRANSACTIONS_PATH, parse_dates=["date"])
    except Exception as e:
        raise SpendingAgentError(f"Failed to read transactions.csv: {e}")

    required_cols = {"customer_id", "date", "category", "amount", "direction"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SpendingAgentError(f"transactions.csv is missing expected columns: {missing}")
    return df


def _load_customers():
    if not os.path.exists(CUSTOMERS_PATH):
        raise SpendingAgentError(
            f"customers.csv not found at {CUSTOMERS_PATH}. "
            f"Make sure it's in the same folder as this script, or re-run generate_all_data.py."
        )
    try:
        return pd.read_csv(CUSTOMERS_PATH)
    except Exception as e:
        raise SpendingAgentError(f"Failed to read customers.csv: {e}")


#get customer transactions
def get_customer_transactions(customer_id: str, months: int = None) -> pd.DataFrame:
    #Returns all transactions for a given customer, optionally limited to the last N months.
    if not customer_id or not isinstance(customer_id, str):
        raise SpendingAgentError(f"customer_id must be a non-empty string, got: {customer_id!r}")
    df = _load_transactions()
    cust_txns = df[df["customer_id"] == customer_id].sort_values("date")

    if cust_txns.empty:
        raise SpendingAgentError(
            f"No transactions found for customer_id='{customer_id}'. "
            f"Check the ID is correct and exists in transactions.csv."
        )
    if months:
        cutoff = cust_txns["date"].max() - pd.DateOffset(months=months)
        cust_txns = cust_txns[cust_txns["date"] > cutoff]

    return cust_txns.reset_index(drop=True)


#calculate monthly surplus
def calculate_monthly_surplus(customer_id: str) -> dict:
    #Calculates average monthly income, average monthly spending,and the resulting surplus (or deficit) for a customer.
    txns = get_customer_transactions(customer_id)
    txns["month"] = txns["date"].dt.to_period("M")

    monthly = txns.groupby(["month", "direction"])["amount"].sum().unstack(fill_value=0)

    monthly_income = monthly.get("credit", pd.Series(dtype=float)).mean()
    monthly_spending = monthly.get("debit", pd.Series(dtype=float)).mean()
    surplus = round(float(monthly_income - monthly_spending), 2)

    return {
        "customer_id": customer_id,
        "avg_monthly_income": round(float(monthly_income), 2),
        "avg_monthly_spending": round(float(monthly_spending), 2),
        "monthly_surplus": surplus,
        "months_analyzed": int(len(monthly)),
    }


# detect overspending categories
def detect_overspending_categories(customer_id: str, threshold_pct: float = 30.0,scope: str = "any_month") -> list:
    """
    Detects categories where a customer's spending in a given month is
    significantly higher (threshold_pct) than their own average for that
    category across the OTHER months
    scope:
    "any_month"    -> checks every month in history and flags each month/category combo that stands out (default;
    catches an unusual spike wherever it happened).
    "latest_month" -> only checks the most recent month vs the rest(useful for a "what happened this month" view).
    Returns a list of dicts, one per flagged month/category combo,
    sorted by how far above baseline it is.
    """
    txns = get_customer_transactions(customer_id)
    txns = txns[txns["direction"] == "debit"]
    txns["month"] = txns["date"].dt.to_period("M")

    by_month_cat = txns.groupby(["month", "category"])["amount"].sum().unstack(fill_value=0)

    if len(by_month_cat) < 2:
        return []  # not enough history to compare

    months_to_check = [by_month_cat.index.max()] if scope == "latest_month" else list(by_month_cat.index)

    flagged = []
    for month in months_to_check:
        other_months = by_month_cat.drop(index=month)
        for category in by_month_cat.columns:
            month_amt = by_month_cat.loc[month, category]
            avg_other = other_months[category].mean()

            if avg_other == 0:
                continue

            pct_change = ((month_amt - avg_other) / avg_other) * 100
            if pct_change >= threshold_pct:
                flagged.append({
                    "month": str(month),
                    "category": category,
                    "month_amount": round(float(month_amt), 2),
                    "avg_other_months": round(float(avg_other), 2),
                    "vs_avg_pct": round(float(pct_change), 1),
                })

    return sorted(flagged, key=lambda x: x["vs_avg_pct"], reverse=True)


#categorize_transaction_description
CATEGORY_KEYWORDS = {
    "Rent/Installment": ["rent", "installment", "home loan"],
    "Utilities": ["electricity", "water co", "internet bill"],
    "Groceries": ["carrefour", "spinneys", "market", "grocery"],
    "Transportation": ["uber", "careem", "fuel", "taxi"],
    "Dining/Restaurants": ["mcdonald", "restaurant", "cafe", "dining"],
    "Entertainment": ["cinema", "netflix ppv", "gaming"],
    "Shopping": ["amazon", "mall", "online store", "shopping"],
    "Subscriptions": ["netflix.com", "spotify", "gym membership", "subscription"],
    "Healthcare": ["pharmacy", "clinic", "lab test", "healthcare"],
    "Savings Transfer": ["savings", "internal xfer", "transfer to savings"],
}

def categorize_transaction_description(description: str) -> str:
    """
    Classifies a raw transaction description string into one of the known
    spending categories using simple keyword matching.

    Returns "Uncategorized" if no keyword matches — this is intentional:
    it's safer to flag a transaction as unclassified than to silently
    guess wrong, especially in a banking context.
    """
    if not description or not isinstance(description, str):
        return "Uncategorized"

    desc_lower = description.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in desc_lower for keyword in keywords):
            return category
    return "Uncategorized"


#analyze_customer (unified interface for the Coordinator)
def analyze_customer(customer_id: str, threshold_pct: float = 30.0) -> dict:
    """
    Single entry point the Coordinator agent can call to get the FULL
    spending analysis for a customer in one call.
    Returns a dict ready to be merged into the shared state, or raises
    SpendingAgentError with a clear message if anything goes wrong,
    the Coordinator should catch SpendingAgentError specifically.
    """
    surplus_data = calculate_monthly_surplus(customer_id)
    overspending = detect_overspending_categories(customer_id, threshold_pct=threshold_pct)

    return {
        "customer_id": customer_id,
        "avg_monthly_income": surplus_data["avg_monthly_income"],
        "avg_monthly_spending": surplus_data["avg_monthly_spending"],
        "monthly_surplus": surplus_data["monthly_surplus"],
        "months_analyzed": surplus_data["months_analyzed"],
        "overspending_categories": overspending,
        "analyzed_by": "Spending Agent",
    }


# Quick manual test
if __name__ == "__main__":
    customers = _load_customers()
    sample_id = customers.iloc[0]["customer_id"]
    print(f"Testing Spending Agent tools on customer: {sample_id}\n")

    print("get_customer_transactions (last 2 months preview):")
    print(get_customer_transactions(sample_id, months=2).head(5).to_string(index=False))

    print("\ncalculate_monthly_surplus:")
    print(calculate_monthly_surplus(sample_id))

    print("\ndetect_overspending_categories:")
    result = detect_overspending_categories(sample_id)
    if result:
        for r in result:
            print(" -", r)
    else:
        print(" No overspending categories flagged for this customer.")
    print("\n categorize_transaction_description (spot checks):")
    for desc in ["MCDONALDS #482", "ACH DEBIT - RENT PAYMENT", "UBER *TRIP 213", "SOME RANDOM TEXT"]:
        print(f"  '{desc}' -> {categorize_transaction_description(desc)}")

    print("\n analyze_customer (unified Coordinator-ready output):")
    print(analyze_customer(sample_id))

    print("\n Error handling check (invalid customer_id):")
    try:
        get_customer_transactions("THIS_ID_DOES_NOT_EXIST")
    except SpendingAgentError as e:
        print(f"  Caught expected SpendingAgentError: {e}")