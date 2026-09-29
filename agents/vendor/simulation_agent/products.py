"""
simulation_agent/products.py

Step 5: simulate investing a single amount in a single product.
Step 5b: try every eligible product, split by liquidity, rank each group.
"""

from shared.finance_utils import monthly_cash_flow, get_product, customer_exists
from .errors import SimulationAgentError


def simulate_product_investment(product_code, products, amount, months_ahead, contribution_type="lump_sum"):
    product = get_product(product_code, products)
    if product is None:
        return {"error": f"Unknown product code: {product_code}"}

    annual_rate = product["expected_return"]
    if annual_rate is None:
        return {"error": f"{product['name']} is a loan product, not a savings/investment option"}

    if amount < product["min_amount"]:
        return {"error": f"Minimum for {product['name']} is {product['min_amount']}, you provided {amount}"}

    monthly_rate = annual_rate / 12
    balance = 0
    history = []

    if contribution_type == "lump_sum":
        balance = amount
        for n in range(1, months_ahead + 1):
            balance *= (1 + monthly_rate)
            history.append({"month": n, "balance": round(balance, 2)})
    elif contribution_type == "monthly":
        for n in range(1, months_ahead + 1):
            balance += amount
            balance *= (1 + monthly_rate)
            history.append({"month": n, "balance": round(balance, 2)})

    total_contributed = amount if contribution_type == "lump_sum" else amount * months_ahead
    interest_earned = round(balance - total_contributed, 2)

    warning = None
    if product["duration_months"] > 0 and months_ahead < product["duration_months"]:
        warning = (
            f"Note: {product['name']} locks funds for {product['duration_months']} months. "
            f"You simulated only {months_ahead} months - early withdrawal isn't modeled here."
        )

    return {
        "product": product["name"],
        "annual_rate": annual_rate,
        "contribution_type": contribution_type,
        "months_simulated": months_ahead,
        "total_contributed": round(total_contributed, 2),
        "final_balance": round(balance, 2),
        "interest_earned": interest_earned,
        "warning": warning,
        "history": history,
    }


def get_investable_products(products):
    return [p for p in products if p["expected_return"] is not None and p["expected_return"] > 0]


def rank_product_options(customer_id, transactions_df, products, months_ahead=6, top_n=3):
    """
    Returns two ranked lists instead of one mixed list:
      - flexible: products with no lock-in (duration_months == 0), e.g. a savings account
      - locked:   products with a lock-in period (duration_months > 0), e.g. a certificate
    Ranking within each group is still by interest_earned, but keeping the
    groups separate avoids a high-rate but 5-year-locked product silently
    outranking a same-day-access option in a single combined list.
    """
    if not customer_exists(customer_id, transactions_df):
        raise SimulationAgentError(f"Unknown customer_id: {customer_id!r} has no transaction history")

    real_cust_txns = transactions_df[transactions_df["customer_id"] == customer_id].sort_values("date")
    last_balance = real_cust_txns.iloc[-1]["balance_after"]
    avg_net = monthly_cash_flow(customer_id, transactions_df)["avg_monthly_net"]

    flexible, locked = [], []
    for product in get_investable_products(products):
        candidates_for_product = []

        if last_balance >= product["min_amount"]:
            result = simulate_product_investment(
                product["product_code"], products, round(last_balance, 2), months_ahead, "lump_sum"
            )
            if "error" not in result:
                result["contribution_amount"] = round(last_balance, 2)
                candidates_for_product.append(result)

        if avg_net > 0:
            result = simulate_product_investment(
                product["product_code"], products, round(avg_net, 2), months_ahead, "monthly"
            )
            if "error" not in result:
                result["contribution_amount"] = round(avg_net, 2)
                candidates_for_product.append(result)

        target_list = locked if product["duration_months"] > 0 else flexible
        target_list.extend(candidates_for_product)

    flexible.sort(key=lambda r: r["interest_earned"], reverse=True)
    locked.sort(key=lambda r: r["interest_earned"], reverse=True)

    return {"flexible": flexible[:top_n], "locked": locked[:top_n]}
