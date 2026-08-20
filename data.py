"""
4 datasets needed by the 5-agent architecture:
-customers.csv         -> Coordinator agent (identity + product held)
-transactions.csv      -> Spending agent + Simulation agent (balance_after)
-goals.csv             -> Goals agent
-products_catalog.json -> Advisor / RAG agent (knowledge base)

Design notes:
- product_code values are inspired by real banking product taxonomies
    (Savings, Checking, Fixed Deposit, Housing Loan, Auto Loan...).
- transaction descriptions mimic realistic bank-statement style text
    (mixed casing, merchant-like names) rather than clean "Starbucks Coffee".
- balance_after is computed cumulatively per account, like a real
    bank statement, to support the Simulation agent's forecasting.
- Anomalies are injected intentionally so the Spending/Simulation
    agents have real signal to detect.
"""

import json
import os
import random
import uuid
from datetime import datetime, timedelta

import pandas as pd
from faker import Faker

fake = Faker()
random.seed(42)
Faker.seed(42)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


NUM_CUSTOMERS = 50
NUM_MONTHS = 6
START_DATE = datetime(2026, 1, 1)
ANOMALY_CHANCE = 0.15
ANOMALY_CATEGORIES = ["Shopping", "Dining/Restaurants", "Entertainment", "Healthcare"]

JOBS = ["Engineer", "Teacher", "Doctor", "Accountant", "Sales Rep",
        "Government Employee", "Business Owner", "Student", "Freelancer"]

PRODUCTS = [
    {"product_code": "CHEK", "name": "Current Account", "type": "Checking",
    "expected_return": 0.0, "min_amount": 0, "duration_months": 0,
    "description": "Everyday checking account with no minimum balance. Used for salary deposits and daily spending."},
    {"product_code": "SAVS", "name": "Regular Savings Account", "type": "Savings",
    "expected_return": 0.09, "min_amount": 500, "duration_months": 0,
    "description": "Flexible savings account with instant withdrawal access and a modest annual return."},
    {"product_code": "FIXD3", "name": "3-Year Fixed Deposit Certificate", "type": "Certificate",
    "expected_return": 0.16, "min_amount": 5000, "duration_months": 36,
    "description": "Fixed-term certificate locking funds for 3 years in exchange for a higher guaranteed annual return."},
    {"product_code": "FIXD5", "name": "5-Year Fixed Deposit Certificate", "type": "Certificate",
    "expected_return": 0.185, "min_amount": 10000, "duration_months": 60,
    "description": "Longer-term certificate for customers who can commit funds for 5 years, offering the highest guaranteed return in this catalog."},
    {"product_code": "HOUL", "name": "Housing Loan", "type": "Loan",
    "expected_return": None, "min_amount": 50000, "duration_months": 120,
    "description": "Long-term financing for home purchase or renovation."},
    {"product_code": "AUTL", "name": "Auto Loan", "type": "Loan",
    "expected_return": None, "min_amount": 20000, "duration_months": 48,
    "description": "Financing for new or used vehicle purchases."},
]

CATEGORY_PROFILE = {
    "Rent/Installment":   {"type": "fixed",    "pct": (0.20, 0.35)},
    "Utilities":           {"type": "fixed",    "pct": (0.03, 0.06)},
    "Groceries":           {"type": "variable", "pct": (0.08, 0.15)},
    "Transportation":      {"type": "variable", "pct": (0.04, 0.08)},
    "Dining/Restaurants":   {"type": "variable", "pct": (0.03, 0.10)},
    "Entertainment":       {"type": "variable", "pct": (0.02, 0.07)},
    "Shopping":             {"type": "variable", "pct": (0.03, 0.10)},
    "Subscriptions":        {"type": "fixed",    "pct": (0.01, 0.03)},
    "Healthcare":           {"type": "variable", "pct": (0.01, 0.05)},
    "Savings Transfer":     {"type": "variable", "pct": (0.00, 0.15)},
}

# Realistic-ish merchant/description templates per category (bank-statement style)
MERCHANT_TEMPLATES = {
    "Rent/Installment": ["ACH DEBIT - RENT PAYMENT", "Home Loan Installment - {bank}", "Monthly Rent Transfer"],
    "Utilities": ["ELECTRICITY BILL PMT", "WATER CO. AUTOPAY", "Internet Bill - {bank}"],
    "Groceries": ["CARREFOUR HYPERMARKET", "SPINNEYS #{num}", "POS PURCHASE - LOCAL MARKET"],
    "Transportation": ["UBER *TRIP {num}", "CAREEM RIDE", "FUEL STATION PURCHASE"],
    "Dining/Restaurants": ["MCDONALDS #{num}", "POS - LOCAL RESTAURANT", "CAFE PURCHASE"],
    "Entertainment": ["CINEMA TICKET PURCHASE", "NETFLIX PPV CHARGE", "GAMING STORE PURCHASE"],
    "Shopping": ["AMAZON.EG ORDER", "MALL PURCHASE - POS", "ONLINE STORE CHECKOUT"],
    "Subscriptions": ["NETFLIX.COM SUBSCRIPTION", "SPOTIFY AB", "GYM MEMBERSHIP AUTOPAY"],
    "Healthcare": ["PHARMACY PURCHASE", "CLINIC VISIT FEE", "LAB TEST PAYMENT"],
    "Savings Transfer": ["TRANSFER TO SAVINGS ACCT", "INTERNAL XFER - SAVINGS"],
}

GOAL_TEMPLATES = [
    ("Emergency Fund", 0.5, 1.0, 6, 12),
    ("New Car", 1.0, 3.0, 12, 36),
    ("Wedding", 1.5, 4.0, 12, 24),
    ("Home Down Payment", 3.0, 8.0, 24, 60),
    ("Vacation", 0.3, 0.8, 3, 9),
]


# Customers
def generate_customers(n):
    customers = []
    for _ in range(n):
        income = round(random.uniform(6000, 40000), -2)
        product = random.choice(PRODUCTS[:2])  # customers hold Checking or Savings as their primary account
        customers.append({
            "customer_id": f"CUST{str(uuid.uuid4())[:6].upper()}",
            "name": fake.name(),
            "age": random.randint(22, 60),
            "job": random.choice(JOBS),
            "marital_status": random.choice(["Single", "Married"]),
            "monthly_income": income,
            "account_type": random.choice(["Current", "Savings", "Current+Savings"]),
            "product_code": product["product_code"],
        })
    return pd.DataFrame(customers)


#Transactions (with running balance)
def _split_amount(total, n):
    if n == 1:
        return [round(total, 2)]
    cuts = sorted(random.uniform(0, total) for _ in range(n - 1))
    parts = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, len(cuts))] + [total - cuts[-1]]
    return [round(max(p, 5), 2) for p in parts]


def _fake_description(category):
    template = random.choice(MERCHANT_TEMPLATES.get(category, [category]))
    return template.format(num=random.randint(100, 999), bank=random.choice(["NBE", "AUTOPAY", "ONLINE"]))


def generate_transactions(customers_df, num_months):
    transactions = []

    for _, cust in customers_df.iterrows():
        income = cust["monthly_income"]
        running_balance = round(random.uniform(2000, 15000), 2)  # starting balance
        has_anomaly_month = random.random() < ANOMALY_CHANCE
        anomaly_month_index = random.randint(0, num_months - 1) if has_anomaly_month else -1

        # Each customer gets a FIXED base spending % per category, drawn once.
        # This represents their personal "normal" habit for that category.
        # Month-to-month, we only apply small noise around this base (+-10%),
        # so a real anomaly month stands out clearly instead of being lost
        # in random noise that changes every month for every category.
        base_pct_by_category = {
            category: random.uniform(*profile["pct"])
            for category, profile in CATEGORY_PROFILE.items()
        }

        # collect all txns for this customer first, then sort by date to compute running balance correctly
        cust_txns = []

        for month_idx in range(num_months):
            # Use real calendar-month arithmetic (not a fixed 30-day step) so
            # months never drift into or collide with each other.
            month_date = START_DATE + pd.DateOffset(months=month_idx)

            salary_day = random.randint(1, 3)
            cust_txns.append({
                "date": month_date.replace(day=salary_day),
                "category": "Salary",
                "description": "MONTHLY SALARY CREDIT",
                "amount": round(income, 2),
                "direction": "credit",
                "is_anomaly": False,
            })

            for category, profile in CATEGORY_PROFILE.items():
                is_anomaly = (month_idx == anomaly_month_index and category in ANOMALY_CATEGORIES)

                base_pct = base_pct_by_category[category]
                # small natural month-to-month noise around the customer's own baseline
                noise_factor = random.uniform(0.9, 1.1)
                pct = base_pct * noise_factor

                if is_anomaly:
                    pct *= random.uniform(2.5, 4.0)

                category_total = round(income * pct, 2)
                if category_total <= 0:
                    continue

                num_txns = 1 if profile["type"] == "fixed" else random.randint(2, 5)
                for amt in _split_amount(category_total, num_txns):
                    txn_day = random.randint(1, 28)
                    cust_txns.append({
                        "date": month_date.replace(day=txn_day),
                        "category": category,
                        "description": _fake_description(category),
                        "amount": amt,
                        "direction": "debit",
                        "is_anomaly": is_anomaly,
                    })

        # sort chronologically and compute running balance
        cust_txns.sort(key=lambda t: t["date"])
        for t in cust_txns:
            if t["direction"] == "credit":
                running_balance += t["amount"]
            else:
                running_balance -= t["amount"]
            transactions.append({
                "transaction_id": str(uuid.uuid4())[:10],
                "customer_id": cust["customer_id"],
                "date": t["date"].strftime("%Y-%m-%d"),
                "category": t["category"],
                "description": t["description"],
                "amount": t["amount"],
                "direction": t["direction"],
                "balance_after": round(running_balance, 2),
                "is_anomaly": t["is_anomaly"],
            })

    return pd.DataFrame(transactions)


#Goals
def generate_goals(customers_df):
    goals = []
    for _, cust in customers_df.iterrows():
        # each customer gets 1-2 goals
        num_goals = random.randint(1, 2)
        chosen = random.sample(GOAL_TEMPLATES, num_goals)
        for goal_name, min_income_mult, max_income_mult, min_dur, max_dur in chosen:
            target_amount = round(cust["monthly_income"] * random.uniform(min_income_mult, max_income_mult) * 3, -2)
            duration_months = random.randint(min_dur, max_dur)
            monthly_required = round(target_amount / duration_months, 2)
            goals.append({
                "goal_id": f"GOAL{str(uuid.uuid4())[:6].upper()}",
                "customer_id": cust["customer_id"],
                "goal_name": goal_name,
                "target_amount": target_amount,
                "duration_months": duration_months,
                "monthly_required": monthly_required,
                "start_date": START_DATE.strftime("%Y-%m-%d"),
            })
    return pd.DataFrame(goals)


# Products catalog (for RAG "advisory Agent")
def generate_products_catalog():
    return PRODUCTS


# Main
if __name__ == "__main__":
    print("Generating customers...")
    customers_df = generate_customers(NUM_CUSTOMERS)

    print("Generating transactions (with running balance)...")
    transactions_df = generate_transactions(customers_df, NUM_MONTHS)

    print("Generating savings goals...")
    goals_df = generate_goals(customers_df)

    print("Generating products catalog...")
    products = generate_products_catalog()

    customers_df.to_csv(f"{OUTPUT_DIR}/customers.csv", index=False)
    transactions_df.to_csv(f"{OUTPUT_DIR}/transactions.csv", index=False)
    goals_df.to_csv(f"{OUTPUT_DIR}/goals.csv", index=False)
    with open(f"{OUTPUT_DIR}/products_catalog.json", "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)

    print(f"\nDone:")
    print(f"  customers.csv         -> {len(customers_df)} rows")
    print(f"  transactions.csv      -> {len(transactions_df)} rows")
    print(f"  goals.csv             -> {len(goals_df)} rows")
    print(f"  products_catalog.json -> {len(products)} products")
    print(f"\nAll files saved in ./{OUTPUT_DIR}/")