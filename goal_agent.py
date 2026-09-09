"""
Goal Agent - Instant Installment Advisor (generalized version)
==================================================================
What this agent does (in plain words):

A customer has a financial goal (from goals.csv) and needs to set aside a
certain amount every month (monthly_required) to reach it on time.

The agent:
    1) Calculates how much money the customer actually has "free" each month
       after paying for essentials (disposable income), looking at ONE
       month of transactions at a time.
    2) Compares that to what the goal requires. If disposable income is less
       than what's required, there's a "gap" that needs to be closed.
    3) Looks through the customer's transactions FOR THAT SAME MONTH for
       big, non-essential purchases (Electronics, Travel, Shopping...) that
       are good candidates to convert into installments.
    4) Checks the products catalog for actual installment plans (if any
       exist there). If none exist, it falls back to a default fee rate.
    5) Picks the installment tenor (3/6/12 months by default) that best
       closes the monthly gap, without choosing a longer tenor than
       necessary.

GENERALIZATION
--------------
This version is designed to run on ANY customer and ANY dataset that
follows the expected column names, not just the specific sample data it
was first built against:

  - All bank-specific / dataset-specific knobs (which categories count as
    "non-essential", minimum eligible purchase amount, tenor options,
    default fee) live in `GoalAgentConfig`, separate from the logic. To
    adapt the agent to a different bank's policy, you change the config,
    not the code.
  - The minimum eligible purchase amount scales with the customer's
    income (a flat floor OR a percentage of income, whichever is bigger),
    instead of one fixed number that only makes sense for one income
    bracket.
  - The dataset's shape is validated up front. If a required column is
    missing, the agent stops with a clear error message instead of
    failing deep inside the logic with a confusing KeyError.
  - `recommend_all()` runs the agent for every customer that has a goal,
    instead of requiring a hard-coded customer_id.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set

import pandas as pd
import requests


# Columns each input file MUST have for the agent to work at all.
# (Extra columns beyond these are fine and simply ignored.)
REQUIRED_COLUMNS = {
    "customers.csv": {"customer_id", "monthly_income"},
    "goals.csv": {"goal_id", "customer_id", "goal_name", "target_amount", "monthly_required"},
    "transactions.csv": {"transaction_id", "customer_id", "date", "category", "amount", "direction"},
}


@dataclass
class GoalAgentConfig:
    """
    All the bank-specific / dataset-specific knobs live here, so the
    agent's core logic never has to change when you point it at a new
    dataset or a new bank policy - you just build a different config.
    """
    # Categories treated as "non-essential" (candidates for installments).
    non_essential_categories: Set[str] = field(default_factory=lambda: {
        "Shopping", "Electronics", "Travel", "Dining", "Entertainment", "Furniture"
    })

    # A purchase must be at least this amount, OR at least this fraction
    # of the customer's monthly income (whichever is bigger), to be
    # considered for installments. The percentage makes the threshold
    # scale naturally across income brackets instead of using one flat
    # number for every customer.
    min_eligible_amount_flat: float = 300.0
    min_eligible_amount_pct_of_income: float = 0.02  # 2% of monthly income

    # Tenors to try if the products catalog has no matching installment
    # product for a given duration.
    tenor_options: List[int] = field(default_factory=lambda: [3, 6, 12])

    # Monthly fee used when no real installment product is found in the
    # catalog for a given tenor.
    default_fee_rate: float = 0.015

    # Name of the column that flags suspicious transactions, if present.
    anomaly_column: str = "is_anomaly"

    # ---- Llama / Ollama settings ----
    # Whether to ask a local Llama model (via Ollama) to turn the raw
    # numbers into a friendly customer-facing message. Off by default so
    # the agent still works with pure Python logic even if Ollama isn't
    # running - set to True once you have `ollama serve` up.
    use_llm_message: bool = False
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3.2"  # change to whatever model you `ollama pull`ed
    ollama_timeout_seconds: int = 30


@dataclass
class InstallmentSuggestion:
    transaction_id: str
    description: str
    category: str
    original_amount: float
    tenor_months: int
    fee_rate: float
    monthly_installment: float
    monthly_cash_freed: float
    contribution_to_goal_pct: float


class GoalAgent:
    def __init__(self, customers_path: str, goals_path: str,
                 transactions_path: str, products_path: str,
                 config: Optional[GoalAgentConfig] = None):
        self.config = config or GoalAgentConfig()

        self.customers = pd.read_csv(customers_path)
        self.goals = pd.read_csv(goals_path)
        self.transactions = pd.read_csv(transactions_path)
        with open(products_path, encoding="utf-8") as f:
            self.products = json.load(f)

        self._validate_schema()

        # Parse dates once and tag every transaction with its month, so we
        # can always look at "one month at a time" further down.
        self.transactions["date"] = pd.to_datetime(self.transactions["date"])
        self.transactions["month"] = self.transactions["date"].dt.to_period("M")

        # Only real, short-term "Installment" products count here - long
        # term Loan/Certificate products are a different kind of product
        # and are intentionally not used to price a purchase-to-installment
        # conversion. If the catalog gains a real Installment product for
        # a given duration, it is picked up automatically.
        self.installment_products_by_tenor = {
            p["duration_months"]: p
            for p in self.products
            if str(p.get("type", "")).lower() == "installment"
        }

    # ---------------------------------------------------------------
    # Schema validation - fail fast with a clear message instead of a
    # confusing crash somewhere deep in the calculations.
    # ---------------------------------------------------------------
    def _validate_schema(self):
        frames = {
            "customers.csv": self.customers,
            "goals.csv": self.goals,
            "transactions.csv": self.transactions,
        }
        problems = []
        for file_name, required in REQUIRED_COLUMNS.items():
            missing = required - set(frames[file_name].columns)
            if missing:
                problems.append(f"  - {file_name} is missing columns: {sorted(missing)}")
        if problems:
            raise ValueError(
                "GoalAgent cannot run - the dataset doesn't match the "
                "expected schema:\n" + "\n".join(problems)
            )

    # ---------------------------------------------------------------
    # Step 1: find data for one customer
    # ---------------------------------------------------------------
    def _get_customer(self, customer_id: str) -> Optional[Dict]:
        row = self.customers[self.customers["customer_id"] == customer_id]
        return row.iloc[0].to_dict() if not row.empty else None

    def _get_goals(self, customer_id: str) -> pd.DataFrame:
        return self.goals[self.goals["customer_id"] == customer_id]

    def _get_customer_transactions(self, customer_id: str) -> pd.DataFrame:
        return self.transactions[self.transactions["customer_id"] == customer_id]

    def _get_customer_transactions_for_month(
        self, customer_id: str, month: Optional[pd.Period] = None
    ) -> pd.DataFrame:
        """
        Return this customer's transactions for a SINGLE month. If `month`
        is not given, use the most recent month present in that
        customer's data (i.e. "this month" from the agent's point of view).
        """
        tx = self._get_customer_transactions(customer_id)
        if tx.empty:
            return tx
        target_month = month if month is not None else tx["month"].max()
        return tx[tx["month"] == target_month]

    # ---------------------------------------------------------------
    # Step 2: how much money does the customer actually have free?
    # ---------------------------------------------------------------
    def _min_eligible_amount(self, monthly_income: float) -> float:
        return max(
            self.config.min_eligible_amount_flat,
            monthly_income * self.config.min_eligible_amount_pct_of_income,
        )

    def _disposable_income(self, customer_id: str, monthly_income: float,
                            month: Optional[pd.Period] = None) -> float:
        tx = self._get_customer_transactions_for_month(customer_id, month)
        essential_debits = tx[
            (tx["direction"] == "debit")
            & (~tx["category"].isin(self.config.non_essential_categories))
        ]["amount"].sum()
        # float(...) forces a native Python float (essential_debits is a
        # numpy.float64 coming from pandas' .sum()).
        return round(float(monthly_income - essential_debits), 2)

    # ---------------------------------------------------------------
    # Step 3: which purchases could become installments?
    # ---------------------------------------------------------------
    def _installment_candidates(self, customer_id: str, monthly_income: float,
                                 month: Optional[pd.Period] = None) -> pd.DataFrame:
        tx = self._get_customer_transactions_for_month(customer_id, month)
        min_amount = self._min_eligible_amount(monthly_income)

        candidates = tx[
            (tx["direction"] == "debit")
            & (tx["category"].isin(self.config.non_essential_categories))
            & (tx["amount"] >= min_amount)
        ]
        anomaly_col = self.config.anomaly_column
        if anomaly_col in candidates.columns:
            candidates = candidates[candidates[anomaly_col] == False]  # noqa: E712
        return candidates.sort_values("amount", ascending=False)

    # ---------------------------------------------------------------
    # Step 4: figure out the fee for a given tenor
    # ---------------------------------------------------------------
    def _fee_rate_for_tenor(self, tenor: int) -> float:
        product = self.installment_products_by_tenor.get(tenor)
        if product is not None:
            fee = product.get("expected_return")
            if fee is not None:
                return fee
        return self.config.default_fee_rate

    # ---------------------------------------------------------------
    # Step 5: build installment options for one purchase
    # ---------------------------------------------------------------
    def _build_tenor_options(self, transaction: pd.Series) -> List[InstallmentSuggestion]:
        amount = float(transaction["amount"])
        options = []
        for tenor in self.config.tenor_options:
            fee_rate = self._fee_rate_for_tenor(tenor)
            total_payable = amount * (1 + fee_rate * tenor)
            monthly_installment = round(total_payable / tenor, 2)
            monthly_cash_freed = round(amount - monthly_installment, 2)
            options.append(InstallmentSuggestion(
                transaction_id=transaction["transaction_id"],
                description=transaction.get("description", ""),
                category=transaction["category"],
                original_amount=amount,
                tenor_months=tenor,
                fee_rate=fee_rate,
                monthly_installment=monthly_installment,
                monthly_cash_freed=monthly_cash_freed,
                contribution_to_goal_pct=0.0,  # filled in later, once we know the goal
            ))
        return options

    # ---------------------------------------------------------------
    # Step 6: the main entry point for ONE customer
    # ---------------------------------------------------------------
    def recommend(self, customer_id: str, month: Optional[str] = None) -> Dict:
        """
        month: optional, format "YYYY-MM" (e.g. "2026-03"). If not given,
        the agent uses the most recent month found in that customer's
        transaction history.
        """
        customer = self._get_customer(customer_id)
        if customer is None:
            return {"error": f"Customer {customer_id} not found"}

        goals = self._get_goals(customer_id)
        if goals.empty:
            return {"error": f"No goals found for customer {customer_id}"}

        target_month = pd.Period(month, freq="M") if month else None

        monthly_income = float(customer["monthly_income"])
        disposable = self._disposable_income(customer_id, monthly_income, target_month)
        candidates = self._installment_candidates(customer_id, monthly_income, target_month)

        goals_output = []
        used_transaction_ids: Set[str] = set()  # tracked across ALL goals of this
        # customer, so the same purchase isn't suggested (and its cash-freed
        # counted) more than once when the customer has multiple goals.

        for _, goal in goals.iterrows():
            monthly_required = float(goal["monthly_required"])
            gap = round(monthly_required - disposable, 2)

            goal_result = {
                "goal_id": goal["goal_id"],
                "goal_name": goal["goal_name"],
                "target_amount": float(goal["target_amount"]),
                "monthly_required": monthly_required,
                "disposable_income": disposable,
                "monthly_gap": gap,
                "status": "on_track" if gap <= 0 else "needs_action",
                "suggestions": [],
            }

            remaining_gap = gap
            if remaining_gap > 0:
                # Skip any transaction already suggested for an earlier
                # goal of this same customer in this same run.
                available_candidates = candidates[
                    ~candidates["transaction_id"].isin(used_transaction_ids)
                ]
                for _, tx in available_candidates.iterrows():
                    if remaining_gap <= 0:
                        break

                    options = self._build_tenor_options(tx)
                    best = min(options, key=lambda o: abs(o.monthly_cash_freed - remaining_gap))
                    best.contribution_to_goal_pct = round(
                        min(100.0, (best.monthly_cash_freed / monthly_required) * 100), 1
                    )

                    goal_result["suggestions"].append(asdict(best))
                    used_transaction_ids.add(best.transaction_id)
                    remaining_gap = round(remaining_gap - best.monthly_cash_freed, 2)

                goal_result["remaining_gap_after_suggestions"] = max(remaining_gap, 0)

            # Optional: ask the local Llama model (via Ollama) to turn
            # these numbers into a friendly message. Only runs if
            # config.use_llm_message is True; otherwise this is skipped
            # entirely and the agent behaves exactly as before.
            customer_name = customer.get("name", "")
            goal_result["customer_message"] = self._generate_customer_message(
                customer_name, goal_result
            )

            goals_output.append(goal_result)

        customer_tx = self._get_customer_transactions(customer_id)
        analyzed_month = target_month or (
            customer_tx["month"].max() if not customer_tx.empty else None
        )

        return {
            "customer_id": customer_id,
            "customer_name": customer.get("name"),
            "monthly_income": monthly_income,
            "analyzed_month": str(analyzed_month),
            "goals": goals_output,
        }

    # ---------------------------------------------------------------
    # Step 6b: turn the raw numbers into a friendly message using a
    # local Llama model served by Ollama.
    # ---------------------------------------------------------------
    def _call_ollama(self, prompt: str) -> Optional[str]:
        """
        Send a prompt to a local Ollama server and return the generated
        text, or None if Ollama isn't reachable (so the rest of the
        agent keeps working even if the LLM step fails).

        Requires Ollama running locally:
            ollama pull llama3.2   (one-time)
            ollama serve           (must be running while this agent runs)
        """
        try:
            response = requests.post(
                self.config.ollama_url,
                json={
                    "model": self.config.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=self.config.ollama_timeout_seconds,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.RequestException as e:
            # Ollama not running, wrong port, model not pulled, etc.
            # We don't want a missing LLM to crash the whole recommendation,
            # so we just report it and move on without a message.
            print(f"[GoalAgent] Could not reach Ollama ({e}); skipping LLM message.")
            return None

    def _build_message_prompt(self, customer_name: str, goal_result: Dict) -> str:
        """
        Turn one goal's structured result into a plain-language prompt
        for the LLM, asking for an English customer-facing message.
        """
        lines = [
            "You are a friendly financial coach inside a banking app. "
            "Write a short in-app notification (3-4 sentences, in English) "
            "for the customer, based ONLY on the data below - do not invent "
            "any numbers or suggest anything not mentioned here.",
            "",
            "STRICT RULES - follow all of them:",
            "- Do not wrap the message in quotation marks.",
            "- Do not add a greeting sign-off or signature (no 'Best,', "
            "no '[Your Name]', no '[Financial Coach]', nothing like that).",
            "- Do not use any placeholder text in brackets like [amount] or "
            "[Name] - only use the exact numbers given below.",
            "- All amounts are in Egyptian Pounds - always write them as "
            "'EGP', never '$' or '£' or any other currency symbol.",
            "- Mention at most ONE specific installment suggestion by name, "
            "even if more than one is listed below.",
            "- NEVER mention or imply any app feature, tool, or named "
            "product that is not explicitly listed below (no 'Monthly "
            "Payments feature', no 'Delayed Purchase feature', no named "
            "app tools of any kind) - these do not exist and inventing "
            "them misleads the customer.",
            "- If NO installment suggestions are listed below, do not "
            "invent an installment suggestion. You MAY offer general, "
            "well-known personal finance advice instead (e.g. the 50/30/20 "
            "budgeting rule, reducing discretionary spending, reviewing "
            "recurring subscriptions) as long as it is not presented as a "
            "specific app feature or tool.",
            "- Do not invent or reference any specific bill, payment, "
            "obligation, purchase, or event that is not explicitly listed "
            "in the data below (e.g. do not say things like 'pay your "
            "upcoming utility bill' unless a utility bill is actually "
            "listed as a suggestion) - only speak in general terms about "
            "spending habits, not specific invented events.",
            "",
            f"Customer name: {customer_name}",
            f"Goal name: {goal_result['goal_name']}",
            f"Amount they need to set aside this month: {goal_result['monthly_required']} EGP",
            f"What's actually available after essential expenses: {goal_result['disposable_income']} EGP",
            f"Monthly gap: {goal_result['monthly_gap']} EGP",
        ]

        if goal_result["status"] == "on_track":
            lines.append("The customer is on track with this goal and needs no changes.")
        else:
            if goal_result["suggestions"]:
                lines.append("Suggestions to convert purchases into installments to close the gap:")
                for s in goal_result["suggestions"]:
                    lines.append(
                        f"- Convert \"{s['description']}\" (amount {s['original_amount']} EGP) "
                        f"into a {s['tenor_months']}-month installment plan, freeing up "
                        f"{s['monthly_cash_freed']} EGP per month "
                        f"({s['contribution_to_goal_pct']}% of what's required)."
                    )
            else:
                lines.append(
                    "No installment suggestions are available this month (no eligible "
                    "purchases were found). Do not mention installments at all for this "
                    "goal - only general budgeting advice is appropriate here."
                )
            remaining = goal_result.get("remaining_gap_after_suggestions", 0)
            if remaining > 0:
                if goal_result["suggestions"]:
                    lines.append(f"Even after these suggestions, there's still a {remaining} EGP monthly shortfall.")
                else:
                    lines.append(f"There's a {remaining} EGP monthly shortfall to mention.")

        return "\n".join(lines)

    @staticmethod
    def _clean_llm_message(text: str) -> str:
        """
        Small post-processing safety net: local models sometimes wrap the
        whole reply in quotation marks even when told not to. Strip a
        single layer of leading/trailing quotes (straight or curly) and
        surrounding whitespace.
        """
        text = text.strip()
        quote_pairs = [('"', '"'), ("'", "'"), (""", """), (""", """)]
        for open_q, close_q in quote_pairs:
            if text.startswith(open_q) and text.endswith(close_q) and len(text) > 1:
                text = text[1:-1].strip()
        return text

    def _generate_customer_message(self, customer_name: str, goal_result: Dict) -> Optional[str]:
        if not self.config.use_llm_message:
            return None
        prompt = self._build_message_prompt(customer_name, goal_result)
        message = self._call_ollama(prompt)
        if message:
            message = self._clean_llm_message(message)
        return message

    # ---------------------------------------------------------------
    # Step 7: run the agent for EVERY customer that has a goal
    # ---------------------------------------------------------------
    def recommend_all(self, month: Optional[str] = None) -> List[Dict]:
        """
        This is what makes the agent usable on a whole dataset instead of
        one hard-coded customer_id: it loops through every customer_id
        that appears in goals.csv and runs `recommend` for each one.
        """
        customer_ids = self.goals["customer_id"].unique()
        return [self.recommend(cid, month) for cid in customer_ids]


if __name__ == "__main__":
    import sys

    args = [a for a in sys.argv[1:] if a != "--message"]

    # Always try to use the local Llama model when running this file
    # directly. If Ollama isn't running, the agent still works fine -
    # customer_message will just be None (see _call_ollama's fallback).
    config = GoalAgentConfig(use_llm_message=True)
    agent = GoalAgent(
        customers_path="customers.csv",
        goals_path="goals.csv",
        transactions_path="transactions.csv",
        products_path="products_catalog.json",
        config=config,
    )

    # Usage:
    #   python3 goal_agent.py CUST62A0AF              -> one customer, latest month
    #   python3 goal_agent.py CUST62A0AF 2026-03       -> one customer, specific month
    #   python3 goal_agent.py --all                    -> every customer, latest month each
    #   python3 goal_agent.py --all 2026-03             -> every customer, specific month
    if len(args) > 0 and args[0] != "--all":
        customer_id = args[0]
        month_arg = args[1] if len(args) > 1 else None
        result = agent.recommend(customer_id, month_arg)
    else:
        month_arg = args[1] if len(args) > 1 else None
        result = agent.recommend_all(month_arg)

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))