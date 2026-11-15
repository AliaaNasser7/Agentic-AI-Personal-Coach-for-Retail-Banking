"""
Spending Agent: Unit Tests
Formal tests for spending_agent_tools.py, covering:
- A normal customer (real data)
- A customer with a REAL injected anomaly (real data)
- The lowest-income customer (real data)
- An edge case with only 1 month of history (synthetic/mocked data,
since our current dataset always has 6 months per customer)
- Error handling (invalid customer_id, missing data)
- categorize_transaction_description() keyword matching
- JSON-serializability of analyze_customer() output (numpy float bug)

Python's built-in "unittest" (works even if pytest isn't set up on another machine)
-python -m unittest test_spending_agent.py -v
"""

import json
import unittest
from unittest.mock import patch

import pandas as pd

import spending_agent_tools as sat


class TestNormalCustomer(unittest.TestCase):    #customer from the dataset
    @classmethod
    def setUpClass(cls):
        customers = sat._load_customers()
        # picking any customer with no anomaly for a clean baseline
        txns = sat._load_transactions()
        anomaly_ids = set(txns[txns["is_anomaly"] == True]["customer_id"])
        non_anomaly = customers[~customers["customer_id"].isin(anomaly_ids)]
        cls.customer_id = non_anomaly.iloc[0]["customer_id"]

    def test_get_customer_transactions_returns_rows(self):
        txns = sat.get_customer_transactions(self.customer_id)
        self.assertFalse(txns.empty)
        self.assertTrue((txns["customer_id"] == self.customer_id).all())

    def test_surplus_is_income_minus_spending(self):
        result = sat.calculate_monthly_surplus(self.customer_id)
        expected_surplus = round(
            result["avg_monthly_income"] - result["avg_monthly_spending"], 2
        )
        self.assertAlmostEqual(result["monthly_surplus"], expected_surplus, places=2)

    def test_months_analyzed_matches_data(self):
        result = sat.calculate_monthly_surplus(self.customer_id)
        self.assertEqual(result["months_analyzed"], 6)  # our dataset generates 6 months

    def test_no_false_positive_overspending(self):
        # A customer with no injected anomaly should not be flagged
        flags = sat.detect_overspending_categories(self.customer_id)
        self.assertEqual(flags, [])


class TestAnomalyCustomer(unittest.TestCase):
    #Verify detection actually catches a REAL injected anomaly.

    @classmethod
    def setUpClass(cls):
        txns = sat._load_transactions()
        anomaly_ids = txns[txns["is_anomaly"] == True]["customer_id"].unique()
        if len(anomaly_ids) == 0:
            raise unittest.SkipTest("No anomaly customers in current dataset, regenerate data with anomalies.")
        cls.customer_id = anomaly_ids[0]

    def test_detects_at_least_one_flag(self):
        flags = sat.detect_overspending_categories(self.customer_id)
        self.assertGreater(len(flags), 0, "Expected at least one overspending flag for a known anomaly customer")

    def test_flagged_category_was_actually_anomalous(self):
        txns = sat._load_transactions()
        cust_txns = txns[txns["customer_id"] == self.customer_id]
        true_anomaly_categories = set(cust_txns[cust_txns["is_anomaly"] == True]["category"])

        flags = sat.detect_overspending_categories(self.customer_id)
        flagged_categories = {f["category"] for f in flags}

        # every flagged category should overlap with a real anomaly category
        self.assertTrue(flagged_categories & true_anomaly_categories)


class TestLowIncomeCustomer(unittest.TestCase):
    #The lowest income customer in the dataset should still work correctly (no division by zero, no negative amount crashes)

    @classmethod
    def setUpClass(cls):
        customers = sat._load_customers()
        cls.customer_id = customers.sort_values("monthly_income").iloc[0]["customer_id"]

    def test_analysis_runs_without_error(self):
        result = sat.analyze_customer(self.customer_id)
        self.assertIn("monthly_surplus", result)

    def test_income_is_positive(self):
        result = sat.calculate_monthly_surplus(self.customer_id)
        self.assertGreater(result["avg_monthly_income"], 0)


class TestSingleMonthCustomer(unittest.TestCase):
    """Edge case: a customer with only one month of transaction history,
    the real dataset always has 6 months, so I build a small synthetic
    transactions table for this one and patch it in"""

    def setUp(self):
        self.fake_txns = pd.DataFrame([
            {"transaction_id": "t1", "customer_id": "CUSTSINGLE", "date": pd.Timestamp("2026-01-03"),
            "category": "Salary", "description": "SALARY", "amount": 10000.0,
            "direction": "credit", "balance_after": 10000.0, "is_anomaly": False},
            {"transaction_id": "t2", "customer_id": "CUSTSINGLE", "date": pd.Timestamp("2026-01-10"),
            "category": "Groceries", "description": "CARREFOUR", "amount": 800.0,
            "direction": "debit", "balance_after": 9200.0, "is_anomaly": False},
        ])

    @patch("spending_agent_tools._load_transactions")
    def test_surplus_works_with_single_month(self, mock_load):
        mock_load.return_value = self.fake_txns
        result = sat.calculate_monthly_surplus("CUSTSINGLE")
        self.assertEqual(result["months_analyzed"], 1)
        self.assertEqual(result["monthly_surplus"], 10000.0 - 800.0)

    @patch("spending_agent_tools._load_transactions")
    def test_overspending_returns_empty_not_error(self, mock_load):
        # With only 1 month there's nothing to compare against so the function should return an empty list, not crash
        mock_load.return_value = self.fake_txns
        flags = sat.detect_overspending_categories("CUSTSINGLE")
        self.assertEqual(flags, [])


class TestErrorHandling(unittest.TestCase):

    def test_invalid_customer_id_raises_spending_agent_error(self):
        with self.assertRaises(sat.SpendingAgentError):
            sat.get_customer_transactions("DOES_NOT_EXIST_ID")

    def test_empty_customer_id_raises_spending_agent_error(self):
        with self.assertRaises(sat.SpendingAgentError):
            sat.get_customer_transactions("")

    def test_none_customer_id_raises_spending_agent_error(self):
        with self.assertRaises(sat.SpendingAgentError):
            sat.get_customer_transactions(None)


class TestCategorization(unittest.TestCase):

    def test_known_merchant_keywords(self):
        cases = {
            "MCDONALDS #482": "Dining/Restaurants",
            "ACH DEBIT - RENT PAYMENT": "Rent/Installment",
            "UBER *TRIP 213": "Transportation",
            "CARREFOUR HYPERMARKET": "Groceries",
            "NETFLIX.COM SUBSCRIPTION": "Subscriptions",
            "PHARMACY PURCHASE": "Healthcare",
        }
        for description, expected_category in cases.items():
            with self.subTest(description=description):
                self.assertEqual(sat.categorize_transaction_description(description), expected_category)

    def test_unknown_text_returns_uncategorized(self):
        self.assertEqual(sat.categorize_transaction_description("XYZ RANDOM TEXT 123"), "Uncategorized")

    def test_empty_or_none_returns_uncategorized(self):
        self.assertEqual(sat.categorize_transaction_description(""), "Uncategorized")
        self.assertEqual(sat.categorize_transaction_description(None), "Uncategorized")


class TestJSONSerialization(unittest.TestCase):
    #the Coordinator needs to be able to json.dumps() this output directly

    def test_analyze_customer_output_is_json_serializable(self):
        customers = sat._load_customers()
        customer_id = customers.iloc[0]["customer_id"]
        result = sat.analyze_customer(customer_id)
        try:
            json.dumps(result)
        except TypeError as e:
            self.fail(f"analyze_customer() output is not JSON-serializable: {e}")
            


class TestGuardrails(unittest.TestCase):

    def setUp(self):
        from spending_agent_local import (
            check_for_hallucinated_categories,
            check_for_ignored_real_flags,
            check_for_surplus_deficit_contradiction,
        )
        self.check_hallucinated = check_for_hallucinated_categories
        self.check_ignored = check_for_ignored_real_flags
        self.check_contradiction = check_for_surplus_deficit_contradiction

    def test_catches_invented_category_when_none_flagged(self):
        # The original bug: model invents "dining" when nothing was flagged
        message = "Your dining out expenses are averaging around 2,500 EGP per month."
        result = self.check_hallucinated(message, overspending_summary=[])
        self.assertIn("Dining/Restaurants", result)

    def test_clean_message_with_no_flags_passes(self):
        message = "Great job managing your finances! Consider growing your savings."
        result = self.check_hallucinated(message, overspending_summary=[])
        self.assertEqual(result, [])

    def test_catches_denied_real_overspending(self):
        # The second bug found via the demo: model says "consistent with usual
        # habits" while ignoring real flags
        message = ("it's great to see that your spending is consistent with "
                    "your usual habits, which is a great sign of financial stability.")
        real_flags = [
            {"category": "Shopping", "vs_avg_pct": 208.5},
            {"category": "Entertainment", "vs_avg_pct": 189.6},
        ]
        result = self.check_ignored(message, real_flags)
        self.assertTrue(set(result) >= {"Shopping", "Entertainment"})

    def test_correctly_mentioning_a_real_flag_passes(self):
        message = "I noticed your Shopping spending was up 208.5% this month compared to usual."
        real_flags = [{"category": "Shopping", "vs_avg_pct": 208.5}]
        result = self.check_ignored(message, real_flags)
        self.assertEqual(result, [])

    def test_catches_vague_omission_without_reassurance_phrase(self):
        """where the model didn't use any "consistent with usual habits" phrasing, but still failed to
        name any of the real flagged categories. The guardrail
        must catch this based on the MISSING category name, not on
        specific wording, since the model rephrases itself every run"""
        message = ("I do want to point out that your spending in one category is "
                "running a bit higher than usual, with a notable increase in your "
                "monthly spending of 18931.72 EGP, which is 8.7% above your average income.")
        real_flags = [
            {"category": "Shopping", "vs_avg_pct": 208.5},
            {"category": "Entertainment", "vs_avg_pct": 189.6},
            {"category": "Healthcare", "vs_avg_pct": 184.5},
        ]
        result = self.check_ignored(message, real_flags)
        self.assertTrue(set(result) >= {"Shopping", "Entertainment", "Healthcare"})
    
    def test_no_real_flags_means_nothing_to_ignore(self):
        message = "Everything looks consistent with your usual habits, great job!"
        result = self.check_ignored(message, overspending_summary=[])
        self.assertEqual(result, [])
        
    
    def test_catches_deficit_wrongly_claimed_on_positive_surplus(self):
        """the model reused a real number (avg spending) but mislabeled it 
            as a "deficit" while the customer actually has a positive surplus.
        """
        message = ("your spending is slightly above your income, with an average "
                "monthly deficit of 22557.92 EGP, but it's great that you're managing.")
        result = self.check_contradiction(message, monthly_surplus=5142.08)
        self.assertIsNotNone(result)

    def test_no_contradiction_when_wording_matches_reality(self):
        message = "You're maintaining a healthy monthly surplus, great job!"
        result = self.check_contradiction(message, monthly_surplus=5142.08)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)