"""
Trains the Recommendation Agent's ML priority model.

WHAT IT PREDICTS
-----------------
A 0-100 "priority score": how urgently a customer's situation calls for
action, versus one that's healthy and just needs monitoring. This is
surfaced in the Recommendation Agent's output (priority_score,
priority_label) alongside the existing deterministic recommendations —
it does NOT decide which recommendations are shown or in what order;
that logic is unchanged and still fully rule-based/explainable. The ML
layer adds a second, learned lens on top: "how urgent is this overall?"

HOW LABELS ARE MADE (be upfront about this)
---------------------------------------------
There's no historical "did the customer act on this" outcome data in
this project (no acceptance logs, no missed-goal events over time) to
learn from. So this script builds training LABELS from a transparent,
documented formula that combines four already-computed signals:
overspending severity, this month's goal funding gap, how far behind
the customer's actual saving pace is versus their plan, and how thin
their surplus is. That formula IS the "ground truth" priority score for
every customer in the training set.

The model's job isn't to discover this formula — it already knows it
from the docstring below. Its job is to LEARN A SMOOTH, GENERALIZABLE
APPROXIMATION of it directly from the raw feature vector, the same way
a real deployment would eventually retrain it on actual customer
outcomes (goal misses, overdrafts, churn, accepted vs. ignored
recommendations) once that data exists — at that point, only
`_label_priority()` below needs to change; the feature extraction,
training loop, and the Recommendation Agent's integration all stay
the same.

Run with (from the project root):
    python -m agents.ml.train_priority_model
"""
import os
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data_access import load_customers
from agents.spending_agent import analyze_customer as run_spending_agent, SpendingAgentError
from agents.goals_agent import run as run_goals_agent, GoalsAgentError
from agents.simulation_agent import run as run_simulation_agent, SimulationAgentError
from agents.ml.features import FEATURE_NAMES, extract_features, features_to_vector

MODEL_PATH = os.path.join(_THIS_DIR, "model", "priority_model.joblib")

# Weights for the documented priority formula. Each term is already a
# 0-1ish ratio (see features.py), so the weights double as "how much
# this signal matters relative to the others" and sum to 1.0.
_WEIGHTS = {
    "overspend_severity": 0.25,
    "goal_gap_ratio": 0.30,
    "months_behind_ratio": 0.25,   # divided by 2 below since its natural range is 0-5, not 0-1
    "low_surplus": 0.20,           # = 1 - surplus_ratio, clipped
}


def _label_priority(features: dict) -> float:
    """The documented, transparent priority formula training labels come
    from. See the module docstring for why this exists instead of real
    outcome data, and what changes when real outcome data is available."""
    low_surplus = max(0.0, min(1.0, 1.0 - features["surplus_ratio"]))
    months_behind_component = min(1.0, features["months_behind_ratio"] / 2.0)
    score = (
        _WEIGHTS["overspend_severity"] * min(1.0, features["overspend_severity"])
        + _WEIGHTS["goal_gap_ratio"] * features["goal_gap_ratio"]
        + _WEIGHTS["months_behind_ratio"] * months_behind_component
        + _WEIGHTS["low_surplus"] * low_surplus
    )
    return round(100 * max(0.0, min(1.0, score)), 2)


def build_dataset():
    """Runs the real pipeline (Spending, Goals, Simulation) for every
    customer in data/customers.csv, extracts features, and computes the
    label formula above -- one row per customer."""
    customers = load_customers()["customer_id"].tolist()
    X, y, rows = [], [], []

    for customer_id in customers:
        try:
            spending = run_spending_agent(customer_id)
        except SpendingAgentError:
            spending = None
        try:
            goals = run_goals_agent(customer_id)
        except GoalsAgentError:
            goals = None
        try:
            simulation = run_simulation_agent(customer_id, spending, goals)
        except SimulationAgentError:
            simulation = None

        if spending is None and goals is None and simulation is None:
            continue

        features = extract_features(spending, goals, simulation)
        label = _label_priority(features)

        X.append(features_to_vector(features))
        y.append(label)
        rows.append(customer_id)

    return np.array(X), np.array(y), rows


def train():
    print("Building training set from the real Spending/Goals/Simulation pipeline...")
    X, y, customer_ids = build_dataset()
    print(f"  {len(customer_ids)} customers, {X.shape[1]} features each.")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,          # dataset is small (~50 rows) -- keep the model shallow to avoid overfitting
        min_samples_leaf=3,
        random_state=42,
    )

    # 5-fold cross-validation, since holding out a separate test set from
    # ~50 rows would leave too little data for either split to be meaningful.
    scores = cross_val_score(model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring="r2")
    print(f"  5-fold CV R^2: {scores.mean():.3f} (+/- {scores.std():.3f})")

    model.fit(X, y)

    importances = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda t: -t[1])
    print("  Feature importances:")
    for name, imp in importances:
        print(f"    {name:<26} {imp:.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, MODEL_PATH)
    print(f"  Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train()
