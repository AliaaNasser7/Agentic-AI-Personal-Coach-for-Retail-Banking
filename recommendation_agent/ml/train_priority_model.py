"""
(Re)trains the Recommendation Agent's ML priority model.

NOT required to use the Recommendation Agent — a trained model is
already checked in at ml/model/priority_model.joblib, so nothing needs
to run before merging or demoing. This script is only here for later,
if the team wants to retrain on fresh data or real outcome labels (see
"HOW LABELS ARE MADE" below).

WHAT IT PREDICTS
-----------------
A 0-100 "priority score": how urgently a customer's situation calls for
action, vs. one that's healthy and just needs monitoring. It's surfaced
in the Recommendation Agent's output as `priority: {score, label,
model}`, alongside the normal rule-based recommendations — it does NOT
decide which recommendations are shown or their order; that logic is
unchanged and still fully rule-based/explainable. The ML layer is a
second, additive, learned read on top: "how urgent is this overall?"

HOW LABELS ARE MADE (be upfront about this)
---------------------------------------------
There's no historical "did the customer act on this" outcome data
available yet (no acceptance logs, no missed-goal events over time) to
learn from. So training labels come from a transparent, documented
formula (`_label_priority` below) combining four already-computed
signals: overspending severity, this month's goal funding gap, how far
behind the customer's actual saving pace is vs. their plan, and how
thin their surplus is. The model's job isn't to discover this formula —
it already knows it. Its job is to learn a smooth, generalizable
approximation of it directly from the raw feature vector, the same way
a real deployment would eventually retrain it on actual outcomes (goal
misses, overdrafts, accepted vs. ignored recommendations) once that
data exists — at that point, only `_label_priority()` needs to change;
feature extraction, training, and the Recommendation Agent's
integration all stay the same.

WHY THIS SCRIPT ISN'T SELF-CONTAINED
---------------------------------------
Building a real training set means running the actual Spending, Goals,
and Simulation agents for every customer -- their code isn't vendored
here (see this package's README: "don't recreate teammates' code"), so
this script imports them the exact same flat way agent_stubs.py already
does. That means it only runs successfully from INSIDE the merged
coordinator repo, once goal_agent.py, spending_agent_tools.py, and the
shared/simulation_agent packages are all importable (i.e. after all
branches are merged) -- not from this standalone recommendation_agent
folder on its own.

Run from the merged repo's root (once everything is merged):
    python -m recommendation_agent.ml.train_priority_model
"""
import os
import sys

try:
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold, cross_val_score
except ImportError as e:
    sys.exit(
        f"Missing dependency ({e}). Run: pip install scikit-learn joblib numpy pandas --break-system-packages"
    )

try:
    # Same flat-import convention agent_stubs.py already uses once
    # everything is merged -- see this file's docstring.
    from data_access import load_customers
    from spending_agent_tools import analyze_customer, SpendingAgentError
    from goal_agent import GoalAgent, GoalAgentConfig
    from shared.finance_utils import load_data, load_products
    from simulation_agent import build_simulation_output
except ImportError as e:
    sys.exit(
        f"Could not import the other three agents ({e}).\n"
        "This script only runs from inside the merged coordinator repo, "
        "after goal_agent.py / spending_agent_tools.py / shared+simulation_agent "
        "are all on sys.path -- see this file's docstring. It is NOT required "
        "to use the Recommendation Agent; the trained model is already included."
    )

from .features import FEATURE_NAMES, extract_features, features_to_vector

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_THIS_DIR, "model", "priority_model.joblib")

# Weights for the documented priority formula below. Each term is
# already a 0-1ish ratio (see features.py), so the weights double as
# "how much this signal matters relative to the others" and sum to 1.0.
_WEIGHTS = {
    "overspend_severity": 0.25,
    "goal_gap_ratio": 0.30,
    "months_behind_ratio": 0.25,   # divided by 2 below since its natural range is 0-5, not 0-1
    "low_surplus": 0.20,
}


def _label_priority(features: dict) -> float:
    """The documented, transparent priority formula training labels come
    from -- see the module docstring for why this exists instead of
    real outcome data, and what changes when real outcome data is
    available."""
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
    customer in the shared data folder, extracts features, and computes
    the label formula above -- one row per customer."""
    customers = load_customers()["customer_id"].tolist()
    customers_df, transactions, goals_df, products = load_data("output")

    goal_agent = GoalAgent(
        customers_path="output/customers.csv",
        goals_path="output/goals.csv",
        transactions_path="output/transactions.csv",
        products_path="output/products_catalog.json",
        config=GoalAgentConfig(use_llm_message=False),
    )

    X, y, rows = [], [], []
    for customer_id in customers:
        try:
            spending = analyze_customer(customer_id)
        except SpendingAgentError:
            spending = None

        goals_result = goal_agent.recommend(customer_id)
        goals = None if "error" in goals_result else goals_result

        try:
            simulation = build_simulation_output(
                customer_id, transactions, goals_df, products,
                months_ahead=6, top_n=3, use_llm_polish=False,
            )
            if "error" in simulation:
                simulation = None
        except Exception:
            simulation = None

        if spending is None and goals is None and simulation is None:
            continue

        features = extract_features(spending, goals, simulation)
        X.append(features_to_vector(features))
        y.append(_label_priority(features))
        rows.append(customer_id)

    return np.array(X), np.array(y), rows


def train():
    print("Building training set from the real Spending/Goals/Simulation pipeline...")
    X, y, customer_ids = build_dataset()
    print(f"  {len(customer_ids)} customers, {X.shape[1]} features each.")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,          # dataset is small -- keep the model shallow to avoid overfitting
        min_samples_leaf=3,
        random_state=42,
    )

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
