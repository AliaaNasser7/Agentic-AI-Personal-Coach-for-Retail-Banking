"""
Loads the trained ML priority model (see train_priority_model.py for how
it was trained and why) and exposes a single scoring function.

Fails soft on purpose: if the model file is missing or scikit-learn
isn't installed, predict_priority() returns None rather than raising,
so the Recommendation Agent can fall back to showing no priority score
instead of crashing the whole pipeline over an optional annotation.
"""
import os

from agents.ml.features import features_to_vector

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_THIS_DIR, "model", "priority_model.joblib")

_bundle = None
_load_error = None
_tried_loading = False


def _load():
    global _bundle, _load_error, _tried_loading
    if _tried_loading:
        return
    _tried_loading = True
    try:
        import joblib
        _bundle = joblib.load(MODEL_PATH)
    except Exception as e:  # missing file, missing sklearn/joblib, version mismatch, etc.
        _bundle = None
        _load_error = str(e)


def _label(score: float) -> str:
    if score >= 60:
        return "High"
    if score >= 30:
        return "Medium"
    return "Low"


def predict_priority(features: dict) -> dict:
    """Returns {"score": float 0-100, "label": "High"/"Medium"/"Low",
    "model": "RandomForestRegressor"} or None if the model isn't
    available (missing file, missing dependency)."""
    _load()
    if _bundle is None:
        return None

    model = _bundle["model"]
    vector = [features_to_vector(features)]
    try:
        score = float(model.predict(vector)[0])
    except Exception:
        return None

    score = max(0.0, min(100.0, score))
    return {
        "score": round(score, 1),
        "label": _label(score),
        "model": type(model).__name__,
    }


def is_available() -> bool:
    _load()
    return _bundle is not None


def load_error() -> str:
    _load()
    return _load_error
