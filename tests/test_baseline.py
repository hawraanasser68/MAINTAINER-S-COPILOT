"""Tests for the TF-IDF + LogReg baseline model."""

import json
from pathlib import Path

import joblib
import pytest

MODEL_PATH = Path(__file__).parent.parent / "models" / "baseline" / "tfidf_logreg.pkl"
METRICS_PATH = Path(__file__).parent.parent / "models" / "baseline" / "metrics.json"
CLASSES = {"bug", "feature", "docs", "question"}


@pytest.fixture(scope="module")
def pipeline():
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH} — run scripts/train_baseline.py"
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def metrics():
    assert METRICS_PATH.exists(), f"Metrics not found: {METRICS_PATH}"
    return json.loads(METRICS_PATH.read_text())


def test_model_predicts_all_classes(pipeline):
    samples = [
        ("IndexError in fit method when input has NaN", "Traceback shows line 42 in _fit_internal"),
        ("Add support for custom loss functions", "It would be useful to pass a callable loss"),
        ("Fix typo in LogisticRegression docstring", "The formula on line 3 is missing a bracket"),
        ("How do I use GridSearchCV with pipelines?", "I want to tune both the vectorizer and classifier"),
    ]
    texts = [f"{t} {b}" for t, b in samples]
    preds = pipeline.predict(texts)
    assert set(preds) == CLASSES, f"Model did not predict all 4 classes: {set(preds)}"


def test_model_returns_single_label_per_input(pipeline):
    preds = pipeline.predict(["some issue title body text"])
    assert len(preds) == 1
    assert preds[0] in CLASSES


def test_metrics_macro_f1_above_floor(metrics):
    f1 = metrics["test"]["macro_f1"]
    assert f1 >= 0.75, f"Baseline macro-F1 {f1:.4f} fell below 0.75 floor"


def test_metrics_all_classes_have_f1(metrics):
    per_class = metrics["test"]["per_class_f1"]
    missing = CLASSES - set(per_class.keys())
    assert not missing, f"Missing per-class F1 for: {missing}"
