"""
Integration tests for the model server endpoints.
These tests require the fine-tuned model weights to be present.
Run after scripts/train_classifier.py has completed.
"""

import pytest
from fastapi.testclient import TestClient

MODEL_DIR_EXISTS = (
    __import__("pathlib").Path(__file__).parent.parent / "models" / "classifier" / "model_card.json"
).exists()

pytestmark = pytest.mark.skipif(
    not MODEL_DIR_EXISTS,
    reason="Fine-tuned model not found — run scripts/train_classifier.py first",
)


@pytest.fixture(scope="module")
def client():
    from modelserver.main import app
    with TestClient(app) as c:
        yield c


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "test_macro_f1" in body


def test_classify_returns_valid_label(client):
    resp = client.post("/classify", json={
        "title": "IndexError when calling fit() with NaN values",
        "body": "Traceback shows line 42 raises IndexError after passing NaN in X.",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in {"bug", "feature", "docs", "question"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["latency_ms"] > 0
    assert set(body["all_scores"].keys()) == {"bug", "feature", "docs", "question"}


def test_classify_scores_sum_to_one(client):
    resp = client.post("/classify", json={"title": "Add support for sparse matrices", "body": ""})
    assert resp.status_code == 200
    scores = resp.json()["all_scores"]
    total = sum(scores.values())
    assert abs(total - 1.0) < 0.01, f"Scores sum to {total}, expected ~1.0"


def test_classify_empty_body_uses_title(client):
    resp = client.post("/classify", json={"title": "Documentation typo in README", "body": ""})
    assert resp.status_code == 200
    assert resp.json()["label"] in {"bug", "feature", "docs", "question"}


def test_ner_returns_entities(client):
    resp = client.post("/ner", json={
        "text": "ValueError raised in sklearn.preprocessing.StandardScaler when using numpy 1.24.0",
    })
    assert resp.status_code == 200
    entities = resp.json()["entities"]
    assert isinstance(entities, list)
    for e in entities:
        assert "text" in e and "type" in e
        assert e["type"] in {"function", "error_code", "package", "version"}


def test_summarize_returns_short_text(client):
    resp = client.post("/summarize", json={
        "title": "Memory leak in GridSearchCV with large datasets",
        "body": "When running GridSearchCV on a dataset with 100k samples, memory usage grows unboundedly across folds.",
    })
    assert resp.status_code == 200
    summary = resp.json()["summary"]
    assert isinstance(summary, str)
    assert len(summary.split()) <= 120, f"Summary too long: {len(summary.split())} words"
