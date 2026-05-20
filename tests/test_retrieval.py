"""Unit tests for retrieval utilities — no DB or Docker needed."""

import pytest
from app.infra.bm25_index import BM25Index, build_index
from app.services.retrieval import reciprocal_rank_fusion


# ── BM25 index ────────────────────────────────────────────────────────────────

SAMPLE_RECORDS = [
    {"number": 1, "title": "IndexError in fit with NaN", "body": "Raises IndexError when X has NaN values"},
    {"number": 2, "title": "Add support for sparse matrices", "body": "Feature request for sparse input support"},
    {"number": 3, "title": "Docs typo in LogisticRegression", "body": "Missing bracket in formula on line 3"},
    {"number": 4, "title": "How to use GridSearchCV", "body": "Usage question about hyperparameter tuning"},
]


@pytest.fixture
def bm25_index():
    return build_index(SAMPLE_RECORDS)


def test_bm25_returns_results(bm25_index):
    results = bm25_index.search("IndexError NaN", top_k=2)
    assert len(results) == 2
    numbers = [n for n, _ in results]
    assert 1 in numbers


def test_bm25_top_result_is_most_relevant(bm25_index):
    results = bm25_index.search("sparse matrices feature request", top_k=4)
    top_number = results[0][0]
    assert top_number == 2


def test_bm25_serialization_roundtrip(bm25_index):
    data = bm25_index.to_bytes()
    loaded = BM25Index.from_bytes(data)
    results = loaded.search("GridSearchCV", top_k=1)
    assert results[0][0] == 4


def test_bm25_returns_scores(bm25_index):
    results = bm25_index.search("NaN", top_k=2)
    for number, score in results:
        assert isinstance(score, float)
        assert score >= 0.0


# ── RRF fusion ────────────────────────────────────────────────────────────────

def test_rrf_combines_both_lists():
    dense = [{"number": 1, "score": 0.9}, {"number": 2, "score": 0.8}, {"number": 3, "score": 0.7}]
    bm25 = [(3, 5.0), (1, 4.0), (4, 3.0)]
    fused = reciprocal_rank_fusion(dense, bm25)
    fused_numbers = [r["number"] for r in fused]
    # 1 appears in both lists so should rank highly
    assert fused_numbers[0] == 1 or fused_numbers[1] == 1


def test_rrf_includes_bm25_only_results():
    dense = [{"number": 1, "score": 0.9}]
    bm25 = [(2, 5.0), (1, 3.0)]
    fused = reciprocal_rank_fusion(dense, bm25)
    numbers = [r["number"] for r in fused]
    assert 2 in numbers


def test_rrf_scores_are_positive():
    dense = [{"number": i, "score": 1.0 / (i + 1)} for i in range(5)]
    bm25 = [(i, float(5 - i)) for i in range(5)]
    fused = reciprocal_rank_fusion(dense, bm25)
    for r in fused:
        assert r["rrf_score"] > 0.0


def test_rrf_descending_order():
    dense = [{"number": i, "score": 1.0 / (i + 1)} for i in range(10)]
    bm25 = [(i, float(10 - i)) for i in range(10)]
    fused = reciprocal_rank_fusion(dense, bm25)
    scores = [r["rrf_score"] for r in fused]
    assert scores == sorted(scores, reverse=True)
