"""
BM25 index — built from all issue texts, persisted as pickle in MinIO.
Loaded once at startup via lru_cache.
"""

from __future__ import annotations

import io
import pickle
from functools import lru_cache
from typing import Any

from rank_bm25 import BM25Okapi

MINIO_BUCKET = "models"
MINIO_KEY = "bm25/index.pkl"


def tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Index:
    def __init__(self, issue_numbers: list[int], texts: list[str]) -> None:
        self.issue_numbers = issue_numbers
        tokenized = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Return [(issue_number, score), ...] sorted by score descending."""
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.issue_numbers[i], float(scores[i])) for i in top_indices]

    def to_bytes(self) -> bytes:
        return pickle.dumps(self)

    @classmethod
    def from_bytes(cls, data: bytes) -> "BM25Index":
        return pickle.loads(data)


def build_index(records: list[dict[str, Any]]) -> BM25Index:
    """Build a BM25Index from a list of issue dicts with number/title/body."""
    numbers = [r["number"] for r in records]
    texts = [f"{r.get('title', '')} {r.get('body', '')}".strip() for r in records]
    return BM25Index(numbers, texts)


_cached_index: BM25Index | None = None


def get_index() -> BM25Index:
    global _cached_index
    if _cached_index is None:
        raise RuntimeError("BM25 index not loaded. Call load_index() at startup.")
    return _cached_index


def load_index(index: BM25Index) -> None:
    global _cached_index
    _cached_index = index
