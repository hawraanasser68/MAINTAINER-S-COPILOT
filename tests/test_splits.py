"""
Data integrity tests for train/val/test splits.
These run without network access — only reads local JSONL files.
"""

import json
from collections import Counter
from pathlib import Path

import pytest

SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"
CLASSES = {"bug", "feature", "docs", "question"}
SPLITS = ["train", "val", "test"]


def load_split(name: str) -> list[dict]:
    path = SPLITS_DIR / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def all_splits():
    return {name: load_split(name) for name in SPLITS}


# ── Size checks ───────────────────────────────────────────────────────────────

def test_split_sizes(all_splits):
    train, val, test = all_splits["train"], all_splits["val"], all_splits["test"]
    total = len(train) + len(val) + len(test)
    assert total == 5000, f"Expected 5000 issues, got {total}"
    assert len(train) == 3500, f"Train: expected 3500, got {len(train)}"
    assert len(val) == 500, f"Val: expected 500, got {len(val)}"
    assert len(test) == 1000, f"Test: expected 1000, got {len(test)}"


# ── Class coverage ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("split_name", SPLITS)
def test_all_classes_present(all_splits, split_name):
    labels = {r["label"] for r in all_splits[split_name]}
    missing = CLASSES - labels
    assert not missing, f"{split_name} is missing classes: {missing}"


@pytest.mark.parametrize("split_name", SPLITS)
def test_no_unknown_labels(all_splits, split_name):
    labels = {r["label"] for r in all_splits[split_name]}
    unknown = labels - CLASSES
    assert not unknown, f"{split_name} contains unknown labels: {unknown}"


# ── Imbalance ─────────────────────────────────────────────────────────────────

def test_train_imbalance_below_threshold(all_splits):
    counts = Counter(r["label"] for r in all_splits["train"])
    ratio = max(counts.values()) / min(counts.values())
    assert ratio < 10, f"Train imbalance ratio {ratio:.1f}x exceeds 10x — oversampling required"


# ── Temporal boundary ─────────────────────────────────────────────────────────

def test_temporal_boundaries(all_splits):
    def max_date(records):
        return max(r["closed_at"] for r in records if r.get("closed_at"))

    def min_date(records):
        return min(r["closed_at"] for r in records if r.get("closed_at"))

    train_max = max_date(all_splits["train"])
    val_min = min_date(all_splits["val"])
    val_max = max_date(all_splits["val"])
    test_min = min_date(all_splits["test"])

    assert train_max <= val_min, (
        f"Train/val boundary violated: train max={train_max}, val min={val_min}"
    )
    assert val_max <= test_min, (
        f"Val/test boundary violated: val max={val_max}, test min={test_min}"
    )


# ── Required fields ───────────────────────────────────────────────────────────

REQUIRED_FIELDS = {"number", "title", "body", "label", "closed_at", "raw_labels"}


@pytest.mark.parametrize("split_name", SPLITS)
def test_required_fields_present(all_splits, split_name):
    for i, record in enumerate(all_splits[split_name]):
        missing = REQUIRED_FIELDS - record.keys()
        assert not missing, f"{split_name}[{i}] missing fields: {missing}"


def test_title_never_null(all_splits):
    for split_name in SPLITS:
        for i, record in enumerate(all_splits[split_name]):
            assert record.get("title"), f"{split_name}[{i}] has null/empty title"


# ── No duplicates ─────────────────────────────────────────────────────────────

def test_no_duplicate_issues_across_splits(all_splits):
    all_numbers = [r["number"] for records in all_splits.values() for r in records]
    counts = Counter(all_numbers)
    dupes = {n: c for n, c in counts.items() if c > 1}
    assert not dupes, f"Issue numbers appear in multiple splits: {dupes}"
