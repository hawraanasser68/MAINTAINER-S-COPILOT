"""
Shared preprocessing utilities for all Phase 2 models.

Usage:
    from scripts.preprocess import make_text, load_split, CLASSES
"""

import json
import re
from pathlib import Path

CLASSES = ["bug", "feature", "docs", "question"]
SPLITS_DIR = Path(__file__).parent.parent / "data" / "splits"


def make_text(title: str, body: str) -> str:
    """Combine title and body into a single input string."""
    title = (title or "").strip()
    body = (body or "").strip()
    if body:
        return f"{title} {body}"
    return title


def load_split(name: str) -> tuple[list[str], list[str]]:
    """Return (texts, labels) for a given split name."""
    path = SPLITS_DIR / f"{name}.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    texts = [make_text(r["title"], r["body"]) for r in records]
    labels = [r["label"] for r in records]
    return texts, labels
