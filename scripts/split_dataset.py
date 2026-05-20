"""
Split raw_issues.jsonl into stratified train/val/test splits.
Test set is strictly more recent in time than train.

Usage:
    python scripts/split_dataset.py
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

INPUT = Path("data/raw_issues.jsonl")
OUTPUT_DIR = Path("data/splits")

TEST_RATIO = 0.20
VAL_RATIO = 0.10


def load(path: Path) -> pd.DataFrame:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(records)
    df["closed_at"] = pd.to_datetime(df["closed_at"], utc=True)
    df = df.dropna(subset=["closed_at", "label"])
    df = df.sort_values("closed_at").reset_index(drop=True)
    return df


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Test = most recent 20%, val = next 10%, train = remaining 70%.
    Temporal boundary is strictly enforced: no test issue is older than any train issue."""
    n = len(df)
    test_start = int(n * (1 - TEST_RATIO))
    val_start = int(n * (1 - TEST_RATIO - VAL_RATIO))

    train = df.iloc[:val_start].copy()
    val = df.iloc[val_start:test_start].copy()
    test = df.iloc[test_start:].copy()

    # Verify temporal boundary
    assert train["closed_at"].max() <= val["closed_at"].min(), "Train/val temporal boundary violated"
    assert val["closed_at"].max() <= test["closed_at"].min(), "Val/test temporal boundary violated"

    return train, val, test


def check_class_coverage(df: pd.DataFrame, split_name: str) -> None:
    counts = Counter(df["label"])
    missing = [c for c in ["bug", "feature", "docs", "question"] if c not in counts]
    print(f"  {split_name}: {dict(counts)}")
    if missing:
        print(f"  WARNING: {split_name} is missing classes: {missing}")


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), default=str) + "\n")


def main() -> None:
    print(f"Loading {INPUT}...")
    df = load(INPUT)
    print(f"  Total mapped issues: {len(df)}")
    print(f"  Label distribution : {dict(Counter(df['label']))}")
    print(f"  Date range         : {df['closed_at'].min().date()} → {df['closed_at'].max().date()}")

    print("\nSplitting...")
    train, val, test = temporal_split(df)

    print("\nClass distribution per split:")
    check_class_coverage(train, "train")
    check_class_coverage(val, "val")
    check_class_coverage(test, "test")

    print(f"\nSizes: train={len(train)}, val={len(val)}, test={len(test)}")

    # Save splits
    save(train, OUTPUT_DIR / "train.jsonl")
    save(val, OUTPUT_DIR / "val.jsonl")
    save(test, OUTPUT_DIR / "test.jsonl")

    # Save stats
    stats = {
        "total": len(df),
        "train": {"n": len(train), "labels": dict(Counter(train["label"]))},
        "val": {"n": len(val), "labels": dict(Counter(val["label"]))},
        "test": {"n": len(test), "labels": dict(Counter(test["label"]))},
        "date_range": {
            "min": str(df["closed_at"].min().date()),
            "max": str(df["closed_at"].max().date()),
        },
        "temporal_boundary": {
            "train_max": str(train["closed_at"].max().date()),
            "val_min": str(val["closed_at"].min().date()),
            "val_max": str(val["closed_at"].max().date()),
            "test_min": str(test["closed_at"].min().date()),
        },
    }
    (OUTPUT_DIR / "split_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\nSaved to {OUTPUT_DIR}/")
    print("  train.jsonl, val.jsonl, test.jsonl, split_stats.json")


if __name__ == "__main__":
    main()
