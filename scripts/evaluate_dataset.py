"""
Dataset evaluation script — run before committing to a repo.
Checks: label coverage, class balance, body quality, temporal spread.

Usage:
    export GITHUB_TOKEN=ghp_...
    python scripts/evaluate_dataset.py
"""

import os
import time
from collections import Counter
from datetime import datetime, timezone

from github import Github, RateLimitExceededException

# ── Label mapping ─────────────────────────────────────────────────────────────
LABEL_MAP: dict[str, str] = {
    # bug
    "bug": "bug", "regression": "bug", "broken": "bug", "fix": "bug",
    "type: bug": "bug", "kind/bug": "bug", "bug report": "bug",
    # feature
    "enhancement": "feature", "feature": "feature", "feature request": "feature",
    "new feature": "feature", "type: enhancement": "feature", "kind/feature": "feature",
    "new model": "feature", "feat": "feature",
    # docs
    "documentation": "docs", "docs": "docs", "doc": "docs",
    "type: docs": "docs", "kind/documentation": "docs",
    # question
    "question": "question", "help wanted": "question", "usage": "question",
    "support": "question", "type: question": "question", "kind/question": "question",
}

CANDIDATES = [
    "pandas-dev/pandas",
    "scikit-learn/scikit-learn",
]

MAX_ISSUES = 2000  # cap per repo to stay within rate limits


def map_labels(labels: list[str]) -> str | None:
    for lbl in labels:
        mapped = LABEL_MAP.get(lbl.lower().strip())
        if mapped:
            return mapped
    return None


def evaluate_repo(gh: Github, repo_name: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Evaluating: {repo_name}")
    print(f"{'='*60}")

    repo = gh.get_repo(repo_name)
    issues = repo.get_issues(state="closed")

    total = 0
    mapped = 0
    class_counts: Counter = Counter()
    empty_body = 0
    short_body = 0  # body < 50 chars
    years: Counter = Counter()

    for issue in issues:
        if issue.pull_request:
            continue  # skip PRs
        total += 1

        label_names = [l.name for l in issue.labels]
        cls = map_labels(label_names)

        if cls:
            mapped += 1
            class_counts[cls] += 1

        if not issue.body or issue.body.strip() == "":
            empty_body += 1
        elif len(issue.body.strip()) < 50:
            short_body += 1

        if issue.closed_at:
            years[issue.closed_at.year] += 1

        if total >= MAX_ISSUES:
            break

        if total % 200 == 0:
            print(f"  ... fetched {total} issues")
            try:
                remaining = gh.get_rate_limit().core.remaining
                if remaining < 50:
                    reset = gh.get_rate_limit().core.reset
                    wait = (reset.replace(tzinfo=timezone.utc) - datetime.now(timezone.utc)).seconds + 5
                    print(f"  Rate limit low ({remaining} remaining) — waiting {wait}s")
                    time.sleep(wait)
            except Exception:
                pass

    coverage = mapped / total * 100 if total > 0 else 0
    empty_pct = empty_body / total * 100 if total > 0 else 0

    # Imbalance ratio: max class / min class (lower is better, 1.0 = perfect balance)
    if class_counts and len(class_counts) == 4:
        imbalance = max(class_counts.values()) / max(min(class_counts.values()), 1)
    else:
        imbalance = float("inf")

    result = {
        "repo": repo_name,
        "total_fetched": total,
        "mapped": mapped,
        "coverage_pct": round(coverage, 1),
        "class_counts": dict(class_counts),
        "missing_classes": [c for c in ["bug", "feature", "docs", "question"] if c not in class_counts],
        "imbalance_ratio": round(imbalance, 1),
        "empty_body_pct": round(empty_pct, 1),
        "years": dict(sorted(years.items())),
    }

    print(f"  Total issues fetched : {total}")
    print(f"  Mapped to 4 classes  : {mapped} ({coverage:.1f}%)")
    print(f"  Class distribution   : {dict(class_counts)}")
    print(f"  Missing classes      : {result['missing_classes'] or 'none'}")
    print(f"  Imbalance ratio      : {imbalance:.1f}x  (1.0 = perfect, >10 = problematic)")
    print(f"  Empty body           : {empty_pct:.1f}%")
    print(f"  Issues by year       : {dict(sorted(years.items()))}")

    return result


def score(r: dict) -> float:
    """Higher is better."""
    s = 0.0
    s += min(r["mapped"], 1500) / 15          # up to 100 pts for volume
    s += r["coverage_pct"]                    # up to 100 pts for label coverage
    s -= r["imbalance_ratio"] * 3             # penalise imbalance
    s -= r["empty_body_pct"]                  # penalise missing text
    s -= len(r["missing_classes"]) * 20       # penalise missing classes
    return round(s, 1)


def main() -> None:
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("ERROR: set GITHUB_TOKEN environment variable first")
        return

    import github
    gh = Github(auth=github.Auth.Token(token), per_page=50, retry=5)
    results = []

    for repo_name in CANDIDATES:
        for attempt in range(3):
            try:
                r = evaluate_repo(gh, repo_name)
                r["score"] = score(r)
                results.append(r)
                break
            except Exception as exc:
                print(f"  attempt {attempt+1} failed: {exc}")
                if attempt < 2:
                    print("  retrying in 15s...")
                    time.sleep(15)

    print(f"\n\n{'='*60}")
    print("SUMMARY (sorted by score)")
    print(f"{'='*60}")
    print(f"{'Repo':<35} {'Mapped':>7} {'Cov%':>6} {'Imbal':>7} {'Empty%':>7} {'Score':>7}")
    print("-" * 75)
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        print(
            f"{r['repo']:<35} {r['mapped']:>7} {r['coverage_pct']:>6} "
            f"{r['imbalance_ratio']:>7} {r['empty_body_pct']:>7} {r['score']:>7}"
        )

    best = max(results, key=lambda x: x["score"]) if results else None
    if best:
        print(f"\n→ Recommended: {best['repo']} (score {best['score']})")
        print(f"  Classes: {best['class_counts']}")


if __name__ == "__main__":
    main()
