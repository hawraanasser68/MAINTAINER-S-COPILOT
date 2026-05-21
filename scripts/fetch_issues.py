"""
Fetch closed issues from scikit-learn/scikit-learn via GitHub API.
Saves to data/raw_issues.jsonl — one JSON object per line.

Usage:
    export GITHUB_TOKEN=ghp_...
    python scripts/fetch_issues.py
"""

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import github

REPO = "scikit-learn/scikit-learn"
OUTPUT = Path("data/raw_issues.jsonl")
MAX_ISSUES = 5000

LABEL_MAP: dict[str, str] = {
    "bug": "bug", "regression": "bug", "crash": "bug", "type: bug": "bug",
    "enhancement": "feature", "feature": "feature",
    "new feature": "feature", "type: enhancement": "feature",
    "documentation": "docs", "docs": "docs", "doc": "docs", "type: docs": "docs",
    "question": "question", "help wanted": "question",
    "usage": "question", "support": "question", "needs triage": "question",
}


def map_labels(label_names: list[str]) -> str | None:
    for name in label_names:
        mapped = LABEL_MAP.get(name.lower().strip())
        if mapped:
            return mapped
    return None


def fetch() -> None:
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise SystemExit("ERROR: set GITHUB_TOKEN environment variable")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    gh = github.Github(auth=github.Auth.Token(token), per_page=50, retry=5)
    repo = gh.get_repo(REPO)
    issues = repo.get_issues(state="closed", sort="updated", direction="desc")

    fetched = 0
    skipped_pr = 0
    skipped_no_label = 0

    with OUTPUT.open("w") as f:
        for issue in issues:
            if issue.pull_request:
                skipped_pr += 1
                continue

            label_names = [lbl.name for lbl in issue.labels]
            mapped = map_labels(label_names)

            if mapped is None:
                skipped_no_label += 1
                continue

            record = {
                "number": issue.number,
                "title": issue.title,
                "body": issue.body or "",
                "raw_labels": label_names,
                "label": mapped,
                "repo": REPO,
                "created_at": issue.created_at.isoformat(),
                "closed_at": issue.closed_at.isoformat() if issue.closed_at else None,
            }
            f.write(json.dumps(record) + "\n")
            fetched += 1

            if fetched % 200 == 0:
                print(f"  fetched {fetched} issues...")
                try:
                    remaining = gh.get_rate_limit().core.remaining
                    if remaining < 100:
                        reset = gh.get_rate_limit().core.reset
                        wait = max(
                            (reset.replace(tzinfo=UTC) - datetime.now(UTC)).seconds + 5, 0
                        )
                        print(f"  rate limit low ({remaining}) — sleeping {wait}s")
                        time.sleep(wait)
                except Exception:
                    pass

            if fetched >= MAX_ISSUES:
                break

    print("\nDone.")
    print(f"  Fetched  : {fetched} issues → {OUTPUT}")
    print(f"  Skipped  : {skipped_pr} PRs, {skipped_no_label} unlabelled")


if __name__ == "__main__":
    fetch()
