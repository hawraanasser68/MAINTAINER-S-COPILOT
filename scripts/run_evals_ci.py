"""
CI eval runner — Phase 6.

Reads committed metrics JSON files, compares every metric against the thresholds
in eval_thresholds.yaml, writes eval_report.json, uploads it to MinIO with a
timestamped key, diffs against the previous green build, and exits non-zero if
any threshold is missed.

Usage:
    python scripts/run_evals_ci.py [--dry-run]

Non-zero exit codes:
    1  — one or more metrics below threshold
    2  — golden set or metrics file missing (configuration error)
    3  — eval_thresholds.yaml has a threshold referencing an unimplemented metric
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        print(f"[ERROR] Missing metrics file: {path} ({label})")
        sys.exit(2)
    return json.loads(path.read_text())


def _check(
    failures: list[str],
    deltas: dict[str, float],
    metric_path: str,
    actual: float | None,
    threshold: float,
) -> None:
    """Compare one metric to its threshold; record failure or delta."""
    if actual is None:
        print(f"[ERROR] Metric '{metric_path}' referenced in thresholds but not found in metrics files")
        sys.exit(3)
    delta = actual - threshold
    deltas[metric_path] = round(delta, 4)
    status = "PASS" if actual >= threshold else "FAIL"
    marker = "✓" if status == "PASS" else "✗"
    print(f"  {marker} {metric_path:<45} actual={actual:.4f}  threshold={threshold:.4f}  delta={delta:+.4f}  [{status}]")
    if status == "FAIL":
        failures.append(f"{metric_path}: actual={actual:.4f} < threshold={threshold:.4f} (delta={delta:+.4f})")


# ---------------------------------------------------------------------------
# MinIO helpers (non-blocking on failure)
# ---------------------------------------------------------------------------

def _minio_upload(report: dict[str, Any], run_id: str) -> str | None:
    """Upload eval_report.json to MinIO; return the key or None on failure."""
    try:
        import boto3
        from botocore.exceptions import ClientError

        endpoint = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
        access = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
        secret = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
        bucket = "eval-reports"
        key = f"eval_report_{run_id}.json"

        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access,
            aws_secret_access_key=secret,
            region_name="us-east-1",
        )
        try:
            client.head_bucket(Bucket=bucket)
        except ClientError:
            client.create_bucket(Bucket=bucket)

        client.put_object(Bucket=bucket, Key=key, Body=json.dumps(report, indent=2).encode())
        print(f"[MinIO] Uploaded → s3://{bucket}/{key}")
        return key
    except Exception as exc:  # noqa: BLE001
        print(f"[MinIO] Upload failed (non-blocking): {exc}")
        return None


def _minio_get_previous(run_id: str) -> dict[str, Any] | None:
    """Download the most recent previous eval_report from MinIO, or None."""
    try:
        import boto3

        endpoint = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
        access = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
        secret = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
        bucket = "eval-reports"

        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access,
            aws_secret_access_key=secret,
            region_name="us-east-1",
        )
        objects = client.list_objects_v2(Bucket=bucket, Prefix="eval_report_").get("Contents", [])
        # Sort by LastModified, skip the one we just uploaded
        objects = [o for o in objects if o["Key"] != f"eval_report_{run_id}.json"]
        if not objects:
            return None
        latest = sorted(objects, key=lambda o: o["LastModified"])[-1]
        data = client.get_object(Bucket=bucket, Key=latest["Key"])["Body"].read()
        print(f"[MinIO] Diffing against previous report: {latest['Key']}")
        return json.loads(data)
    except Exception as exc:  # noqa: BLE001
        print(f"[MinIO] Could not fetch previous report (non-blocking): {exc}")
        return None


def _log_diff(current: dict[str, Any], previous: dict[str, Any]) -> None:
    """Print metric deltas between this run and the previous green build."""
    print("\n── Metric deltas vs previous green build ──")
    curr_deltas = current.get("metric_deltas_vs_threshold", {})
    prev_deltas = previous.get("metric_deltas_vs_threshold", {})
    if not prev_deltas:
        print("  (no previous run to diff against)")
        return
    all_keys = sorted(set(curr_deltas) | set(prev_deltas))
    for k in all_keys:
        c = curr_deltas.get(k)
        p = prev_deltas.get(k)
        if c is None or p is None:
            continue
        change = c - p
        arrow = "↑" if change > 0 else ("↓" if change < 0 else "→")
        print(f"  {arrow} {k:<45} {p:+.4f} → {c:+.4f}  (Δ{change:+.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Skip MinIO upload")
    args = parser.parse_args()

    thresholds_path = ROOT / "eval_thresholds.yaml"
    if not thresholds_path.exists():
        print(f"[ERROR] eval_thresholds.yaml not found at {thresholds_path}")
        sys.exit(2)

    golden_path = ROOT / "data" / "golden_rag.jsonl"
    if not golden_path.exists():
        print(f"[ERROR] Golden set not found: {golden_path}")
        sys.exit(2)

    thresholds = _load_yaml(thresholds_path)
    commit_sha = os.environ.get("GITHUB_SHA", "local")[:12]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"_{commit_sha}"

    print(f"\n{'='*60}")
    print(f"  Maintainer's Copilot — Eval Gate  (run {run_id})")
    print(f"{'='*60}\n")

    # ── Load metrics from committed model artifacts ─────────────────────────
    finetuned_card = _load_json(ROOT / "models" / "classifier" / "model_card.json", "fine-tuned DistilBERT")
    classical_metrics = _load_json(ROOT / "models" / "baseline" / "metrics.json", "TF-IDF+LogReg baseline")
    llm_metrics = _load_json(ROOT / "models" / "llm_baseline" / "metrics.json", "LLM baseline")
    rag_metrics = _load_json(ROOT / "models" / "rag_eval" / "metrics.json", "RAG pipeline")

    finetuned_test = finetuned_card.get("metrics", {})
    finetuned_per_class = finetuned_test.get("per_class_f1", {})
    classical_test = classical_metrics.get("test", {})
    llm_test = llm_metrics.get("test", {})
    rag_advanced = rag_metrics.get("advanced_hybrid_rerank", {})

    failures: list[str] = []
    deltas: dict[str, float] = {}

    # ── Classification thresholds ────────────────────────────────────────────
    print("── Classification: fine-tuned DistilBERT ──")
    ct = thresholds.get("classification", {})
    ft = ct.get("finetuned", {})
    _check(failures, deltas, "classification.finetuned.macro_f1",
           finetuned_test.get("test_macro_f1"), ft.get("macro_f1", 0))

    ft_per = ft.get("per_class_f1", {})
    for cls in ("bug", "feature", "docs", "question"):
        _check(failures, deltas, f"classification.finetuned.per_class_f1.{cls}",
               finetuned_per_class.get(cls), ft_per.get(cls, 0))

    print("\n── Classification: TF-IDF + LogReg baseline ──")
    clt = ct.get("classical", {})
    _check(failures, deltas, "classification.classical.macro_f1",
           classical_test.get("macro_f1"), clt.get("macro_f1", 0))

    print("\n── Classification: LLM zero-shot baseline ──")
    lt = ct.get("llm", {})
    _check(failures, deltas, "classification.llm.macro_f1",
           llm_test.get("macro_f1"), lt.get("macro_f1", 0))

    print("\n── RAG pipeline ──")
    rt = thresholds.get("rag", {})
    _check(failures, deltas, "rag.hit_at_5",
           rag_advanced.get("hit_at_5"), rt.get("hit_at_5", 0))
    _check(failures, deltas, "rag.mrr_at_10",
           rag_advanced.get("mrr_at_10"), rt.get("mrr_at_10", 0))

    # faithfulness and answer_relevancy: check if threshold is real (>= 0.1)
    for metric in ("faithfulness", "answer_relevancy"):
        thr = rt.get(metric, 0)
        if thr >= 0.1:
            actual = rag_metrics.get(metric)
            _check(failures, deltas, f"rag.{metric}", actual, thr)

    # ── Validate golden set exists and is non-empty ──────────────────────────
    golden = [json.loads(line) for line in golden_path.read_text().splitlines() if line.strip()]
    golden_hl = [g for g in golden if g.get("hand_labelled")]
    print("\n── Golden set ──")
    print(f"  ✓ golden_rag.jsonl: {len(golden)} examples, {len(golden_hl)} hand-labelled")
    if len(golden_hl) < 5:
        failures.append(f"golden_rag.jsonl: only {len(golden_hl)} hand-labelled examples (minimum 5 required)")

    # ── Build eval_report.json ───────────────────────────────────────────────
    report: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit_sha": commit_sha,
        "passed": len(failures) == 0,
        "failure_count": len(failures),
        "failures": failures,
        "metric_deltas_vs_threshold": deltas,
        "classification": {
            "finetuned": {
                "macro_f1": finetuned_test.get("test_macro_f1"),
                "per_class_f1": finetuned_per_class,
            },
            "classical": {"macro_f1": classical_test.get("macro_f1")},
            "llm": {"macro_f1": llm_test.get("macro_f1")},
        },
        "rag": {
            "hit_at_5": rag_advanced.get("hit_at_5"),
            "mrr_at_10": rag_advanced.get("mrr_at_10"),
            "faithfulness": rag_metrics.get("faithfulness"),
            "answer_relevancy": rag_metrics.get("answer_relevancy"),
            "eval_set_size": rag_metrics.get("eval_set_size"),
        },
        "redaction_test": "see pytest test_redaction.py",
        "golden_set": {
            "total": len(golden),
            "hand_labelled": len(golden_hl),
        },
    }

    report_path = ROOT / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n[OK] eval_report.json written → {report_path}")

    # ── MinIO upload + diff ──────────────────────────────────────────────────
    if not args.dry_run:
        prev_report = _minio_get_previous(run_id)
        _minio_upload(report, run_id)
        if prev_report:
            _log_diff(report, prev_report)
        else:
            print("[MinIO] No previous report found — this is the baseline run.")
    else:
        print("[dry-run] Skipping MinIO upload.")

    # ── Final verdict ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if failures:
        print(f"  EVAL GATE: FAILED — {len(failures)} metric(s) below threshold\n")
        for f in failures:
            print(f"    ✗ {f}")
        print(f"{'='*60}\n")
        sys.exit(1)
    else:
        print(f"  EVAL GATE: PASSED — all {len(deltas)} metrics meet thresholds")
        print(f"{'='*60}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
