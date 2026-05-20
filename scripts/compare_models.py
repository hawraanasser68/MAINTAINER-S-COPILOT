"""
Generate three-way comparison table from saved metrics files.
Prints Markdown table for DECISIONS.md.

Usage:
    python scripts/compare_models.py
"""

import json
from pathlib import Path

BASELINE_METRICS = Path("models/baseline/metrics.json")
CLASSIFIER_CARD  = Path("models/classifier/model_card.json")
LLM_METRICS      = Path("models/llm_baseline/metrics.json")

CLASSES = ["bug", "feature", "docs", "question"]


def load_json(path: Path) -> dict | None:
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping")
        return None
    return json.loads(path.read_text())


def main() -> None:
    baseline   = load_json(BASELINE_METRICS)
    classifier = load_json(CLASSIFIER_CARD)
    llm        = load_json(LLM_METRICS)

    rows = []

    if baseline:
        t = baseline["test"]
        rows.append({
            "model": "TF-IDF + LogReg",
            "accuracy": t["accuracy"],
            "macro_f1": t["macro_f1"],
            "per_class": t["per_class_f1"],
            "latency_ms": t["latency_ms_per_sample"],
            "cost": "~$0 (local)",
            "note": "Fast, interpretable, no GPU needed",
        })

    if classifier:
        m = classifier["metrics"]
        rows.append({
            "model": "DistilBERT (fine-tuned)",
            "accuracy": m["test_accuracy"],
            "macro_f1": m["test_macro_f1"],
            "per_class": m["per_class_f1"],
            "latency_ms": m["latency_ms_per_sample"],
            "cost": "~$0 (local inference)",
            "note": "Best accuracy, ~66M params, GPU optional",
        })

    if llm:
        t = llm["test"]
        rows.append({
            "model": f"Claude Haiku (zero-shot)",
            "accuracy": t["accuracy"],
            "macro_f1": t["macro_f1"],
            "per_class": t["per_class_f1"],
            "latency_ms": t["latency_ms_per_sample"],
            "cost": llm.get("cost_note", "see model pricing"),
            "note": f"No training, sample={llm.get('sample_size', '?')}",
        })

    if not rows:
        print("No metrics found. Run training scripts first.")
        return

    # Sort by macro-F1 descending
    rows.sort(key=lambda r: r["macro_f1"], reverse=True)

    print("\n## Three-Way Model Comparison\n")
    print(f"| Model | Accuracy | Macro-F1 | bug F1 | feature F1 | docs F1 | question F1 | Latency | Cost |")
    print(f"|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        pc = r["per_class"]
        print(
            f"| {r['model']} "
            f"| {r['accuracy']:.4f} "
            f"| **{r['macro_f1']:.4f}** "
            f"| {pc.get('bug', 'N/A')} "
            f"| {pc.get('feature', 'N/A')} "
            f"| {pc.get('docs', 'N/A')} "
            f"| {pc.get('question', 'N/A')} "
            f"| {r['latency_ms']:.1f}ms "
            f"| {r['cost']} |"
        )

    best = rows[0]
    print(f"\n**Deployment recommendation**: {best['model']} — highest macro-F1 ({best['macro_f1']:.4f}).")
    print(f"Note: {best['note']}")


if __name__ == "__main__":
    main()
