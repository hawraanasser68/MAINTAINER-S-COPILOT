"""
LLM zero-shot baseline using Groq API (free tier).
Runs on a 200-issue sample from the test split.

Usage:
    export GROQ_API_KEY=gsk_...
    python scripts/eval_llm_baseline.py [--sample-size N]

Get a free API key at: https://console.groq.com

Outputs:
    models/llm_baseline/metrics.json
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from groq import Groq
from sklearn.metrics import accuracy_score, classification_report, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.preprocess import CLASSES, load_split

OUTPUT_DIR = Path("models/llm_baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are a GitHub issue classifier. Given a GitHub issue title and body, classify it into exactly one of these four categories:

- bug: A defect, crash, unexpected behaviour, or regression
- feature: A request for new functionality or enhancement
- docs: A request for documentation improvement or fix
- question: A usage question, support request, or triage item

Respond with ONLY the category name — no explanation, no punctuation."""


def classify_issue(client: Groq, text: str) -> tuple[str, float]:
    # Truncate to ~400 chars to stay within 6k TPM free tier limit
    truncated = text[:400]
    for attempt in range(5):
        try:
            t0 = time.perf_counter()
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=10,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": truncated},
                ],
            )
            latency = (time.perf_counter() - t0) * 1000
            label = response.choices[0].message.content.strip().lower()
            if label not in CLASSES:
                label = "question"
            return label, latency
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e):
                wait = 10 * (attempt + 1)
                print(f"  Rate limit hit — waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    return "question", 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=200)
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise SystemExit("ERROR: set GROQ_API_KEY environment variable\nGet a free key at https://console.groq.com")

    client = Groq(api_key=api_key)

    print("Loading test split...")
    test_texts, test_labels = load_split("test")

    random.seed(42)
    indices = random.sample(range(len(test_texts)), min(args.sample_size, len(test_texts)))
    sample_texts  = [test_texts[i]  for i in indices]
    sample_labels = [test_labels[i] for i in indices]

    print(f"Running zero-shot classification on {len(sample_texts)} issues...")
    print(f"Model: {MODEL}\n")

    preds = []
    latencies = []

    for i, (text, true_label) in enumerate(zip(sample_texts, sample_labels)):
        pred, latency_ms = classify_issue(client, text)
        preds.append(pred)
        latencies.append(latency_ms)
        if (i + 1) % 20 == 0:
            acc_so_far = sum(p == l for p, l in zip(preds, sample_labels[:len(preds)])) / len(preds)
            status = "✓" if pred == true_label else "✗"
            print(f"  [{i+1}/{len(sample_texts)}] running_acc={acc_so_far:.3f}  {status} pred={pred} true={true_label}")

    accuracy  = accuracy_score(sample_labels, preds)
    macro_f1  = f1_score(sample_labels, preds, average="macro", labels=CLASSES)
    report    = classification_report(sample_labels, preds, labels=CLASSES, output_dict=True)
    per_class_f1 = {cls: round(report[cls]["f1-score"], 4) for cls in CLASSES}
    avg_latency  = sum(latencies) / len(latencies)

    print(f"\nResults on {len(sample_texts)}-issue sample:")
    print(f"  Accuracy   : {accuracy:.4f}")
    print(f"  Macro-F1   : {macro_f1:.4f}")
    print(f"  Avg latency: {avg_latency:.0f}ms/call")
    print("\nPer-class F1:")
    for cls, f in per_class_f1.items():
        print(f"  {cls:<10} {f:.4f}")
    print(f"\n{classification_report(sample_labels, preds, labels=CLASSES)}")

    metrics = {
        "model": MODEL,
        "provider": "groq",
        "mode": "zero-shot",
        "sample_size": len(sample_texts),
        "test": {
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class_f1": per_class_f1,
            "latency_ms_per_sample": round(avg_latency, 1),
        },
        "cost_note": "Free tier (Groq) — $0",
    }

    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved metrics → {metrics_path}")


if __name__ == "__main__":
    main()
