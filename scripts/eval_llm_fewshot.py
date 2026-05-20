"""
LLM few-shot baseline using Groq API (free tier).
Uses the same 200-issue sample as the zero-shot eval for a fair comparison.

Usage:
    export GROQ_API_KEY=gsk_...
    python scripts/eval_llm_fewshot.py [--sample-size N]

Outputs:
    models/llm_fewshot/metrics.json
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

OUTPUT_DIR_BASE = Path("models/llm_fewshot")

MODEL_DEFAULT = "llama-3.1-8b-instant"

# 2 real examples per class, drawn from train split (seed=0, different from test sample seed=42)
FEW_SHOT_EXAMPLES = [
    {
        "label": "bug",
        "title": "index type np.int32_t causes issue in _csr_polynomial_expansion",
        "body": "I ran into an issue when trying to construct a polynomial expansion feature with a large sparse matrix input.",
    },
    {
        "label": "bug",
        "title": "LatentDirichletAllocation's number of iterations",
        "body": "When I print lda.n_iter_ after fitting, it reports 81. That's more than max_iter=10.",
    },
    {
        "label": "feature",
        "title": "Refactor class_weight in RidgeClassifier",
        "body": "It should reuse the code in utils that is used in SVM and SGD.",
    },
    {
        "label": "feature",
        "title": "Possibility of setting weight coefficients in OPTICS algorithm",
        "body": "We are very interested in enhancement of OPTICS. It would be very important to support sample weights.",
    },
    {
        "label": "docs",
        "title": "LogisticRegression see also should include LogisticRegressionCV",
        "body": "See also section in the class docstring lacks this obvious reference.",
    },
    {
        "label": "docs",
        "title": "model_selection.StratifiedKFold should not require the data array",
        "body": "When importing sklearn.cross_validation I get a DeprecationWarning saying to use sklearn.model_selection instead.",
    },
    {
        "label": "question",
        "title": "Add an example about how to visualize the results of GridSearchCV",
        "body": "I think it would be useful to add an example about how to visualize GridSearchCV results with two hyperparameters.",
    },
    {
        "label": "question",
        "title": "Matplotlib warnings in CircleCI after switching to matplotlib 3.X",
        "body": "We get warnings: UserWarning: Matplotlib is currently using agg, which is a non-GUI backend.",
    },
]


def build_system_prompt() -> str:
    examples_text = "\n\n".join(
        f"Title: {ex['title']}\nBody: {ex['body']}\nLabel: {ex['label']}"
        for ex in FEW_SHOT_EXAMPLES
    )
    return f"""You are a GitHub issue classifier. Classify issues into exactly one of:
- bug: A defect, crash, unexpected behaviour, or regression
- feature: A request for new functionality or enhancement
- docs: A request for documentation improvement or fix
- question: A usage question, support request, or triage item

Here are examples:

{examples_text}

Respond with ONLY the label name — no explanation, no punctuation."""


SYSTEM_PROMPT = build_system_prompt()


def classify_issue(client: Groq, text: str, model: str) -> tuple[str, float]:
    truncated = text[:400]
    for attempt in range(5):
        try:
            t0 = time.perf_counter()
            response = client.chat.completions.create(
                model=model,
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
            err = str(e).lower()
            if "429" in str(e) or "rate_limit" in err:
                wait = 10 * (attempt + 1)
                print(f"  Rate limit hit — waiting {wait}s...")
                time.sleep(wait)
            elif "timeout" in err or "connecttimeout" in err:
                wait = 15 * (attempt + 1)
                print(f"  Timeout — waiting {wait}s and retrying...")
                time.sleep(wait)
            else:
                raise
    return "question", 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--model", type=str, default=MODEL_DEFAULT,
                        help="Groq model ID (e.g. llama-3.1-70b-versatile)")
    args = parser.parse_args()

    model = args.model

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise SystemExit("ERROR: set GROQ_API_KEY environment variable")

    client = Groq(api_key=api_key)

    print("Loading test split...")
    test_texts, test_labels = load_split("test")

    # Same sample as zero-shot eval (seed=42)
    random.seed(42)
    indices = random.sample(range(len(test_texts)), min(args.sample_size, len(test_texts)))
    sample_texts  = [test_texts[i] for i in indices]
    sample_labels = [test_labels[i] for i in indices]

    print(f"Running few-shot classification on {len(sample_texts)} issues...")
    print(f"Model: {model}  |  Examples in prompt: {len(FEW_SHOT_EXAMPLES)} (2 per class)\n")

    preds = []
    latencies = []

    for i, (text, true_label) in enumerate(zip(sample_texts, sample_labels)):
        pred, latency_ms = classify_issue(client, text, model)
        preds.append(pred)
        latencies.append(latency_ms)
        if (i + 1) % 20 == 0:
            acc_so_far = sum(p == label for p, label in zip(preds, sample_labels[:len(preds)])) / len(preds)
            status = "✓" if pred == true_label else "✗"
            print(f"  [{i+1}/{len(sample_texts)}] running_acc={acc_so_far:.3f}  {status} pred={pred} true={true_label}")

    accuracy = accuracy_score(sample_labels, preds)
    macro_f1 = f1_score(sample_labels, preds, average="macro", labels=CLASSES)
    report   = classification_report(sample_labels, preds, labels=CLASSES, output_dict=True)
    per_class_f1 = {cls: round(report[cls]["f1-score"], 4) for cls in CLASSES}
    avg_latency  = sum(latencies) / len(latencies)

    # Load zero-shot results for comparison
    zeroshot_path = Path("models/llm_baseline/metrics.json")
    zeroshot = json.loads(zeroshot_path.read_text()) if zeroshot_path.exists() else None

    print(f"\n{'='*55}")
    print(f"{'Metric':<20} {'Zero-shot':>12} {'Few-shot':>12}  {'Delta':>8}")
    print(f"{'='*55}")
    for metric, fs_val in [
        ("Accuracy",    accuracy),
        ("Macro-F1",    macro_f1),
        ("bug F1",      per_class_f1["bug"]),
        ("feature F1",  per_class_f1["feature"]),
        ("docs F1",     per_class_f1["docs"]),
        ("question F1", per_class_f1["question"]),
    ]:
        if zeroshot:
            zs_val = (
                zeroshot["test"]["accuracy"] if metric == "Accuracy"
                else zeroshot["test"]["macro_f1"] if metric == "Macro-F1"
                else zeroshot["test"]["per_class_f1"].get(metric.split()[0], 0)
            )
            delta = fs_val - zs_val
            sign = "+" if delta >= 0 else ""
            print(f"  {metric:<18} {zs_val:>12.4f} {fs_val:>12.4f}  {sign}{delta:>7.4f}")
        else:
            print(f"  {metric:<18} {'N/A':>12} {fs_val:>12.4f}")

    print(f"\n{classification_report(sample_labels, preds, labels=CLASSES)}")

    metrics = {
        "model": model,
        "provider": "groq",
        "mode": "few-shot",
        "num_examples": len(FEW_SHOT_EXAMPLES),
        "sample_size": len(sample_texts),
        "test": {
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
            "per_class_f1": per_class_f1,
            "latency_ms_per_sample": round(avg_latency, 1),
        },
        "cost_note": "Free tier (Groq) — $0",
    }

    # Save under a model-specific subdirectory
    model_slug = model.replace("/", "_").replace(":", "_")
    output_dir = OUTPUT_DIR_BASE / model_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics → {metrics_path}")


if __name__ == "__main__":
    main()
