"""
Train and evaluate the classical ML baseline: TF-IDF + Logistic Regression.

Usage:
    python scripts/train_baseline.py

Outputs:
    models/baseline/tfidf_logreg.pkl   — sklearn pipeline
    models/baseline/metrics.json       — accuracy, macro-F1, per-class F1, latency
"""

import json
import sys
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.preprocess import CLASSES, load_split

OUTPUT_DIR = Path("models/baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50_000,
            sublinear_tf=True,
            strip_accents="unicode",
            min_df=2,
        )),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=42,
        )),
    ])


def main() -> None:
    print("Loading splits...")
    train_texts, train_labels = load_split("train")
    val_texts,   val_labels   = load_split("val")
    test_texts,  test_labels  = load_split("test")

    print(f"  Train: {len(train_texts)}  Val: {len(val_texts)}  Test: {len(test_texts)}")

    print("\nTraining TF-IDF + LogReg...")
    t0 = time.perf_counter()
    pipe = build_pipeline()
    pipe.fit(train_texts, train_labels)
    train_time = time.perf_counter() - t0
    print(f"  Trained in {train_time:.1f}s")

    # Val metrics
    val_preds = pipe.predict(val_texts)
    val_f1 = f1_score(val_labels, val_preds, average="macro", labels=CLASSES)
    val_acc = accuracy_score(val_labels, val_preds)
    print(f"\nVal  accuracy={val_acc:.4f}  macro-F1={val_f1:.4f}")

    # Test metrics + latency
    t0 = time.perf_counter()
    test_preds = pipe.predict(test_texts)
    inference_time = time.perf_counter() - t0
    latency_ms = (inference_time / len(test_texts)) * 1000

    test_f1 = f1_score(test_labels, test_preds, average="macro", labels=CLASSES)
    test_acc = accuracy_score(test_labels, test_preds)

    print(f"Test accuracy={test_acc:.4f}  macro-F1={test_f1:.4f}  latency={latency_ms:.2f}ms/sample")

    report = classification_report(test_labels, test_preds, labels=CLASSES, output_dict=True)
    per_class_f1 = {cls: round(report[cls]["f1-score"], 4) for cls in CLASSES}

    print("\nPer-class F1 (test):")
    for cls, f in per_class_f1.items():
        print(f"  {cls:<10} {f:.4f}")

    print(f"\n{classification_report(test_labels, test_preds, labels=CLASSES)}")

    # Save model
    model_path = OUTPUT_DIR / "tfidf_logreg.pkl"
    joblib.dump(pipe, model_path)
    print(f"Saved model → {model_path}")

    # Save metrics
    metrics = {
        "model": "tfidf_logreg",
        "val": {"accuracy": round(val_acc, 4), "macro_f1": round(val_f1, 4)},
        "test": {
            "accuracy": round(test_acc, 4),
            "macro_f1": round(test_f1, 4),
            "per_class_f1": per_class_f1,
            "latency_ms_per_sample": round(latency_ms, 3),
        },
        "train_time_s": round(train_time, 1),
    }
    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics → {metrics_path}")


if __name__ == "__main__":
    main()
