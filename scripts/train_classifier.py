"""
Fine-tune DistilBERT for 4-class GitHub issue classification.

Usage:
    python scripts/train_classifier.py [--epochs N] [--batch-size N] [--lr F]

Outputs:
    models/classifier/pytorch_model.bin  — fine-tuned weights
    models/classifier/config.json        — model card
    models/classifier/tokenizer/         — saved tokenizer
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Force line-buffered output so progress is visible in background runs
sys.stdout.reconfigure(line_buffering=True)

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, f1_score  # noqa: E402
from sklearn.metrics import confusion_matrix as sk_confusion_matrix  # noqa: E402
from sklearn.utils.class_weight import compute_class_weight  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.preprocess import CLASSES, load_split  # noqa: E402

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = Path("models/classifier")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL2ID = {cls: i for i, cls in enumerate(CLASSES)}
ID2LABEL = {i: cls for i, cls in enumerate(CLASSES)}


class IssueDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[str], tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        self.labels = torch.tensor([LABEL2ID[lbl] for lbl in labels], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _upload_to_minio(weights_file: Path, run_id: str) -> None:
    endpoint = os.environ.get("MINIO_ENDPOINT", "")
    if not endpoint:
        return
    try:
        import boto3
        from botocore.exceptions import ClientError

        bucket = "model-artifacts"
        key = f"classifier/{run_id}/{weights_file.name}"
        client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
            aws_secret_access_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
            region_name="us-east-1",
        )
        try:
            client.head_bucket(Bucket=bucket)
        except ClientError:
            client.create_bucket(Bucket=bucket)
        client.upload_file(str(weights_file), bucket, key)
        print(f"[MinIO] Uploaded weights → s3://{bucket}/{key}")
    except Exception as exc:
        print(f"[MinIO] Upload failed (non-blocking): {exc}")


def train(epochs: int, batch_size: int, lr: float) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    mlflow_run = None
    try:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("issue-classifier")
        mlflow_run = mlflow.start_run()
        mlflow.log_params({
            "model_name": MODEL_NAME,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": 0.01,
            "max_length": 128,
            "class_weight": "balanced",
        })
        print(f"[MLflow] Run started: {mlflow_run.info.run_id}")
    except Exception as exc:
        print(f"[MLflow] Tracking unavailable: {exc}")
        mlflow_run = None

    print("Loading splits...")
    train_texts, train_labels = load_split("train")
    val_texts,   val_labels   = load_split("val")
    test_texts,  test_labels  = load_split("test")

    print("Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    print("Tokenizing (this takes ~1 min on CPU)...")
    train_dataset = IssueDataset(train_texts, train_labels, tokenizer)
    val_dataset   = IssueDataset(val_texts,   val_labels,   tokenizer)
    test_dataset  = IssueDataset(test_texts,  test_labels,  tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size)

    print("Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASSES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    # Class weights for loss function
    weights_array = compute_class_weight(
        "balanced", classes=np.array(CLASSES), y=train_labels
    )
    class_weights = torch.tensor(weights_array, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs * len(train_loader)
    )

    best_val_f1 = 0.0
    best_epoch = 0
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            input_ids     = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels        = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Val
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend([ID2LABEL[p] for p in preds])
                all_labels.extend([ID2LABEL[lbl.item()] for lbl in batch["labels"]])

        val_f1  = f1_score(all_labels, all_preds, average="macro", labels=CLASSES)
        val_acc = accuracy_score(all_labels, all_preds)
        print(
            f"Epoch {epoch}/{epochs}  loss={avg_loss:.4f}"
            f"  val_f1={val_f1:.4f}  val_acc={val_acc:.4f}"
        )

        if mlflow_run:
            try:
                mlflow.log_metrics(
                    {"train_loss": avg_loss, "val_f1": val_f1, "val_acc": val_acc},
                    step=epoch,
                )
            except Exception:
                pass

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR / "tokenizer")

    train_time = time.perf_counter() - train_start
    print(f"\nBest val macro-F1: {best_val_f1:.4f} at epoch {best_epoch}")
    print(f"Training time: {train_time:.0f}s")

    # Reload best checkpoint and evaluate on test
    print("\nLoading best checkpoint for test evaluation...")
    best_model = DistilBertForSequenceClassification.from_pretrained(
        OUTPUT_DIR, num_labels=len(CLASSES), id2label=ID2LABEL, label2id=LABEL2ID
    ).to(device)
    best_model.eval()

    t0 = time.perf_counter()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            outputs = best_model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend([ID2LABEL[p] for p in preds])
            all_labels.extend([ID2LABEL[lbl.item()] for lbl in batch["labels"]])

    latency_ms = (time.perf_counter() - t0) / len(test_texts) * 1000
    test_f1  = f1_score(all_labels, all_preds, average="macro", labels=CLASSES)
    test_acc = accuracy_score(all_labels, all_preds)

    report = classification_report(all_labels, all_preds, labels=CLASSES, output_dict=True)
    per_class_f1 = {cls: round(report[cls]["f1-score"], 4) for cls in CLASSES}

    cm = sk_confusion_matrix(all_labels, all_preds, labels=CLASSES).tolist()
    confusion_matrix_data = {"labels": CLASSES, "matrix": cm}

    print(  # noqa: E501
        f"\nTest accuracy={test_acc:.4f}  macro-F1={test_f1:.4f}  latency={latency_ms:.2f}ms/sample"
    )
    print(f"\n{classification_report(all_labels, all_preds, labels=CLASSES)}")

    # Compute SHA-256 of weights
    weights_file = OUTPUT_DIR / "model.safetensors"
    if not weights_file.exists():
        weights_file = OUTPUT_DIR / "pytorch_model.bin"
    sha256 = sha256_file(weights_file) if weights_file.exists() else "unknown"

    # Compute training data hash
    train_path = Path("data/splits/train.jsonl")
    train_hash = sha256_file(train_path) if train_path.exists() else "unknown"

    # Save model card
    model_card = {
        "architecture": MODEL_NAME,
        "num_labels": len(CLASSES),
        "classes": CLASSES,
        "max_length": 128,
        "freeze_policy": "all layers unfrozen — full fine-tune with lr=2e-5",
        "hyperparameters": {
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "weight_decay": 0.01,
            "class_weight": "balanced",
        },
        "training_data_hash": train_hash,
        "metrics": {
            "val_macro_f1": round(best_val_f1, 4),
            "test_macro_f1": round(test_f1, 4),
            "test_accuracy": round(test_acc, 4),
            "per_class_f1": per_class_f1,
            "latency_ms_per_sample": round(latency_ms, 3),
            "confusion_matrix": confusion_matrix_data,
        },
        "artifact_path": str(OUTPUT_DIR),
        "sha256": sha256,
        "train_time_s": round(train_time, 0),
    }

    card_path = OUTPUT_DIR / "model_card.json"
    card_path.write_text(json.dumps(model_card, indent=2))
    print(f"\nSaved model card → {card_path}")
    print(f"SHA-256: {sha256}")

    run_id = mlflow_run.info.run_id if mlflow_run else "local"

    if mlflow_run:
        try:
            mlflow.log_metrics({
                "test_macro_f1": round(test_f1, 4),
                "test_accuracy": round(test_acc, 4),
                "latency_ms_per_sample": round(latency_ms, 3),
                "train_time_s": round(train_time, 0),
                **{f"test_f1_{cls}": per_class_f1[cls] for cls in CLASSES},
            })
            mlflow.log_artifact(str(card_path), artifact_path="classifier")
            if weights_file.exists():
                mlflow.log_artifact(str(weights_file), artifact_path="classifier")
            mlflow.set_tag("sha256", sha256)
            mlflow.end_run()
            print(f"[MLflow] Run {run_id} completed.")
        except Exception as exc:
            print(f"[MLflow] Failed to log final artifacts: {exc}")
            try:
                mlflow.end_run()
            except Exception:
                pass

    if weights_file.exists():
        _upload_to_minio(weights_file, run_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
