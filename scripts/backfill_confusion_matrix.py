"""
One-shot script: loads the saved DistilBERT classifier, runs test-set inference,
and writes the confusion_matrix field into models/classifier/model_card.json.

Run from project root:
    python scripts/backfill_confusion_matrix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.preprocess import CLASSES, load_split  # noqa: E402

OUTPUT_DIR = ROOT / "models" / "classifier"
LABEL2ID = {cls: i for i, cls in enumerate(CLASSES)}
ID2LABEL = {i: cls for i, cls in enumerate(CLASSES)}


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[str], labels: list[str], tokenizer):
        self.encodings = tokenizer(
            texts, truncation=True, max_length=128, padding="max_length", return_tensors="pt"
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


def main() -> None:
    card_path = OUTPUT_DIR / "model_card.json"
    if not card_path.exists():
        print("[ERROR] model_card.json not found — run train_classifier.py first")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading tokenizer and model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(OUTPUT_DIR / "tokenizer")
    model = DistilBertForSequenceClassification.from_pretrained(
        OUTPUT_DIR, num_labels=len(CLASSES), id2label=ID2LABEL, label2id=LABEL2ID
    ).to(device)
    model.eval()

    print("Loading test split...")
    test_texts, test_labels = load_split("test")
    loader = DataLoader(_Dataset(test_texts, test_labels, tokenizer), batch_size=32)

    all_preds: list[str] = []
    all_labels: list[str] = []
    with torch.no_grad():
        for batch in loader:
            preds = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend([ID2LABEL[p] for p in preds])
            all_labels.extend([ID2LABEL[lbl.item()] for lbl in batch["labels"]])

    cm = sk_confusion_matrix(all_labels, all_preds, labels=CLASSES).tolist()
    confusion_matrix_data = {"labels": CLASSES, "matrix": cm}

    card = json.loads(card_path.read_text())
    card.setdefault("metrics", {})["confusion_matrix"] = confusion_matrix_data
    card_path.write_text(json.dumps(card, indent=2))

    print("\nConfusion matrix:")
    print(f"  labels: {CLASSES}")
    for row_label, row in zip(CLASSES, cm):
        print(f"  {row_label:8}: {row}")
    print(f"\n[OK] model_card.json updated → {card_path}")


if __name__ == "__main__":
    main()
