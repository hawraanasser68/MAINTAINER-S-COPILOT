"""
DistilBERT classifier — loads weights at startup, validates SHA-256.
Raises RuntimeError if weights are missing or SHA-256 mismatches.
"""

import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

MODEL_DIR = Path(__file__).parent.parent / "models" / "classifier"
CLASSES = ["bug", "feature", "docs", "question"]
LABEL2ID = {cls: i for i, cls in enumerate(CLASSES)}
ID2LABEL = {i: cls for i, cls in enumerate(CLASSES)}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


@lru_cache(maxsize=1)
def load_model() -> tuple:
    """Load model + tokenizer once, validate SHA-256 against model card."""
    card_path = MODEL_DIR / "model_card.json"
    if not card_path.exists():
        raise RuntimeError(
            f"Model card not found at {card_path}. Run scripts/train_classifier.py first."
        )

    card = json.loads(card_path.read_text())
    expected_sha = card.get("sha256", "")

    weights_file = MODEL_DIR / "model.safetensors"
    if not weights_file.exists():
        weights_file = MODEL_DIR / "pytorch_model.bin"
    if not weights_file.exists():
        raise RuntimeError(f"No weight file found in {MODEL_DIR}.")

    actual_sha = _sha256(weights_file)
    if expected_sha and actual_sha != expected_sha:
        raise RuntimeError(
            f"SHA-256 mismatch for {weights_file.name}.\n"
            f"  Expected : {expected_sha}\n"
            f"  Got      : {actual_sha}\n"
            "Model weights may be corrupted or tampered. Refusing to start."
        )

    device = torch.device("cpu")
    model = DistilBertForSequenceClassification.from_pretrained(
        str(MODEL_DIR),
        num_labels=len(CLASSES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)
    model.eval()

    tokenizer_dir = MODEL_DIR / "tokenizer"
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(tokenizer_dir))

    return model, tokenizer, device, card


def classify(title: str, body: str) -> dict:
    model, tokenizer, device, card = load_model()

    text = f"{(title or '').strip()} {(body or '').strip()}".strip()
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=128,
        padding="max_length",
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(**inputs).logits
    latency_ms = (time.perf_counter() - t0) * 1000

    probs = torch.softmax(logits, dim=-1)[0].tolist()
    pred_id = int(torch.argmax(logits, dim=-1).item())
    label = ID2LABEL[pred_id]
    confidence = round(probs[pred_id], 4)

    return {
        "label": label,
        "confidence": confidence,
        "all_scores": {ID2LABEL[i]: round(p, 4) for i, p in enumerate(probs)},
        "model_used": card.get("architecture", "distilbert-base-uncased"),
        "latency_ms": round(latency_ms, 2),
    }
