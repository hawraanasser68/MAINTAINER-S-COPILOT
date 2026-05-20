# Implementation Plan: Phase 2 — Deep Learning Track

**Branch**: `002-dl-track` | **Date**: 2026-05-19 | **Spec**: [spec-phase-2-dl-track.md](../../.specify/memory/spec-phase-2-dl-track.md)

## Summary

Train three classifiers (TF-IDF+LogReg baseline, DistilBERT fine-tune, LLM zero-shot), compare them, wire NER + summarizer, and expose `/classify`, `/ner`, `/summarize` on the model server. All decisions backed by numbers in `DECISIONS.md`.

## Preprocessing Contract (from EDA)

- Input: `f"{title} {body}".strip()`
- Tokenizer: `truncation=True, max_length=512`
- Class weights: `compute_class_weight("balanced")`
- 5.3% of issues exceed 512 words → acceptable truncation

## Task List

### Sub-phase A — Preprocessing pipeline
- [ ] T001: `scripts/preprocess.py` — `make_text(title, body)`, `load_split(name)` helpers used by all 3 models

### Sub-phase B — Classical ML baseline
- [ ] T002: `scripts/train_baseline.py` — TF-IDF + LogReg with `class_weight="balanced"`, saves model + metrics JSON
- [ ] T003: `tests/test_baseline.py` — test that saved model loads and predicts all 4 classes

### Sub-phase C — DistilBERT fine-tune
- [ ] T004: `scripts/train_classifier.py` — fine-tune `distilbert-base-uncased`, save weights + model card to `models/`
- [ ] T005: `scripts/eval_classifier.py` — load weights, run on test split, print + save metrics JSON
- [ ] T006: `tests/test_classifier.py` — load saved model card, verify SHA-256 matches weights

### Sub-phase D — LLM baseline
- [ ] T007: `scripts/eval_llm_baseline.py` — zero-shot via Claude API on 200-issue sample from test split, save metrics
- [ ] T008: Update `DECISIONS.md` with three-way comparison table

### Sub-phase E — NER + Summarizer
- [ ] T009: `modelserver/ner.py` — LLM-driven NER (function names, error codes, packages, versions)
- [ ] T010: `modelserver/summarize.py` — LLM-driven summarizer (< 100 words)

### Sub-phase F — Model server endpoints
- [ ] T011: `modelserver/classifier.py` — load weights at startup, SHA-256 check, inference
- [ ] T012: `modelserver/main.py` — wire `/classify`, `/ner`, `/summarize`, `/health` with model card
- [ ] T013: `tests/test_modelserver.py` — integration tests for all 3 endpoints

## Model Card Schema

```json
{
  "architecture": "distilbert-base-uncased",
  "num_labels": 4,
  "classes": ["bug", "feature", "docs", "question"],
  "max_length": 512,
  "freeze_policy": "all encoder layers frozen, classifier head trained",
  "hyperparameters": {"lr": 2e-5, "epochs": 3, "batch_size": 16},
  "training_data_hash": "<sha256 of train.jsonl>",
  "metrics": {"macro_f1": 0.0, "accuracy": 0.0},
  "artifact_path": "models/classifier/",
  "sha256": "<sha256 of model weights>"
}
```

## Freeze Policy Decision

Fine-tune with all DistilBERT layers **unfrozen** (`unfreeze_all`):
- Dataset is domain-specific (GitHub issues) — different from general web text DistilBERT was trained on
- 3500 training examples is sufficient for full fine-tuning of a 66M-parameter model
- Learning rate 2e-5 (low enough to avoid catastrophic forgetting)
- Alternative (freeze all but last 2 layers) produces ~2% lower macro-F1 in literature for similar tasks
- Defended in `DECISIONS.md` after training comparison

## Directory Layout

```
models/
  classifier/
    config.json          # model card
    pytorch_model.bin    # weights
    tokenizer/           # saved tokenizer
  baseline/
    tfidf_logreg.pkl     # sklearn pipeline
    metrics.json

scripts/
  preprocess.py
  train_baseline.py
  eval_baseline.py       # (reuse in compare)
  train_classifier.py
  eval_classifier.py
  eval_llm_baseline.py
  compare_models.py      # generates DECISIONS table

modelserver/
  main.py
  classifier.py
  ner.py
  summarize.py
```
