# Feature Specification: Phase 2 — Deep Learning Track

**Feature Branch**: `phase-2-dl-track`

**Created**: 2026-05-18

**Status**: Draft

## User Scenarios & Testing

### User Story 1 — Fine-Tuned Classifier Classifies Issues (Priority: P1)

A maintainer submits an issue title and body. The model server returns one of: `bug`, `feature`, `docs`, `question` with a confidence score. The fine-tuned transformer is the primary model.

**Why this priority**: The classifier is the core graded deliverable of the DL track. All three-way comparisons depend on it.

**Independent Test**: POST an issue to `POST /classify` on the model server and receive a label + confidence. Run the golden eval set and confirm macro-F1 meets the committed threshold.

**Acceptance Scenarios**:

1. **Given** an issue title and body, **When** `POST /classify` is called, **Then** the response contains `label` (one of bug/feature/docs/question) and `confidence`.
2. **Given** the fine-tuned model weights, **When** the SHA-256 is checked against the model card, **Then** they match — otherwise the model server refuses to start.
3. **Given** the 25-issue golden eval set, **When** the eval suite runs, **Then** macro-F1 meets or exceeds the committed threshold in `eval_thresholds.yaml`.

---

### User Story 2 — Three-Way Model Comparison Is Documented (Priority: P1)

The fine-tuned transformer, a classical ML baseline (e.g. TF-IDF + LogReg), and an LLM baseline are all evaluated on the same test split. Results appear in `DECISIONS.md` with a deployment recommendation.

**Why this priority**: The three-way comparison is explicitly graded and must be backed by numbers.

**Independent Test**: Run all three inference scripts against the test split and produce a comparison table with accuracy, macro-F1, per-class F1, latency (ms/sample), and cost estimate.

**Acceptance Scenarios**:

1. **Given** the test split, **When** all three models run inference, **Then** each produces accuracy, macro-F1, per-class F1, latency, and cost figures.
2. **Given** the comparison table, **When** `DECISIONS.md` is read, **Then** a deployment choice is stated with a one-line rationale backed by the numbers.

---

### User Story 3 — NER And Summarizer Endpoints Are Available (Priority: P2)

The model server exposes `POST /ner` (extracts code-shaped entities: function names, error codes, package names, version strings) and `POST /summarize` (returns a short summary of an issue thread). Both are called by the chatbot over HTTP.

**Why this priority**: These are tools the chatbot uses. They must be live endpoints before chatbot development (Phase 4).

**Independent Test**: POST sample issue text to each endpoint and receive structured output. Call them from a simple Python script simulating the chatbot.

**Acceptance Scenarios**:

1. **Given** raw issue text, **When** `POST /ner` is called, **Then** response contains a list of extracted entities with types (function, error_code, package, version).
2. **Given** a multi-comment issue thread, **When** `POST /summarize` is called, **Then** response contains a concise summary under 100 words.
3. **Given** the model server is down, **When** the chatbot calls these endpoints, **Then** the chatbot catches the error and falls back gracefully.

---

### Edge Cases

- What if an issue has no body, only a title? Preprocessing must handle empty fields.
- What if the LLM baseline API call times out? Record the timeout as a latency data point, do not crash.
- What if a class has very few examples? Per-class F1 must still be reported; note class imbalance in DECISIONS.md.
- What if the fine-tuned model's SHA-256 mismatches? Model server MUST refuse to start.

## Requirements

### Functional Requirements

- **FR-001**: A preprocessing pipeline MUST clean issue text: strip HTML, normalize whitespace, handle empty fields. Choices defended in `DECISIONS.md`.
- **FR-002**: A small encoder transformer (e.g. DistilBERT, BERT-base) MUST be fine-tuned on the train split for 4-class issue classification.
- **FR-003**: Training MUST be tracked with a run logger (e.g. MLflow, W&B). Artifacts saved to MinIO.
- **FR-004**: A model card MUST be saved alongside weights: architecture, hyperparameters, training data hash, freeze policy, and final metrics.
- **FR-005**: The freeze policy (which layers are frozen/unfrozen) MUST be documented and defended in `DECISIONS.md`.
- **FR-006**: A classical ML baseline (TF-IDF + a scikit-learn classifier) MUST be trained on the same splits.
- **FR-007**: An LLM baseline MUST run zero-shot or few-shot classification on the test split using an LLM API.
- **FR-008**: Three-way comparison table (accuracy, macro-F1, per-class F1, latency, cost) MUST appear in `DECISIONS.md` with a deployment choice.
- **FR-009**: `POST /classify` endpoint MUST be exposed on the model server.
- **FR-010**: `POST /ner` endpoint MUST extract code-shaped entities (function names, error codes, packages, versions).
- **FR-011**: `POST /summarize` endpoint MUST return a concise summary of issue text.
- **FR-012**: Model server MUST refuse to start if classifier weights are missing or SHA-256 mismatches the model card.

### Key Entities

- **ModelCard**: architecture, hyperparameters, training_data_hash, freeze_policy, metrics, artifact_path, sha256.
- **ClassificationResult**: label (bug/feature/docs/question), confidence, model_used, latency_ms.
- **NERResult**: list of Entity(text, type, start, end).
- **SummarizationResult**: summary (str), model_used.

## Success Criteria

- **SC-001**: Fine-tuned model macro-F1 on the test split meets committed threshold in `eval_thresholds.yaml`.
- **SC-002**: All three models produce comparable metrics on the same test split.
- **SC-003**: `DECISIONS.md` contains the three-way comparison table and a defended deployment choice.
- **SC-004**: `POST /classify`, `POST /ner`, `POST /summarize` all return correct structured responses.
- **SC-005**: Model card exists in MinIO with SHA-256 that matches the weights file.
- **SC-006**: Training run is visible in the run logger with loss curves and final metrics.

## Assumptions

- A small encoder (≤110M parameters) is sufficient for 4-class classification on this dataset.
- The LLM baseline uses the same LLM provider as the chatbot (cost tracked separately).
- NER is integration-only — using a pre-trained model or LLM prompt, not trained from scratch.
- Summarizer is pre-trained or LLM-driven — not trained from scratch.
