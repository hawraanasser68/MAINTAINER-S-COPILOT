# Feature Specification: Phase 6 — Evals & CI

**Feature Branch**: `phase-6-evals-ci`

**Created**: 2026-05-18

**Status**: Draft

## User Scenarios & Testing

### User Story 1 — CI Blocks Merge When Evals Regress (Priority: P1)

A developer pushes a commit that accidentally degrades the classifier. The CI pipeline runs the classification eval suite, detects the regression below the committed threshold, and blocks the merge. The developer sees a clear error in CI logs pointing to the failing metric.

**Why this priority**: CI-blocking evals are the core graded requirement. Without this, the thresholds are decoration.

**Independent Test**: Temporarily lower a model's performance (e.g. by corrupting weights), push, and confirm CI fails with a clear regression message. Restore and confirm CI passes.

**Acceptance Scenarios**:

1. **Given** a commit is pushed, **When** CI runs, **Then** both eval suites (classification + RAG) run automatically.
2. **Given** a metric falls below the threshold in `eval_thresholds.yaml`, **When** CI runs, **Then** the pipeline fails with a message identifying the metric and the delta.
3. **Given** all metrics meet thresholds, **When** CI runs, **Then** the pipeline passes and `eval_report.json` is uploaded to MinIO.
4. **Given** any eval threshold is set to zero or disabled in `eval_thresholds.yaml`, **When** the API starts, **Then** it refuses to boot.

---

### User Story 2 — Redaction Is Proven By A Test (Priority: P1)

A test asserts that a message containing a fake API key (e.g. `sk-fake1234`) never appears unredacted in logs, trace spans, or memory writes. This test runs in CI on every push.

**Why this priority**: "Logs are redacted" is a graded requirement. Without a passing test, the claim cannot be verified.

**Independent Test**: Run `pytest tests/test_redaction.py` and confirm the fake API key does not appear in any log output, span attributes, or memory entries after the service processes a message containing it.

**Acceptance Scenarios**:

1. **Given** a message containing `sk-fake1234`, **When** it is processed by any service, **Then** the string `sk-fake1234` does not appear in any log line, span attribute, or memory write.
2. **Given** the redaction patterns defined in `SECURITY.md`, **When** they are applied, **Then** all patterns are covered by at least one test assertion.
3. **Given** the redaction test runs in CI, **When** it fails, **Then** the CI pipeline blocks merge.

---

### User Story 3 — Eval Reports Are Stored And Diffed (Priority: P2)

Every CI run writes `eval_report.json` to MinIO. The pipeline diffs it against the previous green build and logs any metric changes (positive or negative).

**Why this priority**: Tracking eval history allows regression detection and trend visibility over the project lifetime.

**Independent Test**: Run CI twice. Confirm two `eval_report.json` files exist in MinIO and the second run logs a diff against the first.

**Acceptance Scenarios**:

1. **Given** a CI run completes, **When** eval suites finish, **Then** `eval_report.json` is written to MinIO with a timestamped key.
2. **Given** a previous green build's report exists, **When** the current run finishes, **Then** the diff (metric deltas) is logged in CI output.
3. **Given** no previous green build exists, **When** CI runs for the first time, **Then** no diff is attempted — current report is stored as the baseline.

---

### Edge Cases

- What if MinIO is unreachable during CI? The eval suite must still run and fail/pass correctly; the MinIO upload failure must be logged but must NOT cause a false eval failure.
- What if the golden set files are missing or corrupted? CI must fail with a clear error: "golden set not found" rather than a confusing assertion error.
- What if the judge model for RAG eval is unavailable? CI must fail with a clear ToolFailure, not hang indefinitely — add a timeout.
- What if a threshold is added to `eval_thresholds.yaml` but the corresponding metric is not yet implemented? The eval runner must raise a configuration error.

## Requirements

### Functional Requirements

- **FR-001**: `eval_thresholds.yaml` MUST contain committed thresholds for: classifier macro-F1, per-class F1 (all 4 classes), RAG hit@5, MRR@10, faithfulness, answer relevancy.
- **FR-002**: Both eval suites MUST run in CI on every push (GitHub Actions).
- **FR-003**: Regression below any threshold MUST block merge — not warn.
- **FR-004**: `eval_report.json` MUST be written on every CI run and uploaded to MinIO with a timestamped key.
- **FR-005**: The CI pipeline MUST diff `eval_report.json` against the previous green build and log metric deltas.
- **FR-006**: The redaction test MUST assert that a fake API key never appears unredacted in logs, traces, or memory.
- **FR-007**: Redaction patterns MUST be documented in `SECURITY.md`.
- **FR-008**: CI pipeline steps MUST be: lint → type-check → build images → classification eval → RAG eval → redaction test → smoke test.
- **FR-009**: The smoke test MUST verify: stack starts, `/health` returns 200, one classify call succeeds, one RAG call succeeds.
- **FR-010**: The classification eval MUST run against all three models (fine-tuned, classical ML, LLM baseline) and report macro-F1, per-class F1, and confusion matrix for each.
- **FR-011**: The RAG eval MUST use either RAGAS or a frozen judge model (choice defended in `DECISIONS.md`). Pinned version for reproducibility.
- **FR-012**: 5 of the 25 RAG golden examples MUST be hand-labelled; human-judge agreement MUST be reported in `EVALS.md`.

### Key Entities

- **EvalThreshold**: metric_name, minimum_value, model (for classification metrics).
- **EvalReport**: run_id, timestamp, classification_results (per model), rag_results, redaction_passed, commit_sha.
- **GoldenClassificationExample**: issue_id, text, true_label, source (hand-curated).
- **GoldenRAGExample**: question, ideal_answer, ground_truth_chunk_ids, hand_labelled (bool).

## Success Criteria

- **SC-001**: CI fails when any metric drops below its threshold in `eval_thresholds.yaml`.
- **SC-002**: CI passes when all metrics meet thresholds and the redaction test passes.
- **SC-003**: `eval_report.json` is present in MinIO after every CI run.
- **SC-004**: The redaction test explicitly asserts no fake API key appears in logs, traces, or memory.
- **SC-005**: `EVALS.md` documents human-judge agreement for the 5 hand-labelled RAG examples.
- **SC-006**: The eval runner raises a configuration error if a threshold references an unimplemented metric.

## Assumptions

- The golden sets are finalized before Phase 6 CI gates are enabled (classification set in Phase 2, RAG set in Phase 3).
- The judge model version is pinned in `DECISIONS.md` and in the CI config to ensure reproducible results.
- MinIO upload failures are non-blocking for the pass/fail decision — the metric thresholds alone determine CI outcome.
- GitHub Actions is the CI platform.
