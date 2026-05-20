# Research: Phase 1 — Foundations

## Tracing Backend

**Decision**: OpenTelemetry SDK → Jaeger (local, Docker)

**Rationale**: OTEL is the open standard — switching exporters later (Jaeger → Datadog, Honeycomb, etc.) requires only a config change, not a code rewrite. Jaeger runs as a single Docker container with a built-in UI, zero cost, zero config. Every span attribute will already be in OTEL format, satisfying the requirement that every LLM call, tool call, and RAG retrieval is a span.

**Alternatives considered**: Langfuse (LLM-specific, good UI but adds a vendor dependency), OpenTelemetry → stdout (free but no UI for the Friday demo), Weights & Biases (ML-focused, not suitable for general tracing).

---

## GitHub Repo For Dataset

**Decision**: `huggingface/transformers`

**Rationale**: Large volume of closed issues (5000+), well-maintained labelling by an active maintainer team, directly relevant domain (transformer fine-tuning), and clean enough label taxonomy to map to bug/feature/docs/question without ambiguity.

**Label mapping**: Defined in `DECISIONS.md`. Unmapped labels are logged and excluded from splits. Preliminary mapping:
- `bug` → issues labelled `bug`, `regression`, `broken`
- `feature` → issues labelled `feature request`, `enhancement`, `new model`
- `docs` → issues labelled `documentation`, `docs`
- `question` → issues labelled `question`, `help wanted`, `usage`

**EDA**: `notebooks/eda.ipynb` documents label distribution, issue length, temporal spread, class imbalance, and missing body rate before any model training begins.

---

## Vector Store

**Decision**: pgvector (already in Postgres stack)

**Rationale**: Avoids a separate Qdrant container. pgvector supports cosine similarity, IVFFlat and HNSW indexes. For this project's scale (hundreds of thousands of chunks max), pgvector is sufficient. Keeps the compose stack simpler — fewer services to understand and debug.

**Alternatives considered**: Qdrant (better at scale, dedicated vector search, but adds a service). pgvector is the simpler choice for a 5-day solo project.

---

## Python Async Strategy

**Decision**: Fully async FastAPI with SQLAlchemy 2.x async engine.

**Rationale**: FastAPI is async-native. SQLAlchemy 2.x async avoids blocking the event loop during DB queries. Redis-py async client. All infra adapters expose async interfaces. This allows the API to handle concurrent requests without thread-pool overhead.

---

## Split Strategy

**Decision**: Stratified by label, test set is strictly more recent in time than train.

**Implementation**: Sort all issues by `closed_at` ascending. Take the most recent 20% as test. Take the next 10% as val. Remaining 70% as train. Apply stratification within each split using sklearn's `StratifiedShuffleSplit` on the label column. Verify no test issue's `closed_at` is earlier than any train issue's `closed_at`.

**Why temporal split**: Prevents data leakage — the model must generalize to issues that did not exist when it was trained, matching real-world deployment conditions.
