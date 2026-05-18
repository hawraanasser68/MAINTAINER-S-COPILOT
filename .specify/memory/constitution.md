<!--
SYNC IMPACT REPORT
Version change: [UNVERSIONED] → 1.0.0
Added principles: I through VII (all new)
Templates updated: ✅ constitution.md
Deferred: None
-->

# Maintainer's Copilot Constitution

## Core Principles

### I. Architecture Is The Grade
The codebase MUST be layered and each layer's boundary strictly respected:
- `app/api/` — HTTP only; routers MUST NOT touch SQLAlchemy, Redis, or external systems directly.
- `app/services/` — business logic, transaction boundaries, cache and memory invalidation.
- `app/repositories/` — SQL only; MUST NOT raise HTTP errors or invalidate cache.
- `app/domain/` — Pydantic domain models, distinct from SQLAlchemy ORM models.
- `app/infra/` — adapters for Vault, MinIO, Redis, LLM providers, model server, tracing, and redaction.

Layer violations are blocking defects. The boundary will be tested live on Friday.

### II. Evals Are The Grade
Two hand-curated golden sets MUST exist: 25 classification issues and 25 RAG Q/A triples.
- Thresholds committed in `eval_thresholds.yaml`. Zero or disabled threshold = boot-blocking error.
- Both suites MUST run in CI on every push. Regression below threshold MUST block merge.
- `eval_report.json` written every run, stored in MinIO, diffed against previous green build.
- Every architectural decision (embedding model, chunking, retrieval weighting, deployment choice) MUST be backed by a number on the golden set and documented in `DECISIONS.md`.

### III. Secrets In Vault, Never In Code
Every secret MUST resolve from Vault at startup: LLM API keys, tracing keys, JWT signing key, DB password, MinIO credentials.
- `.env` holds ONLY the Vault root token and ports.
- `grep -ri 'sk-' app/` and `grep -ri 'password' app/` MUST return zero matches outside Vault-reading code.
- API MUST refuse to boot if: Vault is unreachable, classifier weights are missing or SHA-256 mismatches, tracing backend is misconfigured, or any eval threshold is zero/disabled.

### IV. Observability And Safe Logging
Every LLM call, tool call, and RAG retrieval MUST be a span. A conversation is a trace tree rooted at the user message.
- Span attributes MUST include: model name, token counts, latency, and tool inputs/outputs after redaction.
- Trace ID MUST be logged alongside every structured log line for the same request.
- A redaction layer MUST run before any log line, trace span, or memory write leaves the service boundary.
- Redaction MUST be explicitly tested: a test asserts a fake API key never appears unredacted in logs, traces, or memory.
- Users MUST never see a stack trace — only a structured error with a code and request ID.

### V. Graceful Error Handling
Tool failures inside the chatbot MUST be caught and recovered. If the classifier is down, the chatbot reports it and falls back — it MUST NOT return HTTP 500.
- Domain exceptions (`NotFoundError`, `PermissionDenied`, `ToolFailure`) MUST be distinct from infrastructure exceptions.
- All domain exceptions map to HTTP responses via a single exception handler.
- Every uncaught exception MUST be logged with trace ID and request ID.

### VI. Simplicity, Mastery, And Scalability
Every abstraction must be justified by a graded requirement — no overengineering.
- Every line shipped must be understood by the author. No black-box copy-paste.
- Best practices applied throughout: type hints, async where it matters, clean interfaces, proper validation at system boundaries only.
- Design for scalability from the start: stateless services, explicit TTLs, clean separation of concerns so components can be swapped independently.
- Complexity not justified by a graded requirement MUST be removed.

### VII. Every Decision Is Backed By A Number
`DECISIONS.md` is a first-class artifact, not an afterthought.
- Embedding model choice, chunking strategy, retrieval weighting, deployment choice — each entry carries a metric from the golden set.
- Label-to-class mapping for the dataset defined here.
- Tracing backend choice defended here.
- Three-way model comparison (accuracy, macro-F1, per-class F1, latency, cost) documented here.

## Technology Stack

| Component | Technology |
|---|---|
| API | FastAPI + fastapi-users (JWT, two roles: user/admin) |
| Chatbot UI | Streamlit (auth, admin config, memory inspector) |
| Widget | React + Vite, single bundled JS served from MinIO or API |
| Model server | FastAPI (classifier, NER, summarizer endpoints) |
| Database | Postgres 16 + pgvector |
| Short-term memory | Redis 7 (explicit, justified TTLs) |
| Long-term memory | Postgres + pgvector (episodic, semantic, or procedural — defend in DECISIONS.md) |
| Blob storage | MinIO |
| Secrets | HashiCorp Vault (dev mode) |
| Migrations | Alembic (migrate container runs before api boots) |
| CI | GitHub Actions: lint → type-check → build → eval suites → redaction test → smoke test |
| Tracing | To be chosen and defended in DECISIONS.md |

## Development Workflow

- `docker-compose up` from a fresh clone after `cp .env.example .env` MUST produce a working stack.
- Prompts live as files in `prompts/`, version-controlled.
- Required docs: `ARCH.md`, `DECISIONS.md`, `RUNBOOK.md`, `EVALS.md`, `SECURITY.md`.
- MinIO holds: model artifacts, `eval_report.json` per CI run, training plots, per-conversation chunk snapshots.
- Submission: public GitHub repo tagged `v0.1.0-week7`.

## Governance

This constitution supersedes all other practices. Amendments require:
1. A rationale referencing a graded requirement or critical defect.
2. Version increment: MAJOR (principle removal/redefinition), MINOR (new principle/section), PATCH (clarification).
3. Propagation check across `plan-template.md`, `spec-template.md`, and `tasks-template.md`.

**Version**: 1.0.0 | **Ratified**: 2026-05-18 | **Last Amended**: 2026-05-18
