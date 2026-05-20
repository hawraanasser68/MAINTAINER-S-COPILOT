# Implementation Plan: Phase 1 — Foundations

**Branch**: `001-foundations` | **Date**: 2026-05-18 | **Spec**: [spec-phase-1-foundations.md](../../.specify/memory/spec-phase-1-foundations.md)

## Summary

Stand up the full Docker Compose stack (10 services), wire Vault as the single secrets source, set up Postgres with Alembic migrations, wire tracing from day one, and produce clean stratified dataset splits from a chosen GitHub repo's closed issues. This phase produces no ML models — only the infrastructure every other phase builds on.

## Technical Context

**Language/Version**: Python 3.11

**Primary Dependencies**: FastAPI, SQLAlchemy 2.x, Alembic, hvac (Vault client), boto3 (MinIO), redis-py, opentelemetry-sdk, PyGitHub (dataset fetch)

**Storage**: Postgres 16 + pgvector (primary), Redis 7 (short-term), MinIO (blob)

**Testing**: pytest, httpx (async API tests)

**Target Platform**: Linux containers (Docker Compose), macOS development

**Project Type**: Multi-service web application

**Performance Goals**: Stack starts in under 5 minutes from a fresh clone. `/health` responds in under 200ms.

**Constraints**: `.env` holds only Vault root token and ports. Zero secrets in `app/` code outside Vault-reading infra adapters.

**Scale/Scope**: Single developer, 5-day build. Dataset: one GitHub repo, ~500–5000 closed issues.

## Constitution Check

| Principle | Status | Notes |
|---|---|---|
| I. Architecture Is The Grade | ✅ | Layer structure defined in project structure below. `app/infra/` holds all adapters. |
| II. Evals Are The Grade | N/A | Eval golden sets created in Phases 2 & 3. `eval_thresholds.yaml` stub created here. |
| III. Secrets In Vault | ✅ | All secrets resolve from Vault. `.env` = Vault token + ports only. Boot check enforced. |
| IV. Observability And Safe Logging | ✅ | Tracing wired in `app/infra/tracing.py` from day one. Redaction layer created here. |
| V. Graceful Error Handling | ✅ | Domain exception hierarchy defined in `app/domain/exceptions.py`. Single handler in `app/api/`. |
| VI. Simplicity, Mastery & Scalability | ✅ | No over-engineering: infra adapters are thin wrappers. Stateless API. Async throughout. |
| VII. Every Decision Backed By A Number | ✅ | Tracing backend choice documented in `DECISIONS.md` before Phase 1 closes. |

## Phase 0: Research & Decisions

**Resolved decisions (must be documented in `DECISIONS.md` before coding starts):**

| Decision | Choice | Rationale |
|---|---|---|
| Tracing backend | OpenTelemetry → Jaeger (local) | Free, Docker-native, OTEL standard means swappable. No vendor lock-in. |
| GitHub repo for dataset | To be chosen Mon morning | Must have 500+ closed, labelled issues. Suggested: `huggingface/transformers`, `fastapi/fastapi`, `langchain-ai/langchain` |
| Label mapping | Defined in `DECISIONS.md` | Map repo-specific labels to bug/feature/docs/question. Document unmapped labels. |
| Long-term memory type | Chosen in Phase 4 | Stub pgvector table created in Phase 1 migration. |
| Vector store | pgvector | Already in Postgres stack. Avoids a separate Qdrant container for now. |

**Output**: `specs/001-foundations/research.md`

## Phase 1: Design & Contracts

### Data Model

**Output**: `specs/001-foundations/data-model.md`

Core tables created in Phase 1 migrations:

| Table | Key Fields | Notes |
|---|---|---|
| `issues` | id, number, title, body, labels, split (train/val/test), label (bug/feature/docs/question), created_at, closed_at | Source dataset |
| `audit_log` | id, actor_id, action, target, timestamp | Append-only, used by all phases |
| `alembic_version` | version_num | Managed by Alembic |

Stub tables (schema only, populated in later phases):

| Table | Phase | Notes |
|---|---|---|
| `users` | Phase 4 | fastapi-users schema |
| `widgets` | Phase 4 | Widget config |
| `long_term_memory` | Phase 4 | pgvector embeddings |
| `conversations` | Phase 4 | Chat sessions |

### Project Structure

```text
maintainers-copilot/
├── app/
│   ├── api/                    # HTTP layer only — routers, request/response models
│   │   ├── __init__.py
│   │   ├── health.py           # GET /health
│   │   └── exceptions.py      # Single exception handler → structured error responses
│   ├── services/               # Business logic, transaction boundaries
│   │   └── __init__.py
│   ├── repositories/           # SQL only — no HTTP, no cache
│   │   └── __init__.py
│   ├── domain/                 # Pydantic domain models + exception hierarchy
│   │   ├── __init__.py
│   │   ├── exceptions.py       # NotFoundError, PermissionDenied, ToolFailure, etc.
│   │   └── models.py
│   └── infra/                  # External system adapters
│       ├── __init__.py
│       ├── vault.py            # hvac wrapper — resolves all secrets
│       ├── database.py         # SQLAlchemy async engine + session factory
│       ├── redis_client.py     # redis-py async wrapper
│       ├── minio_client.py     # boto3 MinIO wrapper
│       ├── tracing.py          # OpenTelemetry SDK setup, span helpers
│       └── redaction.py        # Redaction layer — runs before any log/span/memory write
├── alembic/
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│       └── 0001_initial_schema.py
├── scripts/
│   ├── fetch_issues.py         # GitHub API → raw issues JSONL (huggingface/transformers)
│   └── split_dataset.py        # Stratified splits with temporal ordering
├── notebooks/
│   └── eda.ipynb               # Label distribution, length stats, temporal spread, class imbalance
├── prompts/                    # Version-controlled prompt files (populated Phase 4)
├── tests/
│   ├── test_health.py
│   ├── test_redaction.py       # Asserts fake API key never appears unredacted
│   └── test_splits.py          # Verifies stratification and temporal ordering
├── docker-compose.yml
├── .env.example
├── Dockerfile.api
├── Dockerfile.modelserver
├── Dockerfile.chatbot
├── eval_thresholds.yaml        # Stub — thresholds set to placeholder values, filled Phases 2-3
├── DECISIONS.md
├── ARCH.md
├── SECURITY.md
├── RUNBOOK.md
└── EVALS.md
```

### API Contracts

**Output**: `specs/001-foundations/contracts/`

```
GET /health
Response 200:
{
  "status": "ok",
  "vault": "reachable",
  "db": "reachable",
  "redis": "reachable",
  "tracing": "configured"
}

Response 503 (any dependency down):
{
  "error": { "code": "SERVICE_UNAVAILABLE", "message": "...", "request_id": "..." }
}
```

All error responses follow this envelope (enforced by the single exception handler):
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "request_id": "uuid"
  }
}
```

### Ordered Implementation Steps

Execute strictly in this order — each step unblocks the next:

1. **Repo skeleton** — create directory structure, `pyproject.toml`, `Dockerfile.api`, `.env.example`, empty `DECISIONS.md`, `ARCH.md`, `SECURITY.md`, `RUNBOOK.md`, `EVALS.md`.

2. **Vault adapter** (`app/infra/vault.py`) — `hvac` client, `get_secret(path)` helper, raises `InfrastructureError` if unreachable. Write a unit test with a mock Vault response.

3. **Docker Compose** — all 10 services: `api`, `chatbot`, `modelserver`, `widget`, `host`, `migrate`, `db`, `redis`, `minio`, `vault`. Wire health checks and `depends_on` so `migrate` runs before `api`.

4. **Alembic setup** — `alembic init`, configure async engine in `env.py`, write `0001_initial_schema.py` (issues table + audit_log + pgvector extension). `migrate` container runs `alembic upgrade head` and exits.

5. **Tracing adapter** (`app/infra/tracing.py`) — OpenTelemetry SDK, OTLP exporter → Jaeger. `instrument_app(app)` called at API startup. `create_span(name, attributes)` helper used by every service.

6. **Redaction layer** (`app/infra/redaction.py`) — regex patterns for API keys (`sk-...`), emails, tokens, GitHub PATs. `redact(text) -> text` is called before every log write, span attribute set, and memory write. Patterns documented in `SECURITY.md`.

7. **Domain exceptions** (`app/domain/exceptions.py`) — `AppError` base, `NotFoundError`, `PermissionDenied`, `ToolFailure`, `InfrastructureError`. Single handler in `app/api/exceptions.py` maps each to HTTP status + structured JSON.

8. **Database adapter** (`app/infra/database.py`) — async SQLAlchemy engine, `get_session()` dependency, connection pool config.

9. **Redis adapter** (`app/infra/redis_client.py`) — async redis-py, `get(key)`, `set(key, value, ttl)`, `delete(key)`.

10. **MinIO adapter** (`app/infra/minio_client.py`) — boto3 S3-compatible, `upload(bucket, key, data)`, `download(bucket, key)`, `list(bucket, prefix)`.

11. **API boot checks** — at startup: resolve all secrets from Vault, check DB connectivity, check Redis, check tracing config. Refuse to boot on any failure. Log each check result.

12. **`GET /health` endpoint** — calls all infra adapters, returns structured status. Returns 503 if any dependency is unhealthy.

13. **Dataset fetch script** (`scripts/fetch_issues.py`) — GitHub API via PyGitHub, fetches closed issues with labels, saves to `data/raw_issues.json`. Retries on rate-limit (exponential backoff).

14. **Split script** (`scripts/split_dataset.py`) — applies label mapping, filters unlabelled issues, produces stratified train/val/test splits where all test issues are strictly more recent than all train issues. Saves to `data/splits/`.

15. **EDA notebook** (`notebooks/eda.ipynb`) — after splits are produced: label distribution bar chart, issue body length histogram, temporal spread of issues by month, class imbalance ratio, missing body rate. All findings summarized in a markdown cell at the top. Informs preprocessing and freeze policy decisions in Phase 2.

16. **`eval_thresholds.yaml` stub** — create file with placeholder thresholds. Add boot check: if any threshold is 0 or disabled, refuse to start (thresholds filled in Phase 2 & 3).

17. **Tests** — `test_health.py`, `test_redaction.py` (fake key never in logs/spans/memory), `test_splits.py` (stratification + temporal order).

### Quickstart

**Output**: `specs/001-foundations/quickstart.md`

```bash
git clone <repo>
cd maintainers-copilot
cp .env.example .env
# Fill VAULT_ROOT_TOKEN in .env

docker-compose up --build
# Wait for migrate to exit 0, then api is ready

curl http://localhost:8000/health
# → {"status": "ok", ...}

# Fetch and split dataset
python scripts/fetch_issues.py --repo owner/repo --token $GITHUB_TOKEN
python scripts/split_dataset.py
```

## Complexity Tracking

No constitution violations. All complexity is justified by graded requirements.

---

**Extension Hooks**

**Optional Hook**: git
Command: `/speckit-git-commit`
Description: Commit plan changes after planning is complete.
Prompt: Commit outstanding changes before moving to tasks?
To execute: `/speckit-git-commit`
