# Tasks: Phase 1 — Foundations

**Input**: Design documents from `specs/001-foundations/`

**Branch**: `001-foundations`

**Total tasks**: 47 | **Parallel opportunities**: 18

---

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel with other [P] tasks in the same phase
- **[US#]**: User story this task belongs to
- Paths are relative to repo root

---

## Phase 1: Setup

**Purpose**: Create the skeleton everything else builds on. No logic yet — just structure, config, and empty files.

- [ ] T001 Create top-level directory structure: `app/api/`, `app/services/`, `app/repositories/`, `app/domain/`, `app/infra/`, `scripts/`, `notebooks/`, `tests/`, `prompts/`, `alembic/versions/`
- [ ] T002 [P] Create `pyproject.toml` with all Phase 1 dependencies: fastapi, uvicorn, sqlalchemy[asyncio], alembic, asyncpg, redis, hvac, boto3, opentelemetry-sdk, opentelemetry-exporter-otlp, pygithub, pandas, scikit-learn, pytest, httpx
- [ ] T003 [P] Create `Dockerfile.api` — Python 3.11 slim, installs from pyproject.toml, runs uvicorn
- [ ] T004 [P] Create `Dockerfile.modelserver` — same base, placeholder entrypoint (populated Phase 2)
- [ ] T005 [P] Create `Dockerfile.chatbot` — same base, placeholder streamlit entrypoint (populated Phase 4)
- [ ] T006 [P] Create `.env.example` with: `VAULT_ROOT_TOKEN=`, `API_PORT=8000`, `DB_PORT=5432`, `REDIS_PORT=6379`, `MINIO_PORT=9000`, `VAULT_PORT=8200`, `JAEGER_PORT=16686`
- [ ] T007 [P] Create empty placeholder docs: `DECISIONS.md`, `ARCH.md`, `SECURITY.md`, `RUNBOOK.md`, `EVALS.md` — each with a single header line
- [ ] T008 [P] Create `eval_thresholds.yaml` stub with commented placeholder structure (filled Phases 2–3)
- [ ] T009 [P] Add `__init__.py` to every `app/` sub-package

**Checkpoint**: Repo structure exists. No logic yet.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: All infra adapters, Docker Compose, and Alembic. MUST be complete before any user story work begins.

**⚠️ CRITICAL**: US1, US2, and US3 all block on this phase.

- [ ] T010 Create `app/domain/exceptions.py` — define `AppError` base class, `NotFoundError`, `PermissionDenied`, `ToolFailure`, `InfrastructureError`. Each carries `message` and `code` fields.
- [ ] T011 Create `app/api/exceptions.py` — single FastAPI exception handler that maps domain exceptions to HTTP status codes and returns `{"error": {"code": ..., "message": ..., "request_id": ...}}`. Users never see stack traces.
- [ ] T012 Create `app/infra/vault.py` — `hvac` client, `get_secret(path: str) -> str` async helper. Raises `InfrastructureError("vault_unreachable")` if Vault is down. Write one unit test with a mocked Vault response in `tests/test_vault.py`.
- [ ] T013 Create `app/infra/redaction.py` — `redact(text: str) -> str`. Patterns (at minimum): `sk-[A-Za-z0-9]{20,}` (API keys), `ghp_[A-Za-z0-9]{36}` (GitHub PATs), email addresses. Document all patterns in `SECURITY.md`. Called before every log write, span attribute, and memory write.
- [ ] T014 Create `app/infra/tracing.py` — OpenTelemetry SDK setup, OTLP exporter to Jaeger. `setup_tracing(service_name: str)` called at startup. `get_tracer()` returns the module tracer. `create_span(name, attributes)` context manager used by every service.
- [ ] T015 Create `app/infra/database.py` — async SQLAlchemy engine from connection URL (resolved from Vault). `async_session_factory`. `get_session()` FastAPI dependency. `check_connectivity() -> bool` used in boot checks.
- [ ] T016 [P] Create `app/infra/redis_client.py` — async redis-py. `get(key)`, `set(key, value, ttl)`, `delete(key)`. `check_connectivity() -> bool`.
- [ ] T017 [P] Create `app/infra/minio_client.py` — boto3 S3-compatible. `upload(bucket, key, data)`, `download(bucket, key)`, `list(bucket, prefix)`. Credentials from Vault.
- [ ] T018 Create `alembic/env.py` — configure async engine, import all ORM models so Alembic detects them. Use `run_async_migrations()` pattern.
- [ ] T019 Create `alembic/versions/0001_initial_schema.py` — migration that: enables `pgvector` extension, creates `issues` table (all fields from data-model.md), creates `audit_log` table, creates stub tables for users/widgets/conversations/long_term_memory (empty, populated Phase 4).
- [ ] T020 Create `app/domain/models.py` — Pydantic models for: `Issue`, `AuditLogEntry`, `HealthStatus`, `ErrorResponse`. These are domain models, NOT SQLAlchemy ORM models.
- [ ] T021 Create SQLAlchemy ORM models in `app/repositories/models.py` — `IssueORM`, `AuditLogORM` mapped to the tables in migration 0001.
- [ ] T022 Create `docker-compose.yml` with all 10 services: `api`, `chatbot`, `modelserver`, `widget`, `host`, `migrate`, `db` (postgres:16 + pgvector), `redis` (redis:7), `minio` (minio/minio), `vault` (hashicorp/vault dev mode), `jaeger` (jaegertracing/all-in-one). Wire `depends_on` so `migrate` exits before `api` starts.
- [ ] T023 Add health checks to `docker-compose.yml` for: `db` (pg_isready), `redis` (redis-cli ping), `minio` (curl /minio/health/live), `vault` (vault status).

**Checkpoint**: `docker-compose up` builds. Migrate container runs and exits. All infra adapters exist. No API routes yet.

---

## Phase 3: User Story 1 — Stack Starts And Health Passes (P1) 🎯 MVP

**Goal**: `docker-compose up` from a fresh clone → stack runs → `GET /health` returns 200 confirming all dependencies.

**Independent Test**: `curl http://localhost:8000/health` returns `{"status": "ok", "vault": "reachable", "db": "reachable", "redis": "reachable", "tracing": "configured"}`.

- [ ] T024 [US1] Create `app/main.py` — FastAPI app factory: call `setup_tracing()`, resolve all secrets from Vault via `app/infra/vault.py`, run boot checks (Vault reachable, DB connectable, Redis connectable, tracing configured, no eval threshold is zero/disabled in `eval_thresholds.yaml`). Refuse to start if any check fails — log the specific failure with `InfrastructureError`.
- [ ] T025 [US1] Create `app/api/health.py` — `GET /health` router. Calls each infra adapter's `check_connectivity()`. Returns `HealthStatus` domain model. Returns 503 with structured error if any dependency is unhealthy.
- [ ] T026 [US1] Register health router and exception handler in `app/main.py`. Mount at `/health`.
- [ ] T027 [US1] Write `tests/test_health.py` — test 200 response when all deps healthy (mock infra adapters), test 503 when one dep is down, test that response shape matches `ErrorResponse` schema on failure.
- [ ] T028 [US1] Verify `docker-compose up` starts cleanly: run `docker-compose up -d`, wait for `migrate` to exit 0, then `curl localhost:8000/health`. Fix any compose wiring issues.

**Checkpoint**: Stack starts from a fresh clone. Health endpoint returns 200. Vault boot-check blocks startup if Vault is down (demonstrate by stopping Vault container and restarting api).

---

## Phase 4: User Story 2 — Dataset Fetched, Explored, And Split (P2)

**Goal**: `huggingface/transformers` closed issues downloaded, label-mapped, split into stratified train/val/test with temporal ordering, and explored in an EDA notebook.

**Independent Test**: `python scripts/fetch_issues.py` produces `data/raw_issues.jsonl`. `python scripts/split_dataset.py` produces three split files. `data/splits/split_stats.json` shows stratified label counts. No test issue's `closed_at` is earlier than any train issue's.

- [ ] T029 [US2] Create `scripts/fetch_issues.py` — uses PyGitHub, fetches all closed issues from `huggingface/transformers` with labels. Filters out pull requests. Saves to `data/raw_issues.jsonl` (one JSON object per line). Handles GitHub API rate limiting with exponential backoff. GitHub token read from Vault (path `secret/github/token`).
- [ ] T030 [US2] Define and document label mapping in `DECISIONS.md`: repo labels → bug/feature/docs/question. Log count of unmapped/excluded issues.
- [ ] T031 [US2] Create `scripts/split_dataset.py` — loads `data/raw_issues.jsonl`, applies label mapping, drops unmapped issues, sorts by `closed_at` ascending, takes most recent 20% as test, next 10% as val, remaining 70% as train. Applies stratification within splits using `StratifiedShuffleSplit`. Saves: `data/splits/train.jsonl`, `data/splits/val.jsonl`, `data/splits/test.jsonl`, `data/splits/split_stats.json`.
- [ ] T032 [US2] Write `tests/test_splits.py` — assert: no test issue's `closed_at` is earlier than any train issue's `closed_at`, label distribution is stratified (each class present in each split), no issue appears in more than one split.
- [ ] T033 [US2] Create `notebooks/eda.ipynb` with cells in this order: (1) markdown summary cell, (2) load splits, (3) label distribution bar chart, (4) issue body length histogram (in tokens), (5) temporal spread plot (issues by month), (6) class imbalance ratio table, (7) missing/empty body rate, (8) 3 sample issues per class. Save all plots to `notebooks/figures/`.

**Checkpoint**: All three split files exist. EDA notebook runs end-to-end without errors. `test_splits.py` passes. Label mapping documented in `DECISIONS.md`.

---

## Phase 5: User Story 3 — Tracing Wired And Visible (P2)

**Goal**: Every API call produces a trace in Jaeger. The trace ID appears in every log line for that request.

**Independent Test**: Call `GET /health`, open `http://localhost:16686`, select service `maintainers-copilot-api`, see at least one span with correct attributes.

- [ ] T034 [US3] Add structured logging to `app/main.py` — use Python `logging` with JSON formatter. Every log record includes `trace_id` (from OpenTelemetry context) and `request_id` (UUID generated per request via middleware).
- [ ] T035 [US3] Create `app/api/middleware.py` — `RequestIDMiddleware` that generates a UUID per request, attaches it to the request state, and includes it in every log line and response header (`X-Request-ID`).
- [ ] T036 [US3] Instrument `GET /health` with a span: wrap the health check logic in `create_span("health.check", {"service": "api"})`. Add span attributes for each dependency check result.
- [ ] T037 [US3] Add redaction to all span attributes — call `redact()` on any string value before setting it as a span attribute. Enforce this in `create_span()` so it's automatic for all callers.
- [ ] T038 [US3] Write `tests/test_redaction.py` — assert that a request body containing `sk-fake123456789012345` never appears unredacted in: (a) log output captured during the request, (b) span attributes (use an in-memory OTLP exporter for testing), (c) any string returned by `redact()`. This test MUST pass before Phase 6 CI is wired.

**Checkpoint**: Jaeger UI shows spans for `/health` calls. Log lines contain `trace_id` and `request_id`. Redaction test passes.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Wire everything together, validate the full quickstart, and commit.

- [ ] T039 [P] Fill `ARCH.md` — describe the layered architecture: which layer owns what, why the boundaries exist, diagram (ASCII is fine).
- [ ] T040 [P] Fill `RUNBOOK.md` — how to start the stack, how to run migrations manually, how to fetch/split the dataset, how to run tests, how to access Jaeger and MinIO UIs.
- [ ] T041 [P] Fill `SECURITY.md` — list all redaction patterns with rationale for each. Explain where redaction is called (logs, spans, memory writes).
- [ ] T042 Add boot-check for `eval_thresholds.yaml` — if any threshold is 0 or the file is missing, `app/main.py` raises `InfrastructureError("eval_thresholds_invalid")` at startup. Add note in `eval_thresholds.yaml` that thresholds must be filled before Phase 6 CI is enabled.
- [ ] T043 [P] Add `.gitignore` — exclude: `data/`, `__pycache__/`, `.env`, `*.pyc`, `notebooks/.ipynb_checkpoints/`, `*.egg-info/`
- [ ] T044 Run full quickstart validation per `specs/001-foundations/quickstart.md`: fresh clone simulation, `docker-compose up`, health check, dataset fetch + split, EDA notebook, all tests.
- [ ] T045 [P] `grep -ri 'sk-' app/` — must return zero matches outside `app/infra/vault.py`. Fix any leaks found.
- [ ] T046 [P] `grep -ri 'password' app/` — must return zero matches outside `app/infra/vault.py`. Fix any leaks found.
- [ ] T047 Commit all Phase 1 work: `git add` only tracked files (no `data/` or `.env`), commit with message `feat: phase 1 foundations — stack, infra adapters, dataset, tracing`.

**Checkpoint**: All 47 tasks complete. Phase 1 done criteria from quickstart.md all checked off.

---

## Dependencies & Execution Order

- **Phase 1 (Setup)**: No dependencies — start immediately. All [P] tasks run in parallel.
- **Phase 2 (Foundational)**: Requires Phase 1 complete. T010–T011 first (exceptions needed by adapters). T012–T017 can mostly parallel after T010. T018–T019 require T015 (DB engine). T020–T021 can parallel with T018–T019. T022–T023 (Docker Compose) require all adapters to exist.
- **Phase 3 (US1)**: Requires Phase 2 complete. T024–T026 sequential. T027–T028 after T026.
- **Phase 4 (US2)**: Requires Phase 2 complete. T029–T032 can start in parallel with Phase 3. T033 (EDA) requires T031 (splits) complete.
- **Phase 5 (US3)**: Requires Phase 3 complete (logging middleware added to the running app).
- **Phase 6 (Polish)**: Requires Phases 3, 4, 5 complete.

### Parallel Opportunities Within Phase 2

```
T010 (exceptions) → T012 (vault), T013 (redaction), T014 (tracing), T015 (db), T016 (redis), T017 (minio)  [all parallel after T010]
T015 (db) → T018 (alembic env) → T019 (migration)
T020 (domain models) and T021 (ORM models) parallel with T018
T022 (compose) after T012–T021 all exist
T023 (health checks) parallel with T022
```

---

## Implementation Strategy

### MVP (Phase 1 + 2 + 3 only)

1. Complete Phase 1: Setup (~1 hour)
2. Complete Phase 2: Foundational (~3 hours)
3. Complete Phase 3: US1 — stack starts, health passes (~1 hour)
4. **STOP and validate**: `docker-compose up` + `curl /health` + Vault boot-check demo

### Full Phase 1

After MVP is validated, add Phase 4 (dataset) and Phase 5 (tracing verification) in parallel, then Polish.
