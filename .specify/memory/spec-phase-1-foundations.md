# Feature Specification: Phase 1 — Foundations

**Feature Branch**: `phase-1-foundations`

**Created**: 2026-05-18

**Status**: Draft

## User Scenarios & Testing

### User Story 1 — Developer Clones Repo And Starts The Stack (Priority: P1)

A developer clones the repo, copies `.env.example` to `.env`, fills in the Vault root token, and runs `docker-compose up`. All services start, migrations run, and the API is reachable.

**Why this priority**: Everything else depends on this. No service can be developed or tested without the compose stack running.

**Independent Test**: Run `docker-compose up` from a fresh clone and hit `GET /health` on the API — it returns 200.

**Acceptance Scenarios**:

1. **Given** a fresh clone and a valid `.env`, **When** `docker-compose up` is run, **Then** all services (api, chatbot, modelserver, db, redis, minio, vault, migrate, widget, host) start without errors.
2. **Given** the stack is running, **When** `GET /health` is called, **Then** the API returns 200 and confirms Vault, DB, and Redis connectivity.
3. **Given** Vault is unreachable, **When** the API starts, **Then** it refuses to boot and logs a clear error.
4. **Given** a fresh stack, **When** migrations run, **Then** all Alembic migrations apply cleanly and the `migrate` container exits 0 before `api` starts.

---

### User Story 2 — Dataset Is Fetched, Explored, And Split (Priority: P2)

A script fetches closed issues from `huggingface/transformers` via the GitHub API, cleans them, and produces stratified train/val/test splits where test issues are strictly more recent than train. An EDA notebook documents the dataset characteristics before any model is trained.

**Why this priority**: All ML and RAG work depends on the dataset being ready and understood.

**Independent Test**: Run the fetch script and confirm three split files exist with correct size ratios and no test issue predates any train issue. Open the EDA notebook and confirm all cells run without errors.

**Acceptance Scenarios**:

1. **Given** a GitHub token in Vault, **When** the fetch script runs against `huggingface/transformers`, **Then** closed issues are downloaded and stored as JSONL.
2. **Given** raw issues, **When** the split script runs, **Then** train/val/test are stratified by label and test issues are strictly more recent in time than train.
3. **Given** the splits, **When** label mapping is applied, **Then** all issues have one of: `bug`, `feature`, `docs`, `question`.
4. **Given** the splits, **When** `notebooks/eda.ipynb` is run, **Then** it produces: label distribution chart, issue length distribution, temporal spread plot, class imbalance report, and missing body rate.

---

### User Story 3 — Tracing Is Wired From Day One (Priority: P2)

Every service emits traces to the chosen tracing backend from the first commit. A developer can open the tracing UI and see spans for health check calls.

**Why this priority**: Wiring tracing later is much harder. It must be in `app/infra/` and used by every service from the start.

**Independent Test**: Start the stack, call any API endpoint, open the tracing UI, and see at least one span.

**Acceptance Scenarios**:

1. **Given** the stack is running, **When** any API endpoint is called, **Then** a trace appears in the tracing UI with the correct service name.
2. **Given** a trace, **When** inspected, **Then** span attributes include service name and latency.

---

### Edge Cases

- What happens if the GitHub API rate-limits during fetch? Script must retry with backoff.
- What if an issue has no label? It must be filtered out and counted in a skipped-issues log.
- What if Vault is reachable but a required secret is missing? API must refuse to boot with a specific error identifying the missing key.

## Requirements

### Functional Requirements

- **FR-001**: System MUST provide a `docker-compose.yml` with services: `api`, `chatbot`, `modelserver`, `widget`, `host`, `migrate`, `db`, `redis`, `minio`, `vault`.
- **FR-002**: `migrate` container MUST run `alembic upgrade head` and exit 0 before `api` boots.
- **FR-003**: `.env` MUST contain only Vault root token and service ports — no secrets.
- **FR-004**: API MUST refuse to boot if Vault is unreachable.
- **FR-005**: A dataset fetch script MUST download closed issues from a chosen GitHub repo via API.
- **FR-006**: Splits MUST be stratified by label; test set MUST be strictly more recent in time than train.
- **FR-007**: Label mapping (repo labels → bug/feature/docs/question) MUST be defined in `DECISIONS.md`.
- **FR-008**: Tracing MUST be wired in `app/infra/` and active from the first running commit.
- **FR-009**: `app/infra/` MUST contain adapters for: Vault, MinIO, Redis, LLM provider, tracing backend, redaction layer.
- **FR-010**: An audit log table MUST exist for: role changes, memory writes, widget config changes, conversation deletions.

### Key Entities

- **Issue**: raw GitHub issue with title, body, labels, created_at, closed_at, number.
- **Split**: train/val/test assignment for each issue, with label and timestamp.
- **AuditLog**: actor, action, target, timestamp — append-only.

## Success Criteria

- **SC-001**: `docker-compose up` from a fresh clone produces a fully running stack within 5 minutes.
- **SC-002**: `GET /health` returns 200 confirming Vault, DB, Redis connectivity.
- **SC-003**: Dataset splits exist with correct stratification and temporal ordering verified by a test.
- **SC-004**: At least one trace visible in the tracing UI after any API call.
- **SC-005**: `grep -ri 'sk-' app/` and `grep -ri 'password' app/` return zero matches outside Vault-reading code.

## Assumptions

- The chosen open-source GitHub repo has at least 500 closed, labelled issues.
- Vault runs in dev mode — no production HA setup required.
- Alembic migrations are written for every schema change from day one.
- The tracing backend choice is documented in `DECISIONS.md` before Phase 1 is complete.
