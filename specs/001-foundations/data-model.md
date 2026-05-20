# Data Model: Phase 1 — Foundations

## Core Tables (created in migration 0001)

### `issues`
Stores the raw dataset — one row per GitHub issue.

| Column | Type | Notes |
|---|---|---|
| id | UUID PK | Internal ID |
| number | INTEGER | GitHub issue number |
| title | TEXT NOT NULL | Issue title |
| body | TEXT | Issue body (may be empty) |
| raw_labels | TEXT[] | Original repo labels before mapping |
| label | VARCHAR(20) | Mapped label: bug / feature / docs / question |
| split | VARCHAR(10) | train / val / test |
| created_at | TIMESTAMPTZ | GitHub issue created_at |
| closed_at | TIMESTAMPTZ | GitHub issue closed_at (used for temporal split) |
| repo | VARCHAR(100) | owner/repo |

Indexes: `label`, `split`, `closed_at`.

---

### `audit_log`
Append-only. Every role change, memory write, widget config change, and conversation deletion produces a row.

| Column | Type | Notes |
|---|---|---|
| id | UUID PK | |
| actor_id | UUID | User who performed the action (NULL for system) |
| action | VARCHAR(50) | e.g. MEMORY_WRITE, ROLE_CHANGE, WIDGET_UPDATE, CONVERSATION_DELETE |
| target | TEXT | What was acted on (user_id, memory_id, widget_id, etc.) |
| metadata | JSONB | Additional context |
| timestamp | TIMESTAMPTZ DEFAULT now() | |

No updates or deletes on this table — enforced at the repository layer.

---

## Stub Tables (schema only — populated in later phases)

### `users` (Phase 4)
fastapi-users managed schema. Fields: id, email, hashed_password, is_active, is_superuser, role.

### `widgets` (Phase 4)
Fields: id, widget_id (public UUID), allowed_origins (TEXT[]), theme (JSONB), greeting (TEXT), enabled_tools (TEXT[]), created_by (UUID FK users).

### `conversations` (Phase 4)
Fields: id, user_id, redis_key, created_at.

### `long_term_memory` (Phase 4)
Fields: id, user_id, memory_type, content (TEXT), embedding (vector(1536)), created_at.

---

## Postgres Extensions

- `pgvector` — enabled in migration 0001 via `CREATE EXTENSION IF NOT EXISTS vector`.
- `uuid-ossp` — for UUID generation via `gen_random_uuid()`.
