# Feature Specification: Phase 4 — Chatbot + Memory

**Feature Branch**: `phase-4-chatbot-memory`

**Created**: 2026-05-18

**Status**: Draft

## User Scenarios & Testing

### User Story 1 — Authenticated Maintainer Chats With The Copilot (Priority: P1)

A maintainer logs in via email and password, opens the Streamlit chat interface, and asks the copilot about a GitHub issue. The copilot classifies it, extracts entities, and answers using RAG — all via a single tool-calling LLM that picks which tools to invoke.

**Why this priority**: The chatbot is the primary deliverable of the project. Auth and tool-calling are the minimum viable surface.

**Independent Test**: Log in with a test user, send an issue URL or paste issue text, and receive a response that includes classification, entities, and a grounded answer. Verify the JWT is required — unauthenticated requests return 401.

**Acceptance Scenarios**:

1. **Given** a registered user, **When** they log in with email and password, **Then** a JWT is issued and stored in session.
2. **Given** a valid JWT, **When** the user sends a message, **Then** the LLM selects and calls the appropriate tools (classify, ner, summarize, rag_search).
3. **Given** an expired or missing JWT, **When** a request hits any protected endpoint, **Then** a 401 is returned with a structured error (no stack trace).
4. **Given** the classifier endpoint is down, **When** the LLM calls the classify tool, **Then** the chatbot catches the ToolFailure and tells the user the classifier is unavailable — it does not 500.

---

### User Story 2 — Conversation State Persists Across Messages (Priority: P1)

Within a session, the chatbot remembers what was discussed earlier. If the maintainer asks "what was the label for the last issue?", the chatbot retrieves it from short-term memory.

**Why this priority**: Short-term memory is a core requirement. Without it, the chatbot cannot maintain context across turns.

**Independent Test**: Send 3 messages in sequence. In message 4, reference something from message 1. Confirm the chatbot answers correctly from Redis-stored context.

**Acceptance Scenarios**:

1. **Given** an ongoing conversation, **When** the user references a prior message, **Then** the chatbot uses the Redis-stored conversation state to answer correctly.
2. **Given** a TTL-expired conversation, **When** the user returns after the TTL, **Then** the session is treated as new and the user is informed.
3. **Given** a conversation, **When** inspected in Redis, **Then** the key has an explicit TTL that is documented and justified in `DECISIONS.md`.

---

### User Story 3 — Long-Term Memory Persists Across Conversations (Priority: P2)

An admin invokes `write_memory` to save a maintainer's preference (e.g. "this repo uses semver"). In a later conversation, the chatbot recalls this preference without being told again.

**Why this priority**: Cross-conversation recall is demonstrated on Friday and is explicitly graded.

**Independent Test**: Write a memory entry via the `write_memory` tool. Start a new session (Redis TTL expired). Ask a question that requires the stored preference. Confirm the chatbot uses it.

**Acceptance Scenarios**:

1. **Given** the `write_memory` tool is called, **When** a memory entry is created, **Then** an audit-log row is written: actor, action, target, timestamp.
2. **Given** a new conversation session, **When** the chatbot retrieves relevant long-term memories, **Then** it uses them to answer without the user repeating context.
3. **Given** an unauthorized user, **When** they attempt to write memory, **Then** a PermissionDenied error is returned.

---

### User Story 4 — Admin Manages Users And Widget Config (Priority: P2)

An admin logs in to the Streamlit app, invites a new user, and configures a widget (theme, greeting, enabled tools, allowed origins). The admin sees the embed snippet for each widget.

**Why this priority**: Admin functionality is required for the widget demo on Friday.

**Independent Test**: Log in as admin, invite a user via the admin page, create a widget config, and verify the embed snippet is generated correctly.

**Acceptance Scenarios**:

1. **Given** an admin user, **When** they invite a new email address, **Then** the user is created and can log in.
2. **Given** an admin, **When** they create a widget config, **Then** a `widget_id` is generated and the embed snippet is shown.
3. **Given** a non-admin user, **When** they attempt to access the admin config page, **Then** they receive a 403 PermissionDenied.

---

### Edge Cases

- What if the LLM returns a tool call for a tool that does not exist? The chatbot must log a ToolFailure and ask the user to rephrase.
- What if long-term memory retrieval returns nothing? The chatbot continues without it — no error shown to the user.
- What if Redis is unavailable mid-conversation? The chatbot must degrade gracefully, logging the infra error with trace ID.
- What if two admins edit the same widget config simultaneously? Last write wins; audit log records both.

## Requirements

### Functional Requirements

- **FR-001**: Authentication MUST use `fastapi-users` with JWT. Email and password registration.
- **FR-002**: JWT signing key MUST resolve from Vault at startup.
- **FR-003**: Two roles MUST exist: `user` and `admin`. Admin can invite users and configure widgets.
- **FR-004**: The chatbot MUST be a single tool-calling LLM — not a workflow or multi-agent system.
- **FR-005**: Tools MUST wrap: classifier (`classify`), NER (`extract_entities`), summarizer (`summarize`), RAG pipeline (`rag_search`), and memory (`write_memory`).
- **FR-006**: `write_memory` MUST be an explicit tool — no auto-writes.
- **FR-007**: Prompts MUST be stored as files in `prompts/` and version-controlled.
- **FR-008**: Short-term conversation state MUST be stored in Redis with an explicit, justified TTL.
- **FR-009**: Long-term memory MUST be stored in Postgres with pgvector. At least one of: episodic, semantic, or procedural. Choice defended in `DECISIONS.md`.
- **FR-010**: Every long-term memory write MUST produce an audit-log row: actor, action, target, timestamp.
- **FR-011**: The Streamlit app MUST include: login page, full chat interface, memory inspector, admin config page.
- **FR-012**: A `widget` table MUST exist in Postgres: `widget_id` (public), `allowed_origins` (list), `theme`, `greeting`, `enabled_tools`.
- **FR-013**: Every LLM call MUST be a span with: model name, token counts, latency, tool inputs/outputs after redaction.

### Key Entities

- **User**: id, email, hashed_password, role (user/admin), created_at.
- **Conversation**: id, user_id, redis_key, created_at, ttl_seconds.
- **Message**: id, conversation_id, role (user/assistant), content, tool_calls, created_at.
- **LongTermMemory**: id, user_id, memory_type, content, embedding, created_at.
- **AuditLog**: id, actor_id, action, target, timestamp.
- **Widget**: id, widget_id (public UUID), allowed_origins, theme, greeting, enabled_tools, created_by.

## Success Criteria

- **SC-001**: Unauthenticated requests to protected endpoints return 401 with a structured error.
- **SC-002**: The LLM correctly selects tools for a given maintainer query (verified in the chatbot eval).
- **SC-003**: Short-term memory correctly carries context across messages within a session.
- **SC-004**: Long-term memory is recalled correctly in a new session after Redis TTL expiry.
- **SC-005**: Every long-term write has a corresponding audit-log row.
- **SC-006**: Admin can create a widget config; non-admin cannot.
- **SC-007**: All LLM calls appear as spans in the tracing UI with token counts and latency.

## Assumptions

- The LLM provider is chosen in Phase 1 and its API key is in Vault.
- Redis TTL is set per session, not globally, and the value is justified in `DECISIONS.md`.
- Long-term memory type (episodic/semantic/procedural) is chosen and defended before Phase 4 begins.
- The Streamlit app and API share the same JWT secret from Vault.
