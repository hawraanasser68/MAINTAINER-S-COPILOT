# Implementation Plan: Phase 4 — Chatbot + Memory

**Branch**: `004-chatbot-memory` | **Date**: 2026-05-19 | **Spec**: [spec-phase-4-chatbot-memory.md](../../.specify/memory/spec-phase-4-chatbot-memory.md)

## Summary

Build the authenticated chatbot — a tool-calling LLM that classifies issues, runs RAG, extracts entities, summarizes threads, and reads/writes memory. Wire short-term (Redis) and long-term (pgvector) memory. Build the Streamlit UI with login, chat, memory inspector, and admin page.

## Memory Type Decision

**Long-term memory type: Semantic**
- Stores facts about the repo/project (e.g. "this repo uses semver", "bug label means regression too")
- Recalled by embedding similarity — retrieve memories relevant to the current query
- Simpler than episodic (no event timeline) or procedural (no action sequences)
- Enough to demonstrate cross-session recall for Friday demo

**Redis TTL: 3600 seconds (1 hour)**
- Maintainers typically resolve issues in one sitting
- Short enough that stale context doesn't pollute future sessions
- Long enough to survive browser refresh or short breaks

## Tool Definitions

| Tool | What it does | When LLM calls it |
|---|---|---|
| `classify` | Classify issue into bug/feature/docs/question | User pastes an issue |
| `extract_entities` | Extract function names, errors, packages, versions | User pastes code/traceback |
| `summarize` | Summarize an issue thread | User asks "what is this issue about?" |
| `rag_search` | Search similar past issues | User asks "has anyone seen this before?" |
| `write_memory` | Save a fact to long-term memory | User says "remember that..." |

## Task List

### Sub-phase A — Database schema
- [ ] T001: Alembic migration `0003_chatbot_tables.py` — users, conversations, messages, long_term_memory, widgets

### Sub-phase B — Auth
- [ ] T002: `app/api/auth.py` — fastapi-users routes: register, login, JWT
- [ ] T003: `app/domain/models.py` — add User, Conversation, Message domain models
- [ ] T004: `app/repositories/models.py` — add ORM models for all new tables

### Sub-phase C — Memory services
- [ ] T005: `app/services/short_term_memory.py` — Redis conversation store: get/append/clear, TTL=3600s
- [ ] T006: `app/services/long_term_memory.py` — pgvector semantic memory: write (+ audit log), retrieve by similarity

### Sub-phase D — Chatbot agent
- [ ] T007: `prompts/system.txt` — system prompt defining the copilot persona and tool usage rules
- [ ] T008: `app/services/tools.py` — tool definitions and HTTP callers for classify/ner/summarize/rag_search
- [ ] T009: `app/services/agent.py` — tool-calling loop: send messages → parse tool calls → execute → loop until done

### Sub-phase E — API endpoints
- [ ] T010: `app/api/chat.py` — `POST /chat` endpoint (auth required), calls agent, stores messages
- [ ] T011: Wire chat router into `app/main.py`

### Sub-phase F — Streamlit UI
- [ ] T012: `chatbot/pages/login.py` — email/password login, stores JWT in session_state
- [ ] T013: `chatbot/pages/chat.py` — chat interface, streams responses, shows tool call badges
- [ ] T014: `chatbot/pages/memory.py` — memory inspector: show short-term (Redis) + long-term (pgvector) entries
- [ ] T015: `chatbot/pages/admin.py` — admin page: invite users, create widget configs
- [ ] T016: `chatbot/main.py` — Streamlit app entry point with page routing

### Sub-phase G — Tests
- [ ] T017: `tests/test_auth.py` — register, login, JWT validation, role enforcement
- [ ] T018: `tests/test_memory.py` — short-term TTL, long-term write + retrieve + audit log
- [ ] T019: `tests/test_chat.py` — tool-calling flow end-to-end

## File Layout

```
app/
  api/
    auth.py
    chat.py
  services/
    short_term_memory.py
    long_term_memory.py
    tools.py
    agent.py
  repositories/
    models.py          (extended)

prompts/
  system.txt

chatbot/
  main.py             (extended)
  pages/
    login.py
    chat.py
    memory.py
    admin.py

alembic/versions/
  0003_chatbot_tables.py
```
