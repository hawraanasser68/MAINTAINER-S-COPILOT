# Architecture

## Layer Boundaries

```
app/api/          — HTTP only. Routers, request/response schemas. No DB, no Redis, no external calls.
app/services/     — Business logic, transaction boundaries, cache invalidation.
app/repositories/ — SQL only. No HTTP errors, no cache logic.
app/domain/       — Pydantic domain models + exception hierarchy.
app/infra/        — Adapters: Vault, DB, Redis, MinIO, LLM, tracing, redaction.
```

## Services (Docker Compose)

| Service | Port | Purpose |
|---|---|---|
| api | 8000 | FastAPI — auth, chat, memory, RAG, widget config |
| modelserver | 8001 | FastAPI — classifier, NER, summarizer |
| chatbot | 8501 | Streamlit — admin UI, memory inspector, full chat |
| widget | 8080 | Static server — React bundle + loader script |
| host | 8090 | nginx — demo host app |
| migrate | — | Alembic entrypoint, exits 0 |
| db | 5432 | Postgres 16 + pgvector |
| redis | 6379 | Redis 7 — short-term memory + cache |
| minio | 9000 | MinIO — blob storage |
| vault | 8200 | HashiCorp Vault dev mode — secrets |
| jaeger | 16686 | Jaeger — distributed tracing UI |
