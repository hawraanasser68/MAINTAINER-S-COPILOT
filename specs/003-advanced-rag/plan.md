# Implementation Plan: Phase 3 — Advanced RAG

**Branch**: `003-advanced-rag` | **Date**: 2026-05-19 | **Spec**: [spec-phase-3-advanced-rag.md](../../.specify/memory/spec-phase-3-advanced-rag.md)

## Summary

Embed all 5000 issues into pgvector, build a hybrid retrieval pipeline (BM25 + dense + cross-encoder rerank), add HyDE query rewriting, expose a `/rag` endpoint, and validate against a 25-question golden set. Every design choice backed by numbers in `DECISIONS.md`.

## Architecture

```
User query
    │
    ▼
QueryRewriter (HyDE via LLM)
    │
    ├──► BM25 sparse retrieval  ──┐
    │                              ├──► Reciprocal Rank Fusion ──► Cross-encoder rerank ──► top-k chunks
    └──► Dense retrieval (pgvector)┘
                                                                           │
                                                                           ▼
                                                                    LLM answer generation
                                                                           │
                                                                           ▼
                                                                    RAGResult (answer + chunks)
```

## Design Decisions to Defend

| Decision | Choice | Alternative |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` (384-dim, fast, free) | `text-embedding-3-small` (1536-dim, paid) |
| Chunking | whole-issue (title+body, max 512 tokens) | fixed 256-token sliding window |
| Sparse retrieval | BM25 via `rank_bm25` | TF-IDF |
| Fusion | Reciprocal Rank Fusion (RRF) | linear score combination |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | LLM-based reranker |
| Query transform | HyDE (generate hypothetical answer, embed it) | query expansion |

## Task List

### Sub-phase A — Embedding pipeline
- [ ] T001: `pip install sentence-transformers rank-bm25` — add to pyproject.toml
- [ ] T002: `scripts/embed_issues.py` — embed all 5000 issues, store in pgvector
- [ ] T003: Alembic migration — add `embedding vector(384)` column to issues table
- [ ] T004: `app/repositories/issue_repo.py` — `upsert_embedding`, `search_dense(query_vec, top_k, filters)`

### Sub-phase B — BM25 index
- [ ] T005: `app/infra/bm25_index.py` — build BM25 index from all issue texts, persist as pickle in MinIO
- [ ] T006: `scripts/build_bm25.py` — build + upload BM25 index to MinIO

### Sub-phase C — Hybrid retrieval + reranker
- [ ] T007: `app/services/retrieval.py` — `dense_search`, `bm25_search`, `reciprocal_rank_fusion`, `rerank`
- [ ] T008: `app/services/rag.py` — `RAGPipeline`: query → rewrite → retrieve → rerank → generate answer

### Sub-phase D — Query rewriting
- [ ] T009: `app/services/query_rewriter.py` — HyDE: generate hypothetical issue, embed it, use as query

### Sub-phase E — API endpoint
- [ ] T010: `app/api/rag.py` — `POST /rag` with tracing span (query, rewritten_query, chunk_ids, scores, latency)
- [ ] T011: Wire router into `app/main.py`

### Sub-phase F — Golden eval set + metrics
- [ ] T012: `data/golden_rag.jsonl` — 25 hand-curated questions with ground-truth chunk IDs
- [ ] T013: `scripts/eval_rag.py` — compute hit@5, MRR@10, faithfulness, answer relevancy
- [ ] T014: Run naive baseline vs advanced pipeline, record in `DECISIONS.md`

### Sub-phase G — Tests
- [ ] T015: `tests/test_retrieval.py` — unit tests for RRF, reranking, metadata filtering
- [ ] T016: `tests/test_rag_endpoint.py` — integration test for `POST /rag`

## File Layout

```
app/
  api/rag.py
  services/
    rag.py
    retrieval.py
    query_rewriter.py
  repositories/
    issue_repo.py
  infra/
    bm25_index.py

scripts/
  embed_issues.py
  build_bm25.py
  eval_rag.py

data/
  golden_rag.jsonl

alembic/versions/
  0002_add_embedding_column.py
```

## Eval Thresholds (targets)

| Metric | Target |
|---|---|
| hit@5 (advanced) | ≥ 0.72 |
| MRR@10 (advanced) | ≥ 0.60 |
| hit@5 (naive) | recorded as baseline |
| faithfulness | ≥ 0.70 |
| answer relevancy | ≥ 0.70 |
