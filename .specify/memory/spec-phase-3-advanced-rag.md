# Feature Specification: Phase 3 — Advanced RAG

**Feature Branch**: `phase-3-advanced-rag`

**Created**: 2026-05-18

**Status**: Draft

## User Scenarios & Testing

### User Story 1 — Maintainer Asks A Question And Gets A Grounded Answer (Priority: P1)

A maintainer asks: "How does the project handle X?" The RAG pipeline retrieves the most relevant chunks from the docs and resolved issues, reranks them, and passes them to the LLM to produce a grounded answer with source references.

**Why this priority**: RAG is the core of the chatbot's knowledge. Without it, the chatbot cannot answer maintainer questions accurately.

**Independent Test**: POST a question to the RAG endpoint and receive an answer with retrieved chunks. Run the 25-question golden set and confirm hit@5 and faithfulness meet committed thresholds.

**Acceptance Scenarios**:

1. **Given** a maintainer question, **When** the RAG pipeline runs, **Then** the response contains an answer and the ground-truth chunk in the top-5 retrieved results.
2. **Given** the 25-question golden set, **When** the RAG eval suite runs, **Then** hit@5, MRR@10, faithfulness, and answer relevancy all meet thresholds in `eval_thresholds.yaml`.
3. **Given** a query with metadata filters (e.g. "docs only"), **When** retrieval runs, **Then** only chunks matching the filter are returned.

---

### User Story 2 — Hybrid Retrieval Outperforms Naive Dense Retrieval (Priority: P1)

The hybrid retrieval (sparse BM25 + dense embeddings, tuned weighting) produces higher hit@5 than naive fixed-size chunking + pure dense retrieval on the golden set. The improvement is documented in `DECISIONS.md`.

**Why this priority**: Every RAG design choice must be justified with a number. The baseline comparison is a graded requirement.

**Independent Test**: Run both the naive baseline and the advanced pipeline against the golden set and compare hit@5 and MRR@10.

**Acceptance Scenarios**:

1. **Given** the golden set, **When** naive (fixed-size + pure dense) runs, **Then** hit@5 and MRR@10 are recorded as the baseline.
2. **Given** the golden set, **When** the advanced pipeline runs (smart chunking + hybrid + rerank), **Then** hit@5 and MRR@10 are higher than baseline.
3. **Given** the comparison, **When** `DECISIONS.md` is read, **Then** numbers for both pipelines are present with a rationale for every design choice.

---

### User Story 3 — Query Rewriting Improves Retrieval For Ambiguous Questions (Priority: P2)

When a maintainer asks a vague or ambiguous question, the pipeline rewrites or expands the query before retrieval, producing better results than the original query alone.

**Why this priority**: Query transformation is required by the spec and improves retrieval quality on edge cases.

**Independent Test**: Submit an ambiguous query, inspect the rewritten version, and confirm retrieved chunks are more relevant than with the original query.

**Acceptance Scenarios**:

1. **Given** an ambiguous query, **When** the query transformation step runs, **Then** a rewritten or expanded query is produced and logged as a span attribute.
2. **Given** the rewritten query, **When** retrieval runs, **Then** hit@5 is equal to or higher than with the original query on the test subset.

---

### Edge Cases

- What if a query matches no chunks above a relevance threshold? Return an empty result with a clear message rather than hallucinating.
- What if the embedding model is unavailable? The RAG endpoint must catch the error and return a ToolFailure, not a 500.
- What if a held-out resolved issue appears in the classifier training set? This is a data contamination bug — the split script must prevent it.
- What if reranking takes too long? Log the latency as a span; if it exceeds a threshold, fall back to hybrid-only ranking.

## Requirements

### Functional Requirements

- **FR-001**: The RAG corpus MUST consist of the project's docs plus a held-out slice of resolved issues with maintainer answers. Held-out issues MUST NOT appear in classifier training.
- **FR-002**: A chunking strategy that is NOT naive fixed-size MUST be implemented (e.g. sentence-aware, semantic, markdown-header-based). Choice defended in `DECISIONS.md`.
- **FR-003**: An embedding model MUST be chosen and justified by retrieval-quality numbers against at least one alternative on the golden set. Documented in `DECISIONS.md`.
- **FR-004**: Hybrid retrieval MUST combine sparse (BM25) and dense (embedding) retrieval with a tuned weighting. Weighting choice defended in `DECISIONS.md`.
- **FR-005**: Cross-encoder reranking MUST be applied over the top-k results from hybrid retrieval.
- **FR-006**: At least one query transformation technique MUST be implemented (e.g. HyDE, query expansion, step-back prompting).
- **FR-007**: Metadata filtering MUST be supported (e.g. filter by source type: docs vs. resolved issues).
- **FR-008**: The vector store MUST use pgvector or Qdrant.
- **FR-009**: A 25-question golden set MUST be hand-curated: each entry has a question, ideal answer, and ground-truth chunks.
- **FR-010**: RAG eval MUST report: hit@5, MRR@10, faithfulness, answer relevancy. RAGAS or a frozen judge model — choice defended.
- **FR-011**: 5 of the 25 golden examples MUST be hand-labelled by the developer. Agreement with the judge model must be reported.
- **FR-012**: Per-conversation retrieved-chunk snapshots MUST be stored in MinIO for the last N conversations.
- **FR-013**: Every RAG retrieval MUST be a span with attributes: query, rewritten query, top-k chunk IDs, reranking scores, latency.

### Key Entities

- **Chunk**: id, text, source_type (docs/resolved_issue), metadata (title, url, date), embedding vector.
- **RAGQuery**: original_query, rewritten_query, filters, top_k.
- **RAGResult**: answer, retrieved_chunks (list of Chunk with scores), model_used, latency_ms.
- **GoldenRAGExample**: question, ideal_answer, ground_truth_chunk_ids.

## Success Criteria

- **SC-001**: hit@5 and MRR@10 for the advanced pipeline exceed the naive baseline on the golden set.
- **SC-002**: Faithfulness and answer relevancy meet committed thresholds in `eval_thresholds.yaml`.
- **SC-003**: `DECISIONS.md` contains embedding model comparison numbers and chunking strategy rationale.
- **SC-004**: Metadata filtering correctly restricts results to the specified source type.
- **SC-005**: Every retrieval appears as a span in the tracing UI.
- **SC-006**: 5 hand-labelled examples exist with reported agreement vs. the judge model.

## Assumptions

- The vector store (pgvector or Qdrant) is chosen in Phase 1 and ready before Phase 3 begins.
- The embedding model runs via API (not locally) to keep the compose stack lean.
- The judge model for RAG eval is frozen (pinned version) so CI results are reproducible.
- The held-out resolved issues slice is defined and separated from the classifier splits before Phase 2 training begins.
