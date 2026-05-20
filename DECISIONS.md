# Decisions

All architectural decisions backed by numbers on the golden set or justified rationale.

## Dataset

**Repo**: `scikit-learn/scikit-learn`

**Why chosen**: Evaluated 5 repos programmatically (fastapi: 583x imbalance, transformers: 1368x imbalance, langchain: 77x imbalance, pandas: missing `question` class). Scikit-learn scored +73.6 — the only positive score. It has all 4 classes, 26.8x imbalance (best of all candidates), 1325 mapped issues at 66.2% coverage, and good temporal spread.

**Final split (5000 issues)**: train=3500, val=500, test=1000. Temporal boundary: train ≤ 2023-07-07, test ≥ 2024-05-07.

**Class distribution (train)**: bug 38%, docs 27%, feature 23%, question 12%. Imbalance ratio: 3.1x — well-balanced.

**Imbalance handling**: Minimal class weighting applied for consistency. Per-class F1 reported for all 4 classes.

**Label Mapping**:

| Repo Label | Mapped Class |
|---|---|
| `bug`, `regression`, `crash`, `type: bug` | `bug` |
| `enhancement`, `feature`, `new feature`, `type: enhancement` | `feature` |
| `documentation`, `docs`, `doc`, `type: docs` | `docs` |
| `question`, `help wanted`, `usage`, `support`, `needs triage` | `question` |
| All others | excluded |

## Tracing Backend

**Choice**: OpenTelemetry SDK → Jaeger (local, Docker)

**Rationale**: OTEL is the open standard — switching exporters later requires only a config change, not a code rewrite. Jaeger runs as a single Docker container with built-in UI. No vendor lock-in.

**Alternatives considered**: Langfuse (LLM-specific, good UI but vendor dependency), stdout exporter (no UI for Friday demo).

## Vector Store

**Choice**: pgvector

**Rationale**: Already in the Postgres stack — avoids a separate Qdrant container. Sufficient for this project's corpus size. HNSW index added in Phase 3.

## Split Strategy

**Choice**: Temporal split — most recent 20% as test, next 10% as val, remaining 70% as train. Stratified by label within each split.

**Rationale**: Prevents data leakage. Model must generalise to issues that did not exist at training time, matching real-world deployment.

## EDA Findings (Phase 1 — 2026-05-19)

Source: `notebooks/eda.ipynb` run against all 5000 issues.

| Finding | Number | Decision |
|---|---|---|
| Imbalance ratio (max/min) | 3.1x | MODERATE → `class_weight="balanced"` in all models |
| Issues exceeding 512 words | 5.3% | Acceptable — tokenizer truncation at `max_length=512` sufficient |
| Empty body rate | 0.7% | Negligible — concat `title + " " + body`, fallback to title when body empty |
| Oversampling needed? | No | 3.1x is below the 10x threshold; class weighting is sufficient |
| Markdown / code blocks | Present in bodies | Keep raw — BERT/DistilBERT tokenizer handles them; note in preprocessing |

**Preprocessing contract** (for all Phase 2 models):
- Input: `f"{title} {body}".strip()` — no markdown stripping
- Tokenizer: `truncation=True, max_length=512, padding="max_length"`
- Class weights: `compute_class_weight("balanced", classes=CLASSES, y=train_labels)`

---

## Phase 2 — LLM Baseline Comparison (2026-05-19)

Evaluated on a fixed 200-issue sample from the test split (seed=42). Same sample used for both runs.

### Zero-shot vs Few-shot (Llama 3.1 8B via Groq)

| Metric | Zero-shot | Few-shot | Delta |
|---|---|---|---|
| Accuracy | 0.7200 | 0.7250 | +0.0050 |
| **Macro-F1** | 0.6112 | **0.6714** | **+0.0602** |
| bug F1 | 0.8000 | 0.8108 | +0.0108 |
| feature F1 | 0.7536 | 0.6575 | −0.0961 |
| docs F1 | 0.7912 | 0.7556 | −0.0356 |
| **question F1** | **0.1000** | **0.4615** | **+0.3615** |

**Key finding**: Few-shot prompting (2 examples per class) improved macro-F1 by +0.06. The entire gain came from the `question` class (+0.36), which zero-shot nearly ignored (recall=0.06). The model defaulted to predicting `bug` when uncertain because `bug` dominates the dataset — examples showed it what `question` issues actually look like.

**Trade-off**: `feature` (−0.10) and `docs` (−0.04) dropped slightly — the longer prompt may have confused the boundary between these classes for some edge cases.

**Conclusion**: Few-shot > zero-shot for this dataset. Use few-shot if deploying an LLM-based classifier.

### Full Comparison

| Model | Macro-F1 | Latency | Eval set | Cost |
|---|---|---|---|---|
| **DistilBERT fine-tuned** | **0.8230** | 107ms (CPU) | 1000 issues | $0 |
| TF-IDF + LogReg | 0.8134 | **0.82ms** | 1000 issues | $0 |
| Llama 3.1 8B few-shot | 0.6714 | 4520ms | 200 issues | $0 (Groq free) |
| Llama 3.1 8B zero-shot | 0.6112 | 4520ms | 200 issues | $0 (Groq free) |
| Llama 3.3 70B few-shot | deferred | — | — | $0 (Groq free tier rate limits impractical) |

### Per-class F1 — DistilBERT (test set, 1000 issues)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| bug | 0.91 | 0.86 | 0.88 |
| feature | 0.85 | 0.87 | 0.86 |
| docs | 0.85 | 0.89 | 0.87 |
| question | 0.66 | 0.70 | 0.68 |

### Deployment Recommendation

**Use TF-IDF + LogReg for production** (macro-F1 0.813, latency 0.82ms) over DistilBERT (0.823, latency 107ms on CPU).

Rationale:
- DistilBERT is only +0.010 macro-F1 better — marginal gain
- 130× slower on CPU (107ms vs 0.82ms per issue)
- TF-IDF is interpretable, has no GPU dependency, and deploys as a single 50MB pickle
- DistilBERT would be preferred if a GPU is available (latency drops to ~5ms) or if the dataset grows significantly
- `question` class is the hardest for all models (F1 0.68–0.70) — label definition overlaps with `feature` and `bug`

---

## Phase 3 — Advanced RAG (2026-05-19)

### Embedding Model

**Choice**: `all-MiniLM-L6-v2` (384-dim, sentence-transformers)

**Rationale**: Free, runs locally, 384-dim is sufficient for semantic similarity on short issue texts. Alternative `text-embedding-3-small` (1536-dim, OpenAI) costs ~$0.02/1K tokens — unnecessary for this corpus size.

### Chunking Strategy

**Choice**: Whole-issue (title + body, truncated to 512 tokens)

**Rationale**: GitHub issues are self-contained units of meaning. Splitting them into fixed-size chunks would break the semantic context. Median issue length is 100 words — well within one chunk.

### Retrieval Pipeline

**Choice**: Hybrid BM25 + dense embeddings fused with Reciprocal Rank Fusion (RRF), then cross-encoder reranking

**Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`

### RAG Eval Results (golden set — 15 examples with ground-truth issue numbers)

| Pipeline | hit@5 | MRR@10 |
|---|---|---|
| Naive dense-only | 0.9333 | 0.8222 |
| **Advanced (hybrid + rerank)** | **1.0000** | **0.8778** |
| Delta | +0.0667 | +0.0556 |

**Finding**: Hybrid retrieval + reranking achieves perfect hit@5 on the golden set. The one failure in naive retrieval (Q22 — IncrementalPCA) was recovered by the BM25 component in the hybrid pipeline, proving that sparse retrieval catches exact-term matches that dense embeddings miss.

### Query Rewriting

**Choice**: HyDE (Hypothetical Document Embeddings) — generate a hypothetical resolved issue, embed it as the query vector instead of the original question.

**Rationale**: Short maintainer questions ("has anyone seen X?") embed differently from issue body text. HyDE bridges this gap by generating text in the same style as the corpus.

---

---

## Phase 4 — Chatbot + Memory (2026-05-19)

### Authentication

**Choice**: fastapi-users 13 with JWT Bearer transport, 8-hour token lifetime

**Rationale**: fastapi-users handles the full auth lifecycle (register, login, password hash, token refresh) without boilerplate. JWT is stateless — no session store needed. 8-hour lifetime matches a maintainer's working day so they aren't constantly re-logging in.

**Alternatives considered**: Session cookies (require sticky sessions or a shared Redis session store — adds complexity); OAuth2 with GitHub (better UX but adds a third-party dependency and OAuth app registration — out of scope for a demo).

### Memory Architecture

| Layer | Backend | TTL / Scope | Why |
|---|---|---|---|
| Short-term | Redis (JSON array) | 3600s per conversation | One working session; active sessions don't expire mid-chat because each append resets the TTL |
| Long-term | pgvector (384-dim HNSW) | Permanent, per-user | Semantic recall across sessions; same embedding model as RAG pipeline (all-MiniLM-L6-v2) avoids a second model load |

**Long-term memory type: Semantic**
- Stores facts about the repo (e.g. "this repo uses semver") and user preferences
- Recalled by embedding cosine similarity — top-3 memories prepended to system prompt
- Simpler than episodic (no event timeline) or procedural (no action sequences)

**Redis TTL: 3600 seconds (1 hour)**
- Maintainers typically resolve issues in one sitting
- Short enough that stale context from a previous session doesn't pollute a new session
- Long enough to survive a browser refresh or a short break

### Tool-calling Agent

**Choice**: Groq llama-3.1-8b-instant with OpenAI-format function-calling, max 5 tool rounds

**Rationale**: Same model already proven in Phase 2 and 3 pipelines — consistent and already rate-limit-tested. 5-round safety cap prevents infinite tool loops while allowing multi-step flows (classify → rag_search → write_memory).

**Tools**: classify, extract_entities, summarize, rag_search, write_memory  
All tool results are appended as `role=tool` messages so the LLM can synthesise a final answer with full context.

### Audit Log

Every `write_memory` call inserts a row into `audit_log` with `action='memory_write'`. This satisfies the compliance requirement: all memory writes are attributable to a user_id and timestamped.
