"""
Hybrid retrieval — BM25 + dense embeddings fused with RRF, then cross-encoder reranking.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any

from sentence_transformers import CrossEncoder, SentenceTransformer
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.bm25_index import get_index
from app.repositories.issue_repo import get_issues_by_numbers, search_dense

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60  # standard RRF constant


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL)


def embed_query(query: str) -> list[float]:
    model = get_embedder()
    vec = model.encode(query, normalize_embeddings=True)
    return vec.tolist()


async def async_embed_query(query: str) -> list[float]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, embed_query, query)


async def async_rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, rerank, query, candidates, top_k)


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    bm25_results: list[tuple[int, float]],
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """
    Fuse dense and BM25 ranked lists using Reciprocal Rank Fusion.
    Returns issue numbers sorted by fused score descending.
    """
    scores: dict[int, float] = {}

    for rank, item in enumerate(dense_results):
        num = item["number"]
        scores[num] = scores.get(num, 0.0) + 1.0 / (k + rank + 1)

    for rank, (num, _) in enumerate(bm25_results):
        scores[num] = scores.get(num, 0.0) + 1.0 / (k + rank + 1)

    # Merge metadata from dense results
    meta: dict[int, dict] = {r["number"]: r for r in dense_results}

    sorted_numbers = sorted(scores, key=lambda n: scores[n], reverse=True)
    return [
        {**meta.get(n, {"number": n}), "rrf_score": scores[n]}
        for n in sorted_numbers
    ]


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Cross-encoder reranking over candidates."""
    reranker = get_reranker()
    pairs = [(query, f"{c.get('title', '')} {c.get('body', '')[:300]}") for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [
        {**item, "rerank_score": float(score)}
        for score, item in ranked[:top_k]
    ]


async def hybrid_search(
    session: AsyncSession,
    query: str,
    top_k: int = 5,
    pre_k: int = 20,
    source_type: str | None = None,
) -> list[dict[str, Any]]:
    """
    Full hybrid search pipeline:
    1. Embed query
    2. Dense retrieval from pgvector (top pre_k)
    3. BM25 retrieval (top pre_k)
    4. RRF fusion
    5. Fetch any missing metadata
    6. Cross-encoder rerank to top_k
    """
    # Step 1+2: dense
    query_vec = await async_embed_query(query)
    dense_results = await search_dense(session, query_vec, top_k=pre_k, source_type=source_type)

    # Step 3: BM25
    bm25_index = get_index()
    bm25_results = bm25_index.search(query, top_k=pre_k)

    # Step 4: RRF fusion
    fused = reciprocal_rank_fusion(dense_results, bm25_results)[:pre_k]

    # Step 5: fetch metadata for any BM25-only results not in dense
    dense_numbers = {r["number"] for r in dense_results}
    missing = [r["number"] for r in fused if r["number"] not in dense_numbers]
    if missing:
        extra = await get_issues_by_numbers(session, missing)
        extra_map = {r["number"]: r for r in extra}
        fused = [
            {**extra_map[r["number"]], **r} if r["number"] in extra_map else r
            for r in fused
        ]

    # Step 6: rerank (CPU-bound — run in thread pool)
    return await async_rerank(query, fused, top_k=top_k)


async def dense_only_search(
    session: AsyncSession,
    query: str,
    top_k: int = 5,
    source_type: str | None = None,
) -> list[dict[str, Any]]:
    """Naive baseline — pure dense retrieval, no reranking."""
    query_vec = await async_embed_query(query)
    return await search_dense(session, query_vec, top_k=top_k, source_type=source_type)
