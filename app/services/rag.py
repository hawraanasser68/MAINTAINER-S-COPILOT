"""
RAG pipeline — query rewriting → hybrid retrieval → reranking → answer generation.
"""

from __future__ import annotations

import os
import time
from typing import Any

from groq import Groq
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.bm25_index import get_index
from app.repositories.issue_repo import search_dense
from app.services.query_rewriter import rewrite_query
from app.services.retrieval import embed_query, reciprocal_rank_fusion, rerank

_groq: Groq | None = None

ANSWER_SYSTEM = """You are a helpful assistant for open-source maintainers.
Answer the question using ONLY the provided GitHub issue context.
Be concise (under 150 words). If the context doesn't contain the answer, say so clearly.
Always cite the issue numbers you used (e.g. "See issue #1234")."""


def get_groq() -> Groq:
    global _groq
    if _groq is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        _groq = Groq(api_key=api_key)
    return _groq


def _build_context(chunks: list[dict[str, Any]]) -> str:
    parts = []
    for c in chunks:
        num = c.get("number", "?")
        title = c.get("title", "")
        body = (c.get("body") or "")[:300]
        parts.append(f"Issue #{num}: {title}\n{body}")
    return "\n\n---\n\n".join(parts)


async def run_rag(
    session: AsyncSession,
    query: str,
    top_k: int = 5,
    source_type: str | None = None,
    use_hyde: bool = True,
) -> dict[str, Any]:
    """
    Full RAG pipeline.
    Returns dict with: answer, chunks, rewritten_query, latency_ms.
    """
    t0 = time.perf_counter()

    # Step 1: query rewriting (HyDE)
    if use_hyde:
        rewritten_query, query_vec = rewrite_query(query)
    else:
        rewritten_query = query
        query_vec = embed_query(query)

    # Step 2: hybrid retrieval
    dense_results = await search_dense(session, query_vec, top_k=20, source_type=source_type)

    bm25_index = get_index()
    bm25_results = bm25_index.search(rewritten_query, top_k=20)

    fused = reciprocal_rank_fusion(dense_results, bm25_results)[:20]

    # Step 3: rerank
    chunks = rerank(rewritten_query, fused, top_k=top_k)

    # Step 4: answer generation
    context = _build_context(chunks)
    try:
        groq = get_groq()
        resp = groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=300,
            messages=[
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        answer = f"[Answer generation failed: {e}]"

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "answer": answer,
        "chunks": chunks,
        "rewritten_query": rewritten_query,
        "original_query": query,
        "model_used": "llama-3.1-8b-instant",
        "latency_ms": latency_ms,
    }
