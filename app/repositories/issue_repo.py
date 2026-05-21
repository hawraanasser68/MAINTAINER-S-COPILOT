"""
Issue repository — database operations for issues and embeddings.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def upsert_embedding(
    session: AsyncSession,
    issue_number: int,
    embedding: list[float],
    source_type: str = "resolved_issue",
) -> None:
    await session.execute(
        text("""
            UPDATE issues
            SET embedding = CAST(:embedding AS vector),
                source_type = :source_type
            WHERE number = :number
        """),
        {"embedding": str(embedding), "number": issue_number, "source_type": source_type},
    )


async def search_dense(
    session: AsyncSession,
    query_vector: list[float],
    top_k: int = 20,
    source_type: str | None = None,
) -> list[dict[str, Any]]:
    """Return top_k issues by cosine similarity to query_vector."""
    if source_type:
        sql = text("""
            SELECT number, title, body, label, source_type,
                   1 - (embedding <=> CAST(:vec AS vector)) AS score
            FROM issues
            WHERE embedding IS NOT NULL
              AND source_type = :source_type
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT :k
        """)
        params = {"vec": str(query_vector), "k": top_k, "source_type": source_type}
    else:
        sql = text("""
            SELECT number, title, body, label, source_type,
                   1 - (embedding <=> CAST(:vec AS vector)) AS score
            FROM issues
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT :k
        """)
        params = {"vec": str(query_vector), "k": top_k}

    result = await session.execute(sql, params)
    rows = result.mappings().all()
    return [dict(r) for r in rows]


async def get_issues_for_bm25(
    session: AsyncSession,
) -> list[dict[str, Any]]:
    """Return all issues with their text for BM25 index building."""
    result = await session.execute(
        text("SELECT number, title, body, label FROM issues ORDER BY number")
    )
    rows = result.mappings().all()
    return [dict(r) for r in rows]


async def get_issues_by_numbers(
    session: AsyncSession,
    numbers: list[int],
) -> list[dict[str, Any]]:
    """Fetch issue records by their numbers (used after BM25 retrieval)."""
    result = await session.execute(
        text(
            "SELECT number, title, body, label, source_type FROM issues WHERE number = ANY(:nums)"
        ),
        {"nums": numbers},
    )
    rows = result.mappings().all()
    by_number = {r["number"]: dict(r) for r in rows}
    return [by_number[n] for n in numbers if n in by_number]
