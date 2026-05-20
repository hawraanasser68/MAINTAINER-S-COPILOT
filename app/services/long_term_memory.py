"""
Long-term memory — semantic facts stored in pgvector.

Each memory is embedded with all-MiniLM-L6-v2 (384-dim) and recalled by
cosine similarity.  Every write is also written to audit_log for compliance.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.retrieval import async_embed_query


async def write_memory(
    session: AsyncSession,
    content: str,
    memory_type: str = "semantic",
    user_id: uuid.UUID | None = None,
) -> uuid.UUID:
    """
    Embed `content`, insert into long_term_memory, write audit log row.
    Returns the new memory's UUID.
    """
    embedding = await async_embed_query(content)
    memory_id = uuid.uuid4()

    await session.execute(
        text("""
            INSERT INTO long_term_memory (id, user_id, memory_type, content, embedding)
            VALUES (:id, :user_id, :memory_type, :content, CAST(:embedding AS vector))
        """),
        {
            "id": str(memory_id),
            "user_id": str(user_id) if user_id else None,
            "memory_type": memory_type,
            "content": content,
            "embedding": str(embedding),
        },
    )

    await session.execute(
        text("""
            INSERT INTO audit_log (id, actor_id, action, target, metadata)
            VALUES (:id, :actor_id, 'memory_write', :target, CAST(:metadata AS json))
        """),
        {
            "id": str(uuid.uuid4()),
            "actor_id": str(user_id) if user_id else None,
            "target": f"long_term_memory:{memory_id}",
            "metadata": f'{{"memory_type": "{memory_type}", "content_len": {len(content)}}}',
        },
    )

    await session.commit()
    return memory_id


async def retrieve_memories(
    session: AsyncSession,
    query: str,
    top_k: int = 5,
    user_id: uuid.UUID | None = None,
) -> list[dict[str, Any]]:
    """
    Return top_k memories semantically similar to `query`.
    Scoped to `user_id` when provided (global memories visible to all when user_id is None).
    """
    query_vec = await async_embed_query(query)

    if user_id:
        sql = text("""
            SELECT id, user_id, memory_type, content, created_at,
                   1 - (embedding <=> CAST(:vec AS vector)) AS score
            FROM long_term_memory
            WHERE embedding IS NOT NULL
              AND (user_id = :user_id OR user_id IS NULL)
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT :k
        """)
        params: dict[str, Any] = {"vec": str(query_vec), "user_id": str(user_id), "k": top_k}
    else:
        sql = text("""
            SELECT id, user_id, memory_type, content, created_at,
                   1 - (embedding <=> CAST(:vec AS vector)) AS score
            FROM long_term_memory
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT :k
        """)
        params = {"vec": str(query_vec), "k": top_k}

    result = await session.execute(sql, params)
    rows = result.mappings().all()
    return [dict(r) for r in rows]


async def list_memories(
    session: AsyncSession,
    user_id: uuid.UUID | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return recent memories for the memory inspector UI, newest first."""
    if user_id:
        sql = text("""
            SELECT id, user_id, memory_type, content, created_at
            FROM long_term_memory
            WHERE user_id = :user_id OR user_id IS NULL
            ORDER BY created_at DESC
            LIMIT :limit
        """)
        params_: dict[str, Any] = {"user_id": str(user_id), "limit": limit}
    else:
        sql = text("""
            SELECT id, user_id, memory_type, content, created_at
            FROM long_term_memory
            ORDER BY created_at DESC
            LIMIT :limit
        """)
        params_ = {"limit": limit}

    result = await session.execute(sql, params_)
    return [dict(r) for r in result.mappings().all()]
