"""
Memory API — list and search long-term memories.

GET  /memory          — list recent memories for current user
GET  /memory/search   — semantic search over memories
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import current_active_user
from app.infra.database import get_session
from app.repositories.models import UserORM
from app.services import long_term_memory as ltm

router = APIRouter(prefix="/memory", tags=["memory"])


class MemoryEntry(BaseModel):
    id: uuid.UUID
    memory_type: str
    content: str
    created_at: str
    score: float | None = None


class MemoryListResponse(BaseModel):
    memories: list[MemoryEntry]


@router.get("", response_model=MemoryListResponse)
async def list_memories(
    limit: int = Query(default=50, ge=1, le=200),
    user: UserORM = Depends(current_active_user),
    session: AsyncSession = Depends(get_session),
) -> MemoryListResponse:
    rows = await ltm.list_memories(session, user_id=user.id, limit=limit)
    return MemoryListResponse(
        memories=[
            MemoryEntry(
                id=r["id"],
                memory_type=r["memory_type"],
                content=r["content"],
                created_at=str(r["created_at"]),
            )
            for r in rows
        ]
    )


@router.get("/search", response_model=MemoryListResponse)
async def search_memories(
    query: str = Query(..., min_length=1),
    top_k: int = Query(default=5, ge=1, le=20),
    user: UserORM = Depends(current_active_user),
    session: AsyncSession = Depends(get_session),
) -> MemoryListResponse:
    rows = await ltm.retrieve_memories(session, query, top_k=top_k, user_id=user.id)
    return MemoryListResponse(
        memories=[
            MemoryEntry(
                id=r["id"],
                memory_type=r["memory_type"],
                content=r["content"],
                created_at=str(r["created_at"]),
                score=float(r.get("score", 0)),
            )
            for r in rows
        ]
    )
