"""
POST /rag — RAG endpoint with full OpenTelemetry tracing.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.exceptions import InfrastructureError, ToolFailure
from app.infra.database import get_session
from app.infra.tracing import create_span
from app.services.rag import run_rag

router = APIRouter(prefix="/rag", tags=["rag"])


class RAGRequest(BaseModel):
    query: str
    top_k: int = 5
    source_type: str | None = None
    use_hyde: bool = True


class ChunkResult(BaseModel):
    number: int
    title: str
    label: str | None = None
    rerank_score: float | None = None
    rrf_score: float | None = None


class RAGResponse(BaseModel):
    answer: str
    chunks: list[ChunkResult]
    rewritten_query: str
    original_query: str
    model_used: str
    latency_ms: float


@router.post("", response_model=RAGResponse)
async def rag_endpoint(
    req: RAGRequest,
    session: AsyncSession = Depends(get_session),
) -> RAGResponse:
    with create_span(
        "rag.query",
        attributes={
            "rag.original_query": req.query,
            "rag.top_k": req.top_k,
            "rag.use_hyde": req.use_hyde,
            "rag.source_type": req.source_type or "all",
        },
    ) as span:
        try:
            result = await run_rag(
                session=session,
                query=req.query,
                top_k=req.top_k,
                source_type=req.source_type,
                use_hyde=req.use_hyde,
            )

            # Record retrieved chunk IDs and scores in span
            chunk_ids = [str(c.get("number")) for c in result["chunks"]]
            span.set_attribute("rag.rewritten_query", result["rewritten_query"])
            span.set_attribute("rag.chunk_ids", ",".join(chunk_ids))
            span.set_attribute("rag.latency_ms", result["latency_ms"])

            chunks = [
                ChunkResult(
                    number=c["number"],
                    title=c.get("title", ""),
                    label=c.get("label"),
                    rerank_score=c.get("rerank_score"),
                    rrf_score=c.get("rrf_score"),
                )
                for c in result["chunks"]
            ]

            return RAGResponse(
                answer=result["answer"],
                chunks=chunks,
                rewritten_query=result["rewritten_query"],
                original_query=result["original_query"],
                model_used=result["model_used"],
                latency_ms=result["latency_ms"],
            )

        except (InfrastructureError, ToolFailure) as e:
            raise HTTPException(status_code=503, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
