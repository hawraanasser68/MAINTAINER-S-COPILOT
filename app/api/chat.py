"""
Chat API — POST /chat, POST /chat/reset, GET /chat/history/{conversation_id}.

Authentication is required on all routes.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Response
from opentelemetry import trace
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import current_active_user
from app.infra.database import get_session
from app.repositories.models import ConversationORM, MessageORM, UserORM
from app.services import agent as chat_agent
from app.services import short_term_memory as stm

router = APIRouter(prefix="/chat", tags=["chat"])
tracer = trace.get_tracer(__name__)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    conversation_id: uuid.UUID | None = None


class ToolCallInfo(BaseModel):
    tool: str
    args: dict
    result: str


class ChatResponse(BaseModel):
    reply: str
    conversation_id: uuid.UUID
    tool_calls_made: list[ToolCallInfo]
    rounds: int


class HistoryMessage(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    conversation_id: uuid.UUID
    messages: list[HistoryMessage]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _get_or_create_conversation(
    session: AsyncSession,
    user: UserORM,
    conversation_id: uuid.UUID | None,
) -> uuid.UUID:
    """Return existing conversation_id or create a new one in the DB."""
    if conversation_id:
        return conversation_id

    new_id = uuid.uuid4()
    redis_key = f"conversation:{new_id}"
    convo = ConversationORM(
        id=new_id,
        user_id=user.id,
        redis_key=redis_key,
        ttl_seconds=3600,
    )
    session.add(convo)
    await session.commit()
    return new_id


async def _persist_messages(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    user_message: str,
    assistant_reply: str,
    tool_calls: list[dict],
) -> None:
    """Write user + assistant messages to the messages table."""
    session.add(MessageORM(
        id=uuid.uuid4(),
        conversation_id=conversation_id,
        role="user",
        content=user_message,
    ))
    session.add(MessageORM(
        id=uuid.uuid4(),
        conversation_id=conversation_id,
        role="assistant",
        content=assistant_reply,
        tool_calls=tool_calls if tool_calls else None,
    ))
    await session.commit()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    user: UserORM = Depends(current_active_user),
    session: AsyncSession = Depends(get_session),
) -> ChatResponse:
    with tracer.start_as_current_span("chat.turn") as span:
        conversation_id = await _get_or_create_conversation(session, user, req.conversation_id)
        span.set_attribute("conversation_id", str(conversation_id))
        span.set_attribute("user_id", str(user.id))

        result = await chat_agent.run_agent(
            conversation_id=conversation_id,
            user_message=req.message,
            session=session,
            user_id=user.id,
        )

        await _persist_messages(
            session,
            conversation_id,
            req.message,
            result["reply"],
            result["tool_calls_made"],
        )

        span.set_attribute("tool_rounds", result["rounds"])

        return ChatResponse(
            reply=result["reply"],
            conversation_id=conversation_id,
            tool_calls_made=[
                ToolCallInfo(tool=tc["tool"], args=tc["args"], result=tc["result"])
                for tc in result["tool_calls_made"]
            ],
            rounds=result["rounds"],
        )


@router.post("/reset")
async def reset_conversation(
    conversation_id: uuid.UUID,
    user: UserORM = Depends(current_active_user),
) -> Response:
    """Clear the Redis history for this conversation."""
    await stm.clear(conversation_id)
    return Response(status_code=204)


@router.get("/history/{conversation_id}", response_model=HistoryResponse)
async def get_history(
    conversation_id: uuid.UUID,
    user: UserORM = Depends(current_active_user),
) -> HistoryResponse:
    """Return the in-memory Redis history (not the persisted DB copy)."""
    messages = await stm.get_history(conversation_id)
    return HistoryResponse(
        conversation_id=conversation_id,
        messages=[HistoryMessage(role=m["role"], content=m["content"]) for m in messages],
    )
