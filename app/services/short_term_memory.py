"""
Short-term memory — per-conversation turn history stored in Redis.

Key schema: conversation:{conversation_id}
Value: JSON array of {"role": ..., "content": ...} dicts
TTL: refreshed on every append (default 3600s = 1 hour)
"""

from __future__ import annotations

import json
import uuid

from app.infra import redis_client

_DEFAULT_TTL = 3600


def _key(conversation_id: str | uuid.UUID) -> str:
    return f"conversation:{conversation_id}"


async def get_history(conversation_id: str | uuid.UUID) -> list[dict]:
    """Return message list for this conversation, or [] if expired / not found."""
    raw = await redis_client.get(_key(conversation_id))
    if raw is None:
        return []
    return json.loads(raw)


async def append_message(
    conversation_id: str | uuid.UUID,
    role: str,
    content: str,
    ttl: int = _DEFAULT_TTL,
) -> None:
    """Append one message and reset the TTL so active sessions don't expire mid-chat."""
    history = await get_history(conversation_id)
    history.append({"role": role, "content": content})
    await redis_client.set(_key(conversation_id), json.dumps(history), ttl=ttl)


async def set_history(
    conversation_id: str | uuid.UUID,
    messages: list[dict],
    ttl: int = _DEFAULT_TTL,
) -> None:
    """Overwrite the full history (used when tool-call messages need to be stored together)."""
    await redis_client.set(_key(conversation_id), json.dumps(messages), ttl=ttl)


async def clear(conversation_id: str | uuid.UUID) -> None:
    """Delete the conversation from Redis (used on explicit logout or reset)."""
    await redis_client.delete(_key(conversation_id))
