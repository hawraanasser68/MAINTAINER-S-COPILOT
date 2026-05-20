"""
Memory tests — short-term TTL behaviour, long-term write + retrieve + audit log.

Short-term tests use a mocked Redis client.
Long-term tests use mocked SQLAlchemy session rows.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Short-term memory
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_short_term_get_empty() -> None:
    """get_history returns [] when key is absent from Redis."""
    with patch("app.infra.redis_client.get", new_callable=AsyncMock, return_value=None):
        from app.services.short_term_memory import get_history
        result = await get_history(uuid.uuid4())
    assert result == []


@pytest.mark.anyio
async def test_short_term_append_and_get() -> None:
    """Appending a message stores it under the correct key with TTL."""
    store: dict[str, str] = {}
    convo_id = uuid.uuid4()

    async def mock_get(key: str) -> str | None:
        return store.get(key)

    async def mock_set(key: str, value: str, ttl: int) -> None:
        assert ttl == 3600
        store[key] = value

    with patch("app.infra.redis_client.get", side_effect=mock_get), \
         patch("app.infra.redis_client.set", side_effect=mock_set):
        from app.services import short_term_memory as stm
        await stm.append_message(convo_id, "user", "Hello")
        history = await stm.get_history(convo_id)

    assert len(history) == 1
    assert history[0] == {"role": "user", "content": "Hello"}


@pytest.mark.anyio
async def test_short_term_append_multiple_turns() -> None:
    """Multiple appends accumulate correctly."""
    store: dict[str, str] = {}
    convo_id = uuid.uuid4()

    async def mock_get(key: str) -> str | None:
        return store.get(key)

    async def mock_set(key: str, value: str, ttl: int) -> None:
        store[key] = value

    with patch("app.infra.redis_client.get", side_effect=mock_get), \
         patch("app.infra.redis_client.set", side_effect=mock_set):
        from app.services import short_term_memory as stm
        await stm.append_message(convo_id, "user", "What is issue #123?")
        await stm.append_message(convo_id, "assistant", "It is a bug.")
        history = await stm.get_history(convo_id)

    assert len(history) == 2
    assert history[1]["role"] == "assistant"


@pytest.mark.anyio
async def test_short_term_clear() -> None:
    """clear() removes the Redis key."""
    deleted: list[str] = []

    async def mock_delete(key: str) -> None:
        deleted.append(key)

    with patch("app.infra.redis_client.delete", side_effect=mock_delete):
        from app.services.short_term_memory import clear
        convo_id = uuid.uuid4()
        await clear(convo_id)

    assert len(deleted) == 1
    assert str(convo_id) in deleted[0]


@pytest.mark.anyio
async def test_short_term_ttl_refreshes_on_append() -> None:
    """Each append resets the TTL to the configured value."""
    ttl_values: list[int] = []
    store: dict[str, str] = {}
    convo_id = uuid.uuid4()

    async def mock_get(key: str) -> str | None:
        return store.get(key)

    async def mock_set(key: str, value: str, ttl: int) -> None:
        store[key] = value
        ttl_values.append(ttl)

    with patch("app.infra.redis_client.get", side_effect=mock_get), \
         patch("app.infra.redis_client.set", side_effect=mock_set):
        from app.services import short_term_memory as stm
        await stm.append_message(convo_id, "user", "msg1", ttl=1800)
        await stm.append_message(convo_id, "user", "msg2", ttl=1800)

    assert all(t == 1800 for t in ttl_values)


# ---------------------------------------------------------------------------
# Long-term memory
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_long_term_write_inserts_and_commits() -> None:
    """write_memory calls session.execute twice (insert + audit) and commits."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()

    with patch("app.services.long_term_memory.embed_query", return_value=[0.1] * 384):
        from app.services.long_term_memory import write_memory
        memory_id = await write_memory(session, "scikit-learn uses semver", user_id=None)

    assert isinstance(memory_id, uuid.UUID)
    assert session.execute.call_count == 2  # insert + audit log
    session.commit.assert_called_once()


@pytest.mark.anyio
async def test_long_term_retrieve_returns_rows() -> None:
    """retrieve_memories returns dicts shaped like DB rows."""
    fake_row = {
        "id": str(uuid.uuid4()),
        "user_id": None,
        "memory_type": "semantic",
        "content": "This repo uses semver",
        "created_at": "2026-05-19T00:00:00",
        "score": 0.91,
    }

    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = [fake_row]

    session = AsyncMock()
    session.execute = AsyncMock(return_value=mock_result)

    with patch("app.services.long_term_memory.embed_query", return_value=[0.1] * 384):
        from app.services.long_term_memory import retrieve_memories
        results = await retrieve_memories(session, "what version scheme?", top_k=1)

    assert len(results) == 1
    assert results[0]["content"] == "This repo uses semver"
    assert results[0]["score"] == pytest.approx(0.91)


@pytest.mark.anyio
async def test_long_term_write_creates_audit_entry() -> None:
    """Audit log row is inserted with action='memory_write' in the SQL."""
    sql_statements: list[str] = []

    async def capture_execute(stmt, params=None, **kwargs):  # type: ignore
        sql_statements.append(str(stmt))
        return MagicMock()

    session = AsyncMock()
    session.execute = capture_execute
    session.commit = AsyncMock()

    with patch("app.services.long_term_memory.embed_query", return_value=[0.0] * 384):
        from app.services.long_term_memory import write_memory
        await write_memory(session, "test memory fact", user_id=uuid.uuid4())

    audit_calls = [s for s in sql_statements if "memory_write" in s]
    assert len(audit_calls) == 1
