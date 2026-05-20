"""
Chat / agent tests — tool-calling flow end-to-end (mocked LLM + tools).
"""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — fake Groq response objects
# ---------------------------------------------------------------------------

def _make_groq_text_response(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_groq_tool_response(tool_name: str, args: dict, tool_call_id: str = "tc1") -> MagicMock:
    fn = MagicMock()
    fn.name = tool_name
    fn.arguments = json.dumps(args)

    tc = MagicMock()
    tc.id = tool_call_id
    tc.function = fn

    msg = MagicMock()
    msg.content = ""
    msg.tool_calls = [tc]

    choice = MagicMock()
    choice.message = msg

    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# Agent loop — no tool calls
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_agent_plain_answer() -> None:
    """Agent returns the LLM text directly when no tools are called."""
    convo_id = uuid.uuid4()

    groq_mock = MagicMock()
    groq_mock.chat.completions.create.return_value = _make_groq_text_response("This is a plain answer.")

    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock(
        mappings=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
    ))

    with patch("app.services.agent._get_groq", return_value=groq_mock), \
         patch("app.services.short_term_memory.get_history", new_callable=AsyncMock, return_value=[]), \
         patch("app.services.short_term_memory.set_history", new_callable=AsyncMock), \
         patch("app.services.long_term_memory.retrieve_memories", new_callable=AsyncMock, return_value=[]):

        from app.services.agent import run_agent
        result = await run_agent(convo_id, "Hello!", session)

    assert result["reply"] == "This is a plain answer."
    assert result["tool_calls_made"] == []
    assert result["rounds"] == 1


# ---------------------------------------------------------------------------
# Agent loop — one tool call then text
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_agent_single_tool_call() -> None:
    """Agent calls classify, gets result, then returns final answer."""
    convo_id = uuid.uuid4()
    call_count = 0

    groq_mock = MagicMock()

    def side_effect(**kwargs):  # type: ignore
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_groq_tool_response("classify", {"title": "KeyError in fit()"})
        return _make_groq_text_response("This is a bug.")

    groq_mock.chat.completions.create.side_effect = side_effect

    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock(
        mappings=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
    ))

    with patch("app.services.agent._get_groq", return_value=groq_mock), \
         patch("app.services.short_term_memory.get_history", new_callable=AsyncMock, return_value=[]), \
         patch("app.services.short_term_memory.set_history", new_callable=AsyncMock), \
         patch("app.services.long_term_memory.retrieve_memories", new_callable=AsyncMock, return_value=[]), \
         patch("app.services.tools.execute_tool",
               new_callable=AsyncMock,
               return_value="Classification: **bug** (confidence: 92.0%)"):

        from app.services.agent import run_agent
        result = await run_agent(convo_id, "Classify this issue: KeyError in fit()", session)

    assert result["reply"] == "This is a bug."
    assert len(result["tool_calls_made"]) == 1
    assert result["tool_calls_made"][0]["tool"] == "classify"
    assert result["rounds"] == 2


# ---------------------------------------------------------------------------
# Agent loop — safety limit respected
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_agent_safety_limit() -> None:
    """Agent stops after _MAX_TOOL_ROUNDS and generates a final answer."""
    convo_id = uuid.uuid4()

    groq_mock = MagicMock()
    groq_mock.chat.completions.create.return_value = _make_groq_tool_response(
        "rag_search", {"query": "infinite loop test"}
    )

    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock(
        mappings=MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))
    ))

    with patch("app.services.agent._get_groq", return_value=groq_mock), \
         patch("app.services.agent._MAX_TOOL_ROUNDS", 3), \
         patch("app.services.short_term_memory.get_history", new_callable=AsyncMock, return_value=[]), \
         patch("app.services.short_term_memory.set_history", new_callable=AsyncMock), \
         patch("app.services.long_term_memory.retrieve_memories", new_callable=AsyncMock, return_value=[]), \
         patch("app.services.tools.execute_tool",
               new_callable=AsyncMock,
               return_value="Search result"):

        from app.services.agent import run_agent
        result = await run_agent(convo_id, "trigger infinite tool loop", session)

    # Should not raise; tool_calls should be capped at MAX_TOOL_ROUNDS
    assert len(result["tool_calls_made"]) == 3


# ---------------------------------------------------------------------------
# Tool executor — classify maps to model-server
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_tool_execute_classify() -> None:
    """execute_tool('classify', ...) calls POST /classify and formats result."""
    session = AsyncMock()

    with patch(
        "app.services.tools._post",
        new_callable=AsyncMock,
        return_value={"label": "bug", "confidence": 0.95},
    ):
        from app.services.tools import execute_tool
        result = await execute_tool("classify", {"title": "KeyError"}, session)

    assert "bug" in result
    assert "95.0%" in result


# ---------------------------------------------------------------------------
# Tool executor — write_memory writes to LTM
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_tool_execute_write_memory() -> None:
    """execute_tool('write_memory', ...) calls ltm.write_memory and confirms save."""
    session = AsyncMock()
    memory_id = uuid.uuid4()

    with patch(
        "app.services.long_term_memory.write_memory",
        new_callable=AsyncMock,
        return_value=memory_id,
    ):
        from app.services.tools import execute_tool
        result = await execute_tool(
            "write_memory",
            {"content": "This repo uses semver", "memory_type": "semantic"},
            session,
        )

    assert "Memory saved" in result
    assert str(memory_id) in result
