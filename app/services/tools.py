"""
Tool definitions and HTTP callers for the chatbot agent.

Each tool is described in the OpenAI/Groq function-calling schema format,
and has a corresponding async executor that calls the model-server or
internal services.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.services import long_term_memory as ltm
from app.services.rag import run_rag

_MODELSERVER_URL = os.environ.get("MODELSERVER_URL", "http://modelserver:8001")
_HTTP_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "classify",
            "description": "Classify a GitHub issue into bug, feature, docs, or question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Issue title"},
                    "body": {"type": "string", "description": "Issue body (optional)"},
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_entities",
            "description": (
                "Pass raw issue text to this tool and it will return the technical "
                "entities found in it: function names, error codes, package names, "
                "and version numbers. Input is the raw text string; the tool returns the entities."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The raw issue text to scan for entities",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize",
            "description": "Summarize a GitHub issue thread in under 100 words.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Issue text to summarize"}
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": (
                "Search the corpus of 5000 scikit-learn GitHub issues for past issues "
                "similar to the current query. Always call this when the user asks about "
                "past issues, errors, bugs, or anything that might have been reported before."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_memory",
            "description": (
                "Save a fact to long-term memory so it is recalled in future sessions. "
                "Only call this when the user explicitly asks to remember something."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Fact to remember"},
                    "memory_type": {
                        "type": "string",
                        "description": "Memory type (always 'semantic')",
                        "default": "semantic",
                    },
                },
                "required": ["content"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool executors
# ---------------------------------------------------------------------------

async def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST to model-server and return JSON response."""
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        resp = await client.post(f"{_MODELSERVER_URL}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()


async def execute_tool(
    tool_name: str,
    arguments: dict[str, Any],
    session: AsyncSession,
    user_id: uuid.UUID | None = None,
) -> str:
    """
    Dispatch to the correct executor and return a string result for the
    assistant context window.
    """
    try:
        if tool_name == "classify":
            result = await _post("/classify", arguments)
            label = result.get("label", "unknown")
            confidence = result.get("confidence", 0)
            return f"Classification: **{label}** (confidence: {confidence:.1%})"

        elif tool_name == "extract_entities":
            result = await _post("/ner", arguments)
            entities = result.get("entities", [])
            if not entities:
                return "No entities found."
            lines = [f"- {e['type']}: `{e['text']}`" for e in entities]
            return "Entities found:\n" + "\n".join(lines)

        elif tool_name == "summarize":
            result = await _post("/summarize", arguments)
            return result.get("summary", "[No summary returned]")

        elif tool_name == "rag_search":
            query = arguments.get("query", "")
            top_k = int(arguments.get("top_k", 5))
            rag_result = await run_rag(session, query, top_k=top_k, use_hyde=True)
            chunks = rag_result.get("chunks", [])
            if not chunks:
                return "No similar issues found."
            lines = []
            for c in chunks:
                num = c.get("number", "?")
                title = c.get("title", "")
                score = c.get("score", 0)
                lines.append(f"- Issue #{num}: {title} (score: {score:.3f})")
            header = f"Found {len(chunks)} similar issues:\n"
            return header + "\n".join(lines) + f"\n\n**Answer:** {rag_result.get('answer', '')}"

        elif tool_name == "write_memory":
            content = arguments.get("content", "")
            memory_type = arguments.get("memory_type", "semantic")
            memory_id = await ltm.write_memory(
                session, content, memory_type=memory_type, user_id=user_id
            )
            return f"Memory saved (id: {memory_id})."

        else:
            return f"[Unknown tool: {tool_name}]"

    except httpx.HTTPError as exc:
        return f"[Tool {tool_name} failed: HTTP error — {exc}]"
    except Exception as exc:
        # Roll back any failed DB transaction so the session stays usable for the
        # caller's subsequent operations (e.g. _persist_messages in chat.py).
        try:
            await session.rollback()
        except Exception:
            pass
        return f"[Tool {tool_name} failed: {exc}]"
