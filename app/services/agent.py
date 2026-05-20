"""
Tool-calling agent loop.

Primary path: Anthropic claude-haiku-4-5 with proper tool_choice support.
Fallback path: Groq llama-3.3-70b-versatile with keyword-based rag_search pre-routing
               (used automatically when ANTHROPIC_API_KEY is absent or the primary call fails).
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import anthropic as _anthropic_sdk
from groq import BadRequestError, Groq
from sqlalchemy.ext.asyncio import AsyncSession

from app.services import short_term_memory as stm
from app.services import long_term_memory as ltm
from app.services.tools import TOOL_SCHEMAS, execute_tool

_MAX_TOOL_ROUNDS = 5

# Groq fallback: keyword set that triggers rag_search pre-routing when Anthropic is unavailable.
_RAG_TRIGGERS = {
    "find", "search", "look up", "past issue", "similar issue", "known issue",
    "has anyone", "is there", "have there been", "related issue", "before",
    "history", "corpus", "error", "bug", "crash", "fail",
}
# Messages that start with these words are handled by a single tool — skip rag pre-routing.
_SINGLE_TOOL_PREFIXES = ("summarize", "classify", "extract entities", "extract_entities")

_SYSTEM_PROMPT = (Path(__file__).parent.parent.parent / "prompts" / "system.txt").read_text()

# Convert OpenAI/Groq tool schemas → Anthropic format once at import time.
ANTHROPIC_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": t["function"]["name"],
        "description": t["function"]["description"],
        "input_schema": t["function"]["parameters"],
    }
    for t in TOOL_SCHEMAS
]

_groq_client: Groq | None = None
_anthropic_client: _anthropic_sdk.AsyncAnthropic | None = None


def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _get_anthropic() -> _anthropic_sdk.AsyncAnthropic:
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        _anthropic_client = _anthropic_sdk.AsyncAnthropic(api_key=api_key)
    return _anthropic_client


def _inject_memories(system: str, memories: list[dict]) -> str:
    if not memories:
        return system
    lines = [f"- {m['content']}" for m in memories[:5]]
    mem_block = "## Retrieved memories from past sessions\n" + "\n".join(lines)
    return f"{system}\n\n{mem_block}"


# ---------------------------------------------------------------------------
# Primary path — Anthropic claude-haiku-4-5
# ---------------------------------------------------------------------------

async def _run_anthropic(
    conversation_id: str | uuid.UUID,
    user_message: str,
    session: AsyncSession,
    user_id: uuid.UUID | None,
    history: list[dict],
    system_prompt: str,
    ttl: int,
) -> dict[str, Any] | None:
    """
    Run one agent turn using Anthropic.
    Returns a result dict on success, or None to signal the caller to use the Groq fallback.
    History is NOT mutated on failure so the Groq fallback gets a clean slate.
    """
    try:
        client = _get_anthropic()
    except RuntimeError:
        return None  # No key → fall back immediately

    # Build Anthropic messages from the history (already includes the new user message).
    # Plain text entries from Redis convert directly; Anthropic accepts string content.
    anthropic_msgs: list[dict[str, Any]] = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m["role"] in ("user", "assistant")
    ]

    tool_calls_made: list[dict[str, Any]] = []
    reply = ""
    round_num = 0

    for round_num in range(_MAX_TOOL_ROUNDS):
        try:
            response = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                system=system_prompt,
                messages=anthropic_msgs,
                tools=ANTHROPIC_TOOL_SCHEMAS,
                tool_choice={"type": "required"} if round_num == 0 else {"type": "auto"},
            )
        except Exception:
            if round_num == 0:
                return None  # First call failed → try Groq fallback
            # Mid-loop failure — generate a plain answer from what we have
            try:
                fallback = await client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=256,
                    system=system_prompt,
                    messages=anthropic_msgs + [
                        {"role": "user", "content": "Please give a final answer based on the tool results above."}
                    ],
                )
                reply = fallback.content[0].text if fallback.content else "[No response]"
            except Exception:
                reply = "[No response]"
            history.append({"role": "assistant", "content": reply})
            break

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        if not tool_use_blocks:
            # Final text answer
            reply = text_blocks[0].text if text_blocks else ""
            history.append({"role": "assistant", "content": reply})
            break

        # Add the assistant message (with tool calls) to the conversation
        anthropic_msgs.append({"role": "assistant", "content": response.content})

        tool_results: list[dict[str, Any]] = []
        for block in tool_use_blocks:
            tool_result = await execute_tool(block.name, block.input, session, user_id=user_id)
            tool_calls_made.append({"tool": block.name, "args": block.input, "result": tool_result})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": tool_result,
            })

            if block.name == "write_memory":
                reply = "I've noted that down and saved it to memory."
                history.append({"role": "assistant", "content": reply})
                await stm.set_history(conversation_id, history, ttl=ttl)
                return {"reply": reply, "tool_calls_made": tool_calls_made, "rounds": round_num + 1}

        anthropic_msgs.append({"role": "user", "content": tool_results})

    else:
        # Safety limit
        try:
            fallback = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=256,
                system=system_prompt,
                messages=anthropic_msgs + [
                    {"role": "user", "content": "Please give a final answer based on the tool results above."}
                ],
            )
            reply = fallback.content[0].text if fallback.content else "[No response]"
        except Exception:
            reply = "[No response]"
        history.append({"role": "assistant", "content": reply})

    await stm.set_history(conversation_id, history, ttl=ttl)
    return {"reply": reply, "tool_calls_made": tool_calls_made, "rounds": round_num + 1}


# ---------------------------------------------------------------------------
# Fallback path — Groq llama-3.3-70b-versatile with keyword pre-routing
# ---------------------------------------------------------------------------

async def _run_groq(
    conversation_id: str | uuid.UUID,
    user_message: str,
    session: AsyncSession,
    user_id: uuid.UUID | None,
    history: list[dict],
    system_prompt: str,
    ttl: int,
) -> dict[str, Any]:
    messages_for_llm: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        *history,
    ]

    groq = _get_groq()
    tool_calls_made: list[dict[str, Any]] = []
    reply = ""

    # Keyword pre-routing: run rag_search early for search-type queries so the LLM
    # always has results available (Groq ignores tool_choice="required").
    msg_lower = user_message.lower()
    is_single_tool = any(msg_lower.startswith(p) for p in _SINGLE_TOOL_PREFIXES)
    if not is_single_tool and any(t in msg_lower for t in _RAG_TRIGGERS):
        rag_result_str = await execute_tool("rag_search", {"query": user_message}, session, user_id=user_id)
        tool_calls_made.append({"tool": "rag_search", "args": {"query": user_message}, "result": rag_result_str})
        synthesis_msgs = [
            *messages_for_llm[:-1],
            {
                "role": "user",
                "content": (
                    f"{user_message}\n\n"
                    f"Search results (you MUST cite the issue numbers in your answer):\n"
                    f"{rag_result_str}"
                ),
            },
        ]
        synth = groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=synthesis_msgs,
            max_tokens=512,
        )
        reply = synth.choices[0].message.content or ""
        history.append({"role": "assistant", "content": reply})
        await stm.set_history(conversation_id, history, ttl=ttl)
        return {"reply": reply, "tool_calls_made": tool_calls_made, "rounds": 1}

    round_num = 0
    for round_num in range(_MAX_TOOL_ROUNDS):
        try:
            response = groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages_for_llm,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                max_tokens=512,
            )
        except BadRequestError:
            response = groq.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages_for_llm + [
                    {"role": "user", "content": "Please answer without using any tools."}
                ],
                max_tokens=512,
            )
            reply = response.choices[0].message.content or ""
            history.append({"role": "assistant", "content": reply})
            break

        choice = response.choices[0]
        assistant_msg = choice.message

        if not assistant_msg.tool_calls:
            reply = assistant_msg.content or ""
            messages_for_llm.append({"role": "assistant", "content": reply})
            history.append({"role": "assistant", "content": reply})
            break

        tool_calls_payload = [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in assistant_msg.tool_calls
        ]
        messages_for_llm.append({"role": "assistant", "content": assistant_msg.content or "", "tool_calls": tool_calls_payload})

        for tc in assistant_msg.tool_calls:
            fn_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            tool_result = await execute_tool(fn_name, args, session, user_id=user_id)
            tool_calls_made.append({"tool": fn_name, "args": args, "result": tool_result})
            messages_for_llm.append({"role": "tool", "tool_call_id": tc.id, "content": tool_result})

            if fn_name == "write_memory":
                reply = "I've noted that down and saved it to memory."
                history.append({"role": "assistant", "content": reply})
                await stm.set_history(conversation_id, history, ttl=ttl)
                return {"reply": reply, "tool_calls_made": tool_calls_made, "rounds": round_num + 1}

    else:
        response = groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages_for_llm + [
                {"role": "user", "content": "Please give me a final answer based on the tool results above."}
            ],
            max_tokens=512,
        )
        reply = response.choices[0].message.content or "[No response]"
        history.append({"role": "assistant", "content": reply})

    await stm.set_history(conversation_id, history, ttl=ttl)
    return {"reply": reply, "tool_calls_made": tool_calls_made, "rounds": round_num + 1}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_agent(
    conversation_id: str | uuid.UUID,
    user_message: str,
    session: AsyncSession,
    user_id: uuid.UUID | None = None,
    ttl: int = 3600,
) -> dict[str, Any]:
    """
    Run one full agent turn.
    Tries Anthropic first; falls back to Groq automatically if Anthropic is
    unavailable or returns an error on the first call.
    """
    history = await stm.get_history(conversation_id)
    memories = await ltm.retrieve_memories(session, user_message, top_k=3, user_id=user_id)
    system_prompt = _inject_memories(_SYSTEM_PROMPT, memories)

    history.append({"role": "user", "content": user_message})

    # Primary: Anthropic
    result = await _run_anthropic(
        conversation_id, user_message, session, user_id, history, system_prompt, ttl
    )
    if result is not None:
        return result

    # Fallback: Groq (history already has the user message appended above)
    return await _run_groq(
        conversation_id, user_message, session, user_id, history, system_prompt, ttl
    )
