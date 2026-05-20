"""
HyDE query rewriting — generate a hypothetical resolved issue, embed it.
The embedding of the hypothetical answer is often closer to real answers
than the embedding of the original short question.
"""

from __future__ import annotations

import os

from groq import Groq

from app.services.retrieval import embed_query

_client: Groq | None = None

HYDE_SYSTEM = """You are a GitHub issue assistant for scikit-learn.
Given a maintainer's question, write a short hypothetical GitHub issue (title + 2-sentence body)
that would perfectly answer the question if it existed in the issue tracker.
Write ONLY the hypothetical issue text — no preamble, no explanation."""


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        _client = Groq(api_key=api_key)
    return _client


def rewrite_query(query: str) -> tuple[str, list[float]]:
    """
    HyDE: generate a hypothetical issue text and return
    (hypothetical_text, embedding_of_hypothetical_text).
    Falls back to original query if LLM call fails.
    """
    try:
        client = get_client()
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=150,
            messages=[
                {"role": "system", "content": HYDE_SYSTEM},
                {"role": "user", "content": query},
            ],
        )
        hypothetical = resp.choices[0].message.content.strip()
    except Exception:
        hypothetical = query

    embedding = embed_query(hypothetical)
    return hypothetical, embedding
