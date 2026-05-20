"""Summarizer endpoint — returns a concise summary of an issue thread using Groq."""

import os

import groq

_client: groq.Groq | None = None

SUMMARIZE_SYSTEM = """You are a GitHub issue summarizer for open-source maintainers.
Given an issue title and body, write a concise summary in under 100 words.

Focus on:
1. What the user is reporting or requesting
2. The key technical detail (error, version, component)
3. What would resolve it (if stated)

Write in third person. Do not include greetings or meta-commentary. Just the summary."""


def get_client() -> groq.Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        _client = groq.Groq(api_key=api_key)
    return _client


def summarize_issue(text: str) -> str:
    client = get_client()
    msg = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=150,
        messages=[
            {"role": "system", "content": SUMMARIZE_SYSTEM},
            {"role": "user", "content": text[:3000]},
        ],
    )
    return msg.choices[0].message.content.strip()
