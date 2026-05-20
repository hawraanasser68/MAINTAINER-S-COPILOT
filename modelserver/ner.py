"""NER endpoint — extracts code-shaped entities from issue text using Groq."""

import json
import os

import groq
from pydantic import BaseModel

_client: groq.Groq | None = None

ENTITY_TYPES = ["function", "error_code", "package", "version"]

NER_SYSTEM = """You are a technical entity extractor for GitHub issues.
Extract named entities from the text and return them as a JSON array.

Entity types:
- function: Python/C function or method names (e.g. fit(), _validate_data, sklearn.metrics.f1_score)
- error_code: Exception types or HTTP/exit codes (e.g. ValueError, ImportError, 404)
- package: Library or package names (e.g. numpy, scikit-learn, pandas)
- version: Version strings (e.g. 1.3.0, v2.1, Python 3.11)

Return ONLY a JSON array like:
[{"text": "ValueError", "type": "error_code"}, {"text": "numpy", "type": "package"}]

If no entities found, return [].
"""


class Entity(BaseModel):
    text: str
    type: str


def get_client() -> groq.Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        _client = groq.Groq(api_key=api_key)
    return _client


def extract_entities(text: str) -> list[Entity]:
    client = get_client()
    msg = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=512,
        messages=[
            {"role": "system", "content": NER_SYSTEM},
            {"role": "user", "content": text[:3000]},
        ],
    )
    raw = msg.choices[0].message.content.strip()
    # Strip markdown code fences if the model wrapped the JSON
    if "```" in raw:
        raw = raw.split("```")[-2] if raw.count("```") >= 2 else raw
        raw = raw.lstrip("json").strip()
    # Extract first JSON array if model added prose around it
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    try:
        items = json.loads(raw)
        return [
            Entity(text=item["text"], type=item["type"])
            for item in items
            if item.get("type") in ENTITY_TYPES
        ]
    except Exception:
        return []
