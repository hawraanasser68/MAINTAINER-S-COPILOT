"""
Model server — exposes /classify, /ner, /summarize, /health.
Loads the fine-tuned classifier at startup and refuses to start if SHA-256 mismatches.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.infra import vault as vault_infra
from modelserver.classifier import classify, load_model
from modelserver.ner import Entity, extract_entities
from modelserver.summarize import summarize_issue




# ── Request / Response models ─────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    title: str
    body: str = ""


class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    all_scores: dict[str, float]
    model_used: str
    latency_ms: float


class NERRequest(BaseModel):
    text: str


class NERResponse(BaseModel):
    entities: list[Entity]
    model_used: str = "llama-3.1-8b-instant"


class SummarizeRequest(BaseModel):
    text: str = ""      # full thread text (preferred)
    title: str = ""     # legacy fields — used when text is absent
    body: str = ""


class SummarizeResponse(BaseModel):
    summary: str
    model_used: str = "llama-3.1-8b-instant"


# ── Lifespan — validate classifier weights at startup ────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    groq_key = vault_infra.get_secret("llm/groq_api_key")
    os.environ["GROQ_API_KEY"] = groq_key
    load_model()  # raises RuntimeError on SHA-256 mismatch or missing weights
    yield


app = FastAPI(title="Model Server", version="2.0.0", lifespan=lifespan)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    try:
        _, _, _, card = load_model()
        return {
            "status": "ok",
            "classifier": card.get("architecture"),
            "test_macro_f1": card.get("metrics", {}).get("test_macro_f1"),
        }
    except RuntimeError as e:
        return {"status": "degraded", "error": str(e)}


@app.post("/classify", response_model=ClassifyResponse)
async def classify_endpoint(req: ClassifyRequest) -> ClassifyResponse:
    try:
        result = classify(req.title, req.body)
        return ClassifyResponse(**result)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/ner", response_model=NERResponse)
async def ner_endpoint(req: NERRequest) -> NERResponse:
    entities = extract_entities(req.text)
    return NERResponse(entities=entities)


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_endpoint(req: SummarizeRequest) -> SummarizeResponse:
    text = req.text or f"{req.title}\n\n{req.body}".strip()
    summary = summarize_issue(text)
    return SummarizeResponse(summary=summary)
