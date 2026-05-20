"""
RAG quality evaluation — faithfulness and answer relevancy.

For each of the 25 golden examples:
  1. Retrieve top-5 issues with the production pipeline (hybrid BM25+dense + rerank)
  2. Generate an answer with Groq llama-3.1-8b-instant (same model as production)
  3. Score faithfulness  — judge LLM: is every claim grounded in the retrieved context?
  4. Score answer relevancy — judge LLM: does the answer address the question?

Judge model: llama-3.3-70b-versatile (larger, more reliable for rating)
Requires: GROQ_API_KEY

Usage:
    GROQ_API_KEY=gsk_... python scripts/eval_faithfulness.py

Outputs:
    models/rag_eval/metrics.json  — adds faithfulness + answer_relevancy keys
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.preprocess import make_text  # noqa: E402

GOLDEN_PATH = Path("data/golden_rag.jsonl")
METRICS_PATH = Path("models/rag_eval/metrics.json")

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEN_MODEL = "llama-3.1-8b-instant"
JUDGE_MODEL = "llama-3.3-70b-versatile"

ANSWER_SYSTEM = (
    "You are a helpful assistant for open-source maintainers. "
    "Answer the question using ONLY the provided GitHub issue context. "
    "Be concise (under 150 words). If the context does not contain the answer, say so clearly. "
    "Always cite the issue numbers you used (e.g. 'See issue #1234')."
)

FAITHFULNESS_PROMPT = """\
You are an evaluation judge. Assess whether an AI-generated answer is faithful to the provided context.

Context (retrieved GitHub issues):
{context}

Question: {question}

Answer: {answer}

Faithfulness measures whether every factual claim in the answer is directly supported by the context above.
- 1.0 = every claim is explicitly in the context
- 0.5 = most claims are supported; minor extrapolations
- 0.0 = the answer contains unsupported or fabricated claims

Reply with a single decimal number between 0.0 and 1.0. Nothing else."""

RELEVANCY_PROMPT = """\
You are an evaluation judge. Assess whether an AI-generated answer is relevant to the question asked.

Question: {question}

Answer: {answer}

Answer relevancy measures how well the answer addresses the specific question.
- 1.0 = the answer directly and completely addresses the question
- 0.5 = the answer partially addresses the question or goes off-topic
- 0.0 = the answer does not address the question at all

Reply with a single decimal number between 0.0 and 1.0. Nothing else."""


def _parse_score(text: str) -> float | None:
    """Extract a 0-1 float from judge output; return None if unparseable."""
    text = text.strip()
    for tok in text.split():
        try:
            v = float(tok.rstrip(".,"))
            if 0.0 <= v <= 1.0:
                return v
        except ValueError:
            continue
    return None


def main() -> None:
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        print("ERROR: GROQ_API_KEY not set.")
        sys.exit(1)

    import groq as _groq
    client = _groq.Groq(api_key=groq_key)

    golden = [json.loads(l) for l in GOLDEN_PATH.read_text().splitlines() if l.strip()]
    print(f"Golden set: {len(golden)} examples\n")

    # ── Build local corpus (same as eval_rag.py) ──────────────────────────────
    print(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
    reranker = CrossEncoder(RERANK_MODEL)

    all_records: list[dict] = []
    for split in ("train", "val", "test"):
        for line in Path(f"data/splits/{split}.jsonl").read_text().splitlines():
            if line.strip():
                all_records.append(json.loads(line))

    texts = [make_text(r["title"], r["body"]) for r in all_records]
    numbers = [r["number"] for r in all_records]
    num_to_idx = {n: i for i, n in enumerate(numbers)}

    print(f"Building BM25 index over {len(texts)} issues...")
    bm25 = BM25Okapi([t.lower().split() for t in texts])

    print(f"Embedding {len(texts)} issues...")
    embeddings = embedder.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )

    # ── Retrieval helpers ─────────────────────────────────────────────────────

    def retrieve(query: str, top_k: int = 5) -> list[dict]:
        """Hybrid BM25 + dense + RRF + rerank — production pipeline."""
        qvec = embedder.encode(query, normalize_embeddings=True)
        dense_scores = embeddings @ qvec
        dense_top = list(np.argsort(dense_scores)[::-1][:20])

        bm25_scores = bm25.get_scores(query.lower().split())
        bm25_top = list(np.argsort(bm25_scores)[::-1][:20])

        # RRF fusion
        rrf: dict[int, float] = {}
        for rank, i in enumerate(dense_top):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + rank + 1)
        for rank, i in enumerate(bm25_top):
            rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + rank + 1)
        fused = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:20]

        # Cross-encoder rerank
        pairs = [(query, texts[i][:300]) for i in fused]
        rerank_scores = reranker.predict(pairs)
        ranked = sorted(zip(rerank_scores, fused), key=lambda x: x[0], reverse=True)
        top_indices = [i for _, i in ranked[:top_k]]

        return [
            {"number": numbers[i], "title": all_records[i]["title"], "body": all_records[i].get("body", "")}
            for i in top_indices
        ]

    def build_context(chunks: list[dict]) -> str:
        parts = []
        for c in chunks:
            body = (c.get("body") or "")[:300]
            parts.append(f"Issue #{c['number']}: {c['title']}\n{body}")
        return "\n\n---\n\n".join(parts)

    def generate_answer(question: str, context: str) -> str:
        try:
            resp = client.chat.completions.create(
                model=GEN_MODEL,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": ANSWER_SYSTEM},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[generation failed: {e}]"

    def judge_score(prompt: str) -> float:
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                max_tokens=10,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.choices[0].message.content.strip()
            score = _parse_score(raw)
            if score is None:
                print(f"    [WARN] unparseable judge output: {raw!r} — defaulting 0.5")
                return 0.5
            return score
        except Exception as e:
            print(f"    [WARN] judge call failed: {e} — defaulting 0.5")
            return 0.5

    # ── Evaluate ──────────────────────────────────────────────────────────────

    faithfulness_scores: list[float] = []
    relevancy_scores: list[float] = []

    print(f"\n{'─'*65}")
    print(f"{'Q':<5} {'faithful':>10} {'relevant':>10}  question (truncated)")
    print("─" * 65)

    for ex in golden:
        q = ex["question"]
        chunks = retrieve(q, top_k=5)
        context = build_context(chunks)
        answer = generate_answer(q, context)

        f_score = judge_score(FAITHFULNESS_PROMPT.format(
            context=context, question=q, answer=answer
        ))
        r_score = judge_score(RELEVANCY_PROMPT.format(
            question=q, answer=answer
        ))

        faithfulness_scores.append(f_score)
        relevancy_scores.append(r_score)

        print(f"Q{ex['id']:02d}   {f_score:>10.3f} {r_score:>10.3f}  {q[:40]}")

    avg_faithfulness = round(sum(faithfulness_scores) / len(faithfulness_scores), 4)
    avg_relevancy = round(sum(relevancy_scores) / len(relevancy_scores), 4)

    print(f"\n{'='*45}")
    print(f"  Faithfulness (avg):     {avg_faithfulness:.4f}")
    print(f"  Answer relevancy (avg): {avg_relevancy:.4f}")
    print(f"  Evaluated over:         {len(golden)} examples")
    print(f"  Generator model:        {GEN_MODEL}")
    print(f"  Judge model:            {JUDGE_MODEL}")
    print(f"{'='*45}")

    # ── Merge into existing metrics.json ──────────────────────────────────────
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    metrics["faithfulness"] = avg_faithfulness
    metrics["answer_relevancy"] = avg_relevancy
    metrics["quality_eval"] = {
        "faithfulness": avg_faithfulness,
        "answer_relevancy": avg_relevancy,
        "n_examples": len(golden),
        "generator_model": GEN_MODEL,
        "judge_model": JUDGE_MODEL,
        "per_example": [
            {"id": ex["id"], "faithfulness": f, "answer_relevancy": r}
            for ex, f, r in zip(golden, faithfulness_scores, relevancy_scores)
        ],
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved → {METRICS_PATH}")
    print("\nNext: update eval_thresholds.yaml with these numbers (floor ~5 pts below actual).")


if __name__ == "__main__":
    main()
