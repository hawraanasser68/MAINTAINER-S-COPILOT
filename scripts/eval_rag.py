"""
RAG ablation study — 6 pipeline variants.

  1. dense_only           — cosine similarity, raw query
  2. bm25_only            — keyword search, raw query
  3. hybrid_no_rerank     — BM25 + dense + RRF, no cross-encoder
  4. dense_rerank         — dense + cross-encoder, no BM25
  5. hybrid_rerank        — BM25 + dense + RRF + cross-encoder  (production pipeline)
  6. hybrid_rerank_hyde   — same as 5 but retrieval uses HyDE-rewritten query
                            (requires GROQ_API_KEY; skipped if not set)

Usage:
    python scripts/eval_rag.py
    GROQ_API_KEY=sk-... python scripts/eval_rag.py   # enables HyDE variant

Outputs:
    models/rag_eval/metrics.json
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

OUTPUT_DIR = Path("models/rag_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GOLDEN_PATH = Path("data/golden_rag.jsonl")

HYDE_SYSTEM = (
    "You are a GitHub issue assistant for scikit-learn. "
    "Given a maintainer's question, write a short hypothetical GitHub issue "
    "(title + 2-sentence body) that would perfectly answer the question if it "
    "existed in the issue tracker. "
    "Write ONLY the hypothetical issue text — no preamble, no explanation."
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def hit_at_k(retrieved: list[int], ground_truth: list[int], k: int) -> float:
    if not ground_truth:
        return 0.0
    return float(bool(set(retrieved[:k]) & set(ground_truth)))


def mrr_at_k(retrieved: list[int], ground_truth: list[int], k: int) -> float:
    if not ground_truth:
        return 0.0
    for rank, num in enumerate(retrieved[:k], start=1):
        if num in ground_truth:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# HyDE
# ---------------------------------------------------------------------------

def hyde_rewrite(query: str, client) -> str:  # type: ignore[no-untyped-def]
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=150,
            messages=[
                {"role": "system", "content": HYDE_SYSTEM},
                {"role": "user", "content": query},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return query


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    golden = [json.loads(line) for line in GOLDEN_PATH.read_text().splitlines() if line.strip()]
    eval_set = [g for g in golden if g["ground_truth_issue_numbers"]]
    print(f"Golden set: {len(golden)} total, {len(eval_set)} with ground-truth issue numbers\n")

    # Optional HyDE via Groq
    groq_api_key = os.environ.get("GROQ_API_KEY", "")
    groq_client = None
    if groq_api_key:
        import groq as _groq
        groq_client = _groq.Groq(api_key=groq_api_key)
        print("GROQ_API_KEY found — HyDE variant enabled.")
    else:
        print("GROQ_API_KEY not set — HyDE variant will be skipped.")

    # Load models
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
    print(f"Loading reranker: {RERANK_MODEL}")
    reranker = CrossEncoder(RERANK_MODEL)

    # Build corpus
    print("\nLoading all issues and building index...")
    all_records: list[dict] = []
    for split in ("train", "val", "test"):
        for line in Path(f"data/splits/{split}.jsonl").read_text().splitlines():
            if line.strip():
                all_records.append(json.loads(line))

    texts = [make_text(r["title"], r["body"]) for r in all_records]
    numbers = [r["number"] for r in all_records]
    num_to_idx = {n: i for i, n in enumerate(numbers)}

    print(f"  Building BM25 index over {len(texts)} issues...")
    bm25 = BM25Okapi([t.lower().split() for t in texts])

    print(f"  Embedding {len(texts)} issues (this takes a few minutes)...")
    embeddings = embedder.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )

    # ── Primitive search functions ────────────────────────────────────────────

    def dense_search(query: str, top_k: int = 20) -> list[int]:
        qvec = embedder.encode(query, normalize_embeddings=True)
        scores = embeddings @ qvec
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [numbers[i] for i in top_idx]

    def bm25_search(query: str, top_k: int = 20) -> list[int]:
        scores = bm25.get_scores(query.lower().split())
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [numbers[i] for i in top_idx]

    def rrf(dense: list[int], sparse: list[int], k: int = 60) -> list[int]:
        sc: dict[int, float] = {}
        for rank, n in enumerate(dense):
            sc[n] = sc.get(n, 0.0) + 1.0 / (k + rank + 1)
        for rank, n in enumerate(sparse):
            sc[n] = sc.get(n, 0.0) + 1.0 / (k + rank + 1)
        return sorted(sc, key=lambda n: sc[n], reverse=True)

    def rerank(query: str, candidates: list[int], top_k: int = 5) -> list[int]:
        pairs = [(query, texts[num_to_idx[n]][:300]) for n in candidates if n in num_to_idx]
        if not pairs:
            return candidates[:top_k]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [n for _, n in ranked[:top_k]]

    # ── Pipeline definitions ──────────────────────────────────────────────────

    def p_dense(q: str) -> list[int]:
        return dense_search(q, top_k=5)

    def p_bm25(q: str) -> list[int]:
        return bm25_search(q, top_k=5)

    def p_hybrid(q: str) -> list[int]:
        return rrf(dense_search(q, 20), bm25_search(q, 20))[:5]

    def p_dense_rerank(q: str) -> list[int]:
        return rerank(q, dense_search(q, 20), top_k=5)

    def p_hybrid_rerank(q: str) -> list[int]:
        fused = rrf(dense_search(q, 20), bm25_search(q, 20))[:20]
        return rerank(q, fused, top_k=5)

    def p_hybrid_rerank_hyde(q: str) -> list[int]:
        hyp = hyde_rewrite(q, groq_client)
        fused = rrf(dense_search(hyp, 20), bm25_search(hyp, 20))[:20]
        return rerank(q, fused, top_k=5)  # rerank with original query for accuracy

    pipelines: dict[str, object] = {
        "dense_only":         p_dense,
        "bm25_only":          p_bm25,
        "hybrid_no_rerank":   p_hybrid,
        "dense_rerank":       p_dense_rerank,
        "hybrid_rerank":      p_hybrid_rerank,
    }
    if groq_client:
        pipelines["hybrid_rerank_hyde"] = p_hybrid_rerank_hyde

    # ── Evaluate all pipelines ────────────────────────────────────────────────

    data: dict[str, dict[str, list[float]]] = {
        name: {"hits": [], "mrrs": []} for name in pipelines
    }

    col_w = 16
    print(f"\n{'Q':<5}" + "".join(f"{n:>{col_w}}" for n in pipelines))
    print("─" * (5 + col_w * len(pipelines)))

    for ex in eval_set:
        q = ex["question"]
        gt = ex["ground_truth_issue_numbers"]
        row = f"Q{ex['id']:02d}  "
        for name, fn in pipelines.items():
            top5 = fn(q)  # type: ignore[operator]
            h = hit_at_k(top5, gt, 5)
            m = mrr_at_k(top5, gt, 10)
            data[name]["hits"].append(h)
            data[name]["mrrs"].append(m)
            row += f"{'hit' if h else 'MISS':>{col_w}}"
        print(row)

    # ── Summary ───────────────────────────────────────────────────────────────

    print(f"\n{'='*55}")
    print(f"  {'Pipeline':<24} {'hit@5':>7} {'MRR@10':>8}")
    print(f"  {'─'*24} {'─'*7} {'─'*8}")

    summary: dict[str, dict[str, float]] = {}
    for name, d in data.items():
        h5 = round(sum(d["hits"]) / len(d["hits"]), 4)
        mrr10 = round(sum(d["mrrs"]) / len(d["mrrs"]), 4)
        summary[name] = {"hit_at_5": h5, "mrr_at_10": mrr10}
        tag = "  ← production" if name == "hybrid_rerank" else ""
        print(f"  {name:<24} {h5:>7.4f} {mrr10:>8.4f}{tag}")

    print(f"{'='*55}")

    # Improvement of each over dense_only baseline
    baseline_h = summary["dense_only"]["hit_at_5"]
    baseline_m = summary["dense_only"]["mrr_at_10"]
    print(f"\n  Deltas vs dense_only baseline:")
    for name, s in summary.items():
        if name == "dense_only":
            continue
        dh = s["hit_at_5"] - baseline_h
        dm = s["mrr_at_10"] - baseline_m
        print(f"  {name:<24}  hit@5 {dh:+.4f}   MRR@10 {dm:+.4f}")

    # ── Write metrics.json ────────────────────────────────────────────────────

    metrics = {
        "eval_set_size": len(eval_set),
        "embedding_model": EMBED_MODEL,
        "reranker": RERANK_MODEL,
        # Original keys — keep for CI gate backward compatibility
        "naive_dense": summary["dense_only"],
        "advanced_hybrid_rerank": summary["hybrid_rerank"],
        "delta": {
            "hit_at_5":  round(summary["hybrid_rerank"]["hit_at_5"]  - summary["dense_only"]["hit_at_5"],  4),
            "mrr_at_10": round(summary["hybrid_rerank"]["mrr_at_10"] - summary["dense_only"]["mrr_at_10"], 4),
        },
        # Full ablation results
        "ablation": summary,
    }

    out = OUTPUT_DIR / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
