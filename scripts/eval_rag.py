"""
RAG evaluation — hit@5, MRR@10 for naive vs hybrid pipeline.
Runs locally without Docker by using the precomputed embeddings and BM25 index.

Usage:
    python scripts/eval_rag.py

Outputs:
    models/rag_eval/metrics.json
"""

import json
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.preprocess import load_split, make_text

OUTPUT_DIR = Path("models/rag_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GOLDEN_PATH = Path("data/golden_rag.jsonl")


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN_PATH.read_text().splitlines() if line.strip()]


def hit_at_k(retrieved_numbers: list[int], ground_truth: list[int], k: int) -> float:
    if not ground_truth:
        return 0.0
    return float(bool(set(retrieved_numbers[:k]) & set(ground_truth)))


def mrr_at_k(retrieved_numbers: list[int], ground_truth: list[int], k: int) -> float:
    if not ground_truth:
        return 0.0
    for rank, num in enumerate(retrieved_numbers[:k], start=1):
        if num in ground_truth:
            return 1.0 / rank
    return 0.0


def main() -> None:
    golden = load_golden()
    # Only evaluate examples with ground_truth_issue_numbers
    eval_set = [g for g in golden if g["ground_truth_issue_numbers"]]
    print(f"Golden set: {len(golden)} total, {len(eval_set)} with ground-truth issue numbers")

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)
    reranker = CrossEncoder(RERANK_MODEL)

    print("Loading all issues and building local index...")
    all_texts, all_labels = load_split("train")
    val_texts, val_labels = load_split("val")
    test_texts, test_labels = load_split("test")

    import json as _json
    all_records = []
    for split in ["train", "val", "test"]:
        for line in Path(f"data/splits/{split}.jsonl").read_text().splitlines():
            if line.strip():
                all_records.append(_json.loads(line))

    texts = [make_text(r["title"], r["body"]) for r in all_records]
    numbers = [r["number"] for r in all_records]

    print(f"  Building BM25 index over {len(texts)} issues...")
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    print(f"  Embedding {len(texts)} issues (this takes a few minutes)...")
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    def dense_search(query: str, top_k: int = 20) -> list[int]:
        qvec = embedder.encode(query, normalize_embeddings=True)
        scores = embeddings @ qvec
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [numbers[i] for i in top_idx]

    def bm25_search(query: str, top_k: int = 20) -> list[int]:
        tokens = query.lower().split()
        scores = bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [numbers[i] for i in top_idx]

    def rrf_fusion(dense: list[int], sparse: list[int], k: int = 60) -> list[int]:
        scores: dict[int, float] = {}
        for rank, n in enumerate(dense):
            scores[n] = scores.get(n, 0.0) + 1.0 / (k + rank + 1)
        for rank, n in enumerate(sparse):
            scores[n] = scores.get(n, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores, key=lambda n: scores[n], reverse=True)

    def hybrid_rerank(query: str, top_k: int = 5) -> list[int]:
        dense = dense_search(query, 20)
        sparse = bm25_search(query, 20)
        fused = rrf_fusion(dense, sparse)[:20]
        num_to_idx = {r["number"]: i for i, r in enumerate(all_records)}
        pairs = [(query, texts[num_to_idx[n]][:300]) for n in fused if n in num_to_idx]
        if not pairs:
            return fused[:top_k]
        rerank_scores = reranker.predict(pairs)
        ranked = sorted(zip(rerank_scores, fused), key=lambda x: x[0], reverse=True)
        return [n for _, n in ranked[:top_k]]

    print("\nEvaluating naive (dense only) vs advanced (hybrid + rerank)...\n")

    naive_hits, naive_mrr = [], []
    advanced_hits, advanced_mrr = [], []

    for example in eval_set:
        q = example["question"]
        gt = example["ground_truth_issue_numbers"]

        naive_top5 = dense_search(q, top_k=5)
        adv_top5 = hybrid_rerank(q, top_k=5)

        naive_hits.append(hit_at_k(naive_top5, gt, 5))
        naive_mrr.append(mrr_at_k(naive_top5, gt, 10))
        advanced_hits.append(hit_at_k(adv_top5, gt, 5))
        advanced_mrr.append(mrr_at_k(adv_top5, gt, 10))

        print(f"  Q{example['id']:02d}: naive_hit={naive_hits[-1]:.0f} adv_hit={advanced_hits[-1]:.0f} | {q[:55]}")

    naive_h5   = round(sum(naive_hits) / len(naive_hits), 4)
    naive_mrr10 = round(sum(naive_mrr) / len(naive_mrr), 4)
    adv_h5     = round(sum(advanced_hits) / len(advanced_hits), 4)
    adv_mrr10  = round(sum(advanced_mrr) / len(advanced_mrr), 4)

    print(f"\n{'='*50}")
    print(f"{'Metric':<20} {'Naive (dense)':>15} {'Advanced (hybrid)':>18}  {'Delta':>8}")
    print(f"{'='*50}")
    print(f"  {'hit@5':<18} {naive_h5:>15.4f} {adv_h5:>18.4f}  {adv_h5-naive_h5:>+8.4f}")
    print(f"  {'MRR@10':<18} {naive_mrr10:>15.4f} {adv_mrr10:>18.4f}  {adv_mrr10-naive_mrr10:>+8.4f}")

    metrics = {
        "eval_set_size": len(eval_set),
        "naive_dense": {"hit_at_5": naive_h5, "mrr_at_10": naive_mrr10},
        "advanced_hybrid_rerank": {"hit_at_5": adv_h5, "mrr_at_10": adv_mrr10},
        "delta": {"hit_at_5": round(adv_h5 - naive_h5, 4), "mrr_at_10": round(adv_mrr10 - naive_mrr10, 4)},
        "embedding_model": EMBED_MODEL,
        "reranker": RERANK_MODEL,
    }
    out = OUTPUT_DIR / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
