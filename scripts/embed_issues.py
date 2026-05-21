"""
Embed all issues from the splits into pgvector using all-MiniLM-L6-v2.

Usage:
    python scripts/embed_issues.py

Requires: Docker stack running (db service healthy).
"""

import asyncio
import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.preprocess import make_text

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64
SPLITS_DIR = Path("data/splits")
DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/copilot"


async def embed_all() -> None:
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"  Embedding dim: {model.get_sentence_embedding_dimension()}")

    # Load all issues from splits
    records = []
    for split in ["train", "val", "test"]:
        path = SPLITS_DIR / f"{split}.jsonl"
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                records.append(r)

    print(f"Loaded {len(records)} issues")

    # Embed in batches
    texts = [make_text(r["title"], r["body"]) for r in records]
    print(f"Embedding {len(texts)} issues in batches of {BATCH_SIZE}...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"Embeddings shape: {embeddings.shape}")

    # Write to pgvector
    engine = create_async_engine(DATABASE_URL, echo=False)
    Session = async_sessionmaker(engine, expire_on_commit=False)

    print("Writing embeddings to pgvector...")
    async with Session() as session:
        # Ensure issues exist in DB first
        for i, (record, emb) in enumerate(zip(records, embeddings)):
            await session.execute(
                text("""
                    INSERT INTO issues (number, title, body, label, source_type, embedding)
                    VALUES (:number, :title, :body, :label, 'resolved_issue',
                        CAST(:embedding AS vector))
                    ON CONFLICT (number) DO UPDATE
                    SET embedding = CAST(:embedding AS vector),
                        source_type = 'resolved_issue'
                """),
                {
                    "number": record["number"],
                    "title": record["title"],
                    "body": (record.get("body") or "")[:2000],
                    "label": record["label"],
                    "embedding": str(emb.tolist()),
                },
            )
            if (i + 1) % 500 == 0:
                await session.commit()
                print(f"  {i + 1}/{len(records)} committed")

        await session.commit()

    print(f"\nDone — {len(records)} embeddings written to pgvector")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(embed_all())
