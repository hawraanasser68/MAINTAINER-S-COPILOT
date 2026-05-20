"""
Build the BM25 index from all splits and upload to MinIO.

Usage:
    python scripts/build_bm25.py

Requires: Docker stack running (MinIO healthy).
"""

import io
import json
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.infra.bm25_index import build_index

SPLITS_DIR = Path("data/splits")
import os
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "http://localhost:9000")
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET = "models"
KEY = "bm25/index.pkl"


def main() -> None:
    print("Loading issues from splits...")
    records = []
    for split in ["train", "val", "test"]:
        path = SPLITS_DIR / f"{split}.jsonl"
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    print(f"  Loaded {len(records)} issues")

    print("Building BM25 index...")
    index = build_index(records)
    data = index.to_bytes()
    print(f"  Index size: {len(data) / 1024 / 1024:.1f} MB")

    print("Uploading to MinIO...")
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
    )

    try:
        s3.head_bucket(Bucket=BUCKET)
    except ClientError:
        s3.create_bucket(Bucket=BUCKET)
        print(f"  Created bucket: {BUCKET}")

    s3.put_object(Bucket=BUCKET, Key=KEY, Body=data)
    print(f"  Uploaded → s3://{BUCKET}/{KEY}")
    print("Done.")


if __name__ == "__main__":
    main()
