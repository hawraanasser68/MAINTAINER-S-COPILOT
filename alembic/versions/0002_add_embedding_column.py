"""Add embedding column to issues table.

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-19
"""

from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add 384-dim embedding column (all-MiniLM-L6-v2 output size)
    op.execute("ALTER TABLE issues ADD COLUMN IF NOT EXISTS embedding vector(384)")

    # HNSW index for fast approximate nearest-neighbour search
    op.execute("""
        CREATE INDEX IF NOT EXISTS issues_embedding_hnsw
        ON issues
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)

    # Add source_type column for metadata filtering (docs vs resolved_issue)
    op.execute("""
        ALTER TABLE issues
        ADD COLUMN IF NOT EXISTS source_type VARCHAR(32) DEFAULT 'resolved_issue'
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS issues_embedding_hnsw")
    op.execute("ALTER TABLE issues DROP COLUMN IF EXISTS embedding")
    op.execute("ALTER TABLE issues DROP COLUMN IF EXISTS source_type")
