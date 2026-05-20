"""Add messages table and complete chatbot columns.

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-19
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # messages — per-conversation turn history
    op.create_table(
        "messages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.VARCHAR(20), nullable=False),   # user | assistant | tool
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("tool_calls", postgresql.JSON, nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True),
                  nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])

    # conversations — add ttl_seconds column
    op.add_column("conversations",
        sa.Column("ttl_seconds", sa.Integer, nullable=False, server_default="3600"))

    # long_term_memory — fix embedding dim to 384 (all-MiniLM-L6-v2)
    op.execute("""
        ALTER TABLE long_term_memory
        DROP COLUMN IF EXISTS embedding
    """)
    op.execute("""
        ALTER TABLE long_term_memory
        ADD COLUMN IF NOT EXISTS embedding vector(384)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ltm_embedding_hnsw
        ON long_term_memory
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_ltm_user_id
        ON long_term_memory (user_id)
    """)

    # users — add is_verified column expected by fastapi-users
    op.execute("""
        ALTER TABLE users
        ADD COLUMN IF NOT EXISTS is_verified BOOLEAN NOT NULL DEFAULT false
    """)


def downgrade() -> None:
    op.drop_table("messages")
    op.drop_column("conversations", "ttl_seconds")
    op.execute("DROP INDEX IF EXISTS ltm_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_ltm_user_id")
    op.execute("ALTER TABLE long_term_memory DROP COLUMN IF EXISTS embedding")
    op.execute("ALTER TABLE long_term_memory ADD COLUMN embedding vector(1536)")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS is_verified")
