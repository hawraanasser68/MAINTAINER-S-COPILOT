"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-05-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Extensions
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # issues — the dataset
    op.create_table(
        "issues",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("number", sa.Integer, nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("body", sa.Text, nullable=True),
        sa.Column("raw_labels", postgresql.ARRAY(sa.Text), nullable=False, server_default="{}"),
        sa.Column("label", sa.VARCHAR(20), nullable=False),
        sa.Column("split", sa.VARCHAR(10), nullable=False),
        sa.Column("repo", sa.VARCHAR(100), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("closed_at", sa.TIMESTAMP(timezone=True), nullable=False),
    )
    op.create_index("ix_issues_label", "issues", ["label"])
    op.create_index("ix_issues_split", "issues", ["split"])
    op.create_index("ix_issues_closed_at", "issues", ["closed_at"])

    # audit_log — append-only event log
    op.create_table(
        "audit_log",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("actor_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("action", sa.VARCHAR(50), nullable=False),
        sa.Column("target", sa.Text, nullable=False),
        sa.Column("metadata", postgresql.JSON, nullable=False, server_default="{}"),
        sa.Column("timestamp", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.text("now()")),
    )

    # Stub tables — schema only, populated in Phase 4
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("email", sa.VARCHAR(255), nullable=False, unique=True),
        sa.Column("hashed_password", sa.Text, nullable=False),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("is_superuser", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("role", sa.VARCHAR(20), nullable=False, server_default="user"),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()")),
    )

    op.create_table(
        "widgets",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("widget_id", postgresql.UUID(as_uuid=True), nullable=False, unique=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("allowed_origins", postgresql.ARRAY(sa.Text), nullable=False, server_default="{}"),
        sa.Column("theme", postgresql.JSON, nullable=False, server_default='{"primary_color": "#0ea5e9", "position": "bottom-right"}'),
        sa.Column("greeting", sa.Text, nullable=False, server_default="Hi! How can I help you?"),
        sa.Column("enabled_tools", postgresql.ARRAY(sa.Text), nullable=False, server_default="{}"),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()")),
    )

    op.create_table(
        "conversations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("redis_key", sa.Text, nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()")),
    )

    op.create_table(
        "long_term_memory",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("memory_type", sa.VARCHAR(20), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        # vector column added after pgvector is confirmed available
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()")),
    )
    # Add pgvector column separately so the extension is guaranteed to exist first
    op.execute("ALTER TABLE long_term_memory ADD COLUMN IF NOT EXISTS embedding vector(1536)")


def downgrade() -> None:
    op.drop_table("long_term_memory")
    op.drop_table("conversations")
    op.drop_table("widgets")
    op.drop_table("users")
    op.drop_table("audit_log")
    op.drop_table("issues")
