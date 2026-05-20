from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class IssueLabel(StrEnum):
    BUG = "bug"
    FEATURE = "feature"
    DOCS = "docs"
    QUESTION = "question"


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Issue(BaseModel):
    id: UUID
    number: int
    title: str
    body: str | None
    raw_labels: list[str]
    label: IssueLabel
    split: Split
    repo: str
    created_at: datetime
    closed_at: datetime


class AuditLogEntry(BaseModel):
    id: UUID
    actor_id: UUID | None
    action: str
    target: str
    metadata: dict = Field(default_factory=dict)
    timestamp: datetime


class HealthStatus(BaseModel):
    status: str
    vault: str
    db: str
    redis: str
    tracing: str
    minio: str


class ErrorResponse(BaseModel):
    class ErrorDetail(BaseModel):
        code: str
        message: str
        request_id: str

    error: ErrorDetail
