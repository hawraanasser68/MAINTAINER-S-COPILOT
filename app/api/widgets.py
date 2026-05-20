"""
Widgets API — CRUD for widget configurations (admin only).

POST /widgets    — create a widget config
GET  /widgets    — list all widgets
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import current_admin_user
from app.infra.database import get_session
from app.repositories.models import UserORM, WidgetORM

router = APIRouter(prefix="/widgets", tags=["widgets"])


class WidgetCreate(BaseModel):
    allowed_origins: list[str] = []
    greeting: str = "Hi! How can I help you?"
    enabled_tools: list[str] = []
    theme: dict = {}


class WidgetResponse(BaseModel):
    id: uuid.UUID
    widget_id: uuid.UUID
    allowed_origins: list[str]
    greeting: str
    enabled_tools: list[str]
    theme: dict
    created_at: str


@router.post("", response_model=WidgetResponse, status_code=201)
async def create_widget(
    body: WidgetCreate,
    user: UserORM = Depends(current_admin_user),
    session: AsyncSession = Depends(get_session),
) -> WidgetResponse:
    widget = WidgetORM(
        id=uuid.uuid4(),
        widget_id=uuid.uuid4(),
        allowed_origins=body.allowed_origins,
        greeting=body.greeting,
        enabled_tools=body.enabled_tools,
        theme=body.theme,
        created_by=user.id,
    )
    session.add(widget)
    await session.commit()
    await session.refresh(widget)
    return WidgetResponse(
        id=widget.id,
        widget_id=widget.widget_id,
        allowed_origins=widget.allowed_origins,
        greeting=widget.greeting,
        enabled_tools=widget.enabled_tools,
        theme=widget.theme,
        created_at=str(widget.created_at),
    )


@router.get("", response_model=list[WidgetResponse])
async def list_widgets(
    user: UserORM = Depends(current_admin_user),
    session: AsyncSession = Depends(get_session),
) -> list[WidgetResponse]:
    from sqlalchemy import select
    result = await session.execute(select(WidgetORM).order_by(WidgetORM.created_at.desc()))
    widgets = result.scalars().all()
    return [
        WidgetResponse(
            id=w.id,
            widget_id=w.widget_id,
            allowed_origins=w.allowed_origins,
            greeting=w.greeting,
            enabled_tools=w.enabled_tools,
            theme=w.theme,
            created_at=str(w.created_at),
        )
        for w in widgets
    ]
