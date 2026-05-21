"""
Widgets API — CRUD for widget configurations (admin only).

POST /widgets    — create a widget config
GET  /widgets    — list all widgets
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import current_admin_user
from app.infra.database import get_session
from app.repositories.models import AuditLogORM, UserORM, WidgetORM

_CHAT_HTML_PATH = Path(__file__).parent.parent / "static" / "chat.html"
_chat_html_cache: str = ""


def _get_chat_html() -> str:
    global _chat_html_cache
    if not _chat_html_cache:
        _chat_html_cache = _CHAT_HTML_PATH.read_text()
    return _chat_html_cache

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
    session.add(AuditLogORM(
        actor_id=user.id,
        action="widget.create",
        target=str(widget.widget_id),
        metadata_={
            "allowed_origins": body.allowed_origins,
            "enabled_tools": body.enabled_tools,
            "has_theme": bool(body.theme),
        },
    ))
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


@router.get("/public/{widget_id}/config")
async def get_widget_config(
    widget_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Public — no auth. Used by widget.js to fetch theme and greeting at load time."""
    result = await session.execute(select(WidgetORM).where(WidgetORM.widget_id == widget_id))
    widget = result.scalar_one_or_none()
    if not widget:
        raise HTTPException(status_code=404, detail="Widget not found")
    return {
        "widget_id": str(widget.widget_id),
        "theme": widget.theme,
        "greeting": widget.greeting,
        "enabled_tools": widget.enabled_tools,
        "allowed_origins": widget.allowed_origins,
    }


@router.get("/embed/{widget_id}", response_class=HTMLResponse)
async def embed_widget(
    widget_id: uuid.UUID,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> HTMLResponse:
    """
    Serves chat.html with Content-Security-Policy: frame-ancestors built from
    the widget's allowed_origins — the browser enforces it, blocking any host
    not in the list from embedding this widget.
    """
    result = await session.execute(select(WidgetORM).where(WidgetORM.widget_id == widget_id))
    widget = result.scalar_one_or_none()
    if not widget:
        return HTMLResponse(
            "<p style='font-family:sans-serif;padding:24px'>Widget not found.</p>",
            status_code=404,
        )

    allowed = widget.allowed_origins or []
    frame_ancestors = " ".join(allowed) if allowed else "'none'"

    headers: dict[str, str] = {
        "Content-Security-Policy": f"frame-ancestors {frame_ancestors}",
    }

    # Per-widget CORS: only allow fetch calls from origins in allowed_origins
    origin = request.headers.get("origin", "")
    if origin and origin in allowed:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Vary"] = "Origin"

    return HTMLResponse(content=_get_chat_html(), headers=headers)
