"""
Authentication — fastapi-users with JWT bearer tokens.
JWT secret resolved from Vault at startup.
"""

from __future__ import annotations

import os
import uuid

from fastapi import Depends
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin
from fastapi_users.authentication import AuthenticationBackend, BearerTransport, JWTStrategy
from fastapi_users.db import SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession

from app.infra.database import get_session
from app.repositories.models import UserORM

SECRET = os.environ.get("JWT_SECRET", "dev-jwt-secret-change-in-prod")
TOKEN_LIFETIME = 3600 * 8  # 8 hours


async def get_user_db(session: AsyncSession = Depends(get_session)):
    yield SQLAlchemyUserDatabase(session, UserORM)


class UserManager(UUIDIDMixin, BaseUserManager[UserORM, uuid.UUID]):
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET


def _init_jwt_secret(secret: str) -> None:
    """Called by _boot() after Vault is ready to replace the dev fallback."""
    global SECRET
    SECRET = secret
    UserManager.reset_password_token_secret = secret
    UserManager.verification_token_secret = secret


async def get_user_manager(user_db=Depends(get_user_db)):
    yield UserManager(user_db)


bearer_transport = BearerTransport(tokenUrl="/auth/jwt/login")


def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=TOKEN_LIFETIME)


auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

fastapi_users = FastAPIUsers[UserORM, uuid.UUID](get_user_manager, [auth_backend])

current_active_user = fastapi_users.current_user(active=True)
current_admin_user  = fastapi_users.current_user(active=True, superuser=True)
