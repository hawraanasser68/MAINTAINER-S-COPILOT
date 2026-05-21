import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.auth import _init_jwt_secret, auth_backend, fastapi_users
from app.api.chat import router as chat_router
from app.api.exceptions import register_exception_handlers
from app.api.health import router as health_router
from app.api.memory import router as memory_router
from app.api.middleware import RequestIDMiddleware
from app.api.rag import router as rag_router
from app.api.widgets import router as widgets_router
from app.domain.exceptions import InfrastructureError
from app.infra import database, minio_client, redis_client, vault
from app.infra.bm25_index import BM25Index, load_index
from app.infra.tracing import setup_tracing


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
    )


def _check_eval_thresholds() -> None:
    """Refuse to boot if any threshold is 0 or the file is missing."""
    path = Path("eval_thresholds.yaml")
    if not path.exists():
        raise InfrastructureError("eval_thresholds", "eval_thresholds.yaml not found")

    thresholds = yaml.safe_load(path.read_text())

    def _walk(obj: object, path_: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{path_}.{k}" if path_ else k)
        elif isinstance(obj, int | float) and obj == 0:
            raise InfrastructureError(
                "eval_thresholds",
                f"threshold '{path_}' is 0 — fill in eval_thresholds.yaml before starting",
            )

    _walk(thresholds)


def _boot(log: structlog.BoundLogger) -> None:
    """Resolve all secrets, initialise infra adapters, run connectivity checks.
    Raises InfrastructureError on any failure — app refuses to start."""

    # 1. Vault must be reachable before anything else
    log.info("boot.vault.check")
    if not vault.check_connectivity():
        raise InfrastructureError("vault", "unreachable at startup")

    # 2. Seed dev secrets (no-op if already seeded or in prod)
    vault.seed_dev_secrets()

    # 3. Resolve secrets — ALL secrets come from Vault, never from .env directly
    log.info("boot.secrets.resolving")
    db_password = vault.get_secret("db/password")
    redis_password = vault.get_secret("redis/password")
    minio_access_key = vault.get_secret("minio/access_key")
    minio_secret_key = vault.get_secret("minio/secret_key")
    otlp_endpoint = vault.get_secret("tracing/endpoint")
    groq_api_key = vault.get_secret("llm/groq_api_key")
    os.environ["GROQ_API_KEY"] = groq_api_key

    # 3b. JWT signing key — replaces the module-level fallback string
    try:
        jwt_secret = vault.get_secret("jwt/signing_key")
        if jwt_secret:
            _init_jwt_secret(jwt_secret)
            log.info("boot.jwt.configured_from_vault")
    except Exception as exc:
        log.warning("boot.jwt.using_dev_fallback", error=str(exc))

    # Anthropic key is optional — if absent the agent falls back to Groq automatically.
    try:
        anthropic_api_key = vault.get_secret("llm/anthropic_api_key")
        if anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
            log.info("boot.anthropic.configured")
        else:
            log.warning("boot.anthropic.key_empty_groq_fallback_active")
    except Exception as exc:
        log.warning("boot.anthropic.unavailable_groq_fallback_active", error=str(exc))

    # 4. Tracing — wire before anything else so all subsequent spans are captured
    log.info("boot.tracing.setup", endpoint=otlp_endpoint)
    setup_tracing("maintainers-copilot-api", otlp_endpoint)

    # 5. Database
    db_host = os.environ.get("DB_HOST", "db")
    db_port = os.environ.get("DB_PORT", "5432")
    db_user = os.environ.get("DB_USER", "postgres")
    db_name = os.environ.get("DB_NAME", "maintainers_copilot")
    db_url = f"postgresql+asyncpg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    database.init_db(db_url)

    # 6. Redis
    redis_url = os.environ.get("REDIS_URL", "redis://redis:6379")
    if redis_password:
        redis_url = redis_url.replace("redis://", f"redis://:{redis_password}@")
    redis_client.init_redis(redis_url)

    # 7. MinIO
    minio_endpoint = os.environ.get("MINIO_ENDPOINT", "http://minio:9000")
    minio_client.init_minio(minio_endpoint, minio_access_key, minio_secret_key)

    # 8. BM25 index — load from MinIO into memory
    log.info("boot.bm25_index.loading")
    try:
        data = minio_client.download("models", "bm25/index.pkl")
        load_index(BM25Index.from_bytes(data))
        log.info("boot.bm25_index.loaded")
    except Exception as exc:
        log.warning("boot.bm25_index.unavailable", error=str(exc))

    # 9. Eval thresholds — must have non-zero values
    log.info("boot.eval_thresholds.check")
    _check_eval_thresholds()

    log.info("boot.complete")


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    log = structlog.get_logger("boot")
    _configure_logging()
    _boot(log)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Maintainer's Copilot",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )

    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8090", "http://localhost:8080"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(rag_router)
    app.include_router(chat_router)
    app.include_router(memory_router)
    app.include_router(widgets_router)

    # fastapi-users auth routes
    import uuid as _uuid

    from fastapi_users import schemas as fu_schemas

    class UserRead(fu_schemas.BaseUser[_uuid.UUID]):
        pass

    class UserCreate(fu_schemas.BaseUserCreate):
        pass

    class UserUpdate(fu_schemas.BaseUserUpdate):
        pass

    app.include_router(
        fastapi_users.get_auth_router(auth_backend),
        prefix="/auth/jwt",
        tags=["auth"],
    )
    app.include_router(
        fastapi_users.get_register_router(UserRead, UserCreate),
        prefix="/auth",
        tags=["auth"],
    )
    app.include_router(
        fastapi_users.get_users_router(UserRead, UserUpdate),
        prefix="/users",
        tags=["users"],
    )

    return app


app = create_app()
