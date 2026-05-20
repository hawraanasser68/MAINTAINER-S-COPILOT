from fastapi import APIRouter

from app.domain.models import HealthStatus
from app.infra import database, minio_client, redis_client, vault
from app.infra.tracing import create_span

router = APIRouter()


@router.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    with create_span("health.check"):
        vault_ok = vault.check_connectivity()
        db_ok = await database.check_connectivity()
        redis_ok = await redis_client.check_connectivity()
        minio_ok = minio_client.check_connectivity()

    status = HealthStatus(
        status="ok" if all([vault_ok, db_ok, redis_ok, minio_ok]) else "degraded",
        vault="reachable" if vault_ok else "unreachable",
        db="reachable" if db_ok else "unreachable",
        redis="reachable" if redis_ok else "unreachable",
        minio="reachable" if minio_ok else "unreachable",
        tracing="configured",
    )
    return status
