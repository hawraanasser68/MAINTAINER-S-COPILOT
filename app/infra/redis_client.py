from redis.asyncio import Redis

from app.domain.exceptions import InfrastructureError

_redis: Redis | None = None


def init_redis(url: str) -> None:
    global _redis
    _redis = Redis.from_url(url, decode_responses=True)


def _get() -> Redis:
    if _redis is None:
        raise InfrastructureError("redis", "not initialised — call init_redis() first")
    return _redis


async def get(key: str) -> str | None:
    return await _get().get(key)


async def set(key: str, value: str, ttl: int) -> None:
    """ttl in seconds — always required, never implicit."""
    await _get().setex(key, ttl, value)


async def delete(key: str) -> None:
    await _get().delete(key)


async def check_connectivity() -> bool:
    if _redis is None:
        return False
    try:
        await _get().ping()
        return True
    except Exception:
        return False
