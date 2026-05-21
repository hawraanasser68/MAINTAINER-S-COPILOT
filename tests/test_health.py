from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def mock_boot(monkeypatch):
    """Bypass _boot() so tests don't need real infra."""
    monkeypatch.setattr("app.main._boot", lambda log: None)


@pytest.fixture
async def client(mock_boot):
    from app.main import create_app
    async with AsyncClient(
        transport=ASGITransport(app=create_app()), base_url="http://test"
    ) as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_all_ok(client):
    with (
        patch("app.api.health.vault.check_connectivity", return_value=True),
        patch("app.api.health.database.check_connectivity", new_callable=AsyncMock, return_value=True),  # noqa: E501
        patch("app.api.health.redis_client.check_connectivity", new_callable=AsyncMock, return_value=True),  # noqa: E501
        patch("app.api.health.minio_client.check_connectivity", return_value=True),
    ):
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["vault"] == "reachable"
    assert body["db"] == "reachable"
    assert body["redis"] == "reachable"
    assert body["minio"] == "reachable"


@pytest.mark.asyncio
async def test_health_degraded_when_db_down(client):
    with (
        patch("app.api.health.vault.check_connectivity", return_value=True),
        patch("app.api.health.database.check_connectivity", new_callable=AsyncMock, return_value=False),  # noqa: E501
        patch("app.api.health.redis_client.check_connectivity", new_callable=AsyncMock, return_value=True),  # noqa: E501
        patch("app.api.health.minio_client.check_connectivity", return_value=True),
    ):
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["db"] == "unreachable"


@pytest.mark.asyncio
async def test_error_response_shape(client):
    """Unhandled routes return structured error, never a stack trace."""
    response = await client.get("/nonexistent-route")
    assert response.status_code == 404
    # FastAPI's default 404 — just confirm no raw exception text leaks
    assert "traceback" not in response.text.lower()


@pytest.mark.asyncio
async def test_request_id_in_response_headers(client):
    with (
        patch("app.api.health.vault.check_connectivity", return_value=True),
        patch("app.api.health.database.check_connectivity", new_callable=AsyncMock, return_value=True),  # noqa: E501
        patch("app.api.health.redis_client.check_connectivity", new_callable=AsyncMock, return_value=True),  # noqa: E501
        patch("app.api.health.minio_client.check_connectivity", return_value=True),
    ):
        response = await client.get("/health")

    assert "x-request-id" in response.headers
    assert len(response.headers["x-request-id"]) == 36  # UUID format
