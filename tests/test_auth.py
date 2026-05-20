"""
Auth tests — register, login, JWT validation, role enforcement.

These tests run against a live stack (Postgres + Redis + Vault).
They are skipped automatically if the API server is not reachable.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

API_URL = "http://localhost:8000"


def _is_api_up() -> bool:
    try:
        resp = httpx.get(f"{API_URL}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _is_api_up(), reason="API server not running")


@pytest.fixture(scope="module")
def test_user() -> dict:
    email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    password = "Test1234!"
    return {"email": email, "password": password}


@pytest.fixture(scope="module")
def registered_user(test_user: dict) -> dict:
    resp = httpx.post(
        f"{API_URL}/auth/register",
        json={"email": test_user["email"], "password": test_user["password"]},
        timeout=10,
    )
    assert resp.status_code in (200, 201), f"Registration failed: {resp.text}"
    return test_user


@pytest.fixture(scope="module")
def jwt_token(registered_user: dict) -> str:
    resp = httpx.post(
        f"{API_URL}/auth/jwt/login",
        data={
            "username": registered_user["email"],
            "password": registered_user["password"],
        },
        timeout=10,
    )
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    return resp.json()["access_token"]


def test_register_creates_user(registered_user: dict) -> None:
    """Registering a new user returns 200 or 201."""
    resp = httpx.post(
        f"{API_URL}/auth/register",
        json={
            "email": f"dup_{uuid.uuid4().hex[:8]}@example.com",
            "password": "Test1234!",
        },
        timeout=10,
    )
    assert resp.status_code in (200, 201)


def test_duplicate_email_rejected(registered_user: dict) -> None:
    """Registering with the same email returns 400."""
    resp = httpx.post(
        f"{API_URL}/auth/register",
        json={"email": registered_user["email"], "password": "AnotherPass1!"},
        timeout=10,
    )
    assert resp.status_code == 400


def test_login_returns_jwt(jwt_token: str) -> None:
    """Login returns a non-empty Bearer token."""
    assert jwt_token and len(jwt_token) > 20


def test_wrong_password_rejected(registered_user: dict) -> None:
    """Login with wrong password returns 400."""
    resp = httpx.post(
        f"{API_URL}/auth/jwt/login",
        data={"username": registered_user["email"], "password": "WrongPass!"},
        timeout=10,
    )
    assert resp.status_code == 400


def test_authenticated_request_succeeds(jwt_token: str) -> None:
    """A request with a valid JWT to /chat/history returns 200."""
    convo_id = str(uuid.uuid4())
    resp = httpx.get(
        f"{API_URL}/chat/history/{convo_id}",
        headers={"Authorization": f"Bearer {jwt_token}"},
        timeout=10,
    )
    assert resp.status_code == 200


def test_unauthenticated_request_rejected() -> None:
    """A request without a JWT to /chat returns 401."""
    resp = httpx.post(
        f"{API_URL}/chat",
        json={"message": "hello"},
        timeout=10,
    )
    assert resp.status_code == 401


def test_admin_route_requires_superuser(jwt_token: str) -> None:
    """A regular user cannot access /widgets (admin-only)."""
    resp = httpx.post(
        f"{API_URL}/widgets",
        json={"allowed_origins": [], "greeting": "hi", "enabled_tools": [], "theme": {}},
        headers={"Authorization": f"Bearer {jwt_token}"},
        timeout=10,
    )
    assert resp.status_code in (401, 403)
