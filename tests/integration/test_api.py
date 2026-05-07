"""Integration tests for auth and document API endpoints."""

from __future__ import annotations

import io
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import Base
from app.main import app

# ── In-memory SQLite engine for tests ────────────────────────────────────────
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DB_URL, echo=False)
TestSessionLocal = async_sessionmaker(
    bind=test_engine, class_=AsyncSession, expire_on_commit=False
)


async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
    async with TestSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_db() -> AsyncGenerator[None, None]:
    """Create tables once for the test session."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture()
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Provide an async test client with DB override."""
    from app.db.session import get_db

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


# ── Auth Tests ────────────────────────────────────────────────────────────────


class TestAuthEndpoints:
    """Integration tests for /api/v1/auth endpoints."""

    @pytest.mark.asyncio
    async def test_register_success(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@test.com",
                "password": "Secure123!",
                "full_name": "Test User",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_register_duplicate_email(self, client: AsyncClient) -> None:
        email = "dup@test.com"
        payload = {"email": email, "password": "pass123"}
        await client.post("/api/v1/auth/register", json=payload)
        resp = await client.post("/api/v1/auth/register", json=payload)
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_login_success(self, client: AsyncClient) -> None:
        email = "login@test.com"
        password = "mypassword"
        await client.post("/api/v1/auth/register", json={"email": email, "password": password})

        resp = await client.post("/api/v1/auth/login", json={"email": email, "password": password})
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/auth/login",
            json={"email": "nobody@test.com", "password": "wrong"},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_refresh_token(self, client: AsyncClient) -> None:
        reg = await client.post(
            "/api/v1/auth/register",
            json={"email": "refresh@test.com", "password": "pass123"},
        )
        refresh_token = reg.json()["refresh_token"]
        resp = await client.post(
            "/api/v1/auth/refresh", json={"refresh_token": refresh_token}
        )
        assert resp.status_code == 200
        assert "access_token" in resp.json()


# ── Health Tests ──────────────────────────────────────────────────────────────


class TestHealthEndpoints:
    """Tests for health and utility endpoints."""

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
