"""Redis cache client with async get/set/delete helpers."""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


class RedisCache:
    """Thin async wrapper around redis.asyncio with serialization helpers.

    Args:
        url: Redis connection URL (includes auth if needed).
    """

    def __init__(self, url: str = settings.redis_url) -> None:
        self._pool = aioredis.ConnectionPool.from_url(
            url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
        )
        self._client: aioredis.Redis = aioredis.Redis(connection_pool=self._pool)

    # ── Core ops ──────────────────────────────────────────────────────────────

    async def get(self, key: str) -> str | None:
        """Retrieve raw string value for *key*, or None if missing."""
        try:
            return await self._client.get(key)
        except Exception as exc:
            log.warning("redis_get_error", key=key, error=str(exc))
            return None

    async def set(self, key: str, value: str, ttl: int = settings.cache_ttl_seconds) -> bool:
        """Set *key* = *value* with TTL in seconds.

        Returns True on success, False on failure.
        """
        try:
            await self._client.setex(key, ttl, value)
            return True
        except Exception as exc:
            log.warning("redis_set_error", key=key, error=str(exc))
            return False

    async def delete(self, key: str) -> int:
        """Delete *key*. Returns number of keys deleted."""
        try:
            return await self._client.delete(key)
        except Exception as exc:
            log.warning("redis_delete_error", key=key, error=str(exc))
            return 0

    async def exists(self, key: str) -> bool:
        """Return True if *key* exists in Redis."""
        try:
            return bool(await self._client.exists(key))
        except Exception as exc:
            log.warning("redis_exists_error", key=key, error=str(exc))
            return False

    # ── JSON helpers ──────────────────────────────────────────────────────────

    async def get_json(self, key: str) -> Any | None:
        """Retrieve and deserialize a JSON-encoded value."""
        raw = await self._get_raw(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def set_json(
        self, key: str, value: Any, ttl: int = settings.cache_ttl_seconds
    ) -> bool:
        """Serialize *value* to JSON and store with TTL."""
        try:
            return await self.set(key, json.dumps(value, default=str), ttl=ttl)
        except Exception as exc:
            log.warning("redis_set_json_error", key=key, error=str(exc))
            return False

    async def _get_raw(self, key: str) -> str | None:
        return await self.get(key)

    # ── Progress helpers (for SSE ingestion progress) ─────────────────────────

    async def set_progress(self, job_id: str, progress: int, ttl: int = 86400) -> None:
        """Publish ingestion progress (0-100) for a job."""
        await self.set(f"progress:{job_id}", str(progress), ttl=ttl)

    async def get_progress(self, job_id: str) -> int:
        """Get current ingestion progress (0-100) for a job."""
        raw = await self.get(f"progress:{job_id}")
        if raw is None:
            return 0
        try:
            return int(raw)
        except ValueError:
            return 0

    # ── Metrics helpers ───────────────────────────────────────────────────────

    async def increment(self, key: str, ttl: int = 86400) -> int:
        """Atomically increment a counter, setting TTL on first creation."""
        try:
            val = await self._client.incr(key)
            if val == 1:
                await self._client.expire(key, ttl)
            return val
        except Exception as exc:
            log.warning("redis_incr_error", key=key, error=str(exc))
            return 0

    async def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            return await self._client.ping()
        except Exception:
            return False


# ── Singleton ─────────────────────────────────────────────────────────────────
_redis_cache: RedisCache | None = None


def get_redis_cache() -> RedisCache:
    """Return the module-level RedisCache singleton."""
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = RedisCache()
    return _redis_cache
