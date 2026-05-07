"""Local in-memory cache and state manager to replace Redis."""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any

from cachetools import TTLCache

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

# Query Cache (Thread-safe TTL Cache)
# Stores max 1000 items, expires based on CACHE_TTL_SECONDS
_query_cache: TTLCache[str, str] = TTLCache(maxsize=1000, ttl=settings.cache_ttl_seconds)

# State Store for Progress Tracking (Job ID -> Status Dict)
# In-memory dictionary for keeping track of background tasks
_job_state: dict[str, dict[str, Any]] = {}


def generate_cache_key(prefix: str, **kwargs: Any) -> str:
    """Generate a stable SHA-256 hash key from a dictionary of arguments."""
    ordered_json = json.dumps(kwargs, sort_keys=True)
    hash_str = hashlib.sha256(ordered_json.encode()).hexdigest()
    return f"{prefix}:{hash_str}"


async def get_cache(key: str) -> str | None:
    """Get a value from the TTL cache."""
    return _query_cache.get(key)


async def set_cache(key: str, value: str) -> None:
    """Set a value in the TTL cache."""
    _query_cache[key] = value


# ── Ingestion Job State ───────────────────────────────────────────────────────

async def set_job_progress(job_id: str, status: str, progress: int, message: str = "") -> None:
    """Update the status and progress of an ingestion job."""
    _job_state[job_id] = {
        "status": status,
        "progress": progress,
        "message": message,
        "updated_at": datetime.utcnow().isoformat()
    }


async def get_job_progress(job_id: str) -> dict[str, Any] | None:
    """Get the current progress of an ingestion job."""
    return _job_state.get(job_id)

