"""FastAPI application factory with lifespan, middleware, and router registration."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.routes import auth, documents, query
from app.cache.redis_client import get_redis_cache
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.db.session import create_tables
from app.retrieval.vector_store import get_vector_store

log = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle."""
    configure_logging()
    log.info("app_starting", name=settings.app_name, env=settings.environment)

    # Ensure DB tables exist (dev only — use Alembic migrations in production)
    if settings.environment != "production":
        await create_tables()

    # Ensure Qdrant collection exists
    vs = get_vector_store()
    await vs.ensure_collection()

    # Verify Redis connectivity
    cache = get_redis_cache()
    redis_ok = await cache.ping()
    log.info("redis_connectivity", ok=redis_ok)

    log.info("app_ready", docs_url=f"{settings.api_v1_prefix}/docs")
    yield

    log.info("app_shutting_down")


# ── App factory ───────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        description=(
            "Production-grade AI Document Processing & RAG System. "
            "Upload documents, query with citations, summarize instantly."
        ),
        version="1.0.0",
        docs_url=f"{settings.api_v1_prefix}/docs",
        redoc_url=f"{settings.api_v1_prefix}/redoc",
        openapi_url=f"{settings.api_v1_prefix}/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else ["https://yourdomain.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Rate limiting ──────────────────────────────────────────────────────────
    from app.api.routes.query import limiter

    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── Correlation ID + request logging middleware ────────────────────────────
    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next) -> Response:  # type: ignore
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            correlation_id=correlation_id,
            method=request.method,
            path=request.url.path,
        )
        start = time.monotonic()
        response: Response = await call_next(request)
        latency_ms = int((time.monotonic() - start) * 1000)
        log.info(
            "http_request",
            status_code=response.status_code,
            latency_ms=latency_ms,
        )
        response.headers["X-Correlation-ID"] = correlation_id
        return response

    # ── Routers ───────────────────────────────────────────────────────────────
    prefix = settings.api_v1_prefix
    app.include_router(auth.router, prefix=prefix)
    app.include_router(documents.router, prefix=prefix)
    app.include_router(query.router, prefix=prefix)

    # ── Health & Metrics ──────────────────────────────────────────────────────
    _register_utility_routes(app)

    return app


# ── Health / Metrics endpoints ────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


class MetricsResponse(BaseModel):
    document_count: int
    query_count: int
    cache_hits: int
    cache_hit_rate: float
    vector_point_count: int


def _register_utility_routes(app: FastAPI) -> None:
    prefix = settings.api_v1_prefix

    @app.get(
        f"{prefix}/health",
        response_model=HealthResponse,
        tags=["Ops"],
        summary="Liveness probe",
    )
    async def health() -> HealthResponse:
        """Return 200 OK if the service is alive. Use for load balancer health checks."""
        return HealthResponse(
            status="ok",
            version="1.0.0",
            environment=settings.environment,
        )

    @app.get(
        f"{prefix}/metrics",
        response_model=MetricsResponse,
        tags=["Ops"],
        summary="System metrics for Grafana dashboards",
    )
    async def metrics() -> MetricsResponse:
        """Return aggregate stats: document count, query volume, cache hit rate.

        Designed to feed a Grafana dashboard via Prometheus scraping or direct polling.
        """
        from sqlalchemy import func, select

        from app.db.models import Document
        from app.db.session import AsyncSessionLocal

        cache = get_redis_cache()
        vs = get_vector_store()

        # DB stats
        async with AsyncSessionLocal() as db:
            doc_count: int = (
                await db.execute(select(func.count(Document.id)))
            ).scalar_one()

        # Redis counters
        query_count_raw = await cache.get("metrics:query_count")
        cache_hits_raw = await cache.get("metrics:cache_hits")
        query_count = int(query_count_raw or 0)
        cache_hits = int(cache_hits_raw or 0)
        cache_hit_rate = round(cache_hits / max(query_count, 1), 4)

        # Qdrant stats
        try:
            info = await vs.get_collection_info()
            vector_count = info.get("points_count", 0) or 0
        except Exception:
            vector_count = 0

        return MetricsResponse(
            document_count=doc_count,
            query_count=query_count,
            cache_hits=cache_hits,
            cache_hit_rate=cache_hit_rate,
            vector_point_count=vector_count,
        )


# ── Entry point ───────────────────────────────────────────────────────────────

app = create_app()
