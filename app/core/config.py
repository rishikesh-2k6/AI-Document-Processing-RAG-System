"""Application configuration using Pydantic Settings v2."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_name: str = Field(default="RAG Document System", alias="APP_NAME")
    environment: Literal["development", "production", "testing"] = Field(
        default="development", alias="ENVIRONMENT"
    )
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    api_v1_prefix: str = "/api/v1"

    # ── Security ──────────────────────────────────────────────────────────────
    jwt_secret_key: str = Field(..., alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(
        default=30, alias="JWT_ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    jwt_refresh_token_expire_days: int = Field(
        default=7, alias="JWT_REFRESH_TOKEN_EXPIRE_DAYS"
    )

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://raguser:ragpassword@localhost:5432/ragdb",
        alias="DATABASE_URL",
    )
    db_pool_size: int = Field(default=10, alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=20, alias="DB_MAX_OVERFLOW")
    db_pool_timeout: int = Field(default=30, alias="DB_POOL_TIMEOUT")

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = Field(
        default="redis://:redispassword@localhost:6379/0", alias="REDIS_URL"
    )
    cache_ttl_seconds: int = Field(default=3600, alias="CACHE_TTL_SECONDS")

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_collection_name: str = Field(
        default="documents", alias="QDRANT_COLLECTION_NAME"
    )

    # ── Groq AI ───────────────────────────────────────────────────────────────
    groq_api_key: str = Field(..., alias="GROQ_API_KEY")
    groq_chat_model: str = Field(
        default="llama-3.3-70b-versatile", alias="GROQ_CHAT_MODEL"
    )
    groq_summarize_model: str = Field(
        default="llama-3.3-70b-versatile", alias="GROQ_SUMMARIZE_MODEL"
    )
    groq_embedding_model: str = Field(
        default="nomic-embed-text", alias="GROQ_EMBEDDING_MODEL"
    )

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")
    embedding_dimensions: int = Field(default=768, alias="EMBEDDING_DIMENSIONS")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    default_top_k: int = Field(default=5, alias="DEFAULT_TOP_K")
    confidence_threshold: float = Field(default=0.4, alias="CONFIDENCE_THRESHOLD")

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    rate_limit_query: str = Field(default="10/minute", alias="RATE_LIMIT_QUERY")

    # ── File Upload ───────────────────────────────────────────────────────────
    max_upload_size_mb: int = Field(default=50, alias="MAX_UPLOAD_SIZE_MB")
    upload_dir: Path = Field(default=Path("/app/uploads"), alias="UPLOAD_DIR")
    allowed_extensions: list[str] = Field(
        default=["pdf", "docx", "eml", "msg"], alias="ALLOWED_EXTENSIONS"
    )

    @field_validator("upload_dir", mode="before")
    @classmethod
    def create_upload_dir(cls, v: str | Path) -> Path:
        """Ensure upload directory exists."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def parse_extensions(cls, v: str | list[str]) -> list[str]:
        """Accept comma-separated string or list."""
        if isinstance(v, str):
            return [ext.strip().lower() for ext in v.split(",")]
        return v

    @property
    def max_upload_size_bytes(self) -> int:
        """Max file size in bytes."""
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def celery_broker_url(self) -> str:
        """Celery broker (Redis)."""
        return self.redis_url

    @property
    def celery_result_backend(self) -> str:
        """Celery result backend (Redis)."""
        return self.redis_url


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings singleton."""
    return Settings()


# Convenience module-level instance
settings: Settings = get_settings()
