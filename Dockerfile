# ─── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir build && \
    pip install --no-cache-dir tomli && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r <(python -c "
import sys
sys.path.insert(0, '.')
try:
    import tomllib
except ImportError:
    import tomli as tomllib
with open('pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
deps = data.get('project', {}).get('dependencies', [])
print('\n'.join(deps))
")

# ─── Stage 2: Production ──────────────────────────────────────────────────────
FROM python:3.11-slim AS production

WORKDIR /app

# Runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install wheels from builder
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/* || \
    pip install --no-cache-dir \
        fastapi==0.110.3 \
        uvicorn[standard]==0.29.0 \
        uvloop==0.19.0 \
        pydantic==2.7.1 \
        pydantic-settings==2.2.1 \
        sqlalchemy==2.0.30 \
        asyncpg==0.29.0 \
        alembic==1.13.1 \
        celery[redis]==5.4.0 \
        redis==5.0.4 \
        qdrant-client==1.9.1 \
        groq==0.9.0 \
        pdfplumber==0.11.0 \
        python-docx==1.1.2 \
        python-multipart==0.0.9 \
        python-jose[cryptography]==3.3.0 \
        passlib[bcrypt]==1.7.4 \
        structlog==24.1.0 \
        slowapi==0.1.9 \
        httpx==0.27.0 \
        rank-bm25==0.2.2 \
        tiktoken==0.7.0 \
        tenacity==8.3.0 \
        flower==2.0.1

# Copy application source
COPY app/ ./app/

# Create upload directory
RUN mkdir -p /app/uploads

# Non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
RUN chown -R appuser:appgroup /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1
