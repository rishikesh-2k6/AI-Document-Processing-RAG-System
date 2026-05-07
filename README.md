# 🧠 AI Document Processing & RAG System

> **Production-grade Retrieval-Augmented Generation backend** — upload PDFs, DOCX, and emails, then ask natural-language questions and receive cited, confidence-scored answers.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/LLM-Groq_llama--3.3--70b-orange.svg)](https://groq.com)
[![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-red.svg)](https://qdrant.tech)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📐 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        FastAPI (8000)                            │
│  /api/v1/auth  │  /api/v1/documents  │  /api/v1/query           │
└────────────────────────┬─────────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            │                         │
    ┌───────▼───────┐        ┌────────▼────────┐
    │  Celery Worker│        │   RAG Pipeline  │
    │  (ingestion)  │        │                 │
    └───────┬───────┘        │ Embed → Retrieve│
            │                │ → Rerank → Gen  │
    ┌───────▼───────┐        └────────┬────────┘
    │  Parse → Chunk│                 │
    │  → Embed      │        ┌────────▼────────┐
    └───────┬───────┘        │   Qdrant (6333) │
            │                │  cosine + BM25  │
    ┌───────▼──────────────┐ │  Hybrid + RRF   │
    │  PostgreSQL (5432)   │ └─────────────────┘
    │  Documents, Chunks,  │
    │  Users, QueryLogs    │
    └──────────────────────┘
    ┌──────────────────────┐
    │    Redis (6379)      │
    │  Cache + Progress    │
    │  + Metrics counters  │
    └──────────────────────┘
```

---

## ✨ Resume-Worthy Features

| Feature | Implementation |
|---|---|
| **Hybrid Search** | Dense (cosine) + BM25 sparse, fused by Reciprocal Rank Fusion |
| **Chunk-level Citations** | Every answer includes filename, page number, and 200-char snippet |
| **Confidence Scoring** | Avg cosine similarity of top-k chunks; warning if < 0.4 |
| **Ingestion Observability** | Celery progress events (0→100%) via SSE endpoint |
| **Redis Query Cache** | SHA-256(query+filters) → TTL 1hr; cache hit rate metric |
| **Rate Limiting** | slowapi: 10 req/min per IP on `/query` routes |
| **JWT Auth** | HS256 access (30min) + refresh (7d) tokens |
| **Async Everything** | FastAPI + asyncpg + async Qdrant client |
| **Structured Logging** | structlog with correlation IDs on every request |
| **Metrics Endpoint** | Query volume, cache hit %, vector count → Grafana-ready |

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Groq API Key (free at [console.groq.com](https://console.groq.com))
- Ollama (for embeddings) — optional, fallback included

### 1. Clone & Configure

```bash
git clone https://github.com/rishikesh-2k6/AI-Document-Processing-RAG-System.git
cd AI-Document-Processing-RAG-System/rag-doc-system

# Copy and fill environment variables
cp .env.example .env
# Edit .env — set GROQ_API_KEY and JWT_SECRET_KEY at minimum
```

### 2. Start the Full Stack

```bash
docker compose up --build
```

This starts:
| Service | Port | Purpose |
|---|---|---|
| FastAPI | 8000 | REST API + Swagger UI |
| PostgreSQL | 5432 | Document metadata |
| Redis | 6379 | Cache + task queue |
| Qdrant | 6333 | Vector storage |
| Celery Worker | — | Background ingestion |
| Flower | 5555 | Celery monitoring |

### 3. Verify

```bash
curl http://localhost:8000/api/v1/health
# {"status":"ok","version":"1.0.0","environment":"production"}
```

Open Swagger UI: **http://localhost:8000/api/v1/docs**

---

## 📡 API Reference

### Auth

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/auth/register` | Register new account |
| `POST` | `/api/v1/auth/login` | Get access + refresh tokens |
| `POST` | `/api/v1/auth/refresh` | Refresh expired access token |

### Documents

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/documents/upload` | Upload PDF/DOCX/EML → returns `job_id` |
| `GET` | `/api/v1/documents/` | Paginated document list |
| `GET` | `/api/v1/documents/{id}` | Document metadata + chunk count |
| `DELETE` | `/api/v1/documents/{id}` | Delete doc + vectors from Qdrant |
| `GET` | `/api/v1/documents/{id}/status` | Ingestion job status |
| `GET` | `/api/v1/documents/{id}/progress` | SSE ingestion progress stream |

### Query

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/query/ask` | RAG query → cited answer |
| `POST` | `/api/v1/query/summarize` | Bullet-point document summary |

### Ops

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Liveness probe |
| `GET` | `/api/v1/metrics` | Stats for Grafana |

---

## 🔄 RAG Pipeline Walkthrough

```
User Query
    │
    ▼
Embed query (nomic-embed-text / Ollama)
    │
    ├──► Dense vector search (Qdrant, cosine, top-25)
    └──► BM25 sparse search (rank-bm25 on retrieved texts)
              │
              ▼
         Reciprocal Rank Fusion (RRF, k=60)
              │
              ▼
    Optional: Cross-encoder reranking (Groq llama-3.1-8b)
              │
              ▼
    Top-5 chunks → build context block with citations
              │
              ▼
    Groq llama-3.3-70b-versatile with strict RAG prompt
              │
              ▼
    Answer + SourceChunks + ConfidenceScore
              │
              ▼
    Cache result in Redis (TTL 1hr)
```

---

## 📋 Example Request/Response

**POST /api/v1/query/ask**
```json
{
  "query": "What are the key financial risks mentioned in the Q4 report?",
  "filters": { "file_type": "pdf" },
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "The Q4 report identifies three key financial risks: (1) supply chain disruptions [Source: Q4_2024.pdf, Page 12], (2) foreign exchange volatility affecting margins by ~3.2% [Source: Q4_2024.pdf, Page 15], and (3) rising interest rates on long-term debt obligations [Source: Q4_2024.pdf, Page 18].",
  "source_chunks": [
    {
      "document_id": "550e8400-...",
      "filename": "Q4_2024.pdf",
      "page_number": 12,
      "chunk_index": 47,
      "snippet": "Supply chain disruptions remained a primary concern, with logistics costs increasing 18% year-over-year...",
      "score": 0.8934
    }
  ],
  "confidence_score": 0.87,
  "low_confidence_warning": false,
  "cache_hit": false,
  "latency_ms": 1842,
  "model_used": "llama-3.3-70b-versatile"
}
```

---

## 🏗️ Project Structure

```
rag-doc-system/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── auth.py         # JWT auth endpoints
│   │   │   ├── documents.py    # Upload, list, delete, SSE progress
│   │   │   └── query.py        # Ask + summarize with caching
│   │   └── deps.py             # Shared dependencies (auth guard)
│   ├── core/
│   │   ├── config.py           # Pydantic Settings v2
│   │   ├── security.py         # JWT + bcrypt
│   │   └── logging.py          # structlog JSON/console
│   ├── ingestion/
│   │   ├── parsers.py          # PDF, DOCX, EML parsers
│   │   ├── chunker.py          # Recursive character splitter
│   │   └── embedder.py         # Batched embedding + cache
│   ├── retrieval/
│   │   ├── vector_store.py     # Qdrant hybrid search + RRF
│   │   └── reranker.py         # LLM cross-encoder reranker
│   ├── generation/
│   │   ├── answer.py           # RAG QA chain
│   │   └── summarizer.py       # Bullet summarizer
│   ├── db/
│   │   ├── models.py           # SQLAlchemy 2.0 ORM
│   │   └── session.py          # Async engine + sessions
│   ├── cache/
│   │   └── redis_client.py     # Async Redis wrapper
│   ├── tasks/
│   │   ├── celery_app.py       # Celery factory
│   │   └── ingest_task.py      # 6-stage ingestion pipeline
│   └── main.py                 # App factory
├── tests/
│   ├── unit/                   # Parser, chunker, security tests
│   └── integration/            # API endpoint tests
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── .env.example
```

---

## ⚙️ Environment Variables

| Variable | Default | Required | Description |
|---|---|---|---|
| `GROQ_API_KEY` | — | ✅ | Groq API key |
| `JWT_SECRET_KEY` | — | ✅ | JWT signing secret (≥32 chars) |
| `DATABASE_URL` | `postgresql+asyncpg://...` | ✅ | Async Postgres URL |
| `REDIS_URL` | `redis://:pass@localhost:6379/0` | ✅ | Redis connection |
| `QDRANT_HOST` | `localhost` | | Qdrant hostname |
| `CHUNK_SIZE` | `512` | | Token limit per chunk |
| `CHUNK_OVERLAP` | `64` | | Overlap tokens between chunks |
| `DEFAULT_TOP_K` | `5` | | Chunks retrieved per query |
| `CONFIDENCE_THRESHOLD` | `0.4` | | Low-confidence warning threshold |

---

## 🧪 Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests with coverage
pytest

# Run only unit tests
pytest tests/unit/ -v

# Run with coverage report
pytest --cov=app --cov-report=html
```

---

## 📊 Performance Targets

| Metric | Target | Strategy |
|---|---|---|
| Cached query p95 | < 800ms | Redis TTL cache |
| Uncached query p95 | < 3s | Hybrid search + async Groq |
| Ingestion throughput | ~10 pages/sec | Celery workers + batched embedding |
| Vector search | < 50ms | Qdrant HNSW index |

---

## 🛠️ Tech Stack

- **API**: FastAPI 0.110, Pydantic v2, uvicorn + uvloop
- **AI**: Groq llama-3.3-70b-versatile, nomic-embed-text (Ollama)
- **Vector DB**: Qdrant with HNSW index, cosine similarity
- **Search**: Dense + BM25 (rank-bm25), fused by RRF
- **Database**: PostgreSQL + SQLAlchemy 2.0 async + asyncpg
- **Cache**: Redis (aioredis) — queries + embeddings + progress
- **Queue**: Celery + Redis broker, Flower monitoring
- **Auth**: JWT (python-jose) + bcrypt (passlib)
- **Logging**: structlog with JSON output in production
- **Rate Limiting**: slowapi

---

## 📝 License

MIT © 2024 Rishikesh
