"""Query routes: /ask (RAG QA) and /summarize, with Redis caching and rate limiting."""

from __future__ import annotations

import hashlib
import json
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, DBSession
from app.cache.local_cache import generate_cache_key, get_cache, set_cache
from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import Document, DocumentChunk, JobStatus, QueryLog
from app.generation.answer import RAGAnswerGenerator
from app.generation.summarizer import DocumentSummarizer
from app.ingestion.embedder import EmbeddingService
from app.retrieval.vector_store import get_vector_store

log = get_logger(__name__)

limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/query", tags=["Query"])


# ── Schemas ───────────────────────────────────────────────────────────────────


class QueryFilters(BaseModel):
    """Optional metadata filters for RAG queries."""

    document_id: str | None = None
    file_type: str | None = None


class AskRequest(BaseModel):
    """RAG query request payload."""

    query: str = Field(..., min_length=3, max_length=2000, description="Natural language question")
    filters: QueryFilters | None = None
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranker: bool = Field(default=False, description="Enable LLM cross-encoder reranking")


class SourceChunkOut(BaseModel):
    """Citation source chunk in the response."""

    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    snippet: str
    score: float


class AskResponse(BaseModel):
    """RAG answer response."""

    answer: str
    source_chunks: list[SourceChunkOut]
    confidence_score: float
    low_confidence_warning: bool
    cache_hit: bool
    latency_ms: int
    model_used: str


class SummarizeRequest(BaseModel):
    """Document summarization request."""

    document_id: str = Field(..., description="UUID of the document to summarize")


class SummarizeResponse(BaseModel):
    """Bullet-point summary response."""

    document_id: str
    filename: str
    bullet_points: list[str]
    model_used: str


# ── Cache key ─────────────────────────────────────────────────────────────────


def _make_cache_key(query: str, filters: QueryFilters | None, top_k: int) -> str:
    """Build deterministic SHA-256 cache key for a query."""
    raw = json.dumps(
        {
            "q": query,
            "f": filters.model_dump() if filters else None,
            "k": top_k,
        },
        sort_keys=True,
    )
    return "query:" + hashlib.sha256(raw.encode()).hexdigest()


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a natural language question over ingested documents",
)
@limiter.limit(settings.rate_limit_query)
async def ask(
    request: Request,
    payload: AskRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> AskResponse:
    """Perform hybrid RAG retrieval and generate a cited answer.

    - Results are cached in Redis (TTL = 1 hour) using query + filters as key.
    - Confidence score < 0.4 returns a low-confidence warning.
    - Each source chunk includes filename, page number, and a 200-char snippet.
    """
    start_ms = int(time.monotonic() * 1000)
    cache_key = generate_cache_key("ask", query=payload.query, filters=payload.filters)
    cached_response = await get_cache(cache_key)

    # ── Cache hit ─────────────────────────────────────────────────────────────
    if cached_response:
        latency = int(time.monotonic() * 1000) - start_ms
        log.info("query_cache_hit", query_preview=payload.query[:60])
        return AskResponse(**{**json.loads(cached_response), "cache_hit": True, "latency_ms": latency})

    # ── Embed query ───────────────────────────────────────────────────────────
    embedder = EmbeddingService()
    query_vectors = await embedder.embed_texts([payload.query])
    query_vector = query_vectors[0]

    # ── Retrieve chunks ───────────────────────────────────────────────────────
    vs = get_vector_store()
    filters = payload.filters.model_dump(exclude_none=True) if payload.filters else None

    chunks = await vs.hybrid_search(
        query=payload.query,
        query_vector=query_vector,
        top_k=payload.top_k * 2,  # fetch more for reranker
        filters=filters,
    )

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No relevant documents found for this query",
        )

    # ── Optional reranking ────────────────────────────────────────────────────
    if payload.use_reranker and chunks:
        from app.retrieval.reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        chunks = await reranker.rerank(payload.query, chunks, top_k=payload.top_k)
    else:
        chunks = chunks[: payload.top_k]

    # ── Generate answer ───────────────────────────────────────────────────────
    generator = RAGAnswerGenerator()
    rag_answer = await generator.generate(payload.query, chunks)

    latency = int(time.monotonic() * 1000) - start_ms

    # ── Build response ────────────────────────────────────────────────────────
    source_chunks_out = [
        SourceChunkOut(
            document_id=sc.document_id,
            filename=sc.filename,
            page_number=sc.page_number,
            chunk_index=sc.chunk_index,
            snippet=sc.snippet,
            score=sc.score,
        )
        for sc in rag_answer.source_chunks
    ]

    response_data = {
        "answer": rag_answer.answer,
        "source_chunks": [s.model_dump() for s in source_chunks_out],
        "confidence_score": rag_answer.confidence_score,
        "low_confidence_warning": rag_answer.low_confidence_warning,
        "cache_hit": False,
        "latency_ms": latency,
        "model_used": rag_answer.model_used,
    }

    # ── Cache result ──────────────────────────────────────────────────────────
    await set_cache(cache_key, json.dumps(response_data), ttl=settings.cache_ttl_seconds)

    # ── Log query to DB ───────────────────────────────────────────────────────
    log_entry = QueryLog(
        user_id=current_user.id,
        query_text=payload.query,
        filters=filters,
        top_k=payload.top_k,
        answer_text=rag_answer.answer,
        confidence_score=rag_answer.confidence_score,
        source_chunks=[s.model_dump() for s in source_chunks_out],
        cache_hit=False,
        latency_ms=latency,
    )
    db.add(log_entry)
    # Session auto-commits via get_db

    log.info(
        "query_answered",
        query_preview=payload.query[:60],
        confidence=rag_answer.confidence_score,
        latency_ms=latency,
    )

    return AskResponse(**response_data)


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="Summarize a document into 3-5 bullet points",
)
async def summarize_document(
    payload: SummarizeRequest,
    current_user: CurrentUser,
    db: DBSession,
) -> SummarizeResponse:
    """Generate a concise 3-5 bullet-point summary of an ingested document.

    The document must have already been successfully ingested (status=done).
    """
    try:
        doc_id = uuid.UUID(payload.document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document_id format")

    doc = await db.get(Document, doc_id)
    if not doc or doc.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc.job_status != JobStatus.DONE:
        raise HTTPException(
            status_code=409,
            detail=f"Document ingestion status is '{doc.job_status.value}' — must be 'done'",
        )

    # Fetch chunks from DB
    result = await db.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == doc_id)
        .order_by(DocumentChunk.chunk_index)
    )
    chunks = result.scalars().all()
    if not chunks:
        raise HTTPException(status_code=422, detail="No chunks found for this document")

    summarizer = DocumentSummarizer()
    summary = await summarizer.summarize(
        document_id=str(doc_id),
        filename=doc.original_filename,
        chunks_text=[c.text for c in chunks],
    )

    return SummarizeResponse(
        document_id=summary.document_id,
        filename=summary.filename,
        bullet_points=summary.bullet_points,
        model_used=summary.model_used,
    )
