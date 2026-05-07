"""Document management routes: upload, list, get, delete, status, progress (SSE)."""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, DBSession
from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import Document, DocumentChunk, FileType, JobStatus
from app.tasks.ingest_task import ingest_document

log = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


# ── Response schemas ──────────────────────────────────────────────────────────


class DocumentOut(BaseModel):
    """Document metadata response."""

    id: str
    filename: str
    original_filename: str
    file_type: str
    file_size_bytes: int
    job_status: str
    job_progress: int
    chunk_count: int
    page_count: int | None
    word_count: int | None
    created_at: str

    model_config = {"from_attributes": True}


class DocumentDetailOut(DocumentOut):
    """Extended document response with metadata."""

    doc_metadata: dict | None
    error_message: str | None


class DocumentListResponse(BaseModel):
    """Paginated document list."""

    items: list[DocumentOut]
    total: int
    page: int
    page_size: int


class JobStatusOut(BaseModel):
    """Ingestion job status."""

    document_id: str
    job_id: str | None
    status: str
    progress: int
    error: str | None


class UploadResponse(BaseModel):
    """Response from a successful upload."""

    document_id: str
    job_id: str
    filename: str
    message: str


# ── Helpers ───────────────────────────────────────────────────────────────────


def _doc_to_out(doc: Document) -> DocumentOut:
    return DocumentOut(
        id=str(doc.id),
        filename=doc.filename,
        original_filename=doc.original_filename,
        file_type=doc.file_type.value,
        file_size_bytes=doc.file_size_bytes,
        job_status=doc.job_status.value,
        job_progress=doc.job_progress,
        chunk_count=doc.chunk_count,
        page_count=doc.page_count,
        word_count=doc.word_count,
        created_at=doc.created_at.isoformat(),
    )


def _validate_upload(file: UploadFile) -> str:
    """Validate file extension and return lowercase extension."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    ext = Path(file.filename).suffix.lstrip(".").lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not supported. Allowed: {settings.allowed_extensions}",
        )
    return ext


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document for ingestion",
)
async def upload_document(
    current_user: CurrentUser,
    db: DBSession,
    file: UploadFile = File(..., description="PDF, DOCX, or EML file"),
) -> UploadResponse:
    """Upload a document and queue it for background ingestion.

    Supported formats: PDF, DOCX, EML, MSG.
    Returns a **job_id** you can use to poll ``/documents/{id}/status``.
    """
    ext = _validate_upload(file)

    file_bytes = await file.read()
    if len(file_bytes) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.max_upload_size_mb} MB limit",
        )

    # Compute hash for deduplication
    content_hash = hashlib.sha256(file_bytes).hexdigest()

    # Check for duplicate in this user's documents
    existing = await db.execute(
        select(Document).where(
            Document.content_hash == content_hash,
            Document.owner_id == current_user.id,
        )
    )
    dup = existing.scalar_one_or_none()
    if dup and dup.job_status == JobStatus.DONE:
        raise HTTPException(
            status_code=409,
            detail=f"Duplicate document already ingested as '{dup.original_filename}'",
        )

    # Save file to disk
    doc_id = uuid.uuid4()
    safe_name = f"{doc_id}.{ext}"
    file_path = settings.upload_dir / safe_name
    file_path.write_bytes(file_bytes)

    # Create DB record
    doc = Document(
        id=doc_id,
        owner_id=current_user.id,
        filename=safe_name,
        original_filename=file.filename or safe_name,
        file_type=FileType(ext),
        file_size_bytes=len(file_bytes),
        file_path=str(file_path),
        content_hash=content_hash,
        job_status=JobStatus.QUEUED,
    )
    db.add(doc)
    await db.flush()

    # Dispatch Celery task
    task = ingest_document.apply_async(
        kwargs={
            "document_id": str(doc_id),
            "file_path": str(file_path),
            "filename": file.filename or safe_name,
            "owner_id": str(current_user.id),
        }
    )

    doc.job_id = task.id
    await db.flush()

    log.info("document_queued", doc_id=str(doc_id), job_id=task.id, ext=ext)
    return UploadResponse(
        document_id=str(doc_id),
        job_id=task.id,
        filename=file.filename or safe_name,
        message="Document queued for ingestion",
    )


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List all ingested documents (paginated)",
)
async def list_documents(
    current_user: CurrentUser,
    db: DBSession,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: str | None = Query(None, alias="status"),
) -> DocumentListResponse:
    """Return a paginated list of the authenticated user's documents."""
    query = select(Document).where(Document.owner_id == current_user.id)
    if status_filter:
        try:
            query = query.where(Document.job_status == JobStatus(status_filter))
        except ValueError:
            pass

    count_q = select(func.count()).select_from(
        query.subquery()
    )
    total = (await db.execute(count_q)).scalar_one()

    query = query.order_by(Document.created_at.desc())
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    docs = result.scalars().all()

    return DocumentListResponse(
        items=[_doc_to_out(d) for d in docs],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentDetailOut,
    summary="Get document metadata and chunk count",
)
async def get_document(
    document_id: uuid.UUID,
    current_user: CurrentUser,
    db: DBSession,
) -> DocumentDetailOut:
    """Retrieve detailed metadata for a specific document."""
    doc = await db.get(Document, document_id)
    if not doc or doc.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentDetailOut(
        **_doc_to_out(doc).model_dump(),
        doc_metadata=doc.doc_metadata,
        error_message=doc.error_message,
    )


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete document and its vectors",
)
async def delete_document(
    document_id: uuid.UUID,
    current_user: CurrentUser,
    db: DBSession,
) -> None:
    """Delete a document, its chunks from Postgres, and its vectors from Qdrant."""
    doc = await db.get(Document, document_id)
    if not doc or doc.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete from Qdrant
    try:
        from app.retrieval.vector_store import get_vector_store

        vs = get_vector_store()
        await vs.delete_document_vectors(str(document_id))
    except Exception as exc:
        log.warning("qdrant_delete_failed", doc_id=str(document_id), error=str(exc))

    # Delete file from disk
    try:
        Path(doc.file_path).unlink(missing_ok=True)
    except Exception:
        pass

    await db.delete(doc)
    log.info("document_deleted", doc_id=str(document_id))


@router.get(
    "/{document_id}/status",
    response_model=JobStatusOut,
    summary="Get ingestion job status",
)
async def get_job_status(
    document_id: uuid.UUID,
    current_user: CurrentUser,
    db: DBSession,
) -> JobStatusOut:
    """Poll the ingestion job status for a document."""
    doc = await db.get(Document, document_id)
    if not doc or doc.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Document not found")

    return JobStatusOut(
        document_id=str(doc.id),
        job_id=doc.job_id,
        status=doc.job_status.value,
        progress=doc.job_progress,
        error=doc.error_message,
    )


@router.get(
    "/{document_id}/progress",
    summary="SSE stream of ingestion progress events",
)
async def stream_progress(
    document_id: uuid.UUID,
    current_user: CurrentUser,
    db: DBSession,
) -> StreamingResponse:
    """Server-Sent Events (SSE) endpoint streaming ingestion progress (0-100).

    Connect with ``EventSource`` or any SSE-compatible client.
    The stream closes automatically when progress reaches 100 or if the job
    has already completed/failed.
    """
    doc = await db.get(Document, document_id)
    if not doc or doc.owner_id != current_user.id:
        raise HTTPException(status_code=404, detail="Document not found")

    job_id = doc.job_id or str(document_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        from app.cache.redis_client import get_redis_cache

        cache = get_redis_cache()
        last_progress = -1

        for _ in range(120):  # Max 120 iterations × 0.5s = 60s timeout
            progress = await cache.get_progress(job_id)
            if progress != last_progress:
                last_progress = progress
                yield f"data: {progress}\n\n"
            if progress >= 100:
                break
            await asyncio.sleep(0.5)

        yield "data: done\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
