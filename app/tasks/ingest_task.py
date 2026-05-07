"""Background task: full document ingestion pipeline with progress events."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from app.cache.local_cache import set_job_progress
from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


async def process_document_pipeline(
    document_id: str,
    file_path: str,
    filename: str,
    file_ext: str,
    job_id: str,
) -> dict:
    """Async implementation of the ingestion pipeline."""
    # Import here to avoid circular imports at module load time
    from app.db.models import Document, DocumentChunk, JobStatus
    from app.db.session import AsyncSessionLocal
    from app.ingestion.chunker import default_chunker
    from app.ingestion.embedder import EmbeddingService
    from app.ingestion.parsers import parse_document
    from app.retrieval.vector_store import get_vector_store

    async def update_progress(pct: int, status: JobStatus | None = None) -> None:
        await set_job_progress(job_id, status.value if status else "processing", pct)
        if status:
            async with AsyncSessionLocal() as db:
                doc = await db.get(Document, uuid.UUID(document_id))
                if doc:
                    doc.job_status = status
                    doc.job_progress = pct
                    await db.commit()

    try:
        # ── 0%: Start ─────────────────────────────────────────────────────────
        await update_progress(0, JobStatus.PROCESSING)
        log.info("ingest_started", document_id=document_id, filename=filename)

        # ── 10%: Read file ────────────────────────────────────────────────────
        file_data = Path(file_path).read_bytes()
        await update_progress(10)

        # ── 25%: Parse ────────────────────────────────────────────────────────
        parsed_doc = parse_document(file_data, filename)
        await update_progress(25)

        # ── 40%: Chunk ────────────────────────────────────────────────────────
        chunks = default_chunker.chunk_document(parsed_doc)
        await update_progress(40)

        if not chunks:
            raise ValueError("No text extracted — document may be empty or image-only.")

        # ── 60%: Embed ────────────────────────────────────────────────────────
        embedder = EmbeddingService()
        chunk_vector_pairs = await embedder.embed_chunks(chunks)
        await update_progress(60)

        # ── 75%: Store vectors ────────────────────────────────────────────────
        vs = get_vector_store()
        await vs.ensure_collection()

        chunk_dicts = [
            {
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "vector": vector,
                "content_hash": chunk.content_hash,
            }
            for chunk, vector in chunk_vector_pairs
        ]
        point_ids = await vs.upsert_chunks(
            document_id=document_id,
            chunks=chunk_dicts,
            filename=filename,
            file_type=file_ext,
        )
        await update_progress(75)

        # ── 90%: Persist chunk metadata to SQLite ─────────────────────────────
        async with AsyncSessionLocal() as db:
            db_chunks = [
                DocumentChunk(
                    document_id=uuid.UUID(document_id),
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    page_number=chunk.page_number,
                    token_count=chunk.token_count,
                    embedding_id=point_id,
                    content_hash=chunk.content_hash,
                )
                for (chunk, _), point_id in zip(chunk_vector_pairs, point_ids, strict=False)
            ]
            db.add_all(db_chunks)

            doc = await db.get(Document, uuid.UUID(document_id))
            if doc:
                doc.chunk_count = len(chunks)
                doc.page_count = parsed_doc.page_count
                doc.word_count = parsed_doc.word_count
                doc.doc_metadata = parsed_doc.doc_metadata
                doc.job_status = JobStatus.DONE
                doc.job_progress = 100

            await db.commit()
        await update_progress(100, JobStatus.DONE)

        log.info(
            "ingest_completed",
            document_id=document_id,
            chunks=len(chunks),
            pages=parsed_doc.page_count,
        )
        return {
            "chunk_count": len(chunks),
            "word_count": parsed_doc.word_count,
            "page_count": parsed_doc.page_count,
        }

    except Exception as exc:
        log.error("ingest_failed", document_id=document_id, error=str(exc))
        async with AsyncSessionLocal() as db:
            doc = await db.get(Document, uuid.UUID(document_id))
            if doc:
                doc.job_status = JobStatus.FAILED
                doc.error_message = str(exc)[:500]
                await db.commit()
        await set_job_progress(job_id, "failed", 0, str(exc))
        raise
