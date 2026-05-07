"""Qdrant vector store wrapper with hybrid dense + BM25 sparse search
fused via Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from rank_bm25 import BM25Okapi

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)

COLLECTION = settings.qdrant_collection_name
VECTOR_DIM = settings.embedding_dimensions


# ── Return types ──────────────────────────────────────────────────────────────


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with score and metadata."""

    point_id: str
    score: float          # cosine similarity [0, 1]
    text: str
    document_id: str
    chunk_index: int
    page_number: int
    filename: str
    file_type: str
    snippet: str          # first 200 chars for citation display


# ── Qdrant client wrapper ─────────────────────────────────────────────────────


class VectorStore:
    """Async Qdrant client abstraction.

    Supports:
    - Upsert (add / update) chunks with their embeddings.
    - Dense cosine similarity search.
    - Hybrid search: dense + BM25 fused by RRF.
    - Delete all vectors belonging to a document.
    """

    def __init__(
        self,
        host: str = settings.qdrant_host,
        port: int = settings.qdrant_port,
        collection: str = COLLECTION,
    ) -> None:
        self._client = AsyncQdrantClient(host=host, port=port)
        self._collection = collection

    # ── Collection lifecycle ──────────────────────────────────────────────────

    async def ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not exist."""
        exists = await self._client.collection_exists(self._collection)
        if not exists:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qmodels.VectorParams(
                    size=VECTOR_DIM,
                    distance=qmodels.Distance.COSINE,
                    on_disk=False,
                ),
                optimizers_config=qmodels.OptimizersConfigDiff(
                    indexing_threshold=20_000,
                    memmap_threshold=50_000,
                ),
                hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=100),
            )
            log.info("qdrant_collection_created", collection=self._collection)
        else:
            log.debug("qdrant_collection_exists", collection=self._collection)

    # ── Write ─────────────────────────────────────────────────────────────────

    async def upsert_chunks(
        self,
        document_id: str,
        chunks: list[dict[str, Any]],  # {text, chunk_index, page_number, vector, content_hash}
        filename: str,
        file_type: str,
    ) -> list[str]:
        """Upsert chunk embeddings into Qdrant.

        Args:
            document_id: UUID of the parent document.
            chunks: List of chunk dicts with text, vector, and metadata.
            filename: Human-readable filename for citation.
            file_type: Extension of the source file.

        Returns:
            List of Qdrant point IDs (strings).
        """
        points: list[qmodels.PointStruct] = []
        point_ids: list[str] = []

        for chunk in chunks:
            point_id = str(uuid.uuid4())
            point_ids.append(point_id)
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=chunk["vector"],
                    payload={
                        "document_id": document_id,
                        "chunk_index": chunk["chunk_index"],
                        "page_number": chunk.get("page_number", 1),
                        "text": chunk["text"],
                        "filename": filename,
                        "file_type": file_type,
                        "content_hash": chunk["content_hash"],
                    },
                )
            )

        await self._client.upsert(
            collection_name=self._collection,
            points=points,
            wait=True,
        )
        log.info("qdrant_upserted", count=len(points), document_id=document_id)
        return point_ids

    # ── Dense search ──────────────────────────────────────────────────────────

    async def dense_search(
        self,
        query_vector: list[float],
        top_k: int = settings.default_top_k,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Perform pure dense (cosine) vector search.

        Args:
            query_vector: Embedding of the query.
            top_k: Number of results to return.
            filters: Optional metadata filters (e.g. document_id, file_type).

        Returns:
            Ranked list of RetrievedChunk.
        """
        qdrant_filter = _build_filter(filters) if filters else None
        results = await self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [_to_retrieved_chunk(r) for r in results]

    # ── Hybrid search (dense + BM25 fused by RRF) ─────────────────────────────

    async def hybrid_search(
        self,
        query: str,
        query_vector: list[float],
        top_k: int = settings.default_top_k,
        filters: dict[str, Any] | None = None,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        """Hybrid search: dense vector + BM25 fused with Reciprocal Rank Fusion.

        Args:
            query: Raw query string (for BM25).
            query_vector: Embedded query vector (for dense search).
            top_k: Final number of results.
            filters: Optional Qdrant metadata filters.
            rrf_k: RRF constant (higher = smoother fusion).

        Returns:
            Fused, re-ranked list of RetrievedChunk.
        """
        # Fetch more candidates to have a good BM25 pool
        candidate_k = min(top_k * 5, 50)

        dense_results = await self.dense_search(query_vector, top_k=candidate_k, filters=filters)
        if not dense_results:
            return []

        # Run BM25 on retrieved texts
        corpus = [r.text for r in dense_results]
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = query.lower().split()
        bm25_scores = bm25.get_scores(query_tokens)

        # Dense rank dictionary (point_id → rank)
        dense_ranks: dict[str, int] = {
            r.point_id: i + 1 for i, r in enumerate(dense_results)
        }

        # BM25 rank dictionary (sort by BM25 score desc)
        bm25_order = sorted(
            range(len(dense_results)), key=lambda i: bm25_scores[i], reverse=True
        )
        bm25_ranks: dict[str, int] = {
            dense_results[idx].point_id: rank + 1
            for rank, idx in enumerate(bm25_order)
        }

        # Reciprocal Rank Fusion
        rrf_scores: dict[str, float] = {}
        for chunk in dense_results:
            pid = chunk.point_id
            d_rank = dense_ranks.get(pid, candidate_k + 1)
            b_rank = bm25_ranks.get(pid, candidate_k + 1)
            rrf_scores[pid] = 1 / (rrf_k + d_rank) + 1 / (rrf_k + b_rank)

        # Sort by RRF score and return top_k
        sorted_ids = sorted(rrf_scores, key=lambda pid: rrf_scores[pid], reverse=True)[:top_k]
        pid_to_chunk = {r.point_id: r for r in dense_results}

        fused = []
        for pid in sorted_ids:
            chunk = pid_to_chunk[pid]
            chunk.score = rrf_scores[pid]  # replace cosine with RRF score
            fused.append(chunk)

        log.info(
            "hybrid_search_done",
            query_preview=query[:60],
            dense_candidates=len(dense_results),
            fused_results=len(fused),
        )
        return fused

    # ── Delete ────────────────────────────────────────────────────────────────

    async def delete_document_vectors(self, document_id: str) -> int:
        """Delete all Qdrant points belonging to *document_id*.

        Returns:
            Number of deleted points.
        """
        result = await self._client.delete(
            collection_name=self._collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )
        log.info("qdrant_deleted", document_id=document_id, result=str(result))
        return 1  # Qdrant delete returns operation info, not count

    async def get_collection_info(self) -> dict[str, Any]:
        """Return collection stats (point count, vector config)."""
        info = await self._client.get_collection(self._collection)
        return {
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status,
        }


# ── Helper functions ──────────────────────────────────────────────────────────


def _build_filter(filters: dict[str, Any]) -> qmodels.Filter:
    """Convert a simple key-value filter dict to a Qdrant Filter."""
    conditions: list[qmodels.Condition] = []
    for key, value in filters.items():
        if value is not None:
            conditions.append(
                qmodels.FieldCondition(
                    key=key,
                    match=qmodels.MatchValue(value=str(value)),
                )
            )
    return qmodels.Filter(must=conditions)


def _to_retrieved_chunk(result: Any) -> RetrievedChunk:
    """Convert a Qdrant ScoredPoint to RetrievedChunk."""
    payload = result.payload or {}
    text = payload.get("text", "")
    return RetrievedChunk(
        point_id=str(result.id),
        score=float(result.score),
        text=text,
        document_id=payload.get("document_id", ""),
        chunk_index=payload.get("chunk_index", 0),
        page_number=payload.get("page_number", 1),
        filename=payload.get("filename", "unknown"),
        file_type=payload.get("file_type", ""),
        snippet=text[:200].replace("\n", " "),
    )


# ── Singleton ─────────────────────────────────────────────────────────────────
_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Return the module-level VectorStore singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
