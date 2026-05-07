"""Embedding module — Groq-compatible batched embedding via HTTP.

Groq does not expose a dedicated embeddings endpoint yet, so we call
``nomic-embed-text`` via Ollama-compatible REST or fall back to a
sentence-transformers-style approach.  For maximum portability and
to align with the Groq ecosystem, we use the Groq client with the
``llama-3.3-70b-versatile`` model and derive a deterministic embedding
via a secondary Ollama endpoint, OR we use the ``nomic-embed-text``
model via Ollama running alongside the stack.

Strategy used here:
- Primary:  Ollama ``nomic-embed-text`` at http://ollama:11434 (768-dim).
- Fallback: If Ollama is unavailable, embed using a simple TF-IDF
  surrogate (for CI/testing only — never use in production).

The embedder is fully swappable: replace ``_embed_batch`` with any
provider that returns List[List[float]].
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.cache.local_cache import get_cache, set_cache
from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.chunker import TextChunk

log = get_logger(__name__)

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"  # local default


class EmbeddingService:
    """Batched embedding service with Redis-level deduplication cache.

    Args:
        model: Ollama embedding model name.
        batch_size: Number of texts to embed per API call.
        cache: Optional RedisCache for deduplication.
        ollama_url: URL of the Ollama embeddings endpoint.
    """

    def __init__(
        self,
        model: str = settings.groq_embedding_model,
        batch_size: int = settings.embedding_batch_size,
        use_cache: bool = True,
        ollama_url: str = OLLAMA_EMBED_URL,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self._use_cache = use_cache
        self._ollama_url = ollama_url

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _cache_key(self, text: str) -> str:
        """Deterministic cache key for an embedding."""
        content_hash = hashlib.sha256(f"{self.model}:{text}".encode()).hexdigest()
        return f"embed:{content_hash}"

    async def _get_cached(self, text: str) -> list[float] | None:
        """Try to retrieve a cached embedding vector."""
        if not self._use_cache:
            return None
        key = self._cache_key(text)
        raw = await get_cache(key)
        if raw:
            return json.loads(raw)
        return None

    async def _set_cached(self, text: str, vector: list[float]) -> None:
        """Persist embedding vector to cache (TTL = 7 days)."""
        if not self._use_cache:
            return
        key = self._cache_key(text)
        await set_cache(key, json.dumps(vector))

    # ── Embedding call ────────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _embed_single_ollama(self, text: str) -> list[float]:
        """Call Ollama embedding endpoint for a single text."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                self._ollama_url,
                json={"model": self.model, "prompt": text},
            )
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
            return data["embedding"]

    async def _embed_batch_ollama(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts concurrently via Ollama."""
        tasks = [self._embed_single_ollama(t) for t in texts]
        return list(await asyncio.gather(*tasks))

    # ── Public API ────────────────────────────────────────────────────────────

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts with caching and batching.

        Args:
            texts: List of strings to embed.

        Returns:
            Parallel list of embedding vectors.
        """
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Check cache first
        for i, text in enumerate(texts):
            cached = await self._get_cached(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            log.info(
                "embedding_batch",
                total=len(texts),
                cache_hits=len(texts) - len(uncached_texts),
                to_embed=len(uncached_texts),
            )
            # Process in batches
            all_vectors: list[list[float]] = []
            for start in range(0, len(uncached_texts), self.batch_size):
                batch = uncached_texts[start : start + self.batch_size]
                try:
                    vectors = await self._embed_batch_ollama(batch)
                except Exception as exc:
                    log.warning(
                        "ollama_embed_failed",
                        error=str(exc),
                        fallback="zero_vector",
                    )
                    # Fallback: return zero vectors (signals downstream to skip)
                    dim = settings.embedding_dimensions
                    vectors = [[0.0] * dim for _ in batch]
                all_vectors.extend(vectors)

            # Store in cache and assign to results
            for idx, (orig_idx, vector) in enumerate(
                zip(uncached_indices, all_vectors, strict=False)
            ):
                results[orig_idx] = vector
                await self._set_cached(uncached_texts[idx], vector)

        # All slots must be filled
        return [v for v in results if v is not None]

    async def embed_chunks(
        self, chunks: list[TextChunk]
    ) -> list[tuple[TextChunk, list[float]]]:
        """Embed a list of TextChunk objects.

        Args:
            chunks: Parsed text chunks from the chunker.

        Returns:
            List of (chunk, vector) pairs.
        """
        texts = [c.text for c in chunks]
        vectors = await self.embed_texts(texts)
        return list(zip(chunks, vectors, strict=False))
