"""Optional cross-encoder reranker using a simple LLM-based relevance score.

For production, replace ``_llm_score`` with a real cross-encoder model
(e.g. ``cross-encoder/ms-marco-MiniLM-L-6-v2`` via sentence-transformers).
In this implementation we use Groq to score relevance as a practical
demonstration that does not require local GPU resources.
"""

from __future__ import annotations

import asyncio

from groq import AsyncGroq

from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.vector_store import RetrievedChunk

log = get_logger(__name__)


class CrossEncoderReranker:
    """LLM-based relevance reranker.

    Scores each chunk against the query using Groq's fast inference,
    then returns the top *top_k* chunks sorted by relevance score.
    """

    SCORE_PROMPT = (
        "Rate the relevance of the following passage to the query on a scale "
        "from 0.0 (completely irrelevant) to 1.0 (perfectly relevant). "
        "Respond with ONLY a decimal number, nothing else.\n\n"
        "Query: {query}\n\nPassage: {passage}\n\nRelevance score:"
    )

    def __init__(self) -> None:
        self._client = AsyncGroq(api_key=settings.groq_api_key)

    async def _llm_score(self, query: str, passage: str) -> float:
        """Ask Groq to rate the relevance of *passage* for *query*."""
        prompt = self.SCORE_PROMPT.format(query=query, passage=passage[:500])
        try:
            resp = await self._client.chat.completions.create(
                model="llama-3.1-8b-instant",  # fast small model for scoring
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            score_text = resp.choices[0].message.content.strip()
            return max(0.0, min(1.0, float(score_text)))
        except Exception as exc:
            log.warning("reranker_score_failed", error=str(exc))
            return 0.5  # neutral fallback

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Rerank *chunks* by LLM relevance score, return top *top_k*.

        Args:
            query: The user's natural-language query.
            chunks: Retrieved chunks from the vector store.
            top_k: Number of chunks to return after reranking.

        Returns:
            Top-k chunks sorted by descending relevance score.
        """
        if not chunks:
            return []

        scores = await asyncio.gather(
            *[self._llm_score(query, c.text) for c in chunks]
        )

        scored = sorted(
            zip(chunks, scores, strict=False),
            key=lambda x: x[1],
            reverse=True,
        )

        result = []
        for chunk, score in scored[:top_k]:
            chunk.score = score
            result.append(chunk)

        log.info("reranker_done", input=len(chunks), output=len(result))
        return result
