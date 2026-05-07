"""RAG answer generation using Groq LLM with chunk-level citations and confidence scoring."""

from __future__ import annotations

from dataclasses import dataclass

from groq import AsyncGroq

from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.vector_store import RetrievedChunk

log = get_logger(__name__)


# ── Response contract ─────────────────────────────────────────────────────────


@dataclass
class SourceChunk:
    """Citation metadata for a single source chunk."""

    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    snippet: str
    score: float


@dataclass
class RAGAnswer:
    """Complete RAG answer with citations and confidence."""

    answer: str
    source_chunks: list[SourceChunk]
    confidence_score: float
    low_confidence_warning: bool
    model_used: str
    tokens_used: int


# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a precise, citation-aware AI assistant that answers questions strictly based on provided context documents.

RULES:
1. Answer ONLY from the provided context. Do NOT use external knowledge.
2. Every factual statement must be followed by a citation in format [Source: <filename>, Page <page>].
3. If the context is insufficient, say "I cannot find sufficient information in the provided documents to answer this question."
4. Be concise and direct. Do not add preambles like "Based on the context...".
5. If multiple sources support a claim, cite all of them.
6. Preserve technical terms, numbers, and proper nouns exactly as they appear."""

_USER_PROMPT_TEMPLATE = """CONTEXT DOCUMENTS:
{context}

---

QUESTION: {question}

Provide a precise, well-cited answer based solely on the context above."""


# ── Answer generator ──────────────────────────────────────────────────────────


class RAGAnswerGenerator:
    """Generate grounded answers from retrieved chunks using Groq.

    Features:
    - Chunk-level citations (filename + page number + snippet).
    - Confidence scoring based on average cosine similarity.
    - Low-confidence warning if score < threshold.
    """

    def __init__(self) -> None:
        self._client = AsyncGroq(api_key=settings.groq_api_key)

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into a numbered context block."""
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            parts.append(
                f"[{i}] Source: {chunk.filename} | Page: {chunk.page_number}\n"
                f"{chunk.text}\n"
            )
        return "\n---\n".join(parts)

    def _compute_confidence(self, chunks: list[RetrievedChunk]) -> float:
        """Compute confidence as average cosine similarity of top chunks."""
        if not chunks:
            return 0.0
        scores = [c.score for c in chunks if 0.0 <= c.score <= 1.0]
        if not scores:
            return 0.0
        return round(sum(scores) / len(scores), 4)

    def _extract_source_chunks(self, chunks: list[RetrievedChunk]) -> list[SourceChunk]:
        """Convert RetrievedChunks to SourceChunk citation objects."""
        return [
            SourceChunk(
                document_id=c.document_id,
                filename=c.filename,
                page_number=c.page_number,
                chunk_index=c.chunk_index,
                snippet=c.snippet,
                score=round(c.score, 4),
            )
            for c in chunks
        ]

    async def generate(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> RAGAnswer:
        """Generate a grounded answer with citations.

        Args:
            query: User's natural-language question.
            chunks: Retrieved chunks from the vector store.

        Returns:
            RAGAnswer with answer text, citations, and confidence.
        """
        if not chunks:
            return RAGAnswer(
                answer="No relevant documents were found to answer this question.",
                source_chunks=[],
                confidence_score=0.0,
                low_confidence_warning=True,
                model_used=settings.groq_chat_model,
                tokens_used=0,
            )

        context = self._build_context(chunks)
        user_prompt = _USER_PROMPT_TEMPLATE.format(context=context, question=query)

        try:
            response = await self._client.chat.completions.create(
                model=settings.groq_chat_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=1500,
            )
        except Exception as exc:
            log.error("rag_generation_failed", query=query[:80], error=str(exc))
            raise RuntimeError(f"LLM generation failed: {exc}") from exc

        answer_text = response.choices[0].message.content or ""
        tokens_used = response.usage.total_tokens if response.usage else 0
        confidence = self._compute_confidence(chunks)
        source_chunks = self._extract_source_chunks(chunks)

        log.info(
            "rag_answer_generated",
            query_preview=query[:60],
            confidence=confidence,
            sources=len(source_chunks),
            tokens=tokens_used,
        )

        return RAGAnswer(
            answer=answer_text,
            source_chunks=source_chunks,
            confidence_score=confidence,
            low_confidence_warning=confidence < settings.confidence_threshold,
            model_used=settings.groq_chat_model,
            tokens_used=tokens_used,
        )
