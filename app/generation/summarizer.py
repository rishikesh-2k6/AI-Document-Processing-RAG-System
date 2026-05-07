"""Document summarizer — 3-5 bullet points via Groq using all chunks of a document."""

from __future__ import annotations

from dataclasses import dataclass

from groq import AsyncGroq

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class SummaryResult:
    """Result of a document summarization request."""

    document_id: str
    filename: str
    bullet_points: list[str]
    model_used: str
    tokens_used: int


_SUMMARIZE_SYSTEM = """You are an expert document analyst. Summarize the provided document
content into exactly 3 to 5 concise bullet points.

RULES:
1. Each bullet point must be a single, information-dense sentence.
2. Cover the most important topics, conclusions, or data points.
3. Do not include preamble text — output ONLY the bullet points.
4. Format: start each line with "• " (bullet character + space).
5. Preserve important numbers, names, and technical terms."""

_SUMMARIZE_USER = """Document: {filename}

Content:
{content}

Provide 3-5 bullet point summary:"""


class DocumentSummarizer:
    """Generate concise bullet-point summaries of entire documents."""

    def __init__(self) -> None:
        self._client = AsyncGroq(api_key=settings.groq_api_key)

    async def summarize(
        self,
        document_id: str,
        filename: str,
        chunks_text: list[str],
        max_context_chars: int = 12_000,
    ) -> SummaryResult:
        """Summarize a document from its text chunks.

        Args:
            document_id: UUID of the document.
            filename: Human-readable filename.
            chunks_text: Ordered list of chunk texts.
            max_context_chars: Maximum characters to send to the LLM.

        Returns:
            SummaryResult with parsed bullet points.
        """
        # Concatenate chunks up to context limit
        full_text = "\n\n".join(chunks_text)
        if len(full_text) > max_context_chars:
            full_text = full_text[:max_context_chars] + "\n[...content truncated...]"

        user_prompt = _SUMMARIZE_USER.format(filename=filename, content=full_text)

        try:
            response = await self._client.chat.completions.create(
                model=settings.groq_summarize_model,
                messages=[
                    {"role": "system", "content": _SUMMARIZE_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=600,
            )
        except Exception as exc:
            log.error("summarize_failed", document_id=document_id, error=str(exc))
            raise RuntimeError(f"Summarization failed: {exc}") from exc

        raw_output = response.choices[0].message.content or ""
        tokens_used = response.usage.total_tokens if response.usage else 0

        # Parse bullet points
        bullets = []
        for line in raw_output.splitlines():
            line = line.strip()
            if line.startswith("•"):
                bullets.append(line[1:].strip())
            elif line.startswith("-") or line.startswith("*"):
                bullets.append(line[1:].strip())
            elif line:
                bullets.append(line)

        # Ensure 3–5 bullets
        bullets = [b for b in bullets if b][:5]
        if not bullets:
            bullets = [raw_output.strip()]

        log.info(
            "document_summarized",
            document_id=document_id,
            bullet_count=len(bullets),
            tokens=tokens_used,
        )

        return SummaryResult(
            document_id=document_id,
            filename=filename,
            bullet_points=bullets,
            model_used=settings.groq_summarize_model,
            tokens_used=tokens_used,
        )
