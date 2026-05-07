"""Recursive character text splitter with configurable chunk size and overlap.

The chunker works in two passes:
1. Split on paragraph / sentence boundaries to keep semantic units intact.
2. If any resulting piece still exceeds *chunk_size*, split on whitespace.
Token counting uses tiktoken (cl100k_base) for accuracy.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import tiktoken

from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.parsers import ParsedDocument

log = get_logger(__name__)

# Use the same tokenizer as modern OpenAI / Groq models
_tokenizer = tiktoken.get_encoding("cl100k_base")

# Separator hierarchy (high → low priority)
_SEPARATORS = ["\n\n\n", "\n\n", "\n", ". ", "? ", "! ", " ", ""]


def _count_tokens(text: str) -> int:
    """Return token count for *text* using cl100k_base encoding."""
    return len(_tokenizer.encode(text))


@dataclass
class TextChunk:
    """A single chunk ready for embedding."""

    text: str
    chunk_index: int
    page_number: int
    token_count: int
    content_hash: str

    @classmethod
    def from_text(cls, text: str, chunk_index: int, page_number: int) -> "TextChunk":
        """Construct a chunk, computing token count and hash automatically."""
        return cls(
            text=text,
            chunk_index=chunk_index,
            page_number=page_number,
            token_count=_count_tokens(text),
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
        )


class RecursiveCharacterChunker:
    """Split text recursively using a hierarchy of separators.

    Args:
        chunk_size: Target maximum token count per chunk.
        chunk_overlap: Number of tokens to overlap between consecutive chunks.
        separators: Ordered list of separator strings to try.
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or _SEPARATORS

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split *text* using the first separator that reduces size."""
        if not text.strip():
            return []

        if _count_tokens(text) <= self.chunk_size:
            return [text.strip()]

        # Try each separator in order
        for sep in separators:
            if sep == "":
                # Last resort: hard split by token count
                tokens = _tokenizer.encode(text)
                parts = []
                for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
                    part_tokens = tokens[i : i + self.chunk_size]
                    parts.append(_tokenizer.decode(part_tokens))
                return parts

            splits = text.split(sep)
            if len(splits) <= 1:
                continue

            good_parts: list[str] = []
            current = ""
            for split in splits:
                candidate = (current + sep + split) if current else split
                if _count_tokens(candidate) <= self.chunk_size:
                    current = candidate
                else:
                    if current:
                        good_parts.append(current.strip())
                    # Recursively handle oversized split
                    remaining_seps = separators[separators.index(sep) + 1 :]
                    good_parts.extend(self._split_text(split, remaining_seps or [""]))
                    current = ""
            if current:
                good_parts.append(current.strip())

            return [p for p in good_parts if p]

        return [text.strip()]

    def _merge_with_overlap(
        self, pieces: list[str], page_number: int
    ) -> list[tuple[str, int]]:
        """Merge small pieces and add overlap, returning (text, page_number) tuples."""
        merged: list[tuple[str, int]] = []
        current = ""
        overlap_buffer = ""

        for piece in pieces:
            candidate = (current + "\n" + piece).strip() if current else piece
            if _count_tokens(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    merged.append((current, page_number))
                    # Build overlap from tail of current chunk
                    tail_tokens = _tokenizer.encode(current)[-self.chunk_overlap :]
                    overlap_buffer = _tokenizer.decode(tail_tokens)
                current = (overlap_buffer + "\n" + piece).strip() if overlap_buffer else piece
                overlap_buffer = ""

        if current:
            merged.append((current, page_number))

        return merged

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk_document(self, parsed_doc: ParsedDocument) -> list[TextChunk]:
        """Split a ParsedDocument into TextChunks.

        Args:
            parsed_doc: Output from one of the document parsers.

        Returns:
            Ordered list of TextChunk objects ready for embedding.
        """
        all_chunks: list[tuple[str, int]] = []

        for page in parsed_doc.pages:
            if not page.text.strip():
                continue
            pieces = self._split_text(page.text, self.separators)
            merged = self._merge_with_overlap(pieces, page.page_number)
            all_chunks.extend(merged)

        result: list[TextChunk] = [
            TextChunk.from_text(text, i, page_num)
            for i, (text, page_num) in enumerate(all_chunks)
            if text.strip()
        ]

        log.info(
            "document_chunked",
            total_chunks=len(result),
            avg_tokens=sum(c.token_count for c in result) // max(len(result), 1),
        )
        return result


# ── Module-level convenience instance ─────────────────────────────────────────
default_chunker = RecursiveCharacterChunker()
