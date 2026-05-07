"""Unit tests for the recursive character chunker."""

from __future__ import annotations

import pytest

from app.ingestion.chunker import RecursiveCharacterChunker, TextChunk
from app.ingestion.parsers import ParsedDocument, ParsedPage


def _make_doc(text: str, page: int = 1) -> ParsedDocument:
    """Create a minimal ParsedDocument for testing."""
    return ParsedDocument(
        pages=[ParsedPage(page_number=page, text=text)],
        doc_metadata={},
        word_count=len(text.split()),
        page_count=1,
    )


class TestRecursiveCharacterChunker:
    """Tests for the text chunker."""

    def test_short_text_single_chunk(self) -> None:
        chunker = RecursiveCharacterChunker(chunk_size=512, chunk_overlap=64)
        doc = _make_doc("This is a short document.")
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "This is a short document."

    def test_long_text_multiple_chunks(self) -> None:
        long_text = " ".join(["word"] * 2000)
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10)
        doc = _make_doc(long_text)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1

    def test_chunk_index_sequential(self) -> None:
        long_text = "\n\n".join(["Paragraph text here. " * 10] * 20)
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10)
        doc = _make_doc(long_text)
        chunks = chunker.chunk_document(doc)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_page_number_preserved(self) -> None:
        doc = ParsedDocument(
            pages=[
                ParsedPage(page_number=3, text="Content on page three. " * 5),
            ],
            doc_metadata={},
            word_count=25,
            page_count=1,
        )
        chunker = RecursiveCharacterChunker(chunk_size=512)
        chunks = chunker.chunk_document(doc)
        assert all(c.page_number == 3 for c in chunks)

    def test_token_count_within_limit(self) -> None:
        chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=5)
        long_text = "The quick brown fox jumps over the lazy dog. " * 100
        doc = _make_doc(long_text)
        chunks = chunker.chunk_document(doc)
        # Allow slight overflow due to overlap mechanics
        for chunk in chunks:
            assert chunk.token_count <= 50 + 20, (
                f"Chunk {chunk.chunk_index} has {chunk.token_count} tokens"
            )

    def test_content_hash_unique_per_chunk(self) -> None:
        text = "\n\n".join([f"Unique paragraph {i}. " * 10 for i in range(30)])
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10)
        doc = _make_doc(text)
        chunks = chunker.chunk_document(doc)
        hashes = [c.content_hash for c in chunks]
        assert len(hashes) == len(set(hashes))

    def test_empty_document_returns_empty(self) -> None:
        chunker = RecursiveCharacterChunker()
        doc = _make_doc("")
        chunks = chunker.chunk_document(doc)
        assert chunks == []

    def test_text_chunk_from_text(self) -> None:
        chunk = TextChunk.from_text("Hello world!", chunk_index=0, page_number=1)
        assert chunk.text == "Hello world!"
        assert chunk.chunk_index == 0
        assert chunk.token_count > 0
        assert len(chunk.content_hash) == 64
