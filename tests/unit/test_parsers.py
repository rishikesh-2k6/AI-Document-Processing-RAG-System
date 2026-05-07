"""Unit tests for the document parsers (PDF, DOCX, EML)."""

from __future__ import annotations

import io
import textwrap

import pytest

from app.ingestion.parsers import DOCXParser, EMLParser, PDFParser, get_parser, parse_document


# ── EML Parser ────────────────────────────────────────────────────────────────


class TestEMLParser:
    """Tests for the EML email parser."""

    SAMPLE_EML = textwrap.dedent("""\
        From: alice@example.com
        To: bob@example.com
        Subject: Quarterly Report
        Date: Fri, 01 Jan 2021 10:00:00 +0000
        Content-Type: text/plain; charset=utf-8

        Hello Bob,

        Please find the Q4 report attached.

        Best regards,
        Alice
    """).encode()

    def test_parse_returns_parsed_document(self) -> None:
        parser = EMLParser()
        result = parser.parse(self.SAMPLE_EML, "test.eml")
        assert result.page_count == 1
        assert len(result.pages) == 1

    def test_parse_extracts_subject(self) -> None:
        parser = EMLParser()
        result = parser.parse(self.SAMPLE_EML, "test.eml")
        assert "Quarterly Report" in result.pages[0].text

    def test_parse_extracts_body(self) -> None:
        parser = EMLParser()
        result = parser.parse(self.SAMPLE_EML, "test.eml")
        assert "Q4 report" in result.pages[0].text

    def test_parse_metadata(self) -> None:
        parser = EMLParser()
        result = parser.parse(self.SAMPLE_EML, "test.eml")
        assert result.doc_metadata["from"] == "alice@example.com"
        assert result.doc_metadata["subject"] == "Quarterly Report"

    def test_word_count_positive(self) -> None:
        parser = EMLParser()
        result = parser.parse(self.SAMPLE_EML, "test.eml")
        assert result.word_count > 0

    def test_content_hash_is_sha256(self) -> None:
        parser = EMLParser()
        result = parser.parse(self.SAMPLE_EML, "test.eml")
        assert len(result.content_hash) == 64  # SHA-256 hex

    def test_page_number_is_one(self) -> None:
        parser = EMLParser()
        result = parser.parse(self.SAMPLE_EML, "test.eml")
        assert result.pages[0].page_number == 1


# ── Parser registry ───────────────────────────────────────────────────────────


class TestParserRegistry:
    """Tests for the parser registry and factory function."""

    def test_get_parser_pdf(self) -> None:
        parser = get_parser("pdf")
        assert isinstance(parser, PDFParser)

    def test_get_parser_docx(self) -> None:
        parser = get_parser("docx")
        assert isinstance(parser, DOCXParser)

    def test_get_parser_eml(self) -> None:
        parser = get_parser("eml")
        assert isinstance(parser, EMLParser)

    def test_get_parser_msg(self) -> None:
        parser = get_parser("msg")
        assert isinstance(parser, EMLParser)

    def test_get_parser_unsupported_raises(self) -> None:
        with pytest.raises(ValueError, match="No parser for extension"):
            get_parser("xlsx")

    def test_get_parser_case_insensitive(self) -> None:
        parser = get_parser("PDF")
        assert isinstance(parser, PDFParser)

    def test_parse_document_routes_by_extension(self) -> None:
        sample = b"From: a@b.com\nSubject: Test\n\nBody"
        result = parse_document(sample, "test.eml")
        assert result.page_count >= 1
