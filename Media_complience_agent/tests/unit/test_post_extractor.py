"""Unit tests for core.post_extractor."""

from __future__ import annotations

import io

import docx as python_docx
import pytest

from core.post_extractor import extract_post


def make_docx_bytes(text: str) -> bytes:
    doc = python_docx.Document()
    for line in text.split("\n"):
        if line.strip():
            doc.add_paragraph(line)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_unsupported_format_raises():
    with pytest.raises(ValueError, match="Unsupported format"):
        extract_post(b"not a real file", "test.txt", "sub-001", None)


def test_docx_extraction():
    text = "Our fund guarantees 15% returns.\nInvest now before it is too late."
    docx_bytes = make_docx_bytes(text)
    result = extract_post(docx_bytes, "post.docx", "sub-001", "linkedin")
    assert result.file_type == "docx"
    assert result.platform == "linkedin"
    assert "15%" in result.raw_text
    assert result.char_count > 0


def test_post_content_fields():
    text = "Simple investment post."
    docx_bytes = make_docx_bytes(text)
    result = extract_post(docx_bytes, "test.docx", "sub-002", "twitter")
    assert result.submission_id == "sub-002"
    assert result.file_name == "test.docx"
    assert result.platform == "twitter"
