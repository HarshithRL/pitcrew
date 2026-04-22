"""Extract raw post text from an uploaded PDF or DOCX file.

Mirrors the pymupdf4llm pattern in `core/nodes.py::ingest_node` so rule and
post extraction stay consistent. DOCX is handled by python-docx — plain
paragraph concatenation, no styling preserved.

Format detection is by magic bytes (no reliance on the file extension).
"""

from __future__ import annotations

import io
import logging

import pymupdf
import pymupdf4llm

from core.types import PostContent

logger = logging.getLogger(__name__)

_PDF_MAGIC = b"%PDF"
_DOCX_MAGIC = b"PK\x03\x04"


def extract_post(
    file_bytes: bytes,
    file_name: str,
    submission_id: str,
    platform: str | None,
) -> PostContent:
    """Return a `PostContent` object for a PDF or DOCX upload.

    Raises ValueError for any other format (the magic-bytes check is
    deliberate — relying on `.pdf` / `.docx` file extensions would let an
    arbitrary binary slip through).
    """
    head = file_bytes[:4]

    if head == _PDF_MAGIC:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        try:
            raw_text = pymupdf4llm.to_markdown(doc)
        finally:
            doc.close()
        file_type = "pdf"

    elif head == _DOCX_MAGIC:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        raw_text = "\n\n".join(paragraphs)
        file_type = "docx"

    else:
        raise ValueError(
            f"Unsupported format: {file_name}. "
            "Only PDF and DOCX files are accepted."
        )

    stripped = raw_text.strip()
    logger.info(
        "extracted post %s (%s, %d chars)",
        file_name, file_type, len(stripped),
    )
    return PostContent(
        submission_id=submission_id,
        raw_text=stripped,
        platform=platform,
        file_name=file_name,
        file_type=file_type,
        char_count=len(raw_text),
    )
