"""Shared state for the structure-aware document summarizer.

Pipeline shape:

    ingest -> parse_structure -> chunk -> summarize_chunk (fan-out) -> aggregate -> output

Each chunk carries the FULL heading path (list[str]) so the aggregation step
can rebuild the exact document hierarchy deterministically — no LLM involved.
Using a list (not just section + subsection) is what prevents sibling
subsections with identical names from being merged downstream.
"""

from operator import add
from typing import Annotated, Optional

from pydantic import BaseModel, Field


class SectionChunk(BaseModel):
    """One LLM-sized unit of content tagged with its full position in the hierarchy."""

    index: int                          # global ordering — matches source-document order
    path: list[str] = Field(default_factory=list)
    content: str                        # joined Markdown content for the LLM


class ChunkSummary(BaseModel):
    """Bullet-point summary of a single SectionChunk — emitted by the map step."""

    index: int
    path: list[str] = Field(default_factory=list)
    summary: str


class DocState(BaseModel):
    """State flowing through the summarizer graph.

    ``chunk_summaries`` uses ``operator.add`` as a reducer so results from
    parallel Send-dispatched summarize_chunk calls concatenate instead of
    overwriting each other.
    """

    source_path: str = ""
    # Markdown-formatted extracted content. PDFs are converted to MD via
    # pymupdf4llm; .md files pass through; .txt is treated as plain paragraphs.
    raw_text: str = ""
    # Ordered list of {"path": list[str], "content": list[str]} groups
    # produced by structure.parse_markdown_hierarchy — NOT fed to the LLM.
    sections: list[dict] = Field(default_factory=list)
    # Flat list of section-aware chunks, one LLM invocation per chunk.
    chunks: list[SectionChunk] = Field(default_factory=list)
    chunk_summaries: Annotated[list[ChunkSummary], add] = Field(default_factory=list)
    final_summary: str = ""
    metadata: dict = Field(default_factory=dict)


class ChunkJob(BaseModel):
    """Payload sent to each parallel summarize_chunk invocation via Send API."""

    index: int
    path: list[str] = Field(default_factory=list)
    content: str
    source_path: Optional[str] = None
