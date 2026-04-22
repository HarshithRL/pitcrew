"""Graph nodes for the structure-aware document summarizer.

Pipeline:
    ingest -> parse_structure -> chunk -> summarize_chunk (fan-out) -> aggregate -> output

The LLM is invoked ONLY inside ``summarize_chunk_node`` — hierarchy extraction,
chunking, and final aggregation are deterministic Python. That keeps numbers
exact, prevents the model from inventing or merging sections, and keeps the
pipeline reproducible.

Key design fixes (vs. the prior two-level section/subsection version):
  * Every chunk carries the FULL heading path (list[str]), not just two levels.
  * Aggregation builds a nested tree keyed by path and renders recursively, so
    sibling subsections with identical names ("(A) Requirement" under two
    different parents) remain distinct instead of being merged.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.messages import SystemMessage
from langgraph.types import Send

from core.config import get_llm
from core.prompts import (
    FEW_SHOT_MESSAGES,
    SUMMARIZE_SYSTEM_PROMPT,
    as_text,
    build_human_message,
)
from core.state import ChunkJob, ChunkSummary, DocState, SectionChunk
from core.structure import parse_markdown_hierarchy, split_large_section

logger = logging.getLogger(__name__)

# Upper bound per LLM call. Only applied inside a single (path) group;
# never used to split across different paths.
SECTION_MAX_CHARS = 6000


# ── Ingest ────────────────────────────────────────────────────────────────

def _read_pdf_as_markdown(path: Path) -> str:
    """Extract PDF as Markdown with headings, lists, and tables preserved."""
    import pymupdf4llm

    # Default overload returns a single Markdown string. Cast for type checkers.
    return str(pymupdf4llm.to_markdown(str(path)))


def ingest_node(state: DocState) -> dict:
    """Load ``source_path`` as Markdown. Supports .txt, .md, .pdf."""
    path = Path(state.source_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    elif suffix == ".pdf":
        raw = _read_pdf_as_markdown(path)
    else:
        raise NotImplementedError(f"Parser for {suffix} not wired yet.")

    if not raw.strip():
        raise ValueError(
            "No extractable text — the document may be a scanned or image-only PDF."
        )

    logger.info("ingested %s (%d chars)", path.name, len(raw))
    return {
        "raw_text": raw,
        "metadata": {"filename": path.name, "char_count": len(raw)},
    }


# ── Structure extraction & chunking (deterministic) ───────────────────────

def parse_structure_node(state: DocState) -> dict:
    """Turn Markdown into an ordered list of (path, content) groups.

    Uses the full-hierarchy parser so every H1-H6 level is captured in
    document order.
    """
    sections = parse_markdown_hierarchy(state.raw_text)
    logger.info("parsed %d hierarchy groups from markdown", len(sections))
    # Per-section path trace — DEBUG so it's silent in prod, on tap when needed.
    # Verifies every group carries the full ancestor chain (parent missing = bug).
    for s in sections:
        logger.debug("PATH: %s", s["path"])
    return {"sections": sections}


def chunk_node(state: DocState) -> dict:
    """Build a flat list of ``SectionChunk``s, one per LLM invocation.

    Each chunk is bound to exactly one path. Groups that fit within
    ``SECTION_MAX_CHARS`` become a single chunk; oversized groups are split
    internally while keeping the full path intact on every piece.
    """
    chunks: list[SectionChunk] = []
    for section in state.sections:
        for piece in split_large_section(section, max_chars=SECTION_MAX_CHARS):
            chunks.append(SectionChunk(
                index=len(chunks),
                path=list(piece["path"]),
                content=piece["content"],
            ))
    logger.info("built %d hierarchy-aware chunks", len(chunks))
    for c in chunks:
        logger.debug("CHUNK %d PATH: %s", c.index, c.path)
    return {"chunks": chunks}


def fan_out_chunks(state: DocState) -> list[Send]:
    """Dispatch one ``summarize_chunk`` invocation per chunk (Send API)."""
    return [
        Send(
            "summarize_chunk",
            ChunkJob(
                index=c.index,
                path=list(c.path),
                content=c.content,
                source_path=state.source_path,
            ),
        )
        for c in state.chunks
    ]


# ── Local summarization (the ONLY LLM call in the pipeline) ───────────────

def summarize_chunk_node(job: ChunkJob) -> dict:
    """Summarize a single path-scoped chunk into bullet points.

    Message layout (all but the last message are constant → cacheable)::

        [ System:  SUMMARIZE_SYSTEM_PROMPT                       ]  cacheable
        [ Human:   <few-shot example 1>                          ]  cacheable
        [ AI:      <few-shot output 1>                           ]  cacheable
        [ Human:   <few-shot example 2 (with table)>             ]  cacheable
        [ AI:      <few-shot output 2>                           ]  cacheable
        [ Human:   Hierarchy: A > B > C\\n\\n<chunk content>     ]  per-chunk

    Per-chunk context lives only in the final human message so the system
    prompt and few-shot stay identical across every invocation, keeping the
    prompt-cache hit rate high.
    """
    messages = [
        SystemMessage(content=SUMMARIZE_SYSTEM_PROMPT),
        *FEW_SHOT_MESSAGES,
        build_human_message(job.path, job.content),
    ]
    llm = get_llm(max_tokens=400)
    out = llm.invoke(messages).content
    return {
        "chunk_summaries": [ChunkSummary(
            index=job.index,
            path=list(job.path),
            summary=as_text(out).strip(),
        )]
    }


# ── Deterministic aggregation (NO LLM) ────────────────────────────────────

def _insert_into_tree(root: dict, path: list[str], summary: str, index: int) -> None:
    """Insert one bullet block into the nested tree at the exact path.

    Each node: {"summaries": [(index, summary), ...], "children": OrderedDict[title -> node]}
    Using an ordered child dict (Python 3.7+ dicts preserve insertion order)
    means child nodes render in first-appearance order, which matches the
    source document when summaries are inserted in ``index`` order.
    """
    node = root
    for title in path:
        if title not in node["children"]:
            node["children"][title] = {"summaries": [], "children": {}}
        node = node["children"][title]
    node["summaries"].append((index, summary))


def _render_tree(node: dict, depth: int, lines: list[str]) -> None:
    """Render the tree as Markdown, using depth N -> (N+1) `#` characters.

    depth=1 -> `##` (top-level section), depth=2 -> `###` (subsection),
    depth=3 -> `####` (sub-subsection), and so on. Clamped at H6.
    """
    for title, child in node["children"].items():
        level = min(depth + 1, 6)
        lines.append(f"{'#' * level} {title}")
        lines.append("")
        # Preserve source order if multiple chunks were produced for the same path.
        for _, bullets in sorted(child["summaries"], key=lambda x: x[0]):
            lines.append(bullets)
            lines.append("")
        _render_tree(child, depth + 1, lines)


def aggregate_node(state: DocState) -> dict:
    """Build a nested tree keyed by full path, then render it recursively.

    Grouping is done by tuple-of-path so that sibling nodes with identical
    names at different parents stay distinct (this is the fix for the
    "(A) Requirement" merge bug). Document order is preserved by inserting
    summaries in ``index`` order — since the tree uses insertion-order dicts,
    the first time a heading is seen it locks into position for rendering.
    """
    ordered = sorted(state.chunk_summaries, key=lambda s: s.index)

    # Nested tree. Top-level children are the outermost headings in the document.
    root: dict = {"summaries": [], "children": {}}
    for s in ordered:
        path = list(s.path) if s.path else ["Introduction"]
        _insert_into_tree(root, path, s.summary, s.index)

    lines: list[str] = []
    # depth=1 at the start -> top-level headings render as `##` (matches spec).
    _render_tree(root, depth=1, lines=lines)

    return {"final_summary": "\n".join(lines).rstrip() + "\n"}


# ── Terminal ──────────────────────────────────────────────────────────────

def output_node(state: DocState) -> dict:
    """Terminal hook — metadata is already in state; reserved for export/formatting."""
    logger.info(
        "summary ready for %s (%d distinct paths, %d chunks)",
        state.metadata.get("filename", "?"),
        len({tuple(s.path) for s in state.chunk_summaries}),
        len(state.chunk_summaries),
    )
    return {}
