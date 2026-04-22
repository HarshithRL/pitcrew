"""Structure-aware Markdown utilities.

Two pure, deterministic helpers that power the structure-first summarization
pipeline. They do NOT touch any LLM — all hierarchy work happens here so the
LLM is only asked to summarize local content.

Design rules enforced:
    * Every heading level ``#`` through ``######`` is captured — not only H1/H2.
      A stack keyed by heading level is maintained so that a shallower heading
      resets every deeper level currently in scope. This prevents subsections
      with identical names ("(A) Requirement" under two different parents) from
      being merged downstream.
    * Fenced code blocks and pipe-tables are emitted as single atomic blocks —
      never split mid-row, never leaked across a heading boundary.
    * Content ordering matches the source document exactly (each group is
      appended when the next heading or end-of-document is reached).
"""

from __future__ import annotations

import re

# Match any Markdown ATX heading H1-H6. The capture groups give us (hashes, title).
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# Content before the first heading is parked here so nothing is lost.
_DEFAULT_SECTION = "Introduction"


def parse_markdown_hierarchy(md_text: str) -> list[dict]:
    """Parse Markdown into ordered groups that carry the FULL heading path.

    Each element of the returned list represents one contiguous block of
    content belonging to exactly one (H1 > H2 > ... > Hn) path, in document
    order::

        {"path": list[str], "content": list[str]}

    ``path`` is the ordered list of heading titles from the outermost heading
    currently in scope to the most specific one. A heading at level L clears
    every heading entry at level >= L before being pushed, so the stack
    always reflects the live hierarchy.

    ``content`` is a list of atomic blocks (paragraphs, lists, tables, fenced
    code). Blank lines between blocks are dropped; the blocks themselves are
    preserved verbatim so numbers, tables, and code survive intact.
    """
    lines = md_text.splitlines()

    groups: list[dict] = []
    # level (1..6) -> current heading title at that level
    headings_by_level: dict[int, str] = {}
    blocks: list[str] = []   # atomic blocks belonging to the current path
    buf: list[str] = []      # lines being accumulated into the current paragraph

    def current_path() -> list[str]:
        if not headings_by_level:
            return [_DEFAULT_SECTION]
        # Sorted by level so path is outermost-first, regardless of insertion order.
        return [title for _, title in sorted(headings_by_level.items())]

    def flush_paragraph() -> None:
        if buf:
            blocks.append("\n".join(buf).rstrip())
            buf.clear()

    def flush_group() -> None:
        if blocks:
            groups.append({
                "path": current_path(),
                "content": list(blocks),
            })
            blocks.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Fenced code block — keep intact as one atomic block.
        if stripped.startswith("```"):
            flush_paragraph()
            code = [line]
            i += 1
            while i < len(lines):
                code.append(lines[i])
                if lines[i].strip().startswith("```"):
                    i += 1
                    break
                i += 1
            blocks.append("\n".join(code))
            continue

        # Pipe-table — keep contiguous rows intact as one atomic block.
        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph()
            table = [line]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                table.append(lines[i])
                i += 1
            blocks.append("\n".join(table))
            continue

        # Any ATX heading pushes onto the stack and flushes the previous group.
        m = _HEADING.match(line)
        if m:
            flush_paragraph()
            flush_group()
            level = len(m.group(1))
            title = m.group(2).strip()
            # Reset every heading at or below the new level.
            for lvl in [lv for lv in headings_by_level if lv >= level]:
                del headings_by_level[lvl]
            headings_by_level[level] = title
            i += 1
            continue

        # Blank line ends the current paragraph.
        if stripped == "":
            flush_paragraph()
            i += 1
            continue

        buf.append(line)
        i += 1

    flush_paragraph()
    flush_group()
    return groups


def split_large_section(section: dict, max_chars: int = 6000) -> list[dict]:
    """Split one parsed (path, content) group into LLM-sized chunks.

    Returns a list of dicts shaped for the ``ChunkJob`` payload::

        {"path": list[str], "content": str}

    Rules:
        * If the joined content fits within ``max_chars`` → one chunk.
        * Otherwise → pack atomic blocks greedily up to ``max_chars`` each.
        * A single block larger than ``max_chars`` (e.g. a huge table) is
          emitted as its own oversized chunk rather than split mid-row.
        * ``path`` is copied intact onto every chunk so downstream aggregation
          can rebuild the hierarchy without any LLM help.
    """
    path: list[str] = list(section.get("path") or [_DEFAULT_SECTION])
    content_blocks: list[str] = section.get("content", [])

    joined = "\n\n".join(b for b in content_blocks if b.strip())
    if not joined.strip():
        return []

    if len(joined) <= max_chars:
        return [{"path": path, "content": joined}]

    chunks: list[dict] = []
    current: list[str] = []
    current_len = 0

    for block in content_blocks:
        if not block.strip():
            continue
        # +2 accounts for the "\n\n" glue between blocks.
        projected = current_len + len(block) + (2 if current else 0)
        if current and projected > max_chars:
            chunks.append({"path": list(path), "content": "\n\n".join(current)})
            current = []
            current_len = 0
        current.append(block)
        current_len += len(block) + (2 if current_len else 0)

    if current:
        chunks.append({"path": list(path), "content": "\n\n".join(current)})

    return chunks


# Backwards-compatible alias for any caller that still imports the old name.
parse_markdown_sections = parse_markdown_hierarchy
