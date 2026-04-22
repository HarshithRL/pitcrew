"""Dataset builders for the three eval stages.

Expected input layout — one directory per document::

    eval_data/
      doc_001/
        source.pdf             # the reference PDF
        parsed.md              # client's reference parsed Markdown
        summary.md             # client's reference summary Markdown
      doc_002/
        ...

Either ``source.pdf`` or ``source.md`` / ``source.txt`` is accepted. The
parsed + summary files are optional: rows still run but only the scorers that
don't need that ground truth will produce a meaningful score.

Three builders turn the on-disk layout into the eval-row shapes MLflow
expects (``inputs`` + ``expectations``):

    build_parser_rows(root)  -> list[dict]   for predict_fns.parser_predict
    build_chunk_rows(root)   -> list[dict]   for predict_fns.chunk_predict
    build_doc_rows(root)     -> list[dict]   for predict_fns.doc_predict
"""

from __future__ import annotations

from pathlib import Path

from core.structure import parse_markdown_hierarchy, split_large_section

_SOURCE_CANDIDATES = ("source.pdf", "source.md", "source.txt")


def _find_source(doc_dir: Path) -> Path | None:
    for name in _SOURCE_CANDIDATES:
        candidate = doc_dir / name
        if candidate.exists():
            return candidate
    return None


def _read_if_exists(path: Path) -> str | None:
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    return None


def _iter_doc_dirs(root: str | Path):
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"eval data root not found: {root_path}")
    for entry in sorted(root_path.iterdir()):
        if entry.is_dir():
            yield entry


# ── A. Parser rows ────────────────────────────────────────────────────────

def build_parser_rows(root: str | Path) -> list[dict]:
    """One row per doc. Ground truth comes from the parsed.md if present."""
    rows: list[dict] = []
    for doc_dir in _iter_doc_dirs(root):
        source = _find_source(doc_dir)
        if source is None:
            continue
        parsed_md = _read_if_exists(doc_dir / "parsed.md")
        expectations: dict = {}
        if parsed_md:
            ref_sections = parse_markdown_hierarchy(parsed_md)
            expectations.update({
                "reference_markdown": parsed_md,
                "expected_paths": [s["path"] for s in ref_sections],
                "expected_section_count": len(ref_sections),
                "expected_depth": max((len(s["path"]) for s in ref_sections), default=0),
                "expected_min_chars": max(1, int(len(parsed_md) * 0.5)),
            })
        rows.append({
            "inputs": {"source_path": str(source)},
            "expectations": expectations,
            "tags": {"doc_id": doc_dir.name, "source": "client_reference"},
        })
    return rows


# ── B. Per-chunk rows ─────────────────────────────────────────────────────

def build_chunk_rows(root: str | Path, max_chars: int = 6000) -> list[dict]:
    """One row per chunk derived from each doc's parsed.md.

    We chunk from the client's reference MD (not the PDF) so the summarizer
    is tested on the exact same content the client considers ground truth for
    the parsing stage.
    """
    rows: list[dict] = []
    for doc_dir in _iter_doc_dirs(root):
        parsed_md = _read_if_exists(doc_dir / "parsed.md")
        if not parsed_md:
            continue
        for section in parse_markdown_hierarchy(parsed_md):
            for piece in split_large_section(section, max_chars=max_chars):
                rows.append({
                    "inputs": {
                        "path": list(piece["path"]),
                        "content": piece["content"],
                    },
                    "expectations": {},
                    "tags": {
                        "doc_id": doc_dir.name,
                        "path": " > ".join(piece["path"]),
                    },
                })
    return rows


# ── C. End-to-end doc rows ────────────────────────────────────────────────

def build_doc_rows(root: str | Path) -> list[dict]:
    """One row per doc. summary.md populates expected_response for Correctness."""
    rows: list[dict] = []
    for doc_dir in _iter_doc_dirs(root):
        source = _find_source(doc_dir)
        if source is None:
            continue
        parsed_md = _read_if_exists(doc_dir / "parsed.md")
        summary_md = _read_if_exists(doc_dir / "summary.md")
        expectations: dict = {}
        if summary_md:
            expectations["expected_response"] = summary_md
            expectations["reference_summary"] = summary_md
        if parsed_md:
            expectations["reference_source_markdown"] = parsed_md
            expectations["expected_paths"] = [
                s["path"] for s in parse_markdown_hierarchy(parsed_md)
            ]
        rows.append({
            "inputs": {"source_path": str(source)},
            "expectations": expectations,
            "tags": {"doc_id": doc_dir.name, "source": "client_reference"},
        })
    return rows
