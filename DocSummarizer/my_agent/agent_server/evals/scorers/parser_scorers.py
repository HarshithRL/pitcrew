"""A1-A9 — Parser-stage scorers (deterministic, no LLM).

All scorers operate on the dict returned by ``evals.predict_fns.parser_predict``.
Each @scorer returns a ``Feedback(value=..., rationale=...)`` so the MLflow UI
shows both the score and the reason per row.

Optional ``expectations`` keys (omit any that aren't available for a given row):

    expected_min_chars        int      A1 floor on len(raw_text)
    expected_section_count    int      A2 exact section count
    expected_paths            list[list[str]]  A3/A9 ground-truth heading paths
    expected_depth            int      A4 exact max depth
    reference_markdown        str      A8/A9 client-provided parsed MD
    max_chunk_chars           int      A6 override (defaults to 6000)
"""

from __future__ import annotations

from mlflow.entities import Feedback
from mlflow.genai.scorers import scorer

from core.structure import parse_markdown_hierarchy

# Keep in sync with SECTION_MAX_CHARS in core.nodes.
DEFAULT_MAX_CHUNK_CHARS = 6000


def _paths_from(items: list[dict]) -> list[tuple]:
    """Return path tuples from a list of {"path": [...]} dicts."""
    return [tuple(item.get("path") or []) for item in items]


def _f1(got: set[tuple], want: set[tuple]) -> tuple[float, float, float]:
    if not got and not want:
        return 1.0, 1.0, 1.0
    inter = got & want
    p = len(inter) / len(got) if got else 0.0
    r = len(inter) / len(want) if want else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


# ── A1 ────────────────────────────────────────────────────────────────────
@scorer
def extraction_completeness(outputs, expectations) -> Feedback:
    """Catch scanned/image-only PDFs — raw_text must clear a min-chars floor."""
    floor = (expectations or {}).get("expected_min_chars")
    got = len(outputs.get("raw_text", ""))
    if floor is None:
        return Feedback(value=got > 0, rationale=f"raw_text chars={got} (no floor set)")
    return Feedback(value=got >= floor, rationale=f"raw_text chars={got} floor={floor}")


# ── A2 ────────────────────────────────────────────────────────────────────
@scorer
def section_count_match(outputs, expectations) -> Feedback:
    """Exact section count vs. expectations.expected_section_count."""
    want = (expectations or {}).get("expected_section_count")
    got = len(outputs.get("sections", []))
    if want is None:
        return Feedback(value=True, rationale=f"sections={got} (no expectation)")
    return Feedback(value=got == want, rationale=f"got={got} want={want}")


# ── A3 ────────────────────────────────────────────────────────────────────
@scorer
def heading_path_f1(outputs, expectations) -> Feedback:
    """F1 on the set of full heading paths vs. expectations.expected_paths."""
    want_raw = (expectations or {}).get("expected_paths")
    if not want_raw:
        return Feedback(value=True, rationale="no expected_paths provided")
    got = set(_paths_from(outputs.get("sections", [])))
    want = {tuple(p) for p in want_raw}
    p, r, f1 = _f1(got, want)
    return Feedback(value=f1, rationale=f"P={p:.2f} R={r:.2f} F1={f1:.2f}")


# ── A4 ────────────────────────────────────────────────────────────────────
@scorer
def hierarchy_depth(outputs, expectations) -> Feedback:
    """Max heading depth matches expectations.expected_depth."""
    want = (expectations or {}).get("expected_depth")
    sections = outputs.get("sections", [])
    got = max((len(s.get("path") or []) for s in sections), default=0)
    if want is None:
        return Feedback(value=True, rationale=f"max_depth={got} (no expectation)")
    return Feedback(value=got == want, rationale=f"got={got} want={want}")


# ── A5 ────────────────────────────────────────────────────────────────────
@scorer
def chunk_boundary_integrity(outputs) -> Feedback:
    """Every chunk path must match a real section path. No orphans."""
    section_paths = set(_paths_from(outputs.get("sections", [])))
    bad = [
        c["index"] for c in outputs.get("chunks", [])
        if tuple(c.get("path") or []) not in section_paths
    ]
    if bad:
        return Feedback(value=False, rationale=f"orphan chunks={bad[:10]}")
    return Feedback(value=True, rationale="all chunk paths align with sections")


# ── A6 ────────────────────────────────────────────────────────────────────
@scorer
def no_oversize_chunk(outputs, expectations) -> Feedback:
    """Every chunk.content is within max_chunk_chars."""
    cap = (expectations or {}).get("max_chunk_chars", DEFAULT_MAX_CHUNK_CHARS)
    over = [(c["index"], len(c["content"])) for c in outputs.get("chunks", []) if len(c["content"]) > cap]
    if over:
        return Feedback(value=False, rationale=f"oversize chunks (>{cap}): {over[:5]}")
    return Feedback(value=True, rationale=f"all chunks <= {cap} chars")


# ── A7 ────────────────────────────────────────────────────────────────────
@scorer
def no_sibling_collision(outputs) -> Feedback:
    """Two different parents with the same child title must stay distinct.

    Fails if two sections share the same (parent_path, child_title) tuple
    producing duplicate path tuples — that's the regression the hierarchy
    parser was fixed to prevent.
    """
    paths = _paths_from(outputs.get("sections", []))
    if len(paths) != len(set(paths)):
        dupes = {p for p in paths if paths.count(p) > 1}
        return Feedback(value=False, rationale=f"collapsed sibling paths: {list(dupes)[:5]}")
    return Feedback(value=True, rationale="all section paths unique")


# ── A8 ────────────────────────────────────────────────────────────────────
@scorer
def chars_tolerance_vs_reference(outputs, expectations) -> Feedback:
    """|len(raw_text) - len(reference_markdown)| / len(ref) < 10% (default)."""
    ref = (expectations or {}).get("reference_markdown")
    if not ref:
        return Feedback(value=True, rationale="no reference_markdown provided")
    tol = (expectations or {}).get("chars_tolerance", 0.10)
    got = len(outputs.get("raw_text", ""))
    ref_len = len(ref)
    drift = abs(got - ref_len) / ref_len if ref_len else 1.0
    return Feedback(
        value=drift < tol,
        rationale=f"got={got} ref={ref_len} drift={drift:.3f} tol={tol}",
    )


# ── A9 ────────────────────────────────────────────────────────────────────
@scorer
def headings_match_reference(outputs, expectations) -> Feedback:
    """F1 on heading paths vs. the client's parsed reference MD."""
    ref = (expectations or {}).get("reference_markdown")
    if not ref:
        return Feedback(value=True, rationale="no reference_markdown provided")
    ref_sections = parse_markdown_hierarchy(ref)
    want = {tuple(s["path"]) for s in ref_sections}
    got = set(_paths_from(outputs.get("sections", [])))
    p, r, f1 = _f1(got, want)
    return Feedback(value=f1, rationale=f"P={p:.2f} R={r:.2f} F1={f1:.2f}")


ALL = [
    extraction_completeness,
    section_count_match,
    heading_path_f1,
    hierarchy_depth,
    chunk_boundary_integrity,
    no_oversize_chunk,
    no_sibling_collision,
    chars_tolerance_vs_reference,
    headings_match_reference,
]
