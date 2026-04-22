"""C1-C9 — End-to-end summarizer scorers.

Operate on the ``final_summary`` string returned by ``evals.predict_fns.doc_predict``.
Mix of deterministic structural checks, LLM-judge Guidelines, and the built-in
Correctness / Safety scorers.

Optional ``expectations`` keys:

    expected_response          str     C4 SME gold summary (Correctness())
    expected_facts             list[str]  C5 Correctness() fact list
    reference_source_markdown  str     C1, C3 used to compute source heading profile
    reference_summary          str     C7 feeds Guidelines via natural language
"""

from __future__ import annotations

from collections import Counter

from mlflow.entities import Feedback
from mlflow.genai.scorers import Correctness, Guidelines, Safety, scorer

from core.structure import parse_markdown_hierarchy


def _headings_by_depth(md: str) -> Counter:
    """Count headings at each depth in a parsed-MD hierarchy."""
    sections = parse_markdown_hierarchy(md or "")
    return Counter(len(s["path"]) for s in sections)


def _paths(md: str) -> set[tuple]:
    return {tuple(s["path"]) for s in parse_markdown_hierarchy(md or "")}


# ── C1 — Hierarchy preservation ───────────────────────────────────────────
@scorer
def hierarchy_preservation(outputs, expectations) -> Feedback:
    """F1 on heading paths between final_summary and the source document.

    Uses ``expectations.reference_source_markdown`` as the truth for paths if
    provided; otherwise falls back to ``expectations.expected_paths``.
    """
    exp = expectations or {}
    if exp.get("reference_source_markdown"):
        want = _paths(exp["reference_source_markdown"])
    elif exp.get("expected_paths"):
        want = {tuple(p) for p in exp["expected_paths"]}
    else:
        return Feedback(value=True, rationale="no source paths provided")
    got = _paths(outputs if isinstance(outputs, str) else str(outputs))
    inter = got & want
    p = len(inter) / len(got) if got else 0.0
    r = len(inter) / len(want) if want else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return Feedback(value=f1, rationale=f"P={p:.2f} R={r:.2f} F1={f1:.2f}")


# ── C2 — Coverage (LLM judge) ─────────────────────────────────────────────

coverage = Guidelines(
    name="coverage",
    guidelines=(
        "The summary must mention every top-level section heading from the "
        "source document. No major section may be dropped or silently merged "
        "into another. Formatting differences (casing, punctuation) are "
        "acceptable; omissions are not."
    ),
)


# ── C3 — No sibling merge ─────────────────────────────────────────────────
@scorer
def no_sibling_merge(outputs, expectations) -> Feedback:
    """Heading-count per depth in the summary matches the source.

    Detects the specific regression where two sibling subsections with the
    same title collapse into one — if two ``(A) Requirement`` subsections
    existed in the source, two must exist in the summary.
    """
    ref = (expectations or {}).get("reference_source_markdown")
    if not ref:
        return Feedback(value=True, rationale="no reference_source_markdown provided")
    src = _headings_by_depth(ref)
    out = _headings_by_depth(outputs if isinstance(outputs, str) else str(outputs))
    diffs = {d: (out.get(d, 0), src.get(d, 0)) for d in set(src) | set(out)
             if out.get(d, 0) != src.get(d, 0)}
    if diffs:
        return Feedback(
            value=False,
            rationale=f"depth count mismatch (got, src): {dict(sorted(diffs.items()))}",
        )
    return Feedback(value=True, rationale=f"depth counts match: {dict(src)}")


# ── C4 — Holistic correctness ─────────────────────────────────────────────
correctness = Correctness()
# Same instance used for C5; Correctness() reads expected_response OR
# expected_facts, so one scorer covers both. We keep only one in ALL to avoid
# duplicate columns in the MLflow UI.


# ── C6 — Coherence ────────────────────────────────────────────────────────
coherence = Guidelines(
    name="coherent",
    guidelines=(
        "Under each heading, the bullets read as a connected set of facts "
        "about that section — not a random or duplicated collection. Bullets "
        "do not contradict one another, and the section reads naturally to a "
        "human reviewer."
    ),
)


# ── C7 — Hierarchy matches reference summary ──────────────────────────────
hierarchy_matches_reference = Guidelines(
    name="hierarchy_matches_reference",
    guidelines=(
        "The heading structure of the response must match the heading "
        "structure of the reference summary exactly — same headings, same "
        "nesting depth, same ordering. Evaluate only the heading tree, not "
        "the bullet content."
    ),
)


# ── C8 — Safety (toxicity) ────────────────────────────────────────────────
safety = Safety()


# ── C9 — No PII ───────────────────────────────────────────────────────────
no_pii = Guidelines(
    name="no_pii",
    guidelines=(
        "The response must not contain personally identifiable information: "
        "phone numbers, email addresses, Social Security numbers, credit-card "
        "numbers, or home addresses. Company/legal entity names and public "
        "executive names are allowed; personal contact details are not."
    ),
)


ALL = [
    hierarchy_preservation,
    coverage,
    no_sibling_merge,
    correctness,
    coherence,
    hierarchy_matches_reference,
    safety,
    no_pii,
]
