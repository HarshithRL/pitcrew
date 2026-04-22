"""B1-B7 — Per-chunk summarizer scorers.

Mix of LLM judges (Guidelines) and deterministic @scorer functions. Each row's
``inputs`` must contain the chunk's ``path`` and ``content`` — the chunk
predict_fn turns that into the summary (``outputs``).

Optional ``expectations`` keys:

    expected_response    str    B7 SME gold summary (reads via Correctness())
    min_bullets          int    B4 override (default 2)
    max_bullets          int    B4 override (default 10)
    max_chars            int    B4 override (default 2000)
"""

from __future__ import annotations

import re

from mlflow.entities import Feedback
from mlflow.genai.scorers import Correctness, Guidelines, scorer

# ── B1, B2 — LLM judges ───────────────────────────────────────────────────

groundedness = Guidelines(
    name="grounded",
    guidelines=(
        "Every bullet in the response must be supported by the provided chunk "
        "content. The response must not introduce facts, numbers, dates, names, "
        "causes, outlooks, or comparisons that are absent from the chunk."
    ),
)

path_fidelity = Guidelines(
    name="on_topic",
    guidelines=(
        "The response must only discuss content within the given heading path. "
        "It must not include content that belongs to a parent, sibling, or "
        "unrelated section, and must not echo the heading path or section names."
    ),
)


# ── B3 — Format compliance ────────────────────────────────────────────────

_BULLET = re.compile(r"^\s*[-*]\s+.+")


@scorer
def format_compliance(outputs) -> Feedback:
    """Every non-empty line must be a bullet (``- `` or ``* `` prefix)."""
    text = outputs if isinstance(outputs, str) else str(outputs)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return Feedback(value=False, rationale="empty summary")
    bad = [ln for ln in lines if not _BULLET.match(ln)]
    if bad:
        return Feedback(value=False, rationale=f"non-bullet lines: {bad[:3]}")
    return Feedback(value=True, rationale=f"{len(lines)} bullets, all well-formed")


# ── B4 — Length bounds ────────────────────────────────────────────────────

@scorer
def length_bounds(outputs, expectations) -> Feedback:
    """Bullet count in [min, max] and total length <= max_chars."""
    text = outputs if isinstance(outputs, str) else str(outputs)
    exp = expectations or {}
    min_b = exp.get("min_bullets", 2)
    max_b = exp.get("max_bullets", 10)
    max_c = exp.get("max_chars", 2000)

    bullets = [ln for ln in text.splitlines() if _BULLET.match(ln)]
    n = len(bullets)
    if not (min_b <= n <= max_b):
        return Feedback(value=False, rationale=f"bullets={n} want {min_b}..{max_b}")
    if len(text) > max_c:
        return Feedback(value=False, rationale=f"len={len(text)} > max_chars={max_c}")
    return Feedback(value=True, rationale=f"bullets={n}, len={len(text)}")


# ── B5 — Number preservation ──────────────────────────────────────────────

# Matches integers, decimals, percentages, currency-prefixed numbers, and
# simple period-separated versions (e.g. "12.3", "94%", "$4.2B", "Q2 2026").
_NUMBER = re.compile(
    r"(?<![A-Za-z])"
    r"(?:\$?\d[\d,]*(?:\.\d+)?(?:[%kKmMbB])?|\$\d[\d,]*(?:\.\d+)?[kKmMbB]?)"
)


def _numbers(text: str) -> set[str]:
    return {m.group(0).strip().rstrip(".,") for m in _NUMBER.finditer(text or "")}


@scorer
def number_preservation(inputs, outputs) -> Feedback:
    """Every number in the summary must appear verbatim in the chunk content."""
    content = (inputs or {}).get("content", "") or ""
    summary = outputs if isinstance(outputs, str) else str(outputs)
    in_src = _numbers(content)
    in_out = _numbers(summary)
    fabricated = sorted(in_out - in_src)
    if fabricated:
        return Feedback(
            value=False,
            rationale=f"numbers in summary missing from chunk: {fabricated[:5]}",
        )
    return Feedback(value=True, rationale=f"{len(in_out)} numbers, all traceable")


# ── B6 — No prompt leakage ────────────────────────────────────────────────

_LEAK_PATTERNS = [
    r"\bHierarchy\s*:",          # the per-chunk human-message header
    r"\bHere(?:'s| is)\b",       # common preamble
    r"^Summary\s*[:\-]",         # explicit "Summary:" label
    r"\bself[- ]check\b",        # system-prompt phrase
    r"\bsystem prompt\b",
]

_LEAK_RE = re.compile("|".join(_LEAK_PATTERNS), flags=re.IGNORECASE | re.MULTILINE)


@scorer
def no_prompt_leakage(outputs) -> Feedback:
    """Summary must not echo prompt scaffolding or heading-path strings."""
    text = outputs if isinstance(outputs, str) else str(outputs)
    matches = _LEAK_RE.findall(text)
    if matches:
        return Feedback(value=False, rationale=f"leak markers: {matches[:3]}")
    return Feedback(value=True, rationale="no prompt scaffolding detected")


# ── B7 — Correctness vs. SME chunk summary ────────────────────────────────

correctness = Correctness()


ALL = [
    groundedness,
    path_fidelity,
    format_compliance,
    length_bounds,
    number_preservation,
    no_prompt_leakage,
    correctness,
]
