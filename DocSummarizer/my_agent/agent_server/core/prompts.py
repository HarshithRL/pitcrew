"""Prompt contract for the section-aware summarizer.

DESIGN NOTES (why this file is shaped the way it is):

1.  The system prompt is a CONSTANT. Per-chunk context (the full heading
    hierarchy) lives in the human message. This lets prompt caching kick in:
    the system prompt + few-shot examples are identical on every call, so
    cache hits cover everything except the final user turn. Interpolating
    path values into the system prompt would defeat caching entirely.

2.  Few-shot examples are delivered as real Human/Assistant message pairs,
    not as quoted blobs inside the system prompt. This is the pattern Claude
    responds to most reliably: the model sees prior "correct" turns and
    matches the structure.

3.  The chunk is presented together with a single ``Hierarchy:`` line
    listing every ancestor heading joined by `` > ``. This is context ONLY —
    the rules below forbid the model from echoing it.

4.  Every rule is phrased as an explicit constraint (must / must not /
    exactly) rather than a preference. Open-ended language is avoided on
    purpose.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logger = logging.getLogger(__name__)


# ── System prompt (constant — cacheable) ──────────────────────────────────

SUMMARIZE_SYSTEM_PROMPT = """You are given content from a structured document. Your only job is to summarize that one chunk.

The user message will start with a `Hierarchy:` line that names the ancestor headings of the chunk, joined by ` > `, followed by a blank line and the chunk content. Treat the hierarchy as CONTEXT ONLY — never repeat the section names, never emit a heading, never refer back to other sections.

OUTPUT FORMAT (all must hold):
- EXACTLY 3 to 5 bullet points.
- Every bullet starts with "- " (dash + single space) and contains one fact.
- No blank lines between bullets. No headings. No numbering. No nested bullets.
- No preamble ("Here is a summary..."), no closing remarks, no meta-commentary.
- No bold, italic, or backticks — except when quoting a table cell value verbatim.

CONTENT RULES:
- Use ONLY facts present in the provided chunk content.
- Every number, date, percentage, currency amount, and proper name in your output MUST appear verbatim in the input. Do not round, convert units, restate in words, or reformat ("$4.2B" stays "$4.2B"; "12.3%" stays "12.3%"; "Q2 2026" stays "Q2 2026").
- Do not paraphrase numeric values.
- Do not infer or invent causes, outlooks, comparisons, or implications unless the chunk states them explicitly.
- If something is unclear or ambiguous, omit it — never guess.
- Treat this chunk as an independent unit. Do NOT merge it with sibling sections and do NOT pull context from anywhere else.
- If a Markdown table is present, identify the 3–5 most information-dense cells and surface them as bullets, quoting specific values verbatim (e.g. "Treatment A response rate: 67.3%"). Never emit a bullet like "the table shows revenue growth" — name the rows and the numbers.

SELF-CHECK (run silently before responding, then output only the bullets):
1. Are there exactly 3 to 5 bullets, each starting with "- "?
2. Does every number, date, and proper name in the output appear verbatim in the input?
3. Did I echo a section name, heading, or any part of the Hierarchy line?
4. Did I add a claim, cause, outlook, or comparison not present in the input?
5. Did I include a preamble, closing remark, or any non-bullet text?
If any check fails, rewrite before responding. Output only the bullets — nothing else."""


# ── Few-shot examples (constant — cacheable) ──────────────────────────────

_EXAMPLE_1_HUMAN = """Hierarchy: Financial Results > Revenue > US

Revenue in the US segment grew 12% year-over-year to $4.2B in Q2 2026, driven by strong enterprise demand. Operating margin improved to 18%, up from 15% in the prior-year quarter. The company opened 3 new fulfillment centers in Texas and Ohio during the quarter. Management highlighted that software subscription renewal rate reached 94%."""

_EXAMPLE_1_ASSISTANT = """- US segment revenue grew 12% year-over-year to $4.2B in Q2 2026
- Growth driven by strong enterprise demand
- Operating margin improved to 18%, up from 15% in the prior-year quarter
- 3 new fulfillment centers opened in Texas and Ohio during the quarter
- Software subscription renewal rate reached 94%"""


_EXAMPLE_2_HUMAN = """Hierarchy: Clinical Results > Phase III Trial > Primary Endpoints

Primary efficacy endpoints for the Phase III trial (n=1,248) are summarized below.

| Arm         | Patients | Response Rate | Median PFS (months) |
|-------------|----------|---------------|---------------------|
| Treatment A | 624      | 67.3%         | 14.2                |
| Control     | 624      | 41.8%         | 8.6                 |

Treatment A achieved statistical significance on the primary endpoint (p < 0.001). Grade 3+ adverse events were reported in 14.2% of Treatment A patients."""

_EXAMPLE_2_ASSISTANT = """- Phase III trial enrolled 1,248 patients split evenly between arms (624 each)
- Treatment A response rate: 67.3% vs Control: 41.8%
- Treatment A median PFS: 14.2 months vs Control: 8.6 months
- Treatment A reached statistical significance on the primary endpoint (p < 0.001)
- Grade 3+ adverse events reported in 14.2% of Treatment A patients"""


FEW_SHOT_MESSAGES: list[BaseMessage] = [
    HumanMessage(content=_EXAMPLE_1_HUMAN),
    AIMessage(content=_EXAMPLE_1_ASSISTANT),
    HumanMessage(content=_EXAMPLE_2_HUMAN),
    AIMessage(content=_EXAMPLE_2_ASSISTANT),
]


# ── Per-chunk human-message builder ───────────────────────────────────────

def build_human_message(path: Iterable[str], content: str) -> HumanMessage:
    """Format a chunk as ``Hierarchy: A > B > C\\n\\n<content>``.

    ``path`` is the full list of ancestor heading titles (outermost first).
    Matching the few-shot shape keeps the model locked to the output contract.
    """
    path_list = [p for p in (path or []) if p]
    hierarchy = " > ".join(path_list) if path_list else "(root)"
    return HumanMessage(content=f"Hierarchy: {hierarchy}\n\n{content}")


# ── Registry-backed prefix loader ─────────────────────────────────────────

# Env var: when set (e.g. "prompts:/catalog.schema.pitcrew_summarize@production"),
# the constant prefix is pulled from the MLflow Prompt Registry instead of the
# in-file defaults. Populated once at import time so the cacheable prefix
# stays byte-identical across every LLM call.
_SUMMARIZE_PROMPT_URI = os.environ.get("SUMMARIZE_PROMPT_URI", "").strip()


def _load_prefix_from_registry(uri: str) -> tuple[str, list[BaseMessage]]:
    """Return (system_prompt, few_shot_messages) loaded from MLflow.

    The registered template is the same shape produced by
    ``register_prompts.py``: a list of ``{"role": ..., "content": ...}`` dicts
    starting with ``system`` followed by alternating ``user``/``assistant``
    few-shot pairs.
    """
    from mlflow.genai import load_prompt  # local import — optional dep at runtime

    prompt = load_prompt(uri)
    template = getattr(prompt, "template", None)
    if not isinstance(template, list):
        raise ValueError(
            f"Prompt at {uri!r} is not a chat template (got {type(template).__name__})."
        )

    system_text = ""
    few_shot: list[BaseMessage] = []
    for msg in template:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system" and not system_text:
            system_text = content
        elif role == "user":
            few_shot.append(HumanMessage(content=content))
        elif role == "assistant":
            few_shot.append(AIMessage(content=content))
        else:
            logger.warning("Unexpected role %r in registered prompt %s", role, uri)
    if not system_text:
        raise ValueError(f"No system message in registered prompt at {uri!r}.")
    return system_text, few_shot


if _SUMMARIZE_PROMPT_URI:
    try:
        SUMMARIZE_SYSTEM_PROMPT, FEW_SHOT_MESSAGES = _load_prefix_from_registry(
            _SUMMARIZE_PROMPT_URI
        )
        logger.info("Loaded summarizer prompt from registry: %s", _SUMMARIZE_PROMPT_URI)
    except Exception as exc:
        logger.exception(
            "Failed to load summarizer prompt from registry (%s) — falling back to in-file constants: %s",
            _SUMMARIZE_PROMPT_URI,
            exc,
        )


# ── Content-block normalization helper ────────────────────────────────────

def as_text(content: object) -> str:
    """ChatDatabricks may return ``content`` as str or list[content-block]. Normalize."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)
