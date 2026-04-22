"""Pydantic v2 models + LangGraph state for the compliance agent.

Two kinds of types live here:
- Pydantic `BaseModel`s — internal/API data contracts (RuleEntry, Violation, ...).
- `ComplianceState` — a LangGraph `TypedDict` threaded through every node.

Single source of truth for shapes. Nodes, routes, and the ResponsesAgent all
import from this file.
"""

from __future__ import annotations

from datetime import datetime
from operator import add
from typing import Annotated, Optional, TypedDict

from pydantic import BaseModel


# ── Rule models ─────────────────────────────────────────────────────────────

class RuleEntry(BaseModel):
    """One discrete rule extracted by the LLM rule-parser."""

    rule_id: str                # e.g. "FINRA 2210(d)(1)(B)"
    section: str                # e.g. "(d)(1)(B)"
    rule_text: str              # verbatim — never paraphrased
    citation_label: str         # e.g. "FINRA Rule 2210, Section (d)(1)(B) — Content Standards"
    jurisdiction: str           # "FINRA" | "SEC"
    severity: str               # "critical" | "major" | "minor"
    is_prohibition: bool        # true if "No member may" / "shall not"
    is_requirement: bool        # true if "must" / "shall be required"
    base_rule_number: str = ""  # e.g. "2210" or "275.206(4)-2" — from registry.yaml
    doc_name: str = ""          # e.g. "FINRA Rule 2210 - Communications with the Public"


class RuleRegistry(BaseModel):
    """All rules currently loaded plus the pre-built injection string.

    The injection_string is built once after parsing and reused in every
    compliance_checker call — no per-request string assembly.

    Per-rule format:
        [RULE: FINRA 2210(d)(1)(B)] [CITE: FINRA Rule 2210 §(d)(1)(B)]
        [SEVERITY: critical]
        <verbatim rule_text>
    """

    rules: list[RuleEntry]
    loaded_at: datetime
    source_documents: list[str]
    injection_string: str


# ── Post + violation models ─────────────────────────────────────────────────

class PostContent(BaseModel):
    """Extracted post text plus submission metadata."""

    submission_id: str
    raw_text: str
    platform: Optional[str] = None
    file_name: str
    file_type: str              # "pdf" | "docx"
    char_count: int


class Violation(BaseModel):
    """A single rule failure found by compliance_checker (and verified by judge)."""

    rule_id: str
    citation_label: str
    rule_text: str              # verbatim — the citation shown to the user
    violated_text: str          # exact excerpt from the post
    explanation: str            # plain English
    severity: str
    confidence: float


# ── Guidance models ─────────────────────────────────────────────────────────

class GuidanceItem(BaseModel):
    item: str                   # specific phrase / claim to remove
    reason: str                 # one sentence plain English
    rule_citation: str          # exact citation_label


class ComplianceGuidance(BaseModel):
    what_to_post: list[str]                 # elements that ARE compliant
    what_to_avoid: list[GuidanceItem]       # each with its rule citation
    compliant_rewrite: str                  # full rewritten post
    summary: str                            # one paragraph overall


# ── Final report ────────────────────────────────────────────────────────────

class ComplianceReport(BaseModel):
    """Assembled output returned to the caller."""

    submission_id: str
    thread_id: str
    post_content: PostContent
    verdict: str                            # "compliant" | "non_compliant" | "needs_review"
    risk_score: float
    risk_tier: str                          # "auto_approve" | "hitl" | "auto_reject"
    violations: list[Violation]
    guidance: ComplianceGuidance
    rule_citations: list[str]               # all citation_labels cited
    human_decision: Optional[str] = None    # "approved" | "rejected" | "edit_requested"
    human_notes: Optional[str] = None
    processed_at: datetime


# ── HITL payload ────────────────────────────────────────────────────────────

class HITLPayload(BaseModel):
    """Payload handed to the reviewer when the graph pauses via interrupt()."""

    submission_id: str
    thread_id: str
    post_text: str
    platform: Optional[str] = None
    risk_score: float
    violations: list[Violation]
    awaiting_since: str                     # ISO datetime string


# ── LangGraph state ─────────────────────────────────────────────────────────

def _merge_dicts(a: dict, b: dict) -> dict:
    """Reducer for `node_metrics`: shallow merge, later keys win."""
    return {**a, **b}


class ComplianceState(TypedDict, total=False):
    """State threaded through every node of the compliance graph.

    `total=False` — most keys are populated incrementally by downstream nodes.

    Reducers:
    - `violations` uses `operator.add` so parallel / repeated node emissions
      concatenate. The compliance_checker appends raw violations here.
    - `verified_violations` uses no reducer (plain assignment). The judge
      reads `violations`, removes hallucinated citations, and writes the
      cleaned list here. All downstream nodes (guidance_generator,
      output_formatter) read from `verified_violations` only — never from
      `violations` directly.
    - `judge_results` uses `operator.add` to accumulate per-violation verdicts.
    - `node_metrics` uses `_merge_dicts` so each node records timings/counts
      without clobbering siblings.
    """

    submission_id: str
    post_content: PostContent
    rule_registry: RuleRegistry
    rule_injection: str
    violations: Annotated[list[Violation], add]
    verified_violations: list[Violation]
    risk_score: float
    risk_tier: str
    judge_results: Annotated[list[dict], add]
    human_decision: Optional[str]
    human_notes: Optional[str]
    guidance: Optional[ComplianceGuidance]
    compliance_report: Optional[ComplianceReport]
    retry_count: int
    node_metrics: Annotated[dict, _merge_dicts]
