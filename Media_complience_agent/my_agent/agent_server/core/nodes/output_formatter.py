"""output_formatter — deterministic `ComplianceReport` assembly.

Terminal node. Reads `verified_violations`, `guidance`, `risk_score`,
`risk_tier`, and optional `human_decision`/`human_notes` from state,
produces the final report object.

Verdict ladder:
  tier == auto_approve OR no verified violations → "compliant"
  tier == auto_reject                            → "non_compliant"
  human_decision == "approved"                   → "compliant"
  human_decision == "rejected"                   → "non_compliant"
  otherwise                                      → "needs_review"
"""

from __future__ import annotations

from datetime import datetime, timezone

from core.types import ComplianceGuidance, ComplianceReport, ComplianceState


def _verdict(
    tier: str,
    verified_count: int,
    human_decision: str | None,
) -> str:
    if tier == "auto_approve" or verified_count == 0:
        return "compliant"
    if tier == "auto_reject":
        return "non_compliant"
    if human_decision == "approved":
        return "compliant"
    if human_decision == "rejected":
        return "non_compliant"
    return "needs_review"


async def run(state: ComplianceState) -> dict:
    verified = state.get("verified_violations") or []
    guidance = state.get("guidance")
    score = float(state.get("risk_score", 0.0))
    tier = str(state.get("risk_tier", "auto_approve"))
    human_decision = state.get("human_decision")

    verdict = _verdict(tier, len(verified), human_decision)
    rule_citations = list(dict.fromkeys(v.citation_label for v in verified))

    report = ComplianceReport(
        submission_id=state["submission_id"],
        thread_id=state["submission_id"],
        post_content=state["post_content"],
        verdict=verdict,
        risk_score=score,
        risk_tier=tier,
        violations=verified,
        guidance=guidance
        or ComplianceGuidance(
            what_to_post=[],
            what_to_avoid=[],
            compliant_rewrite="",
            summary="",
        ),
        rule_citations=rule_citations,
        human_decision=human_decision,
        human_notes=state.get("human_notes"),
        processed_at=datetime.now(timezone.utc),
    )
    return {"compliance_report": report}
