"""risk_aggregator — deterministic severity-weighted scorer + HITL router.

Score = sum(severity_weight) / severity_normalizer, clipped to 1.0.

Tiers (thresholds from compliance.yaml → risk):
  score <  auto_approve_threshold    → "auto_approve"
  score <= hitl_threshold            → "hitl"
  score >  hitl_threshold            → "auto_reject"

`route()` is the conditional-edge function. Not wired into the linear
graph in Step 8 — it's here so Step 9 can drop it in without changing
this file.
"""

from __future__ import annotations

from core.nodes._shared import get_compliance_cfg, get_severity_cfg
from core.types import ComplianceState


async def run(state: ComplianceState) -> dict:
    violations = state.get("violations") or []
    if not violations:
        return {"risk_score": 0.0, "risk_tier": "auto_approve"}

    severity_cfg = get_severity_cfg()
    weights = severity_cfg.get("severity_weights") or {}
    default_w = float(weights.get("default", 0.15))

    compliance_cfg = get_compliance_cfg()
    risk_cfg = compliance_cfg.get("risk") or {}
    normalizer = float(risk_cfg.get("severity_normalizer", 3.0))
    auto_approve = float(risk_cfg.get("auto_approve_threshold", 0.40))
    hitl_thresh = float(risk_cfg.get("hitl_threshold", 0.75))

    raw_score = sum(float(weights.get(v.severity, default_w)) for v in violations)
    score = min(raw_score / normalizer, 1.0)

    if score < auto_approve:
        tier = "auto_approve"
    elif score <= hitl_thresh:
        tier = "hitl"
    else:
        tier = "auto_reject"

    return {"risk_score": round(score, 4), "risk_tier": tier}


def route(state: ComplianceState) -> str:
    """Conditional edge: route HITL-tier posts to the reviewer, else straight to judge."""
    return "hitl" if state.get("risk_tier") == "hitl" else "judge"
