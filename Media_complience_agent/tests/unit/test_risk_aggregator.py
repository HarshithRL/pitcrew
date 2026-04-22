"""Unit tests for the deterministic risk scoring + routing node."""

from __future__ import annotations

import asyncio

from core.nodes.risk_aggregator import route, run
from core.types import Violation


def make_violation(severity: str) -> Violation:
    return Violation(
        rule_id="FINRA 2210(d)(1)(B)",
        citation_label="FINRA Rule 2210, Section (d)(1)(B)",
        rule_text="No member may make any false statement.",
        violated_text="guarantees",
        explanation="Performance guarantee is prohibited.",
        severity=severity,
        confidence=0.95,
    )


def test_no_violations_auto_approve(base_state):
    base_state["violations"] = []
    result = asyncio.run(run(base_state))
    assert result["risk_score"] == 0.0
    assert result["risk_tier"] == "auto_approve"


def test_one_critical_auto_reject(base_state):
    """One critical: 1.0 / 3.0 normalizer = 0.33 → below auto_approve threshold (0.40)."""
    base_state["violations"] = [make_violation("critical")]
    result = asyncio.run(run(base_state))
    assert result["risk_tier"] == "auto_approve"


def test_two_criticals_auto_reject(base_state):
    """Two criticals: 2.0 / 3.0 = 0.67 → hitl band."""
    base_state["violations"] = [
        make_violation("critical"),
        make_violation("critical"),
    ]
    result = asyncio.run(run(base_state))
    assert result["risk_tier"] == "hitl"


def test_three_criticals_auto_reject(base_state):
    """Three criticals: 3.0 / 3.0 = 1.0 → auto_reject."""
    base_state["violations"] = [
        make_violation("critical"),
        make_violation("critical"),
        make_violation("critical"),
    ]
    result = asyncio.run(run(base_state))
    assert result["risk_tier"] == "auto_reject"
    assert result["risk_score"] == 1.0


def test_route_hitl(base_state):
    base_state["risk_tier"] = "hitl"
    assert route(base_state) == "hitl"


def test_route_auto(base_state):
    base_state["risk_tier"] = "auto_approve"
    assert route(base_state) == "judge"

    base_state["risk_tier"] = "auto_reject"
    assert route(base_state) == "judge"
