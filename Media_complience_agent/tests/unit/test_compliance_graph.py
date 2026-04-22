"""Unit tests for the compiled compliance graph and its per-node shims.

Uses AsyncMock (not MagicMock) for the LLM because all nodes call
`await llm.ainvoke(...)` — a plain MagicMock returns a non-awaitable and
`await` would raise TypeError.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.graph import get_compiled_graph


MOCK_VIOLATIONS = [
    {
        "rule_id": "FINRA 2210(d)(1)(B)",
        "citation_label": "FINRA Rule 2210, Section (d)(1)(B)",
        "rule_text": "No member may make any false statement.",
        "violated_text": "guarantees 15% annual returns",
        "explanation": "Performance guarantees are prohibited.",
        "severity": "critical",
        "confidence": 0.97,
    }
]

MOCK_JUDGE_VERDICT = {
    "verified_violations": [
        {
            "rule_id": "FINRA 2210(d)(1)(B)",
            "citation_accuracy": 0.97,
            "explanation_clarity": 0.95,
            "overall_valid": True,
        }
    ],
    "removed_rule_ids": [],
}

MOCK_GUIDANCE = {
    "what_to_post": ["Educational investment content"],
    "what_to_avoid": [
        {
            "item": "guarantees 15% annual returns",
            "reason": "Performance guarantees are prohibited.",
            "rule_citation": "FINRA Rule 2210, Section (d)(1)(B)",
        }
    ],
    "compliant_rewrite": (
        "Our fund focuses on disciplined investment strategies. "
        "Past performance does not guarantee future results."
    ),
    "summary": "One critical violation found and fixed.",
}


def mock_llm_response(content: str):
    resp = MagicMock()
    resp.content = content
    return resp


@pytest.fixture(autouse=True)
def reset_graph():
    """Clear the module-level cached graph between tests."""
    import core.graph as g
    g._graph = None
    yield
    g._graph = None


def test_graph_has_required_nodes():
    graph = get_compiled_graph()
    nodes = list(graph.nodes.keys())
    required = [
        "rule_injector",
        "compliance_checker",
        "risk_aggregator",
        "hitl_node",
        "judge",
        "guidance_generator",
        "output_formatter",
    ]
    for n in required:
        assert n in nodes, f"Missing node: {n}"


def test_rule_injector_populates_state(base_state):
    from core.nodes.rule_injector import run
    result = asyncio.run(run(base_state))
    assert "rule_injection" in result
    assert len(result["rule_injection"]) > 0


def test_output_formatter_compliant_verdict(base_state, sample_guidance):
    """Zero verified violations → compliant verdict."""
    from core.nodes.output_formatter import run
    base_state["verified_violations"] = []
    base_state["risk_tier"] = "auto_approve"
    base_state["risk_score"] = 0.0
    base_state["guidance"] = sample_guidance

    result = asyncio.run(run(base_state))
    report = result["compliance_report"]
    assert report.verdict == "compliant"
    assert report.risk_score == 0.0


def test_output_formatter_non_compliant_verdict(
    base_state, sample_violation, sample_guidance
):
    """auto_reject tier → non_compliant verdict, citations preserved."""
    from core.nodes.output_formatter import run
    base_state["verified_violations"] = [sample_violation]
    base_state["risk_tier"] = "auto_reject"
    base_state["risk_score"] = 1.0
    base_state["guidance"] = sample_guidance

    result = asyncio.run(run(base_state))
    report = result["compliance_report"]
    assert report.verdict == "non_compliant"
    assert len(report.violations) == 1
    assert report.violations[0].citation_label != ""
    assert report.violations[0].rule_text != ""


def test_full_graph_mocked_llm(base_state):
    """End-to-end graph run with LLM calls mocked at every node.

    Exercises: checker → risk_aggregator → (router) → judge →
    guidance_generator → output_formatter.

    The 1 critical violation scores 0.33 (below 0.40 auto_approve threshold),
    so the router stays on the judge path — no HITL interrupt.
    """
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(
        side_effect=[
            mock_llm_response(json.dumps(MOCK_VIOLATIONS)),
            mock_llm_response(json.dumps(MOCK_JUDGE_VERDICT)),
            mock_llm_response(json.dumps(MOCK_GUIDANCE)),
        ]
    )

    with patch("core.nodes.compliance_checker.get_llm", return_value=mock_llm), \
         patch("core.nodes.judge.get_llm", return_value=mock_llm), \
         patch("core.nodes.guidance_generator.get_llm", return_value=mock_llm):

        graph = get_compiled_graph()
        config = {"configurable": {"thread_id": "mock-test-001"}}
        result = asyncio.run(graph.ainvoke(base_state, config=config))

    report = result.get("compliance_report")
    assert report is not None
    assert report.verdict in ("compliant", "non_compliant", "needs_review")
