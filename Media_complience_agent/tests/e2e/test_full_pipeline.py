"""End-to-end test with mocked LLM: DOCX → extract → graph → ComplianceReport.

No DATABRICKS credentials required. The three LLM-calling nodes
(compliance_checker, judge, guidance_generator) are patched.

Three critical violations are seeded so the risk_aggregator lands in
auto_reject tier (score 1.0 ≥ 0.75). Two-critical mocks would score 0.67
and hit the HITL branch, which pauses the graph via interrupt() and never
produces a compliance_report — breaking the end-to-end assertions.
"""

from __future__ import annotations

import asyncio
import io
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import docx as python_docx
import pytest

from core.graph import get_compiled_graph
from core.post_extractor import extract_post
from core.types import RuleEntry, RuleRegistry


VIOLATION_POST = (
    "Our fund guarantees 15% annual returns every year. "
    "Past performance shows we have never lost money. "
    "Invest now before this opportunity closes."
)

MOCK_VIOLATIONS_E2E = [
    {
        "rule_id": "FINRA 2210(d)(1)(B)",
        "citation_label": "FINRA Rule 2210, Section (d)(1)(B)",
        "rule_text": "No member may make any false statement.",
        "violated_text": "guarantees 15% annual returns",
        "explanation": "Performance guarantees are prohibited.",
        "severity": "critical",
        "confidence": 0.97,
    },
    {
        "rule_id": "FINRA 2210(d)(1)(F)",
        "citation_label": "FINRA Rule 2210, Section (d)(1)(F)",
        "rule_text": "Communications may not predict performance.",
        "violated_text": "Past performance shows we have never lost",
        "explanation": "Past performance projections are prohibited.",
        "severity": "critical",
        "confidence": 0.95,
    },
    {
        "rule_id": "FINRA 2210(d)(1)(A)",
        "citation_label": "FINRA Rule 2210, Section (d)(1)(A)",
        "rule_text": "Communications must be fair and balanced.",
        "violated_text": "Invest now before this opportunity closes.",
        "explanation": "Creating false urgency is not fair and balanced.",
        "severity": "critical",
        "confidence": 0.92,
    },
]

MOCK_JUDGE_E2E = {
    "verified_violations": [
        {
            "rule_id": "FINRA 2210(d)(1)(B)",
            "citation_accuracy": 0.97,
            "explanation_clarity": 0.95,
            "overall_valid": True,
        },
        {
            "rule_id": "FINRA 2210(d)(1)(F)",
            "citation_accuracy": 0.95,
            "explanation_clarity": 0.93,
            "overall_valid": True,
        },
        {
            "rule_id": "FINRA 2210(d)(1)(A)",
            "citation_accuracy": 0.92,
            "explanation_clarity": 0.90,
            "overall_valid": True,
        },
    ],
    "removed_rule_ids": [],
}

MOCK_GUIDANCE_E2E = {
    "what_to_post": ["Mention your investment approach"],
    "what_to_avoid": [
        {
            "item": "guarantees 15% annual returns",
            "reason": "Performance guarantees prohibited.",
            "rule_citation": "FINRA Rule 2210, Section (d)(1)(B)",
        },
        {
            "item": "never lost money",
            "reason": "Performance projection prohibited.",
            "rule_citation": "FINRA Rule 2210, Section (d)(1)(F)",
        },
        {
            "item": "Invest now before this opportunity closes.",
            "reason": "Creates false urgency.",
            "rule_citation": "FINRA Rule 2210, Section (d)(1)(A)",
        },
    ],
    "compliant_rewrite": (
        "Our fund employs a disciplined strategy focused on risk management. "
        "Investment involves risk and past performance does not indicate "
        "future results."
    ),
    "summary": "Three critical violations removed in rewrite.",
}


def make_docx(text: str) -> bytes:
    doc = python_docx.Document()
    doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def mock_llm_resp(content: str):
    m = MagicMock()
    m.content = content
    return m


@pytest.fixture(autouse=True)
def reset_graph():
    import core.graph as g
    g._graph = None
    yield
    g._graph = None


def test_post_extraction_to_graph_mocked():
    entries = [
        RuleEntry(
            rule_id="FINRA 2210(d)(1)(B)",
            section="(d)(1)(B)",
            rule_text="No member may make any false statement.",
            citation_label="FINRA Rule 2210, Section (d)(1)(B)",
            jurisdiction="FINRA",
            severity="critical",
            is_prohibition=True,
            is_requirement=False,
            base_rule_number="2210",
            doc_name="FINRA Rule 2210",
        ),
        RuleEntry(
            rule_id="FINRA 2210(d)(1)(F)",
            section="(d)(1)(F)",
            rule_text="Communications may not predict performance.",
            citation_label="FINRA Rule 2210, Section (d)(1)(F)",
            jurisdiction="FINRA",
            severity="critical",
            is_prohibition=True,
            is_requirement=False,
            base_rule_number="2210",
            doc_name="FINRA Rule 2210",
        ),
        RuleEntry(
            rule_id="FINRA 2210(d)(1)(A)",
            section="(d)(1)(A)",
            rule_text="Communications must be fair and balanced.",
            citation_label="FINRA Rule 2210, Section (d)(1)(A)",
            jurisdiction="FINRA",
            severity="critical",
            is_prohibition=True,
            is_requirement=False,
            base_rule_number="2210",
            doc_name="FINRA Rule 2210",
        ),
    ]
    registry = RuleRegistry(
        rules=entries,
        loaded_at=datetime.now(timezone.utc),
        source_documents=["finra_2210.pdf"],
        injection_string="[RULE: FINRA 2210(d)(1)(B)] ...",
    )

    docx_bytes = make_docx(VIOLATION_POST)
    post_content = extract_post(
        docx_bytes, "violation_post.docx", "e2e-001", "linkedin"
    )
    assert post_content.file_type == "docx"
    assert "guarantees" in post_content.raw_text

    state = {
        "submission_id": "e2e-001",
        "post_content": post_content,
        "rule_registry": registry,
        "rule_injection": "",
        "violations": [],
        "verified_violations": [],
        "risk_score": 0.0,
        "risk_tier": "",
        "judge_results": [],
        "human_decision": None,
        "human_notes": None,
        "guidance": None,
        "compliance_report": None,
        "retry_count": 0,
        "node_metrics": {},
    }

    # All three LLM-calling nodes use `await llm.ainvoke(...)`, so AsyncMock.
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=[
        mock_llm_resp(json.dumps(MOCK_VIOLATIONS_E2E)),
        mock_llm_resp(json.dumps(MOCK_JUDGE_E2E)),
        mock_llm_resp(json.dumps(MOCK_GUIDANCE_E2E)),
    ])

    with patch("core.nodes.compliance_checker.get_llm", return_value=mock_llm), \
         patch("core.nodes.judge.get_llm", return_value=mock_llm), \
         patch("core.nodes.guidance_generator.get_llm", return_value=mock_llm):

        graph = get_compiled_graph()
        config = {"configurable": {"thread_id": "e2e-001"}}
        result = asyncio.run(graph.ainvoke(state, config=config))

    report = result.get("compliance_report")

    assert report is not None, "No compliance_report in result"
    assert report.verdict == "non_compliant"

    # All three citations survived the judge pass
    assert len(report.violations) == 3
    for v in report.violations:
        assert v.rule_id != ""
        assert v.citation_label != ""
        assert v.rule_text != ""
        assert v.violated_text != ""
        assert v.explanation != ""

    # Guidance present
    assert report.guidance is not None
    assert len(report.guidance.compliant_rewrite) > 10
    assert len(report.guidance.what_to_avoid) == 3

    for item in report.guidance.what_to_avoid:
        assert "FINRA Rule 2210" in item.rule_citation

    # The rewrite should not contain the violated phrases
    rewrite = report.guidance.compliant_rewrite.lower()
    assert "guarantees 15%" not in rewrite
    assert "never lost money" not in rewrite
