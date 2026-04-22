"""Shared fixtures for the compliance agent test suite."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import pytest

# Make agent_server importable
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "my_agent", "agent_server"
    ),
)

from core.types import (  # noqa: E402
    ComplianceGuidance,
    GuidanceItem,
    PostContent,
    RuleEntry,
    RuleRegistry,
    Violation,
)


@pytest.fixture
def sample_rule_entry():
    return RuleEntry(
        rule_id="FINRA 2210(d)(1)(B)",
        section="(d)(1)(B)",
        rule_text=(
            "No member may make any false, exaggerated, unwarranted, "
            "promissory or misleading statement."
        ),
        citation_label=(
            "FINRA Rule 2210, Section (d)(1)(B) — "
            "Prohibition on False or Misleading Statements"
        ),
        jurisdiction="FINRA",
        severity="critical",
        is_prohibition=True,
        is_requirement=False,
        base_rule_number="2210",
        doc_name="FINRA Rule 2210 - Communications with the Public",
    )


@pytest.fixture
def sample_registry(sample_rule_entry):
    injection = (
        "[RULE: FINRA 2210(d)(1)(B)] "
        "[CITE: FINRA Rule 2210, Section (d)(1)(B)]\n"
        "[SEVERITY: critical]\n"
        "No member may make any false statement."
    )
    return RuleRegistry(
        rules=[sample_rule_entry],
        loaded_at=datetime.now(timezone.utc),
        source_documents=["finra_2210.pdf"],
        injection_string=injection,
    )


@pytest.fixture
def sample_post():
    return PostContent(
        submission_id="test-001",
        raw_text="Our fund guarantees 15% annual returns. Invest now!",
        platform="linkedin",
        file_name="test.pdf",
        file_type="pdf",
        char_count=52,
    )


@pytest.fixture
def sample_violation(sample_rule_entry):
    return Violation(
        rule_id="FINRA 2210(d)(1)(B)",
        citation_label="FINRA Rule 2210, Section (d)(1)(B)",
        rule_text=sample_rule_entry.rule_text,
        violated_text="guarantees 15% annual returns",
        explanation=(
            "Performance guarantees are prohibited under FINRA regulations."
        ),
        severity="critical",
        confidence=0.97,
    )


@pytest.fixture
def sample_guidance(sample_violation):
    return ComplianceGuidance(
        what_to_post=["Educational investment content"],
        what_to_avoid=[
            GuidanceItem(
                item="guarantees 15% annual returns",
                reason="Performance guarantees are prohibited.",
                rule_citation="FINRA Rule 2210, Section (d)(1)(B)",
            )
        ],
        compliant_rewrite=(
            "Our fund focuses on disciplined investment strategies. "
            "Past performance does not guarantee future results."
        ),
        summary=(
            "The post contained a critical violation. The rewrite "
            "removes the performance guarantee."
        ),
    )


@pytest.fixture
def base_state(sample_post, sample_registry):
    return {
        "submission_id": "test-001",
        "post_content": sample_post,
        "rule_registry": sample_registry,
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
