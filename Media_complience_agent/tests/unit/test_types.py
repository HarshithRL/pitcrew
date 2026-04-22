"""Smoke tests for core.types Pydantic models."""

from __future__ import annotations

from datetime import datetime, timezone

from core.types import ComplianceReport, HITLPayload


def test_rule_entry_instantiates(sample_rule_entry):
    assert sample_rule_entry.rule_id == "FINRA 2210(d)(1)(B)"
    assert sample_rule_entry.severity == "critical"
    assert sample_rule_entry.is_prohibition is True
    assert sample_rule_entry.base_rule_number == "2210"


def test_violation_has_citation_fields(sample_violation):
    assert sample_violation.rule_id == "FINRA 2210(d)(1)(B)"
    assert "FINRA Rule 2210" in sample_violation.citation_label
    assert len(sample_violation.rule_text) > 10
    assert sample_violation.violated_text != ""
    assert 0.0 <= sample_violation.confidence <= 1.0


def test_compliance_report_instantiates(
    sample_post, sample_violation, sample_guidance
):
    report = ComplianceReport(
        submission_id="test-001",
        thread_id="test-001",
        post_content=sample_post,
        verdict="non_compliant",
        risk_score=1.0,
        risk_tier="auto_reject",
        violations=[sample_violation],
        guidance=sample_guidance,
        rule_citations=["FINRA Rule 2210, Section (d)(1)(B)"],
        processed_at=datetime.now(timezone.utc),
    )
    assert report.verdict == "non_compliant"
    assert len(report.violations) == 1
    assert report.violations[0].citation_label != ""


def test_hitl_payload_instantiates(sample_violation, sample_post):
    payload = HITLPayload(
        submission_id="test-001",
        thread_id="test-001",
        post_text=sample_post.raw_text,
        platform="linkedin",
        risk_score=0.60,
        violations=[sample_violation],
        awaiting_since=datetime.now(timezone.utc).isoformat(),
    )
    assert payload.risk_score == 0.60
    assert len(payload.violations) == 1
