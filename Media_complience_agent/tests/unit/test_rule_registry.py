"""Unit tests for deterministic pieces of core.rule_registry."""

from __future__ import annotations

import pytest

from core.rule_registry import build_injection_string, get_current_registry


def test_get_current_registry_raises_before_load():
    import core.rule_registry as rr

    original = rr._registry
    rr._registry = None
    try:
        with pytest.raises(RuntimeError, match="not loaded"):
            get_current_registry()
    finally:
        rr._registry = original


def test_build_injection_string_format(sample_rule_entry):
    injection = build_injection_string([sample_rule_entry])
    assert "[RULE: FINRA 2210(d)(1)(B)]" in injection
    assert "[CITE:" in injection
    assert "[SEVERITY: critical]" in injection
    assert sample_rule_entry.rule_text in injection


def test_build_injection_string_groups_by_jurisdiction(sample_rule_entry):
    injection = build_injection_string([sample_rule_entry])
    assert "FINRA" in injection
    # Fix-2 header includes base_rule_number and doc_name
    assert "Rule 2210" in injection


def test_rule_registry_injection_string(sample_registry):
    assert len(sample_registry.injection_string) > 50
