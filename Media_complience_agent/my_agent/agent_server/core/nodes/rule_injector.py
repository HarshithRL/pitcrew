"""rule_injector — copy registry.injection_string into state.

Deterministic, zero LLM. The injection string is pre-built at rule-load
time by `core.rule_registry.build_injection_string`, so this node is
a thin wiring shim.
"""

from __future__ import annotations

from core.types import ComplianceState


async def run(state: ComplianceState) -> dict:
    registry = state["rule_registry"]
    return {"rule_injection": registry.injection_string}
