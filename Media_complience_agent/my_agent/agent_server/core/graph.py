"""Compliance graph with HITL routing.

    rule_injector → compliance_checker → risk_aggregator
                                                │
                                       ┌────────┴────────┐
                                       │ route()         │
                                       ▼                 ▼
                                   hitl_node            judge
                                       │                 │
                                       └────────┬────────┘
                                                ▼
                                      guidance_generator → output_formatter

`risk_aggregator.route()` returns "hitl" when tier=="hitl", else "judge".
The hitl_node calls `interrupt()` and pauses the graph until the reviewer
posts a resume command (handled by the /hitl/decision route in Step 11).

Compiled graph is memoised for cheap reuse when called without a
checkpointer. When a caller (the agent) owns a checkpointer, we build
fresh and return uncached — each owner keeps its own instance.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from core.nodes import (
    compliance_checker,
    guidance_generator,
    hitl_node,
    judge,
    output_formatter,
    risk_aggregator,
    rule_injector,
)
from core.nodes.risk_aggregator import route
from core.types import ComplianceState

_graph = None


def _build(checkpointer):
    builder = StateGraph(ComplianceState)

    builder.add_node("rule_injector", rule_injector.run)
    builder.add_node("compliance_checker", compliance_checker.run)
    builder.add_node("risk_aggregator", risk_aggregator.run)
    builder.add_node("hitl_node", hitl_node.run)
    builder.add_node("judge", judge.run)
    builder.add_node("guidance_generator", guidance_generator.run)
    builder.add_node("output_formatter", output_formatter.run)

    builder.add_edge(START, "rule_injector")
    builder.add_edge("rule_injector", "compliance_checker")
    builder.add_edge("compliance_checker", "risk_aggregator")

    builder.add_conditional_edges(
        "risk_aggregator",
        route,
        {"hitl": "hitl_node", "judge": "judge"},
    )

    builder.add_edge("hitl_node", "judge")
    builder.add_edge("judge", "guidance_generator")
    builder.add_edge("guidance_generator", "output_formatter")
    builder.add_edge("output_formatter", END)

    return builder.compile(checkpointer=checkpointer)


def get_compiled_graph(checkpointer=None):
    """Return a compiled compliance graph.

    - With a checkpointer argument: build fresh (uncached). The caller owns
      the checkpointer — important for HITL resume, where the same saver
      must see the paused state.
    - Without: use a shared cached instance with a default MemorySaver.
    """
    global _graph
    if checkpointer is not None:
        return _build(checkpointer)
    if _graph is None:
        _graph = _build(MemorySaver())
    return _graph
