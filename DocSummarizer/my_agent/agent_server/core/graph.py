"""Structure-aware document summarizer graph.

Linear pipeline with a map step in the middle:

    ingest
      -> parse_structure         (deterministic — extracts section hierarchy)
      -> chunk                   (deterministic — section-aware, no overlap)
      -> summarize_chunk         (LLM, fan-out via Send — ONLY LLM call)
      -> aggregate               (deterministic — rebuilds Markdown hierarchy)
      -> output

No supervisor, no routing, no global LLM merge. Structure is owned by Python,
not the model.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from core.nodes import (
    aggregate_node,
    chunk_node,
    fan_out_chunks,
    ingest_node,
    output_node,
    parse_structure_node,
    summarize_chunk_node,
)
from core.state import DocState


def build_graph(checkpointer=None):
    g = StateGraph(DocState)

    g.add_node("ingest", ingest_node)
    g.add_node("parse_structure", parse_structure_node)
    g.add_node("chunk", chunk_node)
    g.add_node("summarize_chunk", summarize_chunk_node)
    g.add_node("aggregate", aggregate_node)
    g.add_node("output", output_node)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "parse_structure")
    g.add_edge("parse_structure", "chunk")
    g.add_conditional_edges("chunk", fan_out_chunks, ["summarize_chunk"])
    g.add_edge("summarize_chunk", "aggregate")
    g.add_edge("aggregate", "output")
    g.add_edge("output", END)

    return g.compile(checkpointer=checkpointer) if checkpointer else g.compile()


_compiled = None


def get_compiled_graph(checkpointer=None):
    """Cache the compiled graph for the lifetime of the process."""
    global _compiled
    if _compiled is None:
        _compiled = build_graph(checkpointer=checkpointer)
    return _compiled
