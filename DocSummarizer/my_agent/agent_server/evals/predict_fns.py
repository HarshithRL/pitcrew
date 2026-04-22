"""predict_fn variants for the three eval stages.

Each function's kwarg names must match the ``inputs`` keys in the eval dataset
rows — that's the MLflow 3 genai contract. See
``references/eval-dataset-patterns.md`` for the schema.

    parser_predict       inputs={"source_path": str}
    chunk_predict        inputs={"path": list[str], "content": str}
    doc_predict          inputs={"source_path": str}

The parser predict_fn returns intermediates (raw_text + sections + chunks) as
a plain dict so parser scorers can inspect structure without going through
traces. The chunk predict_fn returns just the summary string. The doc
predict_fn returns the final_summary string.
"""

from __future__ import annotations

from pathlib import Path

import mlflow

from core.nodes import (
    chunk_node,
    ingest_node,
    parse_structure_node,
    summarize_chunk_node,
)
from core.state import ChunkJob, DocState
from core.graph import build_graph


@mlflow.trace(name="parser_predict")
def parser_predict(source_path: str) -> dict:
    """Run ingest → parse_structure → chunk and return all intermediates.

    No LLM call. Output shape::

        {
            "raw_text":  str,
            "sections":  list[{"path": list[str], "content": list[str]}],
            "chunks":    list[{"index": int, "path": list[str], "content": str}],
        }
    """
    state = DocState(source_path=source_path)

    ingest_out = ingest_node(state)
    state = state.model_copy(update=ingest_out)

    parse_out = parse_structure_node(state)
    state = state.model_copy(update=parse_out)

    chunk_out = chunk_node(state)

    return {
        "raw_text": state.raw_text,
        "sections": state.sections,
        "chunks": [c.model_dump() for c in chunk_out["chunks"]],
    }


@mlflow.trace(name="chunk_predict")
def chunk_predict(path: list[str], content: str) -> str:
    """Run summarize_chunk_node in isolation — the only LLM call in the pipeline.

    The dataset row supplies a pre-chunked ``(path, content)`` pair so the
    summarizer is tested without any parser variability.
    """
    job = ChunkJob(index=0, path=list(path or []), content=content, source_path=None)
    out = summarize_chunk_node(job)
    return out["chunk_summaries"][0].summary


_compiled = None


@mlflow.trace(name="doc_predict")
def doc_predict(source_path: str) -> str:
    """Run the full graph end-to-end and return ``final_summary``."""
    global _compiled
    if _compiled is None:
        _compiled = build_graph()

    # Guard against missing files up front so the error is obvious in the UI.
    if not Path(source_path).exists():
        raise FileNotFoundError(f"source_path not found: {source_path}")

    result = _compiled.invoke({"source_path": source_path})
    return result["final_summary"]
