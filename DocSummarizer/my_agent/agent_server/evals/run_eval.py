"""Eval entry point — runs A/B/C evaluations against the document summarizer.

Usage::

    cd agent_server
    python -m evals.run_eval --stage all   --data-root ./eval_data
    python -m evals.run_eval --stage parser --data-root ./eval_data
    python -m evals.run_eval --stage chunk  --data-root ./eval_data
    python -m evals.run_eval --stage doc    --data-root ./eval_data

Each stage calls ``mlflow.genai.evaluate()`` with the matching predict_fn +
scorer set. Traces and per-row feedback appear in the MLflow experiment set
via ``MLFLOW_EXPERIMENT_NAME`` (or the default below).
"""

from __future__ import annotations

import argparse
import logging
import sys

import mlflow
import mlflow.langchain

from core.config import settings
from evals import dataset
from evals.predict_fns import chunk_predict, doc_predict, parser_predict
from evals.scorers import chunk_scorers, doc_scorers, parser_scorers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evals")

_DEFAULT_EXPERIMENT = "/Shared/document-summarizer-eval"


def _set_experiment() -> None:
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI or "databricks")
    name = settings.MLFLOW_EXPERIMENT_NAME or _DEFAULT_EXPERIMENT
    mlflow.set_experiment(name)
    logger.info("mlflow experiment=%s", name)


def _run(stage: str, data: list[dict], predict_fn, scorers) -> dict:
    if not data:
        logger.warning("stage=%s has 0 rows — skipping", stage)
        return {}
    with mlflow.start_run(run_name=f"eval-{stage}") as run:
        logger.info("stage=%s rows=%d run_id=%s", stage, len(data), run.info.run_id)
        results = mlflow.genai.evaluate(
            data=data,
            predict_fn=predict_fn,
            scorers=scorers,
        )
        metrics = dict(getattr(results, "metrics", {}) or {})
        logger.info("stage=%s metrics=%s", stage, metrics)
        return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["parser", "chunk", "doc", "all"], default="all")
    parser.add_argument("--data-root", default="./eval_data", help="dir of per-doc folders")
    args = parser.parse_args()

    _set_experiment()
    mlflow.langchain.autolog()  # captures the summarizer's LLM span for RAG-style judges

    stages = ["parser", "chunk", "doc"] if args.stage == "all" else [args.stage]

    summary: dict[str, dict] = {}
    if "parser" in stages:
        summary["parser"] = _run(
            "parser",
            dataset.build_parser_rows(args.data_root),
            parser_predict,
            parser_scorers.ALL,
        )
    if "chunk" in stages:
        summary["chunk"] = _run(
            "chunk",
            dataset.build_chunk_rows(args.data_root),
            chunk_predict,
            chunk_scorers.ALL,
        )
    if "doc" in stages:
        summary["doc"] = _run(
            "doc",
            dataset.build_doc_rows(args.data_root),
            doc_predict,
            doc_scorers.ALL,
        )

    print("\n=== eval summary ===")
    for stage, metrics in summary.items():
        print(f"[{stage}]")
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
