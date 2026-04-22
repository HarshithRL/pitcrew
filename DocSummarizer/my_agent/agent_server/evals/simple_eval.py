"""Manual eval loop — bypasses ``mlflow.genai.evaluate()``'s harness.

We hit an MLflow 3 serverless-runtime bug where the harness crashes in
``_get_new_expectations`` with ``AttributeError: 'NoneType' object has no
attribute 'info'`` before scorers ever run. This module does exactly what
the harness would do — call predict_fn per row, call each scorer, aggregate
metrics, log them to MLflow — but without the broken plumbing.

Usage is identical from the caller's perspective::

    from evals.simple_eval import run_eval
    metrics = run_eval(
        stage="parser",
        data=dataset.build_parser_rows(...),
        predict_fn=parser_predict,
        scorers=parser_scorers.ALL,
    )
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable

import mlflow
from mlflow.entities import Feedback
from mlflow.genai.scorers import Correctness, Guidelines, Safety

logger = logging.getLogger(__name__)


def _scorer_name(scorer) -> str:
    """Best-effort readable name for log output."""
    for attr in ("name", "__name__"):
        v = getattr(scorer, attr, None)
        if v:
            return str(v)
    return scorer.__class__.__name__


def _score_to_float(value) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _call_builtin_judge(scorer, *, inputs: dict, outputs, expectations: dict):
    """Call Guidelines / Correctness / Safety judges the same way evaluate() does.

    These scorers expose a ``__call__`` that accepts kwargs matching the eval
    schema. We pass everything; scorers pick the kwargs they need.
    """
    return scorer(
        inputs=inputs or {},
        outputs=outputs,
        expectations=expectations or {},
    )


def _call_code_scorer(scorer, *, inputs: dict, outputs, expectations: dict):
    """Call @scorer-decorated user functions.

    ``@scorer`` functions take a subset of {inputs, outputs, expectations, trace}
    kwargs by name. We pass all of them — the decorator filters.
    """
    return scorer(
        inputs=inputs or {},
        outputs=outputs,
        expectations=expectations or {},
    )


def _invoke(scorer, *, inputs, outputs, expectations):
    """Dispatch to the right caller based on scorer type."""
    if isinstance(scorer, (Guidelines, Correctness, Safety)):
        return _call_builtin_judge(scorer, inputs=inputs, outputs=outputs, expectations=expectations)
    return _call_code_scorer(scorer, inputs=inputs, outputs=outputs, expectations=expectations)


def run_eval(
    stage: str,
    data: list[dict],
    predict_fn: Callable,
    scorers: Iterable,
) -> dict:
    """Run a full eval over ``data`` and return the aggregated metrics dict.

    Opens one MLflow run named ``eval-{stage}``; under it, logs one per-row
    metric per scorer plus a mean across rows. Prints per-row feedback as it
    goes so failures are visible in the notebook cell output.
    """
    scorers = list(scorers)
    if not data:
        logger.warning("stage=%s has 0 rows — skipping", stage)
        return {}

    with mlflow.start_run(run_name=f"eval-{stage}") as run:
        print(f"[{stage}] rows={len(data)} run_id={run.info.run_id}")

        # name -> list of floats across rows
        bucket: dict[str, list[float]] = {}

        for i, row in enumerate(data):
            inputs = row.get("inputs") or {}
            expectations = row.get("expectations") or {}

            try:
                outputs = predict_fn(**inputs)
            except Exception as e:
                logger.exception("row=%d predict_fn failed: %s", i, e)
                continue

            for scorer in scorers:
                name = _scorer_name(scorer)
                try:
                    feedback = _invoke(scorer, inputs=inputs, outputs=outputs, expectations=expectations)
                except Exception as e:
                    logger.exception("row=%d scorer=%s failed: %s", i, name, e)
                    continue

                value = feedback.value if isinstance(feedback, Feedback) else feedback
                rationale = getattr(feedback, "rationale", "") if isinstance(feedback, Feedback) else ""
                score = _score_to_float(value)

                bucket.setdefault(name, []).append(score)
                print(f"  row={i} {name}: {score:.3f}  {rationale}".rstrip())

                # Per-row metric series so the MLflow UI plots trends.
                mlflow.log_metric(key=f"{name}", value=score, step=i)

        metrics: dict[str, float] = {}
        for name, scores in bucket.items():
            mean = sum(scores) / len(scores)
            metrics[f"{name}/mean"] = mean
            mlflow.log_metric(key=f"{name}_mean", value=mean)

        print(f"[{stage}] metrics={metrics}")
        return metrics
