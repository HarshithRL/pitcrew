"""Offline evaluation harness.

Builds a DataFrame from evaluation/datasets/*.json, invokes the compliance
agent per row, and runs the two @scorer functions via mlflow.genai.evaluate().

Requires DATABRICKS credentials (env or profile) — the agent loads rules
and calls the LLM at import time.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "my_agent", "agent_server"
    ),
)

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.types.responses import ResponsesAgentRequest  # noqa: E402

from evaluation.scorers import (  # noqa: E402
    guidance_completeness,
    rule_citation_accuracy,
)


def build_eval_dataframe() -> pd.DataFrame:
    datasets_dir = os.path.join(os.path.dirname(__file__), "datasets")
    rows = []
    for fname in ["violation_posts.json", "compliant_posts.json"]:
        path = os.path.join(datasets_dir, fname)
        with open(path) as f:
            posts = json.load(f)
        for post in posts:
            rows.append({
                "post_text": post["post_text"],
                "platform": post.get("platform", ""),
                "expected_violations": post.get("expected_violations", []),
                "expected_tier": post.get("expected_tier", ""),
                "notes": post.get("notes", ""),
            })
    return pd.DataFrame(rows)


def run():
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(
        os.environ.get(
            "MLFLOW_EXPERIMENT_NAME",
            "/Users/harshith.r@diggibyte.com/Media-Compliance-Agent",
        )
    )

    from agent import agent as compliance_agent
    from core.rule_registry import get_current_registry

    try:
        registry = get_current_registry()
        print(f"Rules loaded: {len(registry.rules)}")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        print("Run with DATABRICKS credentials set.")
        sys.exit(1)

    df = build_eval_dataframe()
    print(f"Evaluation dataset: {len(df)} posts")

    def predict_fn(row) -> str:
        post_text = row["post_text"]
        platform = row.get("platform", "")
        submission_id = f"eval_{abs(hash(post_text)) % 10000:04d}"

        request = ResponsesAgentRequest(
            input=[{
                "role": "user",
                "content": json.dumps({
                    "submission_id": submission_id,
                    "post_text": post_text,
                    "platform": platform,
                    "file_name": "eval_post.txt",
                    "file_type": "text",
                }),
            }]
        )

        response = asyncio.run(compliance_agent.predict(request))
        item = response.output[0]
        if hasattr(item, "content"):
            block = item.content[0]
            return getattr(block, "text", None) or block.get("text", "")
        return item["content"][0]["text"]

    print("Running mlflow.genai.evaluate()...")
    results = mlflow.genai.evaluate(
        data=df,
        predict_fn=predict_fn,
        scorers=[rule_citation_accuracy, guidance_completeness],
    )

    print("\n=== Evaluation Results ===")
    print(results.metrics)
    print("\nTarget thresholds:")
    print("  rule_citation_accuracy >= 0.90")
    print("  guidance_completeness  >= 0.85")

    return results


if __name__ == "__main__":
    run()
