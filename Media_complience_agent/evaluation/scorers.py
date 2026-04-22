"""MLflow GenAI @scorer functions for the compliance agent.

Uses the `mlflow.genai.scorers.scorer` decorator (NOT `make_metric` or the
classic `mlflow.models.evaluate()` path — those two systems don't interop).

Targets:
  rule_citation_accuracy  >= 0.90
  guidance_completeness   >= 0.85
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__), "..", "my_agent", "agent_server"
    ),
)

from mlflow.genai.scorers import scorer  # noqa: E402


@scorer
def rule_citation_accuracy(outputs, inputs=None, expectations=None):
    """For each violation in the output:
      1. Verify rule_id exists in the current registry.
      2. Verify violated_text appears in the post text.
    Score = valid_citations / total_citations. Target >= 0.90.
    Clean post (no violations) scores 1.0.
    """
    from core.rule_registry import get_current_registry

    try:
        registry = get_current_registry()
        known_ids = {r.rule_id for r in registry.rules}
    except RuntimeError:
        return None

    try:
        output = json.loads(outputs) if isinstance(outputs, str) else outputs
        violations = output.get("violations", [])
        post_text = ""
        if inputs:
            inp = json.loads(inputs) if isinstance(inputs, str) else inputs
            post_text = inp.get("post_text", "").lower()
    except Exception:
        return 0.0

    if not violations:
        return 1.0

    valid = 0
    for v in violations:
        rule_id_ok = v.get("rule_id", "") in known_ids
        text_ok = (
            v.get("violated_text", "").lower() in post_text
            if post_text
            else True
        )
        if rule_id_ok and text_ok:
            valid += 1

    return round(valid / len(violations), 4)


@scorer
def guidance_completeness(outputs, inputs=None, expectations=None):
    """For every item in what_to_avoid, verify the problematic phrase is
    NOT in compliant_rewrite.
    Score = fixed_items / total_items. Target >= 0.85.
    No items to avoid → score 1.0.
    """
    try:
        output = json.loads(outputs) if isinstance(outputs, str) else outputs
        guidance = output.get("guidance", {})
        what_to_avoid = guidance.get("what_to_avoid", [])
        rewrite = guidance.get("compliant_rewrite", "").lower()
    except Exception:
        return 0.0

    if not what_to_avoid:
        return 1.0

    fixed = 0
    for item in what_to_avoid:
        phrase = item.get("item", "").lower().strip()
        if phrase and phrase not in rewrite:
            fixed += 1
        elif not phrase:
            fixed += 1

    return round(fixed / len(what_to_avoid), 4)
