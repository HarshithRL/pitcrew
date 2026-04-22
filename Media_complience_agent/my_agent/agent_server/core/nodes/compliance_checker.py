"""compliance_checker — LLM call 1. Returns list[Violation].

Renders `compliance_checker.j2` with the rule injection string and the
post text, calls the Databricks-hosted LLM, and parses + schema-validates
the JSON response into `Violation` objects.

Retries per `retry.max_attempts` in compliance.yaml. On exhaustion,
returns `[]` and logs a WARNING with the final raw LLM response.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

from core.config import get_llm
from core.nodes._shared import (
    as_text,
    get_compliance_cfg,
    parse_and_validate,
    render_prompt,
)
from core.types import ComplianceState, Violation

logger = logging.getLogger(__name__)


async def run(state: ComplianceState) -> dict:
    cfg = get_compliance_cfg()
    retry_cfg = cfg.get("retry") or {}
    max_attempts = int(retry_cfg.get("max_attempts", 2))
    llm_cfg = cfg.get("llm") or {}
    max_tokens = int(llm_cfg.get("max_tokens", 4000))

    post = state["post_content"]
    prompt = render_prompt(
        "compliance_checker.j2",
        rule_injection=state["rule_injection"],
        post_text=post.raw_text,
        platform=post.platform,
    )

    llm = get_llm(max_tokens=max_tokens)
    last_raw = ""
    last_err: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = await llm.ainvoke([HumanMessage(content=prompt)])
            last_raw = as_text(resp.content)
            try:
                parsed = parse_and_validate(last_raw, "violations.schema.json")
            except Exception as parse_err:
                logger.warning(
                    "compliance_checker attempt %d/%d — JSON/schema error: %s. "
                    "Raw response (first 500 chars): %s",
                    attempt, max_attempts, parse_err, last_raw[:500],
                )
                last_err = parse_err
                continue
            violations = [Violation(**v) for v in parsed]
            logger.info(
                "compliance_checker found %d violation(s) on attempt %d",
                len(violations), attempt,
            )
            return {"violations": violations}

        except Exception as e:
            last_err = e
            logger.warning(
                "compliance_checker attempt %d/%d failed: %s",
                attempt, max_attempts, e,
            )

    logger.warning(
        "compliance_checker exhausted %d retries; returning empty list. "
        "Last raw response: %s. Last error: %s",
        max_attempts, last_raw[:500], last_err,
    )
    return {"violations": []}
