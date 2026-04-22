"""judge — LLM call 2. Verifies citations, removes hallucinations.

Reads `violations` (raw, from compliance_checker) and writes
`verified_violations` (plain assignment, no reducer). Downstream nodes
(guidance_generator, output_formatter) read `verified_violations` only.

Fail-open policy: if the judge LLM or its response fails, the original
violations are forwarded unchanged. Losing a genuine violation is worse
than preserving a possibly-hallucinated one — the compliance officer
will catch the false positive at review time.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage

from core.config import get_llm
from core.nodes._shared import as_text, get_compliance_cfg, parse_and_validate, render_prompt
from core.types import ComplianceState

logger = logging.getLogger(__name__)


async def run(state: ComplianceState) -> dict:
    violations = state.get("violations") or []
    if not violations:
        return {"verified_violations": [], "judge_results": []}

    violations_payload = [v.model_dump() for v in violations]
    prompt = render_prompt(
        "judge.j2",
        post_text=state["post_content"].raw_text,
        rule_injection=state["rule_injection"],
        violations_json=json.dumps(violations_payload, indent=2),
    )

    try:
        judge_cfg = get_compliance_cfg().get("judge") or {}
        max_tokens = int(judge_cfg.get("max_tokens", 2000))
        llm = get_llm(max_tokens=max_tokens)
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = as_text(resp.content)
        verdict = parse_and_validate(raw, "judge_verdict.schema.json")

        valid_ids = {
            vv["rule_id"]
            for vv in verdict.get("verified_violations", [])
            if vv.get("overall_valid", False)
        }
        verified = [v for v in violations if v.rule_id in valid_ids]
        logger.info(
            "judge kept %d / %d violations (removed %d hallucinations)",
            len(verified), len(violations),
            len(violations) - len(verified),
        )
        return {
            "verified_violations": verified,
            "judge_results": [verdict],
        }

    except Exception as e:
        logger.warning(
            "judge failed — failing open, keeping all original violations. Error: %s",
            e,
        )
        return {
            "verified_violations": list(violations),
            "judge_results": [],
        }
