"""guidance_generator — LLM call 3. Produces `ComplianceGuidance`.

Reads `verified_violations` (not `violations`) so any hallucinations the
judge removed stay out of the what-to-avoid list and the rewrite.

On failure returns a minimal guidance with the original post as the
"rewrite" and a note that generation failed — the compliance officer
still gets a usable payload, just without suggestions.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage

from core.config import get_llm
from core.nodes._shared import as_text, get_compliance_cfg, parse_and_validate, render_prompt
from core.types import ComplianceGuidance, ComplianceState, GuidanceItem

logger = logging.getLogger(__name__)


async def run(state: ComplianceState) -> dict:
    verified = state.get("verified_violations") or []
    post = state["post_content"]

    verified_payload = [v.model_dump() for v in verified]
    prompt = render_prompt(
        "guidance_generator.j2",
        post_text=post.raw_text,
        platform=post.platform,
        verified_violations_json=json.dumps(verified_payload, indent=2),
        human_decision=state.get("human_decision"),
        human_notes=state.get("human_notes"),
    )

    try:
        guidance_cfg = get_compliance_cfg().get("guidance") or {}
        max_tokens = int(guidance_cfg.get("max_tokens", 3000))
        llm = get_llm(max_tokens=max_tokens)
        resp = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = as_text(resp.content)
        data = parse_and_validate(raw, "guidance.schema.json")

        guidance = ComplianceGuidance(
            what_to_post=data["what_to_post"],
            what_to_avoid=[GuidanceItem(**item) for item in data["what_to_avoid"]],
            compliant_rewrite=data["compliant_rewrite"],
            summary=data["summary"],
        )
        return {"guidance": guidance}

    except Exception as e:
        logger.warning(
            "guidance_generator failed; returning minimal fallback. Error: %s", e,
        )
        fallback = ComplianceGuidance(
            what_to_post=[],
            what_to_avoid=[],
            compliant_rewrite=post.raw_text,
            summary="Guidance generation failed. Review violations manually.",
        )
        return {"guidance": fallback}
