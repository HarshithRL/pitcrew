"""hitl_node — pause the graph for a human reviewer.

NOT async — LangGraph's `interrupt()` is synchronous. The graph halts
here until a `Command(resume={"action": ..., "notes": ...})` is posted
to the same `thread_id`. Reviewer UI talks to this via the HITL routes
added in Step 11.
"""

from __future__ import annotations

from datetime import datetime, timezone

from langgraph.types import interrupt

from core.types import ComplianceState, HITLPayload


def run(state: ComplianceState) -> dict:
    post = state["post_content"]
    payload = HITLPayload(
        submission_id=state["submission_id"],
        thread_id=state["submission_id"],
        post_text=post.raw_text,
        platform=post.platform,
        risk_score=state["risk_score"],
        violations=state["violations"],
        awaiting_since=datetime.now(timezone.utc).isoformat(),
    )

    decision = interrupt(payload.model_dump())

    return {
        "human_decision": decision.get("action"),
        "human_notes": decision.get("notes", ""),
    }
