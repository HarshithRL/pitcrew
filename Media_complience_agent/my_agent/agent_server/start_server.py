"""Entry point — imports agent (registers handlers) then boots AgentServer.

Mounts the Flask testing UI at /ui via WsgiToAsgi so the same container serves
both /invocations (streaming SSE from AgentServer) and /ui (chat UI). The
Databricks App OAuth proxy sits in front of everything — the UI JS calls
/invocations same-origin and the proxy handles auth.

Custom FastAPI routes for compliance work live on the same `app` object:
  Admin:    POST /rules/upload         — upload + parse a rules PDF/DOCX
            GET  /rules/current        — list loaded rules
            GET  /rules/injection-preview — exact LLM injection string
  User:     POST /upload               — submit a post for compliance check
  HITL:     GET  /hitl/review/{thread_id}  — fetch paused-graph payload
            POST /hitl/decision/{thread_id} — resume the graph with decision
"""

import json
import logging
import os
import uuid

logging.basicConfig(level="INFO", format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

# Must be set BEFORE any mlflow import path in `agent`.
os.environ.setdefault("MLFLOW_TRACKING_URI", "databricks")

import agent  # noqa: F401 — registers @invoke side-effect

from asgiref.wsgi import WsgiToAsgi
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from langgraph.types import Command
from mlflow.genai.agent_server import AgentServer
from mlflow.types.responses import ResponsesAgentRequest

from agent import agent as compliance_agent
from core.config import get_llm, settings
from core.post_extractor import extract_post
from core.rule_registry import get_current_registry, reload_registry
from ui import create_ui_app

logger = logging.getLogger(__name__)

agent_server = AgentServer("ResponsesAgent")
app = agent_server.app

# Mount the Flask UI under /ui. Flask is WSGI, AgentServer is ASGI, so wrap it.
app.mount("/ui", WsgiToAsgi(create_ui_app()))


# ── helpers ────────────────────────────────────────────────────────────────

_ALLOWED_MIME = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _extract_response_text(response) -> str:
    """Pull the text payload out of a ResponsesAgentResponse.

    Output items come back as Pydantic models (OutputItem), not dicts — access
    fields via attributes and fall back to dict-style for robustness.
    """
    item = response.output[0]
    if hasattr(item, "content"):
        block = item.content[0]
        return getattr(block, "text", None) or block.get("text", "")
    # Dict fallback (shouldn't normally hit this branch)
    return item["content"][0]["text"]


# ── admin: rules management ────────────────────────────────────────────────

@app.post("/rules/upload")
async def upload_rules(
    file: UploadFile = File(...),
    jurisdiction: str = Form(...),
    base_rule_number: str = Form(...),
):
    """Admin uploads a new rules PDF/DOCX — parses with pymupdf4llm + LLM, reloads registry."""
    if file.content_type not in _ALLOWED_MIME:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Upload a PDF or DOCX.",
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        # Rule parsing uses the Sonnet endpoint (RULE_PARSER_ENDPOINT_NAME)
        # with max_tokens=6000. See agent.py __init__ for context.
        llm = get_llm(
            endpoint=settings.RULE_PARSER_ENDPOINT_NAME,
            max_tokens=6000,
        )
        new_registry = await reload_registry(
            pdf_bytes=file_bytes,
            jurisdiction=jurisdiction.upper(),
            base_rule_number=base_rule_number,
            doc_name=file.filename,
            llm_client=llm,
        )
    except Exception as e:
        logger.exception("rule parsing failed")
        raise HTTPException(status_code=500, detail=f"Rule parsing failed: {e}")

    preview = [
        {
            "rule_id": r.rule_id,
            "severity": r.severity,
            "citation_label": r.citation_label,
        }
        for r in new_registry.rules[:5]
    ]

    return {
        "status": "loaded",
        "rule_count": len(new_registry.rules),
        "source_document": file.filename,
        "jurisdiction": jurisdiction.upper(),
        "preview": preview,
        "loaded_at": new_registry.loaded_at.isoformat(),
    }


@app.get("/rules/current")
def current_rules():
    """Return every loaded rule with its ID, citation label, and severity."""
    try:
        registry = get_current_registry()
    except RuntimeError as e:
        return JSONResponse(
            status_code=503,
            content={"status": "empty", "message": str(e)},
        )

    return {
        "rule_count": len(registry.rules),
        "source_documents": registry.source_documents,
        "loaded_at": registry.loaded_at.isoformat(),
        "rules": [
            {
                "rule_id": r.rule_id,
                "citation_label": r.citation_label,
                "severity": r.severity,
                "jurisdiction": r.jurisdiction,
                "is_prohibition": r.is_prohibition,
                "is_requirement": r.is_requirement,
            }
            for r in registry.rules
        ],
    }


@app.get("/rules/injection-preview")
def injection_preview():
    """Return the exact injection string the LLM receives for compliance checks."""
    try:
        registry = get_current_registry()
    except RuntimeError as e:
        return JSONResponse(
            status_code=503,
            content={"status": "empty", "message": str(e)},
        )

    return {
        "injection_length_chars": len(registry.injection_string),
        "rule_count": len(registry.rules),
        "injection": registry.injection_string,
    }


# ── user: submit a post ────────────────────────────────────────────────────

@app.post("/upload")
async def upload_post(
    file: UploadFile = File(...),
    platform: str = Form(None),
):
    """User submits a social media post (PDF/DOCX). Returns ComplianceReport or HITL envelope."""
    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    submission_id = str(uuid.uuid4())[:8]

    try:
        post_content = extract_post(
            file_bytes=file_bytes,
            file_name=file.filename,
            submission_id=submission_id,
            platform=platform,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    payload = {
        "submission_id": submission_id,
        "post_text": post_content.raw_text,
        "platform": platform,
        "file_name": file.filename,
        "file_type": post_content.file_type,
    }
    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": json.dumps(payload)}]
    )

    response = await compliance_agent.predict(request)
    result = json.loads(_extract_response_text(response))
    result["thread_id"] = submission_id

    if result.get("status") == "hitl_required":
        result["hitl_review_url"] = f"/hitl/review/{submission_id}"

    return result


# ── HITL: review + decision ────────────────────────────────────────────────

@app.get("/hitl/review/{thread_id}")
async def hitl_review(thread_id: str):
    """Return the paused-graph payload (post, violations, risk score) for the reviewer."""
    graph = compliance_agent._graph
    config = {"configurable": {"thread_id": thread_id}}

    try:
        state = graph.get_state(config)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Thread not found: {e}")

    if not state or not state.tasks:
        raise HTTPException(
            status_code=404,
            detail=f"No active state for thread {thread_id}",
        )

    interrupted = [t for t in state.tasks if t.interrupts]
    if not interrupted:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No pending HITL review for thread {thread_id}. "
                "Check may have already completed."
            ),
        )

    interrupt_value = interrupted[0].interrupts[0].value
    return {
        "thread_id": thread_id,
        "status": "awaiting_review",
        "payload": interrupt_value,
    }


@app.post("/hitl/decision/{thread_id}")
async def hitl_decision(
    thread_id: str,
    action: str = Form(...),
    notes: str = Form(""),
):
    """Resume a paused graph with the reviewer's decision. Returns final ComplianceReport."""
    allowed = {"approved", "rejected", "edit_requested"}
    if action not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"action must be one of: {', '.join(sorted(allowed))}",
        )

    graph = compliance_agent._graph
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = await graph.ainvoke(
            Command(resume={"action": action, "notes": notes}),
            config=config,
        )
    except Exception as e:
        logger.exception("resume failed")
        raise HTTPException(status_code=500, detail=f"Failed to resume graph: {e}")

    report = result.get("compliance_report")
    if report is not None:
        return json.loads(report.model_dump_json())

    return {
        "status": "error",
        "thread_id": thread_id,
        "message": "Graph completed but no report was generated.",
    }


# ── server boot ─────────────────────────────────────────────────────────────

def main():
    agent_server.run(app_import_string="start_server:app")


if __name__ == "__main__":
    main()
