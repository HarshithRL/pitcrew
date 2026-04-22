"""Flask UI: file upload + summary display + feedback capture.

Routes:
    GET  /           — upload form
    POST /summarize  — multipart upload, invokes the graph, returns JSON
    POST /feedback   — logs a user thumbs / comment to the MLflow trace
    GET  /whoami     — proxy auth echo
    GET  /healthz    — liveness
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path

import mlflow
from flask import Flask, jsonify, render_template, request
from mlflow.entities import AssessmentSource, AssessmentSourceType
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

ALLOWED_SUFFIXES = {".txt", ".md", ".pdf"}
MAX_UPLOAD_MB = 20
FEEDBACK_NAME = "user_feedback"


def create_ui_app() -> Flask:
    app = Flask(
        __name__,
        static_folder="static",
        static_url_path="/static",
        template_folder="templates",
    )
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

    @app.get("/")
    def index():
        email = request.headers.get("X-Forwarded-Email", "")
        user = request.headers.get("X-Forwarded-User", "")
        return render_template("index.html", user_email=email, user_name=user)

    @app.post("/summarize")
    def summarize():
        upload = request.files.get("document")
        if not upload or not upload.filename:
            return jsonify({"error": "No file uploaded"}), 400

        filename = secure_filename(upload.filename)
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            return jsonify({
                "error": f"Unsupported file type {suffix!r}. Allowed: {sorted(ALLOWED_SUFFIXES)}",
            }), 400

        request_id = str(uuid.uuid4())
        tmp_dir = Path(tempfile.gettempdir()) / "doc_summarizer"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / f"{request_id}_{filename}"
        upload.save(tmp_path)
        logger.info("saved upload to %s (%d bytes)", tmp_path, tmp_path.stat().st_size)

        # Imported lazily so the Flask factory stays cheap and testable.
        from core.graph import get_compiled_graph
        from core.state import DocState

        try:
            graph = get_compiled_graph()
            result = graph.invoke(
                DocState(source_path=str(tmp_path)),
                config={"configurable": {"thread_id": request_id}},
            )
        except Exception as e:
            logger.exception("graph invocation failed")
            return jsonify({"error": f"Summarization failed: {e}"}), 500
        finally:
            tmp_path.unlink(missing_ok=True)

        # Trace id is what the feedback endpoint needs to attach assessments to.
        # Read it after invoke() so MLflow has already closed the root span.
        trace_id = mlflow.get_last_active_trace_id() or ""
        if not trace_id:
            logger.warning("no active MLflow trace id after summarize (request_id=%s)", request_id)

        return jsonify({
            "request_id": request_id,
            "filename": filename,
            "metadata": result.get("metadata", {}),
            "chunk_count": len(result.get("chunk_summaries", [])),
            "extracted_text": result.get("raw_text", ""),
            "summary": result.get("final_summary", ""),
            "mlflow_trace_id": trace_id,
        })

    @app.post("/feedback")
    def feedback():
        """Attach a human assessment (thumbs + optional comment) to an MLflow trace.

        Expects JSON: {trace_id: str, is_helpful: bool, comment?: str}.
        """
        payload = request.get_json(silent=True) or {}
        trace_id = (payload.get("trace_id") or "").strip()
        is_helpful = payload.get("is_helpful")
        comment = (payload.get("comment") or "").strip() or None

        if not trace_id:
            return jsonify({"error": "trace_id is required"}), 400
        if not isinstance(is_helpful, bool):
            return jsonify({"error": "is_helpful must be a boolean"}), 400

        source_id = request.headers.get("X-Forwarded-Email", "").strip() or "ui-anonymous"

        try:
            mlflow.log_feedback(
                trace_id=trace_id,
                name=FEEDBACK_NAME,
                value=is_helpful,
                rationale=comment,
                source=AssessmentSource(
                    source_type=AssessmentSourceType.HUMAN,
                    source_id=source_id,
                ),
                metadata={"channel": "web"},
            )
        except Exception as e:
            logger.exception("mlflow.log_feedback failed (trace_id=%s)", trace_id)
            return jsonify({"error": f"Failed to record feedback: {e}"}), 500

        logger.info("feedback logged: trace_id=%s helpful=%s source=%s", trace_id, is_helpful, source_id)
        return jsonify({"ok": True})

    @app.get("/whoami")
    def whoami():
        return jsonify({
            "email": request.headers.get("X-Forwarded-Email", ""),
            "user": request.headers.get("X-Forwarded-User", ""),
        })

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    return app
