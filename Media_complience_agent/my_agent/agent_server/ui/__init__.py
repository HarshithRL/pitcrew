"""Flask UI chassis for the compliance agent.

Mounted under /ui by start_server.py. Serves the single-page admin + user
HTML, and proxies the three POST actions (rules upload, post upload, HITL
decision) to the same-process FastAPI routes on localhost so the browser
never talks to FastAPI directly.

Proxy-auth: `X-Forwarded-Email` / `X-Forwarded-User` are passed by the
Databricks App OAuth layer — kept in the index route so the header shows
the signed-in user.

Route paths are *bare* here; the `/ui` mount in start_server.py prepends
the prefix. So Flask route `/upload` is reached by the browser at
`/ui/upload`.
"""

from __future__ import annotations

import logging

import requests
from flask import Flask, jsonify, render_template, request

logger = logging.getLogger(__name__)

MAX_UPLOAD_MB = 20
# Where to reach FastAPI routes from within the same container
_API_BASE = "http://localhost:8000"


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

    @app.post("/rules/upload")
    def ui_upload_rules():
        """Admin uploads a rules document — forwards to FastAPI /rules/upload."""
        file = request.files.get("file")
        jurisdiction = request.form.get("jurisdiction", "")
        base_rule_number = request.form.get("base_rule_number", "")

        if not file or not file.filename:
            return jsonify({"error": "No file provided"}), 400
        if not jurisdiction or not base_rule_number:
            return jsonify({"error": "jurisdiction and base_rule_number are required"}), 400

        file.stream.seek(0)
        resp = requests.post(
            f"{_API_BASE}/rules/upload",
            files={"file": (file.filename, file.stream, file.content_type)},
            data={
                "jurisdiction": jurisdiction,
                "base_rule_number": base_rule_number,
            },
            timeout=300,
        )
        return jsonify(resp.json()), resp.status_code

    @app.post("/upload")
    def ui_upload_post():
        """User submits a social-media post — forwards to FastAPI /upload."""
        file = request.files.get("file")
        platform = request.form.get("platform", "")

        if not file or not file.filename:
            return jsonify({"error": "No file provided"}), 400

        file.stream.seek(0)
        resp = requests.post(
            f"{_API_BASE}/upload",
            files={"file": (file.filename, file.stream, file.content_type)},
            data={"platform": platform} if platform else {},
            timeout=300,
        )
        return jsonify(resp.json()), resp.status_code

    @app.post("/hitl/decision")
    def ui_hitl_decision():
        """Reviewer submits approve/reject/edit_requested — forwards to FastAPI /hitl/decision/{id}."""
        thread_id = request.form.get("thread_id", "")
        action = request.form.get("action", "")
        notes = request.form.get("notes", "")

        if not thread_id or not action:
            return jsonify({"error": "thread_id and action required"}), 400

        resp = requests.post(
            f"{_API_BASE}/hitl/decision/{thread_id}",
            data={"action": action, "notes": notes},
            timeout=120,
        )
        return jsonify(resp.json()), resp.status_code

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    return app
