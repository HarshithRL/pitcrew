"""Entry point — imports agent (registers handlers) then boots AgentServer.

Mounts the Flask testing UI at /ui via WsgiToAsgi so the same container serves
both /invocations (streaming SSE from AgentServer) and /ui (chat UI). The
Databricks App OAuth proxy sits in front of everything — the UI JS calls
/invocations same-origin and the proxy handles auth.
"""

import logging
import os

logging.basicConfig(level="INFO", format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

# Must be set BEFORE any mlflow import path in `agent`.
os.environ.setdefault("MLFLOW_TRACKING_URI", "databricks")

import agent  # noqa: F401 — registers @invoke/@stream side-effect

from asgiref.wsgi import WsgiToAsgi
from mlflow.genai.agent_server import AgentServer

from ui import create_ui_app

agent_server = AgentServer("ResponsesAgent")
app = agent_server.app

# Mount the Flask UI under /ui. Flask is WSGI, AgentServer is ASGI, so wrap it.
app.mount("/ui", WsgiToAsgi(create_ui_app()))


def main():
    agent_server.run(app_import_string="start_server:app")


if __name__ == "__main__":
    main()
