"""ComplianceResponsesAgent — MLflow wrapper around the compliance LangGraph.

Boot sequence (runs at module import):
  1. MLflow tracing + experiment wiring (preserved from the skeleton).
  2. Instantiate the agent, which:
       - builds a MemorySaver checkpointer,
       - compiles the compliance graph with that checkpointer (required so
         HITL interrupt/resume see the same state),
       - best-effort loads default rules from configs/rules/registry.yaml,
  3. Register the agent with AgentServer via `set_model()` + `@invoke()`.

Per-request flow in `predict()`:
  - Expect a JSON payload on the latest user message:
      { submission_id, post_text, platform, file_name, file_type }
  - Build ComplianceState, invoke the graph.
  - If the result carries `__interrupt__` (HITL pause), return a
    {"status": "hitl_required", ...} envelope the UI uses to render the
    reviewer panel.
  - Otherwise return `compliance_report.model_dump()` as JSON.
"""

import asyncio
import json
import logging
import os
import uuid

import mlflow
import mlflow.langchain
from langgraph.checkpoint.memory import MemorySaver
from mlflow.genai.agent_server import invoke
from mlflow.models import set_model
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
)

from core.config import get_llm, settings

from core.graph import get_compiled_graph
from core.rule_registry import get_current_registry, load_default_rules
from core.types import ComplianceState, PostContent

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# ── MLflow boot (preserved from skeleton) ──
_tracking_uri = settings.MLFLOW_TRACKING_URI or "databricks"
mlflow.set_tracking_uri(_tracking_uri)
logger.info("MLflow tracking_uri=%s", mlflow.get_tracking_uri())

logger.info(
    "Databricks env: HOST=%s CLIENT_ID_set=%s CLIENT_SECRET_set=%s TOKEN_set=%s",
    os.environ.get("DATABRICKS_HOST", "<unset>"),
    bool(os.environ.get("DATABRICKS_CLIENT_ID")),
    bool(os.environ.get("DATABRICKS_CLIENT_SECRET")),
    bool(os.environ.get("DATABRICKS_TOKEN")),
)

mlflow.langchain.autolog()
logger.info("mlflow.langchain.autolog() registered")

EXPERIMENT_NAME = settings.MLFLOW_EXPERIMENT_NAME
EXPERIMENT_ID = settings.MLFLOW_EXPERIMENT_ID
logger.info("Experiment config: name=%r id=%r", EXPERIMENT_NAME, EXPERIMENT_ID)

_active_experiment = None
if EXPERIMENT_ID:
    try:
        _active_experiment = mlflow.set_experiment(experiment_id=EXPERIMENT_ID)
        logger.info(
            "set_experiment by id=%s OK → resolved name=%r",
            EXPERIMENT_ID, _active_experiment.name,
        )
    except Exception as e:
        logger.exception("set_experiment by id=%s FAILED: %s", EXPERIMENT_ID, e)

if _active_experiment is None and EXPERIMENT_NAME:
    try:
        _active_experiment = mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info(
            "set_experiment by name=%r OK → id=%s",
            EXPERIMENT_NAME, _active_experiment.experiment_id,
        )
    except Exception as e:
        logger.exception("set_experiment by name=%r FAILED: %s", EXPERIMENT_NAME, e)

if _active_experiment is None:
    logger.error(
        "NO MLflow experiment is active (name=%r id=%r). "
        "Traces WILL NOT reach the intended experiment.",
        EXPERIMENT_NAME, EXPERIMENT_ID,
    )
else:
    try:
        with mlflow.start_span(name="startup_probe") as span:
            span.set_attributes({"probe": True})
        logger.info(
            "Startup probe trace written to experiment id=%s",
            _active_experiment.experiment_id,
        )
    except Exception as e:
        logger.exception(
            "Startup probe trace FAILED — tracing backend rejected write: %s",
            e,
        )

# configs/ lives next to this file
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIGS_PATH = os.path.join(_THIS_DIR, "configs")


def _message_text(msg) -> str:
    """Extract text from a ResponsesAgentRequest input item (str or content-block list)."""
    content = getattr(msg, "content", "")
    if isinstance(content, list):
        return "".join(
            c.get("text", "") for c in content if isinstance(c, dict)
        )
    return content or ""


def _make_message(text: str) -> dict:
    return {
        "type": "message",
        "id": f"msg_{uuid.uuid4().hex[:12]}",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


class ComplianceResponsesAgent(ResponsesAgent):
    """LangGraph compliance agent wrapped for MLflow AgentServer."""

    def __init__(self):
        self._checkpointer = MemorySaver()
        self._graph = get_compiled_graph(checkpointer=self._checkpointer)
        # Rule parsing: separate endpoint (RULE_PARSER_ENDPOINT_NAME — Sonnet
        # by default) with a large max_tokens. Opus 4.7 rejects `temperature`
        # and also inconsistently returns valid JSON / tool calls when asked
        # to parse full-length regulatory PDFs; Sonnet reliably emits clean
        # JSON for both default rule docs.
        self._llm = get_llm(
            endpoint=settings.RULE_PARSER_ENDPOINT_NAME,
            max_tokens=6000,
        )

        # Best-effort startup rule load — never blocks boot.
        try:
            asyncio.run(load_default_rules(CONFIGS_PATH, self._llm))
            registry = get_current_registry()
            logger.info(
                "Rules loaded at startup: %d rules from %d documents",
                len(registry.rules), len(registry.source_documents),
            )
        except Exception as e:
            logger.warning(
                "Default rules failed to load: %s. "
                "App starting with empty registry. Use POST /rules/upload.",
                e,
            )

    @mlflow.trace(span_type="AGENT")
    async def predict(
        self, request: ResponsesAgentRequest,
    ) -> ResponsesAgentResponse:
        request_id = str(uuid.uuid4())
        mlflow.update_current_trace(tags={"request_id": request_id})

        # Pull the latest user message.
        user_message = next(
            (
                m for m in reversed(request.input or [])
                if getattr(m, "role", None) == "user"
            ),
            None,
        )
        if user_message is None:
            return self._error("no_input", "No user message found in request.")

        raw_content = _message_text(user_message)
        try:
            payload = json.loads(raw_content)
        except (json.JSONDecodeError, TypeError):
            return self._error(
                "parse_error",
                "Request content must be a JSON string with post_text.",
            )

        submission_id = payload.get("submission_id") or f"sub-{uuid.uuid4().hex[:8]}"
        post_text = payload.get("post_text", "")
        platform = payload.get("platform")
        file_name = payload.get("file_name", "upload")
        file_type = payload.get("file_type", "text")

        mlflow.update_current_trace(tags={"submission_id": submission_id})

        if not post_text:
            return self._error(submission_id, "post_text is required.")

        try:
            registry = get_current_registry()
        except RuntimeError as e:
            return self._error(submission_id, str(e))

        post_content = PostContent(
            submission_id=submission_id,
            raw_text=post_text,
            platform=platform,
            file_name=file_name,
            file_type=file_type,
            char_count=len(post_text),
        )
        initial_state: ComplianceState = {
            "submission_id": submission_id,
            "post_content": post_content,
            "rule_registry": registry,
            "rule_injection": "",
            "violations": [],
            "verified_violations": [],
            "risk_score": 0.0,
            "risk_tier": "",
            "judge_results": [],
            "human_decision": None,
            "human_notes": None,
            "guidance": None,
            "compliance_report": None,
            "retry_count": 0,
            "node_metrics": {},
        }
        config = {"configurable": {"thread_id": submission_id}}

        result = await self._graph.ainvoke(initial_state, config=config)

        # HITL interrupt path: serialize the reviewer payload.
        if "__interrupt__" in result:
            interrupts = result["__interrupt__"]
            first = interrupts[0] if interrupts else None
            interrupt_value = getattr(first, "value", first)
            output_text = json.dumps({
                "status": "hitl_required",
                "thread_id": submission_id,
                "submission_id": submission_id,
                "interrupt_payload": interrupt_value,
            }, default=str)
        else:
            report = result.get("compliance_report")
            if report is not None:
                output_text = json.dumps(report.model_dump(), default=str)
            else:
                output_text = json.dumps({
                    "status": "error",
                    "submission_id": submission_id,
                    "message": "No compliance_report in graph result.",
                })

        return ResponsesAgentResponse(
            output=[_make_message(output_text)],
            custom_outputs={
                "request_id": request_id,
                "submission_id": submission_id,
            },
        )

    def _error(self, submission_id: str, message: str) -> ResponsesAgentResponse:
        output_text = json.dumps({
            "status": "error",
            "submission_id": submission_id,
            "message": message,
        })
        return ResponsesAgentResponse(output=[_make_message(output_text)])


# ── Register with AgentServer ──
agent = ComplianceResponsesAgent()
set_model(agent)


@invoke()
async def handle_invoke(request):
    return await agent.predict(request)
