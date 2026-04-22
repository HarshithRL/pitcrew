"""Agent entry point — ResponsesAgent subclass registered via set_model()."""

import logging
import os
import uuid

import mlflow
import mlflow.langchain
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from mlflow.genai.agent_server import invoke, stream
from mlflow.models import set_model
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from core.config import settings
from core.graph import get_compiled_graph

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

# ── MLflow setup (runs on import) ──
_tracking_uri = settings.MLFLOW_TRACKING_URI or "databricks"
mlflow.set_tracking_uri(_tracking_uri)
logger.info("MLflow tracking_uri=%s", mlflow.get_tracking_uri())

# Databricks App env vars — log so we know auth context exists.
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
            "set_experiment by id=%s OK → resolved name=%r id=%s",
            EXPERIMENT_ID, _active_experiment.name, _active_experiment.experiment_id,
        )
    except Exception as e:
        logger.exception("set_experiment by id=%s FAILED: %s", EXPERIMENT_ID, e)

if _active_experiment is None and EXPERIMENT_NAME:
    try:
        _active_experiment = mlflow.set_experiment(EXPERIMENT_NAME)
        logger.info(
            "set_experiment by name=%r OK → resolved id=%s",
            EXPERIMENT_NAME, _active_experiment.experiment_id,
        )
    except Exception as e:
        logger.exception("set_experiment by name=%r FAILED: %s", EXPERIMENT_NAME, e)

if _active_experiment is None:
    logger.error(
        "NO MLflow experiment is active (name=%r id=%r). Traces WILL NOT reach the intended experiment.",
        EXPERIMENT_NAME, EXPERIMENT_ID,
    )

# ── Skeleton checkpointer (swap for AsyncCheckpointSaver/Lakebase in prod) ──
_checkpointer = InMemorySaver()


def _extract_query(req: ResponsesAgentRequest) -> str:
    for msg in reversed(req.input or []):
        if getattr(msg, "role", None) == "user":
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
            return content or ""
    return ""


class MyAgent(ResponsesAgent):
    """Routes /invocations to predict() or predict_stream()."""

    async def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        request_id = str(uuid.uuid4())
        user_query = _extract_query(request)
        mlflow.update_current_trace(tags={"request_id": request_id, "user_query": user_query[:250]})

        graph = get_compiled_graph(checkpointer=_checkpointer)
        result = await graph.ainvoke(
            {"user_query": user_query, "messages": [HumanMessage(content=user_query)]},
            config={"configurable": {"thread_id": request_id}},
        )

        text = result.get("response_text", "")
        return ResponsesAgentResponse(
            output=[{
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex[:12]}",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }],
            custom_outputs={"request_id": request_id},
        )

    async def predict_stream(self, request: ResponsesAgentRequest):
        """Minimal streaming: emit one event per graph node update."""
        request_id = str(uuid.uuid4())
        user_query = _extract_query(request)
        mlflow.update_current_trace(tags={"request_id": request_id, "user_query": user_query[:250]})
        graph = get_compiled_graph(checkpointer=_checkpointer)

        async for chunk in graph.astream(
            {"user_query": user_query, "messages": [HumanMessage(content=user_query)]},
            config={"configurable": {"thread_id": request_id}},
            stream_mode="updates",
        ):
            for node_name, update in chunk.items():
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item_id=f"{node_name}_{uuid.uuid4().hex[:8]}",
                    output_index=0,
                    item={
                        "type": "message",
                        "id": f"msg_{uuid.uuid4().hex[:12]}",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": str(update)[:500]}],
                    },
                    custom_outputs={"node": node_name, "request_id": request_id},
                )


# ── Register with AgentServer ──
agent = MyAgent()
set_model(agent)


@invoke()
async def handle_invoke(request):
    return await agent.predict(request)


@stream()
async def handle_stream(request):
    async for chunk in agent.predict_stream(request):
        yield chunk
