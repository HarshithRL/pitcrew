# Databricks LangGraph Agent Skeleton Guide

A from-zero reference for building a **lightweight supervisor + subagent** on LangGraph, serving it as a **Databricks App** via MLflow `ResponsesAgent` + `AgentServer`, tracing it in an **MLflow experiment**, and deploying it with a **Databricks Asset Bundle (DAB)**.

Use this as a starting point when scaffolding a new agent project in Databricks. Copy the scaffolds below into fresh files, wire the names together, and you have a runnable agent you can `databricks bundle deploy`. Extend from there.

All patterns below are distilled from the working `comarketer` repo (same workspace). Section 15 lists the exact source files/lines each scaffold is based on, so you can read the production version if a detail surprises you later.

---

## 1. What this skeleton gives you (and what it doesn't)

**You get:**
- Supervisor LangGraph (`classify → subagent → synthesize`) with a compiled **subagent subgraph** added as a node.
- MLflow `ResponsesAgent` wrapper with `predict()` and `predict_stream()`.
- `AgentServer` entry point exposing `POST /invocations` with streaming SSE.
- MLflow tracing: `autolog()`, `set_experiment()`, custom `@mlflow.trace` + `mlflow.start_span` spans.
- `app.yaml` + `databricks.yml` for DAB deploy with an MLflow experiment resource wired in.
- One command each for `bundle deploy` and local run.

**You don't get (deliberately out of scope — add later as needed):**
- No long-term memory (LTM), Lakebase checkpointer, or vector store — skeleton uses `InMemorySaver`.
- No OBO / service-principal per-request auth — skeleton uses the Databricks App's own workspace identity.
- No Genie / MCP tools — subagent nodes are plain LLM calls.
- No SSE anti-buffer middleware — add only when streams exceed the App proxy's ~60s idle timeout.
- No Flask UI mount — `/invocations` is consumed directly by whatever client you wire up.

---

## 2. Dependencies — `agent_server/requirements.txt`

```
mlflow[databricks]>=3.1.3
langgraph>=0.3.0
langchain>=1.0.0
langchain-core>=0.3.0
databricks-langchain>=0.5.0
databricks-sdk>=0.50.0
pydantic>=2.0.0
```

Why each:
- `mlflow[databricks]>=3.1.3` — provides `ResponsesAgent`, `AgentServer`, `@invoke/@stream`, `set_model`. Below 3.1 you fall back to the old `log_model()` pattern.
- `langgraph>=0.3.0` — `StateGraph`, compiled subgraph as a node, checkpointer interface.
- `langchain` / `langchain-core` — `BaseMessage`, `add_messages`, `RunnableConfig`.
- `databricks-langchain` — `ChatDatabricks` (LLM client that uses workspace auth, no gateway URL).
- `databricks-sdk` — workspace client for any `WorkspaceClient()` needs (optional here).
- `pydantic` — input/output models.

---

## 2b. Project metadata — `pyproject.toml` (repo root)

```toml
[project]
name = "my_agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "mlflow[databricks]>=3.1.3",
    "langgraph>=0.3.0",
    "langchain-core>=0.3.0",
    "databricks-langchain>=0.5.0",
    "databricks-sdk>=0.50.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0", "ruff>=0.8.0"]
```

Install locally with `pip install -e ".[dev]"` (editable, plus dev tools). The Databricks App build itself ignores this file — it reads `agent_server/requirements.txt`. See section 3 for why both exist.

---

## 3. Repo layout

```
my_agent/
├── agent_server/
│   ├── agent.py              # ResponsesAgent subclass + @invoke/@stream handlers
│   ├── start_server.py       # AgentServer entry; sets MLFLOW_TRACKING_URI
│   ├── app.yaml              # Databricks App run config (command + env)
│   ├── requirements.txt      # what the Databricks App installs at container build
│   └── core/
│       ├── __init__.py
│       ├── config.py         # frozen dataclass of env vars + get_llm()
│       ├── state.py          # AgentState TypedDict
│       ├── subagent.py       # compiled LangGraph subagent subgraph
│       └── graph.py          # supervisor StateGraph (subagent added as node)
├── databricks.yml            # DAB bundle (experiment + app resources)
├── pyproject.toml            # project metadata + deps for local dev / packaging
└── README.md
```

**Why both `pyproject.toml` and `requirements.txt`?** They coexist intentionally:
- `pyproject.toml` at repo root — canonical project spec (name, version, Python version, deps, dev extras like `pytest`/`ruff`). Used for local dev (`pip install -e .`), IDE integration, CI linting.
- `agent_server/requirements.txt` — what the **Databricks App container** installs at build time. `databricks apps deploy` uploads `source_code_path` (= `./agent_server`) and runs `pip install -r requirements.txt` inside — it does **not** read `pyproject.toml` from the parent directory. Keep the two lists in sync (or generate one from the other via `uv pip compile` / a small script).

Rule of thumb: one responsibility per file, keep files under ~200 lines.

---

## 4. State schema — `agent_server/core/state.py`

```python
from typing import Annotated, Optional, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """Shared state for supervisor and subagent.

    Overlapping keys (messages, subagent_output) let the compiled subagent
    read/write the same state object when added as a node in the parent graph.
    """
    messages: Annotated[list, add_messages]
    user_query: str
    intent: str                     # set by classify_node: "subagent" | "end"
    subagent_output: Optional[dict] # set by subagent, read by synthesize
    response_text: str              # final text returned to the caller
```

Keep the state flat and small. Nested envelopes (`subagent_output`) are fine; avoid deeply nested schemas.

---

## 5. LLM + config — `agent_server/core/config.py`

```python
import os
from dataclasses import dataclass
from databricks_langchain import ChatDatabricks


@dataclass(frozen=True)
class Settings:
    # Databricks workspace
    DATABRICKS_HOST: str = os.environ.get("DATABRICKS_HOST", "")

    # MLflow
    MLFLOW_TRACKING_URI: str = os.environ.get("MLFLOW_TRACKING_URI", "databricks")
    MLFLOW_EXPERIMENT_NAME: str = os.environ.get("MLFLOW_EXPERIMENT_NAME", "")
    MLFLOW_EXPERIMENT_ID: str = os.environ.get("MLFLOW_EXPERIMENT_ID", "")

    # LLM — one Databricks foundation-model endpoint
    LLM_ENDPOINT_NAME: str = os.environ.get(
        "LLM_ENDPOINT_NAME", "databricks-claude-sonnet-4-5"
    )

    # Runtime
    ENV: str = os.environ.get("AGENT_ENV", "development")
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")


settings = Settings()


def get_llm(endpoint: str | None = None, temperature: float = 0.0) -> ChatDatabricks:
    """Thin factory over ChatDatabricks.

    Prefer ChatDatabricks over ChatOpenAI for Databricks-hosted endpoints:
    - Uses the App's workspace identity automatically (no SP token plumbing).
    - No AI Gateway URL config needed.
    - Works with any `databricks-*` model endpoint (foundation or custom).
    """
    return ChatDatabricks(
        endpoint=endpoint or settings.LLM_ENDPOINT_NAME,
        temperature=temperature,
    )
```

Temperature defaults to 0.0 — deterministic for reasoning/classification tasks. Raise it per-call only when you explicitly want variety.

---

## 6. Subagent subgraph — `agent_server/core/subagent.py`

The subagent is a **compiled LangGraph**. The supervisor adds it as a single node via `add_node("subagent", compiled_subagent)`. LangGraph handles state merging at the boundary automatically.

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from core.state import AgentState
from core.config import get_llm


def plan_node(state: AgentState) -> dict:
    """Decide what to do. Output a short plan string."""
    llm = get_llm()
    prompt = [
        SystemMessage(content="You are a task planner. Produce a 1-sentence plan."),
        HumanMessage(content=state["user_query"]),
    ]
    plan = llm.invoke(prompt).content
    return {"subagent_output": {"plan": plan}}


def execute_node(state: AgentState) -> dict:
    """Execute the plan. Here: single LLM answer. In reality: tool calls, retrieval, etc."""
    llm = get_llm()
    plan = state.get("subagent_output", {}).get("plan", "")
    prompt = [
        SystemMessage(content=f"Execute this plan:\n{plan}"),
        HumanMessage(content=state["user_query"]),
    ]
    answer = llm.invoke(prompt).content
    return {"subagent_output": {"plan": plan, "answer": answer}}


def build_subagent():
    g = StateGraph(AgentState)
    g.add_node("plan", plan_node)
    g.add_node("execute", execute_node)
    g.add_edge(START, "plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", END)
    return g.compile()
```

Scale pattern: when the subagent grows, add nodes (retrieve, validate, tool_call) and edges in this file. Keep it self-contained — the supervisor only sees one node.

---

## 7. Supervisor graph — `agent_server/core/graph.py`

```python
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from core.state import AgentState
from core.config import get_llm
from core.subagent import build_subagent


def classify_node(state: AgentState) -> dict:
    """Decide whether to dispatch to subagent or answer directly."""
    llm = get_llm()
    out = llm.invoke([
        SystemMessage(content=(
            "Classify the user's intent. Reply ONLY with one word: "
            "'subagent' if it needs analysis, 'end' if a greeting/smalltalk."
        )),
        HumanMessage(content=state["user_query"]),
    ]).content.strip().lower()
    intent = "subagent" if "subagent" in out else "end"
    return {"intent": intent}


def route(state: AgentState) -> str:
    return state["intent"]  # "subagent" or "end"


def synthesize_node(state: AgentState) -> dict:
    """Turn subagent_output into the final response_text."""
    out = state.get("subagent_output") or {}
    return {"response_text": out.get("answer", "(no answer)")}


def greeting_node(state: AgentState) -> dict:
    return {"response_text": "Hi — ask me an analysis question."}


def build_graph(checkpointer=None):
    subagent = build_subagent()   # compiled subgraph

    g = StateGraph(AgentState)
    g.add_node("classify", classify_node)
    g.add_node("subagent", subagent)       # <- compiled subgraph as a node
    g.add_node("synthesize", synthesize_node)
    g.add_node("greeting", greeting_node)

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route, {
        "subagent": "subagent",
        "end": "greeting",
    })
    g.add_edge("subagent", "synthesize")
    g.add_edge("synthesize", END)
    g.add_edge("greeting", END)

    return g.compile(checkpointer=checkpointer) if checkpointer else g.compile()


_compiled = None

def get_compiled_graph(checkpointer=None):
    """Cache the compiled graph for the lifetime of the process."""
    global _compiled
    if _compiled is None:
        _compiled = build_graph(checkpointer=checkpointer)
    return _compiled
```

---

## 8. ResponsesAgent wrapper — `agent_server/agent.py`

This file is the MLflow contract. Everything above (graph, state, LLM) is plain Python; everything below plugs it into MLflow + `AgentServer`.

```python
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
mlflow.langchain.autolog()
EXPERIMENT_NAME = settings.MLFLOW_EXPERIMENT_NAME
try:
    mlflow.set_experiment(EXPERIMENT_NAME)
except Exception:
    exp_id = settings.MLFLOW_EXPERIMENT_ID
    if exp_id:
        mlflow.set_experiment(experiment_id=exp_id)

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

    @mlflow.trace(name="agent_invoke")
    async def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        request_id = str(uuid.uuid4())
        user_query = _extract_query(request)
        mlflow.update_current_trace(tags={"request_id": request_id, "user_query": user_query[:250]})

        with mlflow.start_span(name="graph_invoke") as span:
            span.set_attributes({"request_id": request_id})
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

    @mlflow.trace(name="agent_stream")
    async def predict_stream(self, request: ResponsesAgentRequest):
        """Minimal streaming: emit one event per graph node update."""
        request_id = str(uuid.uuid4())
        user_query = _extract_query(request)
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
```

**Why `set_model()` and not `mlflow.pyfunc.log_model()`?** With MLflow 3.1+, `AgentServer` + `@invoke/@stream` decorators replace the old log-and-deploy flow. `set_model(agent)` registers the instance with the agent server at import time — the Databricks App executes `start_server.py`, which imports this file, which runs `set_model` as a side effect.

---

## 9. AgentServer entrypoint — `agent_server/start_server.py`

```python
"""Entry point — imports agent (registers handlers) then boots AgentServer."""

import logging
import os

logging.basicConfig(level="INFO", format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

# Must be set BEFORE any mlflow import path in `agent`.
os.environ.setdefault("MLFLOW_TRACKING_URI", "databricks")

import agent  # noqa: F401 — registers @invoke/@stream side-effect

from mlflow.genai.agent_server import AgentServer

agent_server = AgentServer("ResponsesAgent")
app = agent_server.app


def main():
    agent_server.run(app_import_string="start_server:app")


if __name__ == "__main__":
    main()
```

**Add-ons you may want later (all optional):**
- SSE no-buffer ASGI middleware (when streams > ~60s) — see comarketer's `agent_server/start_server.py:26-56`.
- Flask UI mount at `/ui` via `asgiref.wsgi.WsgiToAsgi`.
- `setup_mlflow_git_based_version_tracking()` from `mlflow.genai.agent_server` to tag traces with git SHA.

---

## 10. MLflow tracing & experiment

All three layers come into play automatically once the env vars are set. You do not need any explicit `mlflow.log_model()` call.

**Layer 1 — LangChain autolog.** `mlflow.langchain.autolog()` in `agent.py` instruments every LLM call and every LangGraph node as a span.

**Layer 2 — Method-level trace decorator.** `@mlflow.trace(name="agent_invoke")` on `predict()`/`predict_stream()` wraps the entire request in a named root span. Trace tags via `mlflow.update_current_trace(tags={...})` for filtering in the MLflow UI.

**Layer 3 — Manual spans for sections worth isolating.**

```python
with mlflow.start_span(name="graph_invoke") as span:
    span.set_attributes({"request_id": request_id, "user_query": user_query[:200]})
    result = await graph.ainvoke(...)
```

**Span naming convention.** `snake_case` with optional dot-separated sub-ops, e.g. `graph_invoke`, `checkpoint_load`, `subagent.plan`, `subagent.execute`.

**Experiment routing.**
1. Set `MLFLOW_TRACKING_URI=databricks` (done in `start_server.py` and `app.yaml`).
2. Set `MLFLOW_EXPERIMENT_NAME=/Users/<you>/my_experiment` OR `MLFLOW_EXPERIMENT_ID=<id>`.
3. DAB injects `MLFLOW_EXPERIMENT_ID` automatically when you declare the experiment as an app resource (see section 12).
4. Traces land in the experiment's Traces tab in the Databricks UI.

---

## 11. `agent_server/app.yaml`

```yaml
command:
  - python
  - start_server.py
env:
  - name: MLFLOW_TRACKING_URI
    value: "databricks"
  - name: AGENT_ENV
    value: "production"
  # The mlflow-experiment resource is bound in the App config UI OR by DAB (section 12).
  # Make sure the App service principal has CAN_MANAGE on the experiment.
  - name: MLFLOW_EXPERIMENT_ID
    valueFrom: mlflow-experiment
  - name: MLFLOW_EXPERIMENT_NAME
    value: "/Users/<you>/my_experiment"
  - name: LLM_ENDPOINT_NAME
    value: "databricks-claude-sonnet-4-5"
```

`valueFrom: mlflow-experiment` pulls the experiment ID from the App resource of that name. The resource itself is declared either in `databricks.yml` (preferred — reproducible) or manually in the App config UI.

---

## 12. `databricks.yml` (DAB bundle)

```yaml
bundle:
  name: my_agent

variables:
  username:
    default: "<you>@<company>.com"

workspace:
  host: https://<workspace>.cloud.databricks.com
  profile: DEFAULT

resources:
  experiments:
    my_experiment:
      name: /Users/${var.username}/my_experiment

  apps:
    my_agent:
      name: "my-agent"
      description: "Skeleton supervisor+subagent LangGraph agent"
      source_code_path: ./agent_server
      resources:
        - name: "mlflow-experiment"
          description: "MLflow experiment for agent tracing"
          experiment:
            experiment_id: ${resources.experiments.my_experiment.id}
            permission: "CAN_MANAGE"
      config:
        command:
          - python
          - start_server.py
        env:
          - name: MLFLOW_TRACKING_URI
            value: "databricks"
          - name: MLFLOW_EXPERIMENT_ID
            value: ${resources.experiments.my_experiment.id}
          - name: MLFLOW_EXPERIMENT_NAME
            value: /Users/${var.username}/my_experiment
          - name: AGENT_ENV
            value: "production"
          - name: LLM_ENDPOINT_NAME
            value: "databricks-claude-sonnet-4-5"
```

**Notes:**
- The `experiments.my_experiment` block creates the MLflow experiment as part of the bundle — idempotent.
- The `resources:` block under the app links that experiment with `CAN_MANAGE`, so the App's service principal can write traces.
- Extra secret / SQL-warehouse resources follow the same shape (`- name: ..., secret/sql_warehouse: {...}, permission: ...`). Lakebase (Postgres) is not yet a supported DAB app-resource type — add it through the App config UI after deploy.
- Variables (`${var.username}`) keep the file portable across users/workspaces.

---

## 13. Deployment & local dev

**Local run (uses the same code path the App runs):**
```bash
cd agent_server
pip install -r requirements.txt
python start_server.py
# then POST to http://localhost:8000/invocations
```

**Deploy the bundle + app:**
```bash
databricks bundle validate --profile DEFAULT
databricks bundle deploy   --profile DEFAULT
databricks apps deploy my-agent --source-code-path agent_server --profile DEFAULT
```

`bundle deploy` creates/updates the experiment resource and registers the app spec. `apps deploy` uploads the source code and restarts the container.

**Test the deployed endpoint:**
```bash
curl -X POST "$APP_URL/invocations" \
  -H "Authorization: Bearer $(databricks auth token -p DEFAULT | jq -r .access_token)" \
  -H "Content-Type: application/json" \
  -d '{"input":[{"role":"user","content":"hello"}]}'
```

---

## 14. Verification checklist

- [ ] `python start_server.py` locally — logs show `MLFLOW_TRACKING_URI=databricks`, agent module imports OK, AgentServer boots.
- [ ] Local `POST /invocations` returns a `ResponsesAgentResponse` with `output[0].content[0].text` populated.
- [ ] After `bundle deploy`: MLflow experiment exists at the path you set.
- [ ] After `apps deploy`: App logs (Databricks UI → Compute → Apps → Logs) show the same startup sequence.
- [ ] Hit `/invocations` on the deployed App URL — response returns within a few seconds.
- [ ] Open the MLflow experiment's **Traces** tab — a trace appears with nested spans: `agent_invoke → graph_invoke → classify → subagent → synthesize`.
- [ ] `graph.get_graph().draw_mermaid()` renders the supervisor graph with `subagent` as a nested compiled-graph node (visual sanity check).

If any step fails, the first thing to check is the env vars in `app.yaml` — 80% of Databricks App failures are a missing `MLFLOW_EXPERIMENT_ID` or `MLFLOW_TRACKING_URI`.

---

## 15. Source references (from the `comarketer` repo)

When a pattern above needs more context, read the production version:

| What | File | Lines |
|---|---|---|
| ResponsesAgent subclass, `@mlflow.trace`, trace tags | `agent_server/agent.py` | 293-356 |
| `set_model` + `@invoke`/`@stream` registration | `agent_server/agent.py` | 1432-1478 |
| MLflow experiment setup (name + ID fallback) | `agent_server/agent.py` | 102-136 |
| Checkpointer init (Lakebase → InMemorySaver fallback) | `agent_server/agent.py` | 60-99 |
| AgentServer boot + SSE anti-buffer middleware | `agent_server/start_server.py` | 1-82 |
| StateGraph build + compile | `agent_server/core/graph.py` | 242-289 |
| State TypedDict | `agent_server/core/state.py` | 18-45 |
| Frozen settings dataclass | `agent_server/core/config.py` | 14-61 |
| App run config | `agent_server/app.yaml` | full |
| DAB bundle (experiment + app resources) | `databricks.yml` | full |
| Package pins (Databricks App build) | `agent_server/requirements.txt` | full |
| Project metadata + dev deps (local / CI) | `pyproject.toml` | full |

---

## 16. Preferred-practice cheatsheet

- **One file, one responsibility.** `state.py` holds state; `graph.py` wires nodes; `subagent.py` is its own graph. Resist collapsing into one module.
- **Type-hint every function signature.** Pydantic for any boundary input/output.
- **Temperature 0.0** unless you explicitly need variability.
- **All derived metrics in Python**, never asked-for from the LLM.
- **Never hardcode credentials.** Env vars in `app.yaml`, secrets via App secret resources, never in code.
- **Log, don't print.** `logging` module with `%(asctime)s | %(name)s | %(levelname)s | %(message)s`.
- **ChatDatabricks over ChatOpenAI** when you're calling Databricks-hosted endpoints — skips the AI Gateway URL and per-request SP-token plumbing.
- **Cache the compiled graph.** `_compiled` module-level singleton in `graph.py` — compilation is non-trivial.
- **Async in LangGraph nodes.** Use `asyncio.run(...)` inside a sync node if the underlying IO is async; or make the whole graph async and call `graph.ainvoke`.
- **Trace span names:** `snake_case` with optional dot sub-op (`subagent.plan`, `graph.checkpoint_load`).
- **Before you build**, check whether a helper already exists in the repo — reuse trumps re-implementation.
