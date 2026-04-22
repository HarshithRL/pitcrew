# my_agent

Minimal supervisor + subagent LangGraph skeleton, served as a Databricks App via MLflow
`ResponsesAgent` + `AgentServer`.

## Behavior

- **Greeting** (e.g. `"Hi"`, `"Hello"`) — supervisor answers directly with a greeting.
- **Databricks question** (e.g. `"What is Databricks?"`) — supervisor routes to the subagent,
  which plans and produces an answer.

## Layout

```
my_agent/
├── agent_server/
│   ├── agent.py              # ResponsesAgent subclass + @invoke/@stream handlers
│   ├── start_server.py       # AgentServer entry; sets MLFLOW_TRACKING_URI
│   ├── app.yaml              # Databricks App run config (command + env)
│   ├── requirements.txt
│   └── core/
│       ├── __init__.py
│       ├── config.py         # frozen dataclass of env vars + get_llm()
│       ├── state.py          # AgentState TypedDict
│       ├── subagent.py       # compiled LangGraph subagent subgraph
│       └── graph.py          # supervisor StateGraph (subagent added as node)
├── databricks.yml            # DAB bundle (experiment + app resources)
└── README.md
```

## Local run

```bash
cd agent_server
pip install -r requirements.txt
python start_server.py
# then POST to http://localhost:8000/invocations
```

## Deploy

```bash
databricks bundle validate --profile DEFAULT
databricks bundle deploy   --profile DEFAULT
databricks apps deploy my-agent --source-code-path agent_server --profile DEFAULT
```

## Test the deployed endpoint

```bash
curl -X POST "$APP_URL/invocations" \
  -H "Authorization: Bearer $(databricks auth token -p DEFAULT | jq -r .access_token)" \
  -H "Content-Type: application/json" \
  -d '{"input":[{"role":"user","content":"What is Databricks?"}]}'
```
