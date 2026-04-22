import os
from dataclasses import dataclass
from databricks_langchain import ChatDatabricks
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

@dataclass(frozen=True)
class Settings:
    # Databricks workspace
    DATABRICKS_HOST: str = os.environ.get("DATABRICKS_HOST")

    # MLflow
    MLFLOW_TRACKING_URI: str = os.environ.get("MLFLOW_TRACKING_URI", "databricks")
    MLFLOW_EXPERIMENT_NAME: str = os.environ.get("MLFLOW_EXPERIMENT_NAME", "")
    MLFLOW_EXPERIMENT_ID: str = os.environ.get("MLFLOW_EXPERIMENT_ID", "")

    # LLM — one Databricks foundation-model endpoint for the runtime graph
    # (compliance_checker, judge, guidance_generator)
    LLM_ENDPOINT_NAME: str = os.environ.get(
        "LLM_ENDPOINT_NAME", "databricks-claude-opus-4-7"
    )

    # Rule-parsing endpoint — startup-only cost. Kept separate because
    # Opus 4.7 inconsistently returns valid JSON / structured tool calls
    # when asked to parse full-length regulatory documents; Sonnet 4.5
    # has parsed both default PDFs (FINRA 2210 + SEC 275) reliably.
    RULE_PARSER_ENDPOINT_NAME: str = os.environ.get(
        "RULE_PARSER_ENDPOINT_NAME", "databricks-claude-sonnet-4-5"
    )

    # Runtime
    ENV: str = os.environ.get("AGENT_ENV", "development")
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")


settings = Settings()


def get_llm(
    endpoint: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatDatabricks:
    """Thin factory over ChatDatabricks.

    Prefer ChatDatabricks over ChatOpenAI for Databricks-hosted endpoints:
    - Uses the App's workspace identity automatically (no SP token plumbing).
    - No AI Gateway URL config needed.
    - Works with any `databricks-*` model endpoint (foundation or custom).

    `temperature` defaults to None — not passed to the endpoint — because
    reasoning models like `databricks-claude-opus-4-7` reject the parameter
    outright (HTTP 400). Callers who need deterministic sampling on a model
    that accepts it can pass `temperature=0.0` explicitly.

    `max_tokens` defaults to None (endpoint default). Parse-heavy callers
    (rule_parser, guidance_generator) should pass a larger value — Opus on
    its default max truncates long JSON responses mid-string.
    """
    kwargs: dict = {"endpoint": endpoint or settings.LLM_ENDPOINT_NAME}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatDatabricks(**kwargs)
