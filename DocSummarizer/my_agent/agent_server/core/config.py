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

    # LLM — one Databricks foundation-model endpoint
    LLM_ENDPOINT_NAME: str = os.environ.get(
        "LLM_ENDPOINT_NAME", "databricks-claude-sonnet-4-5"
    )

    # Runtime
    ENV: str = os.environ.get("AGENT_ENV", "development")
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")


settings = Settings()


def get_llm(
    endpoint: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> ChatDatabricks:
    """Thin factory over ChatDatabricks.

    Prefer ChatDatabricks over ChatOpenAI for Databricks-hosted endpoints:
    - Uses the App's workspace identity automatically (no SP token plumbing).
    - No AI Gateway URL config needed.
    - Works with any `databricks-*` model endpoint (foundation or custom).

    ``max_tokens`` is optional — the summarizer passes a tight cap (e.g. 400)
    to prevent runaway output since the contract is 3–5 bullets.
    """
    kwargs: dict = {
        "endpoint": endpoint or settings.LLM_ENDPOINT_NAME,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatDatabricks(**kwargs)
