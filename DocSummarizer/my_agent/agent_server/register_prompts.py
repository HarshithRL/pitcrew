"""One-shot registration of the summarizer chat prompt with the MLflow Prompt Registry.

Registers the constant part of the summarizer prompt — system message + two
few-shot (human, assistant) pairs — as a single chat-format prompt. The
per-chunk user turn (``Hierarchy: ...\\n\\n<content>``) is built in code by
``core.prompts.build_human_message`` and is NOT part of the registered
template; that keeps the registered asset stable across invocations and
preserves prompt caching on the constant prefix.

Defaults are wired for this project:
  * UC location: ``pitcrew_databricks_dev.default.pitcrew_summarize``
  * Experiment tag: MLflow experiment id 4348770528290097 (document-summarizer),
    so the registered version is traceable back to the deployed app.

Usage (defaults → UC three-level name)::

    python agent_server/register_prompts.py --alias production

Override any piece::

    python agent_server/register_prompts.py \\
        --catalog pitcrew_databricks_dev \\
        --schema default \\
        --prompt-name pitcrew_summarize \\
        --alias production

Actual linkage to the experiment's Prompts tab happens at runtime when the
deployed agent calls ``mlflow.genai.load_prompt`` — registration only tags
the prompt version with the experiment id so the relationship is discoverable.

Docs:
  * https://mlflow.org/docs/latest/genai/prompt-registry/
  * https://docs.databricks.com/aws/en/mlflow3/genai/prompt-version-mgmt/prompt-registry/
"""

from __future__ import annotations

import argparse
import logging
import sys

import mlflow
from mlflow.genai import register_prompt, set_prompt_alias

from core.prompts import (
    FEW_SHOT_MESSAGES,
    SUMMARIZE_SYSTEM_PROMPT,
    as_text,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Project-specific defaults. Override via CLI flags.
DEFAULT_CATALOG = "pitcrew_databricks_dev"
DEFAULT_SCHEMA = "default"
DEFAULT_PROMPT_NAME = "pitcrew_summarize"
DEFAULT_EXPERIMENT_ID = "4348770528290097"  # document-summarizer


def _build_chat_template() -> list[dict]:
    """Assemble the registered chat template from the in-code source of truth.

    Order: system → (human, ai) × 2. Matches the layout in
    ``summarize_chunk_node`` minus the final per-chunk user turn.
    """
    template: list[dict] = [{"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT}]
    for msg in FEW_SHOT_MESSAGES:
        # LangChain HumanMessage → "user"; AIMessage → "assistant".
        role = "user" if msg.type == "human" else "assistant"
        template.append({"role": role, "content": as_text(msg.content)})
    return template


def _is_uc_name(name: str) -> bool:
    """Three-level (catalog.schema.name) implies Unity Catalog registry."""
    return name.count(".") == 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Register the summarizer chat prompt with the MLflow Prompt Registry."
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Fully-qualified prompt name. If omitted, built from --catalog/--schema/--prompt-name.",
    )
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--prompt-name", default=DEFAULT_PROMPT_NAME)
    parser.add_argument(
        "--alias",
        default="production",
        help="Alias to move to the new version (default: 'production'). Pass '' to skip.",
    )
    parser.add_argument(
        "--commit-message",
        default="Register section-aware summarizer chat prompt (system + 2 few-shot pairs).",
    )
    parser.add_argument(
        "--experiment-id",
        default=DEFAULT_EXPERIMENT_ID,
        help="MLflow experiment id to tag on the prompt version (default: document-summarizer).",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable. Extra tags to attach to this prompt version.",
    )
    parser.add_argument(
        "--tracking-uri",
        default="databricks",
        help="MLflow tracking URI (default: 'databricks').",
    )
    args = parser.parse_args()

    name = args.name or f"{args.catalog}.{args.schema}.{args.prompt_name}"

    tags: dict[str, str] = {
        "app": "pitcrew-document-parser",
        "task": "section_aware_summarization",
        "output_contract": "3-5_bullets_markdown",
    }
    if args.experiment_id:
        # UC prompt tags cannot contain . , - = / : in keys.
        tags["experiment_id"] = args.experiment_id
        tags["experiment_name"] = "document_summarizer"
    for raw in args.tag:
        if "=" not in raw:
            logger.error("Bad --tag %r (expected KEY=VALUE)", raw)
            return 2
        k, v = raw.split("=", 1)
        tags[k] = v

    mlflow.set_tracking_uri(args.tracking_uri)
    if _is_uc_name(name):
        # UC prompt registry — matches UC model registry wiring.
        mlflow.set_registry_uri("databricks-uc")
        logger.info("Registry URI set to databricks-uc (detected three-level name)")

    # Set experiment so any autolog-side tracking during this script run lands there.
    if args.experiment_id:
        try:
            mlflow.set_experiment(experiment_id=args.experiment_id)
            logger.info("Active experiment id=%s", args.experiment_id)
        except Exception as exc:
            logger.warning("set_experiment failed (%s) — continuing", exc)

    template = _build_chat_template()
    logger.info("Registering %s (%d messages)", name, len(template))

    prompt = register_prompt(
        name=name,
        template=template,
        commit_message=args.commit_message,
        tags=tags,
    )

    version = getattr(prompt, "version", None)
    logger.info("Registered %s version=%s", name, version)

    alias = args.alias or None
    if alias and version is not None:
        set_prompt_alias(name=name, alias=alias, version=version)
        logger.info("Alias %r -> version %s", alias, version)

    versioned_uri = f"prompts:/{name}/{version}" if version else None
    alias_uri = f"prompts:/{name}@{alias}" if alias else None

    print()
    print("Load at runtime with:")
    if versioned_uri:
        print(f'  mlflow.genai.load_prompt("{versioned_uri}")')
    if alias_uri:
        print(f'  mlflow.genai.load_prompt("{alias_uri}")')
    print()
    print("For app.yaml (SUMMARIZE_PROMPT_URI):")
    print(f"  {alias_uri or versioned_uri}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
