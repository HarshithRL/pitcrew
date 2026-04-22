"""Shared helpers for compliance graph nodes.

Owns: prompt rendering env, JSON response parsing + schema validation,
cached YAML loaders for compliance.yaml and severity_weights.yaml.

Kept module-private (underscore name) — imported by sibling nodes only.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from jsonschema import validate as jsonschema_validate

_AGENT_SERVER_DIR = Path(__file__).resolve().parent.parent.parent
_CONFIGS_DIR = _AGENT_SERVER_DIR / "configs"
_TEMPLATES_DIR = _AGENT_SERVER_DIR / "templates"
_PROMPTS_DIR = _TEMPLATES_DIR / "prompts"
_SCHEMAS_DIR = _TEMPLATES_DIR / "schemas"

_prompt_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=select_autoescape(disabled_extensions=("j2",)),
    keep_trailing_newline=True,
)


@lru_cache(maxsize=1)
def get_compliance_cfg() -> dict:
    """Load configs/compliance.yaml once per process."""
    with (_CONFIGS_DIR / "compliance.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_severity_cfg() -> dict:
    """Load configs/rules/severity_weights.yaml once per process."""
    with (_CONFIGS_DIR / "rules" / "severity_weights.yaml").open(
        "r", encoding="utf-8"
    ) as f:
        return yaml.safe_load(f) or {}


def render_prompt(template_name: str, **ctx) -> str:
    return _prompt_env.get_template(template_name).render(**ctx)


def as_text(content: Any) -> str:
    """Normalize LLM content (str | list[block]) into a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def strip_fences(text: str) -> str:
    """Strip ```json / ``` fences if the LLM wrapped the JSON output."""
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.splitlines()[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_and_validate(raw_text: str, schema_filename: str) -> Any:
    """Strip fences, parse JSON, validate against a schema in templates/schemas/.

    Raises json.JSONDecodeError or jsonschema.ValidationError on failure.
    """
    clean = strip_fences(as_text(raw_text))
    data = json.loads(clean)
    with (_SCHEMAS_DIR / schema_filename).open("r", encoding="utf-8") as f:
        schema = json.load(f)
    jsonschema_validate(instance=data, schema=schema)
    return data
