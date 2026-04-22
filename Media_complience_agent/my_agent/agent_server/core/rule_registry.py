"""Two-stage rule loader + thread-safe registry singleton.

Pipeline:
  Stage 1: PDF bytes  → pymupdf4llm markdown   (deterministic, zero LLM)
  Stage 2: Markdown   → list[RuleEntry]        (one LLM call, rule_parser.j2)
  Stage 3: RuleEntry  → injection_string       (deterministic, zero LLM)

Stage 1 is a thin adapter over the same pymupdf4llm pattern used by
`core/nodes.py::ingest_node` — kept consistent so rules and posts parse
identically. Note: default pymupdf4llm margins=(0, 50, 0, 50) already
strip the top/bottom 50pt zones where page headers/footers live, so no
extra flags are needed.

Called at startup for default rules and on every admin /rules/upload.
Startup is wrapped in try/except — the app boots with an empty registry
if the LLM or PDFs are unavailable; admin can upload later.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pymupdf
import pymupdf4llm
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from jsonschema import validate as jsonschema_validate
from langchain_core.messages import HumanMessage

from core.types import RuleEntry, RuleRegistry

logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────
_AGENT_SERVER_DIR = Path(__file__).resolve().parent.parent
_TEMPLATES_DIR = _AGENT_SERVER_DIR / "templates"
_PROMPTS_DIR = _TEMPLATES_DIR / "prompts"
_SCHEMAS_DIR = _TEMPLATES_DIR / "schemas"
_DEFAULT_CONFIGS_DIR = _AGENT_SERVER_DIR / "configs"

_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=select_autoescape(disabled_extensions=("j2",)),
    keep_trailing_newline=True,
)


# ── Stage 1: deterministic PDF extraction ───────────────────────────────────

def extract_markdown_from_bytes(pdf_bytes: bytes) -> str:
    """Convert PDF bytes to LLM-optimized markdown via pymupdf4llm.

    Matches the extraction pattern in `core/nodes.py::ingest_node`. Default
    margins strip running page headers/footers (~50pt top/bottom exclusion).
    """
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        return pymupdf4llm.to_markdown(doc)
    finally:
        doc.close()


# ── Stage 2: LLM rule parser ────────────────────────────────────────────────

def _as_text(content: Any) -> str:
    """Normalize LLM content (str or list[content-block]) to a plain string."""
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


def _strip_fences(text: str) -> str:
    """Strip ```json / ``` code fences if the LLM wrapped the JSON output."""
    t = text.strip()
    if not t.startswith("```"):
        return t
    lines = t.splitlines()
    lines = lines[1:]  # drop opening fence
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _load_severity_config() -> dict:
    path = _DEFAULT_CONFIGS_DIR / "rules" / "severity_weights.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _assign_severity(rule_id: str, is_prohibition: bool, cfg: dict) -> str:
    """Lookup in known_severities, else default by prohibition flag."""
    known = (cfg.get("known_severities") or {})
    if rule_id in known:
        return known[rule_id]
    return "critical" if is_prohibition else "minor"


async def parse_rules_with_llm(
    markdown_text: str,
    jurisdiction: str,
    base_rule_number: str,
    llm_client,
    doc_name: str = "",
) -> list[RuleEntry]:
    """Render rule_parser.j2, call LLM, validate JSON, build RuleEntry list.

    `base_rule_number` and `doc_name` come from registry.yaml (admin upload or
    startup config) — they are threaded into each RuleEntry here so
    `build_injection_string` can emit per-document headers later.
    """
    template = _env.get_template("rule_parser.j2")
    prompt = template.render(
        markdown_text=markdown_text,
        jurisdiction=jurisdiction,
        base_rule_number=base_rule_number,
    )

    response = await llm_client.ainvoke([HumanMessage(content=prompt)])
    raw_text = _strip_fences(_as_text(response.content))

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.error(
            "LLM returned non-JSON for %s. First 500 chars: %s",
            jurisdiction, raw_text[:500],
        )
        raise ValueError(f"rule_parser returned invalid JSON: {e}") from e

    schema_path = _SCHEMAS_DIR / "parsed_rules.schema.json"
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    jsonschema_validate(instance=parsed, schema=schema)

    severity_cfg = _load_severity_config()

    entries: list[RuleEntry] = []
    for p in parsed:
        severity = _assign_severity(
            rule_id=p["rule_id"],
            is_prohibition=p["is_prohibition"],
            cfg=severity_cfg,
        )
        entries.append(
            RuleEntry(
                rule_id=p["rule_id"],
                section=p["section"],
                rule_text=p["rule_text"],
                citation_label=p["citation_label"],
                jurisdiction=jurisdiction,
                severity=severity,
                is_prohibition=p["is_prohibition"],
                is_requirement=p["is_requirement"],
                base_rule_number=base_rule_number,
                doc_name=doc_name,
            )
        )
    return entries


# ── Stage 3: deterministic injection-string builder ─────────────────────────

def build_injection_string(rules: list[RuleEntry]) -> str:
    """Group rules by (jurisdiction, base_rule_number, doc_name) and format
    for LLM prompt injection.

    Header per group:
        === {jurisdiction} Rule {base_rule_number} — {doc_name} ===

    When base_rule_number / doc_name are empty (pre-Fix-2 or unknown source),
    the header degrades gracefully to `=== {jurisdiction} Rules ===`.

    Per-rule format:
        [RULE: <rule_id>] [CITE: <citation_label>]
        [SEVERITY: <severity>]
        <verbatim rule_text>

        (blank line between rules)
    """
    groups: dict[tuple[str, str, str], list[RuleEntry]] = {}
    for r in rules:
        groups.setdefault((r.jurisdiction, r.base_rule_number, r.doc_name), []).append(r)

    parts: list[str] = []
    for (jurisdiction, base_rule_number, doc_name), group in groups.items():
        if base_rule_number and doc_name:
            header = f"=== {jurisdiction} Rule {base_rule_number} — {doc_name} ==="
        elif base_rule_number:
            header = f"=== {jurisdiction} Rule {base_rule_number} ==="
        else:
            header = f"=== {jurisdiction} Rules ==="
        parts.append(header)
        parts.append("")
        for r in group:
            parts.append(f"[RULE: {r.rule_id}] [CITE: {r.citation_label}]")
            parts.append(f"[SEVERITY: {r.severity}]")
            parts.append(r.rule_text)
            parts.append("")
    return "\n".join(parts).rstrip() + "\n"


# ── Registry singleton ──────────────────────────────────────────────────────

_registry: RuleRegistry | None = None
_lock = asyncio.Lock()


def get_current_registry() -> RuleRegistry:
    """Return the current registry, or raise if nothing has been loaded yet."""
    if _registry is None:
        raise RuntimeError(
            "Rule registry not loaded. POST /rules/upload to load rules."
        )
    return _registry


async def reload_registry(
    pdf_bytes: bytes,
    jurisdiction: str,
    base_rule_number: str,
    doc_name: str,
    llm_client,
) -> RuleRegistry:
    """Admin-upload path: parse one PDF end-to-end and replace the registry."""
    markdown = extract_markdown_from_bytes(pdf_bytes)
    entries = await parse_rules_with_llm(
        markdown, jurisdiction, base_rule_number, llm_client, doc_name=doc_name,
    )
    injection = build_injection_string(entries)

    new_registry = RuleRegistry(
        rules=entries,
        loaded_at=datetime.now(),
        source_documents=[doc_name],
        injection_string=injection,
    )

    async with _lock:
        global _registry
        _registry = new_registry

    logger.info(
        "registry reloaded: %d rules from %s (%s)",
        len(entries), doc_name, jurisdiction,
    )
    return new_registry


async def load_default_rules(configs_path: str, llm_client) -> None:
    """Startup path: load every PDF listed in configs/rules/registry.yaml.

    Accumulates rules across all documents into a single combined registry
    (unlike `reload_registry`, which replaces on every admin upload). Wrapped
    in try/except so a missing PDF or LLM failure never blocks app boot.
    """
    try:
        cfg_path = Path(configs_path) / "rules" / "registry.yaml"
        with cfg_path.open("r", encoding="utf-8") as f:
            reg_cfg = yaml.safe_load(f) or {}

        default_rules = reg_cfg.get("default_rules") or []
        if not default_rules:
            logger.warning("registry.yaml has no default_rules; skipping startup load.")
            return

        all_entries: list[RuleEntry] = []
        source_docs: list[str] = []

        for entry in default_rules:
            pdf_path = Path(entry["path"])
            if not pdf_path.is_absolute():
                pdf_path = _AGENT_SERVER_DIR / pdf_path
            if not pdf_path.exists():
                logger.warning(
                    "default rule PDF not found: %s — skipping.", pdf_path
                )
                continue

            with pdf_path.open("rb") as f:
                pdf_bytes = f.read()

            markdown = extract_markdown_from_bytes(pdf_bytes)
            entries = await parse_rules_with_llm(
                markdown,
                entry["jurisdiction"],
                entry["base_rule_number"],
                llm_client,
                doc_name=entry["name"],
            )
            all_entries.extend(entries)
            source_docs.append(entry["name"])
            logger.info(
                "loaded %d rules from %s", len(entries), entry["name"],
            )

        if not all_entries:
            logger.warning(
                "Default rules failed to load. App starting with empty registry. "
                "Use /rules/upload."
            )
            return

        injection = build_injection_string(all_entries)
        new_registry = RuleRegistry(
            rules=all_entries,
            loaded_at=datetime.now(),
            source_documents=source_docs,
            injection_string=injection,
        )

        async with _lock:
            global _registry
            _registry = new_registry

        logger.info(
            "default registry loaded: %d rules across %d documents",
            len(all_entries), len(source_docs),
        )

    except Exception as e:
        logger.warning(
            "Default rules failed to load. App starting with empty registry. "
            "Use /rules/upload. Error: %s",
            e,
        )
