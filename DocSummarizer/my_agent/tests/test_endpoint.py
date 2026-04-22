"""Integration tests for the deployed pitcrew-agent endpoint.

Required env vars:
  APP_URL               - e.g. https://pitcrew-agent-XXXX.aws.databricksapps.com
  DATABRICKS_APP_TOKEN  - OAuth access token; get via:
                              databricks auth token -p DEFAULT | jq -r .access_token

Run:
  pytest tests/test_endpoint.py -v
"""

import json
import os
import urllib.error
import urllib.request

import pytest


APP_URL = os.environ.get("APP_URL", "").rstrip("/")
TOKEN = os.environ.get("DATABRICKS_APP_TOKEN", "")

pytestmark = pytest.mark.skipif(
    not APP_URL or not TOKEN,
    reason="set APP_URL and DATABRICKS_APP_TOKEN to run endpoint tests",
)


def _invoke(user_text: str, timeout: int = 60) -> dict:
    payload = json.dumps({"input": [{"role": "user", "content": user_text}]}).encode("utf-8")
    req = urllib.request.Request(
        url=f"{APP_URL}/invocations",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        assert resp.status == 200, f"expected 200, got {resp.status}"
        return json.loads(resp.read().decode("utf-8"))


def _extract_text(body: dict) -> str:
    output = body.get("output") or []
    assert output, f"empty output in response: {body}"
    content = output[0].get("content") or []
    assert content, f"empty content in output[0]: {output[0]}"
    return content[0].get("text", "")


def test_greeting_routes_to_greeting_node():
    """'Hi' is classified as greeting → supervisor answers directly (no subagent)."""
    body = _invoke("Hi")
    text = _extract_text(body)
    assert text, "greeting response text was empty"
    assert "Ask me a question about Databricks" in text, (
        f"expected the hardcoded greeting, got: {text!r}"
    )


def test_databricks_question_routes_to_subagent():
    """'What is Databricks?' is classified as subagent → plan → execute."""
    body = _invoke("What is Databricks?")
    text = _extract_text(body)
    assert text, "databricks response text was empty"
    assert "databricks" in text.lower(), (
        f"expected 'Databricks' in subagent answer, got: {text!r}"
    )
    assert len(text) > 50, f"subagent answer suspiciously short: {text!r}"


def test_custom_outputs_include_request_id():
    body = _invoke("Hi")
    custom = body.get("custom_outputs") or {}
    assert "request_id" in custom, f"request_id missing from custom_outputs: {custom}"
