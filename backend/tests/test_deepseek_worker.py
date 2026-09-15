import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest


SPEC = importlib.util.spec_from_file_location("deepseek_worker", Path(__file__).resolve().parents[2] / "tools/deepseek-worker.py")
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def test_packet_uses_current_model_without_tools_or_automatic_execution():
    with patch.object(worker.time, "time", return_value=0):
        payload, reserved, limits, digest = worker.prepare("Review a source-to-world similarity transform.", 2048)
    assert payload["model"] == "deepseek-flash"
    assert payload["max_tokens"] == 2048
    assert "tools" not in payload
    assert payload["stream"] is False
    assert payload["thinking"]["type"] == "disabled"
    assert "reasoning_effort" not in payload
    assert limits["thinking"] == "disabled"
    assert reserved > 20000
    assert limits["automatic_retries"] == 0
    assert limits["references_uploaded"] == 0
    assert len(digest) == 64


def test_optional_thinking_is_explicit_recorded_and_bound_into_request_hash():
    with patch.object(worker.time, "time", return_value=0):
        disabled = worker.prepare("Review", 2048)
        enabled = worker.prepare("Review", 2048, "enabled")
    assert enabled[0]["thinking"] == {"type": "enabled"}
    assert enabled[0]["reasoning_effort"] == "low"
    assert enabled[2]["thinking"] == "enabled"
    assert enabled[3] != disabled[3]
    assert enabled[1] >= disabled[1]


@pytest.mark.parametrize("thinking", [None, True, "auto", "none"])
def test_invalid_thinking_mode_refuses(thinking):
    with pytest.raises(worker.BudgetRefused, match="Thinking"):
        worker.prepare("Review", 2048, thinking)


@pytest.mark.parametrize("prompt,tokens", [("", 2048), ("x" * 32769, 2048), ("x", True), ("x", 8193)])
def test_packet_limits(prompt, tokens):
    with patch.object(worker.time, "time", return_value=0), pytest.raises(worker.BudgetRefused):
        worker.prepare(prompt, tokens)


def test_expired_pricing_refuses():
    with patch.object(worker.time, "time", return_value=worker.RATE_EXPIRY), pytest.raises(worker.BudgetRefused):
        worker.prepare("Review this", 2048)


def test_vault_extracts_only_requested_key_without_eval():
    result = type("Result", (), {"returncode": 0, "stdout": "export DEEPSEEK_API_KEY=synthetic-test-only\nexport OTHER_KEY=ignored\nexport NEXUS_INJECT_STATUS=ok\n"})()
    with patch.object(worker.subprocess, "run", return_value=result) as run:
        assert worker.vault_key() == "synthetic-test-only"
    assert run.call_args.args[0][2] == "Deepseek"
    assert run.call_args.kwargs["capture_output"] is True


def test_unknown_endpoint_refuses_before_transport():
    with pytest.raises(worker.WorkerFailure, match="endpoint-not-allowed"):
        worker.request_api("https://unrelated.example/", "synthetic-test-only")


def test_versioned_response_alias_is_recorded_not_discarded():
    answer, observed = worker.extract_response({"model": "versioned-alias", "choices": [{"finish_reason": "stop", "message": {"content": "Proposal", "reasoning_content": "not retained"}}], "usage": {"prompt_tokens": 12, "completion_tokens": 25}})
    assert answer == "Proposal"
    assert observed["requested_model"] == "deepseek-flash"
    assert observed["response_model"] == "versioned-alias"
    assert observed["response_alias_differs"] is True
    assert observed["status"] == "needs-review"
    assert "reasoning_content" not in str(observed)


def test_empty_token_limited_answer_is_recorded_as_incomplete():
    answer, observed = worker.extract_response({"model": "deepseek-flash", "choices": [{"finish_reason": "length", "message": {"content": None}}], "usage": {"completion_tokens": 4096}})
    assert answer == ""
    assert observed["status"] == "incomplete-needs-review"
    assert observed["usage"]["completion_tokens"] == 4096


def test_missing_response_fields_do_not_claim_completed_work():
    answer, observed = worker.extract_response({"choices": [], "usage": None})
    assert answer == ""
    assert observed["status"] == "incomplete-needs-review"
    assert observed["response_model"] is None
