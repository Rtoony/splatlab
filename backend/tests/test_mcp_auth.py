"""Tests for the Blender MCP endpoint's optional bearer-token gate, plus the
first protocol-level test of the MCP server itself.

The unit tests drive the ASGI middleware directly and need nothing but the app
venv. The protocol test at the bottom spawns the real server out of its isolated
SDK environment and speaks JSON-RPC to it, the same way
test_blender_headless_integration gates real-binary work; it skips when that
environment is absent.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dcc import mcp_auth  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MCP_VENV_PYTHON = REPO / "integrations" / "blender" / ".venv" / "bin" / "python"
GOOD_TOKEN = "0123456789abcdef0123456789abcdef"


# ---------------------------------------------------------------------------
# configured_token
# ---------------------------------------------------------------------------

def test_no_token_configured_means_loopback_trust_only(monkeypatch):
    monkeypatch.delenv(mcp_auth.TOKEN_ENV, raising=False)
    monkeypatch.delenv(mcp_auth.TOKEN_FILE_ENV, raising=False)
    assert mcp_auth.configured_token() is None


def test_blank_token_is_treated_as_unset(monkeypatch):
    monkeypatch.delenv(mcp_auth.TOKEN_FILE_ENV, raising=False)
    monkeypatch.setenv(mcp_auth.TOKEN_ENV, "   ")
    assert mcp_auth.configured_token() is None


def test_token_from_env_is_returned(monkeypatch):
    monkeypatch.delenv(mcp_auth.TOKEN_FILE_ENV, raising=False)
    monkeypatch.setenv(mcp_auth.TOKEN_ENV, f"  {GOOD_TOKEN}  ")
    assert mcp_auth.configured_token() == GOOD_TOKEN


def test_token_file_wins_and_keeps_the_secret_off_disk(tmp_path, monkeypatch):
    """The file form exists so the value can come from /dev/shm/nexus-env-*."""
    drop = tmp_path / "nexus-env-splatlab-mcp"
    drop.write_text(GOOD_TOKEN + "\n")
    monkeypatch.setenv(mcp_auth.TOKEN_ENV, "ignored-because-file-wins")
    monkeypatch.setenv(mcp_auth.TOKEN_FILE_ENV, str(drop))

    assert mcp_auth.configured_token() == GOOD_TOKEN


def test_short_token_is_refused_not_silently_accepted(monkeypatch):
    monkeypatch.delenv(mcp_auth.TOKEN_FILE_ENV, raising=False)
    monkeypatch.setenv(mcp_auth.TOKEN_ENV, "hunter2")
    with pytest.raises(mcp_auth.MCPAuthConfigError, match="refusing to start"):
        mcp_auth.configured_token()


def test_unreadable_token_file_is_fatal_not_a_downgrade(tmp_path, monkeypatch):
    """A configured-but-broken token must never fall back to no auth."""
    monkeypatch.delenv(mcp_auth.TOKEN_ENV, raising=False)
    monkeypatch.setenv(mcp_auth.TOKEN_FILE_ENV, str(tmp_path / "absent"))
    with pytest.raises(mcp_auth.MCPAuthConfigError, match="unreadable"):
        mcp_auth.configured_token()


def test_empty_token_file_is_fatal(tmp_path, monkeypatch):
    drop = tmp_path / "empty"
    drop.write_text("\n")
    monkeypatch.delenv(mcp_auth.TOKEN_ENV, raising=False)
    monkeypatch.setenv(mcp_auth.TOKEN_FILE_ENV, str(drop))
    with pytest.raises(mcp_auth.MCPAuthConfigError, match="is empty"):
        mcp_auth.configured_token()


# ---------------------------------------------------------------------------
# bearer_token_middleware — driven as a real ASGI app
# ---------------------------------------------------------------------------

class _Recorder:
    """Minimal ASGI app that records whether it was ever reached."""

    def __init__(self):
        self.calls = 0

    async def __call__(self, scope, receive, send):
        self.calls += 1
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b'{"ok":true}'})


def _call(app, headers=None, scope_type="http"):
    import asyncio

    sent = []
    scope = {"type": scope_type, "method": "POST", "path": "/mcp",
             "headers": [(k.encode(), v.encode()) for k, v in (headers or {}).items()]}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    return sent


def _status(sent):
    return next(m["status"] for m in sent if m["type"] == "http.response.start")


def test_request_without_authorization_is_rejected():
    inner = _Recorder()
    sent = _call(mcp_auth.bearer_token_middleware(inner, GOOD_TOKEN))

    assert _status(sent) == 401
    assert inner.calls == 0, "the tool layer must never be reached"


def test_wrong_token_is_rejected():
    inner = _Recorder()
    sent = _call(mcp_auth.bearer_token_middleware(inner, GOOD_TOKEN),
                 {"authorization": "Bearer " + "f" * 32})

    assert _status(sent) == 401
    assert inner.calls == 0


def test_non_bearer_scheme_is_rejected():
    inner = _Recorder()
    sent = _call(mcp_auth.bearer_token_middleware(inner, GOOD_TOKEN),
                 {"authorization": "Basic " + GOOD_TOKEN})

    assert _status(sent) == 401
    assert inner.calls == 0


def test_token_as_a_prefix_is_rejected():
    """Guards against any accidental startswith-style comparison."""
    inner = _Recorder()
    sent = _call(mcp_auth.bearer_token_middleware(inner, GOOD_TOKEN),
                 {"authorization": f"Bearer {GOOD_TOKEN[:20]}"})

    assert _status(sent) == 401
    assert inner.calls == 0


def test_correct_token_passes_through():
    inner = _Recorder()
    sent = _call(mcp_auth.bearer_token_middleware(inner, GOOD_TOKEN),
                 {"authorization": f"Bearer {GOOD_TOKEN}"})

    assert _status(sent) == 200
    assert inner.calls == 1


def test_bearer_scheme_is_case_insensitive():
    inner = _Recorder()
    sent = _call(mcp_auth.bearer_token_middleware(inner, GOOD_TOKEN),
                 {"authorization": f"bearer {GOOD_TOKEN}"})

    assert _status(sent) == 200


def test_rejection_body_is_a_jsonrpc_error_and_never_echoes_the_token():
    sent = _call(mcp_auth.bearer_token_middleware(_Recorder(), GOOD_TOKEN),
                 {"authorization": "Bearer sekrit-value-that-must-not-leak"})
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    payload = json.loads(body)

    assert payload["jsonrpc"] == "2.0"
    assert payload["error"]["code"] == -32001
    assert b"sekrit-value-that-must-not-leak" not in body


def test_rejection_advertises_the_scheme():
    sent = _call(mcp_auth.bearer_token_middleware(_Recorder(), GOOD_TOKEN))
    headers = dict(next(m for m in sent if m["type"] == "http.response.start")["headers"])
    assert headers[b"www-authenticate"].startswith(b"Bearer")


def test_lifespan_scope_is_not_gated():
    """Startup must not be blocked by an auth check meant for requests."""
    inner = _Recorder()
    sent = _call(mcp_auth.bearer_token_middleware(inner, GOOD_TOKEN),
                 scope_type="lifespan")
    assert inner.calls == 1
    assert not any(m["type"] == "http.response.start" and m["status"] == 401 for m in sent)


# ---------------------------------------------------------------------------
# Protocol-level test against the real server
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _rpc(url, payload, token=None):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"content-type": "application/json",
                 "accept": "application/json, text/event-stream",
                 **({"authorization": f"Bearer {token}"} if token else {})})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read())


def _spawn(tmp_path, port, token=None):
    env = {**os.environ,
           "PYTHONPATH": str(REPO / "backend"),
           "SPLATLAB_OUTPUT_ROOT": str(tmp_path)}
    env.pop("SPLATLAB_MCP_TOKEN", None)
    env.pop("SPLATLAB_MCP_TOKEN_FILE", None)
    if token:
        env["SPLATLAB_MCP_TOKEN"] = token
    proc = subprocess.Popen(
        [str(MCP_VENV_PYTHON), "-m", "dcc.blender_mcp_server", "--port", str(port)],
        env=env, cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    deadline = time.time() + 30
    while time.time() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"server exited early:\n{proc.stdout.read().decode()}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return proc
        except OSError:
            time.sleep(0.2)
    proc.kill()
    pytest.fail("MCP server never started listening")


def _stop(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.mark.skipif(not MCP_VENV_PYTHON.exists(),
                    reason="Blender MCP SDK venv not installed")
def test_default_configuration_serves_without_auth_exactly_as_before(tmp_path):
    """The live unit sets no token. This pins that adding the gate did not
    change the deployed behaviour: unauthenticated calls still work."""
    port = _free_port()
    proc = _spawn(tmp_path, port)
    try:
        listed = _rpc(f"http://127.0.0.1:{port}/mcp",
                      {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
        assert len(listed["result"]["tools"]) == 12
    finally:
        _stop(proc)


@pytest.mark.skipif(not MCP_VENV_PYTHON.exists(),
                    reason="Blender MCP SDK venv not installed")
def test_mcp_protocol_handshake_list_tools_and_inspect_job(tmp_path):
    """initialize -> tools/list -> tools/call inspect_job, with auth enabled,
    against a throwaway output root. Never touches a real job tree."""
    job_id = "splat_protocoltest"
    job = tmp_path / job_id
    (job / "_regen").mkdir(parents=True)
    (job / "meta.json").write_text(json.dumps(
        {"id": job_id, "status": "completed", "meters_per_unit": 0.5}))
    # inspect_job resolves the assembled blend; its bytes are never parsed here.
    (job / "_regen" / "scene.blend").write_bytes(b"BLENDER-v450 placeholder")
    (job / "dimensions.json").write_text(json.dumps(
        [{"id": "d1", "label": "door", "length": 2.0}]))

    port = _free_port()
    env = {**os.environ,
           "PYTHONPATH": str(REPO / "backend"),
           "SPLATLAB_OUTPUT_ROOT": str(tmp_path),
           "SPLATLAB_MCP_TOKEN": GOOD_TOKEN}
    proc = subprocess.Popen(
        [str(MCP_VENV_PYTHON), "-m", "dcc.blender_mcp_server", "--port", str(port)],
        env=env, cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}/mcp"

    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.fail(f"server exited early:\n{proc.stdout.read().decode()}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.2)
        else:
            pytest.fail("MCP server never started listening")

        # Auth is actually enforced on the wire.
        with pytest.raises(urllib.error.HTTPError) as unauth:
            _rpc(url, {"jsonrpc": "2.0", "id": 0, "method": "tools/list", "params": {}})
        assert unauth.value.code == 401

        init = _rpc(url, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18",
                       "capabilities": {},
                       "clientInfo": {"name": "splatlab-tests", "version": "0"}},
        }, token=GOOD_TOKEN)
        assert init["result"]["serverInfo"]["name"] == "SplatLab Blender"

        listed = _rpc(url, {"jsonrpc": "2.0", "id": 2, "method": "tools/list",
                            "params": {}}, token=GOOD_TOKEN)
        names = {t["name"] for t in listed["result"]["tools"]}
        assert {"inspect_job", "inspect_blend", "list_blender_versions",
                "snapshot_blend", "toggle_collection", "transform_object",
                "import_world_element", "cleanup_mesh", "run_polish_recipe",
                "export_blend_glb", "restore_blender_version",
                "open_blender"} == names, "the 12 typed tools are the whole surface"

        called = _rpc(url, {
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "inspect_job", "arguments": {"job_id": job_id}},
        }, token=GOOD_TOKEN)
        assert called["result"]["isError"] is False, called
        payload = called["result"].get("structuredContent") or json.loads(
            called["result"]["content"][0]["text"])
        assert payload["job_id"] == job_id

        # Containment still holds over the protocol, not just in unit tests.
        escaped = _rpc(url, {
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "inspect_job", "arguments": {"job_id": "../../etc"}},
        }, token=GOOD_TOKEN)
        assert escaped["result"]["isError"] is True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
