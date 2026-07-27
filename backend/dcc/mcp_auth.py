"""Optional bearer-token gate for the Blender MCP endpoint.

Why this exists
---------------
The MCP server's entire trust model was "it binds to 127.0.0.1". That is a real
boundary against the network, but not against the host: every local process,
including anything running as this user, could drive Blender through it —
`open_blender` spawns an attended GUI, and the mutating tools write into job
trees. The typed-tool design means a caller cannot execute arbitrary Python, but
it can still act on any job.

Deliberately opt-in
-------------------
`SPLATLAB_MCP_TOKEN` unset (the default, and what the live unit does today)
leaves behaviour byte-for-byte unchanged: loopback trust only. Setting it turns
on `Authorization: Bearer <token>` for every request. That way enabling auth is
an explicit operator decision that cannot silently break the registered
`splatlab-blender` MCP client.

Kept free of any `mcp` import so it is testable from the app venv; the SDK lives
in an isolated environment.
"""

from __future__ import annotations

import hmac
import json
import os

TOKEN_ENV = "SPLATLAB_MCP_TOKEN"
TOKEN_FILE_ENV = "SPLATLAB_MCP_TOKEN_FILE"
MIN_TOKEN_LENGTH = 16


class MCPAuthConfigError(RuntimeError):
    """Refuse to start rather than serve with a token that is not a secret."""


def configured_token() -> str | None:
    """The token to require, or None for loopback-trust-only.

    `SPLATLAB_MCP_TOKEN_FILE` is honoured first so the value can be read from
    the RAM-only `/dev/shm/nexus-env-*` drop the vault injection writes, keeping
    the zero-disk policy intact. A configured-but-unusable token is a hard
    error: silently falling back to no auth is exactly the failure an operator
    would never notice.
    """
    path = os.environ.get(TOKEN_FILE_ENV, "").strip()
    if path:
        try:
            token = open(path, encoding="utf-8").read().strip()
        except OSError as exc:
            raise MCPAuthConfigError(f"{TOKEN_FILE_ENV}={path} is unreadable: {exc}") from exc
        if not token:
            raise MCPAuthConfigError(f"{TOKEN_FILE_ENV}={path} is empty")
    else:
        token = os.environ.get(TOKEN_ENV, "").strip()
        if not token:
            return None

    if len(token) < MIN_TOKEN_LENGTH:
        raise MCPAuthConfigError(
            f"MCP token is {len(token)} characters; refusing to start with fewer "
            f"than {MIN_TOKEN_LENGTH}. Generate one with `openssl rand -hex 32`."
        )
    return token


def _unauthorized(detail: str):
    body = json.dumps({
        "jsonrpc": "2.0",
        "error": {"code": -32001, "message": f"unauthorized: {detail}"},
        "id": None,
    }).encode()

    async def send_401(send):
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"www-authenticate", b'Bearer realm="splatlab-blender-mcp"'),
            ],
        })
        await send({"type": "http.response.body", "body": body})

    return send_401


def bearer_token_middleware(app, token: str):
    """Wrap an ASGI app so every HTTP request must carry the bearer token.

    Constant-time comparison; no logging of the presented value. Lifespan and
    any non-HTTP scope pass straight through — the gate is on requests, not on
    server startup.
    """

    async def guarded(scope, receive, send):
        if scope.get("type") != "http":
            await app(scope, receive, send)
            return

        presented = ""
        for key, value in scope.get("headers") or []:
            if key.lower() == b"authorization":
                presented = value.decode("latin-1", "replace").strip()
                break

        if not presented:
            await _unauthorized("missing Authorization header")(send)
            return

        prefix, _, candidate = presented.partition(" ")
        if prefix.lower() != "bearer" or not candidate:
            await _unauthorized("expected an Authorization: Bearer header")(send)
            return

        if not hmac.compare_digest(candidate.strip(), token):
            await _unauthorized("bad token")(send)
            return

        await app(scope, receive, send)

    return guarded
