# SplatLab Blender MCP

This integration is a restricted alternative to the general-purpose Blender MCP
addon. It exposes only typed SplatLab operations and never accepts Python source.

## Safety Contract

- The server binds only to `127.0.0.1` and enables the MCP SDK's localhost DNS
  rebinding protection.
- **Trust model, stated plainly:** loopback binding is a boundary against the
  network, not against this host. By default any local process running as this
  user can drive Blender through the endpoint, including the attended
  `open_blender` GUI spawn. The typed-tool surface means such a caller cannot
  execute arbitrary Python, but it can act on any job under the output root.
  Set `SPLATLAB_MCP_TOKEN` (or `SPLATLAB_MCP_TOKEN_FILE`, pointing at the
  RAM-only `/dev/shm/nexus-env-*` drop) to require
  `Authorization: Bearer <token>` on every request. This is opt-in: unset means
  today's behaviour, unchanged. A token shorter than 16 characters, or a token
  file that is missing or empty, is a startup error rather than a silent
  downgrade to no auth.
- Job ids resolve only under the configured SplatLab output root.
- Blender subprocesses receive an allowlisted environment, not injected vault
  credentials.
- Every mutation creates `_blender/versions/scene-vNNNN.blend` plus a checksum
  receipt. Restore copies an older version forward and never overwrites history.
- Headless commands are limited to inspect, snapshot, collection visibility, and
  absolute object transforms. There is no `exec`, inline Python, URL fetch, or
  arbitrary filesystem tool.

## Run

Install the pinned MCP SDK v2 beta into its isolated environment, then start
either transport. Keeping it separate prevents the beta SDK's Starlette version
from changing SplatLab's FastAPI runtime.

```bash
python3 -m venv integrations/blender/.venv
integrations/blender/.venv/bin/pip install -r integrations/blender/requirements.txt
integrations/blender/run-mcp.sh
integrations/blender/run-mcp.sh --transport stdio
```

The Streamable HTTP endpoint is `http://127.0.0.1:9877/mcp`. No `.env` file is
used or supported.

`base_version` defaults to the latest immutable version. Use `0` to operate from
the assembled `_regen/scene.blend`, or a positive version to branch from that
receipt. `open_blender` is the only attended GUI action.

## Persistent service + launcher (2026-07-26)

The server now runs as a systemd user unit — `splatlab-blender-mcp.service`
(loopback :9877, `Restart=on-failure`, no secrets) — and is registered as the
`splatlab-blender` HTTP MCP server in `~/.claude.json` (user scope), so agent
sessions get the 9 typed tools without any manual start. The general-purpose
:9876 addon bridge (`blender-cockpit`) is a different server; never conflate.

Outbound leg: `~/bin/splatlab-blender <job_id> [--artifact ...]` opens a job's
assets in GPU Blender 4.5 LTS (latest `_blender/versions` blend → the P6
`_regen/scene.blend` → best GLB set, in that order) and prints the matching
polish-upload return path.

Bootstrap convention: the FIRST `snapshot_blend` after a P6 assemble creates
`_blender/versions/scene-v0001.blend`; every mutating op after that lands as a
new immutable version with a receipt, and `export_blend_glb` writes a
validated GLB to `_blender/exports/` ready for the polish-upload route.
