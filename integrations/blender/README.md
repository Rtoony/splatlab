# SplatLab Blender MCP

This integration is a restricted alternative to the general-purpose Blender MCP
addon. It exposes only typed SplatLab operations and never accepts Python source.

## Safety Contract

- The server binds only to `127.0.0.1` and enables the MCP SDK's localhost DNS
  rebinding protection.
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
