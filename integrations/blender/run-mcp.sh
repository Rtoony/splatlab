#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$ROOT/integrations/blender/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  printf 'Blender MCP environment is missing. Run:\n  python3 -m venv integrations/blender/.venv\n  integrations/blender/.venv/bin/pip install -r integrations/blender/requirements.txt\n' >&2
  exit 1
fi

export PYTHONPATH="$ROOT/backend${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m dcc.blender_mcp_server "$@"
