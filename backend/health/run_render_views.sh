#!/usr/bin/env bash
# Compute-gated wrapper for render_views.py (same shape as run_health.sh).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE="$(cd "$HERE/../.." && pwd)/tools/splatlab-compute-gate.sh"
if ! "$GATE" --is-contained; then
  exec "$GATE" --run "$0" "$@"
fi
"$GATE" --check || exit $?
CONFIG="$1"; OUTDIR="$2"; shift 2
HF_PY="${SPLAT_HEALTH_PYTHON:-/home/rtoony/miniconda3/envs/langfield-spike/bin/python}"
unset CPATH LIBRARY_PATH || true
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="/usr/local/cuda/bin:$PATH"
mkdir -p "$OUTDIR"
"$HF_PY" "$HERE/render_views.py" "$CONFIG" "$OUTDIR" "$@"
