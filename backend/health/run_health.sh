#!/usr/bin/env bash
# Capture-health fog gate: one subprocess in the langfield-spike env (Py3.11,
# nerfstudio + gsplat). Clone of run_langfield.sh minus the SAM pass — same env
# hardening so gsplat's JIT never sees the splatops conda CUDA headers.
set -euo pipefail
CONFIG="$1"        # nerfstudio checkpoint config.yml
OUTDIR="$2"        # <job_dir>/_health  (fog.json + receipt webps land here)
shift 2
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE="$(cd "$HERE/../.." && pwd)/tools/splatlab-compute-gate.sh"
if ! "$GATE" --is-contained; then
  exec "$GATE" --run "$0" "$@"
fi
"$GATE" --check || exit $?
HF_PY="${SPLAT_HEALTH_PYTHON:-/home/rtoony/miniconda3/envs/langfield-spike/bin/python}"

# The splatlab backend passes its train-stage env (CPATH/LIBRARY_PATH point at the
# splatops conda CUDA headers). Drop those and pin the system CUDA so gsplat's
# JIT recompile can't mismatch (same fix as run_langfield.sh).
unset CPATH LIBRARY_PATH || true
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="/usr/local/cuda/bin:$PATH"

mkdir -p "$OUTDIR"
"$HF_PY" "$HERE/fog_gate.py" "$CONFIG" "$OUTDIR" "$@"
