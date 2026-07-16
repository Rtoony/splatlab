#!/usr/bin/env bash
# Language-Field BUILD: export frames -> SAM 2.1 masks -> training-free SigLIP lift.
# Emitted as ONE command so the caller's HEAVY_GPU_LOCK wraps the whole ~7-min build
# a single time (never per-subprocess, or TRELLIS could wedge between the mask + lift
# passes and OOM a half-loaded pipeline). Subprocesses two conda envs by absolute path
# (SAM2.1 = sam2/Py3.10; SigLIP+lift = langfield-spike/Py3.11) — nothing in splatlab's venv.
set -euo pipefail
CONFIG="$1"        # nerfstudio checkpoint config.yml (from the just-trained job)
LFDIR="$2"         # <job_dir>/_langfield  (gauss_emb.npz lands here)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE="$(cd "$HERE/../.." && pwd)/tools/splatlab-compute-gate.sh"
if ! "$GATE" --is-contained; then
  exec "$GATE" --run "$0" "$@"
fi
"$GATE" --check || exit $?
LF_PY="${SPLAT_LANGFIELD_PYTHON:-/home/rtoony/miniconda3/envs/langfield-spike/bin/python}"
SAM_PY="${SPLAT_SAM2_PYTHON:-/home/rtoony/miniconda3/envs/sam2/bin/python}"
FEAT="$LFDIR/features"

# The splatlab backend passes its train-stage env (CPATH/LIBRARY_PATH point at the
# splatops conda CUDA headers). The langfield-spike env drives gsplat here with the
# SYSTEM nvcc, so drop those to avoid a header/nvcc mismatch if gsplat recompiles.
unset CPATH LIBRARY_PATH || true
# R-JIT: gsplat recompiles + fails in a non-interactive subprocess unless CUDA_HOME is set.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="/usr/local/cuda/bin:$PATH"

mkdir -p "$FEAT"
echo "[langfield] export frames ($(date +%T))"
"$LF_PY" "$HERE/export_frames.py" "$CONFIG" "$FEAT"
echo "[langfield] SAM 2.1 masks ($(date +%T))"
"$SAM_PY" "$HERE/sam_masks.py" "$FEAT"
echo "[langfield] training-free lift ($(date +%T))"
"$LF_PY" "$HERE/langfield_v2.py" "$CONFIG" "$FEAT" "$LFDIR"
echo "[langfield] prune scratch features ($(date +%T))"
rm -rf "$FEAT"
echo "[langfield] DONE -> $LFDIR/gauss_emb.npz ($(date +%T))"
