#!/usr/bin/env bash
# COLD per-query relevancy render — the fallback when the warm worker (:3417) is
# down. Renders a 3-view heatmap-overlay strip into <lfdir>/q_<safe>.png. The caller
# holds HEAVY_GPU_LOCK around this (it loads SigLIP + the pipeline = ~20-40s cold).
# Same env discipline as the build wrapper.
set -euo pipefail
CONFIG="$1"      # nerfstudio checkpoint config.yml
LFDIR="$2"       # <job_dir>/_langfield (has gauss_emb.npz)
TEXT="$3"        # the natural-language query
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LF_PY="${SPLAT_LANGFIELD_PYTHON:-/home/rtoony/miniconda3/envs/langfield-spike/bin/python}"
unset CPATH LIBRARY_PATH || true
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="/usr/local/cuda/bin:$PATH"
"$LF_PY" "$HERE/query_render_v2.py" "$CONFIG" "$LFDIR" "$TEXT"
