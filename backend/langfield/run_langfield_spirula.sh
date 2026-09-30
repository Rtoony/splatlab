#!/usr/bin/env bash
# Language-Field BUILD for a Raw 360 (Spirula) scene: upright pinhole views cut from the fisheye frames ->
# SAM 2.1 masks -> training-free SigLIP lift onto the published web.ply (PCA 256). Same one-command / one-gate shape
# as run_langfield.sh; the difference is only where geometry and cameras come from (no nerfstudio checkpoint).
set -euo pipefail
JOB="$1"          # <job_dir> (has _spirula/ and _preview/web.ply)
LFDIR="$2"        # <job_dir>/_langfield
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE="$(cd "$HERE/../.." && pwd)/tools/splatlab-compute-gate.sh"
if ! "$GATE" --is-contained; then
  exec "$GATE" --run "$0" "$@"
fi
"$GATE" --check || exit $?
LF_PY="${SPLAT_LANGFIELD_PYTHON:-/home/rtoony/miniconda3/envs/langfield-spike/bin/python}"
SAM_PY="${SPLAT_SAM2_PYTHON:-/home/rtoony/miniconda3/envs/sam2/bin/python}"
FEAT="$LFDIR/features"
unset CPATH LIBRARY_PATH || true
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="/usr/local/cuda/bin:$PATH"
[ -f "$JOB/_preview/web.ply" ] || { echo "[langfield] no _preview/web.ply — nothing to lift onto"; exit 2; }

mkdir -p "$FEAT"
echo "[langfield] upright pinhole views from the fisheye frames ($(date +%T))"
"$LF_PY" "$HERE/spirula_views.py" "$JOB" "$FEAT"
echo "[langfield] SAM 2.1 masks ($(date +%T))"
"$SAM_PY" "$HERE/sam_masks.py" "$FEAT"
echo "[langfield] training-free lift onto web.ply ($(date +%T))"
"$LF_PY" "$HERE/langfield_v2.py" "$JOB" "$FEAT" "$LFDIR"
# What the query worker loads instead of a checkpoint config, and the client's row-identical copy (fmt=langweb):
# the field's rows ARE web.ply's rows, so langweb.ply is a hard link, not a second 200 MB file.
cp "$FEAT/cameras.json" "$LFDIR/cameras.json"
"$LF_PY" - "$JOB" "$LFDIR" <<'PY'
import json, sys
from pathlib import Path
job, lf = Path(sys.argv[1]), Path(sys.argv[2])
(lf / "scene.json").write_text(json.dumps({"kind": "spirula", "job_dir": str(job),
    "ply": str(job / "_preview" / "web.ply"), "cameras": str(lf / "cameras.json")}, indent=2))
PY
ln -f "$JOB/_preview/web.ply" "$JOB/_preview/langweb.ply"
echo "[langfield] prune scratch features ($(date +%T))"
rm -rf "$FEAT"
echo "[langfield] DONE -> $LFDIR/gauss_emb.npz ($(date +%T))"
