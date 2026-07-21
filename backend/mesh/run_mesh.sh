#!/usr/bin/env bash
# Splat→mesh export: the champion TSDF recipe (mesh-trial program, 2026-07-10)
# productionized as a SplatLab stage. One subprocess chain in the dn-splatter-probe
# env: gs-mesh o3dtsdf on the (vanilla splatfacto) checkpoint, then mesh_report.py
# for stable artifact names + measured metrics + receipt renders.
#
# Usage: run_mesh.sh <config.yml> <outdir>
set -eo pipefail
# Containment FIRST, before any arg handling: the re-exec must pass the ORIGINAL
# "$@" through (a shift before this line silently drops the args on re-exec —
# that exact bug cost the first first-light run).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE="$(cd "$HERE/../.." && pwd)/tools/splatlab-compute-gate.sh"
if ! "$GATE" --is-contained; then
  exec "$GATE" --run "$0" "$@"
fi
"$GATE" --check || exit $?

CONFIG="$1"        # nerfstudio checkpoint config.yml (vanilla splatfacto is fine)
OUTDIR="$2"        # <job_dir>/_mesh (mesh.ply/glb + mesh.json + view_*.png land here)
[ -f "$CONFIG" ] || { echo "FATAL: config not found: $CONFIG" >&2; exit 2; }
[ -n "$OUTDIR" ] || { echo "FATAL: outdir argument missing" >&2; exit 2; }

PROBE_ROOT=/home/rtoony/tools/dn-splatter-probe
REPO=$PROBE_ROOT/dn-splatter
PY=/home/rtoony/miniconda3/envs/dn-splatter-probe/bin/python

# Env recipe proven by mesh_trial.sh for THIS env (gsplat JIT-builds against the
# conda CUDA headers; the extensions cache must stay private to the env — a shared
# py/cu-tagged cache dir is a cross-env fight).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dn-splatter-probe
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
unset CUDA_HOME LIBRARY_PATH || true
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include
export TORCH_EXTENSIONS_DIR=$PROBE_ROOT/.torch_ext_finetune
# Champion constant: TSDF integrates only pixels the splat actually covers
# (alpha >= 0.5) — the alpha-mask patch in the fork's export_mesh.py.
export TSDF_ALPHA_MIN="${TSDF_ALPHA_MIN:-0.5}"

mkdir -p "$OUTDIR"
cd "$REPO"
gs-mesh o3dtsdf --load-config "$CONFIG" --output-dir "$OUTDIR" \
  --voxel-size 0.015 --sdf-truc 0.045 --depth-trunc 6

MESH=$(ls "$OUTDIR"/*_mesh.ply 2>/dev/null | head -1)
[ -n "$MESH" ] || { echo "FATAL: gs-mesh produced no *_mesh.ply in $OUTDIR" >&2; exit 1; }

RECIPE='{"method":"o3dtsdf","voxel_size":0.015,"sdf_truc":0.045,"depth_trunc":6,"tsdf_alpha_min":0.5,"checkpoint":"vanilla"}'
"$PY" "$HERE/mesh_report.py" "$MESH" "$OUTDIR" --recipe "$RECIPE"
