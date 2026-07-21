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

# ── optional DN fine-tune escalation (MESH_FINETUNE=1) ───────────────────────
# The recorded rung for scenes whose VANILLA checkpoint meshes fragmentary
# (proven 2026-07-10: 30.2% -> 65% LCC at room scale): ~3k iters of ags-mesh
# depth-gradient normal supervision on the EXISTING checkpoint, all
# densification/culling OFF, means LR 10x down. ~10-15 min GPU vs ~2 min plain.
EXPORT_CONFIG="$CONFIG"
CKPT_LABEL="vanilla"
if [ "${MESH_FINETUNE:-0}" = "1" ]; then
  FT_ITERS="${MESH_FT_ITERS:-3000}"
  LOAD_DIR="$(dirname "$CONFIG")/nerfstudio_models"
  DATA_DIR="$(dirname "$(dirname "$(dirname "$CONFIG")")")"   # <job>/processed
  [ -d "$LOAD_DIR" ] || { echo "FATAL: no nerfstudio_models beside config" >&2; exit 1; }
  [ -f "$DATA_DIR/transforms.json" ] || { echo "FATAL: $DATA_DIR is not a processed dataset" >&2; exit 1; }
  FT_DIR="$OUTDIR/finetune"
  rm -rf "$FT_DIR"   # self-cleaning: a rerun never resumes a half-done fine-tune
  ns-train ags-mesh \
    --data "$DATA_DIR" \
    --output-dir "$FT_DIR" \
    --experiment-name scene \
    --timestamp ft \
    --load-dir "$LOAD_DIR" \
    --max-num-iterations "$FT_ITERS" \
    --steps-per-save 1000 \
    --pipeline.model.warmup-length 999999 \
    --pipeline.model.stop-split-at 0 \
    --pipeline.model.continue-cull-post-densification False \
    --pipeline.model.normal-supervision depth \
    --optimizers.means.scheduler.lr-final 1.6e-5 \
    --optimizers.means.scheduler.max-steps 1 \
    --viewer.quit-on-train-completion True \
    --vis viewer \
    normal-nerfstudio --load-depths False --load-normals False \
    > "$OUTDIR/finetune.log" 2>&1
  EXPORT_CONFIG="$FT_DIR/scene/ags-mesh/ft/config.yml"
  [ -f "$EXPORT_CONFIG" ] || { echo "FATAL: fine-tune produced no config.yml (see $OUTDIR/finetune.log)" >&2; exit 1; }
  CKPT_LABEL="dn-finetune-$FT_ITERS"
fi

# Champion defaults (garden-proven); env-overridable because TSDF params are
# scene-scale-dependent (mesh-trial finding) — large/open scenes may need
# coarser integration. Overrides are recorded in the report via the recipe blob.
VOXEL="${MESH_VOXEL_SIZE:-0.015}"
SDF_TRUNC="${MESH_SDF_TRUNC:-0.045}"
DEPTH_TRUNC="${MESH_DEPTH_TRUNC:-6}"
gs-mesh o3dtsdf --load-config "$EXPORT_CONFIG" --output-dir "$OUTDIR" \
  --voxel-size "$VOXEL" --sdf-truc "$SDF_TRUNC" --depth-trunc "$DEPTH_TRUNC"

MESH=$(ls "$OUTDIR"/*_mesh.ply 2>/dev/null | head -1)
[ -n "$MESH" ] || { echo "FATAL: gs-mesh produced no *_mesh.ply in $OUTDIR" >&2; exit 1; }

RECIPE="{\"method\":\"o3dtsdf\",\"voxel_size\":$VOXEL,\"sdf_truc\":$SDF_TRUNC,\"depth_trunc\":$DEPTH_TRUNC,\"tsdf_alpha_min\":${TSDF_ALPHA_MIN},\"checkpoint\":\"$CKPT_LABEL\"}"
"$PY" "$HERE/mesh_report.py" "$MESH" "$OUTDIR" --recipe "$RECIPE"
