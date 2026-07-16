#!/usr/bin/env bash
# One A/B arm of the operator-masking spike: masked SfM -> masked train -> fog gate.
#
# Replicates the pipeline's glomap path (splat_route.py _glomap_sfm_command /
# _sfm_stage_commands, recon 2026-07-11) standalone in an isolated arm dir —
# NEVER touches the original job. Reuses the original 720 crops via symlink.
#
# Usage: bash run_arm.sh <arm_name> <masks_root>
#   masks_root must contain colmap/ (crop-named) and seq/ (sequential-named)
#   from compose_masks.py. Prints "ARM <name> DONE verdict=<V>" or "ARM <name> FAILED".
set -euo pipefail
ARM_NAME="$1"
MASKS="$(cd "$2" && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
GATE="$REPO/tools/splatlab-compute-gate.sh"
if ! "$GATE" --is-contained; then
  exec "$GATE" --run "$0" "$@"
fi
"$GATE" --check || exit $?
SRC=/home/rtoony/projects/splatcli/outputs/3d/splat_7f3d29f3de
ARM="$HERE/arm_$ARM_NAME"
COLMAP4=/home/rtoony/miniconda3/envs/colmap4/bin/colmap

# _subprocess_env replica (splat_route.py:716-732)
export PATH="/home/rtoony/miniconda3/envs/splatops/bin:/home/rtoony/miniconda3/envs/colmap/bin:$PATH"
export PYTHONUNBUFFERED=1
export CPATH="/home/rtoony/miniconda3/envs/splatops/targets/x86_64-linux/include"
export LIBRARY_PATH="/home/rtoony/miniconda3/envs/splatops/targets/x86_64-linux/lib:/home/rtoony/miniconda3/envs/splatops/lib"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$FREE" -lt 8000 ]; then echo "ARM $ARM_NAME FAILED (only ${FREE}MB GPU free)"; exit 1; fi

echo "=== ARM $ARM_NAME start $(date +%T) (masks: $MASKS) ==="
rm -rf "$ARM"
mkdir -p "$ARM/colmap/sparse"
ln -sfn "$SRC/colmap/images" "$ARM/colmap/images"

echo "--- [1/5] feature_extractor (masked) $(date +%T)"
"$COLMAP4" feature_extractor \
  --database_path "$ARM/colmap/database.db" \
  --image_path "$ARM/colmap/images" \
  --ImageReader.single_camera 1 \
  --ImageReader.mask_path "$MASKS/colmap" \
  --FeatureExtraction.use_gpu 1 > "$ARM/colmap/feature.log" 2>&1

echo "--- [2/5] sequential_matcher $(date +%T)"
"$COLMAP4" sequential_matcher \
  --database_path "$ARM/colmap/database.db" \
  --SequentialMatching.overlap 16 \
  --SequentialMatching.loop_detection 1 \
  --FeatureMatching.use_gpu 1 > "$ARM/colmap/matcher.log" 2>&1

echo "--- [3/5] global_mapper $(date +%T)"
"$COLMAP4" global_mapper \
  --database_path "$ARM/colmap/database.db" \
  --image_path "$ARM/colmap/images" \
  --output_path "$ARM/colmap/sparse" > "$ARM/colmap/mapper.log" 2>&1
test -f "$ARM/colmap/sparse/0/cameras.bin" || { echo "ARM $ARM_NAME FAILED (no sparse model)"; exit 1; }
PTS=$(stat -c%s "$ARM/colmap/sparse/0/points3D.bin")
[ "$PTS" -le 8 ] && { echo "ARM $ARM_NAME FAILED (0 3D points)"; exit 1; }

echo "--- [4/5] ns-process-data + mask inject + ns-train $(date +%T)"
ns-process-data images \
  --data "$ARM/colmap/images" \
  --output-dir "$ARM/processed" \
  --skip-colmap \
  --colmap-model-path ../colmap/sparse/0 \
  --num-downscales 3 > "$ARM/process.log" 2>&1
REG=$(python3 -c "import json;print(len(json.load(open('$ARM/processed/transforms.json'))['frames']))")
echo "    registered frames: $REG/720"
python3 "$HERE/inject_masks.py" "$ARM/processed" "$MASKS/seq"
ns-train splatfacto \
  --data "$ARM/processed" \
  --output-dir "$ARM" \
  --max-num-iterations 7000 \
  --viewer.quit-on-train-completion True > "$ARM/train.log" 2>&1

echo "--- [5/5] fog gate $(date +%T)"
CONFIG=$(find "$ARM" -name config.yml -newer "$ARM/processed/transforms.json" | head -1)
[ -n "$CONFIG" ] || { echo "ARM $ARM_NAME FAILED (no trained config)"; exit 1; }
bash "$REPO/backend/health/run_health.sh" "$CONFIG" "$ARM/_health" > "$ARM/health.log" 2>&1
VERDICT=$(python3 -c "import json;d=json.load(open('$ARM/_health/fog.json'));print(d['verdict'],d['summary']['median_shell_frac'],d['summary']['median_spread'])")
echo "ARM $ARM_NAME DONE verdict=$VERDICT registered=$REG/720 $(date +%T)"
