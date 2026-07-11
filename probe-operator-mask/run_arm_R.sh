#!/usr/bin/env bash
# Arm R — RIG-constrained SfM (the H3 fix): colmap4 panorama_sfm.py on the same
# 90 equirect frames. The rig config forces all virtual cameras of one panorama
# to share a center (zero translation), which the pipeline's unrigged 8-crop
# lane does not (measured same-frame scatter: median 5.1 units vs true step 0.13).
# Then ns-process-data -> ns-train draft -> fog gate. No person/seam masks: one
# variable at a time (panorama_sfm's own per-pixel ownership masks are internal
# to feature extraction, not content masking).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
SRC=/home/rtoony/projects/splatcli/outputs/3d/splat_7f3d29f3de
ARM="$HERE/arm_R"
PANO_PY=/home/rtoony/miniconda3/envs/colmap4/bin/python
PANO_SFM=/home/rtoony/projects/splatcli/outputs/3d/colmap4-src/python/examples/panorama_sfm.py

FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$FREE" -lt 8000 ]; then echo "ARM R FAILED (only ${FREE}MB GPU free)"; exit 1; fi

echo "=== ARM R start $(date +%T) (rig-constrained panorama_sfm, mapper=global) ==="
rm -rf "$ARM"
mkdir -p "$ARM"

echo "--- [1/4] panorama_sfm (render rig + CPU SfM) $(date +%T)"
"$PANO_PY" "$PANO_SFM" \
  --input_image_path "$SRC/colmap/equirect_frames" \
  --output_path "$ARM/pano" \
  --matcher sequential \
  --mapper global > "$ARM/pano_sfm.log" 2>&1
MODEL=$(ls -d "$ARM/pano/sparse"/*/ 2>/dev/null | head -1)
[ -n "$MODEL" ] && [ -f "$MODEL/cameras.bin" ] || { echo "ARM R FAILED (no sparse model)"; exit 1; }
echo "    model: $MODEL"

echo "--- [2/4] ns-process-data $(date +%T)"
export PATH="/home/rtoony/miniconda3/envs/splatops/bin:/home/rtoony/miniconda3/envs/colmap/bin:$PATH"
export PYTHONUNBUFFERED=1
export CPATH="/home/rtoony/miniconda3/envs/splatops/targets/x86_64-linux/include"
export LIBRARY_PATH="/home/rtoony/miniconda3/envs/splatops/targets/x86_64-linux/lib:/home/rtoony/miniconda3/envs/splatops/lib"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
REL_MODEL=$(realpath --relative-to="$ARM/processed" "$MODEL" 2>/dev/null || echo "$MODEL")
mkdir -p "$ARM/processed"
ns-process-data images \
  --data "$ARM/pano/images" \
  --output-dir "$ARM/processed" \
  --skip-colmap \
  --colmap-model-path "$REL_MODEL" \
  --num-downscales 3 > "$ARM/process.log" 2>&1
REG=$(python3 -c "import json;print(len(json.load(open('$ARM/processed/transforms.json'))['frames']))")
echo "    registered frames: $REG"

echo "--- [3/4] ns-train $(date +%T)"
ns-train splatfacto \
  --data "$ARM/processed" \
  --output-dir "$ARM" \
  --max-num-iterations 7000 \
  --viewer.quit-on-train-completion True > "$ARM/train.log" 2>&1

echo "--- [4/4] fog gate $(date +%T)"
CONFIG=$(find "$ARM" -name config.yml | head -1)
[ -n "$CONFIG" ] || { echo "ARM R FAILED (no trained config)"; exit 1; }
bash "$REPO/backend/health/run_health.sh" "$CONFIG" "$ARM/_health" > "$ARM/health.log" 2>&1
VERDICT=$(python3 -c "import json;d=json.load(open('$ARM/_health/fog.json'));print(d['verdict'],d['summary']['median_shell_frac'],d['summary']['median_spread'])")
echo "ARM R DONE verdict=$VERDICT registered=$REG $(date +%T)"
