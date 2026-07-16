#!/usr/bin/env bash
# Arm RP — rig poses (from arm R, reused) + person masks in the training loss.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
GATE="$REPO/tools/splatlab-compute-gate.sh"
if ! "$GATE" --is-contained; then
  exec "$GATE" --run "$0" "$@"
fi
"$GATE" --check || exit $?
export PATH="/home/rtoony/miniconda3/envs/splatops/bin:/home/rtoony/miniconda3/envs/colmap/bin:$PATH"
export PYTHONUNBUFFERED=1
export CPATH="/home/rtoony/miniconda3/envs/splatops/targets/x86_64-linux/include"
export LIBRARY_PATH="/home/rtoony/miniconda3/envs/splatops/targets/x86_64-linux/lib:/home/rtoony/miniconda3/envs/splatops/lib"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
echo "=== ARM RP start $(date +%T) (rig poses + person masks, retrain only) ==="
ns-train splatfacto \
  --data "$HERE/arm_RP/processed" \
  --output-dir "$HERE/arm_RP" \
  --max-num-iterations 7000 \
  --viewer.quit-on-train-completion True > "$HERE/arm_RP/train.log" 2>&1
CONFIG=$(find "$HERE/arm_RP" -name config.yml | head -1)
[ -n "$CONFIG" ] || { echo "ARM RP FAILED (no trained config)"; exit 1; }
bash "$REPO/backend/health/run_health.sh" "$CONFIG" "$HERE/arm_RP/_health" > "$HERE/arm_RP/health.log" 2>&1
VERDICT=$(python3 -c "import json;d=json.load(open('$HERE/arm_RP/_health/fog.json'));print(d['verdict'],d['summary']['median_shell_frac'],d['summary']['median_spread'])")
echo "ARM RP DONE verdict=$VERDICT $(date +%T)"
