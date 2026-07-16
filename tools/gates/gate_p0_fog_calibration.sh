#!/usr/bin/env bash
# Phase-0 acceptance gate for the Capture Coach fog fingerprint (exit-0 = PASS).
#
# Metric-trust doctrine: the gate asserts ONLY on scenes RToony actually graded —
#   FOG      splat_5177f8d99a, splat_98095cb055   (2026-07-10 root-cause session)
#   HEALTHY  splat_32d926d9 (garden)              ("meshes recognizably", photo receipt)
# Everything else is a report-only row for him to grade from the receipts:
#   splat_192e4223fb (pool) — was assumed HEALTHY (90% registration) but its visual
#     vetting was never done; calibration 2026-07-11 found a textbook fog cocoon
#     (depth pinned at the near plane, structureless RGB). PENDING HIS GRADE.
#   kitchen ffb47186 / bonsai aea04ab3 / counter 6b2e82e5 — unlabeled, likely good.
# REPORT-ONLY: never writes meta.json.
#
# Usage: bash tools/gates/gate_p0_fog_calibration.sh [report_dir]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
GATE="$REPO/tools/splatlab-compute-gate.sh"
if ! "$GATE" --is-contained; then
  exec "$GATE" --run "$0" "$@"
fi
"$GATE" --check || exit $?
REPORT_DIR="${1:-$HOME/reports/$(date +%F)-capture-coach-fog-calibration}"

python3 "$REPO/backend/health/backfill_fog.py" \
  --jobs splat_5177f8d99a,splat_98095cb055,splat_32d926d9,splat_192e4223fb,splat_ffb47186,splat_aea04ab3,splat_6b2e82e5 \
  --expected splat_5177f8d99a=FOG,splat_98095cb055=FOG,splat_32d926d9=HEALTHY \
  --strict --max-runtime 120 \
  --report-dir "$REPORT_DIR"

echo
echo "GATE P0: PASS — calibration report at $REPORT_DIR/index.md"
echo "NOTE: splat_192e4223fb (pool) is ungraded — review its receipts before trusting any run built on it."
