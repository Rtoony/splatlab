#!/usr/bin/env bash
# P6e acceptance gate — ground + environment.
# Read-only. No GPU of its own. Exit 0 = P6e holds.
#
# Checks: (1) toolchain files present; (2) IF a live ground.report.json exists
# for the pinned garden scene, artifacts exist and ground-coverage clears the
# floor; (3) top+oblique receipts present.
set -uo pipefail

GARDEN=/home/rtoony/projects/splatcli/outputs/3d/splat_32d926d9
MESH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../backend/mesh" && pwd)"
FAILS=0

say() { printf '%s %s\n' "$1" "$2"; }
check() {  # check <name> <cmd...>
  local name="$1"; shift
  if "$@" >/dev/null 2>&1; then say "OK  " "$name"; else say "FAIL" "$name"; FAILS=$((FAILS+1)); fi
}

for f in ground_mesh_build.py ground_mesh_receipt.py; do
  check "$f present" test -f "$MESH_DIR/$f"
done

REPORT="$GARDEN/_scene/ground/ground.report.json"
if [ -f "$REPORT" ]; then
  check "garden ground.report.json clears the coverage floor" python3 -c "
import json
r = json.load(open('$REPORT'))
assert r.get('provenance') == 'ground-derived', r.get('provenance')
assert (r.get('ground_points') or 0) >= 50, r.get('ground_points')
assert (r.get('triangles') or 0) > 0, r.get('triangles')
assert r.get('finish', {}).get('faces', 0) > 0, 'twin_finish produced no faces'
"
  check "ground_mesh.glb present" test -f "$GARDEN/_scene/ground/ground_mesh.glb"
  check "top+oblique receipts present" bash -c "
test -f '$GARDEN/_scene/ground/receipt_top.png' && test -f '$GARDEN/_scene/ground/receipt_oblique.png'
"
else
  say "SKIP" "ground.report.json not built yet (run POST /jobs/{id}/scene/ground first)"
fi

if [ "$FAILS" -eq 0 ]; then echo "GATE_P6E: PASS"; exit 0; fi
echo "GATE_P6E: FAIL ($FAILS check(s))"; exit 1
