#!/usr/bin/env bash
# P6b acceptance gate — scene instance inventory.
# Read-only. No GPU of its own (sam3_doctor shallow check only). Exit 0 = P6b holds.
#
# Checks: (1) toolchain files present; (2) sam3_doctor shallow preflight healthy;
# (3) noun_consolidate pure-logic sanity (stuff/things split, no heavy-dep import);
# (4) IF a live inventory.json exists for the pinned garden scene, its conservation
# invariant holds and receipts were produced (informative, not required — a fresh
# checkout with no live run yet still passes 1-3).
set -uo pipefail

GARDEN=/home/rtoony/projects/splatcli/outputs/3d/splat_32d926d9
MESH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../backend/mesh" && pwd)"
SAM3_PY=/home/rtoony/miniconda3/envs/sam3/bin/python
APP_PY=/home/rtoony/projects/splatlab/.venv/bin/python
FAILS=0

say() { printf '%s %s\n' "$1" "$2"; }
check() {  # check <name> <cmd...>
  local name="$1"; shift
  if "$@" >/dev/null 2>&1; then say "OK  " "$name"; else say "FAIL" "$name"; FAILS=$((FAILS+1)); fi
}

# (1) toolchain files
for f in scene_views.py noun_consolidate.py scene_sam3_masks.py instance_lift.py sam3_doctor.py; do
  check "$f present" test -f "$MESH_DIR/$f"
done

# (2) sam3 preflight (shallow)
check "sam3_doctor shallow" "$SAM3_PY" "$MESH_DIR/sam3_doctor.py"

# (3) noun_consolidate pure-logic sanity — no heavy deps required to import/call
check "noun_consolidate stuff/things split" "$APP_PY" -c "
import sys; sys.path.insert(0,'$MESH_DIR')
import noun_consolidate as nc
assert nc.classify_stuff('grass lawn')
assert not nc.classify_stuff('fire hydrant')
assert 'torch' not in vars(nc)
report = nc.consolidate(['fire hydrant', 'fire hydrant'], max_nouns=10, dedup_thresh=0.85)
assert report['things'] == ['fire hydrant'] and report['stuff'] == []
"

# (4) informative: live garden inventory, if one has been built
INVENTORY="$GARDEN/_scene/inventory.json"
if [ -f "$INVENTORY" ]; then
  check "garden inventory.json conservation holds" python3 -c "
import json
r = json.load(open('$INVENTORY'))
c = r.get('conservation') or {}
assert c.get('holds') is True, c
assert len(r.get('instances', [])) >= 1
"
  check "garden inventory receipts present" bash -c "ls '$GARDEN/_scene'/receipt_overlay_cam_*.png >/dev/null 2>&1"
else
  say "SKIP" "garden inventory.json not built yet (run POST /jobs/{id}/scene/inventory first)"
fi

if [ "$FAILS" -eq 0 ]; then echo "GATE_P6B: PASS"; exit 0; fi
echo "GATE_P6B: FAIL ($FAILS check(s))"; exit 1
