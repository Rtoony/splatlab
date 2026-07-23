#!/usr/bin/env bash
# P6c acceptance gate — batch isolation + background remainder.
# Read-only. No GPU of its own. Exit 0 = P6c holds.
#
# Checks: (1) toolchain file present; (2) IF a live batch_isolate.json exists
# for the pinned garden scene (built on top of its P6b inventory.json), its
# sanity_sum_ok invariant holds, at least one instance built, and the
# background-removed receipt was produced.
set -uo pipefail

GARDEN=/home/rtoony/projects/splatcli/outputs/3d/splat_32d926d9
MESH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../backend/mesh" && pwd)"
FAILS=0

say() { printf '%s %s\n' "$1" "$2"; }
check() {  # check <name> <cmd...>
  local name="$1"; shift
  if "$@" >/dev/null 2>&1; then say "OK  " "$name"; else say "FAIL" "$name"; FAILS=$((FAILS+1)); fi
}

check "batch_isolate.py present" test -f "$MESH_DIR/batch_isolate.py"

REPORT="$GARDEN/_scene/isolated/batch_isolate.json"
if [ -f "$REPORT" ]; then
  check "garden batch_isolate.json sanity_sum_ok" python3 -c "
import json
r = json.load(open('$REPORT'))
s = r.get('sanity') or {}
assert s.get('sanity_sum_ok') is True, s
built = [i for i in r['instances'] if i['status'] == 'built']
assert len(built) >= 1, 'no instances built'
"
  check "garden background.ply present" test -f "$GARDEN/_scene/isolated/background.ply"
  check "garden background-removed receipt present" bash -c \
    "ls '$GARDEN/_scene/isolated'/receipt_background_cam_*.png >/dev/null 2>&1"
  check "recall_expand flag recorded in report" python3 -c "
import json
r = json.load(open('$REPORT'))
assert 'recall_expand' in r, 'batch_isolate.py predates the recall_expand field'
"
else
  say "SKIP" "batch_isolate.json not built yet (run POST /jobs/{id}/scene/isolate first)"
fi

if [ "$FAILS" -eq 0 ]; then echo "GATE_P6C: PASS"; exit 0; fi
echo "GATE_P6C: FAIL ($FAILS check(s))"; exit 1
