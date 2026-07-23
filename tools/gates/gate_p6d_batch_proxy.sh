#!/usr/bin/env bash
# P6d acceptance gate — batch proxy + gated registration.
# Read-only. No GPU of its own. Exit 0 = P6d holds.
#
# Checks: (1) toolchain files present; (2) IF a live batch_proxy.json exists
# for the pinned garden scene, every built element carries registration
# numerics AND the in-file generative PLY tag (P6a provenance rails) —
# missing tag = non-zero, per the plan's explicit gate requirement.
set -uo pipefail

GARDEN=/home/rtoony/projects/splatcli/outputs/3d/splat_32d926d9
MESH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../backend/mesh" && pwd)"
FAILS=0

say() { printf '%s %s\n' "$1" "$2"; }
check() {  # check <name> <cmd...>
  local name="$1"; shift
  if "$@" >/dev/null 2>&1; then say "OK  " "$name"; else say "FAIL" "$name"; FAILS=$((FAILS+1)); fi
}

for f in proxy_triptych.py; do
  check "$f present" test -f "$MESH_DIR/$f"
done

REPORT="$GARDEN/_scene/proxied/batch_proxy.json"
if [ -f "$REPORT" ]; then
  check "garden batch_proxy.json has >=1 built element" python3 -c "
import json
r = json.load(open('$REPORT'))
built = [i for i in r['instances'] if i['status'] == 'built']
assert len(built) >= 1, 'no proxies built'
for e in built:
    assert e.get('icp_fitness') is not None, e
    assert e.get('transform_4x4') is not None, e
"
  check "every built proxy.ply carries the in-file generative tag" python3 -c "
import json, sys
sys.path.insert(0, '$MESH_DIR')
import provenance
r = json.load(open('$REPORT'))
built = [i for i in r['instances'] if i['status'] == 'built']
for e in built:
    p = '$GARDEN/_scene/proxied/' + e['slug'] + '/proxy.ply'
    assert provenance.ply_is_generative(p), f'{p} missing the generative tag'
"
  check "triptych receipts present" bash -c "
for d in '$GARDEN/_scene/proxied'/*/; do
  [ -f \"\${d}triptych.png\" ] || exit 1
done
"
else
  say "SKIP" "batch_proxy.json not built yet (run POST /jobs/{id}/scene/proxy first)"
fi

if [ "$FAILS" -eq 0 ]; then echo "GATE_P6D: PASS"; exit 0; fi
echo "GATE_P6D: FAIL ($FAILS check(s))"; exit 1
