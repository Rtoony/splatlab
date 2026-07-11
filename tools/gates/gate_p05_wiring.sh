#!/usr/bin/env bash
# Phase-0.5 acceptance gate for the Capture Coach wiring (exit-0 = PASS).
#
# 1. Health-stage bookkeeping tests (CPU-only): failure never fails the job,
#    verdict persists in meta["health"], plan guard + SPLAT_HEALTH_GATE=0.
# 2. Frontend builds with the health contract/badge/card.
# 3. Live smoke: the running :3416 API serves at least one job with a
#    health.fog.verdict, and its first receipt image is fetchable via the new
#    receipt route (auth: Bearer PORTAL_TOKEN, same contract as main.py:81).
#
# Usage: bash tools/gates/gate_p05_wiring.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

echo "[1/3] backend bookkeeping tests"
pytest "$REPO/backend/tests/test_health_stage_bookkeeping.py" -q

echo "[2/3] frontend build"
(cd "$REPO/frontend" && npm run build >/dev/null 2>&1)
echo "  built OK"

echo "[3/3] live smoke against :3416"
[ -n "${PORTAL_TOKEN:-}" ] || eval "$(nexus-inject --group 'Rtoony Portal' --quiet)"
# NOTE: python3 - reads its PROGRAM from stdin, so the status fetch must happen
# inside the script (a curl pipe would be clobbered by the heredoc redirect).
python3 - "$PORTAL_TOKEN" <<'PY'
import json, subprocess, sys

token = sys.argv[1]
status = subprocess.run(
    ["curl", "-sf", "-H", f"Authorization: Bearer {token}",
     "http://127.0.0.1:3416/api/splat/status"],
    capture_output=True, text=True, check=True).stdout
jobs = json.loads(status)["jobs"]
rows = [(j["job_id"], (j.get("health") or {}).get("fog") or {}) for j in jobs]
with_verdict = [(i, f) for i, f in rows if f.get("verdict")]
for i, f in with_verdict:
    print(f"  {i}: {f['verdict']}")
if not with_verdict:
    sys.exit("  FAIL: no job in the live API carries health.fog.verdict")

job_id, fog = with_verdict[0]
receipt = fog["receipts"][0]
url = f"http://127.0.0.1:3416/api/splat/jobs/{job_id}/health/receipt/{receipt}"
ctype = subprocess.run(
    ["curl", "-sf", "-o", "/dev/null", "-w", "%{content_type}",
     "-H", f"Authorization: Bearer {token}", url],
    capture_output=True, text=True, check=True).stdout
assert ctype.startswith("image/"), f"receipt route returned {ctype!r}"
print(f"  receipt route OK ({receipt} -> {ctype})")
PY

echo
echo "GATE P05: PASS"
