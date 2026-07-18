#!/usr/bin/env bash
# gate_robustness_wave.sh — executable acceptance gate for the 2026-07-18
# robustness wave: glomap photo default-flip, escalation surfacing, optional-
# stage bookkeeping, Capture Coach Phase 1 (probe) + Phase 2 (precheck).
#
# Run BARE and check $? — never pipe through tail (masks the exit code).
set -uo pipefail

REPO="/home/rtoony/projects/splatlab"
PYTEST="$HOME/.local/bin/pytest"
FAIL=0

step() { printf '\n== %s ==\n' "$1"; }

step "wave test files"
"$PYTEST" -q \
    "$REPO/backend/tests/test_photo_glomap_flip.py" \
    "$REPO/backend/tests/test_escalation_surfacing.py" \
    "$REPO/backend/tests/test_langfield_stage_bookkeeping.py" \
    "$REPO/backend/tests/test_compress_webopt_stage_bookkeeping.py" \
    "$REPO/backend/tests/test_health_stage_bookkeeping.py" \
    "$REPO/backend/tests/test_capture_probe.py" \
    "$REPO/backend/tests/test_capture_precheck.py" \
    || FAIL=1

step "full backend suite (no regressions)"
"$PYTEST" -q "$REPO/backend/tests/" || FAIL=1

step "frontend typecheck at the recorded baseline (23 pre-existing errors)"
TSC_ERRORS="$(cd "$REPO/frontend" && npx tsc --noEmit 2>&1 | grep -c 'error TS')"
echo "tsc errors: $TSC_ERRORS (baseline 23)"
[[ "$TSC_ERRORS" -le 23 ]] || FAIL=1

step "frontend build"
(cd "$REPO/frontend" && npm run build >/dev/null 2>&1) && echo "build OK" || { echo "build FAILED"; FAIL=1; }

step "verdict"
if [[ "$FAIL" -eq 0 ]]; then
    echo "GATE: PASS"
else
    echo "GATE: FAIL"
fi
exit "$FAIL"
