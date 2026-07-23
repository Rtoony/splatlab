#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE="$HERE/splatlab-compute-gate.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Source the functions without running main so parsing can be tested against a
# disposable marker without weakening the production marker path.
source "$GATE"
unset SPLAT_TRAINING_DISABLED_REASON
export SPLAT_COMPUTE_UNLOCK_FILE="$TMP/absent-unlock.json"
export SPLAT_GPU_WATCHER_STATUS_FILE="$TMP/absent-watcher.json"

if maintenance_reason "$TMP/missing" >/dev/null; then
    echo "FAIL: absent marker reported maintenance" >&2
    exit 1
fi
check_gate "$TMP/missing"

printf '%s\n' 'SPLAT_TRAINING_DISABLED_REASON="test hardware hold"' > "$TMP/hold.conf"
[[ "$(maintenance_reason "$TMP/hold.conf")" == "test hardware hold" ]]

future="$(date -u -d '+30 minutes' '+%Y-%m-%dT%H:%M:%SZ')"
cat > "$TMP/unlock.json" <<JSON
{
  "schema": "splatlab.compute-unlock.v1",
  "enabled": true,
  "mode": "supervised",
  "reason": "shell test window",
  "operator": "test",
  "created_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "expires_at": "$future",
  "max_active_jobs": 1
}
JSON
python3 - "$TMP/watcher.json" <<'PY'
import json
import sys
import time

with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(
        {
            "run_success": True,
            "finished_at_epoch": time.time(),
            "fault_counts": {
                "aer_current": 0,
                "aer_previous": 1,
                "aer_severe": 0,
                "gpu_unreadable": 0,
                "platform_fatal": 0,
                "xid": 0,
            },
        },
        handle,
    )
PY
export SPLAT_COMPUTE_UNLOCK_FILE="$TMP/unlock.json"
export SPLAT_GPU_WATCHER_STATUS_FILE="$TMP/watcher.json"
check_gate "$TMP/hold.conf"
"$GATE" --status | grep -Fq 'READY: supervised compute unlock is active'
export SPLAT_COMPUTE_UNLOCK_FILE="$TMP/absent-unlock.json"
export SPLAT_GPU_WATCHER_STATUS_FILE="$TMP/absent-watcher.json"

set +e
check_gate "$TMP/hold.conf" >/dev/null 2>&1
rc=$?
set -e
[[ $rc -eq 75 ]] || {
    echo "FAIL: active marker returned $rc instead of 75" >&2
    exit 1
}

printf '%s\n' '# malformed marker still blocks' > "$TMP/malformed.conf"
[[ "$(maintenance_reason "$TMP/malformed.conf")" == "$DEFAULT_REASON" ]]

ln -s "$TMP/missing-target" "$TMP/dangling.conf"
[[ "$(maintenance_reason "$TMP/dangling.conf")" == "$DEFAULT_REASON" ]]

printf '%s\n' \
    'REDIS_HOST=10.0.0.5' \
    'REDIS_PORT=6380' \
    'REDIS_PASSWORD="test password"' \
    > "$TMP/coord.env"
unset REDIS_HOST REDIS_PORT REDIS_PASSWORD
SPLAT_COORDINATOR_ENV_FILES="$TMP/coord.env" load_coordinator_env
[[ "${REDIS_HOST:-}" == "10.0.0.5" ]]
[[ "${REDIS_PORT:-}" == "6380" ]]
[[ "${REDIS_PASSWORD:-}" == "test password" ]]
unset REDIS_HOST REDIS_PORT REDIS_PASSWORD SPLAT_COORDINATOR_ENV_FILES

SPLAT_TRAINING_DISABLED_REASON="environment hold"
[[ "$(maintenance_reason "$TMP/missing")" == "environment hold" ]]
unset SPLAT_TRAINING_DISABLED_REASON

[[ ! -e "$MAINTENANCE_FILE" && ! -L "$MAINTENANCE_FILE" ]] || {
    echo "FAIL: retired production hardware-maintenance marker is active" >&2
    exit 1
}
"$GATE" --status | grep -Fq 'READY: no SplatLab GPU maintenance marker is active'

grep -Fq 'gpu_command_runner.py' "$GATE"
grep -Fq -- '--vram-mb "$MANUAL_VRAM_MB"' "$GATE"

echo "PASS: SplatLab gate honors explicit holds and routes --run through GPU coordination"
