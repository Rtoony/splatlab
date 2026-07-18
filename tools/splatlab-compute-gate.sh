#!/usr/bin/env bash
set -euo pipefail

readonly MAINTENANCE_FILE="/home/rtoony/.config/splatlab/gpu-hardware-maintenance.conf"
readonly DEFAULT_UNLOCK_FILE="/home/rtoony/.config/splatlab/gpu-compute-unlock.json"
readonly DEFAULT_WATCHER_STATUS_FILE="/home/rtoony/.local/state/nexus-watchers/gpu_health_watch_status.json"
readonly DEFAULT_REASON="SplatLab GPU hardware maintenance is active."
readonly UNLOCK_SCHEMA="splatlab.compute-unlock.v1"
readonly COORDINATOR_PYTHON="/home/rtoony/projects/splatlab/.venv/bin/python"
readonly COORDINATOR="/home/rtoony/projects/splatlab/backend/gpu_command_runner.py"
readonly MANUAL_VRAM_MB="${SPLAT_GPU_MANUAL_VRAM_MB:-24000}"
readonly DEFAULT_COORDINATOR_ENV_FILES="/dev/shm/nexus-env-splatlab:/dev/shm/nexus-env-splatlab-langfield"

coordinator_env_files() {
    local configured="${SPLAT_COORDINATOR_ENV_FILES:-$DEFAULT_COORDINATOR_ENV_FILES}"
    local env_file=""

    while [[ "$configured" == *:* ]]; do
        env_file="${configured%%:*}"
        [[ -n "$env_file" ]] && printf '%s\n' "$env_file"
        configured="${configured#*:}"
    done
    [[ -n "$configured" ]] && printf '%s\n' "$configured"
}

load_coordinator_env() {
    [[ -n "${REDIS_PASSWORD:-}" ]] && return

    local env_file=""
    local key=""
    local value=""

    while IFS= read -r env_file; do
        [[ -r "$env_file" ]] || continue
        while IFS='=' read -r key value; do
            key="${key#"${key%%[![:space:]]*}"}"
            key="${key%"${key##*[![:space:]]}"}"
            key="${key#export }"
            case "$key" in
                REDIS_HOST|REDIS_PORT|REDIS_PASSWORD)
                    value="${value%$'\r'}"
                    value="${value#"${value%%[![:space:]]*}"}"
                    value="${value%"${value##*[![:space:]]}"}"
                    if [[ ${#value} -ge 2 ]]; then
                        if [[ "${value:0:1}" == '"' && "${value: -1}" == '"' ]] \
                            || [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
                            value="${value:1:${#value}-2}"
                        fi
                    fi
                    export "$key=$value"
                    ;;
            esac
        done < "$env_file"
        [[ -n "${REDIS_PASSWORD:-}" ]] && return
    done < <(coordinator_env_files)
}

unlock_active() {
    local unlock="${SPLAT_COMPUTE_UNLOCK_FILE:-$DEFAULT_UNLOCK_FILE}"
    local watcher="${SPLAT_GPU_WATCHER_STATUS_FILE:-$DEFAULT_WATCHER_STATUS_FILE}"
    /usr/bin/python3 - "$unlock" "$UNLOCK_SCHEMA" "$watcher" <<'PY'
import json
import sys
import time
from datetime import datetime, timezone

path, schema, watcher_path = sys.argv[1:4]
try:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
except (OSError, json.JSONDecodeError, UnicodeError):
    raise SystemExit(1)

if not isinstance(value, dict):
    raise SystemExit(1)
if value.get("schema") != schema or value.get("enabled") is not True:
    raise SystemExit(1)
if value.get("mode") != "supervised" or value.get("max_active_jobs", 1) != 1:
    raise SystemExit(1)
if not isinstance(value.get("reason"), str) or not value["reason"].strip():
    raise SystemExit(1)
if not isinstance(value.get("operator"), str) or not value["operator"].strip():
    raise SystemExit(1)
try:
    expires = datetime.fromisoformat(str(value.get("expires_at", "")).replace("Z", "+00:00"))
except ValueError:
    raise SystemExit(1)
if expires.tzinfo is None:
    raise SystemExit(1)
remaining = (expires.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds()
if not 0 < remaining <= 7200:
    raise SystemExit(1)

try:
    with open(watcher_path, "r", encoding="utf-8") as handle:
        watcher = json.load(handle)
except (OSError, json.JSONDecodeError, UnicodeError):
    raise SystemExit(1)
if not isinstance(watcher, dict) or watcher.get("run_success") is not True:
    raise SystemExit(1)
try:
    age = time.time() - float(watcher.get("finished_at_epoch"))
except (TypeError, ValueError):
    raise SystemExit(1)
if age < 0 or age > 360:
    raise SystemExit(1)
faults = watcher.get("fault_counts")
if not isinstance(faults, dict):
    raise SystemExit(1)
for key in ("gpu_unreadable", "xid", "aer_current", "aer_severe", "platform_fatal"):
    if int(faults.get(key, 0) or 0) > 0:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

maintenance_reason() {
    local marker="${1:-$MAINTENANCE_FILE}"
    local line=""
    local value=""

    if unlock_active; then
        return 1
    fi

    # Marker presence is authoritative and fail-closed, even if its contents
    # are unreadable or malformed.
    if [[ -e "$marker" || -L "$marker" ]]; then
        if [[ -f "$marker" && -r "$marker" ]]; then
            line="$(grep -m1 '^SPLAT_TRAINING_DISABLED_REASON=' "$marker" 2>/dev/null || true)"
            value="${line#*=}"
            value="${value#"${value%%[![:space:]]*}"}"
            value="${value%"${value##*[![:space:]]}"}"
            if [[ ${#value} -ge 2 ]]; then
                if [[ "${value:0:1}" == '"' && "${value: -1}" == '"' ]] \
                    || [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
                    value="${value:1:${#value}-2}"
                fi
            fi
        fi
        printf '%s\n' "${value:-$DEFAULT_REASON}"
        return 0
    fi

    if [[ -n "${SPLAT_TRAINING_DISABLED_REASON:-}" ]]; then
        printf '%s\n' "$SPLAT_TRAINING_DISABLED_REASON"
        return 0
    fi

    return 1
}

check_gate() {
    local marker="${1:-$MAINTENANCE_FILE}"
    local reason=""

    if reason="$(maintenance_reason "$marker")"; then
        printf 'SplatLab compute startup blocked: %s\n' "$reason" >&2
        return 75
    fi
}

is_contained() {
    grep -Eq '(^|/)splatlab[.]slice(/|$)' /proc/self/cgroup
}

run_contained() {
    [[ $# -gt 0 ]] || {
        printf 'Usage: %s --run COMMAND [ARG ...]\n' "$0" >&2
        return 64
    }

    check_gate

    load_coordinator_env
    if is_contained; then
        exec /usr/bin/taskset --cpu-list 8-15 /usr/bin/nice -n 10 \
            "$COORDINATOR_PYTHON" "$COORDINATOR" --vram-mb "$MANUAL_VRAM_MB" -- "$@"
    fi

    # No fallback is permitted: if the user manager cannot create the bounded
    # scope, the GPU command must not run outside the workstation safety limits.
    exec /usr/bin/systemd-run \
        --user \
        --scope \
        --quiet \
        --collect \
        --no-ask-password \
        --same-dir \
        --description="SplatLab bounded manual GPU command" \
        --slice=splatlab.slice \
        --nice=10 \
        --property=CPUAccounting=yes \
        --property=CPUQuota=400% \
        --property=MemoryAccounting=yes \
        --property=MemoryHigh=32G \
        --property=MemoryMax=48G \
        --property=MemorySwapMax=8G \
        --property=TasksAccounting=yes \
        --property=TasksMax=512 \
        -- /usr/bin/taskset --cpu-list 8-15 \
            "$COORDINATOR_PYTHON" "$COORDINATOR" --vram-mb "$MANUAL_VRAM_MB" -- "$@"
}

main() {
    case "${1:---check}" in
        --check)
            [[ $# -le 1 ]] || return 64
            check_gate
            ;;
        --status)
            [[ $# -le 1 ]] || return 64
            local reason=""
            if unlock_active; then
                printf 'READY: supervised compute unlock is active\nmarker: %s\nunlock: %s\n' "$MAINTENANCE_FILE" "${SPLAT_COMPUTE_UNLOCK_FILE:-$DEFAULT_UNLOCK_FILE}"
            elif reason="$(maintenance_reason)"; then
                printf 'BLOCKED: %s\nmarker: %s\n' "$reason" "$MAINTENANCE_FILE"
            else
                printf 'READY: no SplatLab GPU maintenance marker is active\n'
            fi
            ;;
        --is-contained)
            [[ $# -le 1 ]] || return 64
            is_contained
            ;;
        --run)
            shift
            [[ "${1:-}" == "--" ]] && shift
            run_contained "$@"
            ;;
        *)
            printf 'Usage: %s [--check|--status|--is-contained|--run COMMAND [ARG ...]]\n' "$0" >&2
            return 64
            ;;
    esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
