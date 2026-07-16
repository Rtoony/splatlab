#!/usr/bin/env bash
set -euo pipefail

readonly MAINTENANCE_FILE="/home/rtoony/.config/splatlab/gpu-hardware-maintenance.conf"
readonly DEFAULT_REASON="SplatLab GPU hardware maintenance is active."
readonly COORDINATOR_PYTHON="/home/rtoony/projects/splatlab/.venv/bin/python"
readonly COORDINATOR="/home/rtoony/projects/splatlab/backend/gpu_command_runner.py"
readonly MANUAL_VRAM_MB="${SPLAT_GPU_MANUAL_VRAM_MB:-24000}"

maintenance_reason() {
    local marker="${1:-$MAINTENANCE_FILE}"
    local line=""
    local value=""

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
            if reason="$(maintenance_reason)"; then
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
