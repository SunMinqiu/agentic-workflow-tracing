#!/usr/bin/env bash

results_owner() {
    printf '%s' "${RESULTS_OWNER:-${SUDO_USER:-${SSH_USER:-${USER:-Minqiu}}}}"
}

default_lustre_results_root() {
    local mount_root owner
    mount_root="${MOUNT_PATH:-/mnt/lustrefs}"
    owner="$(results_owner)"
    printf '%s/%s/pi-ebpf-tracing-handoff/results' "$mount_root" "$owner"
}

require_lustre_base_out() {
    local path="$1"
    local mount_root="${MOUNT_PATH:-/mnt/lustrefs}"
    if [ -z "$path" ]; then
        echo "Error: BASE_OUT is empty" >&2
        exit 1
    fi
    if [[ "$path" != /* ]]; then
        echo "Error: BASE_OUT must be an absolute Lustre path, got: $path" >&2
        echo "       Use BASE_OUT=$(default_lustre_results_root)/<run_id>." >&2
        exit 1
    fi
    case "$path" in
        "$mount_root"/*) ;;
        *)
            echo "Error: BASE_OUT must live under $mount_root, got: $path" >&2
            echo "       Use BASE_OUT=$(default_lustre_results_root)/<run_id>." >&2
            exit 1
            ;;
    esac
    if ! awk -v m="$mount_root" '$2 == m { found = 1 } END { exit found ? 0 : 1 }' /proc/mounts; then
        echo "Error: $mount_root is not mounted. Refusing to write trace output to root/home." >&2
        exit 1
    fi
    mkdir -p "$path"
    if [ "$(df -P "$path" | awk 'NR == 2 { print $6 }')" = "/" ]; then
        echo "Error: BASE_OUT resolves to the root filesystem: $path" >&2
        echo "       Mount Lustre first, then rerun." >&2
        exit 1
    fi
}

# Stop the bcc tracer started as `sudo -E env ... bcc_tracer ... &`.
#
# $1 is the PID of the *sudo wrapper*, not of the tracer itself. sudo does not
# reliably relay a signal to its child when the signal comes from a process in
# its own process group, so `sudo kill -INT $TRACER_PID` can be swallowed and
# the following `wait` then blocks forever. Signal the python child directly,
# and escalate INT -> TERM -> KILL on a bounded timer so a stuck tracer can
# never hang the run. SIGINT first: the tracer drains its perf buffer and
# flushes ebpf_events.log on that path, so KILL would truncate the trace.
stop_tracer() {
    local sudo_pid="$1"
    local grace="${2:-30}"
    [ -n "$sudo_pid" ] || return 0

    # Capture the children before signalling: they are gone once INT lands.
    local kids
    kids="$(pgrep -P "$sudo_pid" 2>/dev/null || true)"

    # "Alive" means the wrapper OR the tracer itself: if sudo were reaped first
    # we would otherwise return while the tracer is still writing the log.
    _stop_tracer_alive() {
        local p
        for p in "$sudo_pid" $kids; do
            kill -0 "$p" >/dev/null 2>&1 && return 0
        done
        return 1
    }

    local sig p waited
    for sig in INT TERM KILL; do
        _stop_tracer_alive || break
        for p in $kids "$sudo_pid"; do
            kill -"$sig" "$p" >/dev/null 2>&1 || true
        done
        waited=0
        while _stop_tracer_alive && [ "$waited" -lt "$grace" ]; do
            sleep 1
            waited=$((waited + 1))
        done
        _stop_tracer_alive || break
        echo "  Warning: tracer still alive ${waited}s after SIG$sig; escalating" >&2
        grace=5
    done

    wait "$sudo_pid" >/dev/null 2>&1 || true
}
