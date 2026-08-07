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

# The one place a run directory is named: <Workflow>_<what ran>_<timestamp>.
# Every trace script calls this instead of pasting its own prefix, which is how
# GenoMAS runs ended up stamped "phase4" years after that phase ended.
#
#   run_dir_name GenoMAS A_c2_w1        -> GenoMAS_A_c2_w1_20260806_101500
#   run_dir_name GenoMAS A_c2_w1 A_c4_w2 -> GenoMAS_2tasks_20260806_101500
#   RUN_LABEL=fanout run_dir_name GenoMAS ... -> GenoMAS_fanout_20260806_101500
#
# RUN_LABEL is for a study whose name says more than its task list does.
run_dir_name() {
    local workflow="$1"; shift
    local stamp="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
    if [ -n "${RUN_LABEL:-}" ]; then
        printf '%s_%s_%s' "$workflow" "$RUN_LABEL" "$stamp"
    elif [ "$#" -eq 1 ]; then
        printf '%s_%s_%s' "$workflow" "$1" "$stamp"
    elif [ "$#" -eq 0 ]; then
        printf '%s_%s' "$workflow" "$stamp"
    else
        printf '%s_%dtasks_%s' "$workflow" "$#" "$stamp"
    fi
}

# Task names the run will actually execute: the first field of every WORKLOADS
# entry, kept only if RUN_WORKLOADS selects it. Same filter the run loop uses,
# so the directory name cannot disagree with what is inside it.
selected_task_names() {
    local entry name
    # fanout builds its cells from axis variables and defines no WORKLOADS; it
    # names its runs with RUN_LABEL instead
    [ "${#WORKLOADS[@]}" -gt 0 ] 2>/dev/null || return 0
    for entry in "${WORKLOADS[@]}"; do
        name="${entry%%|*}"
        if [ -n "${RUN_WORKLOADS:-}" ]; then
            case " ${RUN_WORKLOADS//,/ } " in
                *" $name "*) ;;
                *) continue ;;
            esac
        fi
        printf '%s\n' "$name"
    done
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

workload_selected() {
    local name="$1"
    local selection="${RUN_WORKLOADS:-}"
    [ -z "$selection" ] && return 0
    local item
    local old_ifs="$IFS"
    IFS=','
    for item in $selection; do
        item="${item#"${item%%[![:space:]]*}"}"
        item="${item%"${item##*[![:space:]]}"}"
        if [ "$item" = "$name" ]; then
            IFS="$old_ifs"
            return 0
        fi
    done
    IFS="$old_ifs"
    return 1
}

validate_workload_selection() {
    [ -z "${RUN_WORKLOADS:-}" ] && return 0
    local requested entry available="" found
    local old_ifs="$IFS"
    for entry in "$@"; do
        available="$available ${entry%%|*}"
    done
    IFS=','
    for requested in $RUN_WORKLOADS; do
        requested="${requested#"${requested%%[![:space:]]*}"}"
        requested="${requested%"${requested##*[![:space:]]}"}"
        found=0
        for entry in "$@"; do
            [ "${entry%%|*}" = "$requested" ] && found=1
        done
        if [ "$found" -ne 1 ]; then
            IFS="$old_ifs"
            echo "Error: unknown workload in RUN_WORKLOADS: $requested" >&2
            echo "Available workloads:$available" >&2
            return 2
        fi
    done
    IFS="$old_ifs"
}

selected_workload_count() {
    local entry count=0
    for entry in "$@"; do
        workload_selected "${entry%%|*}" && count=$((count + 1))
    done
    printf '%s' "$count"
}

wait_for_trace_file() {
    local tracer_pid="$1"
    local trace_file="$2"
    local attempts="${3:-100}"
    local attempt=0
    while [ "$attempt" -lt "$attempts" ]; do
        [ -s "$trace_file" ] && return 0
        kill -0 "$tracer_pid" >/dev/null 2>&1 || return 1
        sleep 0.1
        attempt=$((attempt + 1))
    done
    return 1
}

run_kvcache_report() {
    local python="$1"
    local run_root="$2"
    local log_path="$run_root/kvcache_report.log"
    if ! find "$run_root" -mindepth 2 -maxdepth 2 -name messages.jsonl -print -quit |
        grep -q .; then
        echo "Skipping KV-cache report because no messages.jsonl was captured."
        return 0
    fi
    echo "Generating run-level KV-cache report..."
    "$python" -m agent_io_tracing.analysis.kvcache.report \
        --results "$run_root" --runs . --dump-prefixes >"$log_path" 2>&1
}

return_results_ownership() {
    local path="$1"
    chmod -R a+rX "$path" || true
    if [ -n "${SUDO_UID:-}" ] && [ -n "${SUDO_GID:-}" ]; then
        chown -R "$SUDO_UID:$SUDO_GID" "$path" 2>/dev/null || true
        echo "Returned ownership of $path to ${SUDO_USER:-uid=$SUDO_UID}"
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
