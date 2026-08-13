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
    local stamp="${RUN_STAMP:-$(env TZ=America/New_York date +%Y%m%d_%H%M%S)}"
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

# Start every vLLM-backed cell with an independent prefix cache. Without this,
# a task can reuse prefixes loaded by the task before it and the comparison no
# longer measures that task alone. Provider-managed APIs have no reset endpoint
# and leave VLLM_URL unset, so this is a no-op for them.
prepare_vllm_cache_for_cell() {
    local base="${VLLM_URL:-}"
    if [ -z "$base" ]; then
        CACHE_STATE="provider_managed"
        export CACHE_STATE
        return 0
    fi
    base="${base%/}"
    base="${base%/v1}"
    case "${VLLM_KEEP_PREFIX_CACHE:-0}" in
        1|true|TRUE|yes|YES)
            CACHE_STATE="warm_inherited"
            export CACHE_STATE
            echo "  Prefix cache: keeping existing vLLM state" >&2
            return 0
            ;;
    esac
    if ! curl -fsS --connect-timeout 3 --max-time 30 \
        -X POST "$base/reset_prefix_cache" >/dev/null; then
        echo "Error: could not reset $base/reset_prefix_cache." >&2
        echo "Start vLLM with VLLM_SERVER_DEV_MODE=1, or set" \
             "VLLM_KEEP_PREFIX_CACHE=1 for an intentional warm run." >&2
        return 1
    fi
    CACHE_STATE="cold_by_reset"
    export CACHE_STATE
    echo "  Prefix cache: reset before cell" >&2
}

# Some vLLM responses omit usage.prompt_tokens_details.cached_tokens even when
# --enable-prompt-tokens-details is set. Prometheus counters provide the
# fallback. Snapshot them around a cell and the delta is that cell's realized
# reuse when no other traffic overlaps. The counters are in tokens: across three
# runs the queries delta equalled the cell's total input tokens exactly,
# which both fixes the unit and proves no other client shared the server.
#
#   vllm_cache_snapshot <output_json_path>
# Writes nothing and returns 0 when no vLLM endpoint is reachable, so runs
# against OpenAI or a down tunnel are unaffected.
vllm_cache_snapshot() {
    local out="$1"
    local base="${VLLM_URL:-}"
    # Say so when the reading is skipped.  A skipped snapshot leaves realized
    # at 0, which in a report is indistinguishable from a measured 0 -- one
    # GenoMAS run was read as "the cache is not working" when in fact the
    # server was at 29% and nobody had asked it.
    if [ -z "$base" ]; then
        echo "  Note: VLLM_URL unset; not reading the server's prefix-cache" \
             "counters. Realized reuse will be UNMEASURED, not zero." >&2
        return 0
    fi
    base="${base%/}"
    base="${base%/v1}"

    local metrics
    if ! metrics="$(curl -s --connect-timeout 3 --max-time 10 "$base/metrics" 2>/dev/null)" \
       || [ -z "$metrics" ]; then
        echo "  Warning: $base/metrics unreachable; realized reuse will be" \
             "UNMEASURED for this cell." >&2
        return 0
    fi

    local queries hits
    queries="$(printf '%s\n' "$metrics" | awk '/^vllm:prefix_cache_queries_total/ {print $2; exit}')"
    hits="$(printf '%s\n' "$metrics" | awk '/^vllm:prefix_cache_hits_total/ {print $2; exit}')"
    if [ -z "$queries" ] || [ -z "$hits" ]; then
        echo "  Warning: $base/metrics has no vllm:prefix_cache_* counters;" \
             "realized reuse will be UNMEASURED for this cell." >&2
        return 0
    fi

    printf '{"unix_time": %s, "prefix_cache_queries_total": %s, "prefix_cache_hits_total": %s}\n' \
        "$(date +%s)" "$queries" "$hits" > "$out"

    # The serving configuration this cell actually ran against: cache_dtype,
    # block_size, prefix_match_unit, capacity, and everything else vLLM
    # publishes.  Read once per cell, next to the counters -- a knob sweep is
    # unattributable without it, and the server's live config answers for the
    # server as it is now, not as it was during the run.
    printf '%s\n' "$metrics" \
        | awk '/^vllm:cache_config_info\{/ {print; exit}' \
        | python3 -c '
import json, re, sys
line = sys.stdin.read().strip()
inner = re.search(r"\{(.*)\}", line)
config = {}
if inner:
    for pair in re.findall(r"([A-Za-z0-9_]+)=\"([^\"]*)\"", inner.group(1)):
        config[pair[0]] = pair[1]
json.dump(config, open(sys.argv[1], "w"), indent=1, sort_keys=True)
' "${out%.json}_serving_config.json" 2>/dev/null || true
}

# Turn the two snapshots into the cell's realized token-level reuse.
#   vllm_cache_delta <before.json> <after.json> <output_json>
vllm_cache_delta() {
    local before="$1" after="$2" out="$3"
    [ -s "$before" ] && [ -s "$after" ] || return 0
    python3 - "$before" "$after" "$out" <<'PY' || true
import json, sys
before, after, out = (json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3])
dq = before["prefix_cache_queries_total"]
dq = after["prefix_cache_queries_total"] - dq
dh = after["prefix_cache_hits_total"] - before["prefix_cache_hits_total"]
json.dump({
    "source": "vllm_prometheus_prefix_cache",
    # Verified against three runs: queries equals the cell's total input
    # tokens exactly, so these counters are in tokens, and hits/queries is
    # directly comparable to a vendor-reported cacheRead/input.
    "unit": "tokens",
    "queries": dq,
    "hits": dh,
    "hit_rate": (dh / dq) if dq > 0 else None,
    # queries != this cell's total input tokens means another client was
    # hitting the same server, and the delta is not attributable to this run.
    "validity": "compare queries against total_input_tokens; equal means sole client",
}, open(out, "w"), indent=1)
PY
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

# Run one Python post-processing module, capture its log, and add its label to
# the caller's failure list. Callers keep control of ordering and arguments.
run_postprocess_module() {
    local python="$1" run_dir="$2" step="$3" log_name="$4" module="$5"
    shift 5
    local rc
    if "$python" -m "$module" "$@" >"$run_dir/$log_name" 2>&1; then
        rc=0
    else
        rc=$?
    fi
    if [ "$rc" -ne 0 ]; then
        failed_step="${failed_step:+$failed_step,}$step"
    fi
    return "$rc"
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
#
# The tracer runs as root while this script may run as the regular user, and a
# regular user cannot signal a root process: plain `kill` returns EPERM, which
# `|| true` used to swallow.  Every signal then did nothing, the escalation ran
# to SIGKILL, and the trace was truncated with an empty bcc.err.  So retry each
# signal through sudo whenever the direct attempt is refused.
_signal_tracer() {
    local sig="$1" pid="$2"
    kill -"$sig" "$pid" >/dev/null 2>&1 && return 0
    [ -d "/proc/$pid" ] || return 0              # already gone, nothing to do
    sudo -n kill -"$sig" "$pid" >/dev/null 2>&1 || true
}

stop_tracer() {
    local sudo_pid="$1"
    local grace="${2:-30}"
    [ -n "$sudo_pid" ] || return 0

    # Capture the children before signalling: they are gone once INT lands.
    local kids
    kids="$(pgrep -P "$sudo_pid" 2>/dev/null || true)"

    # "Alive" means the wrapper OR the tracer itself: if sudo were reaped first
    # we would otherwise return while the tracer is still writing the log.
    #
    # Tested on the cluster: `kill -0` against the root-owned tracer returns
    # EPERM for a regular user, which is indistinguishable from "no such
    # process" and would report a running tracer as dead.  /proc answers
    # regardless of who owns the process.
    _stop_tracer_alive() {
        local p
        for p in "$sudo_pid" $kids; do
            [ -d "/proc/$p" ] && return 0
        done
        return 1
    }

    local sig p waited
    for sig in INT TERM KILL; do
        _stop_tracer_alive || break
        for p in $kids "$sudo_pid"; do
            _signal_tracer "$sig" "$p"
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
