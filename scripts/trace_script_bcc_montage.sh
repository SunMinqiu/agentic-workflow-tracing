#!/usr/bin/env bash
# Trace a fixed-input Montage mosaic through the direct stage driver.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$ROOT_DIR/config/config_montage.env}"
CALLER_BASE_OUT="${BASE_OUT:-}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib_results.sh"

CURRENT_AGENT_PID=""
CURRENT_TRACER_PID=""
CURRENT_LUSTRE_PID=""
cleanup_interrupted_cell() {
    trap - INT TERM
    if [ -n "$CURRENT_AGENT_PID" ]; then
        kill -CONT "$CURRENT_AGENT_PID" >/dev/null 2>&1 || true
        kill -TERM "$CURRENT_AGENT_PID" >/dev/null 2>&1 || true
        wait "$CURRENT_AGENT_PID" >/dev/null 2>&1 || true
    fi
    if [ -n "$CURRENT_TRACER_PID" ]; then
        stop_tracer "$CURRENT_TRACER_PID" || true
    fi
    if [ -n "$CURRENT_LUSTRE_PID" ]; then
        kill -INT "$CURRENT_LUSTRE_PID" >/dev/null 2>&1 || true
        wait "$CURRENT_LUSTRE_PID" >/dev/null 2>&1 || true
    fi
    exit 130
}
trap cleanup_interrupted_cell INT TERM

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config file not found: $CONFIG_FILE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG_FILE"
[ -n "$CALLER_BASE_OUT" ] && BASE_OUT="$CALLER_BASE_OUT"

for python_var in TRACER_PYTHON AGENT_PYTHON POST_PYTHON; do
    python_path="${!python_var}"
    if [ ! -x "$python_path" ]; then
        echo "Error: $python_var=$python_path is not executable" >&2
        exit 1
    fi
done
if ! "$TRACER_PYTHON" -c "from bcc import BPF" >/dev/null 2>&1; then
    echo "Error: $TRACER_PYTHON cannot import bcc.BPF" >&2
    exit 1
fi
if ! "$AGENT_PYTHON" -c "import MontagePy; import agent_io_tracing" >/dev/null 2>&1; then
    echo "Error: $AGENT_PYTHON cannot import MontagePy and agent_io_tracing" >&2
    exit 1
fi
if [ "${#WORKLOADS[@]}" -eq 0 ]; then
    echo "Error: WORKLOADS array is empty" >&2
    exit 1
fi
if [ "$MONTAGE_OFFLINE" != "1" ] && [ "$MONTAGE_OFFLINE" != "true" ]; then
    echo "Error: formal Montage traces require MONTAGE_OFFLINE=1 and fixed input" >&2
    exit 1
fi
if ! [[ "$BCC_PERF_PAGES" =~ ^[1-9][0-9]*$ ]] || [ $((BCC_PERF_PAGES & (BCC_PERF_PAGES - 1))) -ne 0 ]; then
    echo "Error: BCC_PERF_PAGES must be a positive power of two" >&2
    exit 1
fi

BASE_OUT="${BASE_OUT:-$(default_lustre_results_root)/$(run_dir_name Montage $(selected_task_names))}"
require_lustre_base_out "$BASE_OUT"
BASE_OUT="$(cd "$BASE_OUT" && pwd)"

validate_workload_selection "${WORKLOADS[@]}"

echo "=== Montage fixed-input FS tracer ==="
echo "Input root:    $MONTAGE_INPUT_ROOT"
echo "Output:        $BASE_OUT"
echo "Runtime:       $AGENT_PYTHON"
echo "Offline:       $MONTAGE_OFFLINE"
echo "Perf pages:    $BCC_PERF_PAGES per CPU"
echo "Cells:         ${#WORKLOADS[@]}"
echo ""

RUN_FAIL_COUNT=0
for entry in "${WORKLOADS[@]}"; do
    NAME="${entry%%|*}"
    REST="${entry#*|}"
    SIZE="${REST%%|*}"
    REP="${REST#*|}"
    if [ -z "$NAME" ] || ! [[ "$SIZE" =~ ^[0-9]+([.][0-9]+)?$ ]] || [ -z "$REP" ]; then
        echo "Skipping malformed cell: $entry" >&2
        continue
    fi
    workload_selected "$NAME" || { echo "Skipping: $NAME"; continue; }

    SIZE_TAG="${SIZE/./p}deg"
    INPUT_DIR="$MONTAGE_INPUT_ROOT/m17_${SIZE_TAG}/raw"
    FIRST_FITS="$(find "$INPUT_DIR" -maxdepth 1 -type f -name '*.fits' -print -quit 2>/dev/null || true)"
    if [ ! -d "$INPUT_DIR" ] || [ -z "$FIRST_FITS" ]; then
        echo "Error: fixed FITS input is missing for $NAME: $INPUT_DIR" >&2
        exit 1
    fi
    if [ ! -s "$MONTAGE_INPUT_ROOT/m17_${SIZE_TAG}/input_manifest.sha256" ]; then
        echo "Error: input manifest is missing: $MONTAGE_INPUT_ROOT/m17_${SIZE_TAG}/input_manifest.sha256" >&2
        exit 1
    fi
    if ! (cd "$MONTAGE_INPUT_ROOT/m17_${SIZE_TAG}" && sha256sum -c input_manifest.sha256 >/dev/null); then
        echo "Error: fixed input checksum validation failed for $NAME" >&2
        exit 1
    fi

    OUT="$BASE_OUT/$NAME"
    WORK="$OUT/work"
    mkdir -p "$WORK"
    OUT="$(cd "$OUT" && pwd)"
    WORK="$(cd "$WORK" && pwd)"
    echo "=== $NAME: size=${SIZE}deg rep=$REP ==="

    set +e
    "$AGENT_PYTHON" -m agent_io_tracing.adapters.classic.launcher \
        "$WORK" "$OUT" \
        --cmd "$WORKFLOW_CMD" \
        --input "$INPUT_DIR:raw" \
        --env "TMPDIR=$MONTAGE_ROOT/tmp" \
        --env "PYTHONPYCACHEPREFIX=$MONTAGE_ROOT/cache/python" \
        --env "MPLCONFIGDIR=$MONTAGE_ROOT/cache/matplotlib" \
        --env "XDG_CACHE_HOME=$MONTAGE_ROOT/cache/xdg" \
        --env "XDG_CONFIG_HOME=$MONTAGE_ROOT/cache/config" \
        -- \
        --work-dir "$WORK" \
        --size-deg "$SIZE" \
        --execution-units-log "$OUT/execution_units.jsonl" \
        --offline \
        >"$OUT/classic_launcher.log" 2>&1 &
    AGENT_PID=$!
    CURRENT_AGENT_PID="$AGENT_PID"

    STOPPED=0
    for _ in $(seq 1 300); do
        if [ -r "/proc/$AGENT_PID/stat" ]; then
            PROCESS_STATE="$(awk '{print $3}' "/proc/$AGENT_PID/stat" 2>/dev/null)"
            if [ "$PROCESS_STATE" = "T" ]; then
                STOPPED=1
                break
            fi
        fi
        kill -0 "$AGENT_PID" >/dev/null 2>&1 || break
        sleep 0.1
    done
    if [ "$STOPPED" != "1" ]; then
        echo "Error: launcher did not reach its trace-ready stop: $NAME" >&2
        kill -CONT "$AGENT_PID" >/dev/null 2>&1 || true
        kill -TERM "$AGENT_PID" >/dev/null 2>&1 || true
        wait "$AGENT_PID"
        LAUNCH_SETUP_RC=$?
        echo "  launcher exit=$LAUNCH_SETUP_RC; see $OUT/classic_launcher.log" >&2
        RUN_FAIL_COUNT=$((RUN_FAIL_COUNT + 1))
        CURRENT_AGENT_PID=""
        set -e
        continue
    fi

    INSTRUMENTATION_LEVEL="ebpf"
    if [ "$COLLECT_LUSTRE_COUNTERS" = "1" ] || [ "$COLLECT_LUSTRE_COUNTERS" = "true" ]; then
        INSTRUMENTATION_LEVEL="ebpf+lustre-counters"
    fi
    INPUT_COUNT="$(find "$INPUT_DIR" -maxdepth 1 -type f -name '*.fits' | wc -l)"
    EXTRA_JSON="{\"offline\":true,\"survey\":\"2MASS J\",\"location\":\"M 17\",\"mosaic_size_deg\":$SIZE,\"rep\":$REP,\"input_fits_count\":$INPUT_COUNT,\"input_manifest\":\"$MONTAGE_INPUT_ROOT/m17_${SIZE_TAG}/input_manifest.sha256\"}"
    "$POST_PYTHON" -m agent_io_tracing.experiments.run_manifest \
        --output "$OUT/manifest.json" \
        --workload "montage-classic" \
        --task-id "$NAME" \
        --agent-count 1 \
        --pid "$AGENT_PID" \
        --data-dir "$INPUT_DIR" \
        --work-dir "$WORK" \
        --output-dir "$OUT" \
        --instrumentation "$INSTRUMENTATION_LEVEL" \
        --extra-json "$EXTRA_JSON" \
        >"$OUT/manifest.log" 2>&1 || true

    LUSTRE_SAMPLER_PID=""
    if [ "$COLLECT_LUSTRE_COUNTERS" = "1" ] || [ "$COLLECT_LUSTRE_COUNTERS" = "true" ]; then
        "$POST_PYTHON" -m agent_io_tracing.tracing.lustre_counters \
            --output "$OUT/lustre_counters.jsonl" \
            --interval "$LUSTRE_COUNTER_INTERVAL_SEC" \
            >"$OUT/lustre_counters.log" 2>&1 &
        LUSTRE_SAMPLER_PID=$!
        CURRENT_LUSTRE_PID="$LUSTRE_SAMPLER_PID"
    fi

    if [ "$BCC_INCLUDE_NET" = "1" ] || [ "$BCC_INCLUDE_NET" = "true" ]; then
        NET_ARG="--include-net"
    else
        NET_ARG="--no-include-net"
    fi
    sudo -E env "PYTHONPATH=$PYTHONPATH" "$TRACER_PYTHON" -m agent_io_tracing.tracing.bcc_tracer \
        --root-pid "$AGENT_PID" \
        --output "$OUT/ebpf_events.log" \
        --perf-pages "$BCC_PERF_PAGES" \
        "$NET_ARG" \
        >"$OUT/bcc.out" 2>"$OUT/bcc.err" &
    TRACER_PID=$!
    CURRENT_TRACER_PID="$TRACER_PID"

    if ! wait_for_trace_file "$TRACER_PID" "$OUT/ebpf_events.log"; then
        echo "Error: tracer was not ready; refusing to run untraced cell: $NAME" >&2
        stop_tracer "$TRACER_PID"
        kill -CONT "$AGENT_PID" >/dev/null 2>&1 || true
        kill -TERM "$AGENT_PID" >/dev/null 2>&1 || true
        wait "$AGENT_PID" >/dev/null 2>&1 || true
        if [ -n "$LUSTRE_SAMPLER_PID" ]; then
            kill -INT "$LUSTRE_SAMPLER_PID" >/dev/null 2>&1 || true
            wait "$LUSTRE_SAMPLER_PID" >/dev/null 2>&1 || true
        fi
        RUN_FAIL_COUNT=$((RUN_FAIL_COUNT + 1))
        CURRENT_AGENT_PID=""
        CURRENT_TRACER_PID=""
        CURRENT_LUSTRE_PID=""
        set -e
        continue
    fi

    kill -CONT "$AGENT_PID" >/dev/null 2>&1
    wait "$AGENT_PID"
    EXIT_CODE=$?
    stop_tracer "$TRACER_PID"
    if [ -n "$LUSTRE_SAMPLER_PID" ]; then
        kill -INT "$LUSTRE_SAMPLER_PID" >/dev/null 2>&1 || true
        wait "$LUSTRE_SAMPLER_PID" >/dev/null 2>&1 || true
    fi
    set -e
    CURRENT_AGENT_PID=""
    CURRENT_TRACER_PID=""
    CURRENT_LUSTRE_PID=""
    echo "  completed exit=$EXIT_CODE"
    [ "$EXIT_CODE" -eq 0 ] || RUN_FAIL_COUNT=$((RUN_FAIL_COUNT + 1))
done

echo "=== Montage post-processing ==="
POST_FAIL_COUNT=0
for ws_out in "$BASE_OUT"/*/; do
    [ -f "$ws_out/ebpf_events.log" ] || continue
    NAME="$(basename "$ws_out")"
    failed_step=""
    set +e
    run_postprocess_module "$POST_PYTHON" "$ws_out" parse_ebpf parse.log \
        agent_io_tracing.parsing.ebpf "$ws_out"
    PARSE_RC=$?
    if [ "$PARSE_RC" -eq 0 ] && [ -f "$ws_out/parsed.json" ]; then
        run_postprocess_module "$POST_PYTHON" "$ws_out" artifact_sizes artifact_sizes.log \
            agent_io_tracing.analysis.artifact_sizes "$ws_out"
        for module in lineage.analyzer analysis.parallelism analysis.phase1_metrics analysis.execution_units analysis.trace_quality viz.trace; do
            log_name="${module//./_}.log"
            run_postprocess_module "$POST_PYTHON" "$ws_out" "$module" "$log_name" \
                "agent_io_tracing.$module" "$ws_out"
            STEP_RC=$?
            if [ "$module" = "lineage.analyzer" ] && [ "$STEP_RC" -eq 0 ]; then
                run_postprocess_module "$POST_PYTHON" "$ws_out" per_run_io_char per_run_io_char.log \
                    agent_io_tracing.analysis.per_run_io_char --results "$ws_out" --runs .
            fi
        done
    fi
    set -e
    if [ -n "$failed_step" ]; then
        echo "  $NAME: partial failure: $failed_step" >&2
        POST_FAIL_COUNT=$((POST_FAIL_COUNT + 1))
    else
        echo "  $NAME: parsed.json, metrics, lineage, figures, index.html"
    fi
done

return_results_ownership "$BASE_OUT"
echo "Results: $BASE_OUT"
[ "$RUN_FAIL_COUNT" -eq 0 ] && [ "$POST_FAIL_COUNT" -eq 0 ]
