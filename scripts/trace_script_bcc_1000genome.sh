#!/usr/bin/env bash
# Trace the classic 1000genome DAG through its direct Python driver.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CFG_DIR="$ROOT_DIR/config"
CONFIG_FILE="${CONFIG_FILE:-$CFG_DIR/config_1000genome.env}"
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

if [ -f "$ROOT_DIR/.env.1000genome" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.env.1000genome"
    set +a
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config file not found: $CONFIG_FILE" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG_FILE"
[ -n "$CALLER_BASE_OUT" ] && BASE_OUT="$CALLER_BASE_OUT"

if [ ! -d "$WORKFLOW_REPO/bin" ]; then
    echo "Error: invalid WORKFLOW_REPO=$WORKFLOW_REPO (missing bin/)" >&2
    exit 1
fi
if [ ! -x "$TRACER_PYTHON" ]; then
    echo "Error: TRACER_PYTHON=$TRACER_PYTHON is not executable" >&2
    exit 1
fi
if ! "$TRACER_PYTHON" -c "from bcc import BPF" >/dev/null 2>&1; then
    echo "Error: $TRACER_PYTHON cannot import bcc.BPF" >&2
    exit 1
fi
if [ ! -x "$AGENT_PYTHON" ]; then
    echo "Error: AGENT_PYTHON=$AGENT_PYTHON is not executable" >&2
    exit 1
fi
if [ ! -x "$POST_PYTHON" ]; then
    echo "Error: POST_PYTHON=$POST_PYTHON is not executable" >&2
    exit 1
fi
if [ ! -f "$COLUMNS_FILE" ]; then
    echo "Error: columns file not found: $COLUMNS_FILE" >&2
    exit 1
fi
if [ "${#WORKLOADS[@]}" -eq 0 ]; then
    echo "Error: WORKLOADS array is empty" >&2
    exit 1
fi
if ! [[ "$CLASSIC_VCF_RECORD_LIMIT" =~ ^[0-9]+$ ]]; then
    echo "Error: CLASSIC_VCF_RECORD_LIMIT must be a non-negative integer" >&2
    exit 1
fi
if ! [[ "$BCC_PERF_PAGES" =~ ^[1-9][0-9]*$ ]] || [ $((BCC_PERF_PAGES & (BCC_PERF_PAGES - 1))) -ne 0 ]; then
    echo "Error: BCC_PERF_PAGES must be a positive power of two" >&2
    exit 1
fi
if [ "$CLASSIC_VCF_RECORD_LIMIT" -gt 0 ] && [ "$ROWS_PER_CHROMOSOME" -gt "$CLASSIC_VCF_RECORD_LIMIT" ]; then
    echo "Error: ROWS_PER_CHROMOSOME cannot exceed CLASSIC_VCF_RECORD_LIMIT" >&2
    exit 1
fi

BASE_OUT="${BASE_OUT:-$(default_lustre_results_root)/$(run_dir_name 1000Genome $(selected_task_names))}"
require_lustre_base_out "$BASE_OUT"
BASE_OUT="$(cd "$BASE_OUT" && pwd)"

validate_workload_selection "${WORKLOADS[@]}"

echo "=== Classic 1000genome FS tracer ==="
echo "Repo:          $WORKFLOW_REPO"
echo "Output:        $BASE_OUT"
echo "Task workers:  $MAX_WORKERS"
echo "Chunk jobs:    $INDIVIDUAL_JOBS per chromosome"
echo "Populations:   $POPULATIONS"
echo "Offline:       $CLASSIC_OFFLINE"
echo "VCF record cap: $CLASSIC_VCF_RECORD_LIMIT (0 = complete inputs)"
echo "Perf pages:    $BCC_PERF_PAGES per CPU"
echo "Cells:         ${#WORKLOADS[@]}"
echo ""

RUN_FAIL_COUNT=0
for entry in "${WORKLOADS[@]}"; do
    NAME="${entry%%|*}"
    REST1="${entry#*|}"
    SCALE="${REST1%%|*}"
    REST2="${REST1#*|}"
    REP="${REST2%%|*}"
    EXTRA="${REST2#*|}"
    if [ -z "$NAME" ] || ! [[ "$SCALE" =~ ^[0-9]+$ ]] || [ "$SCALE" -lt 1 ] || [ -z "$REP" ]; then
        echo "Skipping malformed cell: $entry" >&2
        continue
    fi
    workload_selected "$NAME" || { echo "Skipping: $NAME"; continue; }

    OUT="$BASE_OUT/$NAME"
    WORK="$OUT/work"
    mkdir -p "$WORK"
    OUT="$(cd "$OUT" && pwd)"
    WORK="$(cd "$WORK" && pwd)"
    CHROMOSOMES="$(seq -s, 1 "$SCALE")"

    subset_vcf() {
        local source="$1" destination="$2" limit="$3"
        awk -v limit="$limit" '
            /^#/ { print; next }
            records < limit { print; records++; if (records >= limit) exit }
        ' "$source" > "$destination"
        local records
        records="$(awk '!/^#/ { records++ } END { print records + 0 }' "$destination")"
        if [ "$records" -ne "$limit" ]; then
            echo "Error: requested $limit records but found $records in $source" >&2
            return 1
        fi
    }

    INPUT_ARGS=(--input "$COLUMNS_FILE")
    IFS=',' read -r -a POPULATION_ARRAY <<< "$POPULATIONS"
    for population in "${POPULATION_ARRAY[@]}"; do
        population="$(echo "$population" | xargs)"
        [ -n "$population" ] || continue
        population_file="$POPULATION_DIR/$population"
        if [ ! -f "$population_file" ]; then
            echo "Error: population file not found: $population_file" >&2
            exit 1
        fi
        INPUT_ARGS+=(--input "$population_file")
    done
    for chromosome in $(seq 1 "$SCALE"); do
        main_vcf="${MAIN_VCF_SOURCE_TEMPLATE//\{chromosome\}/$chromosome}"
        annotation_vcf="${ANNOTATION_VCF_SOURCE_TEMPLATE//\{chromosome\}/$chromosome}"
        if [ ! -f "$main_vcf" ] || [ ! -f "$annotation_vcf" ]; then
            echo "Error: decompressed inputs missing for chromosome $chromosome" >&2
            echo "  main:       $main_vcf" >&2
            echo "  annotation: $annotation_vcf" >&2
            exit 1
        fi
        if [ "$CLASSIC_VCF_RECORD_LIMIT" -gt 0 ]; then
            subset_dir="$OUT/input_subset"
            mkdir -p "$subset_dir"
            subset_main="$subset_dir/$(basename "$main_vcf")"
            subset_annotation="$subset_dir/$(basename "$annotation_vcf")"
            subset_vcf "$main_vcf" "$subset_main" "$CLASSIC_VCF_RECORD_LIMIT"
            subset_vcf "$annotation_vcf" "$subset_annotation" "$CLASSIC_VCF_RECORD_LIMIT"
            INPUT_ARGS+=(--input "$subset_main" --input "$subset_annotation")
        else
            INPUT_ARGS+=(--input "$main_vcf" --input "$annotation_vcf")
        fi
    done

    EXTRA_ARGS=()
    [ -n "$EXTRA" ] && read -r -a EXTRA_ARGS <<< "$EXTRA"
    OFFLINE_ARGS=()
    OFFLINE_JSON=false
    if [ "$CLASSIC_OFFLINE" = "1" ] || [ "$CLASSIC_OFFLINE" = "true" ]; then
        OFFLINE_ARGS=(--offline)
        OFFLINE_JSON=true
    fi
    echo "=== $NAME: chromosomes=$CHROMOSOMES rep=$REP ==="

    set +e
    "$AGENT_PYTHON" -m agent_io_tracing.adapters.classic.launcher \
        "$WORK" "$OUT" \
        --cmd "$WORKFLOW_CMD" \
        --repo "$WORKFLOW_REPO" \
        "${INPUT_ARGS[@]}" \
        -- \
        --chromosomes "$CHROMOSOMES" \
        --populations "$POPULATIONS" \
        --individual-jobs "$INDIVIDUAL_JOBS" \
        --max-workers "$MAX_WORKERS" \
        --rows-per-chromosome "$ROWS_PER_CHROMOSOME" \
        --execution-units-log "$OUT/execution_units.jsonl" \
        --main-vcf-template "$MAIN_VCF_NAME_TEMPLATE" \
        --annotation-vcf-template "$ANNOTATION_VCF_NAME_TEMPLATE" \
        "${OFFLINE_ARGS[@]}" \
        "${EXTRA_ARGS[@]}" \
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
    EXTRA_JSON="{\"offline\":$OFFLINE_JSON,\"scale_chromosomes\":$SCALE,\"chromosomes\":\"$CHROMOSOMES\",\"rep\":$REP,\"rows_per_chromosome\":$ROWS_PER_CHROMOSOME,\"vcf_record_limit\":$CLASSIC_VCF_RECORD_LIMIT,\"individual_jobs\":$INDIVIDUAL_JOBS,\"max_workers\":$MAX_WORKERS,\"populations\":\"$POPULATIONS\"}"
    "$POST_PYTHON" -m agent_io_tracing.experiments.run_manifest \
        --output "$OUT/manifest.json" \
        --workload "1000genome-classic" \
        --task-id "$NAME" \
        --agent-count "$MAX_WORKERS" \
        --pid "$AGENT_PID" \
        --data-dir "$DATASET_DIR" \
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

echo "=== Classic post-processing ==="
POST_FAIL_COUNT=0
for ws_out in "$BASE_OUT"/*/; do
    [ -f "$ws_out/ebpf_events.log" ] || continue
    NAME="$(basename "$ws_out")"
    failed_step=""
    set +e
    "$POST_PYTHON" -m agent_io_tracing.parsing.ebpf "$ws_out" >"$ws_out/parse.log" 2>&1
    PARSE_RC=$?
    [ "$PARSE_RC" -ne 0 ] && failed_step="parse_ebpf"
    if [ "$PARSE_RC" -eq 0 ] && [ -f "$ws_out/parsed.json" ]; then
        "$POST_PYTHON" -m agent_io_tracing.analysis.artifact_sizes "$ws_out" \
            >"$ws_out/artifact_sizes.log" 2>&1
        ARTIFACT_RC=$?
        [ "$ARTIFACT_RC" -ne 0 ] && failed_step="${failed_step:+$failed_step,}artifact_sizes"

        if [ -s "$ws_out/pi_events.jsonl" ]; then
            "$POST_PYTHON" -m agent_io_tracing.analysis.summary "$ws_out" \
                >"$ws_out/analysis_summary.log" 2>&1
            STEP_RC=$?
            [ "$STEP_RC" -ne 0 ] && failed_step="${failed_step:+$failed_step,}analysis.summary"
        fi
        for module in lineage.analyzer analysis.parallelism analysis.phase1_metrics analysis.execution_units analysis.trace_quality viz.trace; do
                log_name="${module//./_}.log"
                "$POST_PYTHON" -m "agent_io_tracing.$module" "$ws_out" >"$ws_out/$log_name" 2>&1
                STEP_RC=$?
                [ "$STEP_RC" -ne 0 ] && failed_step="${failed_step:+$failed_step,}$module"
                if [ "$module" = "lineage.analyzer" ] && [ "$STEP_RC" -eq 0 ]; then
                    "$POST_PYTHON" -m agent_io_tracing.analysis.per_run_io_char \
                        --results "$ws_out" --runs . >"$ws_out/per_run_io_char.log" 2>&1
                    IO_RC=$?
                    [ "$IO_RC" -ne 0 ] && failed_step="${failed_step:+$failed_step,}per_run_io_char"
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
