#!/bin/bash
#
# Trace GenoMAS (https://github.com/Liu-Hy/GenoMAS) using eBPF/BCC.
#
# Mirrors trace_script_bcc_scilink.sh: STOP the agent, start the tracer,
# wait for 'ready' on a FIFO, CONT the agent, wait, then run parse +
# summarize + parallelism + visualize.  Each WORKLOAD entry encodes one
# Phase-4 matrix cell (max_workers × rep).
#
# Differences from the SciLink version:
#   - GenoMAS is non-interactive (no REPL, no --prompt field needed).
#   - WORKLOADS entries are "name|max_workers|rep|extra_args" — 4 fields,
#     not SciLink's 5.
#   - Inter-run sleep (config_genomas.env: INTER_RUN_SLEEP_SEC) lets the
#     OpenAI per-minute rate-limit bucket drain between cells.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CFG_DIR="$ROOT_DIR/config"
CONFIG_FILE="${CONFIG_FILE:-$CFG_DIR/config_genomas.env}"
CALLER_BASE_OUT="${BASE_OUT:-}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib_results.sh"

# Source .env.genomas so API keys + AGENT_PYTHON (absolute, sudo-safe)
# survive `sudo -E`.  deploy_genomas_to_client.sh writes this file with
# hard-coded absolute paths so HOME=/root under sudo doesn't break python
# resolution.  We INTENTIONALLY do NOT source the sibling .env or
# .env.scilink files (those belong to other harnesses).
if [ -f "$ROOT_DIR/.env.genomas" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.env.genomas"
    set +a
else
    echo "Warning: $ROOT_DIR/.env.genomas not found." >&2
    echo "         Re-run deploy_genomas_to_client.sh on your laptop to generate it." >&2
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config file not found: $CONFIG_FILE" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

if [ -n "$CALLER_BASE_OUT" ]; then
    BASE_OUT="$CALLER_BASE_OUT"
fi

# --- Verify paths and interpreters -----------------------------------------
mkdir -p "$WORK_DIR"
mkdir -p "$DATA_DIR" || { echo "Error: cannot create DATA_DIR=$DATA_DIR" >&2; exit 1; }
BASE_OUT="${BASE_OUT:-$(default_lustre_results_root)/phase4_$(date +%Y%m%d_%H%M%S)}"
require_lustre_base_out "$BASE_OUT"
BASE_OUT="$(cd "$BASE_OUT" && pwd)"

if [ ! -x "$TRACER_PYTHON" ]; then
    echo "Error: TRACER_PYTHON=$TRACER_PYTHON is not executable" >&2
    exit 1
fi
if ! "$TRACER_PYTHON" -c "from bcc import BPF" >/dev/null 2>&1; then
    echo "Error: $TRACER_PYTHON cannot import bcc.BPF" >&2
    echo "Install on CentOS Stream 8: sudo dnf install python3-bcc bcc-tools" >&2
    exit 1
fi
if [ ! -x "$AGENT_PYTHON" ]; then
    echo "Error: AGENT_PYTHON=$AGENT_PYTHON is not executable" >&2
    echo "Re-run deploy_genomas_to_client.sh to (re)build the uv venv." >&2
    exit 1
fi
if ! "$AGENT_PYTHON" -c "import sys; sys.path.insert(0,'$GENOMAS_REPO'); import utils.llm" >/dev/null 2>&1; then
    echo "Error: GenoMAS utils.llm not importable by $AGENT_PYTHON" >&2
    echo "       GENOMAS_REPO=$GENOMAS_REPO" >&2
    exit 1
fi

[ -d "$ROOT_DIR/src/agent_io_tracing" ] || {
    echo "Error: package not found: $ROOT_DIR/src/agent_io_tracing" >&2
    exit 1
}

if [ "${#WORKLOADS[@]}" -eq 0 ]; then
    echo "Error: WORKLOADS array is empty (configure config_genomas.env)" >&2
    exit 1
fi

echo "=== GenoMAS FS+Net Tracer (BCC) — Phase 4 matrix ==="
echo "Repo:       $GENOMAS_REPO"
echo "Work dir:   $WORK_DIR"
echo "Data dir:   $DATA_DIR"
echo "Output dir: $BASE_OUT"
echo "Model:      $GENOMAS_MODEL"
echo "Cells:      ${#WORKLOADS[@]} (= max-workers values × reps)"
echo "Inter-run sleep: ${INTER_RUN_SLEEP_SEC}s"
[ -n "${RUN_WORKLOADS:-}" ] && echo "RUN_WORKLOADS filter: $RUN_WORKLOADS"
echo ""

IFS=',' read -r -a RUN_NAME_ARRAY <<< "${RUN_WORKLOADS:-}"

should_run_workload() {
    local name="$1"
    if [ "${#RUN_NAME_ARRAY[@]}" -eq 0 ] || [ -z "${RUN_NAME_ARRAY[0]}" ]; then
        return 0
    fi
    local item
    for item in "${RUN_NAME_ARRAY[@]}"; do
        item="$(echo "$item" | xargs)"
        if [ -n "$item" ] && [ "$item" = "$name" ]; then
            return 0
        fi
    done
    return 1
}

# Each entry: "name|max_workers|rep|extra_args"
CELL_IDX=0
for entry in "${WORKLOADS[@]}"; do
    NAME="${entry%%|*}"
    REST1="${entry#*|}"
    MW="${REST1%%|*}"
    REST2="${REST1#*|}"
    REP="${REST2%%|*}"
    EXTRA="${REST2#*|}"

    if [ -z "$NAME" ] || [ -z "$MW" ] || [ -z "$REP" ]; then
        echo "Skip malformed cell (need name|max_workers|rep|extra): '$entry'" >&2
        continue
    fi
    if ! should_run_workload "$NAME"; then
        echo "Skipping: $NAME (not in RUN_WORKLOADS)"
        continue
    fi

    CELL_IDX=$((CELL_IDX + 1))
    OUT="$BASE_OUT/$NAME"
    mkdir -p "$OUT"
    OUT="$(cd "$OUT" && pwd)"

    WORK="$OUT/work"
    mkdir -p "$WORK"
    WORK="$(cd "$WORK" && pwd)"

    echo "=== Cell $CELL_IDX/${#WORKLOADS[@]}: $NAME (max-workers=$MW, rep=$REP) ==="
    echo "  Start time:  $(date +%H:%M:%S)"
    echo "  Output:      $OUT"

    # Build smoke-traits flag only if explicitly configured.
    TRAITS_FLAG=()
    if [ -n "${GENOMAS_SMOKE_TRAITS:-}" ]; then
        TRAITS_FLAG=("--smoke-traits" "$GENOMAS_SMOKE_TRAITS")
    fi

    set +e
    # Detect "fullpipeline" sentinel in EXTRA: if present, drop --quick-test
    # so GenoMAS actually runs the statistician/regression phase.  Otherwise
    # default is --quick-test (preprocess only) for cheaper runs.
    QUICK_TEST_FLAG="--quick-test"
    if [[ "$EXTRA" == *fullpipeline* ]]; then
        QUICK_TEST_FLAG=""
        EXTRA="${EXTRA//fullpipeline/}"
        echo "  fullpipeline mode: regression phase enabled"
    fi

    # --version embeds NAME so GenoMAS's output/log_<version>.txt is unique
    # per cell (otherwise checkpoint-resume across cells would corrupt data).
    "$AGENT_PYTHON" -m agent_io_tracing.adapters.genomas.launcher \
        "$WORK" "$OUT" \
        --data-root "$DATA_DIR" \
        --model "$GENOMAS_MODEL" \
        --api 1 \
        --version "$NAME" \
        $QUICK_TEST_FLAG \
        --parallel-mode cohorts \
        --max-workers "$MW" \
        "${TRAITS_FLAG[@]}" \
        $EXTRA \
        > "$OUT/genomas.log" 2>&1 &
    AGENT_PID=$!
    kill -STOP "$AGENT_PID" >/dev/null 2>&1 || true

    INSTRUMENTATION_LEVEL="ebpf"
    if [ "${COLLECT_LUSTRE_COUNTERS:-0}" = "1" ] || [ "${COLLECT_LUSTRE_COUNTERS:-0}" = "true" ]; then
        INSTRUMENTATION_LEVEL="ebpf+lustre-counters"
    fi

    "$POST_PYTHON" -m agent_io_tracing.experiments.run_manifest \
        --output "$OUT/manifest.json" \
        --workload "GenoMAS" \
        --task-id "$NAME" \
        --model "$GENOMAS_MODEL" \
        --api "1" \
        --replay-mode "$GENOMAS_LLM_REPLAY" \
        --llm-cache-path "${GENOMAS_LLM_CACHE_PATH:-$OUT/llm_cache.jsonl}" \
        --agent-count "$MW" \
        --pid "$AGENT_PID" \
        --genomas-repo "$GENOMAS_REPO" \
        --data-dir "$DATA_DIR" \
        --work-dir "$WORK" \
        --output-dir "$OUT" \
        --instrumentation "$INSTRUMENTATION_LEVEL" \
        > "$OUT/manifest.log" 2>&1 || true

    LUSTRE_SAMPLER_PID=""
    if [ "${COLLECT_LUSTRE_COUNTERS:-0}" = "1" ] || [ "${COLLECT_LUSTRE_COUNTERS:-0}" = "true" ]; then
        "$POST_PYTHON" -m agent_io_tracing.tracing.lustre_counters \
            --output "$OUT/lustre_counters.jsonl" \
            --interval "$LUSTRE_COUNTER_INTERVAL_SEC" \
            > "$OUT/lustre_counters.log" 2>&1 &
        LUSTRE_SAMPLER_PID=$!
    fi

    BCC_NET_FLAG="${BCC_INCLUDE_NET:-1}"
    if [ "$BCC_NET_FLAG" = "1" ] || [ "$BCC_NET_FLAG" = "true" ]; then
        NET_ARG="--include-net"
    else
        NET_ARG="--no-include-net"
    fi

    sudo -E env "PYTHONPATH=$PYTHONPATH" "$TRACER_PYTHON" -m agent_io_tracing.tracing.bcc_tracer \
        --root-pid "$AGENT_PID" \
        --output "$OUT/ebpf_events.log" \
        $NET_ARG \
        >"$OUT/bcc.out" 2>"$OUT/bcc.err" &
    TRACER_PID=$!

    TRACER_READY=0
    for _ in $(seq 1 100); do
        if [ -s "$OUT/ebpf_events.log" ]; then
            TRACER_READY=1
            break
        fi
        if ! kill -0 "$TRACER_PID" >/dev/null 2>&1; then
            break
        fi
        sleep 0.1
    done
    if [ "$TRACER_READY" != "1" ]; then
        echo "  Warning: tracer did not create ebpf_events.log before agent resume" >&2
    fi
    kill -CONT "$AGENT_PID" >/dev/null 2>&1 || true

    wait "$AGENT_PID"
    EXIT_CODE=$?

    stop_tracer "$TRACER_PID"
    if [ -n "$LUSTRE_SAMPLER_PID" ]; then
        kill -INT "$LUSTRE_SAMPLER_PID" >/dev/null 2>&1 || true
        wait "$LUSTRE_SAMPLER_PID" >/dev/null 2>&1 || true
    fi
    set -e

    echo "  End time:    $(date +%H:%M:%S)  (exit=$EXIT_CODE)"

    # Inter-cell sleep so the next cell starts with a fresh OpenAI rpm bucket.
    # Skip after the last cell so total wall-clock isn't padded for nothing.
    if [ "$CELL_IDX" -lt "${#WORKLOADS[@]}" ] && [ "$INTER_RUN_SLEEP_SEC" -gt 0 ]; then
        echo "  Sleeping ${INTER_RUN_SLEEP_SEC}s before next cell..."
        sleep "$INTER_RUN_SLEEP_SEC"
    fi
    echo ""
done

echo "=== Post-processing (parse + summarize + parallelism + visualize) ==="
POST_FAIL_COUNT=0
POST_FAIL_NAMES=()

for ws_out in "$BASE_OUT"/*/; do
    [ -d "$ws_out" ] || continue
    NAME="$(basename "$ws_out")"

    if [ ! -f "$ws_out/ebpf_events.log" ]; then
        echo "Skipping $NAME (no ebpf_events.log)"
        continue
    fi

    echo "Processing: $NAME"
    set +e
    failed_step=""

    "$POST_PYTHON" -m agent_io_tracing.parsing.ebpf "$ws_out" \
        > "$ws_out/parse.log" 2>&1
    PARSE_RC=$?
    [ $PARSE_RC -ne 0 ] && failed_step="parse_ebpf"
    sed 's/^/    /' "$ws_out/parse.log" | tail -5 || true

    if [ -f "$ws_out/parsed.json" ] && [ -f "$ws_out/pi_events.jsonl" ]; then
        "$POST_PYTHON" -m agent_io_tracing.analysis.summary "$ws_out" \
            > "$ws_out/summarize.log" 2>&1
        SUM_RC=$?
        [ $SUM_RC -ne 0 ] && failed_step="${failed_step:+$failed_step,}summarize"

        "$POST_PYTHON" -m agent_io_tracing.analysis.parallelism "$ws_out" \
            > "$ws_out/parallelism.log" 2>&1
        PAR_RC=$?
        [ $PAR_RC -ne 0 ] && failed_step="${failed_step:+$failed_step,}parallelism"

        "$POST_PYTHON" -m agent_io_tracing.lineage.analyzer "$ws_out" \
            > "$ws_out/lineage.log" 2>&1
        LIN_RC=$?
        [ $LIN_RC -ne 0 ] && failed_step="${failed_step:+$failed_step,}lineage"

        "$POST_PYTHON" -m agent_io_tracing.analysis.phase1_metrics "$ws_out" \
            > "$ws_out/phase1_metrics.log" 2>&1
        P1_RC=$?
        [ $P1_RC -ne 0 ] && failed_step="${failed_step:+$failed_step,}phase1_metrics"

        "$POST_PYTHON" -m agent_io_tracing.analysis.per_run_io_char --results "$ws_out" --runs . \
            > "$ws_out/per_run_io_char.log" 2>&1
        PRC_RC=$?
        [ $PRC_RC -ne 0 ] && failed_step="${failed_step:+$failed_step,}per_run_io_char"

        "$POST_PYTHON" -m agent_io_tracing.viz.trace "$ws_out" \
            > "$ws_out/visualize.log" 2>&1
        VIZ_RC=$?
        [ $VIZ_RC -ne 0 ] && failed_step="${failed_step:+$failed_step,}visualize"

        if [ $PAR_RC -eq 0 ] && [ -f "$ws_out/parallelism_summary.json" ]; then
            WCS=$("$POST_PYTHON" -c "import json; print(json.load(open('$ws_out/parallelism_summary.json'))['wall_clock_s'])" 2>/dev/null || echo "?")
            CF=$("$POST_PYTHON" -c "import json; print(json.load(open('$ws_out/parallelism_summary.json'))['workload_concurrency_factor'])" 2>/dev/null || echo "?")
            echo "    wall_clock=${WCS}s  concurrency_factor=${CF}"
        fi
    else
        echo "  Skipping post-processing (parsed.json or pi_events.jsonl missing)"
        failed_step="${failed_step:+$failed_step,}missing_inputs"
    fi
    set -e

    if [ -n "$failed_step" ]; then
        echo "  ⚠  Partial failure for $NAME: $failed_step"
        POST_FAIL_COUNT=$((POST_FAIL_COUNT + 1))
        POST_FAIL_NAMES+=("$NAME($failed_step)")
    fi
    echo ""
done

# --- Aggregate matrix summary CSV (one row per cell) ----------------------
SUMMARY_CSV="$BASE_OUT/matrix_summary.csv"
echo "cell_name,max_workers,rep,wall_clock_s,total_llm_s,concurrency_factor,n_llm_calls" > "$SUMMARY_CSV"
for ws_out in "$BASE_OUT"/*/; do
    [ -d "$ws_out" ] || continue
    [ -f "$ws_out/parallelism_summary.json" ] || continue
    NAME="$(basename "$ws_out")"
    # Workers are encoded as the trailing _w<N> in the cell name (e.g. A_c8_w4,
    # B_t2_w2).  The cell_name column itself carries the trait/cohort spec.
    MW="$(echo "$NAME" | sed -E 's/.*_w([0-9]+).*/\1/')"
    REP="1"
    LINE=$("$POST_PYTHON" - "$ws_out" << 'PYEOF'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
ps = json.loads((out/"parallelism_summary.json").read_text())
n_calls = sum(1 for _ in (out/"pi_events.jsonl").open()
              if '"message_end"' in _) if (out/"pi_events.jsonl").exists() else 0
print(f"{ps.get('wall_clock_s','')},{ps.get('total_self_time_s','')},"
      f"{ps.get('workload_concurrency_factor','')},{n_calls}")
PYEOF
)
    echo "$NAME,$MW,$REP,$LINE" >> "$SUMMARY_CSV"
done

echo "=== Matrix summary written to $SUMMARY_CSV ==="
cat "$SUMMARY_CSV" 2>/dev/null || true

if [ "$POST_FAIL_COUNT" -gt 0 ]; then
    echo ""
    echo "Post-processing finished with $POST_FAIL_COUNT failed cell(s):"
    for n in "${POST_FAIL_NAMES[@]}"; do echo "  - $n"; done
    echo "(See <cell>/{parse,summarize,parallelism,visualize}.log for details)"
fi

chmod -R a+rX "$BASE_OUT" || true
if [ -n "${SUDO_UID:-}" ] && [ -n "${SUDO_GID:-}" ]; then
    chown -R "$SUDO_UID:$SUDO_GID" "$BASE_OUT" 2>/dev/null || true
    echo "Returned ownership of $BASE_OUT to ${SUDO_USER:-uid=$SUDO_UID}"
fi

echo ""
echo "=== Phase 4 matrix run complete ==="
echo "Results in: $BASE_OUT"
echo "Headline CSV: $SUMMARY_CSV"
echo "Per-cell: $BASE_OUT/<cell>/{visualizations,parallelism_summary.json,ebpf_events.log,pi_events.jsonl}"
