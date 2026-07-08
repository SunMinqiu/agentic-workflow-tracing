#!/bin/bash
#
# Trace ChemGraph XANES MCP workflows using eBPF/BCC.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CFG_DIR="$ROOT_DIR/config"
CONFIG_FILE="${CONFIG_FILE:-$CFG_DIR/config_chemgraph.env}"
CALLER_BASE_OUT="${BASE_OUT:-}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib_results.sh"

if [ -f "$ROOT_DIR/.env.chemgraph" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.env.chemgraph"
    set +a
else
    echo "Warning: $ROOT_DIR/.env.chemgraph not found." >&2
    echo "         Re-run deploy_chemgraph_to_client.sh on your laptop." >&2
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

mkdir -p "$WORK_DIR"
mkdir -p "$DATA_DIR" || { echo "Error: cannot create DATA_DIR=$DATA_DIR" >&2; exit 1; }
BASE_OUT="${BASE_OUT:-$(default_lustre_results_root)/chemgraph_$(date +%Y%m%d_%H%M%S)}"
require_lustre_base_out "$BASE_OUT"
BASE_OUT="$(cd "$BASE_OUT" && pwd)"
DATA_DIR="$(cd "$DATA_DIR" && pwd)"

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
    echo "Re-run deploy_chemgraph_to_client.sh to build the uv venv." >&2
    exit 1
fi
if ! "$AGENT_PYTHON" -c "import chemgraph, langchain_mcp_adapters" >/dev/null 2>&1; then
    echo "Error: chemgraph/langchain_mcp_adapters not importable by $AGENT_PYTHON" >&2
    exit 1
fi
if [ ! -x "$FDMNES_EXE" ]; then
    echo "Error: FDMNES_EXE=$FDMNES_EXE is not executable" >&2
    exit 1
fi

[ -d "$ROOT_DIR/src/agent_io_tracing" ] || {
    echo "Error: package not found: $ROOT_DIR/src/agent_io_tracing" >&2
    exit 1
}

if [ "${#WORKLOADS[@]}" -eq 0 ]; then
    echo "Error: WORKLOADS array is empty (configure config_chemgraph.env)" >&2
    exit 1
fi

echo "=== ChemGraph XANES MCP FS+Net Tracer (BCC) ==="
echo "Repo:       $CHEMGRAPH_REPO"
echo "Data dir:   $DATA_DIR"
echo "Output dir: $BASE_OUT"
echo "FDMNES:     $FDMNES_EXE"
echo "Workloads:  ${#WORKLOADS[@]}"
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

for entry in "${WORKLOADS[@]}"; do
    NAME="${entry%%|*}"
    REST1="${entry#*|}"
    MODEL="${REST1%%|*}"
    PROMPT_TEMPLATE="${REST1#*|}"

    if [ -z "$NAME" ] || [ -z "$MODEL" ] || [ -z "$PROMPT_TEMPLATE" ]; then
        echo "Skip malformed workload entry (need name|model|prompt): '$entry'" >&2
        continue
    fi
    if ! should_run_workload "$NAME"; then
        echo "Skipping: $NAME (not in RUN_WORKLOADS)"
        continue
    fi

    OUT="$BASE_OUT/$NAME"
    mkdir -p "$OUT"
    OUT="$(cd "$OUT" && pwd)"

    WORK="$OUT/work"
    mkdir -p "$WORK"
    WORK="$(cd "$WORK" && pwd)"

    DATA="$DATA_DIR/$NAME"
    mkdir -p "$DATA" "$DATA/tmp"
    DATA="$(cd "$DATA" && pwd)"
    export OUT DATA TMPDIR="$DATA/tmp"

    if [ -f "$DATA_DIR/Fe2O3.cif" ] && [ ! -f "$DATA/Fe2O3.cif" ]; then
        cp "$DATA_DIR/Fe2O3.cif" "$DATA/Fe2O3.cif"
    fi
    if [ -d "$DATA_DIR/ensemble_structs" ] && [ ! -d "$DATA/ensemble_structs" ]; then
        cp -R "$DATA_DIR/ensemble_structs" "$DATA/ensemble_structs"
    fi

    PROMPT="$(eval "printf '%s' \"$PROMPT_TEMPLATE\"")"

    echo "=== Processing: $NAME ==="
    echo "  Model:      $MODEL"
    echo "  Start time: $(date +%H:%M:%S)"
    echo "  Output:     $OUT"
    echo "  Work:       $WORK"
    echo "  Data:       $DATA"
    echo "  Prompt:     $PROMPT"

    set +e
    "$AGENT_PYTHON" -m agent_io_tracing.adapters.chemgraph.launcher \
        "$WORK" "$OUT" \
        --model "$MODEL" \
        --prompt "$PROMPT" \
        > "$OUT/chemgraph.log" 2>&1 &
    AGENT_PID=$!
    kill -STOP "$AGENT_PID" >/dev/null 2>&1 || true

    READY_FIFO="$OUT/bcc.ready.fifo"
    rm -f "$READY_FIFO"
    mkfifo "$READY_FIFO"

    BCC_NET_FLAG="${BCC_INCLUDE_NET:-1}"
    if [ "$BCC_NET_FLAG" = "1" ] || [ "$BCC_NET_FLAG" = "true" ]; then
        NET_ARG="--include-net"
    else
        NET_ARG="--no-include-net"
    fi

    "$TRACER_PYTHON" -m agent_io_tracing.tracing.bcc_tracer \
        --root-pid "$AGENT_PID" \
        --output "$OUT/ebpf_events.log" \
        $NET_ARG \
        --ready-fd 3 \
        3>"$READY_FIFO" \
        >"$OUT/bcc.out" 2>"$OUT/bcc.err" &
    TRACER_PID=$!

    READY_MSG=""
    if read -r READY_MSG <"$READY_FIFO"; then
        :
    fi
    rm -f "$READY_FIFO"

    if [ "$READY_MSG" != "ready" ]; then
        echo "  Warning: tracer did not signal readiness; continuing anyway" >&2
    fi
    kill -CONT "$AGENT_PID" >/dev/null 2>&1 || true

    wait "$AGENT_PID"
    EXIT_CODE=$?

    sudo kill -INT "$TRACER_PID" >/dev/null 2>&1 || true
    wait "$TRACER_PID" >/dev/null 2>&1 || true
    set -e

    echo "  End time:   $(date +%H:%M:%S)"
    echo "  Exit code:  $EXIT_CODE"
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
    sed 's/^/    /' "$ws_out/parse.log" | tail -8 || true

    if [ -f "$ws_out/parsed.json" ] && [ -f "$ws_out/pi_events.jsonl" ]; then
        "$POST_PYTHON" -m agent_io_tracing.analysis.summary "$ws_out" \
            > "$ws_out/summarize.log" 2>&1
        SUM_RC=$?
        [ $SUM_RC -ne 0 ] && failed_step="${failed_step:+$failed_step,}summarize"

        "$POST_PYTHON" -m agent_io_tracing.analysis.parallelism "$ws_out" \
            > "$ws_out/parallelism.log" 2>&1
        PAR_RC=$?
        [ $PAR_RC -ne 0 ] && failed_step="${failed_step:+$failed_step,}parallelism"

        "$POST_PYTHON" -m agent_io_tracing.viz.trace "$ws_out" \
            > "$ws_out/visualize.log" 2>&1
        VIZ_RC=$?
        [ $VIZ_RC -ne 0 ] && failed_step="${failed_step:+$failed_step,}visualize"

        [ -f "$ws_out/call_dag.html" ] && echo "    DAG HTML: $ws_out/call_dag.html"
        [ -f "$ws_out/parallelism_summary.json" ] && echo "    Metrics:  $ws_out/parallelism_summary.json"
        [ -f "$ws_out/visualizations/index.html" ] && echo "    Viz:      $ws_out/visualizations/index.html"
    else
        echo "    Skipping pi summary/viz (parsed.json or pi_events.jsonl missing)"
        failed_step="${failed_step:+$failed_step,}missing_logs"
    fi
    set -e

    if [ -n "$failed_step" ]; then
        echo "  Post-processing partial failure for $NAME: $failed_step"
        POST_FAIL_COUNT=$((POST_FAIL_COUNT + 1))
        POST_FAIL_NAMES+=("$NAME($failed_step)")
    fi
    echo ""
done

if [ "$POST_FAIL_COUNT" -gt 0 ]; then
    echo "Post-processing finished with $POST_FAIL_COUNT failed workload(s):"
    for n in "${POST_FAIL_NAMES[@]}"; do
        echo "  - $n"
    done
fi

chmod -R a+rX "$BASE_OUT" || true
if [ -n "${SUDO_UID:-}" ] && [ -n "${SUDO_GID:-}" ]; then
    chown -R "$SUDO_UID:$SUDO_GID" "$BASE_OUT" 2>/dev/null || true
    echo "Returned ownership of $BASE_OUT to ${SUDO_USER:-uid=$SUDO_UID}"
fi

echo "All done. Results in: $BASE_OUT"
echo "Look for: visualizations/index.html, call_dag.html, parallelism_summary.json"
