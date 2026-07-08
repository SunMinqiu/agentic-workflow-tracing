#!/bin/bash
#
# Trace pi-coding-agent filesystem I/O using eBPF/BCC on a CloudLab client node.
# Repositories are expected to live on a mounted Lustre path.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CFG_DIR="$ROOT_DIR/config"
CONFIG_FILE="${CONFIG_FILE:-$CFG_DIR/config.env}"
PI_ANALYZE_ARGS="${PI_ANALYZE_ARGS:-}"
CALLER_BASE_OUT="${BASE_OUT:-}"
AGENT_PYTHON="${AGENT_PYTHON:-/mnt/lus_fs/software/views/piostack/bin/python3}"
POST_PYTHON="${POST_PYTHON:-python3}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib_results.sh"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config file not found: $CONFIG_FILE" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

# If the caller provided BASE_OUT (e.g. comparison runner), keep it.
if [ -n "$CALLER_BASE_OUT" ]; then
    BASE_OUT="$CALLER_BASE_OUT"
fi

if [ -z "${CASES_DIR:-}" ]; then
    echo "Error: CASES_DIR is not set in config.env" >&2
    exit 1
fi

if [ ! -d "$CASES_DIR" ]; then
    echo "Error: CASES_DIR does not exist: $CASES_DIR" >&2
    exit 1
fi

if ! command -v pi >/dev/null 2>&1; then
    echo "Error: pi CLI not found in PATH. Install with 'npm install -g @mariozechner/pi-coding-agent'" >&2
    exit 1
fi

if ! pi --version >/dev/null 2>&1; then
    echo "Error: pi CLI is installed but not runnable on this host." >&2
    echo "Likely cause: Node.js is too old (pi requires Node 20+)." >&2
    echo "Fix options:" >&2
    echo "  1) Upgrade Node.js (recommended: Node 20+), then reinstall @mariozechner/pi-coding-agent" >&2
    echo "  2) Ensure your shell PATH points at the correct global npm bin directory" >&2
    exit 1
fi

[ -d "$ROOT_DIR/src/agent_io_tracing" ] || {
    echo "Error: package not found: $ROOT_DIR/src/agent_io_tracing" >&2
    exit 1
}

BASE_OUT="${BASE_OUT:-$(default_lustre_results_root)/$(date +%Y%m%d_%H%M%S)}"
require_lustre_base_out "$BASE_OUT"
BASE_OUT="$(cd "$BASE_OUT" && pwd)"

echo "=== CloudLab Lustre Agent FS Tracer (BCC + Pi) ==="
echo "Cases directory: $CASES_DIR"
echo "Output directory: $BASE_OUT"
echo "Workflow directory: $SCRIPT_DIR"
echo "Package path: $ROOT_DIR/src"
echo ""

IFS=',' read -r -a RUN_REPO_ARRAY <<< "${RUN_REPOS:-}"

should_run_repo() {
    local name="$1"
    if [ "${#RUN_REPO_ARRAY[@]}" -eq 0 ] || [ -z "${RUN_REPO_ARRAY[0]}" ]; then
        return 0
    fi
    local item
    for item in "${RUN_REPO_ARRAY[@]}"; do
        # Trim optional surrounding spaces
        item="$(echo "$item" | xargs)"
        if [ -n "$item" ] && [ "$item" = "$name" ]; then
            return 0
        fi
    done
    return 1
}

for repo in "$CASES_DIR"/*/; do
    [ -d "$repo" ] || continue
    REPO_NAME="$(basename "$repo")"

    if ! should_run_repo "$REPO_NAME"; then
        echo "Skipping: $REPO_NAME (not in RUN_REPOS)"
        continue
    fi

    OUT="$BASE_OUT/$REPO_NAME"
    mkdir -p "$OUT"

    echo "=== Processing: $REPO_NAME ==="
    echo "  Start time: $(date +%H:%M:%S)"

    set +e
    # Start the agent paused so no tool activity runs before probes are active.
    "$AGENT_PYTHON" -m agent_io_tracing.adapters.pi.launcher "$repo" "$OUT" $PI_ANALYZE_ARGS \
        >"$OUT/pi.out" 2>"$OUT/pi.err" &
    AGENT_PID=$!
    kill -STOP "$AGENT_PID" >/dev/null 2>&1 || true

    READY_FIFO="$OUT/bcc.ready.fifo"
    rm -f "$READY_FIFO"
    mkfifo "$READY_FIFO"

    # Start tracer and wait for explicit readiness before continuing agent.
    "$AGENT_PYTHON" -m agent_io_tracing.tracing.bcc_tracer \
        --root-pid "$AGENT_PID" \
        --output "$OUT/ebpf_events.log" \
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

    echo "  End time: $(date +%H:%M:%S)"
    echo "  Exit code: $EXIT_CODE"
    echo "  Output: $OUT/"
    echo ""
done

echo "=== Running parse and visualization ==="
for repo_out in "$BASE_OUT"/*/; do
    [ -d "$repo_out" ] || continue
    REPO_NAME="$(basename "$repo_out")"

    if [ ! -f "$repo_out/ebpf_events.log" ]; then
        echo "Skipping $REPO_NAME (no ebpf_events.log)"
        continue
    fi

    echo "Processing: $REPO_NAME"
    echo "  Parsing eBPF logs..."
    "$POST_PYTHON" -m agent_io_tracing.parsing.ebpf \
        "$repo_out" \
        2>&1 | sed 's/^/    /'

    if [ -f "$repo_out/parsed.json" ]; then
        if [ -f "$repo_out/pi_events.jsonl" ] && [ -f "$repo_out/tool_calls.log" ]; then
            echo "  Summarizing pi events..."
            "$POST_PYTHON" -m agent_io_tracing.analysis.summary "$repo_out" 2>&1 | sed 's/^/    /'
        else
            echo "  Skipping pi summary (pi_events.jsonl or tool_calls.log missing)"
        fi

        echo "  Generating visualizations..."
        "$POST_PYTHON" -m agent_io_tracing.viz.trace "$repo_out" 2>&1 | sed 's/^/    /'
    else
        echo "  Skipping visualization (parsed.json not found)"
    fi
    echo ""
done

chmod -R a+rX "$BASE_OUT" || true
echo "All done. Results in: $BASE_OUT"
