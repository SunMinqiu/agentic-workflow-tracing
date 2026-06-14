#!/bin/bash
#
# Trace SRAgent (https://github.com/ArcInstitute/SRAgent) using eBPF/BCC.
#
# Mirrors trace_script_bcc_pi.sh: STOP the agent, start the tracer, wait for
# 'ready' on a FIFO, CONT the agent, wait, then run parse + visualize.
#
# Workload model (different from pi):
#   pi version  : loop over CASES_DIR/*/  (one repo = one trace)
#   sragent ver.: loop over WORKLOADS array in config_sragent.env
#                 (one subcommand+args entry = one trace)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/config_sragent.env}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$SCRIPT_DIR}"
CALLER_BASE_OUT="${BASE_OUT:-}"
# TRACER_PYTHON / AGENT_PYTHON / POST_PYTHON are populated by config_sragent.env
# (sourced below).  Two interpreters are required because BCC bindings are
# tied to the system Python (3.6 on CentOS Stream 8) while SRAgent needs ≥3.11.

# Source .env if present so API keys survive `sudo -E` (deploy_sragent_to_client.sh
# writes it).  set -a auto-exports everything sourced.
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env"
    set +a
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

if [ -z "${WORK_DIR:-}" ]; then
    echo "Error: WORK_DIR is not set in config_sragent.env" >&2
    exit 1
fi
mkdir -p "$WORK_DIR"

if [ -z "${DATA_DIR:-}" ]; then
    echo "Error: DATA_DIR is not set in config_sragent.env" >&2
    exit 1
fi
mkdir -p "$DATA_DIR" || {
    echo "Error: cannot create DATA_DIR=$DATA_DIR (Lustre not mounted?)" >&2
    exit 1
}

# --- Verify the two Python interpreters ---
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
    echo "Run deploy_sragent_to_client.sh to build the uv venv." >&2
    exit 1
fi
if ! "$AGENT_PYTHON" -c "import SRAgent" >/dev/null 2>&1; then
    echo "Error: SRAgent not importable by $AGENT_PYTHON" >&2
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/analyze_codebase_sragent.py" ]; then
    echo "Error: analyze_codebase_sragent.py not found in $SCRIPT_DIR" >&2
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/langchain_tool_logger.py" ]; then
    echo "Error: langchain_tool_logger.py not found in $SCRIPT_DIR" >&2
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/bcc_tracer.py" ] || [ ! -f "$SCRIPT_DIR/parse_ebpf.py" ]; then
    echo "Error: bcc tracer/parser scripts not found in $SCRIPT_DIR" >&2
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/summarize_pi_events.py" ]; then
    echo "Error: summarize_pi_events.py not found in $SCRIPT_DIR" >&2
    exit 1
fi

if [ ! -f "$ANALYSIS_DIR/visualize_strace.py" ]; then
    echo "Error: visualization script not found in $ANALYSIS_DIR" >&2
    exit 1
fi

if [ "${#WORKLOADS[@]}" -eq 0 ]; then
    echo "Error: WORKLOADS array is empty (configure config_sragent.env)" >&2
    exit 1
fi

BASE_OUT="${BASE_OUT:-$SCRIPT_DIR/traces/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$BASE_OUT"

echo "=== SRAgent FS+Net Tracer (BCC) ==="
echo "Work dir:   $WORK_DIR"
echo "Output dir: $BASE_OUT"
echo "Workflow:   $SCRIPT_DIR"
echo "Workloads:  ${#WORKLOADS[@]} entries"
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

# Each entry in WORKLOADS is "<name>|<global_args>|<subcommand>|<args>".
# - global_args are SRAgent flags BEFORE the subcommand (e.g. --no-summaries);
#   leave empty (just `||`) when no global flag is needed.
# - args are forwarded verbatim to `SRAgent <subcommand>`.
for entry in "${WORKLOADS[@]}"; do
    NAME="${entry%%|*}"
    REST="${entry#*|}"
    GLOBAL_ARGS="${REST%%|*}"
    REST2="${REST#*|}"
    SUBCMD="${REST2%%|*}"
    SRARGS="${REST2#*|}"

    if [ -z "$NAME" ] || [ -z "$SUBCMD" ]; then
        echo "Skip malformed workload entry: '$entry'" >&2
        continue
    fi

    if ! should_run_workload "$NAME"; then
        echo "Skipping: $NAME (not in RUN_WORKLOADS)"
        continue
    fi

    OUT="$BASE_OUT/$NAME"
    mkdir -p "$OUT"

    # Local cwd for the agent (cwd; ChromaDB / appdirs cache land here).
    WORK="$WORK_DIR/$NAME"
    mkdir -p "$WORK"

    # Lustre-side download dir.  WORKLOADS entries reference this via $DATA.
    # TMPDIR also points here so fastq-dump tempfiles go to Lustre, per the
    # placement standard.
    DATA="$DATA_DIR/$NAME"
    mkdir -p "$DATA" "$DATA/tmp"
    export DATA
    export TMPDIR="$DATA/tmp"

    echo "=== Processing: $NAME (SRAgent $SUBCMD) ==="
    [ -n "$GLOBAL_ARGS" ] && echo "  Global flags: $GLOBAL_ARGS"
    echo "  Args:        $SRARGS"
    echo "  Start time:  $(date +%H:%M:%S)"
    echo "  Output:      $OUT"
    echo "  Work (local):$WORK"
    echo "  Data (Lustre):$DATA"

    set +e
    # Re-tokenize SRARGS so quoted multi-word inputs stay single argv elements
    # AND \$DATA / $DATA expand to the per-workload Lustre dir.
    # shellcheck disable=SC2294
    eval "set -- $SRARGS"
    SRARGS_ARRAY=("$@")

    # Build optional --pre flag for analyze_codebase_sragent.py (only when
    # GLOBAL_ARGS is non-empty).  Use `--pre=value` (equals) so argparse
    # doesn't peel off a leading `--flag` as one of its own options.
    PRE_FLAG=()
    if [ -n "$GLOBAL_ARGS" ]; then
        PRE_FLAG=("--pre=$GLOBAL_ARGS")
    fi

    # Start the agent paused so no tool activity runs before probes are active.
    "$AGENT_PYTHON" "$SCRIPT_DIR/analyze_codebase_sragent.py" \
        "$WORK" "$OUT" "$SUBCMD" "${PRE_FLAG[@]}" -- "${SRARGS_ARRAY[@]}" \
        > "$OUT/sragent.log" 2>&1 &
    # `&>` style merge: stdout (Rich banner / final results) and stderr (agent
    # diagnostic + SRAgent rendering) end up in one file in time-correct order.
    # `cat sragent.log` ≈ what you'd see typing `SRAgent <sub>` in a real shell.
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

    # Start tracer with TRACER_PYTHON (system py3 with python3-bcc bindings),
    # wait for explicit readiness before continuing agent.
    "$TRACER_PYTHON" "$SCRIPT_DIR/bcc_tracer.py" \
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

    echo "  End time:  $(date +%H:%M:%S)"
    echo "  Exit code: $EXIT_CODE"
    echo ""
done

echo "=== Running parse and visualization ==="
# Each step in this loop is wrapped with set +e so a single workload's
# parse/summarize/visualize failure does NOT abort the whole post-processing
# pass.  Per-workload failures get logged and the loop moves on.
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

    echo "  Parsing eBPF logs..."
    "$POST_PYTHON" "$SCRIPT_DIR/parse_ebpf.py" \
        "$ws_out" \
        > "$ws_out/parse.log" 2>&1
    PARSE_RC=$?
    sed 's/^/    /' "$ws_out/parse.log" || true
    [ $PARSE_RC -ne 0 ] && failed_step="parse_ebpf"

    if [ -f "$ws_out/parsed.json" ]; then
        if [ -f "$ws_out/pi_events.jsonl" ] && [ -f "$ws_out/tool_calls.log" ]; then
            echo "  Summarizing pi-compat events..."
            "$POST_PYTHON" "$SCRIPT_DIR/summarize_pi_events.py" "$ws_out" \
                > "$ws_out/summarize.log" 2>&1
            SUM_RC=$?
            sed 's/^/    /' "$ws_out/summarize.log" || true
            [ $SUM_RC -ne 0 ] && failed_step="${failed_step:+$failed_step,}summarize"
        else
            echo "  Skipping pi summary (pi_events.jsonl or tool_calls.log missing)"
        fi

        echo "  Generating visualizations..."
        "$POST_PYTHON" "$ANALYSIS_DIR/visualize_strace.py" "$ws_out" \
            > "$ws_out/visualize.log" 2>&1
        VIZ_RC=$?
        sed 's/^/    /' "$ws_out/visualize.log" || true
        [ $VIZ_RC -ne 0 ] && failed_step="${failed_step:+$failed_step,}visualize"
    else
        echo "  Skipping visualization (parsed.json not found)"
        failed_step="${failed_step:+$failed_step,}no_parsed_json"
    fi
    set -e

    if [ -n "$failed_step" ]; then
        echo "  ⚠  Post-processing partial failure for $NAME: $failed_step"
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
    echo "(See <workload>/{parse,summarize,visualize}.log for details)"
fi

chmod -R a+rX "$BASE_OUT" || true

# When invoked via `sudo -E`, root owns every trace artefact, which makes it
# annoying to re-parse, re-visualize, scp, or even rm without sudo.  Hand
# ownership back to the invoking user.  No-op if not running under sudo.
if [ -n "${SUDO_UID:-}" ] && [ -n "${SUDO_GID:-}" ]; then
    chown -R "$SUDO_UID:$SUDO_GID" "$BASE_OUT" 2>/dev/null || true
    echo "Returned ownership of $BASE_OUT to ${SUDO_USER:-uid=$SUDO_UID}"
fi

echo "All done. Results in: $BASE_OUT"
