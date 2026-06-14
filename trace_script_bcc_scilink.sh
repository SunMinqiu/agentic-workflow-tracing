#!/bin/bash
#
# Trace SciLink (https://github.com/ziatdinovmax/SciLink) using eBPF/BCC.
#
# Mirrors trace_script_bcc_sragent.sh: STOP the agent, start the tracer, wait
# for 'ready' on a FIFO, CONT the agent, wait, then run parse + visualize.
#
# Two differences from the SRAgent version:
#   - SciLink's CLI is interactive (input()-based REPL).  analyze_codebase_scilink.py
#     pre-loads sys.stdin with the workload's prompt and relies on EOFError
#     to terminate the REPL.  --mode autonomous (in each WORKLOAD's args)
#     suppresses follow-up prompts.
#   - WORKLOADS entries have a 5th '|prompt' field.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/config_scilink.env}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$SCRIPT_DIR}"
CALLER_BASE_OUT="${BASE_OUT:-}"
# TRACER_PYTHON / AGENT_PYTHON / POST_PYTHON come from config_scilink.env
# (sourced below).  Two interpreters are required because BCC bindings are
# tied to the system Python (3.6 on CentOS Stream 8) while SciLink needs ≥3.12.

# Source .env.scilink so API keys + AGENT_PYTHON (absolute, sudo-safe) survive
# `sudo -E`.  deploy_scilink_to_client.sh writes this file with hard-coded
# absolute paths so HOME=/root under sudo doesn't break python resolution.
# We INTENTIONALLY do NOT source the sibling .env (that's owned by the
# SRAgent harness and would override AGENT_PYTHON to the wrong venv).
if [ -f "$SCRIPT_DIR/.env.scilink" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env.scilink"
    set +a
else
    echo "Warning: $SCRIPT_DIR/.env.scilink not found." >&2
    echo "         Re-run deploy_scilink_to_client.sh on your laptop to generate it." >&2
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
    echo "Error: WORK_DIR is not set in config_scilink.env" >&2
    exit 1
fi
mkdir -p "$WORK_DIR"

if [ -z "${DATA_DIR:-}" ]; then
    echo "Error: DATA_DIR is not set in config_scilink.env" >&2
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
    echo "Run deploy_scilink_to_client.sh to build the uv venv." >&2
    exit 1
fi
if ! "$AGENT_PYTHON" -c "import scilink, litellm" >/dev/null 2>&1; then
    echo "Error: scilink / litellm not importable by $AGENT_PYTHON" >&2
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/analyze_codebase_scilink.py" ]; then
    echo "Error: analyze_codebase_scilink.py not found in $SCRIPT_DIR" >&2
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/litellm_tool_logger.py" ]; then
    echo "Error: litellm_tool_logger.py not found in $SCRIPT_DIR" >&2
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

if [ ! -f "$SCRIPT_DIR/compute_parallelism.py" ]; then
    echo "Error: compute_parallelism.py not found in $SCRIPT_DIR" >&2
    exit 1
fi

if [ ! -f "$ANALYSIS_DIR/visualize_strace.py" ]; then
    echo "Error: visualization script not found in $ANALYSIS_DIR" >&2
    exit 1
fi

if [ "${#WORKLOADS[@]}" -eq 0 ]; then
    echo "Error: WORKLOADS array is empty (configure config_scilink.env)" >&2
    exit 1
fi

BASE_OUT="${BASE_OUT:-$SCRIPT_DIR/traces/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$BASE_OUT"
BASE_OUT="$(cd "$BASE_OUT" && pwd)"
WORK_DIR="$(mkdir -p "$WORK_DIR" && cd "$WORK_DIR" && pwd)"
DATA_DIR="$(mkdir -p "$DATA_DIR" && cd "$DATA_DIR" && pwd)"

echo "=== SciLink FS+Net Tracer (BCC) ==="
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

# Each entry in WORKLOADS is "<name>|<global_args>|<subcommand>|<args>|<prompt>".
#   - global_args: empty for SciLink (no global flags).
#   - subcommand : always 'analyze' for now.
#   - args       : forwarded to `scilink analyze`; --mode autonomous required.
#   - prompt     : fed via stdin as the agent's first message.  No '|' inside.
for entry in "${WORKLOADS[@]}"; do
    NAME="${entry%%|*}"
    REST1="${entry#*|}"
    GLOBAL_ARGS="${REST1%%|*}"
    REST2="${REST1#*|}"
    SUBCMD="${REST2%%|*}"
    REST3="${REST2#*|}"
    SRARGS="${REST3%%|*}"
    PROMPT="${REST3#*|}"

    if [ -z "$NAME" ] || [ -z "$SUBCMD" ] || [ -z "$PROMPT" ]; then
        echo "Skip malformed workload entry (need 5 |-separated fields incl. prompt): '$entry'" >&2
        continue
    fi

    if ! should_run_workload "$NAME"; then
        echo "Skipping: $NAME (not in RUN_WORKLOADS)"
        continue
    fi

    OUT="$BASE_OUT/$NAME"
    mkdir -p "$OUT"
    OUT="$(cd "$OUT" && pwd)"

    # Keep every workload self-contained under OUT, but do not dump generated
    # SciLink working files into the trace root.  Root stays for trace /
    # post-processing artefacts; SciLink cwd goes under work/ and session
    # state goes under scilink_session/.
    WORK="$OUT/work"
    mkdir -p "$WORK"
    WORK="$(cd "$WORK" && pwd)"

    DATA="$DATA_DIR/$NAME"
    mkdir -p "$DATA" "$DATA/tmp"
    DATA="$(cd "$DATA" && pwd)"
    # Export OUT/DATA so $OUT / $DATA inside the WORKLOADS entry resolve
    # in the `eval set --` below (config_scilink.env left them deferred).
    export OUT DATA
    export TMPDIR="$DATA/tmp"

    echo "=== Processing: $NAME (scilink $SUBCMD) ==="
    [ -n "$GLOBAL_ARGS" ] && echo "  Global flags: $GLOBAL_ARGS"
    echo "  Args:        $SRARGS"
    echo "  Prompt:      $PROMPT"
    echo "  Start time:  $(date +%H:%M:%S)"
    echo "  Output:      $OUT"
    echo "  Trace root:  $OUT"
    echo "  Work (local):$WORK"
    echo "  Session dir: $OUT/scilink_session"
    echo "  Data (Lustre):$DATA"

    set +e
    # Re-tokenize SRARGS so quoted multi-word inputs stay single argv elements
    # AND \$OUT / \$SCILINK_REPO / \$DATA expand to their per-workload values.
    # shellcheck disable=SC2294
    eval "set -- $SRARGS"
    SRARGS_ARRAY=("$@")

    # Build optional --pre flag (always empty for SciLink today, kept for
    # SRAgent-harness parity).  Use the `key=value` form so argparse doesn't
    # peel off a leading `--flag` as one of its own options.
    PRE_FLAG=()
    if [ -n "$GLOBAL_ARGS" ]; then
        PRE_FLAG=("--pre=$GLOBAL_ARGS")
    fi

    # Run SciLink under the litellm-based logger.  --prompt feeds the REPL
    # via stdin; --mode autonomous (inside SRARGS) silences follow-ups.
    "$AGENT_PYTHON" "$SCRIPT_DIR/analyze_codebase_scilink.py" \
        "$WORK" "$OUT" "$SUBCMD" \
        --prompt "$PROMPT" "${PRE_FLAG[@]}" \
        -- "${SRARGS_ARRAY[@]}" \
        > "$OUT/scilink.log" 2>&1 &
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
# Each step wrapped with set +e so a single workload's post-processing
# failure does NOT abort the whole pass.
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

            echo "  Computing DAG + parallelism metrics..."
            "$POST_PYTHON" "$SCRIPT_DIR/compute_parallelism.py" "$ws_out" \
                > "$ws_out/parallelism.log" 2>&1
            PAR_RC=$?
            sed 's/^/    /' "$ws_out/parallelism.log" || true
            [ $PAR_RC -ne 0 ] && failed_step="${failed_step:+$failed_step,}parallelism"
            if [ $PAR_RC -eq 0 ]; then
                [ -f "$ws_out/call_dag.html" ] && echo "    DAG HTML: $ws_out/call_dag.html"
                [ -f "$ws_out/parallelism_summary.json" ] && echo "    Metrics:  $ws_out/parallelism_summary.json"
            fi
        else
            echo "  Skipping pi summary (pi_events.jsonl or tool_calls.log missing)"
        fi

        echo "  Generating visualizations..."
        "$POST_PYTHON" "$ANALYSIS_DIR/visualize_strace.py" "$ws_out" \
            > "$ws_out/visualize.log" 2>&1
        VIZ_RC=$?
        sed 's/^/    /' "$ws_out/visualize.log" || true
        [ $VIZ_RC -ne 0 ] && failed_step="${failed_step:+$failed_step,}visualize"
        if [ $VIZ_RC -eq 0 ]; then
            [ -f "$ws_out/visualizations/index.html" ] && echo "    Viz index: $ws_out/visualizations/index.html"
            [ -f "$ws_out/visualizations/agent_timeline.html" ] && echo "    Agent timeline: $ws_out/visualizations/agent_timeline.html"
        fi
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

# When invoked via `sudo -E`, root owns every artefact.  Hand ownership
# back to the invoking user.  No-op if not running under sudo.
if [ -n "${SUDO_UID:-}" ] && [ -n "${SUDO_GID:-}" ]; then
    chown -R "$SUDO_UID:$SUDO_GID" "$BASE_OUT" 2>/dev/null || true
    echo "Returned ownership of $BASE_OUT to ${SUDO_USER:-uid=$SUDO_UID}"
fi

echo "All done. Results in: $BASE_OUT"
echo "Per-workload outputs now stay under: $BASE_OUT/<workload>/"
echo "Layout:"
echo "  $BASE_OUT/<workload>/                    trace root"
echo "  $BASE_OUT/<workload>/visualizations/     HTML/PNG visualization output"
echo "  $BASE_OUT/<workload>/scilink_session/    SciLink session state"
echo "  $BASE_OUT/<workload>/work/               SciLink cwd/generated artifacts"
echo "Look for: visualizations/index.html, visualizations/agent_timeline.html, call_dag.html, call_tree.txt, parallelism_summary.json"
