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

if [ "$(id -u)" -eq 0 ]; then
    echo "Error: run this script as the regular SSH user, not through sudo." >&2
    echo "The script elevates only the BCC tracer. Running the whole script as root writes SciLink's multi-gigabyte model cache to /root." >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CFG_DIR="$ROOT_DIR/config"
CONFIG_FILE="${CONFIG_FILE:-$CFG_DIR/config_scilink.env}"
CALLER_BASE_OUT="${BASE_OUT:-}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib_results.sh"
# TRACER_PYTHON / AGENT_PYTHON / POST_PYTHON come from config_scilink.env
# (sourced below).  Two interpreters are required because BCC bindings are
# tied to the system Python (3.6 on CentOS Stream 8) while SciLink needs ≥3.12.

# Source .env.scilink so API keys + AGENT_PYTHON (absolute, sudo-safe) survive
# `sudo -E`.  deploy_scilink_to_client.sh writes this file with hard-coded
# absolute paths so HOME=/root under sudo doesn't break python resolution.
# We INTENTIONALLY do NOT source the sibling .env (that's owned by the
# SRAgent harness and would override AGENT_PYTHON to the wrong venv).
if [ -f "$ROOT_DIR/.env.scilink" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.env.scilink"
    set +a
else
    echo "Warning: $ROOT_DIR/.env.scilink not found." >&2
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
if ! "$POST_PYTHON" -c "import markdown, tiktoken" >/dev/null 2>&1; then
    echo "Error: $POST_PYTHON is missing Markdown or tiktoken." >&2
    echo "Re-run scripts/deploy_scilink_to_client.sh before tracing." >&2
    exit 1
fi
SCILINK_BIN="$(dirname "$AGENT_PYTHON")/scilink"
if [ ! -x "$SCILINK_BIN" ]; then
    echo "Error: SciLink CLI not found: $SCILINK_BIN" >&2
    exit 1
fi

[ -d "$ROOT_DIR/src/agent_io_tracing" ] || {
    echo "Error: package not found: $ROOT_DIR/src/agent_io_tracing" >&2
    exit 1
}

if [ "${#WORKLOADS[@]}" -eq 0 ]; then
    echo "Error: WORKLOADS array is empty (configure config_scilink.env)" >&2
    exit 1
fi

BASE_OUT="${BASE_OUT:-$(default_lustre_results_root)/$(run_dir_name SciLink $(selected_task_names))}"
require_lustre_base_out "$BASE_OUT"
BASE_OUT="$(cd "$BASE_OUT" && pwd)"
WORK_DIR="$(mkdir -p "$WORK_DIR" && cd "$WORK_DIR" && pwd)"
DATA_DIR="$(mkdir -p "$DATA_DIR" && cd "$DATA_DIR" && pwd)"

echo "=== SciLink FS+Net Tracer (BCC) ==="
echo "Work dir:   $WORK_DIR"
echo "Output dir: $BASE_OUT"
echo "Workflow:   $SCRIPT_DIR"
echo "Workloads:  ${#WORKLOADS[@]} entries"
echo ""

validate_workload_selection "${WORKLOADS[@]}"

# Each entry in WORKLOADS is "<name>|<global_args>|<subcommand>|<args>|<prompt>".
#   - global_args: empty for SciLink (no global flags).
#   - subcommand : always 'analyze' for now.
#   - args       : forwarded to `scilink analyze`; --mode autonomous required.
#   - prompt     : fed via stdin as the agent's first message.  No '|' inside.
RUN_CELL_COUNT=0
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

    if ! workload_selected "$NAME"; then
        echo "Skipping: $NAME (not in RUN_WORKLOADS)"
        continue
    fi
    RUN_CELL_COUNT=$((RUN_CELL_COUNT + 1))

    # The no-cache arm writes to a distinct cell directory so it can share a
    # BASE_OUT with the cached arm and land in the same report without
    # overwriting it.  Only the prompt tagging differs; the workload is the same.
    CELL="$NAME"
    case "${SCILINK_NOCACHE:-0}" in
        1|true|TRUE|yes|YES) CELL="${NAME}_nocache" ;;
    esac
    # SCILINK_CELL_SUFFIX appends a further tag, so arms that differ in
    # something the cell name does not encode -- vendor, model, a repeat --
    # land in separate cells instead of overwriting each other.
    CELL="${CELL}${SCILINK_CELL_SUFFIX:-}"
    OUT="$BASE_OUT/$CELL"
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

    prepare_vllm_cache_for_cell
    set +e
    # Re-tokenize SRARGS so quoted multi-word inputs stay single argv elements
    # AND \$OUT / \$SCILINK_REPO / \$DATA expand to their per-workload values.
    # shellcheck disable=SC2294
    eval "set -- $SRARGS"
    SRARGS_ARRAY=("$@")

    if ! "$SCILINK_BIN" "$SUBCMD" --help >/dev/null 2>&1; then
        echo "Error: installed SciLink does not support subcommand '$SUBCMD'" >&2
        exit 1
    fi
    for path_flag in --data --metadata --data-dir --knowledge-dir --code-dir; do
        for ((arg_i = 0; arg_i < ${#SRARGS_ARRAY[@]}; arg_i++)); do
            if [ "${SRARGS_ARRAY[$arg_i]}" = "$path_flag" ]; then
                path_i=$((arg_i + 1))
                path_value="${SRARGS_ARRAY[$path_i]:-}"
                if [ -z "$path_value" ] || [ ! -e "$path_value" ]; then
                    echo "Error: $NAME requires existing $path_flag path: ${path_value:-<missing>}" >&2
                    exit 1
                fi
            fi
        done
    done

    # Build optional --pre flag (always empty for SciLink today, kept for
    # SRAgent-harness parity).  Use the `key=value` form so argparse doesn't
    # peel off a leading `--flag` as one of its own options.
    PRE_FLAG=()
    if [ -n "$GLOBAL_ARGS" ]; then
        PRE_FLAG=("--pre=$GLOBAL_ARGS")
    fi

    # Run SciLink under the litellm-based logger.  --prompt feeds the REPL
    # via stdin; --mode autonomous (inside SRARGS) silences follow-ups.
    "$AGENT_PYTHON" -m agent_io_tracing.adapters.scilink.launcher \
        "$WORK" "$OUT" "$SUBCMD" \
        --prompt "$PROMPT" "${PRE_FLAG[@]}" \
        -- "${SRARGS_ARRAY[@]}" \
        > "$OUT/scilink.log" 2>&1 &
    AGENT_PID=$!
    kill -STOP "$AGENT_PID" >/dev/null 2>&1 || true

    vllm_cache_snapshot "$OUT/vllm_cache_before.json"

    INSTRUMENTATION_LEVEL="ebpf"
    if [ "${COLLECT_LUSTRE_COUNTERS:-0}" = "1" ] || [ "${COLLECT_LUSTRE_COUNTERS:-0}" = "true" ]; then
        INSTRUMENTATION_LEVEL="ebpf+lustre-counters"
    fi

    # pull_agentic_run.sh treats manifest.json as the marker of a result cell,
    # so every cell must write one even though SciLink has no replay cache.
    "$POST_PYTHON" -m agent_io_tracing.experiments.run_manifest \
        --output "$OUT/manifest.json" \
        --workload "SciLink" \
        --task-id "$NAME" \
        --model "$SCILINK_MODEL" \
        --agent-count 1 \
        --pid "$AGENT_PID" \
        --data-dir "$DATA" \
        --work-dir "$WORK" \
        --output-dir "$OUT" \
        --instrumentation "$INSTRUMENTATION_LEVEL" \
        --extra-json "{\"scilink_repo\": \"$SCILINK_REPO\", \"subcommand\": \"$SUBCMD\", \"cell\": \"$CELL\", \"embedding_model\": \"${SCILINK_EMBEDDING_MODEL:-}\"}" \
        > "$OUT/manifest.log" 2>&1 || true

    BCC_NET_FLAG="${BCC_INCLUDE_NET:-1}"
    if [ "$BCC_NET_FLAG" = "1" ] || [ "$BCC_NET_FLAG" = "true" ]; then
        NET_ARG="--include-net"
    else
        NET_ARG="--no-include-net"
    fi

    sudo -n -E env "PYTHONPATH=$PYTHONPATH" "$TRACER_PYTHON" -m agent_io_tracing.tracing.bcc_tracer \
        --root-pid "$AGENT_PID" \
        --output "$OUT/ebpf_events.log" \
        $NET_ARG \
        >"$OUT/bcc.out" 2>"$OUT/bcc.err" &
    TRACER_PID=$!

    # sudo closes non-standard file descriptors on this CloudLab image, so the
    # --ready-fd FIFO handshake cannot be used for the privileged tracer.
    # Wait until the tracer creates the JSONL stream (meta record written) or
    # exits with an error before resuming the paused agent.
    if ! wait_for_trace_file "$TRACER_PID" "$OUT/ebpf_events.log"; then
        echo "  Warning: tracer did not create ebpf_events.log before agent resume" >&2
    fi
    kill -CONT "$AGENT_PID" >/dev/null 2>&1 || true

    wait "$AGENT_PID"
    EXIT_CODE=$?

    vllm_cache_snapshot "$OUT/vllm_cache_after.json"
    vllm_cache_delta "$OUT/vllm_cache_before.json" "$OUT/vllm_cache_after.json" \
        "$OUT/vllm_cache_realized.json"

    stop_tracer "$TRACER_PID"
    set -e

    echo "  End time:  $(date +%H:%M:%S)"
    echo "  Exit code: $EXIT_CODE"
    echo ""
done

if [ "$RUN_CELL_COUNT" -eq 0 ]; then
    echo "Error: RUN_WORKLOADS selected zero SciLink cells: ${RUN_WORKLOADS:-<empty>}" >&2
    echo "Check the workload names in config/config_scilink.env." >&2
    exit 2
fi

echo "=== Running parse and visualization ==="
# Each step wrapped with set +e so a single workload's post-processing
# failure does NOT abort the whole pass.
POST_FAIL_COUNT=0
POST_FAIL_NAMES=()

for ws_out in "$BASE_OUT"/*/; do
    [ -d "$ws_out" ] || continue
    NAME="$(basename "$ws_out")"
    WS_OUT_ABS="$(cd "$ws_out" && pwd)"
    LINEAGE_WORKLOAD_DATA="$DATA_DIR/$NAME"

    if [ ! -f "$ws_out/ebpf_events.log" ]; then
        echo "Skipping $NAME (no ebpf_events.log)"
        continue
    fi

    echo "Processing: $NAME"
    set +e
    failed_step=""
    LINEAGE_DATA_PATH_PREFIXES="$WS_OUT_ABS/scilink_session/:$WS_OUT_ABS/work/:$LINEAGE_WORKLOAD_DATA/:$SCILINK_REPO/examples/" \
    LINEAGE_EXCLUDE_PATH_SUBSTRINGS="/.venv/" \
    "$POST_PYTHON" -m agent_io_tracing.analysis.postprocess "$ws_out" \
        --python "$POST_PYTHON" \
        --max-workers "${POSTPROCESS_MAX_WORKERS:-2}" \
        >"$ws_out/postprocess.log" 2>&1
    POSTPROCESS_RC=$?
    sed 's/^/    /' "$ws_out/postprocess.log" || true
    [ "$POSTPROCESS_RC" -ne 0 ] && failed_step="postprocess"
    set -e

    if [ -n "$failed_step" ]; then
        echo "  ⚠  Post-processing partial failure for $NAME: $failed_step"
        POST_FAIL_COUNT=$((POST_FAIL_COUNT + 1))
        POST_FAIL_NAMES+=("$NAME($failed_step)")
    fi
    echo ""
done

set +e
run_kvcache_report "$POST_PYTHON" "$BASE_OUT"
KVCACHE_RC=$?
set -e
if [ "$KVCACHE_RC" -ne 0 ]; then
    POST_FAIL_COUNT=$((POST_FAIL_COUNT + 1))
    POST_FAIL_NAMES+=("kvcache_report")
fi

if [ "$POST_FAIL_COUNT" -gt 0 ]; then
    echo "Post-processing finished with $POST_FAIL_COUNT failed workload(s):"
    for n in "${POST_FAIL_NAMES[@]}"; do
        echo "  - $n"
    done
    echo "(See <workload>/{parse,summarize,visualize}.log for details)"
fi

return_results_ownership "$BASE_OUT"

echo "All done. Results in: $BASE_OUT"
echo "Per-workload outputs now stay under: $BASE_OUT/<workload>/"
echo "Layout:"
echo "  $BASE_OUT/<workload>/                    trace root"
echo "  $BASE_OUT/<workload>/visualizations/     HTML/PNG visualization output"
echo "  $BASE_OUT/<workload>/scilink_session/    SciLink session state"
echo "  $BASE_OUT/<workload>/work/               SciLink cwd/generated artifacts"
echo "Look for: kvcache_report.html, visualizations/index.html, call_dag.html, call_tree.txt, parallelism_summary.json"
