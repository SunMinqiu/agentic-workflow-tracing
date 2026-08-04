#!/usr/bin/env bash
#
# Rebuild visualizations for a completed fanout run on the CloudLab client,
# then rsync the results back to this local workspace.
#
# Usage:
#   source cloudlab_env.sh
#   bash scripts/visualize_and_pull_fanout_remote.sh
#
# Overrides:
#   REMOTE_RUN=/mnt/lustrefs/Minqiu/pi-ebpf-tracing-handoff/results/fanout_20260623_231616
#   LOCAL_OUT=results/fanout_20260623_231616
#   JOBS=4
#   SKIP_CODE_SYNC=1

set -euo pipefail

SSH_USER="${SSH_USER:-Minqiu}"
CLIENT_NODE="${CLIENT_NODE:?source cloudlab_env.sh first}"
REMOTE_HARNESS_NAME="${REMOTE_HARNESS_NAME:-pi-ebpf-tracing-handoff}"
RESULTS_OWNER="${RESULTS_OWNER:-${SSH_USER:-Minqiu}}"
REMOTE_RUN="${REMOTE_RUN:-/mnt/lustrefs/$RESULTS_OWNER/pi-ebpf-tracing-handoff/results/fanout_20260623_231616}"
LOCAL_OUT="${LOCAL_OUT:-results/$(basename "$REMOTE_RUN")}"
JOBS="${JOBS:-4}"
SKIP_CODE_SYNC="${SKIP_CODE_SYNC:-0}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Remote client: $SSH_USER@$CLIENT_NODE"
echo "==> Remote run:    $REMOTE_RUN"
echo "==> Local out:     $ROOT_DIR/$LOCAL_OUT"
echo "==> Parallel jobs: $JOBS"

if [ "$SKIP_CODE_SYNC" != "1" ]; then
    echo "==> Syncing current src/ and scripts/ to remote harness"
    rsync -az --delete \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        "$ROOT_DIR/src/" \
        "$SSH_USER@$CLIENT_NODE:$REMOTE_HARNESS_NAME/src/"
    rsync -az \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        "$ROOT_DIR/scripts/" \
        "$SSH_USER@$CLIENT_NODE:$REMOTE_HARNESS_NAME/scripts/"
fi

echo "==> Remote parallel post-processing + visualization"
ssh -T "$SSH_USER@$CLIENT_NODE" \
    "REMOTE_RUN='$REMOTE_RUN' JOBS='$JOBS' REMOTE_HARNESS_NAME='$REMOTE_HARNESS_NAME' bash -s" <<'REMOTE'
set -euo pipefail

cd "$HOME/$REMOTE_HARNESS_NAME"
if [ -f .env.genomas ]; then
    # shellcheck disable=SC1091
    source .env.genomas
fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
POST_PYTHON="${POST_PYTHON:-python3}"

if [ ! -d "$REMOTE_RUN" ]; then
    echo "ERROR: remote run not found: $REMOTE_RUN" >&2
    exit 2
fi

echo "Remote python: $POST_PYTHON"
"$POST_PYTHON" - <<'PY'
import matplotlib, pandas, plotly
print("viz deps ok")
PY

CELL_LIST="$(mktemp)"
trap 'rm -f "$CELL_LIST"' EXIT
find "$REMOTE_RUN" -mindepth 1 -maxdepth 1 -type d \
    ! -name figures \
    -exec test -f '{}/manifest.json' ';' \
    -print | sort > "$CELL_LIST"

if [ ! -s "$CELL_LIST" ]; then
    echo "ERROR: no result cells with manifest.json under $REMOTE_RUN" >&2
    exit 3
fi

echo "==> Remote raw-trace integrity check"
INTEGRITY_FAIL=0
while IFS= read -r cell; do
    for required in \
        ebpf_events.log \
        parsed.json \
        pi_events.jsonl \
        tool_calls.log \
        manifest.json
    do
        if [ ! -s "$cell/$required" ]; then
            echo "ERROR: missing or empty: $cell/$required" >&2
            INTEGRITY_FAIL=1
        fi
    done
done < "$CELL_LIST"
if [ "$INTEGRITY_FAIL" -ne 0 ]; then
    echo "ERROR: refusing to rebuild/pull an incomplete remote run" >&2
    exit 3
fi

echo "Cells:"
sed 's/^/  /' "$CELL_LIST"
echo

export POST_PYTHON
run_cell() {
    cell="$1"
    name="$(basename "$cell")"
    echo "[$name] start"

    # Each step is guarded so one failing step never aborts the others, the
    # other parallel cells, or (crucially) the rsync pull-back at the end.
    "$POST_PYTHON" -m agent_io_tracing.lineage.analyzer "$cell" > "$cell/lineage.log" 2>&1 || echo "[$name] lineage FAILED (see lineage.log)"
    "$POST_PYTHON" -m agent_io_tracing.analysis.per_run_io_char \
        --results "$cell" --runs . > "$cell/per_run_io_char.log" 2>&1 || echo "[$name] per-run figures FAILED (see per_run_io_char.log)"
    "$POST_PYTHON" -m agent_io_tracing.analysis.parallelism "$cell" > "$cell/parallelism.log" 2>&1 || echo "[$name] parallelism FAILED (see parallelism.log)"
    "$POST_PYTHON" -m agent_io_tracing.analysis.phase1_metrics "$cell" > "$cell/phase1_metrics.log" 2>&1 || echo "[$name] phase1 FAILED (see phase1_metrics.log)"
    "$POST_PYTHON" -m agent_io_tracing.viz.trace "$cell" > "$cell/visualize.log" 2>&1 || echo "[$name] visualize FAILED (see visualize.log)"

    echo "[$name] done"
}
export -f run_cell

# `|| true`: a non-zero from any cell must not abort the run under `set -e`,
# so the run-level figures and the pull-back below always execute.
xargs -P "$JOBS" -n 1 bash -c 'run_cell "$0"' < "$CELL_LIST" || true

echo "==> Run-level fanout figures + index"
"$POST_PYTHON" -m agent_io_tracing.viz.fanout_input_sizes "$REMOTE_RUN" > "$REMOTE_RUN/input_sizes.log" 2>&1 || echo "make_fanout_input_sizes FAILED (see input_sizes.log)"
"$POST_PYTHON" -m agent_io_tracing.viz.fanout_plot "$REMOTE_RUN" > "$REMOTE_RUN/plot_fanout.log" 2>&1 || echo "plot_fanout FAILED (see plot_fanout.log)"
"$POST_PYTHON" -m agent_io_tracing.viz.fanout_index "$REMOTE_RUN" > "$REMOTE_RUN/make_fanout_index.log" 2>&1 || echo "make_fanout_index FAILED (see make_fanout_index.log)"

echo "==> Remote derived-output integrity check"
INTEGRITY_FAIL=0
while IFS= read -r cell; do
    for required in \
        phase1_metrics.json \
        parallelism_summary.json \
        lineage/artifacts.csv \
        lineage/io_summary.json \
        visualizations/file_access_volume.png \
        visualizations/rw_asymmetry.png \
        visualizations/index.html
    do
        if [ ! -s "$cell/$required" ]; then
            echo "ERROR: missing or empty after post-processing: $cell/$required" >&2
            INTEGRITY_FAIL=1
        fi
    done
done < "$CELL_LIST"
for required in \
    figures/fanout_tidy.csv \
    index.html
do
    if [ ! -s "$REMOTE_RUN/$required" ]; then
        echo "ERROR: missing or empty run-level output: $REMOTE_RUN/$required" >&2
        INTEGRITY_FAIL=1
    fi
done
if [ "$INTEGRITY_FAIL" -ne 0 ]; then
    echo "ERROR: refusing to pull an incomplete post-processing result" >&2
    exit 4
fi

echo "==> Output check"
find "$REMOTE_RUN" -maxdepth 3 \( \
    -path '*/visualizations/index.html' -o \
    -path '*/lineage/io_summary.json' -o \
    -path '*/call_dag.html' -o \
    -path '*/phase1_metrics.json' -o \
    -path '*/parallelism_summary.json' -o \
    -path '*/figures/fanout_tidy.csv' -o \
    -path '*/figures/input_files_tidy.csv' -o \
    -path '*/figures/input_size_distribution.png' -o \
    -name index.html \
\) -print | sort
REMOTE

echo "==> Pulling results back"
mkdir -p "$ROOT_DIR/$LOCAL_OUT"
rsync -az --progress \
    --checksum \
    --partial \
    --exclude 'work/' \
    --exclude 'bcc.out' \
    --exclude 'bcc.err' \
    "$SSH_USER@$CLIENT_NODE:$REMOTE_RUN/" \
    "$ROOT_DIR/$LOCAL_OUT/"

echo "==> Local pull integrity check"
REMOTE_CELLS_FILE="$(mktemp)"
LOCAL_CELLS_FILE="$(mktemp)"
cleanup_integrity_files() {
    rm -f "$REMOTE_CELLS_FILE" "$LOCAL_CELLS_FILE"
}
trap cleanup_integrity_files EXIT

ssh -T "$SSH_USER@$CLIENT_NODE" \
    "find '$REMOTE_RUN' -mindepth 1 -maxdepth 1 -type d ! -name figures -exec test -f '{}/manifest.json' ';' -printf '%f\\n' | sort" \
    > "$REMOTE_CELLS_FILE"
find "$ROOT_DIR/$LOCAL_OUT" -mindepth 1 -maxdepth 1 -type d \
    ! -name figures \
    -exec test -f '{}/manifest.json' ';' \
    -print | while IFS= read -r cell; do basename "$cell"; done \
    | sort > "$LOCAL_CELLS_FILE"

if ! diff -u "$REMOTE_CELLS_FILE" "$LOCAL_CELLS_FILE"; then
    echo "ERROR: local and remote cell sets differ after rsync" >&2
    exit 5
fi

INTEGRITY_FAIL=0
while IFS= read -r name; do
    cell="$ROOT_DIR/$LOCAL_OUT/$name"
    for required in \
        ebpf_events.log \
        parsed.json \
        pi_events.jsonl \
        tool_calls.log \
        manifest.json \
        phase1_metrics.json \
        parallelism_summary.json \
        lineage/artifacts.csv \
        lineage/io_summary.json \
        visualizations/file_access_volume.png \
        visualizations/rw_asymmetry.png \
        visualizations/index.html
    do
        if [ ! -s "$cell/$required" ]; then
            echo "ERROR: local file missing or empty after rsync: $cell/$required" >&2
            INTEGRITY_FAIL=1
        fi
    done
done < "$LOCAL_CELLS_FILE"
for required in figures/fanout_tidy.csv index.html; do
    if [ ! -s "$ROOT_DIR/$LOCAL_OUT/$required" ]; then
        echo "ERROR: local run-level output missing or empty: $ROOT_DIR/$LOCAL_OUT/$required" >&2
        INTEGRITY_FAIL=1
    fi
done
if [ "$INTEGRITY_FAIL" -ne 0 ]; then
    echo "ERROR: pull completed but local integrity verification failed" >&2
    exit 6
fi

cleanup_integrity_files
trap - EXIT

echo
echo "Done. Remote preflight, checksum transfer, and local integrity checks passed."
echo "Open:"
echo "  $ROOT_DIR/$LOCAL_OUT/index.html"
echo "  $ROOT_DIR/$LOCAL_OUT/base/visualizations/index.html"
