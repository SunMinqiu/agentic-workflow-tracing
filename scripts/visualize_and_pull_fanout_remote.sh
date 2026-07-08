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
find "$REMOTE_RUN" -mindepth 1 -maxdepth 1 -type d \
    ! -name figures \
    -exec test -f '{}/parsed.json' ';' \
    -exec test -f '{}/pi_events.jsonl' ';' \
    -print | sort > "$CELL_LIST"

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
    "$POST_PYTHON" -m agent_io_tracing.analysis.parallelism "$cell" > "$cell/parallelism.log" 2>&1 || echo "[$name] parallelism FAILED (see parallelism.log)"
    "$POST_PYTHON" -m agent_io_tracing.analysis.phase1_metrics "$cell" > "$cell/phase1_metrics.log" 2>&1 || echo "[$name] phase1 FAILED (see phase1_metrics.log)"
    "$POST_PYTHON" -m agent_io_tracing.viz.trace "$cell" > "$cell/visualize.log" 2>&1 || echo "[$name] visualize FAILED (see visualize.log)"

    echo "[$name] done"
}
export -f run_cell

# `|| true`: a non-zero from any cell must not abort the run under `set -e`,
# so the run-level figures and the pull-back below always execute.
xargs -P "$JOBS" -n 1 bash -c 'run_cell "$0"' < "$CELL_LIST" || true
rm -f "$CELL_LIST"

echo "==> Run-level fanout figures + index"
"$POST_PYTHON" -m agent_io_tracing.viz.fanout_input_sizes "$REMOTE_RUN" > "$REMOTE_RUN/input_sizes.log" 2>&1 || echo "make_fanout_input_sizes FAILED (see input_sizes.log)"
"$POST_PYTHON" -m agent_io_tracing.viz.fanout_plot "$REMOTE_RUN" > "$REMOTE_RUN/plot_fanout.log" 2>&1 || echo "plot_fanout FAILED (see plot_fanout.log)"
"$POST_PYTHON" -m agent_io_tracing.viz.fanout_index "$REMOTE_RUN" > "$REMOTE_RUN/make_fanout_index.log" 2>&1 || echo "make_fanout_index FAILED (see make_fanout_index.log)"

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
    --exclude 'work/' \
    --exclude 'bcc.out' \
    --exclude 'bcc.err' \
    "$SSH_USER@$CLIENT_NODE:$REMOTE_RUN/" \
    "$ROOT_DIR/$LOCAL_OUT/"

echo
echo "Done."
echo "Open:"
echo "  $ROOT_DIR/$LOCAL_OUT/index.html"
echo "  $ROOT_DIR/$LOCAL_OUT/base/visualizations/index.html"
