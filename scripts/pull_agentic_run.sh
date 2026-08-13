#!/usr/bin/env bash
# Pull one agentic run back from the client node, verify it, and open the
# reports.  One KV report per task is always regenerated locally so its figures match
# local code rather than whatever version the cluster had deployed.
#
# Usage, from the repo root with cloudlab_env.sh already sourced:
#   bash scripts/pull_agentic_run.sh logs/genomas_A_c2_w1_20260806_101500.log
#   bash scripts/pull_agentic_run.sh /mnt/lustrefs/.../results/GenoMAS_A_c2_w1_20260806_101500
#
# The first form reads "Output dir:" out of $HOME/<log> on the client node.
# The second form takes the remote run directory directly, for runs that were
# not tee'd into a log.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

arg="${1:-}"
if [ -z "$arg" ]; then
    echo "Usage: bash scripts/pull_agentic_run.sh <remote_log|remote_run_dir>" >&2
    exit 2
fi
if [ -z "${SSH_USER:-}" ] || [ -z "${CLIENT_NODE:-}" ]; then
    echo "ERROR: SSH_USER and CLIENT_NODE are unset. Source cloudlab_env.sh first." >&2
    exit 2
fi

if [ "${arg#/}" != "$arg" ]; then
    remote_run="$arg"
else
    remote_run=$(ssh "$SSH_USER@$CLIENT_NODE" \
        "sed -n 's/^Output dir: //p' \"\$HOME/$arg\" | head -1")
    if [ -z "$remote_run" ]; then
        echo "ERROR: result path not found in $arg" >&2
        exit 1
    fi
fi

remote_cells=$(ssh "$SSH_USER@$CLIENT_NODE" \
    "find '$remote_run' -mindepth 2 -maxdepth 2 -type f -name manifest.json | wc -l")
if [ "$remote_cells" -eq 0 ]; then
    echo "ERROR: remote run contains zero result cells: $remote_run" >&2
    exit 1
fi

local_out="results/$(basename "$remote_run")"
mkdir -p "$local_out"

# Two passes.  parsed.json is the parser's expansion of ebpf_events.log and can
# reach tens of GB on a long run; it is regenerable from the raw log, so the
# second pass drops it past PARSED_MAX_SIZE.  Everything else, the raw log
# included, comes back whole.
PARSED_MAX_SIZE=100m
rsync -az --progress --checksum --partial \
    --exclude 'work/' --exclude 'bcc.out' --exclude 'parsed.json' \
    "$SSH_USER@$CLIENT_NODE:$remote_run/" "$local_out/"
rsync -az --progress --checksum --partial --max-size="$PARSED_MAX_SIZE" \
    --include '*/' --include 'parsed.json' --exclude '*' \
    "$SSH_USER@$CLIENT_NODE:$remote_run/" "$local_out/"

echo "Regenerating the KV report locally so figures match local code."
PYTHONPATH=src python -m agent_io_tracing.analysis.kvcache.report \
    --results "$local_out" --runs . --dump-prefixes || exit 1

echo "Rebuilding the results index over every local task."
PYTHONPATH=src python -m agent_io_tracing.analysis.results_index \
    --results results || exit 1

failed=0
cells=0
for cell in "$local_out"/*; do
    [ -f "$cell/manifest.json" ] || continue
    cells=$((cells + 1))

    # Report why the cell is degraded before listing what it is missing.  A
    # cell whose LLM calls were rejected produces a clean I/O trace and an
    # empty tool_calls.log, so the file-level errors below read as a plotting
    # bug unless the cause is stated first.
    PYTHONPATH=src python -m agent_io_tracing.analysis.trace_quality_report "$cell" || failed=1

    # parsed.json is exempt from the required list: the size cap above skips the
    # big ones on purpose.  Say so, rather than letting it read as a lost file.
    if [ ! -s "$cell/parsed.json" ]; then
        echo "NOTE: $cell/parsed.json skipped, over $PARSED_MAX_SIZE on the client." \
             "Regenerate it from ebpf_events.log, or rsync that one file by hand."
    fi

    for required in \
        ebpf_events.log manifest.json pi_events.jsonl tool_calls.log \
        messages.jsonl kvcache_demand.json kvcache_logical.json \
        kvcache_report.md kvcache_report.html \
        phase1_metrics.json parallelism_summary.json trace_quality.json \
        lineage/artifacts.csv lineage/execution_unit_io.csv \
        visualizations/file_access_volume.png visualizations/rw_asymmetry.png \
        visualizations/request_size_rw_cdf.png \
        visualizations/byte_normalized_summary.png visualizations/index.html; do
        if [ ! -s "$cell/$required" ]; then
            echo "ERROR: missing or empty: $cell/$required" >&2
            failed=1
        fi
    done
    if [ ! -f "$cell/bcc.err" ] || ! tail -1 "$cell/bcc.err" | grep -q 'lost_events=0'; then
        echo "ERROR: lost-event check failed: $cell/bcc.err" >&2
        failed=1
    fi
done

if [ ! -s "results/index.html" ]; then
    echo "ERROR: missing or empty: results/index.html" >&2
    failed=1
fi

[ "$cells" -gt 0 ] || failed=1
if [ "$failed" -eq 0 ]; then
    open "$local_out"/*/visualizations/index.html
    open "$local_out"/*/kvcache_report.html
    open results/index.html
else
    echo "Result pull failed integrity checks; not opening incomplete output." >&2
    exit 1
fi
