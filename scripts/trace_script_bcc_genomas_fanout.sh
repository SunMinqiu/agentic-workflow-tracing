#!/bin/bash
#
# GEO-only fanout study for GenoMAS (Phase-1 input-shape scaling).
#
# ONE harness runs BOTH axes from the image:
#   Axis 1 (cohort) : one high-cohort trait, C = 1,2,4,8 cohorts
#   Axis 2 (trait)  : T = 1,2,4,8 single-cohort traits
# The C=1 / T=1 point is shared (1 trait x 1 cohort) and run once as "base".
#
# Per cell it: stages a symlink data-root view (stage_geo_view.py) -> STOPs the
# agent -> starts the BCC tracer -> CONTs the agent -> runs the full
# parse/summarize/lineage/phase1 post-pipeline -> appends one row to
# fanout_summary.csv with the axis-relevant numbers.
#
# Mirrors trace_script_bcc_genomas.sh; only the cell-generation + staging and
# the summary columns differ.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CFG_DIR="$ROOT_DIR/config"
CONFIG_FILE="${CONFIG_FILE:-$CFG_DIR/config_genomas_fanout.env}"
CALLER_BASE_OUT="${BASE_OUT:-}"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib_results.sh"

if [ -f "$ROOT_DIR/.env.genomas" ]; then
    set -a; source "$ROOT_DIR/.env.genomas"; set +a
else
    echo "Warning: $ROOT_DIR/.env.genomas not found (run deploy_genomas_to_client.sh)." >&2
fi

[ -f "$CONFIG_FILE" ] || { echo "Error: config not found: $CONFIG_FILE" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CONFIG_FILE"
[ -n "$CALLER_BASE_OUT" ] && BASE_OUT="$CALLER_BASE_OUT"

# --- Verify paths and interpreters -----------------------------------------
BASE_OUT="${BASE_OUT:-$(default_lustre_results_root)/fanout_$(date +%Y%m%d_%H%M%S)}"
require_lustre_base_out "$BASE_OUT"
mkdir -p "$WORK_DIR" "$VIEW_ROOT"
BASE_OUT="$(cd "$BASE_OUT" && pwd)"

[ -x "$TRACER_PYTHON" ] || { echo "Error: TRACER_PYTHON=$TRACER_PYTHON not executable" >&2; exit 1; }
"$TRACER_PYTHON" -c "from bcc import BPF" >/dev/null 2>&1 || {
    echo "Error: $TRACER_PYTHON cannot import bcc.BPF (dnf install python3-bcc bcc-tools)" >&2; exit 1; }
[ -x "$AGENT_PYTHON" ] || { echo "Error: AGENT_PYTHON=$AGENT_PYTHON not executable" >&2; exit 1; }

[ -d "$ROOT_DIR/src/agent_io_tracing" ] || {
    echo "Error: package not found: $ROOT_DIR/src/agent_io_tracing" >&2
    exit 1
}

# --- Generate the cell list from the fanout knobs --------------------------
# Each cell: "name|axis|param|geospec|n_traits|smoke_traits_csv"
read -r -a SCT_ARR <<< "$SINGLE_COHORT_TRAITS"
CELLS=()

# Shared baseline (= C1 = T1).
CELLS+=("base|shared|1|${HIGH_COHORT_TRAIT}:1|1|${HIGH_COHORT_TRAIT}")

# Axis 1 — cohort scaling (skip C=1, covered by base).
for C in $COHORT_AXIS_C; do
    [ "$C" = "1" ] && continue
    CELLS+=("a1_c${C}|cohort|${C}|${HIGH_COHORT_TRAIT}:${C}|1|${HIGH_COHORT_TRAIT}")
done

# Axis 2 — trait fanout (skip T=1, covered by base).
for T in $TRAIT_AXIS_T; do
    [ "$T" = "1" ] && continue
    if [ "${#SCT_ARR[@]}" -lt "$T" ]; then
        echo "NOTE: skipping a2_t${T} — need $T single-cohort traits, have ${#SCT_ARR[@]}." \
             "Add more to SINGLE_COHORT_TRAITS." >&2
        continue
    fi
    geospec=""; csv=""
    for ((i=0; i<T; i++)); do
        geospec="${geospec:+$geospec,}${SCT_ARR[$i]}:1"
        csv="${csv:+$csv,}${SCT_ARR[$i]}"
    done
    CELLS+=("a2_t${T}|trait|${T}|${geospec}|${T}|${csv}")
done

# RUN_CELLS filter.
IFS=',' read -r -a RUN_NAME_ARRAY <<< "${RUN_CELLS:-}"
should_run() {
    [ "${#RUN_NAME_ARRAY[@]}" -eq 0 ] || [ -z "${RUN_NAME_ARRAY[0]}" ] && return 0
    local item; for item in "${RUN_NAME_ARRAY[@]}"; do
        [ "$(echo "$item" | xargs)" = "$1" ] && return 0; done
    return 1
}

echo "=== GenoMAS GEO-only Fanout study (BCC) ==="
echo "Repo:        $GENOMAS_REPO"
echo "Real data:   $DATA_DIR"
echo "View root:   $VIEW_ROOT"
echo "Output dir:  $BASE_OUT"
echo "Model:       $GENOMAS_MODEL  (live, --quick-test, 1 worker)"
echo "Cells:       ${#CELLS[@]}"
for c in "${CELLS[@]}"; do echo "   - ${c%%|*}  (${c})"; done
echo ""

# --- Run each cell ----------------------------------------------------------
CELL_IDX=0
for entry in "${CELLS[@]}"; do
    IFS='|' read -r NAME AXIS PARAM GEOSPEC NTRAITS SMOKE_CSV <<< "$entry"
    should_run "$NAME" || { echo "Skipping $NAME (not in RUN_CELLS)"; continue; }
    CELL_IDX=$((CELL_IDX + 1))

    OUT="$BASE_OUT/$NAME"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
    WORK="$OUT/work"; mkdir -p "$WORK"; WORK="$(cd "$WORK" && pwd)"
    VIEW="$VIEW_ROOT/$NAME"

    echo "=== Cell $CELL_IDX/${#CELLS[@]}: $NAME  axis=$AXIS param=$PARAM ==="
    echo "  Start: $(date +%H:%M:%S)   geospec=$GEOSPEC"

    # 1) Stage the symlink data-root view; skip the cell if data missing.
    if ! "$POST_PYTHON" -m agent_io_tracing.experiments.stage_geo_view \
            --src-root "$DATA_DIR" --dest "$VIEW" --geo "$GEOSPEC" \
            > "$OUT/stage.log" 2>&1; then
        echo "  ⚠  staging failed (data prerequisite not met) — see $OUT/stage.log; skipping"
        continue
    fi
    # Keep the stage manifest next to the cell outputs so the summary CSV can
    # report n_traits/n_cohorts (it lives in the view dir by default).
    cp "$VIEW/stage_manifest.json" "$OUT/stage_manifest.json" 2>/dev/null || true

    # Clear any prior GenoMAS checkpoint for this cell name. --version reuse
    # (the cell name) makes GenoMAS resume from output/<NAME>/.../completed_tasks.json
    # and silently skip all work, leaving an empty pi_events.jsonl. Output dir is
    # $GENOMAS_REPO/output/<NAME> (symlinked to Lustre).
    rm -rf "$GENOMAS_REPO/output/$NAME" 2>/dev/null || true

    # 2) Launch GenoMAS (STOPped) against the staged view.
    set +e
    "$AGENT_PYTHON" -m agent_io_tracing.adapters.genomas.launcher \
        "$WORK" "$OUT" \
        --data-root "$VIEW" \
        --model "$GENOMAS_MODEL" \
        --api 1 \
        --version "$NAME" \
        --quick-test \
        --parallel-mode none \
        --max-workers 1 \
        --smoke-traits "$SMOKE_CSV" \
        > "$OUT/genomas.log" 2>&1 &
    AGENT_PID=$!
    kill -STOP "$AGENT_PID" >/dev/null 2>&1 || true

    INSTRUMENTATION_LEVEL="ebpf"
    [ "${COLLECT_LUSTRE_COUNTERS:-0}" = "1" ] && INSTRUMENTATION_LEVEL="ebpf+lustre-counters"

    "$POST_PYTHON" -m agent_io_tracing.experiments.run_manifest \
        --output "$OUT/manifest.json" --workload "GenoMAS" --task-id "$NAME" \
        --model "$GENOMAS_MODEL" --api "1" --replay-mode "$GENOMAS_LLM_REPLAY" \
        --llm-cache-path "${GENOMAS_LLM_CACHE_PATH:-$OUT/llm_cache.jsonl}" \
        --agent-count "1" --pid "$AGENT_PID" --genomas-repo "$GENOMAS_REPO" \
        --data-dir "$VIEW" --work-dir "$WORK" --output-dir "$OUT" \
        --instrumentation "$INSTRUMENTATION_LEVEL" \
        > "$OUT/manifest.log" 2>&1 || true

    LUSTRE_SAMPLER_PID=""
    if [ "${COLLECT_LUSTRE_COUNTERS:-0}" = "1" ]; then
        "$POST_PYTHON" -m agent_io_tracing.tracing.lustre_counters \
            --output "$OUT/lustre_counters.jsonl" \
            --interval "$LUSTRE_COUNTER_INTERVAL_SEC" \
            > "$OUT/lustre_counters.log" 2>&1 &
        LUSTRE_SAMPLER_PID=$!
    fi

    # 3) Start the tracer; wait for ready; CONT the agent.
    READY_FIFO="$OUT/bcc.ready.fifo"; rm -f "$READY_FIFO"; mkfifo "$READY_FIFO"
    NET_ARG="--include-net"; [ "${BCC_INCLUDE_NET:-1}" = "1" ] || NET_ARG="--no-include-net"
    "$TRACER_PYTHON" -m agent_io_tracing.tracing.bcc_tracer \
        --root-pid "$AGENT_PID" --output "$OUT/ebpf_events.log" \
        $NET_ARG --ready-fd 3 3>"$READY_FIFO" \
        >"$OUT/bcc.out" 2>"$OUT/bcc.err" &
    TRACER_PID=$!
    READY_MSG=""; read -r READY_MSG <"$READY_FIFO" || true; rm -f "$READY_FIFO"
    [ "$READY_MSG" = "ready" ] || echo "  Warning: tracer not ready; continuing" >&2
    kill -CONT "$AGENT_PID" >/dev/null 2>&1 || true

    wait "$AGENT_PID"; EXIT_CODE=$?
    sudo kill -INT "$TRACER_PID" >/dev/null 2>&1 || true
    wait "$TRACER_PID" >/dev/null 2>&1 || true
    if [ -n "$LUSTRE_SAMPLER_PID" ]; then
        kill -INT "$LUSTRE_SAMPLER_PID" >/dev/null 2>&1 || true
        wait "$LUSTRE_SAMPLER_PID" >/dev/null 2>&1 || true
    fi
    set -e
    echo "  End:   $(date +%H:%M:%S)  (exit=$EXIT_CODE)"

    if [ "$CELL_IDX" -lt "${#CELLS[@]}" ] && [ "${INTER_RUN_SLEEP_SEC:-0}" -gt 0 ]; then
        echo "  Sleeping ${INTER_RUN_SLEEP_SEC}s..."; sleep "$INTER_RUN_SLEEP_SEC"
    fi
    echo ""
done

# --- Post-processing --------------------------------------------------------
echo "=== Post-processing (parse + summarize + lineage + phase1 + viz) ==="
for ws_out in "$BASE_OUT"/*/; do
    [ -d "$ws_out" ] || continue
    NAME="$(basename "$ws_out")"
    [ -f "$ws_out/ebpf_events.log" ] || { echo "Skip $NAME (no ebpf_events.log)"; continue; }
    echo "Processing: $NAME"
    set +e
    "$POST_PYTHON" -m agent_io_tracing.parsing.ebpf "$ws_out" > "$ws_out/parse.log" 2>&1
    if [ -f "$ws_out/parsed.json" ] && [ -f "$ws_out/pi_events.jsonl" ]; then
        "$POST_PYTHON" -m agent_io_tracing.analysis.summary "$ws_out" > "$ws_out/summarize.log" 2>&1
        "$POST_PYTHON" -m agent_io_tracing.lineage.analyzer "$ws_out" > "$ws_out/lineage.log" 2>&1
        "$POST_PYTHON" -m agent_io_tracing.analysis.parallelism "$ws_out" > "$ws_out/parallelism.log" 2>&1
        "$POST_PYTHON" -m agent_io_tracing.analysis.phase1_metrics "$ws_out" > "$ws_out/phase1_metrics.log" 2>&1
        "$POST_PYTHON" -m agent_io_tracing.viz.trace "$ws_out" > "$ws_out/visualize.log" 2>&1
    else
        echo "  Skipping post (parsed.json or pi_events.jsonl missing)"
    fi
    set -e
done

# --- Fanout summary CSV (one row per cell, axis-relevant numbers) ----------
SUMMARY_CSV="$BASE_OUT/fanout_summary.csv"
"$POST_PYTHON" - "$BASE_OUT" "$SUMMARY_CSV" << 'PYEOF'
import csv, json, sys
from pathlib import Path
base, out_csv = Path(sys.argv[1]), Path(sys.argv[2])

def jload(p, d=None):
    try: return json.loads(Path(p).read_text())
    except Exception: return d

cols = ["cell", "axis", "param", "n_traits", "n_cohorts",
        "generated_files", "distinct_files",
        "storage_metadata_ops", "data_ops", "metadata_to_data",
        "read_bytes", "write_bytes", "file_count_amplification",
        "wall_clock_s", "total_llm_s"]
rows = []
for d in sorted(base.iterdir()):
    if not d.is_dir(): continue
    p1 = jload(d / "phase1_metrics.json", {})
    lin = jload(d / "lineage" / "io_summary.json", {})
    par = jload(d / "parallelism_summary.json", {})
    stage = jload(d / "stage_manifest.json", {})
    wl = (lin or {}).get("workload", {})
    ratios = (p1 or {}).get("metadata_data_ratio", {})
    opt = (p1 or {}).get("analytical_optimum_amplification", {})
    traits = (stage or {}).get("traits", [])
    n_traits = len(traits) or ""
    n_cohorts = sum(t.get("staged", 0) for t in traits) or ""
    name = d.name
    axis = ("shared" if name == "base" else
            "cohort" if name.startswith("a1_") else
            "trait"  if name.startswith("a2_") else "?")
    param = name.split("_c")[-1].split("_t")[-1] if name != "base" else "1"
    rows.append([
        name, axis, param, n_traits, n_cohorts,
        opt.get("actual_generated_files", ""),
        wl.get("distinct_files", ""),
        ratios.get("storage_metadata_ops", ""),
        ratios.get("data_ops", ""),
        ratios.get("storage_metadata_to_data_ops", ""),
        wl.get("read_bytes", ""), wl.get("write_bytes", ""),
        opt.get("file_count_amplification", ""),
        par.get("wall_clock_s", ""),
        par.get("total_self_time_s", ""),
    ])
with out_csv.open("w", newline="") as f:
    w = csv.writer(f); w.writerow(cols); w.writerows(rows)
print(f"Wrote {out_csv} ({len(rows)} rows)")
for r in rows: print("  " + ",".join(str(x) for x in r))
PYEOF

echo "=== Fanout summary: $SUMMARY_CSV ==="
cat "$SUMMARY_CSV" 2>/dev/null || true

set +e
"$POST_PYTHON" -m agent_io_tracing.viz.fanout_input_sizes "$BASE_OUT" > "$BASE_OUT/input_sizes.log" 2>&1
"$POST_PYTHON" -m agent_io_tracing.viz.fanout_plot "$BASE_OUT" > "$BASE_OUT/plot_fanout.log" 2>&1
"$POST_PYTHON" -m agent_io_tracing.viz.fanout_index "$BASE_OUT" > "$BASE_OUT/make_fanout_index.log" 2>&1
set -e

chmod -R a+rX "$BASE_OUT" || true
if [ -n "${SUDO_UID:-}" ] && [ -n "${SUDO_GID:-}" ]; then
    chown -R "$SUDO_UID:$SUDO_GID" "$BASE_OUT" 2>/dev/null || true
fi
echo ""
echo "=== Fanout study complete. Results in: $BASE_OUT ==="
echo "Headline CSV: $SUMMARY_CSV"
