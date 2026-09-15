#!/usr/bin/env bash
# Start one trace run on the client node and print the two commands you will
# need next: watch its log, pull its results.  Both are printed with the log
# path already filled in, so they paste into any terminal.
#
# Usage, from the repo root with cloudlab_env.sh already sourced:
#   bash scripts/launch_run.sh genomas A_c8_w4
#   bash scripts/launch_run.sh scilink eels_plasmons_basic
#   bash scripts/launch_run.sh 1000genome classic_chr1_r1
#   bash scripts/launch_run.sh montage montage_m17_0p10_r1
#
# The second argument is RUN_WORKLOADS: one cell, a comma list, or omitted for
# every cell the trace script defines.  GenoMAS and SciLink read VLLM_URL from
# this shell when it is set; otherwise the remote .env file picks the backend.
# GENOMAS_REFERENCE_TOKENS and VLLM_SERVED_MODEL are passed through the same way.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

workflow="${1:-}"
workloads="${2:-}"
if [ -z "$workflow" ]; then
    echo "Usage: bash scripts/launch_run.sh <genomas|scilink|1000genome|montage> [RUN_WORKLOADS]" >&2
    exit 2
fi
: "${SSH_USER:?source cloudlab_env.sh first}"
: "${WORKFLOW_NODE:?source cloudlab_env.sh first}"
: "${MOUNT_PATH:?source cloudlab_env.sh first}"

# Optional variables forwarded to the remote script only when set here.
passthrough=""
forward=""
case "$workflow" in
    genomas) forward="VLLM_URL VLLM_SERVED_MODEL GENOMAS_REFERENCE_TOKENS GENOMAS_REFERENCE_POSITION GENOMAS_REFERENCE_OUTPUT_RESERVE GENOMAS_MAX_TIME VLLM_KEEP_PREFIX_CACHE" ;;
    scilink) forward="VLLM_URL VLLM_SERVED_MODEL" ;;
esac
for var in $forward; do
    if [ -n "${!var:-}" ]; then
        passthrough="$passthrough $var='${!var}'"
    fi
done

lustre_user="$MOUNT_PATH/$SSH_USER"
case "$workflow" in
    genomas)
        runner="sudo -n -E env RUN_WORKLOADS='$workloads'$passthrough"
        script="scripts/trace_script_bcc_genomas.sh"
        ;;
    scilink)
        # SciLink runs as the SSH user; the script starts the BCC tracer with sudo itself.
        runner="env RUN_WORKLOADS='$workloads' SCILINK_FORCE_STREAM=1$passthrough"
        script="scripts/trace_script_bcc_scilink.sh"
        ;;
    1000genome)
        repo="$lustre_user/1000genome-workflow"
        runner="sudo -n -E env
            WORKFLOW_REPO='$repo'
            DATASET_DIR='$repo/data/20130502'
            POPULATION_DIR='$repo/data/populations'
            AGENT_PYTHON='$repo/.venv/bin/python'
            POST_PYTHON='$repo/.venv/bin/python'
            CLASSIC_OFFLINE=1 RUN_WORKLOADS='$workloads'"
        script="scripts/trace_script_bcc_1000genome.sh"
        ;;
    montage)
        root="$lustre_user/montage"
        runner="sudo -n -E env
            MONTAGE_ROOT='$root'
            MONTAGE_INPUT_ROOT='$root/input'
            MONTAGE_PYTHON='$root/trace-venv/bin/python'
            AGENT_PYTHON='$root/trace-venv/bin/python'
            POST_PYTHON='$root/trace-venv/bin/python'
            MONTAGE_OFFLINE=1 RUN_WORKLOADS='$workloads'"
        script="scripts/trace_script_bcc_montage.sh"
        ;;
    *)
        echo "ERROR: unknown workflow '$workflow'. Use genomas, scilink, 1000genome or montage." >&2
        exit 2
        ;;
esac
runner="$(printf '%s' "$runner" | tr -s ' \n' ' ')"

if [ "$workflow" = genomas ] || [ "$workflow" = scilink ]; then
    if [ -z "${VLLM_URL:-}" ]; then
        echo "VLLM_URL is unset; the remote .env file decides the backend."
    fi
fi

tag="${workloads:-all}"
tag="${tag//,/+}"
log="logs/${workflow}_${tag}_$(date +%Y%m%d_%H%M%S).log"

# The checks and the nohup must be separate statements.  Written as one
# "a && b && nohup c &" list, bash backgrounds the whole list in a subshell
# whose stdout/stderr are still the ssh pipes, so ssh blocks until the run
# finishes.  Only the nohup line goes to the background here.
ssh -n "$SSH_USER@$WORKFLOW_NODE" \
    "mkdir -p \"\$HOME/logs\" && cd pi-ebpf-tracing-handoff && sudo -n true || exit 1
     nohup $runner bash $script >\"\$HOME/$log\" 2>&1 </dev/null &
     echo PID=\$! LOG=\$HOME/$log"

echo ""
echo "Watch:  source cloudlab_env.sh; ssh $SSH_USER@$WORKFLOW_NODE 'tail -f ~/$log'"
echo "Pull:   source cloudlab_env.sh; bash scripts/pull_run.sh $log"
