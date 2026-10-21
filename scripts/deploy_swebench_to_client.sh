#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SSH_USER:?source cloudlab_env.sh first}"
: "${WORKFLOW_NODE:?source cloudlab_env.sh first}"
DEST="$SSH_USER@$WORKFLOW_NODE"
ssh "$DEST" 'mkdir -p ~/pi-ebpf-tracing-handoff/{scripts,config,workloads}'
rsync -az scripts/swebench_local.py scripts/swebench_proxy.py scripts/summarize_swebench.py scripts/postprocess_swebench.py scripts/ccr_vllm_stream.js scripts/setup_swebench_client.sh scripts/setup_swebench_tasks.sh scripts/configure_swebench_ccr.sh scripts/run_swebench_measured.sh scripts/evaluate_swebench_local.sh "$DEST:pi-ebpf-tracing-handoff/scripts/"
rsync -az config/config_swebench.env config/config_vllm_endpoint.env "$DEST:pi-ebpf-tracing-handoff/config/"
rsync -az src/ "$DEST:pi-ebpf-tracing-handoff/src/"
rsync -az workloads/swebench workloads/swebench_evaluation "$DEST:pi-ebpf-tracing-handoff/workloads/"
ssh "$DEST" 'cd ~/pi-ebpf-tracing-handoff && bash scripts/setup_swebench_client.sh && bash scripts/setup_swebench_tasks.sh && bash scripts/configure_swebench_ccr.sh'
