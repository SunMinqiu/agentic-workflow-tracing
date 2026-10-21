#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/swe-kv-experiment"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export KV_TRACE_SESSION="${1:?session required}"
export KV_TRACE_DIR="$PWD/output/$KV_TRACE_SESSION/physical"
mkdir -p "$KV_TRACE_DIR"
export NUM_GPU_BLOCKS_OVERRIDE="${2:?GPU blocks required}"
export KV_OFFLOADING_SIZE="${3:?CPU GiB required}"
export TP=2 MAX_MODEL_LEN=65536 MAX_NUM_SEQS=1 PORT=8000
export MAMBA_CACHE_MODE=align ENABLE_PREFIX_CACHING=true
source config/config_vllm_observability.sh
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/2026/cuda/13.2
export PATH="$CUDA_HOME/bin:$HOME/.venvs/vllm-qwen36/bin:$PATH"
export CPATH="/opt/nvidia/hpc_sdk/Linux_x86_64/2026/math_libs/13.2/include${CPATH:+:$CPATH}"
export OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
CPU_ARGS=()
if [ "$KV_OFFLOADING_SIZE" != 0 ]; then
    CPU_ARGS=(--kv-offloading-size "$KV_OFFLOADING_SIZE" --kv-offloading-backend native)
fi
echo "Slurm step: $SLURM_JOB_ID.$SLURM_STEP_ID"
exec "$HOME/.venvs/vllm-qwen36/bin/vllm" serve Qwen/Qwen3.6-27B \
    --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 \
    --host 0.0.0.0 --port "$PORT" --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" \
    --num-gpu-blocks-override "$NUM_GPU_BLOCKS_OVERRIDE" \
    --enable-prefix-caching --mamba-cache-mode align \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --no-async-scheduling --max-num-batched-tokens 2048 \
    "${CPU_ARGS[@]}" "${VLLM_OBSERVABILITY_ARGS[@]}"
