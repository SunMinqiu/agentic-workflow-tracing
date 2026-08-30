#!/usr/bin/env bash
# Source this file in the shell that launches vLLM, then append "${VLLM_OBSERVABILITY_ARGS[@]}" to the existing vllm serve command.

export VLLM_SERVER_DEV_MODE=1
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

VLLM_KV_EVENTS_PUBLISH_BIND="${VLLM_KV_EVENTS_PUBLISH_BIND:-tcp://*:5557}"
VLLM_KV_EVENTS_REPLAY_BIND="${VLLM_KV_EVENTS_REPLAY_BIND:-tcp://*:5558}"
VLLM_KV_EVENTS_TOPIC="${VLLM_KV_EVENTS_TOPIC:-kv-events}"
VLLM_KV_METRICS_SAMPLE="${VLLM_KV_METRICS_SAMPLE:-1.0}"

VLLM_KV_EVENTS_CONFIG="$(printf '{"enable_kv_cache_events":true,"publisher":"zmq","endpoint":"%s","replay_endpoint":"%s","buffer_steps":10000,"topic":"%s"}' \
    "$VLLM_KV_EVENTS_PUBLISH_BIND" \
    "$VLLM_KV_EVENTS_REPLAY_BIND" \
    "$VLLM_KV_EVENTS_TOPIC")"

VLLM_OBSERVABILITY_ARGS=(
    --enable-prompt-tokens-details
    --enable-per-request-metrics
    --enable-force-include-usage
    --enable-log-requests
    --kv-cache-metrics
    --kv-cache-metrics-sample "$VLLM_KV_METRICS_SAMPLE"
    --enable-logging-iteration-details
    --kv-events-config "$VLLM_KV_EVENTS_CONFIG"
)

if [ -n "${VLLM_OTLP_TRACES_ENDPOINT:-}" ]; then
    VLLM_OBSERVABILITY_ARGS+=(
        --otlp-traces-endpoint "$VLLM_OTLP_TRACES_ENDPOINT"
        --collect-detailed-traces all
    )
fi
