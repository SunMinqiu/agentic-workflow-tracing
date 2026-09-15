#!/usr/bin/env bash
# Record one replay arm against the vLLM server behind the Mac tunnel.
#
# Usage, from the repo root:
#   BUNDLE=replay/bundles/<bundle>.json bash scripts/replay_arm.sh <sweep_dir> <label> [sweep arm options]
#
# Fixed for every arm: the Mac tunnel URL, the KV-event replay endpoint and the
# output length, so arms inside one sweep stay comparable.  Everything after
# <label> goes to `agent_io_tracing.replay.sweep arm` unchanged, e.g.
# --mode serial --cache-salt-mode shared --warmup-repetitions 1 --strict-config.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

sweep_dir="${1:-}"
label="${2:-}"
if [ -z "$sweep_dir" ] || [ -z "$label" ]; then
    echo "Usage: BUNDLE=<bundle.json> bash scripts/replay_arm.sh <sweep_dir> <label> [sweep arm options]" >&2
    exit 2
fi
shift 2
: "${BUNDLE:?set BUNDLE to the bundle json for this sweep}"
[ -s "$BUNDLE" ] || { echo "ERROR: bundle missing or empty: $BUNDLE" >&2; exit 1; }

MAC_VLLM_URL="${MAC_VLLM_URL:-http://127.0.0.1:18000}"
VLLM_KV_EVENTS_REPLAY_ENDPOINT="${VLLM_KV_EVENTS_REPLAY_ENDPOINT:-tcp://127.0.0.1:15558}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-256}"

curl -fsS "$MAC_VLLM_URL/v1/models" >/dev/null || {
    echo "ERROR: no vLLM behind $MAC_VLLM_URL; rebuild the Perlmutter tunnel." >&2
    exit 1
}

PYTHONPATH=src python3 -m agent_io_tracing.replay.sweep arm \
    --bundle "$BUNDLE" \
    --sweep-dir "$sweep_dir" \
    --label "$label" \
    --url "$MAC_VLLM_URL" \
    --output-tokens "$OUTPUT_TOKENS" \
    --kv-events-replay-endpoint "$VLLM_KV_EVENTS_REPLAY_ENDPOINT" \
    "$@"

test -s "$sweep_dir/$label/rep0/summary.json" && echo "arm-ready: $sweep_dir/$label"
