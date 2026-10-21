#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .env.swebench
mkdir -p "$HOME/.claude-code-router"
CFG="$HOME/.claude-code-router/config.json"
if [ -f "$CFG" ]; then cp "$CFG" "$CFG.backup.$(date +%s)"; fi
TEMP_CFG="$(mktemp)"
# The generator requires a new file.
rm "$TEMP_CFG"
VLLM_URL=http://127.0.0.1:18081 python scripts/swebench_local.py ccr-config --config-output "$TEMP_CFG"
install -m 600 "$TEMP_CFG" "$CFG"
rm "$TEMP_CFG"
ccr stop || true
nohup ccr start > "$SWE_STORAGE/ccr.log" 2>&1 < /dev/null &
sleep 2
curl -fsS http://127.0.0.1:3456/ >/dev/null
echo ccr-ready
