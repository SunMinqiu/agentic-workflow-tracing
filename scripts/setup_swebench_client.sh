#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
STORAGE="${SWE_STORAGE:-/mnt/lustrefs/$USER/pi-ebpf-tracing-handoff/swebench}"
if [ ! -d "$STORAGE" ]; then
  sudo mkdir -p "$STORAGE"
  sudo chown "$USER:$(id -gn)" "$STORAGE"
fi
mkdir -p "$HOME/.local/bin"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
if [ ! -x "$STORAGE/node/bin/node" ]; then
  curl -fsSL https://nodejs.org/dist/v22.16.0/node-v22.16.0-linux-x64.tar.xz -o "$STORAGE/node.tar.xz"
  tar -xJf "$STORAGE/node.tar.xz" -C "$STORAGE"
  mv "$STORAGE/node-v22.16.0-linux-x64" "$STORAGE/node"
fi
export PATH="$STORAGE/node/bin:$PATH"
export npm_config_prefix="$STORAGE/npm"
export PATH="$npm_config_prefix/bin:$PATH"
npm install -g @anthropic-ai/claude-code@2.1.197 @musistudio/claude-code-router@1.0.72
[ -x "$STORAGE/venv/bin/python" ] || uv venv --python 3.11 "$STORAGE/venv"
uv pip install --python "$STORAGE/venv/bin/python" aiohttp Markdown matplotlib numpy pandas plotly tiktoken
"$STORAGE/venv/bin/python" scripts/swebench_local.py prepare --work-root "$STORAGE/work"
cat > .env.swebench <<EOF
export PATH="$STORAGE/node/bin:$STORAGE/npm/bin:$STORAGE/venv/bin:\$HOME/.local/bin:\$PATH"
export SWE_STORAGE="$STORAGE"
export VLLM_URL="http://127.0.0.1:18080"
export VLLM_SERVED_MODEL="Qwen3.6-27B"
EOF
chmod 600 .env.swebench
claude --version
ccr version
