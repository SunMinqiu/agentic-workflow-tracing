#!/bin/bash
#
# Deploy ChemGraph XANES MCP tracing target to a CloudLab client node.
#
# Run locally after:
#   source cloudlab_env.sh
#   bash scripts/deploy_chemgraph_to_client.sh

set -euo pipefail

SSH_USER="${SSH_USER:-Minqiu}"
CLIENT_NODE="${CLIENT_NODE:?source cloudlab_env.sh first}"
LOCAL_HARNESS="${LOCAL_HARNESS:-$HOME/Desktop/Benchmarking_Agents/pi-ebpf-tracing-handoff}"

REMOTE_HARNESS_NAME="${REMOTE_HARNESS_NAME:-pi-ebpf-tracing-handoff}"
REMOTE_CHEMGRAPH_DIR="${REMOTE_CHEMGRAPH_DIR:-ChemGraph}"
CHEMGRAPH_GIT_URL="${CHEMGRAPH_GIT_URL:-https://github.com/argonne-lcf/ChemGraph.git}"
CHEMGRAPH_GIT_REF="${CHEMGRAPH_GIT_REF:-e9e83bc}"

REMOTE_DATA_DIR_CHEMGRAPH="${REMOTE_DATA_DIR_CHEMGRAPH:-/mnt/lustrefs/chemgraph_data}"
REMOTE_FDMNES_DIR="${REMOTE_FDMNES_DIR:-/mnt/lustrefs/fdmnes}"
FDMNES_URL="${FDMNES_URL:-https://hub.neel.cnrs.fr/index.php/s/Jm5KCDif9QwbEwa/download}"
CHEMGRAPH_MODEL="${CHEMGRAPH_MODEL:-gpt-4o-mini}"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Error: OPENAI_API_KEY is not set. Source cloudlab_env.sh first." >&2
    exit 1
fi
if [ -z "${MP_API_KEY:-}" ]; then
    echo "Error: MP_API_KEY is not set. Source cloudlab_env.sh first." >&2
    exit 1
fi
if [ ! -d "$LOCAL_HARNESS" ]; then
    echo "Error: local harness dir not found: $LOCAL_HARNESS" >&2
    exit 1
fi

echo "==> [1/5] SSH + Lustre check ($SSH_USER@$CLIENT_NODE)"
ssh "$SSH_USER@$CLIENT_NODE" "true"
ssh "$SSH_USER@$CLIENT_NODE" "bash -s" << REMOTE_LUSTRE
set -euo pipefail
if ! grep -qE '[[:space:]]/mnt/lustrefs[[:space:]]' /proc/mounts; then
    echo 'Error: /mnt/lustrefs is not mounted on client' >&2
    exit 1
fi
grp=\$(id -gn)
sudo mkdir -p "$REMOTE_DATA_DIR_CHEMGRAPH" "$REMOTE_FDMNES_DIR" \
    /mnt/lustrefs/chemgraph_venv /mnt/lustrefs/uv_cache_chemgraph
sudo chown -R "\$USER:\$grp" "$REMOTE_DATA_DIR_CHEMGRAPH" "$REMOTE_FDMNES_DIR" \
    /mnt/lustrefs/chemgraph_venv /mnt/lustrefs/uv_cache_chemgraph
touch "$REMOTE_DATA_DIR_CHEMGRAPH/.deploy_writetest"
rm "$REMOTE_DATA_DIR_CHEMGRAPH/.deploy_writetest"
REMOTE_LUSTRE

echo "==> [1b/5] rsync harness -> $CLIENT_NODE:~/$REMOTE_HARNESS_NAME/"
rsync -av \
    --exclude '.git/' \
    --exclude '.DS_Store' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'traces/' \
    --exclude 'results/' \
    --exclude '*_traces/' \
    --exclude 'GenoMAS_traces/' \
    --exclude 'summarize_pi/' \
    --exclude '.venv/' \
    --exclude '.env' \
    "$LOCAL_HARNESS/" \
    "$SSH_USER@$CLIENT_NODE:$REMOTE_HARNESS_NAME/"

echo "==> [2/5] FDMNES binary on Lustre"
ssh -T "$SSH_USER@$CLIENT_NODE" "bash -s" << REMOTE_FDMNES
set -euo pipefail
cd "$REMOTE_FDMNES_DIR"
if [ ! -x "$REMOTE_FDMNES_DIR/fdmnes" ]; then
    if ! command -v unzip >/dev/null 2>&1; then
        sudo dnf install -y --setopt=install_weak_deps=False unzip
    fi
    curl -L -o fdmnes_linux.zip "$FDMNES_URL"
    rm -rf unpack
    mkdir -p unpack
    unzip -q -o fdmnes_linux.zip -d unpack
    exe=\$(find unpack -type f -name fdmnes_linux | head -1)
    if [ -z "\$exe" ]; then
        echo "Error: fdmnes_linux executable not found after unzip" >&2
        exit 1
    fi
    chmod +x "\$exe"
    ln -sfn "\$exe" "$REMOTE_FDMNES_DIR/fdmnes"
fi
"$REMOTE_FDMNES_DIR/fdmnes" < /dev/null > fdmnes_probe.out 2> fdmnes_probe.err || true
grep -q 'FDMNES program' fdmnes_probe.out || {
    echo "Error: FDMNES executable did not start cleanly" >&2
    cat fdmnes_probe.out >&2
    cat fdmnes_probe.err >&2
    exit 1
}
ls -lh "$REMOTE_FDMNES_DIR/fdmnes"
REMOTE_FDMNES

echo "==> [3/5] uv + ChemGraph ($REMOTE_CHEMGRAPH_DIR @ $CHEMGRAPH_GIT_REF)"
ssh -T "$SSH_USER@$CLIENT_NODE" "bash -s" << REMOTE_UV
set -euo pipefail
cd ~
if ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
    sudo dnf install -y --setopt=install_weak_deps=False git curl
fi
if ! command -v uv >/dev/null 2>&1 && [ ! -x "\$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="\$HOME/.local/bin:\$PATH"
uv --version

if [ -d "$REMOTE_CHEMGRAPH_DIR/.git" ]; then
    git -C "$REMOTE_CHEMGRAPH_DIR" fetch --quiet origin
else
    git clone --quiet "$CHEMGRAPH_GIT_URL" "$REMOTE_CHEMGRAPH_DIR"
fi
cd "$REMOTE_CHEMGRAPH_DIR"
git checkout --quiet "$CHEMGRAPH_GIT_REF"
echo "  HEAD: \$(git rev-parse --short HEAD)"

export UV_CACHE_DIR=/mnt/lustrefs/uv_cache_chemgraph
export UV_LINK_MODE=copy
VENV_DIR=/mnt/lustrefs/chemgraph_venv
mkdir -p "\$UV_CACHE_DIR"
uv venv --clear --python 3.11 "\$VENV_DIR"
rm -rf .venv
ln -s "\$VENV_DIR" .venv

uv pip install --python .venv/bin/python -e '.[calculators]'
uv pip install --python .venv/bin/python 'langchain-mcp-adapters<0.2' parsl plotly

.venv/bin/python - << 'PYEOF'
import importlib
for pkg in ("chemgraph", "langchain_mcp_adapters"):
    importlib.import_module(pkg)
    print(f"{pkg}: import ok")
PYEOF
REMOTE_UV

echo "==> [4/5] Data + env"
REMOTE_HOME=$(ssh -T "$SSH_USER@$CLIENT_NODE" 'echo $HOME')
ssh "$SSH_USER@$CLIENT_NODE" "cat > '$REMOTE_DATA_DIR_CHEMGRAPH/Fe2O3.cif'" << 'CIF'
data_Fe2O3
_symmetry_space_group_name_H-M   'R -3 c'
_cell_length_a   5.0350
_cell_length_b   5.0350
_cell_length_c   13.7470
_cell_angle_alpha   90
_cell_angle_beta    90
_cell_angle_gamma   120
_symmetry_Int_Tables_number 167
loop_
  _symmetry_equiv_pos_as_xyz
  'x,y,z'
  '-y,x-y,z'
  '-x+y,-x,z'
  '-x,-y,-z'
  'y,-x+y,-z'
  'x-y,x,-z'
  'x-y,x,z+1/2'
  '-x,-x+y,z+1/2'
  'y,x,z+1/2'
  '-x+y,-y,-z+1/2'
  'x,x-y,-z+1/2'
  '-y,-x,-z+1/2'
loop_
  _atom_site_label
  _atom_site_type_symbol
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
  _atom_site_occupancy
  Fe1 Fe 0.00000 0.00000 0.35530 1.0
  O1  O  0.30560 0.00000 0.25000 1.0
CIF

ssh "$SSH_USER@$CLIENT_NODE" "bash -s" << REMOTE_ENSEMBLE_DATA
set -euo pipefail
mkdir -p "$REMOTE_DATA_DIR_CHEMGRAPH/ensemble_structs"
cp "$REMOTE_DATA_DIR_CHEMGRAPH/Fe2O3.cif" "$REMOTE_DATA_DIR_CHEMGRAPH/ensemble_structs/Fe2O3_hematite_a.cif"
cp "$REMOTE_DATA_DIR_CHEMGRAPH/Fe2O3.cif" "$REMOTE_DATA_DIR_CHEMGRAPH/ensemble_structs/Fe2O3_hematite_b.cif"
cp "$REMOTE_DATA_DIR_CHEMGRAPH/Fe2O3.cif" "$REMOTE_DATA_DIR_CHEMGRAPH/ensemble_structs/Fe2O3_hematite_c.cif"
REMOTE_ENSEMBLE_DATA

ssh "$SSH_USER@$CLIENT_NODE" \
    "cat > '$REMOTE_HARNESS_NAME/.env.chemgraph' && chmod 600 '$REMOTE_HARNESS_NAME/.env.chemgraph'" \
    << REMOTE_ENV
# Auto-generated by deploy_chemgraph_to_client.sh -- do not commit.
export OPENAI_API_KEY="${OPENAI_API_KEY}"
export MP_API_KEY="${MP_API_KEY}"
export PYTHONDONTWRITEBYTECODE=1

export CHEMGRAPH_REPO="${REMOTE_HOME}/$REMOTE_CHEMGRAPH_DIR"
export TRACER_PYTHON="/usr/bin/python3"
export AGENT_PYTHON="${REMOTE_HOME}/$REMOTE_CHEMGRAPH_DIR/.venv/bin/python"
export POST_PYTHON="${REMOTE_HOME}/$REMOTE_CHEMGRAPH_DIR/.venv/bin/python"
export FDMNES_EXE="$REMOTE_FDMNES_DIR/fdmnes"
export CHEMGRAPH_MODEL="$CHEMGRAPH_MODEL"
export CHEMGRAPH_MCP_SERVER_MODULE="agent_io_tracing.adapters.chemgraph.xanes_mcp_stdio"
export CHEMGRAPH_MCP_SERVER_MODULE="chemgraph.mcp.xanes_mcp"
export COMPUTE_SYSTEM="polaris"
export REMOTE_DATA_DIR_CHEMGRAPH="$REMOTE_DATA_DIR_CHEMGRAPH"
REMOTE_ENV

echo "==> [5/5] Smoke checks (imports + FDMNES path)"
ssh "$SSH_USER@$CLIENT_NODE" "bash -s" << REMOTE_SMOKE
set -euo pipefail
source "$REMOTE_HARNESS_NAME/.env.chemgraph"
"\$AGENT_PYTHON" - << 'PYEOF'
import os
import importlib
for pkg in ("chemgraph", "langchain_mcp_adapters"):
    importlib.import_module(pkg)
print("imports ok")
print("FDMNES_EXE", os.environ["FDMNES_EXE"])
PYEOF
test -x "\$FDMNES_EXE"
test -f "$REMOTE_DATA_DIR_CHEMGRAPH/Fe2O3.cif"
REMOTE_SMOKE

cat <<EOF

Deploy complete.

Remote trace run:
  ssh $SSH_USER@$CLIENT_NODE
  cd ~/$REMOTE_HARNESS_NAME
  sudo -E bash scripts/trace_script_bcc_chemgraph.sh

For an untraced smoke run:
  ssh $SSH_USER@$CLIENT_NODE
  cd ~/$REMOTE_HARNESS_NAME
  source .env.chemgraph
  AGENT_PYTHON=\$AGENT_PYTHON DATA_DIR=$REMOTE_DATA_DIR_CHEMGRAPH \\
    PYTHONPATH=src "\$AGENT_PYTHON" -m agent_io_tracing.adapters.chemgraph.launcher /tmp/chemgraph_smoke /tmp/chemgraph_smoke_log \\
    --model "$CHEMGRAPH_MODEL" \\
    --prompt "Run a XANES calculation on the file $REMOTE_DATA_DIR_CHEMGRAPH/Fe2O3.cif at the Fe K-edge with Z_absorber=26 and a cluster radius of 3.0 Angstrom."
EOF
