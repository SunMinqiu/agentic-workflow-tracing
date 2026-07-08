#!/bin/bash
#
# Deploy the SRAgent eBPF tracing harness to the CloudLab CLIENT node.
# Run this AFTER setup_lustre_simple.sh has succeeded, ideally after
#   source ~/Desktop/workload-char/cloudlab_env.sh
# so node hostnames and API keys are picked up automatically.
#
# Target OS: CentOS Stream 8 (RHEL family).  Uses dnf, not apt.
#
# What it does on the client:
#   1. Verifies SSH + Lustre mount + creates DATA_DIR on Lustre
#   2. dnf-installs BCC bindings (tied to system python 3.6) + git/curl
#   3. rsyncs ~/Desktop/pi-ebpf-tracing-handoff/ → client:~/pi-ebpf-tracing-handoff/
#   4. Installs uv (one-line curl), clones SRAgent, runs `uv sync` to build
#      ~/SRAgent/.venv with Python 3.11+ and all SRAgent deps
#   5. Adds matplotlib + plotly to the venv (visualize_strace.py needs them)
#   6. Writes ~/pi-ebpf-tracing-handoff/.env from your local API key env vars
#
# Re-runnable: rsync uses --delete, uv sync is idempotent, .env is overwritten.
#

set -euo pipefail

# ============================================================
# Vars (defaults pull from cloudlab_env.sh if you sourced it)
# ============================================================
SSH_USER="${SSH_USER:-Minqiu}"
CLIENT_NODE="${CLIENT_NODE:-c220g2-011108.wisc.cloudlab.us}"
MOUNT_PATH="${MOUNT_PATH:-/mnt/lustrefs}"
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-${MOUNT_PATH}/sragent_data}"

# Local source dir
LOCAL_HARNESS="${LOCAL_HARNESS:-$HOME/Desktop/Benchmarking_Agents/pi-ebpf-tracing-handoff}"

# Remote layout (under remote $HOME)
REMOTE_HARNESS_NAME="${REMOTE_HARNESS_NAME:-pi-ebpf-tracing-handoff}"
REMOTE_SRAGENT_DIR="${REMOTE_SRAGENT_DIR:-SRAgent}"
SRAGENT_GIT_URL="${SRAGENT_GIT_URL:-https://github.com/ArcInstitute/SRAgent.git}"
SRAGENT_GIT_REF="${SRAGENT_GIT_REF:-main}"

# ----------------------------------------------------------------
# Pre-flight (local)
# ----------------------------------------------------------------
if [ ! -d "$LOCAL_HARNESS" ]; then
    echo "Error: local harness dir not found: $LOCAL_HARNESS" >&2
    exit 1
fi

if [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "Warning: neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set locally." >&2
    echo "         Edit ~/$REMOTE_HARNESS_NAME/.env on the client before running." >&2
fi

if [ -z "${CORE_API_KEY:-}" ]; then
    echo "Warning: CORE_API_KEY is not set locally." >&2
    echo "         SRAgent papers download will skip CORE as a source and fall" >&2
    echo "         back to Unpaywall / Europe PMC etc.  Register a free key at" >&2
    echo "         https://core.ac.uk/services/api if you want CORE coverage." >&2
fi

# ----------------------------------------------------------------
# 1) SSH + Lustre + DATA_DIR
# ----------------------------------------------------------------
echo "==> [1/6] Verifying SSH and Lustre on $CLIENT_NODE"
ssh "$SSH_USER@$CLIENT_NODE" "true"

ssh "$SSH_USER@$CLIENT_NODE" "bash -s" << REMOTE_LUSTRE
set -euo pipefail
if ! mountpoint -q "$MOUNT_PATH"; then
    echo "ERROR: $MOUNT_PATH is not a mountpoint.  Did setup_lustre_simple.sh succeed?" >&2
    exit 1
fi
echo "  Lustre mount OK: $MOUNT_PATH"

# Ensure DATA_DIR on Lustre exists and is writable by the user.
sudo mkdir -p "$REMOTE_DATA_DIR"
sudo chown "$SSH_USER" "$REMOTE_DATA_DIR" || sudo chown "\$(id -un)" "$REMOTE_DATA_DIR"
echo "  Lustre data dir ready: $REMOTE_DATA_DIR"
REMOTE_LUSTRE

# ----------------------------------------------------------------
# 2) Install OS deps (BCC bindings tied to system python 3.6, plus git/curl)
# ----------------------------------------------------------------
echo "==> [2/6] Installing OS deps via dnf (BCC, kernel-devel, git, curl)"
# -T forces no pty: heredoc EOF reaches the remote shell so the session closes
# cleanly.  CloudLab has passwordless sudo, so no TTY is needed.  -tt would
# hang here on EL8's openssh after heredoc EOF.
ssh -T "$SSH_USER@$CLIENT_NODE" "bash -s" << 'REMOTE_DNF'
set -euo pipefail

. /etc/os-release
echo "Client OS: $PRETTY_NAME ($VERSION_ID)"
case "$ID" in
    centos|rhel|rocky|almalinux) ;;
    *)
        echo "Unsupported OS: $ID (expected RHEL family / CentOS)" >&2
        exit 1
        ;;
esac

# bcc-tools pulls in libbcc; python3-bcc gives us the Python bindings.
# kernel-devel is needed by BCC at runtime to compile BPF programs.
sudo dnf install -y --setopt=install_weak_deps=False \
    bcc-tools \
    python3-bcc \
    "kernel-devel-$(uname -r)" \
    git curl rsync

# Smoke-test: BCC importable in system python (the only one with bindings).
/usr/bin/python3 -c "from bcc import BPF; print('BCC import OK in system python')"
REMOTE_DNF

# ----------------------------------------------------------------
# 3) rsync the harness
# ----------------------------------------------------------------
echo "==> [3/6] rsync $LOCAL_HARNESS/ → $CLIENT_NODE:~/$REMOTE_HARNESS_NAME/"
rsync -av --delete \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'traces/' \
    --exclude 'results/' \
    --exclude '.env' \
    "$LOCAL_HARNESS/" \
    "$SSH_USER@$CLIENT_NODE:$REMOTE_HARNESS_NAME/"

# ----------------------------------------------------------------
# 4) Install uv, clone SRAgent, uv sync (creates .venv with Python 3.11+)
# ----------------------------------------------------------------
echo "==> [4/6] uv + SRAgent (uv sync builds ~/$REMOTE_SRAGENT_DIR/.venv)"
ssh -T "$SSH_USER@$CLIENT_NODE" "bash -s" << REMOTE_UV
set -euo pipefail
cd ~

# Install uv (idempotent; the installer no-ops if already present).
if ! command -v uv >/dev/null 2>&1 && [ ! -x "\$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="\$HOME/.local/bin:\$PATH"
uv --version

# Clone SRAgent (idempotent: pull if already present).
if [ -d "$REMOTE_SRAGENT_DIR/.git" ]; then
    echo "  SRAgent repo exists; pulling latest"
    git -C "$REMOTE_SRAGENT_DIR" fetch --quiet origin
    git -C "$REMOTE_SRAGENT_DIR" checkout --quiet $SRAGENT_GIT_REF
    git -C "$REMOTE_SRAGENT_DIR" pull --quiet --ff-only origin $SRAGENT_GIT_REF
else
    git clone --quiet --branch $SRAGENT_GIT_REF $SRAGENT_GIT_URL $REMOTE_SRAGENT_DIR
fi

cd "$REMOTE_SRAGENT_DIR"

# Patch SRAgent upstream bug: settings.yml ships step_summary reasoning_effort
# as "" (empty), which gpt-5-mini rejects with HTTP 400.  Replace with "low".
# Re-applied on every deploy so a fresh git pull doesn't reintroduce it.
if grep -q 'step_summary: ""' SRAgent/settings.yml; then
    sed -i 's/step_summary: ""/step_summary: "low"/g' SRAgent/settings.yml
    echo "  Patched SRAgent/settings.yml: step_summary reasoning_effort '' -> 'low'"
fi

# Recover from prior `sudo -E` SRAgent runs that left:
#   (a) root-owned __pycache__/*.pyc that this user can't touch, and/or
#   (b) half-uninstalled site-packages where uv aborted partway and the
#       dist-info is missing RECORD or METADATA (uv then refuses to
#       re-read or re-install the package, e.g. matplotlib).
# After this block, PYTHONDONTWRITEBYTECODE=1 in .env keeps (a) from
# happening again, so future deploys this whole block is a no-op.
if [ -d .venv ]; then
    sudo chown -R "$SSH_USER" .venv
    # Wipe corrupt dist-info dirs (and the matching package dir) so uv
    # treats them as missing and reinstalls cleanly.
    find .venv/lib -maxdepth 4 -type d -name "*.dist-info" 2>/dev/null | \
    while read -r d; do
        if [ ! -f "\$d/METADATA" ] || [ ! -f "\$d/RECORD" ]; then
            base=\$(basename "\$d")
            pkg=\$(echo "\$base" | sed -E 's/-[0-9][^/]*\$//')
            # Refuse to act if regex failed to extract a distinct pkg name,
            # to avoid accidentally rm -rf'ing the whole site-packages dir.
            if [ -n "\$pkg" ] && [ "\$pkg" != "\$base" ]; then
                echo "  Recovering corrupt install: \$pkg"
                rm -rf "\$d" "\$(dirname "\$d")/\$pkg"
            fi
        fi
    done
fi

# uv sync: installs the right Python (≥3.11 per pyproject), creates .venv,
# resolves and installs the lockfile.  Editable install of SRAgent itself.
uv sync

# Add post-processing extras for visualize_strace.py.  pandas/numpy are
# already in SRAgent's deps; matplotlib + plotly are extra.
# pysqlite3-binary works around CentOS Stream 8's old sqlite3 (3.26.x) —
# ChromaDB needs >=3.35.0.  The shim in analyze_codebase_sragent.py
# swaps it into sys.modules before SRAgent imports.
uv pip install matplotlib plotly pysqlite3-binary

# Smoke imports
.venv/bin/python -c "import SRAgent; print('SRAgent: OK')"
.venv/bin/python -c "import langchain_core, langgraph, langchain_openai; print('LangChain stack: OK')"
.venv/bin/python -c "import matplotlib, plotly, pandas, numpy; print('Post-proc stack: OK')"

# BCC bindings live in /usr/lib*/python3.6/site-packages — NOT importable
# from this Py 3.11+ venv (different Python minor version, different ABI).
# That's expected; the trace script invokes bcc_tracer.py with /usr/bin/python3.
echo "  (BCC binding mismatch with venv is expected; tracer uses /usr/bin/python3)"
REMOTE_UV

# ----------------------------------------------------------------
# 5) Write .env (sourced by trace_script at runtime; survives sudo -E)
# ----------------------------------------------------------------
# Resolve the *remote* user's home dir BEFORE writing .env.  We can't use
# `$HOME` literally because trace_script runs under `sudo -E`, where $HOME
# becomes /root and breaks AGENT_PYTHON resolution.
REMOTE_HOME=$(ssh -T "$SSH_USER@$CLIENT_NODE" 'echo $HOME')
echo "==> [5/6] Writing .env to ${REMOTE_HOME}/$REMOTE_HARNESS_NAME/.env"
ssh "$SSH_USER@$CLIENT_NODE" "cat > $REMOTE_HARNESS_NAME/.env && chmod 600 $REMOTE_HARNESS_NAME/.env" << REMOTE_ENV
# Auto-generated by deploy_sragent_to_client.sh -- do not commit.
# Sourced by trace_script_bcc_sragent.sh so sudo -E doesn't strip these.
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
export NCBI_API_KEY="${NCBI_API_KEY:-}"
export NCBI_EMAIL="${NCBI_EMAIL:-mqsun@udel.edu}"
export CORE_API_KEY="${CORE_API_KEY:-}"

# Prevent Python from writing .pyc files into the venv.  Without this,
# `sudo -E` runs of SRAgent leave root-owned __pycache__/*.pyc that the
# regular user can't delete, breaking the next `uv sync` on re-deploy.
export PYTHONDONTWRITEBYTECODE=1


# Pin both Pythons explicitly with the absolute remote-home path so sudo -E
# (which sets HOME=/root) can't break resolution.
export TRACER_PYTHON="/usr/bin/python3"
export AGENT_PYTHON="${REMOTE_HOME}/$REMOTE_SRAGENT_DIR/.venv/bin/python"
export POST_PYTHON="${REMOTE_HOME}/$REMOTE_SRAGENT_DIR/.venv/bin/python"

# Lustre data dir (matches deploy-time choice).
export DATA_DIR="$REMOTE_DATA_DIR"
REMOTE_ENV

# ----------------------------------------------------------------
# 6) Final pointer
# ----------------------------------------------------------------
cat <<EOF

==> [6/6] Deploy complete

Run the trace on the client:

  ssh $SSH_USER@$CLIENT_NODE
  cd ~/$REMOTE_HARNESS_NAME
  cat .env                                     # confirm keys + python paths
  \$EDITOR config_sragent.env                  # tweak WORKLOADS / RUN_WORKLOADS

  # sudo is required for BCC; -E preserves env (.env will also be sourced
  # by the trace script, so it's belt-and-suspenders).
  sudo -E bash trace_script_bcc_sragent.sh

Outputs land in /mnt/lustrefs/$SSH_USER/pi-ebpf-tracing-handoff/results/<timestamp>/<workload>/
including parsed.json, pi_summary.json, visualizations/index.html.

Lustre side ($REMOTE_DATA_DIR/<workload>/) holds the big stuff:
papers PDFs, fastq files, fastq-dump tempfiles.

EOF
