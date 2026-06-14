#!/bin/bash
#
# Phase-1 deploy of SciLink (https://github.com/ziatdinovmax/SciLink) to a
# CloudLab CentOS Stream 8 client node.  This pass is INTENTIONALLY MINIMAL:
# it does NOT install BCC, write a tracing .env, or touch anything that
# deploy_sragent_to_client.sh set up.  Goal: confirm that
#
#     scilink analyze --data examples/eels_plasmons_demo/datacube.npy \
#                     --metadata examples/eels_plasmons_demo/datacube.json
#
# runs end-to-end against OpenAI.  Once that's green, a second pass will
# layer the eBPF tracer (separate script).
#
# Run from the laptop, AFTER sourcing cloudlab_env.sh so SSH_USER /
# CLIENT_NODE / OPENAI_API_KEY are picked up:
#
#     source ~/Desktop/Benchmarking_Agents/pi-ebpf-tracing-handoff/cloudlab_env.sh
#     bash deploy_scilink_to_client.sh
#
# Re-runnable: clone uses pull-if-exists, uv venv / pip install are idempotent,
# the SciLink .env file is overwritten in place.
#

set -euo pipefail

# ============================================================
# Vars (defaults pull from cloudlab_env.sh if you sourced it)
# ============================================================
SSH_USER="${SSH_USER:-Minqiu}"
CLIENT_NODE="${CLIENT_NODE:-c220g1-031107.wisc.cloudlab.us}"

# Local source dir for the tracing harness (rsync'd to remote in step 3).
LOCAL_HARNESS="${LOCAL_HARNESS:-$HOME/Desktop/Benchmarking_Agents/pi-ebpf-tracing-handoff}"

# Remote layout (under remote $HOME).  Independent of pi-ebpf-tracing-handoff
# and ~/SRAgent so SRAgent / SciLink coexist without stepping on each other.
REMOTE_HARNESS_NAME="${REMOTE_HARNESS_NAME:-pi-ebpf-tracing-handoff}"
REMOTE_SCILINK_DIR="${REMOTE_SCILINK_DIR:-SciLink}"
SCILINK_GIT_URL="${SCILINK_GIT_URL:-https://github.com/ziatdinovmax/SciLink.git}"
# Pin to main for now; replace with a commit SHA once we verify a working
# revision so future upstream changes don't break the trace harness.
SCILINK_GIT_REF="${SCILINK_GIT_REF:-main}"

# Optional: install Meta's Segment Anything (README mentions it).  EELS demo
# doesn't actually need it, but other SciLink demos do.  Toggle if you want
# faster deploys for the smoke test.
INSTALL_SAM="${INSTALL_SAM:-0}"

# ----------------------------------------------------------------
# Pre-flight (local)
# ----------------------------------------------------------------
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Error: OPENAI_API_KEY is not set locally." >&2
    echo "       source cloudlab_env.sh first (or export OPENAI_API_KEY=...)." >&2
    exit 1
fi

# ----------------------------------------------------------------
# 1) SSH reachability
# ----------------------------------------------------------------
echo "==> [1/4] SSH check ($SSH_USER@$CLIENT_NODE)"
ssh "$SSH_USER@$CLIENT_NODE" "true"

# ----------------------------------------------------------------
# 1b) rsync the tracing harness (litellm_tool_logger.py +
#     analyze_codebase_scilink.py + config_scilink.env +
#     trace_script_bcc_scilink.sh, plus all the shared
#     bcc_tracer.py / parse_ebpf.py / visualize_strace.py etc.).
#
# Note: this same path is also where deploy_sragent_to_client.sh writes,
# so SRAgent state is preserved.  --exclude .env keeps the SRAgent .env
# (with its keys) intact; --exclude traces/ avoids shipping local trace
# artefacts back and forth.
# ----------------------------------------------------------------
if [ ! -d "$LOCAL_HARNESS" ]; then
    echo "Error: local harness dir not found: $LOCAL_HARNESS" >&2
    exit 1
fi
echo "==> [1b] rsync $LOCAL_HARNESS/ → $CLIENT_NODE:~/$REMOTE_HARNESS_NAME/"
rsync -av \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'traces/' \
    --exclude '.venv/' \
    --exclude '.env' \
    "$LOCAL_HARNESS/" \
    "$SSH_USER@$CLIENT_NODE:$REMOTE_HARNESS_NAME/"

# ----------------------------------------------------------------
# 2) Install uv (idempotent), clone SciLink, build a Py 3.12 venv
# ----------------------------------------------------------------
echo "==> [2/4] uv + SciLink (~/$REMOTE_SCILINK_DIR/.venv on Python 3.12)"
# -T forces no pty: heredoc EOF reaches the remote shell cleanly on EL8.
ssh -T "$SSH_USER@$CLIENT_NODE" "bash -s" << REMOTE_UV
set -euo pipefail
cd ~

# git + curl might already be there from the SRAgent deploy; install only if
# this is a fresh client.  We do NOT install BCC here.
if ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
    sudo dnf install -y --setopt=install_weak_deps=False git curl
fi

# uv: portable Python installer.  Idempotent — installer no-ops if present.
if ! command -v uv >/dev/null 2>&1 && [ ! -x "\$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="\$HOME/.local/bin:\$PATH"
uv --version

# Clone or update SciLink.
if [ -d "$REMOTE_SCILINK_DIR/.git" ]; then
    echo "  SciLink repo exists; fetching"
    git -C "$REMOTE_SCILINK_DIR" fetch --quiet origin
    git -C "$REMOTE_SCILINK_DIR" checkout --quiet $SCILINK_GIT_REF
    git -C "$REMOTE_SCILINK_DIR" pull --quiet --ff-only origin $SCILINK_GIT_REF || true
else
    git clone --quiet --branch $SCILINK_GIT_REF $SCILINK_GIT_URL $REMOTE_SCILINK_DIR
fi

cd "$REMOTE_SCILINK_DIR"
echo "  HEAD: \$(git rev-parse --short HEAD)"

# Patch SciLink upstream bug: orchestrator broadcasts analysis_depth kwarg
# to every agent class, but HyperspectralAnalysisAgent.__init__ doesn't accept
# **kwargs and crashes with TypeError.  Other agents (ImageAnalysisAgent)
# already have **kwargs; this brings hyperspectral in line.  Re-applied every
# deploy so a fresh git pull doesn't reintroduce it.
python3 - << 'PATCH_PY'
import pathlib
p = pathlib.Path.home() / "SciLink/scilink/agents/exp_agents/hyperspectral_analysis_agent.py"
if not p.exists():
    print(f"  WARNING: cannot patch {p} (not found)")
else:
    s = p.read_text()
    # Idempotent: check the __init__ signature for **kwargs before patching.
    init_sig = s.split("def __init__", 1)[1].split("):", 1)[0]
    if "**kwargs" in init_sig:
        print(f"  HyperspectralAnalysisAgent already accepts **kwargs (no patch needed)")
    else:
        old = "enable_human_feedback: bool = True\n    ):"
        new = "enable_human_feedback: bool = True,\n        **kwargs,\n    ):"
        assert old in s, "patch anchor not found — SciLink upstream signature changed"
        p.write_text(s.replace(old, new))
        print(f"  Patched HyperspectralAnalysisAgent.__init__ to accept **kwargs")
PATCH_PY

# Patch the in-process dynamic-analysis exec into a one-line method.  The
# tracing harness wraps this method as a ScriptExec event; keeping the logic
# in-process avoids serializing large hyperspectral cubes to a subprocess.
python3 - << 'PATCH_PY'
import pathlib
p = pathlib.Path.home() / "SciLink/scilink/agents/exp_agents/controllers/hyperspectral_controllers.py"
if not p.exists():
    print(f"  WARNING: cannot patch {p} (not found)")
else:
    s = p.read_text()
    if "def _run_generated_code(self, code_str, global_scope, local_scope):" in s:
        print("  RunDynamicAnalysisController._run_generated_code already present (no patch needed)")
    elif "exec(code_str, global_scope, local_scope)" not in s:
        print("  WARNING: dynamic-analysis exec anchor not found; ScriptExec seam not patched")
    elif "class RunDynamicAnalysisController" not in s:
        print("  WARNING: RunDynamicAnalysisController class not found; ScriptExec seam not patched")
    else:
        s = s.replace(
            "exec(code_str, global_scope, local_scope)",
            "self._run_generated_code(code_str, global_scope, local_scope)",
            1,
        )
        class_start = s.index("class RunDynamicAnalysisController")
        next_class = s.find("\nclass ", class_start + 1)
        method = (
            "\n"
            "    def _run_generated_code(self, code_str, global_scope, local_scope):\n"
            "        \"\"\"Seam for external instrumentation; do not inline.\"\"\"\n"
            "        exec(code_str, global_scope, local_scope)\n"
        )
        if next_class == -1:
            s = s.rstrip() + method + "\n"
        else:
            s = s[:next_class].rstrip() + method + "\n" + s[next_class:]
        p.write_text(s)
        print("  Patched RunDynamicAnalysisController._run_generated_code seam")
PATCH_PY

# Py 3.12 venv.  uv downloads a portable interpreter if the host has none —
# CentOS Stream 8 ships 3.6 system-wide, so this always pulls a fresh 3.12.
# --clear nukes any prior .venv left over from a half-failed deploy, so
# re-running this script is always safe.  Note: "uv venv" does NOT seed pip
# into the venv by design -- we use "uv pip install" instead of
# ".venv/bin/python -m pip install".  (Do not use backticks in comments
# inside this unquoted heredoc: bash treats them as command substitution
# and would execute them on the LOCAL machine before sending stdin to ssh.)
# CloudLab home is often ~16GB with only ~2GB free.  SciLink + CPU torch +
# matplotlib/scipy/scikit-image easily blows that out.  If Lustre is
# mounted, place BOTH the venv and uv cache there; keep the SciLink source
# tree on home (it's small).  We then symlink ~/SciLink/.venv to the real
# Lustre dir, so every later ".venv/bin/python" relative path still works.
# Detection: /proc/mounts is the authoritative list on every Linux and
# requires no extra tools (mountpoint(1) is missing on some EL images).
if grep -qE "[[:space:]]/mnt/lustrefs[[:space:]]" /proc/mounts && [ -w /mnt/lustrefs ]; then
    VENV_DIR=/mnt/lustrefs/scilink_venv
    export UV_CACHE_DIR=/mnt/lustrefs/uv_cache_scilink
    mkdir -p "\$UV_CACHE_DIR"
    echo "  venv:     \$VENV_DIR (Lustre)"
    echo "  uv cache: \$UV_CACHE_DIR (Lustre)"
else
    VENV_DIR=.venv
    echo "  WARNING: Lustre not mounted; venv on local home (may run out)"
fi
# Drop any partial download from the prior failed install.
rm -rf "\$HOME/.cache/uv/.tmp"* 2>/dev/null || true

uv venv --clear --python 3.12 "\$VENV_DIR"

# Symlink ~/SciLink/.venv -> Lustre dir so the rest of this script (and any
# future trace harness) can keep using the relative ".venv/bin/..." paths.
if [ "\$VENV_DIR" != ".venv" ]; then
    rm -rf .venv
    ln -s "\$VENV_DIR" .venv
fi

# Install CPU-only PyTorch FIRST.  Otherwise scilink -> atomai -> torch
# transitively pulls ~4GB of NVIDIA CUDA wheels (cudnn, cublas, cusparse...)
# that the EELS demo never uses on CloudLab CPU nodes.  PyTorch's CPU index
# ships torch with a +cpu local version that satisfies the dep without any
# nvidia-* wheels.
echo "  Installing CPU-only PyTorch (replaces ~4GB CUDA bloat)..."
uv pip install --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

# Install SciLink in editable mode (so we can monkey-patch later without
# reinstalling).  Skip [ui]/[sim] extras — CLI smoke test doesn't need them.
# --python pins the install target so uv uses the venv we just created.
# No --quiet here: progress is reassuring for a multi-minute install.
echo "  Installing SciLink and remaining deps..."
uv pip install --python .venv/bin/python -e .

# Post-processing extras used by visualize_strace.py.  SciLink's own deps
# already include matplotlib + pandas + numpy; plotly is the only extra
# the visualization layer needs that SciLink doesn't pull in itself.
echo "  Installing visualization extras (plotly)..."
uv pip install --python .venv/bin/python plotly

# Optional: Segment Anything (Meta) — only if INSTALL_SAM=1.
if [ "$INSTALL_SAM" = "1" ]; then
    echo "  Installing segment-anything (Meta)..."
    uv pip install --python .venv/bin/python \
        "git+https://github.com/facebookresearch/segment-anything.git"
fi

# Smoke imports.  Use importlib.metadata for versions because some packages
# (litellm in particular) don't expose a __version__ attribute and raise
# AttributeError via __getattr__ when you try to read it.
.venv/bin/python - << 'PYEOF'
import importlib, importlib.metadata as md
for pkg in ("scilink", "litellm"):
    importlib.import_module(pkg)
    try:
        v = md.version(pkg)
    except md.PackageNotFoundError:
        v = "(unknown)"
    print(f"{pkg}: {v}")
PYEOF

# Confirm the CLI entry point landed on PATH.
test -x .venv/bin/scilink || {
    echo "ERROR: .venv/bin/scilink missing — entry point not installed" >&2
    exit 1
}
.venv/bin/scilink --help | head -5
REMOTE_UV

# ----------------------------------------------------------------
# 3) Write two .env files:
#    a) ~/SciLink/.env             — used by the bare-hand smoke test
#       (`cd ~/SciLink; source .env; scilink analyze ...`).
#    b) ~/pi-ebpf-tracing-handoff/.env.scilink — used by
#       trace_script_bcc_scilink.sh.  Holds ABSOLUTE paths so that
#       `sudo -E` (which sets HOME=/root) doesn't break AGENT_PYTHON
#       resolution.  This file is SEPARATE from ~/pi-ebpf-tracing-handoff/.env
#       (which deploy_sragent_to_client.sh owns) so the two harnesses
#       don't fight over the same file.
# ----------------------------------------------------------------
# Resolve the *remote* user's home dir BEFORE writing the .env files.
REMOTE_HOME=$(ssh -T "$SSH_USER@$CLIENT_NODE" 'echo $HOME')

echo "==> [3a/4] Writing ${REMOTE_HOME}/$REMOTE_SCILINK_DIR/.env"
ssh "$SSH_USER@$CLIENT_NODE" \
    "cat > $REMOTE_SCILINK_DIR/.env && chmod 600 $REMOTE_SCILINK_DIR/.env" \
    << REMOTE_ENV
# Auto-generated by deploy_scilink_to_client.sh -- do not commit.
# Source this file (or pass --api-key) before running scilink analyze
# in interactive smoke-test mode (without the eBPF tracer).
export OPENAI_API_KEY="${OPENAI_API_KEY}"
REMOTE_ENV

echo "==> [3b/4] Writing ${REMOTE_HOME}/$REMOTE_HARNESS_NAME/.env.scilink"
ssh "$SSH_USER@$CLIENT_NODE" \
    "cat > $REMOTE_HARNESS_NAME/.env.scilink && chmod 600 $REMOTE_HARNESS_NAME/.env.scilink" \
    << REMOTE_TRACE_ENV
# Auto-generated by deploy_scilink_to_client.sh -- do not commit.
# Sourced by trace_script_bcc_scilink.sh so sudo -E (which sets HOME=/root)
# doesn't strip these.  Paths are ABSOLUTE (resolved at deploy time) for
# the same reason.
export OPENAI_API_KEY="${OPENAI_API_KEY}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"

# Prevent Python from writing .pyc files into the venv.  Without this,
# sudo -E runs leave root-owned __pycache__/*.pyc that the regular user
# can't delete, breaking the next uv pip install on re-deploy.
export PYTHONDONTWRITEBYTECODE=1

# Bypass SciLink's interactive sandbox-approval prompt (scilink/executors.py).
# On CloudLab bare metal (not Docker/VM/Colab), the score-based sandbox
# detector falls below threshold and SciLink would otherwise abort the
# code-exec step with "No sandbox detected and non-interactive terminal."
# We're already running under sudo + eBPF supervision; SciLink can't escape
# anything our trace harness doesn't already permit, so the prompt is a
# false-positive in this context.
export UNSAFE_EXECUTION_OK=true

# Pin all paths absolutely (HOME=/root under sudo -E would break the rest).
export SCILINK_REPO="${REMOTE_HOME}/$REMOTE_SCILINK_DIR"
export TRACER_PYTHON="/usr/bin/python3"
export AGENT_PYTHON="${REMOTE_HOME}/$REMOTE_SCILINK_DIR/.venv/bin/python"
export POST_PYTHON="${REMOTE_HOME}/$REMOTE_SCILINK_DIR/.venv/bin/python"

# Default model + Lustre data dir (matches deploy-time choices).
export SCILINK_MODEL="gpt-4o-mini"
export DATA_DIR="/mnt/lustrefs/scilink_data"
REMOTE_TRACE_ENV

# ----------------------------------------------------------------
# 4) Verification recipe
# ----------------------------------------------------------------
cat <<EOF

==> [4/4] Deploy complete

A. Interactive smoke-test (NO eBPF, NO autorun) — confirms scilink works:

  ssh $SSH_USER@$CLIENT_NODE
  cd ~/$REMOTE_SCILINK_DIR
  source .env
  .venv/bin/scilink analyze \\
      --model gpt-4o-mini \\
      --mode autonomous \\
      --data examples/eels_plasmons_demo/datacube.npy \\
      --metadata examples/eels_plasmons_demo/datacube.json \\
      --session-dir ./smoke_session
  # type a prompt at "👤 You:" e.g. "Find plasmon peaks"

B. Full eBPF-traced run (everything end-to-end, prompt fed via stdin):

  ssh $SSH_USER@$CLIENT_NODE
  cd ~/$REMOTE_HARNESS_NAME
  cat .env.scilink                  # confirm AGENT_PYTHON / OPENAI_API_KEY
  \${EDITOR:-vi} config_scilink.env # tweak WORKLOADS / RUN_WORKLOADS

  # sudo required for BCC; -E preserves OPENAI_API_KEY via .env.scilink.
  sudo -E bash trace_script_bcc_scilink.sh

  # Outputs land in:
  #   ~/$REMOTE_HARNESS_NAME/traces/<timestamp>/<workload>/
  # including parsed.json, pi_summary.json, visualizations/index.html.
Once this is green, we layer on eBPF in the next phase.

EOF
