#!/bin/bash
#
# Phase-1 deploy of GenoMAS (https://github.com/Liu-Hy/GenoMAS) to a CloudLab
# CentOS Stream 8 client node.  This pass is INTENTIONALLY MINIMAL: it does
# NOT install BCC, does NOT install the eBPF .env, does NOT touch the SRAgent
# or SciLink state.  Goal: prove that
#
#   python main.py --version smoke --model gpt-5-mini-2025-08-07 --api 1 \
#       --quick-test --data-root /mnt/lustrefs/genomas_data
#
# runs end-to-end against OpenAI on the client.  Once that's green, Phase 2
# layers in the LLM-call logger (genomas_tool_logger.py), and Phase 3 adds
# the eBPF tracer.
#
# Run from the laptop, AFTER sourcing cloudlab_env.sh:
#
#     source ~/Desktop/Benchmarking_Agents/pi-ebpf-tracing-handoff/cloudlab_env.sh
#     bash deploy_genomas_to_client.sh
#
# Re-runnable: clone uses pull-if-exists, uv venv is --clear, .env files
# are overwritten.  Data dir is NOT touched (you populate it manually from
# the GenoTEX Google Drive — see the bottom of this file).
#
# Key differences from deploy_scilink_to_client.sh:
#   - Python 3.10 (GenoMAS requirements.txt is pinned via README)
#   - .env uses `_1` suffix for keys (GenoMAS load-balancing convention)
#   - OpenAI requires BOTH OPENAI_API_KEY_1 AND OPENAI_ORGANIZATION_1
#   - Data dir layout is `<root>/GEO/<cohort_dirs>` + `<root>/TCGA/<trait>/<files>`
#

set -euo pipefail

# ============================================================
# Vars (defaults pull from cloudlab_env.sh if you sourced it)
# ============================================================
SSH_USER="${SSH_USER:-Minqiu}"
CLIENT_NODE="${CLIENT_NODE:-c220g1-030602.wisc.cloudlab.us}"
MOUNT_PATH="${MOUNT_PATH:-/mnt/lustrefs}"

# Local source dir for the tracing harness (rsync'd to remote).
LOCAL_HARNESS="${LOCAL_HARNESS:-$HOME/Desktop/Benchmarking_Agents/pi-ebpf-tracing-handoff}"

# Remote layout — independent of ~/SRAgent and ~/SciLink so all three coexist.
REMOTE_HARNESS_NAME="${REMOTE_HARNESS_NAME:-pi-ebpf-tracing-handoff}"
REMOTE_GENOMAS_DIR="${REMOTE_GENOMAS_DIR:-GenoMAS}"
GENOMAS_GIT_URL="${GENOMAS_GIT_URL:-https://github.com/Liu-Hy/GenoMAS.git}"
# Pin to main now; replace with a commit SHA once we verify a working revision.
GENOMAS_GIT_REF="${GENOMAS_GIT_REF:-main}"

# Lustre-side data dir.  You populate this manually from the GenoTEX Google
# Drive — see the END_OF_DEPLOY_NOTES section at the bottom.
REMOTE_DATA_DIR="${REMOTE_DATA_DIR_GENOMAS:-${MOUNT_PATH}/genomas_data}"

# ----------------------------------------------------------------
# Pre-flight (local)
# ----------------------------------------------------------------
if [ ! -d "$LOCAL_HARNESS" ]; then
    echo "Error: local harness dir not found: $LOCAL_HARNESS" >&2
    exit 1
fi

# GenoMAS mandates OPENAI_API_KEY + OPENAI_ORGANIZATION for any OpenAI model.
# Other providers (Anthropic / Google) are optional.
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "Error: OPENAI_API_KEY is not set locally." >&2
    echo "       source cloudlab_env.sh first (or export OPENAI_API_KEY=...)." >&2
    exit 1
fi
if [ -z "${OPENAI_ORGANIZATION:-}" ]; then
    echo "Warning: OPENAI_ORGANIZATION is not set; GenoMAS will fail at" >&2
    echo "         ModelConfig.create() when using any OpenAI model." >&2
    echo "         Add to cloudlab_env.sh: export OPENAI_ORGANIZATION='org-...'." >&2
fi

# ----------------------------------------------------------------
# 1) SSH reachability + Lustre mount sanity
# ----------------------------------------------------------------
echo "==> [1/5] SSH + Lustre check ($SSH_USER@$CLIENT_NODE)"
ssh "$SSH_USER@$CLIENT_NODE" "true"

ssh "$SSH_USER@$CLIENT_NODE" "bash -s" << REMOTE_LUSTRE
set -euo pipefail
if ! mountpoint -q "$MOUNT_PATH"; then
    echo "ERROR: $MOUNT_PATH is not a mountpoint.  Did setup_lustre_simple.sh succeed?" >&2
    exit 1
fi
echo "  Lustre mount OK: $MOUNT_PATH"
# Three user-owned subdirs on Lustre (root of $MOUNT_PATH is owned by root,
# can't be chmod'd; we instead create per-project siblings and chown them):
#   - $REMOTE_DATA_DIR             : GenoMAS input data (GEO/, TCGA/)
#   - $MOUNT_PATH/genomas_venv     : Python venv (kept separate from data
#                                    tree so we can tar the data tree alone
#                                    as evidence)
#   - $MOUNT_PATH/genomas_output   : GenoMAS ./output symlink target
#   - $MOUNT_PATH/$SSH_USER/pi-ebpf-tracing-handoff/results : trace outputs
#   - $MOUNT_PATH/uv_cache_genomas : uv wheel cache (deduplicated across
#                                    deploy re-runs)
sudo mkdir -p "$REMOTE_DATA_DIR/GEO" "$REMOTE_DATA_DIR/TCGA" \
    "$MOUNT_PATH/genomas_venv" "$MOUNT_PATH/genomas_output" \
    "$MOUNT_PATH/$SSH_USER/pi-ebpf-tracing-handoff/results" "$MOUNT_PATH/uv_cache_genomas"
sudo chown -R "$SSH_USER" "$REMOTE_DATA_DIR" "$MOUNT_PATH/genomas_venv" \
    "$MOUNT_PATH/genomas_output" "$MOUNT_PATH/$SSH_USER/pi-ebpf-tracing-handoff/results" \
    "$MOUNT_PATH/uv_cache_genomas" || true
echo "  Lustre data dir ready: $REMOTE_DATA_DIR"
echo "  Lustre venv dir  ready: $MOUNT_PATH/genomas_venv"
echo "  Lustre output dir ready: $MOUNT_PATH/genomas_output"
echo "  Lustre results dir ready: $MOUNT_PATH/$SSH_USER/pi-ebpf-tracing-handoff/results"
echo "  Lustre cache dir ready: $MOUNT_PATH/uv_cache_genomas"
# Quick smoke write to confirm we still have write perms (catches the case
# where a prior 'sudo -E' run reverted ownership to root).
touch "$REMOTE_DATA_DIR/.deploy_writetest" && rm "$REMOTE_DATA_DIR/.deploy_writetest"
REMOTE_LUSTRE

# ----------------------------------------------------------------
# 1b) rsync the tracing harness (so future Phase-2 logger + Phase-3 tracer
#      land on the client without needing a second sync round).
# ----------------------------------------------------------------
echo "==> [1b] rsync $LOCAL_HARNESS/ → $CLIENT_NODE:~/$REMOTE_HARNESS_NAME/"
rsync -av \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'traces/' \
    --exclude 'results/' \
    --exclude '.venv/' \
    --exclude '.env' \
    --exclude '.env.sragent' \
    --exclude '.env.scilink' \
    --exclude '.env.genomas' \
    "$LOCAL_HARNESS/" \
    "$SSH_USER@$CLIENT_NODE:$REMOTE_HARNESS_NAME/"

# ----------------------------------------------------------------
# 2) uv + GenoMAS clone + uv venv (Python 3.10, on Lustre to dodge the 16GB
#    root-disk limit), uv pip install -r requirements.txt
# ----------------------------------------------------------------
echo "==> [2/5] uv + GenoMAS (~/$REMOTE_GENOMAS_DIR/.venv on Python 3.10)"
ssh -T "$SSH_USER@$CLIENT_NODE" "bash -s" << REMOTE_UV
set -euo pipefail
cd ~

# Common tools (idempotent; install only if missing — SRAgent deploy may
# have already installed them).
if ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
    sudo dnf install -y --setopt=install_weak_deps=False git curl
fi

# uv installer (idempotent).
if ! command -v uv >/dev/null 2>&1 && [ ! -x "\$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="\$HOME/.local/bin:\$PATH"
uv --version

# Clone or update GenoMAS.
if [ -d "$REMOTE_GENOMAS_DIR/.git" ]; then
    echo "  GenoMAS repo exists; fetching"
    git -C "$REMOTE_GENOMAS_DIR" fetch --quiet origin
    git -C "$REMOTE_GENOMAS_DIR" checkout --quiet $GENOMAS_GIT_REF
    git -C "$REMOTE_GENOMAS_DIR" pull --quiet --ff-only origin $GENOMAS_GIT_REF || true
else
    git clone --quiet --branch $GENOMAS_GIT_REF $GENOMAS_GIT_URL $REMOTE_GENOMAS_DIR
fi

cd "$REMOTE_GENOMAS_DIR"
echo "  HEAD: \$(git rev-parse --short HEAD)"

# GenoMAS writes sizeable preprocess artefacts to ./output.  Keep that path
# as a symlink into Lustre so live runs do not fill the 16 GB root disk.
mkdir -p /mnt/lustrefs/genomas_output
if [ -L output ]; then
    ln -sfn /mnt/lustrefs/genomas_output output
elif [ -d output ]; then
    if find output -mindepth 1 -print -quit | grep -q .; then
        echo "  Preserving existing ./output contents by moving them to Lustre"
        rsync -a output/ /mnt/lustrefs/genomas_output/
    fi
    rm -rf output
    ln -s /mnt/lustrefs/genomas_output output
else
    ln -s /mnt/lustrefs/genomas_output output
fi
echo "  GenoMAS output -> \$(readlink -f output)"

# CloudLab home is ~16 GB with little free.  GenoMAS deps are smaller than
# SciLink+torch (no CUDA wheels) but still ~1-2 GB.  Put the venv + uv cache
# on Lustre and symlink, same pattern as SciLink.
# Writability test targets genomas_data (user-owned subdir), not the mount
# root (which is root-owned and would always fail).
if grep -qE "[[:space:]]/mnt/lustrefs[[:space:]]" /proc/mounts && [ -w /mnt/lustrefs/genomas_data ]; then
    VENV_DIR=/mnt/lustrefs/genomas_venv
    export UV_CACHE_DIR=/mnt/lustrefs/uv_cache_genomas
    mkdir -p "\$UV_CACHE_DIR"
    echo "  venv:     \$VENV_DIR (Lustre)"
    echo "  uv cache: \$UV_CACHE_DIR (Lustre)"
else
    VENV_DIR=.venv
    echo "  WARNING: Lustre not mounted; venv on local home (may run out)"
fi
rm -rf "\$HOME/.cache/uv/.tmp"* 2>/dev/null || true

# GenoMAS README pins Python 3.10 (conda example).  uv downloads a portable
# 3.10 interpreter if the host doesn't have one — CentOS Stream 8 ships 3.6.
uv venv --clear --python 3.10 "\$VENV_DIR"

if [ "\$VENV_DIR" != ".venv" ]; then
    rm -rf .venv
    ln -s "\$VENV_DIR" .venv
fi

# Install requirements.  GenoMAS' requirements.txt has unpinned versions
# (a known footgun), so add --upgrade-strategy=eager to make this
# deterministic-ish on re-deploy.  No CUDA, no torch — much faster than
# SciLink's deploy.
echo "  Installing GenoMAS requirements..."
uv pip install --python .venv/bin/python -r requirements.txt

# Post-processing extras for visualize_strace.py (Phase 3).  pandas/numpy/
# matplotlib are already in requirements.txt; only plotly is extra.
echo "  Installing visualization extras (plotly)..."
uv pip install --python .venv/bin/python plotly

# Smoke imports.  These are the imports main.py does at module load time.
# If any of these fail, no point continuing.
.venv/bin/python - << 'PYEOF'
import importlib, importlib.metadata as md
errors = []
# Note: PyPI 'biopython' imports as 'Bio'; 'python-dotenv' imports as 'dotenv'.
for pkg in ("openai", "anthropic", "google.generativeai", "ollama",
            "dotenv", "backoff", "pandas", "numpy", "statsmodels",
            "Bio", "matplotlib", "seaborn", "networkx", "plotly"):
    try:
        importlib.import_module(pkg)
    except Exception as e:
        errors.append(f"{pkg}: {e!r}")
if errors:
    print("IMPORT ERRORS:")
    for e in errors:
        print("  " + e)
    raise SystemExit(1)
print("All imports OK")
# Show key versions for reproducibility.
for pkg in ("openai", "anthropic", "google-generativeai"):
    try:
        print(f"{pkg}: {md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"{pkg}: (unknown)")
PYEOF

# Confirm GenoMAS entry point file is present.
test -f main.py || { echo "ERROR: GenoMAS main.py missing"; exit 1; }

# --- FreeInference (custom OpenAI-compatible endpoint) model registration ---
# GenoMAS's utils/llm.py has a CLOSED MODEL_INFO registry; validate_model()
# rejects any model not in it ("Could not detect a provider").  When a custom
# OPENAI_BASE_URL + GENOMAS_MODEL are configured, register that model under
# MODEL_INFO['openai'] so it routes through OpenAIClient, whose AsyncOpenAI is
# built with NO base_url and therefore auto-reads OPENAI_BASE_URL from env.
# Idempotent; no-op unless both values are set (i.e. FreeInference mode).
GENOMAS_MODEL_REG="${GENOMAS_MODEL:-}"
OPENAI_BASE_REG="${OPENAI_BASE_URL:-}"
if [ -n "\$OPENAI_BASE_REG" ] && [ -n "\$GENOMAS_MODEL_REG" ]; then
  .venv/bin/python - "\$GENOMAS_MODEL_REG" << 'PYEOF'
import sys
model = sys.argv[1]
path = "utils/llm.py"
src = open(path).read()
if f"'{model}'" in src:
    print(f"  [freeinference] model {model} already in MODEL_INFO; skip")
else:
    anchor = "    'openai': {\n"
    if anchor in src:
        entry = (f"        '{model}': {{'input_price': 0.0, 'output_price': 0.0}},"
                 f"  # FreeInference (OpenAI-compatible via OPENAI_BASE_URL)\n")
        open(path, "w").write(src.replace(anchor, anchor + entry, 1))
        print(f"  [freeinference] registered {model} under MODEL_INFO['openai']")
    else:
        print("  [freeinference] WARNING: MODEL_INFO['openai'] anchor not found; skipped")
PYEOF
fi

# --- cohort-count knob for I/O experiments (GENOMAS_MAX_COHORTS) ---
# GenoMAS's environment.py lists ALL GEO cohorts per trait with no cap, so the
# workload (and its I/O) can't be swept by cohort count.  Patch the enumeration
# to honour GENOMAS_MAX_COHORTS (0/unset = all).  Idempotent.
.venv/bin/python - << 'PYEOF'
path = "environment.py"
src = open(path).read()
if "GENOMAS_MAX_COHORTS" in src:
    print("  [max-cohorts] environment.py already patched; skip")
else:
    needle = "cohorts = os.listdir(geo_trait_dir) + ['TCGA']"
    idx = src.find(needle)
    if idx == -1:
        print("  [max-cohorts] WARNING: cohort-listing line not found; skipped")
    else:
        line_start = src.rfind("\n", 0, idx) + 1
        indent = src[line_start:idx]
        block = (
            f"{indent}_mc = int(os.environ.get('GENOMAS_MAX_COHORTS', '0') or '0')\n"
            f"{indent}if _mc > 0:\n"
            f"{indent}    cohorts = sorted(os.listdir(geo_trait_dir))[:_mc]\n"
            f"{indent}else:\n"
            f"{indent}    cohorts = os.listdir(geo_trait_dir) + ['TCGA']"
        )
        src = src[:line_start] + block + src[idx + len(needle):]
        open(path, "w").write(src)
        print("  [max-cohorts] patched environment.py cohort enumeration")
PYEOF
.venv/bin/python -c "from utils.config import setup_arg_parser; setup_arg_parser().parse_args(['--version','probe','--model','gpt-5-mini-2025-08-07']); print('CLI parser OK')"
REMOTE_UV

# ----------------------------------------------------------------
# 3) Write two .env files:
#    a) ~/GenoMAS/.env             — used by main.py via python-dotenv
#       (`cd ~/GenoMAS; python main.py ...`)
#    b) ~/pi-ebpf-tracing-handoff/.env.genomas — used by the future
#       trace_script_bcc_genomas.sh (Phase 3).  Absolute paths so
#       `sudo -E` (HOME=/root) doesn't break AGENT_PYTHON resolution.
# ----------------------------------------------------------------
REMOTE_HOME=$(ssh -T "$SSH_USER@$CLIENT_NODE" 'echo $HOME')

# FreeInference / any OpenAI-compatible endpoint (optional).  Emitted into both
# env files only when OPENAI_BASE_URL is set locally (e.g. cloudlab_env.sh).
# NOTE: GenoMAS uses a custom _1-suffixed key convention in its own LLM client;
# whether that client honours OPENAI_BASE_URL must be confirmed on the client
# (test one cell) — writing the var is harmless but may need a GenoMAS-side tweak.
if [ -n "${OPENAI_BASE_URL:-}" ]; then
    OPENAI_BASE_BLOCK="export OPENAI_BASE_URL=\"${OPENAI_BASE_URL}\"
export OPENAI_API_BASE=\"${OPENAI_API_BASE:-$OPENAI_BASE_URL}\""
    # ~/GenoMAS/.env is a plain dotenv (no `export` keyword needed).
    OPENAI_BASE_BLOCK_DOTENV="OPENAI_BASE_URL=${OPENAI_BASE_URL}
OPENAI_API_BASE=${OPENAI_API_BASE:-$OPENAI_BASE_URL}"
else
    OPENAI_BASE_BLOCK="# No custom OPENAI_BASE_URL set; using provider default (api.openai.com)."
    OPENAI_BASE_BLOCK_DOTENV="# No custom OPENAI_BASE_URL set; using provider default (api.openai.com)."
fi

echo "==> [3a/5] Writing ${REMOTE_HOME}/$REMOTE_GENOMAS_DIR/.env"
ssh "$SSH_USER@$CLIENT_NODE" \
    "cat > $REMOTE_GENOMAS_DIR/.env && chmod 600 $REMOTE_GENOMAS_DIR/.env" \
    << REMOTE_GENOMAS_ENV
# Auto-generated by deploy_genomas_to_client.sh -- do not commit.
# python-dotenv loads this when main.py imports utils.llm (via load_dotenv()).
# GenoMAS uses the _1 / _2 / _3 suffix convention for API load balancing.
# Phase 1: single key per provider with _1 suffix.

# --- OpenAI (REQUIRED: both key and organization for gpt-* / o* models) ---
OPENAI_API_KEY_1=${OPENAI_API_KEY}
OPENAI_ORGANIZATION_1=${OPENAI_ORGANIZATION:-}

# --- Anthropic (optional; needed for --model claude-*) ---
ANTHROPIC_API_KEY_1=${ANTHROPIC_API_KEY:-}

# --- Google (optional; needed for --model gemini-*) ---
GOOGLE_API_KEY_1=${GOOGLE_API_KEY:-}

# --- Novita (optional; needed for --use-api with open-source models) ---
NOVITA_API_KEY_1=${NOVITA_API_KEY:-}

# --- Custom OpenAI-compatible endpoint (e.g. FreeInference), if set ---
${OPENAI_BASE_BLOCK_DOTENV}
REMOTE_GENOMAS_ENV

echo "==> [3b/5] Writing ${REMOTE_HOME}/$REMOTE_HARNESS_NAME/.env.genomas"
ssh "$SSH_USER@$CLIENT_NODE" \
    "cat > $REMOTE_HARNESS_NAME/.env.genomas && chmod 600 $REMOTE_HARNESS_NAME/.env.genomas" \
    << REMOTE_TRACE_ENV
# Auto-generated by deploy_genomas_to_client.sh -- do not commit.
# Sourced by trace_script_bcc_genomas.sh (Phase 3) so sudo -E (HOME=/root)
# doesn't strip these.  Paths absolute.
export OPENAI_API_KEY_1="${OPENAI_API_KEY}"
export OPENAI_ORGANIZATION_1="${OPENAI_ORGANIZATION:-}"
export ANTHROPIC_API_KEY_1="${ANTHROPIC_API_KEY:-}"
export GOOGLE_API_KEY_1="${GOOGLE_API_KEY:-}"
export NOVITA_API_KEY_1="${NOVITA_API_KEY:-}"
${OPENAI_BASE_BLOCK}

# Prevent root-owned __pycache__ files under sudo -E.
export PYTHONDONTWRITEBYTECODE=1

# Absolute paths (HOME=/root under sudo -E breaks the rest).
export GENOMAS_REPO="${REMOTE_HOME}/$REMOTE_GENOMAS_DIR"
export TRACER_PYTHON="/usr/bin/python3"
export AGENT_PYTHON="${REMOTE_HOME}/$REMOTE_GENOMAS_DIR/.venv/bin/python"
export POST_PYTHON="${REMOTE_HOME}/$REMOTE_GENOMAS_DIR/.venv/bin/python"

# Defaults for the workload matrix (Phase 4).
# Override GENOMAS_MODEL in the local env (cloudlab_env.sh) to switch providers,
# e.g. export GENOMAS_MODEL="openai/glm-5.1" for FreeInference.
export GENOMAS_MODEL="${GENOMAS_MODEL:-gpt-5-mini-2025-08-07}"
export DATA_DIR="${REMOTE_DATA_DIR}"
REMOTE_TRACE_ENV

# ----------------------------------------------------------------
# 4) Data presence check (does NOT download; tells the user what to do)
# ----------------------------------------------------------------
echo "==> [4/5] Data dir status"
ssh -T "$SSH_USER@$CLIENT_NODE" "bash -s" << REMOTE_DATA
set -u
GEO_COUNT=\$(ls -1 "${REMOTE_DATA_DIR}/GEO" 2>/dev/null | wc -l)
TCGA_COUNT=\$(ls -1 "${REMOTE_DATA_DIR}/TCGA" 2>/dev/null | wc -l)
echo "  GEO subdirs:  \$GEO_COUNT  (expected: 4-8 cohort folders for smoke)"
echo "  TCGA subdirs: \$TCGA_COUNT (expected: ≥1 trait folder)"
if [ "\$GEO_COUNT" -eq 0 ] && [ "\$TCGA_COUNT" -eq 0 ]; then
    echo "  → Data dir is empty.  Smoke run will fail until you populate it."
fi
REMOTE_DATA

# ----------------------------------------------------------------
# 5) Final pointer + verification recipe
# ----------------------------------------------------------------
cat <<EOF

==> [5/5] Deploy complete

A. Confirm the venv works (no LLM call, no data needed):

  ssh $SSH_USER@$CLIENT_NODE
  cd ~/$REMOTE_GENOMAS_DIR
  .venv/bin/python -c "import sys; print(sys.version)"
  .venv/bin/python main.py --help | head -30

B. Populate the data dir from the GenoTEX Google Drive:

   1. Open https://drive.google.com/drive/folders/1kxHOyW5wNnY3Rk15xwLaM7ZZS01wGzRO
      on your laptop.
   2. Download 4–8 cohort folders from GEO/ and 1 trait folder from TCGA/.
      Each cohort is typically a few hundred MB.  Don't bother with the full 42 GB.
   3. scp them to the client:
        scp -r ./GEO/<cohort_name> $SSH_USER@$CLIENT_NODE:${REMOTE_DATA_DIR}/GEO/
        scp -r ./TCGA/<trait_name> $SSH_USER@$CLIENT_NODE:${REMOTE_DATA_DIR}/TCGA/
   4. Validate (optional):
        ssh $SSH_USER@$CLIENT_NODE
        cd ~/$REMOTE_GENOMAS_DIR
        .venv/bin/python download/validator.py --data-dir ${REMOTE_DATA_DIR} --validate

C. Smoke run (no tracer, no logger, just GenoMAS):
   Note: this will iterate over EVERY pair in metadata/task_info.json.
   For a true small-scale smoke, Phase 2 will install a task_info.json
   slicer in analyze_codebase_genomas.py.  For now, expect the full
   benchmark run unless you slice metadata/task_info.json yourself.

  ssh $SSH_USER@$CLIENT_NODE
  cd ~/$REMOTE_GENOMAS_DIR
  .venv/bin/python main.py \\
      --version smoke \\
      --model gpt-5-mini-2025-08-07 \\
      --api 1 \\
      --quick-test \\
      --data-root ${REMOTE_DATA_DIR}

  # Logs land in ./output/log_smoke.txt
  # Per-cohort preprocess output in ./output/preprocess/<trait>/

D. Next phases:
   - Phase 2: write genomas_tool_logger.py + analyze_codebase_genomas.py
              (slices task_info.json, installs LLMClient.generate_completion
              monkey-patch BEFORE GenoMAS imports).
   - Phase 3: write config_genomas.env + trace_script_bcc_genomas.sh
              (BCC tracer for FS+net syscalls, parses to parsed.json).

EOF
