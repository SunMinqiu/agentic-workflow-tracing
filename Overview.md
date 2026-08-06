# Agentic Scientific Workflow I/O：运行手册与研究概述

## 第一部分：运行手册

### 0. 从当前状态开始

三台机器的职责如下。

| 机器 | 变量 | 职责 |
| --- | --- | --- |
| Mac | 无 | 发起操作并拉回结果 |
| Workflow cluster | `WORKFLOW_NODE` | 运行 workflow、BCC 和后处理 |
| vLLM cluster | `VLLM_NODE` | 运行 vLLM server |

现有脚本读取 `CLIENT_NODE`。本手册始终令 `CLIENT_NODE="$WORKFLOW_NODE"`。

根据当前状态选择入口，然后按章节编号向后执行。

| 当前状态 | 执行顺序 |
| --- | --- |
| 新 Workflow cluster，GenoMAS 使用 OpenAI | 1 → 2.1 → 2.2 → 4.1 → 6.1 → 7 → 8 |
| 新 Workflow cluster，GenoMAS 使用 FreeInference | 1 → 2.1 → 2.2 → 4.2 → 6.1 → 7 → 8 |
| 新 Workflow cluster，GenoMAS 或 SciLink 使用 vLLM | 1 → 2.1 → 2.2 → 3 → 4.3 → 6 → 7 → 8 |
| 新 Workflow cluster，SciLink 使用 OpenAI | 1 → 2.1 → 2.2 → 4.1 → 6.2 → 7 → 8 |
| 新 Workflow cluster，运行 1000 Genomes | 1 → 2.1 → 2.3 → 5 → 6.3 → 7 → 8 |
| 新 Workflow cluster，运行 Montage | 1 → 2.1 → 2.3 → 2.4 → 5 → 6.4 → 7 → 8 |
| 两个 cluster 已配置，新开 Mac terminal | 1 → 4.4 → 6 → 7 → 8 |
| 只改了代码 | 1 → 5 → 6 → 7 → 8 |
| 只换了 API key、后端或模型 | 1 → 4 → 6 → 7 → 8 |
| 只换了 Workflow cluster | 按上方对应的新 Workflow cluster 路径执行 |
| 只换了 vLLM cluster | 1 → 3 → 4 → 6 → 7 → 8 |
| 只运行新实验 | 1 → 4.4 → 6 → 7 → 8 |
| 实验正在运行 | 1 → 7 → 8 |
| 实验已经完成 | 1 → 8 |

### 1. 每个新 Mac terminal

在 Mac 的仓库根目录执行：

```zsh
source cloudlab_env.sh
export WORKFLOW_NODE="${WORKFLOW_NODE:-$CLIENT_NODE}"
export CLIENT_NODE="$WORKFLOW_NODE"
ssh "$SSH_USER@$WORKFLOW_NODE" true
printf 'WORKFLOW=%s\nVLLM=%s\n' "$WORKFLOW_NODE" "${VLLM_NODE:-not-set}"
```

看到 `[cloudlab_env] keys OK`，且 SSH 没有报错，即可继续。

### 2. 第一次拿到 Workflow cluster

本节只在新节点执行。部署会创建虚拟环境。代码更新不得执行本节。

#### 2.1 公共配置

检查 Lustre：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE" "mountpoint -q '$MOUNT_PATH'"
```

失败时先执行：

```zsh
bash scripts/setup_lustre_simple.sh
```

安装 BCC：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE" '
  sudo dnf install -y --setopt=install_weak_deps=False \
    bcc-tools python3-bcc "kernel-devel-$(uname -r)" git curl rsync &&
  /usr/bin/python3 -c "from bcc import BPF"
'
```

#### 2.2 Agentic workflow

部署所需 workflow：

```zsh
bash scripts/deploy_genomas_to_client.sh
bash scripts/deploy_scilink_to_client.sh
```

只执行需要的一行。两个都需要时依次执行。分别检查：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE" 'test -x "$HOME/GenoMAS/.venv/bin/python"'
ssh "$SSH_USER@$WORKFLOW_NODE" 'test -x "$HOME/SciLink/.venv/bin/python"'
```

只检查已经部署的 workflow。

使用 OpenAI 或 FreeInference 时前往第 4 节。使用 vLLM 时前往第 3 节。

#### 2.3 1000 Genomes

在 Workflow cluster 执行一次：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE"
```

```bash
LUSTRE_USER_DIR="${MOUNT_PATH:-/mnt/lustrefs}/$USER"
mkdir -p "$LUSTRE_USER_DIR"
git clone https://github.com/pegasus-isi/1000genome-workflow.git \
  "$LUSTRE_USER_DIR/1000genome-workflow"
cd "$LUSTRE_USER_DIR/1000genome-workflow"
mkdir -p data/20130502/sifting
bash prepare_input.sh
curl -LsSf https://astral.sh/uv/install.sh | sh
"$HOME/.local/bin/uv" venv --python 3.10 .venv
"$HOME/.local/bin/uv" pip install --python .venv/bin/python \
  numpy matplotlib pillow pandas plotly
exit
```

运行前必须存在所选 chromosome 的主 VCF 和 annotation VCF。文件必须解压为 `.vcf`。

完成后前往第 5 节。

#### 2.4 Montage

Montage 使用 1000 Genomes 的 Python 创建独立环境。在 Mac 执行一次：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE" 'bash -s' <<'REMOTE'
set -euo pipefail
ROOT="/mnt/lustrefs/$USER/montage"
BASE_PYTHON="/mnt/lustrefs/$USER/1000genome-workflow/.venv/bin/python"
TRACE_VENV="$ROOT/trace-venv"
test -x "$BASE_PYTHON"
mkdir -p "$ROOT"/{tmp,logs,input,cache/pip,cache/python,cache/matplotlib,cache/xdg,cache/config}
"$BASE_PYTHON" -m venv "$TRACE_VENV"
"$TRACE_VENV/bin/python" -m pip install \
  MontagePy astropy numpy matplotlib pillow pandas plotly importlib_resources
"$TRACE_VENV/bin/python" -c 'import MontagePy, numpy, pandas, plotly'
REMOTE
```

完成后前往第 5 节。

### 3. 第一次使用或更换 vLLM cluster

本仓库不启动 vLLM server。保持 vLLM 的 `srun` terminal 运行，在新的 Mac terminal 设置计算节点：

```zsh
export VLLM_NODE="<运行 vLLM 的 Perlmutter compute node>"
```

建立 Perlmutter 到 Mac 的 tunnel。该 terminal 保持运行：

```zsh
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=60 \
  -L "18000:$VLLM_NODE:8000" \
  mqsun@perlmutter.nersc.gov
```

在另一个 Mac terminal 建立 Mac 到 Workflow cluster 的 reverse tunnel。该 terminal 也保持运行：

```zsh
source cloudlab_env.sh
export WORKFLOW_NODE="${WORKFLOW_NODE:-$CLIENT_NODE}"
export CLIENT_NODE="$WORKFLOW_NODE"

ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=60 \
  -R 18080:127.0.0.1:18000 \
  "$SSH_USER@$WORKFLOW_NODE"
```

在第三个 Mac terminal 检查 endpoint：

```zsh
source cloudlab_env.sh
export WORKFLOW_NODE="${WORKFLOW_NODE:-$CLIENT_NODE}"
export CLIENT_NODE="$WORKFLOW_NODE"

ssh "$SSH_USER@$WORKFLOW_NODE" \
  "curl --connect-timeout 5 --max-time 10 -fsS \
  http://127.0.0.1:18080/v1/models | python3 -m json.tool"
```

将返回结果中的 `data[0].id` 设为模型：

```zsh
export VLLM_URL="http://127.0.0.1:18080"
export VLLM_SERVED_MODEL="<data[0].id>"
unset VLLM_API_KEY
```

连接路径为 `Workflow cluster:18080 → Mac:18000 → Perlmutter compute node:8000`。更换 vLLM compute node 后重建第一个 tunnel。更换 Workflow cluster 后重建第二个 tunnel。成功后执行第 4.3 节。

### 4. 配置推理后端

GenoMAS 支持三种后端。SciLink 的图像 workload 需要视觉模型。

| 后端 | Key | Base URL | Model |
| --- | --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | 不设置 | OpenAI 模型 |
| FreeInference | `GENOMAS_OPENAI_API_KEY` | `GENOMAS_BASE_URL` | 模型裸名 |
| vLLM | `VLLM_API_KEY` 或占位值 | `VLLM_URL/v1` | served model |

#### 4.1 OpenAI

在 Mac 设置模型：

```zsh
export GENOMAS_MODEL="gpt-4o-mini-2024-07-18"
export SCILINK_MODEL="gpt-4o-mini"
```

写入 Workflow cluster：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE" \
  "if [ -f pi-ebpf-tracing-handoff/.env.genomas ]; then sed -i -E '/^(export )?(OPENAI_API_KEY_1|OPENAI_ORGANIZATION_1|OPENAI_BASE_URL|OPENAI_API_BASE|GENOMAS_MODEL|GENOMAS_VENDOR)=/d' pi-ebpf-tracing-handoff/.env.genomas; cat >> pi-ebpf-tracing-handoff/.env.genomas; chmod 600 pi-ebpf-tracing-handoff/.env.genomas; fi" <<EOF
export OPENAI_API_KEY_1="$OPENAI_API_KEY"
export OPENAI_ORGANIZATION_1="$OPENAI_ORGANIZATION"
export GENOMAS_MODEL="$GENOMAS_MODEL"
export GENOMAS_VENDOR="OpenAI"
EOF

ssh "$SSH_USER@$WORKFLOW_NODE" \
  "if [ -f pi-ebpf-tracing-handoff/.env.scilink ]; then sed -i -E '/^(export )?(OPENAI_API_KEY|OPENAI_BASE_URL|OPENAI_API_BASE|SCILINK_MODEL|SCILINK_VENDOR)=/d' pi-ebpf-tracing-handoff/.env.scilink; cat >> pi-ebpf-tracing-handoff/.env.scilink; chmod 600 pi-ebpf-tracing-handoff/.env.scilink; fi" <<EOF
export OPENAI_API_KEY="$OPENAI_API_KEY"
export SCILINK_MODEL="$SCILINK_MODEL"
export SCILINK_VENDOR="OpenAI"
EOF
```

#### 4.2 FreeInference

在 Mac 设置：

```zsh
export GENOMAS_MODEL="qwen3.6-35b"
```

写入 Workflow cluster：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE" \
  "test -f pi-ebpf-tracing-handoff/.env.genomas && sed -i -E '/^(export )?(OPENAI_API_KEY_1|OPENAI_ORGANIZATION_1|OPENAI_BASE_URL|OPENAI_API_BASE|GENOMAS_MODEL|GENOMAS_VENDOR)=/d' pi-ebpf-tracing-handoff/.env.genomas && cat >> pi-ebpf-tracing-handoff/.env.genomas && chmod 600 pi-ebpf-tracing-handoff/.env.genomas" <<EOF
export OPENAI_API_KEY_1="$GENOMAS_OPENAI_API_KEY"
export OPENAI_BASE_URL="$GENOMAS_BASE_URL"
export OPENAI_API_BASE="$GENOMAS_BASE_URL"
export GENOMAS_MODEL="$GENOMAS_MODEL"
export GENOMAS_VENDOR="FreeInference"
EOF
```

FreeInference 不用于当前 SciLink 图像 workload。

#### 4.3 vLLM

在 Mac 执行：

```zsh
source config/config_vllm_endpoint.env
```

写入 Workflow cluster：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE" \
  "if [ -f pi-ebpf-tracing-handoff/.env.genomas ]; then sed -i -E '/^(export )?(OPENAI_API_KEY_1|OPENAI_ORGANIZATION_1|OPENAI_BASE_URL|OPENAI_API_BASE|GENOMAS_MODEL|GENOMAS_VENDOR)=/d' pi-ebpf-tracing-handoff/.env.genomas; cat >> pi-ebpf-tracing-handoff/.env.genomas; chmod 600 pi-ebpf-tracing-handoff/.env.genomas; fi" <<EOF
export OPENAI_API_KEY_1="$OPENAI_API_KEY_1"
export OPENAI_ORGANIZATION_1="local-vllm"
export OPENAI_BASE_URL="$OPENAI_BASE_URL"
export OPENAI_API_BASE="$OPENAI_API_BASE"
export GENOMAS_MODEL="$GENOMAS_MODEL"
export GENOMAS_VENDOR="vLLM"
EOF

ssh "$SSH_USER@$WORKFLOW_NODE" \
  "if [ -f pi-ebpf-tracing-handoff/.env.scilink ]; then sed -i -E '/^(export )?(OPENAI_API_KEY|OPENAI_BASE_URL|OPENAI_API_BASE|SCILINK_MODEL|SCILINK_VENDOR)=/d' pi-ebpf-tracing-handoff/.env.scilink; cat >> pi-ebpf-tracing-handoff/.env.scilink; chmod 600 pi-ebpf-tracing-handoff/.env.scilink; fi" <<EOF
export OPENAI_API_KEY="$OPENAI_API_KEY"
export OPENAI_BASE_URL="$OPENAI_BASE_URL"
export OPENAI_API_BASE="$OPENAI_API_BASE"
export SCILINK_MODEL="$SCILINK_MODEL"
export SCILINK_VENDOR="vLLM"
EOF
```

#### 4.4 每次运行前检查

查看当前后端：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE" '
  for file in .env.genomas .env.scilink; do
    path="$HOME/pi-ebpf-tracing-handoff/$file"
    [[ -f "$path" ]] || continue
    set -a; source "$path"; set +a
    printf "%s vendor=%s model=%s base=%s\n" \
      "$file" "${GENOMAS_VENDOR:-${SCILINK_VENDOR:-}}" \
      "${GENOMAS_MODEL:-${SCILINK_MODEL:-}}" "${OPENAI_BASE_URL:-default}"
    unset GENOMAS_VENDOR SCILINK_VENDOR GENOMAS_MODEL SCILINK_MODEL OPENAI_BASE_URL
  done
'
```

OpenAI：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE" \
  '! grep -q "^export OPENAI_BASE_URL=" pi-ebpf-tracing-handoff/.env.genomas &&
   ! grep -q "^export OPENAI_BASE_URL=" pi-ebpf-tracing-handoff/.env.scilink'
```

FreeInference 或 vLLM：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE" '
  set -a
  source pi-ebpf-tracing-handoff/.env.genomas
  set +a
  curl -fsS "${OPENAI_BASE_URL%/}/models" >/dev/null
'
```

后端检查通过后，代码未变化时前往第 6 节。代码变化时先执行第 5 节。

### 5. 修改代码后同步整个仓库

在 Mac 执行。该命令以整个仓库为同步范围，不重建虚拟环境，不覆盖 key 和结果：

```zsh
rsync -az --delete \
  --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude 'results/' --exclude '.venv/' --exclude '.env*' \
  ./ "$SSH_USER@$WORKFLOW_NODE:pi-ebpf-tracing-handoff/"
```

检查远端脚本：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE" \
  'cd pi-ebpf-tracing-handoff && bash -n scripts/trace_script_bcc_genomas.sh scripts/trace_script_bcc_scilink.sh'
```

成功后前往第 6 节。

### 6. 每次运行

下面以 GenoMAS、vLLM 和 `A_c2_w1` 为例。运行其他组合时修改前三行：

```zsh
WORKFLOW="genomas"
BACKEND="vllm"
WORKLOAD_TAG="A_c2_w1"
RUN_NAME="${WORKFLOW}__${BACKEND}__${WORKLOAD_TAG}__$(date +%Y%m%d_%H%M%S)"
REMOTE_RUN="$MOUNT_PATH/$SSH_USER/pi-ebpf-tracing-handoff/results/$RUN_NAME"
REMOTE_LOG="logs/$RUN_NAME.log"
ssh "$SSH_USER@$WORKFLOW_NODE" 'mkdir -p "$HOME/logs"'
```

运行名只使用字母、数字、点、下划线和连字符。

#### 6.1 GenoMAS

可选 workload：

```text
A_c1_w1,A_c2_w1,A_c2_w2,A_c3_w2,A_c4_w2,A_c4_w4,A_c8_w4
B_t1_w2,B_t2_w2,B_t4_w2
```

在 Mac 设置并启动：

```zsh
RUN_WORKLOADS="A_c2_w1"
ssh "$SSH_USER@$WORKFLOW_NODE" \
  "cd pi-ebpf-tracing-handoff &&
   sudo -n true &&
   nohup sudo -n -E env BASE_OUT='$REMOTE_RUN' RUN_WORKLOADS='$RUN_WORKLOADS' \
     bash scripts/trace_script_bcc_genomas.sh \
     >\"\$HOME/$REMOTE_LOG\" 2>&1 </dev/null &
   echo PID=\$! LOG=\$HOME/$REMOTE_LOG RESULTS='$REMOTE_RUN'"
```

空的 `RUN_WORKLOADS` 运行全部 10 个 cell。

#### 6.2 SciLink

可选 workload：

```text
eels_plasmons_basic
eels_identification_basic
polycrystalline_grains_basic
planning_critical_materials
```

在 Mac 设置并启动：

```zsh
RUN_WORKLOADS="polycrystalline_grains_basic"
ssh "$SSH_USER@$WORKFLOW_NODE" \
  "cd pi-ebpf-tracing-handoff &&
   sudo -n true &&
   nohup sudo -n -E env BASE_OUT='$REMOTE_RUN' RUN_WORKLOADS='$RUN_WORKLOADS' \
     bash scripts/trace_script_bcc_scilink.sh \
     >\"\$HOME/$REMOTE_LOG\" 2>&1 </dev/null &
   echo PID=\$! LOG=\$HOME/$REMOTE_LOG RESULTS='$REMOTE_RUN'"
```

#### 6.3 1000 Genomes

该 workflow 不使用第 4 节。先运行最小 cell：

```zsh
RUN_WORKLOADS="classic_chr1_r1"
ssh "$SSH_USER@$WORKFLOW_NODE" \
  "cd pi-ebpf-tracing-handoff &&
   sudo -n true &&
   nohup sudo -n -E env \
     BASE_OUT='$REMOTE_RUN' \
     WORKFLOW_REPO='$MOUNT_PATH/$SSH_USER/1000genome-workflow' \
     DATASET_DIR='$MOUNT_PATH/$SSH_USER/1000genome-workflow/data/20130502' \
     POPULATION_DIR='$MOUNT_PATH/$SSH_USER/1000genome-workflow/data/populations' \
     AGENT_PYTHON='$MOUNT_PATH/$SSH_USER/1000genome-workflow/.venv/bin/python' \
     POST_PYTHON='$MOUNT_PATH/$SSH_USER/1000genome-workflow/.venv/bin/python' \
     CLASSIC_OFFLINE=1 RUN_WORKLOADS='$RUN_WORKLOADS' \
     bash scripts/trace_script_bcc_1000genome.sh \
     >\"\$HOME/$REMOTE_LOG\" 2>&1 </dev/null &
   echo PID=\$! LOG=\$HOME/$REMOTE_LOG RESULTS='$REMOTE_RUN'"
```

默认矩阵包含 1、2、4 个 chromosome，各重复三次。正式运行前先确认所需 VCF 已准备。

#### 6.4 Montage

该 workflow 不使用第 4 节。每个规模先准备一次固定输入：

```zsh
SIZE=0.10
ssh "$SSH_USER@$WORKFLOW_NODE" "bash -s -- $SIZE" <<'REMOTE'
set -euo pipefail
SIZE="$1"
ROOT="/mnt/lustrefs/$USER/montage"
TAG="${SIZE/./p}deg"
cd "$HOME/pi-ebpf-tracing-handoff"
PYTHONPATH="$PWD/src" "$ROOT/trace-venv/bin/python" \
  -m agent_io_tracing.adapters.montage.prepare_input \
  --size-deg "$SIZE" --output "$ROOT/input/m17_${TAG}"
REMOTE
```

准备完成后启动：

```zsh
RUN_WORKLOADS="montage_m17_0p10_r1"
ssh "$SSH_USER@$WORKFLOW_NODE" \
  "cd pi-ebpf-tracing-handoff &&
   sudo -n true &&
   nohup sudo -n -E env \
     BASE_OUT='$REMOTE_RUN' \
     MONTAGE_ROOT='$MOUNT_PATH/$SSH_USER/montage' \
     MONTAGE_INPUT_ROOT='$MOUNT_PATH/$SSH_USER/montage/input' \
     MONTAGE_PYTHON='$MOUNT_PATH/$SSH_USER/montage/trace-venv/bin/python' \
     AGENT_PYTHON='$MOUNT_PATH/$SSH_USER/montage/trace-venv/bin/python' \
     POST_PYTHON='$MOUNT_PATH/$SSH_USER/montage/trace-venv/bin/python' \
     MONTAGE_OFFLINE=1 RUN_WORKLOADS='$RUN_WORKLOADS' \
     bash scripts/trace_script_bcc_montage.sh \
     >\"\$HOME/$REMOTE_LOG\" 2>&1 </dev/null &
   echo PID=\$! LOG=\$HOME/$REMOTE_LOG RESULTS='$REMOTE_RUN'"
```

规模为 `0p10`、`0p25` 和 `0p50`。每个规模有 `r1`、`r2` 和 `r3`。

启动成功后保留当前 terminal 中的 `RUN_NAME`、`REMOTE_RUN` 和 `REMOTE_LOG`，前往第 7 节。

### 7. 查看进度

查看日志：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE" "tail -f \"\$HOME/$REMOTE_LOG\""
```

`Ctrl-C` 只停止查看。检查 workflow、tracer 和后处理：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE" \
  "pgrep -af 'trace_script_bcc_|bcc_tracer|phase1_metrics|viz.trace|run_1000genome|run_montage' || true"
```

日志出现 `Results:`、`Results in:` 或 `All done. Results in:`，且进程列表为空，表示完成。进程为空但没有完成标记，表示失败：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE" "tail -100 \"\$HOME/$REMOTE_LOG\""
```

完成后前往第 8 节。

### 8. 拉回结果

#### 8.1 GenoMAS 和 SciLink

```zsh
bash scripts/pull_agentic_run.sh "$REMOTE_LOG"
```

脚本读取日志中的精确结果目录，拉回到 `results/$RUN_NAME`，检查必需文件和 `lost_events=0`，再打开报告。

#### 8.2 1000 Genomes 和 Montage

```zsh
LOCAL_RUN="results/$RUN_NAME"
mkdir -p "$LOCAL_RUN"
rsync -az --progress --checksum --partial \
  --exclude 'work/' --exclude 'bcc.out' \
  "$SSH_USER@$WORKFLOW_NODE:$REMOTE_RUN/" "$LOCAL_RUN/"
```

检查每个 cell：

```zsh
failed=0
cells=0
for cell in "$LOCAL_RUN"/*; do
  [[ -f "$cell/manifest.json" ]] || continue
  ((cells += 1))
  [[ -s "$cell/parsed.json" ]] || failed=1
  [[ -s "$cell/phase1_metrics.json" ]] || failed=1
  [[ -s "$cell/visualizations/index.html" ]] || failed=1
  tail -1 "$cell/bcc.err" | grep -q 'lost_events=0' || failed=1
done
(( cells > 0 )) || failed=1
(( failed == 0 )) && open "$LOCAL_RUN"/*/visualizations/index.html
```

不要按目录时间猜测结果。若当前 terminal 已关闭，先从运行日志找回精确路径：

```zsh
REMOTE_LOG="logs/<run-name>.log"
REMOTE_RUN=$(ssh "$SSH_USER@$WORKFLOW_NODE" \
  "sed -n -E 's/^(Output dir: |Results: |Results in: |All done\\. Results in: )//p' \"\$HOME/$REMOTE_LOG\" | tail -1")
RUN_NAME=$(basename "$REMOTE_RUN")
```

---

## 第二部分：研究概述



### 1. 研究动机

传统科学工作流通常具有固定的 DAG、明确的任务依赖和稳定的生产者—消费者数据流。因此，已有研究主要通过任务结构、文件复用、访问类型、操作次数、数据流规模和带宽解释工作流的 I/O 行为。

Agentic scientific workflow 的执行路径还受 LLM agent 的运行时决策影响。Agent 可以选择工具、检查文件、重试失败步骤、调试错误，并配置下游科学任务。部分 agentic workflow 用于处理过去依赖人工判断的任务，本身没有可直接对照的传统工作流。

本项目研究真实部署的 agentic scientific workflow，区分其从传统科学工作流继承的 I/O 行为与 agent 执行引入或改变的 I/O 行为。研究不要求每个目标系统都存在传统工作流对照。

### 2. 核心研究问题

当 LLM agent 执行科学工作流时会产生哪些 I/O 模式？这些模式与固定 DAG 的传统科学工作流有何不同？

具体问题包括：

- 正常科学任务产生多少 I/O？
- 探索、调试、重试和重复读取等 agent 行为引入多少 I/O？
- 可证明为次优的配置产生多少 I/O？这些配置可能来自 agent 生成的代码，也可能来自工作流的固定脚本。
- 同一科学目标在重复运行中的 I/O footprint 有多稳定？其中多少变化可归因于 agent 行为？



### 3. 研究范围

本项目分析 agent 行为和科学任务执行产生的文件系统 I/O。以下内容不在研究范围内：

- LLM 模型加载
- KV cache paging
- 模型 offloading
- LLM serving 系统内部的存储行为

本项目将 LLM 视为工作流控制器，而非待刻画的存储负载。

#### 3.1 结果存储规范

仓库中的持久化本地结果统一存放在：

```text
results/
```

每次追踪使用以下目录结构：

```text
results/<run_id>/<workload>/
```

`<run_id>` 通常是时间戳或实验名称，`<workload>` 是被追踪的 cell 或用例。每个 cell 的完整结果必须保存在同一目录，包括 `ebpf_events.log`、`parsed.json`、`pi_events.jsonl`、`tool_calls.log`、`phase1_metrics.json`、`lineage/`、`visualizations/`，以及 `scilink_session/` 或 `work/` 等系统专用目录。

`remote_results/` 只能作为从 CloudLab 或其他远端机器传输结果时的临时缓存。验证后的持久化结果必须移至 `results/`，随后删除临时副本。

远端机器在 Lustre 上使用相同的逻辑入口：

```text
/mnt/lustrefs/<user>/pi-ebpf-tracing-handoff/results/
```

CloudLab client 不得将新的追踪结果写入仓库 checkout、home 目录或根文件系统。新的追踪脚本应将 `BASE_OUT` 默认为 `/mnt/lustrefs/<user>/pi-ebpf-tracing-handoff/results/<run_id>`。拉回的本地副本必须保持 `results/<run_id>/<workload>/` 结构。

### 4. 目标系统与编排固定度

不同 agentic scientific workflow 的执行路径固定程度差异很大。本项目将编排固定度视为独立变量，不把所有 agentic workflow 当作同一类系统，也不强制为每个系统寻找传统工作流基线。

对目标仓库的直接检查得到以下结论：

- **GenoMAS**：用于基因表达分析、GEO/TCGA 预处理和回归。每个 agent 角色都按照 `prompts/action_units/base/*.json` 中的有序 Action Units 运行。Agent 决定每个固定阶段生成什么代码，但不决定阶段及其顺序。`environment.py` 中的 trait、condition、cohort、checkpoint 和目录管理均由确定性 Python 代码执行。
- **SRAgent** 与 **ChemGraph**：分别用于 NCBI/SRA 元数据处理和 XANES 模拟。两者都使用 LangGraph ReAct 风格的 supervisor。Supervisor 根据消息历史在运行时选择下一个子 agent 或工具，没有固定阶段列表。工具调用数量和顺序也是运行时决策。
- **SciLink**：用于显微和材料表征。其 autonomous 模式包含 `best_of_n_orchestrator`、`refinement_loop` 和 `multiskill_autoselect`。分析技能的选择和 refinement 次数均由 agent 在运行时决定。
- **CMBAgent**：基于 AG2/AutoGen 的通用多 agent 科研系统。`planning_and_control` 模式先由 `planner` 与 `plan_reviewer` 在 `max_n_attempts` 范围内生成有序计划，再由 `control` 或 `controller` 逐步分派给 `engineer`、`researcher` 或领域 agent。计划结构由 LLM 动态生成，但执行阶段相对固定。`one_shot` 模式没有规划阶段，适合作为低成本的首次集成目标。

GenoMAS 位于固定端，SRAgent、ChemGraph 和 autonomous SciLink 位于动态端，CMBAgent 的 `planning_and_control` 模式位于两者之间。实验报告必须标明每个系统的编排固定度。

### 5. I/O 归因类别

每个观测到的 I/O 单元只能归入以下三类之一。分类依据是该 I/O 的具体成因，不需要构造全局最优基线或“必要 I/O 下限”。

#### 5.1 Agent-induced I/O

这类 I/O 由本次运行中的 agent 行为产生。在任务和配置相同的条件下，行为不同的 agent 不一定会产生这些 I/O。

典型实例包括：

- 代码执行失败后的重复调试和重试
- 同一任务内对同一文件的冗余读取
- 读取错误日志
- 多次检查中间结果
- 被放弃的代码尝试留下的文件

较高的 agent-induced I/O 表明 agent 尚不能用少量直接步骤完成任务，或现有文件系统和工具接口不适合 agent 的搜索、验证与错误恢复方式。这一现象本身就是研究结果。

#### 5.2 Task-misconfigured I/O

同一任务语义存在更优配置，且当前配置可证明产生了更高 I/O 成本时，相关 I/O 归入此类。检测标准只判断配置是否次优，不判断配置来源。

确认实例后，再标记其来源：

- **agent-caused**：次优选择来自本次运行中 agent 生成的代码。例如，使用逐文件 POSIX 读取而不使用已有的批处理接口，或反复解析大型原始文件而不缓存结果。
- **script-caused**：次优选择来自工作流固定的编排代码或工具代码，因此每次运行都会出现。例如，GenoMAS 的 `environment.py` 在每次 cohort 循环中调用 `os.listdir()`，或 `tools/preprocess.py` 的 `validate_and_save_cohort_info` 在每个 cohort 完成后持有 `fcntl` 锁执行完整 JSON 的读取、修改和写回。

比较 agent-caused 和 script-caused 的发生率，可以判断 agent 生成代码是否比人工编写的固定脚本引入更多配置问题。

#### 5.3 Workflow task-induced I/O

排除前两类后，剩余 I/O 归入此类。它包括读取必要输入数据和写入最终结果等行为。该类别由排除法定义，不代表已经计算出任务的绝对最小 I/O。

### 6. 指标



#### 6.1 通用 I/O 指标

以下指标覆盖所有 I/O，并按三类归因结果聚合：

- 读写字节数
- 读写操作数
- 元数据操作数
- 访问的唯一文件数
- 小文件访问数和小 I/O 操作数
- I/O 时间、有效带宽和 duty cycle
- 读写比

Duty cycle 定义为：

```text
|union(read/write syscall intervals)| / group wall-clock time
```

全局指标使用整个运行的 wall time。Phase 和 role 指标使用 tool-call 时间区间的并集。Inference-busy 和 inference-idle 指标分别使用 LLM 时间区间并集与剩余运行时间。

#### 6.2 Agent-induced I/O 指标

- 目录扫描数
- 失败的 `open` 和 `stat` 数
- 同一文件和同一版本文件的重复读取数
- 错误日志读取数
- 输出检查数
- 重试产生的 I/O 字节数和操作数
- 临时文件数
- 废弃 artifact 数
- 冗余读取比例
- 非生产性 I/O 比例



#### 6.3 Task-misconfigured I/O 指标

目标系统运行在使用共享或并行文件系统的真实 HPC 集群上，包括 CloudLab、DARWIN 和 RCCS 上的 Lustre。以下指标默认适用于所有目标系统。每个系统需要确认 I/O 实际落在本地 scratch 还是 Lustre。

- I/O 接口，包括 POSIX、批处理库调用和适用场景中的 parallel I/O
- 输出文件数和平均文件大小
- checkpoint 和元数据写入频率
- scratch 与共享或并行文件系统之间的存储位置选择
- 多 worker 或多 rank 任务的 I/O 规模与时间不均衡
- agent-caused 与 script-caused 的比例

多 worker 场景包括 GenoMAS 的 `--parallel-mode cohorts` 和 ChemGraph 的 ensemble 或 FDMNES 运行。

#### 6.4 运行间方差指标

这些指标用于衡量同一科学目标重复运行时的 I/O 可预测性，并判断方差来自 agent-induced、task-misconfigured 还是 workflow task-induced I/O：

- 总 I/O 字节数和操作数方差
- 元数据操作数方差
- 唯一文件数方差
- 三类归因结果各自的 I/O 方差
- I/O 时间方差
- 总运行时间方差



### 7. 研究流程



#### Phase 1：完整采集指标

先在真实目标系统上验证第 6 节的指标。初始验证使用低成本的小规模实验，例如 GenoMAS 对 1 至 2 个 trait 或 cohort 的 quick test。只有确认现有 eBPF/BCC 基础设施能够提取所需指标后，才能扩大实验规模并接入更多系统。

GenoMAS 完整运行需要 3 至 5 天，成本超过 300 美元，因此必须先完成低成本验证。

#### Phase 2：基于 provenance 的归因

将原始 I/O trace 与 tool call 或 Action Unit 的时间区间结合，把每个被标记的 I/O 结果归入 agent-induced、task-misconfigured 或 workflow task-induced。

时间关联可以使用目标系统的执行日志，也可以使用仓库现有的时间窗口匹配逻辑。具体实现不改变三类归因框架。

#### Phase 3：优化案例

选择少量证据最充分的实例，展示修复方法和 I/O 改善。优先选择 script-caused 的 task-misconfigured I/O，因为这类路径是确定性的，一次修复可作用于后续所有运行，也便于进行清晰的前后对比。

### 8. 关键对比



#### 8.1 传统脚本工作流与 agentic workflow

只有在同一科学目标确实存在传统脚本实现时，才进行直接对比。该对比用于识别 agentic 编排引入的额外 I/O。

对于 GenoMAS 等没有直接传统对照的系统，比较两类执行特征：

- 传统工作流：固定 DAG、稳定的生产者—消费者边和稳定的 I/O 模式
- Agentic workflow：动态执行路径、更多文件探索、更高元数据量和更高运行间方差



#### 8.2 同一目标的重复 agent 运行

对同一 prompt 运行多次，衡量 I/O 的可预测性，并判断变化来自 agent 行为还是底层任务与配置。

#### 8.3 次优配置的局部反事实对比

Task-misconfigured I/O 不需要全局推荐配置作为基线。只需证明修复某个具体实例后，在保持任务语义不变的条件下降低了 I/O。

Script-caused 实例使用补丁前后对比。Agent-caused 实例可以比较重复运行中采用和未采用次优配置的结果。

### 9. 预期发现

- 真实 agentic scientific workflow 中存在不可忽略的 agent-induced I/O。其占比反映 agent 可靠性及文件系统和工具接口是否适合 agent 的访问模式。
- 在重复运行中，agent-induced I/O 的方差高于 task-misconfigured 和 workflow task-induced I/O。
- Agent 生成代码与工作流固定脚本都可能产生 task-misconfigured I/O。比较两者的发生率，可以判断 agent 生成代码是否构成额外的配置错误来源。
- 编排更动态的 SRAgent、ChemGraph 和 autonomous SciLink 可能比固定 Action Unit 序列的 GenoMAS 表现出更高的 I/O 运行间方差。



### 10. 主要贡献

本项目的主要贡献不是提出单个新指标，而是：

1. 刻画 GenoMAS、SRAgent、ChemGraph、SciLink 和 CMBAgent 等真实 agentic scientific workflow 的 I/O 行为，并覆盖不同的编排固定度。
2. 提出基于 provenance 的三类 I/O 归因框架，将 agent 行为、次优配置和工作流任务本身产生的 I/O 分开。次优配置进一步标记为 agent-caused 或 script-caused。
3. 通过少量具体案例，量化修复次优配置后的 I/O 改善。



### 11. 与既有工作的关系

传统 HPC I/O characterization 研究分析大规模系统中的访问模式、文件复用、共享行为、读写特征、带宽、操作数和方差。Workflow-centric 研究进一步把这些行为与 DAG、阶段和生产者—消费者关系关联起来。

本项目沿用这些分析方法，但研究对象具有新的执行特征：

- 执行路径可能由 agent 动态生成，不同系统的动态程度不同。
- 同一科学目标在重复运行中可能产生不同的 I/O footprint。
- Agent 可能引入探索、调试、重试和冗余读取。
- Agent 生成代码和人工编写的固定脚本都可能造成 I/O 次优配置。

因此，本项目关注 agentic execution 条件下的 I/O 行为，而非只分析固定工作流。
