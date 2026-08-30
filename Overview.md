# Agentic Scientific Workflow I/O：运行与研究设计

## 第一部分：运行手册

### 0. 机器与执行路径

| 机器 | 变量 | 职责 |
| --- | --- | --- |
| Mac | 无 | 配置节点、部署代码、启动实验、拉回结果 |
| Lustre MGS + MDT | `MGS_NODE` | 管理 Lustre |
| Lustre OST | `OST_NODE` | 保存 Lustre 数据 |
| Workflow client | `CLIENT_NODE`、`WORKFLOW_NODE` | 运行 workflow、BCC 和后处理 |
| vLLM cluster | `VLLM_NODE` | 运行 vLLM server |

`WORKFLOW_NODE` 始终等于 `CLIENT_NODE`。

| 当前状态 | 执行顺序 |
| --- | --- |
| 新节点，GenoMAS + OpenAI | 1 → 2.1 → 2.2 → 4.1 → 6.1 → 7 → 8 |
| 新节点，GenoMAS + FreeInference | 1 → 2.1 → 2.2 → 4.2 → 6.1 → 7 → 8 |
| 新节点，GenoMAS 或 SciLink + vLLM | 1 → 2.1 → 2.2 → 3 → 4.3 → 6 → 7 → 8 |
| 新节点，SciLink + OpenAI | 1 → 2.1 → 2.2 → 4.1 → 6.2 → 7 → 8 |
| 新节点，1000 Genomes | 1 → 2.1 → 2.3 → 5 → 6.3 → 7 → 8 |
| 新节点，Montage | 1 → 2.1 → 2.3 → 2.4 → 5 → 6.4 → 7 → 8 |
| 已配置，新开 Mac terminal | 4.4 → 6 → 7 → 8 |
| 只改代码 | 5 → 6 → 7 → 8 |
| 只换 API key、后端或模型 | 4 → 6 → 7 → 8 |
| 只换 vLLM cluster | 3 → 4 → 6 → 7 → 8 |
| 实验正在运行 | 7 → 8 |
| 实验已经完成 | 8 |

### 1. 更换 CloudLab allocation

在 Mac 编辑 `cloudlab_env.sh`：

```bash
export MGS_NODE="MGS 的完整主机名"
export OST_NODE="OST 的完整主机名"
export CLIENT_NODE="Workflow client 的完整主机名"
```

在仓库根目录加载新配置：

```zsh
source cloudlab_env.sh
printf 'MGS=%s\nOST=%s\nWORKFLOW=%s\n' \
  "$MGS_NODE" "$OST_NODE" "$WORKFLOW_NODE"
```

主机名必须与本次 CloudLab allocation 一致。

验证三台节点：

```zsh
for node in "$MGS_NODE" "$OST_NODE" "$CLIENT_NODE"; do
  ssh "$SSH_USER@$node" hostname
done
```

出现 `REMOTE HOST IDENTIFICATION HAS CHANGED` 时删除旧指纹：

```zsh
for node in "$MGS_NODE" "$OST_NODE" "$CLIENT_NODE"; do
  ssh-keygen -R "$node"
done
```

重新连接：

```zsh
for node in "$MGS_NODE" "$OST_NODE" "$CLIENT_NODE"; do
  ssh "$SSH_USER@$node" hostname
done
```

三条命令都必须返回远端 hostname。

### 2. 初始化新的 Workflow cluster

#### 2.1 配置 Lustre 和 BCC

三台节点全部更换时执行：

```zsh
bash scripts/setup_lustre_simple.sh
```

该脚本会格式化 MGS 和 OST 的非系统盘。仅用于新分配的空节点。

MGS + MDT 已完成，OST 或 client 配置中断时执行：

```zsh
bash scripts/setup_lustre_ost_client_only.sh
```

该脚本会重新格式化 OST 的非系统盘。

验证三个挂载点：

```zsh
ssh "$SSH_USER@$MGS_NODE" 'mountpoint -q /mnt/mgs_mdt && echo mgs-mounted'
ssh "$SSH_USER@$OST_NODE" 'mountpoint -q /mnt/ost0 && echo ost-mounted'
ssh "$SSH_USER@$CLIENT_NODE" "mountpoint -q '$MOUNT_PATH' && echo client-mounted"
```

输出必须包含 `mgs-mounted`、`ost-mounted` 和 `client-mounted`。

在 Workflow client 安装 BCC：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE" '
  sudo dnf install -y --setopt=install_weak_deps=False \
    bcc-tools python3-bcc "kernel-devel-$(uname -r)" git curl rsync &&
  /usr/bin/python3 -c "from bcc import BPF" &&
  echo bcc-ready
'
```

输出最后一行必须是 `bcc-ready`。

#### 2.2 Agentic workflow

##### 2.2.1 GenoMAS

该命令部署 tracing harness 和 GenoMAS，创建虚拟环境，并将 Mac 数据同步到 Lustre：

```zsh
source cloudlab_env.sh
LOCAL_GENOMAS_DATASET="/Users/minqiu/Desktop/Benchmarking_Agents/GenoMAS_Datasets" \
bash scripts/deploy_genomas_to_client.sh
```

输出最后一行必须是 `genomas-ready`。

##### 2.2.2 SciLink

```zsh
source cloudlab_env.sh
bash scripts/deploy_scilink_to_client.sh
ssh "$SSH_USER@$WORKFLOW_NODE" \
  'test -x "$HOME/SciLink/.venv/bin/python" && echo scilink-ready'
```

输出最后一行必须是 `scilink-ready`。

使用 OpenAI 或 FreeInference 时前往第 4 节。使用 vLLM 时前往第 3 节。

#### 2.3 1000 Genomes

只在新的 Workflow cluster 上执行一次。以下命令会克隆仓库并创建虚拟环境。在 Workflow cluster 执行：

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
test -x .venv/bin/python
test -d data/20130502/sifting
echo 1000genome-ready
exit
```

通过标准：输出最后一行为 `1000genome-ready`。运行前用以下命令确认所选 chromosome 的主 VCF 和 annotation VCF 已解压为 `.vcf`：

```bash
find data/20130502 -type f -name '*.vcf' -print
```

目标 chromosome 的两个文件都出现在输出中才可运行。

完成后前往第 5 节。

#### 2.4 Montage

Montage 使用 1000 Genomes 的 Python 创建独立环境。只在新的 Workflow cluster 上执行一次。在 Mac 执行：

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
echo montage-ready
REMOTE
```

通过标准：输出最后一行为 `montage-ready`。完成后前往第 5 节。

### 3. 第一次使用或更换 vLLM cluster

本仓库不启动 vLLM server。保持 vLLM 的 `srun` terminal 运行，在新的 Mac terminal 设置计算节点：

```zsh
export VLLM_NODE="nid008384"
```

建立 Perlmutter 到 Mac 的 tunnel。该 terminal 保持运行：

```zsh
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=60 \
  -L "18000:$VLLM_NODE:8000" \
  -L "15558:$VLLM_NODE:5558" \
  mqsun@perlmutter.nersc.gov
```

在另一个 Mac terminal 建立 Mac 到 Workflow cluster 的 reverse tunnel。该 terminal 也保持运行：

```zsh
source cloudlab_env.sh
export WORKFLOW_NODE="$CLIENT_NODE"

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

通过标准：输出包含非空的 `data[0].id`。

将返回结果中的 `data[0].id` 设为模型：

```zsh
export VLLM_URL="http://127.0.0.1:18080"
export VLLM_SERVED_MODEL="Qwen3.6-27B"
unset VLLM_API_KEY
```

HTTP 连接路径为 `Workflow cluster:18080 → Mac:18000 → Perlmutter compute node:8000`。KV event replay 路径为 `Mac:15558 → Perlmutter compute node:5558`，仅供 Mac 上的固定输入测试使用。更换 vLLM compute node 后重建第一个 tunnel。更换 Workflow cluster 后重建第二个 tunnel。成功后执行第 4.3 节。

### 4. 配置推理后端

GenoMAS 支持三种后端。SciLink 的图像 workload 需要视觉模型。


| 后端            | Key                      | Base URL           | Model        |
| ------------- | ------------------------ | ------------------ | ------------ |
| OpenAI        | `OPENAI_API_KEY`         | 不设置                | OpenAI 模型    |
| FreeInference | `GENOMAS_OPENAI_API_KEY` | `GENOMAS_BASE_URL` | 模型裸名         |
| vLLM          | `VLLM_API_KEY` 或占位值      | `VLLM_URL/v1`      | served model |


#### 4.1 OpenAI

在 Mac 设置模型：

```zsh
source cloudlab_env.sh
unset VLLM_URL
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
source cloudlab_env.sh
unset VLLM_URL
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
source cloudlab_env.sh
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

选择本次 workflow 的环境文件：

```zsh
source cloudlab_env.sh
REMOTE_ENV=".env.genomas"  # SciLink 改为 .env.scilink
```

验证配置和 endpoint：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE" "bash -s -- '$REMOTE_ENV'" <<'REMOTE'
set -euo pipefail
source "$HOME/pi-ebpf-tracing-handoff/$1"
vendor="${GENOMAS_VENDOR:-${SCILINK_VENDOR:-}}"
model="${GENOMAS_MODEL:-${SCILINK_MODEL:-}}"
key="${OPENAI_API_KEY_1:-${OPENAI_API_KEY:-}}"
base="${OPENAI_BASE_URL:-${OPENAI_API_BASE:-}}"
test -n "$vendor" && test -n "$model" && test -n "$key"
if [[ "$vendor" == "OpenAI" ]]; then
  test -z "$base"
else
  test -n "$base"
  curl -fsS "${base%/}/models" >/dev/null
fi
printf 'backend-ok vendor=%s model=%s base=%s\n' \
  "$vendor" "$model" "${base:-default}"
REMOTE
```

通过标准：输出 `backend-ok` 及正确的 vendor、model 和 base。代码未变化时前往第 6 节，代码变化时先执行第 5 节。

### 5. 修改代码后同步整个仓库

在 Mac 执行。该命令以整个仓库为同步范围，不重建虚拟环境，不覆盖 key 和结果：

```zsh
source cloudlab_env.sh
rsync -az --delete \
  --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude 'results/' --exclude '.venv/' --exclude '.env*' \
  ./ "$SSH_USER@$WORKFLOW_NODE:pi-ebpf-tracing-handoff/"
```

检查远端脚本：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE" \
  'cd pi-ebpf-tracing-handoff && bash -n scripts/trace_script_bcc_genomas.sh scripts/trace_script_bcc_scilink.sh && echo sync-ok'
```

通过标准：输出 `sync-ok`。只运行其他 workflow 时，将对应 trace 脚本加入 `bash -n` 参数。通过后前往第 6 节。

### 6. 每次运行

结果目录由 trace 脚本命名，格式为 `<Workflow>_<task>_<时间戳>`，例如 `GenoMAS_A_c2_w1_20260806_101500`。多个 task 的运行写作 `<Workflow>_<N>tasks_<时间戳>`。目录名和报告统一使用 `America/New_York`，不受 Workflow 节点或 Mac 的系统时区影响。命名逻辑在 `scripts/lib_results.sh` 的 `run_dir_name`。

远端日志目录。每个新 Workflow cluster 建一次即可，已经建过的节点可以跳过；重复执行无副作用：

```zsh
source cloudlab_env.sh
ssh "$SSH_USER@$WORKFLOW_NODE" 'mkdir -p "$HOME/logs"'
```

下面每一节先设置 `RUN_WORKLOADS` 和 `REMOTE_LOG`，再启动。结果目录在运行开始后从日志读取，见第 7 节。

#### 6.1 GenoMAS

可选 workload：

```text
A_c1_w1,A_c2_w1,A_c3_w1,A_c4_w1,A_c8_w1,A_c2_w2,A_c3_w2,A_c4_w2,A_c4_w4,A_c8_w4
B_t1_w2,B_t2_w2,B_t4_w2
full_c1_w2
```

在 Mac 设置并启动：

```zsh
: "${VLLM_URL:?先 export VLLM_URL=http://127.0.0.1:18080；使用 OpenAI 或 FreeInference 时跳过此检查}"
RUN_WORKLOADS="A_c8_w4"
REMOTE_LOG="logs/genomas_${RUN_WORKLOADS}_$(date +%Y%m%d_%H%M%S).log"
ssh "$SSH_USER@$WORKFLOW_NODE" \
  "cd pi-ebpf-tracing-handoff &&
   sudo -n true &&
   nohup sudo -n -E env RUN_WORKLOADS='$RUN_WORKLOADS' VLLM_URL='$VLLM_URL' \
     bash scripts/trace_script_bcc_genomas.sh \
     >\"\$HOME/$REMOTE_LOG\" 2>&1 </dev/null &
   echo PID=\$! LOG=\$HOME/$REMOTE_LOG"
```

空的 `RUN_WORKLOADS` 运行全部 11 个 cell。

#### 6.2 SciLink

可选 workload：

```text
eels_plasmons_basic
eels_identification_basic
polycrystalline_grains_basic
# planning_critical_materials
```

以普通 SSH 用户启动 SciLink 的 trace 脚本。脚本会单独用 `sudo` 启动 BCC tracer。不要为整个脚本添加 `sudo`，否则模型缓存会写入 `/root`，脚本也会直接退出。命令中的 `sudo -n true` 只检查免密 sudo 是否可用。

第一步，验证流式响应重建。更换 vLLM server、模型或 launcher 代码后执行一次：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE" \
  'cd pi-ebpf-tracing-handoff && PYTHONPATH=src ~/SciLink/.venv/bin/python \
     tools/dump_stream_chunks.py --base-url http://127.0.0.1:18080 \
     --model Qwen3.6-27B --max-tokens 3000'
```

命令会发送一个纯文本请求和一个工具调用请求。退出码 0 表示流式内容可以完整重建。思考型模型需要足够的 `--max-tokens`，否则可能只返回推理内容，导致 agent 收到空回答并重试。

第二步，在 Mac 启动运行：

```zsh
: "${VLLM_URL:?先 export VLLM_URL=http://127.0.0.1:18080；使用 OpenAI 时跳过此检查}"
RUN_WORKLOADS="eels_plasmons_basic"
REMOTE_LOG="logs/scilink_${RUN_WORKLOADS}_$(date +%Y%m%d_%H%M%S).log"
ssh "$SSH_USER@$WORKFLOW_NODE" \
  "cd pi-ebpf-tracing-handoff &&
   sudo -n true &&
   nohup env RUN_WORKLOADS='$RUN_WORKLOADS' VLLM_URL='$VLLM_URL' \
     SCILINK_FORCE_STREAM=1 \
     bash scripts/trace_script_bcc_scilink.sh \
     >\"\$HOME/$REMOTE_LOG\" 2>&1 </dev/null &
   echo PID=\$! LOG=\$HOME/$REMOTE_LOG"
```

Launcher 会拦截重试循环。连续发送 20 次相同 prompt 时，进程以退出码 3 终止。`SCILINK_MAX_REPEAT_CALLS` 可以修改该阈值，设为 0 可关闭检查。`SCILINK_MAX_CALLS` 限制总调用数，默认值 0 表示不限制。

SciLink 默认不采集 `futex`、`epoll_wait` 等运行时等待事件，也不采集匿名 `mmap`。这些事件不能直接证明 agent 的有效 I/O，却会显著增加 trace 体积。文件映射仍会采集。只有调试调度和阻塞问题时才应直接为 tracer 添加 `--include-waits`。后处理默认最多并行执行两个互不依赖的步骤，可通过 `POSTPROCESS_MAX_WORKERS` 调整。

如需事后统计调用数，执行：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE" "grep -c message_end \$(ls -td /mnt/lustrefs/$SSH_USER/pi-ebpf-tracing-handoff/results/SciLink_* | head -1)/*/pi_events.jsonl"
```

`planning_critical_materials` 最容易触发重试。SciLink 上游在 `scilink/agents/planning_agents/orchestrator_tools.py:4168` 将 `max_output_tokens` 固定为 1024。思考型模型可能在生成正文前耗尽预算，随后进入解析失败和重试循环。`eels_plasmons_basic` 不经过这段代码。

#### 6.3 1000 Genomes

该 workflow 不使用第 4 节。先运行最小 cell：

```zsh
RUN_WORKLOADS="classic_chr1_r1"
REMOTE_LOG="logs/1000genome_${RUN_WORKLOADS}_$(date +%Y%m%d_%H%M%S).log"
ssh "$SSH_USER@$WORKFLOW_NODE" \
  "cd pi-ebpf-tracing-handoff &&
   sudo -n true &&
   nohup sudo -n -E env \
     WORKFLOW_REPO='$MOUNT_PATH/$SSH_USER/1000genome-workflow' \
     DATASET_DIR='$MOUNT_PATH/$SSH_USER/1000genome-workflow/data/20130502' \
     POPULATION_DIR='$MOUNT_PATH/$SSH_USER/1000genome-workflow/data/populations' \
     AGENT_PYTHON='$MOUNT_PATH/$SSH_USER/1000genome-workflow/.venv/bin/python' \
     POST_PYTHON='$MOUNT_PATH/$SSH_USER/1000genome-workflow/.venv/bin/python' \
     CLASSIC_OFFLINE=1 RUN_WORKLOADS='$RUN_WORKLOADS' \
     bash scripts/trace_script_bcc_1000genome.sh \
     >\"\$HOME/$REMOTE_LOG\" 2>&1 </dev/null &
   echo PID=\$! LOG=\$HOME/$REMOTE_LOG"
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
test -n "$(find "$ROOT/input/m17_${TAG}" -type f -print -quit)"
echo "input-ready: $ROOT/input/m17_${TAG}"
REMOTE
```

通过标准：输出 `input-ready:` 和对应目录。

准备完成后启动：

```zsh
RUN_WORKLOADS="montage_m17_0p10_r1"
REMOTE_LOG="logs/montage_${RUN_WORKLOADS}_$(date +%Y%m%d_%H%M%S).log"
ssh "$SSH_USER@$WORKFLOW_NODE" \
  "cd pi-ebpf-tracing-handoff &&
   sudo -n true &&
   nohup sudo -n -E env \
     MONTAGE_ROOT='$MOUNT_PATH/$SSH_USER/montage' \
     MONTAGE_INPUT_ROOT='$MOUNT_PATH/$SSH_USER/montage/input' \
     MONTAGE_PYTHON='$MOUNT_PATH/$SSH_USER/montage/trace-venv/bin/python' \
     AGENT_PYTHON='$MOUNT_PATH/$SSH_USER/montage/trace-venv/bin/python' \
     POST_PYTHON='$MOUNT_PATH/$SSH_USER/montage/trace-venv/bin/python' \
     MONTAGE_OFFLINE=1 RUN_WORKLOADS='$RUN_WORKLOADS' \
     bash scripts/trace_script_bcc_montage.sh \
     >\"\$HOME/$REMOTE_LOG\" 2>&1 </dev/null &
   echo PID=\$! LOG=\$HOME/$REMOTE_LOG"
```

规模为 `0p10`、`0p25` 和 `0p50`。每个规模有 `r1`、`r2` 和 `r3`。

启动成功后保留当前 terminal 中的 `REMOTE_LOG`，前往第 7 节。

四种启动命令的通过标准相同：立即输出非空的 `PID=` 和 `LOG=`。这只证明后台进程已创建，最终结果必须按第 7 节判断。

### 7. 查看进度

查看日志：

```zsh
source cloudlab_env.sh
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

结果目录由脚本命名，运行开始后从日志读取：

```zsh
REMOTE_RUN=$(ssh "$SSH_USER@$WORKFLOW_NODE" \
  "sed -n -E 's/^(Output dir: |Results: |Results in: |All done\\. Results in: )//p' \"\$HOME/$REMOTE_LOG\" | tail -1")
RUN_NAME=$(basename "$REMOTE_RUN")
printf 'RUN_NAME=%s\n' "$RUN_NAME"
```

完成后前往第 8 节。

### 8. 拉回结果



#### 8.1 GenoMAS 和 SciLink

```zsh
source cloudlab_env.sh
bash scripts/pull_agentic_run.sh "$REMOTE_LOG"
```

脚本读取日志中的精确结果目录，拉回到 `results/$RUN_NAME`，检查必需文件和 `lost_events=0`，再打开报告。

#### 8.2 1000 Genomes 和 Montage

GenoMAS 和 SciLink 不需要本节，8.1 的脚本自己建目录。下面第二行是每次运行都要执行的：每个 run 有各自的 `$RUN_NAME`，接收目录不会重复。

```zsh
source cloudlab_env.sh
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
if (( failed == 0 )); then
  echo "verified: $cells cells"
  open "$LOCAL_RUN"/*/visualizations/index.html
else
  echo "verification failed" >&2
  exit 1
fi
```

通过标准：输出 `verified: <数量> cells`，并打开每个 cell 的报告。任何必需文件为空或 `lost_events` 非零都会返回非零退出码。

不要按目录时间猜测结果。若当前 terminal 已关闭，设置 `REMOTE_LOG="logs/<日志名>.log"` 后重新执行第 7 节末尾的读取命令。

#### 8.3 用 vLLM 自己的分词器重算 KV 指标

本节只用于 vLLM，并且需要在隧道关闭前执行。本地 tiktoken 无法准确还原服务器使用的 chat template、tools schema 和图像输入。vLLM 的 `/tokenize` 使用服务器实际的分词器和 chat template，因此适合重算 KV cache 指标。

在 Mac 执行。Mac 上的隧道端口是 18000，Workflow cluster 上的 18080 端口在这里不可用。

```zsh
PYTHONPATH=src python3 -m agent_io_tracing.analysis.kvcache.server_tokens \
  --results results --url http://127.0.0.1:18000
PYTHONPATH=src python3 -m agent_io_tracing.analysis.kvcache.report \
  --results results --runs "$RUN_NAME" --dump-prefixes
PYTHONPATH=src python3 -m agent_io_tracing.analysis.results_index --results results
test -s "results/$RUN_NAME/server_tokens.json" 2>/dev/null || \
  find "results/$RUN_NAME" -mindepth 2 -maxdepth 2 -name server_tokens.json -size +0 -print -quit | grep -q .
test -s results/index.html && echo kvcache-report-ready
```

通过标准：输出最后一行为 `kvcache-report-ready`。第一条为每个 cell 写入 `server_tokens.json`，第二条重算当前 run，第三条刷新总索引。没有 `server_tokens.json` 的 cell 继续使用 tiktoken。

### 9. 固定输入测试

本节比较 vLLM 启动参数对前缀缓存、prefill 和 decode 性能的影响。重放不会运行 agent，也不会采集 BCC I/O trace。每个请求固定生成 32 个 token，因此不同 arm 的 TTFT、TPOT 和请求延迟可以直接比较。

重复运行 agent 会改变调用次数、prompt 和工具顺序。重放先固定一次真实运行的 prompt token 序列，再把相同输入发送给不同配置的 vLLM。

本节使用三个术语：

- Bundle：从一个 cell 提取的固定请求集合。
- Arm：在一种 vLLM 配置下重放 bundle 得到的结果。
- Sweep：使用同一 bundle 记录的多个 arm。

执行顺序为：创建一次 bundle，依次记录各个 arm，最后生成报告。

#### 9.1 创建 bundle

选择一个已拉回 Mac 的 cell。该目录必须包含 `messages.jsonl` 和 `pi_events.jsonl`。

```zsh
RUN_NAME="GenoMAS_A_c8_w4_20260818_153414"
CELL="results/$RUN_NAME/A_c8_w4"
test -s "$CELL/messages.jsonl" && test -s "$CELL/pi_events.jsonl"
```

确认第 3 节的 Perlmutter 到 Mac 隧道仍在运行，然后执行：

```zsh
MAC_VLLM_URL="http://127.0.0.1:18000"
VLLM_SERVED_MODEL="$(curl -fsS "$MAC_VLLM_URL/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
BUNDLE="replay/bundles/${RUN_NAME}_$(basename "$CELL").json"

curl -fsS "$MAC_VLLM_URL/v1/models" >/dev/null && echo tunnel-ok
mkdir -p replay/bundles
PYTHONPATH=src python3 -m agent_io_tracing.replay.bundle \
  "$CELL" \
  --output "$BUNDLE" \
  --url "$MAC_VLLM_URL" \
  --model "$VLLM_SERVED_MODEL"
python3 -c 'import json,sys; x=json.load(open(sys.argv[1])); assert x' "$BUNDLE" && echo bundle-ready
```

命令使用 vLLM 的 `/tokenize` 接口生成 prompt token ID。`--limit N` 只打包前 N 个请求，适合检查链路，不能用于正式结果。

通过标准：输出最后一行为 `bundle-ready`。

一个 sweep 的所有 arm 必须使用同一个 `BUNDLE`。文件名包含 `$RUN_NAME` 的时间戳，不需要另外创建 bundle ID。

#### 9.2 记录一个 arm

先在 Perlmutter 上使用待测参数启动 vLLM。重放工具不会设置 server knob。它会读取 `cache_config_info`，记录服务器实际使用的配置。

第一次在 Mac 的仓库目录运行 replay 时安装 KV event collector 依赖：

```bash
python3 -m pip install -e '.[observability]'
```

在 Mac 将观测配置复制到 Perlmutter。登录节点和 compute node 共享 `$HOME`，每次代码更新后执行一次：

```zsh
scp config/config_vllm_observability.sh \
  mqsun@perlmutter.nersc.gov:~/config_vllm_observability.sh
```

在 Perlmutter 原本启动 vLLM 的 `srun` shell 中加载配置：

```bash
source "$HOME/config_vllm_observability.sh"
printf 'observability arguments loaded: %s\n' "${VLLM_OBSERVABILITY_ARGS[*]}"
```

通过标准：输出包含 `--enable-prompt-tokens-details` 和 `--kv-events-config`。随后重新执行你当前使用的完整 `vllm serve` 命令，只在命令末尾增加 `"${VLLM_OBSERVABILITY_ARGS[@]}"`。本仓库不保存 Perlmutter 的模型路径、容器命令和 tensor-parallel 参数，因此教程不能安全地重写这条启动命令。不要改动它的模型路径、tensor parallel、Mamba 或其他原有参数。

这会启用逐请求 cached tokens、逐请求 server timing、KV residency histograms、调度 iteration 日志、prefix-cache reset 和带 replay buffer 的 KV block events。replay 保存原始 `kv_events.jsonl`，并将 `BlockRemoved` hash 与此前的 `BlockStored` 对齐后写入 `kv_removals.jsonl`。后者包含可恢复的 block token IDs 和 bundle request 候选位置。Mamba 或 SSM group 的事件会标为 ambiguous，不据此硬推 token 到 hash 的映射。第 3 节的 Perlmutter 到 Mac tunnel 已经转发 5558。回到 Mac 的仓库 terminal 设置：

```zsh
export VLLM_KV_EVENTS_REPLAY_ENDPOINT="tcp://127.0.0.1:15558"
```

正式 latency 运行应将 `VLLM_KV_METRICS_SAMPLE=0.01`，并按需从启动参数数组中移除 iteration logging 和 KV events。`1.0` 采样和全量事件适合诊断 eviction，不适合作为无观测开销的性能结果。

```zsh
MAC_VLLM_URL="http://127.0.0.1:18000"
VLLM_KV_EVENTS_REPLAY_ENDPOINT="tcp://127.0.0.1:15558"
SWEEP_DIR="replay/c8w4_warmup_$(date +%Y%m%d_%H%M%S)"

test -s "$BUNDLE"
curl -fsS "$MAC_VLLM_URL/v1/models" >/dev/null

PYTHONPATH=src python3 -m agent_io_tracing.replay.sweep arm \
  --bundle "$BUNDLE" \
  --sweep-dir "$SWEEP_DIR" \
  --label prefix_on_cold \
  --url "$MAC_VLLM_URL" \
  --mode serial \
  --cache-salt-mode shared \
  --output-tokens 256 \
  --kv-events-replay-endpoint "$VLLM_KV_EVENTS_REPLAY_ENDPOINT"
```

同一个 server 不重启，随后测预热过一次的相同 bundle：

```zsh
PYTHONPATH=src python3 -m agent_io_tracing.replay.sweep arm \
  --bundle "$BUNDLE" \
  --sweep-dir "$SWEEP_DIR" \
  --label prefix_on_prewarmed \
  --url "$MAC_VLLM_URL" \
  --mode serial \
  --cache-salt-mode shared \
  --warmup-repetitions 1 \
  --output-tokens 256 \
  --strict-config \
  --kv-events-replay-endpoint "$VLLM_KV_EVENTS_REPLAY_ENDPOINT"

PYTHONPATH=src python3 -m agent_io_tracing.replay.sweep report \
  --sweep-dir "$SWEEP_DIR" \
  --results results
open "$SWEEP_DIR/kvcache_report.html"
```

这两个 arm 都保持 `enable_prefix_caching=True`。`prefix_on_cold` 从空 cache 开始，但 bundle 后面的请求仍可命中前面请求写入的 prefix。`prefix_on_prewarmed` 在正式测量前多执行一次 bundle，但预热内容可能因容量不足被淘汰。这是一组 warm-up/retention 补充实验，不是原设计的 prefix cache OFF/ON 对照。

#### 9.2.1 原设计 A0/A1：prefix cache OFF 对 ON

新建一个 sweep，并固定 bundle、mode、输出长度和观测参数：

```zsh
SWEEP_DIR="replay/c8w4_prefix_off_on_$(date +%Y%m%d_%H%M%S)"
```

先停止 vLLM，在原启动命令中明确增加 `--no-enable-prefix-caching`，同时保留 `"${VLLM_OBSERVABILITY_ARGS[@]}"`。重启成功后确认实际配置：

```zsh
curl -fsS "$MAC_VLLM_URL/server_info?config_format=json" |
python3 -c 'import json,sys; c=json.load(sys.stdin)["vllm_config"]["cache_config"]; print("prefix=",c["enable_prefix_caching"],"mamba=",c["mamba_cache_mode"])'
```

通过标准：输出 `prefix= False mamba= none`。然后记录 A0：

```zsh
PYTHONPATH=src python3 -m agent_io_tracing.replay.sweep arm \
  --bundle "$BUNDLE" \
  --sweep-dir "$SWEEP_DIR" \
  --label prefix_off \
  --url "$MAC_VLLM_URL" \
  --mode packed \
  --output-tokens 256 \
  --kv-events-replay-endpoint "$VLLM_KV_EVENTS_REPLAY_ENDPOINT"
```

再次停止 vLLM，把 `--no-enable-prefix-caching` 改为 `--enable-prefix-caching`，其他参数不变。重启后用同一检查命令确认输出 `prefix= True mamba= align`，然后记录 A1：

```zsh
PYTHONPATH=src python3 -m agent_io_tracing.replay.sweep arm \
  --bundle "$BUNDLE" \
  --sweep-dir "$SWEEP_DIR" \
  --label prefix_on \
  --url "$MAC_VLLM_URL" \
  --mode packed \
  --output-tokens 256 \
  --strict-config \
  --allow-config-change enable_prefix_caching \
  --kv-events-replay-endpoint "$VLLM_KV_EVENTS_REPLAY_ENDPOINT"
```

vLLM 0.26 在关闭 prefix caching 时会自动把 hybrid 模型的 `mamba_cache_mode` 设为 `none`，开启时则使用当前解析出的 `align`。因此 A0/A1 测量的是 vLLM 完整的 prefix-caching feature OFF/ON，包括这项必然的 Mamba 派生变化，并不是只切换 hash lookup 的内部 microbenchmark。

#### 9.2.2 保持 Mamba 不变，只测跨请求 reuse

本实验不重启 server。确保 server 保持 `enable_prefix_caching=True` 和 `mamba_cache_mode=align`，然后在 Mac 仓库 terminal 从头定义本节使用的全部变量：

```zsh
cd /Users/minqiu/Desktop/Benchmarking_Agents/pi-ebpf-tracing-handoff

RUN_NAME="GenoMAS_A_c8_w4_20260818_153414"
BUNDLE="replay/bundles/${RUN_NAME}_A_c8_w4.json"
MAC_VLLM_URL="http://127.0.0.1:18000"
VLLM_KV_EVENTS_REPLAY_ENDPOINT="tcp://127.0.0.1:15558"
SWEEP_DIR="replay/c8w4_reuse_unique_shared_$(date +%Y%m%d_%H%M%S)"

test -s "$BUNDLE"
curl -fsS "$MAC_VLLM_URL/server_info?config_format=json" |
python3 -c 'import json,sys; c=json.load(sys.stdin)["vllm_config"]["cache_config"]; assert c["enable_prefix_caching"] is True and c["mamba_cache_mode"] == "align"; print("server-ready: prefix=True mamba=align")'
```

先测禁止跨请求复用。工具在 reset 后为 108 个请求分别生成不同 cache salt。Token IDs、Mamba mode 和其他 server 配置不变，但任何请求都不能读取其他请求写入的 prefix：

```zsh
PYTHONPATH=src python3 -m agent_io_tracing.replay.sweep arm \
  --bundle "$BUNDLE" \
  --sweep-dir "$SWEEP_DIR" \
  --label reuse_blocked_unique_salt \
  --url "$MAC_VLLM_URL" \
  --mode serial \
  --cache-salt-mode unique \
  --output-tokens 256 \
  --kv-events-replay-endpoint "$VLLM_KV_EVENTS_REPLAY_ENDPOINT"
```

同一个 server 不重启，随后测允许真实跨请求复用。工具再次 reset cache，但本次 108 个请求共享一个 cache salt：

```zsh
PYTHONPATH=src python3 -m agent_io_tracing.replay.sweep arm \
  --bundle "$BUNDLE" \
  --sweep-dir "$SWEEP_DIR" \
  --label reuse_allowed_shared_salt \
  --url "$MAC_VLLM_URL" \
  --mode serial \
  --cache-salt-mode shared \
  --output-tokens 256 \
  --strict-config \
  --kv-events-replay-endpoint "$VLLM_KV_EVENTS_REPLAY_ENDPOINT"

PYTHONPATH=src python3 -m agent_io_tracing.replay.sweep report \
  --sweep-dir "$SWEEP_DIR" \
  --results results
open "$SWEEP_DIR/kvcache_report.html"
```

预期 `reuse_blocked_unique_salt` 的逐请求 `cached_tokens` 全部为 0，`reuse_allowed_shared_salt` 则恢复 bundle 内自然形成的命中。两者都不使用 warm-up。这个差值回答跨请求 reuse 本身对 TTFT、TPOT 和 wall time 的影响，不包含 prefix OFF 导致的 Mamba mode 变化。完成 serial 机制验证后，可以另建 sweep，把两个 arm 的 mode 同时改成 `packed`，测量原设计的并发性能。

通过标准：进度到达 `completed <总数>/<总数> requests`，命令退出码为 0，新增 arm 的 `rep0/summary.json` 非空。

命令在前台运行，并实时显示 arm 名称、repetition 和已完成请求数：

```text
arm: block784_dtype-auto_prefix-on_gpu0.9_20260811T153012Z; repetition 1/1
replay: starting 11 requests (mode=packed, workers=1)
replay: completed 1/11 requests
replay: completed 2/11 requests
...
replay: completed 11/11 requests
```

工具根据服务器配置和纽约时间自动命名 arm 目录，例如 `block784_dtype-auto_prefix-on_gpu0.9_20260811T113012`。报告使用同一个时间，并以与普通 task 相同的 `月-日 时:分` 格式显示。`--label <名称>` 可以覆盖自动名称。名称只决定结果目录，不会改变服务器配置。已有的 arm 目录不会被覆盖。

重放支持三种请求节奏：

- `packed` 是默认模式。它移除原运行中 agent 和工具执行产生的空档，以原 trace 的峰值并发数连续发送请求。该模式用于机制、容量和 block size 扫描。
- `paced` 按原 trace 的请求到达时间发送，保留请求间隔。该模式用于 retention 和定时清缓存扫描。
- `serial` 严格一次只发送一个请求，用于隔离 intrinsic TTFT、server queue 和 cache hit 的关系，不代表 c8_w4 的并发行为。

同一个 sweep 不能混用不同请求模式。使用 `paced` 时在 arm 命令中添加 `--mode paced`。

`--cache-salt-mode shared` 让 bundle 内所有请求共享一个不进入模型输入的 cache namespace。`--cache-salt-mode unique` 为每个请求使用不同 salt，在 prefix caching 和 Mamba mode 保持开启的情况下禁止跨请求复用。两者使用相同 token IDs，可以替代 prefix caching on/off 对照。

`--warmup-repetitions 1` 在 reset 后先完整重放一次 bundle，再抓取正式测量的 metrics-before 并重放同一个 bundle。预热和正式测量复用同一个 cache salt。正式 summary 和 Prometheus delta 不包含预热，`warmup0_requests.jsonl` 保留预热逐请求结果。

输出默认仍固定为 32 tokens。`--output-tokens N` 可用于额外的 TTFT 或稳态 TPOT microbenchmark，`--max-concurrency N` 可以限制 replay 客户端并发，但不会改变原始 workflow 的 `max-workers`。

capacity sweep 在 server 启动命令中使用 `--num-gpu-blocks-override N`。从第二个 arm 开始建议增加强校验：

```zsh
PYTHONPATH=src python3 -m agent_io_tracing.replay.sweep arm \
  --bundle "$BUNDLE" \
  --sweep-dir "$SWEEP_DIR" \
  --url "$MAC_VLLM_URL" \
  --mode paced \
  --cache-salt-mode shared \
  --strict-config \
  --allow-config-change num_gpu_blocks_override
```

每个 repetition 都保存 `server_info.json`。它包含 vLLM version、served models 和 dev-mode `/server_info?config_format=json` 返回的完整 `VllmConfig`。强校验会比较完整 `VllmConfig` 和 Prometheus `CacheConfig`，只允许目标 knob 及其派生的 `num_gpu_blocks`、`kv_cache_size_tokens` 和 `kv_cache_max_concurrency` 变化。Mamba mode、block size 或其他字段意外变化时，replay 会在发送请求前停止。

工具默认在每次重放前调用 `/reset_prefix_cache`，让每次结果都从冷缓存开始。启动 vLLM 时需要设置 `VLLM_SERVER_DEV_MODE=1` 才能启用该接口。`--repeat N` 只适用于能够清空缓存的 server，报告会对 N 次结果取中位数。

无法启用清缓存接口时，重启 vLLM 后添加 `--keep-cache`，并且只运行一次。不要同时使用 `--keep-cache` 和 `--repeat`，否则后续运行会继承第一次留下的缓存。

#### 9.3 记录其余 arm

对每种待测配置重复以下步骤：

1. 在 Perlmutter 上停止 vLLM。
2. 使用新的参数启动 vLLM。
3. 确认 `curl -fsS "$MAC_VLLM_URL/v1/models"` 成功。
4. 再执行一次第 9.2 节的 arm 命令。

保持 `BUNDLE`、`SWEEP_DIR`、请求模式和缓存初始状态不变。vLLM 仍在同一个 compute node 上时可以继续使用现有隧道。更换 compute node 后，只需重建第 3 节的 Perlmutter 到 Mac 隧道。

#### 9.4 生成报告

```zsh
PYTHONPATH=src python3 -m agent_io_tracing.replay.sweep report \
  --sweep-dir "$SWEEP_DIR"
test -s "$SWEEP_DIR/kvcache_report.html"
test -s results/index.html
echo report-ready
open results/index.html
```

通过标准：输出 `report-ready` 并打开总索引。命令生成 `$SWEEP_DIR/kvcache_report.html`，刷新 `results/index.html`。总索引显示每个 arm 的输入、输出、缓存命中率、TTFT、TPOT、请求延迟和墙钟时间，并链接到完整报告。

`median request latency` 是从发出请求到收完响应的请求耗时中位数。`wall` 是整组请求从开始到全部完成的墙钟时间。

出现 `Not comparable` 时，bundle、调用数、总输入、固定输出长度、请求模式或缓存状态不一致。出现 `Every arm reports the same serving config` 时，vLLM 没有使用新的配置启动。包含冷缓存和热缓存 repetition 的 arm 也会单独报警。

TTFT 在 Mac 上测量，包含 Mac 到 Perlmutter 的隧道延迟。所有 arm 使用相同网络路径时，可以比较相对差异。

---



## 第二部分：研究设计



### 1. 目标与范围

本项目回答一个问题：LLM agent 控制科学工作流时，探索、重试和动态工具选择会增加多少文件系统 I/O？

研究对象是 agent 与科学任务产生的文件系统 I/O。模型加载、KV cache paging、模型 offloading 和 serving 内部存储不在范围内。固定输入重放只用于解释推理缓存，不纳入文件系统 I/O 对比。

### 2. 研究问题

1. 每个 workflow 的读写字节、操作数、元数据操作和文件数是多少？
2. 其中多少来自 agent 的探索、检查、失败和重试？
3. 次优 I/O 配置来自 agent 生成代码还是固定脚本？
4. 同一任务重复运行时，I/O 总量和归因结果是否稳定？
5. 有传统脚本基线时，agentic 运行增加了哪些 I/O？



### 3. 目标系统


| 系统                 | 编排方式                           | 实验标签 |
| ------------------ | ------------------------------ | ---- |
| GenoMAS            | 固定 Action Unit 顺序，agent 生成阶段代码 | 固定   |
| CMBAgent           | LLM 生成计划，控制器按计划分派              | 半动态  |
| SRAgent            | Supervisor 动态选择子 agent 和工具     | 动态   |
| ChemGraph          | Supervisor 动态选择子 agent 和工具     | 动态   |
| SciLink autonomous | 动态选择技能并决定 refinement 次数        | 动态   |


每份实验报告必须记录系统、版本、模型、任务、编排标签、worker 数、输入规模和重复编号。

### 4. I/O 归因

每个 I/O 单元只归入一类：


| 类别                    | 判定标准                     | 例子                 |
| --------------------- | ------------------------ | ------------------ |
| Agent-induced         | 没有本次 agent 行为就不会发生       | 重试、读错误日志、重复检查、废弃输出 |
| Task-misconfigured    | 保持任务语义不变时，存在已验证的低 I/O 配置 | 逐文件读取、重复解析、频繁重写元数据 |
| Workflow task-induced | 不属于前两类                   | 读取输入、写最终结果         |


Task-misconfigured 再标记来源：`agent-caused` 表示配置来自 agent 生成代码，`script-caused` 表示配置来自固定脚本。只有补丁前后实验或等价的局部反事实证明 I/O 降低后，才能标记为 Task-misconfigured。

### 5. 指标

所有运行报告以下指标：

- 读写字节和操作数
- 元数据操作数
- 唯一文件数
- 小文件数和小 I/O 操作数
- I/O 时间、有效带宽、读写比和 duty cycle
- 目录扫描、失败的 `open` 和 `stat`、重复读取、错误日志读取和输出检查次数
- 三类归因的字节数、操作数和占比

Duty cycle 为读写系统调用时间区间并集除以对应 wall time。全局指标使用运行 wall time。Phase 和 role 指标使用各自时间区间。

同一任务至少重复三次，并报告均值、标准差和变异系数。样本不足三次时只报告原始值，不做稳定性结论。

### 6. 实验设计



#### 6.1 采集验证

先运行最小 workload。只有第 7 节的结构、trace 和指标检查全部通过，才扩大输入或增加 workflow。

#### 6.2 归因

用 `pi_events.jsonl` 和 `tool_calls.log` 中的时间区间连接 `parsed.json` 的 I/O 事件。无法从 provenance 证明成因的事件保留为 Workflow task-induced，并在报告中记录限制。

#### 6.3 对比

- 有相同科学目标的传统实现时，对比传统实现与 agentic 实现。
- 没有传统实现时，只比较同一系统、任务和输入的重复运行。
- 次优配置使用补丁前后对比。除待测配置外，输入、worker 数、模型和运行环境保持一致。



#### 6.4 优化案例

只选择证据完整且可复现的案例。每个案例必须给出原始配置、修改内容、语义等价检查和 I/O 差值。

### 7. 输出与验收

本地结果固定存放在 `results/<run_id>/<workload>/`。远端结果固定存放在 `/mnt/lustrefs/<user>/pi-ebpf-tracing-handoff/results/`。`remote_results/` 只作临时传输目录。

每个 cell 必须包含以下输出：


| 输出                          | 用途        | 通过标准                   |
| --------------------------- | --------- | ---------------------- |
| `manifest.json`             | 运行配置      | 文件非空且 JSON 可解析         |
| `ebpf_events.log`           | 原始 trace  | 文件非空                   |
| `parsed.json`               | 解析后的 I/O  | 文件非空且 JSON 可解析         |
| `phase1_metrics.json`       | 聚合指标      | 文件非空且 JSON 可解析         |
| `bcc.err`                   | tracer 状态 | 最后一行包含 `lost_events=0` |
| `visualizations/index.html` | 单 cell 报告 | 文件非空                   |
| `trace_stats.json`          | 采集事件统计    | 文件非空且 JSON 可解析         |
| `postprocess_timings.json`  | 后处理耗时统计   | 所有步骤退出码为 0             |


GenoMAS 和 SciLink 还必须包含 `pi_events.jsonl` 和 `tool_calls.log`。`pi_events.jsonl` 的每个非空行必须是合法 JSON。

拉回结果后，在 Mac 执行：

```zsh
RUN_DIR="results/$RUN_NAME"
PYTHONPATH=src python3 -m agent_io_tracing.analysis.results_index --results results

RUN_DIR="$RUN_DIR" python3 - <<'PY'
import json
import os
from pathlib import Path

run = Path(os.environ["RUN_DIR"])
required = [
    "manifest.json",
    "ebpf_events.log",
    "parsed.json",
    "phase1_metrics.json",
    "bcc.err",
    "visualizations/index.html",
]
cells = [p for p in run.iterdir() if (p / "manifest.json").is_file()]
assert cells, f"no cells in {run}"
for cell in cells:
    for rel in required:
        path = cell / rel
        assert path.exists(), f"missing: {path}"
        assert path.stat().st_size > 0, f"empty: {path}"
    for rel in ("manifest.json", "parsed.json", "phase1_metrics.json"):
        json.loads((cell / rel).read_text())
    manifest = json.loads((cell / "manifest.json").read_text())
    if manifest.get("workload") in {"GenoMAS", "SciLink"}:
        for rel in ("pi_events.jsonl", "tool_calls.log"):
            assert (cell / rel).exists(), f"missing: {cell / rel}"
        for line in (cell / "pi_events.jsonl").read_text().splitlines():
            if line.strip():
                json.loads(line)
    assert "lost_events=0" in (cell / "bcc.err").read_text().splitlines()[-1]
print(f"verified {len(cells)} cells in {run}")
PY

test -s results/index.html && echo index-ready
```

通过标准：脚本输出 `verified <数量> cells` 和 `index-ready`，退出码为 0。检查失败的 run 不进入分析。

### 8. 交付结果

项目最终交付三类结果：

1. 不同编排固定度下的 I/O 测量结果。
2. Agent-induced、Task-misconfigured 和 Workflow task-induced 的归因结果。
3. 可复现的优化案例及补丁前后差值。
