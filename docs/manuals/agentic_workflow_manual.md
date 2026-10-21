# Agentic workflow 运行手册

GenoMAS 和 SciLink 的部署、运行、拉回和固定输入重放。SWE-bench / SWE-bench Pro 的 Claude Code + vLLM 运行步骤见[第 12 节](#12-swe-bench--swe-bench-proclaude-code--vllm)。KVCache eviction 与 swap 的实测步骤见[第 11 节](#11-实测-kvcache-eviction-与-swap)。传统 workflow 见 `traditional_workflow_manual.md`，工具范围和能力见 [README](../../README.md)。

## 0. 机器与执行路径

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
| 新节点，GenoMAS + OpenAI | 1 → 2 → 4.1 → 6 → 7 |
| 新节点，GenoMAS + FreeInference | 1 → 2 → 4.2 → 6 → 7 |
| 新节点，GenoMAS 或 SciLink + vLLM | 1 → 2 → 3 → 4.3 → 6 → 7 |
| 新节点，SciLink + OpenAI | 1 → 2 → 4.1 → 6 → 7 |
| 已配置，新开 Mac terminal | 4.4 → 6 → 7 |
| 只改代码 | 5 → 6 → 7 |
| 只换 key、后端或模型 | 4 → 6 → 7 |
| 只换 vLLM cluster | 3 → 4.3 → 6 → 7 |
| 实验正在运行或已完成 | 7 |
| SWE-bench / Pro + vLLM | 1 → 2.1 → 3 → 12 |

## 1. 更换 CloudLab allocation

在 Mac 编辑 `cloudlab_env.sh`：

```bash
export MGS_NODE="MGS 的完整主机名"
export OST_NODE="OST 的完整主机名"
export CLIENT_NODE="Workflow client 的完整主机名"
```

验证三台节点：

```zsh
source cloudlab_env.sh
for node in "$MGS_NODE" "$OST_NODE" "$CLIENT_NODE"; do
  ssh "$SSH_USER@$node" hostname
done
```

三条命令都必须返回远端 hostname。出现 `REMOTE HOST IDENTIFICATION HAS CHANGED` 时删除旧指纹后重试：

```zsh
for node in "$MGS_NODE" "$OST_NODE" "$CLIENT_NODE"; do
  ssh-keygen -R "$node"
done
```

## 2. 初始化新的 Workflow cluster

### 2.1 配置 Lustre 和 BCC

三台节点全部更换时执行。该脚本会格式化 MGS 和 OST 的非系统盘，仅用于新分配的空节点：

```zsh
bash scripts/setup_lustre_simple.sh
```

MGS + MDT 已完成，OST 或 client 配置中断时执行。该脚本会重新格式化 OST 的非系统盘：

```zsh
bash scripts/setup_lustre_ost_client_only.sh
```

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

### 2.2 GenoMAS

部署 tracing harness 和 GenoMAS，创建虚拟环境，并将 Mac 数据同步到 Lustre：

```zsh
source cloudlab_env.sh
LOCAL_GENOMAS_DATASET="/Users/minqiu/Desktop/Benchmarking_Agents/GenoMAS_Datasets" \
bash scripts/deploy_genomas_to_client.sh
```

输出最后一行必须是 `genomas-ready`。

### 2.3 SciLink

```zsh
source cloudlab_env.sh
bash scripts/deploy_scilink_to_client.sh
ssh "$SSH_USER@$WORKFLOW_NODE" \
  'test -x "$HOME/SciLink/.venv/bin/python" && echo scilink-ready'
```

输出最后一行必须是 `scilink-ready`。

使用 OpenAI 或 FreeInference 时前往第 4 节。使用 vLLM 时前往第 3 节。

## 3. 第一次使用或更换 vLLM cluster

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
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=60 \
  -R 18080:127.0.0.1:18000 \
  "$SSH_USER@$WORKFLOW_NODE"
```

在第三个 Mac terminal 检查 endpoint：

```zsh
source cloudlab_env.sh
ssh "$SSH_USER@$WORKFLOW_NODE" \
  "curl --connect-timeout 5 --max-time 10 -fsS \
  http://127.0.0.1:18080/v1/models | python3 -m json.tool"
```

通过标准：输出包含非空的 `data[0].id`。将它设为模型：

```zsh
export VLLM_URL="http://127.0.0.1:18080"
export VLLM_SERVED_MODEL="Qwen3.6-27B"
unset VLLM_API_KEY
```

HTTP 路径为 `Workflow cluster:18080 → Mac:18000 → Perlmutter compute node:8000`。KV event 路径为 `Mac:15558 → Perlmutter compute node:5558`，只供第 10 节的重放使用。更换 vLLM compute node 后重建第一个 tunnel，更换 Workflow cluster 后重建第二个。成功后执行 4.3。

## 4. 配置推理后端

GenoMAS 支持三种后端。SciLink 的图像 workload 需要视觉模型，只用 OpenAI 或 vLLM。

| 后端 | Key | Base URL | Model |
| --- | --- | --- | --- |
| OpenAI | `OPENAI_API_KEY` | 不设置 | OpenAI 模型 |
| FreeInference | `GENOMAS_OPENAI_API_KEY` | `GENOMAS_BASE_URL` | 模型裸名 |
| vLLM | `VLLM_API_KEY` 或占位值 | `VLLM_URL/v1` | served model |

### 4.1 OpenAI

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

### 4.2 FreeInference

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

### 4.3 vLLM

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

### 4.4 每次运行前检查

选择本次 workflow 的环境文件，SciLink 改为 `.env.scilink`：

```zsh
source cloudlab_env.sh
REMOTE_ENV=".env.genomas"
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

通过标准：输出 `backend-ok` 及正确的 vendor、model 和 base。

## 5. 修改代码后同步整个仓库

在 Mac 执行。不重建虚拟环境，不覆盖 key 和结果：

```zsh
source cloudlab_env.sh
rsync -az --delete \
  --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude 'results/' --exclude '.venv/' --exclude '.env*' \
  ./ "$SSH_USER@$WORKFLOW_NODE:pi-ebpf-tracing-handoff/"
ssh "$SSH_USER@$WORKFLOW_NODE" \
  'cd pi-ebpf-tracing-handoff && bash -n scripts/trace_script_bcc_genomas.sh scripts/trace_script_bcc_scilink.sh && echo sync-ok'
```

通过标准：输出 `sync-ok`。

## 6. 运行

启动命令都是 `bash scripts/launch_run.sh <workflow> <workloads>`。第二个参数是一个 cell 名或逗号分隔的多个，留空时运行该 workflow 的全部 cell。命令立即返回 `PID=` 和 `LOG=`，然后打印两行 `Watch:` 和 `Pull:`，它们就是第 7 节要用的全部命令。

使用 vLLM 时先在当前 shell `export VLLM_URL=http://127.0.0.1:18080`。不设置时，远端 `.env` 文件决定后端。

结果目录由 trace 脚本命名，格式为 `<Workflow>_<cell>_<时间戳>`，多个 cell 时为 `<Workflow>_<N>tasks_<时间戳>`，时间为纽约时间。

### 6.1 GenoMAS

可选 cell：

```text
A_c1_w1,A_c2_w1,A_c3_w1,A_c4_w1,A_c8_w1,A_c2_w2,A_c3_w2,A_c4_w2,A_c4_w4,A_c8_w4
B_t1_w2,B_t2_w2,B_t4_w2
full_c1_w2
```

GenoMAS 默认给每个 cohort 加载对应的 primary paper 作为参考上下文，每个角色最多 8,192 tokens，只加给 `GEOAgent` 和 `DomainExpertAgent`，插在 `**Task**` 之前。

启动：

```zsh
source cloudlab_env.sh
bash scripts/launch_run.sh genomas A_c8_w4
```

第一次运行先用串行 smoke cell `A_c2_w1`。关闭论文上下文得到原始 baseline：

```zsh
GENOMAS_REFERENCE_TOKENS=0 bash scripts/launch_run.sh genomas A_c2_w1
```

`4096`、`12288`、`16384` 对应 4K、12K、16K context。一个比较矩阵内保持该值不变。

### 6.2 SciLink

可选 cell：

```text
eels_plasmons_basic
eels_identification_basic
polycrystalline_grains_basic
planning_critical_materials
```

更换 vLLM server、模型或 launcher 代码后先验证一次流式响应重建。退出码 0 表示流式内容可以完整重建：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE" \
  'cd pi-ebpf-tracing-handoff && PYTHONPATH=src ~/SciLink/.venv/bin/python \
     tools/dump_stream_chunks.py --base-url http://127.0.0.1:18080 \
     --model Qwen3.6-27B --max-tokens 3000'
```

思考型模型需要足够的 `--max-tokens`，否则可能只返回推理内容，agent 收到空回答后重试。

启动：

```zsh
source cloudlab_env.sh
bash scripts/launch_run.sh scilink eels_plasmons_basic
```

Launcher 拦截重试循环：连续 20 次相同 prompt 时进程以退出码 3 终止，`SCILINK_MAX_REPEAT_CALLS` 改阈值，`SCILINK_MAX_CALLS` 限制总调用数。`planning_critical_materials` 最容易触发，因为 SciLink 上游把它的 `max_output_tokens` 固定为 1024，思考型模型会在生成正文前耗尽预算。

SciLink 不采集 `futex`、`epoll_wait` 等等待事件和匿名 `mmap`，文件映射仍采集。只在调试调度问题时为 tracer 加 `--include-waits`。

## 7. 看日志与拉回

第 6 节打印的两行就是本节。看日志：

```zsh
source cloudlab_env.sh; ssh "$SSH_USER@$WORKFLOW_NODE" 'tail -f ~/logs/<日志名>.log'
```

日志最后出现 `Results in:` 表示跑完。`Ctrl-C` 只停止查看。拉回：

```zsh
source cloudlab_env.sh; bash scripts/pull_run.sh logs/<日志名>.log
```

脚本把结果拉到 `results/<run 名>`，在本地重算 KV 报告，刷新 `results/index.html`，检查脚本要求的必需文件和 `lost_events=0`，然后打开报告。输出 `verified: <数量> cells in results/<run 名>` 表示通过。`results/<run 名>` 就是后面各节的 `RUN_NAME`。

远端日志的最后一行也是一条 `Pull to Mac:` 命令，带绝对路径，同样可以直接粘贴。

## 8. 出问题时

日志长时间不动时，看三类进程是否还活着。列表为空且日志没有 `Results in:` 表示中途失败，看最后 100 行找原因：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE" "pgrep -af 'trace_script_bcc_|bcc_tracer|phase1_metrics|viz.trace' || true"
ssh "$SSH_USER@$WORKFLOW_NODE" 'tail -100 ~/logs/<日志名>.log'
```

忘了日志名时列出最近的：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE" 'ls -t ~/logs | head'
```

拉回失败时，每个 `ERROR:` 行指出缺哪个文件。文件缺失前的一行是原因，例如 LLM 调用被拒导致没有工具调用，图就画不出来。

## 9. 用 vLLM 自己的分词器重算 KV 指标

只用于 vLLM，且要在 Perlmutter 隧道关闭前执行。本地 tiktoken 无法还原服务器的 chat template、tools schema 和图像输入，vLLM 的 `/tokenize` 用服务器实际的分词器。Mac 上的隧道端口是 18000。

```zsh
RUN_NAME="GenoMAS_A_c8_w4_20260818_153414"
PYTHONPATH=src python3 -m agent_io_tracing.analysis.kvcache.server_tokens \
  --results results --url http://127.0.0.1:18000
PYTHONPATH=src python3 -m agent_io_tracing.analysis.kvcache.report \
  --results results --runs "$RUN_NAME" --dump-prefixes
PYTHONPATH=src python3 -m agent_io_tracing.analysis.results_index --results results
test -s results/index.html && echo kvcache-report-ready
```

通过标准：最后一行 `kvcache-report-ready`。第一条为每个 cell 写入 `server_tokens.json`，第二条重算当前 run，第三条刷新总索引。

## 10. 固定输入重放

固定输入重放的范围见 [README 下篇](../../README.md#part-ii-vllm-kv-cache-behavior)。步骤是：做一次 bundle，给 Perlmutter 的 vLLM 加观测参数，逐个记录 arm，出报告。所有命令在 Mac 仓库目录执行，第 3 节的 Perlmutter 隧道必须在运行。

### 10.1 创建 bundle

选择一个已拉回的 cell，它必须包含 `messages.jsonl` 和 `pi_events.jsonl`：

```zsh
RUN_NAME="GenoMAS_A_c8_w4_20260818_153414"
CELL="results/$RUN_NAME/A_c8_w4"
BUNDLE="replay/bundles/${RUN_NAME}_$(basename "$CELL").json"
MAC_VLLM_URL="http://127.0.0.1:18000"
VLLM_SERVED_MODEL="$(curl -fsS "$MAC_VLLM_URL/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')"
mkdir -p replay/bundles
PYTHONPATH=src python3 -m agent_io_tracing.replay.bundle "$CELL" \
  --output "$BUNDLE" --url "$MAC_VLLM_URL" --model "$VLLM_SERVED_MODEL"
python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))' "$BUNDLE" && echo bundle-ready
```

通过标准：最后一行 `bundle-ready`。`--limit N` 只打包前 N 个请求，只用于检查链路。一个 sweep 的所有 arm 使用同一个 `BUNDLE`。

### 10.2 给 vLLM 加观测参数

第一次在 Mac 运行重放时安装依赖：

```zsh
python3 -m pip install -e '.[observability]'
```

把观测配置复制到 Perlmutter，代码更新后重做一次：

```zsh
scp config/config_vllm_observability.sh mqsun@perlmutter.nersc.gov:~/config_vllm_observability.sh
```

在 Perlmutter 启动 vLLM 的 `srun` shell 中加载，然后在你原来的 `vllm serve` 命令末尾加上 `"${VLLM_OBSERVABILITY_ARGS[@]}"`，其他参数不动：

```bash
source "$HOME/config_vllm_observability.sh"
printf 'observability arguments loaded: %s\n' "${VLLM_OBSERVABILITY_ARGS[*]}"
```

通过标准：输出包含 `--enable-prompt-tokens-details` 和 `--kv-events-config`。

### 10.3 记录 arm

每个 arm 一条命令。`BUNDLE` 沿用 10.1，`SWEEP_DIR` 每组实验新建一次：

```zsh
SWEEP_DIR="replay/c8w4_<实验名>_$(date +%Y%m%d_%H%M%S)"
BUNDLE="$BUNDLE" bash scripts/replay_arm.sh "$SWEEP_DIR" <label> <参数>
```

通过标准：进度到达 `completed <总数>/<总数> requests`，最后一行 `arm-ready:`。脚本固定隧道地址、KV event 端口和 256 个输出 token，`OUTPUT_TOKENS` 可改。

四组实验的 label 和参数：

| 实验 | label | 参数 | server |
| --- | --- | --- | --- |
| 预热对照 | `prefix_on_cold` | `--mode serial --cache-salt-mode shared` | 不重启 |
| | `prefix_on_prewarmed` | `--mode serial --cache-salt-mode shared --warmup-repetitions 1 --strict-config` | 不重启 |
| 跨请求复用 | `reuse_blocked_unique_salt` | `--mode serial --cache-salt-mode unique` | 不重启 |
| | `reuse_allowed_shared_salt` | `--mode serial --cache-salt-mode shared --strict-config` | 不重启 |
| prefix OFF/ON | `prefix_off` | `--mode packed` | 加 `--no-enable-prefix-caching` 重启 |
| | `prefix_on` | `--mode packed --strict-config --allow-config-change enable_prefix_caching` | 改回 `--enable-prefix-caching` 重启 |
| 容量 | 任意 | `--mode paced --cache-salt-mode shared --strict-config --allow-config-change num_gpu_blocks_override` | 每个 arm 换 `--num-gpu-blocks-override N` 重启 |

需要重启 server 的实验，每次重启后先确认配置再记录 arm：

```zsh
curl -fsS http://127.0.0.1:18000/server_info?config_format=json |
python3 -c 'import json,sys; c=json.load(sys.stdin)["vllm_config"]["cache_config"]; print("prefix=",c["enable_prefix_caching"],"mamba=",c["mamba_cache_mode"])'
```

关闭 prefix caching 时应输出 `prefix= False mamba= none`，开启时 `prefix= True mamba= align`。

同一 sweep 内不改 `BUNDLE`、`SWEEP_DIR`、`--mode` 和缓存初始状态。

### 10.4 生成报告

```zsh
PYTHONPATH=src python3 -m agent_io_tracing.replay.sweep report --sweep-dir "$SWEEP_DIR" --results results
test -s "$SWEEP_DIR/kvcache_report.html" && echo report-ready
open "$SWEEP_DIR/kvcache_report.html" results/index.html
```

通过标准：输出 `report-ready`。总索引显示每个 arm 的输入、输出、缓存命中率、TTFT、TPOT、请求延迟和墙钟时间。

### 10.5 逻辑 block 移除与返回

配置页列出每次运行，并链接有事件数据的 `kv_block_report.html`。无事件数据的运行在配置页说明缺失状态，不生成空 block 页。总表、配置页和单轮明细使用同一份 `kv_report_data.json`。

`kv_lifecycle.jsonl` 记录同一逻辑内容从 GPU `BlockRemoved` 到下一次 GPU `BlockStored` 的区间。它使用 publisher 时间，按进程、缓存组和重启边界隔离身份。未返回表示观察窗口内没有再次出现，不能解释为永不复用。

仅凭逻辑事件得到的 `cpu_backed_return_candidates` 是 CPU 副本候选。确认实际搬回需要传输计划、全部 worker 完成以及后续计算证据。当前已实现这些插桩；物理 eviction 与 swap 的完整操作见第 11 节。历史日志仍保持原有证据级别。

## 11. 实测 KVCache eviction 与 swap

本节说明如何重复执行已经验证的流程。目标是测清实际驱逐、CPU 写出与搬回、完成的输入计算量和请求等待，并生成可追溯报告。模型数值状态验证是独立研究，不属于本节验收条件。

### 11.1 环境与实验选择

本节的启动命令适用于已部署的 Perlmutter 环境：Qwen3.6-27B、四张 A100、TP=4、vLLM 0.26.0 镜像 `da9136512c30`。远端部署目录为 `$SCRATCH/vllm/kv-trace-deploy-20260911`，包含七个源码覆盖文件、hook 模块和 `launch.sh`。新环境使用 `tools/prepare_vllm_trace_patch.py` 核对 `config/vllm_trace_v026_source.json` 中的源码指纹并安装插桩。物理槽位和传输完成记录需要服务端 hook。

先选择实验，再固定 bundle、输出长度和运行顺序：

| 实验 | GPU blocks | CPU offload | 输入与输出 | 用途 |
| --- | ---: | ---: | --- | --- |
| GPU 复用与 CPU 恢复短控制 | 274 | 32 GiB | 原 C2 首条请求，输出 8 tokens | 检查首次计算、直接重复、清空 GPU 后恢复 |
| 压力控制 | 274 | 32 GiB | A → 96 个不同前缀 → A，输出 1 token | 观察容量驱逐及最后 A 是否重新计算 |
| 原 bundle 测量 | 274 | 32 GiB | C2 packed、输出 32；C8 serial、输出 256 | 测双向传输、恢复和计算 |
| GPU 容量对比 | 192、274、548 | 关闭 | 同上 | 测容量改变后的驱逐和计算量 |

每个正式容量与重复组合重新启动服务，先 C2、后 C8。三个容量各重复三轮，保存执行顺序。短控制使用单独服务会话；不要把压力控制留下的 CPU 内容带入正式对照。GPU reset 不保证清空 CPU offload 缓存。

### 11.2 设置参数并连接

所有本地命令在仓库根目录执行。安装依赖：

```zsh
python3 -m pip install -e '.[observability]'
```

先在 Perlmutter 取得一个四 GPU 的有效 allocation。下面将 `实际作业ID`、`实际计算节点` 和用户名替换为本次值，不复用历史 job ID。可在登录节点用 `squeue -u "$USER"` 查看作业节点和剩余时间。C8 请求本身约需八分钟，另留模型启动、快照和收尾时间。

```zsh
export KV_REMOTE='mqsun@perlmutter.nersc.gov'
export KV_JOB='实际作业ID'
export KV_NODE='实际计算节点'
export KV_REMOTE_ROOT='/pscratch/sd/m/mqsun/vllm'
export KV_SOCKET="$HOME/.ssh/kv-evict-manual-control"
export KV_SESSION="evict-$(date -u +%Y%m%dT%H%M%SZ)"
export KV_GPU=274
export KV_CPU=32
export KV_SWEEP_C2="replay/${KV_SESSION}_c2"
export KV_SWEEP_C8="replay/${KV_SESSION}_c8"
export KV_SNAPSHOT="output/${KV_SESSION}-snapshot"
export KV_ARM="gpu${KV_GPU}_cpu${KV_CPU}"

ssh -M -S "$KV_SOCKET" -o ControlPersist=2h -o ExitOnForwardFailure=yes -fnNT \
  -L "127.0.0.1:18001:${KV_NODE}:8000" \
  -L "127.0.0.1:15559:${KV_NODE}:5558" "$KV_REMOTE"
```

18001 是本节 API 端口，15559 是 KV 事件 replay 端口，与第 10 节端口不同。SSH 认证在此完成；后续命令复用该连接。端口被占用时，先检查已有隧道属于哪个节点，不要把新运行发给旧服务。

### 11.3 启动插桩服务

确认远端部署及观测配置存在：

```zsh
ssh -S "$KV_SOCKET" "$KV_REMOTE" \
  "test -f '$KV_REMOTE_ROOT/kv-trace-deploy-20260911/launch.sh' && test -f ~/config_vllm_observability.sh"
```

从本地启动服务，将启动日志保存到本地。`launch.sh` 在指定 allocation 中运行容器，并给引擎和全部 worker 传入 trace 配置：

```zsh
mkdir -p output
ssh -S "$KV_SOCKET" -o BatchMode=yes "$KV_REMOTE" \
  "env KV_TRACE_JOB_ID='$KV_JOB' KV_TRACE_SESSION='$KV_SESSION' KV_GPU_BLOCKS='$KV_GPU' KV_OFFLOADING_SIZE_GIB='$KV_CPU' KV_TRACE_ENABLED=1 bash '$KV_REMOTE_ROOT/kv-trace-deploy-20260911/launch.sh'" \
  > "output/${KV_SESSION}-server.log" 2>&1 &
export KV_LAUNCH_PID=$!
```

查看启动日志，等 `/health` 成功后再运行实验：

```zsh
tail -n 30 "output/${KV_SESSION}-server.log"
curl -fsS http://127.0.0.1:18001/health
curl -fsS 'http://127.0.0.1:18001/server_info?config_format=json' \
  > "output/${KV_SESSION}-server-info.json"
```

实验脚本会检查实际 GPU blocks、CPU offload、KV dtype、有效 block size 和采样比例。已验证环境的有效 block size 为 784，hash 匹配单位为 16；不能只看启动参数中的 `--block-size 16` 推断实际缓存布局。脚本要求 KV dtype 为 `auto`、有效 block size 为 784、histogram 采样比例为 1.0。

### 11.4 先做短控制或压力检查

第一次部署先运行短控制。三个 case 在同一服务、同一 salt 下执行，脚本自动控制是否重置 GPU 缓存：

```zsh
export KV_CONTROL="replay/${KV_SESSION}_controls"
python3 tools/run_kv_causal_controls.py \
  --url http://127.0.0.1:18001 \
  --replay-endpoint tcp://127.0.0.1:15559 \
  --bundle replay/kv_c2_capacity_20260910/bundles/C2-12.json \
  --output "$KV_CONTROL" --salt "$KV_SESSION"
```

已验证样例的首条输入为 2,991 tokens：首次计算 2,991；直接重复计算 15；清空 GPU 后重复，从 CPU 恢复 2,352、计算 639。恢复确认应有四个 worker 完成记录。缓存命中数只用于初步检查，完整证据需在第 11.6 节导入物理日志后查看。

需要压力控制时，在新的服务会话中运行同一命令，改用新的 `--output`、新的 `--salt`，加 `--pressure`。它会执行 A、96 个修改首 token 的前缀、再执行 A，共 98 个请求。已验证运行出现 1,004 次 GPU 物理驱逐，最后 A 重新计算全部 2,991 tokens；这些数值是参考结果，不是要求所有配置都相同的断言。

短控制结束后，先按第 11.6 节保存、导入日志，再停止服务。正式实验重新设置 `KV_SESSION` 和输出路径并启动新服务。

### 11.5 运行原 C2、C8 bundle

下面执行一个 GPU/CPU 配置的一轮，先 C2 后 C8。每条命令的 `--repetitions 1` 表示本例只生成 `rep0`：

```zsh
python3 tools/run_kv_experiment_cell.py \
  --url http://127.0.0.1:18001 \
  --replay-endpoint tcp://127.0.0.1:15559 \
  --bundle replay/kv_c2_capacity_20260910/bundles/C2-12.json \
  --sweep-dir "$KV_SWEEP_C2" --label "$KV_ARM" --workload C2-12 \
  --gpu-blocks "$KV_GPU" --cpu-offload-gib "$KV_CPU" \
  --mode packed --output-tokens 32 --repetitions 1

python3 tools/run_kv_experiment_cell.py \
  --url http://127.0.0.1:18001 \
  --replay-endpoint tcp://127.0.0.1:15559 \
  --bundle replay/kv_c8_capacity_20260910/bundles/C8.json \
  --sweep-dir "$KV_SWEEP_C8" --label "$KV_ARM" --workload C8 \
  --gpu-blocks "$KV_GPU" --cpu-offload-gib "$KV_CPU" \
  --mode serial --output-tokens 256 --repetitions 1
```

通过标准：分别完成 12/12 和 108/108 个请求，日志输出 `collection_errors: []`。脚本保存原 bundle 指纹、temperature=0 和 seed=0 的回放快照、实际服务配置及请求输出。刚生成的报告还不包含节点上的物理日志，必须继续执行快照和导入。

做三轮比较时，固定两个 sweep 路径、arm 和 `--repetitions 3`，每轮分别指定 `--only-rep 0`、`1`、`2`，每轮重启服务并换 `KV_SESSION` 与 `KV_SNAPSHOT`。容量对比只修改 `KV_GPU` 为 192、274、548，固定 `KV_CPU=0`，同步更新 `KV_ARM`，然后按相同流程运行。不要把模型、输出长度或 CPU 开关同时当成一个容量变量。

`tools/run_kv_capacity_matrix.py` 和 `tools/run_kv_same_node_offload.py` 是 2026-09-11 这次实验的自动续跑工具，其源码固定了该日期的 replay 目录。`--output` 只改变快照和完成凭据目录，不能隔离新的实验数据。新实验使用上述单元命令和新 sweep 路径；只有恢复原实验时才使用那两个脚本。

### 11.6 在释放 allocation 前保存物理日志

日志位于计算节点的 `/dev/shm/kv-physical-$KV_SESSION`，不是登录节点。两个 bundle 完成后，服务保持运行，先创建不可变快照：

```zsh
python3 tools/pull_kv_physical_snapshot.py \
  --ssh-host "$KV_REMOTE" --control-path "$KV_SOCKET" --job-id "$KV_JOB" \
  --trace-dir "/dev/shm/kv-physical-$KV_SESSION" \
  --remote-output "$KV_REMOTE_ROOT/trace-snapshots/$KV_SESSION" \
  --output "$KV_SNAPSHOT"
```

工具只复制完整 JSONL 行，保存远端快照及 `snapshot_manifest.json`，核对本地文件 SHA-256。开启 offload 且发生传输时应收齐引擎和四个 worker 文件；关闭 offload 时没有 worker 传输文件不代表丢日志。不要对仍增长的日志直接打包，也不要在快照完成前结束 allocation。

将同一服务快照分别导入本轮 C2 和 C8。导入按请求标记关联，不会把两个 bundle 的全部事件各重复计入一次：

```zsh
python3 -m agent_io_tracing.serving.kv_physical \
  "$KV_SWEEP_C2/$KV_ARM/rep0" "$KV_SNAPSHOT"/*.jsonl
python3 -m agent_io_tracing.serving.kv_physical \
  "$KV_SWEEP_C8/$KV_ARM/rep0" "$KV_SNAPSHOT"/*.jsonl
```

三轮实验将这里的 `rep0` 换为本轮编号。短控制则依次导入 `$KV_CONTROL/first/rep0`、`$KV_CONTROL/gpu_repeat/rep0`、`$KV_CONTROL/cpu_repeat/rep0`；压力控制只导入其 `pressure/rep0`。一个会话的快照只能用于该会话实际执行的请求，不能拿新服务日志补旧运行。

### 11.7 生成报告并判断是否测到了

```zsh
python3 tools/rebuild_kv_reports.py
python3 tools/check_kv_report_links.py
open "$KV_SWEEP_C8/kvcache_report.html" results/index.html
```

Mac 使用 `open`；其他平台用浏览器打开对应 HTML。配置页列出每轮入口，单轮的 `kv_block_report.html` 展示物理驱逐、传输、候选分类和等待，并链接以下文件：

| 文件 | 用户查看什么 |
| --- | --- |
| `physical_evictions.jsonl` | GPU 槽位、分配代次、驱逐内容和空闲时间 |
| `cpu_evictions.jsonl` | CPU 淘汰内容、来源和保存时长；有相关事件才生成 |
| `causal_transfers.jsonl` | 每项双向传输及 worker 完成记录 |
| `confirmed_reloads.jsonl` | 确认搬回的内容及 GPU 目标证据 |
| `candidate_reconciliation.jsonl` | CPU 副本候选最终是搬回、重算还是未解决 |
| `causal_requests.jsonl` | 每个请求的缓存分配和实际完成计算量核算 |
| `request_load_waits.jsonl` | 调度器观测的缓存加载等待 |
| `kv_report_data.json` | 页面共享的统计、数据来源及完整性说明 |

验收时检查请求数量是否齐全、物理日志序号是否有缺口、传输任务是否收齐预期 worker、请求 token 核算是否匹配输入。`PhysicalEviction` 才是本节的 GPU 容量驱逐；histogram 观测数、逻辑 hash 移除和 CPU 写出量分别解释。worker CUDA 时间之和与请求等待时间分开阅读。

已有样例中，C8 的 148 个 CPU 副本候选分为 127 次确认搬回和 21 次确认重算。新运行若不同，应检查配置、缓存初始状态及逐条证据，不能为了匹配样例修改分类。没有观察到 swap 是允许的结果，不应生成空的 swap 明细表。采集缺口保持缺失状态；请求核算通过也不代表已覆盖进程的整个生命周期。

### 11.8 停止服务与中断恢复

确认快照和导入成功后，停止本会话容器：

```zsh
ssh -S "$KV_SOCKET" "$KV_REMOTE" \
  "srun --jobid='$KV_JOB' --overlap -N1 -n1 podman-hpc stop --time 30 'kv-trace-$KV_SESSION'"
wait "$KV_LAUNCH_PID"
```

只有该 allocation 的全部实验结束后才释放它，并关闭本节创建的隧道：

```zsh
ssh -S "$KV_SOCKET" "$KV_REMOTE" "scancel '$KV_JOB'"
ssh -S "$KV_SOCKET" -O exit "$KV_REMOTE"
```

遇到中断按以下顺序处理：

1. 查看启动日志、`squeue` 和 `sacct`，区分认证失败、启动失败和 Slurm TIMEOUT。
2. 若原服务仍在，优先保存它的物理快照；若 allocation 已结束，先找远端 `trace-snapshots`。节点内存日志不保证仍存在。
3. 保留部分运行目录和采集错误。仅有 `summary.json` 不等于物理日志已经完整保存。
4. 无法恢复同一会话日志时，给该运行保留缺失状态，使用新会话及新目录重跑。不要拼接旧请求与新进程日志。
5. SSH 隧道恢复后重新核对实际节点和 `/server_info`。Mac 长时间实验可用 `caffeinate -i python3 ...` 防止本地休眠。

## 12. SWE-bench / SWE-bench Pro：Claude Code + vLLM

三个固定任务为 Django `14999`、Sphinx `8548` 和 SWE-bench Pro 的 NodeBB `70b4a0e2`。Claude Code 在 CloudLab 上实际读写仓库、运行命令并生成补丁。每次回答自然结束；不固定输出长度。任务与 base commit 见 `workloads/swebench/manifest.json`。

### 12.1 安装

先完成第 2.1 节的 Lustre、BCC 和第 3 节的 vLLM 隧道。在 Mac 执行：

```zsh
source cloudlab_env.sh
bash scripts/deploy_swebench_to_client.sh
```

部署脚本安装 Node.js 22.16.0、Claude Code 2.1.197、CCR 1.0.72、Python 环境和三个任务的依赖。代码仓库放在 `/mnt/lustrefs/$USER/pi-ebpf-tracing-handoff/swebench/work`。NodeBB 使用独立 Redis 端口 `16379`，数据库 `0` 和测试数据库 `1`。

该步骤用于首次准备干净任务。已经运行并修改过的仓库会被保留，脚本会拒绝覆盖；只更新 harness 代码时用第 5 节同步，重跑任务用 12.4 的新目录。

在 CloudLab 的 `~/pi-ebpf-tracing-handoff/.env.swebench` 中设置实际 `VLLM_URL` 和 `VLLM_SERVED_MODEL`。本次验证使用：

```bash
export VLLM_URL=http://127.0.0.1:18080
export VLLM_SERVED_MODEL=Qwen3.6-27B
```

`18080` 是 CloudLab 的 reverse tunnel 端口。更换模型后，在 CloudLab 执行 `bash scripts/configure_swebench_ccr.sh`。该命令备份原 CCR 配置并重启 CCR。

### 12.2 执行并测量

在 CloudLab 执行：

```bash
cd ~/pi-ebpf-tracing-handoff
source .env.swebench
curl -fsS "$VLLM_URL/v1/models"
for task in django sphinx nodebb; do
  bash scripts/run_swebench_measured.sh "$task" || break
done
```

请求路径是 `Claude Code → CCR:3456 → 测量代理:18081 → vLLM:18080`。测量脚本自动启动和停止代理，暂停 agent 等待 BCC 就绪，完成后停止 tracer 并解析文件访问事件。它使用 `bypassPermissions` 在任务环境中执行命令和修改代码。

测量代理串行发送模型请求，每次运行使用独立 cache salt，同一任务内部共享前缀缓存。固定会话名称避免自动生成标题；禁用子 agent 委派和后台调度。代理拒绝图像、音频和视频输入。CCR transformer 处理工具调用之间的多余文本 delta。vLLM 需启用对应模型的工具调用解析器。

结果位于 `results/swebench-measured/<task>-<UTC时间>/`：

| 文件 | 内容 |
| --- | --- |
| `timing_summary.json` | 请求数、TTFT、TPOT、服务端 prefill / inference 时间比例 |
| `per_request.json` | 每次请求的时间、usage 和可用的服务端时间字段 |
| `http/` | 实际请求、原始流式响应和请求计时 |
| `metrics-before.txt`、`metrics-after.txt` | vLLM 指标快照 |
| `server-info.json` | 测量时的服务端配置 |
| `agent/*/` | agent 事件、带时间戳事件、补丁、stderr 和运行状态 |
| `ebpf_events.log`、`parsed.json`、`tool_calls.log` | 文件访问追踪、解析结果和工具执行区间 |
| `exit_code.txt` | 执行状态；补丁是否修复问题由 12.3 验证 |

`client_ttft_fraction` 为 `Σ首 token 等待时间 / Σ请求耗时`。`server_prefill_fraction` 为服务端 `Σprefill 时间 / Σinference 时间`。运行时应独占该 vLLM 服务，并确认 `server_requests_match=true`；请求数量一致是检查项，不替代独占运行条件。服务端 prefill 指标描述阶段耗时。正式比较策略时保持模型、追踪开关和请求负载一致。

查看 `bcc.err` 和 `trace_stats.json` 中的丢事件统计。丢事件影响文件访问分析的完整性，HTTP 请求计时另行保存。只测 LLM 时间可设置 `SWE_TRACE=0`。

### 12.3 验证补丁

在 CloudLab 将 `RUN` 设置为实际结果目录：

```bash
RUN=results/swebench-measured/django-实际时间
PATCH=$(find "$RUN/agent" -name patch.diff -print -quit)
bash scripts/evaluate_swebench_local.sh django "$PATCH" baseline
bash scripts/evaluate_swebench_local.sh django "$PATCH" patched
```

`baseline` 在原始代码上运行数据集的测试补丁，预期目标回归测试失败；`patched` 在独立 checkout 上应用 agent 补丁后再测试。将任务名和 `RUN` 换成 `sphinx` 或 `nodebb` 即可。测试日志放在 `$SWE_STORAGE/evaluation/`，不修改 agent 工作目录。

Django 运行 migration operations 测试；Sphinx 运行 autodoc autoclass 测试；NodeBB 运行数据集指定的两个数据库测试文件，使用 Redis 后端。这是固定数据集测试的本地验证，不代表完整官方容器评测或 NodeBB 所有数据库后端验证。

### 12.4 重跑与拉回

使用新目录保留已有补丁。在 CloudLab 执行，以 Django 为例：

```bash
source .env.swebench
export SWE_WORK_ROOT="$SWE_STORAGE/repeat-$(date -u +%Y%m%dT%H%M%S)"
python scripts/swebench_local.py prepare --task django --work-root "$SWE_WORK_ROOT"
bash scripts/run_swebench_measured.sh django
```

Django 和 Sphinx 复用各自的 Python 依赖，通过 `PYTHONPATH` 加载新 checkout。NodeBB 的新 checkout 还需复用依赖。在运行前执行：

```bash
cp "$SWE_STORAGE/work/nodebb/package.json" "$SWE_STORAGE/work/nodebb/config.json" "$SWE_WORK_ROOT/nodebb/"
ln -s "$SWE_STORAGE/work/nodebb/node_modules" "$SWE_WORK_ROOT/nodebb/node_modules"
printf '/node_modules\n' >> "$SWE_WORK_ROOT/nodebb/.git/info/exclude"
```


在 Mac 拉回结果和验证日志：

```zsh
source cloudlab_env.sh
rsync -az "$SSH_USER@$WORKFLOW_NODE:pi-ebpf-tracing-handoff/results/swebench-measured/" results/swebench-measured/
rsync -az --include='*.test.log' --include='*.exit_code' --exclude='*' \
  "$SSH_USER@$WORKFLOW_NODE:/mnt/lustrefs/$SSH_USER/pi-ebpf-tracing-handoff/swebench/evaluation/" \
  results/swebench-evaluation/
```

### 12.5 重放 SWE 请求并测量 eviction / CPU offload

2026-09-16 的实验在 Perlmutter 停机期间临时使用 `ai-l40s-01`、两张 L40S、vLLM 0.29.0 和 `Qwen/Qwen3.6-27B`。远端部署目录为 `~/swe-kv-experiment`，启动脚本为 `scripts/launch_swe_kv_riken.sh`。七个 vLLM 源文件经 `config/vllm_trace_v029_source.json` 的哈希检查后插桩。物理日志同时记录 scheduler 和两个 GPU worker。

本节重放第 12.2 节保存的纯文本请求，测量跨请求 KV 保留、淘汰和恢复。输入按新服务的 tokenizer 重新核对，输出长度逐请求采用原轨迹记录。重放不执行工具，生成内容也不改变下一条已保存的输入，因此耗时表示 LLM 请求重放时间。每个任务串行运行，每组开始清空 GPU cache 并使用独立 cache salt。

实验固定 TP=2、最大上下文 65,536、`max_num_seqs=1`、同步调度和每个 chunk 最多 2,048 tokens。三组依次为 `gpu200_cpu0`、`gpu96_cpu0`、`gpu96_cpu32`。CPU 容量单位为 GiB；关闭 offload 时省略 `--kv-offloading-size`，不要传 `0`。这三组使用 vLLM 原生淘汰与 CPU offload 策略。

Mac 的 API 隧道为 `127.0.0.1:18000 → GPU节点:8000`，KV event 隧道为 `127.0.0.1:15558 → GPU节点:5558`。通过 `/v1/models` 获取实际模型别名。新 allocation 需要更新节点名与 job ID，并先启动第一组服务。在临时集群的登录节点执行，替换实际 job ID 和节点：

```bash
SWE_JOB=384217
SWE_NODE=ai-l40s-01
SWE_SESSION=swe-g200-c0-新的唯一名称
nohup srun --jobid="$SWE_JOB" --overlap -N1 -n1 -c16 -w "$SWE_NODE" \
  bash "$HOME/swe-kv-experiment/launch_swe_kv_riken.sh" "$SWE_SESSION" 200 0 \
  > "$HOME/swe-kv-experiment/output/$SWE_SESSION.log" 2>&1 < /dev/null &
echo $!
```

记下 srun PID。在 Mac 仓库中执行，填写相同 session、PID 和 allocation。使用新的输出目录保留已有实验：

```bash
python tools/run_swe_kv_matrix.py \
  --output replay/swe-riken-新的唯一名称 \
  --host u0001609@login.cloud.r-ccs.riken.jp \
  --job 384217 --node ai-l40s-01 \
  --remote-root /home/users/u0001609/swe-kv-experiment \
  --initial-launch-pid 实际PID --initial-session 实际SESSION
python tools/validate_swe_kv_matrix.py replay/swe-riken-新的唯一名称
```

驱动从 `results/swebench-measured/selected_results.json` 选择三个任务，使用现有 bundle、replay、KV event collector、物理日志导入和报告代码。`--output-tokens 0` 表示原轨迹逐请求长度。每组完成后，驱动拉取带 SHA-256 校验的物理日志快照，再导入报告；切换配置时停止自己启动的服务。最后一组服务保持运行。

本次结果目录为 `replay/swe-riken-20260916/`。`validation.json` 检查九个实验的请求数、输入输出长度、逐请求 token 计算账目、日志序号和传输完成状态。报告写回三个原任务目录的 `kvcache_report.html`，并更新 `results/index.html` 与 `results/all-experiments.html`。本次三组比较已登记在 `config/kv_report_comparisons.json`；更换输出目录后，应同步登记对应比较路径。

判断 CPU 恢复要同时检查 `confirmed_load_jobs`、实际 load 字节数和 worker 完成记录。CPU store 表示写出，不能计为恢复。`ΣTTFT / Σ请求耗时` 是首 token 等待比例；服务端 prefill 指标与实际计算 token 另列。正式轨迹和显式清空 GPU 后的短恢复检查分别保存。

本次九组通过 `validation.json` 的全部检查。三个任务在 96 blocks 下分别发生 13、60、27 次物理 eviction，逐请求命中量与 200-block 基线相同；开启 CPU offload 后均无 load。独立短检查保存在 `controls/validation.json`：清空 GPU 后，从 CPU 恢复 9,408 tokens，计算 380 tokens，确认一个 load 任务和两个 worker 的完成记录。该短检查验证恢复路径，不计入任务性能比较。
