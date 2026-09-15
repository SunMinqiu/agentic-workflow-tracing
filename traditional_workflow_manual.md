# Traditional workflow 运行手册

1000 Genomes 和 Montage 的部署、运行和拉回。两者由固定脚本驱动，不含 LLM，不需要推理后端。Agentic workflow 见 `agentic_workflow_manual.md`，研究设计见 `Design.md`。

## 0. 机器与执行路径

| 机器 | 变量 | 职责 |
| --- | --- | --- |
| Mac | 无 | 配置节点、部署代码、启动实验、拉回结果 |
| Lustre MGS + MDT | `MGS_NODE` | 管理 Lustre |
| Lustre OST | `OST_NODE` | 保存 Lustre 数据 |
| Workflow client | `CLIENT_NODE`、`WORKFLOW_NODE` | 运行 workflow、BCC 和后处理 |

`WORKFLOW_NODE` 始终等于 `CLIENT_NODE`。

| 当前状态 | 执行顺序 |
| --- | --- |
| 新节点，1000 Genomes | 1 → 2.1 → 2.2 → 3 → 4.1 → 5 |
| 新节点，Montage | 1 → 2.1 → 2.2 → 2.3 → 3 → 4.2 → 5 |
| 只改代码 | 3 → 4 → 5 |
| 已配置，新开 Mac terminal | 4 → 5 |
| 实验正在运行或已完成 | 5 |

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

### 2.2 1000 Genomes

只在新的 Workflow cluster 上执行一次。登录 Workflow cluster：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE"
```

克隆仓库、准备输入并创建虚拟环境：

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

通过标准：输出最后一行为 `1000genome-ready`。

运行前确认所选 chromosome 的主 VCF 和 annotation VCF 都已解压为 `.vcf`：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE" "find $MOUNT_PATH/$SSH_USER/1000genome-workflow/data/20130502 -type f -name '*.vcf' -print"
```

目标 chromosome 的两个文件都出现在输出中才可运行。

### 2.3 Montage

Montage 用 1000 Genomes 的 Python 创建独立环境。只在新的 Workflow cluster 上执行一次。在 Mac 执行：

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

通过标准：输出最后一行为 `montage-ready`。

## 3. 修改代码后同步整个仓库

在 Mac 执行。不重建虚拟环境，不覆盖结果：

```zsh
source cloudlab_env.sh
rsync -az --delete \
  --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude 'results/' --exclude '.venv/' --exclude '.env*' \
  ./ "$SSH_USER@$WORKFLOW_NODE:pi-ebpf-tracing-handoff/"
ssh "$SSH_USER@$WORKFLOW_NODE" \
  'cd pi-ebpf-tracing-handoff && bash -n scripts/trace_script_bcc_1000genome.sh scripts/trace_script_bcc_montage.sh && echo sync-ok'
```

通过标准：输出 `sync-ok`。

## 4. 运行

启动命令都是 `bash scripts/launch_run.sh <workflow> <workloads>`。第二个参数是一个 cell 名或逗号分隔的多个，留空时运行该 workflow 的全部 cell。命令立即返回 `PID=` 和 `LOG=`，然后打印两行 `Watch:` 和 `Pull:`，它们就是第 5 节要用的全部命令。

结果目录由 trace 脚本命名，格式为 `<Workflow>_<cell>_<时间戳>`，多个 cell 时为 `<Workflow>_<N>tasks_<时间戳>`，时间为纽约时间。

### 4.1 1000 Genomes

默认矩阵是 1、2、4 个 chromosome 各三次，cell 名为 `classic_chr<N>_r<k>`。先跑最小 cell：

```zsh
source cloudlab_env.sh
bash scripts/launch_run.sh 1000genome classic_chr1_r1
```

### 4.2 Montage

每个规模先准备一次固定输入。规模为 `0.10`、`0.25`、`0.50`：

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

启动。cell 名为 `montage_m17_<规模>_r<k>`，规模写作 `0p10`、`0p25`、`0p50`：

```zsh
source cloudlab_env.sh
bash scripts/launch_run.sh montage montage_m17_0p10_r1
```

## 5. 看日志与拉回

第 4 节打印的两行就是本节。看日志：

```zsh
source cloudlab_env.sh; ssh "$SSH_USER@$WORKFLOW_NODE" 'tail -f ~/logs/<日志名>.log'
```

日志最后出现 `Results in:` 表示跑完。`Ctrl-C` 只停止查看。拉回：

```zsh
source cloudlab_env.sh; bash scripts/pull_run.sh logs/<日志名>.log
```

脚本把结果拉到 `results/<run 名>`，刷新 `results/index.html`，检查 `Design.md` 第 8 节列出的必需文件和 `lost_events=0`，然后打开报告。输出 `verified: <数量> cells in results/<run 名>` 表示通过。

远端日志的最后一行也是一条 `Pull to Mac:` 命令，带绝对路径，同样可以直接粘贴。

## 6. 出问题时

日志长时间不动时，看进程是否还活着。列表为空且日志没有 `Results in:` 表示中途失败，看最后 100 行找原因：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE" "pgrep -af 'trace_script_bcc_|bcc_tracer|run_1000genome|run_montage|phase1_metrics|viz.trace' || true"
ssh "$SSH_USER@$WORKFLOW_NODE" 'tail -100 ~/logs/<日志名>.log'
```

忘了日志名时列出最近的：

```zsh
ssh "$SSH_USER@$WORKFLOW_NODE" 'ls -t ~/logs | head'
```

拉回失败时，每个 `ERROR:` 行指出缺哪个文件。
