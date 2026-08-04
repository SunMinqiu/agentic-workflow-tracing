# Agentic Scientific Workflow I/O：研究概述与追踪手册

本文先说明如何在 CloudLab 上运行工作流并使用 eBPF/BCC 采集 I/O，再定义研究问题、分析框架和预期贡献。

## 第一部分：eBPF/BCC 追踪操作手册

> `$SSH_USER`、`$CLIENT_NODE`、API key、base URL 和模型名称均来自本地且不纳入 Git 的 `cloudlab_env.sh`。更换节点或 Provider 时只修改该文件。

三个已接入系统使用不同的运行依赖和 Provider 配置。不要混用。


| 系统                     | 配置文件                           | 远端 env 文件                         | key 变量             | 模型                             | Provider           |
| ---------------------- | ------------------------------ | --------------------------------- | ------------------ | ------------------------------ | ------------------ |
| **SciLink**            | `config/config_scilink.env`    | `.env.scilink`                    | `OPENAI_API_KEY`   | `gpt-4o-mini`                  | litellm + OpenAI   |
| **GenoMAS**            | `config/config_genomas.env`    | `.env.genomas` + `~/GenoMAS/.env` | `OPENAI_API_KEY_1` | `qwen3.6-35b`（FreeInference，默认）/ `gpt-4o-mini-2024-07-18`（OpenAI，可选） | OpenAI-compatible SDK，vendor 由 base URL 决定 |
| **1000genome classic** | `config/config_1000genome.env` | 可选 `.env.1000genome`              | 不需要                | 不适用                            | 本地 Python DAG，支持离线 |


SciLink 使用 OpenAI。GenoMAS 默认使用 FreeInference 的 `qwen3.6-35b`，配置来自本地的 `GENOMAS_OPENAI_API_KEY` 和 `GENOMAS_BASE_URL`。FreeInference 已实测返回 `cached_tokens`。GenoMAS 也可以切换到 OpenAI 的 `gpt-4o-mini-2024-07-18`。两种 vendor 的缓存资格、粒度和保留策略可能不同，不能把两个 vendor 的 cell 放在同一个比较实验中。1000 Genomes classic 不使用 Provider 或 API key。

执行频率如下：

- 🟥 首次部署或更换节点时执行一次。
- 🟨 每次打开新终端时执行一次。
- 🟩 每次实验运行时执行。
- 🔧 仅在对应配置或代码发生变化时执行。

除非标题注明在 CloudLab 节点执行，以下命令均在 Mac 上执行。

### 1. 配置环境



#### 🟨 每次打开新终端

```zsh
source cloudlab_env.sh
REMOTE_HOME=$(ssh "$SSH_USER@$CLIENT_NODE" 'printf %s "$HOME"')
```

成功标准：终端输出 `[cloudlab_env] keys OK` 和当前的 `CLIENT=…`。

#### 🔧 修改代码后同步代码

`.env*` 只保存在远端并包含 API key，因此同步代码时会明确排除这些文件。

```zsh
rsync -az --delete \
  --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude 'results/' --exclude '.venv/' --exclude '.env*' \
  ./ "$SSH_USER@$CLIENT_NODE:pi-ebpf-tracing-handoff/"
```



#### 🔧 修改 Provider、API key 或模型后同步环境变量

`rsync` 不会修改远端的 `.env.*`。以下命令只替换 Provider 相关字段，保留 Python 和数据路径，并删除旧的重复配置。

```zsh
source cloudlab_env.sh

# SciLink：OpenAI key，不设置 FreeInference base URL
ssh "$SSH_USER@$CLIENT_NODE" \
  "sed -i -E '/^(export )?(OPENAI_API_KEY|OPENAI_BASE_URL|OPENAI_API_BASE|SCILINK_MODEL)=/d' pi-ebpf-tracing-handoff/.env.scilink; cat >> pi-ebpf-tracing-handoff/.env.scilink; chmod 600 pi-ebpf-tracing-handoff/.env.scilink" <<EOF
export OPENAI_API_KEY="$OPENAI_API_KEY"
export SCILINK_MODEL="$SCILINK_MODEL"
EOF

# GenoMAS：FreeInference 专用 key/base URL；模型使用裸名
ssh "$SSH_USER@$CLIENT_NODE" \
  "sed -i -E '/^(export )?(OPENAI_API_KEY_1|OPENAI_ORGANIZATION_1|OPENAI_BASE_URL|OPENAI_API_BASE|GENOMAS_MODEL|GENOMAS_VENDOR|GENOMAS_CAPTURE_STREAM_TIMING)=/d' pi-ebpf-tracing-handoff/.env.genomas; cat >> pi-ebpf-tracing-handoff/.env.genomas; chmod 600 pi-ebpf-tracing-handoff/.env.genomas" <<EOF
export OPENAI_API_KEY_1="$GENOMAS_OPENAI_API_KEY"
export OPENAI_BASE_URL="$GENOMAS_BASE_URL"
export OPENAI_API_BASE="$GENOMAS_BASE_URL"
export GENOMAS_MODEL="qwen3.6-35b"
export GENOMAS_VENDOR="FreeInference"
export GENOMAS_CAPTURE_STREAM_TIMING=1
EOF
ssh "$SSH_USER@$CLIENT_NODE" \
  "sed -i -E '/^(export )?(OPENAI_API_KEY_1|OPENAI_BASE_URL|OPENAI_API_BASE)=/d' GenoMAS/.env; cat >> GenoMAS/.env; chmod 600 GenoMAS/.env" <<EOF
OPENAI_API_KEY_1=$GENOMAS_OPENAI_API_KEY
OPENAI_BASE_URL=$GENOMAS_BASE_URL
OPENAI_API_BASE=$GENOMAS_BASE_URL
EOF
```

上面是 GenoMAS 的默认 FreeInference 配置。模型必须使用裸名 `qwen3.6-35b`。`GENOMAS_VENDOR` 用于报告显示真实服务商，不能用 SDK 中的 `provider=openai` 代替，因为 FreeInference 也走 OpenAI-compatible SDK。

如需改用 OpenAI，执行下面的完整切换命令。它会删除 FreeInference base URL，并明确固定 vendor 和模型。

```zsh
source cloudlab_env.sh

ssh "$SSH_USER@$CLIENT_NODE" \
  "sed -i -E '/^(export )?(OPENAI_API_KEY_1|OPENAI_ORGANIZATION_1|OPENAI_BASE_URL|OPENAI_API_BASE|GENOMAS_MODEL|GENOMAS_VENDOR|GENOMAS_CAPTURE_STREAM_TIMING)=/d' pi-ebpf-tracing-handoff/.env.genomas; cat >> pi-ebpf-tracing-handoff/.env.genomas; chmod 600 pi-ebpf-tracing-handoff/.env.genomas" <<EOF
export OPENAI_API_KEY_1="$OPENAI_API_KEY"
export OPENAI_ORGANIZATION_1="$OPENAI_ORGANIZATION"
export GENOMAS_MODEL="gpt-4o-mini-2024-07-18"
export GENOMAS_VENDOR="OpenAI"
export GENOMAS_CAPTURE_STREAM_TIMING=1
EOF
ssh "$SSH_USER@$CLIENT_NODE" \
  "sed -i -E '/^(export )?(OPENAI_API_KEY_1|OPENAI_ORGANIZATION_1|OPENAI_BASE_URL|OPENAI_API_BASE)=/d' GenoMAS/.env; cat >> GenoMAS/.env; chmod 600 GenoMAS/.env" <<EOF
OPENAI_API_KEY_1=$OPENAI_API_KEY
OPENAI_ORGANIZATION_1=$OPENAI_ORGANIZATION
EOF
```

写入后运行 GenoMAS API 预检。只有看到 `GENOMAS_PREFLIGHT_OK`、非空 usage 和 `FIRST_TOKEN_MS` 才能开始追踪。流式预检保证后续运行能产生真实 TTFT 和 TPOT 图。

```zsh
ssh "$SSH_USER@$CLIENT_NODE" 'cd "$HOME/pi-ebpf-tracing-handoff" && bash -lc '\''
  set -a; source .env.genomas; set +a
  cd "$GENOMAS_REPO"
  "$AGENT_PYTHON" - <<"PY"
import os, time
from openai import OpenAI
c = OpenAI(api_key=os.environ["OPENAI_API_KEY_1"], organization=os.environ.get("OPENAI_ORGANIZATION_1") or None)
started = time.time()
stream = c.chat.completions.create(
    model=os.environ["GENOMAS_MODEL"],
    messages=[{"role": "user", "content": "Reply exactly OK"}],
    max_tokens=8,
    stream=True,
    stream_options={"include_usage": True},
)
first = None
usage = None
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content and first is None:
        first = time.time()
    if chunk.usage is not None:
        usage = chunk.usage
assert first is not None and usage is not None
print("GENOMAS_PREFLIGHT_OK", usage.total_tokens, "FIRST_TOKEN_MS", round((first-started)*1000, 1))
PY
'\''
```

若 GenoMAS 改用 FreeInference，预检要带 `base_url`（其余相同）：

```zsh
ssh "$SSH_USER@$CLIENT_NODE" 'cd "$HOME/pi-ebpf-tracing-handoff" && bash -lc '\''
  set -a; source .env.genomas; set +a
  cd "$GENOMAS_REPO"
  "$AGENT_PYTHON" - <<"PY"
import os, time
from openai import OpenAI
c = OpenAI(api_key=os.environ["OPENAI_API_KEY_1"], base_url=os.environ["OPENAI_BASE_URL"])
started = time.time()
stream = c.chat.completions.create(
    model=os.environ["GENOMAS_MODEL"],
    messages=[{"role": "user", "content": "Reply exactly OK"}],
    max_tokens=8,
    stream=True,
    stream_options={"include_usage": True},
)
first = None
usage = None
for chunk in stream:
    delta = chunk.choices[0].delta if chunk.choices else None
    if delta is not None and (delta.content or getattr(delta, "reasoning_content", None)) and first is None:
        first = time.time()
    if chunk.usage is not None:
        usage = chunk.usage
assert first is not None and usage is not None
print("GENOMAS_PREFLIGHT_OK", usage.total_tokens, "FIRST_TOKEN_MS", round((first-started)*1000, 1))
PY
'\''
```



#### 🟥 首次部署或更换节点后执行全量部署

```zsh
bash scripts/deploy_scilink_to_client.sh    # 或 deploy_genomas_to_client.sh
```

此命令中的 `uv venv --clear` 会重建整个虚拟环境，通常需要数分钟。仅更换 API key、模型或 Provider 时，应执行上一节的环境变量同步命令。

### 2. 运行实验

所有运行命令都使用 `nohup … >log 2>&1 </dev/null &`。SSH 返回提示符后，任务会继续在节点后台运行，此时可以断开连接。使用逗号分隔的 `RUN_WORKLOADS` 选择工作负载子集；留空表示运行全部工作负载。

#### GenoMAS

当前矩阵共 9 个 cell。`A_c{1,2,3,4,8}_w*` 是 cohort sweep，`B_t{1,2,4}_w2` 是 trait-count sweep。`A_c4_w2` 和 `A_c4_w4` 使用相同的 4-cohort 输入，专门比较 2 workers 与 4 workers。最小测试使用 `A_c1_w1`。

```zsh
ssh "$SSH_USER@$CLIENT_NODE" \
  'cd pi-ebpf-tracing-handoff || exit 1
   sudo -n true || exit 1
   nohup sudo -n -E env \
     GENOMAS_VENDOR="FreeInference" \
     GENOMAS_MODEL="qwen3.6-35b" \
     GENOMAS_CAPTURE_STREAM_TIMING=1 \
     RUN_WORKLOADS="A_c1_w1" \
     bash scripts/trace_script_bcc_genomas.sh \
     >"$HOME/genomas_run.log" 2>&1 </dev/null &
   echo "GenoMAS PID $!"'
```

运行多组时只改逗号分隔的 `RUN_WORKLOADS`，例如 `A_c1_w1,A_c2_w2,B_t2_w2`。删除 `RUN_WORKLOADS` 可运行全部 9 个 cell。不要删除命令中的 `GENOMAS_VENDOR` 和 `GENOMAS_MODEL`，它们保证一个结果目录内不会因远端残留环境变量而混用 vendor 或模型。脚本会在所有 cell 后处理完成后自动生成根目录下的 `kvcache_report.md` 和 `kvcache_report.html`。比较 4-cohort 并发度时使用 `RUN_WORKLOADS="A_c4_w2,A_c4_w4"`。

使用 OpenAI 时，先执行上一节的 OpenAI 环境切换命令，再把运行命令中的两行改为：

```zsh
GENOMAS_VENDOR="OpenAI" \
GENOMAS_MODEL="gpt-4o-mini-2024-07-18" \
```


#### SciLink


| workload                       | 类型      | 内容                                                                                       |
| ------------------------------ | ------- | ---------------------------------------------------------------------------------------- |
| `eels_plasmons_basic`          | analyze | EELS 等离激元 mapping                                                                        |
| `eels_identification_basic`    | analyze | 1D EELS 谱识别                                                                              |
| `polycrystalline_grains_basic` | analyze | 2D 晶粒分割                                                                                  |
| `planning_critical_materials`  | plan    | 规划 agent，使用实验数据、知识目录和 embedding |


```zsh
ssh "$SSH_USER@$CLIENT_NODE" \
  'cd pi-ebpf-tracing-handoff || exit 1
   sudo -n true || exit 1
   nohup sudo -n -E env RUN_WORKLOADS="polycrystalline_grains_basic" \
     bash scripts/trace_script_bcc_scilink.sh \
     >"$HOME/scilink_run.log" 2>&1 </dev/null &
   echo "SciLink PID $!"'
```

运行多组时使用逗号分隔，例如 `RUN_WORKLOADS="eels_plasmons_basic,eels_identification_basic,polycrystalline_grains_basic"`。加入 `planning_critical_materials` 可同时测 planning workflow。脚本会先检查 SciLink 子命令和全部输入路径，然后才启动 BCC。每个 inference 会记录 `messages.jsonl`、token usage、`cacheRead`、provider request ID 和端到端时间。运行结束后会自动生成 KV-cache JSON、图表、Markdown 和 HTML 报告。

SSH 输出 PID 并返回提示符后，任务已脱离终端，可以断开连接。`sudo -n true` 失败表示当前节点需要先建立 sudo 凭据，命令不会静默启动一个无法运行的后台任务。

#### 查看并拉回 GenoMAS 和 SciLink 结果

查看日志：

```zsh
# GenoMAS
ssh "$SSH_USER@$CLIENT_NODE" 'tail -f "$HOME/genomas_run.log"'

# SciLink
ssh "$SSH_USER@$CLIENT_NODE" 'tail -f "$HOME/scilink_run.log"'
```

`Ctrl-C` 只退出 `tail`，不会停止实验。查看进程：

```zsh
# GenoMAS
ssh "$SSH_USER@$CLIENT_NODE" \
  "pgrep -af '[t]race_script_bcc_genomas|adapters.[g]enomas.launcher|[b]cc_tracer|analysis.[p]hase1_metrics|viz.[t]race' || true"

# SciLink
ssh "$SSH_USER@$CLIENT_NODE" \
  "pgrep -af '[t]race_script_bcc_scilink|adapters.[s]cilink.launcher|[b]cc_tracer|analysis.[p]hase1_metrics|viz.[t]race' || true"
```

没有输出表示该系统已无运行中的工作流、tracer 或后处理进程。GenoMAS 日志出现 `Results in:`，或 SciLink 日志出现 `All done. Results in:`，才表示后处理、KV-cache 报告和 Index 已完成。

以下函数从指定日志读取结果目录，拉回结果，检查完整性，并打开每个 cell 的 Index。

```zsh
pull_agentic_run() {
  local remote_log="$1"
  local remote_run remote_cells local_out cell required failed=0 cells=0

  remote_run=$(ssh "$SSH_USER@$CLIENT_NODE" \
    "sed -n 's/^Output dir: //p' \"\$HOME/$remote_log\" | head -1")
  if [[ -z "$remote_run" ]]; then
    echo "ERROR: result path not found in $remote_log" >&2
    return 1
  fi
  remote_cells=$(ssh "$SSH_USER@$CLIENT_NODE" \
    "find '$remote_run' -mindepth 2 -maxdepth 2 -type f -name manifest.json | wc -l")
  if (( remote_cells == 0 )); then
    echo "ERROR: remote run contains zero result cells: $remote_run" >&2
    return 1
  fi

  local_out="results/$(basename "$remote_run")"
  mkdir -p "$local_out"
  rsync -az --progress --checksum --partial \
    --exclude 'work/' --exclude 'bcc.out' \
    "$SSH_USER@$CLIENT_NODE:$remote_run/" "$local_out/"

  echo "Regenerating the KV report locally so figures match local code."
  PYTHONPATH=src python -m agent_io_tracing.analysis.kvcache.report \
    --results "$local_out" --runs . --dump-prefixes || return 1

  for cell in "$local_out"/*; do
    [[ -f "$cell/manifest.json" ]] || continue
    ((cells += 1))
    for required in \
      ebpf_events.log parsed.json manifest.json pi_events.jsonl tool_calls.log \
      messages.jsonl kvcache_demand.json kvcache_logical.json \
      kvcache_report.md kvcache_report.html \
      phase1_metrics.json parallelism_summary.json trace_quality.json \
      lineage/artifacts.csv lineage/execution_unit_io.csv \
      visualizations/file_access_volume.png visualizations/rw_asymmetry.png \
      visualizations/request_size_rw_cdf.png \
      visualizations/byte_normalized_summary.png visualizations/index.html; do
      if [[ ! -s "$cell/$required" ]]; then
        echo "ERROR: missing or empty: $cell/$required" >&2
        failed=1
      fi
    done
    if [[ ! -f "$cell/bcc.err" ]] || ! tail -1 "$cell/bcc.err" | grep -q 'lost_events=0'; then
      echo "ERROR: lost-event check failed: $cell/bcc.err" >&2
      failed=1
    fi
  done

  for required in kvcache_report.md kvcache_report.html; do
    if [[ ! -s "$local_out/$required" ]]; then
      echo "ERROR: missing or empty: $local_out/$required" >&2
      failed=1
    fi
  done

  (( cells > 0 )) || failed=1
  if (( failed == 0 )); then
    open "$local_out"/*/visualizations/index.html
    open "$local_out/kvcache_report.html"
  else
    echo "Result pull failed integrity checks; not opening incomplete output." >&2
    return 1
  fi
}
```

定义函数后，按系统执行一行：

```zsh
pull_agentic_run genomas_run.log
pull_agentic_run scilink_run.log
```

每次只需运行其中一行。函数使用日志记录的 `Output dir`，不会根据目录时间猜测，因此不会误拉 1000 Genomes 或 Montage 结果。

### 3. 1000 Genomes classic baseline

该路径不使用 Pegasus、HTCondor 或 LLM。`run_1000genome.py` 先执行 `individuals → individuals_merge` 和并行的 `sifting` 分支，再执行 `mutation_overlap` 与 `frequency`。每个 task 使用独立 sandbox，整个 DAG 共享一个全局 worker 上限。

默认矩阵为 1、2、4 个 chromosome，各重复 3 次：

```text
classic_chr1_r1  classic_chr1_r2  classic_chr1_r3
classic_chr2_r1  classic_chr2_r2  classic_chr2_r3
classic_chr4_r1  classic_chr4_r2  classic_chr4_r3
```

默认固定 `INDIVIDUAL_JOBS=2`、`MAX_WORKERS=4`、`POPULATIONS=ALL`。

#### 首次联网准备【CloudLab client，只做一次】

运行阶段不会下载数据；必须先把代码、Python 依赖和解压后的输入准备好：

```bash
LUSTRE_USER_DIR="${MOUNT_PATH:-/mnt/lustrefs}/$USER"
mkdir -p "$LUSTRE_USER_DIR"
git clone https://github.com/pegasus-isi/1000genome-workflow.git \
  "$LUSTRE_USER_DIR/1000genome-workflow"
cd "$LUSTRE_USER_DIR/1000genome-workflow"

# upstream 脚本假定这个目录已经存在。
mkdir -p data/20130502/sifting
bash prepare_input.sh

# 不安装 Pegasus/HTCondor。使用兼容当前 Python 的科学计算包；不要强制
# 安装 upstream 为旧 Python 固定的版本号。
curl -LsSf https://astral.sh/uv/install.sh | sh
"$HOME/.local/bin/uv" venv --python 3.10 .venv
"$HOME/.local/bin/uv" pip install --python .venv/bin/python \
  numpy matplotlib pillow pandas plotly
```

开始前必须存在：

```text
$WORKFLOW_REPO/bin/{individuals,individuals_merge,sifting,mutation_overlap,frequency}.py
$DATASET_DIR/columns.txt
$DATASET_DIR/ALL.chr1.250000.vcf
$DATASET_DIR/sifting/ALL.chr1.phase3_shapeit2_mvncall_integrated_v5.20130502.sites.annotation.vcf
$POPULATION_DIR/ALL
```

2-chromosome 和 4-chromosome cell 还需要 chr2 或 chr2 至 chr4 的两类 VCF。输入必须是解压后的 `.vcf`，不能只保留 `.vcf.gz`。

#### 隔离的小规模追踪

以下测试只读取原始 VCF，并在独立的 `BASE_OUT` 下创建 task sandbox、trace、指标、图和 `visualizations/index.html`。测试不会修改 upstream checkout、原始数据或已有结果。不要并发运行这些测试，否则 I/O 竞争会污染对比。

运行时间仅是目标区间。首次运行应先用最小档测量当前节点速度，再决定是否运行后两档。

#### 1. 最小：1 chromosome × 300 VCF lines，目标约 5 分钟

```zsh
ssh "$SSH_USER@$CLIENT_NODE" '
  cd "$HOME/pi-ebpf-tracing-handoff"
  REPO="/mnt/lustrefs/$USER/1000genome-workflow"
  OUT="/mnt/lustrefs/$USER/pi-ebpf-tracing-handoff/results/classic_smoke_$(date +%Y%m%d_%H%M%S)"
  sudo -E env \
    BASE_OUT="$OUT" \
    WORKFLOW_REPO="$REPO" \
    DATASET_DIR="$REPO/data/20130502" \
    POPULATION_DIR="$REPO/data/populations" \
    AGENT_PYTHON="$REPO/.venv/bin/python" \
    POST_PYTHON="$REPO/.venv/bin/python" \
    CLASSIC_OFFLINE=1 \
    BCC_PERF_PAGES=1024 \
    RUN_WORKLOADS=classic_chr1_r1 \
    ROWS_PER_CHROMOSOME=300 \
    CLASSIC_VCF_RECORD_LIMIT=300 \
    INDIVIDUAL_JOBS=1 \
    MAX_WORKERS=1 \
    nohup bash scripts/trace_script_bcc_1000genome.sh \
      > "$HOME/classic_smoke.log" 2>&1 < /dev/null &
'
```



#### 2. 小：1 chromosome × 750 VCF lines，目标约 5–10 分钟

```zsh
ssh "$SSH_USER@$CLIENT_NODE" '
  cd "$HOME/pi-ebpf-tracing-handoff"
  REPO="/mnt/lustrefs/$USER/1000genome-workflow"
  OUT="/mnt/lustrefs/$USER/pi-ebpf-tracing-handoff/results/classic_small_$(date +%Y%m%d_%H%M%S)"
  sudo -E env \
    BASE_OUT="$OUT" \
    WORKFLOW_REPO="$REPO" \
    DATASET_DIR="$REPO/data/20130502" \
    POPULATION_DIR="$REPO/data/populations" \
    AGENT_PYTHON="$REPO/.venv/bin/python" \
    POST_PYTHON="$REPO/.venv/bin/python" \
    CLASSIC_OFFLINE=1 \
    BCC_PERF_PAGES=1024 \
    RUN_WORKLOADS=classic_chr1_r1 \
    ROWS_PER_CHROMOSOME=750 \
    CLASSIC_VCF_RECORD_LIMIT=750 \
    INDIVIDUAL_JOBS=1 \
    MAX_WORKERS=2 \
    nohup bash scripts/trace_script_bcc_1000genome.sh \
      > "$HOME/classic_small.log" 2>&1 < /dev/null &
'
```



#### 3. 中：1 chromosome × 2,000 VCF lines，目标约 10–20 分钟

```zsh
ssh "$SSH_USER@$CLIENT_NODE" '
  cd "$HOME/pi-ebpf-tracing-handoff"
  REPO="/mnt/lustrefs/$USER/1000genome-workflow"
  OUT="/mnt/lustrefs/$USER/pi-ebpf-tracing-handoff/results/classic_medium_$(date +%Y%m%d_%H%M%S)"
  sudo -E env \
    BASE_OUT="$OUT" \
    WORKFLOW_REPO="$REPO" \
    DATASET_DIR="$REPO/data/20130502" \
    POPULATION_DIR="$REPO/data/populations" \
    AGENT_PYTHON="$REPO/.venv/bin/python" \
    POST_PYTHON="$REPO/.venv/bin/python" \
    CLASSIC_OFFLINE=1 \
    BCC_PERF_PAGES=1024 \
    RUN_WORKLOADS=classic_chr1_r1 \
    ROWS_PER_CHROMOSOME=2000 \
    CLASSIC_VCF_RECORD_LIMIT=2000 \
    INDIVIDUAL_JOBS=1 \
    MAX_WORKERS=2 \
    nohup bash scripts/trace_script_bcc_1000genome.sh \
      > "$HOME/classic_medium.log" 2>&1 < /dev/null &
'
```

`CLASSIC_VCF_RECORD_LIMIT` 会同时截取 main VCF 和 annotation VCF 并保留 header。副本写入本次 `BASE_OUT`，截取过程在 eBPF tracer 启动前完成。1000 Genomes 的 `individuals.py` 会逐列处理约 2,504 个样本，因此 2,000 行并非最小测试。

三个档位用于验证 tracing、DAG 并发和指标链路，不作为正式科学结果。目标时间以当前节点首次正确部署后的实测为准。

每次运行结束后必须确认 `bcc.err` 的最后一行是 `lost_events=0`。否则该 trace 只能用于调试和估算运行成本，不能进入正式对比。

运行完整的 9-cell 矩阵时，使用默认值 `ROWS_PER_CHROMOSOME=250000` 和 `CLASSIC_VCF_RECORD_LIMIT=0`，并删除 `RUN_WORKLOADS` 或将其设为空字符串。

#### 完全离线运行

“离线”表示运行期间不访问公网且不调用 API。CloudLab 内部的 Lustre 挂载和 SSH 控制连接仍可使用。Classic runner 本身没有下载或网络调用，并默认设置 `CLASSIC_OFFLINE=1`。该标记会写入 `manifest.json` 和 `work/classic_run_summary.json`，便于审计。

在隔离节点执行前，从可联网机器一次性传入以下内容：

- `1000genome-workflow` checkout，包括 `bin/`、`data/populations/`、`columns.txt` 和所需的全部解压 VCF
- 可直接使用的 Python 3.10+ 环境，或包含 `numpy`、`matplotlib`、`pillow`、`pandas`、`plotly` 及其依赖的本地 wheelhouse
- 系统级 BCC 包和与当前 kernel 匹配的 headers。普通 Python wheelhouse 无法替代 BCC

如果使用 wheelhouse，在离线节点安装时禁止访问索引：

```bash
python3.10 -m venv "$WORKFLOW_REPO/.venv"
"$WORKFLOW_REPO/.venv/bin/pip" install \
  --no-index --find-links /path/to/wheelhouse \
  numpy matplotlib pillow pandas plotly
```

确认输入和依赖已落盘后，执行上一节的命令。不需要 API key、`.env.genomas` 或 `.env.scilink`。Trace 脚本会在启动 tracer 前检查 repo、Python、`columns.txt`、population 文件和每个 chromosome 的两个 VCF。缺少任何一项时脚本会直接失败，不会尝试联网补齐。

每个成功 cell 应至少生成：

```text
ebpf_events.log
parsed.json
artifact_sizes.json
manifest.json
work/classic_run_summary.json
execution_units.jsonl
phase1_metrics.json
parallelism_summary.json
trace_quality.json
lineage/artifacts.csv
lineage/execution_unit_io.csv
lineage/execution_unit_summary.json
visualizations/file_access_volume.png
visualizations/rw_asymmetry.png
visualizations/request_size_rw_cdf.png
visualizations/byte_normalized_summary.png
visualizations/index.html
```

Classic run 不生成 LLM summary。Universal lineage、parallelism、metrics 和 dashboard 与 agentic trace 使用同一套后处理。

#### 查看过程并判断结束

按正在运行的档位选择对应日志：

```zsh
# 300-line smoke
ssh "$SSH_USER@$CLIENT_NODE" 'tail -f "$HOME/classic_smoke.log"'

# 750-line small
ssh "$SSH_USER@$CLIENT_NODE" 'tail -f "$HOME/classic_small.log"'

# 2,000-line medium
ssh "$SSH_USER@$CLIENT_NODE" 'tail -f "$HOME/classic_medium.log"'
```

`Ctrl-C` 只退出 `tail`，不会停止远端实验。查看 tracer、workflow 和后处理进程：

```zsh
ssh "$SSH_USER@$CLIENT_NODE" \
  "pgrep -af 'trace_script_bcc_1000genome|run_1000genome.py|phase1_metrics|visualize' || true"
```

查看当前日志记录的远端结果目录：

```zsh
ssh "$SSH_USER@$CLIENT_NODE" '
  for log in "$HOME/classic_smoke.log" "$HOME/classic_small.log" "$HOME/classic_medium.log"; do
    [[ -f "$log" ]] || continue
    printf "%s: " "$(basename "$log")"
    awk "/^Output:/ {print \$2; found=1; exit} END {if (!found) print \"output path not written yet\"}" "$log"
  done
'
```

日志最后出现以下内容才表示整个 trace 和后处理完成：

```text
Results: /mnt/lustrefs/.../results/classic_...
```

只出现 `=== Classic post-processing ===` 表示工作流已完成，但指标、图或 Index 仍在生成。进程列表为空且日志没有 `Results:` 时，脚本已经异常退出，应先查看日志末尾。

```zsh
ssh "$SSH_USER@$CLIENT_NODE" 'tail -100 "$HOME/classic_smoke.log"'
```

将 `classic_smoke.log` 替换为实际运行的 `classic_small.log` 或 `classic_medium.log`。

#### 拉回、校验并打开结果

以下命令选择最新的 `classic_*` 结果，不会误选 SciLink、GenoMAS 或 Montage：

```zsh
REMOTE_RUN=$(ssh "$SSH_USER@$CLIENT_NODE" \
  'ls -1dt /mnt/lustrefs/$USER/pi-ebpf-tracing-handoff/results/classic_*/ 2>/dev/null | head -1')
if [[ -z "$REMOTE_RUN" ]]; then
  echo "ERROR: no classic result found" >&2
else
  LOCAL="results/$(basename "$REMOTE_RUN")"
  mkdir -p "$LOCAL"
  rsync -az --progress --checksum --partial \
    --exclude 'work/' --exclude 'bcc.out' \
    "$SSH_USER@$CLIENT_NODE:$REMOTE_RUN" "$LOCAL/"

  failed=0
  cells=0
  for cell in "$LOCAL"/*; do
    [[ -f "$cell/manifest.json" ]] || continue
    ((cells += 1))
    for required in \
      ebpf_events.log parsed.json artifact_sizes.json manifest.json \
      execution_units.jsonl phase1_metrics.json parallelism_summary.json \
      trace_quality.json lineage/artifacts.csv \
      lineage/execution_unit_io.csv lineage/execution_unit_summary.json \
      visualizations/file_access_volume.png visualizations/rw_asymmetry.png \
      visualizations/request_size_rw_cdf.png \
      visualizations/byte_normalized_summary.png visualizations/index.html; do
      if [[ ! -s "$cell/$required" ]]; then
        echo "ERROR: missing or empty: $cell/$required" >&2
        failed=1
      fi
    done
    for required in pi_events.jsonl tool_calls.log; do
      if [[ ! -f "$cell/$required" ]]; then
        echo "ERROR: missing: $cell/$required" >&2
        failed=1
      fi
    done
    if [[ ! -f "$cell/bcc.err" ]] || ! tail -1 "$cell/bcc.err" | grep -q 'lost_events=0'; then
      echo "ERROR: lost-event check failed: $cell/bcc.err" >&2
      failed=1
    fi
  done

  if (( cells == 0 )); then
    echo "ERROR: no completed cells found under $LOCAL" >&2
    failed=1
  fi
  if (( failed == 0 )); then
    open "$LOCAL"/*/visualizations/index.html
  else
    echo "1000 Genomes pull failed integrity checks; not opening incomplete output." >&2
  fi
fi
```

指定某次实验时，不要使用“最新结果”查找。直接设置完整路径：

```zsh
REMOTE_RUN="/mnt/lustrefs/$SSH_USER/pi-ebpf-tracing-handoff/results/classic_smoke_YYYYMMDD_HHMMSS/"
```



### 4. Montage traditional baseline

日常运行包含两个步骤。更换节点时还需执行一次环境配置。

1. 在 trace 开始前下载并冻结 FITS 输入。
2. 运行正式 eBPF trace，生成本项目的 metrics、lineage、I/O 图和 Index。

`mosaic.png` 是 Montage 的科学结果，不是 characterization 图。正式结果应查看 `visualizations/index.html`。当前节点已完成 `0.10°` 输入准备和正式 smoke。重复运行 `0.10°` 实验时，直接从“最小正式 eBPF trace”开始。

当前节点没有 HTCondor，正式实验使用项目内的 direct stage driver。该 driver 保留 11 个 Montage stage 及其依赖，并记录每个 stage 的 PID 和时间区间。Pegasus 不进入被测进程。

#### 1. 配置正式环境【更换节点时执行一次】

当前节点已经配置完成，无需重复执行。更换节点时，使用 1000 Genomes 的 Python 解释器创建独立的 Montage venv。此过程不会修改 1000 Genomes 环境，所有新文件都写入 Lustre。

```zsh
ssh "$SSH_USER@$CLIENT_NODE" 'bash -s' <<'REMOTE'
set -euo pipefail

ROOT="/mnt/lustrefs/$USER/montage"
BASE_PYTHON="/mnt/lustrefs/$USER/1000genome-workflow/.venv/bin/python"
TRACE_VENV="$ROOT/trace-venv"
[[ -x "$BASE_PYTHON" ]] || { echo "Missing Python 3.10+ runtime: $BASE_PYTHON" >&2; exit 1; }

mkdir -p \
  "$ROOT/tmp" "$ROOT/logs" "$ROOT/input" \
  "$ROOT/cache/pip" "$ROOT/cache/python" \
  "$ROOT/cache/matplotlib" "$ROOT/cache/xdg" "$ROOT/cache/config"
export TMPDIR="$ROOT/tmp"
export PIP_CACHE_DIR="$ROOT/cache/pip"
export PYTHONPYCACHEPREFIX="$ROOT/cache/python"
export MPLCONFIGDIR="$ROOT/cache/matplotlib"
export XDG_CACHE_HOME="$ROOT/cache/xdg"
export XDG_CONFIG_HOME="$ROOT/cache/config"

"$BASE_PYTHON" -m venv "$TRACE_VENV"
"$TRACE_VENV/bin/python" -m pip install --upgrade pip
"$TRACE_VENV/bin/python" -m pip install \
  MontagePy astropy numpy matplotlib pillow pandas plotly importlib_resources
"$TRACE_VENV/bin/python" --version
"$TRACE_VENV/bin/python" -c 'import MontagePy, numpy, pandas, plotly'
REMOTE
```

修改代码后只需执行“一、配环境”中的“改了代码 → 推代码”，不需要重建 venv。

#### 2. 准备固定输入【每个规模执行一次】

当前节点的 `0.10°` 输入已经准备完成，无需重复下载。准备新规模时只修改第一行的 `SIZE`。下载、manifest 和校验都写入 Lustre，并在追踪开始前完成。

```zsh
SIZE=0.25
ssh "$SSH_USER@$CLIENT_NODE" "bash -s -- $SIZE" <<'REMOTE'
set -euo pipefail

SIZE="$1"
ROOT="/mnt/lustrefs/$USER/montage"
TAG="${SIZE/./p}deg"
cd "$HOME/pi-ebpf-tracing-handoff"

export TMPDIR="$ROOT/tmp"
export PIP_CACHE_DIR="$ROOT/cache/pip"
export PYTHONPYCACHEPREFIX="$ROOT/cache/python"
export MPLCONFIGDIR="$ROOT/cache/matplotlib"
export XDG_CACHE_HOME="$ROOT/cache/xdg"
export XDG_CONFIG_HOME="$ROOT/cache/config"

PYTHONPATH="$PWD/src" "$ROOT/trace-venv/bin/python" \
  -m agent_io_tracing.adapters.montage.prepare_input \
  --size-deg "$SIZE" \
  --output "$ROOT/input/m17_${TAG}"
REMOTE
```

准备顺序为 `0.10`、`0.25`、`0.50`。已存在的输入只做 checksum 校验，不会覆盖。

#### 3. 运行最小正式 eBPF trace

以下命令只运行 `0.10° × 0.10°` 的第 1 次重复。结果、运行日志、cache 和临时文件都写入 Lustre。

```zsh
ssh "$SSH_USER@$CLIENT_NODE" 'bash -s' <<'REMOTE'
set -euo pipefail

cd "$HOME/pi-ebpf-tracing-handoff"
ROOT="/mnt/lustrefs/$USER/montage"
OUT="/mnt/lustrefs/$USER/pi-ebpf-tracing-handoff/results/classic_montage_smoke_$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/logs/montage_trace_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$ROOT/logs"
ln -sfn "$LOG" "$ROOT/latest_trace_log"

sudo -E env \
  BASE_OUT="$OUT" \
  MONTAGE_ROOT="$ROOT" \
  MONTAGE_INPUT_ROOT="$ROOT/input" \
  MONTAGE_PYTHON="$ROOT/trace-venv/bin/python" \
  AGENT_PYTHON="$ROOT/trace-venv/bin/python" \
  POST_PYTHON="$ROOT/trace-venv/bin/python" \
  MONTAGE_OFFLINE=1 \
  BCC_PERF_PAGES=1024 \
  RUN_WORKLOADS=montage_m17_0p10_r1 \
  nohup bash scripts/trace_script_bcc_montage.sh \
    > "$LOG" 2>&1 < /dev/null &
printf 'RESULTS=%s\nLOG=%s\n' "$OUT" "$LOG"
REMOTE
```

从小到大只需替换 `RUN_WORKLOADS`：


| 规模      | 第一次运行                 | 三次重复                                                          |
| ------- | --------------------- | ------------------------------------------------------------- |
| `0.10°` | `montage_m17_0p10_r1` | `montage_m17_0p10_r1,montage_m17_0p10_r2,montage_m17_0p10_r3` |
| `0.25°` | `montage_m17_0p25_r1` | `montage_m17_0p25_r1,montage_m17_0p25_r2,montage_m17_0p25_r3` |
| `0.50°` | `montage_m17_0p50_r1` | `montage_m17_0p50_r1,montage_m17_0p50_r2,montage_m17_0p50_r3` |


该 trace 包含 `header → raw_table → projection → projected_table → overlaps → difference_fitting → background_model → background_correction → corrected_table → coadd → render`。每个 stage 都写入 `execution_units.jsonl`，并进入统一的 lineage、metrics、figures 和 Index。

#### 4. 查看过程

```zsh
ssh "$SSH_USER@$CLIENT_NODE" '
  LOG="$(readlink -f "/mnt/lustrefs/$USER/montage/latest_trace_log")"
  tail -f "$LOG"
'
```

查看 tracer、workflow 和后处理进程：

```zsh
ssh "$SSH_USER@$CLIENT_NODE" \
  "pgrep -af 'trace_script_bcc_montage|run_montage|phase1_metrics|visualize' || true"
```

日志最后出现 `Results: /mnt/lustrefs/.../classic_montage_smoke_...` 才表示工具图和 Index 已生成。只出现 `=== Montage post-processing ===` 表示后处理仍在运行。

#### 5. 拉回结果并打开 Index

```zsh
REMOTE_RUN=$(ssh "$SSH_USER@$CLIENT_NODE" \
  'ls -1dt /mnt/lustrefs/$USER/pi-ebpf-tracing-handoff/results/classic_montage_*/ 2>/dev/null | head -1')
if [[ -z "$REMOTE_RUN" ]]; then
  echo "ERROR: no formal Montage trace found" >&2
else
  LOCAL="results/$(basename "$REMOTE_RUN")"
  mkdir -p "$LOCAL"
  rsync -az --progress --checksum --partial \
    --exclude 'work/' --exclude 'bcc.out' \
    "$SSH_USER@$CLIENT_NODE:$REMOTE_RUN" "$LOCAL/"

  failed=0
  cells=0
  for cell in "$LOCAL"/*; do
    [[ -f "$cell/manifest.json" ]] || continue
    ((cells += 1))
    for required in \
      ebpf_events.log parsed.json manifest.json execution_units.jsonl \
      phase1_metrics.json parallelism_summary.json trace_quality.json \
      lineage/artifacts.csv lineage/execution_unit_io.csv \
      visualizations/file_access_volume.png visualizations/rw_asymmetry.png \
      visualizations/request_size_rw_cdf.png \
      visualizations/byte_normalized_summary.png visualizations/index.html; do
      if [[ ! -s "$cell/$required" ]]; then
        echo "ERROR: missing or empty: $cell/$required" >&2
        failed=1
      fi
    done
    if [[ ! -f "$cell/bcc.err" ]] || ! tail -1 "$cell/bcc.err" | grep -q 'lost_events=0'; then
      echo "ERROR: lost-event check failed: $cell/bcc.err" >&2
      failed=1
    fi
  done
  (( cells > 0 )) || failed=1
  if (( failed == 0 )); then
    open "$LOCAL"/*/visualizations/index.html
  else
    echo "Formal Montage pull failed integrity checks; not opening incomplete output." >&2
  fi
fi
```

`mosaic.png` 不在上述拉取清单中。命令打开的是本项目生成的 `visualizations/index.html` 和 I/O characterization 图。

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
