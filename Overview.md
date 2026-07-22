# Overview: Characterizing I/O in Agentic Scientific Workflows

# eBPF/BCC 追踪操作手册

> `$SSH_USER` / `$CLIENT_NODE` / key / base URL / model 都来自 `cloudlab_env.sh`（本地、git-ignored）；换节点改那里。

在 CloudLab 节点上跑 workflow 并用 eBPF/BCC 采集 I/O。三个已接入系统的运行依赖**不一样**，别混：


| 系统          | 配置文件                        | 远端 env 文件                         | key 变量             | 模型写法（FreeInference）                  | 走什么           |
| ----------- | --------------------------- | --------------------------------- | ------------------ | ------------------------------------ | ------------- |
| **SciLink** | `config/config_scilink.env` | `.env.scilink`                    | `OPENAI_API_KEY`   | `openai/qwen3.6-35b`（**要** `openai/` 前缀） | litellm       |
| **GenoMAS** | `config/config_genomas.env` | `.env.genomas` + `~/GenoMAS/.env` | `OPENAI_API_KEY_1` | `qwen3.6-35b`（**裸名**，加前缀会 404）           | openai SDK 直连 |
| **1000genome classic** | `config/config_1000genome.env` | 可选 `.env.1000genome` | 不需要 | 不适用 | 本地 Python DAG，支持离线 |


前两个 agentic 系统的 Provider = FreeInference（OpenAI 兼容），base URL `https://freeinference.org/v1`，Bearer key 放 `OPENAI_API_KEY`。1000genome classic 不使用 provider 或 key。

⚠ **FreeInference 的 `/v1/models` 目录虚标**：列出的模型不都真部署了。实测(2026-07-15)`glm-5.1` / `glm-5-turbo` / `minimax-m3` / `minimax-m2.5` 都 **404**，只有 **`qwen3.6-35b`**（标了「no concurrency limit」，最适合 agentic 多次调用）和 **`deepseek-v4-flash`** 返回 200。换模型前先 `curl .../v1/chat/completions` 实打一次确认,别信目录。SciLink `polycrystalline_grains_basic` 已用 `qwen3.6-35b` 端到端验证通过。

🟥 换节点/首次才做一次 · 🟨 每开新终端一次 · 🟩 每个 run · 🔧 改了什么才做。以下命令除注明【节点】外都在 **Mac** 跑。

---



# 一、配环境



## 🟨 每开新终端【Mac】

```zsh
source cloudlab_env.sh
```

✅ 打印 `[cloudlab_env] keys OK` + 当前 `CLIENT=…`。

## 🔧 改了代码 → 推代码【Mac】

`.env*` 被**故意排除**（只在远端、装着 key），推代码不碰 key。

```zsh
rsync -az --delete \
  --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude 'results/' --exclude '.venv/' --exclude '.env*' \
  ./ "$SSH_USER@$CLIENT_NODE:pi-ebpf-tracing-handoff/"
```



## 🔧 改了 Provider / key / model → 推 env【Mac】

rsync 不碰远端 `.env.*`，换 provider 必须单独推。`.env.*` 从上往下 source、**末尾行覆盖前面**，所以追加一个 override 块即可 —— **秒级、不重建 venv**。先在 `cloudlab_env.sh` 填好 FreeInference 块再 `source`。

```zsh
source cloudlab_env.sh
# SciLink：一个文件，模型带 openai/ 前缀（$SCILINK_MODEL 应为 openai/qwen3.6-35b）
ssh "$SSH_USER@$CLIENT_NODE" "cat >> pi-ebpf-tracing-handoff/.env.scilink" <<EOF

export OPENAI_API_KEY="$OPENAI_API_KEY"
export OPENAI_BASE_URL="$OPENAI_BASE_URL"
export OPENAI_API_BASE="$OPENAI_API_BASE"
export SCILINK_MODEL="$SCILINK_MODEL"
EOF

# GenoMAS：两个文件，key 变量带 _1，模型裸名（$GENOMAS_MODEL 应为 qwen3.6-35b）
ssh "$SSH_USER@$CLIENT_NODE" "cat >> pi-ebpf-tracing-handoff/.env.genomas" <<EOF

export OPENAI_API_KEY_1="$OPENAI_API_KEY"
export OPENAI_BASE_URL="$OPENAI_BASE_URL"
export OPENAI_API_BASE="$OPENAI_API_BASE"
export GENOMAS_MODEL="$GENOMAS_MODEL"
EOF
ssh "$SSH_USER@$CLIENT_NODE" "cat >> GenoMAS/.env" <<EOF

OPENAI_API_KEY_1=$OPENAI_API_KEY
OPENAI_BASE_URL=$OPENAI_BASE_URL
OPENAI_API_BASE=$OPENAI_API_BASE
EOF
```

✅ 确认：`ssh "$SSH_USER@$CLIENT_NODE" "tail -6 pi-ebpf-tracing-handoff/.env.scilink"` 里有 FreeInference base URL + 带 `openai/` 的模型。

## 🟥 首次 / 换节点 → 全量部署【Mac，慢】

```zsh
bash scripts/deploy_scilink_to_client.sh    # 或 deploy_genomas_to_client.sh
```

⚠ 只想换 key/model 时**别**跑这个 —— 它 `uv venv --clear` 会重建整个 venv（分钟级）。换 provider 用上面的「推 env」。

---



# 二、跑一个 run（🟩）

命令都用 `nohup … >log 2>&1 </dev/null &`：ssh **立刻返回**就能断线，任务在节点后台跑。用 `RUN_WORKLOADS` 选子集（逗号分隔），留空 = 全部。

## GenoMAS【Mac】

矩阵 12 格：`mw{1,2,4,8}_rep{1,2,3}`（max-workers × rep）。

```zsh
ssh "$SSH_USER@$CLIENT_NODE" \
  "cd pi-ebpf-tracing-handoff && sudo -E RUN_WORKLOADS='mw1_rep1,mw4_rep1' \
     nohup bash scripts/trace_script_bcc_genomas.sh > ~/genomas_run.log 2>&1 < /dev/null &"
```



## SciLink【Mac】


| workload                       | 类型      | 内容                                    |
| ------------------------------ | ------- | ------------------------------------- |
| `eels_plasmons_basic`          | analyze | EELS 等离激元 mapping                     |
| `eels_identification_basic`    | analyze | 1D EELS 谱识别                           |
| `polycrystalline_grains_basic` | analyze | 2D 晶粒分割                               |
| `planning_critical_materials`  | plan    | 需 embedding；FreeInference 有 `bge-m3`，把 `SCILINK_EMBEDDING_MODEL` 设成 `openai/bge-m3`（未实测） |


```zsh
ssh "$SSH_USER@$CLIENT_NODE" \
  "cd pi-ebpf-tracing-handoff && sudo -E RUN_WORKLOADS='polycrystalline_grains_basic' \
     nohup bash scripts/trace_script_bcc_scilink.sh > ~/scilink_run.log 2>&1 < /dev/null &"
```

✅ ssh 秒回、拿回提示符 = 已脱离终端，可断开 ssh。

## 1000genome classic baseline

该路径不使用 Pegasus、HTCondor 或 LLM。`run_1000genome.py` 直接执行
`individuals → individuals_merge` 与并行的 `sifting` 分支，然后执行
`mutation_overlap` / `frequency`。每个 task 有独立 sandbox，整个 DAG 共享一个
全局 worker 上限。

默认矩阵为 1、2、4 个 chromosome，各重复 3 次：

```text
classic_chr1_r1  classic_chr1_r2  classic_chr1_r3
classic_chr2_r1  classic_chr2_r2  classic_chr2_r3
classic_chr4_r1  classic_chr4_r2  classic_chr4_r3
```

默认固定 `INDIVIDUAL_JOBS=2`、`MAX_WORKERS=4`、`POPULATIONS=ALL`。

### 首次联网准备【CloudLab client，只做一次】

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

2/4-chromosome cell 还分别需要 chr2，以及 chr2–chr4 的两类 VCF。输入必须是
已经解压的 `.vcf`，不能只保留 `.vcf.gz`。

### 跑一个最小 trace【Mac 发起】

以下命令只跑 `classic_chr1_r1`。路径变量在远端普通用户 shell 中展开后再传给
`sudo`。workflow checkout、VCF 和 venv 都放在 Lustre，不能放进容量很小的
`$HOME`/root filesystem：

```zsh
ssh "$SSH_USER@$CLIENT_NODE" '
  cd "$HOME/pi-ebpf-tracing-handoff"
  REPO="/mnt/lustrefs/$USER/1000genome-workflow"
  sudo -E env \
    WORKFLOW_REPO="$REPO" \
    DATASET_DIR="$REPO/data/20130502" \
    POPULATION_DIR="$REPO/data/populations" \
    AGENT_PYTHON="$REPO/.venv/bin/python" \
    POST_PYTHON="$REPO/.venv/bin/python" \
    CLASSIC_OFFLINE=1 \
    RUN_WORKLOADS=classic_chr1_r1 \
    nohup bash scripts/trace_script_bcc_1000genome.sh \
      > "$HOME/classic_1000genome_run.log" 2>&1 < /dev/null &
'
```

跑完整 9-cell 矩阵时去掉 `RUN_WORKLOADS=classic_chr1_r1`，或设为空字符串。
也可以临时覆盖固定参数，例如：

```zsh
RUN_WORKLOADS=classic_chr1_r1 INDIVIDUAL_JOBS=2 MAX_WORKERS=4 POPULATIONS=ALL
```

### 完全离线运行

这里的“离线”指运行期间不访问公网、不调用 API；CloudLab 内部的 Lustre 挂载和
SSH 控制连接仍可使用。classic runner 本身没有下载或网络调用，且默认
`CLASSIC_OFFLINE=1`。该标记会同时写入 `manifest.json` 和
`work/classic_run_summary.json`，便于之后审计。

在隔离节点执行前，从可联网机器一次性传入以下内容：

- `1000genome-workflow` checkout，包括 `bin/`、`data/populations/`、
  `columns.txt` 和所需的全部解压 VCF；
- 可直接使用的 Python 3.10+ 环境，或者包含 `numpy`、`matplotlib`、`pillow`、
  `pandas`、`plotly` 及其依赖的本地 wheelhouse；
- 系统级 BCC 包和与当前 kernel 匹配的 headers。BCC 不能由普通 Python
  wheelhouse 替代。

如果使用 wheelhouse，在离线节点安装时禁止访问索引：

```bash
python3.10 -m venv "$WORKFLOW_REPO/.venv"
"$WORKFLOW_REPO/.venv/bin/pip" install \
  --no-index --find-links /path/to/wheelhouse \
  numpy matplotlib pillow pandas plotly
```

确认输入和依赖已落盘后，执行上一节的命令即可；不需要任何 API key 或
`.env.genomas` / `.env.scilink`。trace 脚本会在启动 tracer 前检查 repo、Python、
`columns.txt`、population 文件和每个 chromosome 的两个 VCF，缺失时直接失败，
不会尝试联网补齐。

每个成功 cell 应至少生成：

```text
ebpf_events.log
parsed.json
artifact_sizes.json
manifest.json
work/classic_run_summary.json
visualizations/file_access_volume.png
visualizations/rw_asymmetry.png
```

classic run 不生成 LLM summary、lineage、parallelism 或 `viz.trace` dashboard。

## 看进度 / 判断结束【Mac】

```zsh
ssh "$SSH_USER@$CLIENT_NODE" "tail -f ~/scilink_run.log"           
ssh "$SSH_USER@$CLIENT_NODE" "pgrep -af trace_script_bcc_scilink"  
# classic：把日志换成 ~/classic_1000genome_run.log，进程名换成 trace_script_bcc_1000genome
# 空 = 已结束
# 实时（GenoMAS 换 genomas_run.log）
```

✅ Agentic 脚本日志出现 `All done. Results in: …`，或 classic 日志最后出现
`Results: …`，表示脚本已结束。Agentic 日志里若有 `401` / `LLM Provider NOT
provided` / `NotFoundError`，说明 provider 没配对，回「推 env」重来。

---



# 三、拉回结果（🟩【Mac】）

```zsh
RUN=$(ssh "$SSH_USER@$CLIENT_NODE" \
  "ls -1dt /mnt/lustrefs/$SSH_USER/pi-ebpf-tracing-handoff/results/*/ | head -1")
LOCAL="results/$(basename "$RUN")"
mkdir -p "$LOCAL"
rsync -az --progress --exclude 'work/' --exclude 'bcc.out' --exclude 'bcc.err' \
  "$SSH_USER@$CLIENT_NODE:$RUN" "$LOCAL/"
open "$LOCAL"/*/visualizations/index.html 2>/dev/null || open "$LOCAL"
```

✅ Agentic run 应出现 `visualizations/index.html`；classic run 没有 dashboard，
应检查 `visualizations/file_access_volume.png`、`visualizations/rw_asymmetry.png`
和 `artifact_sizes.json`。

---



## 1. Project Motivation

Traditional scientific workflows usually have relatively fixed DAGs, known task dependencies, and stable producer-consumer dataflows. Existing workflow I/O characterization studies therefore focus on how task structure, file reuse, access type, operation count, dataflow size, and bandwidth explain workflow-level I/O behavior.

Agentic scientific workflows are different, and not always in the way that "traditional workflow + LLM wrapper" would suggest. Many real agentic scientific workflows exist precisely because the underlying task has no clean traditional-workflow counterpart — e.g., harmonizing heterogeneous raw datasets that previously required manual, judgment-heavy preprocessing. For these systems there is no fixed DAG to compare against in the first place. An LLM agent may decide which tool to call, which files to inspect, whether to retry, how to debug failures, and how to configure downstream scientific tasks. As a result, the I/O behavior is no longer determined only by a fixed workflow DAG. It is also shaped by agent decisions.

This project studies the I/O behavior of real, deployed agentic scientific workflows, with the goal of understanding what is inherited from traditional scientific workflow I/O and what is newly introduced or reshaped by agentic execution — without requiring that a traditional counterpart exist for every system studied.

## 2. Core Research Question

What I/O patterns arise when LLM agents execute scientific workflows, and how do these patterns differ from traditional fixed-DAG scientific workflows?

More specifically:

- How much I/O comes from normal scientific task execution?
- How much I/O is introduced purely by agent behavior — exploration, debugging, retry, repeated reads — the kind of overhead that a differently-behaved agent, running the exact same task and configuration, would not produce?
- How much I/O comes from a demonstrably suboptimal configuration, regardless of whether that configuration was chosen by the agent's generated code or hard-coded into the workflow's own orchestration script?
- How predictable is the I/O footprint of the same scientific goal across repeated agent runs, and how much of that unpredictability traces back to agent behavior specifically?



## 3. Scope

This project focuses on filesystem I/O caused by agent actions and scientific task execution.

Out of scope:

- LLM model loading
- KV cache paging
- model offloading
- internal LLM serving storage behavior

Those topics belong more to LLM serving and inference systems. In this project, the LLM is treated as the controller of the workflow, not as the storage workload being characterized.

## 3.1 Result Storage Standard

All durable local results for this repository should live under:

```text
results/
```

Per-run trace directories should live under:

```text
results/<run_id>/<workload>/
```

where `<run_id>` is usually a timestamp or named campaign, and `<workload>` is
the traced cell/use case. A trace cell should keep the complete bundle in one
place: `ebpf_events.log`, `parsed.json`, `pi_events.jsonl`, `tool_calls.log`,
`phase1_metrics.json`, `lineage/`, `visualizations/`, and any workflow-specific
session/output directory such as `scilink_session/` or `work/`.

`remote_results/` is not a durable result location. It may be used only as a
temporary pull/staging cache while transferring data from CloudLab or another
remote machine. After validation, anything worth keeping must be moved into
`results/`; then the staging copy should be removed.

Remote machines should expose the same logical entry point on Lustre:

```text
/mnt/lustrefs/<user>/pi-ebpf-tracing-handoff/results/
```

CloudLab clients must not write new trace output to the repo checkout, home
directory, or root filesystem. New tracing scripts should default `BASE_OUT` to
`/mnt/lustrefs/<user>/pi-ebpf-tracing-handoff/results/<run_id>`. Pulled local
copies should land under the repo-relative `results/` path with the same
`results/<run_id>/<workload>/` shape.

## 4. Target Systems and the Orchestration-Fixedness Spectrum

This project does not study "agentic workflows" as a monolithic category. Real open-source agentic scientific workflows differ substantially in how much of their execution path is fixed versus decided at runtime by the LLM. This spectrum is itself a useful axis for comparison, in place of a forced traditional-vs-agentic baseline.

Confirmed from direct inspection of the target repositories:

- **GenoMAS** (gene expression analysis, GEO/TCGA preprocessing + regression): the task sequence is fixed at the prompt level. Each agent role (GEO, TCGA, Statistician) is driven by an ordered list of "Action Units" defined in static JSON files (`prompts/action_units/base/*.json`) — e.g., the GEO agent always proceeds through Initial Data Loading → Dataset Analysis and Clinical Feature Extraction → Gene Data Extraction → Gene Identifier Review → Gene Annotation → Gene Identifier Mapping → Data Normalization and Linking, in that order, every run. What the agent decides is the *content* of the code written for each fixed stage, not which stages run or in what order. The outer orchestration loop (`environment.py`) — iterating over trait/condition pairs, cohorts, checkpointing, directory creation — is ordinary deterministic Python with no LLM involvement at all.
- **SRAgent** (NCBI/SRA metadata agent) and **ChemGraph** (XANES simulation agent): both use a LangGraph ReAct-style supervisor pattern (`create_react_agent` / `StateGraph`). The supervisor agent decides at each turn which sub-agent or tool to invoke next based on the running message history; there is no fixed stage list. SRAgent's own system prompt explicitly instructs it to "try multiple approaches if the first attempt fails" and to track what has and hasn't been tried — the number of tool calls and their order is a runtime decision, not a configuration.
- **SciLink** (microscopy/materials characterization agent): exposes an explicit `--mode autonomous` alongside components named `best_of_n_orchestrator`, `refinement_loop`, and `multiskill_autoselect` — i.e., both which analysis skill to run and how many refinement iterations to perform are runtime, agent-driven decisions.
- **CMBAgent** (github.com/CMBAgents/cmbagent; general-purpose multi-agent scientific research system, built on AG2/AutoGen, originally cosmology-focused but domain-agnostic): sits in the **middle** of the spectrum, and is a useful new data point precisely because it makes the "plan, then execute the plan" structure explicit as two separate phases rather than blending them. In `planning_and_control` mode, a `planner` agent and `plan_reviewer` agent iterate (bounded by `max_n_attempts`) until a concrete, ordered step list is agreed on — this planning phase is itself fully LLM-driven, so the *shape* of the plan (how many steps, which agent handles each) is not fixed across runs the way GenoMAS's action-unit list is. Once a plan exists, a `control`/`controller` agent executes it step-by-step, handing each step to the `engineer` (writes and runs code via AG2's `LocalCommandLineCodeExecutor`) or a domain `researcher`/specialized agent. So within one run, control-phase execution follows a plan that was itself dynamically generated — neither a static fixed-DAG (GenoMAS) nor a pure per-turn ReAct loop with no separate planning artifact (SRAgent/ChemGraph). CMBAgent also exposes a simpler `one_shot` mode (single task, no planning phase, default `agent='engineer'`) that is closer to a single-tool-call ReAct turn and is the cheapest entry point for a first integration.

GenoMAS sits at the fixed-DAG end of this spectrum; SRAgent, ChemGraph, and SciLink (in autonomous mode) sit at the dynamically-orchestrated end; CMBAgent's `planning_and_control` mode sits in between (dynamic plan generation, then comparatively fixed execution of that generated plan). The project should report where each studied system falls on this spectrum, and treat "how fixed is the orchestration" as an independent variable rather than assuming all agentic workflows are equally dynamic.

## 5. I/O Attribution Categories

Every unit of observed I/O is assigned to exactly one of three categories. The categories are defined so that assignment is based on evidence about *what specifically caused this I/O*, not on a global notion of an ideal or minimal I/O footprint — no external baseline or "necessary floor" is required to apply this scheme.

### 5.1 Agent-Induced I/O

I/O that exists purely because of the agent's behavior on this run, and that a differently-behaved agent — given the identical task and configuration — would not have produced. This is a behavioral category, not a configuration category.

Examples: repeated debug/retry cycles after a failed code execution, redundant re-reads of a file the agent already read earlier in the same task, reading error logs, inspecting intermediate outputs multiple times, abandoned artifacts from a discarded code attempt.

A large volume of agent-induced I/O is not a flaw in the categorization scheme — it is itself one of the project's central findings. It is direct evidence that either (a) current agents are not yet reliable enough to solve the task in a small, direct number of steps, or (b) the filesystem/tooling interface an agent is given does not match how agents actually search, verify, and recover from errors (e.g., no cheap way to check "have I already read this file" without re-opening it). Both are worth reporting explicitly rather than normalizing away.

### 5.2 Task-Misconfigured I/O

I/O that is more expensive than it needs to be because of a demonstrably suboptimal configuration choice — for the *same* task semantics, a better configuration exists. The detection criterion is deliberately source-agnostic: it asks only "is there a better configuration for this exact task," not "who chose it."

Once an instance is identified, it is additionally tagged along a second, source-specific axis:

- **agent-caused**: the suboptimal choice was made in code the agent generated for a given run (e.g., choosing a POSIX file-per-call read pattern where the provided library already offers a batched alternative; re-parsing a large raw file from scratch each time instead of caching the parsed result within the same code attempt).
- **script-caused**: the suboptimal choice is baked into the workflow's own fixed orchestration or tool code, and is present on every run regardless of what the agent does (e.g., GenoMAS's `environment.py` calling `os.listdir()` on every cohort-loop iteration instead of caching the directory listing once; `validate_and_save_cohort_info` in `tools/preprocess.py` performing a full JSON read-modify-write under an `fcntl` lock on every single cohort completion).

This source tag is what makes the category agent-relevant despite the source-agnostic detection rule: comparing the agent-caused rate to the script-caused rate answers "does agentic code generation introduce misconfiguration at a higher rate than a human-written script would have," which is a genuinely agentic research question even though the underlying bug-finding criterion does not care who wrote the code.

### 5.3 Workflow Task-Induced I/O (Residual)

Everything left over after 5.1 and 5.2 are subtracted out. This is I/O that is neither attributable to agent exploratory behavior nor to an identifiable suboptimal configuration — e.g., reading a required input dataset, writing the final output file. It is defined by exclusion, not by computing an absolute minimum or "necessary floor" for the task; no such floor needs to be constructed for this framework to work.

## 6. Metrics to Characterize



### 6.1 Universal I/O Metrics

Collected for all I/O, then aggregated by the three categories above.

- read bytes / write bytes
- read / write operation count
- metadata operation count
- unique files touched
- small-file access count, small-I/O count
- I/O time, effective bandwidth, and duty cycle. Duty cycle is
`|union(read/write syscall intervals)| / group wall-clock time`, where group
wall time comes from the shared run timeline: run wall for global metrics,
tool-call interval unions for phase/role metrics, and LLM interval union vs
run-wall remainder for inference-busy/idle metrics.
- read/write ratio



### 6.2 Agent-Induced I/O Metrics

- directory scan count, failed open/stat count
- same-file reread count, same-version reread count
- error-log read count, output-inspection count
- retry-induced I/O bytes and operations
- temporary file count, abandoned artifact count
- redundant-read fraction, non-productive I/O fraction



### 6.3 Task-Misconfigured I/O Metrics

These target systems run on real HPC clusters against parallel/shared filesystems (Lustre on CloudLab/DARWIN/RCCS) — this is squarely HPC I/O territory, not an optional extra. All of these metrics are in scope for every target system by default; the per-system work is confirming which storage tier (local scratch vs Lustre) each workflow's I/O actually lands on, not deciding whether HPC-style metrics apply at all.

- I/O interface used (POSIX / batched-library-call / parallel-IO where relevant)
- output file count, average output file size
- checkpoint/metadata-write frequency
- storage location choice (scratch vs shared/parallel filesystem)
- rank-level I/O size/time imbalance (where the task launches multiple workers/ranks, e.g. GenoMAS `--parallel-mode cohorts`, ChemGraph ensemble/FDMNES runs)
- agent-caused vs script-caused split (see 5.2)



### 6.4 Run-to-Run Variance Metrics

The purpose is to quantify how predictable the I/O footprint is when the same scientific goal is executed repeatedly by an agent, and to determine how much of that variance is attributable to agent-induced I/O specifically (5.1) versus task-misconfigured or residual I/O (5.2, 5.3), which should be comparatively stable across runs of the same fixed configuration.

Metrics: total I/O bytes/operations variance, metadata operation variance, unique files touched variance, agent-induced I/O variance, task-misconfigured I/O variance, residual I/O variance, I/O time variance, runtime variance.

## 7. Research Pipeline (Three Phases)



### Phase 1 — Comprehensive metric collection

Find as complete a metric set as possible (Section 6), run it against a real target system (e.g., GenoMAS first), and collect the resulting telemetry. Start with a small-scale run (e.g., GenoMAS `--quick-test` on 1-2 traits/cohorts) to validate that every metric in Section 6 is actually extractable from the current eBPF/bcc tracing infrastructure before committing to full-scale runs — GenoMAS full runs cost 3-5 days and $300+, so the metric-extraction pipeline must be validated cheaply first. Scale up, and extend to additional target systems, only after this validation.

### Phase 2 — Provenance-based attribution

Combine the raw I/O trace with execution provenance (tool-call/action-unit time windows) to assign every abnormal/flagged I/O result to exactly one of the three categories from Section 5: **agent-induced**, **task-misconfigured**, **workflow task-induced (residual)**. The specific mechanism for correlating trace timestamps with agent turn/tool-call boundaries (e.g., using each workflow's own execution logs as ground truth, or the existing time-window matching in this repo's tracing code) is an implementation detail of the tracing pipeline, not a conceptual commitment of this document.

### Phase 3 — Mitigation case studies

Select a small number of the clearest, highest-confidence findings — prioritizing task-misconfigured (script-caused) instances, since these are deterministic code paths where a single fix applies on every future run and the before/after comparison is clean without needing to average over agent run-to-run noise — and show the fix and its I/O impact.

## 8. Key Comparisons



### 8.1 Scripted Workflow vs. Agentic Workflow (optional, where a counterpart exists)

Where a traditional scripted counterpart genuinely exists for the same scientific goal, compare it against the agentic execution to identify extra I/O introduced by agentic orchestration. This comparison is not required for systems (like GenoMAS) where no traditional counterpart exists; for those, use the high-level qualitative framing instead — fixed DAG / stable producer-consumer edges / stable I/O pattern (traditional) vs. dynamic execution path / heavy file exploration / high metadata volume / high run-to-run variance (agentic).

### 8.2 Repeated Agent Runs of the Same Goal

Run the same prompt multiple times to measure I/O predictability and determine whether variation comes from agent-induced behavior (5.1) or from the underlying task/configuration (5.2, 5.3).

### 8.3 Local Counterfactual Comparison for Misconfiguration

For a task-misconfigured instance (5.2), the comparison needed is local, not a global "recommended configuration" baseline: show that applying the identified fix to that specific instance reduces I/O for the same task semantics. This can be a before/after patch comparison (for script-caused instances) or a comparison across repeated agent runs where some runs happened to avoid the suboptimal choice and others didn't (for agent-caused instances).

## 9. Expected Findings

- A non-trivial share of total I/O in real agentic scientific workflows is agent-induced (5.1) rather than task-induced — and the size of this share is itself evidence about current agent reliability and about how well existing filesystem/tool interfaces suit agentic access patterns, not merely overhead to be subtracted out.
- Agent-induced I/O is more variable across repeated runs than task-misconfigured or residual I/O, which should be comparatively stable given a fixed configuration.
- Task-misconfigured I/O occurs in both agent-generated code and in workflows' own fixed orchestration/tooling scripts; comparing the two rates indicates whether agentic code generation is a net-additional source of configuration error beyond what already exists in hand-written scientific workflow code.
- Systems with more dynamically-orchestrated execution (SRAgent, ChemGraph, SciLink-autonomous) are expected to show higher run-to-run I/O variance than systems with a fixed action-sequence (GenoMAS), independent of task-misconfiguration.



## 10. Main Contribution

The main contribution is not a new I/O metric by itself. It is:

1. An I/O characterization of real, deployed agentic scientific workflows (GenoMAS, SRAgent, ChemGraph, SciLink, CMBAgent) spanning a range of orchestration-fixedness, without forcing a traditional-workflow baseline where none exists.
2. A causal, provenance-based three-way I/O attribution scheme (agent-induced / task-misconfigured / residual task-induced) that separates behavioral overhead from configuration error from unavoidable task I/O, using source-agnostic detection criteria plus an agent-caused/script-caused sub-tag for configuration errors.
3. A small number of concrete mitigation case studies, demonstrating measurable I/O reduction from fixing identified misconfigurations.

This connects traditional I/O characterization with the new execution behavior of LLM-based scientific agents.

## 11. Positioning Against Prior Work

Traditional HPC I/O characterization studies analyze access patterns, file reuse, sharing, read/write behavior, bandwidth, operation counts, and variability in large-scale systems.

Workflow-centric I/O characterization studies connect I/O behavior to workflow DAGs, stages, and producer-consumer relationships.

This project builds on both directions, but focuses on a new setting:

- the workflow execution path may be dynamically generated by an agent, to a degree that varies by system (Section 4) — some agentic scientific workflows have no fixed DAG at all
- the same scientific goal may produce different I/O footprints across runs
- the agent may introduce exploration, debugging, retry, and redundant reads that a non-agentic execution of the same task would not
- both agent-generated code and the workflow's own hand-written orchestration can independently misconfigure I/O, and comparing the two rates isolates what is specifically attributable to agentic execution

Thus, the project studies I/O behavior under agentic execution rather than under fixed workflow execution alone.
