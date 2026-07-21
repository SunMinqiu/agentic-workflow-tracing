# 计划 3：图表清理与指标校正

前置：`Plans/Index-Rework-Plan.md` 和 `Plans/VFS-Offset-and-Fixes-Plan.md` 已实现。本计划处理跑通新代码后暴露的问题：图面遮挡、图表重复、指标不准、口径无依据。

**贯穿全篇的一条硬规则：画布里只有数据。** 所有标量、口径说明、公式一律进 HTML 图注（`figure_card` 的 caption 参数，目前几乎没被用），不进 matplotlib 画布。标题只回答"这是什么"，不回答"怎么算的"。

P1–P6 全部离线，改完在现有 trace 上重跑分析即可看到效果。P7 需要重跑 trace。

---

## P1 · 图面统一清理

### 问题

两类，遍布所有图：

1. **标量塞进画布**：`ax.text(0.02, 0.97, ...)` / `ax.text(0.98, 0.04, ...)` 这类 axes 坐标的文本框，数据一旦靠上或靠右就必然被遮挡。已确认受害图：Directory Rescans（右上角框压住 `>=10` 那根柱）、File Access Frequency × Volume（左上角框压图）、Artifact Lifecycle（右上角框压图）、Effective BW by Phase（图内 `n=`/`duty` 标签与柱子重叠 + 右下角 global BW 框）。
2. **标题里塞公式和括号**：`"Effective bandwidth by phase (duty = I/O-busy wall-clock union / phase wall-clock union)"`、`"Per-syscall I/O request size (informs block size / read-ahead)"`、`(Darshan bins)` 之类。读者不看括号，只觉得挤。

### 改法

| 位置 | 动作 |
| --- | --- |
| 所有 `ax.text(...)` 统计摘要框 | 删除，内容移到 `figure_card(caption=...)` |
| 所有标题里的括号口径 | 删除；标题只留名词短语 |
| 图例 | `bbox_to_anchor` 移出绘图区，不再压数据 |
| `Effective BW by Phase` 的 `n=` | 图注写成人话：「n = 该 phase 的读写 syscall 数」 |
| `Access Pattern` 的 `offset coverage X%` | 进图注 |
| `File Size Distribution` 的分箱说明 | 进图注（"分箱沿用 Darshan 标准 10 桶"），标题不出现 `(Darshan bins)` |

涉及文件：`viz/_trace_impl.py`（`create_directory_scan_matplotlib`、`create_effective_bandwidth_matplotlib`、`create_access_pattern_matplotlib`、`create_io_rate_matplotlib`）、`analysis/per_run_io_char.py`（`plot_file_access_volume`、`plot_rw_asymmetry`）、`lineage/_analyzer_impl.py`（`fig_size_distribution`、`fig_lifecycle_spans`、`fig_reader_fanout`）。

---

## P2 · 删除重复图表

### 删

| 图/表 | 位置 | 理由 |
| --- | --- | --- |
| **Reader × Writer Heatmap** | `lineage/_analyzer_impl.py:1387` `fig_reader_writer_heatmap` + 调用点 + index card | 11×11 网格绝大多数格子是空的，唯一有价值的信息是 P-C 四分类。**改为四个数**（见下） |
| **Who Does the I/O** | `lineage/_analyzer_impl.py:1210` `fig_role_io_attribution` + 调用点 + index card | 与 `call_dag` 重复且更粗：DAG 节点已带 per-tool-call 读写字节（`io_by_tool`），role 挂在 tool-call 上。去掉「按 file category 分段上色」（我们已不关心 raw/intermediate）之后，只剩「每 role 读写多少字节」，而 `effective_bandwidth.by_role` 和 `bytes_ops_by_phase` 里都有 |
| **Agent Activity Timeline** | `viz/_trace_impl.py` `create_agent_concurrency_matplotlib` / `_plotly` + `AGENT_VISUALIZATIONS["agent_concurrency"]` + index card | 与 Agent Timeline 重复。独家信息只有「跨 agent 并行结构」，而 Agent Timeline 的三泳道甘特已能表达调用结构 |
| **Detail 三图**：`timeline` / `tool_syscalls` / `tool_syscall_durations` | `viz/_trace_impl.py` 的 `create_timeline_*`、`create_tool_syscalls_plotly`、`create_tool_syscall_durations_plotly` + `STRACE_VISUALIZATIONS` 注册 + index 的 `detail_links` | 全是调试期产物，没有任何 finding 依赖。`tool_syscalls` 最多画 100 个子图，很慢。`tool_syscall_durations` 测的是返回延迟（含 page cache），该结论已由 effective bandwidth + duty cycle 覆盖 |
| **Intensity phases 表** | index `intensity_table` panel | 单位是「bin 个数」，读者无从解读；信息已被 io_rate 折线完全覆盖。`compute_intensity_phases` 的计算保留在 JSON，只是不上 index |
| **File size 表** | index `fsz_table` panel | 与 File Size Distribution 图下半 panel 重复 |
| **Measured interface layers 表** + **Measured STDIO/POSIX byte mix 表** | index 两个 panel | 与 Measured I/O Interface Mix 图同源同数据。三者合成**一张图**（各层 ops + bytes） |

### 保留

- **RH/WH/RW 表**：依据与 Patel FAST'20 图 3a 完全一致（按 `read_share = rb/(rb+wb)` 三分，RH ≥ 2/3、WH ≤ 1/3、RW 居中；他们报 22% / 7% / 71%，我们逐数字可比）。这是跨论文可比的对位数字，值得单独留表。
- **I/O Autocorrelation**：原样不动。当前 run 只有 16 分钟，5/25min 窗 bin 太少、暂无统计意义，但后续会有更长的 run。

### 新增（替代 heatmap）

Reader & Writer Fan-out 图的**图注**里给出 **P-C 四分类**四个数：`1-1` / `1-n` / `n-1` / `n-n` 各多少文件、占比多少。数据已在 `io_summary.json` 的 `reader_writer_joint` 里，只需把 `(w, r)` 计数折叠成四格。这保住了 Tang IPDPS'26 表 I 的对位（并按 `axis_metric_mapping.md` §0.4 的限定：我们不预设 DAG，只报经验分布），同时不再占一张图。

---

## P3 · 修准确性：三个指标没有 workload 过滤

`compute_interface_byte_mix(parsed)`、`compute_measured_interface_layers(parsed)`、`compute_io_autocorrelation(parsed)` —— **三个函数的签名里根本没有 `artifacts`，也就是完全没有 workload 过滤**（`grep` 可验：`build_metrics` 里三处调用都只传 `parsed`）。

后果：Interface Mix 现在把 `.venv` 的 import 读、我们自己 tracer 写 `pi_events.jsonl` / `tool_calls.log` 的 write 全算进去了 → **STDIO / POSIX 的比例是整个进程树的，不是 workload 的，现在的数不准**。Autocorrelation 的时间序列同理，混着解释器和 tracer 的 I/O。

**改法**：三个函数加 `artifacts` 参数 + `make_workload_filter`，与其余所有指标口径统一（这是既定原则：artifacts.csv 是 workload 的唯一真源）。

**顺带**：`mmap` 层的「字节」是**映射区长度，是上界不是实际读量**（页访问不产生 syscall）。这条 caveat 目前只在 docstring 里，要写进 interface 图的图注。

**VFS 之后要不要换接口层的抓法？不用。** VFS kprobe 只提供 offset，不改变层次归属；接口层判定靠 uprobe(`fread`/`fwrite`) 与 syscall(`read`/`write`) 的两层观测，思路正确且已做 de-overlap（`posix_direct = max(kernel_bytes − stdio_bytes, 0)`）。

---

## P4 · I/O Batching Efficiency 换指标

### 现在为什么没有现实意义

`compute_analytical_optimum` 的 `optimum = ceil(总字节 / 4MB)`，隐含假设「这个 run 的所有字节可以攒成一条流、按 4MB 一发地发出去」。**现实里做不到**：字节分散在不同文件（跨文件不能合并）、格式解析器要按记录/行读、流式处理不允许全缓冲。所以「读侧 44×」**不是「可以省 44 倍」**，只是「离一个物理上不可达的理想有多远」。

**文献里没有人这么报**。Darshan / Bez HPDC'22 / Patel 报的是 request size 的**分布**、**小 I/O 占比**（`pct < 4KB` 等）、seq/consec 比例。谈「合并能省多少」的是 mitigation 类工作（Nawaz 的 bulk transfer 合并小文件传输），且是**针对具体可合并对象**估算，不是一个全局除法。

### 换成什么

**基于 offset 的可合并性**（有了 VFS offset 才能做）：在同一 `(tid, fd, open_generation)` 流内，把**物理连续（consecutive）**的相邻 op 按 ≤4MB 边界合并，出：

- `mergeable_ops / actual_ops`（合并后能少发多少次调用）
- `bytes_in_consecutive_runs / total_bytes`（多少字节处在可合并的连续段里）

这些 op **本来就首尾相接**，合并是真的可行（把 buffer 调大即可），所以这个数有物理依据，可以写进 paper，并直接呼应「agent 生成的代码用了默认小 buffer」这个论点。

同时把**文献标准量**标进 request-size 直方图（图注）：`pct_ops < 4KB`、`pct_ops < 64KB`、bytes-weighted 平均 request size。

---

## P5 · Access Pattern 改四格

现在画了四类（consecutive / sequential-gap / backward-random / append），是把 Darshan 的内部计数原样倒出来了，过度细分。Tang（*Characterizing Dataflow for I/O-Aware Scheduling in HPC Workflows*）的口径是 **seq / random 二分**（其结论正是「small random read/write 如今更普遍」）。

**改为四格**：**Seq R / Rand R / Seq W / Rand W**。

- `consecutive` + `sequential_gap` → **Seq**
- `backward_or_random` → **Rand**
- `append` → 归 **Seq W**（追加写在物理上就是连续写）

stride 直方图撤掉（过度细节）。offset 覆盖率进图注。

---

## P6 · Inter-arrival 转直方图 + Write→Read Gap 改分箱

**Inter-arrival CDF → 直方图**。它和 Directory Rescans 不是一回事，必须说清：inter-arrival 是**同一文件两次访问的时间间隔**（有时间维度，对位 Patel FAST'20 图 5a：他们均值 47 小时，我们 93.9% 在 1 秒内）；directory rescan 是**同一目录被 getdents 扫了几次**（纯计数）。

两图共用同一套 log 时间 bin，按 **page-cache 语义**命名：

| bin | 语义 |
| --- | --- |
| `<1s` | 必然 page-cache 命中，根本不必碰盘 |
| `1–30s` | 仍在 cache 内，但可能已 writeback |
| `30s–5min` | cache 命中不确定 |
| `5–30min` | 大概率已被驱逐 |
| `>30min` | 几乎必然回盘 |

**图注必须声明：这套边界是我们按 page-cache / writeback 语义定的，不是文献分箱**（区别于 size 轴用的 Darshan 标准 10 桶，那个是文献的）。

---

## P7 · LLM intensity（需要重跑 trace）

### 显示 bug（离线可修）

`I/O Rate Over Time` 右轴出现 `1.5` —— **那不是数据，是 matplotlib 自动给的坐标轴刻度**。intensity 本身是整数（`sum(1 for ls, le in llm_intervals if le > s and ls < e)` = 该 bin 内活跃的 LLM call 数）。修：整数刻度（`MaxNLocator(integer=True)`）+ 阶梯画法（`drawstyle="steps-post"`）。

### 采集升级（需重跑）

「活跃 call 数」是 0/1 方波，粒度太粗。launcher 其实**已经能看到 delta 事件**（[`_is_delta_event`](../src/agent_io_tracing/adapters/pi/launcher.py#L179)，区分 `text_delta` / `thinking_delta`），只是默认不落盘（要 `PI_LOG_DELTA_EVENTS=1`）。

改动：把 delta 事件的**时间戳 + 类型 + token 计数**写进 `pi_events.jsonl`，`io_rate` 的右轴换成 **output tokens/s** —— 真实的推理强度曲线，而不是方波，且与「存储为何空闲」的因果链更直接。

**不做 TTFT**（prefill/decode 拆分暂不需要）。

---

## P8 · 执行顺序与重跑命令

P1 → P2 → P3 → P4 → P5 → P6 全部离线；P7 的显示 bug 也是离线的，采集部分随下一次 trace 重跑（可与充值后的 SciLink 四个 workload 一起跑）。

### 离线重跑（现有 trace 直接重算）

`viz` 必须最后跑（它读前两步的产物）。SciLink 的 cell 必须带 lineage 前缀，否则 artifacts 为空、所有指标静默归零。

```bash
CELL=results/<run_id>/<workload>

python -m agent_io_tracing.lineage.analyzer        "$CELL"
python -m agent_io_tracing.analysis.phase1_metrics "$CELL"
python -m agent_io_tracing.viz.trace               "$CELL"
open "$CELL/visualizations/index.html"
```

批量：

```bash
for CELL in results/*/*/; do
  [ -f "$CELL/parsed.json" ] || continue
  python -m agent_io_tracing.lineage.analyzer        "$CELL" >/dev/null 2>&1
  python -m agent_io_tracing.analysis.phase1_metrics "$CELL" >/dev/null 2>&1
  python -m agent_io_tracing.viz.trace               "$CELL" >/dev/null 2>&1
  echo "done: $CELL"
done
python -m agent_io_tracing.analysis.per_run_io_char --results results
```

### 集群重跑（P7 采集部分）

```bash
source cloudlab_env.sh

# --exclude '.env*' 必须带：.env.scilink 只存在于远端，--delete 会把它删掉
rsync -az --delete \
  --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude 'results/' --exclude '.venv/' --exclude '.env*' \
  ./ "$SSH_USER@$CLIENT_NODE:pi-ebpf-tracing-handoff/"

ssh -t "$SSH_USER@$CLIENT_NODE" \
  "cd pi-ebpf-tracing-handoff && sudo -E RUN_WORKLOADS='eels_plasmons_basic,eels_identification_basic,polycrystalline_grains_basic,planning_critical_materials' bash scripts/trace_script_bcc_scilink.sh"

RUN=$(ssh "$SSH_USER@$CLIENT_NODE" \
  "ls -1dt /mnt/lustrefs/$SSH_USER/pi-ebpf-tracing-handoff/results/*/ | head -1")
LOCAL="results/$(basename "$RUN")"
mkdir -p "$LOCAL"
rsync -az --progress --exclude 'work/' --exclude 'bcc.out' --exclude 'bcc.err' \
  "$SSH_USER@$CLIENT_NODE:$RUN" "$LOCAL/"
open "$LOCAL"/*/visualizations/index.html
```

---

## P9 · 清理后的 index

```
Global
  fig0_io_volume_summary · Agent Timeline · Time Accounting · Call DAG
  File Access Frequency × Volume · Read/Write Asymmetry

Axis 1 · Context-Limited State Persistence
  图: Directory Rescans · Inter-arrival (hist) · Reread Attribution
      Reader & Writer Fan-out (图注含 P-C 四分类) · Write→Read Gap (hist)
  表: Access type RH/WH/RW · State file rewrite frequency

Axis 2 · Measured I/O Interface Layer
  图: Measured I/O Interface Mix (唯一，ops + bytes)
  表: —

Axis 3 · Exploratory Branching and Backtracking
  图: Artifact Lifecycle
  表: Failed open/stat · Error-log reads · Bytes/ops by phase

Axis 4 · Reasoning-Step / Data-Granularity Alignment
  图: File Size Distribution · Access Pattern (Seq/Rand × R/W)
  表: I/O Batching Efficiency (基于 offset 的可合并性) · Access pattern 摘要

Axis 5 · Uncoordinated Agent Concurrency
  图: I/O Rate Over Time (bytes/s + output tokens/s) · Effective BW by Phase
      I/O Autocorrelation
  表: Workflow concurrency · I/O concurrency

Data & Artifacts
  (Detail 区整个删除)
```
