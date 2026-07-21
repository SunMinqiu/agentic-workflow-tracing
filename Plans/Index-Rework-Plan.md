# Index 重构计划：从「验筛选」图表转向 I/O behavior characterization

## 0. 目标与判据

现在 index 上的图表分两代：一代是当初为了**验证过滤逻辑对不对、有没有混进不该统计的文件**而画的（逐文件列名的 fan-out / staleness / lifecycle、top-5 重扫目录），一代是真正的 characterization。前一代已经完成使命，现在关心的是**次数 / 整体统计量 / 分布**，不再关心具体是哪个文件、是 raw data 还是 intermediate data。

本次重构的判据，任何一项图表要留在 index 上，必须同时满足：

1. 它是**分布或计数**，不是逐文件的枚举；
2. 它要么和传统 scientific workflow 的 I/O 特征化文献**可直接并排比较**（同样的量、同样的分箱），要么它刻画的是 agent 特有、传统 workflow 不会出现的 pattern；
3. 它测得准——测不准的（如原 Axis 1 决策时机）不留在主线，即使概念上诱人。

有图的项目**不再配表**：图上能标的数字标在图上，不另开 panel。

---

## 1. Axis 1（Runtime Decision Timing）下架，五轴重编号

### 为什么

现有实现把「决策时刻」取为路径字符串首次出现在可观测 tool input / 生成代码 / assistant 文本的时间。planning-heavy 的系统（GenoMAS 的 action unit 列表、CMBAgent 的 planner、orchestrator 内部拼出来的路径）根本不会把路径写进可观测文本，于是决策时刻被系统性低估、lead time 被系统性压小。测出来的 `pct_unprefetchable_lt_1s` 高，主要反映的是「我们看不见决策时刻」，不是 agent 真的临阵决定。不能作为 finding。

### 怎么做

新建 `src/agent_io_tracing/legacy/axis1_decision_timing.py`，把 `compute_decision_access_lead_time`（[phase1_metrics.py:1435](../src/agent_io_tracing/analysis/phase1_metrics.py#L1435)）和 `create_decision_access_lead_time_matplotlib`（[_trace_impl.py:1753](../src/agent_io_tracing/viz/_trace_impl.py#L1753)）原样搬过去。**该文件顶部写一段注释说明下架原因，这是全项目唯一保留说明的地方。**

其余位置**全部直接删除，不留注释、不留 "removed" 字样、不留空 section**：

| 文件 | 删什么 |
| --- | --- |
| `analysis/phase1_metrics.py` | `compute_decision_access_lead_time` 函数体（搬走）；`build_metrics` 里 `"decision_access_lead_time": ...` 那行（:1708）；`write_markdown` 里 `Decision→access lead time` 那行（:1650） |
| `viz/_trace_impl.py` | `create_decision_access_lead_time_matplotlib`（搬走）；`AGENT_VISUALIZATIONS` 里的注册项（:4589）；index 里 `lead` / `lead_dist` / `lead_table` 的构造（:4012 起）；`ax1_figs`（:4316）；`ax1_tables`（:4352）；`<h2>Axis 1 · Runtime Decision Timing</h2>` 整个 section（:4528-4530） |
| `axis_metric_mapping.md` | §0.1 六轴表的轴1 行；§1 判决总表的轴1 行；§2 的「轴 1 · 决策粒度/时机」整节 |
| `axis_metric_mapping_sections_0_2_en.md` | 同上三处的英文对应 |

### 重编号

剩余五轴按下表重编，index 的 `<h2>` 和两个 md 文档同步：

| 新编号 | 轴名 | 原编号 |
| --- | --- | --- |
| Axis 1 | Context-Limited State Persistence（状态持久性） | 2 |
| Axis 2 | Measured I/O Interface Layer（接口抽象层） | 3 |
| Axis 3 | Exploratory Branching and Backtracking（分支回溯） | 4 |
| Axis 4 | Reasoning-Step / Data-Granularity Alignment（粒度对齐） | 5 |
| Axis 5 | Uncoordinated Agent Concurrency（I/O 并发） | 6 |

注意 `viz/fanout_plot.py` / `fanout_index.py` / `experiments/stage_geo_view.py` 里的 "Axis 1 — cohort scaling" / "Axis 2 — trait scaling" 指的是 **fanout 实验的自变量轴**，与六轴框架无关，**不要动**。

---

## 2. 删除清单（删干净，不留尾巴）

| 项 | 位置 | 理由 |
| --- | --- | --- |
| `Logical→physical amplification` panel **及其字段** | `_trace_impl.py:4137` 的 `logphys_table` + `:4363` 的 panel；`compute_analytical_optimum` 返回里的 `file_count_amplification` / `metadata_op_amplification` / `optimum_files` / `optimum_storage_metadata_ops` / `actual_generated_files` / `actual_storage_metadata_ops` | 分母是硬编码常数（optimum_files=1、optimum_metadata_ops=3），没有物理含义；名字对应的是 axis_metric_mapping 里「tool-call 层逻辑字节 / subprocess 层物理字节」，但算的完全不是那个东西 |
| Directory rescans 的空 panel | `_trace_impl.py:4356` `panel("Directory rescans", "", ds_note)` | 第二参数是空串，压根没有表，只是一行 caption，和同名图重复 |
| Request size 表 | `_trace_impl.py` 的 `req_table` panel | 和 size 直方图重复；p50 / p95 / pct<4KB 三个数改为标注在直方图上 |
| `fig6_reuse_pattern` | `_analyzer_impl.py:1607` 的 `fig_reuse_pattern` 调用 + 函数；index 的 lineage_card | 全局聚合类，`axis_metric_mapping.md` §4 自己判定「不作 finding」 |
| `intensity_phases` **图** | `_trace_impl.py` 的 `create_intensity_phases_matplotlib` + 注册项 + index 卡片 | 60s 分箱在 16 分钟的 run 上只有 16 个 bin，「4 个高强度段全是单 bin 突发」无统计意义；被新的 io_rate 图覆盖。**`compute_intensity_phases` 的数值保留**，段数/段长进 Axis 5 |

`fig0_io_volume_summary` **保留**在 index 图区。

---

## 3. 四张图改直方图（不再逐文件枚举）

统一原则：**x 轴是量，y 轴是文件数/目录数；不出现文件名；不按 category 上色；不出现灰块**。图上角标注关键标量。

| 图 | x 轴 | y 轴 | 图上标注 | 数据来源 |
| --- | --- | --- | --- | --- |
| Directory Rescans | 一个目录被 `getdents` 扫的次数，离散 bin，≥10 合并为一箱 | 目录数 | total scans / unique dirs / rescan ratio = total/unique / p95 | `phase1_metrics.json['directory_scan']` **需要改**：现在只出 top-5，要改成出**完整的 per-dir 计数直方图**（`scans_per_dir_hist`） |
| Reader & Writer Fan-out | fan-out k，离散 bin，≥10 合并 | 文件数 | mean / max / %(k≥2)，reader 与 writer 各一组 | `per_artifact[*]['reader_tool_ids']` / `['writer_tool_ids']` |
| Write→Read Staleness | 间隔秒数，log 分箱：<10ms / 10-100ms / 0.1-1s / 1-10s / 10-60s / 1-10min / >10min | read 事件数 | n / median / %(<1s) | 见 §5，重新定义 |
| Artifact Lifecycle | `dead_seconds / (run_end − t_create)`，0→1 分 10 箱 | 文件数 | 生成文件数 / write-once-leaf 占比 / 中位 dead 时长 | `per_artifact[*]['dead_seconds']`、`['t_create']`、`['lifecycle_class']` |

改的位置：`lineage/_analyzer_impl.py` 的 `fig_reader_fanout`（:1353）、`fig_staleness_cdf`（:1392）、`fig_lifecycle_spans`（:1435）三个函数整体重写；`viz/_trace_impl.py` 的 `create_directory_scan_matplotlib`（:2064）重写；`phase1_metrics.compute_directory_scan_count`（:592）加 per-dir 计数直方图输出，`top_rescanned` 字段删掉。

---

## 4. Writer fan-out 新增 + reader×writer 联合分布

`writer_tool_ids` 早已在采（[_analyzer_impl.py:807](../src/agent_io_tracing/lineage/_analyzer_impl.py#L807)），从未被使用。

1. `summarize_artifacts` 里把 reader / writer fan-out 一起 rollup 进 `io_summary.json`：`{mean, p50, p95, max, pct_ge_2, hist: {k: n_files}}` 各一份。
2. `fig2_fanout.png` 出两个 panel：左 reader、右 writer，同一 x 轴范围。
3. 新增 **reader×writer 二维计数热图**：x = writer 数，y = reader 数，格子里是文件数。

为什么值得做：传统 workflow 的 DAG 边保证一个文件基本只有**一个 producer**；agent 把文件当外置内存反复 read-modify-write（`cohort_info.json`、`*_state.json`、checkpoint `.tmp`），writer fan-out 会显著 >1。**writer fan-out ≥2 的文件占比**是 Axis 1（状态持久性）最硬的证据，同时这张热图就是 Tang IPDPS'26 的 P-C 四分类（1-1 / 1-n / n-1 / n-n）在**无静态 DAG 场景下的经验版本**——按 axis_metric_mapping §0.4 的限定，这样用是合法的（我们不预设 DAG，只报经验分布）。

---

## 5. Staleness 重新定义为「相邻配对」

### 现在为什么是错的

[_analyzer_impl.py:809-835](../src/agent_io_tracing/lineage/_analyzer_impl.py#L809-L835) 取 `first_write_ts` = 该路径**第一条** write syscall，`first_read_after_write_ts` = **之后第一条** read syscall。文件被增量写时这会给出荒谬的数：一个 run log 在 t=0 首次 append、agent 在 t=500s 读它（读的是刚 append 进去的内容），会被记成 staleness = 500 秒。

### 新定义

对**每一条 read 事件**，取该路径上**紧邻它之前的那条 write**，`gap = t_read − t_last_write_before`。没有前置 write 的 read（纯输入文件）不入账。得到的是**以 read 事件为单位的 gap 分布**，不是每文件一个数。

好处：read-modify-write 的状态文件会自然打出一堆亚秒级 gap，正是「拿文件系统当内存用」的指纹；agent 写完立刻回读自查的行为不会被增量写污染；`pct(gap < 1s)` 直接对应存储侧结论——**这部分读本可以完全在 page cache 里解决，根本不必碰盘**。

原 `first_write_ts` / `first_read_after_write_ts` / `staleness_s` 三个字段及其在 `artifacts.csv`、`io_summary.json`、index 的 `staleness_summary` / `staleness_title` 里的所有痕迹**全部删掉**，换成新字段 `write_read_gap_s`（事件级列表 + 直方图）。

### 必须声明的 caveat

- 写是 buffered 的：Python 侧 `f.write()` 不等于 write syscall，我们看到的写时刻实际是 libc 缓冲区 flush（小文件通常就是 `close()`）。
- 读命中 page cache 时 read syscall 照常发生，所以测的是**逻辑上的写→读间隔**，不是介质上的。

---

## 6. io_rate 重做：bandwidth + inference 双折线

现在的 `create_io_rate_matplotlib`（[_trace_impl.py:829](../src/agent_io_tracing/viz/_trace_impl.py#L829)）画的是 **syscalls/100ms**，是调用计数，会被大量 4KB 小读拉高，误导。

新版单张图：

- **左 y 轴**：wall-clock **bytes/s**，read 和 write 两条折线（1s 或 2s 分箱，随 run 长度自适应）。
- **右 y 轴**：**inference intensity** 折线 = 每个 bin 内同时活跃的 LLM 调用数（单 agent 是 0/1 方波；GenoMAS fanout 多 worker 时是 0..N 的强度曲线）。数据来自 `parallelism._load_llm_events` 已解析的 LLM `start_ms` / `end_ms`。
- 不画甘特带（`agent_timeline` / `agent_concurrency` 已经是甘特图，不重复）。

新增标量 `compute_io_vs_inference`（进 `phase1_metrics.json`）：`pct_time_in_inference`、`pct_bytes_during_inference`、`bandwidth_ratio`（推理期间 bytes/s ÷ 非推理期间 bytes/s）、inference gap 时长分布。

**要说的话**：存储带宽是被 LLM 推理斩断的锯齿波，**idle 段长度由模型延迟决定、与数据规模无关**。传统 workflow 的 I/O 段长由数据量和 stage 边界决定（Patel SC'19 图7：读阶段均值 6.62 分钟），这是形状上的根本差异，也是给存储系统的直接结论——预取 / burst-buffer flush / 资源共享的机会窗口全在这些 inference gap 里。

---

## 7. Effective bandwidth 与 duty cycle：按 phase 分解（不是全局标量）

`grep` 确认：`effective_bandwidth` / `io_time` 目前在 codebase 里**一个都没有**，尽管 `Overview.md` §6.1 把它列为必测项。

### 为什么必须分解

`axis_metric_mapping.md` §0.3 已经把这条写成硬约束：「末端高吞吐掩盖上游瓶颈」陷阱（Tang IPDPS'26 Finding 13 + Raj CPU-Centric 的交叉验证）要求**按 phase 逐段分解，而非只报末端聚合吞吐**。压成一个全局标量就正好踩进这个陷阱。

### 分组

沿用现成的 [phase_for_entry](../src/agent_io_tracing/analysis/phase1_metrics.py#L359)（`compute_bytes_ops_by_phase` 已在用），三级粒度：

- **phase**（GenoMAS action unit、SciLink skill 阶段、`action_unit_backtrack`）—— **主粒度**，与 §0.3 要求一致；
- **role**（GEO / TCGA / Statistician，`build_role_io_attribution` 已有）—— 多 worker 场景的第二视角；
- **inference-busy vs inference-idle** —— 回答「推理期间存储到底闲不闲」。

### 定义（读写必须分开，write 是 buffered，两者不可混）

对任一组 syscall S：

- `io_time(S)` = Σ `duration`（只取 read/write，不含 metadata）。`duration` 已在 `parsed.json` 每条 fs_entry 上。
- **effective bandwidth（单 worker 视角）** = `bytes(S) / Σduration(S)` —— 存储在被真正使用的那些微秒里交付了多快。
- **aggregate throughput（系统视角）** = `bytes(S) / |∪intervals(S)|` —— 区间**并集**，不是 Σduration。多 worker 并发时 Σduration 会重叠，直接相加会低估系统聚合吞吐。区间并集逻辑复用 `parallelism.active_degree` 里算 `io_busy_workers` 时已有的 read/write 区间合并。
- **duty cycle** = `|∪intervals(S)| / wall(S)`，其中 `wall(S)` 必须来自共享 run 时间轴，而不是该组 I/O 自己的首尾跨度：global 用 run wall，phase/role 用对应 tool-call 区间并集，inference-busy/idle 用 LLM 区间并集与 run wall 的差。

两个带宽的比值应当与 `io_busy_workers` 互相印证——对不上说明区间逻辑有 bug，是个免费自检。

### 出图

**按 phase 的 effective bandwidth 条形图**：每个 phase 两根柱（read / write），柱上标该 phase 的字节数与 duty cycle；全局数字（总 read/write 带宽、全局 duty cycle）标注在图角，不单独开表。挂 Axis 5。

预期能读出的形状：wall-clock 平均速率很低（几 MB/s，看上去对存储毫无压力），但 effective bandwidth 很高（存储被使用的那一刻是被打满的），duty cycle 极低。**这三个数放在一起才是完整画像，单报任何一个都会误导。**

### 必须声明的 caveat

syscall `duration` 测的是**调用返回**，不是数据落盘。write 通常只是 memcpy 进 page cache 就返回，read 命中 cache 也不碰盘。所以这是**含 page cache 的应用侧感知带宽，不是设备带宽**。I/O 字节按 syscall 结束时刻归属到 inference-busy/idle；毫秒级 syscall 相对 LLM 区间很短时误差可忽略。要拿设备带宽需要 block-layer 探针或看 fsync 时长——本轮不做，但结论里必须写清这一层。

---

## 8. Darshan 标准分箱统一到所有 size 轴

### 文献依据

Darshan 的 access-size 直方图分箱硬编码在 `DARSHAN_BUCKET_INC` 宏里（`darshan-runtime/lib/darshan-common.h`，已核对源码比较值）：

| bin | 源码比较值 |
| --- | --- |
| 0–100 B | `< 101` |
| 100 B–1 KiB | `< 1025` |
| 1–10 KiB | `< 10241` |
| 10–100 KiB | `< 102401` |
| 100 KiB–1 MiB | `< 1048577` |
| 1–4 MiB | `< 4194305` |
| 4–10 MiB | `< 10485761` |
| 10–100 MiB | `< 104857601` |
| 100 MiB–1 GiB | `< 1073741825` |
| ≥1 GiB | else |

这 10 个 bin 就是 Bez HPDC'22 图3/图4 的 request-size CDF、Patel 的 access-size 分布、以及几乎所有 Darshan-based 特征化论文的 x 轴。**它不是等对数间隔的**——在 1 MiB 附近加密（1–4M、4–10M），因为那正是对文件系统影响的分级点：

- **< 4 KiB**：小于一个 page，syscall 开销主导，Lustre 上每次都要走 RPC；
- **4 KiB – 1 MiB**：低于 Lustre 默认 stripe size，单次请求打不满一个 OST 条带，吃不到条带并行；
- **1 – 4 MiB**：达到 stripe 级别，开始吃到并行；Bez HPDC'22 图11/12 里 POSIX vs STDIO 的差距从这个区间开始拉开；
- **> 4 MiB**：bandwidth-bound 区，syscall 开销可忽略——我们的 `optimal_request_bytes` 默认 4 MiB 正是这条线；
- **≥ 1 GiB**：单请求超大，另算。

### 现状问题

- `fig_size_distribution`（[_analyzer_impl.py:1250](../src/agent_io_tracing/lineage/_analyzer_impl.py#L1250)）用 `np.logspace(0, log10(1GB), 40)`：40 个等对数宽 bin，跟任何一篇论文的图都没法并排看。
- `per_run_io_char.py` 更糙：`np.logspace(floor(log10(min)), ceil(log10(max)), ...)`，**bin 边界随数据浮动**，两次 run 之间的图都不可比。

### 改法

1. 新增公共常量（放 `analysis/` 下一个小模块，lineage 和 viz 共用）：
   `DARSHAN_SIZE_BINS = [0, 100, 1024, 10*1024, 100*1024, 1<<20, 4<<20, 10<<20, 100<<20, 1<<30, inf]` + 对应 label。
2. `fig1_size_distribution` 上下两 panel 换成这套 bin 的**离散类目条形图**（不是连续 log 轴），刻度直接写 `≤100B / 1K / 10K / 100K / 1M / 4M / 10M / 100M / 1G / >1G`。
3. `per_run_io_char.plot_file_access_volume`（x = 每文件数据量）和 `plot_rw_asymmetry` panel (a)（x = 读字节、y = 写字节）换成同一套固定边界。bin 数不再随数据浮动，跨 run、跨系统、跟文献都能并排。

---

## 9. 重构后的 index 结构

```
Global
  fig0_io_volume_summary · Agent Timeline · Time Accounting · Call DAG
  File Access Frequency × Volume (Darshan bins) · Read/Write Asymmetry (Darshan bins)

Axis 1 · Context-Limited State Persistence
  图: Directory Rescans (hist) · Inter-arrival (hist) · Reread Attribution
      Reader & Writer Fan-out (hist ×2) · reader×writer 热图 · Write→Read Staleness (hist)
  表: Access type RH/WH/RW · State file rewrite frequency

Axis 2 · Measured I/O Interface Layer
  图: Measured I/O Interface Mix
  表: Measured interface layers · Measured STDIO/POSIX byte mix

Axis 3 · Exploratory Branching and Backtracking
  图: Artifact Lifecycle (dead-fraction hist)
  表: Failed open/stat · Error-log reads · Bytes/ops by phase

Axis 4 · Reasoning-Step / Data-Granularity Alignment
  图: File Size Distribution (Darshan bins, 数字标在图上)
  表: I/O Batching Efficiency · File size · Access pattern

Axis 5 · Uncoordinated Agent Concurrency
  图: I/O Rate (bytes/s + inference 双折线) · Effective BW by phase (bar)
      Agent Activity Timeline · Who Does the I/O · I/O Autocorrelation
  表: Workflow concurrency · I/O concurrency · Intensity phases (数值)
```

---

## 10. 执行顺序

全部离线，**不需要重跑 trace**。

1. §1 Axis 1 下架 + 重编号
2. §2 删除清单
3. §8 Darshan 分箱
4. §3 四张图改直方图 + §4 writer fan-out
5. §5 staleness 重定义
6. §6 io_rate 重做
7. §7 effective bandwidth / duty cycle

1–3 是纯删改，风险最低，先做完跑一遍确认 index 不炸。

## 11. 重跑命令

改的是分析和绘图层，本地对已有 trace 直接重跑即可。**顺序有讲究**：index.html 由 `viz.trace` 生成，但它要读 `lineage/` 和 `phase1_metrics.json`，所以 viz 必须**最后**跑。

```bash
CELL=results/genomas_base_phaseb_20260708_091105/<workload>

python -m agent_io_tracing.lineage.analyzer   "$CELL"
python -m agent_io_tracing.analysis.phase1_metrics "$CELL"
python -m agent_io_tracing.analysis.per_run_io_char --results results --runs "$(basename "$(dirname "$CELL")")"
python -m agent_io_tracing.viz.trace          "$CELL"

open "$CELL/visualizations/index.html"
```

批量重跑所有已有 cell：

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

若要在 CloudLab 上跑新 trace（本轮不需要，但改完 scripts 后的完整链路是）：

```bash
source cloudlab_env.sh
rsync -az --delete \
  --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude 'results/' --exclude '.venv/' \
  ./ "$SSH_USER@$CLIENT_NODE:pi-ebpf-tracing-handoff/"

ssh -t "$SSH_USER@$CLIENT_NODE" \
  "cd pi-ebpf-tracing-handoff && sudo -E RUN_WORKLOADS='mw1_rep1' bash scripts/trace_script_bcc_genomas.sh"

RUN=$(ssh "$SSH_USER@$CLIENT_NODE" \
  "ls -1dt /mnt/lustrefs/$SSH_USER/pi-ebpf-tracing-handoff/results/*/ | head -1")
LOCAL="results/$(basename "$RUN")"
mkdir -p "$LOCAL"
rsync -az --progress --exclude 'work/' --exclude 'bcc.out' --exclude 'bcc.err' \
  "$SSH_USER@$CLIENT_NODE:$RUN" "$LOCAL/"
open "$LOCAL"/*/visualizations/index.html
```

---

## 12. 本轮明确不做

- **重读的「内容是否变化」判据**：技术上可行（判据 = 同一路径两次 read 之间 trace 里有没有 write，不需要 hash，因为我们 trace 整个进程树的全部 write），但本项目本轮不做。`compute_reread_attribution` 保持现状。
- **每个 reasoning step 的 I/O 足迹分布**：太细，我们做的是整体 characterization，不是逐 workflow 调查。
- **run-to-run variance**：属于实验设计，不在本轮分析改造范围。
- **设备级带宽**：需要 block-layer 探针，本轮只测应用侧感知带宽并声明 caveat。
