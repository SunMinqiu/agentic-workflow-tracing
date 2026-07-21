# Agentic Scientific Workflow I/O Characterization

## 结构性差异轴 × Metric × 文献出处 映射表

---

## 0. 地基：两个前提定义

### 0.1 五个结构性差异轴的来源

不是从文献 metric 表倒推"我们能测什么"，而是从**研究问题本身**推导：agentic workflow 的定义就是"决策主体从静态 scheduler 换成了自主 agent"。把这个替换在哪些方面改变了执行方式想清楚，异常的来源类别就是有限且可枚举的。五个轴即为此枚举：


| #   | 轴       | agent vs 传统 workflow 的结构性差异           |
| --- | ------- | ------------------------------------- |
| 1   | 状态持久性   | 内存状态不丢 → context window 会截断/摘要        |
| 2   | 接口抽象层   | 领域专家手写针对格式的 I/O → agent 生成的通用代码       |
| 3   | 分支与回溯   | 走完即止的 DAG → explore-abandon-retry     |
| 4   | 执行粒度对齐  | task 按数据规模切 → LLM 主观切 reasoning step  |
| 5   | I/O 并发（跨 agent 无协调） | scheduler 以全局资源视角协调 I/O → 多个互不感知的 worker/LLM call 并发 I/O；**首要测真实 I/O 并发与突发性，workflow 并行度为次要** |




### 0.2 Oracle 定义（全表地基）

> **Oracle = agent 成功 trace 经过「静态去冗余」后蒸馏出的确定性脚本。**

「静态去冗余」的含义：对同一文件、同一区间、且两次访问之间无写入的重复读，合并为一次；对被 abandon 的探索性分支产生的 I/O，剔除；保留任务真正必需的确定性 I/O 序列。

**Oracle 的角色**：它和 agent 处理**同一份数据、完成同一个任务**，因此「数据/任务本身要求的 I/O」在 oracle 和 agent 两侧相等、被抵消。**agent 版本 relative to oracle 版本的差，就是 agent 运行时适应性（adaptivity）独有的开销。**

**归因原则（贯穿全表）**：不在单条访问上做「agent-caused vs task-caused」归因（单条访问上分不了，也不该分）；只在 **agent 分布/计数 relative to oracle 分布/计数的差值**上归因。数据引起的部分被 oracle 减掉，剩下的才是 agent 的。

**已知软肋（写作时须诚实声明）**：oracle 从 agent 成功 run 蒸馏，不是理论最优。所测为「运行时适应性的代价」，**不 claim** 为「距离最优有多远」。静态去冗余给了 oracle 规范性（不是原样复制 agent I/O），但仍是一个基线而非下界。

### 0.3 分析陷阱声明（不是 metric，是做 makespan/吞吐分解时必须声明的 caveat）

> **「末端高吞吐掩盖上游瓶颈」陷阱**：做整体 I/O 效率或 makespan 分解时，不能只看末端 task 的吞吐数字，否则会误判整体高效。

Tang IPDPS'26 Finding 13（图5，DeepDriveMD 的 train→final 组写 checkpoint/embedding 达 2.4GB/s，但上游 n-1 的 sim→agg / sim→train 组吞吐低得多）与 Raj CPU-Centric 的「末端 GPU 高吞吐掩盖 CPU 侧工具执行瓶颈」是**同一个分析框架的两次独立出现**——一个在传统 HPC workflow，一个在 agentic serving，交叉验证了这个陷阱的普遍性。

对本项目的约束：v3 第 9 类「makespan 分解（计算时间 vs I/O 等待时间）」必须加这条 caveat——按 agent / reasoning phase 逐段分解，而非只报末端聚合吞吐。

### 0.4 Producer-Consumer taxonomy 的 oracle 限定（防止误用 Tang 的四分类）

Tang IPDPS'26 的 P-C 四分类（1-1 / 1-n / n-1 / n-n，表I）**预设静态 DAG**（producer 和 consumer 提前可知）。agentic workflow **没有固定 DAG**，因此这套 taxonomy 不能直接搬用。

> **限定：P-C taxonomy 只在相对 oracle 基线时才有解释力。** 即测量「oracle 版本的 P-C 关系」vs「agent 版本的 P-C 关系」的偏离（fan-in/fan-out divergence），而非把 Tang 的四分类当成 agent workflow 的固有静态属性。

写作/建模时若要引用 Tang 的 P-C 形式化，必须挂在此限定下，否则会隐含「agent workflow 有静态 DAG」这一与前提冲突的假设。

### 0.5 本项目五轴 vs Raj 三轴的关系（防 reviewer 质疑两套分类冲突）

Raj CPU-Centric 用三个正交 compile-time 轴给 agent workload 分类：**编排者（LLM / Host）× 路径（静态 / 动态）× 重复性（单步 / 多步）**（图1）。本项目的五轴是「agent 在哪些方面偏离传统 workflow」。两者关系明确为**正交、分工，不冲突**：


|          | Raj 三轴                                                     | 本项目五轴                          |
| -------- | ---------------------------------------------------------- | ------------------------------ |
| 回答的问题    | agent **长什么样**（workload 分类）                                | agent 会犯什么 **I/O 异常**（异常来源分类）  |
| 在本项目中的角色 | **benchmark 采样框架**——确保所测 agent 覆盖不同象限（如动态路径×多步 vs 静态路径×单步） | **测量框架**——负责量化每个 agent 在五轴上的偏离 |


**可写成的联合命题**：不同 Raj 象限的 agent，在本项目五轴上的偏离程度不同（预期：动态路径×多步的 agent 在分支回溯、状态持久性和并发轴上偏离更大）。这样两套分类互相引用、互不打架，且给了本项目一个可检验的假设。

---



## 1. 判决总表：五轴 × 文献对应强度


| 轴       | 文献对应强度       | 能借的测量工具（含出处）                   | 必须自己造的空档                           | 是否需 oracle 对照 |
| ------- | ------------ | ------------------------------ | ---------------------------------- | ------------- |
| 1 状态持久性 | 有但机制相反       | inter-arrival time、RH/WH/RW 分类 | 「两次读之间内容是否变化」判据                    | 是（核心）         |
| 2 接口抽象层 | **强，可直接借**   | 接口分布、跨层 reshape        | agent 选的 vs oracle 该选的接口差距         | 部分            |
| 3 分支回溯  | 零            | —                              | 探索性 I/O 开销比                        | 是（核心）         |
| 4 粒度对齐  | 中，需 oracle 配 | request size 分布、小文件聚合开销        | agent vs oracle 的 request size 分布差 | 是（核心）         |
| 5 I/O 并发  | **强，可直接借**   | 自相关、I/O 并行度 CDF、突发性阶段划分            | inference gap 与 bandwidth/duty-cycle 分解                 | 否             |


---



## 2. 逐轴详表：Metric × 处理方式 × 文献出处 × Oracle 角色



### 轴 1 · 状态持久性（重读）


| Metric                                                           | 处理方式                                          | 文献出处（图/表）                                                                                          | Oracle 角色                                                                 | 是否实现 |
| ---------------------------------------------------------------- | --------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- | --- |
| 重复访问 inter-arrival time：同一文件被再次访问的时间间隔分布                         | 借工具改定义                                        | Patel FAST'20 图5a（读均值47hr/写均值55hr，80%文件需50-55hr再访问）；Patel FAST'20 图6b（跨应用 inter-arrival 均值31hr）    | 借测量工具。传统场景间隔反映正常复用；agent 场景需 oracle 对照才能区分「合理复用」与「失忆重读」                   | ✅ 已实现。compute_inter_arrival 从 fs_entries 时间戳按文件算相邻访问间隔分布（p50/p95/p99/mean、pct_lt_1s）。实测 GenoMAS p5：1116 文件重访、93.9% 重访在 1s 内。 |
| RH/WH/RW 文件分类：按读/写数据量把文件分为 read-heavy / write-heavy / read-write | 直接借用（作为描述性维度）                                 | Patel FAST'20 图3a（22% RH / 7% WH / 71% RW，Finding 1）                                               | 描述性，不需 oracle                                                             | ✅ 已实现。compute_access_type_rhwhrw 从 artifacts.csv 每文件读写字节按 read_share=rb/(rb+wb) 三分（RH≥2/3、WH≤1/3、RW 其间），出计数/占比/字节。实测 GenoMAS 37.7%RH/59.4%WH/2.8%RW，ChemGraph 54.7/4.0/41.3，已能区分两系统。 |
| **失忆重读量（核心自造）**：两次读之间文件内容/mtime 无变化的重复读，其超出 oracle 版本的额外次数与字节数   | 借 inter-arrival 工具 + 自造「内容是否变化」判据 + oracle 减法 | 测量脚手架借 Patel FAST'20 图5b（连续同类型 run 数分布，producer-consumer 不对称，Finding 4）；但「两次读之间内容零变化」这一判据文献无对应，须自造 | **核心**。归因不在单次访问，而在「agent 读某文件 N 次 − oracle 读同文件 M 次 = 超额 N−M 次为 agent 失忆」 | ⚠️ 部分实现。做了：phase1_metrics.compute_reread_attribution 以 (path, tool_call_id, fd) 去重后，把重复读分为 agent-induced（same_step_reopen、reread_after_backtrack）与 residual 跨阶段复用。还差（非 oracle 缺口）：「两次读之间内容/mtime 零变化」判据尚未实现，现用 action_unit_backtrack phase 代理且依赖 GenoMAS 特定标注。 |
| read amplification（逻辑读字节 / 物理必需字节）                               | 借工具改定义 + oracle 减法                            | Patel FAST'20 图10a/b（数据量 CoV 仅12% 但 I/O 耗时 CoV 达39%，RH 文件最高68%，Finding 10）作为「跨 run 稳定性」参照          | agent 的 read amp 绝对值无法归因；agent read amp − oracle read amp 才是 agent 超额部分   | ✅ 已实现（唯一未做的是 oracle 减法，按约定不计缺口）。lineage read_amplification = 总读字节 / true_size（每文件），build_reuse_summary read_reuse_factor = 总读 / 唯一读字节。 |
| 目录重复扫描：同一目录被 `getdents`/listing 反复扫，反映状态/context 未保留导致的路径集合重建 | 借 metadata/reuse 计数工具改定义 | 测量脚手架借 Patel FAST'20 inter-arrival/reuse 思路；机制是 agent 状态遗忘，不是传统 workflow 正常复用 | 描述性；若要归因为 agent 超额，需 oracle/stateful baseline 减法 | ✅ 已实现。compute_directory_scan_count 产出 total scans、unique dirs、rescanned dirs、scans_per_dir_hist、p95_scans_per_dir。 |
| 状态文件重写频率：小型协调/进度状态文件（cohort_info / completed_tasks / *_state.json / manifest / .lock）的重写次数，反映内存状态不持久、改用外部文件逐 task read-modify-write | 借 reuse 计数工具改用途 | 无直接对应；per-task 重写是 agent 外置状态的症状 | 描述性；n_writes 相对逻辑 task 数偏高即 read-modify-write-per-task 误配候选，归因超额需 oracle | ✅ 已实现。compute_state_file_rewrite_frequency 按 STATE_FILE_PATH_HINTS 匹配，出 state_shaped_files / total_writes / 每文件 n_writes/n_reads。挂轴1（状态持久性）：外部状态文件正因内存状态会截断/丢失才被反复重写。 |


---



### 轴 2 · 接口抽象层


| Metric                                                                                                                | 处理方式                       | 文献出处（图/表）                                                                                                                                                                                         | Oracle 角色                                                                            | 是否实现 |
| --------------------------------------------------------------------------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | --- |
| I/O 接口类型分布（POSIX / MPI-IO / STDIO / 高层库）在 workflow 内的占比                                                               | 直接借用                       | Bez HPDC'22 表6（Summit SCNL 层 STDIO 是 POSIX 的4.37×、MPI-IO 的200×+）；Bez 综述 图13/14 及接口占比统计（MPI-IO 57.53% / POSIX 47.26% / STDIO 3.42%）；Patel FAST'20 图8a-c（按接口的每 run 每文件传输量与 rank 间标准差）               | 描述性，不需 oracle                                                                        | ⚠️ 部分实现。做了：io_api_classifier 对 agent 生成代码 AST 分类为 stdio / posix_raw / structured / mpiio / vector_index，聚合 layer_exec_counts、pct_stdio_only、pct_structured_any；该图/表是 static, from generated source — not measured syscall bytes。Phase A 新增 libc `fread/fwrite` uprobe + 既有 read/write syscall 的 STDIO/POSIX 字节估计。还差：实际 trace 无 MPI-IO 字节场景。 |
| 接口选择的性能后果：不同接口在相同传输规模下的读写吞吐                                                                                           | 直接借用（作为「选错接口代价多大」的量化依据）    | Bez HPDC'22 图11a/b、图12a/b（POSIX 全面优于 STDIO，100GB-1TB 区间快达40×，小文件快3×）                                                                                                                              | 描述性                                                                                  | ❌ 未实现。无按接口测读写吞吐/代价的功能。 |
| 逻辑-物理放大系数：Tool-call 层逻辑读写字节 / Subprocess 层物理读写字节                                                                      | 借工具（跨层 reshape 思想）+ 自造分层定义 | Bez 综述 图15/16（应用层 4MB 连续 HDF5 请求经 MPI-IO collective 拆成多个 1MB POSIX 请求，因 stripe 配置变非连续，context 逐层丢失）——直接理论支撑「pattern 随层 reshape」                                                                   | 部分。放大系数本身可独立测；「该不该放大」需 oracle 对照                                                     | ❌ 未实现。当前没有单列「tool-call 层逻辑字节 / subprocess 层物理字节」跨层比值；旧的 file_count/metadata_op 假分母已删除，避免误报。 |
| 接口跳变层数：一次逻辑访问穿越的软件栈层数                                                                                                 | 借「逐层 reshape」思想自造          | Bez 综述 图15/16（同上，论证需区分 local/global/system-wide scope，必须指明在哪一层描述 pattern）                                                                                                                         | 描述性                                                                                  | ❌ 基本未实现。generated_code 记录了每段代码触及的 io_layers（层集合），但非「一次逻辑访问穿越软件栈层数」，无 local/global/system-wide scope 刻画。 |
| **接口错配（核心自造）**：agent 实际选用的接口 vs oracle 版本该用的接口之间的差距（如 agent 用通用 open/read 整读，oracle 用 HDF5 slice 增量读）                 | 完全自造 + oracle 对照           | 无直接对应。传统研究里接口是人主动选（专家知道用 collective），无「该选却没选」的对照                                                                                                                                                  | **核心**。oracle 脚本代表「领域专家/去冗余后该用的接口」，agent 接口相对它的偏离即错配                                 | ✅ 已实现（「该用的接口」靠 H1 启发式判定；缺 oracle 规范基线，按约定不计缺口）。io_api_classifier 的 H1 判据（该用 structured 却用 stdio）+ verdicts 对照 HPC 期望，pct_stdio_only / pct_structured_any 即错配信号。 |
| **格式-layout 对齐失配（核心自造，比接口错配更深一层）**：agent 即使选对了结构化格式（如 HDF5），其访问模式与文件内部 layout（chunk size / chunked vs contiguous）是否对齐 | 借工具 + oracle 对照            | Tang IPDPS'26 Finding 14（图5，train 在 HDF5 上吞吐显著高于 infer，因访问模式与 chunk 布局对齐更好；PtychoNN/Montage 亦同）+ Case Study c（HDF5 chunked→contiguous 调优，高并发下 1.9× 加速，表VI）——**直接证明「选对格式 ≠ 高效，还要 layout 与访问模式对齐」** | **核心**。回答了「接口错配」的遗留缺口：agent 的失配不止在选错接口，还在选对格式后 layout 配错。oracle 代表访问模式与 layout 对齐的版本 | ❌ 未实现。io_api_classifier 能识别 structured 层（h5py / create_dataset 等），但不检查 HDF5 chunk 配置（chunk size、chunked vs contiguous），也不与访问 stride 对比；无 layout 对齐指标。且现有 trace（GenoMAS 等）用 csv/json 而非 HDF5，暂无 chunked 格式场景可测。 |


---



### 轴 3 · 分支与回溯


| Metric                                                                           | 处理方式             | 文献出处（图/表）                                                                                                    | Oracle 角色                                                | 是否实现 |
| -------------------------------------------------------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------- | --- |
| **探索性 I/O 开销比（核心自造）**：被 abandon 的探索性分支所产生的读写字节 / 总读写字节                           | 完全自造 + oracle 减法 | 无。传统 workflow 无「跑到一半放弃重来」；Bharathi 中 SIPHT 的 Patser job 数依输入而变（图6/表5）是**数据规模决定的确定性 fan-out，非探索性回溯**，须在文中明确区分 | **核心**。oracle 蒸馏时剔除被 abandon 分支；agent 全量 − oracle = 探索开销 | ❌ 不能作为 finding。无 oracle 时 dead_write 无法区分最终输出 vs 探索废物，不可作 finding；需 oracle 减法。报告中的 Exploration overhead 面板已移除，避免把 raw number 当贡献。 |
| failed open/stat：探索候选路径时产生的失败 open/stat/access 调用                                             | 完全自造描述性信号             | 无直接对应；这是 branch/search 症状，不是传统 workflow metric                | 描述性；若任务本身有合理 missing check，则用 oracle 减掉                      | ✅ 已实现。compute_failed_open_stat_count 产出 agent 级 `total_failed` + 归一化 `failed_rate`（分母 = agent 级 open/stat/access 尝试数）+ by_syscall/top paths。**已过滤 CPython import 探针**（每次 `import` 对 sys.path 逐目录 stat/open、绝大多数 ENOENT 是解释器噪声而非 agent 候选路径探测），另出 `import_probe_failed_excluded` / `total_failed_raw` 供透明核对。 |
| error-log 回读：agent 读回被 lineage 分类为 `logs` 的文件（读自己/workflow 的日志输出），是典型探索/debug 信号 | 完全自造描述性信号 | 无直接对应；branch/search-debug 症状，非传统 workflow metric | 描述性；oracle 中成功脚本无需回读日志诊断，agent 侧回读量即偏离 | ✅ 已实现。compute_error_log_reads 出 log_files / log_files_ever_read / total_reads / total_read_bytes。 |
| retry 次数分布：同一 reasoning 目标被重复执行的次数（如 SciLink trace 中 RunFinalInterpretation 调两次） | 完全自造             | 无                                                                                                            | oracle 中每目标执行 1 次；agent 超出部分为探索性 retry                   | ⚠️ 部分实现。做了：action_unit_backtrack phase 存在且 bytes_ops_by_phase 计其量。还差：无直接产出「同一 reasoning 目标重复执行 N 次」的分布；SciLink 那种 RunFinalInterpretation 调两次的通用计数未实现。 |


---



### 轴 4 · 执行粒度对齐


| Metric                                                                          | 处理方式                   | 文献出处（图/表）                                                                                                                                        | Oracle 角色                                                                                           | 是否实现 |
| ------------------------------------------------------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------- | --- |
| **request size 分布差（核心）**：同一任务同一数据下，agent 版本 vs oracle 版本的单次 read/write 请求大小分布差异 | 借工具 + oracle 对照        | 测量工具借 Bez HPDC'22 图3/图4（文件传输大小 CDF、单进程请求大小 CDF，Darshan 标准 bin）；Bez 综述统计 request size 为第二常用特征（67.81%，图13）                                         | **核心**。数据本身要求的碎在两侧相等被抵消；agent 分布相对 oracle 分布左移（更碎）即粒度错配                                             | ✅ 已实现（agent 侧分布已产出；缺 oracle 对照，按约定不计缺口）。compute_request_size_cdf 给 p50/p95/p99、pct_lt_4kb、pct_lt_10mb。 |
| 小 I/O 聚合潜力：若把 agent 碎片化访问按 oracle 边界合并，可节省的固定开销                                 | 借工具改用途                 | Nawaz 图4/图5（mConcatFit 读6173个均值0.3KB小文件，占 makespan 22-29%）；mitigation「bulk transfer 合并小文件传输请求」直接作为「若对齐能省多少」的估算方法；Bez HPDC'22 图3（97%读/99%写文件<1GB） | oracle 提供合并后的目标边界                                                                                   | ✅ 已实现。compute_analytical_optimum 现同时出写侧 write_call_amplification 与读侧 read_call_amplification（actual/optimum，4MB 请求边界）。实测 GenoMAS p5 读侧 44.4×（3689 读调用 vs 最优 83）。 |
| 碎片化归因标签：碎片化访问中，源于 reasoning step 边界（agent）vs 源于数据格式本身（task）的比例                  | 借 oracle 减法实现「不在单条上归因」 | 无直接对应                                                                                                                                            | **核心方法**。单条 4KB read 无法归因；agent request size 分布 − oracle request size 分布，差值即 reasoning-step 边界造成的碎片 | ✅ 已实现（复用 compute_request_size_cdf 的 agent 分布；该指标唯一缺口是 oracle 分布相减，按约定不计缺口）。 |


---



### 轴 5 · I/O 并发（跨 agent 无协调）

**本轴以「真实 I/O 并发」为主线，workflow/LLM-call 并行度为次要背景。** 主线指标是 io_busy_workers（同时 I/O-busy 的 worker 数）、I/O 读写自相关、高/低强度阶段划分——量化 I/O 在时间上的并发与突发；workflow concurrency (opportunity)、role_io_attribution、interference 重定义是次要维度，解释「谁在并发、为何无协调」，不是本轴的度量重点。

| Metric                                                                                                               | 处理方式        | 文献出处（图/表）                                                                                                            | Oracle 角色     | 是否实现 |
| -------------------------------------------------------------------------------------------------------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------- | ------------- | --- |
| I/O 读/写自相关（1/5/25 分钟窗口，多 lag）+ 读写互相关                                                                                 | 直接借用        | Patel SC'19 图9a-c（≥5分钟窗口自相关显著，1分钟窗口弱；读写互相关全程弱）                                                                       | 描述性，不需 oracle | ✅ 已实现。compute_io_autocorrelation 在 1/5/25min 窗口对读/写各出 lag1-3 自相关 + lag0 读写互相关。注：短 trace（如 16min）5/25min 窗 bin 太少，需较长 run 才稳。 |
| I/O 并行度 CDF：同时 I/O-busy 的 worker 单元数                                                                                       | 直接借用        | Patel SC'19 图11a/b（平均并行度仅5.58读/5.93写，>85%场景用<10个OST，共248可用）；表1（写在高强度阶段并行度比读高16%-240%）                                | 描述性           | ✅ 已实现。parallelism.active_degree 新增 `io_busy_workers`，按 read/write-family syscall 区间与 libc `fread/fwrite` probe（若有）重叠计算；原 semantic-event degree 改名为 workflow concurrency (opportunity)，表示并发机会而非真实 I/O 并发。 |
| 高/低强度 I/O 阶段划分（前25%/后25%分位）+ 阶段长度与间隔                                                                                 | 直接借用        | Patel SC'19 图7（读阶段更长6.62min但更少见，写更频繁更短）、图8（低强度阶段呈相反对应）；Patel FAST'20 图9a-c（3-5am 数据量最大最耗时，耗时与CoV强负相关 Spearman -0.94） | 描述性           | ✅ 已实现。compute_intensity_phases 按 60s 窗分箱，用非空箱 75/25 分位阈值切高/低强度段，出段数/平均段长/最长段。实测 GenoMAS p5：16 箱、4 个高强度段（均为单箱突发）。 |
| 并发 rank/agent 间 I/O 时间倾斜（快等慢造成的计算周期浪费）                                                                               | 直接借用，改归因来源  | Patel FAST'20 图8a-c（OST负载不均导致并发rank间I/O时间差异极大，快rank等慢rank，Finding 8）                                                 | 描述性           | ⚠️ 部分实现。做了：compute_self_intervals 按 role 去嵌套算每 worker 自时间，parallel_time_ratio、observed-pid 并行、build_role_io_attribution 每角色字节；支持 GenoMAS fanout。还差：无专门的「快等慢」倾斜/浪费周期量。 |
| 并发写 run 数与系统级 contention                                                                                             | 直接借用        | Patel FAST'20 图6c（写run单次25GB vs 读run 17GB，Finding 6，mitigation「限制并发写run数」）                                           | 描述性           | ⚠️ 弱部分实现。做了：fanout 可视化（fanout_plot、fanout_input_sizes）跨多 run 汇总。还差：系统级并发写 contention 指标未计算。 |
| **interference 来源重定义（须自做的概念工作，非新 metric）**：传统 interference 来自 MPI rank 间 / 并发应用间；agent 场景来自多个独立 LLM call 互不感知彼此的资源占用 | 测量工具全借，故事重写 | 概念参照 Bez 综述 图15/16（两并发 IOR 实例请求交错，并发应用间明显 interference）                                                              | 描述性           | ✅ 已实现（概念 + 工具）。本轴重定义已落地为「测 agent/LLM/tool 之间的时间重叠 + role_io_attribution 谁并发占用资源」而非 MPI rank；parallelism 模块即其操作化。 |


---



## 3. 明确排除的文献 Metric + 排除理由

以下文献 metric 经五轴筛查，**不对应任何轴**，明确排除（列出以证明是筛过的，非遗漏）：


| 被排除的 metric                   | 出处                                                              | 排除理由                                                                                                                                                               |
| ----------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 云 SSD burst credit 耗尽导致吞吐骤降   | Nawaz 图9（Amazon 前60次128MB/s 后骤降至9MB/s以下）                        | 云基础设施计费/限流机制的产物，与 agent 决策方式无关，属部署环境噪声                                                                                                                             |
| 传输工具 TCP 连接/TLS 握手行为          | Nawaz 图8（aws-cli vs gsutil vs pegasus-s3 的连接复用差异）               | 传输客户端实现差异，与 agent 无关；且我方为单节点/超算节点，非跨云传输场景                                                                                                                          |
| 空文件下载固定开销的时间稳定性验证             | Nawaz 图6/图11                                                    | 云平台方法学稳定性验证，与本项目无关                                                                                                                                                 |
| 多线程 vs 单线程传输 makespan         | Nawaz 图10                                                       | 传输并行度调优，属传统优化，非 agent 特性                                                                                                                                           |
| OSS/OST 服务端 CPU 利用率及其自相关      | Patel SC'19 图15/图16（OSS 均值利用率<2%）                               | 存储服务端硬件资源刻画，本项目观测点在 agent 框架层 + subprocess 层，不含服务端遥测                                                                                                               |
| OST 级数据量长期负载不均                | Patel SC'19 图10 / Patel FAST'20 图7（最不活跃 OST 仅为最活跃的13%）          | Lustre 服务端 MDS 均衡策略产物，与 agent 决策无关；且依赖 LMT 服务端日志，我方用 eBPF 应用侧采集                                                                                                    |
| MDS 文件 open/close 累计操作与自相关    | Patel SC'19 图13/图14                                             | 服务端 metadata 遥测，观测层不匹配；其中「20%文件从未关闭」仅作为轴4「废弃产物可量化」的方法学先例引用，metric 本身不采纳                                                                                            |
| in-system 层 vs PFS 层的用户使用偏好分布 | Bez HPDC'22 表3/表5/图7（PFS 访问数是 in-system 的3.63×-28.87×）          | 多层存储的用户 staging 行为刻画，与 agent 决策方式无关；属存储系统使用习惯研究                                                                                                                    |
| 合成 workflow 生成器的数据规模缩放律       | Bharathi 图1-6/表1-5、缩放律说明                                        | benchmark 构建型贡献，非性能 finding；且预设 DAG 静态可刻画，与 agentic 前提冲突                                                                                                           |
| 全年读写总量对比（读反超写）                | Patel SC'19 图3（读是写的1.75×）                                       | 系统级宏观统计，反映的是传统 workflow 化趋势下的正常复用，非 agent 特性；可作背景引用但不作 finding                                                                                                     |
| CPU/GPU 动态能耗占比随 BS 变化         | Raj CPU-Centric 图11（RAG 的 CPU 动态能耗占比稳定 61%，多数 workload 呈先升后降驼峰） | **不是 I/O characterization，移入范围外/未来工作**。能耗属 CPU-centric serving 维度，收编会稀释「I/O 特征」主线。明确不进五轴，避免主题漂移。（这是一个取舍决定，非「不相关」——若日后拓展到 agent 能效可复用其 nvidia-smi module power 相减法） |


---



## 4. 给 v3 九大类 metric 的体检结论（挂轴检查）

v3 方法论的「九大类」逐条挂轴，挂不上的当场标注：


| v3 类别                                   | 挂到的轴              | 判断                                         |
| --------------------------------------- | ----------------- | ------------------------------------------ |
| 1 全局聚合类（总读写字节、读写比）                      | 无                 | ⚠️ 纯 setup 描述性统计，可留作背景，**不作 finding**，勿当贡献 |
| 2 数据量类（逻辑-物理放大、文件大小分布）                  | 轴3（放大系数）、轴5（大小分布） | ✅ 挂得上                                      |
| 3 操作计数类（探索开销比、未关闭文件占比）                  | 轴4                | ✅ 挂得上，探索开销比是轴4核心                           |
| 4 访问模式类（Sequential/Random、request size） | 轴4/轴5            | ✅ 挂得上。Sequential/Random 由 VFS/POSIX offset 按 fd open-generation 计算，读写分流并报告 `pct_ops_with_offset`；request size 继续挂粒度/批量效率。Caveat：mmap 读不可见，缓冲层下发顺序不等于应用逻辑顺序，逻辑 offset 不等于物理连续性。 |
| 5 IO Interface 类（接口分布、跳变层数）             | 轴3                | ✅ 挂得上                                      |
| 6 时间/并发/突发性类（自相关、阶段划分、突发性）              | 轴6                | ✅ 挂得上，工具可直接借                               |
| 7 任务级 Profile 类（runtime/输入输出分布）         | 轴5（部分）、轴6（部分）     | △ 部分挂得上，需明确每个子项归哪个轴，否则是无意识抄 Bharathi       |
| 8 重复访问/局部性类（重读归因、重复率）                   | 轴2                | ✅ 挂得上，是轴2核心                                |
| 9 Oracle-Replay 对比类                     | 轴1/2/4/5 全部       | ✅ 这是所有 oracle 依赖轴的公共地基，最该展开                |


**体检提示**：第 1 类和第 7 类是最容易「无意识抄文献」的地方——凡挂不上具体轴的子项，要么降级为 setup 背景，要么删。
