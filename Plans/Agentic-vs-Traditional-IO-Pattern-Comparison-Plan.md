# Agentic 与 Traditional Scientific Workflow I/O Pattern 对比计划

## 0. 目标与归一化规则

目标：刻画 agentic scientific workflow 的 I/O pattern，并与 traditional scientific workflow 的同类 pattern 比较。

跨 workflow 只按实际传输的 read bytes 和 write bytes 归一化：

- 读操作计数：每 GiB read bytes。
- 写操作计数：每 GiB write bytes。
- metadata 和 namespace 计数：每 GiB total I/O，分母为 read bytes 加 write bytes。
- 分布、比例、时长、相关性、并发度和带宽不再归一化，表中记为 `--`。
- 同时保留绝对值和归一化值。
- 分母为零时记为 `N/A`。

缩写：`R/GiB` 为每 GiB read bytes，`W/GiB` 为每 GiB write bytes，`T/GiB` 为每 GiB total I/O。

## 1. 现有指标、文献对位和代码修改

本轮已完成不改变实验定义的通用代码：workload-scoped byte denominators、read/write
request-size CDF、read/write inter-arrival、active/wall bandwidth、通用 same-file
reopen、burst summary、扩展 byte-normalized rates、traditional task/stage attribution、
`execution_unit_io.csv`、execution-unit skew、四类 P-C summary、lifecycle quantiles、
failed-path errno 分类和 `trace_quality.json`。HDF5/MPI-IO 仍需新 trace 验证；Montage
和 MuMMI 的实验选择不在本轮固定。


下表的 Index 名称逐字取自当前每个 cell 的 `visualizations/index.html`。`图`表示图片卡片，`表`表示 Index 中的指标表；没有独立卡片的数值注明其数据文件。

| 现有指标 | Index 中的对应名称 | 对比文献 | 归一化 | 需要修改 | 修改理由 |
| --- | --- | --- | --- | --- | --- |
| Read/write bytes 和 data-op counts | 图：`I/O Volume Summary`；图：`Byte-normalized I/O Counts` | [R3] Fig. 2-3；[R6] profile tables；[R9] Fig. 2-3、Table III；[R11] Sec. 5.4.5 | Read ops `R/GiB`；write ops `W/GiB`；bytes `--` | 新增统一的 byte-normalized summary，供 agentic 和 traditional 共用。 | 跨 workflow 统一 |
| Read/write byte share 和 op share | 图：`I/O Volume Summary`；图：`Byte-normalized I/O Counts` | [R1] Fig. 3；[R3] workload summary | `--` | 同时报 byte share 和 op share。 | 文献对齐 |
| Metadata op counts 和 metadata/data-op ratio | 图：`Byte-normalized I/O Counts` | [R3] Fig. 17-18；[R11] Fig. 20；[R9] stage analysis | Count `T/GiB`；ratio `--` | 新增 metadata ops per GiB total I/O。统一 strict metadata 和 storage metadata 定义。 | 跨 workflow 统一 |
| Read/write request-size distribution | 图：`Read/Write Request-size CDF` | [R4] Fig. 4-5；[R3] Fig. 19-21；[R9] 1000 Genomes read-size statistics | `--` | 分开输出 read 和 write CDF。增加 byte-weighted mean、`<4 KiB`、`<64 KiB`、`<1 MiB`。 | 文献对齐 |
| File-size 和 per-file transfer distribution | 图：`File and Request Size`；图：`File Access Frequency x Volume` | [R4] Fig. 3、9；[R7] Fig. 1-2；[R5] task I/O tables | CDF `--`；read files `R/GiB`；written files `W/GiB` | 区分 traced bytes per file 和 on-disk file size。新增文件计数的 byte-normalized rate。 | 正确定义并统一尺度 |
| RH/WH/RW 文件分类 | 图：`Read/Write Asymmetry`；表：`Access type RH/WH/RW` | [R1] Fig. 3a；[R4] Fig. 6、8 | `--` | 算法不改。注明 [R1] 使用相同的 `2/3` 阈值；[R4] 的 RO/RW/WO 是不同定义。 | 文献对齐 |
| Measured POSIX/STDIO/MPI-IO interface mix | 图：`Measured I/O Interface Mix` | [R3] Fig. 15-16；[R4] Table 6、Fig. 9、11-12 | Byte share `--`；interface ops 用 `R/GiB`、`W/GiB` | 跨 workflow 只使用实测 interface bytes 和 ops。generated-code classifier 仅用于解释 agentic 结果。按实际 workload 补 MPI-IO/HDF5 tracing。 | 避免混用实测与推断结果 |
| Effective bandwidth | 图：`Active BW by Phase` | [R4] Fig. 11-12；[R9] Fig. 3；[R10] Fig. 5 | `--` | 统一输出 active bandwidth 和 wall-time bandwidth。禁止混用 syscall-duration sum 与 I/O-busy union time。 | 指标正确性 |
| I/O-busy time 和 wall-time fraction | 图：`Time Accounting` | [R9] Fig. 2、Table II-III；[R3] Fig. 4、17-18 | `--` | 拆成 universal I/O-busy/wall 和 agentic-only I/O-vs-inference。 | 分离通用指标与 agentic 指标 |
| Bytes、ops、latency by phase | 图：`Time Accounting`；表：`Bytes/ops by phase` | [R8] Fig. 6；[R9] Fig. 2-3；[R10] Fig. 5 | Phase ops 用 `R/GiB`、`W/GiB`；byte/time share `--` | 建立通用 `execution_unit`。Agentic 使用 reasoning/tool phase；traditional 使用 task/stage。 | 跨 workflow 统一 |
| Same-file inter-arrival distribution | 图：`Inter-arrival Histogram` | [R1] Fig. 5a、6b | `--` | 分开输出 read 和 write。保留 numeric CDF，便于叠加文献时间尺度。 | 文献对齐 |
| Reread 和 read amplification | 图：`Reread Attribution` | [R1] Fig. 5b、10；[R8] Fig. 2、4-5 | Reread count `R/GiB`；amplification `--` | 新增通用 same-file reopen count、bytes 和 read-reuse factor。Backtrack 分类仅用于 agentic。 | 跨 workflow 统一 |
| Directory scan 和 rescan | 图：`Directory Re-scans (getdents64)` | [R3] Fig. 14、18 仅提供 many-file/metadata 参照 | `T/GiB` | 新增 scans 和 rescans per GiB total I/O。Traditional workflow 必须运行相同 detector。 | 跨 workflow 统一 |
| Failed open/stat/access | 表：`Failed open/stat` | [R11] Fig. 20 仅提供 metadata root-cause 参照 | `T/GiB` | 按 syscall 和 errno 输出。修正失败路径不在 `artifacts.csv` 时的 workload scoping。 | 指标正确性 |
| Error-log reads | 表：`Error-log reads` | 无直接文献对位 | `R/GiB` | 同时报 count 和 bytes。Agentic 与 traditional 使用同一 log-path classifier。 | 跨 workflow 统一 |
| State/checkpoint file rewrites | 表：`State file rewrite frequency` | 无直接文献对位 | Writes `W/GiB`；reads `R/GiB` | 将 detector 分为通用规则和 workflow-specific 规则。通用规则覆盖 state、checkpoint、manifest 和 lock 文件。 | 避免 workflow-specific 规则污染对比 |
| Sequential/random read/write | 图：`Access Pattern` | [R3] Darshan counters；[R10] Fig. 5 | `--` | 使用现有 VFS offset 和 open generation。输出 Seq R、Rand R、Seq W、Rand W，并报告 offset coverage。 | 指标正确性 |
| Consecutive-request mergeability | 表：`I/O Batching Efficiency` | [R4] Fig. 4-5；[R7] Fig. 1-5 | Saved read ops `R/GiB`；saved write ops `W/GiB`；mergeable-byte share `--` | 删除 global analytical optimum。只保留同一 `tid`、`fd`、open generation 内的连续请求合并，单次上限 4 MiB。 | 指标正确性 |
| Read/write autocorrelation | 图：`I/O Autocorrelation` | [R2] Fig. 9a-c | `--` | 保留 1、5、25 分钟窗口。bin 数不足时不输出相关系数。 | 文献对齐并避免无效结果 |
| High/low I/O intensity phases | 图：`I/O Rate Over Time` | [R2] Fig. 7-8；[R1] Fig. 9 | `--` | 保留 60 秒窗口和 25/75 percentile 分段。 | 文献对齐 |
| Burst duration、gap 和 duty cycle | 图：`I/O Rate Over Time`；数值：`phase1_metrics.json` | [R2] Fig. 7-9；[R9] Fig. 3 | `--` | 输出 universal burst summary。LLM interval 和 output-token intensity 只作为 agentic overlay。 | 分离通用指标与 agentic 指标 |
| Workflow concurrency 和 I/O-busy worker degree | 图：`Agent Timeline`；表：`Workflow concurrency (opportunity)`；表：`I/O concurrency` | [R2] Fig. 11、Table I；[R8] Fig. 2、4-5 | `--` | Traditional trace 增加 PID-to-task/stage mapping。两类 workflow 均使用 syscall overlap 计算 I/O-busy workers。 | 跨 workflow 统一 |
| Per-worker 或 per-agent I/O skew | 表：`Execution-unit I/O skew` | [R1] Fig. 8a-c | `--` | 新增 execution-unit 的 max/median 和 p95/p50，分别统计 bytes 与 I/O-busy time。 | 跨 workflow 统一 |
| Reader/writer fan-out 和 P-C classes | 图：`Reader & Writer Fan-out`；表：`Producer-consumer classes` | [R8] Fig. 2、4-5；[R10] Table I 和 workflow figures | `--` | 统一输出 `1-1`、`1-n`、`n-1`、`n-n` 的 count 和 percentage。Agentic 结果只表示观测关系，不视为静态 DAG。 | 跨 workflow 统一并限制结论范围 |
| Write-to-first-read gap、artifact lifecycle 和 dead writes | 图：`Write→Read Gap`；图：`Artifact Lifecycle` | [R8] Fig. 2、4-5 | Gap/lifetime/share `--`；dead files `W/GiB` | 输出 gap 和 reclaimable lifetime quantiles。增加 dead-file count、dead-file rate 和 dead-write byte share。 | 文献对齐并统一输出 |
| I/O by agent role/tool call | 图：`Call DAG with I/O`；表：`Bytes/ops by phase`；数据：`lineage/execution_unit_io.csv` | [R8] Fig. 2；[R10] workflow grouping | Ops 用 `R/GiB`、`W/GiB`；byte share `--` | 新增通用 `execution_unit_io.csv`。Agent role 和 tool-call 细节保留在 agentic extension。 | 分离通用指标与 agentic 指标 |
| Retry/backtrack I/O | 图：`Reread Attribution`；表：`Bytes/ops by phase` | 无 traditional counterpart；[R5] 的 deterministic fan-out 不能作为 retry 对照 | Ops 用 `R/GiB`、`W/GiB`；byte share `--` | 保留现有 agentic 指标。 | 保留 agentic 特有指标 |




## 2. Agentic workflow 还需要测量的内容

- [ ] 采集一个真实使用 HDF5 或其他 structured I/O 的 agentic workflow。
- [ ] 使用新 trace 验证 STDIO/POSIX byte attribution 和 VFS offset coverage。



## 3. Traditional workflow 实验

所有 traditional workflow 使用相同的 eBPF schema、workload filter、byte normalization 和 universal analysis。LLM、reasoning、role 和 backtrack 字段不生成。


| 优先级 | Workflow         | 代码状态                                                                                         | 运行规模                                                      | 测量重点                                                                                                                       | 文献校验                                                                            |
| --- | ---------------- | -------------------------------------------------------------------------------------------- | --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| P0  | **1000 Genomes** | Driver、sandbox、task/stage event、PID attribution、统一后处理和三档隔离测试命令已完成；300-line CloudLab smoke 通过 | 1、2、4 chromosomes；固定 `ALL` population 和 worker cap；每档 3 次 | 全部 universal patterns，重点测 task/stage share、reuse、directory rescan、state rewrite、P-C class、concurrency 和 lifecycle          | [R8] Fig. 2、4-6；[R9] Fig. 2-3、Table III；[R10] workflow figures；[R11] Sec. 5.4.5 |
| P1  | **Montage**      | Direct stage adapter、PID/time attribution、固定输入校验、common tracer 和统一后处理已实现；`0.10°` CloudLab 正式 smoke 通过 | 同一 survey 的 3 个 mosaic sizes；每档 3 次 | 全部 universal patterns，重点测 file/request-size CDF、metadata `T/GiB`、files `W/GiB`、mergeability、stage share 和 burst structure | [R5] Montage profile；[R7] Fig. 1-5；[R8] Fig. 2；[R9] Fig. 2-3、Table II |
| 候选  | **MuMMI**        | 未实现；等待代码、数据、GPU 和 structured-I/O 环境评估                                                        | 环境允许时选择 3 个原生 ensemble sizes；固定资源配置；每档 3 次                | 全部 universal patterns，重点测 interface bytes、request-size CDF、bandwidth、producer-consumer reuse、concurrency、skew 和 duty cycle | [R9] Fig. 2-3 和 MuMMI tables                                                    |


MuMMI 仅作为候选实验。代码、数据、GPU 和 structured-I/O 环境全部满足后再纳入实验。

## 4. 测试与执行计划



### 4.1 已完成的代码级验证

- [x] 冻结 common schema：绝对值、`R/GiB`、`W/GiB`、`T/GiB` 和零分母 `N/A`。
- [x] Universal metrics 可在空 `pi_events.jsonl` 的 traditional trace 上运行。
- [x] 1000 Genomes 无条件生成 lineage、parallelism、phase metrics、execution-unit summary、trace quality、图和 index。
- [x] Traditional task event 记录 task ID、stage、PID、起止时间、依赖、输入和输出。
- [x] PID 和时间区间可映射为统一 execution unit；agentic tool call 使用同一输出 schema。
- [x] 新增 byte-normalized rates、read/write CDF、read/write inter-arrival、same-file reopen、burst、skew 和四类 P-C summary。
- [x] 单元测试覆盖 classic DAG、sandbox、execution-unit event、PID attribution 和比较指标。



### 4.2 1000 Genomes 小规模 trace

- [x] 完成旧版 `1 chromosome × 2,000 lines` 校准运行。Workflow 成功；因 main/annotation scale 不一致、缺少 execution-unit metadata 且丢失 1,429 个 eBPF 样本，仅保留为 debug/cost-calibration trace。
- [x] 部署当前代码和 1,024-page perf buffer 配置，远端通过 shell 语法与 Python 内存编译检查。
- [x] 按 `Overview.md` 运行 `1 chromosome × 300 VCF lines`，不与其他 trace 并发。
- [x] 确认 `classic_run_summary.json` 中 5/5 tasks 成功，`execution_units.jsonl` 的 task 数、stage、PID 和时间完整。
- [x] 确认 `trace_quality.json` 中 read/write denominators 非零，新 trace 的 `lost_events=0`。
- [x] 检查 execution-unit byte attribution、VFS offset 和 workload-path coverage：99.84%、100%、read 29.75% / write 94.50%。
- [ ] 检查 `artifacts.csv`、P-C classes、same-file reopen、lifecycle 和 task/stage share 是否符合 1000 Genomes DAG。
- [ ] 最小档通过后，依次运行 `1 chromosome × 750 VCF lines` 和 `1 chromosome × 2,000 VCF lines`。
- [ ] 比较三档 wall time 和 trace 大小；确认 10–20 分钟档位后再固定正式规模。



### 4.3 正式 traditional experiments

- [ ] 运行 1000 Genomes 的 1、2、4 chromosomes，每档 3 次，共 9 cells。
- [ ] 每个 cell 保存 manifest、trace quality、execution-unit CSV、phase metrics、lineage 和 index。
- [x] 完成 Montage `0.10°` Lustre-only application smoke；该运行不进入正式对比。
- [ ] 完成 Montage `0.25°` 和 `0.50°` application smoke，校准文件数和运行时间。
- [x] 实现 Montage direct stage adapter、execution-unit attribution、固定输入 checksum、common tracer 和统一后处理。
- [x] 固定 `0.10°` 输入并在 trace 前通过 SHA-256 校验，排除下载和 staging I/O。
- [x] 完成 `0.10°` 正式 eBPF smoke：11/11 stages 成功、`lost_events=0`、execution-unit byte attribution 100%，生成完整 tool Index。
- [ ] 固定 `0.25°` 和 `0.50°` 输入，再运行三个规模、每档 3 次。
- [ ] 仅评估 MuMMI 的代码、数据、GPU、structured-I/O library 和 process containment；条件满足后另行决定是否实现。



### 4.4 Agentic 重算与最终比较

- [ ] 用冻结后的 schema 重算所有入选 agentic traces，生成 `execution_unit_io.csv` 和 `trace_quality.json`。
- [ ] 完成 Section 2 的 structured-I/O agentic trace，并验证 STDIO/POSIX attribution 与 VFS offset coverage。
- [ ] Comparison figure 同时显示绝对值与 byte-normalized rate；分布、比例、时长、相关性、并发度和带宽不归一化。
- [ ] 仅在定义一致时叠加文献结果；RH/WH/RW 和 Darshan size bins 可直接叠加。
- [ ] 结论仅描述通过 trace-quality 检查后的跨 workflow I/O pattern 差异。



## 5. 阅读清单

- [ ] **[R1]** T. Patel et al. *Uncovering Access, Reuse, and Sharing Characteristics of I/O-Intensive Files on Large-Scale Production HPC Systems*. FAST 2020. Fig. 3、5-6、8、10. [https://www.usenix.org/conference/fast20/presentation/patel-hpc-systems](https://www.usenix.org/conference/fast20/presentation/patel-hpc-systems)
- [ ] **[R2]** T. Patel et al. *Revisiting I/O Behavior in Large-Scale Storage Systems: The Expected and the Unexpected*. SC 2019. Fig. 7-9、11，Table I. [https://doi.org/10.1145/3295500.3356183](https://doi.org/10.1145/3295500.3356183)
- [ ] **[R3]** H. Luu et al. *A Multiplatform Study of I/O Behavior on Petascale Supercomputers*. HPDC 2015. Fig. 2-5、14-18. [https://sdm.lbl.gov/~sbyna/research/papers/201506-HPDC-iologs.pdf](https://sdm.lbl.gov/~sbyna/research/papers/201506-HPDC-iologs.pdf)
- [ ] **[R4]** J. L. Bez et al. *Access Patterns and Performance Behaviors of Multi-layer Supercomputer I/O Subsystems under Production Load*. HPDC 2022. Fig. 3-5、8-12，Table 6. [https://sdm.lbl.gov/~sbyna/research/papers/2022/2022-HPDC-](https://sdm.lbl.gov/~sbyna/research/papers/2022/2022-HPDC-Bez-IO-perf-analysis.pdf)
- [ ] **[R5]** G. Juve et al. *Characterizing and Profiling Scientific Workflows*. FGCS 2013. Cross-workflow profile tables. [https://deelman.isi.edu/wordpress/wp-content/papercite-data/pdf/juve2013characterizing.pdf](https://deelman.isi.edu/wordpress/wp-content/papercite-data/pdf/juve2013characterizing.pdf)
- [ ] **[R6]** F. Nawaz et al. *Performance Analysis of an I/O-Intensive Workflow Executing on Clouds*. APDCM 2016. Fig. 1-5. [https://deelman.isi.edu/wordpress/wp-content/papercite-data/pdf/nawaz-apdcm-2016.pdf](https://deelman.isi.edu/wordpress/wp-content/papercite-data/pdf/nawaz-apdcm-2016.pdf)
- [ ] **[R7]** H. Lee et al. *Data Flow Lifecycles for Optimizing Workflow Coordination*. SC 2023. Fig. 2、4-6. [https://cs.iit.edu/~scs/assets/files/lee2023data.pdf](https://cs.iit.edu/~scs/assets/files/lee2023data.pdf)
- [ ] **[R8]** O. Kogiou et al. *I/O Characterization of Heterogeneous Workflows*. SC 2024 poster. Fig. 2-3 和 workflow tables. [https://sc24.supercomputing.org/proceedings/poster/poster_files/post222s2-file2.pdf](https://sc24.supercomputing.org/proceedings/poster/poster_files/post222s2-file2.pdf)
- [x] **[R9]** M. Tang et al. *Characterizing Dataflow for I/O-Aware Scheduling in HPC Workflows*. IPDPS 2026. Table I、Fig. 5 和 workflow P-C analyses. [https://doi.org/10.1109/IPDPS65963.2026.00076](https://doi.org/10.1109/IPDPS65963.2026.00076)
- [ ] **[R10]** I. Yildirim et al. *WisIO: Automated I/O Bottleneck Detection with Multi-Perspective Views for HPC Workflows*. ICS 2025. Sec. 5.4.5、Fig. 20. [https://akougkas.io/assets/pdf/wisio.pdf](https://akougkas.io/assets/pdf/wisio.pdf)
