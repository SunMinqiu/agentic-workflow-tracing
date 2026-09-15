# Agentic Scientific Workflow I/O：研究设计

运行步骤见 `agentic_workflow_manual.md` 和 `traditional_workflow_manual.md`。

## 1. 目标与范围

本项目回答一个问题：LLM agent 控制科学工作流时，探索、重试和动态工具选择会增加多少文件系统 I/O？

研究对象是 agent 与科学任务产生的文件系统 I/O。模型加载、KV cache paging、模型 offloading 和 serving 内部存储不在范围内。固定输入重放只用于解释推理缓存，不纳入文件系统 I/O 对比。

## 2. 研究问题

1. 每个 workflow 的读写字节、操作数、元数据操作和文件数是多少？
2. 其中多少来自 agent 的探索、检查、失败和重试？
3. 次优 I/O 配置来自 agent 生成代码还是固定脚本？
4. 同一任务重复运行时，I/O 总量和归因结果是否稳定？
5. 有传统脚本基线时，agentic 运行增加了哪些 I/O？

## 3. 目标系统

| 系统 | 编排方式 | 实验标签 |
| --- | --- | --- |
| GenoMAS | 固定 Action Unit 顺序，agent 生成阶段代码 | 固定 |
| CMBAgent | LLM 生成计划，控制器按计划分派 | 半动态 |
| SRAgent | Supervisor 动态选择子 agent 和工具 | 动态 |
| ChemGraph | Supervisor 动态选择子 agent 和工具 | 动态 |
| SciLink autonomous | 动态选择技能并决定 refinement 次数 | 动态 |

传统基线是 1000 Genomes 和 Montage，由固定脚本驱动，不含 LLM。

每份实验报告必须记录系统、版本、模型、任务、编排标签、worker 数、输入规模和重复编号。

## 4. I/O 归因

每个 I/O 单元只归入一类：

| 类别 | 判定标准 | 例子 |
| --- | --- | --- |
| Agent-induced | 没有本次 agent 行为就不会发生 | 重试、读错误日志、重复检查、废弃输出 |
| Task-misconfigured | 保持任务语义不变时，存在已验证的低 I/O 配置 | 逐文件读取、重复解析、频繁重写元数据 |
| Workflow task-induced | 不属于前两类 | 读取输入、写最终结果 |

Task-misconfigured 再标记来源：`agent-caused` 表示配置来自 agent 生成代码，`script-caused` 表示配置来自固定脚本。只有补丁前后实验或等价的局部反事实证明 I/O 降低后，才能标记为 Task-misconfigured。

归因用 `pi_events.jsonl` 和 `tool_calls.log` 中的时间区间连接 `parsed.json` 的 I/O 事件。无法从 provenance 证明成因的事件保留为 Workflow task-induced，并在报告中记录限制。

## 5. 指标

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

## 6. 实验设计

先运行最小 workload。只有第 8 节的检查全部通过，才扩大输入或增加 workflow。

对比规则：

- 有相同科学目标的传统实现时，对比传统实现与 agentic 实现。
- 没有传统实现时，只比较同一系统、任务和输入的重复运行。
- 次优配置使用补丁前后对比。除待测配置外，输入、worker 数、模型和运行环境保持一致。

优化案例只选择证据完整且可复现的。每个案例必须给出原始配置、修改内容、语义等价检查和 I/O 差值。

## 7. 固定输入重放

重放比较 vLLM 启动参数对前缀缓存、prefill 和 decode 的影响。重复运行 agent 会改变调用次数、prompt 和工具顺序，所以重放先固定一次真实运行的 prompt token 序列，再把相同输入发给不同配置的 vLLM。重放不运行 agent，也不采集 BCC trace。

三个术语：

- Bundle：从一个 cell 提取的固定请求集合，用 vLLM 的 `/tokenize` 接口生成 token ID。
- Arm：在一种 vLLM 配置下重放 bundle 得到的结果。
- Sweep：使用同一 bundle 记录的多个 arm。

### 7.1 请求节奏

- `packed` 移除原运行中 agent 和工具执行产生的空档，以原 trace 的峰值并发数连续发送请求。用于机制、容量和 block size 扫描。
- `paced` 按原 trace 的请求到达时间发送。用于 retention 和定时清缓存扫描。
- `serial` 一次只发一个请求，用于隔离 intrinsic TTFT、server queue 和 cache hit 的关系。

同一个 sweep 不混用节奏。

### 7.2 缓存控制

工具默认在每次重放前调用 `/reset_prefix_cache`，每个 arm 从冷缓存开始。这需要 vLLM 以 `VLLM_SERVER_DEV_MODE=1` 启动。无法清缓存时用 `--keep-cache` 且只跑一次。

`--cache-salt-mode shared` 让 bundle 内所有请求共享一个不进入模型输入的 cache namespace。`--cache-salt-mode unique` 为每个请求使用不同 salt，在 prefix caching 保持开启时禁止跨请求复用。两者 token ID 相同，因此 unique 对 shared 的差值就是跨请求复用本身的影响。

`--warmup-repetitions 1` 在正式测量前完整重放一次 bundle。预热和正式测量共用 cache salt。正式 summary 和 Prometheus delta 不含预热，`warmup0_requests.jsonl` 保留预热的逐请求结果。

### 7.3 配置校验

每个 repetition 保存 `server_info.json`，含 vLLM version、served models 和完整 `VllmConfig`。`--strict-config` 比较完整 `VllmConfig` 和 Prometheus `CacheConfig`，只允许 `--allow-config-change` 指定的 knob 及其派生的 `num_gpu_blocks`、`kv_cache_size_tokens` 和 `kv_cache_max_concurrency` 变化。其他字段变化时，replay 在发送请求前停止。

vLLM 0.26 关闭 prefix caching 时会把 hybrid 模型的 `mamba_cache_mode` 设为 `none`，开启时为 `align`。因此 prefix OFF/ON 对照测的是完整 feature 的开关，包含这项 Mamba 派生变化。只想测跨请求复用时用 cache salt 对照，server 不重启。

### 7.4 观测开销

Perlmutter 侧的 `config_vllm_observability.sh` 启用逐请求 cached tokens、逐请求 server timing、KV residency histograms、调度 iteration 日志、prefix-cache reset 和 KV block events。replay 保存原始 `kv_events.jsonl`，并把 `BlockRemoved` hash 与此前的 `BlockStored` 对齐后写入 `kv_removals.jsonl`。Mamba 或 SSM group 的事件标为 ambiguous。

正式 latency 运行应设 `VLLM_KV_METRICS_SAMPLE=0.01`，并移除 iteration logging 和 KV events。全量采样只用于诊断 eviction。

### 7.5 结果解读

Arm 目录默认按服务器配置和纽约时间命名，`--label` 可覆盖。已有目录不会被覆盖。

报告中 `median request latency` 是从发出请求到收完响应的中位数，`wall` 是整组请求的墙钟时间。出现 `Not comparable` 时，bundle、调用数、总输入、输出长度、节奏或缓存状态不一致。出现 `Every arm reports the same serving config` 时，vLLM 没有用新配置启动。

TTFT 在 Mac 上测量，包含 Mac 到 Perlmutter 的隧道延迟。所有 arm 使用相同网络路径时可以比较相对差异。

## 8. 输出与验收

本地结果存放在 `results/<run>/<cell>/`，远端在 `/mnt/lustrefs/<user>/pi-ebpf-tracing-handoff/results/`。

每个 cell 必须包含：

| 输出 | 用途 | 通过标准 |
| --- | --- | --- |
| `manifest.json` | 运行配置 | JSON 可解析 |
| `ebpf_events.log` | 原始 trace | 非空 |
| `parsed.json` | 解析后的 I/O | 非空；超过 100 MB 时留在远端 |
| `phase1_metrics.json` | 聚合指标 | 非空 |
| `parallelism_summary.json` | 并发统计 | 非空 |
| `trace_quality.json` | 采集质量 | 非空 |
| `lineage/artifacts.csv`、`lineage/execution_unit_io.csv` | 归因数据 | 非空 |
| `bcc.err` | tracer 状态 | 最后一行含 `lost_events=0` |
| `visualizations/index.html` | 单 cell 报告 | 非空 |

GenoMAS 和 SciLink 还必须包含 `pi_events.jsonl`、`tool_calls.log`、`messages.jsonl`、`kvcache_demand.json`、`kvcache_logical.json` 和 `kvcache_report.html`。

`scripts/pull_run.sh` 在拉回时执行以上检查，任何一项失败都不打开报告，该 run 不进入分析。

## 9. 交付结果

1. 不同编排固定度下的 I/O 测量结果。
2. Agent-induced、Task-misconfigured 和 Workflow task-induced 的归因结果。
3. 可复现的优化案例及补丁前后差值。
