# Time Taxonomy & Resource Breakdown — Plan

状态:设计已定,待实现。本文记录我们对"agentic workflow 里到底有几种时间"的讨论结论,
以及由此驱动的画图重构与 tracer 增强计划。

---

## 1. 目标(Why)

回答一个核心问题:**在给定 workload 下,agent 的时间花在哪、谁是瓶颈,以及那段时间到底在
消耗什么资源(算 / 文件 IO / 等待)。**

现有口径太粗,只分两类:
- LLM 时间
- 非 LLM 时间(被叫作 "code execution")

且 LLM 时间是用**残差**算的:`llm_time = overall − tool_time`
([src/summarize_pi_events.py:258](../src/summarize_pi_events.py#L258))。

这个残差是最大的 overclaim:所有"既不是 LLM、又没被工具计时器抓到"的时间(框架开销、
排队、未归类 syscall、调度延迟)全被塞进了 LLM。同时 "code execution" 也轻微 overclaim——
它其实是工具的**墙钟**,里面大头往往是文件 IO 和等待,而不是 CPU 计算。

### 1.1 时间的两个正交维度

讨论的结论:时间要按**两个正交维度**理解,不要混。

- **维度 A — 角色(谁在消耗 / 原因)**:LLM inference、Tool/code execution、Orchestration、Idle。
- **维度 B — 资源(消耗了什么)**:File-IO、CPU compute、Wait/idle(blocking)。
  - 注意:**网络 IO 不在本项目考虑范围**(用户明确)。LLM 调用本质上对本机就是 network-wait,
    我们不拆它的资源,整段当黑盒标 "LLM"。

### 1.2 类别(最终)

**核心原则:能测到的一律单独列,残差只留给真正测不到的"暗时间"。**

任何被归属的时间 = 一个**角色**(谁)× 一个**资源**(消耗什么)。两维都尽量实测;
拼不出来的才进 `Unaccounted`。

**角色(维度 A)** —— 都按 PID 归属:

| 角色 | 含义 | 资源是否再分解 |
|---|---|---|
| `LLM (API)` | 一次 completion 调用的客户端墙钟,黑盒 | 否(整段当 LLM) |
| `Tool / code-exec` | 真正执行代码的工具(CodeExec/ScriptExec)等叶子工具 | 是 → File-IO/CPU/Wait |
| `Orchestration` | 框架自身的工作(拼 prompt、解析 response、路由、日志),跑在 orchestrator PID 上 | 是 → File-IO/CPU/Wait |
| `Process-mgmt` | **单列**:clone/execve/wait4(起子进程、等子进程) | 自成一类 |
| `Unaccounted` | **纯残差**:wall − 所有已归属时间。测不到的暗时间,随埋点完善而缩小,本身就是覆盖率指标 | — |

**资源(维度 B)** —— 资源维度上**互斥**(一个 CPU 时刻只可能在算、在等、或在 syscall 里):

| 资源 | 含义 | 怎么来 |
|---|---|---|
| `File-IO` | read/write/openat/stat/... 的 syscall latency | eBPF |
| `CPU compute` | 真在核上跑 | 残差 **或** sched_switch 实测(二选一,见 §3.3) |
| `Wait (blocking)` | epoll/futex/poll/select/sleep 的阻塞 | eBPF |

> **关于 Orchestration(回应"能测到的就别塞残差"):**
> Orchestration **不是**纯残差。它是 orchestrator 进程自己的真实活动,可被 eBPF 测出:
> 它的 File-IO(读 config、写日志)、Wait(futex/epoll 等子进程或等 LLM 返回)、
> CPU(拼 prompt / 解析 response 的字符串处理,= 该 PID 的 syscall 间隙)。
> 这些都按 §3 的同一套方法实测并分解。
> **真正的纯残差只剩 `Unaccounted`** —— 墙钟在走、但没有任何 LLM / 工具 / syscall 信号的暗时间。

---

## 2. 要画什么图(What)

**两类图,各答一个问题,互补且都不撒谎。** 这套结构现已存在,我们是在它上面增强,不是重写。

### 2.1 饼图 ×2 — 角色时间占用(已存在)

代码:`create_phase_breakdown_plotly` / `_matplotlib`
([src/visualize_strace.py:1740](../src/visualize_strace.py#L1740))。

并行(`max_active=2`,`concurrency≈1.86`)导致"角色时间之和 ≈ 2× 墙钟",这是正常的,
靠**两张饼从两个角度**表达。**两张都要改成新类别,不再是两类。**

- **饼 1 — Sum view(工作量基准)**:**所有类别**的 self-time sum,不再只有 LLM vs Tool。
  切片 = `LLM / File-IO / CPU / Wait / Process-mgmt / Orchestration(分解后)/ Unaccounted`
  (即 §1.2 的角色×资源全集)。
  - 用 `compute_self_intervals` 减掉嵌套(子 agent 的 LLM 不再重复计进父 tool)。
  - 中心数字 = phase span = 各类别之和。回答"总精力按类别怎么分"。
- **饼 2 — Wall view(墙钟基准)**:e2e 墙钟精确 tiled。**不枚举所有重叠组合**(角色多了,
  两两/三三重叠组合爆炸),改成 **n+1 片**:
  - **每个角色的独占时间**(该时刻只有这一个角色在跑)—— n 片。
  - **并行时间**(该时刻 ≥2 个角色/lane 同时在跑)—— **统一 1 片**(就是那个 +1)。
  - `Idle`(无任何活动)始终画出,0% 也保留(本身是信息)。
  - 这些片精确铺满 e2e。回答"墙钟里各角色独占 vs 并行 vs 空闲各占多少"。
- 底部:`speedup = phase span / e2e`(1.0× 全串行,≈N× = N worker 持续忙)。

**改动:Sum view 从 2 类扩到全类别(LLM 改实测口径,Tool 按资源细分,Orchestration 分解,
Process-mgmt 单列,Unaccounted 露出);Wall view 从 `LLM only/Tool only/LLM+Tool/Idle`
改成 `各角色独占 ×n + 并行 ×1 + Idle`。**

### 2.2 甘特图 — 每 agent 一条 lane,堆叠资源段(要改 / 要加)

代码:agent timeline / agent concurrency
(`_load_agent_timeline_data`、`_agent_concurrency_data`,
[src/visualize_strace.py:1936](../src/visualize_strace.py#L1936) 附近)。

- **一个 agent / worker 一条 lane**(水平时间轴 = 真实墙钟)。
- 每条 lane 内,按时间堆叠资源段:`LLM / File-IO / CPU compute / Wait / Process-mgmt / Orchestration`。
- **跨 worker 的并行靠"多条 lane 同时有色块"表达**,lane 之间互不吃时间。
- **每条 lane 的 idle 必须显式画成空白**:agent 在等别的 agent、或在等 LLM 返回的空隙。
  否则 lane 内资源段之和 ≠ 该 lane 的墙钟跨度。

这是 §1.2 各类别真正落地、能看出"IO-bound vs compute-bound vs wait-bound"的地方。

---

## 3. 怎么测出来的(How)— 每个桶的数据来源与去重规则

### 3.1 LLM (API)

- **来源**:`pi_events.jsonl` 的 `message_start`/`message_end`,即 litellm 的
  `start_time`/`end_time`([src/litellm_tool_logger.py:290](../src/litellm_tool_logger.py#L290))。
- **包含**:请求序列化、网络 RTT、服务端排队 + prefill(TTFT)、decode/streaming、litellm 解析 response。
- **不包含**:completion 调用**之外**的拼 prompt、解析 tool call、状态更新 → 落 orchestration。
- **不往里塞资源桶**:整段黑盒(用户不拆网络)。

### 3.2 File-IO / Wait — 来自 eBPF syscall latency

- 每个 syscall 带 `latency_ns`(线程待在 syscall 里的墙钟,含阻塞),已现成
  ([src/bcc_tracer.py](../src/bcc_tracer.py)、[src/parse_ebpf.py:611](../src/parse_ebpf.py#L611))。
- `SYSCALL_CATEGORIES`([src/visualize_strace.py:131](../src/visualize_strace.py#L131))
  重映射到资源桶:
  - `metadata + data + control + modify` → **File-IO**
  - `blocking`(epoll/futex/poll/select/sleep)→ **Wait**
  - `process`(clone/execve/wait4)→ **单列 `Process-mgmt`**(已定)
  - `network` → 折叠/忽略(不在范围)

### 3.3 CPU compute — 两种模式是**二选一**,不是相加

> **回应"两种模式能同时用吗":不能,也不会相加(相加=重复计)。** 它们是同一个数字的两种
> **测法**,由"这次 trace 跑没跑 sched_switch"决定用哪个。分析代码两种输入都支持,
> **有实测就用实测,没有就回退残差**。永远只取其一进图。

- **模式 A — 残差(默认,sched_switch 关时)**:per 叶子 code-exec 工具,细粒度。

  ```
  CPU_compute(这次 CodeExec) = wall − Σ(它的 File-IO latency) − Σ(它的 Wait latency)
  ```

  - 只对**真正执行代码的工具**(CodeExec / ScriptExec)算;Read/Write 等纯 IO 工具残差≈0,不算 CPU。
  - 数据已支持:`compute_tool_summaries` 本就是 per-tool `by_syscall`
    ([src/parse_ebpf.py:681](../src/parse_ebpf.py#L681)),只需把 File-IO 与 Wait 拆成两桶。
  - **已知缺陷**:残差里混了未 trace 的 syscall、调度抢占、mmap 页错误;且**天花板是单核墙钟**,
    多核(numpy/BLAS)会系统性低估 CPU。诚实标 "CPU compute (residual estimate)"。

- **模式 B — sched_switch 实测(开 `--cpu-sched` 时,见 §5.1)**:直接量 on-CPU 时间,
  多核可 > 墙钟,修掉模式 A 的低估。

- **二者关系**:开了 sched,CPU 这一桶**取实测值**,残差**不再用于 CPU**(此时残差只剩
  Orchestration / Unaccounted)。没开 sched,CPU 用残差。**同一张图里 CPU 只来自一个来源。**
  实测可顺便和残差对照(差值揭示多核/未 trace IO 的规模),但那是诊断,不进堆叠。

### 3.4 去重规则(关键)

- **嵌套(LLM in tool)**:只在**叶子层**记账。子 agent wrapper 的 tool 时间不算 code execution,
  其未被叶子覆盖的部分算 orchestration。Sum view 已用 self-time 做了这件事。
- **LLM 窗口的 syscall**:**per-PID / per-lane** 去重,**不是 per-wall-clock**。
  - parse_ebpf 本就按 PID 归属(`matched_tool_call` 走进程树,
    [src/parse_ebpf.py:540](../src/parse_ebpf.py#L540))。
  - 同一 PID 在 LLM 窗口内的 epoll/futex 被 LLM 段吸收,不再单列 Wait。
  - **另一个 worker 的真实 IO 在它自己的 lane,不受影响** —— 这就是为什么必须 per-lane。

---

## 4. 现状盘点:已有 / 要改 / 要加(第一批 — 便宜必做)

### 4.1 已经有了(无需新埋点)

| 能力 | 位置 |
|---|---|
| LLM 真实 start/end | `pi_events.jsonl`(litellm callback) |
| 文件 IO syscall + latency | eBPF `read/write/openat/stat/...` |
| blocking/wait syscall + latency | eBPF `epoll_wait/futex/poll/select/sleep/wait4` |
| 进程 fork/exec | eBPF `clone/execve` |
| per-tool syscall 聚合 + wall vs syscall gap | `compute_tool_summaries` |
| 按 PID/进程树归属 syscall 到工具 | `match_event_to_tool` |
| 时钟对齐(tz 校准) | `_compute_tz_offset` |
| 两张角色饼 + speedup | `create_phase_breakdown_*` |
| per-agent lane 甘特 / 并行图 | `_agent_concurrency_data` 等 |

### 4.2 要改

1. **LLM 时间退残差,改实测**:`summarize_pi_events.py` 用 `pi_events.jsonl` 的
   message_start/end 计算 LLM 时间。残差**不再叫 LLM**:能测到的归 `Orchestration`(orchestrator
   PID 的 File-IO/CPU/Wait),真正测不到的归 `Unaccounted`。
2. **syscall → 资源/角色桶映射**:在聚合层把 `SYSCALL_CATEGORIES` 重映射成
   `File-IO / Wait / Process-mgmt`,而不只是按 syscall 名。
3. **per-leaf-tool 资源分解**:每个 code-exec 工具算出 File-IO / CPU / Wait 三段。
4. **Orchestration 也实测分解**:orchestrator PID 的 syscall 按同套映射出 File-IO/CPU/Wait;
   只把测不到的留给 `Unaccounted`。
5. **per-lane 去重**:LLM 窗口内同 PID 的 blocking syscall 被 LLM 吸收。
6. **甘特 lane 堆叠资源段**(LLM/File-IO/CPU/Wait/Process-mgmt/Orchestration)+ **显式画 idle 空白**。
7. **饼图改新类别**:Sum view 扩到全类别;Wall view 改成 `各角色独占 ×n + 并行 ×1 + Idle`。

### 4.3 要加(tracer,纯加 case,低成本高回报)

1. `readv / writev / preadv / pwritev` —— 向量化 IO(numpy/h5py/pandas)。
   **现在 tracer 没抓但 parse 已分类**,等于空桶,必补。
2. `fsync / fdatasync / sync_file_range` —— Lustre 持久化 stall,写密集步骤的隐藏瓶颈。

---

## 5. 第二批增强(更准,按需,可独立上)

各治一病,互不依赖。

### 5.1 sched_switch 实测 CPU(修多核低估)

- **原理**:tracepoint 在每次上下文切换触发,带 `prev_pid / next_pid / cpu / 时间戳`。
  对每个核记"当前线程何时被切入",切出时用现时间减去得到该轮 on-CPU 时长,按 TID 累加。
  多核各自累加 → CPU-seconds 可 > 墙钟,正是残差测不到的。
- **与 syscall latency 互补**:阻塞 IO 期间线程被切出(不计 CPU),计算期间在核上(计 CPU),
  两者不重叠,可直接并排堆。
- **overhead**:事件高频(满载多核每秒可数十万次切换)。**唯一关键:内核内用 BPF_HASH 聚合,
  用户态只一次性读总数,绝不逐事件 perf_output**;handler 第一步查 `tracked_pids` 早过滤。
  这样典型 overhead 个位数百分比(BCC `cpudist`/`runqlat` 同款)。逐事件落盘则会到几十个百分点,禁止。
- **测准的坑**:
  - 按 **TID** 累加再归并到 tgid(多线程 BLAS 否则漏非主线程)。
  - sched_switch 里同时用 `next_pid` 初始化起跑时间戳,避免首段算错。
  - 时钟用 `bpf_ktime_get_ns()`,与现有 syscall 同时钟,天然对齐。
  - 若甘特要在工具窗口内分 CPU,map key 用 `(tid, time_bucket)` 保留时间分辨率。
- **建议**:做成开关(如 `--cpu-sched`),不强制所有 trace 背这份数据量;残差作默认。

### 5.2 major page-fault(修 mmap 文件 IO 隐身)

- **为什么**:mmap/memmap 文件读**不产生 read syscall**,IO 发生在访问内存的 **major page fault**;
  现在这部分时间被错误算进 CPU 残差。
- **抓什么**:major page fault(真正从磁盘/Lustre 取页;minor fault 是内存命中,不算 IO),
  其耗时即 mmap 文件的真实 IO 等待。**只加 `mmap` syscall 没用**——抓不到读时间。
- **overhead**:只有缺页才触发,远低于 sched_switch,可忽略;同样 map 聚合。
- **前置确认**:是否现在做取决于 code-exec 是否大量用 `numpy.memmap` / `h5py`(mmap 模式)/ `mmap`。
  若主要是普通 `open/read`,则 §4.3 补完后覆盖已够,page-fault 可缓。**待确认 GEOAgent 的读数据方式。**

---

## 6. 落地顺序

**第一批(便宜、必做)**
1. tracer 补 `readv/writev/preadv/pwritev` + `fsync/fdatasync/sync_file_range`。
2. parse/summarize:LLM 改实测、syscall 重映射(File-IO/Wait/Process-mgmt)、per-leaf-tool 资源分解、
   Orchestration 实测分解、Unaccounted 露出、per-lane 去重。
3. 甘特每 lane 堆叠 `LLM / File-IO / CPU(residual) / Wait / Process-mgmt / Orchestration` + 显式 idle 空白;
   饼图改新类别(Sum view 全类别;Wall view = 各角色独占 ×n + 并行 ×1 + Idle)。

**第二批(更准、按需)**
4. `sched_switch` 实测 CPU(修多核低估,**做成 `--cpu-sched` 开关**)。
5. major page-fault(修 mmap IO 遗漏)。

## 7. 已定决策

- [x] `process`(clone/execve/wait4)→ **单列 `Process-mgmt`**。
- [x] sched_switch → **做成开关**(`--cpu-sched`),残差作默认,二选一不相加。
- [x] Orchestration **不是纯残差**:能测到的(orchestrator PID 的 File-IO/CPU/Wait)单独列,
      纯残差只留给 `Unaccounted`。

## 8. 待定决策(实现前敲定)

- [ ] GEOAgent / code-exec 是否大量 mmap/memmap 读 → 决定 §5.2 是否进第一批之后立刻做。
