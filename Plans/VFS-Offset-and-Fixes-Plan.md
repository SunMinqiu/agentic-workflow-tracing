# 计划 2：VFS offset 采集（Access Pattern）+ 已实现代码的缺陷修复

前置：`Plans/Index-Rework-Plan.md` 的内容已实现。本计划接在它后面，分两条线：

- **A 线（离线，改分析层）**：修复已实现代码里四个会让结论出错的缺陷。不需要重跑 trace。
- **B 线（重采，改采集层）**：加 VFS offset，把 sequential/random 从死代码变成真指标。**必须重跑 trace**，现有 20 个 trace 无法回填。

两条线互不阻塞。

---

## A 线 · 已实现代码的缺陷修复

### A1. `compute_io_vs_inference` 没有做 workload 过滤（严重）

[phase1_metrics.py:1253](../src/agent_io_tracing/analysis/phase1_metrics.py#L1253) 遍历 `fs_entries` 时只筛了 syscall 类型，**没有调 `make_workload_filter`**——而隔壁的 `compute_effective_bandwidth`（[:1353](../src/agent_io_tracing/analysis/phase1_metrics.py#L1353)）是筛了的。两个函数口径不一致。

**为什么这个特别毒**：漏进来的不只是 `.venv` 的 import 读，还有**我们自己 tracer 的写**——`pi_events.jsonl`、`tool_calls.log` 是 launcher 进程写的，launcher 在 tracked pids 里。而这些写**恰好发生在 LLM 调用的开始和结束时刻**（每收到一个 LLM 事件就 append 一行）。于是它们会被系统性地记进 "inference-busy" 区间，**`pct_bytes_during_inference` 和 `bandwidth_ratio` 这两个我们最想拿来做 finding 的数，正好被我们自己的测量行为污染**。这是典型的观测者效应，必须修。

**修**：`compute_io_vs_inference` 加 `artifacts` 参数，用 `make_workload_filter(artifacts)` 过滤 path，和 `compute_effective_bandwidth` 完全一致。`build_metrics` 里的调用点同步改。

### A2. `duty_cycle` 的分母是错的（严重）

[`_bandwidth_stats`:1322](../src/agent_io_tracing/analysis/phase1_metrics.py#L1322) 里 `wall_s = max(end) - min(start)`，取的是**该组自己那些 I/O 事件的时间跨度**。

于是：一个持续 300 秒的 phase，如果它的 I/O 全挤在头 2 秒，`busy_s ≈ 2`、`wall_s ≈ 2`，算出来 **duty_cycle ≈ 1.0**——"存储一直在忙"。而真相恰恰相反：这个 phase 有 298 秒存储完全 idle，duty cycle 应该是 0.7%。**这个 bug 会把我们整个论点（"agentic workflow 的存储 duty cycle 极低"）算成反的。**

**修**：每个分组的 wall 时间必须来自**该组的真实时间边界**，不是它的 I/O 事件跨度：

- `by_phase`：phase 的起止用 `read_event_phase_index`（已有）里该 phase 的时间窗；多段不连续的 phase 用各段时长之和。
- `by_role`：该 role 名下所有 tool-call 区间的**并集长度**。
- `by_inference_state`：`inference_busy` 用 LLM 区间并集长度，`inference_idle` 用 run wall − 该并集。
- `global`：用 run 的真实 wall clock（`parallelism_summary.json` 的 `wall_clock_s`，或 trace 首尾事件时间戳）。

`busy_time_s`（I/O 区间并集）和 `io_time_s`（Σduration）的算法不用动，它们是对的。

### A3. `pct_time_in_inference` 的分母偏小

[:1277](../src/agent_io_tracing/analysis/phase1_metrics.py#L1277) 的 `wall_start / wall_end` 是从 **I/O 事件 + LLM 事件的端点**里取 min/max 得到的。run 开头的初始化、结尾的清理（既没 I/O 也没 LLM 的时间）被排除在分母之外，`pct_time_in_inference` 因此偏大。

**修**：分母统一用 run 的真实 wall clock（同 A2 的 `global` 口径），三个函数（`compute_io_vs_inference`、`compute_effective_bandwidth`、`compute_intensity_phases`）**共用同一个 run 时间轴**，不各算各的。

### A4. I/O 字节按"结束时刻"归属 inference 区间（轻，但要声明）

[:1268](../src/agent_io_tracing/analysis/phase1_metrics.py#L1268) / [:1280](../src/agent_io_tracing/analysis/phase1_metrics.py#L1280)：一次 I/O 的全部字节按它的**结束时间戳**落在哪一侧来归属。跨越 LLM 边界的长 op 会被整体记到一侧。

我们的 op 基本都是毫秒级，影响很小。**修法二选一**：按区间重叠比例分摊字节（准确但复杂），或保持现状但在 caveat 里写明"按 op 结束时刻归属，op 时长远小于 LLM 时长时误差可忽略"。**推荐后者**，把复杂度留给真正需要的地方。

### A5. role 提取逻辑重复（中）

[:1362-1367](../src/agent_io_tracing/analysis/phase1_metrics.py#L1362-L1367) 在 `compute_effective_bandwidth` 里现场拼了一套 role 提取启发式（`input_params.role` / `genomas_role` / phase index 三处兜底），而 lineage 里已经有 `build_role_io_attribution` 在做同样的事。两套逻辑各自演化，迟早对不上——而且 index 上这两个数会并排显示，对不上就是直接可见的矛盾。

**修**：抽一个共用的 `role_for_entry(entry, tool_calls, phases)`，两边都调它。

---

## B 线 · VFS offset 采集 + Access Pattern

### B0. 为什么必须下沉一层（结论：我们会比 Darshan 更准）

`read(fd, buf, count)` 的**参数里没有 offset**，内核用 `file->f_pos` 这个隐式游标。我们现在挂的是 `raw_syscalls:sys_enter/sys_exit` tracepoint，只能拿到 arg0..arg4——所以 offset **在这一层根本不存在**，不是我们没去取。

而 `vfs_read(struct file *file, char __user *buf, size_t count, loff_t *pos)` 的**第四个参数就是显式偏移**。kprobe 挂上去读出来即可。

**Darshan 是怎么做的（源码核对结论）**：它是 LD_PRELOAD 用户态库，**根本没问过内核**，而是自己维护一份**影子文件位置** `rec_ref->offset`：`open` 时清零、每次 read/write 后 `offset += 返回值`、`lseek` 时 `offset = lseek 的返回值`（返回值就是跳转后的绝对位置，一行代码统一处理 SEEK_SET/CUR/END）、`pread/pwrite` 用参数里的显式 offset。也就是说，**Darshan 用的正是"按 fd 记账"这条路**，并且为此吃下三个真实失真（均在 `darshan-runtime/lib/darshan-posix.c` 里可验证）：

1. **同一文件被 open 两次，两个 fd 共用一份影子位置**：`POSIX_RECORD_OPEN` 按**路径**查 `rec_id_hash`，命中就复用同一个 `rec_ref` **并把 offset 重置为 0**——第二次 open 会把第一个 fd 的游标也清零，两个 fd 交错读互相踩。
2. **`pread` 污染后续 `read` 的位置**：`POSIX_RECORD_READ` 宏里 `rec_ref->offset = this_offset + __ret` 是无条件执行的，而真正的内核里 **pread 不动 `f_pos`**。同一 fd 混用 pread 和 read，Darshan 后续所有 offset 都错。
3. **`O_APPEND` 无任何特殊处理**（全文件 grep 不到）：追加打开已有文件时影子位置从 0 起，而数据实际落在 EOF——绝对偏移错（有意思的是 seq/consec 的**判定结果**碰巧还对，因为纯追加流看起来就是连续写：**结论对、坐标错**）。

`dup/dup2/dup3` 它处理对了（新 fd 别名到同一个 `rec_ref`，因为 dup 在内核里共享 file description、也就共享 `f_pos`）。

**我们挂 VFS kprobe 拿的是内核马上要用的那个 `*pos` 真值**，所以上面三个坑**结构上就不存在**：不同 fd 各有各的 `struct file`；pread 传的是栈上临时变量、不碰 `f_pos`；O_APPEND 写完内核会把 `f_pos` 回写成真实落点（只有**每个 fd 的第一次追加**可能读到陈旧的 `f_pos`，用已有的 `open_flags` 标出来排除即可）。而且我们**根本不需要跟踪 lseek**。

**双方一样躲不掉的**（写作时必须声明）：

- **mmap 读看不见**：走缺页中断（`filemap_fault`），既没有 read syscall 也不经过 `vfs_read`。Darshan 也只能数一下 `mmap` 调用次数。
- **缓冲层语义**：CPython 的 `BufferedReader` 会把 `f.read(10)` 攒成 8KB/128KB 块再下发，所以我们看到的"顺序"是**缓冲层下发的顺序**，不是应用逻辑的访问顺序。Darshan 在 POSIX 层看到的是同一个东西——所以**口径可比**，但都不能宣称成"agent 的逻辑访问模式"。
- **page cache 在 VFS 之下**：拿到 offset **不等于**拿到设备侧带宽，也不等于物理连续性。逻辑顺序 ≠ 物理顺序。effective bandwidth 的 page-cache caveat **不因这次改动而放宽**。

### B1. 采集改动（`tracing/bcc_tracer.py`）

1. `struct syscall_state_t` 和 event struct 各加 `u64 file_offset` + `u8 offset_src`（0=无 / 1=vfs kprobe / 2=pread 参数）。
2. 新增 `kprobe__vfs_read` / `kprobe__vfs_write`：
   - 先 `is_tracked(pid)` 过滤（和现有探针一致，最先做，省开销）；
   - `inflight.lookup(&pid_tgid)` 拿到**已存在的** in-flight syscall 状态；
   - `bpf_probe_read_kernel(&off, sizeof(off), (void *)PT_REGS_PARM4(ctx))` 读出 `*pos`；
   - **只在 `offset_src == 0` 时写入**（守卫）：万一同一次 syscall 里 `vfs_read` 被调了两次，要记**第一次**的起点，无脑覆盖会记成最后一次。
3. 同样挂 `vfs_iter_read` / `vfs_iter_write`，覆盖 `readv`/`writev`（它们走 iter 路径，**不经过 `vfs_read`**，不挂就静默丢失）。符号不存在时 **warn 而非 fatal**（和 `attach_hpc_io_probes` 现有的容错风格一致）。
4. `pread64` / `pwrite64` 的 offset 直接从 `arg3` 取，不必走 kprobe（顺带把 SciLink 现有 trace 里那 342 条 pread 救回来——**这部分是唯一能回填到旧 trace 的**）。

**关键：kprobe 不发事件。** 它只往已有的 in-flight 状态里塞一个字段，事件仍然只在 `sys_exit` 出口发出一条。所以：事件条数不变 → `fs_entries` 数量不变 → **所有现有的字节统计、op 计数、直方图结果一个字都不会变**；路径归属仍走现有 fd→path 映射（VFS 层拿不到路径，`bpf_d_path` 在 `vfs_read` 上不在内核 allowlist）；`matched_tool_call` 归属逻辑不碰。

**与现有探针的交互，四点**：

- kprobe 会为"没有 in-flight 记录"的调用触发（内核线程、io_uring worker、我们白名单外的 `sendfile`/`splice`）→ 查不到就 return，**是遗漏不是重复**，安全。
- 和 libc `fread`/`fwrite` uprobe（[bcc_tracer.py:685](../src/agent_io_tracing/tracing/bcc_tracer.py#L685)）不冲突：那是**同一次 I/O 在 STDIO 层的另一个视图**，现有的 `compute_measured_interface_layers` 已经在处理"别数两遍"。新的 offset 字段只挂在 syscall 事件上、不进任何字节汇总，**不引入新的重复计数**。
- `vfs_read` 也服务 pipe / `/proc` / tty，它们的 offset 无意义 → sequentiality 计算**必须复用 `is_storage_file_io()` + workload 过滤**，否则 `/dev/urandom` 那种读会污染分布。
- **开销**：每条 read/write 多一个 kprobe（动态插桩，比 tracepoint 贵）。GenoMAS 一个 cell 才 5607 次读，可忽略；fanout 大 run 可能上百万次 → **先用 `agent-trace-w0`（`experiments/w0_microbench`）量一次开销再上真 workload**，不要直接上 GenoMAS。

### B2. 解析改动（`parsing/_ebpf_impl.py`）

`offset_src != 0` 时给 fs_entry 写 `offset` 字段，并保留 `offset_src`。

[`compute_sequentiality`](../src/agent_io_tracing/analysis/phase1_metrics.py#L906) 一直在读 `e["offset"]`，而解析器从未产出过该字段——**它是死代码，index 上那个 "Not measurable here" 的 note 就是这么来的**。这一步之后它变成活的。

### B3. 指标改动（重写 `compute_sequentiality`）

现在的实现有三个错，必须一起改：

| 现在 | 改成 | 为什么 |
| --- | --- | --- |
| 按 **path** 分组 | 按 **(tid, fd, open 世代)** 分组，同时也出 (pid, fd) 口径 | fd 号 close 后会回收给别的文件，不按 open 世代断流会把两个不相干文件接成一条流、判出一堆假回退；多线程共享 fd 时两条各自顺序的流会被揉成一条假随机流——两个口径的差值本身就是"并发访问同一文件"的信号 |
| **读写混在一条序列** | 读 / 写**各自独立**成流 | 同一文件上交错的读和写会被误判成 random |
| 用 `requested_size` | 用**实际字节数**（`return_value`） | 短读时 `prev_end = off + 请求量` 会把正常顺序读误判成"有 gap"（Darshan 也是用返回值，这点它做对了） |

分类口径**对齐 Darshan**（`POSIX_SEQ_*` / `POSIX_CONSEC_*`）：

- `consecutive`：`this_offset == last_byte + 1`——严格紧接，一个字节不差。**预读有效、Lustre 条带能连续吃满。**
- `sequential`（含 gap）：`this_offset > last_byte`——方向朝前但允许跳空。**CONSEC 是 SEQ 的子集。** 预读会读进用不上的数据（浪费带宽）。
- `backward` / `random`：`this_offset <= last_byte`——原地或回退。**每次都是新的 RPC，延迟主导。**
- `append`：`O_APPEND` 打开的 fd 上的写，单列一类，**不参与 seq/random 判定**（见 B0）。
- `stride`：向前跳时 `this_offset - last_byte - 1`，出直方图（复用 `analysis/size_bins.py` 的 Darshan 分箱）。Bez 综述把 stride 列为常用特征之一。

**必须同时出的诚实指标**：`pct_ops_with_offset`——"sequentiality 是基于百分之多少的数据操作算出来的"。Darshan 从不报这个数，我们报。

### B4. 画图

挂 **Axis 4（Reasoning-Step / Data-Granularity Alignment）**，一张图两个 panel：

- 左：**堆叠条形图**，两根柱（read / write），每根按 consecutive / sequential-gap / backward / append 堆叠，柱上标 transition 总数。这是文献标准形态，可直接和 Darshan-based 论文的 seq/consec 比例并排看。
- 右：**stride 分布直方图**（Darshan 分箱）。

图上角标注 `pct_ops_with_offset`。

同时**删掉** index 里 `panel("Access pattern", "", seq_note)` 那个占位 note——它被真图取代（"有图不配表"）。

---

## C 线 · 文档连带修改

- `Plans/Index-Rework-Plan.md` §5 的 caveat 里"部分重读测不了（无 offset）"这句删掉——有 offset 之后按字节区间判重读是可行的。重读本轮仍不做，但**理由不再是"测不了"**。
- `axis_metric_mapping.md` 轴4（粒度对齐）的「Sequential/Random」行：从"❌ 未实现"改为实现说明，并写清 mmap 盲区 + 缓冲层语义 + 逻辑≠物理三条 caveat。
- `Overview.md` §6.1 的 "I/O time, effective bandwidth" 已实现，但要补上 A2 修好之后的 duty-cycle 定义。

---

## D 线 · 执行顺序

**A 线（先做，纯离线）**

1. A1 workload 过滤（一行参数，影响最大）
2. A2 duty_cycle 分母（会翻转结论，必须做对）
3. A3 统一 run 时间轴
4. A5 role 提取合并
5. A4 只补 caveat 文字

做完在现有 trace 上重跑一遍分析，**重点核对 A2 修好后 duty cycle 是不是从接近 1.0 掉到个位数百分比**——如果没掉，说明我对数据的判断错了，要先查清楚再往下走。

**B 线（需要重跑 trace）**

6. B1 采集改动 → **先跑 `agent-trace-w0` 量开销**
7. B2 解析 → 用一个小 cell 验证 `offset` 字段确实出现、`pct_ops_with_offset` 合理（Python 全走 `read`，预期接近 100%；若很低说明 kprobe 没挂上或 iter 路径漏了）
8. B3 指标 + B4 画图
9. 重跑一个 GenoMAS + 一个 SciLink cell，确认图有意义

---

## E 线 · 重跑命令

### A 线（离线，现有 trace 直接重算）

```bash
CELL=results/genomas_base_phaseb_20260708_091105/<workload>

python -m agent_io_tracing.lineage.analyzer        "$CELL"
python -m agent_io_tracing.analysis.phase1_metrics "$CELL"
python -m agent_io_tracing.viz.trace               "$CELL"   # viz 必须最后，它读前两步的产物
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

### B 线（改了采集，必须重跑 trace）

```bash
source cloudlab_env.sh

rsync -az --delete \
  --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude 'results/' --exclude '.venv/' \
  ./ "$SSH_USER@$CLIENT_NODE:pi-ebpf-tracing-handoff/"

# 先量开销，别直接上 GenoMAS
ssh -t "$SSH_USER@$CLIENT_NODE" \
  "cd pi-ebpf-tracing-handoff && sudo -E .venv/bin/agent-trace-w0"

# 小 cell 验证 offset 字段
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

验证 offset 覆盖率：

```bash
python3 -c "
import json,glob,collections
f=sorted(glob.glob('results/<new_run>/*/parsed.json'))[0]
d=json.load(open(f))
c=collections.Counter(('offset' in e) for e in d['fs_entries']
                      if e.get('syscall') in ('read','write','pread64','pwrite64'))
print('with offset:', c[True], ' without:', c[False])
"
```

---

## F 线 · 论文里必须声明的 caveat（不声明就是 overclaim）

1. **mmap 读不可见**——页错误路径无 syscall，Darshan 同样看不见。
2. **顺序性是缓冲层下发的顺序**，不是应用逻辑的访问顺序（CPython `BufferedReader` 会攒块）。Darshan 在 POSIX 层同理，**口径可比**。
3. **逻辑偏移 ≠ 物理偏移**——VFS 在 page cache 之上，顺序读在设备上可能是碎的。要拿物理连续性需挂块层（`submit_bio` / `block:block_rq_issue`），本轮不做。
4. **effective bandwidth 含 page cache**——`duration` 测的是调用返回，不是数据落盘；write 常常只是 memcpy 进 cache 就返回。这是**应用侧感知带宽，不是设备带宽**。
5. **`O_APPEND` 每个 fd 的第一次写偏移可能陈旧**，已用 `open_flags` 标出并排除。
6. **sequentiality 的覆盖率**（`pct_ops_with_offset`）随图报出，不假装 100%。
