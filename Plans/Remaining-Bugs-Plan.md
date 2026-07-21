# 计划 4：剩余 bug 修复

基于对 `results/20260714_125710/polycrystalline_grains_basic`（第一个带 VFS offset 的 trace）的逐图检查。范围只覆盖 **SciLink 最新 trace + 潜在的 GenoMAS**；SRAgent 和 ChemGraph 的 cell 完全不管（它们的 trace 脚本从未设过 `LINEAGE_DATA_PATH_PREFIXES`，新的 fail-loud 守卫会让它们 lineage 直接报错——这是预期行为，不修）。

## 新 trace 验证通过的部分（不用动）

- VFS offset 覆盖率 **100%**（306 个数据 op 全部带 offset），kprobe 在集群上工作正常。
- Access Pattern 四格出来了：Seq R = 71，Rand R = 56。
- 可合并性（P4 新指标）：306 个 op 中 71 个可省（**23.2%**），80.3% 的字节位于连续段内。
- 图面清理生效：Directory Rescans / Access Pattern / Effective BW / I/O Rate / File Size Distribution / Artifact Lifecycle 均无文字框遮挡，图例移出画布，LLM 折线是整数阶梯。
- 两个守卫按设计触发：lineage 拒绝写空 artifacts；phase1 检测到时钟错位直接拒算。

---

## B1 · STDIO 层没有 workload scoping（严重，会让接口层结论出错）

### 现象

`Measured I/O Interface Mix` 图上：**STDIO = 129 calls / 24.5 MB，POSIX = 306 calls / 4.8 MB**。但这个 run 的 workload 总数据量只有约 **5 MB**（read 4.23 MB + write 0.83 MB，与 POSIX 的 4.8 MB 吻合）。**STDIO 的字节数超过了整个 workload 的数据量，物理上不可能。**

### 根因

libc `fread`/`fwrite` uprobe 只能看到 `FILE*`，**拿不到路径**——实测 parsed.json 里所有 STDIO 事件 `path: null`。于是 `make_workload_filter` 对它们无效：POSIX 侧是**过滤后**的 workload 字节，STDIO 侧是**未过滤**的整个进程树（含 Python 解释器 / numpy / 各种库自己的 fread）。**图上把过滤过的和没过滤的并排比，结论必然错。**

（实测样本：`fread` 单次 262144 B 的调用多次出现——256 KB 的块读，很可能是库在读自己的数据文件，不是 workload。）

### 修法

**根治（需重跑 trace）**：在 `fread`/`fwrite` 的 uprobe 里从 `FILE*` 取出 fd —— glibc 的 `struct _IO_FILE` 有 `_fileno` 字段（x86_64 上偏移固定），`bpf_probe_read_user` 读出来即可。事件带上 fd 之后，parser 就能用**现成的 fd→path 表**解析路径，STDIO 事件从此和 syscall 事件走**同一个 workload 过滤**，两层才能对比。

**过渡（离线，立刻可做）**：在 STDIO 层的 fd/path 解析上线之前，**interface 图不要把 STDIO 和 POSIX 并排画**——要么只画 POSIX（workload-scoped），要么把 STDIO 明确标成 `process-tree scope, not workload-scoped`，并在图注里说明二者口径不同、不可相加也不可相除。当前的 `stdio_pct_deoverlapped` / `posix_direct_pct_deoverlapped` 在 fd 解析上线前**都是无意义的**，应从 JSON 和图上撤下。

---

## B2 · `per_run_io_char` 硬编码 run 列表，新 run 被静默跳过

`analysis/per_run_io_char.py:43` 的 `DEFAULT_RUN_ROOTS` 是一个**写死的 run 名列表**。新的 run（如 `20260714_125710`）不在列表里 → 直接跳过 → `file_access_volume.png` / `rw_asymmetry.png` **停留在旧版本**（旧版还带着被删掉的文字框），而且**不报错**。

**修法**：默认扫描 `results/` 下所有含 `parsed.json` 的 cell，`--runs` 只作为可选过滤器。同时**把它接进 trace 脚本**的后处理链（目前六个 trace 脚本都没有调用它，新跑的 run 根本不会生成这两张 Global 图）。

---

## B3 · Access Pattern 写侧显示为 0，但实际有 91 次写

图上 `Seq W = 0` / `Rand W = 0`，看起来像"没有写"。真相是：**每个写流只有 1 个 op，没有"相邻 op"就没有 transition 可分类**（Darshan 也是这个口径：第一个 op 没有前驱，不入账）。

**修法**：图上（或图注）同时给出 **ops 数**与 **transitions 数**两个量，例如 `write: 91 ops / 0 transitions（每文件单次写，无可分类的相邻对）`。否则读者会把"没有 transition"误读成"没有写"。

---

## B4 · P7 未实现：LLM intensity 仍是 "active LLM calls"

launcher 里只有旧的 `PI_LOG_DELTA_EVENTS` 调试开关（`adapters/pi/launcher.py:308`），**没有把 delta 事件的时间戳和 token 数落盘**，任何脚本也没有开启它。实测新 cell 的 `pi_events.jsonl` 只有 `message_start` / `message_end` / `message_update` / `tool_execution_end`，没有 delta。

所以 `I/O Rate Over Time` 的右轴仍是"同时活跃的 LLM 调用数"（显示 bug 已修：整数刻度 + 阶梯）。

**修法**（需重跑 trace）：launcher 把 delta 事件的**时间戳 + 类型（`text_delta` / `thinking_delta`）+ token 计数**写入 `pi_events.jsonl`；`io_rate` 右轴换成 **output tokens/s**。**不做 TTFT。**

---

## B5 · 死重清理（离线）

| 项 | 位置 | 动作 |
| --- | --- | --- |
| `interface_mix` viz 仍注册 | `viz/_trace_impl.py` 的 `AGENT_VISUALIZATIONS` | index 根本不显示它，每次白生成一张 PNG。删注册 + 删函数 |
| 旧的 4MB 除法字段 | `analytical_optimum_amplification` 的 `optimum_read_ops` / `optimum_write_ops` / `read_call_amplification` / `write_call_amplification` | index 已不显示（改用 `sequentiality.mergeability`），但仍在 JSON 里当死数据。删干净 |
| 轴标签里的公式 | `lineage/_analyzer_impl.py:1315` 的 `"dead_seconds / (run_end - t_create)"`；`viz/_trace_impl.py:1384` 的 `"effective bandwidth (MiB/s) = bytes / sum(syscall duration)"` | 公式移进图注，轴标签只留 `dead time share` / `effective bandwidth (MiB/s)` |
| 零带宽 phase 画成空行 | `create_effective_bandwidth_matplotlib` | `Examine_data` / `Load_metadata` 这类没有 I/O 的 phase 不入图 |

---

## B6 · 陈旧 PNG 残留

已停止生成的图（`timeline.png`、`agent_concurrency.png`、`intensity_phases.png`、`tool_syscall*.png`、`interface_mix.png`）仍留在各 cell 的 `visualizations/` 里。index 不再链接它们，但会误导人（我第一轮 review 就被 14:10 的旧 PNG 骗过一次）。

**修法**：`viz.trace` 在写图前，把 `visualizations/` 里**不在当前注册表内**的 PNG/HTML 清掉（只删自己产出的那类文件名，不碰 lineage/ 和数据文件）。

---

## 执行顺序

**离线（先做，改完直接重跑分析看效果）**
1. B2（per_run_io_char 扫全部 cell + 接进 trace 脚本）
2. B1 过渡措施（撤下无意义的 STDIO/POSIX 混合口径）
3. B3（ops vs transitions）
4. B5、B6（死重与陈旧文件清理）

**需重跑 trace**
5. B1 根治（uprobe 里取 `FILE*._fileno` → fd → path）
6. B4（delta / token 落盘 → output tokens/s）

**5 和 6 应当一起改完再重跑**，否则要跑两次。目标 workload：SciLink 剩余三个（`eels_plasmons_basic` / `eels_identification_basic` / `planning_critical_materials`）+ GenoMAS。

## 重跑命令

离线（现有 trace 直接重算，viz 必须最后）：

```bash
CELL=results/20260714_125710/polycrystalline_grains_basic
export PYTHONPATH=src

python3 -m agent_io_tracing.parsing.ebpf            "$CELL"   # 老 trace 补 ts_ms/tid/open_generation
python3 -m agent_io_tracing.lineage.analyzer        "$CELL"
python3 -m agent_io_tracing.analysis.phase1_metrics "$CELL"
python3 -m agent_io_tracing.analysis.per_run_io_char --results results --runs 20260714_125710
python3 -m agent_io_tracing.viz.trace               "$CELL"
open "$CELL/visualizations/index.html"
```

集群重跑（B1 根治 + B4 之后）：

```bash
source cloudlab_env.sh

# --exclude '.env*' 必须带：.env.scilink 只存在于远端
rsync -az --delete \
  --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude 'results/' --exclude '.venv/' --exclude '.env*' \
  ./ "$SSH_USER@$CLIENT_NODE:pi-ebpf-tracing-handoff/"

ssh -t "$SSH_USER@$CLIENT_NODE" \
  "cd pi-ebpf-tracing-handoff && sudo -E RUN_WORKLOADS='eels_plasmons_basic,eels_identification_basic,planning_critical_materials' bash scripts/trace_script_bcc_scilink.sh"

RUN=$(ssh "$SSH_USER@$CLIENT_NODE" \
  "ls -1dt /mnt/lustrefs/$SSH_USER/pi-ebpf-tracing-handoff/results/*/ | head -1")
LOCAL="results/$(basename "$RUN")"
mkdir -p "$LOCAL"
rsync -az --progress --exclude 'work/' --exclude 'bcc.out' --exclude 'bcc.err' \
  "$SSH_USER@$CLIENT_NODE:$RUN" "$LOCAL/"
open "$LOCAL"/*/visualizations/index.html
```
