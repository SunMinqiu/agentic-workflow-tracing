# Phase-1 GEO-only Fanout 实验计划（Axis 1 cohort + Axis 2 trait，合二为一）

条件：**GEO-only · 单模型 · live · 全程 `--quick-test` · 单 worker（串行）**。
目标：固定 model/workers/mode，只缩放**输入形状**，量化 agent 的 I/O 如何随它被交付的工作量增长。

---

## 0. 选模型（最便宜的 OpenAI）

| 模型 | 约定价（输入/输出 /1M tok） | 说明 |
|---|---|---|
| **`gpt-5-nano-2025-08-07`** ✅ 选它 | ~$0.05 / ~$0.40 | gpt-5 家族最便宜档；和现有 `gpt-5-mini` 同一 API 路径，GenoMAS `ModelConfig` 已支持，零改动 |
| `gpt-4.1-nano` | ~$0.10 / ~$0.40 | 备选（若 nano 在该工作流上表现异常） |
| `gpt-4o-mini` | ~$0.15 / ~$0.60 | 再备选 |

已写入 [config/config_genomas_fanout.env](config/config_genomas_fanout.env) 的 `GENOMAS_MODEL`。
> 注：价格按记忆给出，下单前用 `https://platform.openai.com/docs/pricing` 复核一眼即可；换模型只改这一个变量。

---

## 1. 为什么两个 Axis 能合二为一

两个 Axis 本质是同一个操作——**给 GenoMAS 一个 `--data-root`，其 `GEO/<trait>/` 下恰好放我们想让它处理的 cohort**。GenoMAS 从文件系统发现 cohort（glob `GEO/<trait>/GSE*`），所以**控制目录数 = 控制工作量**，不用改 GenoMAS、不用动 task_info.json 的 cohort 列表。

- **Axis 1（cohort）**：1 个高-cohort trait，C=1,2,4,8 个 cohort → geospec `Type_1_Diabetes:8`
- **Axis 2（trait）**：T=1,2,4,8 个单-cohort trait → geospec `A:1,B:1,C:1,...`
- **C=1 与 T=1 是同一次运行**（1 trait × 1 cohort）→ 只跑一次，命名 `base`，作为两条轴共同的原点。

cohort 取「排序后的前 N 个」，所以 C=1 ⊂ C=2 ⊂ C=4 ⊂ C=8（嵌套），增量纯加性，`files∝C` 这种缩放最干净。
视图用**绝对符号链接**搭建，几乎零字节，绝不复制/改动 Lustre 上的真实数据。

一条 harness 跑完全部 cell：[scripts/trace_script_bcc_genomas_fanout.sh](scripts/trace_script_bcc_genomas_fanout.sh)。
每个 cell：搭 view（[src/stage_geo_view.py](src/stage_geo_view.py)）→ STOP agent → 起 BCC tracer → CONT → 跑全套 parse/summarize/lineage/phase1/viz → 往 `fanout_summary.csv` 追加一行。

---

## 2. 数据前提与部署状态

**数据（已就绪 ✅，2026-06-23 核查）**——Lustre `genomas_data/GEO/`：
```
Type_1_Diabetes                        -> 10 cohort   (Axis 1 高-cohort trait)
Type_2_Diabetes / Vitamin_D_Levels /
Von_Hippel_Lindau / Von_Willebrand_Disease /
Werner_Syndrome / X-Linked_Lymphoproliferative_Syndrome /
lower_grade_glioma_and_glioblastoma    -> 各 1 cohort  (Axis 2 trait 池)
```
→ **Axis 1（C=1,2,4,8）和 Axis 2（T=1,2,4,8）现在都能完整跑。**
`SINGLE_COHORT_TRAITS` 已填好这 8 个（Type_1_Diabetes 居首 = 共享原点）。
> 其中 `Von_Hippel_Lindau`、`X-Linked_Lymphoproliferative_Syndrome` 两个名字尚未对 task_info.json
> 核对过（其余 6 个历史 run 已确认有效）；GenoMAS 部署后 stage→slice 会自动报「missing from
> task_info.json」并跳过，到时按提示微调名字即可。

**⚠️ 部署（当前缺失，必须先做）**——该节点已被 ChemGraph 重装：`~/GenoMAS` 不存在、
genomas venv 不在 Lustre、`.env.genomas` 缺失。**跑 fanout 前必须先重新部署 GenoMAS：**
```bash
source ~/Desktop/Benchmarking_Agents/pi-ebpf-tracing-handoff/cloudlab_env.sh
bash ~/Desktop/Benchmarking_Agents/pi-ebpf-tracing-handoff/scripts/deploy_genomas_to_client.sh
```
（会克隆 GenoMAS、建 Py3.10 venv、写 .env.genomas；数据目录不动。）

---

## 3. 测试阶段（先验证新写的代码，**未测的先测**）

新写/改动且需先验证的：`stage_geo_view.py`、`trace_script_bcc_genomas_fanout.sh`、以及被改过的 `phase1_metrics.py / lineage_analyzer.py / parse_ebpf.py`。

**T0 — 本地静态自检（已在本机跑通 ✅）**
- `bash -n` 语法、`stage_geo_view.py` 在假目录建 view、unsatisfiable 返回码=2、`phase1_metrics.py` schema 对齐 CSV 提取器。
- 你无需重跑；这是给你看的「已绿」清单。

**T1 — 集群单 cell 冒烟（最小花费，确认端到端 + 真 LLM 计费）**
只跑 `base`（1 trait × 1 cohort），确认 stage→trace→parse→phase1 全链路：
```bash
ssh Minqiu@<CLIENT>
cd ~/pi-ebpf-tracing-handoff
sudo -E env RUN_CELLS=base \
  bash scripts/trace_script_bcc_genomas_fanout.sh
```
看：`traces/fanout_*/base/` 下应有 `ebpf_events.log`、`parsed.json`、`phase1_metrics.json`、`lineage/io_summary.json`，且 `fanout_summary.csv` 有 1 行非空。
若 `base` 绿 → 放心跑全量；若红 → 看对应 `base/{stage,genomas,parse}.log`。

---

## 4. 正式运行（一条命令跑完两轴）

```bash
ssh Minqiu@<CLIENT>
cd ~/pi-ebpf-tracing-handoff
sudo -E bash scripts/trace_script_bcc_genomas_fanout.sh
```
默认 cell（按当前数据）：`base, a1_c2, a1_c4, a1_c8`（Axis 2 的 t2/t4/t8 因缺数据自动跳过）。
补齐 8 个 trait 并填好 `SINGLE_COHORT_TRAITS` 后，会自动多出 `a2_t2, a2_t4, a2_t8`。

可选：`COLLECT_LUSTRE_COUNTERS=1` 打开 `lctl` 采样（H3「MDS metadata storm」的机制证据）。

---

## 5. 输出在哪 / 看什么 / 为什么

根目录：`traces/fanout_<时间戳>/`

| 文件 | 看什么 | 为什么 |
|---|---|---|
| **`fanout_summary.csv`** ← 头号产物 | 每 cell 一行：`generated_files, distinct_files, storage_metadata_ops, data_ops, metadata_to_data, read_bytes, write_bytes, file_count_amplification, wall_clock_s, total_llm_s` | 直接做两张缩放图 |
| `<cell>/phase1_metrics.json` | `analytical_optimum_amplification`、`metadata_data_ratio`、`six_numbers` | Phase-1 的六个数 + 与「the book」的对比 |
| `<cell>/lineage/io_summary.json` | `workload.distinct_files / read_bytes / write_bytes / bytes_by_category` | 文件数、输入读字节、按类别分桶 |
| `<cell>/stage_manifest.json` | 该 cell 实际 staged 的 trait/cohort | 核对 C/T 与图的 x 轴对齐 |
| `<cell>/visualizations/` | agent-phase timeline + metadata ops/s | H4「memory/summary 阶段触发 metadata 爆发」 |
| `<cell>/lustre_counters.jsonl`（若开） | MDT create/lookup RPC 速率 | H3 机制定位 |

### 两张关键图（用 `fanout_summary.csv`）

**Axis 1 — cohort（x = C ∈ {1,2,4,8}，取 `base,a1_c2,a1_c4,a1_c8`）**
- `generated_files` vs C → 期望 **≈3C**（每 cohort ~3 个产物）。线性=符合预期；超线性=有共享小文件重复重写/churn。
- `read_bytes` vs C → 期望 **∝C**（输入读随 cohort 数线性）。
- `storage_metadata_ops` 与 `metadata_to_data` vs C → 小文件是否把 metadata/data 比推高（H3）。
- **为什么**：验证「单 trait 内堆 cohort」时 I/O 是否线性、小文件重读/churn 是否随 C 放大。

**Axis 2 — trait/fanout（x = T ∈ {1,2,4,8}，取 `base,a2_t2,a2_t4,a2_t8`）**
- `distinct_files`/目录数 vs T → 期望 **≈3T**（每 trait ~3 个输出目录）。
- `storage_metadata_ops` vs T → 跨目录 metadata 扩散、MDS create/lookup 随 T 增长（H3）。
- **为什么**：验证「横向铺开到多 trait」是否制造跨目录 metadata 压力，且与 Axis 1 的纵向堆叠形状不同。

**共享原点**：`base` 同时是 Axis 1 的 C=1 和 Axis 2 的 T=1，两图共用一个点 → 两轴可直接叠在一张图上对比「纵向堆 cohort vs 横向铺 trait，哪种 I/O 增长更陡」。

---

## 6. 判读（呼应 Overview 的假设）

- **线性且温和** → 该规模下 agent I/O 由输入量驱动，无异常放大（某些轴「其实还行」，可信度更高）。
- **超线性 / metadata 比随 C 或 T 抬升** → 命中 H2/H3（小文件 + metadata storm），值得上 CloudLab 扩大实验。
- **`file_count_amplification`（vs 分析最优 = 1 个 batched 文件）随 C/T 增大** → H2 的核心证据。
- 注意 wall-clock 被 LLM 等待主导（`total_llm_s` ≫ FS-I/O）→ 这是「distinctive 但不在关键路径」，按 Phase-1 §8 当**结论**报告，不是 kill。

---

## 7. 复跑命令（上传 → 跑 → 拉回）

```bash
# 1) 本机 → 集群（同步新代码）
source ~/Desktop/Benchmarking_Agents/pi-ebpf-tracing-handoff/cloudlab_env.sh
rsync -av --exclude __pycache__ --exclude 'traces/' --exclude '.venv/' \
  ~/Desktop/Benchmarking_Agents/pi-ebpf-tracing-handoff/ \
  "$SSH_USER@$CLIENT_NODE:pi-ebpf-tracing-handoff/"

# 2) 集群上跑（先 T1 冒烟，再全量）
ssh "$SSH_USER@$CLIENT_NODE" \
  'cd ~/pi-ebpf-tracing-handoff && sudo -E env RUN_CELLS=base bash scripts/trace_script_bcc_genomas_fanout.sh'
ssh "$SSH_USER@$CLIENT_NODE" \
  'cd ~/pi-ebpf-tracing-handoff && sudo -E bash scripts/trace_script_bcc_genomas_fanout.sh'

# 3) 拉回结果
rsync -av "$SSH_USER@$CLIENT_NODE:pi-ebpf-tracing-handoff/traces/" \
  ~/Desktop/Benchmarking_Agents/pi-ebpf-tracing-handoff/traces/
```
