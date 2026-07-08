# Phase 1 Plan — Minimal Closed Loop

Goal: with **one model + one workflow (GenoMAS)**, bring up the logical-to-physical attribution
pipeline and produce 6 numbers that can sit alongside the literature, to judge whether the premise
has signal. **No** factorial, cross-model, or multi-agent in Phase 1.

---

## Implementation-status legend

This plan is annotated against the current codebase so the next agent knows what to build vs. what
already exists. Tags:

- **[✅ DONE]** — implemented and usable today (file pointer given).
- **[🟡 PARTIAL]** — ingredients exist but the Phase-1 deliverable is not assembled / not emitted.
- **[⬜ TODO]** — not in the code; a future agent must build it.

**Scope note (long-term vs. Phase 1).** The repo already supports five targets (pi / SRAgent /
SciLink / GenoMAS / ChemGraph) and a DAG/parallelism layer. These are **correct long-term
infrastructure, not scope creep** — Phase 1 simply *exercises only the GenoMAS path*. The one thing
that is genuinely off-axis for this paper is the **speedup / concurrency-proof** artifacts (e.g.
`GenoMAS_traces/.../make_concurrency_proof_4mw.py`): that answers "orchestration parallelism," which
is a different question from I/O characterization, and should not be carried as a Phase-1 figure. The
DAG itself stays — I/O attribution needs a workflow dependency graph (Lüttgau PDSW'18 anchor).

---

## 0. Baseline route — DECISION: **A (locked)**

Reference = *literature distributions ("the book")* + *analytical optimum*. We do **not** author an
"expert" baseline, and we do **not** require a matched one-to-one baseline run.

**Framing correction (important).** This is a *sampling characterization*, like Luu/Paul: they
sampled many apps/jobs and reported **distributions** ("3/4 of jobs POSIX-only," "most requests
<10MB"), not a paired comparison against a specific reference run. We do the same for agentic
workflows and compare **distribution shapes** to the published numbers. So "is there a perfectly
matched baseline?" is the wrong question — fairness here means *comparable sampling + comparable
metric definitions*, not a rigid 1:1 control.

Concretely, "reference" appears in two distinct roles; only the second ever implied a paired run:

- **§4 comparison table** — pure "compare to the book." Targets are Luu/Paul's *published* numbers.
  **No baseline run needed.**
- **6th number (amplification)** — the word "amplification" smuggled in a paired denominator. **Fix:
  compute it against the *analytical optimum*** (N small files vs. 1 batched file is, by definition,
  more metadata ops on Lustre), **not** against a matched `.h5ad` pipeline run. This removes the
  paired-baseline burden entirely.

A self-built "expert-optimized" version may be added later as an *auxiliary, clearly-caveated*
control, but it is **not** a Phase-1 deliverable and **not** a kill criterion.

---

## 1. Workload — GenoMAS (first and only in Phase 1)

GenoMAS (arXiv:2507.21035): six specialized LLM agents, typed message-passing, **code-driven** gene
expression analysis on the GenoTEX benchmark. Why it is a good first workload:

- **The I/O abstraction choice is literally the agent's and is observable.** Agents *generate /
  revise / validate executable code* rather than chaining fixed tools, so the chosen I/O layer
  (pandas/CSV vs HDF5/AnnData) shows up directly in generated code.
- **Natural agent-characteristic I/O phases:** a dynamic memory mechanism that stores validated code
  snippets (-> memory write-back I/O); Action Units that advance/revise/bypass/**backtrack** (->
  retry-cleanup I/O).
- **Input is "multiple large, semi-structured files"** (GEO-style transcriptomics) -> dataset-read
  characterization.
- **Reproducibility:** GenoTEX defines the tasks (dataset selection, preprocessing, statistical
  analysis), so runs are pinnable and citable.
- **Phase-2 ready:** GenoMAS supports heterogeneous LLM composition, so cross-model is a natural
  later axis (keep `model` as a first-class schema field now).

GenoMAS-specific hypotheses to instrument for:
- preprocessing emits per-sample / per-step CSV/TSV/JSON via pandas STDIO, vs the canonical
  AnnData/`.h5ad` single-object layout (H1/H2);
- the memory-snippet store and summary write-back trigger metadata bursts (H3/H4);
- backtrack/revise produces create->write->unlink churn (retry-cleanup signature).

**W0 microbench is now the *calibration vehicle* (see §6).** Run a **W0 microbench** (IOR/mdtest + a
small agent-memory microbench) **[⬜ TODO]** for two purposes: (1) calibrate the CloudLab Lustre
small-file / metadata saturation knee (hardware ceiling, does not participate in agent-vs-reference);
(2) because it emits a **known ground-truth I/O pattern**, it is also what we use to validate that the
eBPF pipeline reproduces Darshan-defined counters — *without running Darshan* (§6).

## 2. Versions per workload (Phase 1 = two + one reference)

| Version | Purpose | Phase | Status |
|---|---|---|---|
| `live agent` | realism | 1 | **[✅ DONE]** `analyze_codebase_genomas.py` + `genomas_tool_logger.py` |
| `replay / cached LLM` | remove API randomness, isolate pure I/O, test whether I/O is masked by LLM latency | 1 | **[⬜ TODO]** no replay/cache harness exists; required for the 5th number's denominator |
| `reference` (analytical optimum; see DECISION A) | is the agent's choice actually worse | 1 | **[⬜ TODO]** analytical-optimum computation, no run needed |
| `non-agent scripted` (same tool calls, no LLM reasoning/retry) | separate agent orchestration from tool-intrinsic I/O | **1.5** | [⬜ TODO] |
| self-built `expert-optimized` | only if revisiting DECISION B/C | optional | — |

## 3. Data to collect (by layer; folds in the solid parts of the original plan)

### 3.1 Run manifest **[🟡 PARTIAL]**
workload/task ID, dataset hash, prompt/template version, agent version;
**model/API/replay mode, temperature, seed, cached-response hash** (model is a first-class field);
agent count, client node, pid/container;
Lustre config (stripe count/size, # MDS/OSS, mount options, cache mode);
cache state (cold/warm, drop-cache, prewarm);
instrumentation level (none / agent-only / eBPF / eBPF+Lustre counters).

> Status: `config/config_genomas.env` carries paths and model; there is **no structured per-run
> manifest** emitting seed/temperature/cached-response-hash/Lustre-stripe/cache-state/instrumentation
> level. A future agent should emit a `manifest.json` per run with these fields.

### 3.2 Agent logical trace (the core differentiator from Darshan-style work)
per step: `run_id, agent_id, step_id, parent_step_id` (join keys) **[✅ DONE]** `genomas_tool_logger.py`;
`phase` (LLM reasoning / tool call / code exec / memory read / memory write / summarization / retry)
**[🟡 PARTIAL]** — LLM / tool-call / code-exec phases captured; **`memory_snippet_write`,
`action_unit_backtrack`, `summary_write` are NOT tagged [⬜ TODO]** (these are exactly the H4 phases);
LLM start/end, tokens, cache hit **[✅ DONE]** (`message_start/message_end`);
tool name/command/cwd/pid/exit **[✅ DONE]**;
**generated-code hash + imports + I/O-API usage [⬜ TODO]** — logger records code *length* only, not
content/hash/imports. This is the join key for H1 and is the single most important missing piece;
declared input/output path (or path hash, to join with eBPF) **[🟡 PARTIAL]**;
logical op type (artifact_read/write, tool_input_read, tool_output_write, semantic_query,
index_update, summary_write). For GenoMAS, add: `memory_snippet_write`, `action_unit_backtrack`
**[⬜ TODO]**.

### 3.3 I/O abstraction classification (premise-critical, easiest to get wrong) **[⬜ TODO — top priority]**
For each generated script / tool call, record the I/O layer used. **Nothing in the code does this
today.** This is the entire basis of H1 (interface choice) and the stated novelty in Overview §2; it
must be built (offline parse of the generated code captured per 3.2).

| Layer | Signals |
|---|---|
| STDIO / Python file I/O | `open()/read()/write()`, `json`, `csv`, `pandas.read_csv/to_csv` |
| **unbuffered POSIX syscall I/O** (NOT "direct I/O"; direct I/O means O_DIRECT specifically) | `os.open/os.read/os.write`, shell `cp/cat/grep/awk/sed` |
| structured formats | `h5py`, `anndata/scanpy` (`.h5ad`), `netCDF4`, `zarr`, `parquet/pyarrow`, `sqlite`, `duckdb` |
| MPI-IO | `mpi4py`, `MPI.File`, uprobe on `MPI_File_*` **or** code parse (presence of libmpi in `/proc/pid/maps` is *necessary-not-sufficient*: many MPI programs do their I/O via POSIX) |
| vector/index | FAISS rewrite, SQLite writes, Qdrant/Chroma calls, JSONL memory logs |

### 3.4 eBPF syscall trace (Phase 1 workhorse) **[✅ DONE]** `bcc_tracer.py` + `parse_ebpf.py`
Cover: open/close (`openat/openat2/close`); metadata
(`statx/newfstatat/fstat/access/getdents64`); data
(`read/write/pread64/pwrite64/readv/writev`); durability
(`fsync/fdatasync/sync_file_range`); namespace
(`mkdirat/renameat/renameat2/unlinkat/rmdir/symlinkat`); lineage (`fork/clone/execve/exit`).
**Status: syscall coverage is implemented**, including `readv/writev/preadv/pwritev` and
`fsync/fdatasync/sync_file_range`.
Per event: `timestamp_ns, pid/tid, comm, parent_pid/tool_call_id, syscall, ret/errno, latency_ns,
fd, path/path_hash, offset, requested_size, actual_size, flags, cwd/cgroup`.
**Status: core fields emitted** (`syscall, ret, latency_ns, path, raw arg0/arg1/arg2`); size/offset/
flags currently live in **raw args, not named fields** — a future agent should decode them into
named `requested_size/actual_size/offset/flags` for the request-size CDF.

**Mandatory preprocessing (else polluted by Python startup noise):** **[🟡 PARTIAL]** `lineage_analyzer.py`
- keep only workload dirs (e.g. `/mnt/lustre/<run_id>/...`); drop warm-up / venv import phase **[✅ DONE]**;
- filter by tool-call interval (time-window) **[🟡 PARTIAL]**;
- classify paths into five classes: `dataset / artifact / memory / temp / environment` **[🟡 PARTIAL]**
  (current classes are data-flow oriented: raw_input/intermediate/terminal_output/coordination/...).

> **Fix the log-filtering bug.** `lineage_analyzer.py` currently *excludes* the agent's top-level log
> (`/output/log_`, `EXCLUDE_PATH_SUBSTRINGS`). That is agent-induced I/O and is exactly what `logs/`
> (§3.5) and the kill criteria must measure — **do not pre-filter it out**; measure it, then decide.
> Also: `DATA_PATH_PREFIXES` is hardcoded to `/users/Minqiu/GenoMAS/...`; make the default robust
> (derive from the run manifest / cwd) so a different user/path does not silently drop all artifacts.

### 3.5 Namespace / small-file scan (after each run) **[🟡 PARTIAL]** `lineage_analyzer.py`
total file count; file-size CDF; files per logical task / per tool call; directory fanout;
temp-file count & lifetime; rename/unlink/create ratio; (Phase 2) duplicated bytes; index/db rewrite size.
> Status: file-size distribution, reader fan-out, write→first-read staleness, lineage depth, and
> read-amplification are implemented. **Files-per-tool-call, directory fanout, temp-file lifetime,
> and the create/rename/unlink churn ratio are not yet emitted [⬜ TODO]** — the churn ratio is the
> retry-cleanup (H4) signature.
Path classes: `dataset/ tool_inputs/ tool_outputs/ agent_memory/ summaries/ semantic_index/ logs/ tmp/ environment/`
**[⬜ TODO]** — align the current data-flow classes to these agent-phase classes (esp. `agent_memory/
summaries/ logs/`, which carry H4).

### 3.6 Temporal / phase metrics **[🟡 PARTIAL]** `visualize_strace.py` + `compute_parallelism.py`
ops/s and bytes/s time series; metadata ops/s vs data bytes/s; p50/p95/p99 syscall latency **by
phase**; overlap with LLM/tool phases; inter-arrival distribution (burstiness).
> Status: per-phase / per-tool syscall aggregation, role×resource time decomposition, and the
> agent-lane timeline exist. **p50/p95/p99 percentile latency is NOT computed anywhere [⬜ TODO]**
> (only a single 0.75-quantile for an unrelated purpose). Burstiness / inter-arrival distribution is
> also [⬜ TODO].
**Target figure:** agent-phase timeline on top, metadata ops/s, write bytes/s, (if available) MDS CPU,
tool subprocess below — so a reviewer immediately sees "the memory write-back / summary phase triggers
a metadata burst."

### 3.7 Lustre client/server counters (collect even without Darshan) **[⬜ TODO]**
client `llite/mdc/osc` (metadata RPCs, getattr/setattr, create/unlink/rename, object read/write RPCs,
RPC size distribution if available); server `MDT/OST` (op rate, MDS CPU, bytes, IOPS, queues, if
available); system (CPU/mem/disk util/net).
**Why:** eBPF can only say "the process issued a syscall"; it cannot by itself prove the bottleneck is
the MDS/OST/cache/network/Python runtime. Counters are the mechanism-localization evidence (the H3
"metadata storm on MDS" claim depends on this). Use `lctl get_param`; no Darshan needed.
> Status: **no `lctl`/counter collection in the repo.** A future agent must add a sampler.

## 4. Comparison protocol (compare to the book) **[⬜ TODO]**

Each Phase-1 metric -> which figure/table in which paper -> the same/different criterion. **Compare
shapes/ratios only, never absolute values.** This is a *sampling* comparison to published
distributions (see DECISION A), **not** a paired baseline run. **None of this mapping is implemented
yet** — `lineage_analyzer.py` emits raw distributions, but nothing maps them onto the Darshan-defined
buckets or emits a same/different verdict.

| Metric | Comparison target | same / different criterion |
|---|---|---|
| Interface mix | Luu HPDC'15: 3/4 of jobs POSIX-only | same = agent also POSIX-dominant; different = ~zero structured/MPI-IO, or heavy STDIO text |
| Request-size CDF | Paul MASCOTS'21: most <10MB | same = lands in the same small-size regime; different = even more sub-4KB |
| Metadata time fraction | Luu: >1/3 of jobs metadata>data; global/nonglobal/data split | same = within their range; different = pushed higher by a small-file storm |
| File size / bytes per task | Luu Fig 2 (**Edison = Lustre column**); ~half of apps <1GB | same = few large files; different = many small files, low bytes/task |
| Sequential vs random/consecutive | Luu/Darshan standard ratio | same = highly sequential; different = more random/backward seeks (Ellexus "bad I/O") |
| Text vs binary fraction | Luu: ~1/5 of apps text-only | highest probability of "different" -> potential headline |

## 5. The 6 numbers Phase 1 must produce (per DECISION A)

1. metadata ops / data ops ratio — **[🟡 PARTIAL]** syscalls are categorized (parse_ebpf
   `SYSCALL_CATEGORIES`), but a single ratio is not emitted.
2. file-size CDF — **[✅ DONE]** `lineage_analyzer.py` (size distribution).
3. files created per agent step / per tool call — **[⬜ TODO]** per-tool-call file count not emitted.
4. p50/p95/p99 syscall latency **by phase** — **[⬜ TODO]** no percentile computation exists.
5. **FS-I/O time as % of non-LLM runtime** (replay mode). **Revised definition (honest, measurable):**
   eBPF cleanly gives three buckets — **LLM time** (agent trace `message_start/end`), **FS-I/O time**
   (sum of entry→exit latency over the file-I/O syscalls), and **the residual** (wall − the other two).
   The number is `FS-I/O time / (wall − LLM time)`. The old phrasing ("numerator must separate
   tool-compute from I/O") is dropped — separating CPU-compute from I/O cleanly needs sched_switch
   on-CPU measurement (residual CPU has a single-core ceiling), which is deferred. **[🟡 PARTIAL]** —
   constituents exist (`summarize_pi_events.py` LLM time; `parse_ebpf.py` file_io resource latency);
   the single ratio is not emitted, and **replay mode (the denominator's regime) is [⬜ TODO]**.
   **Pin the FS-I/O bucket:** `metadata + data + durability(fsync/fdatasync/sync_file_range) +
   open/close`. **Exclude `mmap/munmap/ioctl/fcntl`** from "storage time" (mmap setup is not data
   transfer); today they fall under the `control` category mapped to `file_io` — split them out.
6. agent vs **analytical optimum** (N small files vs 1 batched file; **not** a self-built expert,
   **not** a matched run) write/metadata amplification — **[⬜ TODO]**.

## 6. Counter-definition calibration via W0 microbench (replaces "run Darshan") **[⬜ TODO]**

Every comparison target in §4 is a Darshan-defined number, so definitions must be aligned. **Decision
(revised): do NOT run Darshan on GenoMAS.** GenoMAS is a fork-heavy, non-MPI Python workload where
Darshan's LD_PRELOAD path is unreliable, and Darshan itself carries quirks if treated as ground truth.

Instead: Darshan's counter definitions (request-size buckets, metadata/data time, POSIX/MPI-IO counts)
are **public and explicit**. Align the eBPF pipeline to those documented definitions, then **validate
against the W0 microbench**, whose I/O pattern is known ground truth. This is stronger than matching
Darshan (we validate against a *known* truth, not against another tool's output) and needs **no
Darshan binary and no MPI**.

> Feasibility-first conclusion: "whether Darshan runs on GenoMAS is irrelevant." A standalone Darshan
> cross-check can live in the **long-term** backlog if a reviewer ever demands it; it is **not** a
> Phase-1 task.

Paper answer for "why not Darshan as the primary mechanism":
> Darshan gives application/job-level profiles. Our first contribution is agent-aware
> logical-to-syscall attribution, so we start with eBPF + agent traces; counter definitions are
> aligned to Darshan's public definitions and validated on a ground-truth microbench, not by running
> Darshan as the primary instrument.

## 7. (removed) Observer-effect controls

**Removed as a Phase-1 pillar.** This paper *characterizes agentic workflows*; it is **not** a study
of eBPF tracing overhead. The original "mandatory 3-mode (no-trace / eBPF aggregate / eBPF raw)"
framing was over-elevated (also in Overview §5.3) and is dropped. Additionally, this workload's
wall-clock is dominated by LLM/API wait, so any e2e overhead measurement would be washed out and
read as a misleading "no perturbation."

*Optional, footnote-level only:* if a reviewer asks whether tracing perturbed the small-file/metadata
pattern, a single sanity run (measured on the **non-LLM / replay** segment, not e2e) suffices. This is
a validity check, not a research axis, and is not required to ship Phase 1.

## 8. Kill criteria — **two orthogonal questions, kept fully separate**

The original kill #1 conflated two independent questions. They are now split; **bytes and wall-clock
appear ONLY in the importance question (#3), never in the distinctiveness question (#1).**

1. **Distinctiveness — "is agentic I/O distinctive?"** Judged **only** on metadata ops / file count /
   namespace fanout / create-write-unlink churn, vs. the analytical reference. **Bytes and wall-clock
   do not appear here at all.** Kill iff the agent shows *no* difference on these structural axes.
   (Using the analytical reference removes null-result ambiguity.)
2. *(reserved)* — kept aligned with #3; not a separate kill.
3. **Importance — "is I/O on the critical path?"** This is where **bytes / time / critical-path** live.
   In replay mode, if storage is a tiny fraction of non-LLM runtime and does not affect progress,
   then I/O is not on the critical path.

**Key design consequence:** #1 and #3 can disagree, and that combination is the *most likely and most
publishable* outcome — agent I/O can be **distinctive (small-file metadata storm) yet not on the
critical path (masked by LLM wait)**. Per Overview §7, that is **not a dead end**: it reframes the
contribution toward characterization/observability rather than a storage-performance motivation.
Design the pipeline so "distinctive-but-not-critical" is reported as a finding, not treated as a
double-kill.

## 9. Execution order (pilot first, do not scale CloudLab first)

1. On existing traces: keep only eBPF events under the GenoMAS workload dir **[✅ DONE]** (but stop
   excluding agent logs — §3.4 fix);
2. join to tool-call intervals **[✅ DONE]**;
3. per tool call, emit file count / metadata ops / bytes / p95 latency **[🟡 PARTIAL]** — bytes done,
   file-count/metadata-ratio/percentiles TODO;
4. classify generated code/command into I/O APIs (3.3) **[⬜ TODO — top priority]**;
5. prepare the reference (DECISION A: **analytical optimum**) **[⬜ TODO]**;
6. compare metadata amplification and file-size CDF **[🟡 PARTIAL]**.

If the pilot already shows the agent producing significantly more small files / metadata ops than the
reference -> worth investing in CloudLab Lustre expansion. Otherwise, do not build the large
experiment yet.

---

### Open action items
1. Build the I/O-API classifier (3.3) + capture generated-code hash/imports (3.2) — these two unlock
   H1 and are the critical path for Phase 1.
2. Pin the GenoMAS model + GenoTEX task subset for the first runs (single model for Phase 1).
3. Build replay/cached-LLM mode (needed for the 5th number's denominator and any validity check).
4. Add Lustre `lctl` counter sampling (3.7) for the H3 mechanism-localization evidence.
5. Tag GenoMAS-specific phases (`memory_snippet_write`, `action_unit_backtrack`, `summary_write`) for
   H4, and align path classes to `agent_memory/ summaries/ logs/`.
