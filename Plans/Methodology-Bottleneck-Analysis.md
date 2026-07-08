# Methodology: Accounting → Ranking → Diagnosis

## Step 1 — Accounting (no classification)

Establish the baseline for this run before looking for anything specific:

- total read bytes, total write bytes
- total I/O time (`fs_io_ms_sum` — sum of storage-syscall durations; **this already exists**, computed in `compute_fs_io_non_llm` in `phase1_metrics.py`, not a new metric)
- total operation count, total metadata operation count
- effective bandwidth (bytes / `fs_io_ms_sum`) — not yet an explicit field, trivial to add from the two numbers above

Nothing here is split into categories. This is the denominator everything in Step 2 gets compared against.

## Step 2 — Ranking (systematic, not manual exploration)

Sort the same underlying I/O events (`fs_entries` from `parsed.json`, joined with `artifacts.csv` and the tool-call/phase index) along each of the following dimensions, and take the top N contributors on each. This is the step that was missing before — the four findings we already have (checkpoint `.tmp` rewrite frequency, directory rescans, reread attribution, backtrack-window I/O) were each found by manually picking one dimension to look at. Running all eight dimensions systematically on every trace is what makes this repeatable instead of one-off.

| # | Dimension | What it ranks | Status |
|---|---|---|---|
| 1 | **Top files by bytes** | Which files contribute the most read+write bytes | Data exists (`artifacts.csv` has `total_read_bytes`/`total_write_bytes` per file) — ranking/top-N extraction not yet built |
| 2 | **Top files by I/O time** | Which files' syscalls consumed the most cumulative duration | Not built — `duration` lives on individual `fs_entries`, not yet summed per file path |
| 3 | **Top tool calls by I/O** | Which specific action-unit execution (`tool_call_id`, not the coarser phase label) contributed the most bytes/ops | Not built as a ranked list — we have the underlying join (used for `reread_attribution` and `bytes_ops_by_phase`), just not aggregated *per tool_call_id* and sorted |
| 4 | **Top phases by I/O** | Which phase (`code_exec`, `summary_write`, `action_unit_backtrack`, `orchestration`) contributed the most bytes/ops/time | **Already have the data** — `bytes_ops_by_phase` in `phase1_metrics.json`; just needs sorting/display as a ranked list instead of an unordered table |
| 5 | **Top metadata-heavy windows** | Which file/directory/tool_call has the highest ratio of metadata ops (open/stat/getdents64) to data ops | Partially have building blocks (`directory_scan`, `failed_open_stat`) but no unified per-window metadata-density ranking |
| 6 | **Top repeated-read windows** | Which specific reread instances (not just bucket totals) are the largest, by bytes | Have the classification (`compute_reread_attribution`, validated against known backtrack counts) but currently reports bucket totals, not the individual top-N reread instances |
| 7 | **Top slow-read/write operations** | Which individual syscalls had the highest latency (outliers, not just phase-level p50/p95/p99) | Not built — `latency_by_phase` gives percentiles per phase, not a ranked list of the actual slowest individual operations |
| 8 | **Top temporary/abandoned artifacts** | Which temp files or dead-writes (written, never read back) are the largest | Have the classification (`classify_artifact` → `tmp`, `reuse_class` → `dead_write`) but not ranked by size |

None of these eight rankings requires new tracing infrastructure — all of them are derivable from data already captured (`parsed.json`, `artifacts.csv`, `generated_code.jsonl`, `pi_events.jsonl`). What's missing is the aggregation/sorting code itself, consistently applied across all eight dimensions rather than picked ad hoc.

## Step 3 — Diagnosis (fixed checklist, applied to each top-N contributor from Step 2)

For every contributor that shows up in a Step-2 ranking, answer the same four questions — don't freestyle a new explanation each time:

1. **Necessary or avoidable?** Would a correctly-behaving run of this exact task still need to produce this I/O? (Answer by opening the corresponding node in the interactive Call DAG — click-to-view now shows the generated code and the LLM's decision output for that step — and judging whether the I/O follows necessarily from the task.)
2. **If avoidable, caused by behavior or by design?** Did this happen because of *this run's* agent behavior (a retry, redundant exploration — would a different, better-behaved agent run not have produced it), or is it baked into how the code/library is written regardless of agent behavior?
3. **If it's a design/code issue, whose code?** Was the inefficient choice made in code the agent generated for this run, or is it fixed in `tools/preprocess.py` / `environment.py` (present on every run regardless of what the agent does)?
4. **Record one line**: what the contributor is, its size/duration, the diagnosis from questions 1-3, a suggested fix if there is one, and the `run_id` / DAG node as evidence so the finding can be independently re-checked later.

This checklist is where labels like "agent-induced" or "misconfigured" still get used — but only as a per-instance diagnosis result recorded in Step 3, never as a mandatory partition applied to all I/O up front.

## Full inventory of measured metrics (GenoMAS, as of this session)

### A. Universal / aggregate accounting metrics

| Metric | What it measures | Why it matters | Status |
|---|---|---|---|
| Read/write bytes total | Total bytes read and written | Denominator for everything else | Done — real data (`io_summary.json`) |
| Read/write op count | Total read/write operation count | Same, at the operation-count granularity | Done |
| Metadata operation count | open/stat/getdents64-class operation count | Signals metadata-heavy vs. data-heavy workload | Done (`strict_metadata_ops`, `storage_metadata_ops`) |
| Unique files touched | Distinct file count | File-granularity vs. byte-granularity view | Done |
| Small-file / small-I/O CDF | Distribution of file sizes and per-request sizes (p50/p95/p99, % under 4KB / 1MB) | Detects the classic HPC "small-file storm" pattern | Done |
| I/O interface used | Which I/O layer (POSIX / stdio / structured / MPI-IO) agent-generated code actually calls | Detects wrong-abstraction choices (e.g. row-by-row read where a batched library call exists) | Done — classifier blind spot for library-wrapper calls fixed this session, verified 1/12 → 11/12 detected |
| Output file count / avg file size | Number and average size of produced files | "Few large files" vs. "many small files" | Done |
| Storage location choice | Local scratch vs. shared Lustre | Confirms data lands on the right storage tier | Done — plus real Lustre `lctl` MDS/OST counters captured automatically |
| FS-I/O % of non-LLM time | I/O time as a fraction of wall-clock time excluding LLM wait | Tells you whether I/O is on the critical path | Done |
| I/O time total (`fs_io_ms_sum`) | Sum of storage-syscall durations | The actual "how long did I/O take" number | Done — already existed in `compute_fs_io_non_llm`, initially mis-marked as missing, corrected |
| Effective bandwidth | Bytes / `fs_io_ms_sum` | I/O efficiency | Not done — trivial to derive from the two numbers above, no field computes it yet |
| Analytical-optimum amplification | File count / write-call / metadata-op count vs. a theoretical "one batched file" optimum | Quantifies write-side waste without needing a reference run | Done, but deprioritized — not part of the Step 1-3 methodology going forward per direction |

### B. Known anomaly-pattern detectors

| Metric | What it measures | Why it matters | Status |
|---|---|---|---|
| Directory scan / rescan count | How many times each directory was listed via `os.listdir()`-class calls, and how many were scanned more than once | Finds "should have cached, didn't" waste | Done, visualized — found all 59 directories in `a1_c8` rescanned at least once, one 16 times |
| Failed open/stat count | Failed open/stat-class syscalls | Finds path-resolution errors and probing behavior | Done |
| Same-file reread (agent-induced vs. residual cross-stage reuse) | Rereads of the same file, split into behavioral overhead vs. expected reuse under the fixed action-unit sequence | The most direct evidence of whether the agent did unnecessary work | Done, cross-validated — `a1_c8` reread-after-backtrack count = 4, `a2_t2` = 2, both exactly matching independently-counted backtrack events; visualized |
| Retry-induced I/O bytes/ops (`action_unit_backtrack` phase) | I/O volume during retry/rollback windows | Quantifies the cost of retries | Done — `a1_c8`: 29 ops, 67.7MB read, 80.7KB write |
| Error-log read count | Whether the agent reads back its own or the workflow's log files | Detects debugging/exploration behavior | Done — currently 0 in every cell checked, a real measured value, not a missing metric |
| Checkpoint / metadata-write frequency | How many times checkpoint-shaped files are rewritten | Finds "should write once, rewrites repeatedly" patterns | Done, visualized — major finding: `cohort_info.json.tmp` rewritten 23 times in an 8-cohort cell vs. 1 time in a 1-cohort cell |
| Temporary file count / abandoned artifact count | Temp files created; files written but never read back | Finds wasted output | Done |
| Redundant-read fraction / non-productive I/O fraction | % of read bytes that are rereads; % of write bytes that are dead writes | A single summary "waste rate" number | Done |

### C. Supporting infrastructure

| Capability | What it does | Why it's needed | Status |
|---|---|---|---|
| Per-role I/O attribution | Attributes every byte to the agent role (GEO/TCGA/Statistician) that produced it | Locates which role is responsible for a finding | Done (`tool_call_attribution.csv`) |
| File lifecycle / reuse classification | Classifies each file as input-read-once, produced-read-many, dead-write, etc. | Underlies the Section B detectors | Done (`classify_reuse`) |
| Agent timeline, phase-time breakdown, concurrency view | Splits wall-clock time into LLM wait / tool execution / residual | Locates which phase is slow | Done, pre-existing |
| Interactive Call DAG (click a node to see code / LLM output) | Per-node drill-down into the generated code and the LLM's decision output for that step | The evidence tool for Step 3 diagnosis | Done this session — verified: 98/98 LLM-output lookups resolved on `a1_c8`, 12/12 code snippets resolved on the CloudLab `base` cell |

### D. Explicit gaps (not built)

| Metric | Why it's missing | Plan |
|---|---|---|
| Same-version reread (content/mtime aware) | Current reread detection uses `tool_call_id` windows, not content hashing or mtime comparison | Low priority, not scheduled |
| Output-inspection count | Listed in Overview.md §6.2, never implemented | Not scheduled |
| Rank-level I/O imbalance | No collected trace currently exercises multi-worker/multi-rank execution | Wait for a `--parallel-mode cohorts` or multi-worker trace |
| Run-to-run variance | Needs genuine repeated-task runs; the existing fanout cells vary task shape, not just repetition | Deferred per instruction — not manufacturing data for this |
| Sequential vs. random I/O | `read`/`write` syscalls carry no offset argument at the kernel level — an inherent limitation of syscall-argument-level eBPF tracing, not specific to GenoMAS or fixable by better parsing | Decided not to fix |
| Generic "sort and take top-N" ranking utility (Step 2 of this methodology) | Not yet built — the eight dimensions above have the underlying data but no consistent ranking code | Next planned work item |
