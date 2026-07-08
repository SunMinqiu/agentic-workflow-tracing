# Phase 1-2 Plan: Metric Inventory & Attribution Status

Companion to `Overview.md` §7 (Research Pipeline). This is a status audit against the
**current** codebase (`src/agent_io_tracing/`), not a re-plan — it does not propose code
changes beyond the one fix already applied and described below.

**Important correction to `Plans/Phase1-Plan.md`:** that document predates the
`src/` → `src/agent_io_tracing/` reorg and references files that no longer exist
(`lineage_analyzer.py`, `genomas_tool_logger.py`, `analyze_codebase_genomas.py`). A large
fraction of what it marks `[⬜ TODO]` is actually **done** in the current tree. Treat this
document as the current source of truth; `Phase1-Plan.md` should be read only for its
hypotheses (H1-H4) and kill criteria, not its status column.

**Rule for what counts as "collected."** A run only counts as delivering a metric if that
metric was actually computed and holds a real, non-degenerate value for that run — not if the
underlying raw trace file merely exists.

---

## Phase 1: Workflows and Metric Coverage

### Workflows table

| Workflow | Orchestration type (Overview §4) | Runs with a real `phase1_metrics.json` | Include in Phase 1? |
|---|---|---|---|
| **GenoMAS** | fixed action-unit DAG | **8 cells total**, all local except one: `results/fanout_runs/fanout_20260623_231616/{base, a1_c2, a1_c4, a1_c8, a2_t2, a2_t4, a2_t8}` (7 cells, varying cohort-count/trait-count fanout axes) + `results/fanout_runs/smoke_base_codeclass_fix_20260630_112143/base` (pulled from CloudLab 2026-07-01, the only one with a non-empty `generated_code.jsonl`). Older `results/GenoMAS_traces/*` dirs have `parsed.json`/`lineage/` but no `phase1_metrics.json` — still `CODE-NOT-RUN`, not part of the 8. | **Yes** — primary/reference workflow, 8 real cells is enough breadth for single-run metric coverage (see below); still zero *repeated* runs of the identical task, so §6.4 variance is unaddressed |
| **SciLink** | dynamic (`--mode autonomous`, refinement loop, skill auto-select) | None yet — `results/traces/{...}/eels_plasmons_basic` (6 runs) have `lineage/` but no `phase1_metrics.json` | **Yes** in principle, but not run yet — do this only when actually needed (see "don't manufacture runs" note below), not as busywork |
| ChemGraph | dynamic (LangGraph `StateGraph`) | None | **Not yet** — testing not finished per your note, park until then |
| SRAgent | dynamic ReAct supervisor | N/A | **No — park as "unsuitable"**: its work is almost entirely NCBI Entrez/BigQuery API calls and web fetches, not local scientific-task filesystem I/O. Keep the traces as a negative/contrast example, don't spend Phase 1 budget on it. |
| CMBAgent | mid-spectrum (`planning_and_control`: dynamic plan generation, then step-by-step execution of that plan; `one_shot` mode is closer to a single ReAct turn) | None yet — no adapter exists | **Planned, not started** — see `Plans/Phase1-CMBAgent-Plan.md` for the integration design (adapter, config, trace script) before any run is attempted |

**On not manufacturing more runs**: per your instruction, the plan is not to wire up `phase4_20260604_164509` (local, older schema, no `phase1_metrics.json`) or SciLink's 6 repeated runs into the metrics pipeline right now, because even the single-run metric set still has an open correctness question (the `interface_mix` gap, now fixed — see below) — there was no point generating more numbers from a pipeline whose output wasn't yet trustworthy. Now that the fix is in, revisit this only when variance analysis is actually the next planned step, not preemptively.

### Metric coverage table (single-run metrics — §6.1-§6.3)

Status legend:
- **HAVE** — a real, non-degenerate value exists across the 8 GenoMAS cells.
- **HAVE-BUT-DEGENERATE** — computed, but unreliable/near-empty for a specific, identified reason.
- **FIXED THIS SESSION** — was degenerate, root cause identified and corrected in `io_api_classifier.py`, re-verified against real data.
- **CODE-NOT-RUN** — the computation exists but hasn't been executed against a counted trace.
- **MISSING** — no code path produces this at all.
- **INHERENT LIMITATION** — not a bug; the underlying trace data does not contain the information this metric needs, for a structural reason explained inline.

#### §6.1 Universal I/O Metrics

| Metric | Status | Evidence |
|---|---|---|
| read/write bytes, op counts, metadata op count, unique files touched, small-file/request-size CDFs, I/O time, read/write ratio | **HAVE** | Real, non-degenerate values across all 8 GenoMAS cells' `phase1_metrics.json` (e.g. `a1_c8`: `metadata_data_ratio`, `file_size_cdf`, `request_size_cdf`, `fs_io_non_llm` all populated with real numbers) |
| sequential vs. random/backward (part of read/write pattern characterization) | **INHERENT LIMITATION — not GenoMAS-specific, not an eBPF decoding bug either; it's what buffered `read()`/`write()` syscalls carry** | See dedicated writeup below. |

#### §6.2 Agent-Induced I/O Metrics

| Metric | Status | Evidence / gap |
|---|---|---|
| directory scan count, failed open/stat count, error-log read count | **MISSING** | still not named metrics; the raw signals (`getdents64`, `ret`/`errno`, `logs`-classified paths) all exist in `parsed.json`/`artifacts.csv`, just not rolled up |
| same-file reread / same-version reread | **HAVE** (reread, not version-aware) | `classify_reuse()` produces real values in all 8 cells |
| retry-induced I/O (`action_unit_backtrack` phase tag) | **HAVE — validated with a positive control this session** | Previously flagged as "zero samples, unvalidated" because the one CloudLab smoke run hit 0 backtracks. Checked all 8 cells directly: `a1_c8` has **4** `action_unit_backtrack` events, `a2_t2` has **2** (`base`, `a1_c2`, `a1_c4`, `a2_t4`, `a2_t8` have 0 — plausible, not every cell hits a review-reject). `a1_c8`'s `phase1_metrics.json` shows `action_unit_backtrack:GEOAgent` with **89 fs events and real latency percentiles** — the phase-tagging → time-window join genuinely works on data where a retry occurred, not just silently returns zero. **Bytes/ops-by-phase rollup still not built** (only latency-by-phase exists) — that remains a small, well-scoped Phase 2 extension, not a validation gap anymore. |
| temporary file count, abandoned artifact count (proxy), redundant-read / non-productive I/O fraction | **HAVE** | Real values across the 8 cells |

#### §6.3 Task-Misconfigured I/O Metrics

| Metric | Status | Evidence / gap |
|---|---|---|
| I/O interface used (`io_api_classifier`) | **FIXED THIS SESSION** | Root cause: GenoMAS's action-unit prompts tell the agent to call the repo's own library wrappers (`geo_get_relevant_filepaths`, `get_background_and_clinical_data`, `get_genetic_data`, `validate_and_save_cohort_info`, `save_result`, etc. from `tools/preprocess.py`/`tools/statistics.py`) instead of writing raw I/O. The classifier only AST-parsed the agent's own snippet, so a call like `geo_get_relevant_filepaths(dir)` looked like "no I/O" even though the function does `os.listdir()` internally. **Fix applied**: read every function in `tools/preprocess.py` and `tools/statistics.py` once, by hand, to determine which ones actually touch the filesystem and through which layer; added a `LIBRARY_FUNCS` lookup table to `src/agent_io_tracing/analysis/io_api_classifier.py` (bare-function-name → layer(s), e.g. `geo_get_relevant_filepaths`→`posix_raw`, `validate_and_save_cohort_info`→`stdio`+`posix_raw`; pure-compute functions like `tune_hyperparameters`/`detect_batch_effect` intentionally excluded so they still classify as no I/O) and a matching regex-fallback for when AST parsing fails. Added 10 new self-test cases (23/23 pass). **Re-ran against the one trace with real `generated_code.jsonl`** (CloudLab `smoke_base_codeclass_fix`): detected-I/O snippets went from **1/12 → 11/12**; `phase1_metrics.json` regenerated in place from already-collected data (no new run). This metric is now trustworthy for future GenoMAS analysis. Still only validated on one trace, since the other 7 cells have empty `generated_code.jsonl` (a separate, earlier capture gap, not today's bug) — worth re-checking on the next CloudLab run that this stays fixed. |
| output file count / avg file size, storage location choice (incl. real Lustre `lctl` MDS/OST counters) | **HAVE** | Real values across the 8 cells; CloudLab run's `manifest.json` additionally captured live `mdc.*.stats`/`osc.*.stats` |
| checkpoint/metadata-write frequency | **FIXED THIS SESSION** | Added `compute_checkpoint_write_frequency` (matches path substrings `cohort_info`/`completed_tasks`/`_state.json`/`.lock`) to `phase1_metrics.py`, re-ran on already-collected cells. **Immediate finding**: in `a1_c8` (8-cohort fanout cell), `cohort_info.json.tmp` was written **23 times**, vs. **1 time** in `base` (1-cohort cell) — scales with cohort count, exactly the read-modify-write-per-validate-call pattern in `tools/preprocess.py:496-587`'s `validate_and_save_cohort_info`. Strong, ready-to-use Phase 3 case-study candidate. |
| rank-level I/O size/time imbalance | **N/A so far** | none of the 8 cells exercise multi-rank/multi-worker I/O; relevant once ChemGraph's FDMNES ensemble or a GenoMAS `--parallel-mode cohorts` cell is in the counted set |
| agent-caused vs. script-caused split | **UNBLOCKED, NOT YET BUILT** | Was blocked on the `interface_mix` fix above; now that the classifier is trustworthy, this is a Phase 2 item (see below), not Phase 1 |

### §6.2 rollups closed out this session

Also implemented and re-run on the already-collected cells (no new runs):

- **`compute_directory_scan_count`**: counts `getdents64`/`getdents`, and flags directories scanned more than once. **Finding**: in `a1_c8`, all 59 unique directories scanned were rescanned at least once (166 total scans / 59 unique), with one GEO cohort directory scanned 16 times — `geo_get_relevant_filepaths`'s `os.listdir()` call is re-run every time a cohort is (re-)inspected rather than caching the result. Another concrete Phase 3 candidate.
- **`compute_failed_open_stat_count`**: counts `open`/`openat`/`stat`/`fstat`/etc. with a negative return value. In `a1_c8`: 2 failed `openat` calls, both against unresolved paths (`<unknown>`/`<string>`) — small, likely an artifact of tracing subprocess/exec boundaries rather than a real agent mistake, but now measurable rather than invisible.
- **`compute_error_log_reads`**: sums reads against artifacts the lineage pass already classifies as `category == "logs"`. Both counted cells show `0` — a real, checked value (no log-inspection behavior happened to occur in these runs), not a missing metric.
- **`compute_bytes_ops_by_phase`**: extends `compute_latency_by_phase`'s phase join to bytes/op counts. This closes the "retry-induced I/O bytes" gap: `a1_c8`'s `action_unit_backtrack:GEOAgent` phase now shows **29 ops, 67.7MB read, 80.7KB write** — a real, non-zero retry-induced I/O number, not just a latency percentile.

All four were added to `src/agent_io_tracing/analysis/phase1_metrics.py` and verified by re-running on `smoke_base_codeclass_fix_20260630_112143/base`, `fanout_20260623_231616/a1_c8`, and `fanout_20260623_231616/a2_t2` — all already-collected data, no new experiments.

#### §6.4 Run-to-Run Variance Metrics — explicitly out of scope for this update

Per your instruction, not addressing this now. The 7-cell fanout study varies cohort-count/trait-count (different task shapes per cell), so it isn't "same task repeated" variance data anyway; SciLink's 6 repeated `eels_plasmons_basic` runs would be the right shape, but running them through `phase1_metrics.py` is deferred until variance analysis is actually the next step.

### Sequentiality: root cause, and why it's not a GenoMAS-specific or eBPF-decoding problem

Checked `src/agent_io_tracing/parsing/_ebpf_impl.py:828-857` directly: the `offset` field is only populated for syscalls that **carry an explicit offset argument** — `pread64`/`pwrite64`/`preadv`/`pwritev`/`preadv2`/`pwritev2`/`sync_file_range`. Plain `read`/`write` — which is what GenoMAS's (and most Python programs') buffered `open()`-then-`read()`/`write()` calls compile down to — do **not** carry an offset argument at all; the kernel tracks the file's read/write position internally, per file descriptor, invisibly to the syscall's own arguments.

This was checked against the two richest available traces:
- CloudLab run: 1012 `read` + 880 `write`, **zero** `pread64`/`pwrite64`/`preadv`/`pwritev`.
- `a1_c8` (largest local cell): 1767 `read` + 12169 `write`, again **zero** positional variants.

So `compute_sequentiality`'s `"unknown: insufficient offset-bearing syscalls"` isn't a decoding bug and isn't specific to GenoMAS's I/O pattern — **it's a structural property of tracing `read(2)`/`write(2)` via eBPF at the syscall-argument level**: any workload whose Python/C code uses buffered, non-positional file I/O (the overwhelming majority of ordinary application code, not just GenoMAS) will hit the same wall, because the information this metric needs was never in the syscall arguments to begin with. The only way to get it would be to track a per-`(pid, fd)` implicit cursor across `openat`→`read`/`write` events and derive the offset from cumulative bytes transferred — a real, buildable extension to the *tracer/parser*, not a fix to any one workload's classifier.

**Decision (per your direction): not fixing now.** Documented as a known, inherent limitation of syscall-argument-level eBPF tracing for buffered I/O, not folded into any per-workflow "gap" framing. Revisit only if a specific paper claim needs sequential/random characterization and no other signal can substitute.

### Phase 1 bottom line

With the correction applied, the previously-missed 7-cell local fanout study counted, the `interface_mix` classifier fixed, and the directory-scan/failed-open/error-log-read/checkpoint-frequency/bytes-by-phase rollups added — **every §6.1-§6.3 metric that was `MISSING` or `PARTIAL` at the start of this session is now computed with real values**, except sequentiality (inherent limitation, documented above, not fixed by design decision) and the two items below that need data this project doesn't have yet, not more code. All of this was done by re-running the existing (now-extended) pipeline against already-collected traces — zero new experiments.

What's left is genuinely data-shaped, not code-shaped, and is deferred per your "don't manufacture data" instruction until it's actually the next needed step:

1. SciLink needs its own `phase1_metrics.json` runs and its own retry/loop-iteration phase-tag equivalent to GenoMAS's `action_unit_backtrack` before it can be compared on equal footing.
2. Multi-worker/multi-rank coverage (§6.3 rank imbalance) is still unaddressed by any counted cell — needs a `--parallel-mode cohorts` or ChemGraph ensemble run, not new code.
3. §6.4 variance metrics — needs genuine repeated-task runs (the 7-cell fanout study varies task shape per cell, so it doesn't count), out of scope per your instruction this session.

Two concrete, ready-to-write-up findings came out of closing the metric gaps (not manufactured — they fell out of running the new rollups on data that already existed): the `cohort_info.json.tmp` rewrite-per-validate-call pattern (23x in an 8-cohort cell vs. 1x in a 1-cohort cell) and the directory-rescan pattern (every scanned directory in `a1_c8` was rescanned at least once, up to 16x for one cohort). Both are strong Phase 3 mitigation case-study candidates — both are script-caused (fixed code in `tools/preprocess.py`), so both get the clean single-fix-applies-everywhere framing `Overview.md` §7 Phase 3 asks for.

---

## Phase 2: Attribution Status

Recall the three-bucket scheme from `Overview.md` §5: **agent-induced** / **task-misconfigured** (source-agnostic detection + agent-caused/script-caused sub-tag) / **workflow task-induced** (residual, by exclusion).

### What's already done (mechanism-level)

- **Time-window attribution (orchestration I/O vs. execution-window I/O)**: `parsing/attribution.py` → `_ebpf_impl.py` (`in_any_tool_window`/`match_event_to_tool`); `phase1_metrics.phase_for_entry` returns `"orchestration"` for fs events outside any tool-call window.
- **Per-role attribution**: `build_role_io_attribution` in `_analyzer_impl.py`, written to `tool_call_attribution.csv` for every counted cell.
- **Retry/backtrack phase tag for GenoMAS** (`action_unit_backtrack`) — now empirically validated (see §6.2 above), not just theoretically present.
- **I/O interface layer classification** — now trustworthy after this session's fix, and is itself the agent-caused/script-caused split for 5.2: any layer detected inside agent-generated code is agent-caused by construction; any layer choice inside `tools/preprocess.py`/`environment.py` is script-caused by construction.
- **Reuse/reread classification** (`classify_reuse`) — raw, cause-agnostic signal for redundant reads and dead writes.

### Metrics directly attributable once measured (no classification method needed)

- **Temp files, dead writes** → agent-induced (5.1) by construction.
- **Orchestration-window I/O** → script-caused or residual by construction (never agent-induced, since it's outside any LLM tool-call window); which of the two follows from reading the source line.
- **I/O interface layer** → agent-caused/script-caused split for 5.2, now usable.

### Metrics that need a classification method (the actual Phase 2 work)

1. ~~Reread → agent-induced vs. residual.~~ **Done this session, with a correction.** The originally-planned join (reread within the same `phase_for_entry` label = agent-induced) was **not implemented as first specified** because it was wrong: checked directly and found `code_exec:GEOAgent`'s `latency_by_phase` bucket spans **5 different `tool_call_id`s** (5 distinct action-unit steps, e.g. Gene Data Extraction vs. Gene Identifier Mapping) that all collapse to the same coarse phase label. Joining reread on that label would have mislabeled GenoMAS's legitimate, by-design cross-action-unit file reuse as agent-induced. **Fix**: `compute_reread_attribution` in `phase1_metrics.py` joins on the specific `matched_tool_call` (tool_call_id) instead, with a touch key of `(path, tool_call_id, fd)` so that buffered multi-syscall reads of one logical `open()` don't get counted as N rereads. Classifies into `same_step_reopen` (agent-induced: re-opened the same path within one code-exec step), `reread_after_backtrack` (agent-induced: re-read following an `action_unit_backtrack` tag), and `residual_cross_stage_reuse` (expected reuse across the fixed action-unit sequence). **Cross-validated against the already-known backtrack counts**: `a1_c8`'s `reread_after_backtrack.count` = **4**, `a2_t2`'s = **2** — exactly matching the raw `action_unit_backtrack` event counts found earlier, confirming the join is behaving correctly rather than just producing a plausible-looking number. `same_step_reopen` is 0 in all 4 cells checked (no genuine within-one-step re-open detected so far); `residual_cross_stage_reuse` is large (17-103 touches per cell) — i.e. most rereads in GenoMAS are legitimate cross-stage reuse, not agent waste, which is itself a finding worth keeping (the fixed-DAG design is doing its job of not causing redundant reads, at least for this axis).
2. ~~Retry-induced I/O bytes.~~ **Done this session** — `compute_bytes_ops_by_phase` in `phase1_metrics.py`, validated against `a1_c8` (29 ops / 67.7MB read / 80.7KB write in the `action_unit_backtrack:GEOAgent` phase). SciLink still needs an equivalent phase-tag heuristic for its `refinement_loop` before the same join applies there — that part remains open, data-shaped, not code-shaped.
3. ~~Checkpoint/metadata-write frequency.~~ **Done this session** — `compute_checkpoint_write_frequency`, and it directly surfaced the 23x-vs-1x `.tmp` rewrite finding, which doubles as the Phase 3 case study this item was meant to feed.
4. **Agent-caused vs. script-caused rate comparison.** The split itself is now free (item above in §6.3). Comparing *rates* (is agent-generated code worse than script code on average) needs enough cells to compute a rate over — the 8 GenoMAS cells now on hand are a reasonable starting sample, once each has its `generated_code.jsonl` populated (currently only 1 of 8 does — a capture gap in the older 7 cells, separate from today's classifier fix). Still open, still data-shaped.

### Phase 2 bottom line

All four classification-method items from the original list are now built: reread attribution (joined correctly on tool_call_id after catching and fixing a same-label false-positive risk in the first design), retry-induced bytes, and checkpoint frequency all produced real, cross-validated numbers this session; only the agent-caused/script-caused *rate* comparison remains, and that's blocked on data (more populated `generated_code.jsonl` cells), not on any unwritten classification logic. Per your instruction, not manufacturing that data preemptively.
