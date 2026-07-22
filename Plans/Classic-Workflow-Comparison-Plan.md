# Classic (Non-Agent) Workflow Comparison Plan

## Goal

Add the ability to trace and characterize **classic, non-agent scientific workflows** (starting with [1000genome-workflow](https://github.com/pegasus-isi/1000genome-workflow)) as a comparison baseline against the existing agentic targets (pi, genomas, scilink, sragent, chemgraph).

Everything here is **additive**. No existing file is modified, no existing capability is removed. The classic path reuses the entire tracing + I/O-characterization half of the pipeline as-is and simply bypasses the LLM-specific analysis half.

## Decisions (already confirmed)

- **Execution mode: a Python driver that reproduces the DAG — NOT Pegasus/HTCondor.** The upstream 1000genome default "local" site still dispatches through the host `condor_schedd`, so tasks are children of `condor_starter`, **outside** the launcher's process tree — the root-pid tracer would miss the real compute. We therefore bypass Pegasus entirely and run the workflow's per-task scripts (`individuals`, `individuals_merge`, `sifting`, `mutation_overlap`, `frequency`) from a driver that honors the real DAG (see the DAG figure and concurrency decision below — **not** a linear chain). Every task is then a descendant of the launcher PID, so the existing `bcc_tracer` root-pid + `sched_process_fork` tracking captures all of it with **no tracer changes**. This is also exactly the compute-task I/O we want to characterize (not Pegasus orchestration I/O).
- **Driver concurrency: faithful DAG parallelism, under a global worker cap.** The driver reproduces the real DAG's parallelism rather than serializing: `individuals` chunk jobs run in parallel and `sifting` runs concurrently with the individuals branch; after both `individuals_merge` and `sifting` finish, `mutation_overlap` and `frequency` run in parallel per population. The driver owns the join/barrier points. This keeps the measured I/O timeline and concurrency representative of the real workflow (a strictly serial driver would misrepresent both). Parallel tasks are still descendants of the launcher PID, so fork-following captures them.
  - **Global concurrency cap is required.** Backgrounding every ready task would bypass HTCondor's resource limits, and 1/2/4-chromosome runs would then hit wildly different (and unrealistically high) concurrency. The driver is therefore a **Python** driver with a single global `--max-workers` pool that schedules the entire DAG **across all chromosomes** (not per-chromosome), so concurrency is bounded and comparable regardless of scale. `--individual-jobs` only sets the per-chromosome chunk count fed into that global scheduler.
- **Per-task sandbox directories (data-race fix).** Pegasus gives every job its own scratch dir; our driver must too. Running all tasks in one shared `$OUT/work` is unsafe — `mutation_overlap.py` and `frequency.py` each untar the same `chr<N>n.tar.gz` into `./chr<N>n/`, and parallel jobs (multiple populations, both consumers) would write the same path concurrently and race. So the driver gives **each job its own CWD** `$WORK/tasks/<task-id>/`, symlinks that job's inputs into it, and on success registers the job's declared outputs into `$WORK/artifacts/`. A downstream task is scheduled **only when all its declared input artifacts are present**; downstream inputs are symlinked from `$WORK/artifacts/` into the consumer's task CWD. If any task fails, its dependents are not run and the driver ultimately exits non-zero. All jobs still run through the one global `--max-workers` pool. We **accept the small extra "direct-driver orchestration I/O"** (driver-created dirs/symlinks and the artifact registry) as the cost of race-free, Pegasus-like job isolation; it is orders of magnitude below the task compute I/O and does not distort the characterization.
- **Per-cell isolated working directory.** Because the driver `chdir`s to run and the workflow writes outputs into the CWD, each cell copies/symlinks its inputs into `$OUT/work` and executes there. This keeps cell outputs isolated (no cross-run contamination or shared Pegasus/state), so reruns and parallel cells give clean, comparable I/O measurements.
- **Figures/analyses: what actually runs.** Verified against the code — the only syscall/`parsed.json`-based figure generator is `per_run_io_char` (with `size_bins` as its internal helper, no CLI of its own). `artifact_sizes` also consumes `parsed.json` but emits the `artifact_sizes.json` sidecar (no figure). `io_api_classifier` reads `generated_code.jsonl` (LLM-generated code) and is **irrelevant** to classic workflows. All LLM-dependent analyses (`summary`, `parallelism`, `lineage`, `phase1_metrics`) are skipped.
  - Classic post-processing = `parsing.ebpf` → `analysis.per_run_io_char` (+ `size_bins` internal) → `analysis.artifact_sizes` (sidecar). `io_api_classifier` skipped.
- **`viz.trace` skipped for classic runs.** It early-exits when `parsed.json.tool_calls` is empty, so it produces nothing for a non-agent run anyway. We do not modify it (preserves the "only add files" constraint). If a syscall-only timeline is wanted later, add a **new** viz module rather than editing `viz.trace`.

## Why the change is small

The pipeline splits cleanly into two halves:

**Tracing half — already workflow-agnostic, reused verbatim:**

- `tracing/bcc_tracer.py` — traces any `--root-pid` and follows `fork`/`clone` into children via `sched_process_fork` ([bcc_tracer.py:156](../src/agent_io_tracing/tracing/bcc_tracer.py#L156)). It has no concept of an "agent".
- `tracing/lustre_counters.py` — Lustre OST counter sampler, PID-agnostic.
- `parsing/ebpf.py` — turns `ebpf_events.log` into `parsed.json`, pure syscall parsing with zero LLM coupling.
- `experiments/run_manifest.py` — run metadata, generic.

**Analysis half — two classes:**

- Pure I/O, consumes only `parsed.json` → **reused as-is**: `analysis/per_run_io_char.py` (with `analysis/size_bins.py` as its internal helper) and `analysis/artifact_sizes.py`. (`analysis/io_api_classifier.py` is **not** a `parsed.json` analysis — it reads `generated_code.jsonl` and does not apply to classic runs.)
- LLM-event dependent → **bypassed for classic runs**: `analysis/summary.py` (hard `raise` on missing `pi_events.jsonl` at [summary.py:350](../src/agent_io_tracing/analysis/summary.py#L350)), `analysis/parallelism.py` ([parallelism.py:290](../src/agent_io_tracing/analysis/parallelism.py#L290)), `lineage/analyzer.py`, `analysis/phase1_metrics.py`.

## New files

### 1. `src/agent_io_tracing/adapters/classic/launcher.py`  (~80 lines)

A generic process launcher modeled on [adapters/genomas/launcher.py](../src/agent_io_tracing/adapters/genomas/launcher.py) but with the LLM logger and `task_info.json` slicing removed.

Responsibilities:

- CLI: `launcher.py <work_dir> <log_dir> --cmd "<command>" [--repo <dir>] [--input SRC[:DEST] ...] [--copy-input SRC[:DEST] ...] [--env KEY=VAL ...] -- <extra args>`.
- **Isolated CWD:** the launcher `chdir`s to `work_dir`; the driver in turn runs each job in its own `$WORK/tasks/<task-id>/` (per-task sandbox, see Decisions). Do **not** `chdir` into the shared repo checkout.
- **Repo passed explicitly to the driver:** `--repo` does not change CWD; the launcher exports it as the `CLASSIC_WORKFLOW_REPO` environment variable (and passes `--repo` through in `--cmd` if templated). The driver resolves each per-task script as `$CLASSIC_WORKFLOW_REPO/bin/<script>` (exact subpath confirmed against the repo layout at build time). This gives a defined resolution rule instead of relying on CWD.
- **Input staging format:** `--input SRC[:DEST]` creates a **read-only symlink** at `work_dir/DEST` → `SRC` (`DEST` defaults to `basename(SRC)`); this is the default for the large read-only inputs (the already-decompressed main/annotation `.vcf`, the shared `columns.txt` and population files). Use `--copy-input SRC[:DEST]` for the few inputs a task mutates in place. This keeps each cell's `work_dir` self-contained without duplicating gigabytes of reference data. Note the tasks read decompressed `.vcf`, so VCFs are decompressed once upstream (deploy/`prepare_input.sh`) and symlinked here — the launcher does not gunzip per cell.
- **Write empty stub logs** `pi_events.jsonl` and `tool_calls.log` into `log_dir` — done as part of the pre-run setup (below), before self-stopping. These let the shared trace script's downstream steps run without hitting the LLM-file `raise` paths, and let us still opt into the LLM steps behind a non-empty check.
- **Self-stop for a race-free trace boundary (see sync note).** After input staging and stub creation are fully done — but before spawning the driver — the launcher sends itself `SIGSTOP` (`os.kill(os.getpid(), SIGSTOP)`). Only after the trace script observes the stop and issues `SIGCONT` does the launcher `subprocess.Popen(cmd)` the driver in `work_dir` and wait for exit.
- Capture the workflow's stdout/stderr into `log_dir` (`classic.stdout` / `classic.stderr`), and propagate the child's exit code as the launcher's exit code.
- Register `adapters/classic/__init__.py` so it is importable as `python -m agent_io_tracing.adapters.classic.launcher`.

Design notes:
- **Race-free start (fixes a real bug in the copied pattern).** The existing scripts `kill -STOP $AGENT_PID` from the *outside* right after launch. For this launcher that is racy: staging happens inside the launcher, so the external STOP could land before staging (staging I/O then gets counted as workload), after staging, or even after the driver already started — the trace boundary is nondeterministic. Instead the **launcher self-`SIGSTOP`s after staging+stubs and before the driver**; the trace script waits until the process is in state `T` (stopped), starts the tracer, waits for it ready, then `SIGCONT`s. Result: the trace begins at a precise point — staging excluded, driver execution included — with **no tracer change**.
- The `--cmd` for 1000genome is the thin driver that calls the per-task scripts directly — no Pegasus, no `pegasus-plan`, no `condor_submit` — so the whole DAG runs inside this process tree. The driver must reproduce the **real** DAG (the naive linear chain is wrong — it omits `individuals_merge`, which produces the `chr<N>n.tar.gz` that the downstream scripts require):

  ```
  individuals[*] ──→ individuals_merge ──┐
                                         ├─→ mutation_overlap[*]
  sifting ───────────────────────────────┘
                                         └─→ frequency[*]
  ```

  So per chromosome: run all `individuals` chunk jobs, then `individuals_merge` (→ `chr<N>n.tar.gz`); `sifting` runs independently of the individuals branch; once both `individuals_merge` and `sifting` are done, run `mutation_overlap` and `frequency` per population. The driver is a **Python** file added under `adapters/classic/` (e.g. `run_1000genome.py`) that builds the whole cross-chromosome DAG and executes it through one global `--max-workers` pool (see the concurrency decision above). A shell driver is insufficient because it cannot enforce a single global worker cap across chromosomes.

### 2. `scripts/trace_script_bcc_1000genome.sh`

Copied from [scripts/trace_script_bcc_genomas.sh](../scripts/trace_script_bcc_genomas.sh) with these edits:

- **Change the sync handshake to launcher-driven self-stop:** launch the launcher (it stages inputs, writes stubs, then `SIGSTOP`s itself), poll `/proc/$AGENT_PID/stat` (or `ps -o state=`) until state is `T`, then start `bcc_tracer`, wait for `ebpf_events.log` to appear, then `kill -CONT $AGENT_PID`. Do **not** issue the external `kill -STOP` the genomas script uses — that race is exactly what self-stop removes.
- Drop GenoMAS-specific preflight (`utils.llm` importability check, `--smoke-traits`, `fullpipeline`, `--quick-test`, `--parallel-mode`).
- Replace the agent invocation with `python -m agent_io_tracing.adapters.classic.launcher "$WORK" "$OUT" --cmd "$WORKFLOW_CMD" --repo "$WORKFLOW_REPO" --input ... ` (each cell gets its own `$WORK = $OUT/work`).
- Post-processing section runs only: `parsing.ebpf` → `analysis.per_run_io_char` → `analysis.artifact_sizes`. `size_bins` is pulled in internally by `per_run_io_char` (no separate invocation). `io_api_classifier` and `viz.trace` are **not** run for classic cells.
- Keep the LLM-only steps (`summary`, `parallelism`, `lineage`, `phase1_metrics`) present but guarded by `if [ -s "$ws_out/pi_events.jsonl" ]`, so the same script still works if a future classic target does emit LLM events (they just skip when the stub is empty).

### 3. `config/config_1000genome.env`

Copied from [config/config_genomas.env](../config/config_genomas.env):

- Remove LLM keys (`GENOMAS_MODEL`, `GENOMAS_LLM_REPLAY`, API-key plumbing, `INTER_RUN_SLEEP_SEC` rate-limit padding).
- Add `WORKFLOW_REPO` (path to the checked-out 1000genome-workflow), `WORKFLOW_CMD` (the Python DAG driver), and the input lists.
- **Inputs — decompressed, and mostly shared:**
  - **Per chromosome, two dedicated files:** the main VCF and the annotation VCF. The tasks read the **decompressed `.vcf`**, not `.vcf.gz` — both the main and annotation VCFs must be decompressed during staging (or pre-decompressed once and symlinked). The annotation VCFs are **not** in the repo; fetch them beforehand via the upstream `prepare_input.sh` (deploy step or manual).
  - **Shared across all chromosomes:** a single `columns.txt` and the selected population file set (`ALL`, plus `EUR/EAS/AFR/AMR/SAS/GBR` if the population axis is later expanded). Stage these once per cell, not per chromosome.
- **`WORKLOADS` scale axis = number of chromosomes** — encoded in the cell name (`name|n_chromosomes|rep|extra`). The number of `individuals` chunk jobs is a **task-partition / concurrency** parameter, *not* input size — kept separate (fixed by default; swept as its own axis only for an explicit concurrency study).

Default experiment matrix (runnable baseline):

| Parameter | Value |
|---|---|
| scale | 1 / 2 / 4 chromosomes |
| chromosomes | `1` ; `1,2` ; `1,2,3,4` |
| individual-jobs | fixed **2** |
| max-workers | fixed **4** |
| populations | fixed **ALL** only |
| repetitions | **3** |

Populations are pinned to `ALL` for the baseline on purpose: upstream `frequency.py` runs **1000 iterations per population** and emits many files, so defaulting to all seven populations would make even a single-chromosome cell very heavy. The population axis is a deliberate later extension, not part of the runnable baseline.

### 4. (Optional) `scripts/deploy_1000genome_to_client.sh`

Copied from [scripts/deploy_genomas_to_client.sh](../scripts/deploy_genomas_to_client.sh), with the uv-venv dependency set swapped for what the per-task scripts need (numpy etc.). **No Pegasus/HTCondor** — the direct driver does not use them. It must also run the upstream `prepare_input.sh` once to download the annotation VCFs (see inputs note below), since those are not shipped in the repo. Defer this until the cluster dependency-install approach is confirmed; if dependencies + inputs are staged manually, this file is not needed for a first run.

## Compatibility checks (read-only, verified)

- `analysis/per_run_io_char.py` consumes only `parsed.json` — confirmed compatible; `size_bins` is its internal helper (no CLI).
- `analysis/artifact_sizes.py` consumes `parsed.json`, stats real files, writes `artifact_sizes.json` sidecar — compatible, no LLM dependency.
- `analysis/io_api_classifier.py` reads `generated_code.jsonl` — **not applicable** to classic runs; skipped.
- `viz/trace.py` early-exits on empty `parsed.json.tool_calls` — produces nothing for classic runs; **not invoked** (left unmodified).
- `bcc_tracer` fork-following reaches all tasks because the direct driver keeps them as descendants of the launcher PID.

## Risks

- **Process-tree escape (designed out).** Pegasus/HTCondor spawns tasks under `condor_starter`, outside the launcher subtree — the direct Python driver avoids Pegasus entirely, so this cannot happen. Revisit only if full Pegasus orchestration ever becomes a measurement target (would need comm/cgroup-based tracer filtering — a tracer change, out of scope here).
- **Empty-LLM-log assumptions.** A few downstream modules assume `pi_events.jsonl` exists and is non-empty. Handled by (a) writing empty stubs in the launcher and (b) `-s` guards in the new trace script.
- **Driver fidelity.** Calling the per-task scripts directly means we, not Pegasus, own task ordering, concurrency, and file staging. The driver must reproduce the real DAG — `individuals[*] → individuals_merge` and `sifting`, both feeding `mutation_overlap[*]` / `frequency[*]` (the figure above), **not** a linear chain — enforce the global worker cap, and stage the right decompressed inputs per cell. Otherwise the I/O profile won't match a real run. Validate against the repo's DAX / task definitions.

## Effort

- Generic launcher (staging + stubs + self-`SIGSTOP`): ~100 lines.
- 1000genome Python DAG driver (dependency graph + global worker pool + per-task script invocation): the substantive piece, ~150–250 lines, and the part that most needs validation against the repo's DAX.
- Trace script + config + optional deploy: copy-and-trim.

Roughly 1–1.5 days, dominated by getting the DAG driver faithful (task order, decompressed inputs, global concurrency cap). The process-tree tracing risk is designed out by avoiding Pegasus.

## Suggested build order

1. `adapters/classic/launcher.py` + `__init__.py` (generic, workflow-agnostic).
2. The 1000genome Python DAG driver `adapters/classic/run_1000genome.py` — per-task sandboxes, artifact registry, global worker pool, fail-fast — reproducing the DAG without Pegasus.
3. `scripts/trace_script_bcc_1000genome.sh`.
4. `config/config_1000genome.env`.
5. Smoke test on a small 1000genome input; verify `ebpf_events.log`, `parsed.json`, the `per_run_io_char` figures, and the `artifact_sizes.json` sidecar all land, and that tasks show up in the trace (fork-following works).
6. (Optional) `deploy_1000genome_to_client.sh` once cluster deps are settled.
