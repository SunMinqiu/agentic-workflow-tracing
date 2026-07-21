# Classic (Non-Agent) Workflow Comparison Plan

## Goal

Add the ability to trace and characterize **classic, non-agent scientific workflows** (starting with [1000genome-workflow](https://github.com/pegasus-isi/1000genome-workflow)) as a comparison baseline against the existing agentic targets (pi, genomas, scilink, sragent, chemgraph).

Everything here is **additive**. No existing file is modified, no existing capability is removed. The classic path reuses the entire tracing + I/O-characterization half of the pipeline as-is and simply bypasses the LLM-specific analysis half.

## Decisions (already confirmed)

- **Execution mode: local / shell.** The workflow runs so that all compute tasks stay inside the launcher's process tree (Pegasus `local` site, or a plain bash/python driver). This lets the existing `bcc_tracer` root-pid + `sched_process_fork` tracking capture every task automatically, with no tracer changes.
- **Figures: I/O characterization only.** Reuse the `parsed.json`-only analyses (`per_run_io_char`, `size_bins`, `io_api_classifier`, `artifact_sizes`). Skip all LLM-dependent analyses (`summary`, `parallelism`, `lineage`, `phase1_metrics`) for classic runs.

## Why the change is small

The pipeline splits cleanly into two halves:

**Tracing half — already workflow-agnostic, reused verbatim:**

- `tracing/bcc_tracer.py` — traces any `--root-pid` and follows `fork`/`clone` into children via `sched_process_fork` ([bcc_tracer.py:156](../src/agent_io_tracing/tracing/bcc_tracer.py#L156)). It has no concept of an "agent".
- `tracing/lustre_counters.py` — Lustre OST counter sampler, PID-agnostic.
- `parsing/ebpf.py` — turns `ebpf_events.log` into `parsed.json`, pure syscall parsing with zero LLM coupling.
- `experiments/run_manifest.py` — run metadata, generic.

**Analysis half — two classes:**

- Pure I/O, consumes only `parsed.json` → **reused as-is**: `analysis/per_run_io_char.py`, `analysis/size_bins.py`, `analysis/io_api_classifier.py`, `analysis/artifact_sizes.py`.
- LLM-event dependent → **bypassed for classic runs**: `analysis/summary.py` (hard `raise` on missing `pi_events.jsonl` at [summary.py:350](../src/agent_io_tracing/analysis/summary.py#L350)), `analysis/parallelism.py` ([parallelism.py:290](../src/agent_io_tracing/analysis/parallelism.py#L290)), `lineage/analyzer.py`, `analysis/phase1_metrics.py`.

## New files

### 1. `src/agent_io_tracing/adapters/classic/launcher.py`  (~80 lines)

A generic process launcher modeled on [adapters/genomas/launcher.py](../src/agent_io_tracing/adapters/genomas/launcher.py) but with the LLM logger and `task_info.json` slicing removed.

Responsibilities:

- CLI: `launcher.py <work_dir> <log_dir> --cmd "<command>" [--repo <dir>] [--env KEY=VAL ...] -- <extra args>`.
- `os.chdir(repo)` if `--repo` given, then `subprocess.Popen(cmd, shell-split)` and wait for exit.
- Capture the workflow's stdout/stderr into `log_dir` (`classic.stdout` / `classic.stderr`), and propagate the child's exit code as the launcher's exit code.
- **Before launch, write empty stub logs** `pi_events.jsonl` and `tool_calls.log` into `log_dir`. This is the single trick that lets the shared trace script's downstream steps run without hitting the LLM-file `raise` paths — and lets us still opt into the LLM steps behind a non-empty check.
- Register `adapters/classic/__init__.py` so it is importable as `python -m agent_io_tracing.adapters.classic.launcher`.

Design note: keeping this as a real launcher (rather than starting the workflow inline in bash) preserves the STOP → start tracer → wait-ready → CONT synchronization the other trace scripts rely on, so the tracer is attached before the workload touches the FS.

### 2. `scripts/trace_script_bcc_1000genome.sh`

Copied from [scripts/trace_script_bcc_genomas.sh](../scripts/trace_script_bcc_genomas.sh) with these edits:

- Keep the STOP → start `bcc_tracer` → wait for `ebpf_events.log` → CONT → `wait` structure unchanged.
- Drop GenoMAS-specific preflight (`utils.llm` importability check, `--smoke-traits`, `fullpipeline`, `--quick-test`, `--parallel-mode`).
- Replace the agent invocation with `python -m agent_io_tracing.adapters.classic.launcher "$WORK" "$OUT" --cmd "$WORKFLOW_CMD" --repo "$WORKFLOW_REPO" ...`.
- Post-processing section runs only: `parsing.ebpf` → `analysis.per_run_io_char` → `analysis.size_bins` → `viz.trace`.
- Guard the LLM-only steps with `if [ -s "$ws_out/pi_events.jsonl" ]` so the same script still works if a future classic target does emit LLM events (it just skips them when the stub is empty).
- Verify `viz.trace` tolerates empty `pi_events.jsonl`; if it does not, wrap it in the same `-s` guard. Any such change lives in this new script only — the viz source is not touched.

### 3. `config/config_1000genome.env`

Copied from [config/config_genomas.env](../config/config_genomas.env):

- Remove LLM keys (`GENOMAS_MODEL`, `GENOMAS_LLM_REPLAY`, API-key plumbing, `INTER_RUN_SLEEP_SEC` rate-limit padding).
- Add `WORKFLOW_REPO` (path to the checked-out 1000genome-workflow) and `WORKFLOW_CMD` (the local/shell driver command).
- Redefine `WORKLOADS` cells around a **non-agent workload axis** — e.g. input size / number of chromosomes / individuals × repetition — encoded in the cell name (`name|scale|rep|extra`).

### 4. (Optional) `scripts/deploy_1000genome_to_client.sh`

Copied from [scripts/deploy_genomas_to_client.sh](../scripts/deploy_genomas_to_client.sh), with the uv-venv dependency set swapped for 1000genome's (numpy etc., plus Pegasus if the Pegasus driver is used). Defer this until the cluster dependency-install approach is confirmed; if dependencies are installed manually, this file is not needed for a first run.

## Compatibility checks (read-only, no edits)

- `analysis/per_run_io_char.py` consumes only `parsed.json` — confirmed compatible.
- Confirm `viz/trace.py` does not crash with empty `pi_events.jsonl`; gate it in the new script if it does.
- Confirm `bcc_tracer` fork-following reaches all workflow tasks under local/shell execution (it should, since they are descendants of the launcher PID).

## Risks

- **Process-tree escape (mitigated).** If the workflow is ever run under Pegasus + HTCondor, compute tasks are spawned by `condor_starter` daemons outside the launcher's subtree and would be missed. Avoided by the confirmed local/shell execution mode. Revisit only if HTCondor execution becomes a requirement (would need comm/cgroup-based tracer filtering).
- **Empty-LLM-log assumptions.** A few downstream modules assume `pi_events.jsonl` exists and is non-empty. Handled by (a) writing empty stubs in the launcher and (b) `-s` guards in the new trace script.

## Effort

One new launcher (~80 lines) plus three copy-and-trim scripts/configs. Roughly half a day. The main technical risk (process-tree tracing) is already designed out via local/shell mode.

## Suggested build order

1. `adapters/classic/launcher.py` + `__init__.py`.
2. `scripts/trace_script_bcc_1000genome.sh`.
3. `config/config_1000genome.env`.
4. Smoke test on a small 1000genome input; verify `ebpf_events.log`, `parsed.json`, and the `per_run_io_char` / `size_bins` figures land.
5. (Optional) `deploy_1000genome_to_client.sh` once cluster deps are settled.
