# Overview: Characterizing I/O in Agentic Scientific Workflows

## 1. Project Motivation

Traditional scientific workflows usually have relatively fixed DAGs, known task dependencies, and stable producer-consumer dataflows. Existing workflow I/O characterization studies therefore focus on how task structure, file reuse, access type, operation count, dataflow size, and bandwidth explain workflow-level I/O behavior.

Agentic scientific workflows are different, and not always in the way that "traditional workflow + LLM wrapper" would suggest. Many real agentic scientific workflows exist precisely because the underlying task has no clean traditional-workflow counterpart — e.g., harmonizing heterogeneous raw datasets that previously required manual, judgment-heavy preprocessing. For these systems there is no fixed DAG to compare against in the first place. An LLM agent may decide which tool to call, which files to inspect, whether to retry, how to debug failures, and how to configure downstream scientific tasks. As a result, the I/O behavior is no longer determined only by a fixed workflow DAG. It is also shaped by agent decisions.

This project studies the I/O behavior of real, deployed agentic scientific workflows, with the goal of understanding what is inherited from traditional scientific workflow I/O and what is newly introduced or reshaped by agentic execution — without requiring that a traditional counterpart exist for every system studied.

## 2. Core Research Question

What I/O patterns arise when LLM agents execute scientific workflows, and how do these patterns differ from traditional fixed-DAG scientific workflows?

More specifically:

- How much I/O comes from normal scientific task execution?
- How much I/O is introduced purely by agent behavior — exploration, debugging, retry, repeated reads — the kind of overhead that a differently-behaved agent, running the exact same task and configuration, would not produce?
- How much I/O comes from a demonstrably suboptimal configuration, regardless of whether that configuration was chosen by the agent's generated code or hard-coded into the workflow's own orchestration script?
- How predictable is the I/O footprint of the same scientific goal across repeated agent runs, and how much of that unpredictability traces back to agent behavior specifically?

## 3. Scope

This project focuses on filesystem I/O caused by agent actions and scientific task execution.

Out of scope:

- LLM model loading
- KV cache paging
- model offloading
- internal LLM serving storage behavior

Those topics belong more to LLM serving and inference systems. In this project, the LLM is treated as the controller of the workflow, not as the storage workload being characterized.

## 3.1 Result Storage Standard

All durable local results for this repository should live under:

```text
results/
```

Per-run trace directories should live under:

```text
results/<run_id>/<workload>/
```

where `<run_id>` is usually a timestamp or named campaign, and `<workload>` is
the traced cell/use case. A trace cell should keep the complete bundle in one
place: `ebpf_events.log`, `parsed.json`, `pi_events.jsonl`, `tool_calls.log`,
`phase1_metrics.json`, `lineage/`, `visualizations/`, and any workflow-specific
session/output directory such as `scilink_session/` or `work/`.

`remote_results/` is not a durable result location. It may be used only as a
temporary pull/staging cache while transferring data from CloudLab or another
remote machine. After validation, anything worth keeping must be moved into
`results/`; then the staging copy should be removed.

Remote machines should expose the same logical entry point on Lustre:

```text
/mnt/lustrefs/<user>/pi-ebpf-tracing-handoff/results/
```

CloudLab clients must not write new trace output to the repo checkout, home
directory, or root filesystem. New tracing scripts should default `BASE_OUT` to
`/mnt/lustrefs/<user>/pi-ebpf-tracing-handoff/results/<run_id>`. Pulled local
copies should land under the repo-relative `results/` path with the same
`results/<run_id>/<workload>/` shape.

## 4. Target Systems and the Orchestration-Fixedness Spectrum

This project does not study "agentic workflows" as a monolithic category. Real open-source agentic scientific workflows differ substantially in how much of their execution path is fixed versus decided at runtime by the LLM. This spectrum is itself a useful axis for comparison, in place of a forced traditional-vs-agentic baseline.

Confirmed from direct inspection of the target repositories:

- **GenoMAS** (gene expression analysis, GEO/TCGA preprocessing + regression): the task sequence is fixed at the prompt level. Each agent role (GEO, TCGA, Statistician) is driven by an ordered list of "Action Units" defined in static JSON files (`prompts/action_units/base/*.json`) — e.g., the GEO agent always proceeds through Initial Data Loading → Dataset Analysis and Clinical Feature Extraction → Gene Data Extraction → Gene Identifier Review → Gene Annotation → Gene Identifier Mapping → Data Normalization and Linking, in that order, every run. What the agent decides is the *content* of the code written for each fixed stage, not which stages run or in what order. The outer orchestration loop (`environment.py`) — iterating over trait/condition pairs, cohorts, checkpointing, directory creation — is ordinary deterministic Python with no LLM involvement at all.
- **SRAgent** (NCBI/SRA metadata agent) and **ChemGraph** (XANES simulation agent): both use a LangGraph ReAct-style supervisor pattern (`create_react_agent` / `StateGraph`). The supervisor agent decides at each turn which sub-agent or tool to invoke next based on the running message history; there is no fixed stage list. SRAgent's own system prompt explicitly instructs it to "try multiple approaches if the first attempt fails" and to track what has and hasn't been tried — the number of tool calls and their order is a runtime decision, not a configuration.
- **SciLink** (microscopy/materials characterization agent): exposes an explicit `--mode autonomous` alongside components named `best_of_n_orchestrator`, `refinement_loop`, and `multiskill_autoselect` — i.e., both which analysis skill to run and how many refinement iterations to perform are runtime, agent-driven decisions.
- **CMBAgent** (github.com/CMBAgents/cmbagent; general-purpose multi-agent scientific research system, built on AG2/AutoGen, originally cosmology-focused but domain-agnostic): sits in the **middle** of the spectrum, and is a useful new data point precisely because it makes the "plan, then execute the plan" structure explicit as two separate phases rather than blending them. In `planning_and_control` mode, a `planner` agent and `plan_reviewer` agent iterate (bounded by `max_n_attempts`) until a concrete, ordered step list is agreed on — this planning phase is itself fully LLM-driven, so the *shape* of the plan (how many steps, which agent handles each) is not fixed across runs the way GenoMAS's action-unit list is. Once a plan exists, a `control`/`controller` agent executes it step-by-step, handing each step to the `engineer` (writes and runs code via AG2's `LocalCommandLineCodeExecutor`) or a domain `researcher`/specialized agent. So within one run, control-phase execution follows a plan that was itself dynamically generated — neither a static fixed-DAG (GenoMAS) nor a pure per-turn ReAct loop with no separate planning artifact (SRAgent/ChemGraph). CMBAgent also exposes a simpler `one_shot` mode (single task, no planning phase, default `agent='engineer'`) that is closer to a single-tool-call ReAct turn and is the cheapest entry point for a first integration.

GenoMAS sits at the fixed-DAG end of this spectrum; SRAgent, ChemGraph, and SciLink (in autonomous mode) sit at the dynamically-orchestrated end; CMBAgent's `planning_and_control` mode sits in between (dynamic plan generation, then comparatively fixed execution of that generated plan). The project should report where each studied system falls on this spectrum, and treat "how fixed is the orchestration" as an independent variable rather than assuming all agentic workflows are equally dynamic.

## 5. I/O Attribution Categories

Every unit of observed I/O is assigned to exactly one of three categories. The categories are defined so that assignment is based on evidence about *what specifically caused this I/O*, not on a global notion of an ideal or minimal I/O footprint — no external baseline or "necessary floor" is required to apply this scheme.

### 5.1 Agent-Induced I/O

I/O that exists purely because of the agent's behavior on this run, and that a differently-behaved agent — given the identical task and configuration — would not have produced. This is a behavioral category, not a configuration category.

Examples: repeated debug/retry cycles after a failed code execution, redundant re-reads of a file the agent already read earlier in the same task, reading error logs, inspecting intermediate outputs multiple times, abandoned artifacts from a discarded code attempt.

A large volume of agent-induced I/O is not a flaw in the categorization scheme — it is itself one of the project's central findings. It is direct evidence that either (a) current agents are not yet reliable enough to solve the task in a small, direct number of steps, or (b) the filesystem/tooling interface an agent is given does not match how agents actually search, verify, and recover from errors (e.g., no cheap way to check "have I already read this file" without re-opening it). Both are worth reporting explicitly rather than normalizing away.

### 5.2 Task-Misconfigured I/O

I/O that is more expensive than it needs to be because of a demonstrably suboptimal configuration choice — for the *same* task semantics, a better configuration exists. The detection criterion is deliberately source-agnostic: it asks only "is there a better configuration for this exact task," not "who chose it."

Once an instance is identified, it is additionally tagged along a second, source-specific axis:

- **agent-caused**: the suboptimal choice was made in code the agent generated for a given run (e.g., choosing a POSIX file-per-call read pattern where the provided library already offers a batched alternative; re-parsing a large raw file from scratch each time instead of caching the parsed result within the same code attempt).
- **script-caused**: the suboptimal choice is baked into the workflow's own fixed orchestration or tool code, and is present on every run regardless of what the agent does (e.g., GenoMAS's `environment.py` calling `os.listdir()` on every cohort-loop iteration instead of caching the directory listing once; `validate_and_save_cohort_info` in `tools/preprocess.py` performing a full JSON read-modify-write under an `fcntl` lock on every single cohort completion).

This source tag is what makes the category agent-relevant despite the source-agnostic detection rule: comparing the agent-caused rate to the script-caused rate answers "does agentic code generation introduce misconfiguration at a higher rate than a human-written script would have," which is a genuinely agentic research question even though the underlying bug-finding criterion does not care who wrote the code.

### 5.3 Workflow Task-Induced I/O (Residual)

Everything left over after 5.1 and 5.2 are subtracted out. This is I/O that is neither attributable to agent exploratory behavior nor to an identifiable suboptimal configuration — e.g., reading a required input dataset, writing the final output file. It is defined by exclusion, not by computing an absolute minimum or "necessary floor" for the task; no such floor needs to be constructed for this framework to work.

## 6. Metrics to Characterize

### 6.1 Universal I/O Metrics

Collected for all I/O, then aggregated by the three categories above.

- read bytes / write bytes
- read / write operation count
- metadata operation count
- unique files touched
- small-file access count, small-I/O count
- I/O time, effective bandwidth
- read/write ratio

### 6.2 Agent-Induced I/O Metrics

- directory scan count, failed open/stat count
- same-file reread count, same-version reread count
- error-log read count, output-inspection count
- retry-induced I/O bytes and operations
- temporary file count, abandoned artifact count
- redundant-read fraction, non-productive I/O fraction

### 6.3 Task-Misconfigured I/O Metrics

These target systems run on real HPC clusters against parallel/shared filesystems (Lustre on CloudLab/DARWIN/RCCS) — this is squarely HPC I/O territory, not an optional extra. All of these metrics are in scope for every target system by default; the per-system work is confirming which storage tier (local scratch vs Lustre) each workflow's I/O actually lands on, not deciding whether HPC-style metrics apply at all.

- I/O interface used (POSIX / batched-library-call / parallel-IO where relevant)
- output file count, average output file size
- checkpoint/metadata-write frequency
- storage location choice (scratch vs shared/parallel filesystem)
- rank-level I/O size/time imbalance (where the task launches multiple workers/ranks, e.g. GenoMAS `--parallel-mode cohorts`, ChemGraph ensemble/FDMNES runs)
- agent-caused vs script-caused split (see 5.2)

### 6.4 Run-to-Run Variance Metrics

The purpose is to quantify how predictable the I/O footprint is when the same scientific goal is executed repeatedly by an agent, and to determine how much of that variance is attributable to agent-induced I/O specifically (5.1) versus task-misconfigured or residual I/O (5.2, 5.3), which should be comparatively stable across runs of the same fixed configuration.

Metrics: total I/O bytes/operations variance, metadata operation variance, unique files touched variance, agent-induced I/O variance, task-misconfigured I/O variance, residual I/O variance, I/O time variance, runtime variance.

## 7. Research Pipeline (Three Phases)

### Phase 1 — Comprehensive metric collection

Find as complete a metric set as possible (Section 6), run it against a real target system (e.g., GenoMAS first), and collect the resulting telemetry. Start with a small-scale run (e.g., GenoMAS `--quick-test` on 1-2 traits/cohorts) to validate that every metric in Section 6 is actually extractable from the current eBPF/bcc tracing infrastructure before committing to full-scale runs — GenoMAS full runs cost 3-5 days and $300+, so the metric-extraction pipeline must be validated cheaply first. Scale up, and extend to additional target systems, only after this validation.

### Phase 2 — Provenance-based attribution

Combine the raw I/O trace with execution provenance (tool-call/action-unit time windows) to assign every abnormal/flagged I/O result to exactly one of the three categories from Section 5: **agent-induced**, **task-misconfigured**, **workflow task-induced (residual)**. The specific mechanism for correlating trace timestamps with agent turn/tool-call boundaries (e.g., using each workflow's own execution logs as ground truth, or the existing time-window matching in this repo's tracing code) is an implementation detail of the tracing pipeline, not a conceptual commitment of this document.

### Phase 3 — Mitigation case studies

Select a small number of the clearest, highest-confidence findings — prioritizing task-misconfigured (script-caused) instances, since these are deterministic code paths where a single fix applies on every future run and the before/after comparison is clean without needing to average over agent run-to-run noise — and show the fix and its I/O impact.

## 8. Key Comparisons

### 8.1 Scripted Workflow vs. Agentic Workflow (optional, where a counterpart exists)

Where a traditional scripted counterpart genuinely exists for the same scientific goal, compare it against the agentic execution to identify extra I/O introduced by agentic orchestration. This comparison is not required for systems (like GenoMAS) where no traditional counterpart exists; for those, use the high-level qualitative framing instead — fixed DAG / stable producer-consumer edges / stable I/O pattern (traditional) vs. dynamic execution path / heavy file exploration / high metadata volume / high run-to-run variance (agentic).

### 8.2 Repeated Agent Runs of the Same Goal

Run the same prompt multiple times to measure I/O predictability and determine whether variation comes from agent-induced behavior (5.1) or from the underlying task/configuration (5.2, 5.3).

### 8.3 Local Counterfactual Comparison for Misconfiguration

For a task-misconfigured instance (5.2), the comparison needed is local, not a global "recommended configuration" baseline: show that applying the identified fix to that specific instance reduces I/O for the same task semantics. This can be a before/after patch comparison (for script-caused instances) or a comparison across repeated agent runs where some runs happened to avoid the suboptimal choice and others didn't (for agent-caused instances).

## 9. Expected Findings

- A non-trivial share of total I/O in real agentic scientific workflows is agent-induced (5.1) rather than task-induced — and the size of this share is itself evidence about current agent reliability and about how well existing filesystem/tool interfaces suit agentic access patterns, not merely overhead to be subtracted out.
- Agent-induced I/O is more variable across repeated runs than task-misconfigured or residual I/O, which should be comparatively stable given a fixed configuration.
- Task-misconfigured I/O occurs in both agent-generated code and in workflows' own fixed orchestration/tooling scripts; comparing the two rates indicates whether agentic code generation is a net-additional source of configuration error beyond what already exists in hand-written scientific workflow code.
- Systems with more dynamically-orchestrated execution (SRAgent, ChemGraph, SciLink-autonomous) are expected to show higher run-to-run I/O variance than systems with a fixed action-sequence (GenoMAS), independent of task-misconfiguration.

## 10. Main Contribution

The main contribution is not a new I/O metric by itself. It is:

1. An I/O characterization of real, deployed agentic scientific workflows (GenoMAS, SRAgent, ChemGraph, SciLink, CMBAgent) spanning a range of orchestration-fixedness, without forcing a traditional-workflow baseline where none exists.
2. A causal, provenance-based three-way I/O attribution scheme (agent-induced / task-misconfigured / residual task-induced) that separates behavioral overhead from configuration error from unavoidable task I/O, using source-agnostic detection criteria plus an agent-caused/script-caused sub-tag for configuration errors.
3. A small number of concrete mitigation case studies, demonstrating measurable I/O reduction from fixing identified misconfigurations.

This connects traditional I/O characterization with the new execution behavior of LLM-based scientific agents.

## 11. Positioning Against Prior Work

Traditional HPC I/O characterization studies analyze access patterns, file reuse, sharing, read/write behavior, bandwidth, operation counts, and variability in large-scale systems.

Workflow-centric I/O characterization studies connect I/O behavior to workflow DAGs, stages, and producer-consumer relationships.

This project builds on both directions, but focuses on a new setting:

- the workflow execution path may be dynamically generated by an agent, to a degree that varies by system (Section 4) — some agentic scientific workflows have no fixed DAG at all
- the same scientific goal may produce different I/O footprints across runs
- the agent may introduce exploration, debugging, retry, and redundant reads that a non-agentic execution of the same task would not
- both agent-generated code and the workflow's own hand-written orchestration can independently misconfigure I/O, and comparing the two rates isolates what is specifically attributable to agentic execution

Thus, the project studies I/O behavior under agentic execution rather than under fixed workflow execution alone.
