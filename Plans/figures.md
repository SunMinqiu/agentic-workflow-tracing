# Figure Reference

Descriptions for every figure in the GenoMAS I/O-characterization pipeline.
The index pages show figure **names only**; this file is the single source for
what each one means. Figures are grouped by the question they answer.

## Storage / I/O shape (per cell)

| Figure | Description |
|---|---|
| File Size Distribution | Per-syscall I/O request size and per-file artifact size, bucketed by category. |
| I/O Volume Summary | Total read and write bytes, broken down by file category (dataset / intermediate / output / memory / log…). |
| Reuse Pattern | Share of written bytes that are never read back (dead writes), and the read-reuse factor (re-reads a cache could save). |
| Reader Fan-out | Number of distinct CodeExec calls that read each artifact (caching candidates). |
| Write→First-Read Staleness | Gap from an artifact's first write to its first read, per artifact. |
| Artifact Lifecycle | Per-artifact reclaimable window over the run (create → last read → end). |
| Who Does the I/O | Read and write bytes attributed to each agent role, split by file category. |

## Time & attribution (per cell)

| Figure | Description |
|---|---|
| Agent Timeline | Three-lane Gantt — LLM and subagents, real tools, and filesystem syscalls by category — plus a side bar of total seconds per syscall category. |
| Time Accounting | Busy time (sum of activity spans) split into LLM, tool, and residual; subtitle reports speedup = busy / wall. |
| I/O Rate Over Time | Syscalls per 100 ms with tool-call windows overlaid. |
| Call DAG with I/O | Parent-child execution graph with each node annotated by its read bytes, write bytes, and metadata ops (logical node → physical I/O attribution). |

## Concurrency (per cell)

| Figure | Description |
|---|---|
| Agent Activity Timeline | One lane per agent, colored by resource (LLM / File-IO / Tool); vertical overlap between lanes shows agents running in parallel. |

## Detail (per cell, link-only)

| Figure | Description |
|---|---|
| Timeline View | Tool calls and filesystem operations over time. |
| Syscalls Per Tool Call | Per-tool breakdown of syscalls as horizontal duration bars. |
| Syscall Duration Distributions | Per-tool violin plots of each syscall type's duration. |

## Cross-cell scaling (per run)

| Figure | Description |
|---|---|
| Axis 1 — Cohort Scaling | Generated files, input read bytes, storage metadata ops, and metadata/data ratio as a function of cohort count C (one trait). |
| Axis 2 — Trait Fanout | The same metrics as a function of the number of single-cohort traits T. |
