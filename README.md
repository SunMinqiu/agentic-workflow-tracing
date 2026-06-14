# Agentic Workflow Tracing & Telemetry Toolkit

A toolkit for tracing **agentic / coding-agent workflows** end to end and turning
the raw activity into timing, token, parallelism, and storage-lineage analyses.

For each run it captures three layers and correlates them:

1. **Agent layer** — every tool call the agent makes (`tool_calls.log`) and the
   raw model events (`pi_events.jsonl`).
2. **System layer** — process lifecycle and filesystem syscalls of the whole
   PID tree, captured with BCC/eBPF (`ebpf_events.log`).
3. **Analysis layer** — correlation, visualization, parallelism/DAG, and
   storage-lineage characterization built on top of the first two.

It supports **four target workflows**, each with the same underlying pipeline but
a different way of intercepting tool calls.

## Supported targets

| Target | Launcher | Tool-call logger (interception) | Trace script |
|---|---|---|---|
| **pi** (pi-coding-agent) | `analyze_codebase_pi.py` | `tool_call_logger.ts` (pi extension) | `trace_script_bcc_pi.sh` |
| **SRAgent** | `analyze_codebase_sragent.py` | `langchain_tool_logger.py` (LangChain callback) | `trace_script_bcc_sragent.sh` |
| **SciLink** | `analyze_codebase_scilink.py` | `litellm_tool_logger.py` (litellm hook) | `trace_script_bcc_scilink.sh` |
| **GenoMAS** | `analyze_codebase_genomas.py` | `genomas_tool_logger.py` (provider-SDK monkey-patch) | `trace_script_bcc_genomas.sh` |

All four loggers emit a common `tool_calls.log` format so the shared parsing,
visualization, and analysis layers work unchanged across targets.

## Pipeline

```
launcher + logger ──► tool_calls.log + pi_events.jsonl
bcc_tracer.py ───────► ebpf_events.log
                          │
       parse_ebpf.py ◄────┘   (parse_strace.py = older strace-based path)
                          ▼
                     parsed.json
                          │
        ┌─────────────────┼───────────────────────────┐
        ▼                 ▼                             ▼
 visualize_strace.py  compute_parallelism.py    lineage_analyzer.py
 (Plotly/PNG,         (DAG + parallelism,        (storage / data
  via extract_phases   via summarize_pi_events)   lineage metrics)
  + parse_strace)
        └──────────────► build_phase_index.py (per-phase index.html)
```

### Shared components

- **`bcc_tracer.py`** — BCC/eBPF tracer; records process lifecycle + filesystem
  syscalls for a target PID tree to `ebpf_events.log`.
- **`parse_ebpf.py`** — correlates eBPF events with tool calls; writes
  `parsed.json`. (`parse_strace.py` is the older strace-based equivalent and is
  still imported as a shared parsing utility.)
- **`visualize_strace.py`** — generates interactive HTML (Plotly) + static PNG
  views; uses `extract_phases.py` and `parse_strace.py`.
- **`compute_parallelism.py`** — DAG and parallelism analysis; builds on
  `summarize_pi_events.py` (higher-level timing/token summaries).
- **`lineage_analyzer.py`** — storage / data-lineage characterization (file-size
  distribution, reader fan-out, write→read staleness, etc.).
- **`build_phase_index.py`** — aggregates per-cell outputs into an `index.html`.
- **`reclassify_subagents.py`** — post-hoc subagent reclassification of tool calls.

## Design docs

- [`CONTROLLER_TRACING_PLAN.md`](CONTROLLER_TRACING_PLAN.md) — controller-level
  tracing for the SciLink hyperspectral pipeline (capturing the steps hidden
  inside `Run_analysis`).
- [`PHASE3_DAG_AND_PARALLELISM.md`](PHASE3_DAG_AND_PARALLELISM.md) — DAG
  construction and parallelism methodology behind `compute_parallelism.py`.
- [`STORAGE_PLACEMENT.md`](STORAGE_PLACEMENT.md) — storage placement standard and
  the metrics computed by `lineage_analyzer.py`.

## Setup

```bash
pip install -r requirements.txt   # matplotlib, numpy, pandas, plotly
```

External dependencies (install per target, as needed):

1. **`pi` CLI** (pi target) — e.g. `npm install -g @mariozechner/pi-coding-agent`;
   requires Node.js 20+.
2. **BCC/eBPF** — `bcc_tracer.py` needs the system BCC Python bindings, usually
   from OS packages such as `python3-bcc` / `python3-bpfcc`. Tracing requires a
   Linux host and root.
3. **Target frameworks** — SRAgent / SciLink / GenoMAS must be importable in the
   environment for their respective launchers.

## Usage (pi target example)

1. Edit **`config.env`** (paths, model, repo subset) — it is a trimmed template,
   not a copy of any private environment file.
2. Run the orchestration script:
   ```bash
   ./trace_script_bcc_pi.sh
   ```
3. Inspect per-repo outputs under the configured trace output directory.
4. Open `visualizations/index.html` in each trace result directory.

Other targets follow the same shape via their `trace_script_bcc_<target>.sh`.

### Machine-specific notes

- The pi launcher contains an original hardcoded skill path
  (`/root/.pi/skills/lustre-skill/SKILL.md`); pass `--no-skill` to skip it.
- `trace_script_bcc_pi.sh` defaults `AGENT_PYTHON` to a Lustre interpreter path;
  override interpreters with `AGENT_PYTHON` / `POST_PYTHON`.

## Notes on what is not in this repo

Trace data and generated outputs (`traces/`, `GenoMAS_traces/`, `phase6_out/`,
`summarize_pi/`) and any file containing real credentials (`cloudlab_env.sh`) are
intentionally git-ignored. Provide your own API keys via environment variables;
`config.env` is a non-secret template.
