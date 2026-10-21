# Agentic Workflow Tracing

This project provides tools to collect and analyze file system I/O, LLM calls, and vLLM KV cache activity in agentic workflows. It connects workflow events with process and file activity, captures model requests for replay, and records cache events inside the serving engine.

## Part I: Workflow I/O and LLM Calls

### Purpose and Scope

The workflow tracing pipeline records what files a workflow accesses, when those accesses occur, and which tools or execution stages are active. LLM call records place model requests and responses on the same timeline. Traditional scientific workflows use the same I/O analysis with task and stage identifiers.

File system tracing runs on a Linux workflow node through BCC/eBPF. It follows the launched process and its descendants. Adapters provide the workflow events needed to associate I/O with tool calls, agent roles, and execution units.

### Implemented Components

- Workflow launchers and logging adapters capture tool execution, model calls, generated code, and workflow stages.
- A BCC tracer records file operations, process activity, syscall durations, return values, and file offsets. Optional library probes record STDIO, HDF5, and MPI-IO calls when the corresponding symbols can be attached.
- Parsers resolve file descriptors and paths, align timestamps, and associate file events with workflow execution intervals.
- Lineage analysis connects files with the execution units that read and write them. Aggregation produces metrics by file, tool call, task, phase, and role.
- Lustre counter collection records client and storage activity alongside workflow traces.
- Postprocessing generates structured metrics, workflow timelines, file lineage views, and per-task HTML pages. Collection checks record dropped events, path and offset coverage, attribution coverage, and LLM call failures.

### Workflow Integrations

| Integration | Included support |
| --- | --- |
| Pi | Agent launcher, tool event logging, and file system tracing |
| GenoMAS | Workflow launcher, tool and LLM logging, cohort and role metadata, and workload scaling configurations |
| SciLink | Workflow launcher, tool and LLM logging, streaming events, and repeated-request guards |
| SRAgent | Workflow launcher, tool logging, and tracing scripts |
| ChemGraph | Workflow launcher, tool logging, tracing scripts, and an XANES MCP bridge |
| 1000 Genomes | Direct task DAG execution, bounded worker concurrency, task metadata, and shared I/O analysis |
| Montage | Input preparation, direct stage execution, stage metadata, and shared I/O analysis |
| Selected SWE-bench and SWE-bench Pro tasks | Claude Code execution, vLLM request capture through a proxy, measured-run scripts, and separate evaluation checkouts |

The fields available for analysis depend on the integration's event records. GenoMAS and SciLink provide the structured message captures used by the workflow prefix analysis and replay pipeline.

### Measurements

| Area | Measurements |
| --- | --- |
| I/O volume and operations | Read and write bytes, operation counts, metadata operations, unique files, and rates normalized by transferred bytes |
| Access sizes and patterns | File and request size distributions, sequential and random transitions from recorded offsets, inter-arrival times, same-file reopens, and repeated reads |
| Namespace activity | Directory scans and rescans, failed path operations, error categories, and state-file rewrites |
| File relationships | Readers and writers, producer-consumer relationships, reader and writer fan-out, file lifetimes, and write-to-read gaps |
| Timing and concurrency | I/O duration, active and wall-time bandwidth, duty cycle, bursts, intensity phases, autocorrelation, and overlapping I/O workers |
| Workflow context | Bytes and operations by tool, task, phase, and role; I/O during recorded backtrack intervals; execution-unit timing and I/O distribution |
| LLM calls | Request and response content, generation parameters, model identifiers, input and output tokens, reported cached tokens, call duration, and failures |
| Streaming and overlap | Time to first token, response timing, overlapping model calls, and I/O during and between LLM call intervals |

Offset-based measurements use events with recorded offsets. Streaming timing uses recorded token events. Client TTFT includes transport and server waiting time; server timing fields retain their own measurement boundaries. File and execution-unit associations provide evidence for examining retries and repeated work; identifying an avoidable operation requires examining the task and its execution context.

### Generated Data and Analysis

These files are generated during collection and postprocessing. Experiment traces and generated pages are excluded from Git tracking.

| Output | Contents |
| --- | --- |
| `manifest.json` | Run configuration and workload metadata |
| `ebpf_events.log`, `bcc.err` | Raw file and process events, tracer status, and dropped-event counts |
| `pi_events.jsonl`, `tool_calls.log`, `messages.jsonl` | Workflow events, tool execution, and captured model calls where supplied by the integration |
| `execution_units.jsonl` | Task and stage records from integrations that emit execution units |
| `parsed.json` | Parsed I/O events and their associated workflow context |
| `phase1_metrics.json`, `parallelism_summary.json`, `trace_quality.json` | I/O characterization, concurrency statistics, and collection coverage |
| `lineage/artifacts.csv`, `lineage/execution_unit_io.csv` | Per-file and per-execution-unit I/O records |
| `visualizations/index.html` | Per-task I/O figures, timelines, and linked execution details |

## Part II: vLLM KV Cache Behavior

### Purpose and Scope

The cache pipeline examines how workflow requests share prompt prefixes and how vLLM stores, reuses, evicts, and transfers cached content. It combines captured requests, backend usage fields, Prometheus metrics, logical block events, and instrumented engine events.

Logical analysis describes shared prefixes in the captured inputs. Physical tracing records cache slots, allocation generations, transfers, and request execution inside vLLM. Each analysis retains the source and collection status of its measurements.

### Implemented Components

- LLM capture preserves messages, tool schemas, request parameters, responses, usage, timing, and request identifiers.
- Prefix analysis groups related calls, follows context growth, identifies shared prompt segments, and computes reuse quantities using the available tokenizer and cache geometry.
- Bundle construction uses the serving endpoint to render and tokenize captured requests, preserving token IDs and request arrival offsets for replay.
- The replay runner controls request timing, concurrency, output length, warmup, cache salts, and prefix-cache resets. It saves the serving configuration and metrics for each run.
- A KV event collector records published block events, publisher epochs, sequence numbers, and collection boundaries. Lifecycle analysis follows stored content, removals, and later returns.
- vLLM hooks record physical allocation and eviction, CPU cache state, transfer plans, worker completion, scheduler completion, observed computation, and request waiting.
- Analysis joins client requests with engine records through explicit request identifiers and joins cached content with physical slots through allocation identities.
- Patch tooling checks source fingerprints before installing or restoring hooks. Source fingerprint configurations are included for the audited vLLM 0.26 and 0.29 deployments.

### Experiment Modes and Collection Requirements

| Mode | Included controls and collection |
| --- | --- |
| Live workflow capture | Collect requests and usage while the agent executes its tools and tasks |
| Fixed-input replay | Send recorded prompt token IDs with a chosen output length and concurrency limit |
| Request timing | `packed` removes workflow gaps under a concurrency limit; `paced` preserves recorded arrival offsets; `serial` sends one request at a time |
| Cache reuse controls | Shared or per-request cache salts, cold-cache resets, warmup repetitions, and scheduled global cache resets |
| Serving configuration experiments | Record runs under selected cache capacities, prefix-caching settings, KV dtypes, and CPU offload settings; validate the serving configuration across arms |
| Controlled sequences | Repeated prompts, reset sequences, and cache-pressure sequences for observing reuse, eviction, and restoration |
| Context construction | GenoMAS can insert cohort-matched paper segments with configurable placement and token budgets, tracking consumption separately for each cohort and role |

Token replay requires a vLLM endpoint that accepts prompt token IDs. Cache resets require the server's reset interface. Logical block collection requires a configured KV event publisher. Physical slot and transfer measurements require hooks installed against the matching source fingerprints. Replay executes model requests; tool execution and workflow file I/O are collected during live workflow runs.

### Measurements

| Area | Measurements |
| --- | --- |
| Prompt demand | Input and output token distributions, context growth, and demand by role and phase |
| Prefix structure | Shared prefix families, block-aligned reuse opportunities, reuse distance, reuse age, and working-set quantities |
| Observed reuse | Per-request cached tokens and cache-hit counters within recorded metric windows |
| Logical cache lifecycle | Store, remove, and return events; residency intervals; CPU-copy candidates; publisher gaps and collection boundaries |
| Physical cache lifecycle | GPU slot allocation, access, release, hash assignment, invalidation, capacity eviction, and CPU cache eviction |
| Transfers | GPU-to-CPU and CPU-to-GPU plans, content identities, expected workers, completion status, reported bytes, and worker transfer times |
| Restoration and computation | Confirmed reloads followed by observed use, completed prompt computation, and candidates classified as restored, recomputed, or unresolved |
| Request waiting and latency | Scheduler-observed load waits, client TTFT, inter-token timing, request latency, output throughput, and server prefill and decode metrics where exposed |
| Run controls | Serving configuration, cache geometry, request counts, output lengths, cache initialization, warmup, and timing mode |

Logical hash removal and physical capacity eviction have separate counts. Transfer submission, worker completion, and later use are separate recorded steps. Worker transfer time and scheduler-observed request waiting are reported separately. Missing events or ambiguous cache-group identities retain an explicit collection or resolution status.

### Generated Data and Analysis

Cache analyses and replay runs generate the following files. Captured bundles, raw cache traces, and experiment results are excluded from Git tracking. The tracked paper texts and their manifest are inputs for context-construction experiments.

| Output | Contents |
| --- | --- |
| Replay bundle JSON | Prompt token IDs, request parameters, arrival offsets, and source-workflow metadata |
| `requests.jsonl`, `summary.json`, `run_manifest.json` | Per-request replay records, aggregate measurements, and replay controls |
| `server_info.json`, `metrics_before.prom`, `metrics_after.prom`, `metrics_window.json` | Serving configuration and metrics over the measurement window |
| `kvcache_demand.json`, `kvcache_logical.json`, `kvcache_segments.json` | Token demand, prefix relationships, reuse quantities, and segment analysis |
| `kv_events.jsonl`, `kv_removals.jsonl` | Published cache events and derived logical removals |
| `physical_traces/`, `physical_evictions.jsonl`, `cpu_evictions.jsonl` | Imported engine traces and derived physical eviction records |
| `causal_transfers.jsonl`, `confirmed_reloads.jsonl`, `candidate_reconciliation.jsonl`, `request_load_waits.jsonl` | Transfer completion, reload evidence, candidate outcomes, and load waits |
| `kv_report_data.json`, `kvcache_report.html` | Shared analysis data and the task's cache analysis page, including associated replay experiments |

`results/index.html` links workflow I/O and KV cache pages. `results/all-experiments.html` provides an experiment index. These pages are generated from local experiment data.

## Repository Structure

| Path | Contents |
| --- | --- |
| `src/agent_io_tracing/adapters/` | Workflow launchers and tool and LLM logging |
| `src/agent_io_tracing/tracing/` | BCC/eBPF tracing and Lustre counters |
| `src/agent_io_tracing/parsing/` | Event parsing, timestamp conversion, and phase association |
| `src/agent_io_tracing/analysis/` | Workflow I/O metrics, collection checks, result indexes, and logical KV cache analysis |
| `src/agent_io_tracing/lineage/` | File lineage and execution-unit I/O attribution |
| `src/agent_io_tracing/replay/` | Request bundles, replay execution, and configuration comparisons |
| `src/agent_io_tracing/serving/` | vLLM endpoint access, cache collection, engine hooks, and physical cache analysis |
| `src/agent_io_tracing/viz/` | Workflow figures, timelines, and workload-scaling views |
| `src/agent_io_tracing/experiments/` | Run metadata and input staging helpers |
| `scripts/` | Deployment, workflow execution, tracing, result retrieval, and SWE task tooling |
| `tools/` | Experiment drivers, vLLM patch preparation, cache analysis, and page rebuilding |
| `config/` | Workflow settings, serving settings, source fingerprints, and comparison definitions |
| `workloads/` | Selected SWE task prompts, metadata, and evaluation patches |
| `output/genomas_references/` | Tracked paper texts and cohort manifest used as experiment inputs |
| `docs/manuals/` | [Agentic workflow manual](docs/manuals/agentic_workflow_manual.md) and [traditional workflow manual](docs/manuals/traditional_workflow_manual.md) |
