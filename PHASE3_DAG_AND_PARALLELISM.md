# Phase 3: Sub-agent Visibility, DAG Reconstruction, Parallelism Metrics

Working document for the next instrumentation pass on `pi-ebpf-tracing-handoff/`.
Sits alongside `~/Desktop/research_plan.html` — implements what that plan's
Block 3-5 needs as substrate for Axes I, II, V.

---

## 1. Current state

### Done (Phase 1 + 2)

- SciLink deploys cleanly on CloudLab via `deploy_scilink_to_client.sh`.
- `scilink analyze` runs end-to-end against OpenAI (gpt-4o-mini), interactively
  and non-interactively (UNSAFE_EXECUTION_OK=true + stdin pre-load).
- eBPF (BCC) tracer captures FS + network syscalls at the orchestrator process.
- `litellm.success_callback` captures every LLM call's usage / start / end.
- Monkey-patch on `AnalysisOrchestratorTools.execute_tool` captures every
  orchestrator-level tool call.
- The existing `parse_ebpf.py` / `summarize_pi_events.py` / `visualize_strace.py`
  pipeline produces a 3-panel timeline that already shows the orchestrator's
  view correctly.

Most recent reference trace: 132.6s wall-clock, 17 LLM calls, 4 orchestrator
tools (Examine_data 31ms, Load_metadata 4.93s, Select_agent 579µs, Run_analysis
115.50s). The Run_analysis box contains the real scientific work but its
contents are invisible.

### Broken / missing

**Problem 1 — Sub-agent internals invisible.**
`Run_analysis` dispatches to `HyperspectralAnalysisAgent.analyze()`, which
loops over (a) LLM calls that write Python code and (b) subprocess invocations
that run that code via `ScriptExecutor.execute_script()`. The LLM calls ARE
captured (litellm callback fires unconditionally) but appear flat —
indistinguishable from orchestrator-level decisions. The subprocess executions
are NOT captured at all.

**Problem 2 — Events are flat; no DAG.**
`pi_events.jsonl` is a chronological stream with no parent-child links. Cannot
compute depth, width, critical path, self-time-vs-containment, or any of the
parallelism metrics defined in §3 below.

---

## 2. Plan: four changes to `pi-ebpf-tracing-handoff/`

### Step 1 — Hierarchy via contextvars in `litellm_tool_logger.py`

Add a context variable at module top:

```python
import contextvars
_current_parent: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "scilink_current_parent", default=None
)
```

Modify the existing `execute_tool` monkey-patch in `install_global()`:

```python
def patched(self, tool_name, **kwargs):
    run_id = uuid.uuid4().hex
    started = handler.on_tool_start(tool_name, dict(kwargs), run_id)
    token = _current_parent.set(run_id)              # NEW
    try:
        result = original(self, tool_name, **kwargs)
        handler.on_tool_end(tool_name, result, run_id, started)
        return result
    except Exception as e:
        handler.on_tool_error(tool_name, e, run_id, started)
        raise
    finally:
        _current_parent.reset(token)                 # NEW
```

Modify `on_llm_success` / `on_llm_failure` / `on_tool_start` to read context
and attach `parent_run_id` to every event:

```python
def on_llm_success(self, kwargs, completion_response, start_time, end_time):
    parent = _current_parent.get()                   # NEW
    run_id = uuid.uuid4().hex
    self._append_event({
        "type": "message_start",
        "run_id": run_id,
        **({"parent_run_id": parent} if parent else {}),  # NEW
        "message": {"role": "assistant",
                    "timestamp": _to_epoch_ms(start_time)},
    })
    # message_end gets parent_run_id too.
```

**Result**: every LLM event emitted from inside `execute_tool`'s body inherits
that tool's `run_id` as its parent. Backwards-compatible: events without
`parent_run_id` field are treated as top-level (parent = None).

### Step 2 — Second monkey-patch: `ScriptExecutor.execute_script`

Confirmed location: `scilink/executors.py:273`, signature
`execute_script(self, script_content: str, working_dir: str = None) -> dict`.
Returns `{"status": "success"|"error", "stdout": ..., "stderr": ...}`.

In `install_global()`, after the AnalysisOrchestratorTools patch:

```python
try:
    from scilink.executors import ScriptExecutor   # type: ignore
except ImportError:
    print("[litellm_tool_logger] WARNING: ScriptExecutor not importable — "
          "sub-agent code-exec events will be missing.", flush=True)
    return

if not getattr(ScriptExecutor.execute_script, "_pi_patched", False):
    orig_exec = ScriptExecutor.execute_script

    def patched_exec(self, script_content, working_dir=None):
        run_id = uuid.uuid4().hex
        parent = _current_parent.get()
        # Log script_len, NOT script_content (can be 100s of lines).
        args = {
            "script_len": len(script_content),
            "working_dir": working_dir,
        }
        started_at = handler.on_tool_start("ScriptExec", args, run_id)
        token = _current_parent.set(run_id)
        try:
            result = orig_exec(self, script_content, working_dir)
            handler.on_tool_end("ScriptExec", result, run_id, started_at)
            return result
        except Exception as e:
            handler.on_tool_error("ScriptExec", e, run_id, started_at)
            raise
        finally:
            _current_parent.reset(token)

    patched_exec._pi_patched = True
    ScriptExecutor.execute_script = patched_exec
```

ScriptExec events appear in `tool_calls.log` alongside orchestrator tools, but
are identifiable by the literal name "ScriptExec" + a non-None `parent_run_id`
(typically pointing to Run_analysis).

### Step 3 — New file: `compute_parallelism.py`

**Inputs**: a trace dir (containing `pi_events.jsonl` and `tool_calls.log`).
**Outputs** in the same dir:
- `parallelism_summary.json` — all numeric metrics
- `call_tree.txt` — human-readable hierarchy with self-time annotations

**Core data structures**:

```python
events_by_id: dict[str, dict] = {}                  # run_id -> full event
children_of: dict[str, list[str]] = defaultdict(list)
for ev in load_paired_events(jsonl_path):           # pairs message_start/end
    rid = ev["run_id"]
    events_by_id[rid] = ev
    pid = ev.get("parent_run_id")
    if pid:
        children_of[pid].append(rid)
```

**Self-time**: each event's duration minus the sum of its children's
durations. Critical because without this, Run_analysis (115s) would
double-count the inner LLM + ScriptExec time.

```python
def duration_ms(rid):
    ev = events_by_id[rid]
    return ev["end_ms"] - ev["start_ms"]

def self_time_ms(rid):
    own = duration_ms(rid)
    kids = sum(duration_ms(c) for c in children_of[rid])
    return max(own - kids, 0)   # clamp; see Risks
```

**Interval helpers** for the parallelism metrics (sweep-line):

```python
def union_length(intervals: list[tuple[int, int]]) -> int:
    """Total measure of merged intervals."""
    ...

def intersection_active(A, B):
    """Time when at least one interval from A AND from B is active."""
    ...

def k_active(intervals, k):
    """Time when at least k intervals are simultaneously active."""
    ...
```

**Metrics** (see §3 for definitions). Output schema:

```json
{
  "wall_clock_s": 132.6,
  "total_self_time_s": 131.8,
  "workload_concurrency_factor": 0.994,

  "parallel_time_ratio": {
    "llm_x_tool":  { "overlap_s": 0.0, "union_s": 131.8, "ratio": 0.0 },
    "tool_x_tool": { "overlap_s": 0.0, "union_s": 120.5, "ratio": 0.0 },
    "llm_x_llm":   { "overlap_s": 0.0, "union_s": 105.9, "ratio": 0.0 }
  },

  "parallel_degree": {
    "semantic_events": {
      "avg_active_over_wall": 0.994,
      "avg_active_when_busy": 1.0,
      "max_active": 1,
      "time_at_degree_ge_2_s": 0.0,
      "parallel_time_ratio": 0.0
    },
    "observed_processes": {
      "avg_active_over_wall": 0.96,
      "avg_active_when_busy": 1.0,
      "max_active": 1,
      "unit": "pid"
    }
  },

  "structural": {
    "depth": 3,
    "width_max": 1,
    "top_level_tools": 4,
    "subagent_internal_events": 22
  }
}
```

`call_tree.txt` example:

```
Session (132.6s total, 131.8s self+children, 0.8s unaccounted)
├─ LLM #1  ( 2.71s)                          [orchestrator: decide examine_data]
├─ Tool: Examine_data  (31ms self)
├─ LLM #2  ( 1.43s)
├─ Tool: Load_metadata (   30ms self, 4.93s total)
│  └─ LLM #3  ( 4.90s)                       [metadata normalization]
├─ Tool: Select_agent  (579µs self)
├─ LLM #4  ( 1.21s)
├─ Tool: Run_analysis  ( 8.50s self, 115.50s total)
│  ├─ LLM #5  ( 8.12s)                       [write NMF code]
│  ├─ ScriptExec #1  ( 4.13s)                [run NMF]
│  ├─ LLM #6  ( 7.55s)                       [interpret + plan next step]
│  ├─ ScriptExec #2  ( 2.81s)
│  ├─ ... (× N)
└─ LLM #17 ( 1.85s)                          [orchestrator wraps up]
```

### Step 4 — Update `visualize_strace.py` for nested rows

Current chart: 3 panels (Semantic, Tool, System), each with row(s).

**Semantic panel** — split into 2 rows:
- `LLM (top-level, Nx)` — events with `parent_run_id == None`
- `LLM (subagent,  Mx)` — events with `parent_run_id != None`

This makes "orchestrator decided X" vs "sub-agent wrote code" eyeball-distinguishable.

**Tool panel** — replace flat list with parent-then-children layout. For each
top-level tool (`parent_run_id == None`), in start-time order:

1. Draw the tool's bar on a row labeled `<ToolName> (Kx, avg Xs)`.
2. If the tool has children, append additional indented rows:
   - `  ↳ subagent LLM (M events)` — drawn at child intervals
   - `  ↳ ScriptExec (N events)` — drawn at child intervals
   - One row per child type that's present.

Implementation sketch:

```python
top_level_tools = [ev for ev in tool_events if ev.get("parent_run_id") is None]
rows = []
for tool in sorted(top_level_tools, key=lambda e: e["start_ms"]):
    rows.append(("tool", tool["name"], [tool]))
    kids = children_of[tool["run_id"]]
    kids_by_type = group_by(kids, key=lambda c: events_by_id[c]["type_label"])
    for type_label, kid_list in kids_by_type.items():
        rows.append(("subchild", f"  ↳ {type_label}", kid_list))
```

**System panel** — unchanged.

**Color scheme**:
- Top-level LLM: existing green
- Sub-agent LLM: lighter green (e.g. `#a8e6a3`)
- Top-level tool: existing orange family
- ScriptExec: purple (e.g. `#9b59b6`) to distinguish from orchestrator tools
- System syscalls: unchanged

---

## 3. Parallelism metrics — formal definitions

Notation. Each event has interval `[start, end]`. Group events by type:
- **L** = set of LLM call intervals
- **T** = set of Tool call intervals (Examine_data, Load_metadata, ...)
- (could further split into L_top / L_sub if needed)

Helper functions (all measured in seconds):
- `union(S)` — total measure of ⋃ s∈S after merging overlaps
- `both_active(A, B)` — measure of { t : ∃a∈A with t∈a AND ∃b∈B with t∈b }
- `k_active(S, k)` — measure of { t : at least k intervals in S contain t }

| Metric | Numerator | Denominator | Range |
|---|---|---|---|
| **Workload concurrency factor** | Σ self_time over all events | wall_clock_total | ≥ 0; = 1 if perfectly busy serial, > 1 if true concurrency |
| **LLM ↔ Tool** | both_active(L, T) | union(L ∪ T) | [0, 1] |
| **Tool ↔ Tool** | k_active(T, 2) | union(T) | [0, 1] |
| **LLM ↔ LLM** | k_active(L, 2) | union(L) | [0, 1] |

Interpretation of pairwise time-ratio metrics: "of all the time during which we were in
class X or Y, what fraction was both classes simultaneously active?" Idle
gaps don't dilute (they aren't in either L or T, so excluded from both
numerator and denominator).

Parallel degree metrics are separate: they measure how many execution units are
active at the same time (`avg_active_*`, `max_active`). For semantic events the
unit is an LLM/tool self-time interval; for observed processes the unit is a PID
interval inferred from `parsed.json` syscall observations.

**Concrete worked example**:

```
time(s) ── 0 ── 4 ── 6 ─── 12 ─ 14 ── 18 ─ 19 ── 21
LLM        [════]       [══════════]       [══════]
Tool             [════════════]                    [════]
```

- L = `{[0,4], [8,12], [14,18]}`, union(L) = 4 + 4 + 4 = 12s
- T = `{[6,12], [19,21]}`, union(T) = 6 + 2 = 8s
- L ∪ T = `[0,4] ∪ [6,12] ∪ [14,18] ∪ [19,21]`, union = 4 + 6 + 4 + 2 = 16s
- both_active(L, T) = `[8,12]` = 4s

LLM ↔ Tool = 4 / 16 = **0.25** — during the time "LLM or Tool was active",
both ran simultaneously 25% of the time.

**Expected SciLink single-agent values** (after Phase 3 lands):
- Workload concurrency factor ≈ 0.99 (mostly busy, almost no idle gaps)
- LLM ↔ Tool, Tool ↔ Tool, LLM ↔ LLM all = 0.0

→ Confirming SciLink is **strictly serial** at single-agent granularity is
itself a research finding (Axis I baseline). Pairwise > 0 only appears with
(a) framework-driven speculative branching (ReAct/LATS), or (b) multi-agent
concurrency (Axis IV).

---

## 4. Expected outcomes

Re-running the existing `eels_plasmons_basic` workload after Phase 3:

| Artefact | Before (Phase 2) | After (Phase 3) |
|---|---|---|
| `pi_events.jsonl` event count | ~30 | ~50 (adds ScriptExec events; LLM events gain `parent_run_id`) |
| Distinct tool names in `tool_calls.log` | ~5 | ~5 + `ScriptExec` |
| Timeline rows in visualization | 3 panels × ~6 rows | 3 panels × ~10 rows (indented children) |
| Run_analysis displayed as | Single 115s bar | Container; inner LLM + ScriptExec bars visible below |
| `compute_parallelism` output | n/a | `parallelism_summary.json` + `call_tree.txt` |

Numerical expectations:
- Depth ≈ 3 (root → orchestrator tool → sub-agent step)
- Width_max ≈ 1
- All pairwise parallel time ratios ≈ 0
- Workload concurrency factor ≈ 0.99
- Parallel degree max ≈ 1

---

## 5. Risks and mitigations

**(1) contextvar doesn't propagate across SciLink threads.** SciLink's
`executors.py` keeps `_active_subprocesses: dict[int, set[Popen]]` indexed by
thread id, hinting that some work may happen off the main thread. Python
contextvars are auto-copied into `threading.Thread` constructors (3.7+), so
single-thread-fork-from-main works. But `ThreadPoolExecutor` recycles workers
and may carry over stale context.
*Mitigation*: implement v1 naively; verify by inspecting `pi_events.jsonl`
after a run that `parent_run_id` on inner LLM/ScriptExec events points to the
actual Run_analysis run_id. If `None`, fall back to walking the Python call
stack (`inspect.stack()`) to find the nearest patched frame.

**(2) ScriptExecutor signature drift.** Current signature is at HEAD
(commit 429d48d). Future SciLink refactors could break the patch silently.
*Mitigation*: the patch already checks `_pi_patched` before re-applying;
add a runtime check that prints `WARNING` if the class import fails (mirror
the existing AnalysisOrchestratorTools-not-importable warning).

**(3) Visualization unreadable with many children.** If a future demo has
many parent tools each with many children, the row count explodes.
*Mitigation*: add `max_inline_children` parameter; collapse to a summary row
`↳ subagent LLM (M events, total T s)` when exceeded.

**(4) Negative self-time.** If clock sources for parent (datetime.now via our
patch) and children (litellm-provided datetimes) drift, sum of children might
exceed parent.
*Mitigation*: clamp `self_time` to ≥ 0, emit a one-line WARNING to the run's
log when clamping engages. Long-term: unify on a single clock source.

---

## 6. Tie-in to research_plan.html

| Research axis | What Phase 3 delivers |
|---|---|
| Axis I — Execution-structure anatomy | Depth/width/fan-out from call tree; per-framework structural signature (once ≥2 frameworks instrumented) |
| Axis II — Provenance of parallelism | Baseline "all zero" for SciLink single-agent; useful as comparison point against multi-agent and against alternative frameworks |
| Axis III — Data access and locality | Existing eBPF syscall trace already provides this; no change |
| Axis IV — Multi-agent interference | Not addressed yet; needs N concurrent SciLink instances + shared LLM backend instrumentation |
| Axis V — Cross-run variance | Enabled — `parallelism_summary.json` is diffable across K reruns of the same workload |

Phase 3 is squarely Axis I + setup for II + V. Multi-agent (Axis IV) is a
separate effort that requires SLURM / concurrent-launcher infrastructure.

---

## 7. Execution order

1. Step 1 (contextvar + execute_tool update) — ~25 min
2. rsync + run on CloudLab — verify `parent_run_id` shows up on inner LLM
   events in `pi_events.jsonl`
3. Step 2 (ScriptExecutor patch) — ~20 min
4. rsync + run — verify ScriptExec entries appear in `tool_calls.log` AND
   carry `parent_run_id = <Run_analysis_id>`
5. Step 3 (`compute_parallelism.py`) — ~1 h
6. Run on the existing trace dir locally; eyeball `call_tree.txt`
7. Step 4 (visualize updates) — ~40 min
8. Re-run end-to-end; compare new timeline vs. the current 132.6s reference
9. Archive the new trace as `phase3_reference_trace_<date>/` for cross-comparison

Total ~2.5 h of focused work, plus run+inspect cycles between steps.
