# Controller-level tracing for SciLink hyperspectral agent

## Why this exists

Today the PI tracer captures only 4 events for an entire hyperspectral run:
the orchestrator's outer tool dispatch (`examine_data`, `load_metadata`,
`select_agent`, `Run_analysis`). The 13 internal pipeline steps and the
in-process `exec()` of LLM-generated code that live inside
`Run_analysis` are invisible to `tool_calls.log`, `call_dag.*`, and the
Gantt chart. See trace
`traces/20260520_103610/eels_plasmons_basic/` — `Run_analysis` shows up as
one 139-second box; everything inside it is lost.

Two reasons, both in the SciLink layout, not the tracer:

1. The monkey-patch in `litellm_tool_logger.install_global()` only
   intercepts `AnalysisOrchestratorTools.execute_tool`, which is the
   outer router. Pipeline controllers are plain Python objects called
   directly by `HyperspectralAnalysisAgent` —
   `controller.execute(state)` at
   `scilink/agents/exp_agents/hyperspectral_analysis_agent.py:694`.
   They never go through `execute_tool`.

2. `RunDynamicAnalysisController` runs LLM-generated code via a literal
   `exec(code_str, ...)` call at
   `scilink/agents/exp_agents/controllers/hyperspectral_controllers.py:1627`,
   bypassing `ScriptExecutor` (the only other thing the tracer patches).
   This is documented as deliberate — 100 MB hyperspectral cubes are
   expensive to serialize for a subprocess — so the design stays, the
   tracer adapts.

## What changed from the first draft of this plan

The first draft introduced a `BaseController` ABC and renamed
`execute()` → `_run()` on all 19 controllers as a template-method
seam. That added ceremony with no business value — controllers don't
need a shared base class for anything except "the tracer wants a hook
point." Scrapped.

New approach: **convention-based discovery in the tracer, near-zero
changes in SciLink.**

The only thing SciLink actually has to change is the one place where
we cannot reach via monkey-patching — the bare `exec()` call inside
`RunDynamicAnalysisController`. Everything else stays untouched.

## Scope

- In: hyperspectral controllers (16) + base controllers used by
  hyperspectral (3) + the in-process `exec()` inside dynamic analysis.
- Out: other agents (curve_fitting / image / sam / fft / atomistic).
  Their controllers already use `ScriptExecutor` subprocess execution
  and so are partially visible to the existing patch. Folding them in
  is a follow-up and requires per-agent demos for verification.
- Out: deeper granularity (e.g. per-NMF-iteration inside
  `RunComponentTestLoopController`'s 7-fit sweep). One controller = one
  event in this round. Sub-step granularity is a separate change.

## Design

### Existing infrastructure already does the right thing

These three pieces are in place and need no change:

- `_current_parent` ContextVar at `litellm_tool_logger.py:44-46`
  propagates `parent_run_id` automatically. When the outer `Run_analysis`
  patch sets the contextvar, any nested controller event we record
  inside it inherits the parent link for free.
- `compute_parallelism.render_call_dag_dot()` at
  `compute_parallelism.py:766` already walks `build_children()` and
  emits `contains` edges. Tree shape comes through.
- `visualize_strace.py:1906` `_load_parent_run_ids()` already reads
  `parent_run_id` per event for LLM attribution.

So adding child events with proper `parent_run_id` is enough — the
visualization layer will pick up the hierarchy.

### Convention-based controller discovery

In `litellm_tool_logger.install_global()` we add a third patch step
that:

1. Imports each module that contains pipeline controllers (explicit
   list — predictable, easy to extend).
2. Scans the module with `inspect.getmembers(mod, inspect.isclass)`
   for classes that:
   - are defined in that module (not re-imported), and
   - have a name ending in `Controller`, and
   - have a callable `execute` attribute with signature
     `(self, state)`.
3. Monkey-patches each such class's `execute` method with the same
   `on_tool_start / on_tool_end` wrapper already used for
   `AnalysisOrchestratorTools.execute_tool` and `ScriptExecutor`.

The wrapper:

- Derives a display name from the class name by stripping the
  `Controller` suffix (`RunPreprocessingController` →
  `RunPreprocessing`).
- Sanitizes `state` for logging by picking a small whitelist of keys
  (`iteration_title`, `depth`, `task_idx`, `task_total`,
  `n_components`, `method`, `axis_units`). Never serializes
  `hspy_data` or any image array.
- Sets the contextvar before delegating, so any nested tool calls
  inside the controller correctly attach to it as their parent.

### The one seam we cannot fake from outside

`RunDynamicAnalysisController._run` (the body of `execute`) contains
this line:

```python
exec(code_str, global_scope, local_scope)
```

A free `exec()` call is not a named callable; we cannot
monkey-patch it from the tracer. The minimum SciLink change is to
extract that single line into a method:

```python
def _run_generated_code(self, code_str, global_scope, local_scope):
    """Seam for the PI tracer to wrap as a ScriptExec event.

    Kept as a one-line method so external instrumentation can record
    code-execution as a sibling of the subprocess ScriptExec events
    emitted elsewhere. Do not inline.
    """
    exec(code_str, global_scope, local_scope)
```

…and change the call site to `self._run_generated_code(...)`.

The tracer then patches `_run_generated_code` on the class and emits a
`ScriptExec` event around it, matching the existing subprocess-based
`ScriptExec` event name. `compute_parallelism.py` already colors
`ScriptExec` purple, so the in-process variant gets the same visual
treatment for free.

### What the resulting event tree looks like

For one hyperspectral run with one refinement iteration:

```
Examine_data                 (21 ms)
Load_metadata             (1.1 s)
Select_agent                 (0.5 ms)
Run_analysis              (139 s)
├─ RunPreprocessing            (≈4 s)
├─ GetInitialComponentParams   (≈3 s, mostly LLM call)
├─ RunComponentTestLoop        (≈40 s, 7 NMF fits collapsed)
├─ CreateElbowPlot             (≈1 s)
├─ GetFinalComponentSelection  (≈4 s, mostly LLM call)
├─ RunFinalSpectralUnmixing    (≈7 s)
├─ CreateAnalysisPlots         (≈3 s)
├─ BuildHyperspectralPrompt    (<1 s)
├─ RunFinalInterpretation      (≈13 s, LLM call)
├─ SelectRefinementTarget      (≈6 s, LLM call)
├─ IterativeFeedback           (≈0 s, disabled)
├─ RunDynamicAnalysis          (≈30 s)
│  └─ ScriptExec (in-process)  (≈4 s)
├─ GenerateRefinementTasks     (<1 s)
├─ BuildHolisticSynthesisPrompt(<1 s)
├─ RunFinalInterpretation      (≈13 s, LLM call)
├─ RunSelfReflection           (≈6 s, LLM call)
├─ ApplyReflectionUpdates      (≈11 s, LLM call)
└─ GenerateHTMLReport          (<1 s)
```

Existing LLM events (already emitted by the litellm success callback)
will hang off the appropriate controller via the contextvar — e.g. the
LLM call inside `GetInitialComponentParams` shows up as a child of
that controller, not as a child of `Run_analysis` directly.

## Implementation

### Files to change

| File | Change | Lines |
|---|---|---|
| `scilink/agents/exp_agents/controllers/hyperspectral_controllers.py` | Extract `exec()` into `_run_generated_code` method; update call site. | ~6 |
| `pi-ebpf-tracing-handoff/litellm_tool_logger.py` | Add controller-discovery + patch step inside `install_global()`. Add `ScriptExec` patch for `_run_generated_code`. | ~60 |

That is the full footprint. No `BaseController`, no class renames, no
business-logic edits.

### SciLink-side change (one file)

`scilink/agents/exp_agents/controllers/hyperspectral_controllers.py`,
inside `RunDynamicAnalysisController`:

- At the call site (currently line 1627):
  - Change `exec(code_str, global_scope, local_scope)` to
    `self._run_generated_code(code_str, global_scope, local_scope)`.
- Add the method definition somewhere in the class body:

  ```python
  def _run_generated_code(self, code_str, global_scope, local_scope):
      """Seam for external instrumentation; do not inline."""
      exec(code_str, global_scope, local_scope)
  ```

The docstring is load-bearing: it documents *why* this is a method
instead of a one-liner, so a future refactor doesn't quietly inline it
and silently break tracing.

### Tracer-side change (one file)

`pi-ebpf-tracing-handoff/litellm_tool_logger.py`, inside
`install_global(handler)`, append after the existing
`ScriptExecutor` patch (currently ends around line 567):

```python
# 4) Pipeline-controller granularity.
#
# SciLink controllers are plain classes called directly by the agent
# (no dispatch table). We discover them by convention: any class whose
# name ends in "Controller" and exposes an `execute(self, state)`
# method, in the modules we explicitly know about. Wrap `execute` so
# every pipeline step shows up as a tool event with parent_run_id
# pointing at the enclosing Run_analysis call (already set on the
# contextvar by step 2).
import inspect

CONTROLLER_MODULES = (
    "scilink.agents.exp_agents.controllers.base_controllers",
    "scilink.agents.exp_agents.controllers.hyperspectral_controllers",
)

def _state_summary(state):
    if not isinstance(state, dict):
        return {}
    keys = ("iteration_title", "depth", "task_idx", "task_total",
            "n_components", "method", "axis_units")
    return {k: state[k] for k in keys if k in state}

def _display_name(cls):
    name = cls.__name__
    return name[:-len("Controller")] if name.endswith("Controller") else name

def _patch_controller_class(cls):
    original = getattr(cls, "execute", None)
    if original is None or getattr(original, "_pi_patched", False):
        return False
    name = _display_name(cls)

    def patched(self, state, *a, **kw):
        run_id = uuid.uuid4().hex
        args = _state_summary(state)
        started_at = handler.on_tool_start(name, args, run_id)
        token = _current_parent.set(run_id)
        try:
            result = original(self, state, *a, **kw)
            handler.on_tool_end(name, result, run_id, started_at)
            return result
        except Exception as e:
            handler.on_tool_error(name, e, run_id, started_at)
            raise
        finally:
            _current_parent.reset(token)
    patched._pi_patched = True
    cls.execute = patched
    return True

patched_controllers = []
for mod_name in CONTROLLER_MODULES:
    try:
        mod = __import__(mod_name, fromlist=["*"])
    except ImportError as e:
        print(f"[litellm_tool_logger] WARNING: {mod_name} not importable: {e}",
              flush=True)
        continue
    for cls_name, cls in inspect.getmembers(mod, inspect.isclass):
        if cls.__module__ != mod_name:
            continue
        if not cls_name.endswith("Controller"):
            continue
        if _patch_controller_class(cls):
            patched_controllers.append(cls_name)

if patched_controllers:
    print("[litellm_tool_logger] patched controllers: "
          + ", ".join(patched_controllers), flush=True)

# 5) In-process script execution inside RunDynamicAnalysisController.
try:
    from scilink.agents.exp_agents.controllers.hyperspectral_controllers import (
        RunDynamicAnalysisController,
    )
except ImportError:
    RunDynamicAnalysisController = None

if RunDynamicAnalysisController is not None:
    original_exec = getattr(
        RunDynamicAnalysisController, "_run_generated_code", None
    )
    if original_exec is not None and not getattr(original_exec, "_pi_patched", False):
        def patched_inproc_exec(self, code_str, global_scope, local_scope):
            run_id = uuid.uuid4().hex
            args = {"mode": "in-process",
                    "script_len": len(code_str) if code_str else 0}
            started_at = handler.on_tool_start("ScriptExec", args, run_id)
            token = _current_parent.set(run_id)
            try:
                result = original_exec(self, code_str, global_scope, local_scope)
                handler.on_tool_end("ScriptExec", result, run_id, started_at)
                return result
            except Exception as e:
                handler.on_tool_error("ScriptExec", e, run_id, started_at)
                raise
            finally:
                _current_parent.reset(token)
        patched_inproc_exec._pi_patched = True
        RunDynamicAnalysisController._run_generated_code = patched_inproc_exec
        print("[litellm_tool_logger] patched in-process ScriptExec hook",
              flush=True)
```

That is the entire change.

## Verification

Re-run the demo (same data, same model) and check:

```
cd /users/Minqiu/pi-ebpf-tracing-handoff
bash trace_script_bcc_scilink.sh   # or analyze_codebase_scilink.py directly
```

Expected outputs in the new trace directory:

| Artifact | Before | After |
|---|---|---|
| `tool_calls.log` line count | 4 | ~18-20 (4 outer + 13 controllers + 1 ScriptExec, ×2 if a refinement iteration runs) |
| `pi_events.jsonl` controller events with `parent_run_id` set | none | every controller event references the `Run_analysis` run_id |
| `call_dag.dot` tree | flat, 4 isolated boxes | `Run_analysis` with `contains` edges to 13+ children, one of which (`RunDynamicAnalysis`) contains a `ScriptExec` grandchild |
| Gantt total time accounting | 139 s `Run_analysis` only | controller sub-bars summing to ≈139 s under `Run_analysis` |

Spot checks to run by hand:

- `grep RunPreprocessing tool_calls.log` returns ≥1 line.
- `grep '"parent_run_id"' pi_events.jsonl | head` shows controller
  events linking back to the `Run_analysis` run_id.
- Open `call_dag.html`; the graph is a tree with `Run_analysis` at
  the root.
- `RunDynamicAnalysis` has exactly one `ScriptExec` child.

If the Gantt visual still renders parent and children as parallel rows
(rather than nesting), the rendering layer in `visualize_strace.py`
needs a `depth = chain-length-from-root` field and indented y-axis
layout — handle as a follow-up commit, not part of this change, since
it depends on visual inspection of the output.

## Regression risk

- SciLink side: extracting one line into a method is behaviorally
  identical. No business-logic code reads `_run_generated_code` by
  introspection, so adding a new method on the class is safe.
- Tracer side: the new patch step is gated on import success and on
  the existing `_pi_patched` sentinel, so it is idempotent and
  cannot double-wrap. If the SciLink module structure changes
  (renamed module, removed class), the tracer logs a warning and
  proceeds — runs do not fail.
- Other agents (curve_fitting, image_analysis, sam, fft, atomistic):
  their controller modules are not in `CONTROLLER_MODULES`. Their
  behavior is unchanged. When they are added in a follow-up,
  per-agent verification runs are required.

## Suggested rollout order

1. Land the SciLink change (one method extraction) as its own
   commit / PR. It is a no-op for end users and for the existing
   tracer.
2. Land the tracer change. Re-run the demo and confirm the four
   verification points above.
3. Decide whether Gantt nesting needs a separate visual change based
   on what the new `call_dag.html` and `agent_timeline.html` look
   like.
