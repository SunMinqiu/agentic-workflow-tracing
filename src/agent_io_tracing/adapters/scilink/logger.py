"""
litellm-based replacement for langchain_tool_logger.py.

SciLink (https://github.com/ziatdinovmax/SciLink) drives LLMs through
`litellm.completion(...)` and dispatches every agent-callable function through
`scilink.agents.exp_agents.analysis_orchestrator_tools.AnalysisOrchestratorTools.execute_tool`.
Neither hook point is reachable from a LangChain CallbackHandler, so this
module:

  1. Registers a `litellm.success_callback` / `failure_callback` to record
     each LLM invocation as a pair of pi_events.jsonl entries
     (message_start + message_end + usage), one system-prompt capture, etc.

  2. Monkey-patches `AnalysisOrchestratorTools.execute_tool` to record every
     tool call (tool_calls.log line + tool_execution_end event).

The output file format is byte-identical with langchain_tool_logger.py's so
parse_ebpf.py, summarize_pi_events.py and visualize_strace.py all keep
working without modification.

Output files (all written under log_dir):

  - tool_calls.log              parse_ebpf.py compatible
  - tool_calls.log.system_prompt
  - pi_events.jsonl             summarize_pi_events.py compatible
  - subagent_calls.log          empty placeholder (SciLink v1 has no
                                subagent classification yet; keep file
                                present so downstream code that checks it
                                isn't surprised)
"""

from __future__ import annotations

import json
import os
import traceback
import threading
import time
import uuid
import contextvars
import inspect
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from agent_io_tracing.analysis.io_api_classifier import classify_code
except Exception:
    classify_code = None  # type: ignore[assignment]

_current_parent: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "scilink_current_parent", default=None
)


# ----- formatting helpers (mirror langchain_tool_logger.py exactly) ---------


def _format_time(dt: datetime) -> str:
    return dt.strftime("%H:%M:%S.%f")


def _normalize_tool_name(name: str | None) -> str:
    if not name:
        return "Tool"
    if name.lower() == "bash":
        return "Bash"
    return name[0].upper() + name[1:]


def _python_literal(value: Any) -> str:
    """Round-trippable Python literal for ast.literal_eval."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        if value != value or value in (float("inf"), float("-inf")):
            return "None"
        return repr(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_python_literal(v) for v in value) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(
                f"{_python_literal(str(k))}: {_python_literal(v)}"
                for k, v in value.items()
            )
            + "}"
        )
    return _python_literal(str(value))


def _format_log_line(
    started_at: datetime,
    ended_at: datetime,
    tool_name: str,
    tool_id: str,
    tool_input: Any,
) -> str:
    duration_ms = (ended_at - started_at).total_seconds() * 1000.0
    return (
        f"[{_format_time(started_at)} -> {_format_time(ended_at)}] "
        f"({duration_ms:.1f}ms) {tool_name} (id={tool_id}) "
        f"input={_python_literal(tool_input)}\n"
    )


def _format_system_prompt_entry(captured_at: datetime, prompt: str) -> str:
    return (
        f"[{captured_at.isoformat()}] length={len(prompt)}\n"
        "--- SYSTEM PROMPT START ---\n"
        f"{prompt}\n"
        "--- SYSTEM PROMPT END ---\n"
        "\n"
    )


# ----- usage normalization (litellm response shapes) ------------------------


def _normalize_usage_from_litellm(response: Any) -> dict:
    """
    litellm wraps OpenAI-style completion responses.  Usage lives at
    response.usage with .prompt_tokens / .completion_tokens / .total_tokens
    and (for prompt-cached models) .prompt_tokens_details.cached_tokens.

    Returns the pi-shaped dict consumed by summarize_pi_events.py:
        {"input": int, "output": int, "cacheRead": int, "totalTokens": int}
    """
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return {"input": 0, "output": 0, "cacheRead": 0, "totalTokens": 0}

    def _get(obj: Any, *names: str) -> int:
        for n in names:
            if isinstance(obj, dict):
                v = obj.get(n)
            else:
                v = getattr(obj, n, None)
            if v is not None:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    pass
        return 0

    inp = _get(usage, "prompt_tokens", "input_tokens", "input")
    out = _get(usage, "completion_tokens", "output_tokens", "output")
    total = _get(usage, "total_tokens", "totalTokens") or (inp + out)

    # Anthropic / OpenAI prompt-caching: details nested one level down.
    cache = 0
    details = (
        getattr(usage, "prompt_tokens_details", None)
        or (usage.get("prompt_tokens_details") if isinstance(usage, dict) else None)
    )
    if details is not None:
        cache = _get(details, "cached_tokens", "cache_read", "cacheRead")

    return {
        "input": inp,
        "output": out,
        "cacheRead": cache,
        "totalTokens": total,
    }


def _to_epoch_ms(t: Any) -> float:
    """litellm passes datetime; fall back to wall-clock if something else."""
    if hasattr(t, "timestamp"):
        try:
            return t.timestamp() * 1000.0
        except Exception:
            pass
    if isinstance(t, (int, float)):
        return float(t) * (1000.0 if t < 1e12 else 1.0)
    return time.time() * 1000.0


def _to_datetime(t: Any) -> datetime:
    if isinstance(t, datetime):
        return t
    if isinstance(t, (int, float)):
        return datetime.fromtimestamp(float(t) if t < 1e12 else t / 1000.0)
    return datetime.now()


# ----- handler --------------------------------------------------------------


class _PendingTool:
    __slots__ = ("started_at", "tool_name", "args", "parent_run_id")

    def __init__(
        self,
        started_at: datetime,
        tool_name: str,
        args: Any,
        parent_run_id: str | None,
    ) -> None:
        self.started_at = started_at
        self.tool_name = tool_name
        self.args = args
        self.parent_run_id = parent_run_id


class LiteLLMToolLogger:
    """
    Pi-compatible event logger for SciLink agent runs.

    Construct once, then call `install_global(handler)` to wire litellm
    callbacks + monkey-patch SciLink's tool dispatcher.  Designed to be
    idempotent (install_global calling itself twice is a no-op).
    """

    def __init__(self, log_dir: str | os.PathLike) -> None:
        self._log_dir = Path(log_dir).resolve()
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._tool_log = self._log_dir / "tool_calls.log"
        self._subagent_log = self._log_dir / "subagent_calls.log"
        self._system_prompt_log = self._log_dir / "tool_calls.log.system_prompt"
        self._events_log = self._log_dir / "pi_events.jsonl"
        self._generated_code_log = self._log_dir / "generated_code.jsonl"

        # Truncate at start (mirror analyze_codebase_pi.py behavior).
        for p in (self._tool_log, self._subagent_log,
                  self._system_prompt_log, self._events_log,
                  self._generated_code_log):
            p.write_text("", encoding="utf-8")

        self._pending: dict[str, _PendingTool] = {}
        self._lock = threading.RLock()
        self._system_prompt_captured = False

    # ----- internal IO ---------------------------------------------------

    def _append_tool_log(self, line: str) -> None:
        with self._lock:
            with self._tool_log.open("a", encoding="utf-8") as f:
                f.write(line)

    def _append_system_prompt(self, entry: str) -> None:
        with self._lock:
            with self._system_prompt_log.open("a", encoding="utf-8") as f:
                f.write(entry)

    def _append_event(self, event: dict) -> None:
        with self._lock:
            with self._events_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def _append_generated_code(self, record: dict) -> None:
        with self._lock:
            with self._generated_code_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def capture_generated_code(
        self,
        *,
        run_id: str,
        code: Any,
        started_at: datetime,
        phase: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        code_str = code if isinstance(code, str) else ""
        if not code_str:
            return
        io_class: dict[str, Any] = {}
        if classify_code is not None:
            try:
                io_class = classify_code(code_str)
            except Exception as e:
                print(
                    "[litellm_tool_logger] WARNING: I/O-API capture failed "
                    f"(generated_code.jsonl classification partial): "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
                traceback.print_exc()
        else:
            print(
                "[litellm_tool_logger] WARNING: io_api_classifier not importable; "
                "generated_code.jsonl will contain raw code without precomputed layers",
                flush=True,
            )
        self._append_generated_code({
            "run_id": run_id,
            "role": "scilink",
            "timestamp": _to_epoch_ms(started_at),
            "phase": phase,
            "code_len": len(code_str),
            "code_sha256": io_class.get("code_sha256"),
            "imports": io_class.get("imports"),
            "io_layers": io_class.get("layers"),
            "io_signals": io_class.get("signals"),
            "parsed": io_class.get("parsed"),
            "metadata": metadata or {},
            "code": code_str,
        })

    # ----- litellm LLM events --------------------------------------------

    def _capture_system_prompt_once(self, messages: Any) -> None:
        if self._system_prompt_captured or not messages:
            return
        # messages is a list of {"role": "...", "content": "..."} dicts.
        try:
            for m in messages:
                role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                if not role or role.lower() != "system":
                    continue
                content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
                if isinstance(content, list):
                    # OpenAI multimodal block: extract text fragments.
                    content = "\n".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in content
                    )
                if isinstance(content, str) and content:
                    self._append_system_prompt(
                        _format_system_prompt_entry(datetime.now(), content)
                    )
                    self._system_prompt_captured = True
                    return
        except Exception:
            # System prompt capture is best-effort; never crash the run.
            pass

    def on_llm_success(
        self, kwargs: dict, completion_response: Any,
        start_time: Any, end_time: Any,
    ) -> None:
        """litellm success_callback signature."""
        try:
            self._capture_system_prompt_once(kwargs.get("messages"))
            usage = _normalize_usage_from_litellm(completion_response)
            run_id = uuid.uuid4().hex
            parent = _current_parent.get()

            start_event = {
                "type": "message_start",
                "run_id": run_id,
                "message": {
                    "role": "assistant",
                    "timestamp": _to_epoch_ms(start_time),
                },
            }
            end_event = {
                "type": "message_end",
                "run_id": run_id,
                "message": {
                    "role": "assistant",
                    "timestamp": _to_epoch_ms(end_time),
                    "usage": usage,
                },
            }
            if parent:
                start_event["parent_run_id"] = parent
                end_event["parent_run_id"] = parent
            self._append_event(start_event)
            self._append_event(end_event)
        except Exception as e:
            # Never let logging break the user's run.
            print(f"[litellm_tool_logger] on_llm_success error: {e!r}",
                  flush=True)

    def on_llm_failure(
        self, kwargs: dict, original_exception: BaseException,
        start_time: Any, end_time: Any,
    ) -> None:
        """litellm failure_callback signature."""
        try:
            self._capture_system_prompt_once(kwargs.get("messages"))
            run_id = uuid.uuid4().hex
            parent = _current_parent.get()
            start_event = {
                "type": "message_start",
                "run_id": run_id,
                "message": {
                    "role": "assistant",
                    "timestamp": _to_epoch_ms(start_time),
                },
            }
            end_event = {
                "type": "message_end",
                "run_id": run_id,
                "error": repr(original_exception),
                "message": {
                    "role": "assistant",
                    "timestamp": _to_epoch_ms(end_time),
                    "usage": {"input": 0, "output": 0, "cacheRead": 0, "totalTokens": 0},
                },
            }
            if parent:
                start_event["parent_run_id"] = parent
                end_event["parent_run_id"] = parent
            self._append_event(start_event)
            self._append_event(end_event)
        except Exception as e:
            print(f"[litellm_tool_logger] on_llm_failure error: {e!r}",
                  flush=True)

    # ----- SciLink tool events (drives via monkey-patch in install_global) -

    def on_tool_start(self, tool_name: str, args: dict, run_id: str) -> datetime:
        started_at = datetime.now()
        norm_name = _normalize_tool_name(tool_name)
        parent = _current_parent.get()
        with self._lock:
            self._pending[run_id] = _PendingTool(
                started_at=started_at,
                tool_name=norm_name,
                args=args if isinstance(args, dict) else {"raw": str(args)},
                parent_run_id=parent,
            )
        event = {
            "type": "message_update",
            "assistantMessageEvent": {
                "type": "toolcall_end",
                "toolCall": {
                    "id": run_id,
                    "arguments": args if isinstance(args, dict) else {"raw": str(args)},
                },
            },
        }
        if parent:
            event["parent_run_id"] = parent
        self._append_event(event)
        return started_at

    def on_tool_end(self, tool_name: str, result: Any, run_id: str,
                    started_at: datetime) -> None:
        ended_at = datetime.now()
        with self._lock:
            pending = self._pending.pop(run_id, None)
        if pending is None:
            norm_name = _normalize_tool_name(tool_name)
            args: Any = {}
            parent = None
        else:
            norm_name = pending.tool_name
            args = pending.args
            parent = pending.parent_run_id

        line = _format_log_line(
            started_at=started_at,
            ended_at=ended_at,
            tool_name=norm_name,
            tool_id=run_id,
            tool_input=args if isinstance(args, dict) else {"raw": str(args)},
        )
        self._append_tool_log(line)

        # Coerce result to a string body for the event.
        if result is None:
            body = ""
        elif isinstance(result, str):
            body = result
        else:
            try:
                body = json.dumps(result, ensure_ascii=False, default=str)
            except Exception:
                body = str(result)

        event = {
            "type": "tool_execution_end",
            "toolCallId": run_id,
            "result": {"content": [{"text": body}]},
        }
        if parent:
            event["parent_run_id"] = parent
        self._append_event(event)

    def on_tool_error(self, tool_name: str, error: BaseException,
                      run_id: str, started_at: datetime) -> None:
        self.on_tool_end(tool_name, f"[error] {error!r}", run_id, started_at)

    # ----- shutdown ------------------------------------------------------

    def flush_pending(self) -> None:
        """Drain still-open tool calls (e.g. agent killed mid-run)."""
        now = datetime.now()
        with self._lock:
            for run_id, pending in list(self._pending.items()):
                line = _format_log_line(
                    started_at=pending.started_at,
                    ended_at=now,
                    tool_name=pending.tool_name,
                    tool_id=run_id,
                    tool_input=pending.args if isinstance(pending.args, dict)
                               else {"raw": str(pending.args)},
                )
                with self._tool_log.open("a", encoding="utf-8") as f:
                    f.write(line)
                del self._pending[run_id]


# ----- global installation --------------------------------------------------


def install_global(handler: LiteLLMToolLogger) -> None:
    """
    Wire the handler into litellm + monkey-patch SciLink's tool dispatcher.

    Idempotent: calling twice will not double-register or double-wrap.

    Order matters: this MUST run before `scilink` is imported by user code,
    so that the patched class is the one SciLink references when it sets up
    the orchestrator.
    """
    # 1) litellm callbacks
    import litellm  # type: ignore

    if not isinstance(getattr(litellm, "success_callback", None), list):
        litellm.success_callback = []
    if not isinstance(getattr(litellm, "failure_callback", None), list):
        litellm.failure_callback = []

    if handler.on_llm_success not in litellm.success_callback:
        litellm.success_callback.append(handler.on_llm_success)
    if handler.on_llm_failure not in litellm.failure_callback:
        litellm.failure_callback.append(handler.on_llm_failure)

    # 2) SciLink tool dispatcher.  Importing this module is cheap; it doesn't
    #    instantiate any agents or open any network connections.
    try:
        from scilink.agents.exp_agents.analysis_orchestrator_tools import (  # type: ignore
            AnalysisOrchestratorTools,
        )
    except ImportError as e:
        print(
            f"[litellm_tool_logger] WARNING: could not import "
            f"AnalysisOrchestratorTools — tool-level logging disabled: {e}",
            flush=True,
        )
        return

    if not getattr(AnalysisOrchestratorTools.execute_tool, "_pi_patched", False):
        original = AnalysisOrchestratorTools.execute_tool

        def patched(self, tool_name, **kwargs):
            run_id = uuid.uuid4().hex
            started_at = handler.on_tool_start(tool_name, dict(kwargs), run_id)
            token = _current_parent.set(run_id)
            try:
                result = original(self, tool_name, **kwargs)
                handler.on_tool_end(tool_name, result, run_id, started_at)
                return result
            except Exception as e:
                handler.on_tool_error(tool_name, e, run_id, started_at)
                raise
            finally:
                _current_parent.reset(token)

        patched._pi_patched = True  # type: ignore[attr-defined]
        AnalysisOrchestratorTools.execute_tool = patched

    # 3) SciLink sub-agent script execution.  These subprocess-backed steps
    #    are where the scientific code actually runs inside Run_analysis.
    try:
        from scilink.executors import ScriptExecutor  # type: ignore
    except ImportError as e:
        print(
            f"[litellm_tool_logger] WARNING: could not import "
            f"ScriptExecutor — sub-agent code-exec events disabled: {e}",
            flush=True,
        )
        return

    patched_methods: list[str] = []

    def _patch_script_method(method_name: str) -> None:
        original_exec = getattr(ScriptExecutor, method_name, None)
        if original_exec is None or getattr(original_exec, "_pi_patched", False):
            return

        def patched_exec(self, *args, **kwargs):
            script_content = args[0] if args else kwargs.get("script_content")
            working_dir = (
                args[1] if len(args) > 1
                else kwargs.get("working_dir", kwargs.get("cwd"))
            )
            run_id = uuid.uuid4().hex
            call_args = {
                "method": method_name,
                "script_len": len(script_content) if script_content is not None else 0,
                "working_dir": working_dir,
            }
            started_at = handler.on_tool_start("ScriptExec", call_args, run_id)
            handler.capture_generated_code(
                run_id=run_id,
                code=script_content,
                started_at=started_at,
                phase="script_executor",
                metadata=call_args,
            )
            token = _current_parent.set(run_id)
            try:
                result = original_exec(self, *args, **kwargs)
                handler.on_tool_end("ScriptExec", result, run_id, started_at)
                return result
            except Exception as e:
                handler.on_tool_error("ScriptExec", e, run_id, started_at)
                raise
            finally:
                _current_parent.reset(token)

        patched_exec._pi_patched = True  # type: ignore[attr-defined]
        setattr(ScriptExecutor, method_name, patched_exec)
        patched_methods.append(method_name)

    for method_name in ("execute_script", "execute"):
        _patch_script_method(method_name)

    if patched_methods:
        print(
            "[litellm_tool_logger] patched ScriptExecutor methods: "
            + ", ".join(patched_methods),
            flush=True,
        )
    else:
        print(
            "[litellm_tool_logger] WARNING: no ScriptExecutor execute methods patched",
            flush=True,
        )

    # 4) Pipeline-controller granularity.
    #
    # SciLink hyperspectral controllers are plain classes called directly by
    # the agent, not routed through AnalysisOrchestratorTools.execute_tool.
    # Discover controller classes by convention and wrap execute(self, state)
    # so every pipeline step inherits the enclosing Run_analysis parent_run_id.
    CONTROLLER_MODULES = (
        "scilink.agents.exp_agents.controllers.base_controllers",
        "scilink.agents.exp_agents.controllers.hyperspectral_controllers",
    )
    STATE_SUMMARY_KEYS = (
        "iteration_title",
        "depth",
        "task_idx",
        "task_total",
        "n_components",
        "method",
        "axis_units",
    )

    def _state_summary(state: Any) -> dict:
        if not isinstance(state, dict):
            return {}
        return {k: state[k] for k in STATE_SUMMARY_KEYS if k in state}

    def _display_name(cls: type) -> str:
        name = cls.__name__
        suffix = "Controller"
        return name[:-len(suffix)] if name.endswith(suffix) else name

    def _has_execute_state_signature(method: Any) -> bool:
        try:
            params = list(inspect.signature(method).parameters.values())
        except (TypeError, ValueError):
            return False

        positional = [
            p for p in params
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        return len(positional) >= 2 and positional[1].name == "state"

    def _patch_controller_class(cls: type) -> bool:
        original = getattr(cls, "execute", None)
        if (
            original is None
            or getattr(original, "_pi_patched", False)
            or not callable(original)
            or not _has_execute_state_signature(original)
        ):
            return False

        name = _display_name(cls)

        def patched(self, state, *args, **kwargs):
            run_id = uuid.uuid4().hex
            started_at = handler.on_tool_start(name, _state_summary(state), run_id)
            token = _current_parent.set(run_id)
            try:
                result = original(self, state, *args, **kwargs)
                result_summary = _state_summary(result) if isinstance(result, dict) else result
                handler.on_tool_end(name, result_summary, run_id, started_at)
                return result
            except Exception as e:
                handler.on_tool_error(name, e, run_id, started_at)
                raise
            finally:
                _current_parent.reset(token)

        patched._pi_patched = True  # type: ignore[attr-defined]
        cls.execute = patched  # type: ignore[method-assign]
        return True

    patched_controllers: list[str] = []
    for mod_name in CONTROLLER_MODULES:
        try:
            mod = __import__(mod_name, fromlist=["*"])
        except ImportError as e:
            print(
                f"[litellm_tool_logger] WARNING: {mod_name} not importable: {e}",
                flush=True,
            )
            continue

        for cls_name, cls in inspect.getmembers(mod, inspect.isclass):
            if cls.__module__ != mod_name or not cls_name.endswith("Controller"):
                continue
            if _patch_controller_class(cls):
                patched_controllers.append(cls_name)

    if patched_controllers:
        print(
            "[litellm_tool_logger] patched controllers: "
            + ", ".join(patched_controllers),
            flush=True,
        )

    # 5) In-process script execution inside RunDynamicAnalysisController.
    # The SciLink-side seam is _run_generated_code(); without it, the tracer
    # cannot wrap a bare exec(...) call from outside the module.
    try:
        from scilink.agents.exp_agents.controllers.hyperspectral_controllers import (  # type: ignore
            RunDynamicAnalysisController,
        )
    except ImportError:
        RunDynamicAnalysisController = None  # type: ignore[assignment]

    if RunDynamicAnalysisController is not None:
        original_exec = getattr(
            RunDynamicAnalysisController, "_run_generated_code", None
        )
        if original_exec is None:
            print(
                "[litellm_tool_logger] WARNING: "
                "RunDynamicAnalysisController._run_generated_code not found; "
                "in-process ScriptExec hook disabled",
                flush=True,
            )
        elif not getattr(original_exec, "_pi_patched", False):

            def patched_inproc_exec(self, code_str, global_scope, local_scope):
                run_id = uuid.uuid4().hex
                call_args = {
                    "mode": "in-process",
                    "script_len": len(code_str) if code_str else 0,
                }
                started_at = handler.on_tool_start("ScriptExec", call_args, run_id)
                handler.capture_generated_code(
                    run_id=run_id,
                    code=code_str,
                    started_at=started_at,
                    phase="in_process_dynamic_analysis",
                    metadata=call_args,
                )
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

            patched_inproc_exec._pi_patched = True  # type: ignore[attr-defined]
            RunDynamicAnalysisController._run_generated_code = patched_inproc_exec
            print(
                "[litellm_tool_logger] patched in-process ScriptExec hook",
                flush=True,
            )
