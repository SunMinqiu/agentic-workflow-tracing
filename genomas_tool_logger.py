"""
Monkey-patch logger for GenoMAS (https://github.com/Liu-Hy/GenoMAS).

GenoMAS calls the OpenAI / Anthropic / Google / Ollama SDKs *directly* (not
through LangChain, not through litellm), so neither langchain_tool_logger
nor litellm_tool_logger hooks fire.  However, all six provider clients in
GenoMAS inherit from a single abstract base `LLMClient` and implement the
*same* async signature

    async def generate_completion(self, messages: list[dict]) -> dict

and return the *same* shape

    {"content": str, "usage": {"input_tokens": int, "output_tokens": int,
                                "cost": float}, "raw_response": Any}

So we hook exactly six method overrides and normalise the rest.  This is
strictly less invasive than monkey-patching three SDKs with mismatched
sync/async surface areas.

Output files (all written under log_dir):

  - tool_calls.log              parse_ebpf.py format-compatible
                                (one LLM call → one line tagged with role)
  - tool_calls.log.system_prompt one-shot capture of the first system prompt
  - pi_events.jsonl             summarize_pi_events.py format-compatible
                                (message_start / message_end with usage)
  - subagent_calls.log          empty placeholder (Phase 2 MVP; future
                                phase will hook GenoMAS agent dispatch)

Format is byte-identical with litellm_tool_logger.py so parse_ebpf.py,
summarize_pi_events.py and visualize_strace.py keep working without change.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

# ---------------------------------------------------------------------------
# pi-compat formatting helpers (mirror litellm_tool_logger.py byte-for-byte)
# ---------------------------------------------------------------------------


def _format_time(dt: datetime) -> str:
    return dt.strftime("%H:%M:%S.%f")


def _normalize_tool_name(name: str | None) -> str:
    if not name:
        return "LLMCall"
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


# ---------------------------------------------------------------------------
# Usage normalisation: GenoMAS-shape → pi-shape
# ---------------------------------------------------------------------------


def _to_pi_usage(genomas_result: Any) -> dict:
    """GenoMAS returns {"usage": {"input_tokens","output_tokens","cost"}}.
    pi schema expects {"input","output","cacheRead","totalTokens"}.
    """
    if not isinstance(genomas_result, dict):
        return {"input": 0, "output": 0, "cacheRead": 0, "totalTokens": 0}
    usage = genomas_result.get("usage") or {}
    inp = int(usage.get("input_tokens", 0) or 0)
    out = int(usage.get("output_tokens", 0) or 0)
    # GenoMAS doesn't surface cache-read tokens; raw_response may have them
    # but each provider stores them differently — best-effort, default 0.
    cache = 0
    return {
        "input": inp,
        "output": out,
        "cacheRead": cache,
        "totalTokens": inp + out,
    }


def _epoch_ms(dt: datetime) -> float:
    return dt.timestamp() * 1000.0


# ---------------------------------------------------------------------------
# Role attribution: walk one frame up the call stack and find an agent.
# Each LLM call goes through `await self.client.generate_completion(...)`
# from inside an Agent subclass method, so the caller's `self` is the agent.
# ---------------------------------------------------------------------------


_AGENT_CLASS_HINTS = (
    "PIAgent",
    "GEOAgent",
    "TCGAAgent",
    "StatisticianAgent",
    "CodeReviewerAgent",
    "DomainExpertAgent",
)


def _infer_role_from_stack() -> str:
    """Look up the stack for the first frame whose `self` looks like a
    GenoMAS Agent subclass.  Falls back to 'unknown'.
    """
    try:
        # Skip our own frame and the patched method's frame.
        frame = inspect.currentframe()
        if frame is None:
            return "unknown"
        # Walk up at most 15 frames to keep cost bounded.
        for _ in range(15):
            frame = frame.f_back
            if frame is None:
                return "unknown"
            caller_self = frame.f_locals.get("self")
            if caller_self is None:
                continue
            cls_name = type(caller_self).__name__
            if cls_name in _AGENT_CLASS_HINTS:
                return cls_name
            # Also accept any class whose name ends with "Agent" — covers
            # subclasses or renamed roles without touching this allowlist.
            if cls_name.endswith("Agent") and cls_name != "LLMClient":
                return cls_name
        return "unknown"
    finally:
        # Break the reference cycle the inspect module famously warns about.
        del frame


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class GenoMASToolLogger:
    """pi-compatible event logger for GenoMAS agent runs.

    Construct once, then call `install_global(handler)` to monkey-patch
    GenoMAS's six LLMClient subclasses.  install_global is idempotent.
    """

    def __init__(self, log_dir: str | os.PathLike) -> None:
        self._log_dir = Path(log_dir).resolve()
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._tool_log = self._log_dir / "tool_calls.log"
        self._subagent_log = self._log_dir / "subagent_calls.log"
        self._system_prompt_log = self._log_dir / "tool_calls.log.system_prompt"
        self._events_log = self._log_dir / "pi_events.jsonl"

        # Truncate at start (mirror analyze_codebase_pi.py behaviour).
        for p in (self._tool_log, self._subagent_log,
                  self._system_prompt_log, self._events_log):
            p.write_text("", encoding="utf-8")

        self._lock = threading.RLock()
        self._system_prompt_captured = False

    # ---- IO ------------------------------------------------------------

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

    # ---- system prompt capture (first call only) -----------------------

    def _capture_system_prompt_once(self, messages: Any) -> None:
        if self._system_prompt_captured or not messages:
            return
        try:
            for m in messages:
                role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
                if not role or role.lower() != "system":
                    continue
                content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
                if isinstance(content, str) and content:
                    self._append_system_prompt(
                        _format_system_prompt_entry(datetime.now(), content)
                    )
                    self._system_prompt_captured = True
                    return
        except Exception:
            pass  # Best-effort.

    # ---- LLM call lifecycle --------------------------------------------

    def on_llm_call(
        self,
        client: Any,
        messages: Any,
        result: Any,
        started_at: datetime,
        ended_at: datetime,
        error: BaseException | None = None,
    ) -> None:
        """One LLM call's start+end events, plus the tool_calls.log line."""
        try:
            self._capture_system_prompt_once(messages)

            run_id = uuid.uuid4().hex
            role = _infer_role_from_stack()
            provider = getattr(getattr(client, "config", None), "provider", "?")
            model = getattr(getattr(client, "config", None), "model_name", "?")
            # Tool name must be plain \w+ for summarize_pi_events.py's regex;
            # model+provider live in the input dict, not the name.
            tool_label = role

            # pi_events.jsonl
            start_event = {
                "type": "message_start",
                "run_id": run_id,
                "message": {
                    "role": "assistant",
                    "timestamp": _epoch_ms(started_at),
                },
                "genomas_role": role,
                "provider": provider,
                "model": model,
            }
            end_event = {
                "type": "message_end",
                "run_id": run_id,
                "message": {
                    "role": "assistant",
                    "timestamp": _epoch_ms(ended_at),
                    "usage": _to_pi_usage(result) if error is None else
                             {"input": 0, "output": 0, "cacheRead": 0, "totalTokens": 0},
                },
                "genomas_role": role,
                "provider": provider,
                "model": model,
            }
            if error is not None:
                end_event["error"] = repr(error)
            self._append_event(start_event)
            self._append_event(end_event)

            # NOTE: We deliberately do NOT write LLM calls to tool_calls.log.
            # That file is reserved for actual tool/code-exec invocations
            # (parse_ebpf.py + summarize_pi_events.py treat every line there
            # as a non-LLM tool call).  Writing LLM calls here was a Phase 2
            # bug that made downstream visualize_strace.py render "100% tool
            # time" instead of "100% LLM time".  When we add a real code-exec
            # hook (Step B), it will write to tool_calls.log via a separate
            # on_tool_call() method, not here.
        except Exception as e:
            # Never break the user's run because of logging.
            print(f"[genomas_tool_logger] on_llm_call error: {e!r}", flush=True)


# ---------------------------------------------------------------------------
# Async wrapper factory
# ---------------------------------------------------------------------------


def _make_async_wrapper(
    original: Callable[..., Awaitable[Any]],
    handler: GenoMASToolLogger,
) -> Callable[..., Awaitable[Any]]:
    """Build an async wrapper around an LLMClient.generate_completion."""
    async def wrapper(self, messages, *args, **kwargs):
        started_at = datetime.now()
        try:
            result = await original(self, messages, *args, **kwargs)
            ended_at = datetime.now()
            handler.on_llm_call(self, messages, result, started_at, ended_at)
            return result
        except BaseException as e:
            ended_at = datetime.now()
            handler.on_llm_call(self, messages, None, started_at, ended_at,
                                error=e)
            raise
    wrapper._genomas_patched = True  # type: ignore[attr-defined]
    return wrapper


# ---------------------------------------------------------------------------
# Global installation
# ---------------------------------------------------------------------------


_CLIENT_CLASS_NAMES = (
    "OpenAIClient",
    "AnthropicClient",
    "GoogleClient",
    "OllamaClient",
    "NovitaClient",
    "DeepSeekClient",
)


# ---------------------------------------------------------------------------
# Code-exec hook (Step B)
# ---------------------------------------------------------------------------
# GenoMAS funnels every LLM-generated Python snippet through a single
# entry point: core.execution.CodeExecutor.execute(code, timeout).  The
# method is async and dispatches the actual exec() to a background thread
# via loop.run_in_executor.  Wrapping it captures EVERY code-exec event
# (pandas, statsmodels, sparse_lmm, etc.) with one hook.  This closes the
# "unaccounted gap" between LLM calls in pi_events.jsonl.
#
# Events are written to BOTH pi_events.jsonl (as tool_execution_start /
# tool_execution_end, matching the SciLink schema) AND tool_calls.log
# (one line per exec, with code length and stdout length in the input dict).
# Downstream summarize_pi_events.py + visualize_strace.py then attribute
# wall-clock time correctly into LLM-time + code-exec-time + gap.


def _hook_code_executor(handler: GenoMASToolLogger) -> bool:
    """Patch CodeExecutor.execute to emit code_exec events. Returns True iff patched."""
    try:
        import core.execution as ce_mod  # type: ignore
    except ImportError as e:
        print(
            f"[genomas_tool_logger] WARNING: core.execution not importable; "
            f"code-exec hook disabled: {e}",
            flush=True,
        )
        return False

    Executor = getattr(ce_mod, "CodeExecutor", None)
    if Executor is None:
        print(
            "[genomas_tool_logger] WARNING: core.execution.CodeExecutor not found",
            flush=True,
        )
        return False

    original = Executor.execute
    if getattr(original, "_genomas_patched", False):
        return True  # already patched (idempotent)
    if not asyncio.iscoroutinefunction(original):
        print(
            "[genomas_tool_logger] WARNING: CodeExecutor.execute is not async; "
            "signature changed upstream — skipping code-exec hook",
            flush=True,
        )
        return False

    async def patched_execute(self, code, timeout=None, *args, **kwargs):
        started_at = datetime.now()
        run_id = uuid.uuid4().hex
        role = _infer_role_from_stack()
        code_len = len(code) if isinstance(code, str) else 0
        # tool_calls.log line + pi_events.jsonl start event
        line = _format_log_line(
            started_at=started_at,
            ended_at=started_at,  # placeholder; rewritten below isn't trivial,
            # so we just emit a complete line at end. Use a temp marker.
            tool_name="CodeExec",
            tool_id=run_id,
            tool_input={"role": role, "code_len": code_len, "timeout": timeout},
        )
        # Defer line write to end (so duration is correct).
        start_event = {
            "type": "tool_execution_start",
            "run_id": run_id,
            "tool_name": "CodeExec",
            "genomas_role": role,
            "code_len": code_len,
            "timestamp": _epoch_ms(started_at),
        }
        handler._append_event(start_event)

        try:
            result = await original(self, code, timeout, *args, **kwargs)
            ended_at = datetime.now()
            stdout_len = len(getattr(result, "stdout", "") or "")
            err = getattr(result, "error", None)
            is_timeout = getattr(result, "is_timeout", False)

            end_event = {
                "type": "tool_execution_end",
                "run_id": run_id,
                "tool_name": "CodeExec",
                "genomas_role": role,
                "timestamp": _epoch_ms(ended_at),
                "stdout_len": stdout_len,
                "is_timeout": is_timeout,
            }
            if err is not None:
                end_event["error"] = f"{type(err).__name__}: {str(err)[:200]}"
            handler._append_event(end_event)

            # Write the tool_calls.log line with correct ended_at and inputs.
            line = _format_log_line(
                started_at=started_at,
                ended_at=ended_at,
                tool_name="CodeExec",
                tool_id=run_id,
                tool_input={
                    "role": role,
                    "code_len": code_len,
                    "stdout_len": stdout_len,
                    "error": end_event.get("error", None),
                    "timeout": is_timeout,
                },
            )
            handler._append_tool_log(line)
            return result
        except BaseException as e:
            ended_at = datetime.now()
            end_event = {
                "type": "tool_execution_end",
                "run_id": run_id,
                "tool_name": "CodeExec",
                "genomas_role": role,
                "timestamp": _epoch_ms(ended_at),
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            }
            handler._append_event(end_event)
            line = _format_log_line(
                started_at=started_at,
                ended_at=ended_at,
                tool_name="CodeExec",
                tool_id=run_id,
                tool_input={"role": role, "code_len": code_len,
                            "error": end_event["error"]},
            )
            handler._append_tool_log(line)
            raise

    patched_execute._genomas_patched = True  # type: ignore[attr-defined]
    Executor.execute = patched_execute  # type: ignore[assignment]
    return True


def install_global(handler: GenoMASToolLogger) -> list[str]:
    """Patch every LLMClient subclass found in utils.llm.

    MUST be called BEFORE any GenoMAS agent is constructed (because the
    agents stash their own `.client` references at construction time and
    we don't want to chase those).  Calling main() *after* install_global
    is correct; calling install_global from inside a running event loop is
    fine.

    Returns the list of patched class names (for logging).
    """
    try:
        import utils.llm as llm_mod  # type: ignore
    except ImportError as e:
        print(
            f"[genomas_tool_logger] WARNING: utils.llm not importable; "
            f"is sys.path set to the GenoMAS repo? Error: {e}",
            flush=True,
        )
        return []

    patched: list[str] = []
    for cls_name in _CLIENT_CLASS_NAMES:
        cls = getattr(llm_mod, cls_name, None)
        if cls is None:
            continue
        original = getattr(cls, "generate_completion", None)
        if original is None:
            continue
        if getattr(original, "_genomas_patched", False):
            patched.append(f"{cls_name}(already)")
            continue
        if not asyncio.iscoroutinefunction(original):
            print(
                f"[genomas_tool_logger] WARNING: {cls_name}.generate_completion "
                f"is not async; skipping",
                flush=True,
            )
            continue
        cls.generate_completion = _make_async_wrapper(original, handler)  # type: ignore[assignment]
        patched.append(cls_name)

    # Code-exec hook (Step B): wrap CodeExecutor.execute so every exec() call
    # of LLM-generated code lands in pi_events.jsonl + tool_calls.log.
    if _hook_code_executor(handler):
        patched.append("CodeExecutor.execute")

    print(
        f"[genomas_tool_logger] patched: {', '.join(patched) if patched else '(none)'}",
        flush=True,
    )
    return patched
