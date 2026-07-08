"""
LangChain BaseCallbackHandler that mimics tool_call_logger.ts (the pi extension).

Outputs:

1. tool_calls.log
   Real tool calls (classified as "not a subagent"), appended on tool_execution_end
   in the exact format expected by parse_ebpf.py's TOOL_CALL_PATTERN regex:

       [HH:MM:SS.uuuuuu -> HH:MM:SS.uuuuuu] (Xms) ToolName (id=<run_id>) input={...}

   `input=...` is a Python literal (repr of a dict) so ast.literal_eval can parse
   it back.

2. subagent_calls.log
   Same line format as tool_calls.log, but for LangChain "tool" invocations whose
   execution subtree contains an LLM call (or whose name matches PI_SUBAGENT_NAME_REGEX).
   These are LangGraph subagents/handoff tools wrapped as tools; the orchestrator
   calls them like tools but they internally drive an agent loop, so counting them
   as ordinary tool calls would double-count agent orchestration time/tokens.

3. tool_calls.log.system_prompt
   First chat-model invocation's system message captured once, in the same
   block format the pi extension used.

4. pi_events.jsonl
   Translated LangChain events emitted as pi-shaped JSON so summarize_pi_events.py
   runs unchanged.  Shapes emitted (matching what summarize_pi_events.py reads):

   - {"type":"message_start", "message":{"role":"assistant", "timestamp":<ms>}}
   - {"type":"message_end",   "message":{"role":"assistant", "timestamp":<ms>,
                                         "usage":{"input":..., "output":...,
                                                  "cacheRead":..., "totalTokens":...}}
                              [, "parent_subagent_run_id":<run_id>]}
   - {"type":"message_update","assistantMessageEvent":{"type":"toolcall_end",
                                                       "toolCall":{"id":<run_id>,
                                                                   "arguments":{...}}}}
   - {"type":"tool_execution_end", "toolCallId":<run_id>,
                                   "result":{"content":[{"text":"..."}]}
                                   [, "is_subagent":true]
                                   [, "parent_subagent_run_id":<run_id>]}

   `parent_subagent_run_id` is present on inner LLM/tool events that ran inside
   a subagent; it points to that subagent's run_id and lets summarize_pi_events.py
   attribute inner LLM tokens / inner tool time to the right subagent.

Concurrency: LangChain may invoke callbacks from multiple threads (sync handler
fanned out by the async manager).  All file writes are protected by a single
RLock; pending tool-start records keyed by run_id.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
except ImportError:
    try:
        from langchain.callbacks.base import BaseCallbackHandler  # type: ignore
    except ImportError:
        # LangChain not installed.  Allow the module to import (so format
        # helpers can be unit-tested), but constructing LangChainToolLogger
        # will fail loudly at runtime.
        class BaseCallbackHandler:  # type: ignore
            _LANGCHAIN_MISSING = True

            def __init_subclass__(cls, **kwargs: Any) -> None:
                super().__init_subclass__(**kwargs)


# ----- formatting helpers (mirror tool_call_logger.ts) -----------------------


def _format_time(dt: datetime) -> str:
    return dt.strftime("%H:%M:%S.%f")  # 6-digit microseconds


def _normalize_tool_name(name: str | None) -> str:
    if not name:
        return "Tool"
    if name.lower() == "bash":
        return "Bash"
    return name[0].upper() + name[1:]


def _python_literal(value: Any) -> str:
    """
    Produce a Python literal that ast.literal_eval can round-trip.  Matches
    tool_call_logger.ts toPythonLiteral semantics closely enough that
    parse_ebpf.py's regex + literal_eval accepts the result.
    """
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return repr(value) if value == value and value not in (float("inf"), float("-inf")) else "None"
    if isinstance(value, str):
        # repr() emits a Python string literal with proper escaping.
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
    # Fallback: stringify and quote.
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


# ----- usage normalization ---------------------------------------------------


def _normalize_usage(llm_output: dict | None) -> dict:
    """
    Map provider-specific token-usage shapes onto pi's:
      {"input": int, "output": int, "cacheRead": int, "totalTokens": int}
    """
    if not isinstance(llm_output, dict):
        return {"input": 0, "output": 0, "cacheRead": 0, "totalTokens": 0}

    usage = (
        llm_output.get("token_usage")
        or llm_output.get("usage")
        or llm_output.get("usage_metadata")
        or {}
    )
    if not isinstance(usage, dict):
        return {"input": 0, "output": 0, "cacheRead": 0, "totalTokens": 0}

    inp = (
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or usage.get("input")
        or 0
    )
    out = (
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or usage.get("output")
        or 0
    )
    cache = (
        usage.get("cache_read_input_tokens")
        or usage.get("cache_read")
        or usage.get("cacheRead")
        or 0
    )
    total = usage.get("total_tokens") or usage.get("totalTokens") or (int(inp) + int(out))
    return {
        "input": int(inp or 0),
        "output": int(out or 0),
        "cacheRead": int(cache or 0),
        "totalTokens": int(total or 0),
    }


# ----- handler ---------------------------------------------------------------


class _PendingTool:
    __slots__ = ("started_at", "tool_name", "args")

    def __init__(self, started_at: datetime, tool_name: str, args: Any) -> None:
        self.started_at = started_at
        self.tool_name = tool_name
        self.args = args


class LangChainToolLogger(BaseCallbackHandler):
    """
    Pi-compatible tool / event logger for LangChain (and LangGraph) runs.

    Construct once per run and pass via RunnableConfig(callbacks=[...]),
    OR install globally by monkey-patching CallbackManager.configure (see
    install_global() in this module).
    """

    # tell LangChain we want chat-model events
    raise_error = False
    run_inline = True

    def __init__(self, log_dir: str | os.PathLike) -> None:
        self._log_dir = Path(log_dir).resolve()
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._tool_log = self._log_dir / "tool_calls.log"
        self._subagent_log = self._log_dir / "subagent_calls.log"
        self._system_prompt_log = self._log_dir / "tool_calls.log.system_prompt"
        self._events_log = self._log_dir / "pi_events.jsonl"

        # truncate at start (mirror analyze_codebase_pi.py behavior)
        for p in (self._tool_log, self._subagent_log, self._system_prompt_log, self._events_log):
            p.write_text("", encoding="utf-8")

        self._pending: dict[str, _PendingTool] = {}
        # RLock so helpers that take the lock can be nested under other
        # locked sections without deadlocking.
        self._lock = threading.RLock()
        self._system_prompt_captured = False

        # ----- subagent classification state -----
        # run_id -> parent_run_id (None for top-level). Populated by every
        # callback that sees a parent_run_id (chain/tool/llm/chat_model).
        self._parent_of: dict[str, str | None] = {}
        # run_ids whose subtree contains an LLM call. Built up on every
        # on_chat_model_start / on_llm_start by walking up parent_run_id.
        self._has_llm_descendant: set[str] = set()
        # run_ids that arrived via on_tool_start. Any of these whose run_id
        # ends up in _has_llm_descendant is a subagent (its tool body invoked
        # the LLM, which a real tool would never do).
        self._tool_run_ids: set[str] = set()
        # run_id -> nearest tool ancestor run_id (== the subagent the run is
        # attributed to). None for top-level work. Computed at on_*_start
        # time so the answer is stable by the time end events fire.
        self._subagent_ancestor: dict[str, str | None] = {}

        # Optional name-based heuristic fallback. Some LangGraph handoff
        # tools never call the LLM directly (they return a Command that the
        # graph router dispatches), so the subtree-LLM signal can miss them.
        # PI_SUBAGENT_NAME_REGEX="(.*_agent$)" classifies those by name.
        pattern = os.environ.get("PI_SUBAGENT_NAME_REGEX", "").strip()
        self._subagent_name_re: re.Pattern[str] | None = (
            re.compile(pattern) if pattern else None
        )

    # ----- internal IO --------------------------------------------------

    def _append_tool_log(self, line: str) -> None:
        with self._lock:
            with self._tool_log.open("a", encoding="utf-8") as f:
                f.write(line)

    def _append_subagent_log(self, line: str) -> None:
        with self._lock:
            with self._subagent_log.open("a", encoding="utf-8") as f:
                f.write(line)

    def _append_system_prompt(self, entry: str) -> None:
        with self._lock:
            with self._system_prompt_log.open("a", encoding="utf-8") as f:
                f.write(entry)

    def _append_event(self, event: dict) -> None:
        with self._lock:
            with self._events_log.open("a", encoding="utf-8") as f:
                # default=str so LangChain's HumanMessage / AIMessage / etc.
                # (which appear in tool args for agent-handoff tools and aren't
                # JSON-serializable by default) get stringified instead of
                # raising TypeError mid-write.
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    # ----- subagent classification helpers ------------------------------
    # All helpers below assume `self._lock` is held.

    def _record_parent_locked(self, run_id: Any, parent_run_id: Any) -> None:
        self._parent_of[str(run_id)] = str(parent_run_id) if parent_run_id else None

    def _mark_ancestors_has_llm_locked(self, parent_run_id: Any) -> None:
        """Walk up parent chain, marking each ancestor as having an LLM descendant."""
        cur: str | None = str(parent_run_id) if parent_run_id else None
        while cur is not None and cur not in self._has_llm_descendant:
            self._has_llm_descendant.add(cur)
            cur = self._parent_of.get(cur)

    def _find_subagent_ancestor_locked(self, parent_run_id: Any) -> str | None:
        """
        Nearest ancestor that came in via on_tool_start. That ancestor is the
        subagent this run is attributed to (since a real tool never contains
        further child runs, anything inside a tool's run subtree is the work
        of a subagent).  None for top-level (no tool ancestor).
        """
        cur: str | None = str(parent_run_id) if parent_run_id else None
        seen: set[str] = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            if cur in self._tool_run_ids:
                return cur
            cur = self._parent_of.get(cur)
        return None

    def _classify_subagent(self, run_key: str, tool_name: str) -> bool:
        """Subagent iff its subtree spawned an LLM call, or name matches regex."""
        with self._lock:
            if run_key in self._has_llm_descendant:
                return True
        if self._subagent_name_re is not None and self._subagent_name_re.search(tool_name):
            return True
        return False

    # ----- chain events (parent-tree only) ------------------------------

    def on_chain_start(
        self,
        serialized: dict,
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        # We don't log chains, but recording them keeps the parent chain
        # unbroken so LLM/tool runs nested under LangGraph internals still
        # walk back up to their owning subagent.
        with self._lock:
            self._record_parent_locked(run_id, parent_run_id)

    # ----- chat model events --------------------------------------------

    def on_chat_model_start(
        self,
        serialized: dict,
        messages: list,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        # System prompt: capture once on the first chat-model invocation.
        if not self._system_prompt_captured:
            sys_text = self._extract_system_prompt(messages)
            if sys_text is not None:
                self._append_system_prompt(_format_system_prompt_entry(datetime.now(), sys_text))
                self._system_prompt_captured = True

        # Parent tree + ancestor mark + subagent attribution.
        with self._lock:
            self._record_parent_locked(run_id, parent_run_id)
            self._mark_ancestors_has_llm_locked(parent_run_id)
            self._subagent_ancestor[str(run_id)] = self._find_subagent_ancestor_locked(
                parent_run_id
            )

        # message_start with run_id (so agent_timeline can pair start/end by id
        # in concurrent fan-out scenarios where sequence order isn't enough).
        self._append_event(
            {
                "type": "message_start",
                "run_id": str(run_id),
                "message": {
                    "role": "assistant",
                    "timestamp": time.time() * 1000.0,
                },
            }
        )

    def on_llm_start(
        self,
        serialized: dict,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            self._record_parent_locked(run_id, parent_run_id)
            self._mark_ancestors_has_llm_locked(parent_run_id)
            self._subagent_ancestor[str(run_id)] = self._find_subagent_ancestor_locked(
                parent_run_id
            )

        self._append_event(
            {
                "type": "message_start",
                "run_id": str(run_id),
                "message": {
                    "role": "assistant",
                    "timestamp": time.time() * 1000.0,
                },
            }
        )

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        llm_output = getattr(response, "llm_output", None)
        usage = _normalize_usage(llm_output)
        with self._lock:
            parent_subagent = self._subagent_ancestor.get(str(run_id))
        event: dict[str, Any] = {
            "type": "message_end",
            "run_id": str(run_id),
            "message": {
                "role": "assistant",
                "timestamp": time.time() * 1000.0,
                "usage": usage,
            },
        }
        if parent_subagent:
            event["parent_subagent_run_id"] = parent_subagent
        self._append_event(event)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """
        Emit a message_end so the agent_timeline can close the LLM bar even on
        failure.  Without this handler, on_llm_start fires but no end pair
        follows, which breaks start/end pairing.
        """
        with self._lock:
            parent_subagent = self._subagent_ancestor.get(str(run_id))
        event: dict[str, Any] = {
            "type": "message_end",
            "run_id": str(run_id),
            "error": repr(error),
            "message": {
                "role": "assistant",
                "timestamp": time.time() * 1000.0,
                "usage": {"input": 0, "output": 0, "cacheRead": 0, "totalTokens": 0},
            },
        }
        if parent_subagent:
            event["parent_subagent_run_id"] = parent_subagent
        self._append_event(event)

    # ----- tool events --------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        inputs: dict | None = None,
        **kwargs: Any,
    ) -> None:
        tool_name = _normalize_tool_name((serialized or {}).get("name"))
        args = self._coerce_inputs(input_str, inputs)
        started_at = datetime.now()
        run_key = str(run_id)

        with self._lock:
            self._record_parent_locked(run_id, parent_run_id)
            self._tool_run_ids.add(run_key)
            self._subagent_ancestor[run_key] = self._find_subagent_ancestor_locked(
                parent_run_id
            )
            self._pending[run_key] = _PendingTool(
                started_at=started_at,
                tool_name=tool_name,
                args=args,
            )

        # message_update / toolcall_end → drives content_tokens_by_tool_id in
        # summarize_pi_events.py.
        self._append_event(
            {
                "type": "message_update",
                "assistantMessageEvent": {
                    "type": "toolcall_end",
                    "toolCall": {
                        "id": run_key,
                        "arguments": args if isinstance(args, dict) else {"raw": str(args)},
                    },
                },
            }
        )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        ended_at = datetime.now()
        run_key = str(run_id)
        with self._lock:
            pending = self._pending.pop(run_key, None)
        if pending is None:
            started_at = ended_at
            tool_name = _normalize_tool_name(kwargs.get("name"))
            args: Any = {}
        else:
            started_at = pending.started_at
            tool_name = pending.tool_name
            args = pending.args

        is_subagent = self._classify_subagent(run_key, tool_name)
        with self._lock:
            parent_subagent = self._subagent_ancestor.get(run_key)

        # 1) tool_calls.log (real tools) or subagent_calls.log (subagents).
        #    Line format is byte-identical with the legacy regex so
        #    parse_ebpf.py continues to work on
        #    tool_calls.log without changes.
        line = _format_log_line(
            started_at=started_at,
            ended_at=ended_at,
            tool_name=tool_name,
            tool_id=run_key,
            tool_input=args if isinstance(args, dict) else {"raw": str(args)},
        )
        if is_subagent:
            self._append_subagent_log(line)
        else:
            self._append_tool_log(line)

        # 2) tool_execution_end event (summarize_pi_events.py output_tokens
        #    + per-subagent attribution).
        event: dict[str, Any] = {
            "type": "tool_execution_end",
            "toolCallId": run_key,
            "result": {
                "content": [{"text": self._coerce_output_text(output)}],
            },
        }
        if is_subagent:
            event["is_subagent"] = True
        if parent_subagent:
            event["parent_subagent_run_id"] = parent_subagent
        self._append_event(event)

    def on_tool_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        # Treat as completion with the exception text as the result.
        self.on_tool_end(f"[error] {error!r}", run_id=run_id, **kwargs)

    # ----- shutdown -----------------------------------------------------

    def flush_pending(self) -> None:
        """
        Drain still-open tool calls (e.g. agent killed mid-run).  Mirrors the
        session_shutdown hook in tool_call_logger.ts.
        """
        now = datetime.now()
        with self._lock:
            for run_key, pending in list(self._pending.items()):
                line = _format_log_line(
                    started_at=pending.started_at,
                    ended_at=now,
                    tool_name=pending.tool_name,
                    tool_id=run_key,
                    tool_input=pending.args if isinstance(pending.args, dict) else {"raw": str(pending.args)},
                )
                is_subagent = (
                    run_key in self._has_llm_descendant
                    or (
                        self._subagent_name_re is not None
                        and self._subagent_name_re.search(pending.tool_name) is not None
                    )
                )
                target = self._subagent_log if is_subagent else self._tool_log
                with target.open("a", encoding="utf-8") as f:
                    f.write(line)
                del self._pending[run_key]

    # ----- helpers ------------------------------------------------------

    @staticmethod
    def _extract_system_prompt(messages: Any) -> str | None:
        """
        LangChain `messages` can be List[List[BaseMessage]] or List[BaseMessage].
        Find the first SystemMessage and return its content.
        """
        if not messages:
            return None

        def _content_of(msg: Any) -> str | None:
            if msg is None:
                return None
            t = getattr(msg, "type", None) or getattr(msg, "role", None)
            if isinstance(t, str) and t.lower() == "system":
                content = getattr(msg, "content", None)
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return "\n".join(
                        c.get("text", "") if isinstance(c, dict) else str(c) for c in content
                    )
                return str(content) if content is not None else None
            return None

        # List[List[BaseMessage]]
        if isinstance(messages, list) and messages and isinstance(messages[0], list):
            for msg in messages[0]:
                got = _content_of(msg)
                if got:
                    return got
        # List[BaseMessage]
        elif isinstance(messages, list):
            for msg in messages:
                got = _content_of(msg)
                if got:
                    return got
        return None

    @staticmethod
    def _coerce_inputs(input_str: Any, inputs: dict | None) -> Any:
        if isinstance(inputs, dict) and inputs:
            return inputs
        if isinstance(input_str, dict):
            return input_str
        if isinstance(input_str, str):
            s = input_str.strip()
            # Try JSON first, then Python literal, then raw.
            if s.startswith("{") or s.startswith("["):
                try:
                    return json.loads(s)
                except Exception:
                    pass
                try:
                    import ast

                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (dict, list)):
                        return parsed
                except Exception:
                    pass
            return {"raw": s}
        return {"raw": str(input_str) if input_str is not None else ""}

    @staticmethod
    def _coerce_output_text(output: Any) -> str:
        if output is None:
            return ""
        if isinstance(output, str):
            return output
        # LangChain ToolMessage / Generation-ish objects have .content / .text
        for attr in ("content", "text"):
            v = getattr(output, attr, None)
            if isinstance(v, str):
                return v
            if isinstance(v, list):
                return "\n".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in v
                )
        try:
            return json.dumps(output, ensure_ascii=False, default=str)
        except Exception:
            return str(output)


# ----- global installation (so we don't need to know SRAgent's API) ---------


def install_global(handler: LangChainToolLogger) -> None:
    """
    Monkey-patch LangChain's CallbackManager / AsyncCallbackManager `configure`
    classmethods so every Runnable picks up the handler, regardless of whether
    SRAgent passes a RunnableConfig anywhere.

    Why monkey-patch instead of `register_configure_hook`?  The hook API has
    moved across LangChain versions; classmethod `configure` has been stable
    since 0.1.x.

    Idempotent: calling twice will not double-wrap.
    """
    import langchain_core.callbacks.manager as _mgr

    for cls_name in ("CallbackManager", "AsyncCallbackManager"):
        cls = getattr(_mgr, cls_name, None)
        if cls is None:
            continue
        if getattr(cls.configure, "_pi_patched", False):
            continue
        original = cls.configure

        def _make_patched(orig):
            def _patched(cls_, *args, **kwargs):
                # Find the current local_callbacks regardless of how the caller
                # passed it (kwarg vs 2nd positional).  Don't touch any other
                # parameter — relay them verbatim to avoid kwarg/positional
                # collisions like "got multiple values for inheritable_callbacks".
                if "local_callbacks" in kwargs:
                    local = kwargs["local_callbacks"]
                    via_kwarg = True
                elif len(args) >= 2:
                    local = args[1]
                    via_kwarg = False
                else:
                    local = None
                    via_kwarg = True  # nothing supplied; inject as kwarg

                # Idempotent injection of `handler`.
                handlers = [handler]
                if local is None:
                    new_local = handlers
                elif isinstance(local, list):
                    if any(h is handler for h in local):
                        new_local = local
                    else:
                        new_local = list(local) + handlers
                else:
                    # Opaque (BaseCallbackManager etc.) — leave alone.
                    new_local = local

                # Splice back where it came from, then relay.
                if via_kwarg:
                    kwargs["local_callbacks"] = new_local
                    return orig(*args, **kwargs)
                else:
                    new_args = args[:1] + (new_local,) + args[2:]
                    return orig(*new_args, **kwargs)

            _patched._pi_patched = True
            return _patched

        cls.configure = classmethod(_make_patched(original))
