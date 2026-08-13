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
import hashlib
import inspect
import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from agent_io_tracing.adapters.trace_writer import TraceFileWriter
from agent_io_tracing.adapters.llm_trace import (
    apply_nocache_tag as _apply_nocache_tag,
    attribute_cache_hit,
    prefix_cache_counters,
    cache_request_config as _cache_request_config,
    capture_messages as _capture_messages,
    capture_request_params as _capture_request_params,
    capture_response as _capture_response,
    format_system_prompt as _format_system_prompt_entry,
    format_tool_log as _format_log_line,
    make_nocache_tag as _make_nocache_tag,
    normalize_messages as _normalize_messages,
    runtime_vendor as _runtime_vendor,
    stable_json as _stable_json,
    trace_cache_config as _trace_cache_config,
)

# I/O abstraction classifier (Phase 1 §3.3 / H1). Guard the
# import so a missing/broken classifier only disables classification, never
# breaks the run.
try:
    from agent_io_tracing.analysis.io_api_classifier import classify_code
except Exception:  # pragma: no cover - best-effort
    classify_code = None  # type: ignore

# ---------------------------------------------------------------------------
# pi-compat formatting helpers (mirror litellm_tool_logger.py byte-for-byte)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Usage normalisation: GenoMAS-shape → pi-shape
# ---------------------------------------------------------------------------


def _field(obj: Any, *names: str) -> Any:
    """First non-None of obj.<name> / obj[name] across the given names."""
    for n in names:
        v = obj.get(n) if isinstance(obj, dict) else getattr(obj, n, None)
        if v is not None:
            return v
    return None


def _cached_tokens_observation(genomas_result: dict) -> tuple[int, bool]:
    """Dig the provider's prompt-cache hit out of the raw response.

    GenoMAS's own ``usage`` dict is stripped to input/output/cost, but it also
    returns ``raw_response`` — the untouched provider object. OpenAI reports the
    reused-prefix length at ``usage.prompt_tokens_details.cached_tokens``; other
    OpenAI-compatible backends (e.g. FreeInference) leave it null. Best-effort,
    handles both attribute- and dict-shaped responses; defaults to 0.
    """
    raw = genomas_result.get("raw_response")
    if raw is None:
        return 0, False
    usage = _field(raw, "usage")
    if usage is None:
        return 0, False
    details = _field(usage, "prompt_tokens_details")
    src = details if details is not None else usage
    val = _field(src, "cached_tokens", "cache_read", "cacheRead")
    if val is None:
        return 0, False
    try:
        return int(val or 0), True
    except (TypeError, ValueError):
        return 0, False


def _cached_tokens_from_raw(genomas_result: dict) -> int:
    """Return cached tokens while preserving the legacy integer helper."""
    return _cached_tokens_observation(genomas_result)[0]


def _to_pi_usage(genomas_result: Any) -> dict:
    """GenoMAS returns {"usage": {"input_tokens","output_tokens","cost"},
    "raw_response": <provider object>}. pi schema expects
    {"input","output","cacheRead","totalTokens"}.
    """
    if not isinstance(genomas_result, dict):
        return {"input": 0, "output": 0, "cacheRead": 0, "totalTokens": 0}
    usage = genomas_result.get("usage") or {}
    inp = int(usage.get("input_tokens", 0) or 0)
    out = int(usage.get("output_tokens", 0) or 0)
    cache, cache_available = _cached_tokens_observation(genomas_result)
    # When the response omits cached_tokens, use an attributable per-request
    # Prometheus delta. An explicit cached_tokens=0 is a real observation and
    # must not be replaced.
    cache_source = "response" if cache_available else None
    if not cache_available:
        measured = genomas_result.get("_trace_cache") or {}
        if measured.get("attributable"):
            cache = int(measured.get("cacheRead") or 0)
            cache_available = True
            cache_source = "prometheus_request_delta"
    return {
        "input": inp,
        "output": out,
        "cacheRead": cache,
        "cacheReadAvailable": cache_available,
        "cacheReadSource": cache_source,
        "totalTokens": inp + out,
    }


def _provider_request_id(genomas_result: Any) -> str | None:
    if not isinstance(genomas_result, dict):
        return None
    raw = genomas_result.get("raw_response")
    value = _field(raw, "id", "request_id", "_request_id") if raw is not None else None
    return str(value) if value is not None else None


def _stream_timing(genomas_result: Any) -> dict[str, float]:
    """Read real streaming timestamps supplied by a provider integration.

    Provider clients can attach ``_trace_timing`` to the normalized result with
    wall-clock millisecond values for ``response_headers_ms``,
    ``first_token_ms``, and ``last_token_ms``. Missing fields stay missing.
    """
    if not isinstance(genomas_result, dict):
        return {}
    timing = genomas_result.get("_trace_timing")
    if not isinstance(timing, dict):
        return {}
    output: dict[str, float] = {}
    for name in ("response_headers_ms", "first_token_ms", "last_token_ms"):
        value = timing.get(name)
        if isinstance(value, (int, float)):
            output[name] = float(value)
    return output


def _epoch_ms(dt: datetime) -> float:
    return dt.timestamp() * 1000.0


def _nocache_enabled() -> bool:
    """No-cache arm: see `apply_nocache_tag` in adapters/llm_trace.py."""
    return os.environ.get("GENOMAS_NOCACHE", "0").lower() in {"1", "true", "yes"}


def _llm_cache_key(client: Any, messages: Any) -> str:
    provider = getattr(getattr(client, "config", None), "provider", "?")
    model = getattr(getattr(client, "config", None), "model_name", "?")
    payload = {"provider": provider, "model": model, "messages": messages}
    return hashlib.sha256(_stable_json(payload).encode("utf-8", "replace")).hexdigest()


def _text_from_messages(messages: Any, limit: int = 12000) -> str:
    chunks: list[str] = []
    try:
        for m in messages or []:
            if isinstance(m, dict):
                c = m.get("content")
            else:
                c = getattr(m, "content", None)
            if c is not None:
                chunks.append(str(c))
            if sum(len(x) for x in chunks) >= limit:
                break
    except Exception:
        return ""
    return "\n".join(chunks)[:limit].lower()


def _infer_llm_phase(messages: Any) -> str:
    text = _text_from_messages(messages)
    if any(k in text for k in ("backtrack", "revise action unit", "rollback", "retry")):
        return "action_unit_backtrack"
    if any(k in text for k in ("summary", "summarize", "summarisation", "summarization")):
        return "summary_write"
    if any(k in text for k in ("memory snippet", "validated code snippet", "snippet store")):
        return "memory_snippet_write"
    return "llm_reasoning"


def _infer_code_phase(role: str, code: str) -> str:
    text = f"{role}\n{code[:12000]}".lower()
    if any(k in text for k in ("memory_snippet", "memory snippet", "validated_code", "snippet_store")):
        return "memory_snippet_write"
    if any(k in text for k in ("backtrack", "rollback", "retry", "cleanup")):
        return "action_unit_backtrack"
    if any(k in text for k in ("summary", "summarize", "summarization", "final_report")):
        return "summary_write"
    return "code_exec"


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


class GenoMASToolLogger(TraceFileWriter):
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
        # Per-code-exec generated-code capture for I/O-API classification (§3.2/§3.3).
        self._generated_code_log = self._log_dir / "generated_code.jsonl"
        self._llm_cache_log = self._log_dir / "llm_cache.jsonl"
        # Raw per-call prompt capture: the full messages array BEFORE it is
        # hashed into cache_key. cache_key alone (a one-way sha256) can only
        # answer "is this whole prompt identical"; the raw messages are what let
        # us compute longest-common-prefix between calls and split the reused
        # prefix by source (system / instructions / history / tool output).
        self._messages_log = self._log_dir / "messages.jsonl"
        self._dump_messages = os.environ.get("GENOMAS_DUMP_MESSAGES", "1").lower() in {
            "1", "true", "yes"
        }

        self._cache_path = Path(
            os.environ.get("GENOMAS_LLM_CACHE_PATH") or str(self._llm_cache_log)
        )
        self._replay_mode = os.environ.get("GENOMAS_LLM_REPLAY", "0").lower() in {
            "1", "true", "yes", "replay"
        }
        self._replay_strict = os.environ.get("GENOMAS_LLM_REPLAY_STRICT", "0").lower() in {
            "1", "true", "yes"
        }
        self._llm_cache: dict[str, Any] = {}
        if self._cache_path.is_file():
            for line in self._cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = rec.get("cache_key")
                if isinstance(key, str) and "result" in rec:
                    self._llm_cache[key] = rec["result"]

        # Truncate at start (mirror analyze_codebase_pi.py behaviour).
        _truncate = [self._tool_log, self._subagent_log,
                     self._system_prompt_log, self._events_log,
                     self._generated_code_log]
        if self._dump_messages:
            _truncate.append(self._messages_log)
        for p in _truncate:
            p.write_text("", encoding="utf-8")
        if not self._replay_mode and self._cache_path == self._llm_cache_log:
            self._llm_cache_log.write_text("", encoding="utf-8")

        self._lock = threading.RLock()
        self._system_prompt_captured = False

    # ---- IO ------------------------------------------------------------

    def _append_llm_cache(self, record: dict) -> None:
        # not TraceFileWriter._append_json: the cache path is redirectable and
        # its directory may not exist yet
        with self._lock:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self._cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def get_cached_llm_result(self, key: str) -> Any | None:
        return self._llm_cache.get(key)

    def record_llm_result(self, key: str, result: Any) -> None:
        cached_result = result
        if isinstance(result, dict) and "_trace_timing" in result:
            cached_result = dict(result)
            cached_result.pop("_trace_timing", None)
        self._llm_cache[key] = cached_result
        if not self._replay_mode:
            self._append_llm_cache({
                "cache_key": key,
                "cached_response_hash": hashlib.sha256(
                    _stable_json(cached_result).encode("utf-8", "replace")
                ).hexdigest(),
                "result": cached_result,
            })

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
        cache_key: str | None = None,
        cache_hit: bool = False,
        run_id: str | None = None,
        started_monotonic_ns: int | None = None,
        ended_monotonic_ns: int | None = None,
        attempt: int = 1,
        nocache_tag: str | None = None,
        request_params: Any = None,
    ) -> None:
        """One LLM call's start+end events, plus the tool_calls.log line."""
        try:
            self._capture_system_prompt_once(messages)

            run_id = run_id or uuid.uuid4().hex
            role = _infer_role_from_stack()
            provider = getattr(getattr(client, "config", None), "provider", "?")
            model = getattr(getattr(client, "config", None), "model_name", "?")
            phase = _infer_llm_phase(messages)
            # Tool name must be plain \w+ for summarize_pi_events.py's regex;
            # model+provider live in the input dict, not the name.
            provider_request_id = _provider_request_id(result)
            timing = _stream_timing(result)
            config_params = getattr(
                getattr(client, "config", None),
                "extra_message_params",
                None,
            )
            captured_params = _capture_request_params(
                config_params,
                request_params,
            )
            captured_response = (
                _capture_response(result)
                if error is None and result is not None else {}
            )
            common = {
                "run_id": run_id,
                "agent_role": role,
                "genomas_role": role,
                "provider": provider,
                "vendor": _runtime_vendor(provider, model),
                "model": model,
                "cache_config": _trace_cache_config(_cache_request_config(client)),
                "phase": phase,
                "cache_key": cache_key,
                "cache_hit": cache_hit,
                "provider_request_id": provider_request_id,
                "attempt": attempt,
                "nocache": bool(nocache_tag),
            }

            # pi_events.jsonl
            request_start_event = {
                **common,
                "type": "message_request_start",
                "wall_time_ms": _epoch_ms(started_at),
                "monotonic_ns": started_monotonic_ns,
                "message": {
                    "role": "assistant",
                    "timestamp": _epoch_ms(started_at),
                },
            }
            start_event = {
                **common,
                "type": "message_start",
                "message": {
                    "role": "assistant",
                    "timestamp": _epoch_ms(started_at),
                },
                "wall_time_ms": _epoch_ms(started_at),
                "monotonic_ns": started_monotonic_ns,
            }
            end_event = {
                **common,
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "timestamp": _epoch_ms(ended_at),
                    "usage": _to_pi_usage(result) if error is None else
                             {"input": 0, "output": 0, "cacheRead": 0, "totalTokens": 0},
                },
                "wall_time_ms": _epoch_ms(ended_at),
                "monotonic_ns": ended_monotonic_ns,
                "stream_timing_available": "first_token_ms" in timing,
            }
            if error is not None:
                end_event["error"] = repr(error)
            self._append_event(request_start_event)
            self._append_event(start_event)
            for event_type, timing_key in [
                ("message_response_headers", "response_headers_ms"),
                ("message_first_token", "first_token_ms"),
                ("message_last_token", "last_token_ms"),
            ]:
                if timing_key not in timing:
                    continue
                self._append_event({
                    **common,
                    "type": event_type,
                    "wall_time_ms": timing[timing_key],
                    "message": {
                        "role": "assistant",
                        "timestamp": timing[timing_key],
                    },
                })
            self._append_event(end_event)

            # Raw prompt dump, joinable to pi_events via run_id. Written from the
            # same `messages` object used to compute cache_key, so it captures the
            # exact bytes that were prefilled — the input to prefix-overlap and
            # source-decomposition analysis (Q3 structure / Q4).
            if self._dump_messages:
                self._append_messages({
                    "run_id": run_id,
                    "cache_key": cache_key,
                    "agent_role": role,
                    "genomas_role": role,
                    "phase": phase,
                    "provider": provider,
                    "model": model,
                    "timestamp": _epoch_ms(started_at),
                    "messages": _normalize_messages(messages),
                    "request": {
                        "messages": _capture_messages(messages),
                        "parameters": captured_params,
                    },
                    "request_params": captured_params,
                    "response": captured_response,
                    "response_text": captured_response.get("text"),
                    "nocache_tag": nocache_tag,
                })

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
        started_monotonic_ns = time.monotonic_ns()
        run_id = uuid.uuid4().hex
        cache_key = _llm_cache_key(self, messages)
        if handler._replay_mode:
            cached = handler.get_cached_llm_result(cache_key)
            if cached is not None:
                ended_at = datetime.now()
                ended_monotonic_ns = time.monotonic_ns()
                handler.on_llm_call(
                    self, messages, cached, started_at, ended_at,
                    cache_key=cache_key, cache_hit=True,
                    run_id=run_id,
                    started_monotonic_ns=started_monotonic_ns,
                    ended_monotonic_ns=ended_monotonic_ns,
                    request_params=kwargs,
                )
                return cached
            if handler._replay_strict:
                ended_at = datetime.now()
                ended_monotonic_ns = time.monotonic_ns()
                err = RuntimeError(f"LLM replay cache miss: {cache_key}")
                handler.on_llm_call(
                    self, messages, None, started_at, ended_at,
                    error=err, cache_key=cache_key, cache_hit=False,
                    run_id=run_id,
                    started_monotonic_ns=started_monotonic_ns,
                    ended_monotonic_ns=ended_monotonic_ns,
                    request_params=kwargs,
                )
                raise err
        nocache_tag = _make_nocache_tag() if _nocache_enabled() else None
        sent_messages = (
            _apply_nocache_tag(messages, nocache_tag) if nocache_tag else messages
        )
        try:
            result = await original(self, sent_messages, *args, **kwargs)
            ended_at = datetime.now()
            ended_monotonic_ns = time.monotonic_ns()
            handler.record_llm_result(cache_key, result)
            if nocache_tag:
                hit = _cached_tokens_from_raw(result if isinstance(result, dict) else {})
                if hit:
                    print(
                        "[genomas_tool_logger] WARNING: no-cache arm got "
                        f"cached_tokens={hit} on run_id={run_id}; this cell is "
                        "not a valid no-cache observation.",
                        flush=True,
                    )
            handler.on_llm_call(
                self, messages, result, started_at, ended_at,
                cache_key=cache_key, cache_hit=False,
                run_id=run_id,
                started_monotonic_ns=started_monotonic_ns,
                ended_monotonic_ns=ended_monotonic_ns,
                nocache_tag=nocache_tag,
                request_params=kwargs,
            )
            return result
        except BaseException as e:
            ended_at = datetime.now()
            ended_monotonic_ns = time.monotonic_ns()
            handler.on_llm_call(self, messages, None, started_at, ended_at,
                                error=e, cache_key=cache_key, cache_hit=False,
                                run_id=run_id,
                                started_monotonic_ns=started_monotonic_ns,
                                ended_monotonic_ns=ended_monotonic_ns,
                                nocache_tag=nocache_tag,
                                request_params=kwargs)
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


def _install_openai_stream_timing(llm_mod: Any) -> bool:
    """Replace GenoMAS's OpenAI-compatible call with a usage-bearing stream."""
    enabled = os.environ.get("GENOMAS_CAPTURE_STREAM_TIMING", "1").lower() in {
        "1", "true", "yes",
    }
    cls = getattr(llm_mod, "OpenAIClient", None)
    if not enabled or cls is None:
        return False
    current = getattr(cls, "generate_completion", None)
    if current is None or getattr(current, "_genomas_stream_timing", False):
        return bool(current)

    async def generate_completion(self, messages):
        try:
            recent = getattr(llm_mod, "check_recent_openai_model", lambda _: False)
            if recent(self.model_name) and messages and messages[0].get("role") == "system":
                messages[0]["role"] = (
                    "assistant" if "o1-mini" in self.model_name.lower() else "developer"
                )
            params = dict(self.config.extra_message_params or {})
            params.pop("stream", None)
            params.pop("stream_options", None)
            cache_before = prefix_cache_counters()
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                **params,
            )
            headers_ms = time.time() * 1000.0
            first_token_ms = None
            last_token_ms = None
            content_parts: list[str] = []
            usage = None
            response_id = None
            response_model = self.model_name
            async for chunk in stream:
                now_ms = time.time() * 1000.0
                response_id = response_id or _field(chunk, "id")
                response_model = _field(chunk, "model") or response_model
                chunk_usage = _field(chunk, "usage")
                if chunk_usage is not None:
                    usage = chunk_usage
                choices = _field(chunk, "choices") or []
                delta = _field(choices[0], "delta") if choices else None
                content = _field(delta, "content") if delta is not None else None
                reasoning = (
                    _field(delta, "reasoning_content", "reasoning")
                    if delta is not None else None
                )
                if content:
                    content_parts.append(str(content))
                if content or reasoning:
                    if first_token_ms is None:
                        first_token_ms = now_ms
                    last_token_ms = now_ms
            if usage is None:
                raise RuntimeError(
                    "stream completed without usage; provider must support "
                    "stream_options.include_usage for KV-cache tracing"
                )
            if first_token_ms is None:
                raise RuntimeError("stream completed without a generated token")
            usage_dict = (
                usage.model_dump()
                if hasattr(usage, "model_dump") else dict(usage)
                if isinstance(usage, dict) else {
                    "prompt_tokens": _field(usage, "prompt_tokens"),
                    "completion_tokens": _field(usage, "completion_tokens"),
                    "prompt_tokens_details": _field(usage, "prompt_tokens_details"),
                }
            )
            raw_response = {
                "id": response_id,
                "model": response_model,
                "usage": usage_dict,
            }
            result = self._format_response(
                content="".join(content_parts),
                input_tokens=int(_field(usage, "prompt_tokens", "input_tokens") or 0),
                output_tokens=int(
                    _field(usage, "completion_tokens", "output_tokens") or 0
                ),
                raw_response=raw_response,
            )
            result["_trace_timing"] = {
                "response_headers_ms": headers_ms,
                "first_token_ms": first_token_ms,
                "last_token_ms": last_token_ms,
            }
            result["_trace_cache"] = attribute_cache_hit(
                cache_before,
                prefix_cache_counters(),
                int(_field(usage, "prompt_tokens", "input_tokens") or 0),
            )
            return result
        except Exception as exc:
            return self.handle_exception(exc)

    generate_completion._genomas_stream_timing = True  # type: ignore[attr-defined]
    cls.generate_completion = generate_completion
    return True


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

    # I/O-API classification depends on io_api_classifier being importable
    # (same dir as this module). If it isn't, the hook still records timing,
    # but generated_code.jsonl stays EMPTY and interface_mix never populates.
    # Make that loud instead of silent — this exact gap cost us a whole run.
    if classify_code is None:
        print(
            "[genomas_tool_logger] WARNING: io_api_classifier not importable — "
            "I/O-API classification DISABLED. generated_code.jsonl will be empty "
            "and phase1 interface_mix will have total_execs=0. Ensure "
            "agent_io_tracing.analysis.io_api_classifier is importable.",
            flush=True,
        )

    # Surface (once, with traceback) any runtime failure of classification or
    # the generated_code.jsonl write — otherwise it is swallowed and the file
    # silently stays empty (exactly the gap that cost us a run).
    _io_err_state = {"logged": False}

    async def patched_execute(self, code, timeout=None, *args, **kwargs):
        started_at = datetime.now()
        run_id = uuid.uuid4().hex
        role = _infer_role_from_stack()
        code_str = code if isinstance(code, str) else ""
        code_len = len(code_str)
        phase = _infer_code_phase(role, code_str)

        # I/O-API classification (H1). Capture the raw snippet + its classified
        # layers so we can (a) substantiate the interface-choice claim and
        # (b) re-classify offline if the rules evolve. Best-effort: never let
        # classification break the actual code execution.
        io_class: dict = {}
        if classify_code is not None and code_str:
            try:
                io_class = classify_code(code_str)
                handler._append_generated_code({
                    "run_id": run_id,
                    "role": role,
                    "timestamp": _epoch_ms(started_at),
                    "phase": phase,
                    "code_len": code_len,
                    "code_sha256": io_class.get("code_sha256"),
                    "imports": io_class.get("imports"),
                    "io_layers": io_class.get("layers"),
                    "io_signals": io_class.get("signals"),
                    "parsed": io_class.get("parsed"),
                    "code": code_str,
                })
            except Exception as e:
                io_class = {}
                if not _io_err_state["logged"]:
                    _io_err_state["logged"] = True
                    import traceback
                    print(
                        "[genomas_tool_logger] WARNING: I/O-API capture failed "
                        f"(generated_code.jsonl will stay empty): {type(e).__name__}: {e}",
                        flush=True,
                    )
                    traceback.print_exc()
        io_layers = io_class.get("layers") or []
        code_sha256 = io_class.get("code_sha256")

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
            "code_sha256": code_sha256,
            "io_layers": io_layers,
            "phase": phase,
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
                "io_layers": io_layers,
                "phase": phase,
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
                    "io_layers": io_layers,
                    "phase": phase,
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
                "phase": phase,
            }
            handler._append_event(end_event)
            line = _format_log_line(
                started_at=started_at,
                ended_at=ended_at,
                tool_name="CodeExec",
                tool_id=run_id,
                tool_input={"role": role, "code_len": code_len,
                            "phase": phase, "error": end_event["error"]},
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
    if _install_openai_stream_timing(llm_mod):
        patched.append("OpenAIClient.stream_timing")
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
