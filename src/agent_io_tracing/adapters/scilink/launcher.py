#!/usr/bin/env python3
"""
Run a SciLink CLI invocation in-process under our litellm tool/event logger.

Mirrors analyze_codebase_sragent.py.  Two key differences from the SRAgent
version:

  1. The logger is litellm-based (litellm.success_callback + monkey-patch
     of AnalysisOrchestratorTools.execute_tool), not LangChain-based.
     `langchain_tool_logger` is irrelevant to SciLink — see comments in
     litellm_tool_logger.py.

  2. SciLink's CLI runs an interactive `input()`-based REPL ("👤 You: ...").
     Our trace runs aren't interactive, so we pre-load sys.stdin with the
     workload's prompt and rely on the loop's EOFError handler to terminate
     cleanly once the prompt has been consumed.

Outputs in <log_dir>:
  - tool_calls.log                parse_ebpf.py format-compatible
  - tool_calls.log.system_prompt  system prompt capture
  - pi_events.jsonl               summarize_pi_events.py format-compatible
  - subagent_calls.log            empty placeholder (no SciLink subagent
                                  classification yet)

Usage:
    python analyze_codebase_scilink.py <work_dir> <log_dir> <subcommand> \\
        --prompt "<prompt text>" -- <scilink_args>...

Example (eels_plasmons_demo):
    python analyze_codebase_scilink.py /tmp/work /tmp/log analyze \\
        --prompt "Find and characterize the plasmon peaks." -- \\
        --mode autonomous --model gpt-4o-mini \\
        --data examples/eels_plasmons_demo/datacube.npy \\
        --metadata examples/eels_plasmons_demo/datacube.json \\
        --session-dir /tmp/log/scilink_session

The text after `--` is forwarded verbatim to `scilink <subcommand>`.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from agent_io_tracing.adapters.cli import split_forwarded_argv
from agent_io_tracing.adapters.llm_trace import (
    apply_nocache_tag, attribute_cache_hit, make_nocache_tag,
    prefix_cache_counters,
)


def request_fingerprint(kwargs: dict[str, Any]) -> str:
    """Identity of the outgoing request: its messages plus its tools.

    Two requests with the same fingerprint ask the model exactly the same
    thing.  An agent that is making progress never does that twice in a row --
    each answer becomes part of the next prompt.
    """
    payload = {
        "messages": kwargs.get("messages"),
        "tools": kwargs.get("tools"),
    }
    try:
        blob = json.dumps(payload, sort_keys=True, default=str)
    except Exception:
        blob = repr(payload)
    return hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()


class RepeatGuard:
    """Abort when the same prompt is sent over and over.

    Consecutive repeats, not a call total: a long analysis makes hundreds of
    calls but never repeats a prompt, because the last answer is in the next
    one.
    """

    def __init__(self, repeat_limit: int = 20, total_limit: int = 0) -> None:
        self.repeat_limit = repeat_limit
        self.total_limit = total_limit
        self.last: str | None = None
        self.repeats = 1
        self.total = 0
        # Set once the first abort has been signalled, so a swallowed
        # SystemExit is answered with an unconditional process exit.
        self.aborted = False

    def check(self, fingerprint: str) -> str | None:
        """Return why the run must stop, or None to let this call through."""
        self.total += 1
        self.repeats = self.repeats + 1 if fingerprint == self.last else 1
        self.last = fingerprint
        if self.repeat_limit > 0 and self.repeats >= self.repeat_limit:
            return (
                f"the identical prompt has now been sent {self.repeats} times "
                f"in a row ({self.total} calls so far). An agent that is making "
                f"progress never repeats a prompt, so this is a retry loop. "
                f"Raise SCILINK_MAX_REPEAT_CALLS if this workload really does "
                f"resample one prompt that often."
            )
        if self.total_limit > 0 and self.total >= self.total_limit:
            return (
                f"{self.total} LLM calls in one cell, over the "
                f"SCILINK_MAX_CALLS={self.total_limit} ceiling."
            )
        return None


def _chunk_has_content(chunk: Any) -> bool:
    """True once a chunk carries generated output, not just role metadata.

    The first chunk of an OpenAI-style stream usually announces the assistant
    role with an empty delta; timing TTFT from it would understate the model's
    real time to first token.  Tool-call deltas count as content -- for a call
    that only emits a tool call, they are the first thing generated.
    """
    try:
        choices = getattr(chunk, "choices", None)
        if choices is None and isinstance(chunk, dict):
            choices = chunk.get("choices")
        for choice in choices or []:
            delta = getattr(choice, "delta", None)
            if delta is None and isinstance(choice, dict):
                delta = choice.get("delta")
            if delta is None:
                continue
            content = getattr(delta, "content", None)
            if content is None and isinstance(delta, dict):
                content = delta.get("content")
            if content:
                return True
            tool_calls = getattr(delta, "tool_calls", None)
            if tool_calls is None and isinstance(delta, dict):
                tool_calls = delta.get("tool_calls")
            if tool_calls:
                return True
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning is None and isinstance(delta, dict):
                reasoning = delta.get("reasoning_content")
            if reasoning:
                return True
    except Exception:
        return False
    return False


def _response_has_output(response: Any) -> bool:
    """True when a rebuilt response carries something the caller can use.

    Narrower than _chunk_has_content on purpose: that accepts reasoning because
    on a thinking model the first token generated is reasoning, which is what
    TTFT measures.  Reasoning alone is not an answer -- the agent retries it.
    """
    try:
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")
        for choice in choices or []:
            message = getattr(choice, "message", None)
            if message is None and isinstance(choice, dict):
                message = choice.get("message")
            if message is None:
                continue
            for name in ("content", "tool_calls"):
                value = getattr(message, name, None)
                if value is None and isinstance(message, dict):
                    value = message.get(name)
                if value:
                    return True
    except Exception:
        return False
    return False


def _delta_field(chunk: Any, name: str) -> Any:
    """One field of a streaming chunk's first delta, object or dict shaped."""
    choices = getattr(chunk, "choices", None)
    if choices is None and isinstance(chunk, dict):
        choices = chunk.get("choices")
    for choice in choices or []:
        delta = getattr(choice, "delta", None)
        if delta is None and isinstance(choice, dict):
            delta = choice.get("delta")
        if delta is None:
            continue
        value = getattr(delta, name, None)
        if value is None and isinstance(delta, dict):
            value = delta.get(name)
        if value:
            return value
    return None


def describe_response(response: Any) -> dict[str, Any]:
    """What the caller will actually see, read the way the caller reads it.

    SciLink normalizes a response with plain attribute access
    (``getattr(choice, "message")`` then ``getattr(message, "content")``), so a
    diagnosis has to look at the same attributes rather than at a dict dump --
    an object can serialize to rich JSON and still expose nothing through the
    attributes the caller uses.
    """
    out: dict[str, Any] = {"type": type(response).__name__}
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    out["n_choices"] = len(choices or [])
    for choice in (choices or [])[:1]:
        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")
        out["finish_reason"] = getattr(choice, "finish_reason", None)
        for name in ("content", "reasoning_content"):
            value = getattr(message, name, None)
            if value is None and isinstance(message, dict):
                value = message.get(name)
            out[f"{name}_len"] = len(value) if isinstance(value, str) else None
            if isinstance(value, str) and value:
                out[f"{name}_head"] = value[:200]
        calls = getattr(message, "tool_calls", None)
        if calls is None and isinstance(message, dict):
            calls = message.get("tool_calls")
        out["n_tool_calls"] = len(calls or [])
        out["tool_names"] = [
            getattr(getattr(call, "function", None), "name", None)
            for call in (calls or [])[:4]
        ]
    usage = getattr(response, "usage", None)
    for name in ("prompt_tokens", "completion_tokens"):
        value = getattr(usage, name, None)
        if value is None and isinstance(usage, dict):
            value = usage.get(name)
        out[name] = value
    return out


def _dump_stream_debug(log_dir: Any, record: dict[str, Any]) -> None:
    """Append one line to stream_debug.jsonl; never break the run doing it."""
    if not log_dir:
        return
    try:
        with open(Path(log_dir) / "stream_debug.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


def streamed_completion(
    litellm: Any, original_completion: Any, logger_module: Any,
    c_args: tuple, c_kwargs: dict, debug_dir: Any = None,
) -> Any:
    """Drain a streamed completion, time the first token, rebuild the response.

    Returns the same ModelResponse shape a non-streamed call would return, so
    the caller cannot tell the difference; the only side effect is that the
    first-token instant becomes observable.
    """
    c_kwargs = dict(c_kwargs)
    c_kwargs["stream"] = True
    # vLLM and OpenAI only emit the usage block on a streamed call when asked;
    # without it the rebuilt response would carry no token counts.
    options = dict(c_kwargs.get("stream_options") or {})
    options.setdefault("include_usage", True)
    c_kwargs["stream_options"] = options

    chunks = []
    first_ms = None
    last_ms = None
    for chunk in original_completion(*c_args, **c_kwargs):
        if _chunk_has_content(chunk):
            now_ms = time.time() * 1000.0
            if first_ms is None:
                first_ms = now_ms
            # Keep advancing: the stream ends with a usage-only chunk that
            # carries no token, and timing TPOT to that would count a round
            # trip the model never spent generating.
            last_ms = now_ms
        chunks.append(chunk)
    if not chunks:
        raise RuntimeError("forced-stream completion produced no chunks")

    rebuilt = litellm.stream_chunk_builder(chunks, messages=c_kwargs.get("messages"))
    if debug_dir:
        _dump_stream_debug(debug_dir, {
            "n_chunks": len(chunks),
            "first_token_ms": first_ms,
            "rebuilt": describe_response(rebuilt),
            # The raw deltas, so a rebuild that drops output is separable from
            # a model that produced none.
            "delta_content_chars": sum(
                len(_delta_field(chunk, "content") or "") for chunk in chunks
            ),
            "delta_reasoning_chars": sum(
                len(_delta_field(chunk, "reasoning_content") or "") for chunk in chunks
            ),
            "delta_tool_fragments": sum(
                len(_delta_field(chunk, "tool_calls") or []) for chunk in chunks
            ),
        })
    if rebuilt is None:
        raise RuntimeError("stream_chunk_builder could not rebuild the response")

    # A rebuild that silently drops the generated text is worse than no
    # streaming at all: the caller sees an empty answer and retries forever.
    # One such run made 930 calls where 17 were expected.  Refuse the rebuilt
    # object unless it actually carries output, and let the caller fall back.
    if not _response_has_output(rebuilt):
        raise RuntimeError(
            "stream_chunk_builder returned a response with no content and no "
            f"tool calls (rebuilt from {len(chunks)} chunks)"
        )

    if first_ms is not None:
        logger_module.observed_first_token_ms.set(first_ms)
    if last_ms is not None:
        logger_module.observed_last_token_ms.set(last_ms)
    return rebuilt


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a SciLink subcommand under the pi-compatible litellm logger.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "work_dir",
        type=Path,
        help="Working directory for SciLink (it cd's here; session artifacts land here).",
    )
    p.add_argument(
        "log_dir",
        type=Path,
        help="Directory for tool_calls.log, pi_events.jsonl, system prompt.",
    )
    p.add_argument(
        "subcommand",
        type=str,
        help="SciLink subcommand (usually 'analyze' for eels_plasmons_demo and friends).",
    )
    p.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text fed to the agent's REPL via stdin.  With --mode autonomous "
             "the agent should run to completion on this one prompt; the EOF "
             "after it makes the REPL exit cleanly.",
    )
    p.add_argument(
        "--pre",
        type=str,
        default="",
        help="Space-separated string of GLOBAL flags to insert BEFORE the "
             "subcommand.  SciLink's CLI currently has no global flags so "
             "this is normally empty; kept for parity with the SRAgent harness.",
    )
    return p


def _stdin_for_scilink(
    subcommand: str,
    prompt: str,
    log_dir: Path,
    scilink_args: list[str] | None = None,
) -> str:
    """Build stdin for SciLink's interactive prompts before the chat loop."""
    lines: list[str] = []
    scilink_args = scilink_args or []

    # SciLink's setup() runs several interactive key prompts BEFORE the
    # pipeline/chat loop, each reading one line via input().  If stdin EOFs at
    # any of them SciLink dies with EOFError.  Feed exactly one Enter per prompt
    # that will actually appear, in SciLink's own order, so each is skipped
    # (auto-detect / skip) without the workload prompt being consumed as a key:
    #   1. Google Gemini key — asked when neither GEMINI_API_KEY nor
    #      GOOGLE_API_KEY is set ("Enter to auto-detect").
    #   2. Optional FutureHouse key — asked when FUTUREHOUSE_API_KEY is absent
    #      ("Enter to skip").
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        lines.append("")
    if not os.getenv("FUTUREHOUSE_API_KEY"):
        lines.append("")

    if subcommand == "plan":
        # `scilink plan` asks for the research objective and then the session
        # directory. In this harness the workload prompt is the objective.
        lines.append(prompt.rstrip("\n"))
        lines.append(str(log_dir / "scilink_session"))
    elif (
        subcommand == "analyze"
        and "--mode" in scilink_args
        and scilink_args[scilink_args.index("--mode") + 1:scilink_args.index("--mode") + 2] == ["autonomous"]
        and "--data" in scilink_args
    ):
        # In current SciLink, analyze --mode autonomous with --data/--metadata
        # runs the full pipeline before entering the chat loop. Feeding the
        # workload prompt after that triggers an extra, unnecessary chat turn.
        pass
    else:
        # `scilink analyze` reaches the main chat loop after setup.
        lines.append(prompt.rstrip("\n"))

    return "\n".join(lines) + "\n"


def main() -> int:
    ours, scilink_args = split_forwarded_argv(sys.argv[1:])
    args = build_arg_parser().parse_args(ours)

    work_dir = args.work_dir.resolve()
    log_dir = args.log_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Install the handler BEFORE importing scilink so the litellm callback
    # list and the monkey-patch on AnalysisOrchestratorTools.execute_tool
    # are in place before any SciLink module wires them in.
    import litellm  # type: ignore
    from agent_io_tracing.adapters.scilink import logger as scilink_logger
    from agent_io_tracing.adapters.scilink.logger import LiteLLMToolLogger, install_global

    # SciLink passes provider/tool parameters such as `tool_choice` through
    # litellm.  Some OpenAI model aliases reject those at litellm's validation
    # layer even when they are harmless for our workload.  Dropping unsupported
    # optional params keeps model swaps from aborting before the orchestrator
    # can do any real work.
    litellm.drop_params = True
    os.environ["LITELLM_DROP_PARAMS"] = "true"

    original_completion = litellm.completion

    # No-cache arm: tag the outgoing prompt so no provider-side prefix cache can
    # hit.  The logger strips the tag back off before writing messages.jsonl, so
    # the recorded prompt stays byte-comparable with the cached arm.
    nocache = os.environ.get("SCILINK_NOCACHE", "0").lower() in {"1", "true", "yes"}

    # TTFT and TPOT are only observable on a streamed response.  SciLink calls
    # litellm without stream=True, so we can stream on its behalf, time the
    # first chunk, then reassemble the chunks into the ordinary ModelResponse
    # it expects.
    #
    # Off by default: reassembly depends on the exact chunk shape a given
    # server emits, and against vLLM 0.26 it produced empty responses that
    # SciLink retried in a loop.  Verify with tools/dump_stream_chunks.py
    # against the server you are about to use, then set
    # SCILINK_FORCE_STREAM=1 for that run.
    force_stream = os.environ.get("SCILINK_FORCE_STREAM", "0").lower() in {
        "1", "true", "yes",
    }

    # Abort a retry storm from inside the process.  Set either limit to 0 to
    # disable it; the repeat guard is the one that matters, the total is a
    # backstop for loops that vary the prompt slightly on each attempt.
    guard = RepeatGuard(
        repeat_limit=int(os.environ.get("SCILINK_MAX_REPEAT_CALLS", "20") or 0),
        total_limit=int(os.environ.get("SCILINK_MAX_CALLS", "0") or 0),
    )

    # SCILINK_STREAM_DEBUG=1 writes stream_debug.jsonl next to the other trace
    # files: one line per streamed call, describing what the rebuild produced
    # and what the raw deltas carried.  Off by default; it is a diagnosis tool,
    # not instrumentation.
    stream_debug = (
        log_dir
        if os.environ.get("SCILINK_STREAM_DEBUG", "0").lower() in {"1", "true", "yes"}
        else None
    )

    def _streamed_completion(c_args: tuple, c_kwargs: dict):
        return streamed_completion(
            litellm, original_completion, scilink_logger, c_args, c_kwargs,
            debug_dir=stream_debug,
        )

    def completion_with_drop_params(*c_args, **c_kwargs):
        c_kwargs.setdefault("drop_params", True)
        # Fingerprint before the no-cache tag is applied: that tag is random
        # per call and would make every prompt look unique.
        stop_reason = guard.check(request_fingerprint(c_kwargs))
        if stop_reason:
            print(
                f"[analyze_codebase_scilink] ABORTING: {stop_reason}",
                file=sys.stderr, flush=True,
            )
            # Second time through means the first abort did not take: either a
            # handler swallowed SystemExit, or the call came from a worker
            # thread, where SystemExit kills only that thread.  Leave no way for
            # the loop to continue -- an unstoppable guard is the whole point.
            if guard.aborted:
                os._exit(3)
            guard.aborted = True
            # SystemExit rather than a normal exception: SciLink wraps model
            # calls in broad `except Exception` handlers that would swallow the
            # abort and keep looping, which is the very failure being stopped.
            raise SystemExit(3)
        if nocache and c_kwargs.get("messages"):
            c_kwargs["messages"] = apply_nocache_tag(
                c_kwargs["messages"], make_nocache_tag()
            )
        scilink_logger.observed_first_token_ms.set(None)
        scilink_logger.observed_last_token_ms.set(None)
        scilink_logger.observed_cache.set(None)
        cache_before = prefix_cache_counters()

        def _finish(response):
            """Attribute this call's cached tokens from the counter delta."""
            usage = getattr(response, "usage", None)
            prompt = getattr(usage, "prompt_tokens", None)
            if prompt is None and isinstance(usage, dict):
                prompt = usage.get("prompt_tokens")
            scilink_logger.observed_cache.set(
                attribute_cache_hit(cache_before, prefix_cache_counters(),
                                    int(prompt or 0))
            )
            return response

        # Never let instrumentation break the workload: if streaming or the
        # reassembly fails for this call, fall back to the plain request and
        # lose only that call's TTFT.
        if force_stream and not c_kwargs.get("stream"):
            try:
                return _finish(_streamed_completion(c_args, c_kwargs))
            except Exception as exc:  # noqa: BLE001 - see comment above
                print(
                    "[analyze_codebase_scilink] forced streaming failed "
                    f"({type(exc).__name__}: {exc}); retrying without stream",
                    file=sys.stderr, flush=True,
                )
                scilink_logger.observed_first_token_ms.set(None)
                scilink_logger.observed_last_token_ms.set(None)
        return _finish(original_completion(*c_args, **c_kwargs))

    completion_with_drop_params._pi_drop_params_patched = True  # type: ignore[attr-defined]
    litellm.completion = completion_with_drop_params
    print(
        "[analyze_codebase_scilink] litellm.drop_params=True; "
        "completion() injects drop_params=True"
        + ("; SCILINK_NOCACHE=1 (no-cache arm)" if nocache else "")
        + ("; forced streaming for TTFT/TPOT" if force_stream else ""),
        file=sys.stderr,
        flush=True,
    )

    handler = LiteLLMToolLogger(log_dir=log_dir)
    install_global(handler)

    # Parity with pi-coding-agent and SRAgent: some downstream code (and our
    # own debug paths) reads PI_TOOL_LOG from env.
    os.environ["PI_TOOL_LOG"] = str(log_dir / "tool_calls.log")

    # cd into work_dir so any local artefacts (SciLink session dir, generated
    # python scripts written by the autonomous code-exec agent, etc.) land
    # under our control and BCC's path filtering still works.
    os.chdir(work_dir)

    # Feed SciLink's setup prompts plus the workload prompt via stdin.  After
    # these lines are consumed, StringIO raises EOFError and SciLink exits its
    # chat loop cleanly.
    prompt_text = _stdin_for_scilink(
        args.subcommand, args.prompt, log_dir, scilink_args
    )
    sys.stdin = io.StringIO(prompt_text)

    pre_argv = args.pre.split() if args.pre else []
    sys.argv = ["scilink", *pre_argv, args.subcommand, *scilink_args]
    print(
        f"[analyze_codebase_scilink] cwd={work_dir} log_dir={log_dir}\n"
        f"[analyze_codebase_scilink] argv={sys.argv}\n"
        f"[analyze_codebase_scilink] stdin lines={prompt_text.count(chr(10))} "
        f"prompt={args.prompt[:120]!r}",
        file=sys.stderr,
        flush=True,
    )

    exit_code = 0
    try:
        # Defer SciLink import until after install_global() has run.
        from scilink.cli.main import main as scilink_main  # type: ignore
    except ImportError as e:
        print(
            f"[analyze_codebase_scilink] failed to import SciLink: {e}\n"
            "Hint: pip install scilink (or pip install -e . from a clone) "
            "in this venv.",
            file=sys.stderr,
        )
        return 2

    try:
        result = scilink_main()
        if isinstance(result, int):
            exit_code = result
    except SystemExit as se:
        exit_code = int(se.code) if isinstance(se.code, int) else (0 if se.code is None else 1)
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        handler.flush_pending()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
