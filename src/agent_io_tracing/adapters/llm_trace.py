"""Shared helpers for adapter-neutral LLM tracing."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from urllib.parse import urlparse
from datetime import datetime
from typing import Any


_CACHE_REQUEST_KEYS = (
    "prompt_cache_key",
    "prompt_cache_retention",
    "prompt_cache_options",
    "cache_control",
)
_CAPTURE_REQUEST_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "stop",
    "max_tokens",
    "max_completion_tokens",
    "min_tokens",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "n",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "reasoning_effort",
    "thinking_token_budget",
    "logprobs",
    "top_logprobs",
    "ignore_eos",
    "add_generation_prompt",
    "continue_final_message",
    "chat_template",
    "chat_template_kwargs",
    "stream",
    "stream_options",
    "extra_body",
)


def field(obj: Any, *names: str) -> Any:
    """Return the first non-null attribute or mapping value."""
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value is not None:
            return value
    return None


def stable_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(value)


# Fields a chat template renders but that live outside `content`.  Dropping
# them cost real tokens: replaying a 19-call SciLink trace through vLLM's own
# /tokenize came in 0.65-2.8% under the server's recorded input on exactly the
# calls that carried tool_calls, and the deficit grew with every tool round.
# Kept optional so a message without them serializes as it always did.
MESSAGE_EXTRA_FIELDS = ("tool_calls", "tool_call_id", "name", "function_call")


def normalize_messages(messages: Any) -> list[dict[str, Any]]:
    """Preserve the outgoing role and content values used for prefix analysis."""
    output: list[dict[str, Any]] = []
    for message in messages or []:
        entry: dict[str, Any] = {
            "role": field(message, "role"),
            "content": field(message, "content"),
        }
        for name in MESSAGE_EXTRA_FIELDS:
            value = field(message, name)
            if value is not None:
                entry[name] = jsonable(value)
        output.append(entry)
    return output


def jsonable(value: Any) -> Any:
    """Convert SDK and Pydantic values into stable JSON-compatible data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return jsonable(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return jsonable(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return {
                str(key): jsonable(item)
                for key, item in vars(value).items()
                if not str(key).startswith("_")
            }
        except Exception:
            pass
    return str(value)


def capture_messages(messages: Any) -> list[dict[str, Any]]:
    """Preserve complete outgoing messages for chat-template replay."""
    captured = jsonable(messages or [])
    return captured if isinstance(captured, list) else []


def capture_request_params(*sources: Any) -> dict[str, Any]:
    """Merge replay-relevant, non-secret generation parameters."""
    output: dict[str, Any] = {}
    for source in sources:
        value = jsonable(source)
        if not isinstance(value, dict):
            continue
        for key in _CAPTURE_REQUEST_KEYS:
            if key in value and value[key] is not None:
                output[key] = value[key]
    return output


def capture_response(response: Any) -> dict[str, Any]:
    """Capture comparable response text, reasoning, tools, and finish state."""
    direct_text = field(response, "content", "text")
    raw = field(response, "raw_response") or response
    choices = field(raw, "choices") or []
    choice = choices[0] if choices else None
    message = field(choice, "message") if choice is not None else None
    text = (
        field(message, "content")
        if message is not None else field(choice, "text")
        if choice is not None else direct_text
    )
    reasoning = (
        field(message, "reasoning_content", "reasoning")
        if message is not None else None
    )
    tool_calls = field(message, "tool_calls") if message is not None else None
    finish_reason = field(choice, "finish_reason") if choice is not None else None
    captured = {
        "text": jsonable(text),
        "reasoning": jsonable(reasoning),
        "tool_calls": jsonable(tool_calls),
        "finish_reason": jsonable(finish_reason),
    }
    return {
        key: value for key, value in captured.items()
        if value is not None
    }


# No-cache arm ---------------------------------------------------------------
# A unique tag in front of the very first token makes every prefix unique, so
# no provider-side prefix cache can ever hit.  The tag goes only into the copy
# sent to the provider; messages.jsonl stores the untagged prompt plus the tag
# itself, so logical reuse and the prefix lineage stay intact and the sent
# prompt can still be rebuilt exactly (tag + original), which must score
# logical == realized == 0.



EMPTY_USAGE = {"input": 0, "output": 0, "cacheRead": 0, "totalTokens": 0}


NOCACHE_TAG_PATTERN = re.compile(r"^\[nocache:[0-9a-f]{32}\]\n")


def make_nocache_tag() -> str:
    return f"[nocache:{uuid.uuid4().hex}]\n"


def apply_nocache_tag(messages: Any, tag: str) -> Any:
    """Return a copy of `messages` with `tag` in front of the first message."""
    if not isinstance(messages, list) or not messages:
        return messages
    first = messages[0]
    if isinstance(first, dict) and isinstance(first.get("content"), str):
        patched = dict(first)
        patched["content"] = tag + first["content"]
        return [patched, *messages[1:]]
    # Multimodal or unexpected shape: prepend a separate leading message so the
    # tag is still the very first token of the prefill.
    return [{"role": "system", "content": tag}, *messages]




def strip_nocache_tag(messages: Any) -> tuple[Any, str | None]:
    """Split a possibly tagged prompt into (untagged messages, tag or None).

    Used where the prompt is only observable after it was sent, such as a
    litellm callback, so the recorded prompt matches the cached arm byte for
    byte and stays comparable across arms.
    """
    if not isinstance(messages, list) or not messages:
        return messages, None
    first = messages[0]
    if not isinstance(first, dict):
        return messages, None
    content = first.get("content")
    if not isinstance(content, str):
        return messages, None
    match = NOCACHE_TAG_PATTERN.match(content)
    if not match:
        return messages, None
    tag = match.group(0)
    remainder = content[len(tag):]
    if remainder == "":
        # Tag was injected as its own leading message.
        return messages[1:], tag
    stripped = dict(first)
    stripped["content"] = remainder
    return [stripped, *messages[1:]], tag


def cache_key(provider: Any, model: Any, messages: Any) -> str:
    """Hash one normalized provider, model, and full-message request."""
    payload = {
        "provider": str(provider or "?"),
        "model": str(model or "?"),
        "messages": capture_messages(messages),
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8", "replace")).hexdigest()


def runtime_vendor(provider: Any, model: Any) -> str:
    """Identify the service vendor, not the SDK protocol adapter."""
    explicit = os.environ.get("GENOMAS_VENDOR") or os.environ.get("SCILINK_VENDOR")
    if explicit:
        return explicit
    serving = serving_config()
    backend = str(serving.get("backend") or serving.get("engine") or "").lower()
    if backend == "vllm":
        return "vLLM"
    base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    if base_url:
        host = (urlparse(base_url).hostname or base_url).lower()
        if "freeinference" in host:
            return "FreeInference"
        if "openai.com" in host:
            return "OpenAI"
        return host
    provider_name = str(provider or "").lower()
    if provider_name == "openai" or str(model or "").startswith("gpt-"):
        return "OpenAI"
    return str(provider or "unknown")


def serving_config() -> dict[str, Any]:
    """Return the arm manifest supplied by the shared serving driver."""
    raw = os.environ.get("KVCACHE_ARM_JSON", "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {
            "manifest_error": f"invalid KVCACHE_ARM_JSON: {exc.msg}",
        }
    if not isinstance(value, dict):
        return {
            "manifest_error": "KVCACHE_ARM_JSON must contain a JSON object",
        }
    return value


def trace_cache_config(request_config: Any = None) -> dict[str, Any]:
    """Combine request cache controls with the serving arm manifest."""
    output = dict(request_config) if isinstance(request_config, dict) else {}
    serving = serving_config()
    if serving:
        output["serving"] = serving
    return output


def cache_request_config(client: Any) -> dict[str, Any]:
    """Return only cache controls actually present in the outgoing request."""
    config = getattr(client, "config", None)
    params = getattr(config, "extra_message_params", None)
    if not isinstance(params, dict):
        return {}
    return {key: params[key] for key in _CACHE_REQUEST_KEYS if key in params}


def provider_request_id(response: Any) -> str | None:
    value = field(response, "id", "request_id", "_request_id")
    return str(value) if value is not None else None


def format_time(value: datetime) -> str:
    return value.strftime("%H:%M:%S.%f")


def python_literal(value: Any) -> str:
    """Return a value that can be read with ast.literal_eval."""
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
        return "[" + ", ".join(python_literal(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(
            f"{python_literal(str(key))}: {python_literal(item)}"
            for key, item in value.items()
        ) + "}"
    return python_literal(str(value))


def format_tool_log(
    started_at: datetime,
    ended_at: datetime,
    tool_name: str,
    tool_id: str,
    tool_input: Any,
) -> str:
    duration_ms = (ended_at - started_at).total_seconds() * 1000.0
    return (
        f"[{format_time(started_at)} -> {format_time(ended_at)}] "
        f"({duration_ms:.1f}ms) {tool_name} (id={tool_id}) "
        f"input={python_literal(tool_input)}\n"
    )


def format_system_prompt(captured_at: datetime, prompt: str) -> str:
    return (
        f"[{captured_at.isoformat()}] length={len(prompt)}\n"
        "--- SYSTEM PROMPT START ---\n"
        f"{prompt}\n"
        "--- SYSTEM PROMPT END ---\n\n"
    )


def infer_phase(messages: Any) -> str:
    text = "\n".join(str(field(message, "content") or "") for message in messages or [])
    text = text[:12000].lower()
    if any(word in text for word in ("backtrack", "rollback", "retry")):
        return "backtrack"
    if any(word in text for word in ("summary", "summarize", "summarisation", "summarization")):
        return "summary"
    if any(word in text for word in ("memory snippet", "validated code snippet", "snippet store")):
        return "memory_write"
    return "reasoning"


# ----- per-call prefix-cache measurement ------------------------------------
#
# vLLM leaves usage.prompt_tokens_details null, so a response never says how
# much of its prompt was served from cache.  Its Prometheus counters do, and
# reading them either side of one request attributes the delta to that request
# -- verified directly: a single request moved queries by exactly its prompt
# token count.  The attribution only holds while nothing else is in flight, so
# every reading carries the check that proves it.

PREFIX_CACHE_QUERIES = "vllm:prefix_cache_queries_total"
PREFIX_CACHE_HITS = "vllm:prefix_cache_hits_total"


def prefix_cache_counters(url: str | None = None, timeout: float = 2.0):
    """(queries, hits) in tokens, or None when no endpoint answers."""
    import urllib.request

    base = (url or os.environ.get("VLLM_URL") or "").strip()
    if not base:
        return None
    base = base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        with urllib.request.urlopen(f"{base}/metrics", timeout=timeout) as response:
            text = response.read().decode("utf-8", "replace")
    except Exception:
        return None
    queries = hits = None
    for line in text.splitlines():
        if line.startswith(PREFIX_CACHE_QUERIES):
            queries = float(line.rsplit(" ", 1)[-1])
        elif line.startswith(PREFIX_CACHE_HITS):
            hits = float(line.rsplit(" ", 1)[-1])
        if queries is not None and hits is not None:
            break
    if queries is None or hits is None:
        return None
    return int(queries), int(hits)


def attribute_cache_hit(before, after, prompt_tokens: int) -> dict[str, Any]:
    """Turn a counter pair into this call's cached tokens, with its own proof.

    ``attributable`` is the whole point: the queries delta must equal this
    request's prompt tokens.  Anything else means another request overlapped
    and the hits delta is not ours, so the caller must not record it.
    """
    if not before or not after:
        return {"attributable": False, "reason": "counters unavailable"}
    queries = after[0] - before[0]
    hits = after[1] - before[1]
    if queries < 0 or hits < 0:
        return {"attributable": False, "reason": "counters reset mid-call"}
    ok = bool(prompt_tokens) and queries == int(prompt_tokens)
    return {
        "attributable": ok,
        "queries_delta": queries,
        "hits_delta": hits,
        "cacheRead": hits if ok else 0,
        "reason": None if ok else (
            f"queries delta {queries} != prompt tokens {prompt_tokens}; "
            "another request overlapped"
        ),
    }
