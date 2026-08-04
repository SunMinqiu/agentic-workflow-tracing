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


EMPTY_USAGE = {"input": 0, "output": 0, "cacheRead": 0, "totalTokens": 0}
_CACHE_REQUEST_KEYS = (
    "prompt_cache_key",
    "prompt_cache_retention",
    "prompt_cache_options",
    "cache_control",
)


def field(obj: Any, *names: str) -> Any:
    """Return the first non-null attribute or mapping value."""
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value is not None:
            return value
    return None


def integer_field(obj: Any, *names: str) -> int:
    value = field(obj, *names)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def stable_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return str(value)


def normalize_messages(messages: Any) -> list[dict[str, Any]]:
    """Preserve the outgoing role and content values used for prefix analysis."""
    output: list[dict[str, Any]] = []
    for message in messages or []:
        output.append({
            "role": field(message, "role"),
            "content": field(message, "content"),
        })
    return output


# No-cache arm ---------------------------------------------------------------
# A unique tag in front of the very first token makes every prefix unique, so
# no provider-side prefix cache can ever hit.  The tag goes only into the copy
# sent to the provider; messages.jsonl stores the untagged prompt plus the tag
# itself, so logical reuse and the prefix lineage stay intact and the sent
# prompt can still be rebuilt exactly (tag + original), which must score
# logical == realized == 0.

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
        "messages": normalize_messages(messages),
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8", "replace")).hexdigest()


def runtime_vendor(provider: Any, model: Any) -> str:
    """Identify the service vendor, not the SDK protocol adapter."""
    explicit = os.environ.get("GENOMAS_VENDOR") or os.environ.get("SCILINK_VENDOR")
    if explicit:
        return explicit
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


def cache_request_config(client: Any) -> dict[str, Any]:
    """Return only cache controls actually present in the outgoing request."""
    config = getattr(client, "config", None)
    params = getattr(config, "extra_message_params", None)
    if not isinstance(params, dict):
        return {}
    return {key: params[key] for key in _CACHE_REQUEST_KEYS if key in params}


def epoch_ms(value: Any) -> float:
    if hasattr(value, "timestamp"):
        try:
            return float(value.timestamp()) * 1000.0
        except Exception:
            pass
    if isinstance(value, (int, float)):
        return float(value) * (1000.0 if value < 1e12 else 1.0)
    return datetime.now().timestamp() * 1000.0


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
