#!/usr/bin/env python3
"""Check that forced streaming can be reassembled on THIS server before using it.

SCILINK_FORCE_STREAM=1 makes the launcher stream every request and rebuild the
chunks into a normal response.  Whether that rebuild keeps the model's output
depends on the exact chunk shape the server emits -- against vLLM 0.26 it once
produced empty responses that SciLink then retried 930 times.  Run this first:
it makes two real requests (one plain, one that should trigger a tool call),
dumps the raw chunks, and reports whether litellm.stream_chunk_builder gets the
content back.

    PYTHONPATH=src python tools/dump_stream_chunks.py \
        --base-url http://127.0.0.1:18080 --model Qwen3.6-27B

Exit code 0 means forced streaming is safe to enable for that server/model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


TOOL_SCHEMA = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]


def _plain(obj):
    """Chunks are pydantic models; get a JSON-able view without assuming which."""
    for attr in ("model_dump", "dict"):
        method = getattr(obj, attr, None)
        if callable(method):
            try:
                return method()
            except Exception:
                pass
    if isinstance(obj, dict):
        return obj
    return json.loads(json.dumps(obj, default=str))


def probe(litellm, name: str, model: str, base_url: str, api_key: str,
          messages: list, tools: list | None, out_dir: Path,
          max_tokens: int = 2048) -> bool:
    from agent_io_tracing.adapters.scilink.launcher import (
        _chunk_has_content, _response_has_output,
    )

    # A reasoning model spends its budget thinking before it answers: at 128
    # tokens this probe burned 120 on reasoning, emitted no content and no tool
    # call, and still passed -- a false all-clear.  Give it room to finish.
    request = dict(
        model=model, messages=messages, api_base=base_url, api_key=api_key,
        max_tokens=max_tokens, stream=True,
        stream_options={"include_usage": True},
        drop_params=True,
    )
    if tools:
        request["tools"] = tools

    chunks = []
    first_index = None
    for index, chunk in enumerate(litellm.completion(**request)):
        if first_index is None and _chunk_has_content(chunk):
            first_index = index
        chunks.append(chunk)

    dump = out_dir / f"chunks_{name}.json"
    dump.write_text(
        json.dumps([_plain(c) for c in chunks], indent=1, default=str),
        encoding="utf-8",
    )

    rebuilt = litellm.stream_chunk_builder(chunks, messages=messages)

    # What the raw stream actually carried, independent of the rebuild.  If the
    # deltas hold text the rebuild does not, the rebuild is the broken part.
    delta_content = 0
    delta_tool_fragments = 0
    for chunk in chunks:
        for choice in (_plain(chunk).get("choices") or []):
            delta = choice.get("delta") or {}
            delta_content += len(delta.get("content") or "")
            delta_tool_fragments += len(delta.get("tool_calls") or [])

    message = _plain(rebuilt).get("choices", [{}])[0].get("message", {}) if rebuilt else {}
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    reasoning = message.get("reasoning_content") or ""
    usage = (_plain(rebuilt).get("usage") or {}) if rebuilt else {}
    finish = (_plain(rebuilt).get("choices", [{}])[0].get("finish_reason")
              if rebuilt else None)

    # Reasoning text alone is not an answer: a call that returns only thinking
    # looks empty to the agent, which retries.  Demand real content or a real
    # tool call, and demand the rebuild keep what the stream delivered.
    answered = bool(content) or bool(tool_calls)
    kept = (len(content) >= delta_content) and (bool(tool_calls) or not delta_tool_fragments)
    ok = rebuilt is not None and answered and kept

    print(f"--- {name}")
    print(f"    chunks received       : {len(chunks)}")
    print(f"    first content chunk   : "
          f"{first_index if first_index is not None else 'NONE (TTFT unmeasurable)'}")
    print(f"    finish_reason         : {finish}")
    print(f"    stream carried        : {delta_content} content chars, "
          f"{delta_tool_fragments} tool_call fragments")
    print(f"    rebuilt content       : {len(content)} chars")
    print(f"    rebuilt tool_calls    : {len(tool_calls)}")
    print(f"    rebuilt reasoning     : {len(reasoning)} chars")
    print(f"    usage                 : {usage or 'MISSING'}")
    print(f"    raw chunks written to : {dump}")
    if not rebuilt:
        print("    VERDICT: rebuild returned None")
    elif not answered:
        print("    VERDICT: no content and no tool call -- the agent would see "
              "an empty answer and retry. Raise --max-tokens if finish_reason "
              "is 'length'; otherwise do NOT set SCILINK_FORCE_STREAM=1")
    elif not kept:
        print("    VERDICT: the stream carried output the rebuild dropped -- "
              "stream_chunk_builder is lossy here; do NOT set "
              "SCILINK_FORCE_STREAM=1")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True,
                        help="e.g. http://127.0.0.1:18080 (with or without /v1)")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument("--api-key", default="local-vllm")
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--litellm-prefix", default="openai/",
                        help="prefix selecting litellm's OpenAI-compatible adapter")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="output budget; a reasoning model needs room to "
                             "think before it answers")
    args = parser.parse_args()

    import litellm  # imported late so --help works without it
    litellm.drop_params = True

    base_url = args.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    model = f"{args.litellm_prefix}{args.model}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    results.append(probe(
        litellm, "text", model, base_url, args.api_key,
        [{"role": "user", "content": "Count from one to five in words."}],
        None, args.out_dir, args.max_tokens,
    ))
    results.append(probe(
        litellm, "toolcall", model, base_url, args.api_key,
        [{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
        TOOL_SCHEMA, args.out_dir, args.max_tokens,
    ))

    print()
    if all(results):
        print("OK: both probes reassembled with output intact.")
        print("Forced streaming is safe here: SCILINK_FORCE_STREAM=1")
        return 0
    print("FAIL: at least one probe lost its output during reassembly.")
    print("Leave SCILINK_FORCE_STREAM unset (default 0); inspect the dumped chunks.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
