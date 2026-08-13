#!/usr/bin/env python3
"""Tokenize each recorded call with the serving engine's own /tokenize, and cache it.

tiktoken cannot reproduce the server's prompt: the tools schema is a separate
request parameter (8421 tokens on SciLink) and base64 images are not text.  The
bias propagates into logical through the input/our_tokens rescaling.

Only ever adds a cache file; logical.py falls back to tiktoken per call, so
OpenAI and FreeInference cells are untouched.

    PYTHONPATH=src python3 -m agent_io_tracing.analysis.kvcache.server_tokens \\
        --results results --url http://127.0.0.1:18000
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

CACHE_NAME = "server_tokens.json"
SOURCE = "vllm_tokenize"


def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's byte-to-unicode table, used by byte-level BPE.

    ``/tokenize`` returns each token as its byte-level surface form, where a
    space is "Ġ" and a newline "Ċ".  Concatenating those strings
    directly prints mojibake, so map every character back to the byte it
    stands for.  Checked against the server's own ``/detokenize`` on three
    ranges of a real 15,225-token prompt: identical text.
    """
    printable = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapped = list(printable)
    spare = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            mapped.append(256 + spare)
            spare += 1
    return {chr(code): byte for byte, code in zip(printable, mapped)}


_BYTE_DECODER = _byte_decoder()


def decode_token_strs(token_strs: list[str]) -> str:
    """Join per-token surface forms back into the text the engine rendered."""
    raw = bytearray()
    for piece in token_strs:
        for char in piece:
            byte = _BYTE_DECODER.get(char)
            if byte is None:
                raw.extend(char.encode("utf-8"))
            else:
                raw.append(byte)
    return raw.decode("utf-8", errors="replace")


def _post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def tokenize_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the request as the server saw it: messages plus tools.

    ``tools`` lives in ``request_params``, not in ``messages`` -- it is a
    sibling request parameter, which is exactly why re-tokenizing ``messages``
    alone lost 8421 tokens per tool-bearing call.
    """
    params = record.get("request_params") or {}
    payload: dict[str, Any] = {
        "model": record.get("model"),
        "messages": record.get("messages") or [],
        # Per-token surface forms, cached alongside the ids so the prefix dump
        # can render any range offline.  Without them a report regenerated
        # after the tunnel is gone could only print token counts.
        "return_token_strs": True,
    }
    tools = params.get("tools")
    if tools:
        payload["tools"] = tools
    extra_body = params.get("extra_body") or {}
    template_kwargs = extra_body.get("chat_template_kwargs")
    if template_kwargs:
        payload["chat_template_kwargs"] = template_kwargs
    return payload


def _recorded_inputs(cell: Path) -> dict[str, int]:
    """run_id -> the input token count the server itself reported."""
    path = cell / "pi_events.jsonl"
    if not path.is_file():
        return {}
    out: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "message_end":
            continue
        run_id = event.get("run_id")
        usage = (event.get("message") or {}).get("usage") or {}
        if run_id:
            out[run_id] = int(usage.get("input") or 0)
    return out


def build_cell(
    cell: Path,
    url: str,
    timeout: float = 180.0,
    force: bool = False,
) -> dict[str, Any] | None:
    """Tokenize every call in one cell through the server; write the cache."""
    messages_path = cell / "messages.jsonl"
    if not messages_path.is_file():
        return None
    cache_path = cell / CACHE_NAME
    if cache_path.is_file() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    endpoint = url.rstrip("/")
    endpoint = endpoint[:-3] if endpoint.endswith("/v1") else endpoint
    endpoint = f"{endpoint}/tokenize"

    recorded = _recorded_inputs(cell)
    calls: list[dict[str, Any]] = []
    for line in messages_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        run_id = record.get("run_id")
        server_input = recorded.get(run_id, 0)
        entry: dict[str, Any] = {
            "run_id": run_id,
            "server_input": server_input,
            "model": record.get("model"),
        }
        try:
            result = _post(endpoint, tokenize_payload(record), timeout)
            tokens = result.get("tokens") or []
            entry["tokens"] = tokens
            if result.get("token_strs"):
                entry["token_strs"] = result["token_strs"]
            entry["count"] = int(result.get("count") or len(tokens))
            entry["exact"] = bool(server_input) and entry["count"] == server_input
        except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        calls.append(entry)

    tokenized = [c for c in calls if "tokens" in c]
    exact = [c for c in tokenized if c.get("exact")]
    cache = {
        "source": SOURCE,
        "endpoint": endpoint,
        "n_calls": len(calls),
        "n_tokenized": len(tokenized),
        "n_exact": len(exact),
        # Calls short of the server's own input lost tool_calls/tool_call_id at
        # record time; see the module docstring.  Kept visible rather than
        # averaged away.
        "residual_pct": (
            round(
                100 * (1 - sum(c["count"] for c in tokenized)
                       / sum(c["server_input"] for c in tokenized)),
                3,
            )
            if tokenized and sum(c["server_input"] for c in tokenized) else None
        ),
        "calls": calls,
    }
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return cache


def load(cell: Path) -> dict[str, list[int]]:
    """run_id -> server token ids, for every call the server could tokenize."""
    path = cell / CACHE_NAME
    if not path.is_file():
        return {}
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        c["run_id"]: c["tokens"]
        for c in cache.get("calls") or []
        if c.get("run_id") and c.get("tokens")
    }


def load_token_strs(cell: Path) -> dict[str, list[str]]:
    """run_id -> per-token surface forms, for rendering reused prefixes."""
    path = cell / CACHE_NAME
    if not path.is_file():
        return {}
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {
        c["run_id"]: c["token_strs"]
        for c in cache.get("calls") or []
        if c.get("run_id") and c.get("token_strs")
    }


def load_meta(cell: Path) -> dict[str, Any]:
    path = cell / CACHE_NAME
    if not path.is_file():
        return {}
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {k: v for k, v in cache.items() if k != "calls"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--url", required=True, help="vLLM base URL, e.g. http://127.0.0.1:18000")
    parser.add_argument("--cell", type=Path, default=None, help="one cell instead of all")
    parser.add_argument("--force", action="store_true", help="re-tokenize even if cached")
    args = parser.parse_args()

    cells = (
        [args.cell]
        if args.cell
        else sorted(
            c for c in args.results.glob("*/*")
            if c.is_dir() and (c / "messages.jsonl").is_file()
        )
    )
    for cell in cells:
        cache = build_cell(cell, args.url, force=args.force)
        if cache is None:
            continue
        print(
            f"{cell.name}: {cache['n_exact']}/{cache['n_tokenized']} exact "
            f"of {cache['n_calls']} calls, residual {cache['residual_pct']}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
