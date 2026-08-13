"""Replay token-exact bundles against an already-running vLLM endpoint."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agent_io_tracing.serving.vllm_endpoint import (
    _base_url,
    parse_cache_config,
    parse_prefix_cache,
    prefix_cache_delta,
    VLLMEndpoint,
)

FIXED_OUTPUT_TOKENS = 32


def _peak_concurrency(requests: list[dict[str, Any]]) -> int:
    events = []
    for request in requests:
        start = float(request.get("arrival_offset_ms", 0.0) or 0.0)
        duration = float(request.get("original_duration_ms", 0.0) or 0.0)
        events.append((start, 1))
        events.append((start + max(duration, 0.001), -1))
    current = peak = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        current += delta
        peak = max(peak, current)
    return max(peak, 1)


def _completion(
    endpoint: VLLMEndpoint,
    model: str,
    item: dict[str, Any],
) -> dict[str, Any]:
    sampling = dict(item.get("sampling_params") or {})
    allowed = {
        "temperature", "top_p", "top_k", "min_p", "seed",
        "frequency_penalty", "presence_penalty", "repetition_penalty",
        "ignore_eos", "min_tokens", "thinking_token_budget",
        "response_format",
    }
    payload = {
        key: value for key, value in sampling.items()
        if key in allowed and value is not None
    }
    payload.update({
        "model": model,
        "prompt": item["prompt_token_ids"],
        "max_tokens": FIXED_OUTPUT_TOKENS,
        "min_tokens": FIXED_OUTPUT_TOKENS,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "add_special_tokens": False,
        "return_token_ids": True,
        "request_id": f"replay-{item['request_id']}",
    })
    headers = {"Content-Type": "application/json"}
    if endpoint.api_key:
        headers["Authorization"] = f"Bearer {endpoint.api_key}"
    request = Request(
        f"{_base_url(endpoint.base_url)}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    first_token = None
    last_token = None
    output_parts = []
    output_token_ids = []
    usage = None
    response_id = None
    try:
        with urlopen(request, timeout=endpoint.timeout_s) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                response_id = response_id or chunk.get("id")
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    text = choice.get("text") or ""
                    token_ids = choice.get("token_ids") or []
                    if text or token_ids:
                        received = time.perf_counter()
                        if first_token is None:
                            first_token = received
                        last_token = received
                    output_parts.append(text)
                    output_token_ids.extend(token_ids)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(
            f"request {item['request_id']} returned HTTP {exc.code}: {detail}"
        ) from exc
    except (URLError, OSError) as exc:
        reason = getattr(exc, "reason", None) or str(exc)
        raise RuntimeError(
            f"request {item['request_id']} failed: {reason}"
        ) from exc
    ended = time.perf_counter()
    ttft_ms = (
        round((first_token - started) * 1000.0, 3)
        if first_token is not None else None
    )
    total_ms = round((ended - started) * 1000.0, 3)
    output_tokens = int(
        (usage or {}).get("completion_tokens") or len(output_token_ids)
    )
    tpot_ms = (
        round((last_token - first_token) * 1000.0 / (output_tokens - 1), 3)
        if first_token is not None
        and last_token is not None
        and output_tokens > 1
        else None
    )
    return {
        "index": item["index"],
        "request_id": item["request_id"],
        "provider_request_id": response_id,
        "prompt_tokens": len(item["prompt_token_ids"]),
        "output_tokens": output_tokens,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "total_ms": total_ms,
        "usage": usage,
        "output_text": "".join(output_parts),
        "output_token_ids": output_token_ids,
        "thread_id": threading.get_ident(),
    }


def run_bundle(
    bundle: dict[str, Any],
    endpoint: VLLMEndpoint,
    mode: str,
) -> list[dict[str, Any]]:
    requests = bundle.get("requests") or []
    if mode not in {"packed", "paced"}:
        raise ValueError("mode must be packed or paced")
    if not requests:
        return []
    if mode == "packed":
        workers = _peak_concurrency(requests)

        def task(item: dict[str, Any]) -> dict[str, Any]:
            return _completion(endpoint, bundle["served_model"], item)
    else:
        workers = min(len(requests), 64)
        replay_started = time.perf_counter()

        def task(item: dict[str, Any]) -> dict[str, Any]:
            target = float(item.get("arrival_offset_ms", 0.0) or 0.0) / 1000.0
            delay = target - (time.perf_counter() - replay_started)
            if delay > 0:
                time.sleep(delay)
            return _completion(endpoint, bundle["served_model"], item)

    print(
        f"replay: starting {len(requests)} requests "
        f"(mode={mode}, workers={workers})",
        file=sys.stderr,
        flush=True,
    )
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(task, item) for item in requests]
        for future in as_completed(futures):
            results.append(future.result())
            print(
                f"replay: completed {len(results)}/{len(requests)} requests",
                file=sys.stderr,
                flush=True,
            )
    return sorted(results, key=lambda row: row["index"])


def _cache_state(reset_before: bool, metrics_before: str) -> str:
    """What the prefix cache held when this replay started.

    A server without VLLM_SERVER_DEV_MODE=1 has no /reset_prefix_cache, so the
    only way to get a cold arm there is to restart the engine -- which a knob
    change requires anyway.  A queries counter still at zero proves the engine
    has served nothing since it came up, which is a measurement rather than a
    claim about what the operator did.
    """
    if reset_before:
        return "cold_by_reset"
    if parse_prefix_cache(metrics_before).get("queries") == 0:
        return "cold_since_restart"
    return "warm_inherited"


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


def replay_to_dir(
    bundle: dict[str, Any],
    endpoint: VLLMEndpoint,
    mode: str,
    output_dir: Path,
    arm: dict[str, Any] | None = None,
    reset_before: bool = False,
    bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Replay the bundle once and write requests, metrics and summary.

    The serving config comes from the server's own counters, so an arm is
    labelled by what was actually serving rather than by what the launch
    command asked for.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if reset_before:
        endpoint.reset_prefix_cache()
    metrics_before = endpoint.metrics()
    started = time.time()
    results = run_bundle(bundle, endpoint, mode)
    ended = time.time()
    metrics_after = endpoint.metrics()

    with (output_dir / "requests.jsonl").open("w", encoding="utf-8") as out:
        for row in results:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "metrics_before.prom").write_text(metrics_before, encoding="utf-8")
    (output_dir / "metrics_after.prom").write_text(metrics_after, encoding="utf-8")

    output_tokens = sum(row.get("output_tokens") or 0 for row in results)
    summary = {
        "bundle": str((bundle_path or Path(bundle.get("source_cell", ""))).resolve()),
        "bundle_requests": len(bundle.get("requests") or []),
        "mode": mode,
        # What the prefix cache held when this replay started.  A warm arm
        # measures the previous arm as much as its own, so the report must
        # never average the two together.
        "cache_state": _cache_state(reset_before, metrics_before),
        "arm": arm or json.loads(os.environ.get("KVCACHE_ARM_JSON", "{}")),
        "serving_config": parse_cache_config(metrics_before),
        "prefix_cache": prefix_cache_delta(metrics_before, metrics_after),
        "started_at_epoch_s": started,
        "ended_at_epoch_s": ended,
        "wall_s": round(ended - started, 3),
        "requests": len(results),
        "prompt_tokens": sum(row["prompt_tokens"] for row in results),
        "output_tokens": output_tokens,
        "fixed_output_tokens_per_request": FIXED_OUTPUT_TOKENS,
        "median_ttft_ms": _median(
            [row["ttft_ms"] for row in results if row["ttft_ms"] is not None]
        ),
        "median_tpot_ms": _median(
            [row["tpot_ms"] for row in results if row.get("tpot_ms") is not None]
        ),
        "median_total_ms": _median([row["total_ms"] for row in results]),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a token-exact bundle.")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("packed", "paced"), default="packed")
    parser.add_argument(
        "--url",
        default=(
            os.environ.get("VLLM_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
        ),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("VLLM_API_KEY") or "",
    )
    parser.add_argument("--reset-before", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.url:
        raise SystemExit("Set the vLLM URL.")
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    endpoint = VLLMEndpoint(args.url, args.api_key, timeout_s=300.0)
    try:
        summary = replay_to_dir(
            bundle,
            endpoint,
            args.mode,
            args.output_dir,
            reset_before=args.reset_before,
            bundle_path=args.bundle,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"replay failed: {exc}") from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
