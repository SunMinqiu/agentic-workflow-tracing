"""Replay token-exact bundles against an already-running vLLM endpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
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
    prometheus_window,
    prefix_cache_delta,
    VLLMEndpoint,
)

FIXED_OUTPUT_TOKENS = 32
CACHE_SALT_MODES = ("none", "shared", "unique")


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
    output_tokens: int = FIXED_OUTPUT_TOKENS,
    cache_salt: str | None = None,
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
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "add_special_tokens": False,
        "return_token_ids": True,
        "request_id": f"replay-{item['request_id']}",
    })
    if cache_salt is not None:
        payload["cache_salt"] = cache_salt
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
    started_at_epoch_s = time.time()
    first_token = None
    last_token = None
    output_parts = []
    output_token_ids = []
    usage = None
    server_metrics = None
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
                if chunk.get("metrics"):
                    server_metrics = chunk["metrics"]
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
    ended_at_epoch_s = time.time()
    ttft_ms = (
        round((first_token - started) * 1000.0, 3)
        if first_token is not None else None
    )
    total_ms = round((ended - started) * 1000.0, 3)
    generated_output_tokens = int(
        (usage or {}).get("completion_tokens") or len(output_token_ids)
    )
    tpot_ms = (
        round((last_token - first_token) * 1000.0 / (generated_output_tokens - 1), 3)
        if first_token is not None
        and last_token is not None
        and generated_output_tokens > 1
        else None
    )
    prompt_details = (usage or {}).get("prompt_tokens_details") or {}
    cached_tokens = prompt_details.get("cached_tokens")
    if not isinstance(cached_tokens, (int, float)):
        cached_tokens = None
    return {
        "index": item["index"],
        "request_id": item["request_id"],
        "provider_request_id": response_id,
        "prompt_tokens": len(item["prompt_token_ids"]),
        "output_tokens": generated_output_tokens,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "total_ms": total_ms,
        "usage": usage,
        "cached_tokens": int(cached_tokens) if cached_tokens is not None else None,
        "server_metrics": server_metrics,
        "cache_salt": cache_salt,
        "output_text": "".join(output_parts),
        "output_token_ids": output_token_ids,
        "thread_id": threading.get_ident(),
        "started_at_epoch_s": started_at_epoch_s,
        "first_token_at_epoch_s": (
            started_at_epoch_s + (first_token - started)
            if first_token is not None else None
        ),
        "ended_at_epoch_s": ended_at_epoch_s,
    }


def run_bundle(
    bundle: dict[str, Any],
    endpoint: VLLMEndpoint,
    mode: str,
    *,
    output_tokens: int = FIXED_OUTPUT_TOKENS,
    max_concurrency: int | None = None,
    cache_salt_mode: str = "none",
    cache_salt: str | None = None,
) -> list[dict[str, Any]]:
    requests = bundle.get("requests") or []
    if mode not in {"packed", "paced", "serial"}:
        raise ValueError("mode must be packed, paced, or serial")
    if output_tokens < 1:
        raise ValueError("output_tokens must be positive")
    if max_concurrency is not None and max_concurrency < 1:
        raise ValueError("max_concurrency must be positive")
    if cache_salt_mode not in CACHE_SALT_MODES:
        raise ValueError(f"cache_salt_mode must be one of {CACHE_SALT_MODES}")
    if not requests:
        return []
    salt_base = cache_salt or secrets.token_urlsafe(32)

    def item_salt(item: dict[str, Any]) -> str | None:
        if cache_salt_mode == "none":
            return None
        if cache_salt_mode == "shared":
            return salt_base
        return f"{salt_base}:{item['index']}:{item['request_id']}"

    def complete(item: dict[str, Any]) -> dict[str, Any]:
        return _completion(
            endpoint,
            bundle["served_model"],
            item,
            output_tokens,
            item_salt(item),
        )

    if mode == "serial":
        workers = 1

        def task(item: dict[str, Any]) -> dict[str, Any]:
            return complete(item)
    elif mode == "packed":
        workers = _peak_concurrency(requests)
        if max_concurrency is not None:
            workers = min(workers, max_concurrency)

        def task(item: dict[str, Any]) -> dict[str, Any]:
            return complete(item)
    else:
        workers = min(len(requests), 64)
        if max_concurrency is not None:
            workers = min(workers, max_concurrency)
        replay_started = time.perf_counter()

        def task(item: dict[str, Any]) -> dict[str, Any]:
            target = float(item.get("arrival_offset_ms", 0.0) or 0.0) / 1000.0
            delay = target - (time.perf_counter() - replay_started)
            if delay > 0:
                time.sleep(delay)
            return complete(item)

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


def percentile(values: list[float], fraction: float) -> float | None:
    """Linearly interpolated percentile for per-request measurements."""
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return round(ordered[low], 3)
    weight = position - low
    return round(ordered[low] * (1 - weight) + ordered[high] * weight, 3)


def _server_metadata(endpoint: VLLMEndpoint) -> dict[str, Any]:
    """Capture full dev-mode config when available plus stable public metadata."""
    metadata: dict[str, Any] = {}
    for key, method_name in (
        ("version", "version"),
        ("models", "models"),
        ("server_info", "server_info"),
    ):
        method = getattr(endpoint, method_name, None)
        if method is None:
            continue
        try:
            metadata[key] = method()
        except (RuntimeError, OSError, ValueError, json.JSONDecodeError):
            metadata[key] = None
    return metadata


def _process_start_time(metrics: str) -> float | None:
    for line in metrics.splitlines():
        if line.startswith("process_start_time_seconds "):
            try:
                return float(line.rsplit(" ", 1)[-1])
            except ValueError:
                return None
    return None


def _replay_root(output_dir: Path) -> Path | None:
    for candidate in (output_dir.resolve(), *output_dir.resolve().parents):
        if candidate.name == "replay":
            return candidate
    fallback = Path.cwd() / "replay"
    return fallback if fallback.is_dir() else None


def _kv_event_sequence_hint(
    output_dir: Path,
    endpoint: str,
    process_started_at: float | None,
) -> int:
    """Resume after the newest event captured from this live server process."""
    root = _replay_root(output_dir)
    if root is None or process_started_at is None:
        return 0
    last_sequence = -1
    for summary_path in root.glob("**/summary.json"):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            events = summary.get("kv_events") or {}
            if events.get("endpoint") != endpoint:
                continue
            candidate = events.get("last_sequence")
            if not isinstance(candidate, int):
                continue
            metrics_path = summary_path.with_name("metrics_before.prom")
            if not metrics_path.is_file():
                continue
            previous_start = _process_start_time(
                metrics_path.read_text(encoding="utf-8")
            )
            if previous_start == process_started_at:
                last_sequence = max(last_sequence, candidate)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return last_sequence + 1


def replay_to_dir(
    bundle: dict[str, Any],
    endpoint: VLLMEndpoint,
    mode: str,
    output_dir: Path,
    arm: dict[str, Any] | None = None,
    reset_before: bool = False,
    bundle_path: Path | None = None,
    output_tokens: int = FIXED_OUTPUT_TOKENS,
    max_concurrency: int | None = None,
    cache_salt_mode: str = "none",
    cache_salt: str | None = None,
    kv_events_replay_endpoint: str | None = None,
    warmup_repetitions: int = 0,
) -> dict[str, Any]:
    """Replay the bundle once and write requests, metrics and summary.

    The serving config comes from the server's own counters, so an arm is
    labelled by what was actually serving rather than by what the launch
    command asked for.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if warmup_repetitions < 0:
        raise ValueError("warmup_repetitions must not be negative")
    effective_cache_salt = cache_salt
    if cache_salt_mode != "none" and not effective_cache_salt:
        effective_cache_salt = secrets.token_urlsafe(32)
    if reset_before:
        endpoint.reset_prefix_cache()
    server_metadata = _server_metadata(endpoint)
    if kv_events_replay_endpoint:
        server_info = server_metadata.get("server_info") or {}
        vllm_config = server_info.get("vllm_config") or {}
        if "kv_events_config" in vllm_config and vllm_config["kv_events_config"] is None:
            raise RuntimeError(
                "the live server reports kv_events_config=null; restart vLLM "
                "with --kv-events-config before passing "
                "--kv-events-replay-endpoint"
            )
    warmup_summaries = []
    for warmup_index in range(warmup_repetitions):
        warmup_results = run_bundle(
            bundle,
            endpoint,
            mode,
            output_tokens=output_tokens,
            max_concurrency=max_concurrency,
            cache_salt_mode=cache_salt_mode,
            cache_salt=effective_cache_salt,
        )
        with (output_dir / f"warmup{warmup_index}_requests.jsonl").open(
            "w", encoding="utf-8"
        ) as output:
            for row in warmup_results:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        warmup_summaries.append({
            "requests": len(warmup_results),
            "prompt_tokens": sum(row["prompt_tokens"] for row in warmup_results),
            "cached_tokens": sum(row.get("cached_tokens") or 0 for row in warmup_results),
            "median_total_ms": _median([row["total_ms"] for row in warmup_results]),
        })
    # Reset and the final warm-up scheduler step publish asynchronously. Drain
    # them into the pre-measurement boundary before choosing the next sequence.
    if kv_events_replay_endpoint:
        time.sleep(0.25)
    next_event_sequence = 0
    if kv_events_replay_endpoint:
        from agent_io_tracing.serving.kv_events import fetch_replay
        process_started_at = _process_start_time(endpoint.metrics())
        sequence_hint = _kv_event_sequence_hint(
            output_dir,
            kv_events_replay_endpoint,
            process_started_at,
        )
        _, initial_transport = fetch_replay(
            kv_events_replay_endpoint,
            start_sequence=sequence_hint,
        )
        previous = initial_transport.get("last_sequence")
        next_event_sequence = (
            int(previous) + 1 if previous is not None else sequence_hint
        )
    metrics_before = endpoint.metrics()
    started = time.time()
    results = run_bundle(
        bundle,
        endpoint,
        mode,
        output_tokens=output_tokens,
        max_concurrency=max_concurrency,
        cache_salt_mode=cache_salt_mode,
        cache_salt=effective_cache_salt,
    )
    ended = time.time()
    metrics_after = endpoint.metrics()
    metrics_window = prometheus_window(metrics_before, metrics_after)

    # Persist the completed inference measurement before optional KV-event
    # collection. A diagnostic side channel must not discard an expensive run.
    with (output_dir / "requests.jsonl").open("w", encoding="utf-8") as out:
        for row in results:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "metrics_before.prom").write_text(metrics_before, encoding="utf-8")
    (output_dir / "metrics_after.prom").write_text(metrics_after, encoding="utf-8")
    (output_dir / "metrics_window.json").write_text(
        json.dumps(metrics_window, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "server_info.json").write_text(
        json.dumps(server_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    kv_event_summary = None
    if kv_events_replay_endpoint:
        from agent_io_tracing.serving.kv_events import (
            derive_removals,
            fetch_replay,
            summarize_events,
            write_events,
        )
        # The publisher drains an asynchronous queue. Give the final scheduler
        # batch a brief chance to reach its replay buffer.
        time.sleep(0.25)
        try:
            event_rows, event_transport = fetch_replay(
                kv_events_replay_endpoint,
                start_sequence=next_event_sequence,
            )
            write_events(output_dir / "kv_events.jsonl", event_rows)
            removal_rows = derive_removals(
                event_rows,
                bundle.get("requests") or [],
                started_at=started,
                ended_at=ended,
            )
            write_events(output_dir / "kv_removals.jsonl", removal_rows)
            kv_event_summary = {
                **event_transport,
                **summarize_events(
                    event_rows,
                    started_at=started,
                    ended_at=ended,
                ),
                "removals": len(removal_rows),
                "removals_matched_to_store": sum(
                    bool(row.get("matched_store")) for row in removal_rows
                ),
                "removals_with_request_candidates": sum(
                    bool(row.get("possible_request_prefixes")) for row in removal_rows
                ),
                "ambiguous_token_mappings": sum(
                    bool(row.get("token_mapping_ambiguous")) for row in removal_rows
                ),
            }
        except RuntimeError as exc:
            kv_event_summary = {
                "endpoint": kv_events_replay_endpoint,
                "collection_error": str(exc),
            }

    total_output_tokens = sum(row.get("output_tokens") or 0 for row in results)
    client_ttft_ms = [
        row["ttft_ms"] for row in results
        if isinstance(row.get("ttft_ms"), (int, float))
    ]
    cache_rows = [
        row for row in results
        if isinstance(row.get("cached_tokens"), (int, float))
    ]
    cached_values = [int(row["cached_tokens"]) for row in cache_rows]
    prefix_window = prefix_cache_delta(metrics_before, metrics_after)

    def server_median(key: str) -> float | None:
        values = [
            (row.get("server_metrics") or {}).get(key) for row in results
        ]
        return _median([v for v in values if isinstance(v, (int, float))])

    summary = {
        "bundle": str((bundle_path or Path(bundle.get("source_cell", ""))).resolve()),
        "bundle_requests": len(bundle.get("requests") or []),
        "mode": mode,
        "max_concurrency": max_concurrency,
        "warmup_repetitions": warmup_repetitions,
        "warmup": warmup_summaries,
        "cache_salt_mode": cache_salt_mode,
        "cache_salt": effective_cache_salt,
        # What the prefix cache held when this replay started.  A warm arm
        # measures the previous arm as much as its own, so the report must
        # never average the two together.
        "cache_state": _cache_state(reset_before, metrics_before),
        "arm": arm or json.loads(os.environ.get("KVCACHE_ARM_JSON", "{}")),
        "serving_config": parse_cache_config(metrics_before),
        "server_metadata": {
            "version": server_metadata.get("version"),
            "models": server_metadata.get("models"),
            "full_config_captured": bool(server_metadata.get("server_info")),
        },
        "prefix_cache": prefix_window,
        "prometheus_window": metrics_window,
        "kv_events": kv_event_summary,
        "started_at_epoch_s": started,
        "ended_at_epoch_s": ended,
        "wall_s": round(ended - started, 3),
        "requests": len(results),
        "prompt_tokens": sum(row["prompt_tokens"] for row in results),
        "output_tokens": total_output_tokens,
        "fixed_output_tokens_per_request": output_tokens,
        "per_request_cache": {
            "reported_requests": len(cached_values),
            "coverage": round(len(cached_values) / len(results), 4) if results else None,
            "cached_tokens": int(sum(cached_values)),
            "hit_rate": (
                round(sum(cached_values) / sum(row["prompt_tokens"] for row in cache_rows), 4)
                if cache_rows and sum(row["prompt_tokens"] for row in cache_rows) else None
            ),
            "matches_prometheus": (
                len(cache_rows) == len(results)
                and prefix_window is not None
                and sum(cached_values) == prefix_window.get("hits")
                and sum(row["prompt_tokens"] for row in results)
                == prefix_window.get("queries")
            ),
        },
        "server_timing": {
            "reported_requests": sum(bool(row.get("server_metrics")) for row in results),
            "median_ttft_ms": server_median("time_to_first_token_ms"),
            "median_queue_ms": server_median("queue_time_ms"),
            "median_generation_ms": server_median("generation_time_ms"),
            "median_mean_itl_ms": server_median("mean_itl_ms"),
            "median_tokens_per_second": server_median("tokens_per_second"),
        },
        "median_ttft_ms": _median(client_ttft_ms),
        "p95_ttft_ms": percentile(client_ttft_ms, 0.95),
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
    parser.add_argument("--mode", choices=("packed", "paced", "serial"), default="packed")
    parser.add_argument("--output-tokens", type=int, default=FIXED_OUTPUT_TOKENS)
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--warmup-repetitions", type=int, default=0)
    parser.add_argument(
        "--cache-salt-mode",
        choices=CACHE_SALT_MODES,
        default="none",
        help="none: global namespace; shared: one isolated namespace; unique: disable cross-request reuse.",
    )
    parser.add_argument(
        "--cache-salt",
        help="Salt base. A random value is generated when a salted mode omits it.",
    )
    parser.add_argument(
        "--kv-events-replay-endpoint",
        default=os.environ.get("VLLM_KV_EVENTS_REPLAY_ENDPOINT"),
        help="ZMQ replay endpoint, for example tcp://127.0.0.1:15558.",
    )
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
            output_tokens=args.output_tokens,
            max_concurrency=args.max_concurrency,
            cache_salt_mode=args.cache_salt_mode,
            cache_salt=args.cache_salt,
            kv_events_replay_endpoint=args.kv_events_replay_endpoint,
            warmup_repetitions=args.warmup_repetitions,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"replay failed: {exc}") from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
