"""Inspect an already-running OpenAI-compatible vLLM endpoint."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from agent_io_tracing.adapters.llm_trace import (
    PREFIX_CACHE_HITS,
    PREFIX_CACHE_QUERIES,
)

CACHE_CONFIG_METRIC = "vllm:cache_config_info{"

OBSERVABILITY_METRIC_FRAGMENTS = (
    "prefix_cache",
    "prompt_tokens_cached",
    "prompt_tokens",
    "kv_cache_usage",
    "kv_block_",
    "num_preemptions",
    "num_requests_",
    "iteration_tokens",
    "time_to_first_token",
    "inter_token_latency",
    "e2e_request_latency",
    "request_queue_time",
    "request_prefill_time",
    "request_decode_time",
    "request_inference_time",
    "request_time_per_output_token",
    "request_prefill_kv_computed_tokens",
)


def _base_url(value: str) -> str:
    url = value.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


@dataclass(frozen=True)
class VLLMEndpoint:
    """Read and explicitly reset a remotely managed vLLM server."""

    base_url: str
    api_key: str = ""
    timeout_s: float = 10.0

    def _request(
        self,
        path: str,
        *,
        method: str = "GET",
        accept_json: bool = True,
    ) -> Any:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{_base_url(self.base_url)}{path}",
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8", "replace")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"{method} {path} returned HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, OSError) as exc:
            reason = getattr(exc, "reason", None) or str(exc)
            raise RuntimeError(
                f"{method} {path} could not reach {self.base_url}: {reason}"
            ) from exc
        if not accept_json:
            return body
        return json.loads(body) if body.strip() else {}

    def health(self) -> None:
        self._request("/health", accept_json=False)

    def models(self) -> list[str]:
        payload = self._request("/v1/models")
        return [
            str(item["id"])
            for item in payload.get("data", [])
            if isinstance(item, dict) and item.get("id")
        ]

    def metrics(self) -> str:
        return str(self._request("/metrics", accept_json=False))

    def version(self) -> Any:
        return self._request("/version")

    def server_info(self) -> dict[str, Any]:
        payload = self._request("/server_info?config_format=json")
        if not isinstance(payload, dict):
            raise RuntimeError("/server_info did not return a JSON object")
        return payload

    def reset_prefix_cache(self) -> Any:
        return self._request("/reset_prefix_cache", method="POST")

    def post_json(self, path: str, payload: dict[str, Any]) -> Any:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{_base_url(self.base_url)}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                body = response.read().decode("utf-8", "replace")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"POST {path} returned HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, OSError) as exc:
            reason = getattr(exc, "reason", None) or str(exc)
            raise RuntimeError(
                f"POST {path} could not reach {self.base_url}: {reason}"
            ) from exc
        return json.loads(body) if body.strip() else {}

    def tokenize_chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **template_options: Any,
    ) -> list[int]:
        payload = {
            "model": model,
            "messages": messages,
            "return_token_strs": False,
            **template_options,
        }
        response = self.post_json("/tokenize", payload)
        tokens = response.get("tokens")
        if not isinstance(tokens, list) or not all(
            isinstance(token, int) for token in tokens
        ):
            raise RuntimeError("/tokenize did not return integer token IDs")
        return tokens


def parse_prefix_cache(text: str) -> dict[str, int]:
    """Prefix-cache counters out of one /metrics scrape, in tokens."""
    counters: dict[str, int] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        for key, metric in (
            ("queries", PREFIX_CACHE_QUERIES),
            ("hits", PREFIX_CACHE_HITS),
        ):
            if line.startswith(metric):
                counters[key] = int(float(line.rsplit(" ", 1)[-1]))
    return counters


def parse_cache_config(text: str) -> dict[str, str]:
    """The engine's own cache_config_info labels: block_size, cache_dtype, ...

    A sweep arm is defined by what the server is actually serving, not by what
    the launch command asked for, so every replay records this.
    """
    for line in text.splitlines():
        if line.startswith(CACHE_CONFIG_METRIC):
            inner = line[line.index("{") + 1:line.rindex("}")]
            return dict(re.findall(r'([A-Za-z0-9_]+)="([^"]*)"', inner))
    return {}


def prefix_cache_delta(before: str, after: str) -> dict[str, Any] | None:
    """Realized reuse over one replay, from two /metrics scrapes."""
    start = parse_prefix_cache(before)
    end = parse_prefix_cache(after)
    if "queries" not in start or "queries" not in end:
        return None
    queries = end["queries"] - start["queries"]
    hits = end.get("hits", 0) - start.get("hits", 0)
    if queries <= 0 or hits < 0:
        return None
    return {
        "unit": "tokens",
        "queries": queries,
        "hits": hits,
        "hit_rate": round(hits / queries, 4),
    }


def _prometheus_totals(text: str) -> dict[str, float]:
    """Aggregate vLLM samples across labels for one metrics scrape."""
    totals: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        sample, _, raw_value = line.rpartition(" ")
        name = sample.split("{", 1)[0]
        # Histogram buckets are cumulative. Summing them across ``le`` labels
        # produces a number with no statistical meaning, so preserve them
        # separately in _prometheus_histogram_buckets.
        if name.endswith("_bucket"):
            continue
        if not name.startswith("vllm:"):
            continue
        if not any(fragment in name for fragment in OBSERVABILITY_METRIC_FRAGMENTS):
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        totals[name] = totals.get(name, 0.0) + value
    return totals


def _prometheus_histogram_buckets(
    text: str,
) -> dict[str, dict[float, float]]:
    """Aggregate cumulative histogram buckets across non-``le`` labels."""
    histograms: dict[str, dict[float, float]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        sample, _, raw_value = line.rpartition(" ")
        name = sample.split("{", 1)[0]
        if not name.startswith("vllm:") or not name.endswith("_bucket"):
            continue
        if not any(fragment in name for fragment in OBSERVABILITY_METRIC_FRAGMENTS):
            continue
        labels = dict(re.findall(r'([A-Za-z0-9_]+)="([^"]*)"', sample))
        raw_upper = labels.get("le")
        if raw_upper is None:
            continue
        try:
            upper = float(raw_upper)
            value = float(raw_value)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        base = name[:-7]
        buckets = histograms.setdefault(base, {})
        buckets[upper] = buckets.get(upper, 0.0) + value
    return histograms


def _histogram_quantile_details(
    cumulative_buckets: dict[float, float],
    fraction: float,
) -> tuple[float | None, float | None, float | None]:
    """Quantile estimate plus the bucket interval containing it."""
    if not cumulative_buckets:
        return None, None, None
    ordered = sorted(cumulative_buckets.items())
    total = cumulative_buckets.get(math.inf, ordered[-1][1])
    if total <= 0:
        return None, None, None
    rank = fraction * total
    previous_upper = 0.0
    previous_count = 0.0
    for upper, count in ordered:
        if count < rank:
            if math.isfinite(upper):
                previous_upper = upper
            previous_count = count
            continue
        if not math.isfinite(upper):
            return previous_upper, previous_upper, None
        bucket_count = count - previous_count
        if bucket_count <= 0:
            return upper, previous_upper, upper
        position = (rank - previous_count) / bucket_count
        return (
            previous_upper + position * (upper - previous_upper),
            previous_upper,
            upper,
        )
    return None, None, None


def prometheus_window(before: str, after: str) -> dict[str, Any]:
    """Summarize the observable vLLM metric changes over one replay window."""
    start = _prometheus_totals(before)
    end = _prometheus_totals(after)
    metrics: dict[str, dict[str, float | None]] = {}
    for name in sorted(start.keys() | end.keys()):
        first = start.get(name, 0.0)
        last = end.get(name, 0.0)
        delta = last - first
        metrics[name] = {
            "before": first,
            "after": last,
            "delta": delta if delta >= 0 else None,
        }

    means: dict[str, dict[str, float]] = {}
    for sum_name, values in metrics.items():
        if not sum_name.endswith("_sum"):
            continue
        base = sum_name[:-4]
        count = metrics.get(f"{base}_count")
        total_delta = values.get("delta")
        count_delta = (count or {}).get("delta")
        if isinstance(total_delta, (int, float)) and isinstance(count_delta, (int, float)) and count_delta > 0:
            means[base] = {
                "observations": count_delta,
                "window_mean": total_delta / count_delta,
            }

    start_buckets = _prometheus_histogram_buckets(before)
    end_buckets = _prometheus_histogram_buckets(after)
    quantiles: dict[str, dict[str, float]] = {}
    for base in sorted(start_buckets.keys() | end_buckets.keys()):
        first = start_buckets.get(base, {})
        last = end_buckets.get(base, {})
        window_buckets = {
            upper: last.get(upper, 0.0) - first.get(upper, 0.0)
            for upper in first.keys() | last.keys()
        }
        if any(value < 0 for value in window_buckets.values()):
            continue
        count = (metrics.get(f"{base}_count") or {}).get("delta")
        if not isinstance(count, (int, float)) or count <= 0:
            continue
        row: dict[str, float] = {"observations": count}
        for label, fraction in (("p50", 0.50), ("p95", 0.95)):
            value, lower, upper = _histogram_quantile_details(
                window_buckets, fraction
            )
            if value is not None:
                row[label] = value
            if lower is not None:
                row[f"{label}_bucket_lower"] = lower
            if upper is not None:
                row[f"{label}_bucket_upper"] = upper
        quantiles[base] = row
    return {
        "metrics": metrics,
        "histogram_window_means": means,
        "histogram_window_quantiles": quantiles,
    }


def _metric_names(text: str) -> list[str]:
    names = set()
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(None, 1)[0]
        if any(fragment in name for fragment in OBSERVABILITY_METRIC_FRAGMENTS):
            names.add(name)
    return sorted(names)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check a remotely managed vLLM endpoint.",
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
        default=(
            os.environ.get("VLLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY_1")
            or ""
        ),
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the prefix cache after the read-only checks.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.url:
        raise SystemExit(
            "Set VLLM_URL or OPENAI_BASE_URL, or pass --url."
        )
    endpoint = VLLMEndpoint(args.url, args.api_key, args.timeout)
    try:
        endpoint.health()
        models = endpoint.models()
        version = endpoint.version()
        try:
            server_info = endpoint.server_info()
        except RuntimeError:
            server_info = None
        metrics = _metric_names(endpoint.metrics())
        reset_result = endpoint.reset_prefix_cache() if args.reset else None
    except (RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"vLLM endpoint check failed: {exc}") from exc
    print(json.dumps({
        "base_url": _base_url(args.url),
        "health": "ok",
        "models": models,
        "version": version,
        "server_info": server_info,
        "kv_metric_names": metrics,
        "prefix_cache_reset": reset_result if args.reset else "not requested",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
