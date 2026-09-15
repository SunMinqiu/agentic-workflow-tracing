#!/usr/bin/env python3
"""Replay one bundle under several server configurations and compare them.

A knob sweep needs the workload held fixed. Re-running the agent does not hold
it fixed: the agent's decisions change, so the call count, the prompts and the
tool sequence all differ between arms, and any latency difference mixes the
knob with the workload. Replaying one bundle sends byte-identical requests to
every arm, so the knob is the only thing that changed.

The knobs that matter here (``block_size``, ``cache_dtype``,
``gpu_memory_utilization``, ``enable_prefix_caching``) are engine startup
flags, so an arm means "the server is currently up with that config". This
tool cannot restart the server; it records what the server reports about
itself, so a mislabelled arm is caught rather than believed.

    arm     replay the bundle once against whatever is serving now
    report  tabulate every arm recorded so far
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any

from agent_io_tracing.experiments.timezone import (
    EXPERIMENT_TIMEZONE,
    from_epoch,
    now,
)
from agent_io_tracing.replay.runner import percentile, replay_to_dir
from agent_io_tracing.serving.kv_events import derive_removals, summarize_events
from agent_io_tracing.serving.vllm_endpoint import (
    parse_cache_config,
    prometheus_window,
    VLLMEndpoint,
)

WARMING_SPREAD = 0.05
EVICTION_TIMELINE_BIN_S = 10.0

# Startup flags a sweep is likely to move. Any other key that differs between
# arms is reported too; this list only fixes the column order.
KNOBS = (
    "block_size",
    "cache_dtype",
    "enable_prefix_caching",
    "gpu_memory_utilization",
    "num_gpu_blocks_override",
    "swap_space",
)

DERIVED_CONFIG_CHANGES = {
    "enable_prefix_caching": {
        "mamba_cache_mode",
        "mamba_page_size_padded",
        "skip_page_size_padded",
        "num_gpu_blocks",
        "kv_cache_size_tokens",
        "kv_cache_max_concurrency",
    },
    "num_gpu_blocks_override": {
        "num_gpu_blocks",
        "kv_cache_size_tokens",
        "kv_cache_max_concurrency",
    },
    "kv_cache_memory_bytes": {
        "num_gpu_blocks",
        "kv_cache_size_tokens",
        "kv_cache_max_concurrency",
    },
    "gpu_memory_utilization": {
        "num_gpu_blocks",
        "kv_cache_size_tokens",
        "kv_cache_max_concurrency",
    },
}

VOLATILE_SERVER_CONFIG_PATHS = {
    "system_env.env_vars.VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME",
    "vllm_config.instance_id",
    "vllm_env.VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME",
}


def _flatten_config(value: Any, prefix: str = "") -> dict[str, str]:
    """Flatten JSON config so strict sweeps can name the exact changed fields."""
    if prefix == "system_env.env_vars" and isinstance(value, str):
        flattened: dict[str, str] = {}
        for line in value.splitlines():
            key, separator, child = line.partition("=")
            if separator:
                flattened[f"{prefix}.{key}"] = json.dumps(child)
        return flattened
    if isinstance(value, dict):
        flattened: dict[str, str] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_config(child, path))
        return flattened
    return {prefix: json.dumps(value, sort_keys=True, default=str)}


def _server_config_changes(old: Any, new: Any) -> set[str]:
    old_flat = _flatten_config(old)
    new_flat = _flatten_config(new)
    return {
        path for path in old_flat.keys() | new_flat.keys()
        if old_flat.get(path) != new_flat.get(path)
        and path not in VOLATILE_SERVER_CONFIG_PATHS
    }


def _allowed_config_path(path: str, allowed: set[str]) -> bool:
    parts = set(path.split("."))
    return any(name == path or name in parts for name in allowed)


def automatic_label(endpoint: VLLMEndpoint) -> str:
    """Build a unique arm name from the live config and New York time."""
    config = parse_cache_config(endpoint.metrics())
    prefix = str(config.get("enable_prefix_caching", "unknown")).lower()
    prefix = {"true": "on", "false": "off"}.get(prefix, prefix)
    values = (
        f"block{config.get('block_size', 'unknown')}",
        f"dtype-{config.get('cache_dtype', 'unknown')}",
        f"prefix-{prefix}",
        f"gpu{config.get('gpu_memory_utilization', 'unknown')}",
        now().strftime("%Y%m%dT%H%M%S"),
    )
    return "_".join(re.sub(r"[^A-Za-z0-9._-]+", "-", value) for value in values)


def record_arm(
    bundle_path: Path,
    sweep_dir: Path,
    label: str,
    endpoint: VLLMEndpoint,
    mode: str = "packed",
    repeat: int = 1,
    reset_before: bool = True,
    output_tokens: int = 32,
    max_concurrency: int | None = None,
    cache_salt_mode: str = "none",
    cache_salt: str | None = None,
    kv_events_replay_endpoint: str | None = None,
    warmup_repetitions: int = 0,
    strict_config: bool = False,
    allowed_config_changes: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Replay the bundle ``repeat`` times into ``sweep_dir/label/rep<i>``.

    The prefix cache is reset before each repetition by default: an arm that
    inherits the previous arm's cache measures the previous arm.
    """
    arm_dir = sweep_dir / label
    if arm_dir.exists():
        raise ValueError(f"arm output already exists: {arm_dir}")
    if repeat < 1:
        raise ValueError("repeat must be positive")
    if not reset_before and repeat != 1:
        raise ValueError("--keep-cache requires --repeat 1")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if strict_config:
        current_config = parse_cache_config(endpoint.metrics())
        baseline_paths = sorted(sweep_dir.glob("*/rep0/summary.json"))
        if baseline_paths:
            baseline = json.loads(baseline_paths[0].read_text(encoding="utf-8"))
            baseline_config = baseline.get("serving_config") or {}
            changed = {
                key for key in baseline_config.keys() | current_config.keys()
                if baseline_config.get(key) != current_config.get(key)
            }
            allowed = set(allowed_config_changes or ())
            for knob in tuple(allowed):
                allowed.update(DERIVED_CONFIG_CHANGES.get(knob, set()))
            unexpected = sorted(changed - allowed)
            baseline_server_path = baseline_paths[0].with_name("server_info.json")
            server_info_method = getattr(endpoint, "server_info", None)
            if baseline_server_path.is_file() and server_info_method is not None:
                baseline_server = json.loads(
                    baseline_server_path.read_text(encoding="utf-8")
                ).get("server_info")
                try:
                    current_server = server_info_method()
                except RuntimeError:
                    current_server = None
                if baseline_server and current_server:
                    server_changed = _server_config_changes(
                        baseline_server, current_server
                    )
                    unexpected.extend(sorted(
                        path for path in server_changed
                        if not _allowed_config_path(path, allowed)
                    ))
            if unexpected:
                raise ValueError(
                    "serving config changed outside the allowlist: "
                    + ", ".join(sorted(set(unexpected)))
                )
    if reset_before:
        # Fail here rather than after minutes of decoding: vLLM registers
        # /reset_prefix_cache only under VLLM_SERVER_DEV_MODE=1.
        try:
            endpoint.reset_prefix_cache()
        except RuntimeError as exc:
            raise RuntimeError(
                "this server does not expose /reset_prefix_cache; vLLM only "
                "registers it when launched with VLLM_SERVER_DEV_MODE=1. "
                "Relaunch it with that variable, or pass --keep-cache to "
                "accept a warm start. Changing a knob restarts the engine and "
                "empties the cache on its own, so the first repetition of each "
                "arm is cold either way -- but later repetitions are not, and "
                f"the report marks them warm. Underlying error: {exc}"
            ) from exc
    summaries = []
    for index in range(repeat):
        print(
            f"arm: {label}; repetition {index + 1}/{repeat}",
            file=sys.stderr,
            flush=True,
        )
        summary = replay_to_dir(
            json.loads(json.dumps(bundle)),
            endpoint,
            mode,
            sweep_dir / label / f"rep{index}",
            arm={"label": label, "repetition": index},
            reset_before=reset_before,
            bundle_path=bundle_path,
            output_tokens=output_tokens,
            max_concurrency=max_concurrency,
            cache_salt_mode=cache_salt_mode,
            cache_salt=cache_salt,
            kv_events_replay_endpoint=kv_events_replay_endpoint,
            warmup_repetitions=warmup_repetitions,
        )
        summaries.append(summary)
    return summaries


def load_repetition_summary(path: Path) -> dict[str, Any]:
    """Load one repetition and backfill analyses from its raw artifacts."""
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("cache_state") == "cold_by_reset":
        summary["cache_state"] = "reset_requested"
    requests_path = path.parent / "requests.jsonl"
    request_rows = _read_jsonl(requests_path)
    summary["_request_rows"] = request_rows
    if not isinstance(summary.get("p95_ttft_ms"), (int, float)):
        ttft_ms = [
            row["ttft_ms"] for row in request_rows
            if isinstance(row.get("ttft_ms"), (int, float))
        ]
        summary["p95_ttft_ms"] = percentile(ttft_ms, 0.95)

    before_path = path.parent / "metrics_before.prom"
    after_path = path.parent / "metrics_after.prom"
    if before_path.is_file() and after_path.is_file():
        summary["prometheus_window"] = prometheus_window(
            before_path.read_text(encoding="utf-8"),
            after_path.read_text(encoding="utf-8"),
        )
    server_info_path = path.parent / "server_info.json"
    if server_info_path.is_file():
        try:
            server_info = json.loads(server_info_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            server_info = {}
        summary["_kv_cache_metrics_sample"] = (
            (((server_info.get("server_info") or {}).get("vllm_config") or {}).get(
                "observability_config"
            ) or {}).get("kv_cache_metrics_sample")
        )

    from agent_io_tracing.serving.kv_report_data import load_report_data

    report_data = load_report_data(path.parent)
    event_rows = [] if report_data else _read_jsonl(path.parent / "kv_events.jsonl")
    removal_rows: list[dict[str, Any]] = []
    if report_data:
        prior = summary.get("kv_events") or {}
        summary["kv_events"] = {
            "buffer_truncated": prior.get("buffer_truncated"),
            "collection_error": prior.get("collection_error"),
            "collection_status": report_data["collection_status"],
            **report_data["events"],
            "lifecycle": report_data["lifecycle"],
        }
        if report_data["logical_events_available"]:
            removal_rows = _read_jsonl(path.parent / "kv_removals.jsonl")
    elif event_rows:
        started_at = summary.get("started_at_epoch_s")
        ended_at = summary.get("ended_at_epoch_s")
        event_analysis = summarize_events(
            event_rows,
            started_at=started_at,
            ended_at=ended_at,
        )
        bundle = {}
        bundle_path = Path(str(summary.get("bundle") or ""))
        if bundle_path.is_file():
            try:
                bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                bundle = {}
        removal_rows = derive_removals(
            event_rows,
            bundle.get("requests") or [],
            started_at=started_at,
            ended_at=ended_at,
        )
        from agent_io_tracing.serving.kv_lifecycle import derive_lifecycle

        _, lifecycle_summary = derive_lifecycle(
            event_rows, started_at=started_at, ended_at=ended_at,
        )
        prior = summary.get("kv_events") or {}
        summary["kv_events"] = {
            **prior,
            **event_analysis,
            "lifecycle": lifecycle_summary,
            "removals": len(removal_rows),
            "removals_matched_to_store": sum(
                bool(row.get("matched_store")) for row in removal_rows
            ),
            "removals_with_request_candidates": sum(
                bool(row.get("content_match_candidates")) for row in removal_rows
            ),
            "ambiguous_token_mappings": sum(
                bool(row.get("token_mapping_ambiguous")) for row in removal_rows
            ),
        }
    summary["_removal_rows"] = removal_rows
    summary["eviction"] = _eviction_analysis(summary, removal_rows)
    return summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _metric_delta(window: dict[str, Any], name: str) -> float | None:
    value = ((window.get("metrics") or {}).get(name) or {}).get("delta")
    return value if isinstance(value, (int, float)) else None


def _eviction_analysis(
    summary: dict[str, Any],
    removals: list[dict[str, Any]],
) -> dict[str, Any]:
    window = summary.get("prometheus_window") or {}
    means = window.get("histogram_window_means") or {}
    quantiles = window.get("histogram_window_quantiles") or {}
    idle_name = "vllm:kv_block_idle_before_evict_seconds"
    lifetime_name = "vllm:kv_block_lifetime_seconds"
    idle = means.get(idle_name) or {}
    lifetime = means.get(lifetime_name) or {}
    idle_q = quantiles.get(idle_name) or {}
    lifetime_q = quantiles.get(lifetime_name) or {}
    events = summary.get("kv_events") or {}
    groups: dict[str, dict[str, Any]] = {}
    for row in removals:
        group = str(row.get("group_idx"))
        target = groups.setdefault(group, {
            "removed_hashes": 0,
            "matched_stores": 0,
            "content_candidate_matches": 0,
            "ambiguous_mappings": 0,
            "kinds": {},
            "block_sizes": {},
            "time_since_last_store_s": [],
        })
        target["removed_hashes"] += 1
        target["matched_stores"] += int(bool(row.get("matched_store")))
        target["content_candidate_matches"] += int(bool(row.get("content_match_candidates")))
        target["ambiguous_mappings"] += int(bool(row.get("token_mapping_ambiguous")))
        kind = str(row.get("kv_cache_spec_kind") or "unknown")
        target["kinds"][kind] = target["kinds"].get(kind, 0) + 1
        size = str(row.get("block_size") or "unknown")
        target["block_sizes"][size] = target["block_sizes"].get(size, 0) + 1
        observed_idle = row.get("time_since_last_store_s")
        if isinstance(observed_idle, (int, float)):
            target["time_since_last_store_s"].append(observed_idle)
    for target in groups.values():
        idle_values = target.pop("time_since_last_store_s")
        target["median_time_since_last_store_s"] = (
            round(statistics.median(idle_values), 3) if idle_values else None
        )
        target["p95_time_since_last_store_s"] = percentile(idle_values, 0.95)
    physical = idle.get("observations")
    prompt_tokens = summary.get("prompt_tokens")
    return {
        "eviction_histogram_observations": physical,
        "metrics_sample": summary.get("_kv_cache_metrics_sample"),
        "eviction_histogram_observations_per_million_input": (
            physical / prompt_tokens * 1_000_000
            if isinstance(physical, (int, float))
            and isinstance(prompt_tokens, (int, float)) and prompt_tokens > 0
            else None
        ),
        "removed_cached_hashes": (events.get("block_hashes") or {}).get("removed"),
        "mean_idle_before_evict_s": idle.get("window_mean"),
        "p50_idle_before_evict_s": idle_q.get("p50"),
        "p95_idle_before_evict_s": idle_q.get("p95"),
        "p95_idle_bucket_lower_s": idle_q.get("p95_bucket_lower"),
        "p95_idle_bucket_upper_s": idle_q.get("p95_bucket_upper"),
        "mean_block_lifetime_s": lifetime.get("window_mean"),
        "p50_block_lifetime_s": lifetime_q.get("p50"),
        "p95_block_lifetime_s": lifetime_q.get("p95"),
        "p95_lifetime_bucket_lower_s": lifetime_q.get("p95_bucket_lower"),
        "p95_lifetime_bucket_upper_s": lifetime_q.get("p95_bucket_upper"),
        "preemptions": _metric_delta(window, "vllm:num_preemptions_total"),
        "clears_during_measurement": (events.get("event_types") or {}).get(
            "AllBlocksCleared", 0
        ) if "event_types" in events else None,
        "collection_status": events.get("collection_status", "unconfirmed"),
        "event_buffer_truncated": bool(events.get("buffer_truncated")),
        "event_collection_error": events.get("collection_error"),
        "removals": len(removals),
        "matched_removals": sum(bool(row.get("matched_store")) for row in removals),
        "mapped_removals": sum(
            bool(row.get("content_match_candidates")) for row in removals
        ),
        "ambiguous_removals": sum(
            bool(row.get("token_mapping_ambiguous")) for row in removals
        ),
        "by_group": groups,
    }


def load_arms(sweep_dir: Path) -> list[dict[str, Any]]:
    """One row per arm, medians taken across its repetitions."""
    from agent_io_tracing.serving.kv_report_data import read_json

    arms: list[dict[str, Any]] = []
    for arm_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        reps = []
        for rep_dir in sorted(arm_dir.glob("rep*")):
            if read_json(rep_dir / "kv_report_metadata.json").get("exclude_from_comparison"):
                continue
            path = rep_dir / "summary.json"
            if path.is_file():
                reps.append(load_repetition_summary(path))
        if not reps:
            continue

        def med(key: str, source: list[dict[str, Any]] = reps) -> float | None:
            values = [r.get(key) for r in source]
            values = [v for v in values if isinstance(v, (int, float))]
            return round(statistics.median(values), 3) if values else None

        eviction_reps = [r.get("eviction") or {} for r in reps]

        def eviction_med(key: str) -> float | None:
            return med(key, eviction_reps)

        group_ids = {
            group
            for eviction in eviction_reps
            for group in (eviction.get("by_group") or {})
        }
        eviction_groups = {}
        for group in sorted(group_ids):
            group_rows = [
                (eviction.get("by_group") or {}).get(group) or {}
                for eviction in eviction_reps
            ]
            kinds: dict[str, int] = {}
            sizes: dict[str, int] = {}
            for group_row in group_rows:
                for kind, count in (group_row.get("kinds") or {}).items():
                    kinds[kind] = kinds.get(kind, 0) + count
                for size, count in (group_row.get("block_sizes") or {}).items():
                    sizes[size] = sizes.get(size, 0) + count
            eviction_groups[group] = {
                "kind": max(kinds, key=kinds.get) if kinds else "unknown",
                "block_size": max(sizes, key=sizes.get) if sizes else "unknown",
                "removed_hashes": med("removed_hashes", group_rows),
                "matched_stores": med("matched_stores", group_rows),
                "content_candidate_matches": med("content_candidate_matches", group_rows),
                "ambiguous_mappings": med("ambiguous_mappings", group_rows),
                "median_time_since_last_store_s": med(
                    "median_time_since_last_store_s", group_rows
                ),
                "p95_time_since_last_store_s": med("p95_time_since_last_store_s", group_rows),
            }

        request_observations: dict[str, list[dict[str, Any]]] = {}
        for rep in reps:
            for request in rep.get("_request_rows") or []:
                request_id = str(request.get("request_id"))
                request_observations.setdefault(request_id, []).append(request)
        per_request = {}
        for request_id, observations in request_observations.items():
            def request_med(key: str) -> float | None:
                values = [
                    row.get(key) for row in observations
                    if isinstance(row.get(key), (int, float))
                ]
                return statistics.median(values) if values else None

            per_request[request_id] = {
                "cached_tokens": request_med("cached_tokens"),
                "ttft_ms": request_med("ttft_ms"),
                "prompt_tokens": request_med("prompt_tokens"),
            }

        # Kept in repetition order, not sorted: the question is whether the
        # later passes inherited what the earlier ones cached.
        hit_rates = [
            (r.get("prefix_cache") or {}).get("hit_rate")
            for r in reps
        ]
        hit_rates = [v for v in hit_rates if isinstance(v, (int, float))]
        query_counts = [
            (r.get("prefix_cache") or {}).get("queries") for r in reps
        ]
        query_counts = [v for v in query_counts if isinstance(v, (int, float))]
        hit_counts = [
            (r.get("prefix_cache") or {}).get("hits") for r in reps
        ]
        hit_counts = [v for v in hit_counts if isinstance(v, (int, float))]
        # A Prometheus window that counted noticeably more queries than this
        # arm itself sent means another client shared the server during the
        # window (e.g. arms replayed concurrently); its delta then measures
        # the mixed traffic, not this arm.
        window_shared = any(
            isinstance((r.get("prefix_cache") or {}).get("queries"), (int, float))
            and isinstance(r.get("prompt_tokens"), (int, float))
            and r["prompt_tokens"] > 0
            and (r.get("prefix_cache") or {}).get("queries")
            > r["prompt_tokens"] * 1.02
            for r in reps
        )
        per_request_rates = [
            (r.get("per_request_cache") or {}).get("hit_rate") for r in reps
        ]
        per_request_rates = [
            v for v in per_request_rates if isinstance(v, (int, float))
        ]
        arms.append({
            "label": arm_dir.name,
            "recorded_at": reps[0].get("started_at_epoch_s") or 0.0,
            "reps": len(reps),
            "bundle": Path(reps[0].get("bundle", "")).name,
            "mode": reps[0].get("mode"),
            "max_concurrency": reps[0].get("max_concurrency"),
            "warmup_repetitions": reps[0].get("warmup_repetitions", 0),
            "cache_salt_mode": reps[0].get("cache_salt_mode", "none"),
            "requests": reps[0].get("requests"),
            "prompt_tokens": reps[0].get("prompt_tokens"),
            "output_tokens": med("output_tokens"),
            "fixed_output_tokens_per_request": reps[0].get(
                "fixed_output_tokens_per_request"
            ),
            "serving_config": reps[0].get("serving_config") or {},
            "cache_state": ",".join(sorted(
                {str(r.get("cache_state")) for r in reps}
            )),
            "hit_rate": (
                round(statistics.median(hit_rates), 4) if hit_rates else None
            ),
            "hit_rates": hit_rates,
            "window_shared": window_shared,
            "per_request_hit_rate": (
                round(statistics.median(per_request_rates), 4)
                if per_request_rates else None
            ),
            "cache_queries": round(statistics.median(query_counts)) if query_counts else None,
            "cache_hits": round(statistics.median(hit_counts)) if hit_counts else None,
            "median_ttft_ms": med("median_ttft_ms"),
            "p95_ttft_ms": med("p95_ttft_ms"),
            "median_tpot_ms": med("median_tpot_ms"),
            "median_total_ms": med("median_total_ms"),
            "median_server_ttft_ms": med(
                "median_ttft_ms",
                [r.get("server_timing") or {} for r in reps],
            ),
            "median_server_queue_ms": med(
                "median_queue_ms",
                [r.get("server_timing") or {} for r in reps],
            ),
            "median_server_itl_ms": med(
                "median_mean_itl_ms",
                [r.get("server_timing") or {} for r in reps],
            ),
            "per_request_cache_coverage": med(
                "coverage",
                [r.get("per_request_cache") or {} for r in reps],
            ),
            "kv_events": [r.get("kv_events") for r in reps],
            "eviction": {
                "eviction_histogram_observations": eviction_med(
                    "eviction_histogram_observations"
                ),
                "metrics_sample": eviction_med("metrics_sample"),
                "eviction_histogram_observations_per_million_input": eviction_med(
                    "eviction_histogram_observations_per_million_input"
                ),
                "removed_cached_hashes": eviction_med("removed_cached_hashes"),
                "mean_idle_before_evict_s": eviction_med(
                    "mean_idle_before_evict_s"
                ),
                "p95_idle_before_evict_s": eviction_med(
                    "p95_idle_before_evict_s"
                ),
                "p95_idle_bucket_lower_s": eviction_med(
                    "p95_idle_bucket_lower_s"
                ),
                "p95_idle_bucket_upper_s": eviction_med(
                    "p95_idle_bucket_upper_s"
                ),
                "mean_block_lifetime_s": eviction_med("mean_block_lifetime_s"),
                "p95_block_lifetime_s": eviction_med("p95_block_lifetime_s"),
                "p95_lifetime_bucket_lower_s": eviction_med(
                    "p95_lifetime_bucket_lower_s"
                ),
                "p95_lifetime_bucket_upper_s": eviction_med(
                    "p95_lifetime_bucket_upper_s"
                ),
                "preemptions": eviction_med("preemptions"),
                "clears_during_measurement": eviction_med(
                    "clears_during_measurement"
                ),
                "matched_removals": eviction_med("matched_removals"),
                "mapped_removals": eviction_med("mapped_removals"),
                "ambiguous_removals": eviction_med("ambiguous_removals"),
                "buffer_truncated": any(
                    bool(row.get("event_buffer_truncated"))
                    for row in eviction_reps
                ),
                "collection_errors": [
                    row.get("event_collection_error") for row in eviction_reps
                    if row.get("event_collection_error")
                ],
                "by_group": eviction_groups,
            },
            "per_request": per_request,
            "removal_rows": (
                reps[0].get("_removal_rows") or [] if len(reps) == 1 else []
            ),
            "started_at_epoch_s": reps[0].get("started_at_epoch_s"),
            "wall_s": med("wall_s"),
        })
    # Oldest first: the arm you recorded first is the baseline the rest are
    # measured against, which alphabetical order would scramble.
    return sorted(arms, key=lambda a: (a["recorded_at"], a["label"]))


def comparability(arms: list[dict[str, Any]]) -> list[str]:
    """Reasons these arms are not measuring the same workload."""
    problems = []
    for field, what in (
        ("bundle", "different bundles"),
        ("mode", "different replay modes"),
        ("max_concurrency", "different replay concurrency limits"),
        ("warmup_repetitions", "different warm-up counts"),
        ("requests", "different call counts"),
        ("prompt_tokens", "different total input"),
        ("fixed_output_tokens_per_request", "different fixed output lengths"),
        ("cache_state", "different cache states at start"),
    ):
        values = {str(a.get(field)) for a in arms}
        if len(values) > 1:
            problems.append(f"{what}: {', '.join(sorted(values))}")
    return problems


def differing_config_keys(arms: list[dict[str, Any]]) -> list[str]:
    """Serving-config keys that are not identical across arms."""
    keys = {k for a in arms for k in a["serving_config"]}
    differing = {
        k for k in keys
        if len({a["serving_config"].get(k) for a in arms}) > 1
    }
    ordered = [k for k in KNOBS if k in differing]
    return ordered + sorted(differing - set(ordered))


def warming_arms(arms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Arms whose repetitions are a warming sequence, not replicates.

    Without a cache reset the first pass loads the bundle's prefixes and every
    later pass reads them back, so a median across repetitions mixes a cold
    measurement with a warm one and describes neither.
    """
    return [
        arm for arm in arms
        if len(arm["hit_rates"]) > 1
        and max(arm["hit_rates"]) - min(arm["hit_rates"]) > WARMING_SPREAD
    ]


def _num(value: Any, suffix: str = "", scale: float = 1.0) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{value * scale:,.2f}{suffix}"


def _count(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{int(value):,}"
    if isinstance(value, str):
        try:
            return f"{int(float(value)):,}"
        except ValueError:
            pass
    return "n/a"


_AUTOMATIC_LABEL = re.compile(
    r"^block(?P<block>[^_]+)_dtype-(?P<dtype>[^_]+)_prefix-(?P<prefix>[^_]+)_"
    r"gpu(?P<gpu>[^_]+)_(?P<stamp>\d{8}T\d{6})(?P<utc>Z?)$"
)


def pretty_arm(arm: dict[str, Any]) -> str:
    """Format an arm like a task name while keeping its directory unchanged."""
    label = str(arm["label"])
    match = _AUTOMATIC_LABEL.match(label)
    if match:
        values = match.groupdict()
        stamp_time = _dt.datetime.strptime(values["stamp"], "%Y%m%dT%H%M%S")
        if values["utc"]:
            stamp_time = stamp_time.replace(
                tzinfo=_dt.timezone.utc
            ).astimezone(EXPERIMENT_TIMEZONE)
        stamp = stamp_time.strftime("%m-%d %H:%M")
        try:
            gpu = f"{float(values['gpu']):.2f}"
        except ValueError:
            gpu = values["gpu"]
        return (
            f"block {values['block']} · {values['dtype']} · "
            f"prefix {values['prefix']} · GPU {gpu} · {stamp}"
        )
    recorded_at = arm.get("recorded_at")
    if isinstance(recorded_at, (int, float)) and recorded_at > 0:
        stamp = from_epoch(recorded_at).strftime("%m-%d %H:%M")
        return f"{label} · {stamp}"
    return label


def render(arms: list[dict[str, Any]]) -> list[str]:
    if not arms:
        return ["No arms recorded yet."]
    config_keys = differing_config_keys(arms)
    if len({arm.get("cache_salt_mode") for arm in arms}) > 1:
        config_keys = ["cache_salt_mode", *config_keys]
    headers = config_keys + [
        "median TTFT (client)", "p95 TTFT (client)",
        "median TTFT (server)", "server queue", "median TPOT (client)",
        "server ITL", "median request latency", "wall",
        "vs baseline",
    ]
    warming = warming_arms(arms)
    baseline = arms[0].get("median_ttft_ms")
    lines = [
        "| arm | " + " | ".join(headers) + " |",
        "| --- |" + " --- |" * len(config_keys)
        + " ---: |" * (len(headers) - len(config_keys)),
    ]
    for arm in arms:
        ttft = arm.get("median_ttft_ms")
        if arm is arms[0]:
            delta = "baseline"
        elif isinstance(ttft, (int, float)) and baseline:
            delta = f"{(ttft / baseline - 1) * 100:+.2f}% TTFT"
        else:
            delta = "n/a"
        lines.append("| " + " | ".join(
            [pretty_arm(arm)]
            + [
                str(
                    arm.get(k, "?")
                    if k == "cache_salt_mode"
                    else arm["serving_config"].get(k, "?")
                )
                for k in config_keys
            ]
            + [
                _num(arm.get("median_ttft_ms"), "s", 0.001),
                _num(arm.get("p95_ttft_ms"), "s", 0.001),
                _num(arm.get("median_server_ttft_ms"), "s", 0.001),
                _num(arm.get("median_server_queue_ms"), "s", 0.001),
                _num(arm.get("median_tpot_ms"), "ms/token"),
                _num(arm.get("median_server_itl_ms"), "ms/token"),
                _num(arm.get("median_total_ms"), "s", 0.001),
                _num(arm.get("wall_s"), "s"),
                delta,
            ]
        ) + " |")
    fixed_output = arms[0].get("fixed_output_tokens_per_request")
    output_description = (
        f"output fixed at {_count(fixed_output)} tokens per request"
        if isinstance(fixed_output, (int, float))
        else "legacy output length"
    )
    lines.append("")
    lines.append(
        f"{arms[0]['requests']} calls, "
        f"{_count(arms[0].get('prompt_tokens'))} total input tokens, "
        f"{output_description}, "
        f"replayed in `{arms[0]['mode']}` mode from `{arms[0]['bundle']}`."
    )
    lines.append(
        f"Cache-salt policy: `{arms[0]['cache_salt_mode']}`; per-request "
        f"cached-token coverage: {_num(arms[0].get('per_request_cache_coverage'), '%', 100)}."
    )
    for arm in warming:
        lines.append("")
        lines.append(
            f"**{arm['label']}: its repetitions are not replicates.** Hit rate "
            f"went {arm['hit_rates'][0]:.2%} → {arm['hit_rates'][-1]:.2%} "
            "across them, so the first pass loaded the prefixes the later ones "
            "read back. Every median on this row mixes a cold pass with a warm "
            "one and describes neither. Use a distinct cache salt per repetition "
            "and verify first-request cached tokens; reset alone does not establish a cold offload cache."
        )
    if any("cold_since_restart" in a["cache_state"] for a in arms):
        lines.append("")
        lines.append(
            "`cold*` = the engine's query counter was still at zero, so it had "
            "served nothing since restart."
        )
    problems = comparability(arms)
    if problems:
        lines.append("")
        lines.append("**Not comparable** — " + "; ".join(problems) + ".")
    if len(arms) == 1:
        lines.append("")
        lines.append("One arm only — nothing to compare against yet.")
    elif not config_keys:
        lines.append("")
        lines.append(
            "Every arm reports the same serving config; the server was not "
            "restarted with a different knob between them, so any difference "
            "above is run-to-run noise, not a knob effect."
        )
    return lines


def _write_physical_capacity_plot(sweep_dir: Path, arms: list[dict[str, Any]]) -> Path | None:
    """Show individual measured runs and medians without imputing missing traces."""
    from agent_io_tracing.serving.kv_report_data import load_report_data, read_json
    import matplotlib.pyplot as plt

    points = []
    for arm in arms:
        for summary in sorted((sweep_dir / arm['label']).glob('rep*/summary.json')):
            rep = summary.parent
            if read_json(rep / 'kv_report_metadata.json').get('exclude_from_comparison'):
                continue
            data = load_report_data(rep)
            physical, causal = data.get('physical_trace'), data.get('causal_trace')
            raw = read_json(summary)
            config = raw.get('serving_config') or {}
            offload = config.get('kv_offloading_size', 'unknown')
            disabled = offload is None or str(offload).lower() in {'none', '0', '0.0'}
            if not physical or not causal or not disabled:
                continue
            requests = causal.get('request_paths') or []
            if len(requests) != raw['requests'] or not all(r.get('token_accounting_matches_prompt') for r in requests):
                continue
            points.append({'run': str(rep.relative_to(sweep_dir)), 'gpu_blocks': int(config['num_gpu_blocks']), 'runtime_s': raw['wall_s'], 'completed_prefill_tokens': causal['completed_prefill_tokens'], 'physical_evictions': physical['physical_evictions']})
    capacities = sorted({p['gpu_blocks'] for p in points})
    if len(capacities) < 2:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, key, label in zip(axes, ['runtime_s', 'completed_prefill_tokens', 'physical_evictions'], ['Replay runtime (s)', 'Completed prompt computation (tokens)', 'Physical cached-content evictions']):
        for capacity in capacities:
            values = [p[key] for p in points if p['gpu_blocks'] == capacity]
            ax.scatter([capacity] * len(values), values, facecolors='none', edgecolors='#007c78', s=55)
            ax.plot(capacity, statistics.median(values), marker='_', color='#172c44', markersize=18, markeredgewidth=2)
        ax.set_xticks(capacities, [f'{c}\nn={sum(p["gpu_blocks"] == c for p in points)}' for c in capacities])
        ax.set_xlabel('Configured GPU blocks')
        ax.set_ylabel(label)
        ax.set_ylim(0, max(1, max(p[key] for p in points)) * 1.1)
        ax.grid(alpha=.2)
    fig.suptitle('CPU offload disabled · circles: individual runs · bars: medians')
    fig.tight_layout()
    directory = sweep_dir / 'visualizations'
    directory.mkdir(exist_ok=True)
    target = directory / 'physical_capacity.png'
    fig.savefig(target, dpi=160, bbox_inches='tight')
    plt.close(fig)
    target.with_suffix('.json').write_text(json.dumps({'points': points, 'scope': 'Only saved physical traces with complete request token accounting. Prompt computation includes first computation and recomputation. Hardware provenance remains in the experiment receipts.'}, indent=2) + '\n')
    return target


def write_report(sweep_dir: Path, results_dir: Path | None = None) -> Path:
    """Refresh the owning task pages after a replay experiment."""
    from agent_io_tracing.analysis.kvcache.catalog import Catalog, root_for
    from agent_io_tracing.serving.kv_report_data import prepare_report_data
    sources = sorted(sweep_dir.glob('*/rep*/summary.json'))
    if not sources:
        raise ValueError('a fixed-input report requires at least one run')
    for source in sources:
        prepare_report_data(source.parent.resolve())
    root = results_dir.resolve().parent if results_dir else root_for(sweep_dir)
    catalog = Catalog(root, sources=None if (root/'replay').is_dir() else sources)
    from agent_io_tracing.analysis.kvcache.replay_report import publish_catalog
    index = publish_catalog(catalog)
    row = catalog.by_rep.get(sources[0].parent.resolve())
    return catalog.page(row) if row else index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    arm = sub.add_parser("arm", help="replay the bundle against the live server")
    arm.add_argument("--bundle", type=Path, required=True)
    arm.add_argument("--sweep-dir", type=Path, required=True)
    arm.add_argument(
        "--label",
        help="Optional result name; defaults to live config plus New York time.",
    )
    arm.add_argument("--mode", choices=("packed", "paced", "serial"), default="packed")
    arm.add_argument("--output-tokens", type=int, default=32)
    arm.add_argument("--max-concurrency", type=int)
    arm.add_argument("--warmup-repetitions", type=int, default=0)
    arm.add_argument(
        "--cache-salt-mode",
        choices=("none", "shared", "unique"),
        default="none",
    )
    arm.add_argument("--cache-salt")
    arm.add_argument(
        "--kv-events-replay-endpoint",
        default=os.environ.get("VLLM_KV_EVENTS_REPLAY_ENDPOINT"),
    )
    arm.add_argument(
        "--strict-config",
        action="store_true",
        help="Reject changes from the sweep baseline outside --allow-config-change.",
    )
    arm.add_argument(
        "--allow-config-change",
        action="append",
        default=[],
        help="CacheConfig key expected to differ. Repeat for multiple keys.",
    )
    arm.add_argument("--repeat", type=int, default=1)
    arm.add_argument(
        "--keep-cache",
        action="store_true",
        help="Do not reset the prefix cache first; only for a warm-start arm.",
    )
    arm.add_argument(
        "--url",
        default=(
            os.environ.get("VLLM_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
        ),
    )
    arm.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY") or "")
    arm.add_argument("--results", type=Path, default=Path("results"))

    report = sub.add_parser("report", help="tabulate the arms recorded so far")
    report.add_argument("--sweep-dir", type=Path, required=True)
    report.add_argument("--results", type=Path, default=Path("results"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "report":
        if not args.sweep_dir.is_dir():
            raise SystemExit(f"{args.sweep_dir} does not exist")
        try:
            path = write_report(args.sweep_dir, args.results)
        except ValueError as exc:
            raise SystemExit(f"report failed: {exc}") from exc
        from agent_io_tracing.analysis.results_index import write_results_index
        index_path = write_results_index(args.results.resolve())
        print("\n".join(render(load_arms(args.sweep_dir))))
        print(f"\nwrote {path}")
        print(f"index → {index_path}")
        return

    if not args.url:
        raise SystemExit("Set VLLM_URL or pass --url.")
    endpoint = VLLMEndpoint(args.url, args.api_key, timeout_s=300.0)
    label = args.label or "automatic"
    try:
        if not args.label:
            label = automatic_label(endpoint)
        summaries = record_arm(
            args.bundle,
            args.sweep_dir,
            label,
            endpoint,
            mode=args.mode,
            repeat=args.repeat,
            reset_before=not args.keep_cache,
            output_tokens=args.output_tokens,
            max_concurrency=args.max_concurrency,
            cache_salt_mode=args.cache_salt_mode,
            cache_salt=args.cache_salt,
            kv_events_replay_endpoint=args.kv_events_replay_endpoint,
            warmup_repetitions=args.warmup_repetitions,
            strict_config=args.strict_config,
            allowed_config_changes=set(args.allow_config_change),
        )
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"arm {label} failed: {exc}") from exc
    print(json.dumps(summaries[-1], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
