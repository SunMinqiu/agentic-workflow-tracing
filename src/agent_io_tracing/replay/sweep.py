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

from agent_io_tracing.analysis.kvcache.summary import PAGE_CSS
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

    event_rows = _read_jsonl(path.parent / "kv_events.jsonl")
    removal_rows: list[dict[str, Any]] = []
    if event_rows:
        started_at = summary.get("started_at_epoch_s")
        ended_at = summary.get("ended_at_epoch_s")
        event_analysis = summarize_events(
            event_rows,
            started_at=started_at,
            ended_at=ended_at,
        )
        saved_removals = _read_jsonl(path.parent / "kv_removals.jsonl")
        removal_rows = [
            row for row in saved_removals
            if (
                started_at is None
                or float(row.get("removed_batch_ts", 0) or 0) >= started_at
            ) and (
                ended_at is None
                or float(row.get("removed_batch_ts", 0) or 0) <= ended_at
            )
        ]
        if not saved_removals:
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
        for row in removal_rows:
            if not isinstance(row.get("observed_idle_s"), (int, float)):
                removed_at = row.get("removed_batch_ts")
                stored_at = row.get("stored_batch_ts")
                if isinstance(removed_at, (int, float)) and isinstance(
                    stored_at, (int, float)
                ):
                    row["observed_idle_s"] = round(removed_at - stored_at, 6)
        prior = summary.get("kv_events") or {}
        summary["kv_events"] = {
            **prior,
            **event_analysis,
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
            "mapped_prefixes": 0,
            "ambiguous_mappings": 0,
            "kinds": {},
            "block_sizes": {},
            "observed_idle_s": [],
        })
        target["removed_hashes"] += 1
        target["matched_stores"] += int(bool(row.get("matched_store")))
        target["mapped_prefixes"] += int(bool(row.get("possible_request_prefixes")))
        target["ambiguous_mappings"] += int(bool(row.get("token_mapping_ambiguous")))
        kind = str(row.get("kv_cache_spec_kind") or "unknown")
        target["kinds"][kind] = target["kinds"].get(kind, 0) + 1
        size = str(row.get("block_size") or "unknown")
        target["block_sizes"][size] = target["block_sizes"].get(size, 0) + 1
        observed_idle = row.get("observed_idle_s")
        if isinstance(observed_idle, (int, float)):
            target["observed_idle_s"].append(observed_idle)
    for target in groups.values():
        idle_values = target.pop("observed_idle_s")
        target["median_observed_idle_s"] = (
            round(statistics.median(idle_values), 3) if idle_values else None
        )
        target["p95_observed_idle_s"] = percentile(idle_values, 0.95)
    physical = idle.get("observations")
    prompt_tokens = summary.get("prompt_tokens")
    return {
        "physical_block_evictions": physical,
        "metrics_sample": summary.get("_kv_cache_metrics_sample"),
        "physical_evictions_per_million_input": (
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
        ),
        "event_buffer_truncated": bool(events.get("buffer_truncated")),
        "event_collection_error": events.get("collection_error"),
        "removals": len(removals),
        "matched_removals": sum(bool(row.get("matched_store")) for row in removals),
        "mapped_removals": sum(
            bool(row.get("possible_request_prefixes")) for row in removals
        ),
        "ambiguous_removals": sum(
            bool(row.get("token_mapping_ambiguous")) for row in removals
        ),
        "by_group": groups,
    }


def load_arms(sweep_dir: Path) -> list[dict[str, Any]]:
    """One row per arm, medians taken across its repetitions."""
    arms: list[dict[str, Any]] = []
    for arm_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        reps = []
        for rep_dir in sorted(arm_dir.glob("rep*")):
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
                "mapped_prefixes": med("mapped_prefixes", group_rows),
                "ambiguous_mappings": med("ambiguous_mappings", group_rows),
                "median_observed_idle_s": med(
                    "median_observed_idle_s", group_rows
                ),
                "p95_observed_idle_s": med("p95_observed_idle_s", group_rows),
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
                "physical_block_evictions": eviction_med(
                    "physical_block_evictions"
                ),
                "metrics_sample": eviction_med("metrics_sample"),
                "physical_evictions_per_million_input": eviction_med(
                    "physical_evictions_per_million_input"
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


def _histogram_estimate(
    value: Any, lower: Any, upper: Any, suffix: str = "s"
) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    estimate = f"≈{value:,.2f}{suffix}"
    if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
        return f"{estimate} ({lower:g}–{upper:g}{suffix} bucket)"
    if isinstance(lower, (int, float)):
        return f"{estimate} (≥{lower:g}{suffix} bucket)"
    return estimate


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
        f"{arms[0]['prompt_tokens']:,} total input tokens, "
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
            "one and describes neither; take the repetitions apart, or give "
            "the server VLLM_SERVER_DEV_MODE=1 so each pass starts cold."
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


def _token_summary(arms: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| arm | calls | total input | total output | cache queries | cache hits | realized reuse | repetitions |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in arms:
        rate = _num(arm.get("hit_rate"), "%", 100)
        lines.append(
            f"| {pretty_arm(arm)} | {arm['requests']} | {arm['prompt_tokens']:,} | "
            f"{_count(arm.get('output_tokens'))} | "
            f"{_count(arm.get('cache_queries'))} | {_count(arm.get('cache_hits'))} | "
            f"{rate} | {arm['reps']} |"
        )
    return lines


def _eviction_summary(arms: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| arm | capacity | physical eviction observations | evictions / 1M input | removed cached hashes | mean idle before eviction | p95 idle before eviction | mean block lifetime | p95 block lifetime | preemptions | measurement coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for arm in arms:
        eviction = arm.get("eviction") or {}
        config = arm.get("serving_config") or {}
        removed = eviction.get("removed_cached_hashes")
        matched = eviction.get("matched_removals")
        mapped = eviction.get("mapped_removals")
        if eviction.get("collection_errors"):
            coverage = "collection error"
        elif eviction.get("buffer_truncated"):
            coverage = "buffer truncated"
        elif isinstance(removed, (int, float)) and removed > 0:
            match_pct = matched / removed if isinstance(matched, (int, float)) else 0
            map_pct = mapped / removed if isinstance(mapped, (int, float)) else 0
            coverage = f"{match_pct:.1%} store match; {map_pct:.1%} prefix mapped"
        elif isinstance(removed, (int, float)):
            coverage = "complete; no removals"
        else:
            coverage = "unmeasured"
        sample = eviction.get("metrics_sample")
        if isinstance(sample, (int, float)):
            coverage = f"eviction sample {sample:.0%}; {coverage}"
        clears = eviction.get("clears_during_measurement")
        if isinstance(clears, (int, float)) and clears > 0:
            coverage += f"; {_count(clears)} cache clear(s) in window"
        lines.append("| " + " | ".join([
            pretty_arm(arm),
            _count(config.get("kv_cache_size_tokens")),
            _count(eviction.get("physical_block_evictions")),
            _num(eviction.get("physical_evictions_per_million_input")),
            _count(removed),
            _num(eviction.get("mean_idle_before_evict_s"), "s"),
            _histogram_estimate(
                eviction.get("p95_idle_before_evict_s"),
                eviction.get("p95_idle_bucket_lower_s"),
                eviction.get("p95_idle_bucket_upper_s"),
            ),
            _num(eviction.get("mean_block_lifetime_s"), "s"),
            _histogram_estimate(
                eviction.get("p95_block_lifetime_s"),
                eviction.get("p95_lifetime_bucket_lower_s"),
                eviction.get("p95_lifetime_bucket_upper_s"),
            ),
            _count(eviction.get("preemptions")),
            coverage,
        ]) + " |")
    lines.extend([
        "",
        "Physical eviction observations come from vLLM's eviction histogram and equal the full count only when the configured sampling rate is 100%. Removed cached hashes come from `BlockRemoved`; the latter also includes hash replacement during partial-block promotion and is not itself a physical-eviction count. Histogram p95 values are interpolated estimates, so the containing bucket is shown beside each value.",
    ])
    return lines


def _eviction_group_summary(arms: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| arm | group | cache kind | block size | removed hashes | store matched | prefix mapped | ambiguous | median observed idle | p95 observed idle |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    found = False
    for arm in arms:
        groups = (arm.get("eviction") or {}).get("by_group") or {}
        for group, row in sorted(groups.items()):
            found = True
            removed = row.get("removed_hashes")

            def group_pct(key: str) -> str:
                value = row.get(key)
                if not isinstance(value, (int, float)) or not removed:
                    return "n/a"
                return f"{value / removed:.1%}"

            lines.append("| " + " | ".join([
                pretty_arm(arm),
                group,
                str(row.get("kind") or "unknown"),
                str(row.get("block_size") or "unknown"),
                _count(removed),
                group_pct("matched_stores"),
                group_pct("mapped_prefixes"),
                group_pct("ambiguous_mappings"),
                _num(row.get("median_observed_idle_s"), "s"),
                _num(row.get("p95_observed_idle_s"), "s"),
            ]) + " |")
    return lines if found else ["KV removal events were not collected."]


def _paired_request_impact(arms: list[dict[str, Any]]) -> list[str]:
    if len(arms) < 2:
        return []
    baseline = arms[0]
    baseline_requests = baseline.get("per_request") or {}
    if not baseline_requests:
        return []
    lines = [
        "| comparison | paired requests | gained cache | recovered cached tokens | lost cache | median TTFT delta among gained-cache requests | gained-cache requests faster | median TTFT delta with unchanged cache |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    found = False
    for arm in arms[1:]:
        current = arm.get("per_request") or {}
        common = sorted(baseline_requests.keys() & current.keys())
        if not common:
            continue
        found = True
        gained = []
        lost = []
        unchanged = []
        recovered_tokens = 0.0
        for request_id in common:
            old = baseline_requests[request_id]
            new = current[request_id]
            old_cache = old.get("cached_tokens")
            new_cache = new.get("cached_tokens")
            old_ttft = old.get("ttft_ms")
            new_ttft = new.get("ttft_ms")
            if not all(isinstance(value, (int, float)) for value in (
                old_cache, new_cache, old_ttft, new_ttft
            )):
                continue
            delta = new_ttft - old_ttft
            if new_cache > old_cache:
                gained.append(delta)
                recovered_tokens += new_cache - old_cache
            elif new_cache < old_cache:
                lost.append(delta)
            else:
                unchanged.append(delta)
        lines.append("| " + " | ".join([
            f"{pretty_arm(baseline)} → {pretty_arm(arm)}",
            _count(len(common)),
            _count(len(gained)),
            _count(recovered_tokens),
            _count(len(lost)),
            _num(statistics.median(gained) if gained else None, "ms"),
            f"{sum(delta < 0 for delta in gained)}/{len(gained)}" if gained else "n/a",
            _num(statistics.median(unchanged) if unchanged else None, "ms"),
        ]) + " |")
    return lines if found else []


def _write_comparison_plot(sweep_dir: Path, arms: list[dict[str, Any]]) -> Path | None:
    if not arms:
        return None
    import matplotlib.pyplot as plt

    labels = [arm["label"] for arm in arms]
    positions = list(range(len(labels)))
    height = max(3.5, 0.7 * len(labels) + 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(14, height))
    rates = [
        arm["hit_rate"] * 100 if isinstance(arm.get("hit_rate"), (int, float)) else 0
        for arm in arms
    ]
    ttft = [
        arm["median_ttft_ms"] / 1000 if isinstance(arm.get("median_ttft_ms"), (int, float)) else 0
        for arm in arms
    ]
    axes[0].barh(positions, rates, color="#4c78a8")
    axes[0].set_xlabel("Realized reuse (%)")
    axes[0].set_yticks(positions, labels=labels)
    axes[0].invert_yaxis()
    axes[1].barh(positions, ttft, color="#f58518")
    axes[1].set_xlabel("Median TTFT (s)")
    axes[1].set_yticks(positions, labels=[])
    axes[1].invert_yaxis()
    fig.suptitle(f"Fixed-input test: {sweep_dir.name}")
    fig.tight_layout()
    viz = sweep_dir / "visualizations"
    viz.mkdir(parents=True, exist_ok=True)
    path = viz / "sweep_comparison.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_eviction_timeline(
    sweep_dir: Path, arms: list[dict[str, Any]]
) -> Path | None:
    """Plot cached-hash removals over each measured replay window."""
    if not any(arm.get("removal_rows") for arm in arms):
        return None
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4.8))
    plotted = False
    durations = [
        float(arm["wall_s"]) for arm in arms
        if isinstance(arm.get("wall_s"), (int, float))
    ]
    max_duration = max(durations, default=1.0)
    bin_width = EVICTION_TIMELINE_BIN_S
    n_bins = max(1, int(max_duration / bin_width) + 1)
    for arm in arms:
        removals = arm.get("removal_rows") or []
        started = arm.get("started_at_epoch_s")
        if not removals or not isinstance(started, (int, float)):
            continue
        counts = [0] * n_bins
        for row in removals:
            removed_at = row.get("removed_batch_ts")
            if not isinstance(removed_at, (int, float)):
                continue
            index = min(max(int((removed_at - started) / bin_width), 0), n_bins - 1)
            counts[index] += 1
        x = [(index + 0.5) * bin_width for index in range(n_bins)]
        ax.plot(x, counts, marker="o", markersize=2.5, linewidth=1.4,
                label=pretty_arm(arm))
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("elapsed replay time (s)")
    ax.set_ylabel(f"removed cached hashes per {bin_width:.1f}s")
    ax.set_title("Cached-hash removal timeline")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    viz = sweep_dir / "visualizations"
    viz.mkdir(parents=True, exist_ok=True)
    path = viz / "eviction_timeline.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _per_arm_sections(sweep_dir: Path, arms: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for arm in arms:
        lines.extend([f"### {pretty_arm(arm)}", ""])
        lines.append(
            f"- Bundle: `{arm['bundle']}` · mode: `{arm['mode']}` · "
            f"cache: `{arm['cache_state']}` · repetitions: {arm['reps']}"
        )
        lines.extend(["", "**Server config**", ""])
        lines.extend(["| key | value |", "| --- | --- |"])
        for key, value in sorted(arm["serving_config"].items()):
            if key == "kv_cache_size_tokens":
                continue
            lines.append(f"| `{key}` | `{value}` |")
        lines.extend(["", "**Artifacts**", ""])
        for index in range(arm["reps"]):
            base = f"{arm['label']}/rep{index}"
            lines.append(
                f"- repetition {index + 1}: [summary]({base}/summary.json) · "
                f"[requests]({base}/requests.jsonl) · "
                f"[metrics before]({base}/metrics_before.prom) · "
                f"[metrics after]({base}/metrics_after.prom)"
                f" · [metrics window]({base}/metrics_window.json)"
                f" · [full server config]({base}/server_info.json)"
            )
            if index < len(arm.get("kv_events") or []) and arm["kv_events"][index]:
                events = arm["kv_events"][index]
                lines.append(
                    f"  - KV events: {events.get('events', 0)} events · "
                    f"{(events.get('block_hashes') or {}).get('stored', 0)} stored hashes · "
                    f"{(events.get('block_hashes') or {}).get('removed', 0)} removed hashes · "
                    f"[raw events]({base}/kv_events.jsonl) · "
                    f"[removal ledger]({base}/kv_removals.jsonl)"
                )
        lines.extend(["", "---", ""])
    return lines


def _capacity_section(arms: list[dict[str, Any]]) -> list[str]:
    """Show each distinct physical KV-cache capacity once per report."""
    capacities = set()
    for arm in arms:
        raw = (arm.get("serving_config") or {}).get("kv_cache_size_tokens")
        try:
            capacities.add(int(raw))
        except (TypeError, ValueError):
            continue
    if not capacities:
        return []
    lines = ["", "**KV-cache capacity used by these experiments**", ""]
    lines.extend(f"- {capacity:,} tokens" for capacity in sorted(capacities))
    return lines


def write_report(sweep_dir: Path, results_dir: Path | None = None) -> Path:
    import markdown

    arms = load_arms(sweep_dir)
    if len(arms) < 2:
        raise ValueError(
            "a fixed-input comparison report requires at least two arms"
        )
    comparison_plot = _write_comparison_plot(sweep_dir, arms)
    eviction_timeline = _write_eviction_timeline(sweep_dir, arms)
    lines = [
        f"# Fixed-input Test — {sweep_dir.name}",
        "",
        f"_Generated {_dt.date.today().isoformat()} · {len(arms)} arm(s) · fixed-input replay._",
        "",
        "## Summary — tokens",
        "",
    ]
    lines.extend(_token_summary(arms))
    lines.extend(_capacity_section(arms))
    lines.extend(["", "## Summary — time and configuration", ""])
    lines.extend(render(arms))
    if comparison_plot is not None:
        lines.extend(["", "![Realized reuse and TTFT by arm](visualizations/sweep_comparison.png)", ""])
    lines.extend(["", "## Eviction pressure", ""])
    lines.extend(_eviction_summary(arms))
    lines.extend(["", "## Eviction by KV group", ""])
    lines.extend(_eviction_group_summary(arms))
    if eviction_timeline is not None:
        lines.extend([
            "",
            "![Cached-hash removal timeline](visualizations/eviction_timeline.png)",
            "",
        ])
    paired_impact = _paired_request_impact(arms)
    if paired_impact:
        lines.extend(["", "## Per-request capacity impact", ""])
        lines.extend(paired_impact)
    lines.extend(["", "## Per arm", ""])
    lines.extend(_per_arm_sections(sweep_dir, arms))
    if results_dir is not None:
        index = results_dir.resolve() / "index.html"
        lines.extend([f"[Back to experiments index]({os.path.relpath(index, sweep_dir.resolve())})", ""])
    text = "\n".join(lines) + "\n"
    body = markdown.markdown(text, extensions=["tables"])
    html_text = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Fixed-input Test — {sweep_dir.name}</title>\n<style>\n{PAGE_CSS}"
        f"</style>\n</head>\n<body>\n{body}"
        + "\n</body>\n</html>\n"
    )
    path = sweep_dir / "kvcache_report.html"
    path.write_text(html_text, encoding="utf-8")
    return path


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
