"""Measure observed GPU removal-to-return intervals from KV event logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent_io_tracing.serving.kv_events import block_identity, event_medium, event_scope, event_in_window


def derive_lifecycle(
    rows: list[dict[str, Any]],
    *,
    started_at: float | None = None,
    ended_at: float | None = None,
    observation_boundaries: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Track each removal until return, reset, or the observation boundary.

    BlockStored also announces recomputation and prefix reuse. A return is
    therefore an observation of GPU presence, not proof of a CPU load.
    """
    cpu: dict[tuple[str, ...], float] = {}
    pending: dict[tuple[str, ...], dict[str, Any]] = {}
    episodes: list[dict[str, Any]] = []
    generations: dict[tuple[str, ...], int] = {}
    last_times = {}
    last_sequences = {}
    seen_events = {}
    missing_sequences = 0
    duplicate_events = 0
    duplicate_removals = 0

    def finish(key: tuple[str, ...], status: str, ts: float) -> None:
        item = pending.pop(key)
        item["status"] = status
        item["observation_end_batch_ts"] = ts
        item["observed_duration_s"] = round(ts - item["removed_batch_ts"], 6)
        item["return_batch_ts"] = ts if status == "returned" else None
        item["removal_to_next_store_s"] = item["observed_duration_s"] if status == "returned" else None
        if item.pop("in_measurement", True):
            episodes.append(item)

    for row in rows:
        scope = event_scope(row)
        stream = scope[:2]
        ts = float(row["batch_ts"])
        if "phase" not in row and ended_at is not None and ts > ended_at:
            continue
        if isinstance(row.get("sequence"), int) and isinstance(row.get("event_index"), int):
            event_id = stream + (str(row["sequence"]), str(row["event_index"]))
            if event_id in seen_events:
                if seen_events[event_id] != row:
                    raise ValueError("Conflicting duplicate KV event")
                duplicate_events += 1
                continue
            seen_events[event_id] = row
        if stream in last_times and ts < last_times[stream]:
            raise ValueError("KV events must be ordered within each publisher epoch")
        previous_ts = last_times.get(stream, ts)
        kind = row.get("event_type")
        sequence = row.get("sequence")
        previous_seq = last_sequences.get(stream)
        implicit_gap = isinstance(sequence, int) and previous_seq is not None and sequence > previous_seq + 1
        if implicit_gap or kind == "ObservationGap":
            if implicit_gap:
                missing_sequences += sequence - previous_seq - 1
            elif kind == "ObservationGap":
                missing_sequences += row.get("end_sequence", 0) - row.get("start_sequence", 0) + 1
            for key in list(pending):
                if key[:2] == stream:
                    pending[key]["cpu_copy_continuously_observed"] = False
                    finish(key, "observation_gap", previous_ts)
            cpu = {key: value for key, value in cpu.items() if key[:2] != stream}
        if isinstance(sequence, int):
            if previous_seq is not None and sequence < previous_seq:
                raise ValueError("KV sequence decreased without a publisher epoch change")
            last_sequences[stream] = sequence
        last_times[stream] = ts
        if kind == "AllBlocksCleared":
            for key in list(pending):
                if key[:3] == scope:
                    finish(key, "reset", ts)
            cpu = {key: value for key, value in cpu.items() if key[:3] != scope}
            generations[scope] = generations.get(scope, 0) + 1
            continue
        medium = event_medium(row)
        for block_hash in row.get("block_hashes") or []:
            key = block_identity(row, block_hash)
            if medium == "CPU":
                if kind == "BlockStored":
                    cpu.setdefault(key, ts)
                elif kind == "BlockRemoved":
                    cpu.pop(key, None)
                    if key in pending:
                        pending[key]["cpu_copy_continuously_observed"] = False
                continue
            if medium != "GPU":
                continue
            if kind == "BlockRemoved":
                if key in pending:
                    duplicate_removals += 1
                    continue
                pending[key] = {
                    "block_hash": block_hash,
                    "group_idx": row.get("group_idx"),
                    "data_parallel_rank": row.get("data_parallel_rank"),
                    "locality": row.get("locality"),
                    "publisher_id": row.get("publisher_id"),
                    "publisher_epoch": row.get("publisher_epoch"),
                    "cache_generation": generations.get(scope, 0),
                    "in_measurement": event_in_window(row, started_at, ended_at),
                    "removed_sequence": row.get("sequence"),
                    "removed_batch_ts": ts,
                    "cpu_stored_batch_ts": cpu.get(key),
                    "cpu_copy_continuously_observed": key in cpu,
                    "transfer_confirmed": False,
                }
            elif kind == "BlockStored" and key in pending:
                pending[key]["return_sequence"] = row.get("sequence")
                finish(key, "returned", ts)

    for boundary in observation_boundaries or []:
        if boundary.get('phase') != 'after_measurement' or not boundary.get('boundary_confirmed'):
            continue
        scope = event_scope(boundary)[:2]
        timestamp = boundary['server_ack']['publisher_epoch_s']
        if scope in last_times:
            if timestamp < last_times[scope]:
                raise ValueError('Publication boundary predates its acknowledged events')
            last_times[scope] = timestamp
    phased = any("phase" in row for row in rows)
    end = max(last_times.values(), default=None) if phased or ended_at is None else ended_at
    for key in list(pending):
        finish(key, "right_censored", last_times[key[:2]] if phased or ended_at is None else ended_at)

    def unique(items: list[dict[str, Any]]) -> int:
        return len({block_identity(item, item["block_hash"]) + (
            str(item["cache_generation"]),
        ) for item in items})

    returned = [item for item in episodes if item["status"] == "returned"]
    candidates = [item for item in returned if item["cpu_copy_continuously_observed"]]
    def distribution(items: list[dict[str, Any]]) -> dict[str, float | None]:
        durations = sorted(item["removal_to_next_store_s"] for item in items)

        def quantile(p: float) -> float | None:
            if not durations:
                return None
            index = (len(durations) - 1) * p
            lo = int(index)
            hi = min(lo + 1, len(durations) - 1)
            return round(durations[lo] + (durations[hi] - durations[lo]) * (index - lo), 6)

        return {"p50": quantile(.5), "p95": quantile(.95), "max": max(durations, default=None)}

    summary = {
        "schema_version": 3,
        "measurement": "GPU BlockRemoved to next GPU BlockStored for the same logical block",
        "identity": "publisher, epoch, rank, group, locality, hash, reset generation",
        "time_basis": "publisher batch timestamps, not DMA timestamps",
        "transfer_confirmation": "unavailable in BlockStored/BlockRemoved events",
        "window_start_epoch_s": started_at,
        "window_end_epoch_s": end,
        "removal_episodes": len(episodes),
        "unique_removed_blocks": unique(episodes),
        "return_episodes": len(returned),
        "distribution_population": "returned episodes only",
        "distribution_sample_size": len(returned),
        "unknown_medium_events": sum(event_medium(row) == "unknown" and row.get("event_type") in {"BlockStored", "BlockRemoved"} for row in rows),
        "unique_returned_blocks": unique(returned),
        "cpu_backed_return_candidates": len(candidates),
        "unique_cpu_backed_return_candidates": unique(candidates),
        "right_censored_episodes": sum(item["status"] == "right_censored" for item in episodes),
        "reset_terminated_episodes": sum(item["status"] == "reset" for item in episodes),
        "duplicate_removals": duplicate_removals,
        "duplicate_events": duplicate_events,
        "gap_terminated_episodes": sum(item["status"] == "observation_gap" for item in episodes),
        "missing_sequence_numbers": missing_sequences,
        "removal_to_next_store_s": distribution(returned),
        "cpu_backed_candidate_removal_to_next_store_s": distribution(candidates),
    }
    return episodes, summary


def main() -> None:
    from agent_io_tracing.serving.kv_report_data import prepare_report_data

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rep_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare_report_data(args.rep_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
