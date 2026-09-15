"""Collect vLLM KV-cache block events from its ZMQ publisher."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _protocol_types(msgspec: Any) -> tuple[type, Any]:
    """Build wire-compatible types without importing the vLLM package."""
    event_base = msgspec.defstruct(
        "KVCacheEvent",
        [],
        tag=True,
        omit_defaults=True,
        gc=False,
    )
    block_stored = msgspec.defstruct(
        "BlockStored",
        [
            ("block_hashes", list[Any]),
            ("parent_block_hash", Any | None),
            ("token_ids", list[int]),
            ("block_size", int),
            ("lora_id", int | None),
            ("medium", str | None),
            ("lora_name", str | None),
            ("extra_keys", list[Any] | None, None),
            ("group_idx", int | None, None),
            ("kv_cache_spec_kind", str | None, None),
            ("kv_cache_spec_sliding_window", int | None, None),
            ("locality", str | None, None),
        ],
        bases=(event_base,),
        omit_defaults=True,
        gc=False,
    )
    block_removed = msgspec.defstruct(
        "BlockRemoved",
        [
            ("block_hashes", list[Any]),
            ("medium", str | None),
            ("group_idx", int | None, None),
            ("locality", str | None, None),
        ],
        bases=(event_base,),
        omit_defaults=True,
        gc=False,
    )
    all_blocks_cleared = msgspec.defstruct(
        "AllBlocksCleared",
        [],
        bases=(event_base,),
        omit_defaults=True,
        gc=False,
    )
    event_union = block_stored | block_removed | all_blocks_cleared
    KVEventBatch = msgspec.defstruct(
        "KVEventBatch",
        [
            ("ts", float),
            ("events", list[event_union]),
            ("data_parallel_rank", int | None, None),
        ],
        array_like=True,
        omit_defaults=True,
        gc=False,
    )
    return KVEventBatch, msgspec.msgpack.Decoder(type=KVEventBatch)


def _dependencies() -> tuple[Any, Any, type, Any]:
    try:
        import msgspec
        import zmq
    except ImportError as exc:
        raise RuntimeError(
            "KV event collection requires pyzmq and msgspec; install the "
            "project's observability extra"
        ) from exc
    batch_type, decoder = _protocol_types(msgspec)
    return zmq, msgspec, batch_type, decoder


def _event_rows(msgspec: Any, sequence: int, batch: Any) -> list[dict[str, Any]]:
    rows = []
    for event_index, event in enumerate(batch.events):
        payload = msgspec.to_builtins(event)
        if not isinstance(payload, dict):
            payload = {"payload": payload}
        rows.append({
            "sequence": sequence,
            "event_index": event_index,
            "batch_ts": batch.ts,
            "data_parallel_rank": getattr(batch, "data_parallel_rank", None),
            "event_type": type(event).__name__,
            **payload,
        })
    return rows


def fetch_replay(
    endpoint: str,
    *,
    start_sequence: int = 0,
    timeout_ms: int = 5000,
    on_batch: Any = None,
    barrier_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch the publisher's buffered event batches from a replay endpoint."""
    zmq, msgspec, _, decoder = _dependencies()
    context = zmq.Context.instance()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(endpoint)
    # ROUTER expects [identity, empty delimiter, start sequence]. DEALER owns
    # the identity frame, so the client supplies the remaining two frames.
    request_payload = json.dumps({"start_sequence": start_sequence, "barrier_id": barrier_id}).encode() if barrier_id else start_sequence.to_bytes(8, "big")
    socket.send_multipart((b"", request_payload))
    barrier_ack = None
    rows: list[dict[str, Any]] = []
    sequences: list[int] = []
    try:
        while socket.poll(timeout_ms):
            frames = socket.recv_multipart()
            if len(frames) != 4 or frames[0] != b"":
                raise RuntimeError(f"unexpected KV replay frame count: {len(frames)}")
            _, _, sequence_bytes, payload = frames
            if sequence_bytes == b"\xff" * 8:
                barrier_ack = json.loads(payload) if payload else None
                break
            if not payload:
                raise RuntimeError("Unexpected empty replay batch")
            sequence = int.from_bytes(sequence_bytes, "big")
            sequences.append(sequence)
            batch = decoder.decode(payload)
            batch_rows = _event_rows(msgspec, sequence, batch)
            if on_batch is not None:
                on_batch({"sequence": sequence, "batch_ts": batch.ts,
                          "data_parallel_rank": getattr(batch, "data_parallel_rank", None),
                          "received_at_epoch_s": time.time(), "events": batch_rows})
            else:
                rows.extend(batch_rows)
        else:
            raise RuntimeError(f"KV replay endpoint {endpoint} timed out")
    finally:
        socket.close()
    first = min(sequences) if sequences else None
    last = max(sequences) if sequences else None
    return rows, {
        "endpoint": endpoint,
        "requested_start_sequence": start_sequence,
        "first_sequence": first,
        "last_sequence": last,
        "buffer_truncated": first is not None and first > start_sequence,
        "batches": len(set(sequences)),
        "events": len(rows),
        "reply_complete": True,
        "barrier_ack": barrier_ack,
    }


def summarize_events(
    rows: list[dict[str, Any]],
    *,
    started_at: float | None = None,
    ended_at: float | None = None,
) -> dict[str, Any]:
    """Count block events, optionally restricted to a replay time window."""
    selected = [row for row in rows if row.get("event_type") not in {"ObservationGap", "EmptyBatch"}
                and event_in_window(row, started_at, ended_at)]
    if not any("phase" in row for row in rows):
        while selected and selected[0].get("event_type") == "AllBlocksCleared":
            selected.pop(0)
    kinds: dict[str, int] = {}
    groups: dict[str, dict[str, int]] = {}
    hashes = {"stored": 0, "removed": 0}
    for row in selected:
        kind = str(row.get("event_type") or "unknown")
        kinds[kind] = kinds.get(kind, 0) + 1
        group = str(row.get("group_idx"))
        group_counts = groups.setdefault(group, {})
        group_counts[kind] = group_counts.get(kind, 0) + 1
        count = len(row.get("block_hashes") or [])
        if kind == "BlockStored":
            hashes["stored"] += count
        elif kind == "BlockRemoved":
            hashes["removed"] += count
    return {
        "events": len(selected),
        "event_types": kinds,
        "block_hashes": hashes,
        "by_group": groups,
        "window_start_epoch_s": started_at,
        "window_end_epoch_s": ended_at,
    }


def _hash_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def event_medium(row: dict[str, Any]) -> str:
    """Keep unspecified storage media unknown in every analysis."""
    value = row.get("medium")
    return str(value).upper() if value else "unknown"


def event_scope(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(_hash_key(row.get(field)) for field in (
        "publisher_id", "publisher_epoch", "data_parallel_rank",
    ))


def event_in_window(row: dict[str, Any], started_at: float | None, ended_at: float | None) -> bool:
    if "phase" in row:
        return row["phase"] == "measurement"
    ts = float(row.get("batch_ts", 0) or 0)
    return (started_at is None or ts >= started_at) and (ended_at is None or ts <= ended_at)


def block_identity(row: dict[str, Any], block_hash: Any) -> tuple[str, ...]:
    """Identify content within one publisher epoch, rank, group and locality."""
    return event_scope(row) + tuple(_hash_key(row.get(field)) for field in (
        "group_idx", "locality",
    )) + (_hash_key(block_hash),)


def _content_candidates(
    token_ids: list[int],
    requests: list[tuple[Any, str]],
) -> list[dict[str, Any]]:
    """Find exact token subsequences, including overlapping occurrences."""
    if not token_ids:
        return []
    pattern = "," + ",".join(map(str, token_ids)) + ","
    candidates = []
    for request_id, prompt in requests:
        cursor = 0
        while (position := prompt.find(pattern, cursor)) >= 0:
            candidates.append({
                "request_id": request_id,
                "token_offset": prompt.count(",", 0, position),
            })
            cursor = position + 1
    return candidates


def derive_removals(
    rows: list[dict[str, Any]],
    requests: list[dict[str, Any]] | None = None,
    *,
    started_at: float | None = None,
    ended_at: float | None = None,
) -> list[dict[str, Any]]:
    """Join in-window removals to stores and token-content candidates."""
    resident: dict[tuple[str, ...], dict[str, Any]] = {}
    removals = []
    request_index = [(request.get("request_id"), "," + ",".join(map(str, request.get("prompt_token_ids") or [])) + ",") for request in requests or []]
    candidate_cache: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    last_sequences = {}
    for row in rows:
        kind = row.get("event_type")
        stream = event_scope(row)[:2]
        sequence = row.get("sequence")
        previous = last_sequences.get(stream)
        if kind == "ObservationGap" or isinstance(sequence, int) and previous is not None and sequence > previous + 1:
            resident = {key: value for key, value in resident.items() if key[:2] != stream}
        if isinstance(sequence, int):
            last_sequences[stream] = sequence
        if kind == "ObservationGap":
            continue
        if kind == "AllBlocksCleared":
            scope = event_scope(row)
            resident = {key: value for key, value in resident.items() if key[:3] != scope}
            continue
        if kind == "BlockStored":
            hashes = row.get("block_hashes") or []
            tokens = row.get("token_ids") or []
            block_size = int(row.get("block_size") or 0)
            expected_tokens = block_size * len(hashes)
            spec_kind = str(row.get("kv_cache_spec_kind") or "").lower()
            ambiguous = (
                block_size <= 0
                or len(tokens) != expected_tokens
                or "mamba" in spec_kind
                or "ssm" in spec_kind
            )
            for index, block_hash in enumerate(hashes):
                block_tokens = (
                    tokens[index * block_size:(index + 1) * block_size]
                    if not ambiguous else []
                )
                key = block_identity(row, block_hash) + (event_medium(row),)
                previous = resident.get(key)
                token_key = tuple(block_tokens)
                if token_key not in candidate_cache:
                    candidate_cache[token_key] = _content_candidates(block_tokens, request_index)
                resident[key] = {
                    "stored_sequence": (
                        previous or {}
                    ).get("stored_sequence", row.get("sequence")),
                    "stored_batch_ts": (
                        previous or {}
                    ).get("stored_batch_ts", row.get("batch_ts")),
                    "last_stored_sequence": row.get("sequence"),
                    "last_stored_batch_ts": row.get("batch_ts"),
                    "store_observations": int(
                        (previous or {}).get("store_observations", 0)
                    ) + 1,
                    "block_hash": block_hash,
                    "block_size": block_size,
                    "token_ids": block_tokens,
                    "token_mapping_ambiguous": ambiguous,
                    "content_match_candidates": candidate_cache[token_key],
                    "kv_cache_spec_kind": row.get("kv_cache_spec_kind"),
                    "kv_cache_spec_sliding_window": row.get(
                        "kv_cache_spec_sliding_window"
                    ),
                }
            continue
        if kind != "BlockRemoved":
            continue
        removed_at = float(row.get("batch_ts", 0) or 0)
        in_window = event_in_window(row, started_at, ended_at)
        for block_hash in row.get("block_hashes") or []:
            key = block_identity(row, block_hash) + (event_medium(row),)
            stored = resident.pop(key, None)
            if not in_window:
                continue
            first_stored_at = (stored or {}).get("stored_batch_ts")
            last_stored_at = (stored or {}).get("last_stored_batch_ts")
            removals.append({
                "publisher_id": row.get("publisher_id"),
                "publisher_epoch": row.get("publisher_epoch"),
                "removed_sequence": row.get("sequence"),
                "removed_batch_ts": row.get("batch_ts"),
                "group_idx": row.get("group_idx"),
                "data_parallel_rank": row.get("data_parallel_rank"),
                "locality": row.get("locality"),
                "medium": event_medium(row),
                "block_hash": block_hash,
                "matched_store": stored is not None,
                "observed_residence_s": (
                    round(removed_at - float(first_stored_at), 6)
                    if first_stored_at is not None else None
                ),
                "time_since_last_store_s": (
                    round(removed_at - float(last_stored_at), 6)
                    if last_stored_at is not None else None
                ),
                **(stored or {}),
            })
    return removals


def write_events(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_live(
    endpoint: str,
    topic: str,
    output: Path,
    duration_s: float | None,
) -> None:
    """Subscribe until interrupted or until the requested duration elapses."""
    zmq, msgspec, _, decoder = _dependencies()
    context = zmq.Context.instance()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(endpoint)
    socket.setsockopt_string(zmq.SUBSCRIBE, topic)
    deadline = time.monotonic() + duration_s if duration_s is not None else None
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", buffering=1) as stream:
        try:
            while deadline is None or time.monotonic() < deadline:
                if not socket.poll(250):
                    continue
                _, sequence_bytes, payload = socket.recv_multipart()
                sequence = int.from_bytes(sequence_bytes, "big")
                for row in _event_rows(msgspec, sequence, decoder.decode(payload)):
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        finally:
            socket.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--endpoint", help="Live ZMQ PUB endpoint.")
    source.add_argument("--replay-endpoint", help="Buffered ZMQ replay endpoint.")
    parser.add_argument("--topic", default="kv-events")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--start-sequence", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.replay_endpoint:
        rows, transport = fetch_replay(
            args.replay_endpoint,
            start_sequence=args.start_sequence,
        )
        write_events(args.output, rows)
        print(json.dumps({**transport, **summarize_events(rows)}, indent=2))
        return
    collect_live(args.endpoint, args.topic, args.output, args.duration)


if __name__ == "__main__":
    main()
