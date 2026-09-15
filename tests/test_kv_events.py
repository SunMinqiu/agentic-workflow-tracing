import threading
import uuid

import pytest

from agent_io_tracing.serving.kv_events import (
    derive_removals,
    fetch_replay,
    summarize_events,
)


def test_summarize_events_counts_hashes_by_kind_group_and_window() -> None:
    rows = [
        {
            "sequence": 4,
            "batch_ts": 10.0,
            "event_type": "BlockStored", "medium": "GPU",
            "group_idx": 0,
            "block_hashes": ["a", "b"],
        },
        {
            "sequence": 5,
            "batch_ts": 11.0,
            "event_type": "BlockRemoved",
            "group_idx": 1,
            "block_hashes": ["a"],
        },
        {
            "sequence": 6,
            "batch_ts": 20.0,
            "event_type": "AllBlocksCleared",
        },
    ]

    summary = summarize_events(rows, started_at=9.0, ended_at=12.0)

    assert summary["events"] == 2
    assert summary["event_types"] == {"BlockStored": 1, "BlockRemoved": 1}
    assert summary["block_hashes"] == {"stored": 2, "removed": 1}
    assert summary["by_group"]["0"] == {"BlockStored": 1}
    assert summary["by_group"]["1"] == {"BlockRemoved": 1}


def test_summarize_events_excludes_async_clear_before_first_measurement_event() -> None:
    rows = [
        {"sequence": 0, "batch_ts": 10.0, "event_type": "AllBlocksCleared"},
        {"sequence": 0, "batch_ts": 10.0, "event_type": "BlockStored", "medium": "GPU",
         "group_idx": 0, "block_hashes": ["a"]},
        {"sequence": 1, "batch_ts": 11.0, "event_type": "AllBlocksCleared"},
    ]

    summary = summarize_events(rows, started_at=9.0, ended_at=12.0)

    assert summary["event_types"] == {
        "BlockStored": 1,
        "AllBlocksCleared": 1,
    }


def test_derive_removals_recovers_tokens_and_candidate_request() -> None:
    rows = [
        {
            "sequence": 2,
            "batch_ts": 10.0,
            "event_type": "BlockStored", "medium": "GPU",
            "group_idx": 0,
            "block_hashes": [["h0"], ["h1"]],
            "block_size": 2,
            "token_ids": [10, 11, 12, 13],
        },
        {
            "sequence": 3,
            "batch_ts": 12.0,
            "event_type": "BlockRemoved",
            "group_idx": 0,
            "medium": "GPU",
            "block_hashes": [["h1"]],
        },
    ]
    requests = [{"request_id": "r1", "prompt_token_ids": [1, 10, 11, 12, 13, 99]}]

    removals = derive_removals(rows, requests)

    assert removals[0]["matched_store"] is True
    assert removals[0]["token_ids"] == [12, 13]
    assert removals[0]["content_match_candidates"] == [
        {"request_id": "r1", "token_offset": 3}
    ]
    assert removals[0]["token_mapping_ambiguous"] is False
    assert removals[0]["store_observations"] == 1
    assert removals[0]["observed_residence_s"] == 2.0
    assert removals[0]["time_since_last_store_s"] == 2.0


def test_derive_removals_keeps_store_history_and_filters_output_window() -> None:
    rows = [
        {"sequence": 1, "batch_ts": 5.0, "event_type": "BlockStored", "medium": "GPU",
         "group_idx": 0, "block_hashes": ["h"], "block_size": 1,
         "token_ids": [1]},
        {"sequence": 2, "batch_ts": 8.0, "event_type": "BlockStored", "medium": "GPU",
         "group_idx": 0, "block_hashes": ["h"], "block_size": 1,
         "token_ids": [1]},
        {"sequence": 3, "batch_ts": 12.0, "event_type": "BlockRemoved",
         "group_idx": 0, "block_hashes": ["h"], "medium": "GPU"},
        {"sequence": 4, "batch_ts": 20.0, "event_type": "BlockRemoved",
         "group_idx": 0, "block_hashes": ["other"], "medium": "GPU"},
    ]

    removals = derive_removals(rows, started_at=10.0, ended_at=15.0)

    assert len(removals) == 1
    assert removals[0]["store_observations"] == 2
    assert removals[0]["observed_residence_s"] == 7.0
    assert removals[0]["time_since_last_store_s"] == 4.0


def test_replay_client_receives_multiple_router_replies() -> None:
    msgspec = pytest.importorskip("msgspec")
    zmq = pytest.importorskip("zmq")
    endpoint = f"inproc://kv-events-{uuid.uuid4()}"
    context = zmq.Context.instance()
    router = context.socket(zmq.ROUTER)
    router.bind(endpoint)

    def serve() -> None:
        identity, delimiter, start = router.recv_multipart()
        assert delimiter == b""
        assert int.from_bytes(start, "big") == 0
        for sequence in (0, 1):
            payload = msgspec.msgpack.encode([
                1.0 + sequence,
                [{"type": "AllBlocksCleared"}],
                0,
            ])
            router.send_multipart((
                identity,
                b"",
                b"kv-events",
                sequence.to_bytes(8, "big"),
                payload,
            ))
        router.send_multipart((
            identity,
            b"",
            b"",
            (-1).to_bytes(8, "big", signed=True),
            b"",
        ))
        router.close()

    thread = threading.Thread(target=serve)
    thread.start()
    rows, transport = fetch_replay(endpoint, timeout_ms=1000)
    thread.join(timeout=1)

    assert transport["batches"] == 2
    assert transport["last_sequence"] == 1
    assert [row["event_type"] for row in rows] == [
        "AllBlocksCleared",
        "AllBlocksCleared",
    ]


def test_content_candidates_preserve_token_boundaries_and_overlaps():
    rows = [
        {"sequence": 0, "batch_ts": 0, "event_type": "BlockStored", "medium": "GPU", "block_hashes": ["h"], "block_size": 2, "token_ids": [1, 1]},
        {"sequence": 1, "batch_ts": 1, "event_type": "BlockRemoved", "medium": "GPU", "block_hashes": ["h"]},
    ]
    requests = [
        {"request_id": "r", "prompt_token_ids": [11, 1, 1, 1]},
        {"request_id": "other", "prompt_token_ids": [11, 11]},
    ]
    assert derive_removals(rows, requests)[0]["content_match_candidates"] == [
        {"request_id": "r", "token_offset": 1},
        {"request_id": "r", "token_offset": 2},
    ]
