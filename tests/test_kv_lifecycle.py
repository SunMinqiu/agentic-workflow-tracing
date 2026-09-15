from agent_io_tracing.serving.kv_events import derive_removals
from agent_io_tracing.serving.kv_lifecycle import derive_lifecycle


def event(ts, kind, medium="GPU", block="a", rank=0, group=0):
    return dict(batch_ts=ts, event_type=kind, medium=medium,
                block_hashes=[block], data_parallel_rank=rank, group_idx=group,
                block_size=1, token_ids=[1])


def test_returns_count_episodes_and_unique_blocks_separately():
    rows = [event(1, "BlockStored", "CPU"), event(2, "BlockRemoved"),
            event(5, "BlockStored"), event(6, "BlockStored"),
            event(7, "BlockRemoved"), event(11, "BlockStored")]
    episodes, summary = derive_lifecycle(rows)
    assert [item["removal_to_next_store_s"] for item in episodes] == [3, 4]
    assert summary["return_episodes"] == 2
    assert summary["unique_returned_blocks"] == 1
    assert summary["cpu_backed_return_candidates"] == 2
    assert all(not item["transfer_confirmed"] for item in episodes)


def test_cpu_eviction_breaks_continuous_residency_even_if_restored():
    rows = [event(1, "BlockStored", "CPU"), event(2, "BlockRemoved"),
            event(3, "BlockRemoved", "CPU"), event(4, "BlockStored", "CPU"),
            event(5, "BlockStored")]
    episodes, summary = derive_lifecycle(rows)
    assert summary["cpu_backed_return_candidates"] == 0
    assert episodes[0]["removal_to_next_store_s"] == 3


def test_window_reset_rank_and_right_censoring():
    rows = [event(1, "BlockRemoved"), event(2, "BlockRemoved", rank=1),
            event(3, "AllBlocksCleared"), event(4, "BlockStored"),
            event(5, "BlockStored", rank=1), event(6, "BlockRemoved"),
            event(9, "BlockStored")]
    episodes, summary = derive_lifecycle(rows, started_at=2, ended_at=8)
    assert [item["status"] for item in episodes] == ["returned", "right_censored"]
    assert episodes[1]["removal_to_next_store_s"] is None
    assert episodes[1]["observed_duration_s"] == 2
    assert summary["return_episodes"] == 1


def test_group_identity_and_duplicate_removals():
    rows = [event(1, "BlockRemoved"), event(2, "BlockRemoved"),
            event(3, "BlockStored", group=1), event(4, "BlockStored")]
    episodes, summary = derive_lifecycle(rows)
    assert episodes[0]["removal_to_next_store_s"] == 3
    assert summary["duplicate_removals"] == 1


def test_removals_do_not_mix_media_or_ranks():
    rows = [event(1, "BlockStored"), event(2, "BlockStored", "CPU"),
            event(3, "BlockStored", rank=1), event(4, "BlockRemoved"),
            event(5, "BlockRemoved", "CPU"), event(6, "BlockRemoved", rank=1)]
    removed = derive_removals(rows)
    assert [item["stored_batch_ts"] for item in removed] == [1, 2, 3]


def test_reset_only_clears_its_own_rank():
    rows = [event(1, "BlockStored", rank=1), event(2, "AllBlocksCleared"),
            event(3, "BlockRemoved", rank=1)]
    assert derive_removals(rows)[0]["stored_batch_ts"] == 1
