import json
import re

import pytest

from agent_io_tracing.replay import runner, sweep
from agent_io_tracing.serving.vllm_endpoint import (
    parse_cache_config,
    prefix_cache_delta,
    VLLMEndpoint,
)

METRICS = """\
# HELP vllm:cache_config_info Cache config
vllm:cache_config_info{block_size="1568",cache_dtype="fp8",enable_prefix_caching="True"} 1.0
vllm:prefix_cache_queries_total{model_name="m"} %d.0
vllm:prefix_cache_hits_total{model_name="m"} %d.0
"""


def test_physical_capacity_plot_handles_disabled_offload_without_imputing_missing(tmp_path, monkeypatch):
    from agent_io_tracing.serving import kv_report_data
    arms = []
    for capacity, offload in [(192, 'None'), (274, None), (548, None)]:
        label = f'gpu{capacity}'
        arms.append({'label': label})
        rep = tmp_path / label / 'rep0'
        rep.mkdir(parents=True)
        (rep / 'summary.json').write_text(json.dumps({'requests': 1, 'wall_s': 2, 'serving_config': {'num_gpu_blocks': str(capacity), 'kv_offloading_size': offload}}))
    measured = {'physical_trace': {'physical_evictions': 0}, 'causal_trace': {'request_paths': [{'token_accounting_matches_prompt': True}], 'completed_prefill_tokens': 10}}
    monkeypatch.setattr(kv_report_data, 'load_report_data', lambda rep: {} if rep.parent.name == 'gpu548' else measured)
    image = sweep._write_physical_capacity_plot(tmp_path, arms)
    points = json.loads(image.with_suffix('.json').read_text())['points']
    assert [p['gpu_blocks'] for p in points] == [192, 274]
    assert all(p['physical_evictions'] == 0 for p in points)


class FakeEndpoint:
    """Serves one scrape before the replay and one after."""

    def __init__(self) -> None:
        self.scrapes = [METRICS % (1000, 100), METRICS % (2000, 700)]
        self.resets = 0

    def metrics(self) -> str:
        return self.scrapes.pop(0)

    def reset_prefix_cache(self) -> None:
        self.resets += 1

    def version(self):
        return {"version": "0.26.0"}

    def models(self):
        return ["m"]

    def server_info(self):
        return {"vllm_config": {"cache_config": {"cache_dtype": "fp8"}}}


def test_automatic_label_uses_live_config_and_timestamp() -> None:
    endpoint = FakeEndpoint()

    label = sweep.automatic_label(endpoint)

    assert re.fullmatch(
        r"block1568_dtype-fp8_prefix-on_gpuunknown_\d{8}T\d{6}",
        label,
    )


def test_eviction_timeline_uses_fixed_ten_second_bins() -> None:
    assert sweep.EVICTION_TIMELINE_BIN_S == 10.0


def test_cache_config_and_prefix_cache_come_from_the_server() -> None:
    config = parse_cache_config(METRICS % (0, 0))

    assert config["block_size"] == "1568"
    assert config["cache_dtype"] == "fp8"
    assert prefix_cache_delta(METRICS % (1000, 100), METRICS % (2000, 700)) == {
        "unit": "tokens",
        "queries": 1000,
        "hits": 600,
        "hit_rate": 0.6,
    }


def test_replay_records_the_config_that_was_actually_serving(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "run_bundle",
        lambda bundle, endpoint, mode, **options: [
            {
                "index": 0, "request_id": "a", "prompt_tokens": 1000,
                "output_tokens": 11, "ttft_ms": 800.0, "total_ms": 1000.0,
                "tpot_ms": 20.0,
            },
        ],
    )
    endpoint = FakeEndpoint()

    summary = runner.replay_to_dir(
        {"served_model": "m", "requests": [{"index": 0}]},
        endpoint,
        "packed",
        tmp_path / "rep0",
        arm={"label": "fp8"},
        reset_before=True,
    )

    assert endpoint.resets == 1
    assert summary["serving_config"]["cache_dtype"] == "fp8"
    assert summary["prefix_cache"]["hit_rate"] == 0.6
    assert summary["median_ttft_ms"] == 800.0
    assert summary["p95_ttft_ms"] == 800.0
    assert summary["median_tpot_ms"] == 20.0
    assert summary["output_tokens"] == 11
    assert summary["fixed_output_tokens_per_request"] == 32
    assert (tmp_path / "rep0" / "metrics_before.prom").is_file()
    assert (tmp_path / "rep0" / "metrics_window.json").is_file()
    server_info = json.loads(
        (tmp_path / "rep0" / "server_info.json").read_text(encoding="utf-8")
    )
    assert server_info["version"] == {"version": "0.26.0"}
    assert server_info["server_info"]["vllm_config"]["cache_config"]["cache_dtype"] == "fp8"
    assert summary["server_metadata"]["full_config_captured"] is True


def test_warmup_is_recorded_but_excluded_from_measurement_window(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_run(bundle, endpoint, mode, **options):
        calls.append(options)
        return [{
            "index": 0,
            "request_id": "a",
            "prompt_tokens": 4,
            "output_tokens": 2,
            "cached_tokens": 4 if len(calls) > 1 else 0,
            "ttft_ms": 10.0,
            "tpot_ms": 2.0,
            "total_ms": 12.0,
            "server_metrics": {},
            "cache_salt": options["cache_salt"],
        }]

    monkeypatch.setattr(runner, "run_bundle", fake_run)
    endpoint = FakeEndpoint()

    summary = runner.replay_to_dir(
        {"served_model": "m", "requests": [{"index": 0}]},
        endpoint,
        "serial",
        tmp_path / "rep0",
        reset_before=True,
        cache_salt_mode="shared",
        cache_salt="stable",
        warmup_repetitions=1,
    )

    assert len(calls) == 2
    assert calls[0]["cache_salt"] == calls[1]["cache_salt"] == "stable"
    assert summary["warmup"][0]["cached_tokens"] == 0
    assert summary["per_request_cache"]["cached_tokens"] == 4
    assert (tmp_path / "rep0" / "warmup0_requests.jsonl").is_file()


def test_replay_requests_32_output_tokens_and_measures_tpot(monkeypatch) -> None:
    sent = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            chunks = []
            for index in range(32):
                chunk = {
                    "id": "result",
                    "choices": [{"text": "x", "token_ids": [index]}],
                    "usage": {
                        "completion_tokens": 32,
                        "prompt_tokens_details": {"cached_tokens": 16},
                    } if index == 31 else None,
                    "metrics": {
                        "time_to_first_token_ms": 12.5,
                        "queue_time_ms": 2.0,
                        "generation_time_ms": 31.0,
                        "mean_itl_ms": 1.0,
                    } if index == 31 else None,
                }
                chunks.append(f"data: {json.dumps(chunk)}\n".encode())
            return iter(chunks + [b"data: [DONE]\n"])

    def fake_urlopen(request, timeout):
        sent.update(json.loads(request.data))
        return Response()

    monkeypatch.setattr(runner, "urlopen", fake_urlopen)
    result = runner._completion(
        VLLMEndpoint("http://server"),
        "model",
        {
            "index": 0,
            "request_id": "a",
            "prompt_token_ids": [1, 2],
            "sampling_params": {"stop": ["STOP"]},
        },
    )

    assert sent["min_tokens"] == 32
    assert sent["max_tokens"] == 32
    assert sent["ignore_eos"] is True
    assert "stop" not in sent
    assert result["output_tokens"] == 32
    assert result["tpot_ms"] is not None
    assert result["cached_tokens"] == 16
    assert result["server_metrics"]["queue_time_ms"] == 2.0


def test_replay_passes_shared_cache_salt_and_custom_output_length(monkeypatch) -> None:
    sent = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            return iter([
                b'data: {"choices":[{"text":"x","token_ids":[1]}],"usage":{"completion_tokens":1}}\n',
                b"data: [DONE]\n",
            ])

    def fake_urlopen(request, timeout):
        sent.update(json.loads(request.data))
        return Response()

    monkeypatch.setattr(runner, "urlopen", fake_urlopen)
    runner._completion(
        VLLMEndpoint("http://server"),
        "model",
        {"index": 0, "request_id": "a", "prompt_token_ids": [1, 2]},
        output_tokens=1,
        cache_salt="shared-test-salt",
    )

    assert sent["min_tokens"] == 1
    assert sent["max_tokens"] == 1
    assert sent["cache_salt"] == "shared-test-salt"


def _write_arm(sweep_dir, label, block_size, ttft, hit_rate, at,
               cache_state="cold_by_reset") -> None:
    path = sweep_dir / label / "rep0"
    path.mkdir(parents=True)
    (path / "summary.json").write_text(json.dumps({
        "bundle": "/x/genomas.json",
        "started_at_epoch_s": at,
        "cache_state": cache_state,
        "mode": "packed",
        "requests": 17,
        "prompt_tokens": 165000,
        "output_tokens": 544,
        "fixed_output_tokens_per_request": 32,
        "serving_config": {"block_size": block_size, "cache_dtype": "fp8"},
        "prefix_cache": {"hit_rate": hit_rate},
        "median_ttft_ms": ttft,
        "p95_ttft_ms": ttft + 500.0,
        "median_tpot_ms": 14.4,
        "median_total_ms": 3000.0,
        "wall_s": 60.0,
        "output_tokens_per_s": 120.0,
    }), encoding="utf-8")


def test_report_columns_only_the_knobs_that_changed(tmp_path) -> None:
    _write_arm(tmp_path, "block16", "16", 1000.0, 0.46, at=1)
    _write_arm(tmp_path, "block1568", "1568", 1500.0, 0.21, at=2)

    text = "\n".join(sweep.render(sweep.load_arms(tmp_path)))

    assert "block_size" in text
    assert "cache_dtype" not in text
    assert "median TPOT" in text
    assert "14.40ms/token" in text
    assert "1.00s" in text
    assert "p95 TTFT" in text
    assert "1.50s" in text
    assert "p99 TTFT" not in text
    assert "3.00s" in text
    assert "60.00s" in text
    assert "+50.00% TTFT" in text
    assert "Not comparable" not in text


def test_old_replay_backfills_tail_ttft_from_requests(tmp_path) -> None:
    rep = tmp_path / "arm" / "rep0"
    rep.mkdir(parents=True)
    (rep / "summary.json").write_text(json.dumps({
        "median_ttft_ms": 20.0,
    }), encoding="utf-8")
    (rep / "requests.jsonl").write_text("\n".join(
        json.dumps({"ttft_ms": value}) for value in [10.0, 20.0, 30.0]
    ), encoding="utf-8")

    summary = sweep.load_repetition_summary(rep / "summary.json")

    assert summary["p95_ttft_ms"] == 29.0


def test_repetition_backfills_windowed_eviction_analysis(tmp_path) -> None:
    rep = tmp_path / "arm" / "rep0"
    rep.mkdir(parents=True)
    (rep / "summary.json").write_text(json.dumps({
        "started_at_epoch_s": 10.0,
        "ended_at_epoch_s": 20.0,
        "prompt_tokens": 1000,
        "kv_events": {"buffer_truncated": False},
    }), encoding="utf-8")
    (rep / "requests.jsonl").write_text(
        json.dumps({"request_id": "r", "ttft_ms": 10.0}),
        encoding="utf-8",
    )
    (rep / "metrics_before.prom").write_text("""\
vllm:kv_block_idle_before_evict_seconds_bucket{le="1"} 0
vllm:kv_block_idle_before_evict_seconds_bucket{le="5"} 0
vllm:kv_block_idle_before_evict_seconds_bucket{le="+Inf"} 0
vllm:kv_block_idle_before_evict_seconds_count 0
vllm:kv_block_idle_before_evict_seconds_sum 0
vllm:num_preemptions_total 0
""", encoding="utf-8")
    (rep / "metrics_after.prom").write_text("""\
vllm:kv_block_idle_before_evict_seconds_bucket{le="1"} 0
vllm:kv_block_idle_before_evict_seconds_bucket{le="5"} 2
vllm:kv_block_idle_before_evict_seconds_bucket{le="+Inf"} 2
vllm:kv_block_idle_before_evict_seconds_count 2
vllm:kv_block_idle_before_evict_seconds_sum 6
vllm:num_preemptions_total 0
""", encoding="utf-8")
    events = [
        {"sequence": 0, "batch_ts": 11.0, "event_type": "AllBlocksCleared"},
        {"sequence": 0, "batch_ts": 11.0, "event_type": "BlockStored", "medium": "GPU",
         "group_idx": 0, "block_hashes": ["h"], "block_size": 1,
         "token_ids": [1], "kv_cache_spec_kind": "full_attention"},
        {"sequence": 1, "batch_ts": 13.0, "event_type": "BlockRemoved",
         "group_idx": 0, "block_hashes": ["h"], "medium": "GPU"},
    ]
    (rep / "kv_events.jsonl").write_text(
        "\n".join(json.dumps(row) for row in events), encoding="utf-8"
    )

    summary = sweep.load_repetition_summary(rep / "summary.json")
    eviction = summary["eviction"]

    assert eviction["eviction_histogram_observations"] == 2
    assert eviction["removed_cached_hashes"] == 1
    assert eviction["mean_idle_before_evict_s"] == 3
    assert eviction["clears_during_measurement"] == 0
    assert eviction["by_group"]["0"]["kinds"] == {"full_attention": 1}


def test_strict_config_ignores_restart_ids_but_checks_other_environment() -> None:
    old = {
        "vllm_config": {"instance_id": "old"},
        "vllm_env": {
            "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME": "buffer-old",
        },
        "system_env": {
            "env_vars": (
                "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME=buffer-old\n"
                "VLLM_KV_METRICS_SAMPLE=0.01"
            ),
        },
    }
    new = {
        "vllm_config": {"instance_id": "new"},
        "vllm_env": {
            "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME": "buffer-new",
        },
        "system_env": {
            "env_vars": (
                "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME=buffer-new\n"
                "VLLM_KV_METRICS_SAMPLE=1.0"
            ),
        },
    }

    assert sweep._server_config_changes(old, new) == {
        "system_env.env_vars.VLLM_KV_METRICS_SAMPLE"
    }


def test_sweep_report_without_source_keeps_evidence_in_index(tmp_path) -> None:
    _write_arm(tmp_path, "block16", "16", 1000.0, 0.46, at=1)
    _write_arm(tmp_path, "block32", "32", 900.0, 0.52, at=2)
    report = sweep.write_report(tmp_path)
    page = report.read_text()
    assert report.name == 'all-experiments.html'
    assert 'Fixed-input replay experiments' in page
    assert 'Not collected' in page
    assert not (tmp_path / 'kvcache_report.html').exists()


def test_a_restarted_engine_counts_as_cold_without_a_reset_endpoint() -> None:
    assert runner._cache_state(False, METRICS % (0, 0)) == "cold_since_restart"
    assert runner._cache_state(False, METRICS % (122473, 37632)) == "warm_inherited"
    assert runner._cache_state(True, METRICS % (122473, 37632)) == "reset_requested"


def test_repetitions_that_warm_each_other_are_not_averaged_silently(tmp_path) -> None:
    _write_arm(tmp_path, "twopass", "784", 695.1, 0.3888, at=1)
    rep1 = tmp_path / "twopass" / "rep1"
    rep1.mkdir()
    row = json.loads((tmp_path / "twopass" / "rep0" / "summary.json").read_text())
    row.update({"prefix_cache": {"hit_rate": 0.9935}, "median_ttft_ms": 340.3})
    rep1.joinpath("summary.json").write_text(json.dumps(row), encoding="utf-8")

    text = "\n".join(sweep.render(sweep.load_arms(tmp_path)))

    assert "38.88% → 99.35%" in text
    assert "not replicates" in text


def test_report_flags_a_warm_arm_against_a_cold_one(tmp_path) -> None:
    _write_arm(tmp_path, "cold", "16", 1000.0, 0.46, at=1)
    _write_arm(tmp_path, "warm", "1568", 400.0, 0.92, at=2,
               cache_state="warm_inherited")

    text = "\n".join(sweep.render(sweep.load_arms(tmp_path)))

    assert "different cache states at start" in text


def test_arm_fails_before_replaying_when_the_cache_cannot_be_reset(tmp_path) -> None:
    class NoResetEndpoint:
        def reset_prefix_cache(self):
            raise RuntimeError("POST /reset_prefix_cache returned HTTP 404")

        def metrics(self):
            raise AssertionError("must not start replaying")

    bundle = tmp_path / "b.json"
    bundle.write_text(json.dumps({"served_model": "m", "requests": []}))

    try:
        sweep.record_arm(bundle, tmp_path / "sweep", "a", NoResetEndpoint())
    except RuntimeError as exc:
        assert "VLLM_SERVER_DEV_MODE=1" in str(exc)
        assert "--keep-cache" in str(exc)
    else:
        raise AssertionError("expected the missing endpoint to abort the arm")


def test_arm_refuses_to_mix_with_an_existing_label(tmp_path) -> None:
    bundle = tmp_path / "b.json"
    bundle.write_text(json.dumps({"served_model": "m", "requests": []}))
    existing = tmp_path / "sweep" / "same-label"
    existing.mkdir(parents=True)

    try:
        sweep.record_arm(bundle, tmp_path / "sweep", "same-label", FakeEndpoint())
    except ValueError as exc:
        assert str(existing) in str(exc)
    else:
        raise AssertionError("expected the existing arm directory to be rejected")


def test_keep_cache_refuses_multiple_repetitions(tmp_path) -> None:
    bundle = tmp_path / "b.json"
    bundle.write_text(json.dumps({"served_model": "m", "requests": []}))

    try:
        sweep.record_arm(
            bundle,
            tmp_path / "sweep",
            "warm",
            FakeEndpoint(),
            repeat=2,
            reset_before=False,
        )
    except ValueError as exc:
        assert "--repeat 1" in str(exc)
    else:
        raise AssertionError("expected warm repetitions to be rejected")


def test_report_refuses_to_compare_different_workloads(tmp_path) -> None:
    _write_arm(tmp_path, "a", "16", 1000.0, 0.46, at=1)
    _write_arm(tmp_path, "b", "1568", 1500.0, 0.21, at=2)
    summary_path = tmp_path / "b" / "rep0" / "summary.json"
    row = json.loads(summary_path.read_text())
    row["requests"] = 12
    summary_path.write_text(json.dumps(row), encoding="utf-8")

    text = "\n".join(sweep.render(sweep.load_arms(tmp_path)))

    assert "Not comparable" in text
    assert "different call counts" in text


def test_excluded_pilot_stays_in_details_but_not_comparison(tmp_path):
    _write_arm(tmp_path, 'pilot', '784', 695.1, 0.3888, at=1)
    rep=tmp_path/'pilot'/'rep0'
    (rep/'kv_report_metadata.json').write_text(json.dumps({'exclude_from_comparison':True,'exclusion_reason':'Cached tokens persisted across reset.'}))
    assert sweep.load_arms(tmp_path)==[]
    page = sweep.write_report(tmp_path).read_text()
    assert 'Excluded from comparison.' in page
