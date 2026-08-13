import json
import re

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


class FakeEndpoint:
    """Serves one scrape before the replay and one after."""

    def __init__(self) -> None:
        self.scrapes = [METRICS % (1000, 100), METRICS % (2000, 700)]
        self.resets = 0

    def metrics(self) -> str:
        return self.scrapes.pop(0)

    def reset_prefix_cache(self) -> None:
        self.resets += 1


def test_automatic_label_uses_live_config_and_timestamp() -> None:
    endpoint = FakeEndpoint()

    label = sweep.automatic_label(endpoint)

    assert re.fullmatch(
        r"block1568_dtype-fp8_prefix-on_gpuunknown_\d{8}T\d{6}",
        label,
    )


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
        lambda bundle, endpoint, mode: [
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
    assert summary["median_tpot_ms"] == 20.0
    assert summary["output_tokens"] == 11
    assert summary["fixed_output_tokens_per_request"] == 32
    assert (tmp_path / "rep0" / "metrics_before.prom").is_file()


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
                    "usage": {"completion_tokens": 32} if index == 31 else None,
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
    assert "3.00s" in text
    assert "60.00s" in text
    assert "+50.00% TTFT" in text
    assert "Not comparable" not in text


def test_sweep_report_has_normal_kv_sections_and_artifacts(tmp_path) -> None:
    _write_arm(tmp_path, "block16", "16", 1000.0, 0.46, at=1)

    report = sweep.write_report(tmp_path)

    assert report.name == "kvcache_report.html"
    page = report.read_text(encoding="utf-8")
    assert "Summary — tokens" in page
    assert "Summary — time and configuration" in page
    assert "Fixed-input Test" in page
    assert "Per arm" in page
    assert "metrics before" in page
    assert (tmp_path / "sweep.html").is_file()
    assert (tmp_path / "visualizations" / "sweep_comparison.png").is_file()


def test_a_restarted_engine_counts_as_cold_without_a_reset_endpoint() -> None:
    assert runner._cache_state(False, METRICS % (0, 0)) == "cold_since_restart"
    assert runner._cache_state(False, METRICS % (122473, 37632)) == "warm_inherited"
    assert runner._cache_state(True, METRICS % (122473, 37632)) == "cold_by_reset"


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
