import pytest

from agent_io_tracing.serving import vllm_endpoint
from agent_io_tracing.serving.vllm_endpoint import (
    VLLMEndpoint,
    _base_url,
    _metric_names,
    prometheus_window,
)


def test_base_url_accepts_openai_v1_form() -> None:
    assert _base_url("http://hpc.example:8000/v1") == "http://hpc.example:8000"
    assert _base_url("http://hpc.example:8000/") == "http://hpc.example:8000"


def test_metric_names_selects_kvcache_experiment_signals() -> None:
    metrics = """
# HELP vllm:prefix_cache_hits Prefix cache hits
vllm:prefix_cache_hits{model_name="qwen"} 12
vllm:kv_cache_usage_perc 0.5
vllm:num_requests_running 2
process_cpu_seconds_total 10
"""

    assert _metric_names(metrics) == [
        "vllm:kv_cache_usage_perc",
        "vllm:num_requests_running",
        "vllm:prefix_cache_hits",
    ]


def test_connection_reset_becomes_readable_error(monkeypatch) -> None:
    def reset(*args, **kwargs):
        raise ConnectionResetError(54, "Connection reset by peer")

    monkeypatch.setattr(vllm_endpoint, "urlopen", reset)

    with pytest.raises(RuntimeError, match="Connection reset by peer"):
        VLLMEndpoint("http://127.0.0.1:18000").health()


def test_prometheus_window_captures_cache_residency_and_request_timing() -> None:
    before = """\
vllm:num_preemptions_total{model_name="m"} 3
vllm:kv_cache_usage_perc{model_name="m"} 0.2
vllm:kv_block_lifetime_seconds_sum{model_name="m"} 10
vllm:kv_block_lifetime_seconds_count{model_name="m"} 2
vllm:kv_block_lifetime_seconds_bucket{model_name="m",le="1"} 1
vllm:kv_block_lifetime_seconds_bucket{model_name="m",le="5"} 2
vllm:kv_block_lifetime_seconds_bucket{model_name="m",le="+Inf"} 2
vllm:request_queue_time_seconds_sum{model_name="m"} 4
vllm:request_queue_time_seconds_count{model_name="m"} 2
"""
    after = """\
vllm:num_preemptions_total{model_name="m"} 5
vllm:kv_cache_usage_perc{model_name="m"} 0.8
vllm:kv_block_lifetime_seconds_sum{model_name="m"} 22
vllm:kv_block_lifetime_seconds_count{model_name="m"} 5
vllm:kv_block_lifetime_seconds_bucket{model_name="m",le="1"} 1
vllm:kv_block_lifetime_seconds_bucket{model_name="m",le="5"} 4
vllm:kv_block_lifetime_seconds_bucket{model_name="m",le="+Inf"} 5
vllm:request_queue_time_seconds_sum{model_name="m"} 10
vllm:request_queue_time_seconds_count{model_name="m"} 4
"""

    window = prometheus_window(before, after)

    assert window["metrics"]["vllm:num_preemptions_total"]["delta"] == 2
    assert window["metrics"]["vllm:kv_cache_usage_perc"]["after"] == 0.8
    assert window["histogram_window_means"]["vllm:kv_block_lifetime_seconds"] == {
        "observations": 3.0,
        "window_mean": 4.0,
    }
    assert window["histogram_window_means"]["vllm:request_queue_time_seconds"] == {
        "observations": 2.0,
        "window_mean": 3.0,
    }
    quantiles = window["histogram_window_quantiles"][
        "vllm:kv_block_lifetime_seconds"
    ]
    assert quantiles["observations"] == 3.0
    assert quantiles["p50"] == 4.0
    assert quantiles["p50_bucket_lower"] == 1.0
    assert quantiles["p50_bucket_upper"] == 5.0
    assert quantiles["p95"] == 5.0
    assert quantiles["p95_bucket_lower"] == 5.0
    assert "p95_bucket_upper" not in quantiles
    assert "vllm:kv_block_lifetime_seconds_bucket" not in window["metrics"]
