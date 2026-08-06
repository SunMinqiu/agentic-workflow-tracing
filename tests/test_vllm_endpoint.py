import pytest

from agent_io_tracing.serving import vllm_endpoint
from agent_io_tracing.serving.vllm_endpoint import (
    VLLMEndpoint,
    _base_url,
    _metric_names,
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
