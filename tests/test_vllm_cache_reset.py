from __future__ import annotations

import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
AGENT_SCRIPTS = (
    "trace_script_bcc_genomas.sh",
    "trace_script_bcc_genomas_fanout.sh",
    "trace_script_bcc_scilink.sh",
    "trace_script_bcc_sragent.sh",
    "trace_script_bcc_pi.sh",
    "trace_script_bcc_chemgraph.sh",
)


def _bash(body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f"set -u\nsource {SCRIPTS / 'lib_results.sh'}\n{body}"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_vllm_cache_is_reset_by_default() -> None:
    result = _bash(
        "VLLM_URL=http://127.0.0.1:18080/v1\n"
        "curl() { echo \"curl:$*\" >&2; return 0; }\n"
        "prepare_vllm_cache_for_cell\n"
        "echo $CACHE_STATE"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "cold_by_reset"
    assert "-X POST http://127.0.0.1:18080/reset_prefix_cache" in result.stderr


def test_provider_managed_backend_does_not_attempt_a_reset() -> None:
    result = _bash(
        "unset VLLM_URL\n"
        "curl() { echo called >&2; return 1; }\n"
        "prepare_vllm_cache_for_cell\n"
        "echo $CACHE_STATE"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "provider_managed"
    assert "called" not in result.stderr


def test_an_intentional_warm_run_requires_an_explicit_override() -> None:
    result = _bash(
        "VLLM_URL=http://127.0.0.1:18080\n"
        "VLLM_KEEP_PREFIX_CACHE=1\n"
        "curl() { echo called >&2; return 1; }\n"
        "prepare_vllm_cache_for_cell\n"
        "echo $CACHE_STATE"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "warm_inherited"
    assert "called" not in result.stderr


def test_a_failed_reset_stops_the_run() -> None:
    result = _bash(
        "VLLM_URL=http://127.0.0.1:18080\n"
        "curl() { return 22; }\n"
        "prepare_vllm_cache_for_cell"
    )
    assert result.returncode != 0
    assert "could not reset" in result.stderr


def test_every_agent_trace_resets_before_its_cell() -> None:
    for name in AGENT_SCRIPTS:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        assert text.count("prepare_vllm_cache_for_cell") == 1, name
