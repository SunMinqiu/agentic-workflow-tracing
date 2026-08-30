"""Tests for the run-time GenoMAS model-registry guard.

GenoMAS refuses any --model absent from the hard-coded MODEL_INFO dict in its
utils/llm.py, and a re-clone of that checkout silently reverts a previous
registration.  scripts/ensure_genomas_model.py re-registers on every run; these
tests pin the properties that matter: it is idempotent, the file it writes still
parses, and it never damages a checkout it cannot handle.
"""
from pathlib import Path
import importlib.util
import sys

import pytest

_HELPER = Path(__file__).resolve().parents[1] / "scripts" / "ensure_genomas_model.py"
_spec = importlib.util.spec_from_file_location("ensure_genomas_model", _HELPER)
ensure_genomas_model = importlib.util.module_from_spec(_spec)
sys.modules["ensure_genomas_model"] = ensure_genomas_model
_spec.loader.exec_module(ensure_genomas_model)

REGISTRY = """\
MODEL_INFO = {
    'openai': {
        'gpt-4o-mini-2024-07-18': {'input_price': 0.15, 'output_price': 0.6},
    },
    'anthropic': {
        'claude-sonnet-4-20250514': {'input_price': 3.0, 'output_price': 15.0},
    },
}
"""


def _repo(tmp_path: Path, source: str = REGISTRY) -> Path:
    (tmp_path / "utils").mkdir(parents=True, exist_ok=True)
    (tmp_path / "utils" / "llm.py").write_text(source, encoding="utf-8")
    return tmp_path


def _models(repo: Path, provider: str) -> list[str]:
    source = (repo / "utils" / "llm.py").read_text(encoding="utf-8")
    return ensure_genomas_model._model_info(source)[provider]


def test_registers_an_unknown_vllm_model_under_openai(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    assert ensure_genomas_model.ensure(repo, "Qwen3.6-27B") == 0

    assert "Qwen3.6-27B" in _models(repo, "openai")
    assert "gpt-4o-mini-2024-07-18" in _models(repo, "openai")
    assert _models(repo, "anthropic") == ["claude-sonnet-4-20250514"]


def test_re_registering_is_a_no_op(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    ensure_genomas_model.ensure(repo, "Qwen3.6-27B")
    once = (repo / "utils" / "llm.py").read_text(encoding="utf-8")

    assert ensure_genomas_model.ensure(repo, "Qwen3.6-27B") == 0

    assert (repo / "utils" / "llm.py").read_text(encoding="utf-8") == once


def test_a_model_already_under_another_provider_is_left_alone(tmp_path: Path) -> None:
    repo = _repo(tmp_path)

    assert ensure_genomas_model.ensure(repo, "claude-sonnet-4-20250514") == 0

    assert "claude-sonnet-4-20250514" not in _models(repo, "openai")


def test_a_corrupt_registry_is_reported_and_never_rewritten(tmp_path: Path) -> None:
    broken = "MODEL_INFO = {\n    'openai': {\n        'x': {'a': 0}},\n"
    repo = _repo(tmp_path, broken)

    assert ensure_genomas_model.ensure(repo, "Qwen3.6-27B") == 1

    assert (repo / "utils" / "llm.py").read_text(encoding="utf-8") == broken


def test_a_registry_without_the_openai_anchor_is_refused(tmp_path: Path) -> None:
    source = "MODEL_INFO = {\n    'anthropic': {\n        'c': {'input_price': 1.0},\n    },\n}\n"
    repo = _repo(tmp_path, source)

    assert ensure_genomas_model.ensure(repo, "Qwen3.6-27B") == 1

    assert (repo / "utils" / "llm.py").read_text(encoding="utf-8") == source


def test_a_missing_checkout_is_reported_rather_than_created(tmp_path: Path) -> None:
    assert ensure_genomas_model.ensure(tmp_path / "absent", "Qwen3.6-27B") == 1

    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("model", ["", "   "])
def test_an_empty_model_name_is_rejected(tmp_path: Path, model: str, monkeypatch) -> None:
    repo = _repo(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["ensure_genomas_model.py", "--repo", str(repo), "--model", model]
    )

    assert ensure_genomas_model.main() == 1
