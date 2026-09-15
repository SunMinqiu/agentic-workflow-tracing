import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def runner(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'tools/run_kv_capacity_matrix.py'
    spec = importlib.util.spec_from_file_location('capacity_matrix', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    return module


def test_partial_cell_cannot_be_reused_with_an_unrelated_trace(runner, tmp_path):
    rep = runner.cell_repetitions(274, 0)[0]
    rep.mkdir(parents=True)
    (rep / 'summary.json').write_text('{}')
    with pytest.raises(RuntimeError, match='Incomplete cell contains evidence'):
        runner.validate_resume_cell(tmp_path / 'receipt.json', 274, 0)


def test_receipt_requires_both_workloads_and_physical_traces(runner, tmp_path):
    marker = tmp_path / 'receipt.json'
    marker.write_text(json.dumps({'gpu_blocks': 192, 'rep': 0}))
    with pytest.raises(RuntimeError, match='missing saved evidence'):
        runner.validate_resume_cell(marker, 192, 0)
    for rep in runner.cell_repetitions(192, 0):
        (rep / 'physical_traces').mkdir(parents=True)
        (rep / 'summary.json').write_text('{}')
        (rep / 'physical_traces/engine.jsonl').write_text('{}\n')
    assert runner.validate_resume_cell(marker, 192, 0)
    with pytest.raises(RuntimeError, match='does not match'):
        runner.validate_resume_cell(marker, 274, 0)


def test_new_cell_can_start(runner, tmp_path):
    assert not runner.validate_resume_cell(tmp_path / 'receipt.json', 274, 0)
