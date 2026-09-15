import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('trace_patch',Path(__file__).parents[1]/'tools/prepare_vllm_trace_patch.py')
patch=importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)


def setup_source(tmp_path,monkeypatch):
    source=tmp_path/'source';pins={}
    for name in patch.TARGETS:
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('x = 1\n')
        pins[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    pin=tmp_path/'pins.json';pin.write_text(json.dumps({'files':pins}))
    monkeypatch.setattr(patch,'PIN',pin)
    return source


def test_patch_round_trip_and_source_mismatch_guard(tmp_path,monkeypatch):
    source=setup_source(tmp_path,monkeypatch);output=tmp_path/'patch'
    patch.prepare(source,output,apply=True)
    assert all('_install_agent_trace' in (source/name).read_text() for name in patch.TARGETS)
    patch.restore(source,output)
    assert all((source/name).read_text()=='x = 1\n' for name in patch.TARGETS)
    (source/list(patch.TARGETS)[-1]).write_text('x = 2\n')
    with pytest.raises(ValueError,match='fingerprint mismatch'):
        patch.prepare(source,output,apply=True)
    assert (source/next(iter(patch.TARGETS))).read_text()=='x = 1\n'


def test_restore_preserves_edits_since_patch(tmp_path,monkeypatch):
    source=setup_source(tmp_path,monkeypatch);output=tmp_path/'patch'
    patch.prepare(source,output,apply=True)
    path=source/next(iter(patch.TARGETS));path.write_text(path.read_text()+'y = 2\n')
    with pytest.raises(ValueError,match='Patched source changed'):
        patch.restore(source,output)
    assert path.read_text().endswith('y = 2\n')
