import hashlib
import json
from pathlib import Path

from agent_io_tracing.replay.sweep import load_repetition_summary, write_report
from agent_io_tracing.serving.kv_events import derive_removals
from agent_io_tracing.serving.kv_lifecycle import derive_lifecycle
from agent_io_tracing.serving.kv_lifecycle_report import section
from agent_io_tracing.analysis.kvcache.catalog import Catalog, stable_id
from agent_io_tracing.analysis.kvcache.replay_report import run_section, refresh_task_pages
from agent_io_tracing.serving.kv_report_data import load_report_data, prepare_report_data


def setup_run(tmp_path, *, offload=0, transport=None, rows=None, rep_name='rep2'):
    rep = tmp_path / 'replay' / 'experiment' / 'arm' / rep_name
    rep.mkdir(parents=True)
    metadata = {'started_at_epoch_s': 0, 'ended_at_epoch_s': 10, 'mode': 'serial', 'requests': 1, 'wall_s': 10, 'serving_config': {'kv_offloading_size': offload}, 'kv_events': transport or {}}
    source = tmp_path / 'results/GenoMAS_task_20260914_120000/task'
    source.mkdir(parents=True, exist_ok=True)
    (source / 'messages.jsonl').write_text('')
    (source / 'kvcache_report.html').write_text('<html><head></head><body><h1>Task KVCache</h1><h2>Per task</h2><p>Original observations</p></body></html>')
    bundle = tmp_path / 'replay/bundle.json'
    bundle.write_text(json.dumps({'source_cell':str(source)}))
    metadata['bundle'] = str(bundle)
    (rep / 'summary.json').write_text(json.dumps(metadata))
    if rows is not None:
        (rep / 'kv_events.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return rep


def render_run(rep):
    catalog = Catalog(rep.parents[3])
    return run_section(catalog, catalog.by_rep[rep.resolve()], rep)


def event(seq, ts, kind, medium='GPU'):
    return {'sequence': seq, 'batch_ts': ts, 'event_type': kind, 'medium': medium, 'block_hashes': ['a'], 'group_idx': 0, 'data_parallel_rank': 0, 'block_size': 1, 'token_ids': [1]}


def test_missing_and_empty_unconfirmed_do_not_prove_zero(tmp_path):
    rep = setup_run(tmp_path)
    data = prepare_report_data(rep)
    assert data['collection_status'] == 'not_collected'
    assert data['lifecycle'] == {}
    assert section(rep) == ''
    assert not (rep / 'kv_block_report.html').exists()
    assert not (rep / 'kv_lifecycle.jsonl').exists()
    (rep / 'kv_events.jsonl').write_text('')
    data = prepare_report_data(rep)
    assert data['collection_status'] == 'unconfirmed'
    assert data['lifecycle'] == {}
    assert section(rep) == ''


def test_explicit_null_cpu_offload_is_disabled_but_missing_config_is_unknown(tmp_path):
    rep=setup_run(tmp_path,offload='None')
    assert prepare_report_data(rep)['offload']=='disabled'
    summary=json.loads((rep/'summary.json').read_text())
    summary['serving_config']['kv_offloading_size']=None
    (rep/'summary.json').write_text(json.dumps(summary))
    assert prepare_report_data(rep)['offload']=='disabled'
    summary['serving_config'].pop('kv_offloading_size')
    (rep/'summary.json').write_text(json.dumps(summary))
    assert prepare_report_data(rep)['offload']=='unknown'


def test_confirmed_empty_log_shows_zero_without_empty_ledgers(tmp_path):
    rep = setup_run(tmp_path, rows=[], transport={'collection_complete': True, 'tail_confirmed': True})
    data = prepare_report_data(rep)
    assert data['lifecycle']['removal_episodes'] == 0
    page = render_run(rep)
    assert 'Complete' in page
    assert 'class="kv-ledger"' not in page
    assert 'CPU-backed' not in page


def test_gpu_only_report_is_read_only_and_links_to_actual_arm(tmp_path):
    rep = setup_run(tmp_path, rows=[event(0, 1, 'BlockStored'), event(1, 2, 'BlockRemoved'), event(2, 5, 'BlockStored')])
    data = prepare_report_data(rep)
    assert data['collection_status'] == 'unconfirmed'
    assert data['lifecycle']['removal_to_next_store_s']['p50'] == 3
    files = [p for p in rep.iterdir() if p.suffix in {'.json', '.jsonl'}]
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}
    page = render_run(rep)
    assert 'CPU-backed' not in page
    assert 'CPU stored at s' not in page
    assert 'removal-to-next-store' in page.lower()
    assert 'only the 1 returned episodes' in page
    assert 'Back to summary' in page
    assert 'View this comparison' not in page
    assert stable_id(rep, tmp_path) in page
    assert 'rep0' not in page
    assert page.count('class=\"kv-ledger\"') == 0
    assert page.count('kv-ledger') == 1
    section(rep)
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in files}
    assert load_repetition_summary(rep / 'summary.json')['kv_events']['lifecycle'] == data['lifecycle']


def test_cpu_candidates_remain_separate_from_transfer_evidence(tmp_path):
    rep = setup_run(tmp_path, offload=32, rows=[event(0, 1, 'BlockStored', 'CPU'), event(1, 2, 'BlockRemoved'), event(2, 5, 'BlockStored')])
    data = prepare_report_data(rep)
    assert data['lifecycle']['cpu_backed_return_candidates'] == 1
    assert data['transfer_status'] == 'not_collected'
    page = render_run(rep)
    assert 'CPU-backed candidate' in page
    assert 'confirming a specific return requires its transfer plan and all worker completions' in page


def test_malformed_or_truncated_collection_is_not_complete(tmp_path):
    rep = setup_run(tmp_path, rows=[event(0, 1, 'BlockRemoved'), event(2, 5, 'BlockStored')], transport={'collection_complete': True, 'tail_confirmed': True})
    assert prepare_report_data(rep)['collection_status'] == 'partial'
    (rep / 'kv_events.jsonl').write_text('{bad')
    data = prepare_report_data(rep)
    assert data['collection_status'] == 'failed'
    assert data['lifecycle'] == {}
    assert section(rep) == ''


def test_unknown_medium_never_matches_a_gpu_store():
    rows = [event(0, 1, 'BlockStored', None), event(1, 2, 'BlockRemoved'), event(2, 3, 'BlockStored', None)]
    assert derive_removals(rows)[0]['matched_store'] is False
    _, lifecycle = derive_lifecycle(rows)
    assert lifecycle['return_episodes'] == 0
    assert lifecycle['unknown_medium_events'] == 2


def test_changed_source_invalidates_shared_artifacts(tmp_path):
    rep = setup_run(tmp_path, rows=[event(0, 1, 'BlockRemoved')])
    prepare_report_data(rep)
    with (rep / 'kv_events.jsonl').open('a') as stream:
        stream.write(json.dumps(event(1, 3, 'BlockStored')) + '\n')
    assert load_report_data(rep) == {}
    assert section(rep) == ''
    assert prepare_report_data(rep)['lifecycle']['return_episodes'] == 1


def test_task_integration_has_namespaced_ledgers_and_is_idempotent(tmp_path):
    rep = setup_run(tmp_path, rows=[event(0, 1, 'BlockRemoved')])
    report = write_report(rep.parent.parent, results_dir=tmp_path / 'results')
    page = report.read_text()
    assert f'id="{stable_id(rep, tmp_path)}"' in page
    assert 'Original observations' in page
    assert 'Fixed-input replay experiments' in page
    assert 'rep2' in page and 'rep0' not in page
    assert 'kv_block_report.html' not in page
    assert not (rep.parent.parent/'kvcache_report.html').exists()
    digest = hashlib.sha256(page.encode()).hexdigest()
    assert hashlib.sha256(write_report(rep.parent.parent, results_dir=tmp_path / 'results').read_bytes()).hexdigest() == digest


def test_comparison_alias_uses_one_canonical_run(tmp_path):
    rep = setup_run(tmp_path, rows=[event(0, 1, 'BlockRemoved')])
    write_report(rep.parent.parent)
    alias_sweep = tmp_path / 'replay' / 'alias-experiment'
    alias_sweep.mkdir()
    (alias_sweep / 'baseline').symlink_to(rep.parent, target_is_directory=True)
    catalog = Catalog(tmp_path)
    assert len(catalog.records) == 1
    page = refresh_task_pages(catalog)[0].read_text()
    assert page.count(f'id="{stable_id(rep, tmp_path)}"') == 1


def test_publication_audit_distinguishes_timeout_incident_from_sequence_loss():
    from agent_io_tracing.serving.kv_report_data import publication_audit
    common={'boundary_confirmed':True,'publisher_id':'p','publisher_epoch':0,'server_ack':{'publisher_epoch':'server'}}
    transport={'boundaries':[{**common,'phase':'measurement','next_sequence':5},{**common,'phase':'after_measurement','next_sequence':7}],'collection_errors':['timeout']}
    rows=[{'publisher_id':'p','publisher_epoch':0,'sequence':n} for n in [5,6]]
    assert publication_audit(rows,transport)['status']=='all_sequences_present'
    assert publication_audit(rows[:1],transport)['status']=='missing_sequences'
    transport['boundaries'][-1]['server_ack']={'publisher_epoch':'restarted'}
    assert publication_audit(rows,transport)['status']=='unconfirmed_boundaries'
