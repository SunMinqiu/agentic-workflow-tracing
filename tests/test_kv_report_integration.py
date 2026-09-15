"""Task ownership, metric provenance, colors and report migration contracts."""
import json
import re
from pathlib import Path

import pytest

from agent_io_tracing.analysis.kvcache.catalog import Catalog, cache_measurement, COLORS
from agent_io_tracing.analysis.kvcache.replay_report import integrate_page, render_replays, comparison_table
from agent_io_tracing.serving.kv_report_data import prepare_report_data
from tools.rebuild_kv_reports import rebuild
from tools.check_kv_report_links import check


def make_run(root, experiment='one', task='task', swap=32, events=None, coverage=None):
    source = root/'results'/f'GenoMAS_{task}_20260914_120000'/task
    source.mkdir(parents=True, exist_ok=True)
    (source/'messages.jsonl').write_text('')
    page=source/'kvcache_report.html'
    if not page.exists():
        page.write_text('<html><head></head><body><h1>Original task</h1><h2>Per task</h2><p>Original observation</p></body></html>')
    rep=root/'replay'/experiment/'same_arm'/'rep2'
    rep.mkdir(parents=True)
    bundle=root/'replay'/f'{task}.json'
    bundle.write_text(json.dumps({'source_cell':str(source)}))
    summary={'bundle':str(bundle),'requests':1,'mode':'serial','wall_s':10,'prompt_tokens':100,'output_tokens':2,
             'per_request_cache':{'coverage':1,'cached_tokens':60,'hit_rate':.6},
             'prefix_cache':{'queries':100,'hits':40,'hit_rate':.4},
             'serving_config':{'kv_offloading_size':swap,'num_gpu_blocks':274},'kv_events':coverage or {}}
    (rep/'summary.json').write_text(json.dumps(summary))
    if events is not None:
        (rep/'kv_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in events))
    prepare_report_data(rep)
    return rep,source


def test_cache_source_prefers_complete_requests_and_rejects_shared_engine():
    assert cache_measurement({'prompt_tokens':100,'per_request_cache':{'coverage':1,'cached_tokens':60},'prefix_cache':{'queries':100,'hits':40}})['rate']==.6
    assert cache_measurement({'prompt_tokens':100,'prefix_cache':{'queries':200,'hits':50}})['rate'] is None
    assert cache_measurement({'prompt_tokens':100,'prefix_cache':{'queries':100}})['rate'] is None
    assert cache_measurement({'prompt_tokens':100,'prefix_cache':{'queries':100,'hits':0}})['rate']==0


@pytest.mark.parametrize('swap,events,coverage,blue',[
    (32,[],{'collection_complete':True,'tail_confirmed':True},True),
    (32,None,{},False),
    (0,[],{'collection_complete':True,'tail_confirmed':True},False),
    (32,[],{},False),
])
def test_blue_means_swap_with_eviction_measurements(tmp_path,swap,events,coverage,blue):
    rep,source=make_run(tmp_path,swap=swap,events=events,coverage=coverage)
    c=Catalog(tmp_path)
    assert c.blue(c.by_rep[rep]) is blue
    assert '!important' not in COLORS
    assert 'tr>td.blank' in COLORS


def test_same_names_across_experiments_have_unique_ids_and_owners(tmp_path):
    a,source=make_run(tmp_path,'one')
    b,_=make_run(tmp_path,'two')
    other,other_source=make_run(tmp_path,'three','other')
    c=Catalog(tmp_path)
    page=integrate_page((source/'kvcache_report.html').read_text(),c,source)
    ids=re.findall(r'\bid="([^"]+)"',page)
    assert len(ids)==len(set(ids))
    assert c.by_rep[a]['id'] in ids and c.by_rep[b]['id'] in ids
    assert c.by_rep[other]['id'] not in ids
    assert 'Original observation' in page
    assert integrate_page(page,c,source)==page


def test_rebased_evidence_ids_and_sources_stay_in_their_run(tmp_path):
    event={'sequence':0,'batch_ts':1,'event_type':'BlockRemoved','medium':'GPU','block_hashes':['a'],'group_idx':0,'data_parallel_rank':0,'block_size':1}
    rep,source=make_run(tmp_path,events=[event])
    c=Catalog(tmp_path)
    row=c.by_rep[rep]
    page=render_replays(c,source,source)
    assert row['id']+'-block-events' in page
    assert 'class="kv-evidence-data"' in page
    assert 'data-evidence-prefix="'+row['id']+'"' in page
    assert 'replay/one/same_arm/rep2/kv_events.jsonl' in page
    assert 'id="block-events"' not in page


def test_rebuild_migrates_and_deletes_old_pages_without_touching_raw_data(tmp_path):
    rep,source=make_run(tmp_path,events=[],coverage={'collection_complete':True,'tail_confirmed':True})
    old=rep.parent.parent/'kvcache_report.html'
    old.write_text('<html>Old sweep</html>')
    block=rep/'kv_block_report.html'
    block.write_text('<section id="block-events"></section>')
    note=tmp_path/'notes.md'
    note.write_text('[Evidence](replay/one/same_arm/rep2/kv_block_report.html#block-events)')
    before=(rep/'summary.json').read_bytes()
    result=rebuild(tmp_path)
    assert not old.exists() and not block.exists()
    assert len(result['removed_reports'])==2
    assert 'results/GenoMAS_task_' in note.read_text()
    assert (rep/'summary.json').read_bytes()==before
    assert check(tmp_path)['errors']==[]
    assert rebuild(tmp_path)['removed_reports']==result['removed_reports']


def test_comparison_rejects_different_task_and_shares_baseline(tmp_path):
    a,source=make_run(tmp_path,'one')
    b,other=make_run(tmp_path,'two','other')
    config=tmp_path/'config';config.mkdir()
    spec={'id':'comparison','title':'Capacity','overview':'capacity','baseline':str(a.relative_to(tmp_path)),
          'runs':[{'label':'Baseline','rep':str(a.relative_to(tmp_path))},{'label':'Other','rep':str(b.relative_to(tmp_path))}]}
    (config/'kv_report_comparisons.json').write_text(json.dumps({'comparisons':[spec]}))
    with pytest.raises(ValueError,match='Incompatible comparison'):
        Catalog(tmp_path)
    spec['runs']=spec['runs'][:1]
    (config/'kv_report_comparisons.json').write_text(json.dumps({'comparisons':[spec]}))
    c=Catalog(tmp_path)
    for base in (source,tmp_path/'results'):
        page=comparison_table(c,c.comparisons[0],base)
        assert '60.00%' in page and '1.000' in page
        assert c.by_rep[a]['id'] in page


def test_meeting_summary_keeps_direct_io_and_kvcache_entries(tmp_path):
    from tools.build_kvcache_config_summary import workflow_reports
    _, source = make_run(tmp_path)
    io = source/'visualizations/index.html'
    io.parent.mkdir()
    io.write_text('<html>IO report</html>')
    page = workflow_reports(tmp_path/'results')
    rel = source.relative_to(tmp_path/'results').as_posix()
    assert f'href="{rel}/visualizations/index.html">IO</a>' in page
    assert f'href="{rel}/kvcache_report.html">KVCache</a>' in page
    assert 'all-experiments.html' not in page
    assert page.count('<a ') == 2


def test_uncollected_run_has_no_eviction_section(tmp_path):
    from agent_io_tracing.analysis.kvcache.replay_report import run_section
    rep, source = make_run(tmp_path, swap=0)
    catalog = Catalog(tmp_path)
    row = catalog.by_rep[rep]
    page = run_section(catalog, row, source)
    assert f'id="{row["id"]}-eviction"' not in page
    assert 'Event records' not in page
    assert catalog.href(row, source, 'eviction').endswith('#' + row['id'])


def test_swap_size_time_does_not_claim_eviction_bytes(tmp_path):
    from agent_io_tracing.serving.kv_lifecycle_report import physical_section, transfer_section
    (tmp_path/'physical_transfers.jsonl').write_text('\n'.join(json.dumps(row) for row in [
        {'success': True, 'direction': 'store', 'transfer_time_s': .002},
        {'success': True, 'direction': 'store', 'transfer_time_s': .004},
        {'success': False, 'direction': 'store', 'transfer_time_s': 100},
    ]))
    data = {'aggregate_transfers': {'directions': {'GPU_to_CPU': {'bytes': 2000000000, 'worker_time_s': .006, 'worker_samples': 2}}},
            'physical_trace': {'physical_evictions': 3, 'scope': 'Matched requests', 'coverage_note': 'Partial'}}
    page = transfer_section(tmp_path, data)
    assert '<td>2.000</td><td>0.006</td><td>2</td><td>2</td><td>3.000</td><td>3.900</td>' in page
    physical = physical_section(tmp_path, data)
    assert '<td>Evicted size in bytes</td><td>Not recorded</td>' in physical
    assert '<td>Eviction operation duration</td><td>Not recorded</td>' in physical
