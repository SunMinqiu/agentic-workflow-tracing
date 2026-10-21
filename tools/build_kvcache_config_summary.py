"""Build the workflow meeting summary with direct IO and KVCache reports."""
from __future__ import annotations

import html
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from agent_io_tracing.analysis.kvcache.catalog import Catalog, COLORS
from agent_io_tracing.analysis.kvcache.replay_report import comparison_table, fmt, pct, refresh_task_pages, table, output_length_description
from agent_io_tracing.analysis.results_index import write_results_index, discover_tasks, IO_REPORT, KV_REPORT

CSS = '''
:root{--ink:#20303b;--muted:#5c6b74;--line:#d7e1e4;--teal:#006c65}
*{box-sizing:border-box}body{margin:0;background:#f4f6f7;color:var(--ink);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1200px;margin:auto;padding:36px 28px 60px}h1{font-size:32px;line-height:1.25;margin:0 0 14px}h2{font-size:24px;margin:0 0 14px}h3{font-size:18px;margin:24px 0 8px}
section{background:white;border:1px solid var(--line);padding:26px;margin:24px 0;border-radius:6px;scroll-margin-top:20px}p{margin:10px 0}.note{font-size:13px;color:var(--muted)}.finding{font-size:18px;font-weight:600}a{color:var(--teal);text-underline-offset:3px}header{border-bottom:2px solid var(--ink);padding-bottom:16px}
.kv-table{overflow-x:auto;margin:14px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid var(--line);padding:11px 12px;vertical-align:top;text-align:right}th{background:#edf3f3}td:first-child,th:first-child{text-align:left}td:last-child{white-space:nowrap}th{font-weight:650}.legend{font-size:13px;color:var(--muted)}.legend span{padding:3px 8px}.swap-key{background:#dff1ff}.missing-key{background:#fff6e3}footer{font-size:13px;color:var(--muted)}
@media(max-width:650px){main{padding:20px 12px}section{padding:16px}h1{font-size:26px}th,td{padding:9px;min-width:75px}td:first-child{min-width:130px}}
@media print{body{background:white}main{max-width:none;padding:0}section{break-inside:avoid}.kv-table{overflow:visible}footer a{display:none}}
'''


def workflow_reports(results: Path) -> str:
    rows = sorted(discover_tasks(results), key=lambda row: (row['workflow'], row['stamp'], row['cell']), reverse=True)
    parts = ['<section id="workflow-reports"><h2>Workflow results</h2>']
    for workflow in sorted({row['workflow'] for row in rows}):
        entries = []
        for row in rows:
            if row['workflow'] != workflow:
                continue
            links = []
            for filename, label in [(IO_REPORT, 'IO'), (KV_REPORT, 'KVCache')]:
                href = html.escape(row['rel'] + '/' + filename, quote=True)
                links.append(f'<a href="{href}">{label}</a>' if filename in row['available_reports'] else 'Not generated')
            stamp = row['stamp']
            date = f'{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]} {stamp[9:11]}:{stamp[11:13]}' if stamp else 'Unknown'
            entries.append([html.escape(row['cell']), date, html.escape(', '.join(row['runtime'].get('models') or ['Unknown'])), row['n_calls'], fmt(row['latency'].get('wall_clock_span_s'),1), *links])
        parts.append('<h3>' + html.escape(workflow) + '</h3>')
        parts.append(table(['Task','Run','Model','Calls','Workflow span s','IO','KVCache'], entries))
    return ''.join(parts) + '</section>'


def physical_section(catalog, results):
    parts = ['<section id="continuous-validation"><h2>Eviction and CPU swap</h2>']
    groups = {}
    for label, family in [('C2-12', 'kv_physical_c2-12_20260911'), ('C8', 'kv_physical_c8_20260911')]:
        for row in catalog.records:
            if row['family'] == family and not row['review'].get('exclude_from_comparison'):
                groups.setdefault(row['rep'].parent, (label, []))[1].append(row)
    for comparison in catalog.comparisons:
        for _, row in comparison['entries']:
            arm = row['rep'].parent
            if arm in groups or not catalog.data(row).get('physical_trace'):
                continue
            runs = [r for r in catalog.records if r['rep'].parent == arm
                    and not r['review'].get('exclude_from_comparison') and catalog.data(r).get('physical_trace')]
            groups[arm] = (comparison['title'], runs)
    physical = []
    for label, runs in groups.values():
        config = runs[0].get('serving_config') or {}
        cpu = config.get('kv_offloading_size')
        cpu = 0 if cpu in (None, 'None', '0', '0.0', 0) else cpu
        def median_value(get):
            values=[get(catalog.data(r)) for r in runs]
            return statistics.median(values) if all(isinstance(v,(int,float)) for v in values) else None
        def transfer_median(direction, field, scale=1):
            value = median_value(lambda data: ((data.get('aggregate_transfers') or {}).get('directions',{}).get(direction) or {}).get(field))
            return fmt(value / scale,3) if value is not None else None
        physical.append([html.escape(label),config.get('num_gpu_blocks'),cpu,len(runs),fmt(statistics.median(r['wall_s'] for r in runs),3),
                         fmt(median_value(lambda d:(d.get('physical_trace') or {}).get('physical_evictions')),0),
                         fmt(median_value(lambda d:((d.get('causal_trace') or {}).get('cpu_cache') or {}).get('eviction_events')),0),
                         transfer_median('GPU_to_CPU','bytes',1e9), transfer_median('GPU_to_CPU','worker_time_s'),
                         transfer_median('CPU_to_GPU','bytes',1e9), transfer_median('CPU_to_GPU','worker_time_s'),
                         catalog.link(runs[0],results,'config','KVCache')])
    parts.append('<p>Rows group repetitions of the same experiment configuration. Values are medians; Runs shows the number of measurements.</p>')
    parts.append(table(['Workload / experiment','GPU blocks','CPU GiB','Runs','Replay runtime s','GPU block evictions','CPU content evictions','GPU → CPU GB','Store worker time s','CPU → GPU GB','Load worker time s','Report'],physical))
    parts.append('<p class="note">Transfer bytes and worker times measure CPU swap. Physical eviction byte size and operation duration were not recorded. Independent restore controls remain in All experiment records.</p></section>')
    return ''.join(parts)


def build(root: Path = ROOT) -> Path:
    catalog = Catalog(root)
    refresh_task_pages(catalog)
    results = root/'results'
    results.mkdir(exist_ok=True)
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Workflow experiments</title><style>',CSS,COLORS,'</style></head><body><main><header><h1>Workflow experiments</h1><p>IO and KVCache results</p></header>']
    parts.append(workflow_reports(results))
    parts.append('<section id="workloads"><h2>KVCache replay workloads</h2>')
    workloads=[]
    for comparison in catalog.comparisons:
        if comparison['overview'] != 'capacity':
            continue
        row=comparison['baseline_record']
        source=row['source']
        timing=json.loads((source/'parallelism_summary.json').read_text()) if (source/'parallelism_summary.json').is_file() else {}
        label=comparison['title'].split(' · ')[0]
        workloads.append([label,row.get('requests'),fmt(row.get('prompt_tokens'),0),html.escape(str(row.get('mode'))),html.escape(output_length_description(row)),fmt(timing.get('wall_clock_s'),1)])
    parts.append(table(['Workload','Requests','Input tokens','Replay mode','Output tokens / request','Original workflow span s'],workloads))
    parts.append('<p class="note">C2 and C8 use 2 and 8 cohorts. Output lengths follow the per-workload setting shown above.</p></section>')
    for key,title in [('capacity','Cache reuse and capacity'),('precision','KV precision')]:
        parts.append(f'<section id="{key}"><h2>{title}</h2>')
        parts.append('<p class="legend"><span class="swap-key">Light blue: CPU swap with eviction data</span> · <span class="missing-key">Yellow: missing measurement</span></p>')
        for comparison in catalog.comparisons:
            if comparison['overview'] != key:
                continue
            label=comparison['title'].split(' · ')[0]
            baseline=comparison['baseline_record']
            parts.append(f'<h3>{html.escape(label)}</h3>')
            if key=='capacity':
                swap=next((r for _,r in comparison['entries'] if r['swap']),None)
                if swap:
                    saved=(1-swap['wall_s']/baseline['wall_s'])*100
                    gain=(swap['measurement']['rate']-baseline['measurement']['rate'])*100 if swap['measurement']['rate'] is not None and baseline['measurement']['rate'] is not None else None
                    finding=f'CPU swap: {abs(saved):.1f}% {"shorter" if saved >= 0 else "longer"} replay runtime'
                    if gain is not None:
                        finding+=f'; cache hit ratio {gain:+.1f} percentage points'
                    parts.append('<p class="finding">'+finding+'.</p>')
            parts.append(comparison_table(catalog,comparison,results))
        parts.append('<p class="note">Relative runtime uses the baseline named in each workload’s comparison. These comparisons have one run per configuration.</p>')
        if key=='precision':
            parts.append('<p class="note">FP8 also changes block geometry, so these results reflect the combined configuration change.</p>')
        parts.append('</section>')
    parts.append(physical_section(catalog, results))
    parts.append('<footer id="reports"><a href="all-experiments.html">All experiment records</a></footer></main></body></html>')
    path=results/'index.html'
    path.write_text(''.join(parts),encoding='utf-8')
    write_results_index(results,catalog)
    return path


if __name__=='__main__':
    print(build())
