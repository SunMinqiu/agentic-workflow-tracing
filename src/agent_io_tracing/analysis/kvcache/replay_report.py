"""Embed replay experiments and their evidence in the source task's KV report."""
from __future__ import annotations

import html
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit, unquote

from agent_io_tracing.analysis.kvcache.catalog import Catalog, COLORS, LEGEND, REPORT, cold_state, eviction_available
from agent_io_tracing.serving.kv_lifecycle_report import REPORT_CSS, STATUS, causal_section, physical_section, section, transfer_section

START = '<!-- task-kv-replays:start -->'
END = '<!-- task-kv-replays:end -->'
STYLE_START = '<!-- task-kv-style:start -->'
STYLE_END = '<!-- task-kv-style:end -->'
CSS = COLORS + REPORT_CSS + '''
.kv-replays{border-top:2px solid #006c65;margin-top:24px;padding-top:16px}
.kv-replays summary{cursor:pointer;font-weight:650;padding:10px 0}
.kv-replays details{border:1px solid #d7e1e4;border-radius:6px;padding:0 14px;margin:14px 0}
.kv-replays .run-identity{font-size:14px;color:#52616b}
.kv-replays a{color:#006c65;text-underline-offset:3px}
.kv-replays :target{outline:2px solid #7d6b45;outline-offset:4px}
.kv-replays img{content-visibility:auto}
.kv-replays table{font-size:14px}
.kv-replays th,.kv-replays td{vertical-align:top;min-width:70px}
.kv-replays nav{display:flex;flex-wrap:wrap;gap:12px}
'''
SCRIPT = '''<script>
(() => {
  const mount = holder => {
    const data = holder.querySelector('script.kv-evidence-data');
    if (!data) return;
    const content = document.createElement('div');
    content.innerHTML = JSON.parse(data.textContent);
    data.replaceWith(content);
    const filters = content.querySelector('.kv-filters');
    if (!filters) return;
    const hash = filters.querySelector('input');
    const selects = filters.querySelectorAll('select');
    const [group, state] = selects;
    const apply = () => {
      let visible = 0;
      content.querySelectorAll('.kv-ledger').forEach(section => {
        let count = 0;
        section.querySelectorAll('tbody tr').forEach(row => {
          row.hidden = !row.dataset.block.toLowerCase().includes(hash.value.toLowerCase()) || (group.value && row.dataset.group !== group.value) || (state.value && section.dataset.state !== state.value);
          if (!row.hidden) count++;
        });
        section.hidden = count === 0;
        visible += count;
      });
      filters.querySelector('p').textContent = visible + ' matching episodes';
    };
    [hash, group, state].forEach(input => input.addEventListener('input', apply));
    apply();
  };
  document.addEventListener('toggle', event => {
    if (event.target.matches('details.kv-evidence') && event.target.open) mount(event.target);
  }, true);
  const reveal = () => {
    let id;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
    let target = document.getElementById(id);
    if (!target) {
      const holder = [...document.querySelectorAll('details.kv-evidence')].find(e => id.startsWith(e.dataset.evidencePrefix + '-'));
      if (holder) { mount(holder); target = document.getElementById(id); }
    }
    if (!target) return;
    let node = target;
    while (node) { if (node.tagName === 'DETAILS') node.open = true; node = node.parentElement; }
    requestAnimationFrame(() => target.scrollIntoView({block:'start'}));
  };
  addEventListener('hashchange', reveal);
  if (document.readyState === 'loading') addEventListener('DOMContentLoaded', reveal); else reveal();
})();
</script>'''


def fmt(value, digits=2):
    return f'{value:,.{digits}f}' if isinstance(value, (int, float)) else None


def pct(value):
    return f'{value:.2%}' if isinstance(value, (int, float)) else None


def table(headers, rows):
    if not rows:
        return ''
    def cell(value):
        return '<td class="blank">Not recorded</td>' if value is None else '<td>' + str(value) + '</td>'
    return '<div class="kv-table"><table><thead><tr>' + ''.join('<th>' + h + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(cell(v) for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def comparison_table(catalog, comparison, base):
    baseline = comparison['baseline_record']
    rows = []
    for label, row in comparison['entries']:
        runtime = row.get('wall_s')
        ratio = runtime / baseline['wall_s'] if runtime is not None and baseline.get('wall_s') else None
        rows.append([html.escape(label), html.escape(str((row.get('serving_config') or {}).get('num_gpu_blocks', 'Not recorded'))),
                     pct(row['measurement']['rate']), fmt(row.get('server_ttft'), 1), fmt(row.get('server_ttft_p90'), 1),
                     fmt(runtime, 3), fmt(ratio, 3), catalog.link(row, base, 'eviction' if catalog.blue(row) else 'run', 'KVCache')])
    return table(['Configuration', 'GPU blocks', 'Hit ratio', 'Server TTFT p50 ms', 'Server TTFT p90 ms', 'Replay runtime s', 'Relative runtime', 'KVCache details'], rows)


def output_length_description(row):
    if row.get('output_length_mode') == 'recorded':
        return 'recorded output length for each request'
    count = row.get('fixed_output_tokens_per_request')
    return f'{count} output tokens per request' if count is not None else 'output length not recorded'


def comparison_section(catalog, comparison, base):
    baseline = comparison['baseline_record']
    cfg = baseline.get('serving_config') or {}
    cpu = cfg.get('kv_offloading_size')
    cpu_label = f'CPU offload {cpu} GiB' if cpu and str(cpu) not in ('None', '0', '0.0') else 'no CPU offload'
    identity = html.escape(f"{cfg.get('cache_dtype')} · {cfg.get('num_gpu_blocks')} GPU blocks · {cpu_label}")
    mode = html.escape(str(baseline.get('mode')))
    output = html.escape(output_length_description(baseline))
    rows = comparison['entries']
    evidence = []
    for label, row in rows:
        if row['rep'] == baseline['rep'] or not baseline.get('wall_s') or row.get('wall_s') is None:
            continue
        change = (row['wall_s'] / baseline['wall_s'] - 1) * 100
        evidence.append(f'{html.escape(label)}: {abs(change):.2f}% {"longer" if change > 0 else "shorter"} runtime')
    conclusion = '<p>' + '; '.join(evidence) + '. These are the recorded single-run comparisons.</p>' if evidence else ''
    note = 'FP8 also changes block geometry; these rows compare the combined configuration.' if comparison['overview'] == 'precision' else 'Unique cache salts prevent reuse across requests; engine prefix caching remains enabled.'
    note = html.escape(comparison.get('note', note))
    geometry = ''
    if comparison['overview'] == 'precision':
        geometry = '<details><summary>KV precision and block geometry</summary>' + table(['Configuration','KV dtype','Block tokens','GPU blocks','GPU utilization','TPOT ms/token'], [[html.escape(label),html.escape(str((row.get('serving_config') or {}).get('cache_dtype'))),(row.get('serving_config') or {}).get('block_size'),(row.get('serving_config') or {}).get('num_gpu_blocks'),(row.get('serving_config') or {}).get('gpu_memory_utilization'),fmt(row.get('median_tpot_ms'))] for label,row in rows]) + '</details>'
    return (f'<section id="{comparison["id"]}"><h3>{html.escape(comparison["title"])}</h3>'
            f'<p>{mode} replay · {output} · baseline: {identity}.</p>'
            + conclusion + comparison_table(catalog, comparison, base) + geometry
            + f'<p>{note} Reset-only records do not confirm an empty initial cache. Hit ratio uses complete per-request measurements when available; evidence retains engine counters separately.</p></section>')


def rebase_evidence(fragment: str, rep: Path, base: Path, prefix: str) -> str:
    ids = re.findall(r'\bid="([^"]+)"', fragment)
    for old in sorted(ids, key=len, reverse=True):
        new = prefix + '-' + old
        fragment = fragment.replace(f'id="{old}"', f'id="{new}"').replace(f"getElementById('{old}')", f"getElementById('{new}')").replace(f'href="#{old}"', f'href="#{new}"')
    def attr(match):
        key, raw = match.groups()
        target = html.unescape(raw)
        url = urlsplit(target)
        if url.scheme or url.netloc or not url.path:
            return match.group(0)
        path = os.path.relpath((rep / unquote(url.path)).resolve(), base)
        if url.fragment:
            path += '#' + url.fragment
        return key + '="' + html.escape(path, quote=True) + '"'
    fragment = re.sub(r'(href|src)="([^"]+)"', attr, fragment)
    fragment = re.sub(r'<(/?)h([23])>', lambda m: f'<{m[1]}h{int(m[2])+2}>', fragment)
    return fragment.replace('<img ', '<img loading="lazy" ')


def run_section(catalog, row, base):
    data = catalog.data(row)
    status = STATUS.get(data.get('collection_status'), 'Analysis not built')
    config = row.get('serving_config') or {}
    cpu_size = config.get('kv_offloading_size')
    cpu_label = 'Off' if cpu_size in (None, 'None', '0', '0.0', 0) else cpu_size
    title = row['rep'].parent.name + ' / ' + row['rep'].name
    source = row['source'].relative_to(catalog.root / 'results').as_posix() if row['source'] else row['source_raw'] or 'Source unresolved'
    rid = row['id']
    parts = [f'<details class="kv-run" id="{rid}"><summary>{html.escape(title)} · {fmt(row.get("wall_s"),3)} s</summary>',
             f'<p class="run-identity">Task: {html.escape(source)}<br>Experiment: {html.escape(row["family"])} · Run: {html.escape(title)}. These observations belong to this replay.</p>',
             '<p><a href="' + os.path.relpath(catalog.root / 'results/index.html', base) + '">Back to summary</a></p>']
    if row['review'].get('exclude_from_comparison'):
        parts.append('<p><strong>Excluded from comparison.</strong> ' + html.escape(row['review'].get('exclusion_reason', '')) + '</p>')
    parts.append(table(['Requests','GPU blocks','CPU swap GiB','Hit ratio','Client TTFT p50 ms','Replay runtime s','Collection'], [[row.get('requests'),config.get('num_gpu_blocks'),cpu_label,pct(row['measurement']['rate']),fmt(row.get('median_ttft_ms'),3),fmt(row.get('wall_s'),3),html.escape(status)]]))
    parts.append('<p>Hit ratio source: ' + html.escape(row['measurement']['source']) + '. Initial cache: ' + html.escape(cold_state(row)) + '.</p>')
    for comparison in catalog.comparisons:
        if any(r['rep'] == row['rep'] for _, r in comparison['entries']):
            baseline = comparison['baseline_record']
            ratio = row.get('wall_s', 0) / baseline['wall_s'] if baseline.get('wall_s') else None
            parts.append(f'<p>{html.escape(comparison["title"])}: relative runtime {fmt(ratio,3)} against {html.escape(baseline["rep"].parent.name)}.</p>')
    if eviction_available(data):
        physical = data.get('physical_trace') or {}
        logical = data.get('lifecycle') or {}
        heading = 'Physical KVCache eviction' if physical else 'Logical cache removals'
        if row['swap']:
            heading += ' and CPU swap'
        parts.append(f'<section id="{rid}-eviction"><h4>{heading}</h4>')
        if physical:
            parts.append(rebase_evidence(physical_section(row['rep'], data), row['rep'], base, rid))
        elif data.get('logical_events_available'):
            parts.append(table(['Logical removals','Logical returns'], [[logical.get('removal_episodes'),logical.get('return_episodes')]]))
            parts.append('<p>Physical eviction size and operation time were not recorded.</p>')
        if row['swap']:
            parts.append(rebase_evidence(transfer_section(row['rep'], data), row['rep'], base, rid))
        content = (causal_section(row['rep'], data) if row['swap'] else '') + section(row['rep'])
        if content:
            embedded = json.dumps(rebase_evidence(content, row['rep'], base, rid)).replace('<', '\\u003c')
            parts.append(f'<details class="kv-evidence" data-evidence-prefix="{rid}"><summary>Event records</summary><script type="application/json" class="kv-evidence-data">' + embedded + '</script></details>')
        parts.append('</section>')
    elif row['swap'] and (data.get('aggregate_transfers') or {}).get('directions'):
        parts.append(rebase_evidence(transfer_section(row['rep'], data), row['rep'], base, rid))
    parts.append('<details><summary>Measurements and configuration</summary>')
    engine = row.get('prefix_cache') or {}
    request = row.get('per_request_cache') or {}
    parts.append(table(['Measurement','Cached tokens','Query / input tokens','Hit ratio'], [['Per-request',request.get('cached_tokens'),row.get('prompt_tokens'),pct(row['measurement']['rate']) if row['measurement']['source']=='per-request' else None],['Engine counters',engine.get('hits'),engine.get('queries'),pct(engine.get('hit_rate'))]]))
    if request.get('matches_prometheus') is False:
        parts.append('<p>The per-request and engine counts differ. The main result uses complete per-request measurements; the engine counters are retained here as separate evidence.</p>')
    parts.append(table(['Setting','Value'], [[html.escape(k),html.escape(str(v))] for k,v in config.items()]))
    paths = ['summary.json','requests.jsonl','server_info.json','metrics_before.prom','metrics_after.prom','kv_report_data.json']
    parts.append('<p>' + ' · '.join('<a href="' + html.escape(os.path.relpath(row['rep']/name,base),quote=True) + '">' + name + '</a>' for name in paths if (row['rep']/name).is_file()) + '</p></details></details>')
    return ''.join(parts)


def render_replays(catalog, source, base):
    records = catalog.task_records(source)
    if not records:
        return ''
    parts = [START, '<section class="kv-replays" id="replay-experiments"><h2>Fixed-input replay experiments</h2><p>Replay measurements use the task inputs with the recorded serving configuration and output limit. Original workflow observations remain in the original execution sections.</p>', LEGEND]
    for sweep in sorted({r['rep'].parent.parent for r in records}):
        figure = sweep/'visualizations/physical_capacity.png'
        owners = {r['source'] for r in catalog.records if r['rep'].parent.parent == sweep}
        if figure.is_file() and len(owners) == 1:
            href = os.path.relpath(figure, base)
            parts.append('<details><summary>Physical capacity comparison: ' + html.escape(sweep.name) + '</summary><p>Circles show individual runs; bars show medians. Completed prompt computation includes first computation and recomputation.</p><a href="' + href + '"><img loading="lazy" src="' + href + '" alt="Runtime, completed computation and physical eviction by capacity"></a><p><a href="' + os.path.relpath(figure.with_suffix('.json'),base) + '">Exact plotted values</a></p></details>')
    groups = defaultdict(list)
    for row in records:
        groups[row['rep'].parent].append(row)
    for arm, runs in groups.items():
        first = runs[0]
        parts.append(f'<details id="{first["arm_id"]}"><summary>{html.escape(arm.parent.name)} · {html.escape(arm.name)} · {len(runs)} run(s)</summary>')
        parts.append('<p>Mode: ' + html.escape(str(first.get('mode'))) + '; ' + html.escape(output_length_description(first)) + '.</p>')
        if len(runs) > 1:
            result_rows = []
            for row in runs:
                data = catalog.data(row)
                physical = data.get('physical_trace') or {}
                result_rows.append([html.escape(row['rep'].name), fmt(row.get('wall_s'),3), pct(row['measurement']['rate']), physical.get('physical_evictions'), 'Excluded' if row['review'].get('exclude_from_comparison') else 'Included'])
            parts.append(table(['Run','Runtime s','Hit ratio','Physical GPU evictions','Comparison'], result_rows))
        parts.extend(run_section(catalog,row,base) for row in runs)
        parts.append('</details>')
    parts.extend(['</section>',SCRIPT,END])
    return ''.join(parts)


def integrate_page(page: str, catalog: Catalog, source: Path) -> str:
    page = re.sub(re.escape(START)+r'.*?'+re.escape(END),'',page,flags=re.S)
    page = re.sub(re.escape(STYLE_START)+r'.*?'+re.escape(STYLE_END),'',page,flags=re.S)
    fragment = render_replays(catalog,source,source)
    if not fragment:
        return page
    style = STYLE_START + '<style>' + CSS + '</style>' + STYLE_END
    page = page.replace('</head>',style+'</head>')
    marker = '<h2>Per task</h2>'
    if marker in page:
        page = page.replace(marker,fragment+'<h2 id="original-details">Original execution details</h2>',1)
    elif '<h2 id="original-details">' in page:
        page = page.replace('<h2 id="original-details">',fragment+'<h2 id="original-details">',1)
    else:
        page = page.replace('</body>',fragment+'</body>')
    nav = '<nav id="kv-task-nav"><a href="#replay-experiments">Replay experiments</a> · <a href="#original-details">Original execution details</a> · <a href="' + os.path.relpath(catalog.root/'results/index.html',source) + '">Experiments overview</a></nav>'
    page = re.sub(r'<nav id="kv-task-nav">.*?</nav>','',page,flags=re.S)
    return re.sub(r'(</h1>)',lambda m:m[0]+nav,page,count=1)


def refresh_task_pages(catalog: Catalog) -> list[Path]:
    from agent_io_tracing.replay.sweep import _write_physical_capacity_plot
    for sweep in sorted({r['rep'].parent.parent for r in catalog.records}):
        records = [r for r in catalog.records if r['rep'].parent.parent == sweep]
        if len({r['source'] for r in records}) != 1 or not any(catalog.data(r).get('physical_trace') for r in records):
            continue
        capacities = {(r.get('serving_config') or {}).get('num_gpu_blocks') for r in records if not r['swap']}
        if len(capacities) < 2:
            continue
        image = sweep/'visualizations/physical_capacity.png'
        newest = max((r['rep']/'kv_report_data.json').stat().st_mtime for r in records)
        if not image.is_file() or image.stat().st_mtime < newest:
            _write_physical_capacity_plot(sweep,[{'label':a} for a in sorted({r['rep'].parent.name for r in records})])
    pages = []
    for source in sorted({r['source'] for r in catalog.records if r['source']}):
        path = source / REPORT
        if not path.is_file():
            from agent_io_tracing.analysis.results_index import task_row
            from agent_io_tracing.analysis.kvcache.report import build_report
            build_report(source,[task_row(source,catalog.root/'results')],catalog.root/'results',cells_root=source.parent,include_replays=False)
        path.write_text(integrate_page(path.read_text(),catalog,source),encoding='utf-8')
        pages.append(path)
    return pages


def refresh_for_run(rep: Path) -> Path:
    from agent_io_tracing.analysis.kvcache.catalog import root_for
    root = root_for(rep)
    catalog = Catalog(root)
    row = catalog.by_rep.get(rep.resolve())
    if row is None and (rep/'summary.json').is_file():
        catalog = Catalog(root, sources=[rep/'summary.json'])
        row = catalog.by_rep.get(rep.resolve())
    publish_catalog(catalog)
    if row and row['source']:
        return catalog.page(row)
    return root/'results/index.html'


def publish_catalog(catalog: Catalog) -> Path:
    """Refresh task reports and both navigation entry points."""
    import runpy
    root = catalog.root
    builder = root/'tools/build_kvcache_config_summary.py'
    if builder.is_file() and (root/'config/kv_report_comparisons.json').is_file():
        runpy.run_path(str(builder),run_name='__main__')
        return root/'results/index.html'
    from agent_io_tracing.analysis.results_index import write_results_index
    results = root/'results'
    results.mkdir(parents=True,exist_ok=True)
    refresh_task_pages(catalog)
    write_results_index(results,catalog)
    rows = [[html.escape(r['source'].name if r['source'] else r['source_kind']),catalog.link(r,results)] for r in catalog.records]
    comparisons = ''.join(comparison_section(catalog,c,results) for c in catalog.comparisons)
    page = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>KVCache experiments</title><style>' + CSS + '</style></head><body><h1>KVCache experiments</h1>' + LEGEND
    page += '<section id="capacity"><h2>Capacity comparisons</h2>' + comparisons + '</section><section id="continuous-validation"><h2>Recorded runs</h2>' + table(['Source','KVCache'],rows) + '</section><section id="supplement">' + '</section><p><a href="all-experiments.html">Full experiment index</a></p></body></html>'
    target = results/'index.html'
    target.write_text(page,encoding='utf-8')
    return target
