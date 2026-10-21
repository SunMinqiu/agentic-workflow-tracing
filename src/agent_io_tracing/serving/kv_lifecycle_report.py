"""Render shared KV analysis with configuration and repetition navigation."""
from __future__ import annotations

import html
import os
import statistics
from pathlib import Path

from agent_io_tracing.serving.kv_report_data import load_report_data, read_json, read_rows

REPORT_CSS = 'body{max-width:1220px;overflow-wrap:anywhere} h1,h2,h3,h4,a{overflow-wrap:anywhere} .kv-table{overflow:auto;max-width:100%} table{display:block;overflow-x:auto;max-width:100%} th,td{overflow-wrap:normal;word-break:normal;min-width:85px} .kv-ledger .kv-table{max-height:600px} figure{margin:20px 0} figure img{display:block;max-width:100%;height:auto} .kv-table th{position:sticky;top:0;background:#edf4f8} details{margin:18px 0} nav{margin:16px 0} section{scroll-margin-top:20px} @media(max-width:600px){h1{font-size:24px}}'
STATUS = {"complete": "Complete", "partial": "Partial collection", "unconfirmed": "Recorded; completeness unconfirmed", "not_collected": "Not collected", "failed": "Collection or analysis failed"}


def link(path: Path, base: Path, label: str, fragment: str = '') -> str:
    href = os.path.relpath(path, base) + (f'#{fragment}' if fragment else '')
    return f'<a href="{html.escape(href, quote=True)}">{html.escape(label)}</a>'


def table(headers: list[str], rows: list[list[object]], *, attributes: list[str] | None = None) -> str:
    if not rows:
        return ''
    def cell(value: object) -> str:
        return html.escape(str(value) if value is not None else 'Unknown')
    return '<div class="kv-table"><table><thead><tr>' + ''.join(f'<th>{cell(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr' + (attributes[index] if attributes else '') + '>' + ''.join(f'<td>{cell(v)}</td>' for v in row) + '</tr>' for index, row in enumerate(rows)) + '</tbody></table></div>'


def number(value: object) -> str:
    return f'{value:,.3f}' if isinstance(value, (int, float)) else 'Unknown'


def section(rep: Path) -> str:
    """Render one repetition without changing analysis files."""
    data = load_report_data(rep)
    if not data.get('logical_events_available'):
        return ''
    stats = data['lifecycle']
    entries = read_rows(rep / 'kv_lifecycle.jsonl')
    cpu = data['show_cpu']
    causal_returns=bool(sum((data.get('causal_trace',{}).get('candidate_reconciliation') or {}).values()))
    timing = stats['removal_to_next_store_s']
    fields = [
        ('Unique removed logical blocks', stats['unique_removed_blocks']),
        ('Removal episodes', stats['removal_episodes']),
        ('Unique returned logical blocks', stats['unique_returned_blocks']),
        ('Return episodes', stats['return_episodes']),
        ('Right-censored episodes', stats['right_censored_episodes']),
        ('Reset-terminated episodes', stats['reset_terminated_episodes']),
        ('Gap-terminated episodes', stats.get('gap_terminated_episodes', 0)),
    ]
    if cpu:
        fields.extend([('CPU-backed candidate blocks', stats['unique_cpu_backed_return_candidates']), ('CPU-backed candidate episodes', stats['cpu_backed_return_candidates'])])
    if stats['return_episodes']:
        fields.extend((f'Removal-to-next-store {q} s', number(timing[q])) for q in ('p50', 'p95', 'max'))
    parts = ['<section id="block-events"><h2>Logical block events</h2>',
             '<p>BlockRemoved → next BlockStored for the same logical content. These events do not identify individual physical evictions or confirm transfers.</p>',
             f'<p>Collection: {STATUS[data["collection_status"]]}.</p>']
    parts.extend('<p>' + html.escape(note) + '</p>' for note in data['quality_notes'])
    audit=data.get('publication_window_audit') or {}
    if audit.get('status')=='all_sequences_present':
        parts.append(f'<p>All {audit["expected_batch_sequences"]} batch sequence numbers between the acknowledged publication boundaries are present. Collector incidents are reported separately above.</p>')
    parts.append(table(['Metric', 'Observed value'], [[k, v] for k, v in fields]))
    parts.append(f'<p>Interval quantiles use only the {stats["distribution_sample_size"]:,} returned episodes. Right-censored, reset-terminated, and gap-terminated episodes are excluded. Times use publisher batch timestamps.</p>')
    if cpu:
        parts.append('<p>CPU-backed returns are candidates. Aggregate transfer counters describe traffic; confirming a specific return requires its transfer plan and all worker completions.</p>')
    for figure in data.get('figures', []):
        parts.append(f'<figure><a href="{html.escape(figure["path"], quote=True)}"><img src="{html.escape(figure["path"], quote=True)}" alt="{html.escape(figure["title"], quote=True)}"></a><figcaption>{html.escape(figure["title"])}</figcaption></figure>')
    parts.append('<p>' + ' · '.join(link(rep / name, rep, label) for name, label in [('kv_report_data.json', 'Analysis and coverage'), ('kv_lifecycle.jsonl', 'Lifecycle JSONL'), ('kv_removals.jsonl', 'Removal JSONL'), ('kv_events.jsonl', 'Raw events')] if (rep / name).is_file()) + '</p>')
    phased = bool((data.get('source_transport') or {}).get('boundaries'))
    start = None if phased else data.get('window_start_epoch_s')
    if start is None:
        start = 0
        parts.append('<p>Ledger timestamps are publisher epoch seconds; cross-clock alignment is unconfirmed.</p>')
    else:
        parts.append('<p>Ledger timestamps are seconds since measurement start.</p>')
    if entries:
        groups = sorted({str(item.get('group_idx')) for item in entries})
        options = ''.join('<option>' + html.escape(group) + '</option>' for group in groups)
        parts.append('<div class="kv-filters"><label>Block hash <input id="kv-hash-filter" type="search"></label> <label>Group <select id="kv-group-filter"><option value="">All groups</option>' + options + '</select></label> <label>Status <select id="kv-state-filter"><option value="">All states</option><option value="returned">Returned</option><option value="right_censored">Right censored</option><option value="reset">Reset</option><option value="observation_gap">Observation gap</option></select></label><p id="kv-filter-count"></p></div>')
    for state, title in [('returned', 'Returned episodes'), ('right_censored', 'Not returned by observation end'), ('reset', 'Ended by reset'), ('observation_gap', 'Observation interrupted by missing events')]:
        ledger = [item for item in entries if item['status'] == state]
        if not ledger:
            continue
        headers = ['Block hash', 'Rank', 'Group', 'Locality', 'Generation', 'Removed at s', 'Next store at s', 'Interval s', 'Observed until s']
        if cpu:
            headers += ['CPU stored at s', 'CPU copy continuously observed']
        if causal_returns and state=='returned':
            headers += ['Return explanation', 'Returning request', 'Completion evidence']
        cells = []
        for item in ledger:
            def relative(key: str) -> str:
                value = item.get(key)
                return number(value - start) if value is not None else 'Unknown'
            row = [item['block_hash'], item.get('data_parallel_rank'), item.get('group_idx'), item.get('locality'), item['cache_generation'], relative('removed_batch_ts'), relative('return_batch_ts'), number(item['removal_to_next_store_s']), relative('observation_end_batch_ts')]
            if cpu:
                continuous = 'Unknown' if data['collection_status'] == 'partial' and (data.get('publication_window_audit') or {}).get('status')!='all_sequences_present' else 'Yes' if item['cpu_copy_continuously_observed'] else 'Not established'
                row += [relative('cpu_stored_batch_ts'), continuous]
            if causal_returns and state=='returned':
                explanation={'confirmed_load':'Confirmed CPU load','computation_observed':'Computed again','unresolved':'Unresolved'}.get(item.get('return_evidence'),'Not classified')
                evidence='jobs '+','.join(map(str,item.get('return_transfer_job_ids',[]))) if item.get('return_transfer_job_ids') else str(item['completed_returned_content_tokens'])+' content tokens computed' if item.get('completed_returned_content_tokens') is not None else 'No matched evidence'
                boundary=item.get('return_cpu_store_boundary_evidence') or {}
                if boundary.get('boundary_inside_completed_chunk'):
                    intervals=', '.join(f'[{a}, {b})' for a,b in boundary['completed_covering_steps'])
                    evidence+=f'; required prefix end {boundary["cache_boundary_tokens"]} lies inside earlier completed step {intervals}; CPU-storing request {boundary.get("cpu_store_source_request_id") or boundary["cpu_store_request_id"]}'
                row += [explanation,item.get('return_request_id'),evidence]
            cells.append(row)
        attrs = [f' data-block="{html.escape(str(item["block_hash"]), quote=True)}" data-group="{html.escape(str(item.get("group_idx")), quote=True)}"' for item in ledger]
        parts.append(f'<details class="kv-ledger" data-state="{state}"><summary>{title}: {len(ledger):,}</summary>' + table(headers, cells, attributes=attrs) + '</details>')
    parts.append('</section>')
    return '\n'.join(parts)


def physical_section(rep: Path, data: dict) -> str:
    stats = data.get('physical_trace')
    if not stats:
        return ''
    rows = [('Physical GPU block evictions', stats['physical_evictions']), ('Evicted size in bytes', 'Not recorded'), ('Eviction operation duration', 'Not recorded')]
    cpu = (data.get('causal_trace') or {}).get('cpu_cache') or {}
    if cpu:
        rows.insert(1, ('CPU content evictions', cpu.get('eviction_events')))
    sources = ' · '.join(link(rep/name, rep, label) for name, label in [('physical_trace_summary.json', 'Trace coverage'), ('physical_evictions.jsonl', 'Physical evictions'), ('physical_transfers.jsonl', 'Transfers')] if (rep/name).is_file() and (rep/name).stat().st_size)
    duration_rows=[]
    labels={'time_since_last_release_s':'Since reference count reached zero','allocation_age_s':'Age of this slot allocation','time_since_last_access_s':'Since last recorded prefix-cache touch'}
    for key,values in (stats.get('duration_statistics') or {}).items():
        duration_rows.append([labels[key],values['count'],number(values['p50']),number(values['p95']),number(values['max'])])
    durations=table(['Interval at cached-content eviction', 'Available observations', 'p50 s', 'p95 s', 'Maximum s'],duration_rows) if duration_rows else ''
    figure=stats.get('idle_figure')
    plot=f'<figure><a href="{figure["path"]}"><img src="{figure["path"]}" alt="{html.escape(figure["title"])}"></a><figcaption>{html.escape(figure["title"])}</figcaption></figure>' if figure else ''
    return '<section id="physical-events"><h2>Physical eviction measurements</h2>' + table(['Metric', 'Observed value'], [[key,value] for key,value in rows]) + '<p>Each event records a GPU block slot being reused. The trace contains no byte size or start/end timestamps for that operation.</p>' + durations + '<p>These intervals measure content age or idle time at eviction.</p><details><summary>Eviction distribution and trace coverage</summary>' + plot + '<p>' + html.escape(stats['scope']) + ' ' + html.escape(stats['coverage_note']) + '</p><p>' + sources + '</p></details></section>'



def transfer_section(rep: Path, data: dict, *, base: Path | None = None) -> str:
    stats = data.get('aggregate_transfers') or {}
    directions = stats.get('directions') or {}
    if not directions:
        return ''
    rows = []
    records = read_rows(rep/'physical_transfers.jsonl')
    for key, label, direction in [('GPU_to_CPU', 'GPU → CPU', 'store'), ('CPU_to_GPU', 'CPU → GPU', 'load')]:
        if key not in directions:
            continue
        row = directions[key]
        times = sorted(r['transfer_time_s'] for r in records if r.get('success') and r.get('direction') == direction and isinstance(r.get('transfer_time_s'), (int,float)))
        p50 = statistics.median(times) * 1000 if times else None
        p95 = (statistics.quantiles(times,n=100,method='inclusive')[94] if len(times)>1 else times[0]) * 1000 if times else None
        rows.append([label, number(row['bytes']/1e9) if row['bytes'] is not None else 'Not recorded', number(row['worker_time_s']), row['worker_samples'], len(times), number(p50), number(p95)])
    return '<section><h2>CPU swap size and time</h2>' + table(['Direction', 'Transferred GB', 'Sum of worker transfer times s', 'Worker transfers in metrics', 'Timed trace records', 'Trace transfer p50 ms', 'Trace transfer p95 ms'], rows) + '<p>GB = 1,000,000,000 bytes. Totals use the run metrics window; percentiles use successful request-matched worker transfers. Worker times add across concurrent workers. CPU writes can precede GPU eviction, so transferred bytes are not the size evicted.</p></section>'


def causal_section(rep: Path, data: dict) -> str:
    stats = data.get('causal_trace') or {}
    if not stats or not stats.get('registered_requests'):
        return ''
    rows = [('Requests joined by explicit ID', f"{stats['registered_requests']} / {stats['expected_requests']}"),
            ('Transfer jobs with all worker completions', f"{stats['confirmed_transfer_jobs']} / {stats['transfer_jobs']}"),
            ('Confirmed CPU → GPU jobs', stats['confirmed_load_jobs']),
            ('Content records in confirmed CPU → GPU jobs', stats['confirmed_reload_content_records']),
            ('Loaded content also present in a completed request step', stats['reload_records_present_in_completed_step']),
            ('Loaded content with an earlier physical capacity eviction', stats['reload_records_with_prior_physical_eviction'])]
    if stats.get('confirmed_load_jobs'):
        rows.append(('Loaded GPU allocations unchanged through completed use', stats.get('reload_records_with_stable_gpu_allocation', 'Not analyzed')))
    if not stats['confirmed_load_jobs']:
        rows=rows[:2] if stats['transfer_jobs'] else rows[:1]
    accounting=data.get('transfer_accounting') or {}
    if accounting:
        rows.append(('Transfer totals agree with metrics in each observed direction',all(r['matches_metrics_window'] for r in accounting.values())))
    title='Which CPU copies were loaded?' if stats['confirmed_load_jobs'] else 'Request cache and computation evidence'
    explanation='<p>A confirmed job has a scheduler plan identifying the content, successful completion records from every expected worker, and the scheduler acknowledgement that all workers finished. A job can contain several content records across cache groups.</p>' if stats['transfer_jobs'] else ''
    parts = ['<section id="confirmed-transfers"><h2>' + title + '</h2>' + explanation, table(['Evidence', 'Observed count'], [[key,value] for key,value in rows])]
    reconciliation=stats.get('candidate_reconciliation') or {}
    if sum(reconciliation.values()):
        parts.append('<h3>What happened to the CPU-backed return candidates?</h3><p>Each candidate is matched to the request that assigned its returning hash in that publication batch. The request, content hash and cache group then identify a confirmed load or completed recomputation.</p>' + table(['Explanation of candidate return', 'Episodes'], [[label,reconciliation.get(key,0)] for key,label in [('confirmed_load','Confirmed CPU load'),('computation_observed','Completed computation observed'),('unresolved','Unresolved')]]))
    miss_history = {}
    for candidate in read_rows(rep/'candidate_reconciliation.jsonl'):
        if candidate.get('status') != 'computation_observed':
            continue
        observations = {(lookup['group_idx'], lookup['history']['prior_cpu_event']) for lookup in candidate.get('cpu_lookups',[]) if lookup.get('result') == 'MISS' and lookup.get('history')}
        for key in observations:
            miss_history[key] = miss_history.get(key, 0) + 1
    if miss_history:
        labels = {'no_prior_cpu_event_observed':'No earlier assignment or write completion observed', 'CPUBlockAssigned':'Assigned; no subsequent write completion observed', 'CPUStoreReady':'Write completion observed; lookup still missed', 'CPUContentEvicted':'CPU content eviction observed after its earlier state'}
        parts.append('<h3>CPU misses in the recomputed candidates</h3>' + table(['Missing cache group', 'Latest preceding CPU evidence', 'Candidate episodes'], [[group,labels[state],count] for (group,state),count in sorted(miss_history.items())]) + '<p>History is matched by the full offload key and process epoch. No earlier record means no earlier record in the available trace; whole-process coverage remains separate. A missing required group can prevent restoring a prefix even when another group has a CPU copy.</p>')
    inside=stats.get('candidate_boundaries_inside_store_request_chunks',0)
    if inside:
        parts.append(f'<p>For {inside} candidates, the required prefix boundary lies inside a completed computation chunk of the request that stored the CPU copy. The original returned-episode ledger includes the boundary, earlier interval, and storing request in its completion-evidence column.</p>')
    figure=stats.get('reload_figure')
    if figure:
        parts.append(f'<figure><a href="{html.escape(figure["path"],quote=True)}"><img src="{html.escape(figure["path"],quote=True)}" alt="{html.escape(figure["title"],quote=True)}"></a><figcaption>{html.escape(figure["title"])}</figcaption></figure>')
    paths = stats.get('request_paths') or []
    if paths:
        parts.append('<details><summary>Request token accounting</summary><p>For each request, compare the prompt length with tokens allocated from the local GPU cache, tokens allocated from external cache, and prompt tokens in completed computation steps. External allocation is confirmed separately by completed load jobs. A mismatch requires investigation.</p>' + table(['Request', 'Prompt tokens', 'Local cached allocation'] + (['External cached allocation'] if data.get('offload')=='enabled' else []) + ['Completed prompt computation', 'Sum matches prompt'] + (['Confirmed load jobs'] if stats['confirmed_load_jobs'] else []), [[r['source_request_id'],r['prompt_tokens'],r['allocated_local_cached_tokens']] + ([r['allocated_external_cached_tokens']] if data.get('offload')=='enabled' else []) + [r['completed_prefill_tokens'],r['token_accounting_matches_prompt']] + ([r['confirmed_load_jobs']] if stats['confirmed_load_jobs'] else []) for r in paths]) + '</details>')
    waits = read_rows(rep/'request_load_waits.jsonl')
    if waits:
        parts.append('<h3>Observed request waits for cache loading</h3><p>The interval starts when the scheduler is observed waiting for remote KV and ends when it allows the request to proceed. It includes scheduler delay and is not an isolated copy duration.</p>' + table(['Request', 'Blocked to allowed-to-proceed s', 'Blocked to completion observed s', 'Completion to allowed-to-proceed s'], [[r['source_request_id'],number(r['observed_blocked_to_unblocked_s']),number(r['observed_blocked_to_completion_s']),number(r['completion_to_unblocked_s'])] for r in waits]))
    cpu=stats.get('cpu_cache') or {}
    if cpu:
        cpu_rows=[['CPU slot assignments',cpu['assignment_events']],['CPU contents marked ready',cpu['store_ready_events']],['CPU capacity evictions',cpu['eviction_events']]]
        if cpu['eviction_events']:
            cpu_rows += [['Evicted contents stored by this run',cpu['victims_created_in_run']],['Evicted contents stored by other requests',cpu['victims_from_other_requests']],['Evicted contents with unknown origin',cpu['victims_with_unknown_origin']]]
        parts.append('<h3>CPU content eviction during these requests</h3><p>' + html.escape(cpu['scope']) + '</p>' + table(['CPU observation','Count'],cpu_rows))
        duration_rows=[]
        for key,label in [('since_assignment_s','CPU slot assignment to eviction'),('since_store_ready_s','CPU store ready to eviction')]:
            values=cpu['duration_statistics'].get(key)
            if values:duration_rows.append([label,values['count'],number(values['p50']),number(values['p95']),number(values['max'])])
        parts.append(table(['CPU content lifetime interval','Observed lifetimes','p50 s','p95 s','Maximum s'],duration_rows))
    parts.append('<p>' + html.escape(stats['coverage_note']) + '</p>')
    sources = [('candidate_reconciliation.jsonl','Candidate-by-candidate explanation'),('causal_requests.jsonl','Request token accounting'),('causal_trace_summary.json','Evidence summary'),('causal_transfers.jsonl','Transfer jobs and content'),('confirmed_reloads.jsonl','Confirmed loaded content'),('request_load_waits.jsonl','Observed request waits'),('recomputed_content.jsonl','Repeated hash-unit computation'),('cpu_evictions.jsonl','CPU eviction identities and lifetimes')]
    parts.append('<p>' + ' · '.join(link(rep/name,rep,label) for name,label in sources if (rep/name).is_file() and (rep/name).stat().st_size) + '</p></section>')
    return ''.join(parts)
