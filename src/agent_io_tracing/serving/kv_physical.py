"""Import physical traces and join requests only by recorded request identity."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def analyze(rows, requests):
    request_map = {}
    for request in requests:
        for key in ('sent_request_id', 'provider_request_id'):
            if request.get(key):
                request_map[request[key]] = request.get('request_id')
    for row in rows:
        if row.get('event_type') == 'RequestRegistered':
            external = (row.get('sampling_extra_args') or {}).get('kv_trace_request_id')
            if external in request_map:
                request_map[row['request_id']] = request_map[external]
    seen = {}
    previous = {}
    gaps = []
    selected = []
    unmatched = 0
    duplicates = 0
    for row in rows:
        scope = (row.get('trace_session_id'), row['process_id'], row['publisher_epoch'])
        key = (*scope, row['sequence'])
        if key in seen:
            if seen[key] != row:
                raise ValueError('Conflicting physical trace sequence')
            duplicates += 1
            continue
        seen[key] = row
        seq = row['sequence']
        if scope in previous and seq <= previous[scope]:
            raise ValueError('Physical trace sequence must increase within each process epoch')
        expected = previous.get(scope, -1) + 1
        if seq > expected:
            gaps.append({'process_id':scope[1], 'publisher_epoch':scope[2], 'first_missing':expected, 'last_missing':seq-1})
        previous[scope] = seq
        request_id = row.get('request_id') or row.get('associated_request_id')
        if request_id not in request_map:
            unmatched += 1
            continue
        selected.append({**row, 'source_request_id':request_map[request_id]})
    evictions = []
    for row in selected:
        if row['event_type'] != 'PhysicalEviction':
            continue
        item = dict(row)
        for source, target in [('last_release_ns', 'time_since_last_release_s'), ('last_access_ns', 'time_since_last_access_s'), ('allocated_at_ns', 'allocation_age_s')]:
            timestamp = row.get(source)
            item[target] = (row['time_monotonic_ns']-timestamp)/1e9 if timestamp is not None else None
            if item[target] is not None and item[target] < 0:
                raise ValueError('Physical trace duration is negative')
        evictions.append(item)
    submissions = {}
    transfers = []
    for row in selected:
        key = (row.get('trace_session_id'),row['process_id'],row['publisher_epoch'],row.get('worker_id'),row.get('job_id'))
        if row['event_type'] == 'TransferSubmitted':
            submissions[key] = row
        elif row['event_type'] == 'TransferCompleted':
            submitted = submissions.pop(key, None)
            transfers.append({**row, 'submission_observed':submitted is not None,
                              'submission_to_observed_completion_s':(row['completion_observed_at_ns']-submitted['submitted_at_ns'])/1e9 if submitted else None})
    summary = {
        'schema_version':1, 'collection_status':'partial' if gaps else 'unconfirmed',
        'scope':'Events with an exact recorded request ID match; other process events are excluded.',
        'coverage_note':'Request matching and contiguous records do not confirm trace start, tail delivery, or coverage of every engine process.',
        'raw_events':len(seen),'matched_events':len(selected),'unmatched_events':unmatched,
        'duplicate_events':duplicates,'sequence_gaps':gaps,
        'physical_evictions':len(evictions),
        'transfer_completions':len(transfers),
        'successful_transfer_bytes_by_direction':{direction:sum(r.get('transfer_bytes') or 0 for r in transfers if r.get('success') and r.get('direction')==direction) for direction in ('load','store')},
        'pending_accepted_transfers':sum(bool(row.get('accepted')) for row in submissions.values()),
        'request_cache_lookups':sum(row['event_type']=='RequestCacheLookup' for row in selected),
    }
    summary['duration_statistics']={}
    for field in ['time_since_last_release_s','allocation_age_s','time_since_last_access_s']:
        values=sorted(r[field] for r in evictions if r.get(field) is not None)
        if values:
            summary['duration_statistics'][field]={'count':len(values),'mean':statistics.mean(values),'p50':statistics.median(values),'p95':statistics.quantiles(values,n=100,method='inclusive')[94] if len(values)>1 else values[0],'max':max(values)}
    return summary, evictions, transfers


def prepare(rep):
    paths = sorted((rep/'physical_traces').glob('*.jsonl'))
    if not paths:
        return {}
    rows = [row for path in paths for row in read_rows(path)]
    rows.sort(key=lambda r:(str(r.get('trace_session_id')),r['process_id'],r['publisher_epoch'],r['sequence']))
    summary, evictions, transfers = analyze(rows, read_rows(rep/'requests.jsonl'))
    from agent_io_tracing.serving.kv_plots import write_eviction_plot
    summary['idle_figure']=write_eviction_plot(rep,evictions)
    summary['sources'] = {str(path.relative_to(rep)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    for name, data in [('physical_evictions.jsonl',evictions),('physical_transfers.jsonl',transfers)]:
        (rep/name).write_text(''.join(json.dumps(row)+'\n' for row in data))
    (rep/'physical_trace_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rep',type=Path)
    parser.add_argument('traces',type=Path,nargs='+')
    args=parser.parse_args()
    if not (args.rep/'requests.jsonl').is_file():
        parser.error('The repetition must contain recorded requests.jsonl')
    target=args.rep/'physical_traces'
    target.mkdir(exist_ok=True)
    for source in args.traces:
        payload=source.read_bytes()
        path=target/(hashlib.sha256(payload).hexdigest()+'.jsonl')
        path.write_bytes(payload)
    from agent_io_tracing.serving.kv_report_data import prepare_report_data
    from agent_io_tracing.analysis.kvcache.replay_report import refresh_for_run
    data=prepare_report_data(args.rep)
    refresh_for_run(args.rep)
    print(json.dumps(data['physical_trace'],indent=2))


if __name__=='__main__':
    main()
