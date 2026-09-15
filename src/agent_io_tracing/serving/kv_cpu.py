"""Measure CPU content lifetimes using pool, slot, and allocation generation."""
from collections import Counter
import statistics


def analyze_cpu_cache(rows, request_map):
    """Consume deduplicated, time-ordered rows; count evictions caused by this run."""
    kinds={'CPUBlockAssigned','CPUStoreReady','CPUContentEvicted'}
    selected=[r for r in rows if r['event_type'] in kinds and r.get('request_id') in request_map]
    if not selected:
        return {}
    assigned={};ready={};evictions=[]
    fields=('cpu_pool_id','cpu_block_id','cpu_generation','offload_key')
    for row in rows:
        if row['event_type'] not in kinds:continue
        key=tuple(row.get(k) for k in fields)
        valid=all(value is not None for value in key)
        if row['event_type']=='CPUBlockAssigned':
            if valid:assigned[key]=row
        elif row['event_type']=='CPUStoreReady':
            if valid:ready[key]=row
        else:
            origin=assigned.pop(key,None) if valid else None
            available=ready.pop(key,None) if valid else None
            if row.get('request_id') not in request_map:continue
            record={k:row.get(k) for k in (*fields,'block_hash','group_idx')}
            record.update(evicting_request_id=row['request_id'],evicting_source_request_id=request_map[row['request_id']],storing_request_id=origin.get('request_id') if origin else None,storing_source_request_id=request_map.get(origin.get('request_id')) if origin else None,origin_observed=origin is not None,origin_request_in_run=bool(origin and origin.get('request_id') in request_map),evicted_at_ns=row['time_monotonic_ns'])
            for name,event in [('since_assignment_s',origin),('since_store_ready_s',available)]:
                delta=(row['time_monotonic_ns']-event['time_monotonic_ns'])/1e9 if event else None
                record[name]=delta if delta is not None and delta>=0 else None
            evictions.append(record)
    counts=Counter(r['event_type'] for r in selected)
    durations={}
    for key in ['since_assignment_s','since_store_ready_s']:
        values=[r[key] for r in evictions if r[key] is not None]
        if values:
            durations[key]={'count':len(values),'p50':statistics.median(values),'p95':statistics.quantiles(values,n=100,method='inclusive')[94] if len(values)>1 else values[0],'max':max(values)}
    return {'assignment_events':counts['CPUBlockAssigned'],'store_ready_events':counts['CPUStoreReady'],'eviction_events':len(evictions),'evictions_by_group':dict(Counter(str(r['group_idx']) for r in evictions)),'victims_created_in_run':sum(r['origin_request_in_run'] for r in evictions),'victims_from_other_requests':sum(r['origin_observed'] and not r['origin_request_in_run'] for r in evictions),'victims_with_unknown_origin':sum(not r['origin_observed'] for r in evictions),'duration_statistics':durations,'evictions':evictions,'scope':'CPU capacity evictions triggered by the matched requests. Victims may have been stored by earlier workloads. Each identity is a CPU pool slot generation shared logically across tensor-parallel workers; it is not a GPU physical eviction or a worker transfer sample.'}
