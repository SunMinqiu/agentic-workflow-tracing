"""Explain logical CPU-backed returns using request-scoped execution evidence."""
from collections import defaultdict


def cpu_lookup_history(rows):
    """Describe preceding CPU events in the same process and full offload key."""
    state = {}; result = {}
    kinds = {'CPUBlockAssigned', 'CPUStoreReady', 'CPUContentEvicted', 'CPULookup'}
    for row in sorted((r for r in rows if r['event_type'] in kinds), key=lambda r: (r.get('time_monotonic_ns', 0), r.get('sequence', 0))):
        required = ('process_id', 'publisher_epoch', 'trace_session_id', 'sequence', 'offload_key')
        if any(row.get(key) is None for key in required):
            continue
        scope = (row['trace_session_id'], row['process_id'], row['publisher_epoch'])
        key = (*scope, row['offload_key'])
        if row['event_type'] == 'CPULookup':
            previous = state.get(key)
            result[(*scope, row['sequence'])] = {'prior_cpu_event': previous['event_type'] if previous else 'no_prior_cpu_event_observed', 'prior_sequence': previous['sequence'] if previous else None, 'lookup_sequence': row['sequence'], 'scope': 'Available trace history in the same process epoch and full offload key; absence is not whole-process coverage proof.'}
        else:
            state[key] = row
    return result


def reconcile(rows, events, episodes, loads, recomputed, prompt_inputs, request_sources=None):
    request_sources=request_sources or {}
    def cpu_identity(row):return tuple(row.get(k) for k in ('cpu_pool_id','cpu_block_id','cpu_generation','offload_key'))
    def scope(row):return (row.get('publisher_id'),row.get('publisher_epoch'),row.get('data_parallel_rank'))
    batches=defaultdict(dict)
    stored=defaultdict(list)
    for row in events:
        batches[scope(row)][row['sequence']]=row['batch_ts']
        if row['event_type']=='BlockStored' and str(row.get('medium','')).upper()=='GPU':
            for index,h in enumerate(row.get('block_hashes') or []):
                stored[(scope(row),row['sequence'],str(h),row.get('group_idx'))].append((row,index))
    prior={}
    for namespace,series in batches.items():
        previous=float('-inf')
        for sequence,ts in sorted(series.items()):
            prior[(namespace,sequence)]=previous;previous=ts
    assigned=defaultdict(list);registrations={};completed=defaultdict(list);lookups=defaultdict(list)
    cpu_assignments={cpu_identity(r):r for r in rows if r['event_type']=='CPUBlockAssigned' and all(value is not None for value in cpu_identity(r))}
    histories = cpu_lookup_history(rows)
    for row in rows:
        if row['event_type']=='BlockHashAssigned':assigned[(str(row['assigned_hash']),row['assigned_group_idx'])].append(row)
        elif row['event_type']=='CPULookup':lookups[(row['request_id'],str(row['block_hash']))].append(row)
        elif row['event_type']=='RequestRegistered':registrations[row['request_id']]=row
        elif row['event_type']=='RequestComputeOutputObserved' and row.get('usable'):completed[row['request_id']].append(row)
    load_map=defaultdict(list);compute_map=defaultdict(list)
    for source,target in [(loads,load_map),(recomputed,compute_map)]:
        for row in source:target[(row['request_id'],str(row['block_hash']),row['group_idx'])].append(row)
    results=[]
    for episode in episodes:
        if episode['status']!='returned' or not episode['cpu_copy_continuously_observed']:
            continue
        key=(str(episode['block_hash']),episode['group_idx'])
        low=prior.get((scope(episode),episode['return_sequence']),float('-inf'))
        assignments=[r for r in assigned[key] if low<r['time_epoch_s']<=episode['return_batch_ts']]
        rids={r['request_id'] for r in assignments}
        record=dict(publisher_id=episode.get('publisher_id'),publisher_epoch=episode.get('publisher_epoch'),data_parallel_rank=episode.get('data_parallel_rank'),cache_generation=episode.get('cache_generation'),block_hash=episode['block_hash'],group_idx=episode['group_idx'],
                    removed_sequence=episode['removed_sequence'],return_sequence=episode['return_sequence'],
                    request_id=None,source_request_id=None,status='unresolved')
        if len(rids)==1:
            rid=next(iter(rids));record['request_id']=rid
            load=[r for r in load_map[(rid,*key)] if r.get('all_workers_observed_at_ns',0)<=max(a['time_monotonic_ns'] for a in assignments)]
            computation=compute_map[(rid,*key)]
            if load:
                record.update(status='confirmed_load',source_request_id=load[0]['source_request_id'],job_ids=sorted({r['job_id'] for r in load}))
            elif computation:
                record.update(status='computation_observed',source_request_id=computation[0]['source_request_id'],cpu_lookups=[{'group_idx':r['group_idx'],'result':r['result'],'cpu':r.get('cpu'), 'history': histories.get((r.get('trace_session_id'), r.get('process_id'), r.get('publisher_epoch'), r.get('sequence')))} for r in lookups[(rid,key[0])]])
                reg=registrations[rid]
                positions=[(i+1)*reg['hash_block_size'] for i,h in enumerate(reg['prompt_hashes']) if str(h)==key[0]]
                source=stored[(scope(episode),episode['return_sequence'],*key)]
                if len(positions)==len(source)==1:
                    event,index=source[0];block_size=event['block_size']
                    tokens=event.get('token_ids',[])[index*block_size:(index+1)*block_size]
                    end=positions[0];start=end-len(tokens)
                    prompt=prompt_inputs.get(record['source_request_id'])
                    if event.get('kv_cache_spec_kind')=='full_attention' and tokens and start>=0 and prompt is not None and prompt[start:end]==tokens:
                        overlaps=[(max(start,r['token_start']),min(end,r['token_end'],r['prompt_tokens'])) for r in completed[rid]]
                        overlaps=[(a,b) for a,b in overlaps if b>a]
                        stop=-1;count=0
                        for a,b in sorted(overlaps):count+=max(0,b-max(a,stop));stop=max(stop,b)
                        record.update(content_token_start=start,content_token_end=end,completed_content_tokens=count,
                                      entire_returned_chunk_computed=count==len(tokens),computed_overlaps=overlaps)
                        origins=[]
                        for lookup in lookups[(rid,key[0])]:
                            if lookup['group_idx']!=key[1] or lookup['result']!='HIT' or not lookup.get('cpu'):
                                continue
                            origin=cpu_assignments.get(cpu_identity(lookup['cpu']))
                            if origin and origin['time_monotonic_ns']<=lookup['time_monotonic_ns']:
                                origins.append((origin,lookup))
                        if len(origins)==1:
                            origin,lookup=origins[0]
                            steps=[r for r in completed[origin['request_id']] if r['time_monotonic_ns']<lookup['time_monotonic_ns'] and r['token_start']<end<=r['token_end']]
                            record['cpu_store_boundary_evidence']={'cpu_store_request_id':origin['request_id'],'cpu_store_source_request_id':request_sources.get(origin['request_id']),'cpu_assignment_sequence':origin['sequence'],'cache_boundary_tokens':end,'completed_covering_steps':[[r['token_start'],r['token_end']] for r in steps],'boundary_inside_completed_chunk':any(r['token_start']<end<r['token_end'] for r in steps),'boundary_at_completed_chunk_end':any(r['token_end']==end for r in steps),'scope':'CPU assignment identifies the storing request; these are its completed computation intervals before the returning lookup. It need not be the original GPU producer.'}
        results.append(record)
    return results
