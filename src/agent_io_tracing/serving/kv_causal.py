"""Join audited scheduler plans, worker completions, and observed computation."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path


def analyze(rows, requests):
    unique = {}
    for row in rows:
        key = (row.get('trace_session_id'), row['process_id'], row['publisher_epoch'], row['sequence'])
        if key in unique and unique[key] != row:
            raise ValueError(f'Conflicting trace sequence: {key}')
        unique[key] = row
    rows = sorted(unique.values(), key=lambda row:(row['time_monotonic_ns'], row['sequence']))
    external = {r['sent_request_id']:r['request_id'] for r in requests if r.get('sent_request_id')}
    request_map = {}
    registrations = {}
    for row in rows:
        if row['event_type'] != 'RequestRegistered':
            continue
        marker = (row.get('sampling_extra_args') or {}).get('kv_trace_request_id')
        if marker in external:
            request_map[row['request_id']] = external[marker]
            registrations[row['request_id']] = row
    selected = [r for r in rows if (r.get('request_id') or r.get('associated_request_id')) in request_map]
    plans = [r for r in selected if r['event_type']=='TransferPlan']
    plan_keys = defaultdict(list)
    for plan in plans:
        plan_keys[(plan.get('trace_session_id'),plan['job_id'])].append(plan)
    worker_completions = defaultdict(list)
    coordinator_completions = {}
    for row in rows:
        if row['event_type']=='TransferCompleted':
            worker_completions[(row.get('trace_session_id'),row['job_id'])].append(row)
        elif row['event_type']=='TransferAllWorkersComplete':
            coordinator_completions[(row.get('trace_session_id'),row['publisher_epoch'],row['connector_id'],row['job_id'])]=row
    eviction_history = defaultdict(list)
    allocation_history = defaultdict(list)
    for row in rows:
        if row['event_type']=='BlockAllocated':
            allocation_history[(row.get('trace_session_id'),row.get('publisher_epoch'),row.get('process_id'),row['block_id'])].append(row)
        if row['event_type']=='PhysicalEviction':
            for identity in row.get('cached_hashes') or []:
                eviction_history[(row.get('trace_session_id'), row['publisher_epoch'], str(identity['block_hash']), identity['group_idx'])].append(row)
    request_events = defaultdict(list)
    for row in selected:
        request_events[row.get('request_id') or row.get('associated_request_id')].append(row)
    transfers = []
    reloads = []
    for plan in plans:
        key=(plan.get('trace_session_id'),plan['job_id'])
        completions=[r for r in worker_completions[key] if r.get('associated_request_id')==plan['request_id'] and r.get('direction')==plan['direction']]
        workers={(r['process_id'],r['publisher_epoch'],r.get('worker_id')) for r in completions}
        coordinator=coordinator_completions.get((plan.get('trace_session_id'),plan['publisher_epoch'],plan['connector_id'],plan['job_id']))
        confirmed=bool(len(plan_keys[key])==1 and coordinator and len(workers)==len(completions)==plan['expected_workers'] and all(r.get('accepted') and r.get('success') and r.get('source')==plan.get('source') and r.get('destination')==plan.get('destination') for r in completions))
        transfer=dict(source_request_id=request_map[plan['request_id']],request_id=plan['request_id'],job_id=plan['job_id'],direction=plan['direction'],expected_workers=plan['expected_workers'],completed_workers=len(workers),confirmed=confirmed,contents=plan['contents'],worker_bytes=sum(r.get('transfer_bytes') or 0 for r in completions),worker_transfer_time_s=sum(r.get('transfer_time_s') or 0 for r in completions),all_workers_observed_at_ns=coordinator['time_monotonic_ns'] if coordinator else None)
        transfers.append(transfer)
        if not confirmed or plan['direction']!='load':
            continue
        computed=[r for r in request_events[plan['request_id']] if r['event_type']=='RequestComputeOutputObserved' and r.get('usable') and r.get('scheduled_at_ns', r['time_monotonic_ns'])>=coordinator['time_monotonic_ns']]
        for identity in plan['contents']:
            key=(plan.get('trace_session_id'),plan['publisher_epoch'],str(identity['block_hash']),identity['group_idx'])
            before=[r for r in eviction_history[key] if r['time_monotonic_ns']<plan['time_monotonic_ns']]
            gpu_blocks=set(identity['gpu_block_ids'])-{0}
            usage=next((r for r in computed if gpu_blocks and identity['group_idx'] < len(r.get('block_ids', [])) and gpu_blocks.issubset(set(r['block_ids'][identity['group_idx']]))),None)
            allocation_proof=[]
            for block_id in sorted(gpu_blocks):
                history=allocation_history[(plan.get('trace_session_id'),plan.get('publisher_epoch'),plan.get('process_id'),block_id)]
                prior=[r for r in history if r['time_monotonic_ns']<=plan['time_monotonic_ns']]
                last=max(prior,key=lambda r:r['time_monotonic_ns']) if prior else None
                reused=bool(usage and any(plan['time_monotonic_ns']<r['time_monotonic_ns']<=usage['time_monotonic_ns'] for r in history))
                status='no_completed_use' if usage is None else 'missing_allocation' if last is None or last.get('pool_id') is None or last.get('allocation_generation') is None or len({r.get('pool_id') for r in prior})!=1 else 'reallocated' if reused else 'same_allocation'
                allocation_proof.append({'block_id':block_id,'pool_id':last.get('pool_id') if last else None,'allocation_generation':last.get('allocation_generation') if last else None,'status':status})
            stable_allocation=bool(allocation_proof and all(r['status']=='same_allocation' for r in allocation_proof))
            reloads.append(dict(source_request_id=request_map[plan['request_id']],request_id=plan['request_id'],job_id=plan['job_id'],all_workers_observed_at_ns=coordinator['time_monotonic_ns'],**identity,previous_physical_eviction_observed=bool(before),evicted_block_id=before[-1]['block_id'] if before else None,evicted_allocation_generation=before[-1]['allocation_generation'] if before else None,loaded_blocks_present_in_completed_request_step=usage is not None,completed_step_id=usage['step_id'] if usage else None,loaded_block_allocation_proof=allocation_proof,loaded_block_allocation_unchanged_through_completed_step=stable_allocation))
    waits=[]
    for rid, events in request_events.items():
        blocked=None;completed=None
        for row in events:
            if row['event_type']=='RequestLoadBlocked':
                blocked=row;completed=None
            elif row['event_type']=='RequestLoadCompletionObserved' and blocked:
                completed=row
            elif row['event_type']=='RequestLoadUnblocked' and blocked:
                waits.append(dict(source_request_id=request_map[rid],request_id=rid,
                                  observed_blocked_to_unblocked_s=(row['time_monotonic_ns']-blocked['time_monotonic_ns'])/1e9,
                                  observed_blocked_to_completion_s=(completed['time_monotonic_ns']-blocked['time_monotonic_ns'])/1e9 if completed else None,
                                  completion_to_unblocked_s=(row['time_monotonic_ns']-completed['time_monotonic_ns'])/1e9 if completed else None))
                blocked=None
    recomputed=[]
    for rid, registration in registrations.items():
        hash_size=registration['hash_block_size']
        positions={str(h):(index+1)*hash_size for index,h in enumerate(registration['prompt_hashes'])}
        completed=[r for r in request_events[rid] if r['event_type']=='RequestComputeOutputObserved' and r.get('usable') and (r.get('completed_prompt_tokens') or 0)>0]
        for key, evictions in eviction_history.items():
            if key[:2]!=(registration.get('trace_session_id'),registration['publisher_epoch']) or key[2] not in positions:
                continue
            candidates=[r for r in evictions if r['time_monotonic_ns']<registration['time_monotonic_ns']]
            if not candidates:
                continue
            eviction=candidates[-1];end=positions[key[2]]
            # A content hash identifies its terminal hash unit. A recurrent state
            # or an aliased physical block can represent a longer prefix.
            start=max(0,end-hash_size)
            overlaps=[(max(start,r['token_start']),min(end,r['token_end'],r['prompt_tokens'])) for r in completed]
            overlaps=[(a,b) for a,b in overlaps if b>a]
            if overlaps:
                recomputed.append(dict(source_request_id=request_map[rid],request_id=rid,block_hash=key[2],group_idx=key[3],previous_evicted_block_id=eviction['block_id'],content_token_start=start,content_token_end=end,computed_overlaps=overlaps))
    union_tokens=0
    for rid in registrations:
        intervals=sorted(interval for row in recomputed if row['request_id']==rid for interval in row['computed_overlaps'])
        stop=-1
        for start,end in intervals:
            union_tokens+=max(0,end-max(start,stop));stop=max(stop,end)
    request_paths=[]
    previous_prompts=defaultdict(list)
    for rid, registration in sorted(registrations.items(), key=lambda item:item[1]['time_monotonic_ns']):
        events=request_events[rid]
        done=[r for r in events if r['event_type']=='RequestComputeOutputObserved' and r.get('usable')]
        allocations=[r for r in events if r['event_type']=='RequestAllocation' and r.get('success')]
        computed=sum(r.get('completed_prompt_tokens') or 0 for r in done)
        local=sum(r.get('local_cached_tokens') or 0 for r in allocations)
        external_tokens=sum(r.get('external_computed_tokens') or 0 for r in allocations)
        prompt_key=(registration.get('trace_session_id'),registration.get('cache_salt'),registration.get('prompt_sha256'))
        prior=[r for r in previous_prompts[prompt_key] if r['time_monotonic_ns']<registration['time_monotonic_ns']] if registration.get('prompt_sha256') else []
        overlapping=[]
        for current in done:
            for earlier in prior:
                lo=max(current['token_start'],earlier['token_start'])
                hi=min(current['token_end'],earlier['token_end'],current['prompt_tokens'])
                if hi>lo:overlapping.append((lo,hi))
        repeated=0;stop=-1
        for lo,hi in sorted(overlapping):
            repeated+=max(0,hi-max(lo,stop));stop=max(stop,hi)
        previous_prompts[prompt_key].extend(done)
        prompt_tokens=registration.get('prompt_tokens')
        request_paths.append(dict(source_request_id=request_map[rid],request_id=rid,prompt_tokens=prompt_tokens,
                                  allocated_local_cached_tokens=local,allocated_external_cached_tokens=external_tokens,
                                  completed_prefill_tokens=computed,token_accounting_matches_prompt=prompt_tokens==local+external_tokens+computed,
                                  confirmed_load_jobs=sum(t['confirmed'] and t['direction']=='load' and t['request_id']==rid for t in transfers),
                                  repeated_same_prompt_computed_tokens=repeated))
    summary=dict(schema_version=1,registered_requests=len(registrations),expected_requests=len(requests),
                 transfer_jobs=len(transfers),confirmed_transfer_jobs=sum(r['confirmed'] for r in transfers),
                 confirmed_load_jobs=sum(r['confirmed'] and r['direction']=='load' for r in transfers),
                 confirmed_reload_content_records=len(reloads),unique_loaded_contents=len({(r['block_hash'],r['group_idx']) for r in reloads}),reload_records_with_prior_physical_eviction=sum(r['previous_physical_eviction_observed'] for r in reloads),
                 reload_records_present_in_completed_step=sum(r['loaded_blocks_present_in_completed_request_step'] for r in reloads),
                 observed_request_wait_intervals=len(waits),completed_prefill_tokens=sum(r.get('completed_prompt_tokens') or 0 for r in selected if r['event_type']=='RequestComputeOutputObserved' and r.get('usable')),
                 recomputed_terminal_hash_unit_tokens=union_tokens,
                 clock_scope='Monotonic timestamps from the same compute node. Blocked intervals are scheduler observations, not isolated DMA or counterfactual latency.',
                 recomputation_scope='Completed prompt intervals overlapping the terminal hash unit of previously evicted content. This is a lower bound on repeated token positions, not all recomputation or eviction-caused overhead. Groups and aliases are unioned per request.',
                 collection_status='unconfirmed',coverage_note='Positive associations require coordinator and all worker completion records. Whole-process start and end coverage remains separate.')
    summary['request_paths']=request_paths
    from agent_io_tracing.serving.kv_cpu import analyze_cpu_cache
    summary['cpu_cache']=analyze_cpu_cache(rows,request_map)
    summary['reload_records_with_stable_gpu_allocation']=sum(r['loaded_block_allocation_unchanged_through_completed_step'] for r in reloads)
    summary['token_accounting_matching_requests']=sum(r['token_accounting_matches_prompt'] for r in request_paths)
    summary['repeated_same_prompt_computed_tokens']=sum(r['repeated_same_prompt_computed_tokens'] for r in request_paths)
    return summary,transfers,reloads,waits,recomputed


def prepare(rep, *, episodes=None, events=None):
    paths=sorted((rep/'physical_traces').glob('*.jsonl'))
    if not paths:
        return {}
    rows=[json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
    requests=[json.loads(line) for line in (rep/'requests.jsonl').read_text().splitlines() if line.strip()]
    summary,*datasets=analyze(rows,requests)
    cpu_evictions=(summary.get('cpu_cache') or {}).pop('evictions',[])
    if summary.get('cpu_cache'):
        (rep/'cpu_evictions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in cpu_evictions))
    elif (rep/'cpu_evictions.jsonl').exists():
        (rep/'cpu_evictions.jsonl').unlink()
    from agent_io_tracing.serving.kv_candidates import reconcile
    from agent_io_tracing.serving.kv_report_data import bundle_path, read_json, read_rows
    source=bundle_path(rep,read_json(rep/'summary.json'))
    inputs={r['request_id']:r['prompt_token_ids'] for r in read_json(source).get('requests',[])} if source else {}
    reconciliation=reconcile(rows,events if events is not None else read_rows(rep/'kv_events.jsonl'),episodes if episodes is not None else read_rows(rep/'kv_lifecycle.jsonl'),datasets[1],datasets[3],inputs,{r['request_id']:r['source_request_id'] for r in summary['request_paths']})
    summary['candidate_reconciliation']={status:sum(r['status']==status for r in reconciliation) for status in ['confirmed_load','computation_observed','unresolved']}
    summary['candidate_boundaries_inside_store_request_chunks']=sum(bool((r.get('cpu_store_boundary_evidence') or {}).get('boundary_inside_completed_chunk')) for r in reconciliation)
    regions=defaultdict(list)
    for row in reconciliation:regions[row['request_id']].extend(row.get('computed_overlaps',[]))
    repeated_tokens=0
    for intervals in regions.values():
        stop=-1
        for lo,hi in sorted(intervals):repeated_tokens+=max(0,hi-max(lo,stop));stop=max(stop,hi)
    summary['recomputed_candidate_content_tokens']=repeated_tokens
    summary['candidate_reconciliation_scope']='Return publication batches locate hash assignments. Exact request ID, content hash and cache group link each assignment to completed load or computation evidence.'
    (rep/'candidate_reconciliation.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in reconciliation))
    from agent_io_tracing.serving.kv_plots import write_reload_plot
    summary['reload_figure']=write_reload_plot(rep,rows,datasets[0])
    for name,data in zip(['causal_transfers','confirmed_reloads','request_load_waits','recomputed_content'],datasets):
        (rep/(name+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in data))
    (rep/'causal_requests.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in summary['request_paths']))
    (rep/'causal_trace_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
