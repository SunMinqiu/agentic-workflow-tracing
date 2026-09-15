from agent_io_tracing.serving.kv_cpu import analyze_cpu_cache


def event(kind, timestamp, generation=1, request='current'):
    return dict(event_type=kind,time_monotonic_ns=timestamp*1_000_000_000,cpu_pool_id='pool',cpu_block_id=0,cpu_generation=generation,offload_key=f'key-{generation}',block_hash=f'h{generation}',group_idx=3,request_id=request)


def test_cpu_eviction_separates_prior_workload_and_current_slot_generation():
    rows=[event('CPUBlockAssigned',1,request='earlier'),event('CPUStoreReady',2,request='earlier'),event('CPUContentEvicted',4),event('CPUBlockAssigned',5,generation=2),event('CPUStoreReady',6,generation=2),event('CPUContentEvicted',7,generation=2)]
    data=analyze_cpu_cache(rows,{'current':'source'})
    assert data['eviction_events']==2
    assert data['assignment_events']==1
    assert data['victims_created_in_run']==data['victims_from_other_requests']==1
    assert [r['since_assignment_s'] for r in data['evictions']]==[3,2]
    assert [r['since_store_ready_s'] for r in data['evictions']]==[2,1]


def test_unobserved_origin_has_no_zero_age_and_gpu_only_has_no_cpu_summary():
    data=analyze_cpu_cache([event('CPUContentEvicted',4)],{'current':'source'})
    assert data['victims_with_unknown_origin']==1
    assert data['evictions'][0]['since_assignment_s'] is None
    assert data['duration_statistics']=={}
    assert analyze_cpu_cache([],{'current':'source'})=={}
