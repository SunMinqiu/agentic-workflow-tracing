from agent_io_tracing.serving.kv_candidates import reconcile, cpu_lookup_history


def test_cpu_miss_history_distinguishes_absent_pending_and_evicted():
    base = dict(process_id=1, publisher_epoch='epoch', trace_session_id='session', offload_key='full-key')
    kinds = ['CPULookup', 'CPUBlockAssigned', 'CPULookup', 'CPUStoreReady', 'CPUContentEvicted', 'CPULookup']
    rows = [dict(base, event_type=kind, sequence=i, time_monotonic_ns=i) for i,kind in enumerate(kinds)]
    history = cpu_lookup_history(list(reversed(rows)))
    assert history[('session',1,'epoch',0)]['prior_cpu_event'] == 'no_prior_cpu_event_observed'
    assert history[('session',1,'epoch',2)]['prior_cpu_event'] == 'CPUBlockAssigned'
    assert history[('session',1,'epoch',5)]['prior_cpu_event'] == 'CPUContentEvicted'
    rows.append(dict(rows[-1], publisher_epoch='new', sequence=0, time_monotonic_ns=10))
    assert cpu_lookup_history(rows)[('session',1,'new',0)]['prior_cpu_event'] == 'no_prior_cpu_event_observed'


def fixture():
    episode=dict(block_hash='h',group_idx=3,status='returned',cpu_copy_continuously_observed=True,removed_sequence=2,return_sequence=3,return_batch_ts=3)
    events=[dict(sequence=2,batch_ts=2,event_type='BlockRemoved'),dict(sequence=3,batch_ts=3,event_type='BlockStored',medium='GPU',block_hashes=['h'],group_idx=3,block_size=2,token_ids=[30,40],kv_cache_spec_kind='full_attention')]
    rows=[dict(event_type='BlockHashAssigned',assigned_hash='h',assigned_group_idx=3,request_id='r',time_epoch_s=2.5,time_monotonic_ns=250),dict(event_type='RequestRegistered',request_id='r',hash_block_size=2,prompt_hashes=['old','h']),dict(event_type='RequestComputeOutputObserved',request_id='r',usable=True,token_start=2,token_end=4,prompt_tokens=4)]
    source=dict(request_id='r',source_request_id='source',block_hash='h',group_idx=3)
    return episode,events,rows,source


def test_candidate_load_requires_returning_request_identity():
    episode,events,rows,source=fixture()
    loads=[dict(source,job_id=7,all_workers_observed_at_ns=240)]
    assert reconcile(rows,events,[episode],loads,[],{})[0]['status']=='confirmed_load'
    loads[0]['request_id']='other'
    assert reconcile(rows,events,[episode],loads,[],{})[0]['status']=='unresolved'


def test_recomputed_full_chunk_requires_exact_prompt_tokens():
    episode,events,rows,source=fixture()
    result=reconcile(rows,events,[episode],[],[source],{'source':[10,20,30,40]})[0]
    assert result['entire_returned_chunk_computed']
    assert result['completed_content_tokens']==2
    mismatch=reconcile(rows,events,[episode],[],[source],{'source':[10,20,99,40]})[0]
    assert 'entire_returned_chunk_computed' not in mismatch


def test_cpu_copy_boundary_links_to_the_exact_storing_allocation():
    episode,events,rows,source=fixture()
    cpu=dict(cpu_pool_id='pool',cpu_block_id=5,cpu_generation=2,offload_key='full-key')
    rows += [dict(cpu,event_type='CPUBlockAssigned',request_id='origin',time_monotonic_ns=100,sequence=1),
             dict(event_type='RequestComputeOutputObserved',request_id='origin',usable=True,token_start=2,token_end=6,prompt_tokens=6,time_monotonic_ns=150),
             dict(event_type='CPULookup',request_id='r',block_hash='h',group_idx=3,result='HIT',cpu=cpu,time_monotonic_ns=200)]
    result=reconcile(rows,events,[episode],[],[source],{'source':[10,20,30,40]},{'origin':'original-source'})[0]
    proof=result['cpu_store_boundary_evidence']
    assert proof['cpu_store_source_request_id']=='original-source'
    assert proof['cache_boundary_tokens']==4
    assert proof['completed_covering_steps']==[[2,6]]
    assert proof['boundary_inside_completed_chunk']
    rows[-1]['cpu']={**cpu,'cpu_generation':3}
    assert 'cpu_store_boundary_evidence' not in reconcile(rows,events,[episode],[],[source],{'source':[10,20,30,40]})[0]


def test_ambiguous_return_batch_does_not_choose_a_request():
    episode,events,rows,source=fixture()
    rows.append({**rows[0],'request_id':'another'})
    assert reconcile(rows,events,[episode],[],[source],{})[0]['status']=='unresolved'
