import copy
import pytest
from agent_io_tracing.serving.kv_causal import analyze


def evidence():
    def row(seq,kind,**fields):
        return dict(trace_session_id='s',process_id=1,publisher_epoch='engine',sequence=seq,time_monotonic_ns=seq*1000000,event_type=kind,request_id='internal',**fields)
    content=dict(block_hash='h',group_idx=0,gpu_block_ids=[8],cpu={'cpu_pool_id':'cpu','cpu_block_id':2,'cpu_generation':3})
    rows=[row(0,'PhysicalEviction',cached_hashes=[{'block_hash':'h','group_idx':0}],block_id=5,allocation_generation=2),
          row(1,'RequestRegistered',sampling_extra_args={'kv_trace_request_id':'sent'},prompt_hashes=['h'],hash_block_size=16),
          row(2,'TransferPlan',job_id=7,connector_id='c',direction='load',expected_workers=2,contents=[content]),
          row(3,'RequestLoadBlocked')]
    for pid in [2,3]:
        rows.append(dict(trace_session_id='s',process_id=pid,publisher_epoch=f'w{pid}',sequence=0,time_monotonic_ns=4000000,event_type='TransferCompleted',request_id=None,associated_request_id='internal',worker_id=f'worker{pid}',job_id=7,direction='load',accepted=True,success=True,transfer_bytes=100,transfer_time_s=.001))
    rows.extend([row(4,'TransferAllWorkersComplete',job_id=7,connector_id='c'),row(5,'RequestLoadCompletionObserved'),row(6,'RequestLoadUnblocked'),row(7,'RequestComputeOutputObserved',usable=True,block_ids=[[8]],step_id=1,completed_prompt_tokens=0,token_start=16,token_end=17,prompt_tokens=16)])
    return rows,[{'request_id':'source','sent_request_id':'sent'}]


def test_confirm_requires_every_worker_and_coordinator():
    rows,requests=evidence()
    summary,transfers,reloads,waits,_=analyze(rows,requests)
    assert summary['confirmed_load_jobs']==1
    assert transfers[0]['worker_bytes']==200
    assert reloads[0]['previous_physical_eviction_observed']
    assert reloads[0]['loaded_blocks_present_in_completed_request_step']
    assert waits[0]['observed_blocked_to_unblocked_s']==.003
    assert analyze([r for r in rows if r['process_id']!=3],requests)[0]['confirmed_load_jobs']==0
    assert analyze([r for r in rows if r['event_type']!='TransferAllWorkersComplete'],requests)[0]['confirmed_load_jobs']==0
    broken=copy.deepcopy(rows);next(r for r in broken if r['process_id']==3)['success']=False
    assert analyze(broken,requests)[0]['confirmed_load_jobs']==0


def test_duplicate_snapshots_do_not_duplicate_confirmed_jobs():
    rows,requests=evidence()
    assert analyze(rows+rows,requests)[0]['confirmed_load_jobs']==1
    with pytest.raises(ValueError,match='Conflicting'):
        analyze(rows+[{**rows[0],'block_id':999}],requests)


def test_loaded_block_in_other_group_does_not_prove_usage():
    rows,requests=evidence();rows[-1]['block_ids']=[[],[8]]
    assert not analyze(rows,requests)[2][0]['loaded_blocks_present_in_completed_request_step']


def test_reallocated_slot_id_does_not_prove_the_loaded_allocation_was_used():
    rows,requests=evidence()
    assert analyze(rows,requests)[0]['reload_records_with_stable_gpu_allocation']==0
    allocation=dict(rows[0],event_type='BlockAllocated',sequence=100,time_monotonic_ns=1500000,block_id=8,pool_id='gpu',allocation_generation=3)
    rows.append(allocation)
    assert analyze(rows,requests)[0]['reload_records_with_stable_gpu_allocation']==1
    rows.append(dict(allocation,sequence=101,time_monotonic_ns=6500000,allocation_generation=4))
    for i,row in enumerate(sorted(rows,key=lambda r:r['time_monotonic_ns'])):
        if row['process_id']==1:row['sequence']=i
    summary,_,reloads,_,_=analyze(rows,requests)
    assert reloads[0]['loaded_blocks_present_in_completed_request_step']
    assert summary['reload_records_with_stable_gpu_allocation']==0
    assert reloads[0]['loaded_block_allocation_proof'][0]['status']=='reallocated'


def test_unrelated_request_does_not_join_by_suffix():
    rows,requests=evidence();requests[0]['sent_request_id']='different'
    assert analyze(rows,requests)[0]['registered_requests']==0


def test_recomputed_hash_units_do_not_expand_cumulative_prefix_length():
    rows,requests=evidence();rows[0]['hash_num_tokens']=1000
    rows[1]['prompt_hashes']=['older','h']
    rows[-1].update(token_start=0,token_end=32,prompt_tokens=32,completed_prompt_tokens=32)
    summary,*_=analyze(rows,requests)
    assert summary['recomputed_terminal_hash_unit_tokens']==16


def test_worker_destination_must_match_scheduler_plan():
    rows,requests=evidence()
    next(r for r in rows if r['event_type']=='TransferPlan')['destination']={'block_ids':[8]}
    for row in rows:
        if row['event_type']=='TransferCompleted':row['destination']={'block_ids':[8]}
    assert analyze(rows,requests)[0]['confirmed_load_jobs']==1
    next(r for r in rows if r['process_id']==3)['destination']={'block_ids':[99]}
    assert analyze(rows,requests)[0]['confirmed_load_jobs']==0


def test_repeated_prompt_counts_only_computation_completed_before_new_request():
    rows,requests=evidence()
    rows[1].update(prompt_sha256='same',cache_salt='salt',prompt_tokens=16)
    rows[-1].update(token_start=0,token_end=16,prompt_tokens=16,completed_prompt_tokens=16)
    second={**rows[1],'sequence':8,'time_monotonic_ns':8000000,'request_id':'second','sampling_extra_args':{'kv_trace_request_id':'sent2'}}
    done={**rows[-1],'sequence':9,'time_monotonic_ns':9000000,'request_id':'second'}
    requests.append({'request_id':'source2','sent_request_id':'sent2'})
    result=analyze(rows+[second,done],requests)[0]
    assert result['repeated_same_prompt_computed_tokens']==16
    second['time_monotonic_ns']=6000000
    assert analyze(rows+[second,done],requests)[0]['repeated_same_prompt_computed_tokens']==0
