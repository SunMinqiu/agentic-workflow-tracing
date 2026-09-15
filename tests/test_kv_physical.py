import pytest
from agent_io_tracing.serving.kv_physical import analyze


def row(seq, kind, **fields):
    return dict(process_id=1,publisher_epoch='epoch',trace_session_id='session',sequence=seq,event_type=kind,request_id='engine-a',time_monotonic_ns=5_000_000_000,**fields)


def test_physical_trace_joins_only_exact_request_and_preserves_worker_identity():
    events=[row(0,'PhysicalEviction',last_release_ns=3_000_000_000,block_id=4,allocation_generation=2),row(1,'RequestAllocation')]
    events.append({**row(2,'PhysicalEviction'),'request_id':'unrelated'})
    result,evictions,transfers=analyze(events,[{'request_id':'source-a','sent_request_id':'engine-a'}])
    assert result['physical_evictions']==1 and result['unmatched_events']==1
    assert evictions[0]['time_since_last_release_s']==2
    assert evictions[0]['source_request_id']=='source-a'
    assert result['collection_status']=='unconfirmed'


def test_transfer_submission_and_completion_link_without_cross_process_join():
    submitted=row(0,'TransferSubmitted',job_id=7,submitted_at_ns=1_000_000_000,accepted=True,direction='load')
    completed=row(1,'TransferCompleted',job_id=7,completion_observed_at_ns=2_000_000_000,transfer_bytes=512,success=True,direction='load')
    foreign={**completed,'process_id':2,'sequence':0}
    result,_,transfers=analyze([submitted,completed,foreign],[{'request_id':'source-a','sent_request_id':'engine-a'}])
    assert [t['submission_observed'] for t in transfers]==[True,False]
    assert transfers[0]['submission_to_observed_completion_s']==1
    assert result['successful_transfer_bytes_by_direction']['load']==1024


def test_gap_and_duplicate_coverage():
    first=row(2,'PhysicalEviction')
    result,_,_=analyze([first,first],[{'request_id':'source-a','sent_request_id':'engine-a'}])
    assert result['duplicate_events']==1
    assert result['collection_status']=='partial'
    with pytest.raises(ValueError,match='Conflicting'):
        analyze([first,{**first,'block_id':8}],[])
