import json
import threading
import time
import uuid

import pytest

from agent_io_tracing.serving.kv_collector import ReplayCollector, materialize_batches
from agent_io_tracing.serving.kv_lifecycle import derive_lifecycle
from agent_io_tracing.serving.kv_events import derive_removals


def event(kind, medium='GPU'):
    return {'type':kind, 'block_hashes':['a'], 'parent_block_hash':None, 'token_ids':[1], 'block_size':1, 'lora_id':None, 'medium':medium, 'lora_name':None, 'group_idx':0}


def batch(seq, events, *, epoch=0, phase='measurement', ts=None):
    return {'publisher_id':'p','publisher_epoch':epoch,'sequence':seq,'batch_ts':float(seq if ts is None else ts),'phase':phase,'data_parallel_rank':0,'events':[{'event_type':e['type'],**e} for e in events]}


def test_empty_batch_prevents_false_gap_and_preserves_initial_cpu_state():
    rows,gaps=materialize_batches([batch(0,[event('BlockStored','CPU')],phase='initial'),batch(1,[event('BlockRemoved')]),batch(2,[]),batch(3,[event('BlockStored')])])
    episodes,summary=derive_lifecycle(rows,started_at=999999,ended_at=1000000)
    assert not gaps
    assert summary['return_episodes']==1
    assert summary['cpu_backed_return_candidates']==1
    assert episodes[0]['removal_to_next_store_s']==2


def test_gap_ends_episode_and_invalidates_resident_store():
    rows,gaps=materialize_batches([batch(0,[event('BlockStored','CPU'),event('BlockStored')]),batch(1,[event('BlockRemoved')]),batch(4,[event('BlockStored'),event('BlockRemoved')])])
    episodes,summary=derive_lifecycle(rows)
    assert len(gaps)==1
    assert summary['return_episodes']==0
    assert summary['gap_terminated_episodes']==1
    assert not episodes[0]['cpu_copy_continuously_observed']
    assert summary['missing_sequence_numbers']==2
    assert derive_removals(rows)[-1]['stored_batch_ts']==4


def test_restart_and_duplicate_batches_do_not_join_lifecycles():
    first=batch(0,[event('BlockRemoved')],ts=20)
    second=batch(0,[event('BlockStored')],epoch=1,ts=10)
    rows,gaps=materialize_batches([first,first,second])
    assert len(rows)==2
    assert derive_lifecycle(rows)[1]['return_episodes']==0


def test_continuous_collector_persists_real_zmq_batches(tmp_path):
    zmq=pytest.importorskip('zmq');msgspec=pytest.importorskip('msgspec')
    endpoint='inproc://collector-'+uuid.uuid4().hex
    context=zmq.Context.instance();router=context.socket(zmq.ROUTER);router.bind(endpoint)
    batches=[(0,[1.0,[event('BlockStored','CPU')],0])]
    stopped=threading.Event()
    def serve():
        try:
            while not stopped.is_set():
                if not router.poll(20):continue
                identity,_,start=router.recv_multipart()
                for seq,payload in list(batches):
                    if seq>=int.from_bytes(start,'big'):
                        router.send_multipart([identity,b'',b'kv-events',seq.to_bytes(8,'big'),msgspec.msgpack.encode(payload)])
                router.send_multipart([identity,b'',b'',b'\xff'*8,b''])
        finally:router.close()
    server=threading.Thread(target=serve);server.start()
    collector=ReplayCollector(endpoint,tmp_path,poll_interval_s=.02,timeout_ms=300,process_identity=lambda:1).start()
    try:
        collector.mark('measurement')
        batches.extend([(1,[2.0,[event('BlockRemoved')],0]),(2,[3.0,[],0]),(3,[4.0,[event('BlockStored')],0])])
        deadline=time.monotonic()+2
        while len(collector.batches)<4 and time.monotonic()<deadline:time.sleep(.01)
        rows,transport=collector.stop()
    finally:
        stopped.set();server.join(1)
    assert len((tmp_path/'kv_batches.jsonl').read_text().splitlines())==4
    assert transport['batches']==4 and transport['sequence_gaps']==[]
    assert transport['tail_confirmed'] is False
    assert derive_lifecycle(rows)[1]['cpu_backed_return_candidates']==1
    assert json.loads((tmp_path/'kv_collection.json').read_text())['collection_method']=='continuous_incremental_replay'


def test_missing_batches_invalidate_removal_store_even_without_materialized_gap():
    rows=[batch(0,[event('BlockStored')]),batch(3,[event('BlockRemoved')])]
    flattened=[{**envelope,**item} for envelope in rows for item in envelope['events']]
    assert derive_removals(flattened)[0]['matched_store'] is False


def test_confirmed_publisher_boundary_extends_censoring_without_client_clock():
    rows,_=materialize_batches([batch(0,[event('BlockRemoved')],ts=10)])
    boundaries=[{'phase':'after_measurement','publisher_id':'p','publisher_epoch':0,'boundary_confirmed':True,'server_ack':{'publisher_epoch_s':20}}]
    episodes,_=derive_lifecycle(rows,ended_at=1000000,observation_boundaries=boundaries)
    assert episodes[0]['observed_duration_s']==10
