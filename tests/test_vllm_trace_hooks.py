from collections import deque
from itertools import count
from queue import Queue
from types import SimpleNamespace
import json
import threading
import uuid

import pytest

from agent_io_tracing.serving import vllm_trace_hooks as hooks
from agent_io_tracing.serving.kv_events import fetch_replay


@pytest.fixture
def sink(tmp_path, monkeypatch):
    monkeypatch.setenv('KV_TRACE_DIR',str(tmp_path))
    monkeypatch.setattr(hooks,'_WRITER',None)
    yield tmp_path
    if hooks._WRITER:
        hooks._WRITER.stream.close()
    hooks._WRITER=None


def test_physical_eviction_is_distinct_from_hash_assignment(sink):
    class Pool:
        def __init__(self):
            self.block=SimpleNamespace(block_id=1,block_hash=('a',0),ref_cnt=0,is_null=False)
        def get_new_blocks(self,num_blocks):
            self._maybe_evict_cached_block(self.block);self.block.ref_cnt=1
            return [self.block]
        def _maybe_evict_cached_block(self,block):
            removed=block.block_hash is not None;block.block_hash=None;return removed
        def evict_blocks(self,block_ids):
            self._maybe_evict_cached_block(self.block)
        def touch(self,blocks):
            for b in blocks:b.ref_cnt+=1
        def free_blocks(self,ordered_blocks):
            for b in ordered_blocks:b.ref_cnt-=1
        def _insert_block_hash(self,block_hash_with_group_id,block,num_tokens):
            block.block_hash=block_hash_with_group_id
        def reset_prefix_cache(self):return True
        def get_cached_block(self,block_hash,kv_cache_group_ids):return [self.block]
    hooks.install_block_pool({'BlockPool':Pool,'get_group_id':lambda h:h[1],'get_block_hash':lambda h:h[0],'maybe_convert_block_hash':lambda h:h})
    pool=Pool();pool.get_new_blocks(1)
    pool._insert_block_hash(('b',0),pool.block,1)
    pool._insert_block_hash(('c',0),pool.block,2)
    pool.free_blocks(iter([pool.block]))
    pool.get_new_blocks(1)
    rows=[json.loads(s) for p in sink.glob('*.jsonl') for s in p.read_text().splitlines()]
    evictions=[r for r in rows if r['event_type']=='PhysicalEviction']
    assert [r['block_hash'] for r in evictions]==['a','c']
    assert all(r['reason']=='capacity_reuse' for r in evictions)
    assert evictions[-1]['allocation_generation']==1
    assert evictions[-1]['last_release_ns'] is not None
    assert len([r for r in rows if r['event_type']=='BlockHashAssigned'])==2


def test_transfer_completion_preserves_generator_results(sink):
    result=SimpleNamespace(job_id=7,success=True,transfer_size=4096,transfer_time=.002)
    class Worker:
        def submit_load(self,*args):return True
        def get_finished(self):yield result
    owner=SimpleNamespace(_load_jobs={7:'request-a'})
    proxy=hooks.TransferWorkerProxy(Worker(),owner)
    proxy.submit_load(7,SimpleNamespace(block_ids=[3]),SimpleNamespace(block_ids=[9]))
    assert proxy.get_finished()==[result]
    rows=[json.loads(s) for p in sink.glob('*.jsonl') for s in p.read_text().splitlines()]
    assert [r['event_type'] for r in rows]==['TransferSubmitted','TransferCompleted']
    assert rows[-1]['transfer_bytes']==4096
    assert rows[-1]['associated_request_id']=='request-a'
    assert rows[-1]['transfer_time_s']==.002


def test_publication_barrier_drains_queue_before_acknowledgement():
    zmq=pytest.importorskip('zmq');msgspec=pytest.importorskip('msgspec')
    endpoint='inproc://barrier-'+uuid.uuid4().hex
    ctx=zmq.Context.instance();router=ctx.socket(zmq.ROUTER);router.bind(endpoint)
    class Publisher:pass
    hooks.install_publisher({'ZmqEventPublisher':Publisher})
    publisher=Publisher();publisher._replay=router;publisher._event_queue=Queue()
    publisher._event_queue.put([1.0,[{'type':'AllBlocksCleared'}],0])
    publisher._buffer=deque();publisher._seq_gen=count();publisher._pack=msgspec.msgpack.Encoder()
    publisher._pub=SimpleNamespace(send_multipart=lambda frames:None)
    publisher._topic_bytes=b'kv-events';publisher.END_SEQ=b'\xff'*8
    def serve():
        publisher._service_replay();router.close()
    thread=threading.Thread(target=serve);thread.start()
    rows,transport=fetch_replay(endpoint,barrier_id='end-marker',timeout_ms=1000)
    thread.join(1)
    assert len(rows)==1
    assert publisher._event_queue.unfinished_tasks==0
    assert transport['barrier_ack']['barrier_id']=='end-marker'
    assert transport['barrier_ack']['last_sequence']==0
    assert transport['barrier_ack']['publication_barrier'] is True


def test_manager_accepts_request_id_for_remove_skipped_blocks(sink):
    class Manager:
        def get_computed_blocks(self, request):return (SimpleNamespace(get_block_ids=lambda:[]),0,0)
        def allocate_slots(self, request, num_new_tokens, num_external_computed_tokens=0, num_new_computed_tokens=0):return []
        def free(self, request):return request.request_id
        def cache_blocks(self, request, num_computed_tokens):return None
        def remove_skipped_blocks(self, request_id, processed_computed_tokens, num_prompt_tokens=None):
            hooks.writer().emit('ReleaseObserved')
            return request_id
        def pop_blocks_for_free(self, request):return []
    hooks.install_manager({'KVCacheManager':Manager})
    manager=Manager()
    assert manager.remove_skipped_blocks('internal-r',2991,2991)=='internal-r'
    assert manager.free(SimpleNamespace(request_id='internal-r'))=='internal-r'
    rows=[json.loads(s) for p in sink.glob('*.jsonl') for s in p.read_text().splitlines()]
    assert rows[-1]['request_id']=='internal-r'
    assert hooks._REQUEST.get() is None
