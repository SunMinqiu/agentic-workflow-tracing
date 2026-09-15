"""Optional hooks for the source-verified vLLM 0.26.0 trace patch."""
from __future__ import annotations

import contextvars
import functools
import inspect
import json
import os
import threading
import time
import uuid
from pathlib import Path

_REQUEST = contextvars.ContextVar('kv_trace_request', default=None)
_OPERATION = contextvars.ContextVar('kv_trace_operation', default='unknown')
_WRITER = None


def safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {'bytes_hex': value.hex()}
    if isinstance(value, (list, tuple, set)):
        return [safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if type(value).__module__.startswith('numpy'):
        return safe(value.tolist())
    return {'type': type(value).__name__}


class TraceWriter:
    def __init__(self, directory):
        self.pid = os.getpid()
        self.epoch = uuid.uuid4().hex
        self.sequence = 0
        self.lock = threading.Lock()
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.stream = (path / f'kv-physical-{self.pid}-{self.epoch}.jsonl').open('a', buffering=1)

    def emit(self, kind, **fields):
        with self.lock:
            row = dict(schema_version=1, event_type=kind, process_id=self.pid,
                       publisher_epoch=self.epoch, trace_session_id=os.environ.get("KV_TRACE_SESSION"), sequence=self.sequence,
                       time_monotonic_ns=time.monotonic_ns(), time_epoch_s=time.time(),
                       request_id=_REQUEST.get(), rank=os.environ.get('RANK'), **fields)
            self.stream.write(json.dumps(safe(row), separators=(',', ':')) + '\n')
            self.sequence += 1


def writer():
    global _WRITER
    directory = os.environ.get('KV_TRACE_DIR')
    if not directory:
        return None
    if _WRITER is None or _WRITER.pid != os.getpid():
        _WRITER = TraceWriter(directory)
    return _WRITER


def wrap(cls, name, hook):
    original = getattr(cls, name)
    signature = inspect.signature(original)
    @functools.wraps(original)
    def traced(self, *args, **kwargs):
        sink = writer()
        if sink is None:
            return original(self, *args, **kwargs)
        values = signature.bind(self, *args, **kwargs)
        values.apply_defaults()
        return hook(sink, original, self, args, kwargs, values.arguments)
    setattr(cls, name, traced)


def install_block_pool(namespace):
    cls = namespace['BlockPool']
    if getattr(cls, '_agent_trace_installed', False):
        return
    cls._agent_trace_installed = True
    def state(pool, block):
        if not hasattr(pool, '_agent_trace_states'):
            pool._agent_trace_states = {}
            pool._agent_trace_pool_id = uuid.uuid4().hex
        return pool._agent_trace_states.setdefault(block.block_id, {'allocation_generation': 0})
    def identity(pool, block):
        st = state(pool, block)
        raw = block.block_hash
        aliases = set(getattr(pool, 'cached_block_hashes_by_block', {}).get(block.block_id, ()))
        if raw is not None:
            aliases.add(raw)
        cached_hashes = [dict(block_hash=namespace['maybe_convert_block_hash'](namespace['get_block_hash'](key)), group_idx=namespace['get_group_id'](key)) for key in sorted(aliases, key=repr)]
        return dict(pool_id=pool._agent_trace_pool_id, block_id=block.block_id,
                    ref_count=block.ref_cnt, cached_hashes=cached_hashes, hash_num_tokens=getattr(block, "block_hash_num_tokens", None), **st,
                    group_idx=namespace['get_group_id'](raw) if raw is not None else None,
                    block_hash=namespace['maybe_convert_block_hash'](namespace['get_block_hash'](raw)) if raw is not None else None)
    def allocation(sink, original, pool, args, kwargs, values):
        token = _OPERATION.set('capacity_reuse')
        try:
            blocks = original(pool, *args, **kwargs)
        finally:
            _OPERATION.reset(token)
        for block in blocks:
            st = state(pool, block)
            st.update(allocation_generation=st['allocation_generation'] + 1,
                      allocated_at_ns=time.monotonic_ns(), last_access_ns=None, last_release_ns=None)
            sink.emit('BlockAllocated', **identity(pool, block))
        return blocks
    def eviction(sink, original, pool, args, kwargs, values):
        block = values['block']
        before = identity(pool, block)
        removed = original(pool, *args, **kwargs)
        if removed:
            kind = 'PhysicalEviction' if _OPERATION.get() == 'capacity_reuse' and before['ref_count'] == 0 else 'CacheInvalidated'
            sink.emit(kind, reason=_OPERATION.get(), **before)
        return removed
    def explicit(sink, original, pool, args, kwargs, values):
        token = _OPERATION.set('explicit_invalidation')
        try:
            return original(pool, *args, **kwargs)
        finally:
            _OPERATION.reset(token)
    def touch(sink, original, pool, args, kwargs, values):
        result = original(pool, *args, **kwargs)
        for block in values['blocks']:
            state(pool, block)['last_access_ns'] = time.monotonic_ns()
            sink.emit('BlockAccessed', **identity(pool, block))
        return result
    def release(sink, original, pool, args, kwargs, values):
        blocks = list(values['ordered_blocks'])
        result = original(pool, blocks)
        for block in blocks:
            if block.ref_cnt == 0 and not block.is_null:
                state(pool, block)['last_release_ns'] = time.monotonic_ns()
                sink.emit('BlockReleased', **identity(pool, block))
        return result
    def stored(sink, original, pool, args, kwargs, values):
        result = original(pool, *args, **kwargs)
        block = values.get('block') or values.get('blk')
        if block is not None:
            key = values['block_hash_with_group_id']
            sink.emit('BlockHashAssigned', assigned_hash=namespace['maybe_convert_block_hash'](namespace['get_block_hash'](key)),
                      assigned_group_idx=namespace['get_group_id'](key), **identity(pool, block))
        return result
    def reset(sink, original, pool, args, kwargs, values):
        result = original(pool, *args, **kwargs)
        sink.emit('CacheReset', success=bool(result), pool_id=getattr(pool, '_agent_trace_pool_id', None))
        return result
    def lookup(sink, original, pool, args, kwargs, values):
        result = original(pool, *args, **kwargs)
        sink.emit('BlockLookup', block_hash=namespace['maybe_convert_block_hash'](values['block_hash']),
                  group_ids=values['kv_cache_group_ids'], all_groups_hit=result is not None,
                  blocks=[identity(pool, b) for b in result] if result else [])
        return result
    for name, hook in [('get_new_blocks', allocation), ('_maybe_evict_cached_block', eviction),
                       ('evict_blocks', explicit), ('touch', touch), ('free_blocks', release),
                       ('_insert_block_hash', stored), ('reset_prefix_cache', reset), ('get_cached_block', lookup)]:
        wrap(cls, name, hook)


def install_manager(namespace):
    cls = namespace['KVCacheManager']
    if getattr(cls, '_agent_trace_installed', False):
        return
    cls._agent_trace_installed = True
    def context(sink, original, manager, args, kwargs, values):
        request = values.get('request')
        token = _REQUEST.set(request.request_id if request is not None else values['request_id'])
        start = time.monotonic_ns()
        try:
            result = original(manager, *args, **kwargs)
            if original.__name__ == 'get_computed_blocks':
                sink.emit('RequestCacheLookup', cached_tokens=result[1],
                          prompt_tokens=request.num_prompt_tokens,
                          lookup_duration_ns=time.monotonic_ns() - start,
                          selected_block_ids=result[0].get_block_ids())
            elif original.__name__ == 'allocate_slots':
                sink.emit('RequestAllocation', success=result is not None,
                          scheduled_compute_tokens=values['num_new_tokens'],
                          external_computed_tokens=values['num_external_computed_tokens'],
                          local_cached_tokens=values['num_new_computed_tokens'])
            return result
        finally:
            _REQUEST.reset(token)
    for name in ('get_computed_blocks', 'allocate_slots', 'free', 'cache_blocks', 'remove_skipped_blocks', 'pop_blocks_for_free'):
        wrap(cls, name, context)


def transfer_spec(spec):
    return {'type': type(spec).__name__, **{key: safe(getattr(spec, key)) for key in ('block_ids', 'group_sizes', 'block_indices', 'num_blocks') if hasattr(spec, key)}}


class TransferWorkerProxy:
    def __init__(self, worker, owner):
        self.worker = worker
        self.owner = owner
        self.worker_id = uuid.uuid4().hex
        self.jobs = {}
    def __getattr__(self, name):
        return getattr(self.worker, name)
    def _submit(self, direction, job_id, src_spec, dst_spec):
        info = dict(worker_id=self.worker_id, job_id=job_id, direction=direction, source=transfer_spec(src_spec),
                    destination=transfer_spec(dst_spec), submitted_at_ns=time.monotonic_ns(),
                    associated_request_id=getattr(self.owner, '_agent_transfer_requests', {}).get(job_id) or self.owner._load_jobs.get(job_id))
        result = getattr(self.worker, 'submit_' + direction)(job_id, src_spec, dst_spec)
        info['accepted'] = bool(result)
        self.jobs[job_id] = info
        writer().emit('TransferSubmitted', **info)
        return result
    def submit_load(self, job_id, src_spec, dst_spec):
        return self._submit('load', job_id, src_spec, dst_spec)
    def submit_store(self, job_id, src_spec, dst_spec):
        return self._submit('store', job_id, src_spec, dst_spec)
    def get_finished(self):
        results = list(self.worker.get_finished())
        for result in results:
            info = self.jobs.pop(result.job_id, {'worker_id':self.worker_id, 'job_id':result.job_id})
            getattr(self.owner, '_agent_transfer_requests', {}).pop(result.job_id, None)
            writer().emit('TransferCompleted', **info, success=bool(result.success),
                          transfer_bytes=result.transfer_size, transfer_time_s=result.transfer_time,
                          completion_observed_at_ns=time.monotonic_ns())
        return results


def install_offload_worker(namespace):
    cls = namespace['OffloadingConnectorWorker']
    if getattr(cls, '_agent_trace_installed', False):
        return
    cls._agent_trace_installed = True
    def initialized(sink, original, owner, args, kwargs, values):
        result = original(owner, *args, **kwargs)
        owner.worker = TransferWorkerProxy(owner.worker, owner)
        return result
    def metadata(sink, original, owner, args, kwargs, values):
        if not hasattr(owner, '_agent_transfer_requests'):
            owner._agent_transfer_requests = {}
        for jobs in (values['metadata'].load_jobs, values['metadata'].store_jobs):
            owner._agent_transfer_requests.update({job_id:job.req_id for job_id, job in jobs.items()})
        return original(owner, *args, **kwargs)
    wrap(cls, '_init_worker', initialized)
    for name in ('start_kv_transfers', 'prepare_store_kv'):
        wrap(cls, name, metadata)


def install_publisher(namespace):
    """Add a publication barrier to replay without changing normal PUB frames."""
    cls = namespace['ZmqEventPublisher']
    if getattr(cls, '_agent_barrier_installed', False):
        return
    cls._agent_barrier_installed = True
    def replay(self):
        import queue
        frame = self._replay.recv_multipart()
        if len(frame) != 3:
            return
        client_id, _, request = frame
        barrier = None
        if len(request) == 8:
            start = int.from_bytes(request, 'big')
        else:
            command = json.loads(request)
            start = int(command['start_sequence'])
            barrier = command['barrier_id']
            # The publisher owns both sockets and sequence allocation. Draining
            # here keeps the barrier ordered after all queued publications.
            queued_at_boundary = self._event_queue.qsize()
            for _ in range(queued_at_boundary):
                try:
                    event = self._event_queue.get_nowait()
                except queue.Empty:
                    break
                if event is None:
                    self._event_queue.task_done()
                    self._event_queue.put(None)
                    break
                sequence = next(self._seq_gen)
                payload = self._pack.encode(event)
                self._pub.send_multipart((self._topic_bytes, sequence.to_bytes(8, 'big'), payload))
                self._buffer.append((sequence, payload))
                self._event_queue.task_done()
        for sequence, payload in self._buffer:
            if sequence >= start:
                self._replay.send_multipart((client_id, b'', self._topic_bytes, sequence.to_bytes(8, 'big'), payload))
        acknowledgement = b''
        if barrier is not None:
            if not hasattr(self, '_agent_publisher_epoch'):
                self._agent_publisher_epoch = uuid.uuid4().hex
            acknowledgement = json.dumps(dict(barrier_id=barrier, publisher_epoch=self._agent_publisher_epoch,
                                             last_sequence=self._buffer[-1][0] if self._buffer else -1,
                                             publisher_epoch_s=time.time(), publication_barrier=True)).encode()
        self._replay.send_multipart((client_id, b'', b'', self.END_SEQ, acknowledgement))
    cls._service_replay = replay
