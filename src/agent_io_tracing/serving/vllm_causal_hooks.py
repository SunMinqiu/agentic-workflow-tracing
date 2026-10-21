"""Request, CPU content, and scheduler completion records for audited vLLM."""
from __future__ import annotations

import hashlib
import json
import time
import uuid

from agent_io_tracing.serving.vllm_trace_hooks import _REQUEST, safe, transfer_spec, wrap


def content(key):
    from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash
    return dict(offload_key=bytes(key).hex(), block_hash=maybe_convert_block_hash(key[:-4]), group_idx=int.from_bytes(key[-4:], 'big'))


def emit_request(sink, kind, request_id, **fields):
    token = _REQUEST.set(request_id)
    try:
        sink.emit(kind, **fields)
    finally:
        _REQUEST.reset(token)


def install_cpu_manager(namespace):
    cls = namespace['CPUOffloadingManager']
    if getattr(cls, '_agent_causal_installed', False):
        return
    cls._agent_causal_installed = True
    def allocate(sink, original, owner, args, kwargs, values):
        blocks = original(owner, *args, **kwargs)
        if not hasattr(owner, '_agent_cpu_id'):
            owner._agent_cpu_id = uuid.uuid4().hex
            owner._agent_cpu_generations = {}
            owner._agent_cpu_keys = {}
        for key, block in zip(values['keys'], blocks):
            generation = owner._agent_cpu_generations.get(block.block_id, 0) + 1
            owner._agent_cpu_generations[block.block_id] = generation
            record = dict(cpu_pool_id=owner._agent_cpu_id, cpu_block_id=block.block_id,
                          cpu_generation=generation, **content(key))
            owner._agent_cpu_keys[key] = record
            sink.emit('CPUBlockAssigned', **record)
        return blocks
    def operation(sink, original, owner, args, kwargs, values):
        token = _REQUEST.set(values['req_context'].req_id)
        try:
            result = original(owner, *args, **kwargs)
            records = getattr(owner, '_agent_cpu_keys', {})
            if original.__name__ == 'lookup':
                sink.emit('CPULookup', result=result.name, **content(values['key']), cpu=records.get(values['key']))
            elif original.__name__ == 'prepare_store' and result is not None:
                for key in result.evicted_keys:
                    sink.emit('CPUContentEvicted', **records.pop(key, content(key)))
            elif original.__name__ in ('complete_store', 'complete_load'):
                for key in values['keys']:
                    sink.emit('CPUStoreReady' if original.__name__ == 'complete_store' else 'CPULoadReleased', **records.get(key, content(key)))
            return result
        finally:
            _REQUEST.reset(token)
    wrap(cls, '_allocate_blocks', allocate)
    for name in ('lookup', 'prepare_store', 'prepare_load', 'complete_store', 'complete_load'):
        wrap(cls, name, operation)


def transfer_contents_v029(owner, req_state, status, job, direction):
    """Map actual transfer slots, including recurrent copy-on-write tails."""
    spec = job.dst_spec if direction == 'load' else job.src_spec
    ends = {bytes(h): (i + 1) * owner.config.tokens_per_hash
            for i, h in enumerate(req_state.req.block_hashes)}
    records = []
    offset = 0
    for group_idx, config in enumerate(owner.config.kv_group_configs):
        size = spec.group_sizes[group_idx]
        ids = list(spec.block_ids[offset:offset + size])
        offset += size
        keys = sorted((key for key in status.keys if int.from_bytes(key[-4:], 'big') == group_idx),
                      key=lambda key: ends[bytes(key[:-4])])
        if owner.config.blocks_per_chunk != 1 or len(keys) != len(ids):
            raise ValueError('Transfer mapping requires one GPU block per offload key')
        for key, block_id in zip(keys, ids):
            end = ends[bytes(key[:-4])]
            start = ((end - 1) // config.tokens_per_chunk) * config.tokens_per_chunk
            records.append(dict(**content(key), token_start=start, token_end=end,
                                gpu_block_ids=[int(block_id)],
                                cpu=getattr(owner.manager, '_agent_cpu_keys', {}).get(key)))
    return records


def install_offload_scheduler(namespace):
    cls = namespace['OffloadingConnectorScheduler']
    if getattr(cls, '_agent_causal_installed', False):
        return
    cls._agent_causal_installed = True
    def build(sink, original, owner, args, kwargs, values):
        meta = original(owner, *args, **kwargs)
        if not hasattr(owner, '_agent_connector_id'):
            owner._agent_connector_id = uuid.uuid4().hex
        for direction, jobs in [('load', meta.load_jobs), ('store', meta.store_jobs)]:
            for job_id, job in jobs.items():
                status = owner._jobs[job_id]
                req_state = owner._req_status[job.req_id]
                contents = []
                gpu_offset = 0
                for group_index, (config, group) in enumerate(zip(owner.config.kv_group_configs, req_state.group_states)):
                    for index, key in enumerate(group.offload_keys):
                        if key not in status.keys:
                            continue
                        first = index * owner.config.blocks_per_chunk
                        if direction == 'load':
                            spec = job.dst_spec
                            group_size = spec.group_sizes[group_index]
                            group_ids = spec.block_ids[gpu_offset:gpu_offset + group_size]
                            base = spec.block_indices[group_index]
                            lo = max(0, first - base)
                            hi = min(group_size, first + owner.config.blocks_per_chunk - base)
                            block_ids = group_ids[lo:hi] if hi > lo else []
                        else:
                            block_ids = group.block_ids[first:first + owner.config.blocks_per_chunk]
                        contents.append(dict(**content(key), token_start=index * config.tokens_per_chunk,
                                             token_end=(index + 1) * config.tokens_per_chunk,
                                             gpu_block_ids=list(block_ids),
                                             cpu=getattr(owner.manager, '_agent_cpu_keys', {}).get(key)))
                    if direction == 'load':
                        gpu_offset += job.dst_spec.group_sizes[group_index]
                if hasattr(owner.config, 'supports_partial_tail'):
                    contents = transfer_contents_v029(owner, req_state, status, job, direction)
                emit_request(sink, 'TransferPlan', job.req_id, connector_id=owner._agent_connector_id,
                             job_id=job_id, direction=direction, expected_workers=status.pending_count,
                             source=transfer_spec(job.src_spec), destination=transfer_spec(job.dst_spec),
                             blocks_per_chunk=owner.config.blocks_per_chunk, contents=contents, keys=[content(key) for key in sorted(status.keys)])
        return meta
    def completion(sink, original, owner, args, kwargs, values):
        before = {job_id:dict(request_id=status.req_id, direction='store' if status.is_store else 'load', pending_workers=status.pending_count)
                  for job_id, status in owner._jobs.items()}
        result = original(owner, *args, **kwargs)
        for job_id, info in before.items():
            if job_id not in owner._jobs:
                emit_request(sink, 'TransferAllWorkersComplete', info['request_id'],
                             connector_id=getattr(owner, '_agent_connector_id', None), job_id=job_id,
                             direction=info['direction'], pending_workers_before=info['pending_workers'])
        return result
    wrap(cls, 'build_connector_meta', build)
    wrap(cls, 'update_connector_output', completion)


def install_scheduler(namespace):
    cls = namespace['Scheduler']
    if getattr(cls, '_agent_causal_installed', False):
        return
    cls._agent_causal_installed = True
    def register(sink, original, owner, args, kwargs, values):
        request = values['request']
        result = original(owner, *args, **kwargs)
        from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash
        emit_request(sink, 'RequestRegistered', request.request_id, cache_salt=request.cache_salt,
                     hash_block_size=owner.kv_cache_manager.block_pool.hash_block_size,
                     prompt_hashes=[maybe_convert_block_hash(h) for h in request.block_hashes],
                     prompt_tokens=request.num_prompt_tokens,
                     prompt_sha256=hashlib.sha256(json.dumps(request.prompt_token_ids, separators=(',', ':')).encode()).hexdigest(),
                     sampling_extra_args=safe(getattr(request.sampling_params, 'extra_args', None)))
        return result
    def schedule(sink, original, owner, args, kwargs, values):
        before = {rid:getattr(req.status, 'name', str(req.status)) for rid, req in owner.requests.items()}
        result = original(owner, *args, **kwargs)
        for rid, req in owner.requests.items():
            if req.status.name == 'WAITING_FOR_REMOTE_KVS' and before.get(rid) != req.status.name:
                emit_request(sink, 'RequestLoadBlocked', rid, num_computed_tokens=req.num_computed_tokens)
        return result
    def scheduled(sink, original, owner, args, kwargs, values):
        output = values['scheduler_output']
        step = getattr(owner, '_agent_step', 0)
        owner._agent_step = step + 1
        if not hasattr(owner, '_agent_outputs'):
            owner._agent_outputs = {}
        plans = {}
        for rid, count in output.num_scheduled_tokens.items():
            req = owner.requests[rid]
            start = req.num_computed_tokens
            plans[rid] = dict(step_id=step, token_start=start, token_end=start + count,
                              prompt_tokens=req.num_prompt_tokens, scheduled_at_ns=time.monotonic_ns(),
                              block_ids=owner.kv_cache_manager.get_block_ids(rid))
            emit_request(sink, 'RequestComputeScheduled', rid, **plans[rid])
        owner._agent_outputs[id(output)] = plans
        return original(owner, *args, **kwargs)
    def finished(sink, original, owner, args, kwargs, values):
        output = values['scheduler_output']
        model_output = values['model_runner_output']
        plans = getattr(owner, '_agent_outputs', {}).pop(id(output), {})
        active = {rid for rid in plans if rid in owner.requests and not owner.requests[rid].is_finished()}
        result = original(owner, *args, **kwargs)
        failed = bool(model_output.kv_connector_output and model_output.kv_connector_output.invalid_block_ids)
        for rid, plan in plans.items():
            usable = rid in active and rid in model_output.req_id_to_index and not failed and not output.scheduled_spec_decode_tokens.get(rid)
            emit_request(sink, 'RequestComputeOutputObserved', rid, **plan, usable=usable,
                         completion_observed_at_ns=time.monotonic_ns(),
                         completed_prompt_tokens=max(0, min(plan['token_end'], plan['prompt_tokens'])-min(plan['token_start'], plan['prompt_tokens'])) if usable else None)
        return result
    def received(sink, original, owner, args, kwargs, values):
        result = original(owner, *args, **kwargs)
        for rid in values['kv_connector_output'].finished_recving or ():
            emit_request(sink, 'RequestLoadCompletionObserved', rid)
        return result
    def promoted(sink, original, owner, args, kwargs, values):
        req = values['request']
        was_blocked = req.status.name == 'WAITING_FOR_REMOTE_KVS'
        result = original(owner, *args, **kwargs)
        if was_blocked and result:
            emit_request(sink, 'RequestLoadUnblocked', req.request_id,
                         num_computed_tokens=req.num_computed_tokens,
                         block_ids=owner.kv_cache_manager.get_block_ids(req.request_id))
        return result
    for name, hook in [('add_request', register), ('schedule', schedule), ('_update_after_schedule', scheduled),
                       ('update_from_output', finished), ('_update_from_kv_xfer_finished', received),
                       ('_try_promote_blocked_waiting_request', promoted)]:
        wrap(cls, name, hook)
