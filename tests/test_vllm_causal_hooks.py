from types import SimpleNamespace as NS

from agent_io_tracing.serving import vllm_causal_hooks as causal
from agent_io_tracing.serving import vllm_trace_hooks as hooks


class Sink:
    def __init__(self):self.rows=[]
    def emit(self,kind,**fields):self.rows.append(dict(event_type=kind,request_id=hooks._REQUEST.get(),**fields))


def test_scheduler_transfer_completion_waits_for_all_workers(monkeypatch):
    sink=Sink();monkeypatch.setattr(hooks,'writer',lambda:sink)
    monkeypatch.setattr(causal,'content',lambda key:{'offload_key':key.hex(),'block_hash':'a','group_idx':0})
    key=b'a0000'
    class Connector:
        def __init__(self):
            self._jobs={8:NS(req_id='req-a',is_store=False,pending_count=4,keys={key})}
            self.config=NS(blocks_per_chunk=1,kv_group_configs=[NS(tokens_per_chunk=16)])
            self._req_status={'req-a':NS(group_states=[NS(offload_keys=[key],block_ids=[])])}
            self.manager=NS(_agent_cpu_keys={key:{'cpu_block_id':3,'cpu_generation':2}})
        def build_connector_meta(self,scheduler_output):
            return NS(load_jobs={8:NS(req_id='req-a',src_spec=NS(block_ids=[3]),dst_spec=NS(block_ids=[42],group_sizes=[1],block_indices=[0]))},store_jobs={})
        def update_connector_output(self,connector_output):
            self._jobs[8].pending_count-=connector_output
            if self._jobs[8].pending_count==0:del self._jobs[8]
    causal.install_offload_scheduler({'OffloadingConnectorScheduler':Connector})
    owner=Connector();owner.build_connector_meta(None)
    plan=sink.rows[0]
    assert plan['contents'][0]['gpu_block_ids']==[42]
    assert plan['contents'][0]['cpu']['cpu_generation']==2
    assert plan['expected_workers']==4 and plan['request_id']=='req-a'
    owner.update_connector_output(2)
    assert len(sink.rows)==1
    owner.update_connector_output(2)
    assert sink.rows[-1]['event_type']=='TransferAllWorkersComplete'
    assert sink.rows[-1]['job_id']==8


def test_compute_output_is_separate_from_schedule_and_excludes_failure(monkeypatch):
    sink=Sink();monkeypatch.setattr(hooks,'writer',lambda:sink)
    class Scheduler:
        def __init__(self):
            self.requests={'r':NS(num_computed_tokens=8,num_prompt_tokens=10,is_finished=lambda:False)}
            self.kv_cache_manager=NS(get_block_ids=lambda rid:[[3]])
        def add_request(self,request):pass
        def schedule(self):pass
        def _update_after_schedule(self,scheduler_output):self.requests['r'].num_computed_tokens+=4
        def update_from_output(self,scheduler_output,model_runner_output):return {}
        def _update_from_kv_xfer_finished(self,kv_connector_output):pass
        def _try_promote_blocked_waiting_request(self,request):pass
    causal.install_scheduler({'Scheduler':Scheduler})
    owner=Scheduler();output=NS(num_scheduled_tokens={'r':4},scheduled_spec_decode_tokens={})
    owner._update_after_schedule(output)
    assert sink.rows[-1]['event_type']=='RequestComputeScheduled'
    assert not any(r['event_type']=='RequestComputeOutputObserved' for r in sink.rows)
    owner.update_from_output(output,NS(req_id_to_index={'r':0},kv_connector_output=None))
    assert sink.rows[-1]['completed_prompt_tokens']==2
    owner._update_after_schedule(output)
    owner.update_from_output(output,NS(req_id_to_index={'r':0},kv_connector_output=NS(invalid_block_ids={3})))
    assert sink.rows[-1]['usable'] is False
    assert sink.rows[-1]['completed_prompt_tokens'] is None
