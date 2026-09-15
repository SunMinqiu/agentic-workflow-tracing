from agent_io_tracing.serving.kv_transfers import window


def sample(direction, value, engine='0'):
    return '\n'.join(f'vllm:{name}{{engine="{engine}",transfer_type="{direction}"}} {value}' for name in ['kv_offload_total_bytes_total','kv_offload_total_time_total','kv_offload_size_count'])


def test_directions_and_worker_series_remain_separate():
    before=sample('GPU_to_CPU',10)+'\n'+sample('CPU_to_GPU',2)+'\n'+sample('GPU_to_CPU',30,'1')
    after=sample('GPU_to_CPU',15)+'\n'+sample('CPU_to_GPU',3)+'\n'+sample('GPU_to_CPU',32,'1')
    result=window(before,after)
    assert result['status']=='observed'
    assert result['directions']['GPU_to_CPU']['bytes']==7
    assert result['directions']['CPU_to_GPU']['bytes']==1


def test_missing_series_and_reset_are_not_zero():
    result=window(sample('CPU_to_GPU',10),sample('CPU_to_GPU',2))
    assert result['directions']['CPU_to_GPU']['bytes'] is None
    assert result['status']=='partial'
    assert window('', '')['status']=='not_collected'
    assert window('',sample('CPU_to_GPU',2))['directions']['CPU_to_GPU']['bytes'] is None


def test_measured_zero_is_available():
    result=window(sample('CPU_to_GPU',0),sample('CPU_to_GPU',0))
    assert result['directions']['CPU_to_GPU']['bytes']==0
    assert result['status']=='observed'


def test_process_restart_invalidates_even_increasing_counters():
    before='process_start_time_seconds 10\n'+sample('CPU_to_GPU',10)
    after='process_start_time_seconds 20\n'+sample('CPU_to_GPU',50)
    assert window(before,after)['directions']['CPU_to_GPU']['bytes'] is None
