"""Run bounded pressure and reset controls."""
import copy
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from agent_io_tracing.replay.runner import replay_to_dir
from agent_io_tracing.replay.sweep import write_report
from agent_io_tracing.serving.vllm_endpoint import VLLMEndpoint
from agent_io_tracing.serving.kv_report_data import read_rows


def main():
    endpoint=VLLMEndpoint('http://127.0.0.1:18000',timeout_s=300)
    config=endpoint.server_info()['vllm_config']['cache_config']
    if int(config['num_gpu_blocks'])!=274 or float(config.get('kv_offloading_size') or 0)!=32:
        raise RuntimeError('Controls require the recorded GPU 274 / CPU 32 GiB configuration')
    source=ROOT/'replay/kv_c2_capacity_20260910/bundles/C2-12.json'
    bundle=json.loads(source.read_text())
    sweep=ROOT/'replay/kv_controls_20260910'
    snapshots=sweep/'bundles';snapshots.mkdir(parents=True,exist_ok=True)
    first=copy.deepcopy(bundle['requests'][0])
    pressure=copy.deepcopy(bundle)
    variants=[token for token in dict.fromkeys(first['prompt_token_ids'][1:]) if token != first['prompt_token_ids'][0]][:96]
    if len(variants)!=96:raise ValueError('Need 96 distinct first-token pressure variants')
    requests=[{**copy.deepcopy(first),'request_id':'control-A-before'}]
    for index,token in enumerate(variants):
        item=copy.deepcopy(first);item['request_id']=f'pressure-{index}'
        item['prompt_token_ids'][0]=token
        requests.append(item)
    requests.append({**copy.deepcopy(first),'request_id':'control-A-after'})
    pressure['requests']=requests
    singleton=copy.deepcopy(bundle);singleton['requests']=[{**first,'request_id':'reset-A'}]
    def run(label,index,data,mode,limit,collect=True):
        snapshot=snapshots/(label+'.json')
        payload=json.dumps(data,sort_keys=True)
        if snapshot.exists() and snapshot.read_text()!=payload:raise ValueError('Control snapshot changed')
        snapshot.write_text(payload)
        rep=sweep/label/f'rep{index}'
        if (rep/'summary.json').exists():return json.loads((rep/'summary.json').read_text())
        if rep.exists() and any(rep.iterdir()):raise ValueError(f'Partial control run: {rep}')
        summary=replay_to_dir(data,endpoint,mode,rep,reset_before=True,bundle_path=snapshot,output_tokens=limit,cache_salt_mode='shared',cache_salt='kv-controls-20260910',kv_events_replay_endpoint='tcp://127.0.0.1:15558' if collect else None)
        print(json.dumps({'control':label,'rep':index,'wall_s':summary['wall_s']}),flush=True)
        return summary
    run('pressure',0,pressure,'serial',1)
    for index in range(2):run('reset',index,singleton,'serial',32)
    pressure_requests=read_rows(sweep/'pressure/rep0/requests.jsonl')
    event_rows=read_rows(sweep/'pressure/rep0/kv_events.jsonl')
    first_store=next(row for row in event_rows if row.get('phase')=='measurement' and row.get('medium')=='GPU' and row.get('event_type')=='BlockStored' and row.get('token_ids',[])[:16]==first['prompt_token_ids'][:16])
    block_hash=first_store['block_hashes'][0]
    sequence=[{'type':row['event_type'],'medium':row.get('medium'),'group':row.get('group_idx'),'sequence':row['sequence']} for row in event_rows if row.get('phase')=='measurement' and all(row.get(key)==first_store.get(key) for key in ('publisher_id','publisher_epoch','data_parallel_rank','group_idx')) and block_hash in (row.get('block_hashes') or [])]
    report=json.loads((sweep/'pressure/rep0/kv_report_data.json').read_text())
    check={'first_cached_tokens':pressure_requests[0]['cached_tokens'],'last_cached_tokens':pressure_requests[-1]['cached_tokens'],'logical_removal_episodes':report['lifecycle']['removal_episodes'],'logical_return_episodes':report['lifecycle']['return_episodes'],'a_first_hash':block_hash,'a_group':first_store.get('group_idx'),'a_hash_events':sequence,'collection_status':report['collection_status']}
    (sweep/'pressure/rep0/pressure_sequence_check.json').write_text(json.dumps(check,indent=2)+'\n')
    findings={'pressure':{'description':'A, 96 variants of A with distinct first prompt tokens, A; output limit 1. Synthetic validation separate from the original-bundle matrix.','first_cached_tokens':pressure_requests[0].get('cached_tokens'),'last_cached_tokens':pressure_requests[-1].get('cached_tokens')},'reset_cached_tokens':[read_rows(sweep/f'reset/rep{index}/requests.jsonl')[0].get('cached_tokens') for index in range(2)]}
    (sweep/'control_findings.json').write_text(json.dumps(findings,indent=2)+'\n')
    write_report(sweep,results_dir=ROOT/'results')
    print(json.dumps(findings,indent=2),flush=True)


if __name__=='__main__':
    main()
