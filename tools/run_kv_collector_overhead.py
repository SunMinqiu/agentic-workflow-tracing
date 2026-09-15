"""Measure client collector overhead with cache namespaces isolated per run."""
import json
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from agent_io_tracing.replay.runner import replay_to_dir
from agent_io_tracing.replay.sweep import write_report
from agent_io_tracing.serving.kv_report_data import read_rows
from agent_io_tracing.serving.vllm_endpoint import VLLMEndpoint


def main():
    endpoint=VLLMEndpoint('http://127.0.0.1:18000',timeout_s=300)
    source=ROOT/'replay/kv_c2_capacity_20260910/bundles/C2-12.json'
    bundle=json.loads(source.read_text())
    sweep=ROOT/'replay/kv_collector_overhead_20260910'
    pairs=[]
    for index in range(3):
        pair={'pair':index}
        for enabled in ([False,True] if index%2==0 else [True,False]):
            config=endpoint.server_info()['vllm_config']['cache_config']
            if int(config['num_gpu_blocks'])!=274 or float(config.get('kv_offloading_size') or 0)!=32:
                raise RuntimeError('Overhead comparison requires GPU 274 / CPU 32 GiB')
            label='on' if enabled else 'off';rep=sweep/label/f'rep{index}'
            if (rep/'summary.json').exists():summary=json.loads((rep/'summary.json').read_text())
            else:
                if rep.exists() and any(rep.iterdir()):raise ValueError(f'Partial repetition: {rep}')
                summary=replay_to_dir(bundle,endpoint,'packed',rep,reset_before=True,bundle_path=source,output_tokens=32,cache_salt_mode='shared',cache_salt=f'kv-cold-overhead-{label}-{index}',kv_events_replay_endpoint='tcp://127.0.0.1:15558' if enabled else None)
            requests=read_rows(rep/'requests.jsonl')
            pair[label]={'wall_s':summary['wall_s'],'ttft_p50_ms':summary['median_ttft_ms'],'first_request_cached_tokens':requests[0].get('cached_tokens'),'total_cached_tokens':sum(row.get('cached_tokens') or 0 for row in requests),'log_bytes':sum(p.stat().st_size for p in rep.glob('kv_*') if p.is_file() and p.suffix in {'.json','.jsonl'})}
            if requests[0].get('cached_tokens')!=0:raise RuntimeError('First request was not cold under its isolated cache salt')
            print(json.dumps({'pair':index,'collector':label,**pair[label]}),flush=True)
        pair['cache_observations_match']=pair['on']['total_cached_tokens']==pair['off']['total_cached_tokens']
        pair['relative_runtime']=pair['on']['wall_s']/pair['off']['wall_s']
        pairs.append(pair)
    result={'pairs':pairs,'median_paired_relative_runtime':statistics.median(pair['relative_runtime'] for pair in pairs),'scope':'Client continuous replay collection only; physical server hooks are not installed. Three pairs do not establish a precise overhead estimate.','cache_rule':'Distinct salt for every repetition; same salt for requests within each repetition; first request cached tokens must be zero.'}
    (sweep/'overhead_findings.json').write_text(json.dumps(result,indent=2)+'\n')
    write_report(sweep,results_dir=ROOT/'results')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
