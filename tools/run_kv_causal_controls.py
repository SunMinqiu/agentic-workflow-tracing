"""Run explicit GPU reuse, CPU reload, and pressure controls on an audited server."""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from agent_io_tracing.replay.runner import replay_to_dir
from agent_io_tracing.serving.vllm_endpoint import VLLMEndpoint


def run(args):
    endpoint=VLLMEndpoint(args.url,timeout_s=300)
    original=json.loads(args.bundle.read_text())
    first=copy.deepcopy(original['requests'][0])
    bundle=copy.deepcopy(original)
    cases=[('first',[first],True),('gpu_repeat',[first],False),('cpu_repeat',[first],True)]
    if args.pressure:
        variants=[t for t in dict.fromkeys(first['prompt_token_ids'][1:]) if t!=first['prompt_token_ids'][0]][:96]
        if len(variants)!=96:raise ValueError('Need 96 distinct first-token variants')
        requests=[copy.deepcopy(first)]
        for token in variants:
            item=copy.deepcopy(first);item['prompt_token_ids'][0]=token;requests.append(item)
        requests.append(copy.deepcopy(first))
        cases=[('pressure',requests,True)]
    for label,requests,reset in cases:
        bundle['requests']=copy.deepcopy(requests)
        for index,item in enumerate(bundle['requests']):
            item['request_id']=f'{label}-{index}';item['index']=index
            item['sampling_params']={**(item.get('sampling_params') or {}),'temperature':0,'seed':0}
        rep=args.output/label/'rep0'
        if rep.exists():raise RuntimeError(f'Preserve existing run: {rep}')
        snapshot=args.output/'bundles'/(label+'.json');snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_text(json.dumps(bundle,sort_keys=True))
        summary=replay_to_dir(bundle,endpoint,'serial',rep,reset_before=reset,bundle_path=snapshot,output_tokens=1 if args.pressure else 8,cache_salt_mode='shared',cache_salt=args.salt,kv_events_replay_endpoint=args.replay_endpoint)
        rows=[json.loads(line) for line in (rep/'requests.jsonl').read_text().splitlines()]
        print(json.dumps({'control':label,'wall_s':summary['wall_s'],'cached_tokens':[row.get('cached_tokens') for row in rows],'output_tokens':[row['output_tokens'] for row in rows]}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',required=True);parser.add_argument('--replay-endpoint',required=True)
    parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--salt',required=True);parser.add_argument('--pressure',action='store_true')
    run(parser.parse_args())
