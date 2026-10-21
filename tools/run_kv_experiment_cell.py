"""Run a reproducible bundle cell after verifying the live serving configuration."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from agent_io_tracing.replay.runner import replay_to_dir
from agent_io_tracing.replay.sweep import write_report
from agent_io_tracing.serving.vllm_endpoint import VLLMEndpoint


def run(args):
    endpoint=VLLMEndpoint(args.url,timeout_s=300)
    source=json.loads(args.bundle.read_text())
    bundle=copy.deepcopy(source)
    for item in bundle['requests']:
        item['sampling_params']={**(item.get('sampling_params') or {}),'temperature':0,'seed':0}
    snapshot=args.sweep_dir/'bundles'/f'{args.workload}.json'
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(bundle,sort_keys=True)
    if snapshot.exists() and snapshot.read_text()!=payload:
        raise RuntimeError('Existing bundle snapshot differs; choose a new experiment directory')
    snapshot.write_text(payload)
    manifest={'source_bundle':str(args.bundle.resolve()),'source_sha256':hashlib.sha256(args.bundle.read_bytes()).hexdigest(),'snapshot_sha256':hashlib.sha256(payload.encode()).hexdigest(),'workload':args.workload,'mode':args.mode,'output_tokens':args.output_tokens,'expected_gpu_blocks':args.gpu_blocks,'expected_cpu_offload_gib':args.cpu_offload_gib,'expected_block_size':args.block_size,'repetitions':args.repetitions}
    arm=args.sweep_dir/args.label
    arm.mkdir(parents=True,exist_ok=True)
    manifest_path=arm/'cell_manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise RuntimeError('Existing cell manifest differs')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    indices=[args.only_rep] if getattr(args,"only_rep",None) is not None else range(args.repetitions)
    if any(index<0 or index>=args.repetitions for index in indices):raise ValueError("Repetition index is outside the declared cell")
    for index in indices:
        info=endpoint.server_info()['vllm_config'];cfg=info['cache_config']
        actual_offload=float(cfg.get('kv_offloading_size') or 0)
        if int(cfg['num_gpu_blocks'])!=args.gpu_blocks or actual_offload!=args.cpu_offload_gib:
            raise RuntimeError(f'Live capacity mismatch: GPU={cfg["num_gpu_blocks"]}, offload={actual_offload}')
        if cfg['cache_dtype']!='auto' or int(cfg['block_size'])!=args.block_size:
            raise RuntimeError(f'Expected auto dtype and {args.block_size}-token blocks')
        if (info.get('observability_config') or {}).get('kv_cache_metrics_sample')!=1.0:
            raise RuntimeError('This matrix requires full histogram observation sampling')
        rep=arm/f'rep{index}'
        if (rep/'summary.json').exists():
            print(f'Already completed: {rep}',flush=True)
            continue
        if rep.exists() and any(rep.iterdir()):
            raise RuntimeError(f'Partial run exists at {rep}; preserve it and choose a fresh cell')
        summary=replay_to_dir(bundle,endpoint,args.mode,rep,reset_before=True,bundle_path=snapshot,output_tokens=args.output_tokens,cache_salt_mode='shared',kv_events_replay_endpoint=args.replay_endpoint)
        write_report(args.sweep_dir,results_dir=ROOT/'results')
        print(json.dumps({'rep':str(rep),'wall_s':summary['wall_s'],'requests':summary['requests'],'collection_errors':(summary.get('kv_events') or {}).get('collection_errors')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--sweep-dir',type=Path,required=True)
    p.add_argument('--label',required=True);p.add_argument('--workload',required=True)
    p.add_argument('--block-size',type=int,default=784);p.add_argument('--gpu-blocks',type=int,required=True);p.add_argument('--cpu-offload-gib',type=float,required=True)
    p.add_argument('--mode',choices=['serial','packed'],required=True);p.add_argument('--output-tokens',type=int,required=True,help='0 preserves each recorded output length')
    p.add_argument('--only-rep',type=int);p.add_argument('--repetitions',type=int,default=3);p.add_argument('--url',default='http://127.0.0.1:18000')
    p.add_argument('--replay-endpoint',default='tcp://127.0.0.1:15558')
    run(p.parse_args())
