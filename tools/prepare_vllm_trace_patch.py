"""Generate or apply a patch only to the audited vLLM source fingerprints."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PIN=ROOT/'config/vllm_trace_v026_source.json'
TARGETS={
 'vllm/v1/core/block_pool.py':'install_block_pool',
 'vllm/v1/core/kv_cache_manager.py':'install_manager',
 'vllm/distributed/kv_transfer/kv_connector/v1/offloading/worker.py':'install_offload_worker',
 'vllm/distributed/kv_events.py':'install_publisher',
 'vllm/v1/core/sched/scheduler.py':'install_scheduler',
 'vllm/v1/kv_offload/cpu/manager.py':'install_cpu_manager',
 'vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py':'install_offload_scheduler',
}


def prepare(source_root, output_dir, apply=False, pin=None):
    pins=json.loads((pin or PIN).read_text())
    changes=[]
    for name,hook in TARGETS.items():
        path=source_root/name
        original=path.read_text()
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=pins['files'][name]:
            raise ValueError(f'Source fingerprint mismatch: {name}. Audit the deployed source before patching.')
        module = 'vllm_causal_hooks' if hook in ('install_scheduler', 'install_cpu_manager', 'install_offload_scheduler') else 'vllm_trace_hooks'
        addition=f'\n\nfrom agent_io_tracing.serving.{module} import {hook} as _install_agent_trace\n_install_agent_trace(globals())\n'
        updated=original+addition
        compile(updated,name,'exec')
        changes.append((path,original,updated))
    output_dir.mkdir(parents=True,exist_ok=True)
    patch=''.join(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+str(path.relative_to(source_root)),tofile='b/'+str(path.relative_to(source_root)))) for path,old,new in changes)
    (output_dir/'vllm-kv-trace.patch').write_text(patch)
    verification = {'source_root':str(source_root.resolve()),'source_pins':pins,'applied':False,'patched_sha256':{str(path.relative_to(source_root)):hashlib.sha256(new.encode()).hexdigest() for path,old,new in changes}}
    if apply:
        for path,old,new in changes:
            backup=output_dir/'original'/path.relative_to(source_root)
            backup.parent.mkdir(parents=True,exist_ok=True)
            backup.write_text(old)
        for path, old, new in changes:
            path.write_text(new)
        verification["applied"] = True
    (output_dir/'source_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    return output_dir/'vllm-kv-trace.patch'


def restore(source_root, output_dir):
    verification=json.loads((output_dir/'source_verification.json').read_text())
    if str(source_root.resolve()) != verification['source_root'] or not verification['applied']:
        raise ValueError('Restore requires the applied source root and its verification record')
    originals=[]
    for name in TARGETS:
        path=source_root/name
        backup=output_dir/'original'/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=verification['patched_sha256'][name]:
            raise ValueError(f'Patched source changed: {name}; preserve the new edits before restoring')
        if hashlib.sha256(backup.read_bytes()).hexdigest()!=verification['source_pins']['files'][name]:
            raise ValueError(f'Original backup fingerprint mismatch: {name}')
        originals.append((path,backup.read_bytes()))
    for path,payload in originals:
        path.write_bytes(payload)
    verification['applied']=False
    verification['restored']=True
    (output_dir/'source_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    return output_dir/'source_verification.json'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--pin',type=Path,default=PIN)
    p.add_argument('--output-dir',type=Path,required=True)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--apply',action='store_true')
    mode.add_argument('--restore',action='store_true')
    args=p.parse_args()
    print(restore(args.source_root,args.output_dir) if args.restore else prepare(args.source_root,args.output_dir,args.apply,args.pin))
