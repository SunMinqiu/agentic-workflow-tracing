"""Collect an offload-enabled comparison in the capacity matrix allocation."""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['ssh-host','control-path','job-id','remote-root','url','replay-endpoint']:
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repetitions',type=int,choices=[1,2,3],default=3)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    ssh=['ssh','-S',args.control_path,'-o','BatchMode=yes',args.ssh_host]
    def remote(command,**kwargs):return subprocess.run(ssh+[shlex.join(command)],check=True,**kwargs)
    receipt=args.output/'same-node-offload.json'
    if receipt.exists():
        raise RuntimeError('This offload comparison already has a completion receipt')
    cell_roots=[ROOT/f'replay/kv_same_node_physical_{w}_20260911' for w in ['c2-12','c8']]
    for root in cell_roots:
        if list(root.glob('*/rep*')):
            raise RuntimeError(f'Preserve and recover existing results before restarting: {root}')
    left=remote(['squeue','-h','-j',args.job_id,'-o','%L'],capture_output=True,text=True).stdout.strip()
    days,clock=left.split('-',1) if '-' in left else ('0',left)
    seconds=int(days)*86400+sum(int(v)*60**i for i,v in enumerate(reversed(clock.split(':'))))
    required_seconds=180+660*args.repetitions
    if seconds<required_seconds:
        print(json.dumps({'status':'allocation_checkpoint','seconds_left':seconds,'required_seconds':required_seconds}),flush=True)
        return
    node=remote(['squeue','-h','-j',args.job_id,'-o','%N'],capture_output=True,text=True).stdout.strip()
    for index in range(args.repetitions):
        session=f'same-node-offload-r{index}-{args.job_id}'
        log=f'{args.remote_root}/logs/kv-{session}.log'
        command=shlex.join(['env',f'KV_TRACE_JOB_ID={args.job_id}',f'KV_TRACE_SESSION={session}','KV_GPU_BLOCKS=274','KV_OFFLOADING_SIZE_GIB=32','KV_TRACE_ENABLED=1','bash',f'{args.remote_root}/kv-trace-deploy-20260911/launch.sh'])+' > '+shlex.quote(log)+' 2>&1'
        process=subprocess.Popen(ssh+[command],stdin=subprocess.DEVNULL)
        ready=False
        for _ in range(180):
            if process.poll() is not None:raise RuntimeError(f'Server exited: {log}')
            try:
                with urllib.request.urlopen(args.url+'/health',timeout=2) as response:ready=response.status==200
            except OSError:pass
            if ready:break
            time.sleep(2)
        if not ready:raise RuntimeError(f'Server startup timed out: {log}')
        for workload,mode,limit,source in [('C2-12','packed','32','replay/kv_physical_c2-12_20260911/bundles/C2-12.json'),('C8','serial','256','replay/kv_physical_c8_20260911/bundles/C8.json')]:
            subprocess.run([sys.executable,str(ROOT/'tools/run_kv_experiment_cell.py'),'--url',args.url,'--replay-endpoint',args.replay_endpoint,'--gpu-blocks','274','--cpu-offload-gib','32','--label','gpu274_cpu32','--repetitions',str(args.repetitions),'--only-rep',str(index),'--bundle',str(ROOT/source),'--sweep-dir',str(ROOT/f'replay/kv_same_node_physical_{workload.lower()}_20260911'),'--workload',workload,'--mode',mode,'--output-tokens',limit],check=True,cwd=ROOT)
        snapshot=args.output/(session+'-snapshot')
        subprocess.run([sys.executable,str(ROOT/'tools/pull_kv_physical_snapshot.py'),'--ssh-host',args.ssh_host,'--control-path',args.control_path,'--job-id',args.job_id,'--trace-dir','/dev/shm/kv-physical-'+session,'--remote-output',args.remote_root+'/trace-snapshots/'+session,'--output',str(snapshot)],check=True)
        remote(['srun','--jobid='+args.job_id,'--overlap','-N1','-n1','podman-hpc','stop','--time','30','kv-trace-'+session])
        process.wait(timeout=45)
        sys.path.insert(0,str(ROOT/'src'))
        from agent_io_tracing.serving.kv_report_data import prepare_report_data
        from agent_io_tracing.replay.sweep import write_report
        for root in cell_roots:
            for rep in sorted(root.glob(f'*/rep{index}')):
                target=rep/'physical_traces';target.mkdir(exist_ok=False)
                for source in snapshot.glob('*.jsonl'):os.link(source,target/source.name)
                prepare_report_data(rep)
            write_report(root,results_dir=ROOT/'results')
        (args.output/f'same-node-offload-rep{index}.json').write_text(json.dumps({'rep':index,'job_id':args.job_id,'node':node,'gpu_blocks':274,'cpu_offload_gib':32,'repetitions':args.repetitions,'session':session,'snapshot':str(snapshot.resolve()),'remote_log':log},indent=2)+'\n')
        print(json.dumps({'status':'same_node_offload_cell_complete','rep':index,'node':node}),flush=True)
    receipt.write_text(json.dumps({'job_id':args.job_id,'node':node,'gpu_blocks':274,'cpu_offload_gib':32,'repetitions':args.repetitions,'protocol':'Fresh server for each repetition; C2 followed by C8, matching the capacity matrix','cells':[json.loads((args.output/f'same-node-offload-rep{i}.json').read_text()) for i in range(args.repetitions)]},indent=2)+'\n')
    print(json.dumps({'status':'same_node_offload_complete','node':node}),flush=True)


if __name__=='__main__':main()
