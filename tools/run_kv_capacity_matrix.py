"""Run a randomized capacity matrix inside an existing GPU allocation."""
import argparse
import json
import random
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def cell_repetitions(capacity, index):
    return [ROOT/f'replay/kv_capacity_physical_{workload}_20260911'/f'gpu{capacity}_cpu0'/f'rep{index}' for workload in ['c2-12','c8']]


def validate_resume_cell(marker, capacity, index):
    repetitions=cell_repetitions(capacity,index)
    if marker.exists():
        receipt=json.loads(marker.read_text())
        if (receipt.get('gpu_blocks'),receipt.get('rep')) != (capacity,index):
            raise RuntimeError(f'Completion receipt does not match the scheduled cell: {marker}')
        for rep in repetitions:
            if not (rep/'summary.json').is_file() or not list((rep/'physical_traces').glob('*.jsonl')):
                raise RuntimeError(f'Completed cell is missing saved evidence: {rep}')
        return True
    for rep in repetitions:
        if rep.exists() and any(rep.iterdir()):
            raise RuntimeError(f'Incomplete cell contains evidence: {rep}. Recover its original trace snapshot or archive the attempt before rerunning.')
    return False


def complete_cell(args, capacity, index, session, process=None):
    cell={'gpu_blocks':capacity,'rep':index}
    ssh=['ssh','-S',args.control_path,'-o','BatchMode=yes',args.ssh_host]
    log=f'{args.remote_root}/logs/kv-{session}.log'
    marker=args.output/f'gpu{capacity}-rep{index}.json'
    node=subprocess.run(ssh+[shlex.join(['squeue','-h','-j',args.job_id,'-o','%N'])],check=True,capture_output=True,text=True).stdout.strip()
    snapshot=args.output/(session+'-snapshot')
    subprocess.run([sys.executable,str(ROOT/'tools/pull_kv_physical_snapshot.py'),'--ssh-host',args.ssh_host,'--control-path',args.control_path,'--job-id',args.job_id,'--trace-dir','/dev/shm/kv-physical-'+session,'--remote-output',args.remote_root+'/trace-snapshots/'+session,'--output',str(snapshot)],check=True)
    stop=shlex.join(['srun','--jobid='+args.job_id,'--overlap','-N1','-n1','podman-hpc','stop','--time','30','kv-trace-'+session])
    subprocess.run(ssh+[stop],check=True)
    if process is not None:process.wait(timeout=45)
    sys.path.insert(0,str(ROOT/'src'))
    from agent_io_tracing.serving.kv_report_data import prepare_report_data
    from agent_io_tracing.replay.sweep import write_report
    for workload in ['c2-12','c8']:
        rep=ROOT/f'replay/kv_capacity_physical_{workload}_20260911'/f'gpu{capacity}_cpu0'/f'rep{index}'
        target=rep/'physical_traces';target.mkdir(exist_ok=False)
        for source in snapshot.glob('*.jsonl'):os.link(source,target/source.name)
        prepare_report_data(rep)
        write_report(rep.parent.parent,results_dir=ROOT/'results')
    marker.write_text(json.dumps({**cell,'job_id':args.job_id,'session':session,'snapshot':str(snapshot.resolve()),'remote_log':log,'node':node},indent=2)+'\n')
    print(json.dumps({'status':'cell_complete',**cell}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['ssh-host','control-path','job-id','remote-root','url','replay-endpoint']:
        p.add_argument('--'+flag,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    ssh=['ssh','-S',args.control_path,'-o','BatchMode=yes',args.ssh_host]
    def remote(command,**kwargs):return subprocess.run(ssh+[command],check=True,**kwargs)
    schedule=[];rng=random.Random(20260911)
    for repeat in range(3):
        capacities=[192,274,548];rng.shuffle(capacities)
        schedule.extend({'gpu_blocks':capacity,'rep':repeat} for capacity in capacities)
    plan=args.output/'schedule.json'
    if plan.exists() and json.loads(plan.read_text())!=schedule:raise RuntimeError('Existing schedule differs')
    plan.write_text(json.dumps(schedule,indent=2)+'\n')
    for cell in schedule:
        capacity,index=cell['gpu_blocks'],cell['rep']
        marker=args.output/f'gpu{capacity}-rep{index}.json'
        if validate_resume_cell(marker,capacity,index):continue
        left=remote(shlex.join(['squeue','-h','-j',args.job_id,'-o','%L']),stdout=subprocess.PIPE,text=True).stdout.strip()
        if not left:raise RuntimeError('Allocation is no longer running')
        days,clock=(left.split('-',1) if '-' in left else ('0',left))
        units=[int(value) for value in clock.split(':')]
        seconds=int(days)*86400+sum(value*60**power for power,value in enumerate(reversed(units)))
        if seconds<900:
            print(json.dumps({'status':'allocation_checkpoint','seconds_left':seconds,'next_cell':cell}),flush=True)
            return
        session=f'capacity-gpu{capacity}-r{index}-{args.job_id}'
        log=f'{args.remote_root}/logs/kv-{session}.log'
        env=[f'KV_TRACE_JOB_ID={args.job_id}',f'KV_TRACE_SESSION={session}',f'KV_GPU_BLOCKS={capacity}','KV_OFFLOADING_SIZE_GIB=0','KV_TRACE_ENABLED=1']
        command=shlex.join(['env',*env,'bash',f'{args.remote_root}/kv-trace-deploy-20260911/launch.sh'])+' > '+shlex.quote(log)+' 2>&1'
        process=subprocess.Popen(ssh+[command],stdin=subprocess.DEVNULL)
        print(json.dumps({'status':'starting','session':session,**cell}),flush=True)
        ready=False
        for attempt in range(180):
            if process.poll() is not None:raise RuntimeError(f'Server exited; inspect {log}')
            try:
                with urllib.request.urlopen(args.url+'/health',timeout=2) as response:
                    ready=response.status==200
            except OSError:pass
            if ready:break
            time.sleep(2)
        if not ready:raise RuntimeError(f'Server startup timed out: {log}')
        for workload,mode,limit,source in [('C2-12','packed','32','replay/kv_c2_capacity_20260910/bundles/C2-12.json'),('C8','serial','256','replay/kv_c8_capacity_20260910/bundles/C8.json')]:
            sweep=ROOT/f'replay/kv_capacity_physical_{workload.lower()}_20260911'
            command=[sys.executable,str(ROOT/'tools/run_kv_experiment_cell.py'),'--url',args.url,'--replay-endpoint',args.replay_endpoint,'--gpu-blocks',str(capacity),'--cpu-offload-gib','0','--label',f'gpu{capacity}_cpu0','--repetitions','3','--only-rep',str(index),'--bundle',str(ROOT/source),'--sweep-dir',str(sweep),'--workload',workload,'--mode',mode,'--output-tokens',limit]
            subprocess.run(command,check=True,cwd=ROOT)
        complete_cell(args,capacity,index,session,process)
    print(json.dumps({'status':'matrix_complete','cells':len(schedule)}),flush=True)


if __name__=='__main__':main()
