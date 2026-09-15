"""Copy complete JSONL records from a running GPU node into an immutable snapshot."""
import argparse
import hashlib
import io
import json
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

REMOTE = '''import hashlib,json,time
from pathlib import Path
source=Path(SOURCE)
target=Path(TARGET)
target.mkdir(parents=True,exist_ok=False)
manifest={'source':str(source),'started_at_epoch_s':time.time(),'files':{}}
for path in sorted(source.glob('*.jsonl')):
    with path.open('rb') as stream:
        limit=path.stat().st_size
        data=stream.read(limit)
    boundary=data.rfind(b'\\n')+1
    data=data[:boundary]
    (target/path.name).write_bytes(data)
    manifest['files'][path.name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'last_sequence':json.loads(data.splitlines()[-1])['sequence'] if data else None}
manifest['ended_at_epoch_s']=time.time()
(target/'snapshot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\\n')
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ['ssh-host','control-path','job-id','trace-dir','remote-output']:
        parser.add_argument('--'+flag,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise RuntimeError('Preserve existing snapshot')
    ssh=['ssh','-S',args.control_path,'-o','BatchMode=yes',args.ssh_host]
    script=REMOTE.replace('Path(SOURCE)',f'Path({args.trace_dir!r})').replace('Path(TARGET)',f'Path({args.remote_output!r})')
    command=shlex.join(['srun','--jobid='+args.job_id,'--overlap','-N1','-n1','python3','-'])
    subprocess.run(ssh+[command],input=script.encode(),check=True)
    command=shlex.join(['tar','-C',args.remote_output,'-czf','-','.'])
    payload=subprocess.run(ssh+[command],stdout=subprocess.PIPE,check=True).stdout
    args.output.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
        archive.extractall(args.output,filter='data')
    manifest=json.loads((args.output/'snapshot_manifest.json').read_text())
    for name,record in manifest['files'].items():
        if hashlib.sha256((args.output/name).read_bytes()).hexdigest()!=record['sha256']:
            raise RuntimeError(f'Snapshot hash mismatch: {name}')
    print(json.dumps({'output':str(args.output),'files':len(manifest['files']),'bytes':sum(row['bytes'] for row in manifest['files'].values())}),flush=True)


if __name__=='__main__':main()
