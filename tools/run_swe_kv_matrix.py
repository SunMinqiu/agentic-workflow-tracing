"""Replay captured SWE tasks with physical KV tracing on a Slurm allocation."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))


def command(argv):
    print(shlex.join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), cwd=ROOT, env=ENV, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--host', required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--node', required=True)
    p.add_argument('--remote-root', required=True)
    p.add_argument('--url', default='http://127.0.0.1:18000')
    p.add_argument('--replay-endpoint', default='tcp://127.0.0.1:15558')
    p.add_argument('--initial-launch-pid', type=int, required=True)
    p.add_argument('--initial-session', required=True)
    p.add_argument('--small-blocks', type=int, default=96)
    args = p.parse_args()
    sys.path.insert(0, str(ROOT / 'src'))
    from agent_io_tracing.serving.vllm_endpoint import VLLMEndpoint
    from agent_io_tracing.replay.bundle import build_bundle
    endpoint = VLLMEndpoint(args.url, timeout_s=300)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'bundles').mkdir(exist_ok=True)
    selected = json.loads((ROOT / 'results/swebench-measured/selected_results.json').read_text())
    ssh = ['ssh', '-o', 'BatchMode=yes', args.host]
    launch_pid = args.initial_launch_pid
    sessions = []
    for index, (blocks, cpu) in enumerate([(200, 0), (args.small_blocks, 0), (args.small_blocks, 32)]):
        label = f'gpu{blocks}_cpu{cpu}'
        session = args.initial_session if index == 0 else f'{args.output.name}-{label}'
        if index:
            command(ssh + [f'kill -TERM {launch_pid}'])
            for _ in range(60):
                try:
                    urllib.request.urlopen(args.url + '/health', timeout=2).close()
                except Exception:
                    break
                time.sleep(2)
            else:
                raise RuntimeError('Previous server did not stop')
            launch = ['srun', '--jobid=' + args.job, '--overlap', '-N1', '-n1', '-c16', '-w', args.node,
                      'bash', args.remote_root + '/launch_swe_kv_riken.sh', session, str(blocks), str(cpu)]
            log = args.remote_root + '/output/' + session + '.log'
            shell = 'nohup ' + shlex.join(launch) + ' > ' + shlex.quote(log) + ' 2>&1 < /dev/null & echo $!'
            launch_pid = int(subprocess.check_output(ssh + [shell], text=True).strip())
        sessions.append(dict(session=session, blocks=blocks, cpu_gib=cpu, launch_pid=launch_pid))
        (args.output / 'sessions.json').write_text(json.dumps(sessions, indent=2) + '\n')
        for attempt in range(90):
            try:
                urllib.request.urlopen(args.url + '/health', timeout=3).close()
                break
            except Exception:
                if attempt % 3 == 0:
                    print(f'{session}: waiting for server', flush=True)
                time.sleep(10)
        else:
            raise RuntimeError('Server startup timeout')
        info = endpoint.server_info()['vllm_config']
        model = endpoint.models()[0]
        for row in selected:
            task = row['task']
            bundle_path = args.output / 'bundles' / f'{task}.json'
            if not bundle_path.exists():
                cell = ROOT / 'results/swebench-measured' / row['run_directory']
                bundle = build_bundle(cell, endpoint, model)
                for request in bundle['requests']:
                    if request['prompt_tokens'] != request['recorded_usage']['input']:
                        raise RuntimeError(f'{task}: tokenized prompt differs from captured usage')
                bundle_path.write_text(json.dumps(bundle) + '\n')
            command([sys.executable, 'tools/run_kv_experiment_cell.py', '--url', args.url,
                     '--replay-endpoint', args.replay_endpoint, '--bundle', bundle_path,
                     '--sweep-dir', args.output / task, '--label', label, '--workload', task,
                     '--gpu-blocks', blocks, '--cpu-offload-gib', cpu,
                     '--block-size', info['cache_config']['block_size'],
                     '--mode', 'serial', '--output-tokens', '0', '--repetitions', '1'])
        snapshot = args.output / 'snapshots' / session
        command([sys.executable, 'tools/pull_kv_physical_snapshot.py', '--ssh-host', args.host,
                 '--control-path', 'none', '--job-id', args.job,
                 '--trace-dir', args.remote_root + '/output/' + session + '/physical',
                 '--remote-output', args.remote_root + '/output/' + session + '-snapshot',
                 '--output', snapshot])
        logs = sorted(snapshot.glob('*.jsonl'))
        if not logs:
            raise RuntimeError('Missing physical trace files')
        for row in selected:
            sweep = args.output / row['task']
            command([sys.executable, '-m', 'agent_io_tracing.serving.kv_physical',
                     sweep / label / 'rep0', *logs])
            command([sys.executable, '-m', 'agent_io_tracing.replay.sweep', 'report',
                     '--sweep-dir', sweep, '--results', ROOT / 'results'])
        print(f'{label}: replay and physical import complete', flush=True)


if __name__ == '__main__':
    main()
