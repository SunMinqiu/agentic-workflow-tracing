#!/usr/bin/env python3
"""Prepare and launch the selected SWE tasks through an existing CCR service."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import signal
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / 'workloads/swebench'


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def load_task(alias):
    return next(t for t in json.loads((TASKS / 'manifest.json').read_text()) if t['alias'] == alias)


def prepare(task, work_root):
    repo = work_root / task['alias']
    if repo.exists():
        if git(repo, 'rev-parse', 'HEAD') != task['base_commit']:
            raise ValueError(f'{repo}: unexpected commit; use a fresh --work-root')
        if git(repo, 'status', '--porcelain'):
            raise ValueError(f'{repo}: contains changes; use a fresh --work-root')
        return repo
    repo.mkdir(parents=True)
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'remote', 'add', 'origin',
                    'https://github.com/' + task['repo'] + '.git'], check=True)
    subprocess.run(['git', '-C', str(repo), 'fetch', '--depth=1', 'origin',
                    task['base_commit']], check=True)
    subprocess.run(['git', '-C', str(repo), 'checkout', '-q', '--detach', 'FETCH_HEAD'], check=True)
    return repo


def endpoint_config(env):
    url = env['VLLM_URL'].rstrip('/')
    if url.endswith('/chat/completions'):
        endpoint = url
    else:
        endpoint = (url if url.endswith('/v1') else url + '/v1') + '/chat/completions'
    model = env['VLLM_SERVED_MODEL']
    route = 'vllm,' + model
    return {
        'HOST': '127.0.0.1', 'PORT': int(env.get('PROV_CCR_PORT', '3456')),
        'APIKEY': env.get('PROV_CCR_TOKEN', 'ccr'),
        'transformers': [{'path': str(ROOT / 'scripts/ccr_vllm_stream.js')}],
        'Providers': [{'name': 'vllm', 'api_base_url': endpoint,
                       'api_key': env.get('VLLM_API_KEY', 'local-vllm'), 'models': [model],
                       'transformer': {'use': ['vllm-stream']}}],
        'Router': {key: route for key in ('default', 'background', 'think', 'longContext')},
    }


def agent_env(env):
    result = dict(env)
    result.pop('ANTHROPIC_API_KEY', None)
    result.pop('CLAUDECODE', None)
    result['CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC'] = '1'
    for key in ('CLAUDE_CODE_USE_BEDROCK', 'CLAUDE_CODE_USE_VERTEX', 'CLAUDE_CODE_USE_FOUNDRY'):
        result.pop(key, None)
    result['ANTHROPIC_BASE_URL'] = env.get('PROV_CCR_URL', 'http://127.0.0.1:3456')
    result['ANTHROPIC_AUTH_TOKEN'] = env.get('PROV_CCR_TOKEN', 'ccr')
    route = 'vllm,' + env['VLLM_SERVED_MODEL']
    for key in ('ANTHROPIC_MODEL', 'ANTHROPIC_DEFAULT_SONNET_MODEL',
                'ANTHROPIC_DEFAULT_OPUS_MODEL', 'ANTHROPIC_DEFAULT_HAIKU_MODEL',
                'ANTHROPIC_SMALL_FAST_MODEL'):
        result[key] = route
    return result


def run(task, args):
    repo = args.work_root / task['alias']
    if git(repo, 'rev-parse', 'HEAD') != task['base_commit'] or git(repo, 'status', '--porcelain'):
        raise ValueError('Run requires a clean checkout at the recorded base commit')
    env = agent_env(os.environ)
    command = ['claude', '-p', '--name', task['instance_id'], '--model', env['ANTHROPIC_MODEL'],
               '--output-format', 'stream-json', '--verbose', '--include-partial-messages',
               '--disallowedTools', 'Agent,Task,Workflow,SendMessage,CronCreate,ScheduleWakeup',
               '--permission-mode', args.permission_mode]
    if not args.execute:
        print(json.dumps({'cwd': str(repo), 'command': command,
                          'router': env['ANTHROPIC_BASE_URL'],
                          'prompt_file': str(TASKS / task['alias'] / 'prompt.txt'),
                          'execute': False}, indent=2))
        return
    if not shutil.which('claude'):
        raise ValueError('Install Claude Code on the execution node first')
    out = args.output / (task['alias'] + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(parents=True, exist_ok=False)
    prompt = (TASKS / task['alias'] / 'prompt.txt').read_text()
    record = {'task': task['instance_id'], 'base_commit': task['base_commit'],
              'model': env['ANTHROPIC_MODEL'], 'cwd': str(repo),
              'started_at': datetime.now(timezone.utc).isoformat()}
    (out / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    with (out / 'agent_events.jsonl').open('w') as stdout, (out / 'agent_events_timed.jsonl').open('w') as timed, (out / 'stderr.log').open('w') as stderr:
        proc = subprocess.Popen(command, text=True, cwd=repo, env=env,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr)
        proc.stdin.write(prompt)
        proc.stdin.close()
        for line in proc.stdout:
            received_ns = time.time_ns()
            stdout.write(line)
            stdout.flush()
            timed.write(json.dumps({'received_ns': received_ns, 'event': json.loads(line)}) + '\n')
            timed.flush()
        result = subprocess.CompletedProcess(command, proc.wait())
    record.update(exit_code=result.returncode, ended_at=datetime.now(timezone.utc).isoformat())
    for line in (out / 'agent_events.jsonl').read_text().splitlines():
        event = json.loads(line)
        if event.get('type') == 'result':
            record['agent_is_error'] = event.get('is_error', False)
            if record['agent_is_error']:
                record['exit_code'] = result.returncode or 1
    (out / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
    (out / 'patch.diff').write_text(git(repo, 'diff', '--binary', 'HEAD') + '\n')
    (out / 'git_status.txt').write_text(git(repo, 'status', '--porcelain') + '\n')
    (out / 'untracked_files.json').write_text(json.dumps(git(repo, 'ls-files', '--others', '--exclude-standard').splitlines(), indent=2)+'\n')
    print(out)
    raise SystemExit(record['exit_code'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'status', 'ccr-config', 'run'])
    parser.add_argument('--task', choices=['django', 'sphinx', 'nodebb', 'all'], default='all')
    parser.add_argument('--work-root', type=Path, default=ROOT / 'replay/swebench_work')
    parser.add_argument('--output', type=Path, default=ROOT / 'results/swebench')
    parser.add_argument('--config-output', type=Path)
    parser.add_argument('--execute', action='store_true', help='Actually launch Claude Code; otherwise print the invocation')
    parser.add_argument('--permission-mode', choices=['default', 'acceptEdits', 'bypassPermissions'], default='default')
    args = parser.parse_args()
    args.work_root = args.work_root.resolve()
    args.output = args.output.resolve()
    if args.action == 'ccr-config':
        if not args.config_output:
            parser.error('--config-output is required; existing files are never overwritten')
        config = endpoint_config(os.environ)
        fd = os.open(args.config_output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(config, stream, indent=2)
            stream.write('\n')
        print(args.config_output)
        return
    if args.action == 'run' and args.task == 'all':
        parser.error('Choose one --task; tasks run sequentially')
    for alias in (['django', 'sphinx', 'nodebb'] if args.task == 'all' else [args.task]):
        task = load_task(alias)
        if args.action == 'prepare':
            print(prepare(task, args.work_root))
        elif args.action == 'run':
            run(task, args)
        else:
            repo = args.work_root / alias
            print(json.dumps({'task': alias, 'exists': repo.exists(),
                              'commit': git(repo, 'rev-parse', 'HEAD') if repo.exists() else None,
                              'expected_commit': task['base_commit'],
                              'clean': not git(repo, 'status', '--porcelain') if repo.exists() else None}))


if __name__ == '__main__':
    if os.environ.get('SWE_START_PAUSED') == '1':
        os.kill(os.getpid(), signal.SIGSTOP)
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
