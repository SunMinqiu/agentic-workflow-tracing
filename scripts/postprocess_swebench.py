#!/usr/bin/env python3
"""Adapt saved SWE captures to the shared agentic report pipeline."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def normalize(cell):
    run = json.loads(next((cell/'agent').glob('*/run.json')).read_text())
    rows = [json.loads(l) for l in (cell/'http/requests.jsonl').read_text().splitlines()]
    events, messages = [], []
    for r in rows:
        rid = r['request_id']
        request = json.loads((cell/'http'/f'{rid}.request.json').read_text())
        start = r['start_unix']*1000
        u = r.get('usage', {})
        usage = {'input':u.get('prompt_tokens',0),'output':u.get('completion_tokens',0),
                 'cacheRead':u.get('prompt_tokens_details',{}).get('cached_tokens',0),
                 'cacheReadAvailable':'cached_tokens' in u.get('prompt_tokens_details',{}),
                 'cacheReadSource':'vllm_stream_usage','totalTokens':u.get('total_tokens',0)}
        for kind, offset in [('message_start',0),('message_first_token',r.get('first_token_ms')),
                             ('message_last_token',r.get('last_token_ms')),('message_end',r['duration_ms'])]:
            if offset is None: continue
            events.append({'type':kind,'run_id':rid,'agent_role':'ClaudeCode','genomas_role':'ClaudeCode',
                           'phase':'solve','timestamp':start+offset,
                           'message':{'role':'assistant','timestamp':start+offset,'usage':usage if kind=='message_end' else {}}})
        params={k:v for k,v in request.items() if k not in ('messages','model')}
        messages.append({'run_id':rid,'timestamp':start,'agent_role':'ClaudeCode','genomas_role':'ClaudeCode',
                         'phase':'solve','provider':'vllm','model':request['model'],'messages':request['messages'],
                         'request_capture':{'messages':request['messages'],'parameters':params},'request_params':params})
    parsed=json.loads((cell/'parsed.json').read_text())
    for call in parsed['tool_calls']:
        params=call.get('input_params',{})
        if isinstance(params,dict) and set(params)=={'raw'}:
            try: params=json.loads(params['raw']);call['input_params']=params
            except ValueError: pass
        for kind,key in [('tool_execution_start','start_time'),('tool_execution_end','end_time')]:
            ts=datetime.fromisoformat(call[key]).replace(tzinfo=timezone.utc).timestamp()*1000
            events.append({'type':kind,'timestamp':ts,'run_id':call['tool_id'],'toolCallId':call['tool_id'],
                           'toolName':call['tool_name'],'args':params,'genomas_role':'ClaudeCode','agent_role':'ClaudeCode','phase':'solve'})
    (cell/'parsed.json').write_text(json.dumps(parsed)+'\n')
    events.sort(key=lambda e:e['timestamp'])
    for name, records in [('pi_events.jsonl',events),('messages.jsonl',messages)]:
        (cell/name).write_text(''.join(json.dumps(r)+'\n' for r in records))
    manifest={'workload':'SWEbench','task_id':run['task'],'model':run['model'],'vendor':'vLLM',
              'replay_mode':'live','agent_count':1,'work_dir':run['cwd'],'output_dir':str(cell),
              'instrumentation':'ebpf+http_proxy','capture_source':'saved SWE live capture',
              'timing_source':'HTTP proxy wall clock; tool event receive times',
              'started_at':run['started_at'],'ended_at':run['ended_at'],'exit_code':run['exit_code']}
    (cell/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def validate(cell):
    required = """manifest.json ebpf_events.log parsed.json phase1_metrics.json
    parallelism_summary.json trace_quality.json lineage/artifacts.csv
    lineage/execution_unit_io.csv pi_events.jsonl tool_calls.log messages.jsonl
    kvcache_demand.json kvcache_logical.json kvcache_report.md kvcache_report.html
    call_dag.html call_dag.dot visualizations/index.html
    visualizations/file_access_volume.png visualizations/rw_asymmetry.png
    visualizations/request_size_rw_cdf.png visualizations/byte_normalized_summary.png
    lineage/fig2_fanout.png lineage/fig4_lifecycle.png""".split()
    missing = [name for name in required if not (cell/name).is_file() or not (cell/name).stat().st_size]
    bcc = (cell/'bcc.err').read_text().splitlines()
    zero_loss = bool(bcc and 'lost_events=0' in bcc[-1])
    rows = [json.loads(line) for line in (cell/'http/requests.jsonl').read_text().splitlines()]
    demand = json.loads((cell/'kvcache_demand.json').read_text())
    checks = {
        'required_files_present': not missing,
        'zero_lost_events': zero_loss,
        'request_count_matches': demand['n_token_calls'] == len(rows),
        'input_tokens_match': demand['total_input_tokens'] == sum(r['usage']['prompt_tokens'] for r in rows),
        'cached_tokens_match': demand['total_cacheread_tokens'] == sum(r['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0) for r in rows),
    }
    result = {'passed': all(checks.values()), 'checks': checks, 'missing': missing}
    (cell/'report_validation.json').write_text(json.dumps(result, indent=2)+'\n')
    if not result['passed']:
        raise RuntimeError(f'Report validation failed: {result}')
    print(f'{cell.name}: report validation passed', flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cell',type=Path);args=p.parse_args()
    cell=args.cell.resolve();normalize(cell)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src')+os.pathsep+os.environ.get('PYTHONPATH',''))
    steps=[('summary','agent_io_tracing.analysis.summary',[str(cell)]),
           ('lineage','agent_io_tracing.lineage.analyzer',[str(cell)]),
           ('parallelism','agent_io_tracing.analysis.parallelism',[str(cell)]),
           ('phase1_metrics','agent_io_tracing.analysis.phase1_metrics',[str(cell)]),
           ('execution_units','agent_io_tracing.analysis.execution_units',[str(cell)]),
           ('trace_quality','agent_io_tracing.analysis.trace_quality',[str(cell)]),
           ('per_run_io_char','agent_io_tracing.analysis.per_run_io_char',['--results',str(cell),'--runs','.']),
           ('visualize','agent_io_tracing.viz.trace',[str(cell)]),
           ('kvcache','agent_io_tracing.analysis.kvcache.report',['--results',str(cell.parent),'--runs',cell.name,'--dump-prefixes'])]
    for name,module,options in steps:
        print(cell.name,name,flush=True)
        with (cell/(name+'.log')).open('w') as log:
            subprocess.run([sys.executable,'-m',module,*options],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    validate(cell)


if __name__=='__main__':main()
