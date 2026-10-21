#!/usr/bin/env python3
"""Summarize proxy observations and vLLM metric deltas for one completed task."""
import json
from pathlib import Path
import re
import statistics
import sys


def metrics(path):
    result = {}
    for line in path.read_text().splitlines():
        if line.startswith('#'):
            continue
        match = re.match(r'(vllm:[^{ ]+)(?:\{[^}]*\})?\s+(\S+)', line)
        if match and not match[1].endswith('_bucket'):
            result[match[1]] = result.get(match[1], 0) + float(match[2])
    return result


def summarize(root):
    request_log = root/'http/requests.jsonl'
    rows = [json.loads(line) for line in request_log.read_text().splitlines()] if request_log.exists() else []
    ok = [r for r in rows if r.get('status') == 200 and r.get('first_token_ms') is not None]
    before, after = metrics(root/'metrics-before.txt'), metrics(root/'metrics-after.txt')
    delta = {k: v-before.get(k, 0) for k, v in after.items() if k.endswith(('_sum', '_count', '_total'))}
    prefill = delta.get('vllm:request_prefill_time_seconds_sum', 0)
    inference = delta.get('vllm:request_inference_time_seconds_sum', 0)
    for row in ok:
        row['ttft_fraction'] = row['first_token_ms']/row['duration_ms']
        output = row.get('usage', {}).get('completion_tokens', 0)
        row['tpot_ms'] = (row['last_token_ms']-row['first_token_ms'])/(output-1) if output>1 else None
    for timed_path in (root/'agent').glob('*/agent_events_timed.jsonl'):
        from datetime import datetime, timezone
        starts = {}
        tools = []
        for line in timed_path.read_text().splitlines():
            entry = json.loads(line)
            event = entry['event']
            content = event.get('message', {}).get('content', [])
            if not isinstance(content, list):
                continue
            for block in content:
                if block.get('type') == 'tool_use':
                    starts[block['id']] = (entry['received_ns'], block)
                elif block.get('type') == 'tool_result' and block.get('tool_use_id') in starts:
                    start, call = starts.pop(block['tool_use_id'])
                    end = entry['received_ns']
                    fmt = lambda ns: datetime.fromtimestamp(ns/1e9, timezone.utc).strftime('%H:%M:%S.%f')
                    tools.append(f"[{fmt(start)} -> {fmt(end)}] ({(end-start)/1e6:.3f}ms) {call['name']} (id={call['id']}) input={json.dumps(call.get('input', {}))}")
        (root/'tool_calls.log').write_text('\n'.join(tools)+'\n')
    (root/'per_request.json').write_text(json.dumps(ok, indent=2)+'\n')
    wall = None
    for path in (root/'agent').glob('*/run.json'):
        from datetime import datetime
        run = json.loads(path.read_text())
        wall = (datetime.fromisoformat(run['ended_at'])-datetime.fromisoformat(run['started_at'])).total_seconds()
    result = {'requests':len(rows), 'successful_streams':len(ok), 'agent_wall_seconds':wall,
              'client_inference_seconds':sum(r['duration_ms'] for r in ok)/1000,
              'client_ttft_seconds':sum(r['first_token_ms'] for r in ok)/1000,
              'client_ttft_fraction':sum(r['first_token_ms'] for r in ok)/sum(r['duration_ms'] for r in ok) if ok else None,
              'client_ttft_median_ms': statistics.median(r['first_token_ms'] for r in ok) if ok else None,
              'server_requests_match': delta.get('vllm:request_success_total') == len(ok),
              'client_tpot_median_ms': statistics.median(r['tpot_ms'] for r in ok if r['tpot_ms'] is not None) if any(r['tpot_ms'] is not None for r in ok) else None,
              'server_prefill_seconds':prefill, 'server_inference_seconds':inference,
              'server_prefill_fraction':prefill/inference if inference else None,
              'server_metric_deltas':delta}
    (root/'timing_summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2))
