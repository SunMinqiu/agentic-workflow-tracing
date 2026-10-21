"""Check recorded lengths and physical request accounting for a SWE replay matrix."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def validate(root):
    rows = []
    failures = []
    for task in ('django', 'sphinx', 'nodebb'):
        bundle = read(root / 'bundles' / f'{task}.json')
        if not (Path(bundle['source_cell']) / 'kvcache_report.html').is_file():
            failures.append(f'{task}: missing integrated KV report')
        arms = sorted((root / task).glob('gpu*/rep0'))
        if len(arms) != 3:
            failures.append(f'{task}: expected three configurations, found {len(arms)}')
        for rep in arms:
            name = str(rep.relative_to(root))
            required = ['summary.json', 'requests.jsonl', 'physical_trace_summary.json',
                        'causal_trace_summary.json']
            missing = [p for p in required if not (rep / p).is_file()]
            if missing:
                failures.append(f'{name}: missing {missing}')
                continue
            requests = [json.loads(line) for line in (rep / 'requests.jsonl').read_text().splitlines()]
            summary = read(rep / 'summary.json')
            physical = read(rep / 'physical_trace_summary.json')
            causal = read(rep / 'causal_trace_summary.json')
            expected = bundle['requests']
            checks = {
                'request_count': len(requests) == len(expected),
                'per_request_metrics_complete': summary.get('server_timing', {}).get('reported_requests') == len(expected),
                'input_lengths': [r['prompt_tokens'] for r in requests] == [r['prompt_tokens'] for r in expected],
                'output_lengths': [r['output_tokens'] for r in requests] == [r['recorded_usage']['output'] for r in expected],
                'collection_errors_absent': summary.get('kv_events', {}).get('collection_errors') == [],
                'sequence_gaps_absent': not physical['sequence_gaps'],
                'all_requests_accounted': causal['token_accounting_matching_requests'] == len(expected),
                'no_pending_transfers': physical['pending_accepted_transfers'] == 0,
            }
            for check, passed in checks.items():
                if not passed:
                    failures.append(f'{name}: {check}')
            prompt = sum(r['prompt_tokens'] for r in requests)
            metrics = summary['prometheus_window']['metrics']
            prefill_s = metrics['vllm:request_prefill_time_seconds_sum']['delta']
            inference_s = metrics['vllm:request_inference_time_seconds_sum']['delta']
            metric_count = metrics['vllm:request_prefill_time_seconds_count']['delta']
            if metric_count != len(expected):
                failures.append(f'{name}: prefill metric count mismatch')
            rows.append(dict(task=task, arm=rep.parent.name, checks=checks,
                             requests=len(requests), wall_s=summary['wall_s'],
                             client_ttft_fraction=sum(r['ttft_ms'] for r in requests) / sum(r['total_ms'] for r in requests),
                             server_prefill_s=prefill_s,
                             server_inference_s=inference_s,
                             server_prefill_fraction=prefill_s / inference_s,
                             median_tpot_ms=summary['median_tpot_ms'],
                             median_server_ttft_ms=summary['server_timing']['median_ttft_ms'],
                             cache_hit_fraction=sum(r.get('cached_tokens') or 0 for r in requests) / prompt,
                             physical_evictions=physical['physical_evictions'],
                             transfer_bytes=physical['successful_transfer_bytes_by_direction'],
                             confirmed_load_jobs=causal['confirmed_load_jobs'],
                             completed_prefill_tokens=causal['completed_prefill_tokens']))
    result = dict(passed=not failures, failures=failures, rows=rows)
    (root / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    result = validate(parser.parse_args().root)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
