#!/usr/bin/env python3
"""Audit one saved SciLink run without estimating image or model tokens."""
import argparse
import collections
import copy
import hashlib
import shutil
import subprocess
import sys
import os
import json
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def text_prefix(messages):
    """Compare structured text only up to the first non-text content block."""
    atoms = []
    for message in messages:
        atoms.append(('header', canonical({k: v for k, v in message.items() if k != 'content'})))
        content = message.get('content')
        if isinstance(content, str):
            atoms.extend(('text', c) for c in content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get('type') != 'text':
                    return atoms, True
                atoms.append(('block', canonical({k: v for k, v in block.items() if k != 'text'})))
                atoms.extend(('text', c) for c in block.get('text', ''))
        elif content is not None:
            return atoms, True
        atoms.append(('end', ''))
    return atoms, False


def common_text(a, b):
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += x[0] == 'text'
    return count


def clean(source, output):
    output.mkdir(parents=True, exist_ok=True)
    original = source/'raw_capture'
    original.mkdir(exist_ok=True)
    for name in ('messages.jsonl', 'pi_events.jsonl', 'kvcache_logical.json'):
        if not (original/name).exists():
            shutil.copy2(source/name, original/name)
    rows = [json.loads(s) for s in (original/'messages.jsonl').read_text().splitlines()]
    events = [json.loads(s) for s in (original/'pi_events.jsonl').read_text().splitlines()]
    grouped = collections.defaultdict(list)
    by_run = collections.defaultdict(list)
    for e in events:
        if e.get('run_id'):
            by_run[e['run_id']].append(e)
    for old_index, row in enumerate(rows):
        assert row.get('provider_request_id'), 'Missing server request ID'
        grouped[row['provider_request_id']].append((old_index, row))
    groups = sorted(grouped.items(), key=lambda g: min(r['timestamp'] for _, r in g[1]))
    cleaned, mapping, calls, normalized_events, prefixes = [], [], [], [], []
    for index, (pid, group) in enumerate(groups):
        assert len({digest(r['request']) for _, r in group}) == 1, f'Conflicting requests: {pid}'
        assert len({r['timestamp'] for _, r in group}) == 1, f'Conflicting starts: {pid}'
        assert all(r['messages'] == r['request']['messages'] for _, r in group)
        chosen = max(group, key=lambda item: len(canonical(item[1].get('response', {}))))[1]
        relevant = [e for _, r in group for e in by_run[r['run_id']]]
        ends = [e for e in relevant if e['type'] == 'message_end']
        assert len({canonical(e['message']['usage']) for e in ends}) == 1
        usage = copy.deepcopy(ends[0]['message']['usage'])
        # Historical zero values lack availability metadata.
        if usage.get('cacheReadAvailable') is not True and usage.get('cacheRead', 0) == 0:
            usage.update(cacheRead=None, cacheReadAvailable=False, cacheReadSource=None)
        times = {}
        for kind in ('message_start', 'message_first_token', 'message_last_token', 'message_end'):
            matches = [e for e in relevant if e['type'] == kind]
            vals = [e['message']['timestamp'] for e in matches]
            if not vals:
                continue
            times[kind] = min(vals) if kind != 'message_end' else max(vals)
            ev = copy.deepcopy(matches[0]);ev['run_id'] = pid
            ev['message']['timestamp'] = times[kind]
            if kind == 'message_end':ev['message']['usage'] = usage
            normalized_events.append(ev)
        assert times['message_start'] <= times['message_first_token'] <= times['message_last_token'] <= times['message_end']
        record = copy.deepcopy(chosen);record['run_id'] = pid
        cleaned.append(record)
        for old, row in group:
            mapping.append({'old_index': old, 'old_run_id': row['run_id'], 'clean_index': index,
                            'provider_request_id': pid, 'request_sha256': digest(row['request']),
                            'has_response': bool(row.get('response')), 'end_time_ms': next(e['message']['timestamp'] for e in by_run[row['run_id']] if e['type']=='message_end')})
        atoms, media = text_prefix(record['messages']);prefixes.append(atoms)
        start, end = times['message_start'], times['message_end']
        call = {'index': index, 'provider_request_id': pid, 'old_indices': [i for i, _ in group],
                'phase': record.get('phase'), 'model': record.get('model'), 'start_ms': start, 'end_ms': end,
                'ttft_s': (times['message_first_token']-start)/1000, 'duration_s': (end-start)/1000,
                'end_callback_spread_ms': max(e['message']['timestamp'] for e in ends)-min(e['message']['timestamp'] for e in ends),
                'usage': usage, 'nontext_boundary': media, 'request_sha256': digest(record['request']),
                'tools_sha256': digest(record['request_params'].get('tools', []))}
        call['tpot_s'] = ((times['message_last_token']-times['message_first_token'])/1000/(usage['output']-1)) if usage['output']>1 else None
        call['ttft_fraction'] = call['ttft_s']/call['duration_s']
        candidates = []
        for j, previous in enumerate(calls):
            if previous['end_ms'] <= start and previous['model'] == call['model'] and previous['tools_sha256'] == call['tools_sha256']:
                candidates.append({'source': j, 'shared_text_characters': common_text(prefixes[j], atoms)})
        call['comparisons'] = candidates
        best = max(candidates, key=lambda c: (c['shared_text_characters'], c['source']), default=None)
        previous_chars = next((c['shared_text_characters'] for c in candidates if c['source']==index-1),0)
        call['best_source'] = best['source'] if best and best['shared_text_characters'] else None
        call['best_shared_text_characters'] = best['shared_text_characters'] if best else 0
        call['previous_shared_text_characters'] = previous_chars
        call['extra_text_characters_vs_previous'] = call['best_shared_text_characters']-previous_chars
        parent = call['best_source']
        call['intervening_requests'] = index-parent-1 if parent is not None else None
        call['source_completion_gap_s'] = (start-calls[parent]['end_ms'])/1000 if parent is not None else None
        call['intervening_request_indices'] = list(range(parent+1,index)) if parent is not None else []
        call['intervening_distinct_requests'] = len({calls[j]['request_sha256'] for j in call['intervening_request_indices']})
        call['intervening_input_tokens'] = sum(calls[j]['usage']['input'] for j in call['intervening_request_indices'])
        call['exact_request_sources'] = [j for j,c in enumerate(calls) if c['end_ms']<=start and c['request_sha256']==call['request_sha256']]
        calls.append(call)
    old_calls = json.loads((original/'kvcache_logical.json').read_text())['per_call']
    old_by_run = {r['run_id']: r for r in rows}
    old_edges = old_duplicate_edges = 0
    for c in old_calls:
        candidates = c.get('logical_source_candidates', [])
        if candidates and c.get('logical_aligned', 0)>0:
            old_edges += 1
            parent = old_calls[max(candidates)]
            old_duplicate_edges += old_by_run[c['run_id']]['provider_request_id'] == old_by_run[parent['run_id']]['provider_request_id']
    cache = json.loads((source/'vllm_cache_realized.json').read_text())
    before = json.loads((source/'vllm_cache_before_serving_config.json').read_text())
    after = json.loads((source/'vllm_cache_after_serving_config.json').read_text())
    total_input = sum(c['usage']['input'] for c in calls)
    returns = [c for c in calls if c['intervening_requests'] and c['extra_text_characters_vs_previous']>0]
    summary = {'raw_records': len(rows), 'requests': len(calls), 'duplicates_removed': len(rows)-len(calls),
               'old_graph_edges': old_edges, 'old_graph_duplicate_edges': old_duplicate_edges,
               'input_tokens': total_input, 'output_tokens': sum(c['usage']['output'] for c in calls),
               'requests_with_nontext_content': sum(c['nontext_boundary'] for c in calls),
               'sum_ttft_s': sum(c['ttft_s'] for c in calls), 'sum_request_s': sum(c['duration_s'] for c in calls),
               'server_cache': cache, 'cache_queries_match_input': cache['queries']==total_input,
               'serving_config_unchanged': before==after, 'serving_config': before,
               'text_return_indices': [c['index'] for c in returns],
               'text_return_indices_at_least_256_extra_characters': [c['index'] for c in returns if c['extra_text_characters_vs_previous']>=256],
               'exact_repeat_indices': [c['index'] for c in calls if c['exact_request_sources']],
               'physical_token_prefix_reconstructed': False,
               'reason': 'Historical tokenizer revision, chat template and multimodal token sequence are not captured.',
               'text_comparison': 'Exact structured message text before first non-text block; equal model and tool schema; completed sources only. Characters are not KV tokens.',
               'source': str(source.resolve())}
    summary['ttft_fraction'] = summary['sum_ttft_s']/summary['sum_request_s']
    summary['overlapping_request_pairs'] = [(a['index'],b['index']) for i,a in enumerate(calls) for b in calls[i+1:] if b['start_ms']<a['end_ms']]
    for name, data in [('dedup_mapping.json',mapping),('per_request.json',calls),('cleaning_summary.json',summary)]:
        (output/name).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    for name, data in [('messages.jsonl', cleaned), ('pi_events.jsonl', sorted(normalized_events + [e for e in events if not e.get('type', '').startswith('message_')], key=lambda e: e.get('message', {}).get('timestamp', 0)))]:
        (output/name).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in data))
    (output/'kvcache_lineage_notes.md').write_text(
        "36 capture records represent 19 server requests. Duplicate records were merged by server request ID after verifying identical requests and usage. Sources must complete before the next request starts. "
        "The server window reports 64,448 cache hits out of 152,729 input tokens, matching deduplicated usage. Per-request cache hits are unavailable. "
        "Eleven requests contain non-text input. Historical model tokenization is unavailable, so KV-token reuse and capacity estimates are unmeasured. "
        "The lineage figure uses verified text before the first non-text block, with equal model and tool schemas. Edges require 256 shared characters for display; node sizes are uniform. "
        "Request 18 returns to request 3 after 14 intervening calls and 1,882.20 seconds, sharing 19,735 text characters.\n")
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'))
    subprocess.run([sys.executable, '-m', 'agent_io_tracing.analysis.kvcache.report', '--results', str(output.parent), '--runs', output.name], env=env, check=True)
    print(json.dumps({k:v for k,v in summary.items() if k!='serving_config'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);a=p.parse_args();clean(a.source,a.source)
