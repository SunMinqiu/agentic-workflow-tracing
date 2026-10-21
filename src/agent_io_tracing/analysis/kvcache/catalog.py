"""Shared identities, measurements and destinations for KV replay reports."""
from __future__ import annotations

import hashlib
import html
import os
import re
from pathlib import Path
from urllib.parse import quote

from agent_io_tracing.serving.kv_report_data import load_report_data, read_json, read_rows

REPORT = 'kvcache_report.html'
COLORS = '''
tr:has(.kv-swap)>td{background:#dff1ff}
.kv-swap{font-weight:650}
tr>td.blank{background:repeating-linear-gradient(135deg,#fffaf0,#fffaf0 8px,#fff6e3 8px,#fff6e3 9px)}
.kv-legend{font-size:14px;padding:12px;border:1px solid #d7e1e4;margin:16px 0}
.kv-legend .swap-key{background:#dff1ff;padding:3px 8px}
.kv-legend .missing-key{background:#fff6e3;padding:3px 8px}
'''
LEGEND = '<p class="kv-legend"><span class="swap-key">Light blue: CPU swap experiment with KVCache eviction data</span> · <span class="missing-key">Yellow: missing measurement</span>. Collection status is shown separately.</p>'


def root_for(path: Path) -> Path:
    for parent in (path.resolve(), *path.resolve().parents):
        if parent.name in ('replay', 'results'):
            return parent.parent
    return path.resolve()


def stable_id(path: Path | str, root: Path, kind: str = 'replay') -> str:
    if isinstance(path, Path):
        value = os.path.relpath(path.resolve(), root.resolve())
    else:
        value = path
    readable = re.sub(r'[^a-zA-Z0-9_-]+', '-', value).strip('-')[-100:]
    return kind + '-' + readable + '-' + hashlib.sha256(value.encode()).hexdigest()[:10]


def resolve_recorded(raw: str, root: Path, marker: str) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    candidates = [path if path.is_absolute() else root / path]
    if marker in path.parts:
        index = len(path.parts) - 1 - path.parts[::-1].index(marker)
        candidates.insert(0, root.joinpath(*path.parts[index:]))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def cache_measurement(summary: dict) -> dict:
    request = summary.get('per_request_cache') or {}
    if request.get('coverage') == 1 and isinstance(request.get('cached_tokens'), (float, int)) and summary.get('prompt_tokens'):
        return {'rate': request['cached_tokens'] / summary['prompt_tokens'], 'hits': request['cached_tokens'], 'queries': summary['prompt_tokens'], 'source': 'per-request'}
    if request.get('coverage') == 1 and isinstance(request.get('hit_rate'), (float, int)):
        return {'rate': request['hit_rate'], 'hits': request.get('cached_tokens'), 'queries': summary.get('prompt_tokens'), 'source': 'per-request'}
    engine = summary.get('prefix_cache') or {}
    queries, prompt = engine.get('queries'), summary.get('prompt_tokens')
    if isinstance(engine.get('hits'), (float, int)) and isinstance(queries, (float, int)) and isinstance(prompt, (float, int)) and prompt > 0 and abs(queries - prompt) <= prompt * .02:
        return {'rate': engine.get('hits', 0) / queries if queries else None, 'hits': engine.get('hits'), 'queries': queries, 'source': 'engine counters'}
    return {'rate': None, 'hits': None, 'queries': None, 'source': 'not attributable / not recorded'}


def cold_state(summary: dict) -> str:
    state = summary.get('cache_state')
    if state in ('cold_by_reset', 'reset_requested'):
        return 'Reset requested; cold state unconfirmed'
    return str(state or 'Not recorded')


def swap_enabled(summary: dict) -> bool:
    try:
        return float((summary.get('serving_config') or {}).get('kv_offloading_size') or 0) > 0
    except (TypeError, ValueError):
        return False


def eviction_available(data: dict) -> bool:
    return bool(data.get('logical_events_available') or data.get('physical_trace') or (data.get('causal_trace') or {}).get('cpu_cache'))


class Catalog:
    """Read each run once and use canonical paths for all report links."""

    def __init__(self, root: Path, sources: list[Path] | None = None):
        self.root = root.resolve()
        self.records = []
        self.by_rep = {}
        self._data = {}
        paths = sources if sources is not None else (self.root / 'replay').rglob('rep*/summary.json')
        for path in sorted({p.resolve() for p in paths}):
            summary = read_json(path)
            if not summary:
                continue
            rep = path.parent
            bundle = resolve_recorded(str(summary.get('bundle') or ''), self.root, 'replay')
            source_raw = str(read_json(bundle).get('source_cell') or '') if bundle else ''
            source = resolve_recorded(source_raw, self.root, 'results') if source_raw and not source_raw.startswith('synthetic:') else None
            if source and not (source / 'messages.jsonl').is_file() and not (source / REPORT).is_file():
                source = None
            if source and not source.is_relative_to(self.root / 'results'):
                source = None
            review = read_json(rep / 'kv_report_metadata.json')
            row = dict(summary, path=path, rep=rep, family=rep.parent.parent.name, source=source,
                       source_kind='task' if source else 'synthetic' if source_raw.startswith('synthetic:') else 'unresolved',
                       source_raw=source_raw, review=review, bundle_path=bundle,
                       id=stable_id(rep, self.root), arm_id=stable_id(rep.parent, self.root, 'config'))
            row['measurement'] = cache_measurement(summary)
            row['swap'] = swap_enabled(summary)
            request_rows = read_rows(rep / 'requests.jsonl')
            for key, metric, quantile in [('server_ttft', 'time_to_first_token_ms', .5), ('server_ttft_p90', 'time_to_first_token_ms', .9), ('queue_p50', 'queue_time_ms', .5)]:
                values = [(r.get('server_metrics') or {}).get(metric) for r in request_rows]
                values = sorted(v for v in values if isinstance(v, (float, int)))
                value = None
                if values and len(values) == summary.get('requests'):
                    rank = quantile * (len(values) - 1)
                    lo = int(rank)
                    value = values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (rank - lo)
                row[key] = value
            preemptions = []
            for name in ['metrics_before.prom', 'metrics_after.prom']:
                mp = rep / name
                values = re.findall(r'^vllm:num_preemptions_total(?:\{[^}]*\})?\s+([0-9.eE+-]+)', mp.read_text(), re.M) if mp.exists() else []
                preemptions.append(sum(map(float, values)) if values else None)
            row['preemptions'] = preemptions[1] - preemptions[0] if all(v is not None for v in preemptions) else None
            self.records.append(row)
            self.by_rep[rep] = row
        self.comparisons = []
        for spec in read_json(self.root / 'config/kv_report_comparisons.json').get('comparisons', []):
            entries = []
            for entry in spec['runs']:
                record = self.by_rep.get((self.root / entry['rep']).resolve())
                if record and not record['review'].get('exclude_from_comparison'):
                    entries.append((entry['label'], record))
            baseline = self.by_rep.get((self.root / spec['baseline']).resolve())
            if not entries or not baseline or baseline['review'].get('exclude_from_comparison'):
                continue
            if any(record['source'] != baseline['source'] or record.get('mode') != baseline.get('mode') or record.get('fixed_output_tokens_per_request') != baseline.get('fixed_output_tokens_per_request') or record.get('bundle_path') != baseline.get('bundle_path') for _, record in entries):
                raise ValueError(f"Incompatible comparison: {spec['id']}")
            self.comparisons.append({**spec, 'entries': entries, 'baseline_record': baseline, 'source': baseline['source']})

    def data(self, row: dict) -> dict:
        if row['rep'] not in self._data:
            self._data[row['rep']] = load_report_data(row['rep'])
        return self._data[row['rep']]

    def blue(self, row: dict) -> bool:
        return row['swap'] and eviction_available(self.data(row))

    def page(self, row: dict) -> Path:
        return row['source'] / REPORT if row['source'] else self.root / 'results/all-experiments.html'

    def href(self, row: dict, base: Path, kind: str = 'run') -> str:
        fragment = row['arm_id'] if kind == 'config' else row['id'] + ('-eviction' if kind == 'eviction' and eviction_available(self.data(row)) else '')
        return quote(os.path.relpath(self.page(row), base), safe='/') + '#' + fragment

    def link(self, row: dict, base: Path, kind: str = 'run', label: str | None = None) -> str:
        text = label or ('CPU swap · KVCache eviction' if kind == 'eviction' and row['swap'] else 'KVCache eviction' if kind == 'eviction' else 'View this replay')
        css = ' class="kv-swap"' if self.blue(row) else ''
        title = (row['source'].name if row['source'] else row['source_kind']) + ' · ' + row['rep'].parent.name + ' · ' + row['rep'].name
        return f'<a{css} title="{html.escape(title, quote=True)}" href="{html.escape(self.href(row, base, kind), quote=True)}">{html.escape(text)}</a>'

    def task_records(self, source: Path | None) -> list[dict]:
        return [r for r in self.records if r['source'] == source]
