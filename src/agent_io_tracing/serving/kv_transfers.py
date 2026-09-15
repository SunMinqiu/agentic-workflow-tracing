"""Preserve transfer direction and counter identity in Prometheus windows."""
from __future__ import annotations

import math
import re
from pathlib import Path

_METRICS = {'vllm:kv_offload_total_bytes_total':'bytes',
            'vllm:kv_offload_total_time_total':'worker_time_s',
            'vllm:kv_offload_size_count':'worker_samples'}
_SAMPLE = re.compile(r'^(\S+?)\{(.*)\}\s+(\S+)')
_LABEL = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')


def parse(text):
    result = {}
    for line in text.splitlines():
        match = _SAMPLE.match(line)
        if not match or match[1] not in _METRICS:
            continue
        labels = tuple(sorted(_LABEL.findall(match[2])))
        value = float(match[3])
        if math.isfinite(value):
            result[(match[1], labels)] = value
    return result


def window(before, after):
    start, end = parse(before), parse(after)
    epoch_pattern=r'^process_start_time_seconds\s+(\S+)'
    before_epoch=re.search(epoch_pattern,before,re.M)
    after_epoch=re.search(epoch_pattern,after,re.M)
    restarted=bool(before_epoch and after_epoch and before_epoch[1]!=after_epoch[1])
    directions = {}
    for direction in ('GPU_to_CPU', 'CPU_to_GPU'):
        fields = {}; problems = []
        for metric, field in _METRICS.items():
            keys = {key for key in start.keys() | end.keys() if key[0]==metric and dict(key[1]).get('transfer_type')==direction}
            if not keys:
                fields[field] = None
                continue
            invalid = [key for key in keys if restarted or key not in start or key not in end or end[key] < start[key]]
            if invalid:
                fields[field] = None
                problems.append(f'{field}: series missing, process restarted, or counter decreased')
            else:
                fields[field] = sum(end[key]-start[key] for key in keys)
        if any(key[0] in _METRICS and dict(key[1]).get('transfer_type')==direction for key in start.keys() | end.keys()):
            directions[direction] = {**fields, 'status':'invalid_window' if problems else 'observed' if all(value is not None for value in fields.values()) else 'partial', 'notes':problems}
    return {'status':'observed' if directions and all(r['status']=='observed' for r in directions.values()) else 'partial' if directions else 'not_collected',
            'directions':directions,
            'scope':'Differences between metrics snapshots, summed across reported worker transfers. Worker time is not elapsed run time or request stall time. Samples are worker transfer observations, not blocks or requests.'}


def prepare(rep: Path):
    before, after = rep/'metrics_before.prom', rep/'metrics_after.prom'
    return window(before.read_text() if before.is_file() else '', after.read_text() if after.is_file() else '')
