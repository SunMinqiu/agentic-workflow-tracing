"""Build versioned, shared KV analysis artifacts before rendering reports."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from agent_io_tracing.serving.kv_events import derive_removals, event_medium, summarize_events, write_events
from agent_io_tracing.serving.kv_lifecycle import derive_lifecycle

SCHEMA_VERSION = 3
ANALYZER_VERSION = "3.17"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []


def bundle_path(rep: Path, metadata: dict[str, Any]) -> Path | None:
    raw = metadata.get("bundle")
    if not raw:
        return None
    path = Path(str(raw))
    if path.is_absolute():
        return path if path.is_file() else None
    for base in (Path.cwd(), *rep.resolve().parents):
        candidate = base / path
        if candidate.is_file():
            return candidate
    return None


def source_fingerprints(rep: Path) -> dict[str, str | None]:
    metadata = read_json(rep / "summary.json")
    paths = {"summary": rep / "summary.json", "events": rep / "kv_events.jsonl", "collection": rep / "kv_collection.json", "metrics_samples": rep / "kv_metrics.jsonl", "bundle": bundle_path(rep, metadata)}
    paths['requests'] = rep / 'requests.jsonl'
    paths['metrics_before'] = rep / 'metrics_before.prom'
    paths['metrics_after'] = rep / 'metrics_after.prom'
    paths.update({str(path.relative_to(rep)): path for path in sorted((rep / 'physical_traces').glob('*.jsonl'))})
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None for name, path in paths.items()}


def load_report_data(rep: Path) -> dict[str, Any]:
    """Read current artifacts without rebuilding or changing any files."""
    data = read_json(rep / "kv_report_data.json")
    if data.get("schema_version") != SCHEMA_VERSION or data.get("analyzer_version") != ANALYZER_VERSION:
        return {}
    if data.get("sources") != source_fingerprints(rep):
        return {}
    return data


def publication_audit(rows, transport):
    boundaries=transport.get('boundaries') or []
    start=next((r for r in boundaries if r.get('phase')=='measurement'),None)
    end=next((r for r in reversed(boundaries) if r.get('phase')=='after_measurement'),None)
    if not start or not end or not start.get('boundary_confirmed') or not end.get('boundary_confirmed'):
        return {'status':'unconfirmed_boundaries'}
    first,last=start.get('next_sequence'),end.get('next_sequence')
    namespace=lambda r:(r.get('publisher_id'),r.get('publisher_epoch'))
    a,b=start.get('server_ack') or {},end.get('server_ack') or {}
    if namespace(start)!=namespace(end) or not a.get('publisher_epoch') or a.get('publisher_epoch')!=b.get('publisher_epoch') or not isinstance(first,int) or not isinstance(last,int) or last<first:
        return {'status':'unconfirmed_boundaries'}
    observed={r['sequence'] for r in rows if namespace(r)==namespace(start) and first<=r['sequence']<last}
    return {'status':'all_sequences_present' if len(observed)==last-first else 'missing_sequences',
            'first_sequence':first,'end_sequence_exclusive':last,'expected_batch_sequences':last-first,'observed_batch_sequences':len(observed),
            'scope':'Publication batch sequences between acknowledged boundaries. This checks delivery of published events, not whole-process trace coverage.'}


def prepare_report_data(rep: Path) -> dict[str, Any]:
    """Analyze raw input once and persist the shared report data."""
    existing = load_report_data(rep)
    figures=list(existing.get("figures", []))+[(existing.get("physical_trace") or {}).get("idle_figure"),(existing.get("causal_trace") or {}).get("reload_figure")]
    artifacts_present = all((rep / figure["path"]).is_file() for figure in figures if figure)
    if ((existing.get('causal_trace') or {}).get('cpu_cache') or {}).get('eviction_events'):
        artifacts_present = artifacts_present and (rep/'cpu_evictions.jsonl').is_file()
    if existing and artifacts_present and (not (rep / "kv_events.jsonl").is_file() or all((rep / name).is_file() for name in ("kv_lifecycle.jsonl", "kv_lifecycle_summary.json", "kv_removals.jsonl"))):
        return existing
    metadata = read_json(rep / "summary.json")
    transport = read_json(rep / "kv_collection.json") or metadata.get("kv_events") or {}
    config = metadata.get("serving_config") or {}
    try:
        raw_offload = config["kv_offloading_size"]
        enabled = False if raw_offload is None or str(raw_offload).lower() == 'none' else float(raw_offload) > 0
        offload = "enabled" if enabled else "disabled"
    except (KeyError, TypeError, ValueError):
        offload = "unknown"
    event_file = rep / "kv_events.jsonl"
    rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    removals: list[dict[str, Any]] = []
    lifecycle: dict[str, Any] = {}
    event_summary: dict[str, Any] = {}
    reasons = []
    status = "not_collected"
    try:
        rows = read_rows(event_file)
        if event_file.is_file():
            episodes, lifecycle = derive_lifecycle(rows, started_at=metadata.get("started_at_epoch_s"), ended_at=metadata.get("ended_at_epoch_s"), observation_boundaries=transport.get("boundaries"))
            source_bundle = bundle_path(rep, metadata)
            requests = read_json(source_bundle).get("requests", []) if source_bundle else []
            removals = derive_removals(rows, requests, started_at=metadata.get("started_at_epoch_s"), ended_at=metadata.get("ended_at_epoch_s"))
            event_summary = summarize_events(rows, started_at=metadata.get("started_at_epoch_s"), ended_at=metadata.get("ended_at_epoch_s"))
            status = "unconfirmed"
            if transport.get("collection_complete") is True and transport.get("tail_confirmed") is True:
                status = "complete"
            else:
                reasons.append("Measurement boundaries and final event delivery were not confirmed.")
            if lifecycle.get("missing_sequence_numbers") or transport.get("buffer_truncated"):
                status = "partial"
                reasons.append("Sequence gaps or replay buffer truncation were observed.")
            if lifecycle.get("unknown_medium_events"):
                reasons.append("Some events have unspecified storage media and are excluded from GPU return statistics.")
        if transport.get("collection_error"):
            status = "partial" if rows else "failed"
            reasons.append(str(transport["collection_error"]))
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        status = "failed"
        reasons.append(f"Event analysis failed: {exc}")
        rows, episodes, removals, lifecycle, event_summary = [], [], [], {}, {}
    cpu_observed = any(event_medium(row) == "CPU" for row in rows)
    if offload == "disabled" and cpu_observed:
        reasons.append("CPU events were observed although offload is configured as disabled.")
    # A missing event file never proves zero events. An empty unconfirmed file
    # also has no valid zero-count population.
    available = bool(rows and lifecycle) or status == "complete"
    from agent_io_tracing.serving.kv_physical import prepare as prepare_physical
    physical = prepare_physical(rep)
    from agent_io_tracing.serving.kv_causal import prepare as prepare_causal
    from agent_io_tracing.serving.kv_transfers import prepare as prepare_transfers
    causal = prepare_causal(rep, episodes=episodes, events=rows)
    aggregate_transfers = prepare_transfers(rep)
    if causal:
        def episode_key(row):
            return tuple(row.get(key) for key in ('publisher_id','publisher_epoch','data_parallel_rank','cache_generation','block_hash','group_idx','removed_sequence','return_sequence'))
        evidence={episode_key(row):row for row in read_rows(rep/'candidate_reconciliation.jsonl')}
        for episode in episodes:
            result=evidence.get(episode_key(episode))
            if result:
                episode['return_evidence']=result['status']
                episode['transfer_confirmed']=result['status']=='confirmed_load'
                episode['return_request_id']=result['source_request_id']
                episode['return_transfer_job_ids']=result.get('job_ids',[])
                episode['completed_returned_content_tokens']=result.get('completed_content_tokens')
                episode['return_cpu_store_boundary_evidence']=result.get('cpu_store_boundary_evidence')
    transfer_accounting={}
    if causal:
        jobs=read_rows(rep/'causal_transfers.jsonl')
        for direction,kind in [('GPU_to_CPU','store'),('CPU_to_GPU','load')]:
            observed=aggregate_transfers.get('directions',{}).get(direction)
            if observed is None:continue
            complete=[r for r in jobs if r['confirmed'] and r['direction']==kind]
            traced={'bytes':sum(r['worker_bytes'] for r in complete),'worker_time_s':sum(r['worker_transfer_time_s'] for r in complete),'worker_samples':sum(r['completed_workers'] for r in complete)}
            matches=all(observed.get(key) is not None and math.isclose(value,observed[key],rel_tol=1e-9,abs_tol=1e-9) for key,value in traced.items())
            transfer_accounting[direction]={'confirmed_job_totals':traced,'matches_metrics_window':matches}
    data = {
        "publication_window_audit": publication_audit(rows,transport),
        "physical_trace": physical,
        "causal_trace": causal,
        "aggregate_transfers": aggregate_transfers,
        "transfer_accounting": transfer_accounting,
        "schema_version": SCHEMA_VERSION,
        "analyzer_version": ANALYZER_VERSION,
        "sources": source_fingerprints(rep),
        "run_id": "/".join(rep.parts[-3:]),
        "window_start_epoch_s": metadata.get("started_at_epoch_s"),
        "window_end_epoch_s": metadata.get("ended_at_epoch_s"),
        "offload": offload,
        "collection_status": status,
        "quality_notes": reasons,
        "logical_events_available": available,
        "show_cpu": available and (cpu_observed or offload == "enabled"),
        "transfer_status": aggregate_transfers["status"],
        "physical_eviction_events_status": physical.get("collection_status", "not_collected"),
        "events": event_summary if available else {},
        "lifecycle": lifecycle if available else {},
        "source_transport": transport,
    }
    from agent_io_tracing.serving.kv_plots import write_plots
    data["figures"], data["return_horizons"] = write_plots(rep, rows, episodes) if available else ([], [])
    if event_file.is_file():
        write_events(rep / "kv_lifecycle.jsonl", episodes)
        write_events(rep / "kv_removals.jsonl", removals)
        (rep / "kv_lifecycle_summary.json").write_text(json.dumps({**data["lifecycle"], "collection_status": status, "source_transport": transport}, indent=2) + "\n", encoding="utf-8")
    (rep / "kv_report_data.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data
