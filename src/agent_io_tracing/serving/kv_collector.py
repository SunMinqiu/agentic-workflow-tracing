"""Persist incremental replay batches before, during, and after measurement."""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Any

from agent_io_tracing.serving.kv_events import fetch_replay, write_events


class ReplayCollector:
    def __init__(self, endpoint: str, output_dir: Path, *, process_identity: Callable[[], Any] | None = None, poll_interval_s: float = .2, timeout_ms: int = 2000):
        self.endpoint = endpoint
        self.output_dir = output_dir
        self.process_identity = process_identity
        self.poll_interval_s = poll_interval_s
        self.timeout_ms = timeout_ms
        self.collection_id = uuid.uuid4().hex
        self.phase = 'initial'
        self.generation = 0
        self.process_start = None
        self.cursor = 0
        self.errors = []
        self.restarts = 0
        self.batches = {}
        self.duplicates = 0
        self.boundaries = []
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None
        self._last_process_check = 0.0
        self._last_poll = 0.0
        self._last_reset_probe = 0.0
        self._journal = None

    def start(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._journal = (self.output_dir / 'kv_batches.jsonl').open('w', encoding='utf-8', buffering=1)
        self._thread = threading.Thread(target=self._run, name='kv-replay-collector', daemon=True)
        self._thread.start()
        self._ready.wait(self.timeout_ms / 1000 + 3)
        return self

    def _receive(self, batch):
        with self._lock:
            batch.update(publisher_id=self.endpoint, publisher_epoch=self.generation,
                         process_start_time=self.process_start, collection_id=self.collection_id,
                         phase=self.phase)
            key = (self.generation, batch['sequence'])
            previous = self.batches.get(key)
            if previous is not None:
                self.duplicates += 1
                if previous['events'] != batch['events'] or previous['batch_ts'] != batch['batch_ts']:
                    self.errors.append('Conflicting batch content for one publisher sequence.')
                return
            self.batches[key] = batch
            self._journal.write(json.dumps(batch, ensure_ascii=False) + '\n')
            self.cursor = max(self.cursor, batch['sequence'] + 1)

    def _poll(self, barrier_id=None):
        with self._lock:
            if self.process_identity and time.monotonic() - self._last_process_check >= 1:
                current = self.process_identity()
                if current is not None and self.process_start is not None and current != self.process_start:
                    self.generation += 1
                    self.restarts += 1
                    self.cursor = 0
                if current is not None:
                    self.process_start = current
                self._last_process_check = time.monotonic()
            _, transport = fetch_replay(self.endpoint, start_sequence=self.cursor,
                                        timeout_ms=self.timeout_ms, on_batch=self._receive, barrier_id=barrier_id)
            if barrier_id is None and transport['batches'] == 0 and self.cursor > 0 and time.monotonic() - self._last_reset_probe > 2:
                self._last_reset_probe = time.monotonic()
                _, previous = fetch_replay(self.endpoint, start_sequence=self.cursor - 1,
                                           timeout_ms=self.timeout_ms, on_batch=self._receive)
                if previous['batches'] == 0:
                    recovered = []
                    _, probe = fetch_replay(self.endpoint, start_sequence=0,
                                            timeout_ms=self.timeout_ms, on_batch=recovered.append)
                    if probe['last_sequence'] is not None and probe['last_sequence'] < self.cursor - 1:
                        self.generation += 1
                        self.restarts += 1
                        self.cursor = 0
                        for batch in recovered:
                            self._receive(batch)
            self._last_poll = time.monotonic()
            return transport

    def _run(self):
        try:
            while not self._stop.is_set():
                try:
                    self._poll()
                except Exception as exc:
                    with self._lock:
                        self.errors.append(f'{type(exc).__name__}: {exc}')
                finally:
                    self._ready.set()
                self._stop.wait(self.poll_interval_s)
        finally:
            self._journal.flush()

    def mark(self, phase: str):
        """Request a publication barrier, with an unconfirmed legacy fallback."""
        with self._lock:
            marker_id = uuid.uuid4().hex
            acknowledgement = None
            before = time.time()
            try:
                transport = self._poll(barrier_id=marker_id)
                acknowledgement = transport.get('barrier_ack')
                if acknowledgement is None:
                    self._poll()
                elif acknowledgement.get('barrier_id') != marker_id:
                    raise RuntimeError('Publication barrier identity mismatch')
            except Exception as exc:
                self.errors.append(f'Boundary poll failed: {exc}')
                acknowledgement = None
            with self._lock:
                item = dict(phase=phase, publisher_id=self.endpoint, publisher_epoch=self.generation,
                            next_sequence=self.cursor, client_epoch_s=time.time(),
                            client_before_epoch_s=before,
                            boundary_confirmed=bool(acknowledgement and acknowledgement.get('publication_barrier') is True and acknowledgement.get('publisher_epoch') and isinstance(acknowledgement.get('publisher_epoch_s'), (int, float)) and acknowledgement.get('last_sequence') == self.cursor - 1),
                            server_ack=acknowledgement)
                self.phase = phase
                self.boundaries.append(item)
                (self.output_dir / 'kv_boundaries.json').write_text(json.dumps(self.boundaries, indent=2) + '\n')
                return item

    def stop(self, *, drain_timeout_s: float = 3.0):
        self._stop.set()
        if self._thread:
            self._thread.join(self.timeout_ms / 1000 + 4)
            if self._thread.is_alive():
                raise RuntimeError('KV collector did not stop within its bounded timeout')
        deadline = time.monotonic() + drain_timeout_s
        stable = 0
        while time.monotonic() < deadline and stable < 2:
            before = len(self.batches)
            try:
                self._poll()
            except Exception as exc:
                self.errors.append(f'Final poll failed: {exc}')
                break
            stable = stable + 1 if len(self.batches) == before else 0
            if stable < 2:
                time.sleep(min(self.poll_interval_s, max(0, deadline - time.monotonic())))
        final_boundary = self.mark("after_measurement")
        self._journal.close()
        rows, gaps = materialize_batches(list(self.batches.values()))
        write_events(self.output_dir / 'kv_events.jsonl', rows)
        epochs = {(b.get('server_ack') or {}).get('publisher_epoch') for b in self.boundaries}
        same_publisher = len(epochs) == 1 and None not in epochs
        transport = {
            'schema_version': 3, 'endpoint': self.endpoint, 'collection_id': self.collection_id,
            'collection_method': 'continuous_incremental_replay',
            'api_process_identity_observed': self.process_start is not None,
            'publisher_identity_confirmed': bool(final_boundary.get('server_ack', {}).get('publisher_epoch')) if final_boundary.get('server_ack') else False,
            'first_sequence': min((b['sequence'] for b in self.batches.values()), default=None),
            'last_sequence': max((b['sequence'] for b in self.batches.values()), default=None),
            'batches': len(self.batches), 'events': sum(len(b['events']) for b in self.batches.values()),
            'duplicates': self.duplicates, 'restarts': self.restarts,
            'sequence_gaps': gaps, 'buffer_truncated': any(g['start_sequence'] == 0 for g in gaps),
            'collection_errors': self.errors, 'collection_error': '; '.join(self.errors) or None,
            'collection_complete': bool(len(self.boundaries) >= 2 and all(b['boundary_confirmed'] for b in self.boundaries) and same_publisher and not gaps and not self.errors and not self.restarts),
            'tail_confirmed': final_boundary['boundary_confirmed'],
            'window_definition': 'publication barriers when confirmed; collector phase otherwise',
            'tail_stable_polls': stable, 'boundaries': self.boundaries,
            'time_basis': 'publisher time for intervals; collector phase for unconfirmed window boundaries',
        }
        (self.output_dir / 'kv_collection.json').write_text(json.dumps(transport, indent=2) + '\n')
        return rows, transport


def materialize_batches(batches):
    """Preserve empty batches and insert state-breaking observations for gaps."""
    unique = {}
    for batch in batches:
        key = (batch.get('publisher_id'), batch.get('publisher_epoch', 0), batch['sequence'])
        if key in unique and unique[key]['events'] != batch['events']:
            raise ValueError('Conflicting duplicate batch')
        unique[key] = batch
    rows, gaps = [], []
    previous = {}
    for key, batch in sorted(unique.items(), key=lambda item: (str(item[0][0]), item[0][1], item[0][2])):
        stream = key[:2]
        expected = previous.get(stream, -1) + 1
        envelope = {k: v for k, v in batch.items() if k != 'events'}
        if batch['sequence'] > expected:
            gap = dict(publisher_id=key[0], publisher_epoch=key[1], start_sequence=expected,
                       end_sequence=batch['sequence'] - 1)
            gaps.append(gap)
            rows.append({**envelope, **gap, 'event_type': 'ObservationGap', 'event_index': -1})
        if not batch['events']:
            rows.append({**envelope, 'event_type': 'EmptyBatch', 'event_index': -1})
        for index, event in enumerate(batch['events']):
            rows.append({**event, **envelope, 'event_index': index})
        previous[stream] = batch['sequence']
    return rows, gaps
