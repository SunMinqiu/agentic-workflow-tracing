"""Shared Darshan-style size bins for per-request and per-file plots."""
from __future__ import annotations

import math
from typing import Iterable

DARSHAN_SIZE_BINS = [
    0,
    100,
    1024,
    10 * 1024,
    100 * 1024,
    1 << 20,
    4 << 20,
    10 << 20,
    100 << 20,
    1 << 30,
    math.inf,
]

DARSHAN_SIZE_LABELS = [
    "<=100B",
    "100B-1K",
    "1K-10K",
    "10K-100K",
    "100K-1M",
    "1M-4M",
    "4M-10M",
    "10M-100M",
    "100M-1G",
    ">1G",
]


def darshan_hist(values: Iterable[float]) -> dict[str, int]:
    counts = {label: 0 for label in DARSHAN_SIZE_LABELS}
    for value in values:
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if x < 0:
            continue
        for lo, hi, label in zip(DARSHAN_SIZE_BINS, DARSHAN_SIZE_BINS[1:], DARSHAN_SIZE_LABELS):
            if lo <= x < hi or (math.isinf(hi) and x >= lo):
                counts[label] += 1
                break
    return counts


def finite_darshan_edges() -> list[float]:
    return [float(x) for x in DARSHAN_SIZE_BINS if not math.isinf(x)]
