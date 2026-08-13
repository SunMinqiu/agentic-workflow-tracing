"""One timezone for experiment directory names and reports."""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo


EXPERIMENT_TIMEZONE_NAME = "America/New_York"
EXPERIMENT_TIMEZONE = ZoneInfo(EXPERIMENT_TIMEZONE_NAME)


def from_epoch(timestamp: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(timestamp, EXPERIMENT_TIMEZONE)


def now() -> dt.datetime:
    return dt.datetime.now(EXPERIMENT_TIMEZONE)
