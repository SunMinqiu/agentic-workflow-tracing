"""Matplotlib figures for lineage reports."""

from agent_io_tracing.lineage._analyzer_impl import (
    fig_io_volume_summary,
    fig_lifecycle_spans,
    fig_reader_fanout,
    fig_reuse_pattern,
    fig_role_io_attribution,
    fig_size_distribution,
    fig_staleness_cdf,
)

__all__ = [
    "fig_io_volume_summary",
    "fig_lifecycle_spans",
    "fig_reader_fanout",
    "fig_reuse_pattern",
    "fig_role_io_attribution",
    "fig_size_distribution",
    "fig_staleness_cdf",
]

