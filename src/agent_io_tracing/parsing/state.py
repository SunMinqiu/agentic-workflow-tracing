"""File-descriptor and process-tree state helpers used by the eBPF parser."""

from agent_io_tracing.parsing._ebpf_impl import FDInfo as EbpfFDInfo
from agent_io_tracing.parsing._ebpf_impl import FDTable as EbpfFDTable
from agent_io_tracing.parsing._ebpf_impl import ProcessTree as EbpfProcessTree

__all__ = [
    "EbpfFDInfo",
    "EbpfFDTable",
    "EbpfProcessTree",
]
