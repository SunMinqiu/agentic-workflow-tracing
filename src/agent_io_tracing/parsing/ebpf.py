"""eBPF trace parsing entry point.

The heavy implementation still lives in ``_ebpf_impl`` while the parser is
being split into schema, event loading, state tracking, and attribution modules.
"""

from agent_io_tracing.parsing._ebpf_impl import *  # noqa: F401,F403
from agent_io_tracing.parsing._ebpf_impl import main


if __name__ == "__main__":
    raise SystemExit(main())
