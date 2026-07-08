"""Trace visualization CLI and compatibility entry point."""

from agent_io_tracing.viz._trace_impl import *  # noqa: F401,F403
from agent_io_tracing.viz._trace_impl import main


if __name__ == "__main__":
    main()
