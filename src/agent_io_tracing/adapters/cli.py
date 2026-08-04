"""Shared command-line helpers for workflow adapters."""


def split_forwarded_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split adapter arguments from arguments after the first standalone --."""
    if "--" not in argv:
        return argv, []
    index = argv.index("--")
    return argv[:index], argv[index + 1:]
