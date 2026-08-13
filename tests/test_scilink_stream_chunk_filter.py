"""A streamed call must be recorded once, not once per chunk.

litellm runs the success callback for every chunk of a streamed response as
well as once at the end.  Recording each chunk as a finished call turned one
real request with 142 chunks into 143 trace entries, all with the same prompt,
an empty response and zero output tokens.  Four real calls were written as 143.
A healthy run then looked like a retry storm and was killed for it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_io_tracing.adapters.scilink.logger import is_stream_chunk  # noqa: E402


def test_a_chunk_is_recognised():
    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"))])
    assert is_stream_chunk(chunk) is True


def test_an_empty_role_only_chunk_is_still_a_chunk():
    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))])
    assert is_stream_chunk(chunk) is True


def test_a_finished_response_is_not_a_chunk():
    done = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])
    assert is_stream_chunk(done) is False


def test_dict_shaped_chunk_and_response():
    assert is_stream_chunk({"choices": [{"delta": {"content": "hi"}}]}) is True
    assert is_stream_chunk({"choices": [{"message": {"content": "hi"}}]}) is False


def test_anything_unrecognised_is_recorded_rather_than_dropped():
    """Losing a real call is worse than logging one odd object."""
    assert is_stream_chunk(None) is False
    assert is_stream_chunk(SimpleNamespace()) is False
    assert is_stream_chunk({"choices": []}) is False
    assert is_stream_chunk(SimpleNamespace(choices=[SimpleNamespace()])) is False
