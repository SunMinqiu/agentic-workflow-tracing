"""Forced streaming must reject a response that only contains thinking.

The failure this guards against is not hypothetical.  With SCILINK_FORCE_STREAM=1
a run once made 930 LLM calls where 17 were expected and wrote a 13.8 GB trace,
because the responses handed back to SciLink carried no answer and the agent
retried each one.  Probing the live server reproduced the mechanism: given a
128-token output budget the reasoning model spent 120 tokens thinking and
emitted no content at all, yet the old check passed it because it accepted
``reasoning_content`` as output.

TTFT is measured from a different rule on purpose -- on a thinking model the
first token generated *is* reasoning, so the first-token check must accept it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_io_tracing.adapters.scilink.launcher import (  # noqa: E402
    _chunk_has_content, _response_has_output,
)


def _response(**message):
    return {"choices": [{"message": message}]}


def _chunk(**delta):
    return {"choices": [{"delta": delta}]}


def test_reasoning_only_response_is_not_output():
    assert _response_has_output(_response(
        content="", tool_calls=None, reasoning_content="let me think about it",
    )) is False


def test_content_is_output():
    assert _response_has_output(_response(content="five")) is True


def test_tool_call_alone_is_output():
    assert _response_has_output(_response(
        content="", tool_calls=[{"id": "1", "function": {"name": "f"}}],
    )) is True


def test_empty_response_is_not_output():
    assert _response_has_output(_response(content="", tool_calls=[])) is False


def test_first_token_timing_accepts_reasoning():
    """TTFT must start at the first generated token, thinking included."""
    assert _chunk_has_content(_chunk(reasoning_content="hmm")) is True


def test_first_token_timing_ignores_the_role_announcement():
    assert _chunk_has_content(_chunk(role="assistant", content="")) is False
