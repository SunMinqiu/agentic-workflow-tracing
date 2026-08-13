"""The retry-storm guard must fire on repeats and stay quiet on progress.

Written after a run asked the model the identical question 183 times: 327
calls where 19 were expected, 13 GB of trace, caught only because a human
happened to notice the clock.  The protection it replaces was a line in the
runbook telling the operator to watch a counter by hand.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_io_tracing.adapters.scilink.launcher import (  # noqa: E402
    RepeatGuard, request_fingerprint,
)


def test_a_growing_conversation_never_trips():
    """Hundreds of calls are fine as long as each prompt differs."""
    guard = RepeatGuard(repeat_limit=20)
    messages = [{"role": "user", "content": "start"}]
    for turn in range(500):
        messages = messages + [{"role": "assistant", "content": f"step {turn}"}]
        assert guard.check(request_fingerprint({"messages": messages})) is None
    assert guard.total == 500


def test_the_same_prompt_repeated_trips_at_the_limit():
    guard = RepeatGuard(repeat_limit=20)
    same = {"messages": [{"role": "user", "content": "analyse this"}]}
    reasons = [guard.check(request_fingerprint(same)) for _ in range(20)]
    assert all(r is None for r in reasons[:19])
    assert "retry loop" in (reasons[19] or "")


def test_a_repeat_run_broken_by_one_new_prompt_resets():
    """Two attempts then progress is normal; only a sustained run is a loop."""
    guard = RepeatGuard(repeat_limit=3)
    a = request_fingerprint({"messages": [{"role": "user", "content": "a"}]})
    b = request_fingerprint({"messages": [{"role": "user", "content": "b"}]})
    assert guard.check(a) is None
    assert guard.check(a) is None
    assert guard.check(b) is None
    assert guard.check(a) is None
    assert guard.check(a) is None
    assert guard.check(a) is not None


def test_tools_are_part_of_the_identity():
    """Same messages with a different tool set is a different question."""
    messages = [{"role": "user", "content": "x"}]
    bare = request_fingerprint({"messages": messages})
    armed = request_fingerprint({"messages": messages, "tools": [{"name": "f"}]})
    assert bare != armed


def test_total_ceiling_is_off_unless_asked_for():
    guard = RepeatGuard(repeat_limit=0, total_limit=0)
    for turn in range(1000):
        assert guard.check(f"unique-{turn}") is None


def test_total_ceiling_catches_a_loop_that_varies_its_prompt():
    guard = RepeatGuard(repeat_limit=0, total_limit=50)
    reasons = [guard.check(f"unique-{turn}") for turn in range(50)]
    assert all(r is None for r in reasons[:49])
    assert "SCILINK_MAX_CALLS" in (reasons[49] or "")
