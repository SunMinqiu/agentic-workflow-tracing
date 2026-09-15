import json

import pytest

from agent_io_tracing.adapters.genomas.logger import (
    GenoMASToolLogger, _cohort_from_messages, _infer_cohort_from_stack,
)
from agent_io_tracing.adapters.genomas.references import ReferenceSchedule


class Endpoint:
    def _request(self, path):
        return {"data": [{"id": "test", "max_model_len": 4096}]}

    def post_json(self, path, payload):
        if path == "/tokenize":
            return {"tokens": list(payload["prompt"].encode("ascii"))}
        return {"prompt": bytes(payload["tokens"]).decode("ascii")}

    def tokenize_chat(self, model, messages, **kwargs):
        return list(range(sum(len(message["content"]) + 10 for message in messages) + 10))


def schedule(tmp_path, position="front", target=1024):
    (tmp_path / "paper.txt").write_text("0123456789" * 150)
    (tmp_path / "manifest.json").write_text(json.dumps({"documents": [{
        "title": "Evidence", "text": "paper.txt", "cohorts": ["GSE123086", "GSE123088"],
    }]}))
    return ReferenceSchedule(Endpoint(), "test", tmp_path / "manifest.json", target, position, 1024, tmp_path)


@pytest.mark.parametrize("position", ["front", "back"])
def test_rotation_exhausts_text_without_repeating_or_mutating_history(tmp_path, position):
    reader = schedule(tmp_path, position)
    messages = [{"role": "system", "content": "rules"},
                {"role": "user", "content": 'cohort: "GSE123086"\nHistory: original work'}]
    cursor = 0
    for _ in range(10):
        patched, metadata = reader.add(messages, "GEOAgent", "GSE123086")
        if metadata["status"] == "skipped":
            assert metadata["reason"] == "exhausted"
            break
        assert metadata["source_start"] == cursor
        cursor = metadata["source_end"]
        assert metadata["actual_tokens"] <= 1024
        assert metadata["prompt_tokens"] + metadata["output_reserve"] <= 4096
        assert patched[0] == messages[0]
        if position == "front":
            assert patched[2] == messages[1]
        else:
            assert patched[1]["content"].startswith(messages[1]["content"])
    assert cursor == 1500
    assert "REFERENCE" not in messages[1]["content"]
    _, other = reader.add(messages, "DomainExpertAgent", "GSE123086")
    assert other["source_start"] == 0
    _, cohort = reader.add(messages, "GEOAgent", "GSE123088")
    assert cohort["source_start"] == 0


def test_context_budget_preserves_unread_content_when_no_room(tmp_path):
    reader = schedule(tmp_path, target=3000)
    messages = [{"role": "user", "content": "x" * 3500}]
    _, skipped = reader.add(messages, "GEOAgent", "GSE123086")
    assert skipped["reason"] == "no_context_room"
    messages[0]["content"] = "x" * 2000
    _, added = reader.add(messages, "GEOAgent", "GSE123086", 1500)
    assert added["source_start"] == 0
    assert added["prompt_tokens"] + 1500 <= 4096
    assert added["actual_tokens"] < 3000


def test_logger_supports_geo_prompt_without_task_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("GENOMAS_REFERENCE_TOKENS", "0")
    monkeypatch.setenv("GENOMAS_PROMPT_PREFIX_TOKENS", "0")
    logger = GenoMASToolLogger(tmp_path)
    logger._reference_schedule = schedule(tmp_path)
    messages = [{"role": "user", "content": 'cohort: "GSE123086"\nCurrent step: 1'}]
    patched, metadata = logger.add_cohort_reference(messages, "GEOAgent")
    assert metadata["status"] == "injected"
    assert patched != messages
    assert logger.add_cohort_reference(messages, "CodeReviewerAgent") == (messages, None)
    assert _cohort_from_messages([{"content": "Compare GSE123086 and GSE123088"}]) is None
    _, missing = logger.add_cohort_reference([{"role": "user", "content": "hello"}], "GEOAgent")
    assert missing["reason"] == "cohort_missing_or_ambiguous"


def test_planning_uses_current_agent_setup_instead_of_shared_previous_cohort(tmp_path, monkeypatch):
    monkeypatch.setenv("GENOMAS_REFERENCE_TOKENS", "0")
    monkeypatch.setenv("GENOMAS_PROMPT_PREFIX_TOKENS", "0")
    logger = GenoMASToolLogger(tmp_path)
    logger._reference_schedule = schedule(tmp_path)

    class GEOAgent:
        def ask(self, cohort):
            self.setups = f'cohort: "{cohort}"'
            return _infer_cohort_from_stack("GEOAgent")

    first, second = GEOAgent(), GEOAgent()
    assert first.ask("GSE123086") == "GSE123086"
    assert second.ask("GSE123088") == "GSE123088"
    _, metadata = logger.add_cohort_reference(
        [{"role": "user", "content": "Plan the next step"}], "GEOAgent",
        runtime_cohort=first.ask("GSE123086"),
    )
    assert metadata["status"] == "injected"
    assert metadata["cohort_source"] == "agent_setups"
