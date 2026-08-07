import tiktoken

from agent_io_tracing.analysis.kvcache.logical import _serialize
from agent_io_tracing.analysis.kvcache.sections import (
    CODE, DOCUMENT, FALLBACK, GENOMAS, HISTORY, INSTRUCTIONS, RAW_DATA, SYSTEM,
    UNLABELED,
    byte_offsets, detect_grammar, overlap_by_tag, parse_sections,
    section_token_ranges, widest_span,
)

ENC = tiktoken.get_encoding("o200k_base")

PROMPT = (
    "You are a Role.GEO_AGENT working on a multi-step workflow.\n"
    "**General Guidelines**:\n"
    "Write clean code and explain nothing.\n"
    "**Task History**:\nSTEP 1\nchose the loader\n"
    "[Code]:\nimport pandas as pd\nprint(1)\n"
    "[Output]:\n1.500529793 7.306074269 GSM3721554\n"
)


def _messages(user=PROMPT, system="You are a Code Reviewer."):
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def test_sections_partition_the_prompt_exactly():
    """Contiguous, non-overlapping, and covering every character."""
    messages = _messages()
    text = _serialize(messages)
    sections = parse_sections(messages, GENOMAS)
    assert sections[0].start == 0
    assert sections[-1].end == len(text)
    for a, b in zip(sections, sections[1:]):
        assert a.end == b.start


def test_each_header_gets_its_own_content_type():
    tags = {s.name: s.tag for s in parse_sections(_messages(), GENOMAS)}
    assert tags["<system>"] == SYSTEM
    assert tags["preamble"] == INSTRUCTIONS
    assert tags["General Guidelines"] == INSTRUCTIONS
    assert tags["Task History"] == HISTORY
    assert tags["Code"] == CODE
    assert tags["Output"] == RAW_DATA


def test_a_header_labels_only_itself():
    """Output Paths is a path setting, not an execution output."""
    user = "top\nOutput Paths:\n- out: ./x.csv\n[Output]:\n3.14\n"
    tags = {s.name: s.tag for s in parse_sections(_messages(user), GENOMAS)}
    assert tags["Output Paths"] == INSTRUCTIONS
    assert tags["Output"] == RAW_DATA


def test_model_written_bold_lines_are_not_headers():
    """Only headers the template emits may cut a section."""
    user = PROMPT + "**Key Suggestions**:\nrename the variable\n"
    names = [s.name for s in parse_sections(_messages(user), GENOMAS)]
    assert "Key Suggestions" not in names
    # the review text stays inside the section it was written into
    output = [s for s in parse_sections(_messages(user), GENOMAS) if s.name == "Output"]
    assert "Key Suggestions" in _serialize(_messages(user))[output[0].start:output[0].end]


def test_preamble_takes_the_kind_of_message_it_opens():
    """The role description above the first header is still an instruction."""
    sections = parse_sections(_messages(), GENOMAS)
    preamble = [s for s in sections if s.name == "preamble"]
    assert preamble and preamble[0].tag == INSTRUCTIONS
    text = _serialize(_messages())
    assert "Role.GEO_AGENT" in text[preamble[0].start:preamble[0].end]


def test_unlabeled_is_only_for_a_message_no_header_touched():
    """The one honest use of the label: the grammar found nothing at all."""
    messages = [{"role": "user", "content": "free-form text with no headers"}]
    sections = parse_sections(messages, GENOMAS)
    assert [s.tag for s in sections] == [UNLABELED]


def test_token_ranges_keep_the_partition_after_conversion():
    messages = _messages()
    text = _serialize(messages)
    tokens = ENC.encode(text, disallowed_special=())
    ranges = section_token_ranges(messages, tokens, ENC, GENOMAS, text)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == len(tokens)
    for a, b in zip(ranges, ranges[1:]):
        assert a[1] == b[0], "a boundary inside a token must resolve the same way twice"


def test_a_token_range_decodes_to_the_text_it_claims():
    messages = _messages()
    text = _serialize(messages)
    tokens = ENC.encode(text, disallowed_special=())
    ranges = section_token_ranges(messages, tokens, ENC, GENOMAS, text)
    for start, end, name, _tag in ranges:
        if name in {"Code", "Output", "Task History"}:
            body = ENC.decode(tokens[start:end])
            assert name in body[:40], f"{name} should open its own range"


def test_byte_offsets_survive_multibyte_characters():
    text = "中文 guidelines\n[Code]:\nprint('好')\n"
    tokens = ENC.encode(text, disallowed_special=())
    offsets = byte_offsets(ENC, tokens)
    assert offsets[-1] == len(text.encode("utf-8"))
    assert len(offsets) == len(tokens) + 1


def test_multibyte_prompt_still_partitions_exactly():
    messages = [{"role": "user", "content": "序言\n[Code]:\nprint('好')\n[Output]:\n1.5\n"}]
    text = _serialize(messages)
    tokens = ENC.encode(text, disallowed_special=())
    ranges = section_token_ranges(messages, tokens, ENC, GENOMAS, text)
    assert ranges[0][0] == 0 and ranges[-1][1] == len(tokens)
    assert sum(b - a for a, b, _, _ in ranges) == len(tokens)


def test_overlap_splits_a_span_across_the_types_it_crosses():
    ranges = [(0, 10, "Code", CODE), (10, 30, "Output", RAW_DATA)]
    assert overlap_by_tag(ranges, 0, 30) == {CODE: 10, RAW_DATA: 20}
    assert overlap_by_tag(ranges, 5, 15) == {CODE: 5, RAW_DATA: 5}
    assert overlap_by_tag(ranges, 12, 12) == {}


def test_widest_span_picks_the_longest_section_of_its_type():
    ranges = [(0, 4, "Code", CODE), (4, 9, "Output", RAW_DATA), (9, 20, "Code", CODE)]
    assert widest_span(ranges, 0, 20, CODE) == (9, 20, "Code")
    # sections are ranked after the hit end clips them, not before
    assert widest_span(ranges, 0, 12, CODE) == (0, 4, "Code")
    assert widest_span(ranges, 0, 15, CODE) == (9, 15, "Code")


def test_genomas_grammar_is_detected_from_its_own_headers():
    assert detect_grammar([_messages()] * 4) is GENOMAS


def test_a_corpus_without_those_headers_falls_back_to_roles():
    pdf = [{"role": "system", "content": "analyse this"},
           {"role": "user", "content": "Figure 1 shows a lattice. " * 200}]
    grammar = detect_grammar([pdf] * 4)
    assert grammar is FALLBACK
    tags = [s.tag for s in parse_sections(pdf, grammar)]
    assert tags == [SYSTEM, DOCUMENT]


def test_one_stray_header_does_not_claim_the_corpus():
    """A single lucky match is not evidence that the template wrote it."""
    odd = [{"role": "user", "content": "prose about [Code]: in a sentence"}]
    assert detect_grammar([odd] + [[{"role": "user", "content": "plain"}]] * 9) is FALLBACK
