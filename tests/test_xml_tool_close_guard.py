# SPDX-License-Identifier: Apache-2.0
"""The Qwen3-Coder XML close guard: what the model may emit after ``</parameter>\\n``.

The fixture ``xml_close_residue_write_call.txt`` is the raw text
Qwen3.8-27B-Atlassian-Q8 emitted (engine tap on the parser input) when the
first ``write`` of goose session 20260925_39 was replayed with its own
request: the content parameter closes, the model emits ``!``, then closes
again. Parsed as-is, the residue becomes the tail of the file content.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx

from rapid_mlx.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
from rapid_mlx.xml_tool_close_guard import (
    XmlToolCloseGuard,
    xml_close_guard_spec,
)

FIXTURES = Path(__file__).parent / "fixtures"
RAW_WRITE = (FIXTURES / "xml_close_residue_write_call.txt").read_text()
TEMPLATE = (FIXTURES / "qwen35_chat_template.jinja").read_text()
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]


class PieceTokenizer:
    """Longest-match tokenizer over a fixed piece list, one char otherwise.

    The pieces reproduce how the Qwen3.5 tokenizer splits the wire markers
    (``</parameter>\\n`` → ``</`` ``parameter`` ``>`` ``\\n``), which is all
    the guard reads.
    """

    PIECES = [
        "<tool_call>",
        "</tool_call>",
        "<think>",
        "</think>",
        "<|im_end|>",
        "</",
        "<",
        "parameter",
        "function",
        ">",
        "=",
        "\n\n",
        "\n",
    ]

    def __init__(self, chat_template: str | None = TEMPLATE):
        self.chat_template = chat_template
        self.vocab: dict[str, int] = {p: i for i, p in enumerate(self.PIECES)}
        self.inverse: dict[int, str] = dict(enumerate(self.PIECES))

    def _id(self, piece: str) -> int:
        if piece not in self.vocab:
            token_id = len(self.vocab)
            self.vocab[piece] = token_id
            self.inverse[token_id] = piece
        return self.vocab[piece]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids, i = [], 0
        while i < len(text):
            piece = next(
                (
                    p
                    for p in sorted(self.PIECES, key=len, reverse=True)
                    if text.startswith(p, i)
                ),
                text[i],
            )
            ids.append(self._id(piece))
            i += len(piece)
        return ids

    def decode(self, ids, **_kwargs) -> str:
        return "".join(self.inverse.get(int(i), "") for i in ids)

    def __len__(self) -> int:
        return VOCAB


VOCAB = 4096


def _spec(tok: PieceTokenizer):
    return xml_close_guard_spec(tok, {tok._id("<|im_end|>")})


def _logits(tok: PieceTokenizer, preferences: dict[str, float]) -> mx.array:
    """Logits with every token at -30 except the given pieces."""
    row = [-30.0] * VOCAB
    for piece, logprob in preferences.items():
        row[tok._id(piece)] = logprob
    return mx.array([row])


def _top(tok: PieceTokenizer, logits: mx.array) -> str:
    return tok.inverse[int(mx.argmax(logits, axis=-1).item())]


def _parse(text: str) -> dict:
    result = Qwen3CoderToolParser(None).extract_tool_calls(text, {"tools": TOOLS})
    assert result.tools_called
    call = result.tool_calls[0]
    return json.loads(
        call["arguments"] if isinstance(call, dict) else call.function.arguments
    )


# The measured distribution after the content parameter's ``</parameter>\n``
# (logprobs of the replayed session request, top candidates).
MEASURED_AFTER_LAST_CLOSE = {"}": -0.5, "]": -1.75, "!": -2.25, "</": -3.0}


def test_the_session_residue_is_payload_without_the_guard():
    """The defect as goose saw it: the file content ends with the residue."""
    args = _parse(RAW_WRITE)
    assert args["content"].endswith("</parameter>\n!")
    assert args["path"].endswith("notes/kickoff.md")


def test_spec_is_derived_from_the_tokenizer_not_assumed():
    tok = PieceTokenizer()
    spec = _spec(tok)
    assert spec is not None
    rules = {
        tok.decode(r.window): {tok.decode([i]) for i in r.allowed} for r in spec.rules
    }
    assert rules == {
        "</parameter>": {"\n"},
        "</parameter>\n": {"<", "</"},
        "</parameter>\n<": {"parameter"},
        "</parameter>\n</": {"function"},
        "</parameter>\n</function": {">"},
        "<tool_call>": {"\n"},
        "<tool_call>\n": {"<"},
        "<tool_call>\n<": {"function"},
        "</tool_call>": {"\n", "<|im_end|>"},
        "</tool_call>\n": {"<tool_call>"},
    }
    inside = {tok.decode(r.window) for r in spec.rules if r.inside_call}
    assert inside == {w for w in rules if w.startswith("</parameter>")}


def test_no_spec_for_a_template_without_the_xml_call():
    assert _spec(PieceTokenizer("{{ tools | tojson }}")) is None
    assert _spec(PieceTokenizer(None)) is None
    assert xml_close_guard_spec(PieceTokenizer(), set()) is None


def test_guard_replaces_the_measured_residue_with_the_function_close():
    """Replay the session's call with the measured preference at the close."""
    tok = PieceTokenizer()
    guard = XmlToolCloseGuard(_spec(tok))
    prompt = tok.encode("<|im_start|>assistant\n")
    close = RAW_WRITE.index("\n!\n</parameter>") + 1  # just after "</parameter>\n"
    history = prompt + tok.encode(RAW_WRITE[:close])

    unguarded = _logits(tok, MEASURED_AFTER_LAST_CLOSE)
    assert _top(tok, unguarded) == "}"
    step = guard(mx.array(history), unguarded)
    assert _top(tok, step) == "</"
    # The model's own next choice after "</" is then honoured within the rule.
    history += tok.encode("</")
    step = guard(mx.array(history), _logits(tok, {"parameter": 0.0, "function": -6.0}))
    assert _top(tok, step) == "function"
    history += tok.encode("function")
    step = guard(mx.array(history), _logits(tok, {">": 0.0}))
    assert _top(tok, step) == ">"
    history += tok.encode(">\n</tool_call>")

    args = _parse(tok.decode(history[len(prompt) :]))
    assert "</parameter>" not in args["content"]
    assert args["content"] == _parse(RAW_WRITE)["content"].removesuffix(
        "\n</parameter>\n!"
    )
    assert args["path"].endswith("notes/kickoff.md")


def test_between_parameters_the_next_opener_is_allowed():
    tok = PieceTokenizer()
    guard = XmlToolCloseGuard(_spec(tok))
    after_path = RAW_WRITE.index("<parameter=content>")
    history = tok.encode(RAW_WRITE[:after_path])
    step = guard(mx.array(history), _logits(tok, {"<": 0.0, "!": -1.0}))
    assert _top(tok, step) == "<"
    history += tok.encode("<")
    step = guard(mx.array(history), _logits(tok, {"parameter": -1.0, "}": 0.0}))
    assert _top(tok, step) == "parameter"


def test_guard_is_inert_outside_a_tool_call_and_inside_values():
    tok = PieceTokenizer()
    guard = XmlToolCloseGuard(_spec(tok))
    junk = {"!": 0.0, "</": -3.0}
    # Prose that quotes the wire, with no open <tool_call>.
    prose = tok.encode("Example:\n<parameter=x>\n1\n</parameter>\n")
    assert _top(tok, guard(mx.array(prose), _logits(tok, junk))) == "!"
    # A closed call earlier in the history does not keep the guard armed.
    closed = tok.encode(RAW_WRITE + "\nthen </parameter>\n")
    assert _top(tok, guard(mx.array(closed), _logits(tok, junk))) == "!"
    # Mid-value and an inline literal close are free text.
    mid = tok.encode(RAW_WRITE[: RAW_WRITE.index("- **Client:**")])
    assert _top(tok, guard(mx.array(mid), _logits(tok, junk))) == "!"
    inline = tok.encode(
        '<tool_call>\n<function=shell>\n<parameter=command>\nprint("</parameter>'
    )
    assert _top(tok, guard(mx.array(inline), _logits(tok, {'"': 0.0}))) == '"'
    # An indented close-looking line is a value line, not the close.
    indented = tok.encode(
        "<tool_call>\n<function=write>\n<parameter=content>\n<a>\n  </parameter>"
    )
    assert _top(tok, guard(mx.array(indented), _logits(tok, junk))) == "!"


def test_the_close_marker_itself_admits_only_the_newline():
    """Studio, guard on rules 1-3: 3 of 10 replays went ``</parameter>!``."""
    tok = PieceTokenizer()
    guard = XmlToolCloseGuard(_spec(tok))
    marker = RAW_WRITE.index("\n!\n</parameter>")  # the close ends here
    history = tok.encode(RAW_WRITE[:marker])
    assert tok.decode(history).endswith("?\n\n</parameter>")
    step = guard(mx.array(history), _logits(tok, {"!": 0.0, "\n": -2.0}))
    assert _top(tok, step) == "\n"
    # The same marker right after the opener's newline (an empty value) too.
    empty = tok.encode(
        "<tool_call>\n<function=shell>\n<parameter=command>\n</parameter>"
    )
    step = guard(mx.array(empty), _logits(tok, {"!": 0.0, "\n": -2.0}))
    assert _top(tok, step) == "\n"


def test_mtp_path_applies_the_same_rule_statelessly():
    tok = PieceTokenizer()
    guard = XmlToolCloseGuard(_spec(tok))
    close = RAW_WRITE.index("\n!\n</parameter>") + 1
    history = tok.encode(RAW_WRITE[:close])
    boundary = guard.mtp_snapshot_state()
    step = guard.mtp_apply(
        mx.array(history, dtype=mx.uint32),
        mx.array(history[-2:], dtype=mx.uint32),
        _logits(tok, MEASURED_AFTER_LAST_CLOSE),
    )
    assert _top(tok, step) == "</"
    guard.mtp_restore_state(boundary)
    assert (
        _top(tok, guard(mx.array(history), _logits(tok, MEASURED_AFTER_LAST_CLOSE)))
        == "</"
    )


def test_after_the_call_only_another_call_or_the_end_of_turn():
    """Measured: closed cleanly, the model emitted ``!`` where the turn ends."""
    tok = PieceTokenizer()
    guard = XmlToolCloseGuard(_spec(tok))
    history = tok.encode("<|im_start|>assistant\n<think>\n\n</think>\n\n" + RAW_WRITE)
    step = guard(mx.array(history), _logits(tok, {"!": 0.0, "<|im_end|>": -1.0}))
    assert _top(tok, step) == "<|im_end|>"
    step = guard(mx.array(history), _logits(tok, {"!": 0.0, "\n": -2.0}))
    assert _top(tok, step) == "\n"
    history += tok.encode("\n")
    step = guard(mx.array(history), _logits(tok, {"!": 0.0, "<tool_call>": -5.0}))
    assert _top(tok, step) == "<tool_call>"
    history += tok.encode("<tool_call>")
    step = guard(mx.array(history), _logits(tok, {"!": 0.0, "\n": -5.0}))
    assert _top(tok, step) == "\n"
    history += tok.encode("\n")
    step = guard(mx.array(history), _logits(tok, {"!": 0.0, "<": -5.0}))
    assert _top(tok, step) == "<"
    # Once the function header starts, its name is the model's own.
    history += tok.encode("<function")
    step = guard(mx.array(history), _logits(tok, {"=": 0.0, "!": -1.0}))
    assert _top(tok, step) == "="


def test_guard_is_inert_inside_an_open_think_block():
    tok = PieceTokenizer()
    guard = XmlToolCloseGuard(_spec(tok))
    upto_close = RAW_WRITE[: RAW_WRITE.index("\n!\n") + 1]
    drafting = tok.encode("<think>\nmaybe " + upto_close)
    step = guard(mx.array(drafting), _logits(tok, {"!": 0.0, "</": -3.0}))
    assert _top(tok, step) == "!"
    closed = tok.encode("<think>\nplan\n</think>\n\n" + upto_close)
    step = guard(mx.array(closed), _logits(tok, {"!": 0.0, "</": -3.0}))
    assert _top(tok, step) == "</"


@pytest.fixture(scope="module")
def qwen_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        # The revision test_tool_grammar_558 pins; cached, never fetched.
        return transformers.AutoTokenizer.from_pretrained(
            "mlx-community/Qwen3.5-4B-MLX-4bit",
            revision="32f3e8ecf65426fc3306969496342d504bfa13f3",
            local_files_only=True,
        )
    except Exception as exc:  # noqa: BLE001 - uncached tokenizer
        pytest.skip(f"Qwen3.5 tokenizer not cached: {exc}")


def test_real_qwen_tokenizer_arms_the_guard_on_the_session_text(qwen_tokenizer):
    spec = xml_close_guard_spec(
        qwen_tokenizer, {qwen_tokenizer.convert_tokens_to_ids("<|im_end|>")}
    )
    assert spec is not None
    guard = XmlToolCloseGuard(spec)
    close = RAW_WRITE.index("\n!\n</parameter>") + 1
    history = qwen_tokenizer.encode(RAW_WRITE[:close], add_special_tokens=False)
    vocab = len(qwen_tokenizer)
    row = [-30.0] * vocab

    def tid(text):
        (only,) = qwen_tokenizer.encode(text, add_special_tokens=False)
        return only

    for text, logprob in MEASURED_AFTER_LAST_CLOSE.items():
        row[tid(text)] = logprob
    out = guard(mx.array(history), mx.array([row]))
    assert int(mx.argmax(out, axis=-1).item()) == tid("</")


def _scheduler_with(tokenizer):
    from unittest.mock import MagicMock

    from rapid_mlx.scheduler import Scheduler, SchedulerConfig

    fallback = MagicMock()
    fallback.encode = lambda x: list(range(len(x.split())))
    config = SchedulerConfig(enable_prefix_cache=True, use_memory_aware_cache=True)
    scheduler = Scheduler(MagicMock(), fallback, config)
    scheduler._actual_tokenizer = tokenizer
    return scheduler


def test_scheduler_admits_the_guard_on_the_mtp_safe_row():
    """A tools request on an XML-wire model carries the guard; MTP keeps the row."""
    from unittest.mock import MagicMock

    from rapid_mlx.repetition_guard import AgentRepetitionLogitsProcessor
    from rapid_mlx.request import Request, SamplingParams

    scheduler = _scheduler_with(PieceTokenizer())
    scheduler.config.hybrid_cache_entries = 8
    scheduler.config.non_trimmable_exact_prefix_reuse = True
    request = Request(
        request_id="req-xml-close-guard",
        prompt="ignored",
        prompt_token_ids=[10, 20, 30, 40],
        sampling_params=SamplingParams(max_tokens=4),
    )
    request.has_tools = True
    request.prefix_boundary = 99
    scheduler.waiting.append(request)
    batch_generator = MagicMock()
    batch_generator.insert_segments.return_value = [105]
    batch_generator.insert.return_value = [105]
    scheduler.batch_generator = batch_generator
    scheduler._ensure_batch_generator = MagicMock(return_value=True)
    scheduler._get_request_sampler = MagicMock(return_value=MagicMock())
    scheduler._register_uid_processors = MagicMock()

    assert scheduler._schedule_waiting() == [request]

    call = batch_generator.insert_segments.call_args or batch_generator.insert.call_args
    admitted = call.kwargs["logits_processors"][0]
    assert isinstance(admitted[0], AgentRepetitionLogitsProcessor)
    assert isinstance(admitted[1], XmlToolCloseGuard)
    assert tuple(admitted) == request._mtp_safe_logits_processors


def test_scheduler_guard_absent_for_other_wires_and_on_opt_out(monkeypatch):
    assert (
        _scheduler_with(PieceTokenizer("{{ tools | tojson }}"))._xml_tool_close_guard()
        is None
    )
    monkeypatch.setenv("RAPID_MLX_XML_CLOSE_GUARD", "0")
    assert _scheduler_with(PieceTokenizer())._xml_tool_close_guard() is None
    monkeypatch.delenv("RAPID_MLX_XML_CLOSE_GUARD")
    assert isinstance(
        _scheduler_with(PieceTokenizer())._xml_tool_close_guard(), XmlToolCloseGuard
    )
