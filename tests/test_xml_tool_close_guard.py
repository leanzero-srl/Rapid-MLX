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
                (p for p in sorted(self.PIECES, key=len, reverse=True) if text.startswith(p, i)),
                text[i],
            )
            ids.append(self._id(piece))
            i += len(piece)
        return ids

    def decode(self, ids, **_kwargs) -> str:
        return "".join(self.inverse[int(i)] for i in ids)


VOCAB = 4096


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
    return json.loads(call["arguments"] if isinstance(call, dict) else call.function.arguments)


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
    spec = xml_close_guard_spec(tok)
    assert spec is not None
    assert tok.decode(spec.trigger) == "</parameter>\n"
    allowed_first = dict(spec.states)[()]
    assert {tok.decode([i]) for i in allowed_first} == {"<", "</"}
    assert {tok.decode(prefix) for prefix, _allowed in spec.states} == {
        "",
        "<",
        "</",
        "</function",
    }


def test_no_spec_for_a_template_without_the_xml_call():
    assert xml_close_guard_spec(PieceTokenizer("{{ tools | tojson }}")) is None
    assert xml_close_guard_spec(PieceTokenizer(None)) is None


def test_guard_replaces_the_measured_residue_with_the_function_close():
    """Replay the session's call with the measured preference at the close."""
    tok = PieceTokenizer()
    guard = XmlToolCloseGuard(xml_close_guard_spec(tok))
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
    guard = XmlToolCloseGuard(xml_close_guard_spec(tok))
    after_path = RAW_WRITE.index("<parameter=content>")
    history = tok.encode(RAW_WRITE[:after_path])
    step = guard(mx.array(history), _logits(tok, {"<": 0.0, "!": -1.0}))
    assert _top(tok, step) == "<"
    history += tok.encode("<")
    step = guard(mx.array(history), _logits(tok, {"parameter": -1.0, "}": 0.0}))
    assert _top(tok, step) == "parameter"


def test_guard_is_inert_outside_a_tool_call_and_inside_values():
    tok = PieceTokenizer()
    guard = XmlToolCloseGuard(xml_close_guard_spec(tok))
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
    inline = tok.encode('<tool_call>\n<function=shell>\n<parameter=command>\nprint("</parameter>')
    assert _top(tok, guard(mx.array(inline), _logits(tok, {'"': 0.0}))) == '"'


def test_mtp_path_applies_the_same_rule_statelessly():
    tok = PieceTokenizer()
    guard = XmlToolCloseGuard(xml_close_guard_spec(tok))
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
    assert _top(tok, guard(mx.array(history), _logits(tok, MEASURED_AFTER_LAST_CLOSE))) == "</"


@pytest.fixture(scope="module")
def qwen_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            "mlx-community/Qwen3.5-4B-MLX-4bit", local_files_only=True
        )
    except Exception as exc:  # noqa: BLE001 - uncached tokenizer
        pytest.skip(f"Qwen3.5 tokenizer not cached: {exc}")


def test_real_qwen_tokenizer_arms_the_guard_on_the_session_text(qwen_tokenizer):
    spec = xml_close_guard_spec(qwen_tokenizer)
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
