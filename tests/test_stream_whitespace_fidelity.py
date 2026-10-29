# SPDX-License-Identifier: Apache-2.0
"""Streamed deltas must concatenate to exactly the non-streamed text.

Regression for the bracket whitespace bug (LeanZero lz.16): every streamed
content delta went through ``sanitize_output``, which ``.strip()``-ed any
delta holding ``<``, ``[`` or ``]``.  Token deltas like ``" [--"`` and
``"]\\n\\n\\n"`` lost their whitespace, so a file written through a
streaming client came out with joined words and missing blank lines
(LongCat-Flash-Lite code_edit 8k: 41/182 lines wrong streamed, exact
non-streamed; Qwen3.8 runs: 12 wrong lines).  The non-streamed path also
trimmed the ends of any answer containing a bracket.
"""

from unittest.mock import MagicMock

import pytest

from rapid_mlx.api.models import AssistantMessage
from rapid_mlx.api.utils import sanitize_content_for_stream, sanitize_output
from rapid_mlx.reasoning.deepseek_r1_parser import DeepSeekR1DistillReasoningParser
from rapid_mlx.service.postprocessor import StreamingPostProcessor

# Token-shaped deltas that the old per-delta strip corrupted.
BRACKET_DELTAS = [
    "Usage:",
    " tool",
    " result.json",
    " [--",
    "out",
    " FILE",
    "]\n\n\n",
    'payload = {"messages":',
    " [",
    "]}\n\n",
    "if a",
    " <",
    " b",
    " &",
    "& c:",
    "\n    x = y",
    " [",
    "0",
    "]  ",
    "\n",
    "    <div",
    ' class="a">',
    "\n\n",
    "  ]\n",
]

# Code-ish text with brackets, angle brackets, ampersands and blank lines.
CODE_TEXT = (
    "def main(argv: list[str]) -> int:\n"
    '    """Run: tool result.json [--out FILE]"""\n'
    "\n"
    "\n"
    "    items = [x for x in argv if x]\n"
    '    if len(items) < 2 && items[0] > "":\n'
    "        return 1\n"
    '    payload = {"messages": [], "tags": ["a", "b"]}\n'
    "\n"
    "    html = \"<p class='x'>a &amp; b</p>\"\n"
    "    return 0\n"
)


def _cfg(**overrides):
    cfg = MagicMock()
    cfg.engine = None
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = None
    cfg.enable_auto_tool_choice = False
    cfg.tool_call_parser = None
    cfg.tool_parser_instance = None
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _out(text, finished=False):
    out = MagicMock()
    out.new_text = text
    out.finished = finished
    out.channel = None
    out.finish_reason = "stop" if finished else None
    out.prompt_tokens = 10
    out.completion_tokens = 5
    out.tokens = []
    out.logprobs = None
    out.tool_calls = None
    return out


def _stream(pp, deltas):
    pp.reset()
    pieces = []
    for i, d in enumerate(deltas):
        for ev in pp.process_chunk(_out(d, finished=(i == len(deltas) - 1))):
            if ev.type in ("content", "finish") and ev.content:
                pieces.append(ev.content)
    for ev in pp.finalize():
        if ev.type in ("content", "finish") and ev.content:
            pieces.append(ev.content)
    return "".join(pieces)


def _char_deltas(text, size):
    return [text[i : i + size] for i in range(0, len(text), size)]


@pytest.mark.parametrize(
    "delta", [" [--", "]\n\n\n", " [", "]\n\n", " <", " &", "  ]\n", "\n<div"]
)
def test_stream_sanitizer_keeps_whitespace(delta):
    assert sanitize_content_for_stream(delta) == delta


@pytest.mark.parametrize(
    "text", [" [--", "]\n\n\n", " [", "]\n\n", " <", " &", CODE_TEXT]
)
def test_final_sanitizer_returns_plain_bracket_text_unchanged(text):
    # Nothing to strip -> byte-for-byte identity (no end trim).
    assert sanitize_output(text) == text
    assert AssistantMessage(content=text).content == text


def test_final_sanitizer_still_strips_markers():
    assert sanitize_output("done<|im_end|>") == "done"
    assert sanitize_output("<|im_end|>") is None
    assert sanitize_output("a [Calling tool: x] b") == "a  b"


def test_stream_sanitizer_strips_markers_but_not_whitespace():
    assert sanitize_content_for_stream(" done<|im_end|>\n") == " done\n"


def test_standard_path_concat_equals_nonstream():
    joined = "".join(BRACKET_DELTAS)
    got = _stream(StreamingPostProcessor(_cfg()), BRACKET_DELTAS)
    assert got == joined
    assert got == sanitize_output(joined)


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 16])
def test_standard_path_code_any_split(size):
    got = _stream(StreamingPostProcessor(_cfg()), _char_deltas(CODE_TEXT, size))
    assert got == CODE_TEXT == sanitize_output(CODE_TEXT)


def test_reasoning_path_content_concat_equals_nonstream():
    cfg = _cfg(reasoning_parser=DeepSeekR1DistillReasoningParser())
    pp = StreamingPostProcessor(cfg)
    deltas = ["<think>", "plan [a]", "</think>"] + _char_deltas(CODE_TEXT, 3)
    got = _stream(pp, deltas)
    assert got == CODE_TEXT
