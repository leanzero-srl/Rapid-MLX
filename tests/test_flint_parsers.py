# SPDX-License-Identifier: Apache-2.0
"""Recursive Flint: the ``flint`` tool parser (``<tool_call>NAME<arg_key>K
</arg_key><arg_value>V</arg_value></tool_call>``, raw string values), the
``flint`` reasoning parser (``<think>`` only where the model writes it) and
the auto-config of a local Flint folder.

Calls are rendered with the checkpoint's OWN chat template
(``tests/fixtures/flint/chat_template.jinja``, RecursiveSI/flint@2a247be), so
the parser reads exactly the bytes the model was trained to write.
"""

import json
import re
import shutil
from pathlib import Path

import pytest

from rapid_mlx.reasoning import get_parser
from rapid_mlx.tool_parsers import FlintToolParser, ToolParserManager

FIXTURE = Path(__file__).parent / "fixtures" / "flint"
TEMPLATE = (FIXTURE / "chat_template.jinja").read_text()


def _tool(name, **props):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": props},
        },
    }


REQUEST = {
    "tools": [
        _tool("write", path={"type": "string"}, content={"type": "string"}),
        _tool("shell", command={"type": "string"}, timeout={"type": "integer"}),
        _tool("tree", path={"type": "string"}),
        _tool("edit", path={"type": "string"}, edits={"type": "array"}),
    ]
}


def _render_calls(*calls):
    """The assistant row the template renders for ``calls`` (name, args)."""
    from transformers.utils.chat_template_utils import _compile_jinja_template

    msgs = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": n, "arguments": a}}
                for n, a in calls
            ],
        },
    ]
    out = _compile_jinja_template(TEMPLATE).render(
        messages=msgs, tools=None, add_generation_prompt=False
    )
    return out[out.index("<|assistant|>") + len("<|assistant|>") :]


def _parse(text, request=REQUEST):
    p = FlintToolParser(None)
    p.reset()
    return p.extract_tool_calls(text, request)


def test_registered():
    assert ToolParserManager.get_tool_parser("flint") is FlintToolParser


def test_the_template_renders_the_wire_the_parser_reads():
    text = _render_calls(("shell", {"command": "ls -la", "timeout": 30}))
    assert text == (
        "<tool_call>shell<arg_key>command</arg_key><arg_value>ls -la</arg_value>"
        "<arg_key>timeout</arg_key><arg_value>30</arg_value></tool_call>"
    )
    r = _parse(text)
    assert r.tools_called and r.content is None
    assert json.loads(r.tool_calls[0]["arguments"]) == {
        "command": "ls -la",
        "timeout": 30,
    }


@pytest.mark.parametrize(
    "content",
    [
        'def f():\n    return "42"\n\n',
        "  leading and trailing spaces  ",
        "42",
        '{"a": 1}',
        "[1, 2]",
        "true",
        "null",
        "line with </arg_value> and <arg_key>x</arg_key> inside\n",
        "docs about <tool_call>x</tool_call> too\n",
        "ünïcödé — 中文\n",
    ],
)
def test_string_values_round_trip_byte_identical(content):
    args = {"path": "src/a.py", "content": content}
    r = _parse(_render_calls(("write", args)))
    assert len(r.tool_calls) == 1
    assert json.loads(r.tool_calls[0]["arguments"]) == args


def test_typed_values_round_trip():
    args = {
        "path": "a.py",
        "edits": [{"old": "a", "new": "b\n"}, {"old": "c", "new": ""}],
    }
    r = _parse(_render_calls(("edit", args)))
    assert json.loads(r.tool_calls[0]["arguments"]) == args


def test_parallel_calls_after_reasoning():
    text = "<think>\nplan it\n</think>" + _render_calls(
        ("tree", {"path": "."}), ("shell", {"command": "pwd"})
    )
    r = _parse(text)
    assert [c["name"] for c in r.tool_calls] == ["tree", "shell"]


def test_narration_then_a_call():
    r = _parse("Let me look.\n" + _render_calls(("tree", {"path": "."})))
    assert r.tools_called and r.tool_calls[0]["name"] == "tree"


def test_no_arguments_prose_and_unclosed():
    assert (
        json.loads(_parse("<tool_call>tree</tool_call>").tool_calls[0]["arguments"])
        == {}
    )
    r = _parse("Nothing to call here.")
    assert not r.tools_called and r.content == "Nothing to call here."
    assert not _parse(
        "<tool_call>write<arg_key>path</arg_key><arg_value>a.py"
    ).tools_called


def test_the_glm47_parser_would_corrupt_these_values():
    """Why Flint does not alias glm47: it strips and json-decodes values."""
    glm = ToolParserManager.get_tool_parser("glm47")(None)
    text = _render_calls(("write", {"path": "a", "content": "42\n"}))
    assert (
        json.loads(glm.extract_tool_calls(text, REQUEST).tool_calls[0]["arguments"])[
            "content"
        ]
        == 42
    )
    assert json.loads(_parse(text).tool_calls[0]["arguments"])["content"] == "42\n"


def _stream(parser, pieces, request=REQUEST):
    emitted, content, prev = [], [], ""
    for piece in pieces:
        cur = prev + piece
        out = parser.extract_tool_calls_streaming(prev, cur, piece, request=request)
        prev = cur
        if out and out.get("tool_calls"):
            emitted.extend(out["tool_calls"])
        if out and out.get("content"):
            content.append(out["content"])
    rest = parser.finalize_legacy_raw_stream(prev, request)
    if rest:
        emitted.extend(rest["tool_calls"])
    return emitted, "".join(content)


def _pieces(text, n=5):
    # Protocol markers are single vocabulary tokens (ids 150006-150017), so a
    # real stream delivers each whole; other text arrives in arbitrary pieces.
    out = []
    for part in re.split(r"(</?(?:tool_call|arg_key|arg_value|think)>)", text):
        if re.fullmatch(r"</?(?:tool_call|arg_key|arg_value|think)>", part or ""):
            out.append(part)
        elif part:
            out.extend(part[i : i + n] for i in range(0, len(part), n))
    return out


def test_streaming_emits_every_call_once_in_order():
    content_value = "a\n</arg_value> b\n"
    text = "Writing now.\n" + _render_calls(
        ("write", {"path": "x.md", "content": content_value}),
        ("shell", {"command": "pwd", "timeout": 5}),
    )
    p = FlintToolParser(None)
    p.reset()
    emitted, content = _stream(p, _pieces(text))
    assert [c["function"]["name"] for c in emitted] == ["write", "shell"]
    assert [c["index"] for c in emitted] == [0, 1]
    assert json.loads(emitted[0]["function"]["arguments"])["content"] == content_value
    assert content == "Writing now.\n"


def test_holds_a_call_in_progress():
    p = FlintToolParser(None)
    assert p.holds_call_in_progress("<tool_call>write<arg_key>path</arg_key>", REQUEST)
    assert not p.holds_call_in_progress(_render_calls(("tree", {"path": "."})), REQUEST)


def test_xing_still_reads_its_own_wire():
    xing = ToolParserManager.get_tool_parser("xing4")(None)
    r = xing.extract_tool_calls(
        "<tool_call>tree<param_key>path</param_key><param_value>.</param_value></tool_call>",
        REQUEST,
    )
    assert r.tools_called and json.loads(r.tool_calls[0]["arguments"]) == {"path": "."}
    # ...and not Flint's.
    assert not xing.extract_tool_calls(
        _render_calls(("tree", {"path": "."})), REQUEST
    ).tools_called


# ---------------------------------------------------------------- reasoning


def _rparser():
    p = get_parser("flint")()
    p.configure_request(enable_thinking=True, prompt_thinking_active=True)
    return p


@pytest.mark.parametrize("enable_thinking", [None, True, False])
def test_untagged_output_is_content(enable_thinking):
    p = _rparser()
    assert p.extract_reasoning("The answer is 4.", enable_thinking=enable_thinking) == (
        None,
        "The answer is 4.",
    )


def test_a_tagged_turn_splits():
    p = _rparser()
    reasoning, content = p.extract_reasoning(
        "<think>\n2+2 is 4\n</think>The answer is 4."
    )
    assert reasoning == "2+2 is 4" and content == "The answer is 4."


def _rstream(p, pieces):
    reasoning, content, prev = [], [], ""
    for piece in pieces:
        cur = prev + piece
        out = p.extract_reasoning_streaming(prev, cur, piece)
        prev = cur
        if out is not None:
            if out.reasoning:
                reasoning.append(out.reasoning)
            if out.content:
                content.append(out.content)
    fin = p.finalize_streaming(prev, finish_reason="stop")
    return "".join(reasoning), "".join(content), fin


def test_streaming_untagged_is_content_from_the_first_character():
    p = _rparser()
    text = "Sure, here it is. " * 3
    pieces = _pieces(text, 3)
    out = p.extract_reasoning_streaming("", pieces[0], pieces[0])
    assert out is not None and out.content == pieces[0] and not out.reasoning
    reasoning, content, fin = _rstream(_rparser(), pieces)
    assert reasoning == "" and content == text and fin is None


def test_streaming_think_then_answer():
    reasoning, content, fin = _rstream(
        _rparser(), _pieces("<think>\nstep one, step two\n</think>Final answer.", 4)
    )
    assert reasoning.strip() == "step one, step two"
    assert content == "Final answer."
    assert fin is None


def test_streaming_think_then_tool_call_reaches_content():
    call = _render_calls(("tree", {"path": "."}))
    reasoning, content, _ = _rstream(
        _rparser(), _pieces("<think>\nlook first\n</think>" + call, 4)
    )
    assert reasoning.strip() == "look first"
    assert content == call


# ---------------------------------------------------------------- auto-config


def test_a_local_flint_folder_gets_its_parsers(tmp_path):
    from rapid_mlx.model_auto_config import detect_model_config

    model = tmp_path / "flint-int4"
    model.mkdir()
    for name in ("config.json", "chat_template.jinja", "generation_config.json"):
        shutil.copy(FIXTURE / name, model / name)
    cfg = detect_model_config(str(model))
    assert cfg is not None
    assert cfg.tool_call_parser == "flint"
    assert cfg.reasoning_parser == "flint"
    assert cfg.is_moe is True
    assert cfg.is_hybrid is False
