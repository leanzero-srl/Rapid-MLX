# SPDX-License-Identifier: Apache-2.0
"""Xing4.0 tool-call parser (``<tool_call>NAME<param_key>..</param_key>
<param_value>..</param_value></tool_call>``, raw string values)."""

import json

from rapid_mlx.tool_parsers import ToolParserManager, Xing4ToolParser


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
    ]
}


def _call(name, **params):
    body = "".join(
        f"<param_key>{k}</param_key><param_value>{v}</param_value>"
        for k, v in params.items()
    )
    return f"<tool_call>{name}{body}</tool_call>"


def _parse(text, request=REQUEST):
    p = Xing4ToolParser(None)
    p.reset()
    return p.extract_tool_calls(text, request)


def test_registered_under_its_names():
    for name in ("xing4", "xing4_0", "xing"):
        assert ToolParserManager.get_tool_parser(name) is Xing4ToolParser


def test_one_call_with_typed_values():
    r = _parse(_call("shell", command="ls -la", timeout="30"))
    assert r.tools_called
    assert r.tool_calls[0]["name"] == "shell"
    assert json.loads(r.tool_calls[0]["arguments"]) == {
        "command": "ls -la",
        "timeout": 30,
    }


def test_string_values_arrive_byte_identical():
    content = 'def f():\n    return "42"\n\n'
    r = _parse(_call("write", path="src/a.py", content=content))
    args = json.loads(r.tool_calls[0]["arguments"])
    assert args["content"] == content
    # A string that LOOKS like JSON stays a string when the schema says so.
    r = _parse(_call("write", path="n.json", content='{"a": 1}'))
    assert json.loads(r.tool_calls[0]["arguments"])["content"] == '{"a": 1}'
    r = _parse(_call("write", path="n", content="42"))
    assert json.loads(r.tool_calls[0]["arguments"])["content"] == "42"


def test_a_literal_close_inside_a_value_does_not_end_the_call():
    content = "the wire is <tool_call>x</tool_call> and </param_value> too\n"
    r = _parse(_call("write", path="docs.md", content=content))
    assert len(r.tool_calls) == 1
    assert json.loads(r.tool_calls[0]["arguments"])["content"] == content


def test_two_calls_and_a_think_block():
    text = (
        "<think>\nplan\n</think>"
        + _call("tree", path=".")
        + _call("shell", command="pwd")
    )
    r = _parse(text)
    assert [c["name"] for c in r.tool_calls] == ["tree", "shell"]
    assert r.content is None


def test_a_call_with_no_arguments():
    r = _parse("<tool_call>tree</tool_call>")
    assert r.tools_called and json.loads(r.tool_calls[0]["arguments"]) == {}


def test_prose_only_is_content():
    r = _parse("I will now look at the files.")
    assert not r.tools_called
    assert r.content == "I will now look at the files."


def test_an_unclosed_call_is_not_a_call():
    r = _parse("<tool_call>write<param_key>path</param_key><param_value>a.py")
    assert not r.tools_called


def test_an_undeclared_name_reaches_the_client_but_not_inside_code():
    r = _parse(_call("read", path="a.py"))
    assert r.tools_called and r.tool_calls[0]["name"] == "read"
    r = _parse("Example:\n```\n" + _call("read", path="a.py") + "\n```")
    assert not r.tools_called


def test_streaming_emits_the_first_call_at_its_close_and_the_rest_at_the_end():
    text = _call("tree", path=".") + _call("shell", command="pwd")
    p = Xing4ToolParser(None)
    p.reset()
    # ``<tool_call>`` / ``</tool_call>`` are SPECIAL tokens in Xing's
    # vocabulary, so a real stream delivers them whole; everything else
    # (the param tags included) arrives in arbitrary pieces.
    import re

    pieces = []
    for part in re.split(r"(</?tool_call>)", text):
        if part in ("<tool_call>", "</tool_call>"):
            pieces.append(part)
        else:
            pieces.extend(part[i : i + 7] for i in range(0, len(part), 7))
    emitted, prev, cur = [], "", ""
    for piece in pieces:
        cur = prev + piece
        out = p.extract_tool_calls_streaming(
            prev, cur, cur[len(prev) :], request=REQUEST
        )
        prev = cur
        if out and out.get("tool_calls"):
            emitted.extend(out["tool_calls"])
        assert not (out and out.get("content")), out
    rest = p.finalize_legacy_raw_stream(text, REQUEST)
    if rest:
        emitted.extend(rest["tool_calls"])
    assert [c["function"]["name"] for c in emitted] == ["tree", "shell"]
    assert [c["index"] for c in emitted] == [0, 1]


def test_holds_a_call_in_progress():
    p = Xing4ToolParser(None)
    assert p.holds_call_in_progress(
        "<tool_call>write<param_key>path</param_key>", REQUEST
    )
    assert not p.holds_call_in_progress(_call("tree", path="."), REQUEST)
