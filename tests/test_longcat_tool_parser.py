# SPDX-License-Identifier: Apache-2.0
"""LongCat tool calls (``<longcat_tool_call>`` wire), parser and route.

Calls are rendered with the checkpoint's OWN chat template
(``tests/fixtures/longcat/chat_template.jinja``, mlx-community/
LongCat-Flash-Lite-4bit), so the parser is tested against the wire the model
was trained to write. Before the ``longcat`` parser the engine returned these
calls as content with ``tool_calls`` empty (2026-10-29).

The route cases also pin the bracket-whitespace fix end to end: a streamed
code answer must concatenate to exactly the non-streamed one.
"""

import json
import re
import shutil
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.routes.chat import router as chat_router
from rapid_mlx.tool_parsers import LongCatToolParser, ToolParserManager

FIXTURE = Path(__file__).parent / "fixtures" / "longcat"
TEMPLATE = (FIXTURE / "chat_template.jinja").read_text()


def _tool(name, **props):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": props},
        },
    }


TOOLS = [
    _tool("write", path={"type": "string"}, content={"type": "string"}),
    _tool("read_file", path={"type": "string"}, max_lines={"type": "integer"}),
    _tool("tree", path={"type": "string"}),
    _tool("edit", path={"type": "string"}, edits={"type": "array"}),
    _tool("noop"),
]
REQUEST = {"tools": TOOLS}


def _render_calls(*calls, content=""):
    """The assistant row the template renders for ``calls`` (name, args)."""
    from transformers.utils.chat_template_utils import _compile_jinja_template

    msgs = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {"type": "function", "function": {"name": n, "arguments": a}}
                for n, a in calls
            ],
        },
    ]
    out = _compile_jinja_template(TEMPLATE).render(
        messages=msgs, tools=None, add_generation_prompt=False
    )
    body = out[out.rindex("<longcat_assistant>") + len("<longcat_assistant>") :]
    return body.removesuffix("</longcat_s>")


def _parse(text, request=REQUEST):
    p = LongCatToolParser(None)
    p.reset()
    return p.extract_tool_calls(text, request)


def test_registered():
    assert ToolParserManager.get_tool_parser("longcat") is LongCatToolParser


def test_the_template_renders_the_wire_the_parser_reads():
    text = _render_calls(("read_file", {"path": "src/main.rs", "max_lines": 20}))
    assert text == (
        "<longcat_tool_call>read_file\n"
        "<longcat_arg_key>path</longcat_arg_key>\n"
        "<longcat_arg_value>src/main.rs</longcat_arg_value>\n"
        "<longcat_arg_key>max_lines</longcat_arg_key>\n"
        "<longcat_arg_value>20</longcat_arg_value>\n"
        "</longcat_tool_call>\n"
    )
    r = _parse(text)
    assert r.tools_called and r.content is None
    assert r.tool_calls[0]["name"] == "read_file"
    assert json.loads(r.tool_calls[0]["arguments"]) == {
        "path": "src/main.rs",
        "max_lines": 20,
    }


def test_live_model_output_shape():
    # What LongCat-Flash-Lite-4bit wrote on 2026-10-29 (engine lz.14).
    text = (
        "<longcat_tool_call>read_file\n<longcat_arg_key>path</longcat_arg_key>\n"
        "<longcat_arg_value>src/main.rs</longcat_arg_value>\n"
        "<longcat_arg_key>max_lines</longcat_arg_key>\n"
        "<longcat_arg_value>20</longcat_arg_value>\n</longcat_tool_call>"
    )
    r = _parse(text)
    assert [c["name"] for c in r.tool_calls] == ["read_file"]
    assert json.loads(r.tool_calls[0]["arguments"]) == {
        "path": "src/main.rs",
        "max_lines": 20,
    }


@pytest.mark.parametrize(
    "content",
    [
        "def f(a: list[int]) -> int:\n    return a [0]\n\n\n",
        "  leading and trailing spaces  ",
        "42",
        '{"messages": [], "a": 1}',
        "if a < b && c > d:\n    pass\n",
        "docs about <longcat_tool_call>x</longcat_tool_call> too\n",
        "ünïcödé — 中文\n",
    ],
)
def test_string_values_round_trip_byte_identical(content):
    args = {"path": "src/a.py", "content": content}
    r = _parse(_render_calls(("write", args)))
    assert len(r.tool_calls) == 1
    assert json.loads(r.tool_calls[0]["arguments"]) == args


def test_typed_values_and_two_calls():
    edits = [{"old": "a", "new": "b\n"}, {"old": "c", "new": ""}]
    r = _parse(
        _render_calls(
            ("edit", {"path": "a.py", "edits": edits}), ("tree", {"path": "."})
        )
    )
    assert [c["name"] for c in r.tool_calls] == ["edit", "tree"]
    assert json.loads(r.tool_calls[0]["arguments"]) == {"path": "a.py", "edits": edits}


def test_call_without_arguments():
    r = _parse("<longcat_tool_call>noop\n</longcat_tool_call>")
    assert [c["name"] for c in r.tool_calls] == ["noop"]
    assert json.loads(r.tool_calls[0]["arguments"]) == {}


def test_narration_before_the_call_stays_content():
    text = _render_calls(("tree", {"path": "."}), content="Looking at the tree.")
    r = _parse(text)
    assert r.tools_called
    assert r.content == "Looking at the tree."


def test_plain_answer_is_content():
    r = _parse("No tools needed: [1, 2] < [3].")
    assert not r.tools_called
    assert r.content == "No tools needed: [1, 2] < [3]."


def test_glm_wire_is_not_read_as_longcat():
    r = _parse(
        "<tool_call>tree<arg_key>path</arg_key><arg_value>.</arg_value></tool_call>"
    )
    assert not r.tools_called


# ---------------------------------------------------------------- auto-config


def test_a_local_longcat_folder_gets_the_longcat_parser(tmp_path):
    from rapid_mlx.model_auto_config import detect_model_config

    model = tmp_path / "LongCat-Flash-Lite-4bit"
    model.mkdir()
    for name in ("config.json", "chat_template.jinja"):
        shutil.copy(FIXTURE / name, model / name)
    cfg = detect_model_config(str(model))
    assert cfg is not None
    assert cfg.tool_call_parser == "longcat"


# ---------------------------------------------------------------- route

_MARKERS = r"(</?longcat_(?:tool_call|arg_key|arg_value)>)"


def _deltas(text, size):
    """Engine-shaped deltas: each special-token tag whole, the rest chunked."""
    out = []
    for part in re.split(_MARKERS, text):
        if re.fullmatch(_MARKERS, part or ""):
            out.append(part)
        elif part:
            out.extend(part[i : i + size] for i in range(0, len(part), size))
    return out


class _Tok:
    chat_template = TEMPLATE


class _Engine:
    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = _Tok()

    def __init__(self, text, size):
        self._text = text
        self._deltas = _deltas(text, size)

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def stream_chat(self, messages, **kwargs):
        text = ""
        for i, delta in enumerate(self._deltas):
            text += delta
            last = i == len(self._deltas) - 1
            yield GenerationOutput(
                text=text,
                new_text=delta,
                prompt_tokens=4,
                completion_tokens=i + 1,
                finished=last,
                finish_reason="stop" if last else None,
                channel=None,
            )

    async def chat(self, messages, **kwargs):
        return GenerationOutput(
            text=self._text,
            new_text=self._text,
            prompt_tokens=4,
            completion_tokens=len(self._deltas),
            finished=True,
            finish_reason="stop",
            channel=None,
        )


def _client(text, size, parser="longcat"):
    cfg = reset_config()
    cfg.engine = _Engine(text, size)
    cfg.model_name = "longcat"
    cfg.model_registry = None
    cfg.enable_auto_tool_choice = parser is not None
    cfg.tool_call_parser = parser
    cfg.reasoning_parser_name = None
    cfg.reasoning_parser = None
    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app)


def _body(stream, tools=True):
    body = {
        "model": "longcat",
        "stream": stream,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "go"}],
    }
    if tools:
        body["tools"] = TOOLS
    return body


def _streamed(text, size, **kw):
    resp = _client(text, size, **kw).post(
        "/v1/chat/completions", json=_body(True, tools=kw.get("parser", 1) is not None)
    )
    assert resp.status_code == 200, resp.text
    content, finish = "", []
    calls = {}
    for line in resp.text.splitlines():
        if not line.startswith("data:") or "[DONE]" in line:
            continue
        for choice in json.loads(line[5:]).get("choices", []):
            delta = choice.get("delta") or {}
            content += delta.get("content") or ""
            for tc in delta.get("tool_calls") or []:
                call = calls.setdefault(tc["index"], {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                call["name"] += fn.get("name") or ""
                call["arguments"] += fn.get("arguments") or ""
            if choice.get("finish_reason"):
                finish.append(choice["finish_reason"])
    return content, [calls[i] for i in sorted(calls)], finish[-1] if finish else None


def _whole(text, **kw):
    resp = _client(text, 4, **kw).post(
        "/v1/chat/completions", json=_body(False, tools=kw.get("parser", 1) is not None)
    )
    assert resp.status_code == 200, resp.text
    choice = resp.json()["choices"][0]
    msg = choice["message"]
    calls = [
        {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
        for c in msg.get("tool_calls") or []
    ]
    return msg.get("content") or "", calls, choice["finish_reason"]


FILE = "fn main() {\n    let v = vec! [1, 2];\n    if v.len() < 3 && true {}\n}\n\n"
WRITE = _render_calls(("write", {"path": "src/main.rs", "content": FILE}))


@pytest.mark.parametrize("size", [1, 5, 100000])
def test_route_streamed_call_is_a_tool_call(size):
    content, calls, finish = _streamed(WRITE, size)
    assert [c["name"] for c in calls] == ["write"]
    assert json.loads(calls[0]["arguments"]) == {"path": "src/main.rs", "content": FILE}
    assert "<longcat_" not in content
    assert finish == "tool_calls"


def test_route_whole_call_is_a_tool_call():
    content, calls, finish = _whole(WRITE)
    assert [c["name"] for c in calls] == ["write"]
    assert json.loads(calls[0]["arguments"]) == {"path": "src/main.rs", "content": FILE}
    assert "<longcat_" not in content
    assert finish == "tool_calls"


CODE = (
    "Usage: tool result.json [--out FILE]\n\n\n"
    'payload = {"messages": [], "tags": ["a"]}\n\n'
    "if a < b && c:\n    x = y [0]  \n\n"
    '<div class="a">\n  ]\n'
)


@pytest.mark.parametrize("parser", [None, "longcat"])
@pytest.mark.parametrize("size", [1, 2, 3, 7])
def test_route_streamed_content_equals_whole_content(size, parser):
    streamed, _, _ = _streamed(CODE, size, parser=parser)
    whole, _, _ = _whole(CODE, parser=parser)
    assert streamed == whole == CODE
