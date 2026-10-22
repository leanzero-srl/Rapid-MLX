# SPDX-License-Identifier: Apache-2.0
"""Flint through the real ``/v1/chat/completions`` route, the way goose calls
it: streaming, tools offered, ``--tool-call-parser flint --reasoning-parser
flint``, and the engine's tokenizer carrying Flint's own chat template (whose
source MENTIONS ``<think>`` for history rows but never primes it).

Each case is a turn shape the model writes: a direct answer, reasoning then an
answer, reasoning then calls, narration then a call; with and without
``reasoning_effort`` (goose's t12 builds send ``medium``, the released app
sends nothing and the engine's tools gate pins thinking off).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.reasoning import get_parser
from rapid_mlx.routes.chat import router as chat_router

TEMPLATE = (Path(__file__).parent / "fixtures" / "flint" / "chat_template.jinja").read_text()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "command": {"type": "string"},
                },
            },
        },
    }
    for name in ("write", "shell", "tree")
]

FILE = 'export const x = "42";\n\n'
WRITE = (
    "<tool_call>write<arg_key>path</arg_key><arg_value>src/x.ts</arg_value>"
    f"<arg_key>content</arg_key><arg_value>{FILE}</arg_value></tool_call>"
)
SHELL = (
    "<tool_call>shell<arg_key>command</arg_key><arg_value>npm test</arg_value>"
    "</tool_call>"
)

_MARKERS = r"(</?(?:think|tool_call|arg_key|arg_value)>)"


def _deltas(text: str, size: int) -> list[str]:
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

    def __init__(self, text: str, size: int):
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


def _client(text: str, size: int) -> TestClient:
    cfg = reset_config()
    cfg.engine = _Engine(text, size)
    cfg.model_name = "flint"
    cfg.model_registry = None
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "flint"
    cfg.reasoning_parser_name = "flint"
    cfg.reasoning_parser = get_parser("flint")()  # as server.py builds it
    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app)


def _body(stream: bool, effort: str | None, tools: bool) -> dict:
    body = {
        "model": "flint",
        "stream": stream,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "go"}],
    }
    if tools:
        body["tools"] = TOOLS
    if effort:
        body["reasoning_effort"] = effort
    return body


def _streamed(text: str, size: int, **kw) -> dict:
    resp = _client(text, size).post("/v1/chat/completions", json=_body(True, **kw))
    assert resp.status_code == 200, resp.text
    content, reasoning, finish = "", "", []
    calls: dict[int, dict] = {}
    for line in resp.text.splitlines():
        if not line.startswith("data:") or "[DONE]" in line:
            continue
        for choice in json.loads(line[5:]).get("choices", []):
            delta = choice.get("delta") or {}
            content += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or delta.get("reasoning") or ""
            for tc in delta.get("tool_calls") or []:
                call = calls.setdefault(tc["index"], {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                call["name"] += fn.get("name") or ""
                call["arguments"] += fn.get("arguments") or ""
            if choice.get("finish_reason"):
                finish.append(choice["finish_reason"])
    return {
        "content": content,
        "reasoning": reasoning,
        "calls": [calls[i] for i in sorted(calls)],
        "finish": finish[-1] if finish else None,
    }


def _whole(text: str, **kw) -> dict:
    resp = _client(text, 4).post("/v1/chat/completions", json=_body(False, **kw))
    assert resp.status_code == 200, resp.text
    choice = resp.json()["choices"][0]
    msg = choice["message"]
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
        "calls": [
            {"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
            for c in msg.get("tool_calls") or []
        ],
        "finish": choice["finish_reason"],
    }


CASES = {
    "direct answer": ("The answer is 4.", "The answer is 4.", "", []),
    "think then answer": (
        "<think>\n2 and 2 make 4.\n</think>The answer is 4.",
        "The answer is 4.",
        "2 and 2 make 4.",
        [],
    ),
    "think then two calls": (
        "<think>\nWrite it, then test.\n</think>" + WRITE + SHELL,
        "",
        "Write it, then test.",
        ["write", "shell"],
    ),
    "narration then a call": ("Writing the file now.\n" + WRITE, "Writing the file now.", "", ["write"]),
}


@pytest.mark.parametrize("effort", [None, "medium"])
@pytest.mark.parametrize("size", [1, 5, 100000])
@pytest.mark.parametrize("case", sorted(CASES))
def test_streamed_turns(case, size, effort):
    text, content, reasoning, calls = CASES[case]
    out = _streamed(text, size, effort=effort, tools=True)
    assert out["content"].strip() == content, out
    assert out["reasoning"].strip() == reasoning, out
    assert [c["name"] for c in out["calls"]] == calls, out
    for leaked in ("<think>", "</think>", "<tool_call>", "<arg_key>"):
        assert leaked not in out["content"], out
    if calls:
        assert out["finish"] == "tool_calls"
        args = json.loads(out["calls"][0]["arguments"])
        assert args == {"path": "src/x.ts", "content": FILE}
    else:
        assert out["finish"] == "stop"


@pytest.mark.parametrize("effort", [None, "medium"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_whole_turns(case, effort):
    text, content, reasoning, calls = CASES[case]
    out = _whole(text, effort=effort, tools=True)
    assert out["content"].strip() == content, out
    assert out["reasoning"].strip() == reasoning, out
    assert [c["name"] for c in out["calls"]] == calls, out
    if calls:
        assert json.loads(out["calls"][0]["arguments"]) == {"path": "src/x.ts", "content": FILE}


@pytest.mark.parametrize("stream", [True, False])
def test_plain_chat_without_tools(stream):
    text = "<think>\nshort\n</think>Hello!"
    out = _streamed(text, 3, effort=None, tools=False) if stream else _whole(text, effort=None, tools=False)
    assert out["content"].strip() == "Hello!" and out["reasoning"].strip() == "short", out
