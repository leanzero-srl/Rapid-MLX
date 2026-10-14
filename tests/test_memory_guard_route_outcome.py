# SPDX-License-Identifier: Apache-2.0
"""Review 8: every chat-shaped route tells the memory guard how a response
ENDED -- "tool_calls" / "length" (the client owes the conversation its next
request: the only evidence that keeps a non-running conversation's cache) or
"stop". /v1/chat/completions threads the holder through its engine calls
(test_memory_guard_session_identity); /v1/messages and /v1/responses get it
from SESSION_OUTCOME for the request and fill it from the response."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.request import SESSION_OUTCOME
from rapid_mlx.routes.session_outcome import (
    anthropic_outcome,
    reports_outcome,
    responses_outcome,
)


class _Engine:
    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self, finish: str, tool_calls: list | None = None):
        self.finish = finish
        self.tool_calls = tool_calls
        self.holders: list[Any] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    def _out(self):
        return GenerationOutput(
            text="ok",
            new_text="ok",
            tokens=[1],
            prompt_tokens=4,
            completion_tokens=1,
            finished=True,
            finish_reason=self.finish,
            channel=None,
            tool_calls=self.tool_calls,
        )

    async def chat(self, messages, **kwargs):
        self.holders.append(SESSION_OUTCOME.get())
        return self._out()

    async def stream_chat(self, messages, **kwargs):
        self.holders.append(SESSION_OUTCOME.get())
        yield self._out()


@pytest.fixture(autouse=True)
def _reset():
    yield
    reset_config()


def _client(engine):
    from rapid_mlx.routes.anthropic import router

    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.reasoning_parser = None
    cfg.tool_parser = None
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("finish", "expected"), [("stop", "stop"), ("length", "length")]
)
def test_anthropic_messages_report_how_the_response_ended(stream, finish, expected):
    engine = _Engine(finish)
    body = {
        "model": "test-model",
        "max_tokens": 8,
        "stream": stream,
        "messages": [{"role": "user", "content": "hi"}],
    }
    resp = _client(engine).post("/v1/messages", json=body)
    assert resp.status_code == 200, resp.text
    resp.read()
    assert engine.holders and engine.holders[0] is not None
    assert engine.holders[0] == [expected], (resp.text[-400:], engine.holders)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"stop_reason":"tool_use"}', "tool_calls"),
        (
            'event: message_delta\ndata: {"delta":{"stop_reason": "max_tokens"}}',
            "length",
        ),
        ('{"stop_reason":"end_turn"}', "stop"),
        ('{"stop_reason":"stop_sequence"}', "stop"),
        ("no terminal event", None),
    ],
)
def test_anthropic_outcome(text, expected):
    assert anthropic_outcome(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            '{"output":[{"type":"function_call","name":"x"}],"status":"completed"}',
            "tool_calls",
        ),
        ('{"type":"response.incomplete","response":{"status":"incomplete"}}', "length"),
        ('{"type":"response.completed","response":{"status":"completed"}}', "stop"),
        ("data: [DONE]", None),
    ],
)
def test_responses_outcome(text, expected):
    assert responses_outcome(text) == expected


def test_a_cut_off_stream_reports_nothing():
    from fastapi.responses import StreamingResponse

    seen = []

    @reports_outcome(anthropic_outcome)
    async def handler():
        seen.append(SESSION_OUTCOME.get())

        async def body():
            yield 'data: {"delta":{"stop_reason":"end_turn"}}\n\n'
            raise RuntimeError("client gone")

        return StreamingResponse(body())

    async def run():
        response = await handler()
        async for _ in response.body_iterator:
            pass

    with pytest.raises(RuntimeError):
        asyncio.run(run())
    assert seen == [[None]]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("finish", "expected"), [("stop", "stop"), ("length", "length")]
)
def test_responses_api_reports_how_the_response_ended(stream, finish, expected):
    from rapid_mlx.routes.responses import router

    engine = _Engine(finish)
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.reasoning_parser = None
    cfg.tool_parser = None
    app = FastAPI()
    app.include_router(router)
    resp = TestClient(app).post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "hi",
            "stream": stream,
            "max_output_tokens": 8,
        },
    )
    assert resp.status_code == 200, resp.text
    resp.read()
    assert engine.holders and engine.holders[0] == [expected], (
        resp.text[-400:],
        engine.holders,
    )


TOOLS_ANTHROPIC = [
    {
        "name": "read_file",
        "description": "read a file",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
    }
]
TOOLS_RESPONSES = [
    {
        "type": "function",
        "name": "read_file",
        "description": "read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    }
]
CALL = [{"name": "read_file", "arguments": '{"path": "a.txt"}'}]


@pytest.mark.parametrize("stream", [False, True])
def test_anthropic_messages_report_a_tool_call(stream):
    """A tool_use response is the evidence the guard keeps a conversation
    for: the client runs the tool and comes back."""
    engine = _Engine("stop", tool_calls=CALL)
    body = {
        "model": "test-model",
        "max_tokens": 64,
        "stream": stream,
        "tools": TOOLS_ANTHROPIC,
        "messages": [{"role": "user", "content": "read a.txt"}],
    }
    resp = _client(engine).post("/v1/messages", json=body)
    assert resp.status_code == 200, resp.text
    resp.read()
    assert "tool_use" in resp.text, resp.text[-600:]
    assert engine.holders[0] == ["tool_calls"], (resp.text[-400:], engine.holders)


@pytest.mark.parametrize("stream", [False, True])
def test_responses_api_reports_a_tool_call(stream):
    from rapid_mlx.routes.responses import router

    engine = _Engine("stop", tool_calls=CALL)
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.reasoning_parser = None
    cfg.tool_parser = None
    app = FastAPI()
    app.include_router(router)
    resp = TestClient(app).post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "read a.txt",
            "stream": stream,
            "tools": TOOLS_RESPONSES,
            "max_output_tokens": 64,
        },
    )
    assert resp.status_code == 200, resp.text
    resp.read()
    assert "function_call" in resp.text, resp.text[-600:]
    assert engine.holders[0] == ["tool_calls"], (resp.text[-400:], engine.holders)
