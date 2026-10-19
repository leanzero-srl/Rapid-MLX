# SPDX-License-Identifier: Apache-2.0
"""A repetition-guard stop is told apart from an output cap on the wire.

Live defect, 2026-10-19 (RUN-ISSUES I-28, Gabee, Forge 2.0 through the goose
app): the exact-loop guard stopped a looping reply (``Stopping agent request
... after exact token loop (period_tokens=16 repeats=5)``) and the engine
remapped its internal abort to the spec-valid ``finish_reason="length"``.
goose read that as an output-token cap, told the model "your reply was cut off
at the output-token limit" three times (the wrong cause: it was looping) and
then ended the session as a normal finish at 68 of 300 calls.

The wire now keeps ``finish_reason="length"`` (spec-valid, unchanged for every
client) and adds ``"rapid_mlx_stop_reason": "repetition"`` on the SAME choice,
streamed and non-streamed. Every other stop carries no such field at all.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.request import RequestOutput
from rapid_mlx.routes.chat import router as chat_router

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "run a command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
        },
    }
]

LOOP = "I'll add it to config.js. " * 6


class _Engine:
    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self, finish_reason: str, stop_reason: str | None):
        self.finish_reason = finish_reason
        self.stop_reason = stop_reason

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def stream_chat(self, messages, **kwargs):
        pieces = [LOOP[i : i + 9] for i in range(0, len(LOOP), 9)]
        text = ""
        for i, piece in enumerate(pieces):
            text += piece
            last = i == len(pieces) - 1
            yield GenerationOutput(
                text=text,
                new_text=piece,
                prompt_tokens=4,
                completion_tokens=i + 1,
                finished=last,
                finish_reason=self.finish_reason if last else None,
                **self._reason(last),
            )

    async def chat(self, messages, **kwargs):
        return GenerationOutput(
            text=LOOP,
            prompt_tokens=4,
            completion_tokens=40,
            finish_reason=self.finish_reason,
            **self._reason(True),
        )

    def _reason(self, last: bool) -> dict:
        return {"stop_reason": self.stop_reason} if last and self.stop_reason else {}


def _client(finish_reason: str, stop_reason: str | None) -> TestClient:
    cfg = reset_config()
    cfg.engine = _Engine(finish_reason, stop_reason)
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "qwen3_coder_xml"
    app = FastAPI()
    app.include_router(chat_router)
    return TestClient(app)


def _post(client: TestClient, stream: bool):
    return client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "stream": stream,
            "max_tokens": 1024,
            "tools": TOOLS,
            "messages": [{"role": "user", "content": "go"}],
        },
    )


def _streamed_choices(client: TestClient) -> list[dict]:
    resp = _post(client, stream=True)
    assert resp.status_code == 200, resp.text
    choices = []
    for line in resp.text.splitlines():
        if line.startswith("data:") and "[DONE]" not in line:
            choices.extend(json.loads(line[5:]).get("choices", []))
    return choices


def test_a_streamed_loop_stop_names_itself_on_the_terminal_choice():
    choices = _streamed_choices(_client("length", "repetition"))
    terminal = [c for c in choices if c.get("finish_reason")]
    assert len(terminal) == 1, terminal
    assert terminal[0]["finish_reason"] == "length"
    assert terminal[0]["rapid_mlx_stop_reason"] == "repetition"
    others = [c for c in choices if not c.get("finish_reason")]
    assert others and all("rapid_mlx_stop_reason" not in c for c in others)


@pytest.mark.parametrize("finish_reason", ["length", "stop"])
def test_a_streamed_ordinary_stop_carries_no_engine_reason(finish_reason):
    choices = _streamed_choices(_client(finish_reason, None))
    assert [c["finish_reason"] for c in choices if c.get("finish_reason")] == [
        finish_reason
    ]
    assert all("rapid_mlx_stop_reason" not in c for c in choices), choices


def test_a_whole_answer_loop_stop_names_itself_on_its_choice():
    resp = _post(_client("length", "repetition"), stream=False)
    assert resp.status_code == 200, resp.text
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "length"
    assert choice["rapid_mlx_stop_reason"] == "repetition"


@pytest.mark.parametrize("finish_reason", ["length", "stop"])
def test_a_whole_answer_ordinary_stop_carries_no_engine_reason(finish_reason):
    resp = _post(_client(finish_reason, None), stream=False)
    assert resp.status_code == 200, resp.text
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == finish_reason
    assert "rapid_mlx_stop_reason" not in choice


def test_the_engine_reads_the_reason_off_the_schedulers_remapped_output():
    """engine_core remaps a repetition abort to finish_reason=length and keeps
    error_kind="repetition"; an output cap is a plain length with no kind."""
    from rapid_mlx.engine.base import engine_stop_reason

    looped = RequestOutput(
        request_id="r",
        finished=True,
        finish_reason="length",
        error_kind="repetition",
    )
    capped = RequestOutput(request_id="r", finished=True, finish_reason="length")
    stopped = RequestOutput(request_id="r", finished=True, finish_reason="stop")
    assert engine_stop_reason(looped) == "repetition"
    assert engine_stop_reason(capped) is None
    assert engine_stop_reason(stopped) is None


def test_the_output_router_keeps_the_reason_on_its_terminal_outputs():
    from rapid_mlx.engine.batched import _CHANNEL_TO_STRING, BatchedEngine

    source = GenerationOutput(
        text="x",
        finished=True,
        finish_reason="length",
        stop_reason="repetition",
    )
    sentinel = BatchedEngine._routed_finish_sentinel(None, source)
    assert sentinel.stop_reason == "repetition"

    class _Event:
        text = "x"
        token_id = 1
        channel = next(iter(_CHANNEL_TO_STRING))
        tool_call = None

    finished = BatchedEngine._make_routed_output(
        None, source, _Event(), finished=True, finish_reason="length"
    )
    assert finished.stop_reason == "repetition"
    midway = BatchedEngine._make_routed_output(None, source, _Event())
    assert midway.stop_reason is None
