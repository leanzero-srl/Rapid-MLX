# SPDX-License-Identifier: Apache-2.0
"""Memory guard, review 5: a conversation's identity comes from the client
and the model's own chat template -- never a fixed token count.

45e157bab took a session's first KV block (256 tokens) as its system prompt.
Tools-first templates (Qwen3.5, Qwen3.8-27B-Atlassian) render the
reasoning-effort line and the sorted tool JSON BEFORE the system prompt, and
goose subagents inherit the session's extensions: main and subagent shared
539-577 tokens and were one "session". Now:

1. the chat route reads goose's ``agent-session-id`` header (goose sets it
   on every chat request through ``session_id_request_builder``; a subagent
   runs under its own session id) and hands it to the engine; engines that
   do not take it never see it;
2. the engine measures the rendered frame (system + tools, whatever the
   template puts first) and the first message with the template itself;
3. both reach the scheduler's ``Request.session_hint``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.request import SessionHint
from rapid_mlx.routes.chat import router as chat_router

pytestmark = pytest.mark.requires_mlx

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} " + "does a thing. " * 30,
            "parameters": {"type": "object", "properties": {}},
        },
    }
    for name in ("analyze", "developer__shell")
]
MAIN_SYSTEM = {"role": "system", "content": "You are goose. MAIN"}
SUB_SYSTEM = {"role": "system", "content": "You are a goose subagent. SUB"}


class _CharacterTokenizer:
    def encode(self, text):
        return [ord(char) for char in text]


def _tools_first(messages, tools=None, *, add_generation_prompt=True, **kwargs):
    """Qwen3.5 / Qwen3.8 shape: reasoning line + tool JSON BEFORE the
    system prompt."""
    head = "<|im_start|>system\nReasoning: high\n"
    if tools:
        head += "<tools>" + json.dumps(tools, sort_keys=True) + "</tools>\n"
    body = "".join(
        f"<|im_start|>{m['role']}\n{m.get('content') or ''}<|im_end|>\n"
        for m in messages
    )
    return head + body + ("<|im_start|>assistant\n" if add_generation_prompt else "")


class _Stub:
    def __init__(self):
        self.kwargs: dict[str, Any] | None = None

    async def generate(self, *, prompt, sampling_params, **kwargs):
        self.kwargs = kwargs
        return _Output()

    async def add_request(self, *, prompt, sampling_params, **kwargs):
        self.kwargs = kwargs
        return "req-stub"

    async def stream_outputs(self, request_id):
        yield _Output()


class _Output:
    output_text = text = new_text = "stub"
    prompt_tokens = completion_tokens = cached_tokens = 0
    finish_reason = "stop"
    usage = logprobs = token_logprobs = None
    tokens = new_token_ids = output_token_ids = []
    finished = True


def _engine(monkeypatch):
    from rapid_mlx.engine.batched import BatchedEngine

    engine = BatchedEngine("test-model")
    engine._loaded = True
    engine._is_mllm = False
    engine._engine = _Stub()
    engine._tokenizer = _CharacterTokenizer()
    monkeypatch.setattr(engine, "_apply_chat_template", _tools_first)
    monkeypatch.setattr(engine, "_is_hybrid_model", lambda: True)
    return engine


def _hint(engine, messages, session_id=None, streaming=False) -> SessionHint:
    async def run():
        if streaming:
            async for _ in engine.stream_chat(
                messages=messages, tools=TOOLS, session_id=session_id
            ):
                break
        else:
            await engine.chat(messages=messages, tools=TOOLS, session_id=session_id)

    asyncio.run(run())
    assert "session_id" not in engine._engine.kwargs  # consumed, never forwarded
    return engine._engine.kwargs["session_hint"]


def _rendered(messages):
    return _CharacterTokenizer().encode(
        _tools_first(messages, TOOLS, add_generation_prompt=True)
    )


@pytest.mark.parametrize("streaming", [False, True])
def test_the_frame_is_measured_with_the_template_not_a_block(monkeypatch, streaming):
    engine = _engine(monkeypatch)
    main = [MAIN_SYSTEM, {"role": "user", "content": "fix the bug"}]
    sub = [SUB_SYSTEM, {"role": "user", "content": "Begin."}]
    main_hint = _hint(engine, main, "20261014_7", streaming)
    sub_hint = _hint(engine, sub, "20261014_8", streaming)
    main_tokens, sub_tokens = _rendered(main), _rendered(sub)
    # What review 5 measured: the tools-first frame shares more than a KV
    # block between the two conversations ...
    shared = 0
    while main_tokens[shared] == sub_tokens[shared]:
        shared += 1
    assert shared > 256
    # ... yet the frames are each conversation's own: they end where the
    # first message's content starts, past the differing system prompt.
    text = "".join(map(chr, main_tokens))
    assert text[: main_hint.frame].endswith("<|im_start|>user\n")
    assert "MAIN" in text[: main_hint.frame]
    assert text[: main_hint.first_end].endswith(
        "fix the bug<|im_end|>\n<|im_start|>assistant\n"
    )
    assert main_tokens[: main_hint.frame] != sub_tokens[: sub_hint.frame]
    assert main_hint.session_id == "20261014_7" and main_hint.turn_start
    assert sub_hint.session_id == "20261014_8"


def test_a_tool_round_is_not_a_turn_start_and_the_frame_is_stable(monkeypatch):
    engine = _engine(monkeypatch)
    first = [MAIN_SYSTEM, {"role": "user", "content": "fix the bug"}]
    later = [
        *first,
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    a, b = _hint(engine, first, "s"), _hint(engine, later, "s")
    assert (a.frame, a.first_end) == (b.frame, b.first_end)
    assert a.turn_start and not b.turn_start


def test_no_header_still_gets_the_measured_frame(monkeypatch):
    engine = _engine(monkeypatch)
    hint = _hint(engine, [MAIN_SYSTEM, {"role": "user", "content": "x"}])
    assert hint.session_id is None and hint.first_end > hint.frame > 256


class _RecordingEngine:
    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self, accepts: bool):
        self.last_chat_kwargs: dict[str, Any] | None = None
        if accepts:
            self.ACCEPTS_SESSION_ID = True

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def chat(self, messages, **kwargs):
        self.last_chat_kwargs = kwargs
        return GenerationOutput(
            text="ok",
            raw_text="ok",
            prompt_tokens=4,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
        )


def _post(engine, headers):
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    app = FastAPI()
    app.include_router(chat_router)
    resp = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
        },
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    return engine.last_chat_kwargs


def test_the_route_hands_goose_session_header_to_the_engine():
    kwargs = _post(_RecordingEngine(True), {"agent-session-id": "20261014_7"})
    assert kwargs["session_id"] == "20261014_7"
    # Absent header, or an engine that does not take it: nothing is passed.
    assert "session_id" not in _post(_RecordingEngine(True), {})
    assert "session_id" not in _post(
        _RecordingEngine(False), {"agent-session-id": "20261014_7"}
    )
