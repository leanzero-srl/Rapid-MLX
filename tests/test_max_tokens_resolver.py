# SPDX-License-Identifier: Apache-2.0
"""Regression tests for max_tokens cap resolution.

These tests do not load a model. They pin the shared resolver used by
chat, responses, Anthropic, and completions routes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _thinking_cfg(*, default_max_tokens_is_explicit: bool):
    from rapid_mlx.config import reset_config

    cfg = reset_config()
    cfg.default_max_tokens = 128
    cfg.default_max_tokens_is_explicit = default_max_tokens_is_explicit
    cfg.thinking_token_budget = 2048
    cfg.reasoning_parser_name = "qwen3"
    return cfg


def test_request_explicit_max_tokens_is_hard_cap_for_thinking_model():
    from rapid_mlx.service.helpers import _resolve_max_tokens

    _thinking_cfg(default_max_tokens_is_explicit=False)

    assert _resolve_max_tokens(64, enable_thinking=True) == 64


def test_operator_explicit_default_max_tokens_is_hard_cap_for_thinking_model():
    from rapid_mlx.service.helpers import _resolve_max_tokens

    _thinking_cfg(default_max_tokens_is_explicit=True)

    assert _resolve_max_tokens(None, enable_thinking=True) == 128


def test_implicit_default_gets_thinking_headroom_when_request_omits_max_tokens():
    from rapid_mlx.service.helpers import _resolve_max_tokens

    _thinking_cfg(default_max_tokens_is_explicit=False)

    assert _resolve_max_tokens(None, enable_thinking=True) == 128 + 2048


def test_non_thinking_request_does_not_get_implicit_headroom():
    from rapid_mlx.service.helpers import _resolve_max_tokens

    _thinking_cfg(default_max_tokens_is_explicit=False)

    assert _resolve_max_tokens(None, enable_thinking=False) == 128


class _RawRequest:
    def __init__(self, body: dict | None = None, headers: dict | None = None):
        self._body = body or {}
        # Mirror Starlette's Request.headers (a .get()-able Mapping). The
        # chat route reads ``raw_request.headers.get("user-agent")`` for
        # the telemetry caller_agent field (routes/chat.py); without this
        # the double raised AttributeError before the resolver ran.
        self.headers = headers or {}

    async def json(self):
        return self._body

    async def is_disconnected(self):
        return False


class _CaptureChatEngine:
    supports_guided_generation = False
    preserve_native_tool_format = False
    is_mllm = False
    model_name = "test-model"
    tokenizer = SimpleNamespace(encode=lambda _text: [1])

    def __init__(self):
        self.captured_max_tokens = None

    async def chat(self, messages, **kwargs):
        from rapid_mlx.engine.base import GenerationOutput

        self.captured_max_tokens = kwargs.get("max_tokens")
        return GenerationOutput(
            text="ok",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
        )


class _CaptureCompletionEngine:
    supports_completion_logprobs = True
    tokenizer = SimpleNamespace(encode=lambda text: [1], decode=lambda ids: "x")

    def __init__(self):
        self.captured_max_tokens = None

    async def generate(self, **kwargs):
        from rapid_mlx.engine.base import GenerationOutput

        self.captured_max_tokens = kwargs.get("max_tokens")
        return GenerationOutput(
            text="ok",
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
        )


async def _await_direct(coro, *_args, **_kwargs):
    return await coro


def _patch_common_route_deps(monkeypatch, module, engine):
    resolver_calls = []

    def fake_resolver(*args, **kwargs):
        resolver_calls.append((args, kwargs))
        return 777

    monkeypatch.setattr(module, "_resolve_max_tokens", fake_resolver)
    monkeypatch.setattr(module, "get_engine", lambda *_args, **_kw: engine)
    monkeypatch.setattr(module, "_validate_model_name", lambda *_args, **_kw: None)
    monkeypatch.setattr(module, "_check_admission_or_503", lambda *_args, **_kw: None)
    monkeypatch.setattr(
        module, "_release_admission_unless_committed", lambda *_args, **_kw: None
    )
    monkeypatch.setattr(module, "_wait_with_disconnect", _await_direct)
    return resolver_calls


@pytest.mark.asyncio
async def test_chat_route_passes_resolved_max_tokens_to_engine(monkeypatch):
    from rapid_mlx.api.models import ChatCompletionRequest
    from rapid_mlx.routes import chat

    engine = _CaptureChatEngine()
    resolver_calls = _patch_common_route_deps(monkeypatch, chat, engine)
    monkeypatch.setattr(
        chat, "validate_content_blocks_for_capabilities", lambda *a, **k: None
    )
    monkeypatch.setattr(chat, "enforce_context_length_for_messages", lambda *a, **k: 1)

    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=None,
    )

    await chat._create_chat_completion_impl(
        request,
        _RawRequest(),
        engine,
        _commit_state=[False],
        _admission_acquired=[False],
    )

    assert engine.captured_max_tokens == 777, "no stated window: the resolved default"
    assert any(args and args[0] is None for args, _kwargs in resolver_calls)


@pytest.mark.asyncio
async def test_chat_route_without_max_tokens_runs_to_the_windows_room(monkeypatch):
    """LeanZero fork (goose Q-65 class): a chat request with no max_tokens on a
    model that states its window gets the room its counted prompt leaves — not
    the typed serve default (32768), which capped every such answer. The
    context guard sees the prompt alone (the default no longer refuses a prompt
    within 32768 of the window)."""
    from rapid_mlx.api.models import ChatCompletionRequest
    from rapid_mlx.routes import chat

    engine = _CaptureChatEngine()
    engine._model = SimpleNamespace(
        args=SimpleNamespace(max_position_embeddings=262_144)
    )
    _patch_common_route_deps(monkeypatch, chat, engine)
    monkeypatch.setattr(
        chat, "validate_content_blocks_for_capabilities", lambda *a, **k: None
    )
    guard_budgets = []

    def fake_guard(*_a, **k):
        guard_budgets.append(k.get("max_tokens"))
        return 240_000

    monkeypatch.setattr(chat, "enforce_context_length_for_messages", fake_guard)

    for max_tokens, expected in ((None, 262_144 - 240_000), (50, 777)):
        await chat._create_chat_completion_impl(
            ChatCompletionRequest(
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=max_tokens,
            ),
            _RawRequest(),
            engine,
            _commit_state=[False],
            _admission_acquired=[False],
        )
        assert engine.captured_max_tokens == expected

    assert guard_budgets == [None, 777], "implicit: the prompt alone is checked"


@pytest.mark.asyncio
async def test_chat_route_prompt_filling_the_window_is_context_length_exceeded(
    monkeypatch,
):
    from fastapi import HTTPException

    from rapid_mlx.api.models import ChatCompletionRequest
    from rapid_mlx.routes import chat

    engine = _CaptureChatEngine()
    engine._model = SimpleNamespace(args=SimpleNamespace(max_position_embeddings=4096))
    _patch_common_route_deps(monkeypatch, chat, engine)
    monkeypatch.setattr(
        chat, "validate_content_blocks_for_capabilities", lambda *a, **k: None
    )
    monkeypatch.setattr(
        chat, "enforce_context_length_for_messages", lambda *a, **k: 4096
    )

    with pytest.raises(HTTPException) as refused:
        await chat._create_chat_completion_impl(
            ChatCompletionRequest(
                model="test-model", messages=[{"role": "user", "content": "hi"}]
            ),
            _RawRequest(),
            engine,
            _commit_state=[False],
            _admission_acquired=[False],
        )
    assert refused.value.status_code == 400
    assert refused.value.detail["error"]["code"] == "context_length_exceeded"
    assert engine.captured_max_tokens is None


@pytest.mark.asyncio
async def test_completions_route_without_max_tokens_runs_to_the_windows_room(
    monkeypatch,
):
    from rapid_mlx.api.models import CompletionRequest
    from rapid_mlx.routes import completions

    engine = _CaptureCompletionEngine()
    engine._model = SimpleNamespace(args=SimpleNamespace(max_position_embeddings=8192))
    engine.tokenizer = SimpleNamespace(
        encode=lambda text, **_k: [1], decode=lambda ids: "x"
    )
    _patch_common_route_deps(monkeypatch, completions, engine)
    prechecks = []
    monkeypatch.setattr(
        completions,
        "enforce_context_length_for_prompt",
        lambda *a, **k: prechecks.append(k.get("max_tokens")),
    )

    await completions.create_completion(
        CompletionRequest(model="test-model", prompt="hi", max_tokens=None),
        _RawRequest(),
    )

    assert engine.captured_max_tokens == 8192 - 1, "the one-token prompt's room"
    assert prechecks == [1], "the gate proves one token of room before the stream"


@pytest.mark.asyncio
async def test_completions_route_passes_resolved_max_tokens_to_engine(monkeypatch):
    from rapid_mlx.api.models import CompletionRequest
    from rapid_mlx.routes import completions

    engine = _CaptureCompletionEngine()
    resolver_calls = _patch_common_route_deps(monkeypatch, completions, engine)
    monkeypatch.setattr(
        completions, "enforce_context_length_for_prompt", lambda *a, **k: None
    )

    await completions.create_completion(
        CompletionRequest(model="test-model", prompt="hi", max_tokens=None),
        _RawRequest(),
    )

    assert engine.captured_max_tokens == 777
    assert any(args and args[0] is None for args, _kwargs in resolver_calls)


@pytest.mark.asyncio
async def test_responses_route_passes_resolved_max_tokens_to_engine(monkeypatch):
    from rapid_mlx.api.models import ChatCompletionRequest
    from rapid_mlx.api.responses_models import ResponsesRequest
    from rapid_mlx.routes import responses

    engine = _CaptureChatEngine()
    resolver_calls = []

    def fake_resolver(*args, **kwargs):
        resolver_calls.append((args, kwargs))
        return 777

    monkeypatch.setattr(responses, "_resolve_max_tokens", fake_resolver)
    monkeypatch.setattr(responses, "_wait_with_disconnect", _await_direct)

    openai_request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=None,
    )
    responses_request = ResponsesRequest(
        model="test-model",
        input=[{"type": "message", "role": "user", "content": "hi"}],
    )

    await responses._non_stream(
        engine,
        openai_request,
        responses_request,
        _RawRequest(),
    )

    assert engine.captured_max_tokens == 777
    assert any(args and args[0] is None for args, _kwargs in resolver_calls)


@pytest.mark.asyncio
async def test_anthropic_route_passes_resolved_max_tokens_to_engine(monkeypatch):
    from rapid_mlx.routes import anthropic

    engine = _CaptureChatEngine()
    resolver_calls = _patch_common_route_deps(monkeypatch, anthropic, engine)
    monkeypatch.setattr(
        anthropic, "enforce_context_length_for_messages", lambda *a, **k: 1
    )

    await anthropic.create_anthropic_message(
        _RawRequest(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 1,
            }
        )
    )

    assert engine.captured_max_tokens == 777
    assert any(args and args[0] == 1 for args, _kwargs in resolver_calls)
