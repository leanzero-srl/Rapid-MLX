# SPDX-License-Identifier: Apache-2.0
"""A text-only engine turns an image part into a text notice, not a 400.

LeanZero fork. goose gave the model a ``read_image`` tool; the model
screenshotted its own UI and read the PNG back, goose sent an image part,
and the ``--text-only`` engine answered HTTP 400 ``image_input_unsupported``.
goose treats that 400 as fatal, so a 6.75-hour benchmark session ended with
no score. Policy ``notice`` (the default) swaps the image for a deterministic
text part; policy ``reject`` keeps the 400.

Every route test renders the engine's messages through the real Qwen3.8
chat template checked in under tests/fixtures, so "the prompt contains the
notice" is checked on what the model would actually read.
"""

from __future__ import annotations

import base64
import json
import pathlib

import jinja2
import jinja2.sandbox
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.api.text_only_images import (
    reset_text_only_image_stats,
    text_only_image_notice,
    text_only_image_stats,
)
from rapid_mlx.config import get_config, reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.engine.batched import _normalize_tool_call_arguments_for_template
from rapid_mlx.routes.anthropic import router as anthropic_router
from rapid_mlx.routes.chat import router as chat_router
from rapid_mlx.routes.health import router as health_router
from rapid_mlx.routes.responses import router as responses_router
from rapid_mlx.utils.chat_template import apply_chat_template

_QWEN38_TEMPLATE = (
    pathlib.Path(__file__).parent / "fixtures" / "qwen38_chat_template.jinja"
)

_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4  # 1032 bytes
_PNG_B64 = base64.b64encode(_PNG_BYTES).decode()
_PNG_DATA_URL = f"data:image/png;base64,{_PNG_B64}"
_NOTICE_PNG = text_only_image_notice(["image/png", f"{len(_PNG_BYTES)} bytes"])


class _Qwen38Applicator:
    def __init__(self):
        def raise_exception(message):
            raise jinja2.exceptions.TemplateError(message)

        env = jinja2.sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=["jinja2.ext.loopcontrols"],
        )
        env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False, **kw)
        env.globals["raise_exception"] = raise_exception
        self.chat_template = _QWEN38_TEMPLATE.read_text()
        self._template = env.from_string(self.chat_template)

    def apply_chat_template(self, messages, **kwargs):
        kwargs.setdefault("add_generation_prompt", True)
        return self._template.render(messages=messages, **kwargs)


def _qwen_render(messages, tools=None) -> str:
    return apply_chat_template(
        _Qwen38Applicator(),
        _normalize_tool_call_arguments_for_template(
            [dict(m) if isinstance(m, dict) else m for m in messages]
        ),
        tools=tools,
        enable_thinking=False,
    )


class _TextOnlyEngine:
    preserve_native_tool_format = True
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self):
        self.chat_calls: list[dict] = []
        self.prompts: list[str] = []

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return _qwen_render(messages, tools)

    def get_stats(self):
        return {}

    async def chat(self, messages, **kwargs):
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        self.prompts.append(_qwen_render(messages, kwargs.get("tools")))
        return GenerationOutput(
            text="ok",
            new_text="ok",
            tokens=[1],
            prompt_tokens=1,
            completion_tokens=1,
            finished=True,
            finish_reason="stop",
            channel=None,
        )


class _VisionEngine(_TextOnlyEngine):
    is_mllm = True


@pytest.fixture(autouse=True)
def _reset():
    reset_text_only_image_stats()
    yield
    reset_config()
    reset_text_only_image_stats()


def _client(engine, *, policy: str | None = None) -> TestClient:
    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "text-only-model"
    cfg.model_registry = None
    cfg.no_thinking = True
    cfg.reasoning_parser = None
    cfg.reasoning_parser_name = None
    cfg.tool_parser = None
    cfg.tool_call_parser = None
    if policy is not None:
        cfg.text_only_image_policy = policy
    app = FastAPI()
    app.include_router(chat_router)
    app.include_router(responses_router)
    app.include_router(anthropic_router)
    app.include_router(health_router)
    return TestClient(app)


def _chat_image_request(url: str = _PNG_DATA_URL) -> dict:
    return {
        "model": "text-only-model",
        "max_tokens": 8,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what does the screenshot show?"},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
    }


def _goose_read_image_request() -> dict:
    """The shape of the failing goose turn: the model called read_image, the
    tool result carries text AND the image, and goose also attaches the
    image to the following user message."""
    return {
        "model": "text-only-model",
        "max_tokens": 8,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_image",
                    "description": "Read an image file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                },
            }
        ],
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Check the UI."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_image",
                            "arguments": '{"path": "/tmp/shot.png"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [
                    {"type": "text", "text": "Read /tmp/shot.png (1032 bytes)."},
                    {"type": "image_url", "image_url": {"url": _PNG_DATA_URL}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _PNG_DATA_URL}},
                ],
            },
        ],
    }


# ---------------------------------------------------------------------------
# The notice itself
# ---------------------------------------------------------------------------


def test_notice_names_mime_and_exact_byte_size():
    assert "image/png" in _NOTICE_PNG
    assert "1032 bytes" in _NOTICE_PNG
    assert _NOTICE_PNG.startswith("[image not shown: this model is serving text-only")
    assert "Describe or check it another way." in _NOTICE_PNG


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------


def test_chat_image_on_text_only_engine_returns_200_and_prompt_has_notice():
    engine = _TextOnlyEngine()
    resp = _client(engine).post("/v1/chat/completions", json=_chat_image_request())
    assert resp.status_code == 200, resp.text
    prompt = engine.prompts[-1]
    assert "<|im_start|>user\nwhat does the screenshot show?\n" + _NOTICE_PNG in prompt
    assert _PNG_B64[:40] not in prompt
    assert text_only_image_stats() == {"images_replaced": 1, "requests": 1}


def test_chat_goose_read_image_turn_keeps_tool_text_and_renders_notice():
    engine = _TextOnlyEngine()
    resp = _client(engine).post(
        "/v1/chat/completions", json=_goose_read_image_request()
    )
    assert resp.status_code == 200, resp.text
    prompt = engine.prompts[-1]
    # The tool reply keeps its text and gains the notice, inside the Qwen
    # tool-response envelope.
    tool_block = prompt.split("<tool_response>", 1)[1].split("</tool_response>", 1)[0]
    assert "Read /tmp/shot.png (1032 bytes)." in tool_block
    assert _NOTICE_PNG in tool_block
    # The follow-up user image is a notice too.
    assert prompt.count(_NOTICE_PNG) == 2
    assert _PNG_B64[:40] not in prompt
    assert text_only_image_stats() == {"images_replaced": 2, "requests": 1}


def test_chat_remote_url_notice_names_the_source():
    engine = _TextOnlyEngine()
    resp = _client(engine).post(
        "/v1/chat/completions",
        json=_chat_image_request("https://example.com/shot.png"),
    )
    assert resp.status_code == 200, resp.text
    assert (
        text_only_image_notice(["from https://example.com/shot.png"])
        in engine.prompts[-1]
    )


def test_chat_reject_policy_keeps_the_400():
    engine = _TextOnlyEngine()
    resp = _client(engine, policy="reject").post(
        "/v1/chat/completions", json=_chat_image_request()
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"]["error"]["code"] == "image_input_unsupported"
    assert engine.chat_calls == []
    assert text_only_image_stats() == {"images_replaced": 0, "requests": 0}


def test_chat_reject_policy_keeps_the_tool_content_400():
    engine = _TextOnlyEngine()
    resp = _client(engine, policy="reject").post(
        "/v1/chat/completions", json=_goose_read_image_request()
    )
    assert resp.status_code == 400, resp.text
    assert "role 'tool' content must be" in resp.text


def test_chat_vision_engine_is_unaffected():
    engine = _VisionEngine()
    resp = _client(engine).post("/v1/chat/completions", json=_chat_image_request())
    assert resp.status_code == 200, resp.text
    content = engine.chat_calls[-1]["messages"][0]["content"]
    assert [part["type"] for part in content] == ["text", "image_url"]
    assert content[1]["image_url"]["url"] == _PNG_DATA_URL
    assert text_only_image_stats() == {"images_replaced": 0, "requests": 0}


def test_chat_malformed_image_part_still_rejected_with_shape_error():
    engine = _TextOnlyEngine()
    body = _chat_image_request()
    body["messages"][0]["content"][1] = {"type": "image_url", "image_url": {}}
    resp = _client(engine).post("/v1/chat/completions", json=body)
    assert resp.status_code in (400, 422), resp.text
    assert engine.chat_calls == []


def test_chat_video_on_text_only_engine_unchanged():
    engine = _TextOnlyEngine()
    body = _chat_image_request()
    body["messages"][0]["content"][1] = {
        "type": "video_url",
        "video_url": {"url": "https://example.com/v.mp4"},
    }
    resp = _client(engine).post("/v1/chat/completions", json=body)
    assert resp.status_code == 400
    assert "does not support video inputs" in resp.text


def test_chat_same_request_renders_identically_for_the_prefix_cache():
    engine = _TextOnlyEngine()
    client = _client(engine)
    for _ in range(2):
        resp = client.post("/v1/chat/completions", json=_goose_read_image_request())
        assert resp.status_code == 200, resp.text
    assert len(engine.prompts) == 2
    assert engine.prompts[0] == engine.prompts[1]


def test_next_turn_repeating_the_image_keeps_the_rendered_prefix():
    """Turn 2 replays turn 1's history (image included) and appends the
    model's answer plus a new user message. The rendered turn-1 prompt, minus
    its trailing generation prompt, must be a prefix of turn 2's, so the
    prefix cache hits on everything before the new turn."""
    engine = _TextOnlyEngine()
    client = _client(engine)
    turn1 = _goose_read_image_request()
    assert client.post("/v1/chat/completions", json=turn1).status_code == 200
    turn2 = json.loads(json.dumps(turn1))
    turn2["messages"] += [
        {"role": "assistant", "content": "I cannot see the image."},
        {"role": "user", "content": "Then read the DOM instead."},
    ]
    assert client.post("/v1/chat/completions", json=turn2).status_code == 200
    first, second = engine.prompts
    generation_prompt = "<|im_start|>assistant\n"
    assert first.endswith(generation_prompt + "<think>\n\n</think>\n\n")
    history = first[: first.rindex(generation_prompt)]
    assert _NOTICE_PNG in history
    assert second.startswith(history)
    assert text_only_image_stats() == {"images_replaced": 4, "requests": 2}


def test_status_reports_the_counter():
    engine = _TextOnlyEngine()
    client = _client(engine)
    before = client.get("/v1/status").json()
    assert before["text_only_images_replaced"] == 0
    assert before["text_only_image"]["policy"] == "notice"
    assert before["text_only_image"]["serving_text_only"] is True
    client.post("/v1/chat/completions", json=_goose_read_image_request())
    client.post("/v1/chat/completions", json=_chat_image_request())
    after = client.get("/v1/status").json()
    assert after["text_only_images_replaced"] == 3
    assert after["text_only_image"]["images_replaced"] == 3
    assert after["text_only_image"]["requests"] == 2


def test_one_warning_per_request_with_the_count(caplog):
    engine = _TextOnlyEngine()
    with caplog.at_level("WARNING", logger="rapid_mlx.api.text_only_images"):
        _client(engine).post("/v1/chat/completions", json=_goose_read_image_request())
    records = [r for r in caplog.records if r.name == "rapid_mlx.api.text_only_images"]
    assert len(records) == 1
    assert "replaced 2 image part(s)" in records[0].getMessage()


# ---------------------------------------------------------------------------
# /v1/responses
# ---------------------------------------------------------------------------


def _responses_body(input_items) -> dict:
    return {"model": "text-only-model", "max_output_tokens": 8, "input": input_items}


def test_responses_input_image_returns_200_and_prompt_has_notice():
    engine = _TextOnlyEngine()
    resp = _client(engine).post(
        "/v1/responses",
        json=_responses_body(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "what is on screen?"},
                        {"type": "input_image", "image_url": _PNG_DATA_URL},
                    ],
                }
            ]
        ),
    )
    assert resp.status_code == 200, resp.text
    prompt = engine.prompts[-1]
    assert "what is on screen?\n" + _NOTICE_PNG in prompt
    assert text_only_image_stats() == {"images_replaced": 1, "requests": 1}


def test_responses_function_call_output_image_becomes_notice_not_base64():
    engine = _TextOnlyEngine()
    resp = _client(engine).post(
        "/v1/responses",
        json=_responses_body(
            [
                {"type": "message", "role": "user", "content": "Check the UI."},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read_image",
                    "arguments": '{"path": "/tmp/shot.png"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [
                        {"type": "input_text", "text": "Read /tmp/shot.png."},
                        {"type": "input_image", "image_url": _PNG_DATA_URL},
                    ],
                },
            ]
        ),
    )
    assert resp.status_code == 200, resp.text
    prompt = engine.prompts[-1]
    assert _PNG_B64[:40] not in prompt
    assert "Read /tmp/shot.png." in prompt
    assert "[image not shown: this model is serving text-only" in prompt
    assert text_only_image_stats() == {"images_replaced": 1, "requests": 1}


def test_responses_function_call_output_dict_image_becomes_notice_not_base64():
    """``output`` may be a single dict (``str | dict | list``); the adapter
    JSON-encodes it verbatim, so an image dict would land as base64."""
    engine = _TextOnlyEngine()
    resp = _client(engine).post(
        "/v1/responses",
        json=_responses_body(
            [
                {"type": "message", "role": "user", "content": "Check the UI."},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read_image",
                    "arguments": '{"path": "/tmp/shot.png"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": {"type": "input_image", "image_url": _PNG_DATA_URL},
                },
            ]
        ),
    )
    assert resp.status_code == 200, resp.text
    prompt = engine.prompts[-1]
    assert _PNG_B64[:40] not in prompt
    tool_block = prompt.split("<tool_response>", 1)[1].split("</tool_response>", 1)[0]
    assert tool_block.strip() == _NOTICE_PNG
    assert text_only_image_stats() == {"images_replaced": 1, "requests": 1}


def test_responses_function_call_output_non_image_dict_untouched():
    engine = _TextOnlyEngine()
    resp = _client(engine).post(
        "/v1/responses",
        json=_responses_body(
            [
                {"type": "message", "role": "user", "content": "Run it."},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "run",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": {"exit_code": 0, "stdout": "ok"},
                },
            ]
        ),
    )
    assert resp.status_code == 200, resp.text
    assert '"exit_code": 0' in engine.prompts[-1]
    assert text_only_image_stats() == {"images_replaced": 0, "requests": 0}


def test_responses_reject_policy_keeps_the_400():
    engine = _TextOnlyEngine()
    resp = _client(engine, policy="reject").post(
        "/v1/responses",
        json=_responses_body(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": _PNG_DATA_URL}],
                }
            ]
        ),
    )
    assert resp.status_code == 400, resp.text
    assert "image_input_unsupported" in resp.text
    assert engine.chat_calls == []


# ---------------------------------------------------------------------------
# /v1/messages (Anthropic)
# ---------------------------------------------------------------------------


def _anthropic_image(data: str = _PNG_B64) -> dict:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data},
    }


def test_anthropic_image_block_returns_200_and_prompt_has_notice():
    engine = _TextOnlyEngine()
    resp = _client(engine).post(
        "/v1/messages",
        json={
            "model": "text-only-model",
            "max_tokens": 8,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        _anthropic_image(),
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    prompt = engine.prompts[-1]
    assert "what is this?\n" + _NOTICE_PNG in prompt
    assert text_only_image_stats() == {"images_replaced": 1, "requests": 1}


def test_anthropic_tool_result_image_keeps_text_and_gains_notice():
    engine = _TextOnlyEngine()
    resp = _client(engine).post(
        "/v1/messages",
        json={
            "model": "text-only-model",
            "max_tokens": 8,
            "tools": [
                {
                    "name": "read_image",
                    "description": "Read an image file",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                }
            ],
            "messages": [
                {"role": "user", "content": "Check the UI."},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "read_image",
                            "input": {"path": "/tmp/shot.png"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [
                                {"type": "text", "text": "Read /tmp/shot.png."},
                                _anthropic_image(),
                            ],
                        }
                    ],
                },
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    prompt = engine.prompts[-1]
    tool_block = prompt.split("<tool_response>", 1)[1].split("</tool_response>", 1)[0]
    assert "Read /tmp/shot.png." in tool_block
    assert _NOTICE_PNG in tool_block
    assert text_only_image_stats() == {"images_replaced": 1, "requests": 1}


def test_anthropic_reject_policy_keeps_the_400():
    engine = _TextOnlyEngine()
    resp = _client(engine, policy="reject").post(
        "/v1/messages",
        json={
            "model": "text-only-model",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": [_anthropic_image()]}],
        },
    )
    assert resp.status_code == 400, resp.text
    assert "does not support image inputs" in resp.text
    assert engine.chat_calls == []


def test_anthropic_document_block_still_rejected_under_notice():
    engine = _TextOnlyEngine()
    resp = _client(engine).post(
        "/v1/messages",
        json={
            "model": "text-only-model",
            "max_tokens": 8,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "JVBERi0=",
                            },
                        }
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 400, resp.text
    assert "does not support document inputs" in resp.text


def test_anthropic_vision_engine_is_unaffected():
    engine = _VisionEngine()
    resp = _client(engine).post(
        "/v1/messages",
        json={
            "model": "text-only-model",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": [_anthropic_image()]}],
        },
    )
    assert resp.status_code == 200, resp.text
    assert len(engine.prompts) == 1
    assert "image not shown" not in engine.prompts[0]
    assert text_only_image_stats() == {"images_replaced": 0, "requests": 0}


# ---------------------------------------------------------------------------
# The tunable
# ---------------------------------------------------------------------------


def test_default_policy_is_notice():
    from rapid_mlx.config.server_config import ServerConfig

    assert ServerConfig().text_only_image_policy == "notice"


@pytest.mark.parametrize(
    ("argv", "expected"),
    [([], "notice"), (["--text-only-image-policy", "reject"], "reject")],
)
def test_cli_flag_parses_and_reaches_the_config(argv, expected):
    import sys
    from unittest import mock

    from rapid_mlx import cli, server

    captured = {}
    with (
        mock.patch.object(cli, "serve_command", lambda a: captured.setdefault("a", a)),
        mock.patch.object(
            sys, "argv", ["rapid-mlx", "serve", "qwen3.5-4b-4bit", *argv]
        ),
    ):
        cli.main()
    assert captured["a"].text_only_image_policy == expected

    saved = server._text_only_image_policy
    try:
        server._text_only_image_policy = captured["a"].text_only_image_policy
        server._sync_config()
        assert get_config().text_only_image_policy == expected
    finally:
        server._text_only_image_policy = saved
        server._sync_config()


def test_cli_flag_rejects_unknown_policy():
    import sys
    from unittest import mock

    from rapid_mlx import cli

    with (
        mock.patch.object(cli, "serve_command", lambda a: None),
        mock.patch.object(
            sys,
            "argv",
            ["rapid-mlx", "serve", "m", "--text-only-image-policy", "drop"],
        ),
        pytest.raises(SystemExit),
    ):
        cli.main()


# ---------------------------------------------------------------------------
# DFlash / native-MTP server: it never 400ed, it dropped the image unseen
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy", ["notice", "reject"])
def test_dflash_server_says_the_image_was_not_shown(policy):
    from unittest.mock import MagicMock

    from rapid_mlx.speculative.dflash.runtime import DFlashRuntime
    from rapid_mlx.speculative.dflash.server import _build_app

    reset_config().text_only_image_policy = policy
    seen: list = []

    def _record(processor, model, request, enable_thinking=None):
        seen.append(request.messages)
        raise RuntimeError("stop after render")

    app = _build_app(
        model=MagicMock(),
        processor=MagicMock(),
        runtime=DFlashRuntime(
            drafter=MagicMock(),
            kind="dflash",
            drafter_repo="z-lab/Qwen3.5-27B-DFlash",
        ),
        served_model_name="text-only-model",
        default_max_tokens=8,
        cors_origins=[],
        render_prompt_fn=_record,
    )
    TestClient(app, raise_server_exceptions=False).post(
        "/v1/chat/completions", json=_chat_image_request()
    )
    assert len(seen) == 1
    parts = [
        p.model_dump(exclude_none=True) if hasattr(p, "model_dump") else p
        for p in seen[0][0].content
    ]
    if policy == "notice":
        assert parts[1] == {"type": "text", "text": _NOTICE_PNG}
        assert text_only_image_stats()["images_replaced"] == 1
    else:
        # ``reject`` leaves this server's behaviour as it was: the image
        # part reaches the renderer, which drops it with a WARN log.
        assert parts[1]["type"] == "image_url"
        assert text_only_image_stats()["images_replaced"] == 0


def test_dflash_render_joins_text_and_notice_with_a_newline(monkeypatch):
    from unittest.mock import MagicMock

    import mlx_vlm.prompt_utils

    from rapid_mlx.api.models import ChatCompletionRequest
    from rapid_mlx.api.text_only_images import replace_images_in_openai_messages
    from rapid_mlx.speculative.dflash.server import _render_prompt

    seen: list = []
    monkeypatch.setattr(
        mlx_vlm.prompt_utils,
        "apply_chat_template",
        lambda processor, config, messages, **kw: seen.append(messages) or "p",
    )
    request = ChatCompletionRequest(**_chat_image_request())
    assert replace_images_in_openai_messages(request.messages) == 1
    _render_prompt(MagicMock(), MagicMock(), request)
    assert seen[0][0]["content"] == "what does the screenshot show?\n" + _NOTICE_PNG
