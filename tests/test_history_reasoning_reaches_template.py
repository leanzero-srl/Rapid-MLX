# SPDX-License-Identifier: Apache-2.0
"""An assistant turn's reasoning, sent back by the client on later turns,
reaches the chat template.

Qwen3.8's template defaults ``preserve_thinking`` on and renders every
history row as ``<think>\\n{reasoning_content}\\n</think>``. The request
model used to drop ``reasoning_content``, so a goose session's history showed
the model an empty think block on every past turn although goose sent the
reasoning back each time."""

from __future__ import annotations

import json
import pathlib

import jinja2
import jinja2.sandbox

from rapid_mlx.api.models import ChatCompletionRequest
from rapid_mlx.api.utils import extract_multimodal_content
from rapid_mlx.engine.batched import _normalize_tool_call_arguments_for_template
from rapid_mlx.utils.chat_template import apply_chat_template

_QWEN38_TEMPLATE = (
    pathlib.Path(__file__).parent / "fixtures" / "qwen38_chat_template.jinja"
)

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
        },
    }
]


class _Qwen38Applicator:
    """The stock Qwen3.8-27B ``chat_template.jinja`` (md5 519239a4…), rendered
    with the Jinja environment transformers builds."""

    def __init__(self):
        self.chat_template = _QWEN38_TEMPLATE.read_text()

        def raise_exception(message):
            raise jinja2.exceptions.TemplateError(message)

        env = jinja2.sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=["jinja2.ext.loopcontrols"],
        )
        env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False, **kw)
        env.globals["raise_exception"] = raise_exception
        self._template = env.from_string(self.chat_template)

    def apply_chat_template(self, messages, **kwargs):
        kwargs.setdefault("add_generation_prompt", True)
        return self._template.render(messages=messages, **kwargs)


def _goose_session(field: str = "reasoning_content", value="I will write the ledger first.") -> list[dict]:
    """The shape goose sends: an assistant tool-call row carrying its
    reasoning, the tool result, then a plain assistant row, then the next
    user turn."""
    return [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Build the app."},
        {
            "role": "assistant",
            "content": None,
            field: value,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "write",
                        "arguments": '{"path":"src/ledger.js","content":"x"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        {"role": "assistant", "content": "Ledger written.", field: "Now the summary."},
        {"role": "user", "content": "Continue."},
    ]


def _render(messages: list[dict], *, preserve_native_format: bool = True) -> str:
    request = ChatCompletionRequest(model="m", messages=messages, tools=_TOOLS)
    processed, _, _ = extract_multimodal_content(
        request.messages, preserve_native_format=preserve_native_format
    )
    return apply_chat_template(
        _Qwen38Applicator(),
        _normalize_tool_call_arguments_for_template(processed),
        tools=_TOOLS,
        enable_thinking=True,
    )


def test_history_reasoning_renders_inside_every_past_think_block():
    prompt = _render(_goose_session())
    assert "<think>\nI will write the ledger first.\n</think>" in prompt
    assert "<think>\nNow the summary.\n</think>" in prompt
    history = prompt.rsplit("<|im_start|>user\nContinue.", 1)[0]
    assert "<think>\n\n</think>" not in history


def test_the_litellm_reasoning_spelling_is_the_same_field():
    prompt = _render(_goose_session(field="reasoning"))
    assert "<think>\nI will write the ledger first.\n</think>" in prompt
    assert "<think>\nNow the summary.\n</think>" in prompt


def test_reasoning_survives_the_text_tool_call_path():
    prompt = _render(_goose_session(), preserve_native_format=False)
    assert "<think>\nI will write the ledger first.\n</think>" in prompt


def test_a_row_without_reasoning_still_renders_an_empty_block():
    messages = _goose_session()
    for message in messages:
        message.pop("reasoning_content", None)
    prompt = _render(messages)
    assert "I will write the ledger first." not in prompt
    assert prompt.count("<think>\n\n</think>") >= 2


def test_a_non_string_reasoning_is_absent_not_rendered():
    prompt = _render(_goose_session(field="reasoning", value={"summary": "SENTINEL"}))
    assert "SENTINEL" not in prompt


def test_the_request_model_keeps_reasoning_content():
    request = ChatCompletionRequest(model="m", messages=_goose_session())
    assert request.messages[2].reasoning_content == "I will write the ledger first."


def test_an_echoed_anthropic_thinking_block_reaches_the_template():
    from rapid_mlx.api.anthropic_adapter import anthropic_to_openai
    from rapid_mlx.api.anthropic_models import AnthropicRequest

    request = anthropic_to_openai(
        AnthropicRequest(
            model="m",
            max_tokens=64,
            messages=[
                {"role": "user", "content": "Build the app."},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Plan the ledger.", "signature": ""},
                        {"type": "text", "text": "Starting."},
                        {"type": "tool_use", "id": "t1", "name": "write", "input": {"path": "a"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
                },
            ],
        )
    )
    assistant = [m for m in request.messages if m.role == "assistant"]
    assert assistant[0].reasoning_content == "Plan the ledger."
    processed, _, _ = extract_multimodal_content(
        request.messages, preserve_native_format=True
    )
    prompt = apply_chat_template(
        _Qwen38Applicator(),
        _normalize_tool_call_arguments_for_template(processed),
        tools=_TOOLS,
        enable_thinking=True,
    )
    assert "<think>\nPlan the ledger.\n</think>" in prompt
