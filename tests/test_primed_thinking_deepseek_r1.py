# SPDX-License-Identifier: Apache-2.0
"""``deepseek_r1`` keeps a primed thought in the reasoning channel.

Qwen3.5/3.6/3.8 templates end the generation prompt with ``<think>\\n`` when
thinking is on, so the streamed output starts inside the thought with no
opening tag. The parser's no-tag threshold used to route everything past the
64th character to content: on the T12 bake-off (2026-10-15) a Qwen3.8 turn
streamed "All confirmed. Now let's write the files. Context usage is at 7" as
reasoning and "9% - should conserve..." plus ~10k tokens of its own thinking
as the visible answer."""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock

from rapid_mlx.reasoning.deepseek_r1_parser import DeepSeekR1ReasoningParser
from rapid_mlx.service.postprocessor import StreamingPostProcessor

_QWEN38 = (pathlib.Path(__file__).parent / "fixtures" / "qwen38_chat_template.jinja").read_text()

# Renders past reasoning in history but never opens a think block for the next
# turn: the template MENTIONS <think> without priming it.
_HISTORY_ONLY = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n"
    "{% if m.reasoning_content %}<think>\n{{ m.reasoning_content }}\n</think>\n{% endif %}"
    "{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

_THOUGHT = (
    "All confirmed. Now let's write the files. Context usage is at 7"
    "9% — should conserve: write the code in this turn and the next few turns, "
    "then do lint/build/test."
)


def _stream(template: str, text: str, enable_thinking=None, chunk=4):
    cfg = MagicMock()
    cfg.engine = MagicMock()
    cfg.engine.tokenizer.chat_template = template
    cfg.reasoning_parser = DeepSeekR1ReasoningParser()
    cfg.reasoning_parser_name = "deepseek_r1"
    cfg.enable_auto_tool_choice = False
    cfg.tool_call_parser = None
    cfg.tool_parser_instance = None
    processor = StreamingPostProcessor(cfg, enable_thinking=enable_thinking)
    processor.reset()
    reasoning, content = "", ""
    # Qwen3.8 emits ``</think>`` as one token, so it never splits across chunks.
    pieces = []
    for part in text.split("</think>"):
        pieces += [part[i : i + chunk] for i in range(0, len(part), chunk)]
        pieces.append("</think>")
    pieces.pop()
    events = []
    for index, piece in enumerate(pieces):
        output = MagicMock()
        output.new_text = piece
        output.finished = False
        output.channel = None
        output.finish_reason = None
        output.prompt_tokens = 10
        output.completion_tokens = index + 1
        output.tokens = []
        output.logprobs = None
        output.tool_calls = None
        events += processor.process_chunk(output)
    events += processor.finalize()
    for event in events:
        if event.type == "reasoning" and event.reasoning:
            reasoning += event.reasoning
        elif event.type == "content" and event.content:
            content += event.content
    return reasoning, content


def test_a_primed_qwen38_thought_stays_reasoning_until_it_closes():
    reasoning, content = _stream(_QWEN38, _THOUGHT + "</think>\n\nWriting the files now.")
    assert "Context usage is at 79% — should conserve" in reasoning
    assert "should conserve" not in content
    assert content.strip() == "Writing the files now."


def test_a_long_primed_thought_still_streaming_is_not_content():
    reasoning, content = _stream(_QWEN38, _THOUGHT * 20)
    assert content == ""
    assert reasoning.count("should conserve") == 20


def test_thinking_off_keeps_a_tagless_answer_as_content():
    answer = "The capital of France is Paris, and it has been since the tenth century."
    reasoning, content = _stream(_QWEN38, answer, enable_thinking=False)
    assert content == answer
    assert reasoning == ""


def test_a_template_that_only_renders_history_thoughts_does_not_prime():
    from rapid_mlx.service.helpers import _should_start_in_thinking

    # A source scan says "primed" (the template contains <think> and
    # add_generation_prompt); the render says the next turn opens no thought.
    assert _should_start_in_thinking(_HISTORY_ONLY, None)
    assert not _should_start_in_thinking(_HISTORY_ONLY, None, rendered=True)
    assert _should_start_in_thinking(_QWEN38, None, rendered=True)
    assert not _should_start_in_thinking(_QWEN38, False, rendered=True)
    answer = "The capital of France is Paris, and it has been since the tenth century."
    reasoning, content = _stream(_HISTORY_ONLY, answer)
    assert content  # not swallowed whole as a primed thought


def test_a_coder_named_checkpoint_is_not_primed_when_the_flag_is_unset():
    # apply_chat_template renders an unset flag as thinking OFF for a model
    # whose name contains "coder": the prompt ends with a CLOSED think block,
    # so a tagless answer must stay the answer.
    from rapid_mlx.service.helpers import _prompt_thinking_active_for

    parser = DeepSeekR1ReasoningParser()
    assert not _prompt_thinking_active_for(
        parser, _QWEN38, None, tools_requested=False, model_name="/m/Qwen3.8-27B-coder-sft"
    )
    assert _prompt_thinking_active_for(
        parser, _QWEN38, None, tools_requested=False, model_name="/m/Qwen3.8-27B"
    )
    assert _prompt_thinking_active_for(
        parser, _QWEN38, True, tools_requested=False, model_name="/m/Qwen3.8-27B-coder-sft"
    )


# --- route level: /v1/messages and /v1/responses configure the parser too ----

import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from rapid_mlx.config import reset_config  # noqa: E402

_ROUTE_THOUGHT = _THOUGHT + " " + "Now the manifest, then the ledger module. " * 4


class _Qwen38Tokenizer:
    chat_template = _QWEN38


class _PrimedQwen38Engine:
    """Streams a primed Qwen3.8 thought (no opening tag), the closing token,
    then a short answer."""

    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    supports_tool_calls = True
    tokenizer = _Qwen38Tokenizer()

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def stream_chat(self, messages, **kwargs):
        pieces = [_ROUTE_THOUGHT[i : i + 5] for i in range(0, len(_ROUTE_THOUGHT), 5)]
        pieces += ["</think>", "\n\nDone."]
        for index, piece in enumerate(pieces, start=1):
            yield SimpleNamespace(
                new_text=piece,
                text="".join(pieces[:index]),
                prompt_tokens=10,
                completion_tokens=index,
                finish_reason=None,
                finished=False,
                channel=None,
                tool_calls=None,
                tokens=[],
                logprobs=None,
            )
        yield SimpleNamespace(
            new_text="",
            text="".join(pieces),
            prompt_tokens=10,
            completion_tokens=len(pieces),
            finish_reason="stop",
            finished=True,
            channel=None,
            tool_calls=None,
            tokens=[],
            logprobs=None,
        )


@pytest.fixture
def _route_cfg():
    from rapid_mlx.middleware.auth import rate_limiter

    saved = rate_limiter.enabled
    rate_limiter.enabled = False
    cfg = reset_config()
    cfg.engine = _PrimedQwen38Engine()
    cfg.model_name = "Qwen3.8-27B"
    cfg.model_path = None
    cfg.reasoning_parser_name = "deepseek_r1"
    cfg.model_registry = None
    cfg.no_thinking = False
    cfg.tool_call_parser = None
    cfg.enable_auto_tool_choice = False
    yield cfg
    reset_config()
    rate_limiter.enabled = saved


def _sse(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


def test_the_messages_route_keeps_a_primed_thought_as_thinking(_route_cfg):
    from rapid_mlx.routes.anthropic import router

    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).post(
        "/v1/messages",
        json={
            "model": "Qwen3.8-27B",
            "max_tokens": 4000,
            "stream": True,
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "messages": [{"role": "user", "content": "Build it."}],
        },
    )
    assert response.status_code == 200, response.text
    deltas = [e.get("delta", {}) for e in _sse(response.text) if e.get("type") == "content_block_delta"]
    thinking = "".join(d.get("thinking", "") for d in deltas if d.get("type") == "thinking_delta")
    text = "".join(d.get("text", "") for d in deltas if d.get("type") == "text_delta")
    assert "79% — should conserve" in thinking
    assert "should conserve" not in text
    assert text.strip() == "Done."


def test_the_responses_route_keeps_a_primed_thought_as_reasoning(_route_cfg):
    from rapid_mlx.middleware.exception_handlers import install_exception_handlers
    from rapid_mlx.routes.responses import router

    app = FastAPI()
    install_exception_handlers(app)
    app.include_router(router)
    response = TestClient(app).post(
        "/v1/responses",
        json={
            "model": "Qwen3.8-27B",
            "stream": True,
            "max_output_tokens": 4000,
            "reasoning": {"effort": "medium"},
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Build it."}]}],
        },
    )
    assert response.status_code == 200, response.text
    events = _sse(response.text)
    reasoning = "".join(
        e.get("delta", "") for e in events if "reasoning" in str(e.get("type", "")) and isinstance(e.get("delta"), str)
    )
    text = "".join(
        e.get("delta", "") for e in events if e.get("type") == "response.output_text.delta"
    )
    assert "79% — should conserve" in reasoning
    assert "should conserve" not in text
    assert text.strip() == "Done."
