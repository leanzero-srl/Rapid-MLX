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
