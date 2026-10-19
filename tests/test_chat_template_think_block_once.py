# SPDX-License-Identifier: Apache-2.0
"""A template that renders the ``<think>`` block on EVERY assistant row, yet
carries the Qwen3.5 live literal and computes ``ns.last_query_index`` (it never
branches on it), must not get a second block from the tool-loop retain pass.

Live case (2026-10-19, laptop, abenzerps/Nex-N2.5-mini-MLX-8bit): every request
ending on a user row (goose's length-cut nudge, its compaction request) rendered
a doubled ``<think>\\n\\n</think>\\n\\n`` on every history tool-loop row, so it
shared only 5,526 of ~110k tokens with the tool round before it and re-prefilled
the conversation (211 s for the compaction request)."""

from __future__ import annotations

import pytest

from rapid_mlx.utils.chat_template import (
    _template_drops_think_from_tool_loop_history,
    apply_chat_template,
)

# Nex-N2.5-mini's assistant/tool rendering, reduced: last_query_index is
# computed but the block is rendered unconditionally.
_ALWAYS_BLOCK = r"""{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) -%}
{%- for message in messages[::-1] -%}
    {%- set index = (messages|length - 1) - loop.index0 -%}
    {%- if ns.multi_step_tool and message.role == "user" and not ((message.content|trim).startswith('<tool_response>') and (message.content|trim).endswith('</tool_response>')) -%}
        {%- set ns.multi_step_tool = false -%}
        {%- set ns.last_query_index = index -%}
    {%- endif -%}
{%- endfor -%}
{%- for message in messages -%}
    {%- set content = message.content|trim -%}
    {%- if message.role == "system" or message.role == "user" -%}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>\n' -}}
    {%- elif message.role == "assistant" -%}
        {%- set reasoning_content = '' -%}
        {%- if message.reasoning_content is string -%}
            {%- set reasoning_content = message.reasoning_content -%}
        {%- endif -%}
        {%- set reasoning_content = reasoning_content|trim -%}
        {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content -}}
        {%- for tool_call in message.tool_calls or [] -%}
            {{- '<tool_call>' + tool_call.function.name + '</tool_call>' -}}
        {%- endfor -%}
        {{- '<|im_end|>\n' -}}
    {%- elif message.role == "tool" -%}
        {{- '<|im_start|>user\n<tool_response>\n' + content + '\n</tool_response><|im_end|>\n' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' -}}
{%- endif -%}"""


class _JinjaApplicator:
    def __init__(self, template: str):
        self.chat_template = template

    def apply_chat_template(self, messages, **kwargs):
        jinja2 = pytest.importorskip("jinja2")
        env = jinja2.Environment(keep_trailing_newline=True)
        return env.from_string(self.chat_template).render(
            messages=messages,
            add_generation_prompt=kwargs.get("add_generation_prompt", True),
        )


def _call(name: str, idx: int) -> dict:
    return {
        "id": f"call_{idx}",
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


_TOOL_ROUND = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "do the task"},
    {"role": "assistant", "content": "plan one", "tool_calls": [_call("shell", 1)]},
    {"role": "tool", "tool_call_id": "call_1", "content": "out one"},
    {"role": "assistant", "content": "plan two", "tool_calls": [_call("tree", 2)]},
    {"role": "tool", "tool_call_id": "call_2", "content": "out two"},
]


def _render(messages):
    return apply_chat_template(
        _JinjaApplicator(_ALWAYS_BLOCK), messages, add_generation_prompt=False
    )


def test_the_retain_pass_is_off_for_a_template_that_keeps_history_blocks():
    # The live literal + ns.last_query_index are present (the literal check
    # alone armed the pass); the probe render shows history keeps its block.
    assert "ns.last_query_index" in _ALWAYS_BLOCK
    assert not _template_drops_think_from_tool_loop_history(_JinjaApplicator(_ALWAYS_BLOCK))


def test_the_retain_pass_stays_on_for_the_qwen35_live_only_template():
    # Control: the asymmetric Qwen3.5 branch still needs the pass.
    from tests.test_chat_template_tool_loop_think import _QWEN35_LIKE

    assert _template_drops_think_from_tool_loop_history(_JinjaApplicator(_QWEN35_LIKE))


def test_an_appended_user_row_keeps_the_tool_round_a_byte_prefix():
    tool_round = _render(_TOOL_ROUND)
    with_nudge = _render(
        _TOOL_ROUND + [{"role": "user", "content": "your reply was cut; continue"}]
    )
    assert with_nudge.startswith(tool_round)


def test_no_history_row_gets_a_second_think_block():
    with_nudge = _render(_TOOL_ROUND + [{"role": "user", "content": "summarize"}])
    assert "<think>\n\n</think>\n\n<think>" not in with_nudge
    assert with_nudge.count("<think>") == 2
