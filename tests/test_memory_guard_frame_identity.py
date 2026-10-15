# SPDX-License-Identifier: Apache-2.0
"""A conversation WITHOUT a session id is one line from its first turn.

RC6 interaction review (2026-10-15): with no session id (any client but
goose), turn 1 got its own line ``('frame', 304)`` and every later turn
``('frame', 306)`` -- one conversation, two lines, on both Qwen3.5 and
Qwen3.8, with or without reasoning in the history. ``first_end`` (where the
first message ends) was the common prefix of two renders followed by
different assistant probe rows; Qwen opens an assistant row with an empty
think block, so the two probes agreed past the first message's end -- past
the end of turn 1's whole prompt, whose generation prompt opens the think
block only. ``_memguard_line`` clamped turn 1 to its prompt length.

Rendered here with the real checked-in templates (the engine's own render
path) and a character tokenizer, so the test needs no model download.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import types

import jinja2
import jinja2.sandbox
import pytest

from rapid_mlx.api.models import ChatCompletionRequest
from rapid_mlx.api.utils import extract_multimodal_content
from rapid_mlx.engine.batched import BatchedEngine
from rapid_mlx.scheduler import Scheduler

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    }
]


class _Template:
    """A checked-in chat_template.jinja, rendered with the Jinja environment
    transformers builds; characters as tokens."""

    def __init__(self, path: pathlib.Path):
        self.chat_template = path.read_text()
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )
        env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False, **kw)

        def raise_exception(message):
            raise jinja2.exceptions.TemplateError(message)

        env.globals["raise_exception"] = raise_exception
        self._template = env.from_string(self.chat_template)

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        kwargs.setdefault("add_generation_prompt", True)
        return self._template.render(messages=messages, **kwargs)

    def encode(self, text, **_kwargs):
        return [ord(c) for c in text]


class _Stub:
    async def generate(self, *, prompt, sampling_params, **kwargs):
        self.prompt, self.kwargs = prompt, kwargs
        return types.SimpleNamespace(
            output_text="x",
            text="x",
            new_text="x",
            prompt_tokens=0,
            completion_tokens=0,
            cached_tokens=0,
            finish_reason="stop",
            usage=None,
            logprobs=None,
            token_logprobs=None,
            tokens=[],
            new_token_ids=[],
            output_token_ids=[],
            finished=True,
        )


def _engine(template: str) -> BatchedEngine:
    engine = BatchedEngine("test-model")
    engine._loaded = True
    engine._is_mllm = False
    engine._engine = _Stub()
    engine._tokenizer = _Template(FIXTURES / template)
    engine._is_hybrid_model = lambda: True
    return engine


def _call(i: int) -> dict:
    return {
        "id": f"call_{i}",
        "type": "function",
        "function": {"name": "write", "arguments": json.dumps({"path": f"f{i}.py"})},
    }


def _round(i: int, narration: str, reasoning: str) -> list[dict]:
    """goose's shape: narration row, tool-call row, tool result."""
    return [
        {"role": "assistant", "content": narration, "reasoning_content": reasoning},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": reasoning,
            "tool_calls": [_call(i)],
        },
        {"role": "tool", "tool_call_id": f"call_{i}", "content": f"wrote f{i}.py"},
    ]


BASE = [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "Build the app."},
]
TURNS = [
    BASE,
    BASE + _round(1, "Now the ledger module:", "I will write the ledger first."),
    BASE
    + _round(1, "Now the ledger module:", "I will write the ledger first.")
    + _round(2, "Now the tests:", "Now the tests."),
    BASE
    + _round(1, "Now the ledger module:", "I will write the ledger first.")
    + _round(2, "Now the tests:", "Now the tests.")
    + [
        {"role": "assistant", "content": "Done.", "reasoning_content": "Summarise."},
        {"role": "user", "content": "Add a README."},
    ],
]


def _lines(template: str, session: str | None, reasoning: bool) -> list:
    engine = _engine(template)
    lines = []
    for messages in TURNS:
        if not reasoning:
            messages = [
                {k: v for k, v in m.items() if k != "reasoning_content"}
                for m in messages
            ]
        request = ChatCompletionRequest(model="m", messages=messages, tools=TOOLS)
        msgs, _, _ = extract_multimodal_content(
            request.messages, preserve_native_format=True
        )
        asyncio.run(engine.chat(messages=msgs, tools=TOOLS, session_id=session))
        prompt = engine._engine.prompt
        tokens = prompt if isinstance(prompt, list) else [ord(c) for c in prompt]
        hint = engine._engine.kwargs["session_hint"]
        request = types.SimpleNamespace(session_hint=hint, prompt_token_ids=tokens)
        lines.append(Scheduler._memguard_line(None, request))
    return lines


@pytest.mark.parametrize(
    "template", ["qwen38_chat_template.jinja", "qwen35_chat_template.jinja"]
)
@pytest.mark.parametrize("reasoning", [True, False])
def test_a_conversation_without_a_session_id_is_one_line_from_turn_one(
    template, reasoning
):
    lines = _lines(template, None, reasoning)
    assert len(set(lines)) == 1, [line[0] for line in lines]
    # control: with a session id it always was one line
    assert len(set(_lines(template, "sess-1", reasoning))) == 1
