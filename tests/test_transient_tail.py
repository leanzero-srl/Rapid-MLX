# SPDX-License-Identifier: Apache-2.0
"""LeanZero fork: the client-marked volatile tail (``rapid_mlx_transient_tail``).

An agent client (goose) ends every request with a block that changes on every
call — its clock, token count and turn budget — and drops it from the next
request, which instead appends the model's tool call and the tool result. On a
hybrid (non-trimmable) cache the upstream boundary snapshot is taken AFTER that
block, so it is never an exact prefix of the next request and reuse falls back
to the newest recurrent checkpoint. The marker moves only the snapshot boundary
to before the block; the rendered prompt is untouched.

The property pinned here is the one that pays: the boundary chosen for request N
is an exact prefix of request N+1's rendered prompt, in the merged shape (tail
appended to the user's own message), the separate-message shape (tail as its own
user turn after a tool result) and the tool shape (tail appended to the tool
result the request ends on).
"""

import asyncio
import logging

import pytest

pytestmark = pytest.mark.requires_mlx

from rapid_mlx.api.models import REQUEST_EXTENSIONS, ChatCompletionRequest, ModelInfo
from rapid_mlx.engine.batched import BatchedEngine

TAIL = "<turn-context>\n<current-time>14:07</current-time>\n</turn-context>"


class _CharacterTokenizer:
    def encode(self, text):
        return [ord(char) for char in text]


def _chatml(messages, tools=None, *, add_generation_prompt=True, **kwargs):
    body = "".join(
        f"<|im_start|>{m['role']}\n{m.get('content') or ''}<|im_end|>\n"
        for m in messages
    )
    return body + ("<|im_start|>assistant\n" if add_generation_prompt else "")


class _Stub:
    def __init__(self):
        self.last_generate_kwargs = None

    async def generate(self, *, prompt, sampling_params, **kwargs):
        self.last_generate_kwargs = kwargs
        return _Output()

    async def add_request(self, *, prompt, sampling_params, **kwargs):
        self.last_generate_kwargs = kwargs
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
    engine = BatchedEngine("test-model")
    engine._loaded = True
    engine._is_mllm = False
    engine._engine = _Stub()
    engine._tokenizer = _CharacterTokenizer()
    monkeypatch.setattr(engine, "_apply_chat_template", _chatml)
    monkeypatch.setattr(engine, "_is_hybrid_model", lambda: True)
    return engine


def _boundary(engine, messages, tail):
    stable = engine._stable_messages_before_transient_tail(messages, None, tail)
    kwargs = {} if stable is None else {"stable_messages": stable}
    return engine._compute_prefix_boundary(
        messages, generation_prompt=_chatml(messages), **kwargs
    )


SYSTEM = {"role": "system", "content": "You are an agent."}
TASK = "Find the request handler."
CALL = {
    "role": "assistant",
    "content": "",
    "tool_calls": [{"id": "c1", "type": "function"}],
}
RESULT = {"role": "tool", "tool_call_id": "c1", "content": "1  fn handle_request()"}


def test_merged_tail_boundary_is_a_prefix_of_the_next_request(monkeypatch):
    engine = _engine(monkeypatch)
    turn1 = [SYSTEM, {"role": "user", "content": f"{TASK}\n{TAIL}"}]
    turn2 = [
        SYSTEM,
        {"role": "user", "content": TASK},
        CALL,
        RESULT,
        {"role": "user", "content": TAIL},
    ]

    marked = _boundary(engine, turn1, "\n" + TAIL)
    upstream = _boundary(engine, turn1, None)

    next_prompt = _chatml(turn2)
    assert _chatml(turn1)[:marked] == next_prompt[:marked]
    assert "<turn-context>" not in _chatml(turn1)[:marked]
    # The upstream boundary includes the tail, so it is NOT reusable next turn.
    assert _chatml(turn1)[:upstream] != next_prompt[:upstream]


def test_separate_message_tail_boundary_is_a_prefix_of_the_next_request(monkeypatch):
    engine = _engine(monkeypatch)
    history = [SYSTEM, {"role": "user", "content": TASK}, CALL, RESULT]
    turn2 = [*history, {"role": "user", "content": TAIL}]
    turn3 = [
        *history,
        CALL,
        RESULT,
        {"role": "user", "content": TAIL.replace("14:07", "14:09")},
    ]

    marked = _boundary(engine, turn2, TAIL)
    upstream = _boundary(engine, turn2, None)

    assert _chatml(turn2)[:marked] == _chatml(turn3)[:marked]
    assert marked >= len(_chatml(history, add_generation_prompt=False)) - 8
    assert upstream > marked
    assert _chatml(turn2)[:upstream] != _chatml(turn3)[:upstream]


def test_a_tail_on_the_tool_results_boundary_is_a_prefix_of_the_next_request(
    monkeypatch,
):
    """goose's chat shape (Q-94): the block is appended to the tool result the
    request ends on. The boundary for this step must be an exact prefix of the
    next step, which carries the same result without the block, the model's next
    call, and a new result with a new block — and the tool message is never
    dropped, even when the block was all it held."""
    engine = _engine(monkeypatch)
    history = [SYSTEM, {"role": "user", "content": TASK}, CALL]
    step = [*history, {**RESULT, "content": f"{RESULT['content']}\n{TAIL}"}]
    after = [
        *history,
        RESULT,
        CALL,
        {**RESULT, "content": f"2  fn route()\n{TAIL.replace('14:07', '14:09')}"},
    ]

    marked = _boundary(engine, step, "\n" + TAIL)
    upstream = _boundary(engine, step, None)

    assert _chatml(step)[:marked] == _chatml(after)[:marked]
    assert "<turn-context>" not in _chatml(step)[:marked]
    assert marked >= len(_chatml([*history, RESULT], add_generation_prompt=False)) - 20
    assert _chatml(step)[:upstream] != _chatml(after)[:upstream]

    only_block = [*history, {**RESULT, "content": TAIL}]
    assert BatchedEngine._stable_messages_before_transient_tail(
        only_block, None, TAIL
    ) == [*history, {**RESULT, "content": ""}]


def test_the_stable_prefix_drops_or_strips_exactly_the_tail():
    whole = [SYSTEM, {"role": "user", "content": TAIL}]
    assert BatchedEngine._stable_messages_before_transient_tail(whole, None, TAIL) == [
        SYSTEM
    ]

    merged = [SYSTEM, {"role": "user", "content": f"{TASK}\n{TAIL}", "name": "u"}]
    assert BatchedEngine._stable_messages_before_transient_tail(
        merged, None, "\n" + TAIL
    ) == [
        SYSTEM,
        {"role": "user", "content": TASK, "name": "u"},
    ]
    assert merged[1]["content"] == f"{TASK}\n{TAIL}", (
        "the submitted messages are never mutated"
    )


def test_the_tail_stops_at_an_earlier_server_priming_boundary():
    messages = [
        SYSTEM,
        {"role": "user", "content": f"{TASK}\n{TAIL}"},
        {"role": "system", "content": "server-only reminder"},
    ]
    assert BatchedEngine._stable_messages_before_transient_tail(
        messages, 2, "\n" + TAIL
    ) == [
        SYSTEM,
        {"role": "user", "content": TASK},
    ]


@pytest.mark.parametrize(
    "messages",
    [
        [SYSTEM, {"role": "user", "content": "no block here"}],
        [SYSTEM, {"role": "user", "content": [{"type": "text", "text": TAIL}]}],
        [SYSTEM],
    ],
)
def test_a_tail_that_is_not_the_exact_suffix_is_ignored_loudly(messages, caplog):
    with caplog.at_level(logging.WARNING, logger="rapid_mlx.engine.batched"):
        assert (
            BatchedEngine._stable_messages_before_transient_tail(messages, None, TAIL)
            is None
        )
    assert "rapid_mlx_transient_tail ignored" in caplog.text


def test_no_tail_is_upstream():
    assert (
        BatchedEngine._stable_messages_before_transient_tail([SYSTEM], None, None)
        is None
    )
    assert (
        BatchedEngine._stable_messages_before_transient_tail([SYSTEM], None, "") is None
    )


@pytest.mark.parametrize("streaming", [False, True])
def test_both_paths_route_the_tail_and_never_forward_it(monkeypatch, streaming):
    engine = _engine(monkeypatch)
    seen = []

    def compute(messages, tools=None, **kwargs):
        seen.append(kwargs)
        return 5

    monkeypatch.setattr(engine, "_compute_prefix_boundary", compute)
    messages = [SYSTEM, {"role": "user", "content": f"{TASK}\n{TAIL}"}]

    async def run(**extra):
        if streaming:
            async for _ in engine.stream_chat(messages=messages, **extra):
                break
        else:
            await engine.chat(messages=messages, **extra)

    asyncio.run(run(transient_tail="\n" + TAIL))
    assert seen[-1]["stable_messages"] == [SYSTEM, {"role": "user", "content": TASK}]
    assert "transient_tail" not in engine._engine.last_generate_kwargs
    assert engine._engine.last_generate_kwargs["prefix_boundary"] == 5

    asyncio.run(run())
    assert "stable_messages" not in seen[-1], "absent marker must be the upstream call"


def test_the_request_field_parses_and_the_server_advertises_it():
    request = ChatCompletionRequest(
        model="m",
        messages=[{"role": "user", "content": f"{TASK}\n{TAIL}"}],
        rapid_mlx_transient_tail="\n" + TAIL,
    )
    assert request.rapid_mlx_transient_tail == "\n" + TAIL
    assert (
        ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "x"}]
        ).rapid_mlx_transient_tail
        is None
    )
    assert "rapid_mlx_transient_tail" in REQUEST_EXTENSIONS
    assert ModelInfo(id="m").model_dump()["request_extensions"] == [
        "rapid_mlx_transient_tail",
        "rapid_mlx_transient_tail_on_tool",
    ]
