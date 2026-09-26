# SPDX-License-Identifier: Apache-2.0
"""The pipeline server keeps a chat request's prefix up to goose's turn-context block
on the tool shape too (``rapid_mlx_transient_tail_on_tool``, goose Q-143).

In process, CPU, no ranks: the server's own HTTP app (``_build_app``) over a
word-level tokenizer with a role-marked chat template, and rank 0's own
prefix-cache decisions (``_PrefixIndex``). A consumer thread stands in for the
batch loop — it admits each row, stores the snapshot rank 0 asked for, and
finishes the request — so what is asserted is exactly what the server would
tell every rank to restore and snapshot. The restore itself (every rank, bit
for bit against a cold run) is ``test_pipeline_qwen4_serve``'s two-rank test.
"""

import json
import threading
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
pytest.importorskip("fastapi")
pytestmark = pytest.mark.requires_mlx

from fastapi.testclient import TestClient  # noqa: E402

SERVED = "tiny-pipeline"
ROLES = ("system", "user", "assistant", "tool")


def _word(i: int) -> str:
    """Token ``i``'s word. Letters only: the fast-tokenizer wrapper splits "w10"
    into "w1" + "0" (measured), so a digit vocabulary is not one word per id."""
    return "t" + chr(97 + i // 26) + chr(97 + i % 26)


def _words(ids) -> str:
    return " ".join(_word(i) for i in ids)


OPEN, CLOSE, EOS = 250, 251, 255
TEMPLATE = (
    "{% for m in messages %}"
    + _word(OPEN)
    + " {{ m['role'] }} {{ m['content'] }} "
    + _word(CLOSE)
    + " {% endfor %}"
    "{% if add_generation_prompt %}" + _word(OPEN) + " assistant {% endif %}"
)
SYSTEM_IDS = [3 + (i * 7) % 190 for i in range(120)]  # ids 3..192
# goose's per-call block is ~300 tokens: longer than the boundary's 8 replay
# tokens (``_PREFIX_BOUNDARY_REPLAY_TOKENS``). A block inside that margin would
# be missed by the unmarked boundary too, by accident.
BLOCK_BODY_IDS = list(range(220, 240))


def _block_ids(step: int) -> list[int]:
    """goose's turn-context block: different on every call (ids 200..239)."""
    return [200 + step, *BLOCK_BODY_IDS]


def _tokenizer_dir(path: Path) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {_word(i): i for i in range(256)}
    vocab.update({role: 256 + i for i, role in enumerate(ROLES)})
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token=_word(1)))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "eos_token": _word(EOS),
                "unk_token": _word(1),
                "chat_template": TEMPLATE,
            }
        )
    )
    return path


class _Kv:
    budgets = [1 << 40, 1 << 40]

    def reserve(self, lengths):
        return [len(lengths) * max(lengths)] * len(self.budgets)

    def entry_bytes(self, tokens):
        return [tokens, tokens]


class _Server:
    """The pipeline server's app with rank 0's prefix index and a stand-in batch loop."""

    def __init__(self, tokenizer_dir: Path, prefix_cache: bool = True):
        from mlx_lm.utils import load_tokenizer

        from rapid_mlx.distributed.pipeline_qwen4_serve import (
            _build_app,
            _PrefixIndex,
            _State,
        )

        tokenizer = load_tokenizer(tokenizer_dir)
        self.state = _State(
            served=SERVED,
            context=4096,
            max_batch=1,
            prefix=_PrefixIndex(_Kv()) if prefix_cache else None,
        )
        self.rows = []
        self.client = TestClient(
            _build_app(self.state, tokenizer, set(tokenizer.eos_token_ids))
        )
        self.loop = threading.Thread(target=self._batch_loop, daemon=True)
        self.loop.start()

    def _batch_loop(self):
        while (job := self.state.jobs.get()) is not None:
            row, index = job.row, self.state.prefix
            if index is not None:
                index.admit(row, [len(row.ids) + row.max_tokens + 1])
                if row.store_id:
                    key = tuple(row.ids[: row.store_at])
                    index.stored(row.store_id, key, index.kv.entry_bytes(row.store_at))
            self.rows.append(row)
            job.push(("done", "length"))

    def chat(self, messages, tail=None):
        body = {"model": SERVED, "messages": messages, "max_tokens": 1}
        if tail is not None:
            body["rapid_mlx_transient_tail"] = tail
        return self.client.post("/v1/chat/completions", json=body)

    def close(self):
        self.state.jobs.put(None)
        self.loop.join(timeout=10)


@pytest.fixture
def tokenizer_dir(tmp_path):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield _tokenizer_dir(tmp_path)
    mx.set_default_device(previous)


def _agent_turn(steps: int, block_on_tool: bool):
    """goose's requests for one agent turn of ``steps`` tool calls, and each one's tail.

    ``block_on_tool``: the block rides joined to the tool results the request
    ends on (an engine declaring ``_on_tool``); otherwise as a user turn of its
    own (the Q-94 shape goose sends an engine declaring only the tail).
    """
    ask = _words(range(10, 30))
    history = [
        {"role": "system", "content": _words(SYSTEM_IDS)},
        {"role": "user", "content": ask},
    ]
    first = _words(_block_ids(0))
    requests = [
        (
            [history[0], {"role": "user", "content": ask + "\n" + first}],
            "\n" + first,
        )
    ]
    for step in range(1, steps + 1):
        history += [
            {"role": "assistant", "content": _words([20 + step, 30 + step])},
            {"role": "tool", "content": _words([40 + step, 50 + step, 60 + step])},
        ]
        block = _words(_block_ids(step))
        if block_on_tool:
            last = history[-1]
            joined = {**last, "content": last["content"] + "\n" + block}
            requests.append(([*history[:-1], joined], "\n" + block))
        else:
            requests.append(([*history, {"role": "user", "content": block}], block))
    return requests


def _cached(response) -> int:
    assert response.status_code == 200, response.text
    return response.json()["usage"]["prompt_tokens_details"]["cached_tokens"]


def test_the_tokenizer_is_one_word_per_id(tokenizer_dir):
    from mlx_lm.utils import load_tokenizer

    ids = [*range(256), *(256 + i for i in range(len(ROLES)))]
    assert (
        load_tokenizer(tokenizer_dir).encode(_words(range(256)) + " " + " ".join(ROLES))
        == ids
    )


def test_models_declares_the_tail_on_tool_messages_only_with_a_prefix_cache(
    tokenizer_dir,
):
    server = _Server(tokenizer_dir)
    off = _Server(tokenizer_dir, prefix_cache=False)
    try:
        listed = server.client.get("/v1/models").json()["data"][0]
        assert listed["request_extensions"] == [
            "rapid_mlx_transient_tail",
            "rapid_mlx_transient_tail_on_tool",
        ]
        assert (
            off.client.get("/v1/models").json()["data"][0]["request_extensions"] == []
        )
    finally:
        server.close()
        off.close()


@pytest.mark.parametrize(
    "block_on_tool", [True, False], ids=["on_tool", "own_user_turn"]
)
def test_each_tool_step_restores_the_prefix_the_previous_step_stored(
    tokenizer_dir, block_on_tool
):
    server = _Server(tokenizer_dir)
    try:
        requests = _agent_turn(4, block_on_tool)
        cached = [_cached(server.chat(messages, tail)) for messages, tail in requests]
        rows = server.rows
        # <open> system <SYSTEM> <close>: what a snapshot at the system prompt holds.
        system_only = len(SYSTEM_IDS) + 3
        assert cached[0] == 0
        for step in range(1, len(requests)):
            before, row = rows[step - 1], rows[step]
            # The next request restores exactly what this one stored ...
            assert cached[step] == before.store_at == before.boundary > system_only
            assert row.ids[: before.store_at] == before.ids[: before.store_at]
            # ... which stops before the block the next request no longer carries:
            # the boundary's 8 replay tokens before the stable text ends — and, when
            # the block was a user turn of its own, before that turn's "<open> user"
            # (the next request's "<open>" follows, so the snapshot takes it too).
            start = before.ids.index(200 + step - 1)
            assert before.ids[start : start + len(BLOCK_BODY_IDS) + 1] == _block_ids(
                step - 1
            )
            own_turn = not block_on_tool and step > 1
            assert start - before.store_at == 8 + (1 if own_turn else 0)
        assert server.state.prefix.status()["hits"] == len(requests) - 1
    finally:
        server.close()


def test_without_the_tail_a_tool_step_restores_nothing(tokenizer_dir):
    """Negative control: the same requests with the block unnamed."""
    server = _Server(tokenizer_dir)
    try:
        cached = [
            _cached(server.chat(m)) for m, _ in _agent_turn(4, block_on_tool=True)
        ]
        assert cached == [0] * len(cached)
    finally:
        server.close()


def test_a_tail_that_is_not_a_string_is_refused_by_name(tokenizer_dir):
    server = _Server(tokenizer_dir)
    try:
        messages, _ = _agent_turn(0, block_on_tool=True)[0]
        refused = server.chat(messages, tail=[_word(200)])
        assert refused.status_code == 400
        assert "rapid_mlx_transient_tail must be a string" in refused.text
        assert server.rows == []
    finally:
        server.close()
