# SPDX-License-Identifier: Apache-2.0
"""A sliding-window conversation whose message-boundary COPY does not fit
keeps its resume point anyway, like a hybrid one does.

Gabee, 2026-10-18 (lz/memory-guard-rc13 e3a92b555, gemma-4-31B-it 8-bit,
64 GB, ceiling ~43 GB). Gemma 4 interleaves 50 sliding-window layers
(``RotatingKVCache``, window 1024) with 10 full-attention layers. Every turn
of a goose Forge build logged

    skipped storing a 4.42GB prefix entry: resident 39.88GB + entry + margin
    0.80GB > ceiling 40.01GB

and the next turn hit only an old 3,265-token boundary (prompt 30,826,
remaining 27,561): 12 % reuse against 98-99 % for the hybrid Qwen3.8-27B on the
same engine. The turn-end prompt+output entry cannot serve the next turn: the
next prompt re-renders the reply (it diverges right after the old prompt), and
a wrapped ``RotatingKVCache`` cannot be trimmed back past the reply
(``is_trimmable()`` is False once the ring has rotated), so LCP reuse is
refused. The hybrid fallback (keep what cannot be trimmed at the boundary,
cut the request's own KV back at the turn end) only accepted
``ArraysCache`` + ``KVCache`` layouts, so a sliding-window model kept
nothing.

The fix keeps the ring layers' state at the boundary (bounded by the window,
not by the context) and hands the full-attention KV over at the turn end.
These tests run the REAL scheduler and mlx-lm ``BatchGenerator`` on a tiny
gemma3_text model (layers 0/2 sliding window 16, 1/3 full attention) on the
CPU device, with a window far smaller than the boundary so the ring wraps.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import KVCache, RotatingKVCache  # noqa: E402

from rapid_mlx.request import Request, SamplingParams  # noqa: E402
from tests.test_lz_singleton_srpf import (  # noqa: E402,F401
    _no_metal,
    _run,
    _scheduler,
    cpu,
)

WINDOW = 16
BOUNDARY_1, PROMPT_1 = 48, 61
BOUNDARY_2, PROMPT_2 = 72, 83


@pytest.fixture(autouse=True)
def _calm_kernel(monkeypatch):
    import rapid_mlx.memory_budget as memory_budget

    monkeypatch.setattr(memory_budget, "system_memory_pressure", lambda: 0)


@pytest.fixture(scope="module")
def sliding4(cpu):
    """Layers 0/2 RotatingKVCache(max_size=16), 1/3 KVCache."""
    from mlx_lm.models.gemma3_text import Model, ModelArgs

    args = ModelArgs(
        model_type="gemma3_text",
        hidden_size=64,
        num_hidden_layers=4,
        intermediate_size=128,
        num_attention_heads=4,
        head_dim=16,
        num_key_value_heads=2,
        vocab_size=128,
        query_pre_attn_scalar=16,
        sliding_window=WINDOW,
        sliding_window_pattern=2,
        max_position_embeddings=4096,
    )
    mx.random.seed(0)
    model = Model(args)
    mx.eval(model.parameters())
    kinds = [type(c) for c in model.make_cache()]
    assert kinds == [RotatingKVCache, KVCache, RotatingKVCache, KVCache], kinds
    return model


def _turn(rid: str, prompt: list[int], boundary: int, max_tokens: int) -> Request:
    request = Request(
        request_id=rid,
        prompt=list(prompt),
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
    )
    request.prefix_boundary = boundary
    return request


def _conversation():
    turn_1 = [(7 * i + 3) % 120 + 1 for i in range(PROMPT_1)]
    # the next turn re-renders the reply: it shares turn 1's prompt only up
    # to the message boundary
    turn_2 = turn_1[:BOUNDARY_1] + [
        (11 * i + 5) % 120 + 1 for i in range(PROMPT_2 - BOUNDARY_1)
    ]
    return turn_1, turn_2


def _cached_scheduler(model, *, refuse_copies: bool):
    s = _scheduler(
        model,
        enable_prefix_cache=True,
        use_memory_aware_cache=True,
        hybrid_cache_entries=8,
    )
    seen = {"copies_refused": 0}
    original = s._memguard_room_for_store

    def room(request, cache, **kwargs):
        has_kv = any(
            isinstance(layer, KVCache) and (layer.offset or 0) > 0
            for layer in (cache or ())
        )
        if refuse_copies and has_kv and not kwargs.get("turn_end"):
            # the copy of the live KV does not fit beside it (Gabee 2026-10-18)
            seen["copies_refused"] += 1
            return False
        return original(request, cache, **kwargs)

    s._memguard_room_for_store = room
    return s, seen


def _two_turns(model, *, refuse_copies: bool, max_tokens: int = 12):
    turn_1, turn_2 = _conversation()
    s, seen = _cached_scheduler(model, refuse_copies=refuse_copies)
    _, tokens_1, _ = _run(s, {0: [_turn("t1", turn_1, BOUNDARY_1, max_tokens)]})
    _, tokens_2, cached_2 = _run(s, {0: [_turn("t2", turn_2, BOUNDARY_2, 8)]})
    return tokens_1["t1"], tokens_2["t2"], cached_2["t2"], seen, s


def test_a_copied_sliding_window_boundary_is_reused(sliding4):
    """Control: when the boundary COPY fits, turn 2 resumes from it."""
    _, _, cached_2, _, _ = _two_turns(sliding4, refuse_copies=False)
    assert cached_2 == BOUNDARY_1


def test_a_sliding_window_boundary_that_cannot_be_copied_is_handed_over(sliding4):
    """RED on 1c513d179 (and e3a92b555): the copy is refused, nothing is kept,
    the turn-end prompt+output entry needs a trim the wrapped ring refuses,
    and turn 2 re-prefills everything (cached 0). GREEN: turn 2 hits the
    48-token boundary and answers exactly as after a copied boundary and as
    after no cache at all."""
    copied_1, copied_2, _, _, _ = _two_turns(sliding4, refuse_copies=False)
    tokens_1, tokens_2, cached_2, seen, s = _two_turns(sliding4, refuse_copies=True)
    assert seen["copies_refused"] >= 1
    assert cached_2 == BOUNDARY_1, (cached_2, sorted(len(k) for k in s.memory_aware_cache._entries))
    assert tokens_1 == copied_1
    assert tokens_2 == copied_2
    cold_s, _ = _cached_scheduler(sliding4, refuse_copies=False)
    _, cold, _ = _run(cold_s, {0: [_turn("cold", _conversation()[1], 0, 8)]})
    assert tokens_2 == cold["cold"]


def test_the_handed_over_entry_is_the_freshly_prefilled_boundary(sliding4):
    """The entry against a fresh prefill of exactly the boundary tokens: the
    full-attention KV equal position by position, the ring layers equal to
    the ring a fresh prefill leaves (same offset, same window contents)."""
    from mlx_lm.models.cache import make_prompt_cache

    turn_1, _ = _conversation()
    s, _ = _cached_scheduler(sliding4, refuse_copies=True)
    _run(s, {0: [_turn("t1", turn_1, BOUNDARY_1, 12)]})
    key = tuple(turn_1[:BOUNDARY_1])
    assert key in s.memory_aware_cache._entries, sorted(
        len(k) for k in s.memory_aware_cache._entries
    )
    entry = s.memory_aware_cache._entries[key]
    fresh = make_prompt_cache(sliding4)
    for start in range(0, BOUNDARY_1, 8):
        sliding4(mx.array([turn_1[start : start + 8]]), cache=fresh)
        mx.eval([c.state for c in fresh])
    for got, want in zip(entry.cache, fresh, strict=True):
        assert type(got) is type(want)
        assert got.offset == want.offset == BOUNDARY_1
        # logits after the entry == logits after a fresh boundary prefill
    probe = mx.array([turn_1[BOUNDARY_1 : BOUNDARY_1 + 4]])
    import copy

    a = sliding4(probe, cache=copy.deepcopy(list(entry.cache)))
    b = sliding4(probe, cache=fresh)
    assert mx.allclose(a, b, rtol=1e-4, atol=1e-4).item()
