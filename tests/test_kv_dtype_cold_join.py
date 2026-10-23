# SPDX-License-Identifier: Apache-2.0
"""A batch's KV keeps the model's dtype when a cold request joins a row that
already holds KV.

Laptop, 2026-10-23 (M4 Max 128 GB, v0.14.3-lz.13, sabicap/sabi-noema-MLX-4bit,
dense Qwen3.5 27B 4-bit, Forge 2.0 through the app): decode ran at 4.0 tok/s
at ~47k context where the bandwidth arithmetic gives ~20. The stored prefix
entries tell why: every entry up to 03:47:10 held 65,536 B of KV per token
(16 full-attention layers x K,V x 4 KV heads x 256 x bf16), every entry after
it 131,072 (float32), and the recurrent state grew 146.8 -> 149.4 MB (the conv
state, which follows the activation dtype, went float32). At 03:47:10 goose's
second request (3,208 tokens) joined while the first (1,405) was mid-prefill
(``running=2``). mlx-lm 0.31.3 ``BatchKVCache.extend`` pads a side that holds
no KV yet with ``mx.array([])``, which is FLOAT32, and ``mx.concatenate``
promotes the whole batch's KV. The attention output, the residual stream and
every later layer then run in float32, ``merge`` carries the first row's dtype
into every restore, and each stored entry passes it on to the next turn: the
whole session decodes in float32 for good. HireTheRobots' Qwen3.8-27B 4-bit
(same app, 2026-10-22) shows the same flip at the same moment (2.9 tok/s at
170k). Ornith 122B-A10B stayed fast only because its two requests arrived in
the same instant and were prefilled together (no extend of an empty side).

The split (goose's tensor ranks) fixed this as Q-447 in its own wrapper; the
single engine never got it.
"""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import (  # noqa: E402
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    KVCache,
)

import rapid_mlx.scheduler  # noqa: E402,F401  (installs the guards)
from rapid_mlx.request import Request, SamplingParams  # noqa: E402
from tests.test_lz_singleton_srpf import (  # noqa: E402,F401
    _no_metal,
    _run,
    _scheduler,
    _tiny_args,
    cpu,
)

H, D = 2, 16


def _filled(cls, rows: int, length: int, dtype=mx.bfloat16, **kwargs):
    cache = cls(left_padding=[0] * rows, **kwargs) if kwargs else cls([0] * rows)
    k = mx.random.normal((rows, H, length, D)).astype(dtype)
    v = mx.random.normal((rows, H, length, D)).astype(dtype)
    cache.update_and_fetch(k, v)
    mx.eval(cache.keys, cache.values)
    return cache


def _empty(cls, rows: int, **kwargs):
    return cls(left_padding=[0] * rows, **kwargs) if kwargs else cls([0] * rows)


@pytest.mark.parametrize(
    "cls,kwargs",
    [(BatchKVCache, {}), (BatchRotatingKVCache, {"max_size": 64})],
    ids=["kv", "rotating"],
)
@pytest.mark.parametrize("filled_first", [True, False], ids=["filled.extend(cold)", "cold.extend(filled)"])
def test_extending_with_a_cold_row_keeps_the_dtype(cpu, cls, kwargs, filled_first):
    mx.random.seed(1)
    filled = _filled(cls, 1, 12, **kwargs)
    cold = _empty(cls, 1, **kwargs)
    reference_keys = mx.array(filled.keys)
    a, b = (filled, cold) if filled_first else (cold, filled)

    a.extend(b)

    assert a.keys.dtype == mx.bfloat16
    assert a.values.dtype == mx.bfloat16
    # same rows, padding and values as upstream: the filled row's KV is
    # unchanged and the cold row is all padding
    row = 0 if filled_first else 1
    live = reference_keys[..., : filled._idx if hasattr(filled, "_idx") else 12, :]
    got = a.keys[row : row + 1, :, a._idx - live.shape[2] : a._idx, :]
    assert mx.array_equal(got, live).item()
    assert int(a.left_padding[1 - row].item()) == a._idx


def _bf16_hybrid():
    """A tiny Qwen3.5 (GatedDeltaNet + attention) in bfloat16, no MTP head:
    the dense 27B's layer kinds, so ``max_num_seqs`` > 1 batches rows."""
    from mlx_lm.models.qwen3_5 import TextModel

    args = _tiny_args(2)
    object.__setattr__(args, "num_hidden_layers", 4)
    mx.random.seed(0)
    model = TextModel(args)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def bf16_hybrid(cpu):
    return _bf16_hybrid()


def _req(rid: str, tokens: list[int], max_tokens: int = 4) -> Request:
    return Request(
        request_id=rid,
        prompt=list(tokens),
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
    )


def _kv_dtypes(cache) -> set[str]:
    return {
        str(layer.keys.dtype).replace("mlx.core.", "")
        for layer in cache or ()
        if isinstance(layer, KVCache) and getattr(layer, "keys", None) is not None
    }


def _conv_dtypes(cache) -> set[str]:
    return {
        str(layer.cache[0].dtype).replace("mlx.core.", "")
        for layer in cache or ()
        if isinstance(layer, ArraysCache) and layer.cache[0] is not None
    }


def _entries(s):
    return {key: entry.cache for key, entry in s.memory_aware_cache._entries.items()}


def _cached_scheduler(model):
    return _scheduler(
        model,
        enable_prefix_cache=True,
        use_memory_aware_cache=True,
        hybrid_cache_entries=8,
        max_num_seqs=4,
    )


def test_a_cold_request_joining_a_row_mid_prefill_keeps_bfloat16(bf16_hybrid):
    """The laptop's 03:47:10 shape: the second request arrives while the
    first is still reading its prompt (CHUNK=8, so 60 tokens take 8 steps)."""
    s = _cached_scheduler(bf16_hybrid)
    first = [(5 * i + 2) % 120 + 1 for i in range(60)]
    second = [(9 * i + 4) % 120 + 1 for i in range(40)]
    _run(s, {0: [_req("main", first)], 2: [_req("side", second)]})

    entries = _entries(s)
    assert entries, "nothing was stored; the test no longer exercises the path"
    for key, cache in entries.items():
        assert _kv_dtypes(cache) == {"bfloat16"}, (len(key), _kv_dtypes(cache))
        assert _conv_dtypes(cache) <= {"bfloat16"}, (len(key), _conv_dtypes(cache))


def test_a_float32_entry_from_before_the_fix_is_not_resumed(bf16_hybrid, caplog):
    """An entry promoted before this fix (still in memory, or persisted to
    disk at shutdown and loaded at boot) would put float32 back into the
    batch on every turn that resumes from it. It is dropped and the prompt
    read again, once."""
    s = _cached_scheduler(bf16_hybrid)
    turn_1 = [(7 * i + 3) % 120 + 1 for i in range(40)]
    _run(s, {0: [_req("t1", turn_1)]})
    assert _entries(s)

    for cache in _entries(s).values():  # what lz.13 left behind
        for layer in cache:
            if isinstance(layer, KVCache) and layer.keys is not None:
                layer.keys = layer.keys.astype(mx.float32)
                layer.values = layer.values.astype(mx.float32)
            elif isinstance(layer, ArraysCache) and layer.cache[0] is not None:
                layer.cache[0] = layer.cache[0].astype(mx.float32)

    turn_2 = turn_1 + [(3 * i + 1) % 120 + 1 for i in range(16)]
    with caplog.at_level(logging.WARNING, logger="rapid_mlx.scheduler"):
        _, _, cached = _run(s, {0: [_req("t2", turn_2)]})

    assert cached["t2"] == 0
    assert any("float32" in r.getMessage() for r in caplog.records)
    for key, cache in _entries(s).items():
        assert _kv_dtypes(cache) == {"bfloat16"}, (len(key), _kv_dtypes(cache))
    assert s.kv_dtype_stats()["promoted_entries_dropped"] >= 1


def test_a_float32_model_keeps_its_float32_entries(cpu):
    """The drop is about a PROMOTION: a model that computes in float32 stores
    float32 KV and resumes from it as before."""
    from mlx_lm.models.qwen3_5 import TextModel

    args = _tiny_args(2)
    object.__setattr__(args, "num_hidden_layers", 4)
    mx.random.seed(0)
    model = TextModel(args)
    mx.eval(model.parameters())
    s = _cached_scheduler(model)
    turn_1 = [(7 * i + 3) % 120 + 1 for i in range(40)]
    _run(s, {0: [_req("t1", turn_1)]})
    turn_2 = turn_1 + [(3 * i + 1) % 120 + 1 for i in range(16)]
    _, _, cached = _run(s, {0: [_req("t2", turn_2)]})
    assert cached["t2"] > 0
    assert s.kv_dtype_stats()["promoted_entries_dropped"] == 0
