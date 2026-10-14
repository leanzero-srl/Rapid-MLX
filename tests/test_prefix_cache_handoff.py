# SPDX-License-Identifier: Apache-2.0
"""Memory guard: the end-of-turn entry takes the finished session over
layer by layer instead of holding two copies of it.

Gabee Q8 (30 GB weights, 50.1 GB ceiling), one growing session: from 87k
the end-of-turn store needed weights + the live session + a full copy and
was skipped (store_skips 1 -> 6); at 113k the store had 0 entries after the
turn and the next turn re-prefilled ~115k (~10 min). Measured here with
MLX's real allocator on the CPU device: storing a finished 16-layer cache
peaks at the session + ONE layer with the handoff, at 2x the session
without it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import KVCache  # noqa: E402

from rapid_mlx.memory_cache import (  # noqa: E402
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
)

LAYERS = 16
TOKENS = 3000
CAPACITY = 3072  # mlx-lm grows KV in 256-token steps
HEADS, DIM = 4, 64


@pytest.fixture(autouse=True)
def _cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _finished_request_cache() -> tuple[list, int]:
    """What mlx-lm hands over at finish for a B=1 row: per layer an
    offset-trimmed LAZY contiguous copy of the live buffer (the singleton
    fast path's ``_copy_exact_layer``); the live layers themselves are
    dropped by the batch."""
    live = []
    for _ in range(LAYERS):
        layer = KVCache()
        layer.keys = mx.random.normal((1, HEADS, CAPACITY, DIM))
        layer.values = mx.random.normal((1, HEADS, CAPACITY, DIM))
        layer.offset = TOKENS
        live.append(layer)
    mx.eval([c.keys for c in live] + [c.values for c in live])
    handed = []
    for layer in live:
        clone = KVCache()
        clone.keys = mx.contiguous(layer.keys[..., :TOKENS, :])
        clone.values = mx.contiguous(layer.values[..., :TOKENS, :])
        clone.offset = TOKENS
        handed.append(clone)
    layer_bytes = 2 * HEADS * TOKENS * DIM * 4
    return handed, layer_bytes


def _store_peak(handoff: bool) -> tuple[int, int, int]:
    mx.clear_cache()
    cache = MemoryAwarePrefixCache(
        model=MagicMock(), config=MemoryCacheConfig(max_memory_mb=4096)
    )
    handed, layer_bytes = _finished_request_cache()
    before = mx.get_active_memory()
    mx.reset_peak_memory()
    kwargs = {"handoff": True} if handoff else {}
    assert cache.store(list(range(TOKENS)), handed, evict_prefixes=False, **kwargs)
    del handed
    peak = mx.get_peak_memory() - before
    held = mx.get_active_memory() - before
    return peak, held, layer_bytes


def test_the_handoff_peaks_at_one_layer_beyond_the_session():
    """RED on 605548ef6 (no handoff): every layer's copy is evaluated while
    every live buffer is still alive."""
    peak, held, layer = _store_peak(handoff=True)
    # The live buffers (with their 256-step slack) die as their copies land.
    assert held <= LAYERS * layer * 0.02 + layer, (held, layer)
    assert peak <= 2.5 * layer, (peak, layer)


def test_negative_control_the_plain_store_holds_two_copies():
    peak, _held, layer = _store_peak(handoff=False)
    assert peak >= 0.9 * LAYERS * layer


def test_the_handed_over_entry_owns_exactly_its_charged_bytes():
    from rapid_mlx.memory_cache import estimate_kv_cache_memory

    cache = MemoryAwarePrefixCache(
        model=MagicMock(), config=MemoryCacheConfig(max_memory_mb=4096)
    )
    handed, layer = _finished_request_cache()
    assert cache.store(list(range(TOKENS)), handed, evict_prefixes=False, handoff=True)
    entry = next(iter(cache._entries.values()))
    assert entry.memory_bytes == estimate_kv_cache_memory(entry.cache) == LAYERS * layer
    for c in entry.cache:
        assert c.keys.shape[2] == TOKENS and c.offset == TOKENS


# -- the admission side: a loan whose entry was evicted is not copied -------


def _loaned_layer(tokens: int = 512) -> KVCache:
    layer = KVCache()
    layer.keys = mx.random.normal((1, HEADS, tokens, DIM))
    layer.values = mx.random.normal((1, HEADS, tokens, DIM))
    layer.offset = tokens
    mx.eval(layer.keys, layer.values)
    return layer


def test_a_handed_over_loan_is_admitted_without_its_copy_once():
    from rapid_mlx.scheduler import _handed_over, _mark_handed_over
    from rapid_mlx.singleton_cache_fastpath import (
        _detach_layer,
        install_singleton_cache_fastpath,
    )

    assert install_singleton_cache_fastpath()  # what the scheduler runs on
    loan = [_loaned_layer() for _ in range(3)]
    assert not _handed_over(loan)
    _mark_handed_over(loan)
    assert _handed_over(loan)
    mx.synchronize()
    before = mx.get_active_memory()
    admitted = [_detach_layer(layer) for layer in loan]
    mx.eval([a.keys for a in admitted])
    assert all(a is b for a, b in zip(admitted, loan, strict=True))
    assert mx.get_active_memory() - before < 4096  # no copy
    # One-shot: the flag is consumed at admission, so a later admission of
    # the same objects (or anything derived from them) copies again.
    assert not _handed_over(loan)
    again = _detach_layer(loan[0])
    assert again is not loan[0] and again.keys is not loan[0].keys


def test_a_loan_that_was_not_handed_over_is_still_copied():
    # Negative control: copy-on-admit protects a loan that shares array
    # objects with a stored entry (LCP / supersequence rewinds).
    from rapid_mlx.singleton_cache_fastpath import _detach_layer

    entry = _loaned_layer()
    loan = KVCache()
    loan.keys, loan.values, loan.offset = entry.keys, entry.values, 256
    admitted = _detach_layer(loan)
    assert admitted is not loan
    assert admitted.keys is not entry.keys


def test_the_sole_owner_flag_never_travels_with_a_copy():
    from rapid_mlx.scheduler import _handed_over, _mark_handed_over
    from rapid_mlx.singleton_cache_fastpath import _copy_exact_layer

    loan = [_loaned_layer()]
    _mark_handed_over(loan)
    clone = _copy_exact_layer(loan[0])  # extract / store / copy-on-admit body
    assert not _handed_over([clone])
    # No data layers at all (fresh request, wrappers only) is never "handed
    # over": the admission charge stays as computed.
    assert not _handed_over([])
    assert not _handed_over([KVCache()])
