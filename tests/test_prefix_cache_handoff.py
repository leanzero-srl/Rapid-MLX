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

