# SPDX-License-Identifier: Apache-2.0
"""Memory guard E3 (cache side): the operator's prefix budget is an upper
bound; the guard sets the budget in force from the device's real headroom."""

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

MB = 1024**2


def _entry(tokens: int) -> list:
    kv = KVCache()
    # 2 x (1, 8, tokens, 128) fp32 = 8 KiB per token
    kv.update_and_fetch(
        mx.zeros((1, 8, tokens, 128), dtype=mx.float32),
        mx.zeros((1, 8, tokens, 128), dtype=mx.float32),
    )
    mx.eval(kv.keys, kv.values)
    return [kv]


def _cache() -> MemoryAwarePrefixCache:
    cache = MemoryAwarePrefixCache(
        model=MagicMock(), config=MemoryCacheConfig(max_memory_mb=64)
    )
    for i in range(4):  # 4 entries x 8 MiB, oldest first
        assert cache.store([1000 * i + t for t in range(1024)], _entry(1024))
    return cache


def test_lowering_the_budget_evicts_oldest_first_down_to_it():
    cache = _cache()
    assert len(cache) == 4
    evicted = cache.set_max_memory(20 * MB)
    assert evicted == 2
    assert cache._current_memory <= 20 * MB
    assert [k[0] for k in cache._entries] == [2000, 3000]
    stats = cache.get_stats()
    assert stats["effective_max_memory_mb"] == 20.0
    assert stats["configured_max_memory_mb"] == 64.0
    assert stats["max_memory_mb"] == 20.0


def test_the_configured_budget_is_an_upper_bound():
    cache = _cache()
    assert cache.set_max_memory(10**12) == 0
    assert cache._max_memory == 64 * MB


def test_negative_control_a_budget_above_usage_evicts_nothing():
    cache = _cache()
    assert cache.set_max_memory(40 * MB) == 0
    assert len(cache) == 4
