# SPDX-License-Identifier: Apache-2.0
"""Memory guard E4: one turn stores one owned copy, and every copy is logged.

goose's text lane (Qwen3.5-27B Q8, 64 GB M3 Max, 2026-10-14) measured
~3.7 GB per 22.5k-token prefix entry and up to three owned copies per turn:
the message-boundary snapshot, the prompt save and the prompt+output save.
The prompt entry of a non-trimmable (hybrid) cache that sits within 64
tokens of a stored message boundary is superseded by that boundary entry
(upstream 1394f1696, ported rule only) and is no longer stored.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import ArraysCache, KVCache  # noqa: E402

from rapid_mlx import memory_cache as mc  # noqa: E402
from rapid_mlx.memory_cache import (  # noqa: E402
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
    entry_bytes_by_component,
    estimate_kv_cache_memory,
)
from rapid_mlx.request import Request, SamplingParams  # noqa: E402
from rapid_mlx.scheduler import Scheduler, SchedulerConfig  # noqa: E402

MB = 1024**2


class _MockArray:
    def __init__(self, nbytes: int):
        self.nbytes = nbytes


class _KVLayer:
    """Stands in for KVCache (trimmable attention layer)."""

    def __init__(self, nbytes: int):
        self.keys = _MockArray(nbytes // 2)
        self.values = _MockArray(nbytes // 2)
        self.offset = 0

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        return n


class _RecurrentLayer(_KVLayer):
    """Stands in for ArraysCache (GatedDeltaNet state, not trimmable)."""

    def is_trimmable(self) -> bool:
        return False


def _hybrid_cache(total_bytes: int) -> list:
    per_layer = total_bytes // 4
    return [_KVLayer(per_layer)] + [_RecurrentLayer(per_layer) for _ in range(3)]


def _scheduler(monkeypatch) -> Scheduler:
    monkeypatch.delenv(mc.PREFIX_CACHE_MAX_BYTES_ENV, raising=False)
    config = SchedulerConfig(
        enable_prefix_cache=True,
        use_memory_aware_cache=True,
        use_paged_cache=False,
        hybrid_cache_entries=8,
    )
    tokenizer = MagicMock()
    tokenizer.encode = lambda s: list(range(len(s)))
    return Scheduler(model=MagicMock(), tokenizer=tokenizer, config=config)


def _register(sched: Scheduler, uid: int, prompt_len: int) -> Request:
    req = Request(
        request_id=f"req-{uid}",
        prompt="x",
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=16),
    )
    sched.requests[req.request_id] = req
    sched.uid_to_request_id[uid] = req.request_id
    sched.request_id_to_uid[req.request_id] = uid
    return req


def test_hybrid_prompt_entry_skipped_after_a_stored_boundary(monkeypatch):
    sched = _scheduler(monkeypatch)
    req = _register(sched, 7, 40)
    req.prefix_boundary = 30
    req._cache_snapshot_stored = True
    sched.memory_aware_cache.store = MagicMock(return_value=True)

    sched._prompt_cache_save_cb(7, _hybrid_cache(4 * MB))

    sched.memory_aware_cache.store.assert_not_called()


def test_hybrid_prompt_entry_kept_when_no_boundary_was_stored(monkeypatch):
    # Negative control: no stored boundary entry -> the prompt entry is the
    # only copy and must be stored.
    sched = _scheduler(monkeypatch)
    req = _register(sched, 8, 40)
    req.prefix_boundary = 30
    sched.memory_aware_cache.store = MagicMock(return_value=True)
    layers = _hybrid_cache(4 * MB)

    sched._prompt_cache_save_cb(8, layers)

    sched.memory_aware_cache.store.assert_called_once_with(
        list(range(40)), layers, evict_prefixes=False
    )


def test_trimmable_prompt_entry_kept_after_a_stored_boundary(monkeypatch):
    # Negative control: a trimmable cache can trim-one into an exact hit, so
    # the N-token entry stays useful.
    sched = _scheduler(monkeypatch)
    req = _register(sched, 9, 40)
    req.prefix_boundary = 30
    req._cache_snapshot_stored = True
    sched.memory_aware_cache.store = MagicMock(return_value=True)
    layers = [_KVLayer(4 * MB)]

    sched._prompt_cache_save_cb(9, layers)

    sched.memory_aware_cache.store.assert_called_once_with(
        list(range(40)), layers, evict_prefixes=False
    )


def test_hybrid_prompt_entry_kept_when_the_boundary_is_far_back(monkeypatch):
    # Negative control: the gap (300 tokens) is above the 64-token bound.
    sched = _scheduler(monkeypatch)
    req = _register(sched, 10, 400)
    req.prefix_boundary = 100
    req._cache_snapshot_stored = True
    sched.memory_aware_cache.store = MagicMock(return_value=True)
    layers = _hybrid_cache(4 * MB)

    sched._prompt_cache_save_cb(10, layers)

    sched.memory_aware_cache.store.assert_called_once_with(
        list(range(400)), layers, evict_prefixes=False
    )


def test_gap_of_exactly_64_tokens_is_superseded_and_65_is_not(monkeypatch):
    sched = _scheduler(monkeypatch)
    layers = _hybrid_cache(4 * MB)
    req = _register(sched, 12, 164)
    req.prefix_boundary = 100
    req._cache_snapshot_stored = True
    assert sched._boundary_snapshot_supersedes(req, layers, 164) is True
    assert sched._boundary_snapshot_supersedes(req, layers, 165) is False


def test_internal_snapshot_is_not_a_message_boundary(monkeypatch):
    sched = _scheduler(monkeypatch)
    req = _register(sched, 11, 40)
    req._cache_snapshot_boundary = 39
    req._cache_snapshot_is_internal = True
    req._cache_snapshot_stored = True
    assert sched._boundary_snapshot_supersedes(req, _hybrid_cache(4 * MB), 40) is False


def _real_hybrid_cache(tokens: int) -> list:
    kv = KVCache()
    kv.update_and_fetch(
        mx.zeros((1, 2, tokens, 64), dtype=mx.bfloat16),
        mx.zeros((1, 2, tokens, 64), dtype=mx.bfloat16),
    )
    rec = ArraysCache(size=2)
    rec[0] = mx.zeros((1, 3, 128), dtype=mx.bfloat16)
    rec[1] = mx.zeros((1, 4, 32, 32), dtype=mx.float32)
    mx.eval(kv.keys, kv.values, rec[0], rec[1])
    return [kv, rec]


def test_entry_bytes_split_sums_to_the_charged_total():
    cache = _real_hybrid_cache(100)
    parts = entry_bytes_by_component(cache)
    assert parts["total"] == estimate_kv_cache_memory(cache)
    assert parts["kv"] + parts["recurrent"] + parts["checkpoints"] == parts["total"]
    assert parts["recurrent"] == 3 * 128 * 2 + 4 * 32 * 32 * 4
    assert parts["kv"] == 2 * (2 * 100 * 64 * 2)


def test_store_logs_one_byte_line_per_entry(caplog):
    cache = MemoryAwarePrefixCache(
        model=MagicMock(),
        config=MemoryCacheConfig(max_memory_mb=64, hybrid_reuse_max_entries=4),
    )
    caplog.set_level(logging.INFO, logger="rapid_mlx.memory_cache")
    assert cache.store(list(range(100)), _real_hybrid_cache(100))
    lines = [
        r.getMessage()
        for r in caplog.records
        if "[cache_store_bytes]" in r.getMessage()
    ]
    assert len(lines) == 1
    assert "tokens=100" in lines[0] and "recurrent=" in lines[0]
    assert "kv=" in lines[0] and "charged=" in lines[0] and "entries=1" in lines[0]
