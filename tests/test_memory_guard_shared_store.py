# SPDX-License-Identifier: Apache-2.0
"""A prefix entry stored MID-TURN shares the live cache's buffers, and the
guard plans the bytes where they land.

``mx.contiguous`` does not copy an array that already is one whole
contiguous buffer (mlx 0.32.2: 0 bytes). A prompt-end or message-boundary
snapshot of a solo row (``BatchKVCache.merge`` sizes its buffer exactly; a
recurrent state is always whole) therefore stores an entry that owns no
bytes yet: the live layer's next write allocates a new buffer and the old
one stays with the entry. The bytes the store's room check made room for
land at that write, the first step after the store, which nothing planned
(the 1/100 sim: +60.5 MB at 7,100 tokens against 17.2 MB planned). RC6's
slack hid it; a fixed MTP verify cost planned beside it crossed the ceiling
(5 times on Gabee, 3 on 128 GB, with the cost planned exactly).

Two ways to make the plan true were measured:

* COPY at the store (62d0a0c3a): one memcpy of the session per turn and a
  higher peak at the next growth -- independent review on Metal, 27B at
  100k: 217-493 ms per store and 8.2-8.4 GB over the live cache, against
  2-12 ms and 6.12 GB shared; on this laptop's CPU at 3 GB of KV: 272 ms
  and 5.55 GB against 1 ms and 3.02 GB. Its copy decision read a
  process-wide memory counter, which an asynchronous step in flight moves
  (a second copy in 10 of 10 stores), and it decided for all layers at
  once (a full sliding-window ring made it copy every full-attention layer
  twice).
* SHARE and PLAN (shipped): the buffers stay shared, found per array by
  buffer address; each counts in the guard's footprint until the live
  layer writes (or the entry or the live layer is gone). Same crossings (0)
  and windows as the copy in the sim, no copy, the RC6 peak.
"""

from __future__ import annotations

import gc
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from rapid_mlx.memory_cache import (
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
    _trim_to_offset,
    own_cache_bytes,
)
from rapid_mlx.scheduler import Scheduler
from tests.test_memory_guard_session import _no_metal, cpu  # noqa: F401

MB = 2**20


def _kv_layer(tokens: int, slack: int = 0, heads: int = 4, dim: int = 64) -> KVCache:
    layer = KVCache()
    layer.keys = mx.random.normal((1, heads, tokens + slack, dim))
    layer.values = mx.random.normal((1, heads, tokens + slack, dim))
    layer.offset = tokens
    return layer


def _arrays(layers) -> list:
    out = []
    for c in layers:
        if isinstance(c, ArraysCache):
            out.extend(a for a in c.cache if a is not None)
        else:
            out.extend((c.keys, c.values))
    return out


def _nbytes(layers) -> int:
    return sum(int(a.nbytes) for a in _arrays(layers))


def _extract(live) -> list:
    """What mlx-lm's ``extract_cache`` hands over for a solo row: new layer
    objects over a contiguous slice up to the write position."""
    out = []
    for c in live:
        if isinstance(c, ArraysCache):
            e = ArraysCache(size=len(c.cache))
            e.cache = [mx.contiguous(a[0:1]) for a in c.cache]
        elif isinstance(c, RotatingKVCache):
            e = RotatingKVCache(max_size=c.max_size)
            e.keys, e.values = mx.contiguous(c.keys), mx.contiguous(c.values)
            e.offset, e._idx = c.offset, c._idx
        else:
            e = KVCache()
            e.keys = mx.contiguous(c.keys[0:1, :, 0 : c.offset, :])
            e.values = mx.contiguous(c.values[0:1, :, 0 : c.offset, :])
            e.offset = c.offset
        out.append(e)
    return out


def _cache(*, hybrid: bool = False) -> MemoryAwarePrefixCache:
    """``hybrid``: keep entries with a non-trimmable layer (a recurrent
    state, a ring), as the engine does when hybrid reuse is on."""
    return MemoryAwarePrefixCache(
        model=MagicMock(),
        config=MemoryCacheConfig(
            max_memory_mb=4096, hybrid_reuse_max_entries=4 if hybrid else 0
        ),
    )


def _scheduler(live, mac) -> Scheduler:
    s = object.__new__(Scheduler)
    s._memory_guard = object()
    s.memory_aware_cache = mac
    s.batch_generator = SimpleNamespace(
        _generation_batch=SimpleNamespace(prompt_cache=live), _prompt_batch=None
    )
    return s


def _store(s, mac, live, tokens: int) -> list[int]:
    key = list(range(tokens))
    assert mac.store(key, _extract(live), evict_prefixes=False)
    s._memguard_note_shared(key)
    return key


def _settle() -> None:
    gc.collect()
    mx.clear_cache()


# -- the mechanism ------------------------------------------------------------


def test_a_store_from_a_live_cache_with_no_slack_shares_and_is_planned(cpu):
    tokens = 2048
    live = [_kv_layer(tokens) for _ in range(8)]
    mx.eval(_arrays(live))
    kv = _nbytes(live)
    mac = _cache()
    s = _scheduler(live, mac)
    _settle()
    before = mx.get_active_memory()
    _store(s, mac, live, tokens)
    # nothing copied: the entry shares the live buffers ...
    assert mx.get_active_memory() - before < 0.01 * kv
    # ... and the guard counts them in the footprint
    assert s._memguard_shared_pending() == kv
    raw = s._current_metal_active_bytes() + s._current_metal_cache_bytes()
    assert s._metal_footprint_bytes() - raw == kv
    probe = s._memguard_probe() if hasattr(s, "_memguard_evict_one") else None
    if probe is not None:
        assert probe.active() - s._current_metal_active_bytes() == kv
    # the live cache's next write (a growth: no slack) allocates them
    _settle()
    before = mx.get_active_memory()
    for c in live:
        c.update_and_fetch(mx.zeros((1, 4, 1, 64)), mx.zeros((1, 4, 1, 64)))
    mx.eval(_arrays(live))
    allocated = mx.get_active_memory() - before
    assert allocated >= kv
    # ... and the charge ends there: the bytes are the entry's own now
    assert s._memguard_shared_pending() == 0
    assert s._memguard_shared_written == kv


def test_a_store_from_a_live_cache_with_slack_copies_and_charges_nothing(cpu):
    """A slice short of the buffer is copied at the store, which planned it:
    nothing is pending."""
    tokens = 2048
    live = [_kv_layer(tokens, slack=256) for _ in range(4)]
    mx.eval(_arrays(live))
    mac = _cache()
    s = _scheduler(live, mac)
    _settle()
    before = mx.get_active_memory()
    _store(s, mac, live, tokens)
    assert mx.get_active_memory() - before >= 0.99 * _nbytes(live) * tokens / (
        tokens + 256
    )
    assert s._memguard_shared_pending() == 0


def test_evicting_the_entry_ends_the_charge(cpu):
    live = [_kv_layer(1024) for _ in range(4)]
    mx.eval(_arrays(live))
    mac = _cache()
    s = _scheduler(live, mac)
    key = _store(s, mac, live, 1024)
    assert s._memguard_shared_pending() == _nbytes(live)
    with mac._lock:
        mac._evict_entry_locked(tuple(key), reason="test")
    assert s._memguard_shared_pending() == 0
    assert int(getattr(s, "_memguard_shared_written", 0) or 0) == 0


def test_the_live_layer_going_away_ends_the_charge_without_a_write(cpu):
    """The row finished: its buffers stay with the entry, nothing new is
    allocated."""
    live = [_kv_layer(1024) for _ in range(4)]
    mx.eval(_arrays(live))
    mac = _cache()
    s = _scheduler(live, mac)
    _store(s, mac, live, 1024)
    assert s._memguard_shared_pending() > 0
    s.batch_generator._generation_batch.prompt_cache = None
    del live
    gc.collect()
    assert s._memguard_shared_pending() == 0
    assert int(getattr(s, "_memguard_shared_written", 0) or 0) == 0


def test_a_ring_written_in_place_ends_the_charge(cpu):
    """A full sliding-window ring IS its buffer: the entry shares it, and
    the ring's next in-place write copies it (same array object, the write
    position moves)."""
    ring = RotatingKVCache(max_size=256)
    ring.keys = mx.random.normal((1, 4, 256, 64))
    ring.values = mx.random.normal((1, 4, 256, 64))
    ring.offset, ring._idx = 1000, 100  # past its window, mid-rotation
    mx.eval(ring.keys, ring.values)
    live = [ring]
    mac = _cache(hybrid=True)
    s = _scheduler(live, mac)
    key = list(range(ring.offset))
    assert mac.store(key, _extract(live), evict_prefixes=False)
    s._memguard_note_shared(key)
    shared = s._memguard_shared_pending()
    assert shared == ring.keys.nbytes + ring.values.nbytes
    _settle()
    before = mx.get_active_memory()
    keys_object = ring.keys
    ring.update_and_fetch(mx.zeros((1, 4, 1, 64)), mx.zeros((1, 4, 1, 64)))
    mx.eval(ring.keys, ring.values)
    assert ring.keys is keys_object  # written in place
    assert mx.get_active_memory() - before >= shared  # copy-on-write
    assert s._memguard_shared_pending() == 0
    assert s._memguard_shared_written == shared


def test_a_recurrent_state_update_ends_the_charge(cpu):
    layer = ArraysCache(size=2)
    layer[0] = mx.random.normal((1, 3, 256))
    layer[1] = mx.random.normal((1, 8, 64, 64))
    mx.eval(layer.cache)
    live = [layer]
    mac = _cache(hybrid=True)
    s = _scheduler(live, mac)
    _store(s, mac, live, 64)
    assert s._memguard_shared_pending() == _nbytes(live)
    layer[1] = layer[1] * 0.5
    assert s._memguard_shared_pending() == layer[0].nbytes
    layer[0] = layer[0] + 1
    assert s._memguard_shared_pending() == 0


def test_a_step_that_wrote_a_shared_buffer_is_not_learned_as_transient(cpu):
    """The write's allocation was planned through the footprint: the
    observation takes it out, so it is not counted twice."""
    seen = {}

    class _Guard:
        def observe(self, shape, transient, **kw):
            seen["transient"] = transient

    s = object.__new__(Scheduler)
    s._memory_guard = _Guard()
    s._memguard_idle_base_bytes = 0
    mx.reset_peak_memory()
    peak = int(mx.get_peak_memory())
    s._memguard_shared_written = 100 * MB
    plan = SimpleNamespace(shape=SimpleNamespace(batch=1), admit_bytes=0)
    s._memguard_observe_dispatch(plan, (peak - 300 * MB, 0, 0))
    assert seen["transient"] == pytest.approx(200 * MB, abs=8 * MB)
    assert s._memguard_shared_written == 0


# -- the review's reproducers (the copy variant's defects; and its cost) -------


def test_a_store_with_a_step_in_flight_copies_once(cpu):
    """Review reproducer race3.py: a 16-layer cache with slack (copied once)
    stored while another stream's step is in flight. The copy variant read
    a process-wide counter and copied again in 10 of 10 stores."""
    tokens, layers = 4096, 8
    live = [_kv_layer(tokens, slack=256, dim=128) for _ in range(layers)]
    mx.eval(_arrays(live))
    stream = mx.new_stream(mx.default_device())
    w = mx.random.normal((1024, 1024))
    mx.eval(w)
    entry_bytes = _nbytes(live) * tokens // (tokens + 256)
    for _ in range(4):
        _settle()
        with mx.stream(stream):
            h = mx.random.normal((8, 1024))
            for _ in range(64):
                h = mx.tanh(h @ w)
            mx.async_eval(h)
        extracted = []
        for c in live:
            e = KVCache()
            e.keys = mx.contiguous(c.keys[0:1, :, 0:tokens, :])
            e.values = mx.contiguous(c.values[0:1, :, 0:tokens, :])
            e.offset = tokens
            extracted.append(e)
        mx.reset_peak_memory()
        start = mx.get_active_memory()
        entry = own_cache_bytes(_trim_to_offset(extracted))
        assert mx.get_peak_memory() - start <= 1.02 * entry_bytes
        del entry, extracted
        mx.eval(h)
        del h


def test_a_sliding_window_ring_does_not_double_the_full_attention_copy(cpu):
    """Review reproducer mixed.py: full-attention layers with slack beside
    full sliding-window rings. The copy variant's all-or-nothing check saw
    the rings as shared and copied every full-attention layer twice."""
    tokens = 4096
    cache = []
    for i in range(8):
        if i % 2 == 0:
            cache.append(_kv_layer(tokens, slack=256))
        else:
            ring = RotatingKVCache(max_size=256)
            ring.keys = mx.random.normal((1, 4, 256, 64))
            ring.values = mx.random.normal((1, 4, 256, 64))
            ring.offset, ring._idx = tokens, 256
            cache.append(ring)
    mx.eval(_arrays(cache))
    full = sum(
        c.keys.nbytes + c.values.nbytes
        for c in cache
        if not isinstance(c, RotatingKVCache)
    )
    trimmed = full * tokens // (tokens + 256)
    _settle()
    mx.reset_peak_memory()
    start = mx.get_active_memory()
    own_cache_bytes(_trim_to_offset(cache))
    assert mx.get_peak_memory() - start <= 1.05 * trimmed


def test_a_shared_store_costs_no_copy_and_no_extra_peak(cpu):
    """Review reproducer cost.py: a solo row with no slack. Shared, the store
    allocates nothing and the next growth peaks at the new buffer (RC6);
    the copy variant allocated the session at the store and peaked at
    ~1.85x it at the growth."""
    tokens, layers = 8192, 8
    live = [_kv_layer(tokens, dim=128) for _ in range(layers)]
    mx.eval(_arrays(live))
    kv = _nbytes(live)
    _settle()
    mx.reset_peak_memory()
    start = mx.get_active_memory()
    entry = own_cache_bytes(_trim_to_offset(_extract(live)))
    assert mx.get_active_memory() - start < 0.01 * kv
    for c in live:
        block = mx.zeros((1, 4, 256, 128))
        c.keys = mx.concatenate([c.keys, block], axis=2)
        c.values = mx.concatenate([c.values, block], axis=2)
    mx.eval(_arrays(live))
    grown = kv * (tokens + 256) // tokens
    assert mx.get_peak_memory() - start <= 1.05 * grown
    del entry
