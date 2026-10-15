# SPDX-License-Identifier: Apache-2.0
"""A conversation's newest prefix entry is the guard's LAST resort.

Gabee T12 control (2026-10-15, engine f39d1d92 = the b06 guard, Qwen3.8-27B
Q8 + MTP, ceiling 50.10 GB, Metal ~55.7 GB): at ~96k the conversation's ONLY
entry -- its 95,813-token message-boundary snapshot, 6.72 GB as charged --
was evicted (``evicted prefix entry tokens=95813 level=4 (E1)
entries_left=0``, later ``tokens=97120 level=4 (pressure)``), and every
later turn re-prefilled ~98k tokens: 657 s to the first token.

The rules these tests pin:

1. An eviction is credited only with the bytes it ACTUALLY frees: an entry
   whose buffers a live cache, a queued loan or another entry still holds
   frees none and is never taken for room.
2. A cheaper action comes first: a narrower MTP verify (it holds S - 1
   restore points of every recurrent layer: 4.77 GB at 32 rows on the 27B),
   a smaller chunk, deferring a concurrent request.
3. A conversation's newest entry (EVICT_LAST) goes last, and only when it
   frees real bytes.
4. With room (128 GB) nothing changes: no narrowing, no eviction.

The numbers are the control's own (20:08:52Z): 38.88 GB resident with no
entry, the entry 6.72 GB, a 1,376-token side entry 0.25 GB, the KV ~6 GB at
95k (5,988 MB in the store line).
"""

from __future__ import annotations

import weakref
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from rapid_mlx.memory_guard import (
    EVICT_HEAD,
    EVICT_LAST,
    EVICT_NEEDED,
    EVICT_REDUNDANT,
    GuardProbe,
    MemoryGuard,
    PeakPrior,
    RowPlan,
)
from rapid_mlx.request import Request, SamplingParams
from rapid_mlx.scheduler import _buffer_address, _SharedBuffer, _write_mark
from tests.test_memory_guard_mtp_decode import (
    _Batch,
    _decoding,
    _guard,
    _model_27b,
)
from tests.test_memory_guard_session import _no_metal, cpu  # noqa: F401

GB = 10**9
CEILING = int(50.10 * GB)
BIG_CEILING = int(0.9 * 115.4 * GB)
LIVE = int(38.88 * GB)
ENTRY = int(6.72 * GB)
SIDE = int(0.25 * GB)
KV_95K = int(5.99 * GB)
PROMPT, BOUNDARY, FETCHED = 95_842, 95_813, 94_245
PER_ROW = 153_944_064  # 48 x (48 x 128 x 128 x 4 + 3 x 10,240 x 2)
MARGIN = int(0.02 * CEILING)


def _scheduler(ceiling: int, *, live: int = LIVE):
    from tests.test_prompt_cache_snapshot import _make_scheduler_with_cache

    s = _make_scheduler_with_cache()
    s.config.spec_decode = "mtp"
    s.config.mtp_max_k = 3
    s.model = _model_27b()  # copy draft of 31: a verify of up to 32 rows
    s._memory_guard = _guard()
    s._memguard_ceiling = lambda: ceiling
    request = Request(
        request_id="main",
        prompt="ignored",
        prompt_token_ids=list(range(PROMPT)),
        sampling_params=SamplingParams(max_tokens=4),
    )
    request.cached_tokens = FETCHED
    s.running = {"main": request}
    s.requests["main"] = request
    _decoding(s, capacity=PROMPT + 200, idx=PROMPT, context=PROMPT)
    mac = s.memory_aware_cache

    def active() -> int:
        return live + sum(
            e.memory_bytes
            for e in mac._entries.values()
            if not getattr(e, "_test_shared", False)
        )

    s._current_metal_active_bytes = active
    s._current_metal_cache_bytes = lambda: 0
    return s


def _entry(s, tokens: list[int], charge: int):
    layer = KVCache()
    layer.keys = mx.zeros((1, 1, 4, 8))
    layer.values = mx.zeros((1, 1, 4, 8))
    layer.offset = 4
    mx.eval(layer.keys, layer.values)
    assert s.memory_aware_cache.store(tokens, [layer], evict_prefixes=False)
    entry = s.memory_aware_cache._entries[tuple(tokens)]
    entry.memory_bytes = charge
    return tuple(tokens)


def _side(s):
    return _entry(s, [10**6 + i for i in range(1376)], SIDE)


def _widest(free: int) -> int:
    return 1 + (free - MARGIN) // PER_ROW


def _shared(s, key, entry) -> dict:
    """The entry's K and V share the decoding layer's buffers (a store from
    a row with no slack): KV_95K allocated at that layer's next write."""
    live = s.batch_generator._generation_batch.prompt_cache[0]
    return {
        _buffer_address(getattr(entry.cache[0], slot)): _SharedBuffer(
            nbytes=KV_95K // 2,
            layer=weakref.ref(live),
            slot=slot,
            array=weakref.ref(getattr(live, slot)),
            mark=_write_mark(live),
            entries=[(key, weakref.ref(entry))],
        )
        for slot in ("keys", "values")
    }


def _plan(s):
    assert s._memguard_plan_dispatch([]) == (None, [])
    return getattr(s.batch_generator, "_lz_verify_rows_cap", None)


# -- the control, replayed --------------------------------------------------


def test_the_decode_after_the_boundary_store_narrows_the_verify_not_the_entry(cpu):
    """20:08:52 then the turn's decode: 38.88 GB + the 6.72 GB entry (+ the
    side's 0.25 GB) + a 4.77 GB verify + margin = 51.6 GB > 50.10. The b06
    guard evicted the entry; now the verify narrows and the entry stays."""
    s = _scheduler(CEILING)
    main = _entry(s, list(range(BOUNDARY)), ENTRY)
    side = _side(s)
    full = s._memguard_decode_growth(s._memguard())
    assert full == pytest.approx(4.77 * GB, rel=0.01)
    cap = _plan(s)
    entries = s.memory_aware_cache._entries
    assert main in entries
    resident = LIVE + ENTRY + (SIDE if side in entries else 0)
    assert cap == _widest(CEILING - resident)
    assert 4 <= cap < 32
    assert resident + (cap - 1) * PER_ROW + MARGIN <= CEILING
    counts = s._memguard().counts
    assert counts["verify_narrowed"] == 1
    assert counts["newest_evictions"] == 0 and counts["decode_over_ceiling"] == 0


def test_a_shared_entry_keeps_its_room_by_a_narrower_verify(cpu):
    """Right after a mid-turn store that shares the live buffers, the next
    write re-allocates the 6 GB of K/V (planned): the verify narrows around
    it and the entry stays. Its eviction is credited with exactly that
    allocation, which it would cancel -- and only as the last resort."""
    s = _scheduler(CEILING)
    main = _entry(s, list(range(BOUNDARY)), ENTRY)
    entry = s.memory_aware_cache._entries[main]
    entry._test_shared = True
    s._memguard_shared = _shared(s, main, entry)
    assert s._metal_footprint_bytes() == LIVE + KV_95K
    cap = _plan(s)
    assert main in s.memory_aware_cache._entries
    assert cap == _widest(CEILING - LIVE - KV_95K)
    # no room even for the floor verify: the last resort, and it is real
    # (the re-allocation it cancels), so the step fits instead of crossing
    tight = _scheduler(CEILING, live=CEILING - KV_95K - MARGIN)
    main = _entry(tight, list(range(BOUNDARY)), ENTRY)
    entry = tight.memory_aware_cache._entries[main]
    entry._test_shared = True
    tight._memguard_shared = _shared(tight, main, entry)
    _plan(tight)
    assert main not in tight.memory_aware_cache._entries
    counts = tight._memguard().counts
    assert counts["newest_evictions"] == 1 and counts["decode_over_ceiling"] == 0
    assert tight._metal_footprint_bytes() == CEILING - KV_95K - MARGIN


def test_an_entry_a_loan_still_holds_frees_nothing_and_stays(cpu):
    """A queued request's loan holds the same array objects: evicting the
    entry would free nothing now -- never credited, never taken."""
    s = _scheduler(CEILING, live=CEILING - MARGIN)
    main = _entry(s, list(range(BOUNDARY)), ENTRY)
    entry = s.memory_aware_cache._entries[main]
    loan = Request(
        request_id="next",
        prompt="ignored",
        prompt_token_ids=list(range(BOUNDARY + 50)),
        sampling_params=SamplingParams(max_tokens=4),
    )
    loan.prompt_cache = list(entry.cache)
    s.waiting.append(loan)
    with s.memory_aware_cache._lock:
        assert s._memguard_freeable_bytes()[main] == 0
    assert s._memguard_evictable_bytes(EVICT_LAST) == 0
    _plan(s)
    assert main in s.memory_aware_cache._entries
    assert s._memguard().counts["newest_evictions"] == 0


def test_the_newest_entry_goes_last_and_only_for_real_bytes(cpu):
    """No verify width fits: the side entry, then an older entry this turn
    stored (NEEDED), and only then the conversation's newest (LAST)."""
    # the floor verify (0.46 GB) is 1.2 GB short with all three entries:
    # the side's and the older entry's 0.75 GB are not enough
    live = CEILING - ENTRY - SIDE - int(0.5 * GB) - MARGIN + int(0.74 * GB)
    s = _scheduler(CEILING, live=live)
    main = _entry(s, list(range(BOUNDARY)), ENTRY)
    older = _entry(s, list(range(95_000)), int(0.5 * GB))
    side = _side(s)
    with s.memory_aware_cache._lock:
        levels = {k: lv for k, lv, _c in s._memguard_ranked_entries()}
    assert levels[main] == EVICT_LAST
    assert levels[older] == EVICT_NEEDED
    order = []
    real = s.memory_aware_cache._evict_entry_locked

    def spy(key, reason):
        order.append(key)
        return real(key, reason)

    s.memory_aware_cache._evict_entry_locked = spy
    real_lru = s.memory_aware_cache._evict_lru

    def spy_lru():
        order.append(next(iter(s.memory_aware_cache._entries)))
        return real_lru()

    s.memory_aware_cache._evict_lru = spy_lru
    cap = _plan(s)
    assert order[-1] == main and set(order) == {side, older, main}
    assert order.index(main) > order.index(older)
    assert s._memguard().counts["newest_evictions"] == 1
    assert cap is None or cap >= 4  # widened again once the room was real


def test_the_128_gb_arm_takes_no_action(cpu):
    s = _scheduler(BIG_CEILING)
    main = _entry(s, list(range(BOUNDARY)), ENTRY)
    side = _side(s)
    assert _plan(s) is None
    assert main in s.memory_aware_cache._entries and side in s.memory_aware_cache._entries
    counts = s._memguard().counts
    assert counts["verify_narrowed"] == 0 and counts["newest_evictions"] == 0
    assert counts["decode_over_ceiling"] == 0


def test_the_verify_is_widened_back_when_room_returns(cpu):
    s = _scheduler(CEILING)
    main = _entry(s, list(range(BOUNDARY)), ENTRY)
    assert _plan(s) is not None
    with s.memory_aware_cache._lock:
        s.memory_aware_cache._evict_entry_locked(main, reason="test")
    assert _plan(s) is None


def test_measured_pressure_does_not_count_bytes_only_planned(cpu):
    """The pressure evictor reacts to MEASURED memory: shared bytes a later
    write will allocate never make it evict (it would free nothing)."""
    s = _scheduler(CEILING, live=CEILING - KV_95K // 2)
    main = _entry(s, list(range(BOUNDARY)), ENTRY)
    entry = s.memory_aware_cache._entries[main]
    entry._test_shared = True
    s._memguard_shared = _shared(s, main, entry)
    assert s._metal_footprint_bytes() >= CEILING
    assert s._metal_footprint_bytes(shared=False) < CEILING
    s._resolve_metal_cap_bytes = lambda: CEILING
    s._memguard_pressure_fraction = lambda: 1.0
    s._memguard_yield_to_system = lambda: None
    assert s.evict_prefix_cache_under_pressure() == 0
    assert main in s.memory_aware_cache._entries


# -- the plan (prefill) and joins -------------------------------------------


class _World:
    """A probe whose entries are ``(bytes freed, level)``, with peek."""

    def __init__(self, active: int, entries):
        self.active = active
        self.entries = list(entries)
        self.log: list[int] = []

    def probe(self) -> GuardProbe:
        def nxt(max_level):
            ranked = sorted(
                (lv, i) for i, (_b, lv) in enumerate(self.entries) if lv <= max_level
            )
            for lv, i in ranked:
                if self.entries[i][0] > 0:
                    return i
            return None

        def evict(max_level):
            i = nxt(max_level)
            if i is None:
                return False
            freed, level = self.entries.pop(i)
            self.active -= freed
            self.log.append(level)
            return True

        def peek(max_level):
            i = nxt(max_level)
            return None if i is None else self.entries[i]

        return GuardProbe(
            active=lambda: self.active,
            cache=lambda: 0,
            clear_cache=lambda: None,
            evict_one=evict,
            entry_count=lambda: len(self.entries),
            evictable=lambda lv: sum(b for b, level in self.entries if level <= lv),
            peek=peek,
        )


def _plain_guard() -> MemoryGuard:
    return MemoryGuard(
        PeakPrior(attention_bytes_per_unit=100.0, token_bytes=0.0, kv_copy_bytes_per_context=0),
        floor_tokens=256,
    )


ROWS = [
    RowPlan("main", offset=60_000, segment=20_000, prompt_tokens=80_000, arrival=1.0),
    RowPlan("side", offset=0, segment=20_000, prompt_tokens=20_000, arrival=2.0),
]


def test_a_concurrent_request_is_deferred_before_a_needed_entry_goes():
    """Two rows: the NEEDED entry frees less than the shortfall and
    deferring the latest arrival covers it -- defer, evict nothing."""
    guard = _plain_guard()
    world = _World(44 * GB, [(int(0.2 * GB), EVICT_NEEDED), (6 * GB, EVICT_LAST)])
    plan = guard.plan(ROWS, configured_chunk=2048, ceiling=48 * GB, probe=world.probe())
    assert plan.refuse is not None and plan.refuse.request_id == "side"
    assert world.log == []
    # an entry that alone covers the shortfall may go (rule 2 does not bar it)
    world = _World(44 * GB, [(1 * GB, EVICT_NEEDED), (6 * GB, EVICT_LAST)])
    plan = guard.plan(ROWS, configured_chunk=2048, ceiling=48 * GB, probe=world.probe())
    assert world.log == [EVICT_NEEDED] and plan.refuse is None


def test_alone_the_newest_entry_goes_after_the_needed_one():
    guard = _plain_guard()
    world = _World(int(46.5 * GB), [(6 * GB, EVICT_LAST), (int(0.2 * GB), EVICT_NEEDED)])
    plan = guard.plan(ROWS[:1], configured_chunk=2048, ceiling=48 * GB, probe=world.probe())
    assert world.log == [EVICT_NEEDED, EVICT_LAST]
    assert plan.refuse is None


def test_an_entry_that_frees_nothing_is_never_evicted():
    guard = _plain_guard()
    world = _World(44 * GB, [(0, EVICT_REDUNDANT), (0, EVICT_HEAD)])
    plan = guard.plan(ROWS[:1], configured_chunk=2048, ceiling=48 * GB, probe=world.probe())
    assert world.log == [] and len(world.entries) == 2
    assert plan.chunk < 2048 or plan.refuse is not None


def test_a_join_defers_rather_than_trading_a_needed_entry_for_less(cpu):
    s = _scheduler(CEILING)
    older = _entry(s, list(range(95_000)), int(0.5 * GB))
    main = _entry(s, list(range(BOUNDARY)), ENTRY)
    room = CEILING - (LIVE + ENTRY + int(0.5 * GB)) - MARGIN
    # 0.3 GB short: the joiner can be deferred, but the older entry alone
    # covers it (0.5 GB): it goes
    assert s._memguard_make_room(room + int(0.3 * GB), CEILING, EVICT_NEEDED, deferrable=True)
    assert older not in s.memory_aware_cache._entries
    assert main in s.memory_aware_cache._entries
    # 1 GB short with only the newest left beyond NEEDED: nothing is evicted
    # (the newest is EVICT_LAST; the joiner is deferred instead)
    room += int(0.5 * GB)
    assert not s._memguard_make_room(room + GB, CEILING, EVICT_NEEDED, deferrable=True)
    assert main in s.memory_aware_cache._entries


def test_a_join_defers_before_a_needed_entry_that_frees_less(cpu):
    s = _scheduler(CEILING)
    older = _entry(s, list(range(95_000)), int(0.5 * GB))
    other = _entry(s, list(range(94_500)), int(0.5 * GB))
    _entry(s, list(range(BOUNDARY)), ENTRY)
    room = CEILING - (LIVE + ENTRY + GB) - MARGIN
    # 0.8 GB short: both NEEDED entries together would cover it, each alone
    # frees less -- with a joiner to defer, neither goes
    assert not s._memguard_make_room(room + int(0.8 * GB), CEILING, EVICT_NEEDED, deferrable=True)
    assert older in s.memory_aware_cache._entries and other in s.memory_aware_cache._entries
    # nobody to defer: they go, cheapest first
    assert s._memguard_make_room(room + int(0.8 * GB), CEILING, EVICT_NEEDED)
    assert older not in s.memory_aware_cache._entries


def test_the_batch_helper_is_the_decoding_one():
    assert _Batch(uids=[1]).__len__() == 1
