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


def test_an_entry_that_frees_nothing_is_skipped_for_one_that_does(cpu):
    """The running request's own loan still holds an older entry's arrays:
    evicting it would give nothing, so the next one in order goes and the
    held one stays (rule 1)."""
    live = CEILING - ENTRY - int(0.5 * GB) - MARGIN + int(0.6 * GB)
    s = _scheduler(CEILING, live=live)
    main = _entry(s, list(range(BOUNDARY)), ENTRY)
    older = _entry(s, list(range(95_000)), int(0.5 * GB))
    held = s.memory_aware_cache._entries[older]
    held._test_shared = True  # its bytes are the loan's, not the entry's
    s.running["main"].prompt_cache = list(held.cache)
    with s.memory_aware_cache._lock:
        assert s._memguard_freeable_bytes()[older] == 0
    _plan(s)
    assert older in s.memory_aware_cache._entries
    assert main not in s.memory_aware_cache._entries  # the last resort, real bytes
    assert s._memguard().counts["newest_evictions"] == 1


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


def test_with_a_concurrent_request_the_newest_entry_waits_for_the_deferral():
    """Only a conversation's newest entry could cover the step: with a
    concurrent request to defer, it is not touched (rule 3)."""
    guard = _plain_guard()
    world = _World(44 * GB, [(6 * GB, EVICT_LAST)])
    plan = guard.plan(ROWS, configured_chunk=2048, ceiling=48 * GB, probe=world.probe())
    assert plan.refuse is not None and plan.refuse.request_id == "side"
    assert world.log == []


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


# -- a side request prefilling while the main decodes (review of f1fe5b3be) --


def _mixed(ceiling: int, live: int):
    """The control state plus a 20k side request mid-prefill (admitted
    before the pressure arrived, as goose's subagents are) while the main
    decodes with MTP."""
    from tests.test_memory_guard_mtp_decode import _Batch

    s = _scheduler(ceiling, live=live)
    main = _entry(s, list(range(BOUNDARY)), ENTRY)
    side = Request(
        request_id="side",
        prompt="ignored",
        prompt_token_ids=[10**6 + i for i in range(20_000)],
        sampling_params=SamplingParams(max_tokens=8),
    )
    side.arrival_time = 2.0
    s.running["main"].arrival_time = 1.0
    s.running["side"] = side
    s.requests["side"] = side
    main_request = s._memguard_request_for_uid(7)
    s._memguard_request_for_uid = lambda uid: side if uid == 8 else main_request
    s._memguard_idle_base_bytes = LIVE - KV_95K - int(1.9 * GB)  # the weights
    bg = s.batch_generator
    bg._prompt_batch = _Batch(uids=[8], tokens=[[0] * 4096])
    bg._currently_processing = [([list(range(16_000))], 4096)]
    bg.prefill_batch_size = 1
    bg.prefill_step_size = 2048
    bg._unprocessed_sequences = []
    deferred: list[str] = []

    def defer(row, plan, others):
        deferred.append(row.request_id)
        s.running.pop(row.request_id, None)
        bg._prompt_batch = _Batch(uids=[], tokens=[])
        bg._currently_processing = []

    s._memguard_defer = defer
    return s, main, deferred


@pytest.mark.parametrize("extra_gb", [0, 1, 2, 3])
def test_a_side_prefill_beside_the_decoding_main_never_costs_its_entry(cpu, extra_gb):
    """The verify narrows and the side keeps a real chunk; where even that
    does not fit, the side is deferred. The main's 95,813 entry stays."""
    s, main, deferred = _mixed(CEILING, LIVE + extra_gb * GB)
    s._memguard_plan_dispatch([])
    counts = s._memguard().counts
    bg = s.batch_generator
    assert main in s.memory_aware_cache._entries
    assert counts["newest_evictions"] == 0
    if deferred:
        assert deferred == ["side"]
    else:
        assert counts["verify_narrowed"] == 1
        assert 4 <= bg._lz_verify_rows_cap < 32
        assert bg.prefill_step_size > 256
    if extra_gb == 0:
        assert not deferred and bg.prefill_step_size >= 1900


def test_the_side_prefill_on_128_gb_takes_no_action(cpu):
    s, main, deferred = _mixed(BIG_CEILING, LIVE)
    s._memguard_plan_dispatch([])
    counts = s._memguard().counts
    assert not deferred and main in s.memory_aware_cache._entries
    assert getattr(s.batch_generator, "_lz_verify_rows_cap", None) is None
    assert s.batch_generator.prefill_step_size == 2048
    assert counts["verify_narrowed"] == 0 and counts["chunks_reduced"] == 0


def test_when_even_the_main_alone_cannot_keep_it_the_side_goes_first(cpu):
    """+4 GB: 42.88 GB + the 6.72 GB entry leave no room for even the 4-row
    verify of the main ALONE. The side is deferred first; only then, the
    main's newest entry is the last resort (its bytes are real)."""
    s, main, deferred = _mixed(CEILING, LIVE + 4 * GB)
    order: list[str] = []
    real_defer = s._memguard_defer

    def defer(row, plan, others):
        order.append("defer")
        real_defer(row, plan, others)

    s._memguard_defer = defer
    real_evict = s.memory_aware_cache._evict_lru

    def evict():
        order.append("evict")
        return real_evict()

    s.memory_aware_cache._evict_lru = evict
    s._memguard_plan_dispatch([])
    assert deferred == ["side"]
    assert order == ["defer", "evict"]
    assert s._memguard().counts["newest_evictions"] == 1


# -- single stream, MTP decode at a KV growth step (Forge control, 354c3ac0e) --
#
# Gabee 03:36-03:37Z, one stream, 85,444-token boundary entry (6.07 GB, a
# copy: evicting it freed 5.5 GB measured). At growth steps the verify
# narrowed to 4-7 of 32 rows, then "evicted prefix entry tokens=85444
# level=5 (join) entries_left=0"; every later turn re-prefilled ~88k
# (562 s). Causes: after a partial reject the generator restored a VIEW of
# the fused verify's stacked restore points, which kept every point (4.6 GB
# at 31 rows) resident and invisible until the next step (fixed in the
# generator: an owned copy; test_mtp_spec_decode.py); the guard charged the
# previous round's held points on top of the new verify at a growth step;
# and the cheapest action -- a plain one-row decode -- was not available
# (the floor was the chain's depth + 1).

HELD = 30 * PER_ROW  # a 31-row round's restore points, still held


def _forge(monkeypatch, footprint: int, *, gap: int, ceiling: int = CEILING, droppable=True):
    """The decode step at the KV buffer's edge: it grows when the verify
    writes more than ``gap`` rows."""
    import rapid_mlx.scheduler as sched

    s = _scheduler(ceiling, live=footprint - ENTRY)
    if droppable:
        s.model.make_mtp_cache = lambda: [KVCache()]
    _decoding(s, capacity=PROMPT + gap, idx=PROMPT, context=PROMPT)
    monkeypatch.setattr(sched, "_held_rollback_bytes", lambda caches: HELD)
    main = _entry(s, list(range(BOUNDARY)), ENTRY)
    return s, main


def test_held_restore_points_are_not_charged_twice_at_a_growth_step(cpu, monkeypatch):
    """47.9 GB resident incl. 4.62 GB of the last round's restore points, a
    step that grows: the new verify replaces them, so the full width fits
    (354c3ac0e charged both and narrowed to 4 rows)."""
    s, main = _forge(monkeypatch, int(47.9 * GB), gap=0, droppable=False)
    assert _plan(s) is None
    assert main in s.memory_aware_cache._entries
    assert s._memguard().counts["verify_narrowed"] == 0


def test_a_plain_decode_step_comes_before_evicting_the_entry(cpu, monkeypatch):
    """48.72 GB resident (the entry + 4.62 GB held), one row of room before
    the buffer grows: any verify grows it and does not fit; a plain one-row
    decode does. 354c3ac0e evicted the 85,444-token entry here; now the step
    drops speculation."""
    s, main = _forge(monkeypatch, int(48.72 * GB), gap=1)
    cap = _plan(s)
    assert main in s.memory_aware_cache._entries
    assert cap == 1
    assert s._memguard_verify_rows_range() == (1, 32)
    counts = s._memguard().counts
    assert counts["newest_evictions"] == 0 and counts["decode_over_ceiling"] == 0


def test_a_narrower_verify_that_avoids_the_growth_is_preferred(cpu, monkeypatch):
    s, main = _forge(monkeypatch, int(48.72 * GB), gap=2)
    assert _plan(s) == 2
    assert main in s.memory_aware_cache._entries


def test_without_a_droppable_chain_the_floor_is_the_chain(cpu, monkeypatch):
    """An MTP head whose cache cannot trim keeps the chain's depth + 1 as the
    floor (a drafted chain is verified whole)."""
    s, _main = _forge(monkeypatch, int(47.9 * GB), gap=0, droppable=False)
    assert s._memguard_verify_rows_range() == (4, 32)


def test_the_forge_step_on_128_gb_takes_no_action(cpu, monkeypatch):
    s, main = _forge(monkeypatch, int(48.72 * GB), gap=1, ceiling=BIG_CEILING)
    assert _plan(s) is None
    assert main in s.memory_aware_cache._entries
    counts = s._memguard().counts
    assert counts["verify_narrowed"] == 0 and counts["newest_evictions"] == 0
