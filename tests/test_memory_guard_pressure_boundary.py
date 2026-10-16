# SPDX-License-Identifier: Apache-2.0
"""An entry a running HYBRID row "already holds" is not redundant unless that
row can rebuild it (RC9 128 GB arm, 2026-10-16).

M4 Max 128 GB, ceiling 99.09 GB, RC9 fc56dc1a2, Qwen3.8-27B Q8 + MTP. The
guard's own planning took zero actions; the kernel sat at memory-pressure
level 2 (warn) on the busy laptop, so the system-pressure evictor gave back
what it ranks below a head. chatcmpl-898 (80,749-token prompt, fetched
79,277) stored its 80,717-token boundary snapshot, was parked by SRPF, and
resumed with 80,717 tokens "cached". While it decoded:

    evicted prefix entry tokens=79277 level=2 (system pressure)
    evicted prefix entry tokens=80717 level=2 (system pressure)

No turn-end entry followed (MTP publishes none), so the next main turn
MISSed: 86,500 tokens, 759 s to the first token. Level 2 is EVICT_REDUNDANT
-- "its bytes are duplicated in the live cache" -- which is false for a
non-trimmable cache: the live cache runs past the boundary and cannot be cut
back. Now any path that drops such an entry first switches the covering row
into boundary-state mode (it keeps the entry's recurrent state, by
reference) and the turn end rebuilds the entry from the row's own KV; when
the switch is impossible the entry ranks EVICT_LAST.

The real scheduler, mlx-lm ``BatchGenerator`` and vendored MTP verifier on a
tiny Qwen3.5 hybrid on the CPU device; the kernel level is stubbed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import rapid_mlx.memory_budget as memory_budget  # noqa: E402
from tests.test_lz_singleton_srpf import _no_metal, _run, cpu  # noqa: E402,F401
from tests.test_memory_guard_boundary_handoff import (  # noqa: E402,F401
    BOUNDARY_1,
    BOUNDARY_2,
    _cached_scheduler,
    _conversation,
    _turn,
    _two_turns,
    hybrid4,
)


@pytest.fixture
def kernel(monkeypatch):
    level = {"value": 0}
    monkeypatch.setattr(memory_budget, "system_memory_pressure", lambda: level["value"])
    return level


def _parked_then_pressed(model, kernel):
    """Turn 1 stores its boundary copy, is parked by SRPF past it, resumes
    with the boundary 'cached', and decodes under kernel pressure 2."""
    turn_1, turn_2 = _conversation()
    s, _ = _cached_scheduler(model, refuse_copies=False)
    evicted: list[int] = []
    mac = s.memory_aware_cache
    original = mac._evict_entry_locked

    def spy(key, reason):
        evicted.append((len(key), reason))
        return original(key, reason)

    mac._evict_entry_locked = spy
    s._lz_test_evicted = evicted
    main = _turn("t1", turn_1, BOUNDARY_1, 6)
    s.add_request(main)
    for _ in range(200):
        s.step()
        if tuple(turn_1[:BOUNDARY_1]) in s.memory_aware_cache._entries:
            break
    s.add_request(_turn("short", [40, 41, 42], 0, 1))
    pressed = False
    for _ in range(400):
        out = s.step()
        resumed = (
            main.request_id in s.running
            and int(main.cached_tokens or 0) >= BOUNDARY_1
            and main.num_output_tokens >= 1
        )
        if resumed and not pressed:
            kernel["value"] = 2
            s._memguard_yield_to_system()
            kernel["value"] = 0
            pressed = True
        if pressed and not s.has_requests():
            break
    assert pressed, "turn 1 never decoded with its boundary counted as cached"
    gone = tuple(turn_1[:BOUNDARY_1]) not in s.memory_aware_cache._entries
    return s, turn_2, gone


def test_rc9_a_boundary_evicted_under_kernel_pressure_is_rebuilt_at_the_turn_end(
    hybrid4, kernel
):
    """RED on fc56dc1a2: the pressure evictor drops turn 1's boundary as
    'redundant' while it decodes; nothing rebuilds it and turn 2 re-prefills
    all 83 tokens. GREEN: the row keeps the boundary's recurrent state when
    the copy goes, its turn end rebuilds the entry, turn 2 hits 48 tokens and
    answers exactly as after the copy that was never evicted."""
    s, turn_2, _gone = _parked_then_pressed(hybrid4, kernel)
    assert s._memguard().counts["system_pressure_evictions"] >= 1
    _, tokens_2, cached_2 = _run(s, {0: [_turn("t2", turn_2, BOUNDARY_2, 8)]})
    assert cached_2["t2"] == BOUNDARY_1
    _, copied_2, *_ = _two_turns(hybrid4, refuse_copies=False)
    assert tokens_2["t2"] == copied_2
    assert s._memguard().counts["boundary_handoffs"] >= 1


def test_the_pressure_evictor_really_took_the_boundary(hybrid4, kernel):
    """Control: the scenario does evict the boundary copy (else the test
    above proves nothing)."""
    s, _turn_2, _gone = _parked_then_pressed(hybrid4, kernel)
    assert BOUNDARY_1 in [n for n, _reason in s._lz_test_evicted], s._lz_test_evicted


def test_an_entry_a_row_cannot_rebuild_ranks_last(hybrid4, kernel):
    """The switch is impossible (a recurrent layout the turn end cannot
    rebuild): the entry ranks EVICT_LAST, so kernel pressure at WARN keeps
    it."""
    from mlx_lm.models.cache import ArraysCache

    from rapid_mlx.memory_guard import EVICT_LAST, EVICT_REDUNDANT

    class OtherState(ArraysCache):
        pass

    turn_1, _ = _conversation()
    s, _ = _cached_scheduler(hybrid4, refuse_copies=False)
    main = _turn("t1", turn_1, BOUNDARY_1, 6)
    s.add_request(main)
    for _ in range(200):
        s.step()
        if tuple(turn_1[:BOUNDARY_1]) in s.memory_aware_cache._entries:
            break
    key = tuple(turn_1[:BOUNDARY_1])
    main.cached_tokens = BOUNDARY_1  # as after an SRPF resume
    mac = s.memory_aware_cache
    with mac._lock:
        levels = {k: lv for k, lv, _c in s._memguard_ranked_entries()}
    assert levels[key] == EVICT_REDUNDANT  # rebuildable: honest
    entry = mac._entries[key]
    entry.cache = [
        (OtherState(len(c.cache)) if type(c) is ArraysCache else c) for c in entry.cache
    ]
    with mac._lock:
        levels = {k: lv for k, lv, _c in s._memguard_ranked_entries()}
    assert levels[key] == EVICT_LAST


def test_an_armable_entry_is_credited_without_the_state_it_hands_on(hybrid4, kernel):
    """Evicting a covered hybrid entry frees its KV, not its recurrent state
    (the row keeps that): the planner is never credited bytes that stay."""
    from rapid_mlx.memory_cache import _layer_is_non_trimmable
    from rapid_mlx.scheduler import _layer_arrays

    turn_1, _ = _conversation()
    s, _ = _cached_scheduler(hybrid4, refuse_copies=False)
    main = _turn("t1", turn_1, BOUNDARY_1, 6)
    s.add_request(main)
    for _ in range(200):
        s.step()
        if tuple(turn_1[:BOUNDARY_1]) in s.memory_aware_cache._entries:
            break
    key = tuple(turn_1[:BOUNDARY_1])
    mac = s.memory_aware_cache
    entry = mac._entries[key]
    recurrent = sum(
        int(a.nbytes)
        for layer, _slot, a in _layer_arrays(entry.cache)
        if _layer_is_non_trimmable(layer)
    )
    with mac._lock:
        before = s._memguard_freeable_bytes()[key]
    main.cached_tokens = BOUNDARY_1  # covered, as after an SRPF resume
    with mac._lock:
        after = s._memguard_freeable_bytes()[key]
    assert recurrent > 0
    total = sum(int(a.nbytes) for _l, _s, a in _layer_arrays(entry.cache))
    assert after == pytest.approx(before * (total - recurrent) / total, abs=2)
