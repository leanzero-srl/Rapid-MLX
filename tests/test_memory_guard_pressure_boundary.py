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


def _plain_hybrid(model):
    """The hybrid without MTP: its turn end stores prompt + output."""
    from tests.test_lz_singleton_srpf import _scheduler

    return _scheduler(
        model,
        enable_prefix_cache=True,
        use_memory_aware_cache=True,
        hybrid_cache_entries=8,
        spec_decode="none",
        max_num_seqs=1,
    )


def _arm_at(request, length, model):
    from mlx_lm.models.cache import ArraysCache

    states = [
        ArraysCache(2) if layer.is_linear else None for layer in model.layers
    ]
    request._memguard_boundary_state = (length, states)


def test_a_boundary_stored_after_an_arm_supersedes_it(hybrid4, kernel):
    """A row armed at a shorter entry (evicted before the row reached its
    boundary) that then stores its boundary copy keeps no armed state: the
    copy is the better resume point."""
    turn_1, _ = _conversation()
    s = _plain_hybrid(hybrid4)
    main = _turn("t1", turn_1, BOUNDARY_1, 4)
    s.add_request(main)
    _arm_at(main, 16, hybrid4)
    for _ in range(200):
        s.step()
        if tuple(turn_1[:BOUNDARY_1]) in s.memory_aware_cache._entries:
            break
    assert tuple(turn_1[:BOUNDARY_1]) in s.memory_aware_cache._entries
    assert main._memguard_boundary_state is None


def test_a_kept_resume_point_keeps_the_end_of_turn_store(hybrid4, kernel):
    """An armed row whose conversation already holds a resume point at
    least as good does not trade its end-of-turn entry for a cut."""
    turn_1, _ = _conversation()
    s = _plain_hybrid(hybrid4)
    main = _turn("t1", turn_1, BOUNDARY_1, 4)
    s.add_request(main)
    for _ in range(200):
        s.step()
        if tuple(turn_1[:BOUNDARY_1]) in s.memory_aware_cache._entries:
            break
    _arm_at(main, 16, hybrid4)  # armed, with the 48-token boundary kept
    while s.has_requests():
        s.step()
    lengths = sorted(len(k) for k in s.memory_aware_cache._entries)
    assert lengths[-1] == len(turn_1) + 4, lengths  # prompt + output stored
    assert 16 not in lengths


# -- review of d03e2e5af (probes in scratchpad rc10probe/) ------------------

HELD = 56  # past the 48-token boundary, inside the 61-token prompt


def test_an_entry_past_the_boundary_never_replaces_the_kept_boundary(hybrid4, kernel):
    """D1 (regression): RC9 kept (48, state) because the copy did not fit;
    the row then holds a requeue-style entry prompt[:56] (a deferral's
    re-fetch) and that entry is evicted. Arming it replaced the kept state:
    the turn end stored a dead 56-token entry and turn 2 cached 8 (RC9: 48).
    An entry past the row's message boundary is no resume point of its next
    prompt: evicting it arms nothing."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    turn_1, turn_2 = _conversation()
    s, _ = _cached_scheduler(hybrid4, refuse_copies=True)
    main = _turn("t1", turn_1, BOUNDARY_1, 6)
    s.add_request(main)
    mac = s.memory_aware_cache
    done = False
    for _ in range(400):
        s.step()
        kept = getattr(main, "_memguard_boundary_state", None)
        if (
            not done
            and kept is not None
            and main.request_id in s.running
            and main.num_output_tokens >= 1
        ):
            assert kept[0] == BOUNDARY_1
            fresh = make_prompt_cache(hybrid4)
            hybrid4(mx.array([turn_1[:HELD]]), cache=fresh)
            mx.eval([c.state for c in fresh])
            assert mac.store(list(turn_1[:HELD]), fresh, evict_prefixes=False)
            main.cached_tokens = HELD
            with mac._lock:
                mac._evict_entry_locked(tuple(turn_1[:HELD]), reason="system pressure")
            assert main._memguard_boundary_state[0] == BOUNDARY_1
            done = True
        if done and not s.has_requests():
            break
    assert done
    assert sorted(len(k) for k in mac._entries) == [BOUNDARY_1]
    _, tokens_2, cached_2 = _run(s, {0: [_turn("t2", turn_2, BOUNDARY_2, 8)]})
    assert cached_2["t2"] == BOUNDARY_1
    _, copied_2, *_ = _two_turns(hybrid4, refuse_copies=False)
    assert tokens_2["t2"] == copied_2


@pytest.mark.parametrize("reason", ["handoff", "system pressure", "lru_evict"])
def test_every_eviction_path_arms_the_covering_row(hybrid4, kernel, reason):
    """D2: an admission handoff evicts prompt[:cached]; an SRPF-parked row
    resumes with exactly its boundary as cached_tokens (RC9 chatcmpl-898:
    80,717), so the handed-over key is the boundary entry itself. The hook
    skipped "handoff": turn 2 cached 0. Every reason arms now."""
    turn_1, turn_2 = _conversation()
    s, _ = _cached_scheduler(hybrid4, refuse_copies=False)
    main = _turn("t1", turn_1, BOUNDARY_1, 6)
    s.add_request(main)
    key = tuple(turn_1[:BOUNDARY_1])
    done = False
    for _ in range(400):
        s.step()
        if (
            not done
            and key in s.memory_aware_cache._entries
            and main.request_id in s.running
            and main.num_output_tokens >= 1
        ):
            main.cached_tokens = BOUNDARY_1  # as after the SRPF park
            mac = s.memory_aware_cache
            with mac._lock:
                mac._evict_entry_locked(key, reason=reason)
            done = True
        if done and not s.has_requests():
            break
    assert done
    _, tokens_2, cached_2 = _run(s, {0: [_turn("t2", turn_2, BOUNDARY_2, 8)]})
    assert cached_2["t2"] == BOUNDARY_1
    _, copied_2, *_ = _two_turns(hybrid4, refuse_copies=False)
    assert tokens_2["t2"] == copied_2


def test_a_true_admission_handoff_answers_as_a_copy_would(hybrid4, kernel):
    """A real admission handoff: turn 2 takes turn 1's boundary entry's
    buffers over (the copy "does not fit"), and the hook arms it with that
    entry's recurrent state by reference. The turn-2 answer is unchanged,
    the state is superseded by turn 2's own stored boundary, and turn 3
    resumes from that boundary."""
    turn_1, turn_2 = _conversation()
    s, _ = _cached_scheduler(hybrid4, refuse_copies=False)
    _run(s, {0: [_turn("t1", turn_1, BOUNDARY_1, 6)]})
    key = tuple(turn_1[:BOUNDARY_1])
    assert key in s.memory_aware_cache._entries
    t2 = _turn("t2", turn_2, BOUNDARY_2, 8)
    real = s._memguard_admit_handoffs
    armed = {}

    def handoff(guard, ceiling, rows):
        # force the hand-over branch: the copy-on-admit "does not fit"
        s._metal_footprint_bytes = lambda **_k: 10**15
        try:
            return real(guard, ceiling, rows)
        finally:
            del s._metal_footprint_bytes
            state = getattr(t2, "_memguard_boundary_state", None)
            if state is not None:
                armed.setdefault("state", state)

    s._memguard_admit_handoffs = handoff
    _, tokens_2, cached_2 = _run(s, {0: [t2]})
    assert cached_2["t2"] == BOUNDARY_1
    assert s._memguard().counts["admission_handoffs"] >= 1
    assert armed.get("state") is not None and armed["state"][0] == BOUNDARY_1
    _, copied_2, *_ = _two_turns(hybrid4, refuse_copies=False)
    assert tokens_2["t2"] == copied_2
    assert t2._memguard_boundary_state is None
    assert tuple(turn_2[:BOUNDARY_2]) in s.memory_aware_cache._entries


# -- RC10: dead weight in one line ------------------------------------------


def test_a_line_keeps_its_newest_boundary_and_one_fallback(hybrid4, kernel):
    """RC10 128 GB: the prefix cache held 7 hybrid entries (40.2 GB) for one
    main conversation and one side request, ``superseded`` 0 -- an MTP turn
    stores no end-of-turn entry, the only path that superseded anything. A
    newer boundary of the same line now supersedes the older ones it extends
    as soon as it is stored, except the one this turn resumed FROM (a branch
    may still extend it; review 4). RED on 150a141ee: one entry per turn,
    all kept. Every turn still resumes from the previous boundary."""
    s, _ = _cached_scheduler(hybrid4, refuse_copies=False)
    prompt = [(7 * i + 3) % 120 + 1 for i in range(40)]
    boundary = len(prompt) - 13
    counts = []
    cached = []
    for turn in range(6):
        rid = f"t{turn}"
        _, _tokens, got = _run(s, {0: [_turn(rid, prompt, boundary, 4)]})
        cached.append(got[rid])
        counts.append(sorted(len(k) for k in s.memory_aware_cache._entries))
        # the next turn: up to this boundary, then the re-rendered reply,
        # the tool result and a new generation prompt
        tail = [(11 * i + 5 * turn) % 120 + 1 for i in range(24 + 13)]
        prompt = prompt[:boundary] + tail
        boundary = len(prompt) - 13
    assert all(len(c) <= 2 for c in counts), counts
    assert cached[1:] == [c[-1] for c in counts[:-1]], (cached, counts)
    assert s._memguard().counts["superseded"] >= 4


def test_a_hybrid_on_a_machine_with_no_room_resumes_through_the_handover(
    hybrid4, kernel, monkeypatch
):
    """The machine leaves the engine nothing for copies (101 GB of others on
    the laptop): the boundary COPY is refused as optional memory, the row
    keeps the recurrent state, and its turn end hands the boundary over
    under the device ceiling. Turn 2 resumes from it, with the tokens of a
    copied boundary."""
    turn_1, turn_2 = _conversation()
    s, _ = _cached_scheduler(hybrid4, refuse_copies=False)
    physical = 10**12
    monkeypatch.setattr(memory_budget, "system_physical_memory", lambda: physical)
    monkeypatch.setattr(
        memory_budget,
        "system_memory_used",
        lambda: memory_budget.MachineMemory(physical, 0),  # all of it in use
    )
    s._memguard_others_read = None
    assert s._memguard_cache_ceiling() == 0
    _run(s, {0: [_turn("t1", turn_1, BOUNDARY_1, 6)]})
    counts = s._memguard().counts
    assert counts["boundary_states_kept"] >= 1 and counts["boundary_handoffs"] >= 1
    _, tokens_2, cached_2 = _run(s, {0: [_turn("t2", turn_2, BOUNDARY_2, 8)]})
    assert cached_2["t2"] == BOUNDARY_1
    _, copied_2, *_ = _two_turns(hybrid4, refuse_copies=False)
    assert tokens_2["t2"] == copied_2
