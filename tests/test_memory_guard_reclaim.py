# SPDX-License-Identifier: Apache-2.0
"""The guard reclaims before it refuses, and learns in a goose-shaped session
(workmule Flash-Next Gauntlet, 2026-10-29).

Gauntlet sb-7.2 on Jundot/Qwen3.8-Flash-Next-oQ4e-mtp (qwen4_exp: hybrid
GatedDeltaNet + QSA layers, PLE offload, YaRN), one goose session, prefix
cache on. At 22:09 local:

    infeasible even at floor chunk 256 with the allocator cache and every
    evictable entry gone | footprint=82.34GB step=18.56GB margin=3.93GB
    ceiling=103.90GB shape=1x256x239903
    refused request chatcmpl-75b: This prompt (241320 tokens) needs 104.8 GB
    but this Mac can hold 103.9 GB

goose then compacted 241k -> 4k. A fresh engine on the same Mac admitted
391,493 tokens (peak 96.46 GB). Two causes, both in this file:

1. The step was priced on the PRIOR (192 B per chunk x context; this model
   measures 0.2-13): in 2 h of turns the guard never calibrated once. Every
   goose prefill step is the FIRST chunk of a turn (a prefix hit), and those
   steps were kept out of the fit; once the prior shrank the chunk, each
   step had a new chunk size (2015, 1888 ... 375, 374), a group of one.
2. The only entry left was the hit itself (6.34 GB). qwen4_exp's caches are
   not admitted by the singleton pass-through, so the hit cannot be handed
   over: admission copies it, and that copy was charged on top. Alone from
   token 0 the prompt fits; it was refused instead.

At 1/100 of the bytes on the simulated device of the session tests.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rapid_mlx.memory_guard import (
    GuardProbe,
    MemoryGuard,
    PeakPrior,
    RowPlan,
    StepShape,
)
from tests.test_memory_guard_session import (  # noqa: F401
    GABEE_64_CEILING,
    GABEE_PHYSICAL,
    _auto_budget_scheduler,
    _Device,
    _drive,
    _new_tokens,
    _no_metal,
    _request,
    _session,
    _tokens,
    cpu,
    llama,
)

import rapid_mlx.scheduler as scheduler_mod


def _no_handover(monkeypatch):
    """The qwen4_exp case: the loan's layers are not singleton pass-through
    layers, so admission copies the hit whatever the guard decides."""
    monkeypatch.setattr(scheduler_mod, "_handover_admissible", lambda caches: False)


def test_a_turn_that_fits_alone_is_not_refused_for_its_own_hit_copy(
    llama, monkeypatch
):
    """REGRESSION (the 241k refusal): a session grows to the edge of the
    honest window with hand-over impossible. RED on v0.14.3-lz.16-ple.5: a
    turn inside the window is refused (needs more than the ceiling: the
    footprint holds the hit's entry AND its admission copy is charged).
    GREEN: the hit is given back, the turn re-prefills from token 0 and
    finishes; no refusal, no step over the ceiling, no OOM."""
    _no_handover(monkeypatch)
    device = _Device(physical=GABEE_PHYSICAL)
    scheduler = _auto_budget_scheduler(llama, monkeypatch, device)
    device.install(monkeypatch, scheduler)

    history, windows = _session(scheduler, delta=500, native=65536)

    guard = scheduler._memguard()
    assert guard.counts["refusals"] == 0
    assert guard.counts["admission_refusals"] == 0
    assert guard.counts["admission_handoffs"] == 0  # impossible here, by setup
    assert guard.counts["cold_requeues"] >= 1, guard.counts
    assert device.ooms == 0
    assert device.max_seen <= device.ceiling, (device.max_seen, device.ceiling)
    assert scheduler.get_stats()["memory_guard"]["steps_over_ceiling"] == 0
    # The session reached the window edge (it was not cut short).
    assert history[-1][1] + 500 >= windows[-1], (history[-1], windows[-1])
    # Every turn but the cold ones still hit.
    hits = sum(1 for _t, length, cached in history[1:] if cached and cached > 0)
    assert hits >= len(history) - 1 - guard.counts["cold_requeues"], history


def test_a_cold_requeue_happens_once_per_request(llama, monkeypatch):
    """A request given back its hit is never re-queued cold again: if it
    still does not fit from token 0, the typed refusal stands."""
    _no_handover(monkeypatch)
    device = _Device(physical=GABEE_PHYSICAL)
    scheduler = _auto_budget_scheduler(llama, monkeypatch, device)
    device.install(monkeypatch, scheduler)
    request = SimpleNamespace(cached_tokens=100, _memguard_cold=True)
    scheduler.running["r"] = request
    row = RowPlan(
        request_id="r",
        offset=100,
        segment=10,
        chunk_cap=0,
        chunk_floor=0,
        prompt_tokens=110,
        admit_bytes=1,
        admitting=True,
        arrival=0.0,
    )
    assert scheduler._memguard_requeue_cold(row) is False
    request._memguard_cold = False
    request.cached_tokens = 0
    assert scheduler._memguard_requeue_cold(row) is False  # no hit to give up
    scheduler.running.pop("r")


def test_a_goose_shaped_session_calibrates(llama, monkeypatch):
    """The only chunk size that repeats in this session is the FIRST chunk
    of each turn (a prefix-hit step); every tail has its own size. RED on
    ple.5: the coefficient stays on the prior for the whole session. GREEN:
    the hit steps' transient is measured without the copy and enters the
    fit."""
    device = _Device(physical=int(GABEE_64_CEILING / 0.97) + 1, scratch_scale=600.0)
    scheduler = _auto_budget_scheduler(llama, monkeypatch, device)
    device.install(monkeypatch, scheduler)
    guard = scheduler._memguard()
    # A deliberately WRONG prior (as on qwen4_exp: ~15-500x the device).
    guard.model.prior = PeakPrior(
        attention_bytes_per_unit=guard.model.prior.attention_bytes_per_unit * 4,
        token_bytes=guard.model.prior.token_bytes,
        kv_copy_bytes_per_context=guard.model.prior.kv_copy_bytes_per_context,
    )
    prompt = _new_tokens(0, 600)
    for turn in range(60):
        rid = f"t{turn}"
        outs = _drive(scheduler, [_request(rid, prompt, 4)])[rid]
        assert not outs[-1].error, (turn, outs[-1].error)
        # One chunk (64) plus a tail of a NEW size every turn, as goose's
        # tool results: only the FIRST chunk of each turn repeats a size,
        # and it is a prefix-hit step.
        prompt = prompt + _tokens(outs) + _new_tokens(turn + 1, 64 + 9 + turn)
        if guard.model.calibrated:
            break
    snap = guard.model.snapshot()
    assert guard.model.calibrated, (snap["groups"], [(o.shape, o.transient) for g in guard.model._groups.values() for o in list(g)[-6:]], guard.model.prior)
    assert device.max_seen <= device.ceiling


def test_hit_step_transient_excludes_the_measured_copy():
    """The copy stays resident after the step (it is the live cache now):
    peak - active_after + the chunk's new KV is the step without it."""
    gb = 1_000_000_000
    fake = SimpleNamespace(
        _current_metal_active_bytes=lambda: 88 * gb,
        _kv_footprint_fn=lambda: (lambda tokens: 100 + 1000 * tokens),
    )
    plan = SimpleNamespace(shape=StepShape(1, 256, 240_000))
    peak, before = 89 * gb, 82 * gb
    transient, admit = scheduler_mod.Scheduler._memguard_hit_step_transient(
        fake, plan, peak, peak - before, 6 * gb
    )
    assert admit == 0
    assert transient == 1 * gb + 1000 * 256
    # A step that also freed the source reads above the whole step: kept whole.
    fake._current_metal_active_bytes = lambda: 80 * gb
    transient, admit = scheduler_mod.Scheduler._memguard_hit_step_transient(
        fake, plan, peak, peak - before, 6 * gb
    )
    assert (transient, admit) == (peak - before, 0)
    # No reading: unchanged (the old, excluded path).
    fake._kv_footprint_fn = lambda: None
    assert scheduler_mod.Scheduler._memguard_hit_step_transient(
        fake, plan, peak, 7 * gb, 6 * gb
    ) == (7 * gb, 6 * gb)


def _probe(fp: int) -> GuardProbe:
    return GuardProbe(
        active=lambda: fp,
        cache=lambda: 0,
        clear_cache=lambda: None,
        entry_count=lambda: 0,
        evict_one=lambda level: False,
    )


@pytest.mark.parametrize("calibrated", [False, True])
def test_a_reduced_chunk_on_the_prior_is_a_power_of_two_of_the_floor(calibrated):
    """On the prior the reduced chunk snaps down to floor x 2^k so the steps
    group and the fit can take over; once calibrated, the largest chunk that
    fits (unchanged)."""
    prior = PeakPrior(
        attention_bytes_per_unit=192.0,
        token_bytes=100_000.0,
        kv_copy_bytes_per_context=2048.0,
    )
    guard = MemoryGuard(prior, floor_tokens=256)
    if calibrated:
        guard.model._a_learned = 192.0
        guard.model._b_learned = 2048.0
    ceiling = 100 * 10**9
    row = RowPlan(
        request_id="r",
        offset=60_000,
        segment=2048,
        chunk_cap=0,
        chunk_floor=0,
        prompt_tokens=62_048,
        admit_bytes=0,
        admitting=False,
        arrival=0.0,
    )
    plan = guard.plan([row], configured_chunk=2048, ceiling=ceiling, probe=_probe(80 * 10**9))
    assert plan.refuse is None
    assert 256 <= plan.chunk < 2048
    is_pow2_of_floor = (plan.chunk // 256) & (plan.chunk // 256 - 1) == 0 and plan.chunk % 256 == 0
    if calibrated:
        # The largest that fits: one more token would not.
        assert not guard.plan(
            [row], configured_chunk=plan.chunk + 1, ceiling=ceiling, probe=_probe(80 * 10**9)
        ).chunk == plan.chunk + 1
    else:
        assert is_pow2_of_floor, plan.chunk


# The real session's 2,048-token chunks (RAPID_MLX_GUARD_TRACE=1, the
# Gauntlet conversation replayed on the workmule): context, measured GB.
_FLAT_2048 = [
    (2048, 2.976), (4096, 2.938), (6144, 2.937), (8192, 2.937), (10240, 2.937),
    (12288, 2.937), (15341, 2.823), (17389, 2.967), (19437, 2.966),
    (22632, 2.692), (28284, 2.764), (31026, 2.790), (33074, 2.968),
    (35122, 2.968), (43998, 2.794),
]


def _workmule_prior() -> PeakPrior:
    # The engine's prior for Jundot/Qwen3.8-Flash-Next-oQ4e-mtp (its log).
    return PeakPrior(
        attention_bytes_per_unit=192.0,
        token_bytes=107_776.0,
        kv_copy_bytes_per_context=2048.0,
    )


def test_a_flat_full_chunk_group_calibrates_instead_of_being_dropped():
    """RED on ple.5: the group's median slope is under one layer's KV copy,
    so it was dropped, the model stayed on the prior and projected 18.7 GB
    for the 2.8 GB step at 44k (and 12.4 GB for a sub-GB step at 240k).
    GREEN: it votes a = 0; the residual window holds the measured step."""
    from rapid_mlx.memory_guard import PeakModel

    model = PeakModel(_workmule_prior(), fit_floor_tokens=256)
    for context, gb in _FLAT_2048:
        model.observe(StepShape(1, 2048, context), int(gb * 1e9), fit=True)
    assert model.calibrated
    assert model.attention_bytes_per_unit == 0.0
    # Every measured step is still covered by its projection (within the
    # model's own 10% error bound), and the 44k step is no longer 18.7 GB.
    for context, gb in _FLAT_2048:
        assert model.project(StepShape(1, 2048, context)) >= 0.9 * gb * 1e9
    assert model.project(StepShape(1, 2048, 43998)) < 4e9


def test_flat_tail_groups_still_do_not_vote():
    """The Gabee rule stands for chunks inside a buffer's slack: a group of
    48-token tails with a slope under the copy is dropped, not a vote."""
    from rapid_mlx.memory_guard import PeakModel

    model = PeakModel(_workmule_prior(), fit_floor_tokens=8)
    for i in range(16):
        context = 4000 + 2000 * i
        model.observe(StepShape(1, 48, context), int(5e6), fit=True)
    assert not model.calibrated
