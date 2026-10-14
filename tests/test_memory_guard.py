# SPDX-License-Identifier: Apache-2.0
"""Memory guard E1 (pure logic): the peak model and the per-chunk decision.

Root cause it answers (goose T12, 2026-10-14, Qwen3.5-27B Q8 on a 64 GB M3
Max): head_dim-256 attention has no fused SDPA path for q_len < k_len on a
non-NAX chip, so each prefill chunk materialises query-heads x chunk x context
score tensors that nothing charged -> ``[METAL] Command buffer execution
failed: Insufficient Memory``. The guard projects each chunk's step peak and
clears the allocator cache, evicts prefix entries, halves the chunk or refuses
BEFORE anything is allocated.
"""

from __future__ import annotations

import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rapid_mlx.memory_guard import (
    MARGIN_CEILING_RATIO,
    MARGIN_TRANSIENT_RATIO,
    GuardProbe,
    MemoryGuard,
    PeakModel,
    PeakPrior,
    RowPlan,
    StepShape,
    is_metal_oom,
    step_shape,
)

# Qwen3.5-27B: 24 query heads, 4 KV heads, head_dim 256, bf16, 16 of 64
# layers full attention (64 KiB/token of KV), hidden 5120, MLP 17408.
QWEN35_27B = PeakPrior.from_dims(
    query_heads=24,
    score_itemsize=2,
    hidden_size=5120,
    mlp_intermediate=17408,
    kv_bytes_per_token=16 * 2 * 4 * 256 * 2,
    layer_kv_bytes_per_token=2 * 4 * 256 * 2,
)


def _truth(a: float, k: float, g: float, c: float = 0.0):
    def t(s: StepShape) -> int:
        return int(a * s.units + k * s.tokens + g * s.context + c)

    return t


def test_prior_is_four_score_copies_per_query_head():
    assert QWEN35_27B.attention_bytes_per_unit == 24 * 2 * 4
    # The OOM step goose measured: 2203-token chunk at a 24.8k context. The
    # prior must charge at least the measured 5-8 GB live score tensors.
    projected = PeakModel(QWEN35_27B).project(StepShape(1, 2203, 24_800))
    assert projected >= 8e9


def test_learns_the_attention_slope_and_covers_every_observation():
    prior = QWEN35_27B
    truth = _truth(a=100.0, k=300_000.0, g=prior.kv_copy_bytes_per_context)
    model = PeakModel(prior)
    shapes = [StepShape(1, 2048, x) for x in range(2048, 40_961, 2048)]
    for s in shapes:
        model.observe(s, truth(s))
    assert model.calibrated
    assert model.attention_bytes_per_unit == pytest.approx(100.0, rel=1e-3)
    for s in shapes:
        assert model.project(s) >= truth(s)
    # Extrapolation (a longer context than ever observed) stays an upper
    # bound and is not wildly conservative.
    far = StepShape(1, 2048, 98_000)
    assert truth(far) <= model.project(far) <= 1.05 * truth(far)


def test_close_observations_do_not_calibrate():
    # Negative control: a span below 4 chunks and 25% of the context is noise.
    model = PeakModel(QWEN35_27B)
    model.observe(StepShape(1, 2048, 20_000), 5_000_000_000)
    model.observe(StepShape(1, 2048, 22_048), 6_000_000_000)
    assert not model.calibrated
    assert model.attention_bytes_per_unit == QWEN35_27B.attention_bytes_per_unit


def test_worst_case_never_decays():
    model = PeakModel(QWEN35_27B)
    s = StepShape(1, 1024, 8192)
    model.observe(s, 9_000_000_000)
    high = model.project(s)
    model.observe(s, 1_000_000_000)
    assert model.project(s) == high


def test_no_credit_for_a_smaller_chunk_until_one_is_observed():
    # Linear part observed only at 2048 tokens: a 256-token chunk keeps that
    # linear charge (a per-step constant could dominate it) ...
    prior = PeakPrior(
        attention_bytes_per_unit=0.0, token_bytes=0.0, kv_copy_bytes_per_context=0.0
    )
    model = PeakModel(prior)
    model.observe(StepShape(1, 2048, 4096), 600_000_000)
    assert model.project(StepShape(1, 256, 4096)) == 600_000_000
    # ... and once 256 is observed, the smaller size gets its own bound.
    model.observe(StepShape(1, 256, 4096), 100_000_000)
    assert model.project(StepShape(1, 256, 4096)) == 100_000_000
    # Between observed sizes: never above the larger size's bound.
    assert model.project(StepShape(1, 1024, 4096)) <= 600_000_000
    # Above the largest observed size it scales linearly.
    assert model.project(StepShape(2, 2048, 4096)) == 1_200_000_000


@settings(max_examples=60, deadline=None)
@given(
    a=st.floats(0.0, 400.0),
    k=st.floats(0.0, 2_000_000.0),
    c=st.floats(0.0, 2e9),
    seed=st.integers(0, 2**31),
)
def test_every_observation_stays_covered(a, k, c, seed):
    """Property: after any sequence of observations (any truth with
    non-negative coefficients, plus +-10% multiplicative noise), every
    recorded observation is covered by its projection plus the guard's own
    transient margin (10%), except at most ONE per residual window -- the
    single largest residual a window ignores so that one noisy step cannot
    move the model (review 2026-10-14; the margin and E5 absorb it)."""
    rng = random.Random(seed)
    truth = _truth(a=a, k=k, g=QWEN35_27B.kv_copy_bytes_per_context, c=c)
    model = PeakModel(QWEN35_27B)
    seen = []
    for _ in range(40):
        s = StepShape(
            rng.choice((1, 1, 2)),
            rng.choice((256, 512, 1024, 2048)),
            rng.randrange(256, 130_000),
        )
        t = int(truth(s) * rng.uniform(0.9, 1.1))
        model.observe(s, t)
        seen.append((s, t))
    by_tokens: dict = {}
    for s, t in seen:
        by_tokens.setdefault(s.tokens, []).append((s, t))
    for _tokens, obs in by_tokens.items():
        uncovered = [
            (s, t)
            for s, t in obs[-16:]
            if (1 + MARGIN_TRANSIENT_RATIO) * model.project(s) < t
        ]
        assert len(uncovered) <= 1, (uncovered, model.snapshot())


# ---------------------------------------------------------------------------
# step_shape
# ---------------------------------------------------------------------------


def test_step_shape_pads_to_the_longest_chunk_and_aligns_context():
    rows = [
        RowPlan("a", offset=10_000, segment=5000),
        RowPlan("b", offset=30_000, segment=300),
        RowPlan("c", offset=50, segment=0),  # moving to generation
    ]
    s = step_shape(rows, 2048)
    assert (s.batch, s.chunk, s.context) == (2, 2048, 32_048)
    capped = [RowPlan("a", offset=1000, segment=5000, chunk_cap=512)]
    assert step_shape(capped, 2048).chunk == 512


# ---------------------------------------------------------------------------
# MemoryGuard.plan
# ---------------------------------------------------------------------------

GB = 10**9


class _World:
    """A fake device: active bytes, allocator cache, prefix entries.

    ``entries``: (bytes, eviction level) oldest first; level 1 = no live
    session needs it, 2 = a running request already copied it, 3 = a live
    session still needs it (see ``memory_guard.EVICT_*``)."""

    def __init__(self, *, active, cache=0, entries=()):
        self.active = active
        self.cache = cache
        self.entries = [e if isinstance(e, tuple) else (e, 1) for e in entries]
        self.log: list[str] = []

    def probe(self) -> GuardProbe:
        def clear():
            if self.cache:
                self.log.append("clear")
            self.cache = 0

        def evict(max_level: int) -> bool:
            if not self.entries:
                return False
            lowest = min(level for _b, level in self.entries)
            if lowest > max_level:
                return False
            i = next(i for i, (_b, lv) in enumerate(self.entries) if lv == lowest)
            nbytes, level = self.entries.pop(i)
            self.active -= nbytes
            self.log.append(f"evict{level}")
            return True

        return GuardProbe(
            active=lambda: self.active,
            cache=lambda: self.cache,
            clear_cache=clear,
            evict_one=evict,
            entry_count=lambda: len(self.entries),
        )


def _guard(a=100.0, k=0.0) -> MemoryGuard:
    return MemoryGuard(
        PeakPrior(
            attention_bytes_per_unit=a, token_bytes=k, kv_copy_bytes_per_context=0
        ),
        floor_tokens=256,
    )


ROW = [RowPlan("req-long", offset=60_000, segment=20_000, prompt_tokens=80_000)]


def test_a_chunk_that_fits_runs_unchanged():
    guard = _guard()
    world = _World(active=30 * GB, cache=5 * GB)
    plan = guard.plan(ROW, configured_chunk=2048, ceiling=50 * GB, probe=world.probe())
    # 100 B x 2048 x 62048 = 12.7 GB step; 35 + 12.7 + margin < 50.
    assert plan.chunk == 2048 and plan.refuse is None
    assert world.log == [] and plan.actions == []


def test_the_allocator_cache_goes_first():
    guard = _guard()
    world = _World(active=30 * GB, cache=8 * GB, entries=[3 * GB, 3 * GB])
    plan = guard.plan(ROW, configured_chunk=2048, ceiling=50 * GB, probe=world.probe())
    assert world.log == ["clear"]
    assert plan.chunk == 2048 and plan.refuse is None
    assert guard.counts["clear_cache"] == 1 and guard.counts["evictions"] == 0


def test_order_is_cache_then_unneeded_entries_then_halving_then_needed_ones():
    guard = _guard()
    # Active 41 GB incl. 3 entries of 3 GB: the one the next turn needs
    # (level 3, stored this turn) is the OLDEST, then the fetched-from one a
    # running request already copied (2), then an unused one (1). A 2048
    # chunk needs 12.7 GB.
    world = _World(
        active=41 * GB, cache=1 * GB, entries=[(3 * GB, 3), (3 * GB, 2), (3 * GB, 1)]
    )
    plan = guard.plan(ROW, configured_chunk=2048, ceiling=48 * GB, probe=world.probe())
    # cache, then level 1, then level 2 (never by age): 35 GB + 12.7 GB step
    # + margin is still over 48 GB, so the chunk halves; at 1024 it fits and
    # the entry the next turn needs survives.
    assert world.log == ["clear", "evict1", "evict2"]
    assert plan.chunk == 1024 and plan.refuse is None
    assert world.entries == [(3 * GB, 3)]
    assert guard.counts["chunk_halvings"] == 1


def test_a_needed_entry_goes_only_after_halving_to_the_floor():
    guard = _guard()
    world = _World(active=int(48.5 * GB), entries=[(4 * GB, 3)])
    plan = guard.plan(ROW, configured_chunk=2048, ceiling=50 * GB, probe=world.probe())
    # Halving to the floor (256: 1.5 GB step + 1.2 GB margin) is not enough
    # at 48.5 GB; the needed entry goes, then it fits.
    assert plan.chunk == 256
    assert world.log == ["evict3"] and plan.refuse is None


def test_cannot_dispatch_at_the_floor_reports_the_longest_row():
    guard = _guard()
    world = _World(active=49 * GB)
    rows = ROW + [RowPlan("req-short", offset=100, segment=50, prompt_tokens=150)]
    plan = guard.plan(rows, configured_chunk=2048, ceiling=50 * GB, probe=world.probe())
    assert plan.chunk == 256
    assert plan.refuse is not None and plan.refuse.request_id == "req-long"
    assert plan.needed > 50 * GB
    # The scheduler decides defer vs refuse (and counts it).
    assert guard.counts["refusals"] == 0


def test_per_request_floor_override_lets_the_backstop_go_lower():
    guard = _guard()
    world = _World(active=48 * GB)
    rows = [
        RowPlan("r", offset=60_000, segment=20_000, chunk_cap=1024, chunk_floor=128)
    ]
    plan = guard.plan(rows, configured_chunk=2048, ceiling=50 * GB, probe=world.probe())
    assert plan.chunk <= 1024
    assert plan.chunk >= 128


def test_margin_is_a_ratio_of_the_ceiling_plus_a_ratio_of_the_step():
    guard = _guard()
    assert guard.margin(10 * GB, 50 * GB) == int(
        MARGIN_CEILING_RATIO * 50 * GB + MARGIN_TRANSIENT_RATIO * 10 * GB
    )


# ---------------------------------------------------------------------------
# Honest max context
# ---------------------------------------------------------------------------


def _kv(per_token=65_536, fixed=150_000_000):
    return lambda tokens: fixed + per_token * tokens


@settings(max_examples=80, deadline=None)
@given(
    base=st.integers(1 * GB, 40 * GB),
    extra=st.integers(0, 30 * GB),
    a=st.floats(0.0, 400.0),
)
def test_max_context_is_monotone_in_available_memory(base, extra, a):
    guard = _guard(a=a, k=100_000.0)
    ceiling = 48 * GB
    smaller = guard.max_context(
        base=base + extra, kv_footprint=_kv(), ceiling=ceiling, upper=262_144
    )
    larger = guard.max_context(
        base=base, kv_footprint=_kv(), ceiling=ceiling, upper=262_144
    )
    assert smaller <= larger


@settings(max_examples=80, deadline=None)
@given(base=st.integers(1 * GB, 45 * GB), a=st.floats(0.0, 400.0))
def test_max_context_never_exceeds_the_kv_only_figure(base, a):
    guard = _guard(a=a, k=100_000.0)
    ceiling = 48 * GB
    kv = _kv()
    honest = guard.max_context(
        base=base, kv_footprint=kv, ceiling=ceiling, upper=262_144
    )
    # The KV-only figure: the largest context whose KV alone fits beside the
    # same base.
    kv_only = max(
        (x for x in range(0, 262_145, 512) if base + kv(x) <= ceiling), default=0
    )
    assert honest <= max(kv_only, 0) + 511
    if honest:
        need = guard.single_sequence_need(
            honest, base=base, kv_footprint=kv, ceiling=ceiling
        )
        assert need <= ceiling


def test_max_context_is_capped_by_the_native_window():
    guard = _guard(a=0.0)
    assert (
        guard.max_context(base=0, kv_footprint=_kv(), ceiling=500 * GB, upper=32_768)
        == 32_768
    )


def test_metal_oom_text_is_recognised():
    assert is_metal_oom(
        RuntimeError(
            "[METAL] Command buffer execution failed: Insufficient Memory "
            "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
        )
    )
    assert not is_metal_oom(RuntimeError("shape mismatch"))


# ---------------------------------------------------------------------------
# Review 2026-10-14: calibration must not ratchet on one noisy observation
# ---------------------------------------------------------------------------

GOOSE_CEILING = int(50.1e9)  # the 64 GB M3 Max's Metal cap
GOOSE_BASE = int(30.5e9)  # Q8 27B weights + fixed state


def _goose_kv(tokens: int) -> int:
    return 156_893_184 + 65_536 * tokens


def _calibrated_guard(seed: int = 7) -> MemoryGuard:
    rng = random.Random(seed)
    guard = MemoryGuard(QWEN35_27B, floor_tokens=256)
    truth = _truth(a=104.0, k=280_000.0, g=QWEN35_27B.kv_copy_bytes_per_context)
    for chunk in (2048, 256):
        for x in range(chunk, 90_000, 2048 if chunk == 2048 else 1536):
            s = StepShape(1, chunk, x)
            guard.observe(s, int(truth(s) * rng.uniform(0.97, 1.03)))
    return guard


def _window(guard: MemoryGuard) -> int:
    return guard.max_context(
        base=GOOSE_BASE, kv_footprint=_goose_kv, ceiling=GOOSE_CEILING, upper=262_144
    )


@pytest.mark.parametrize(
    "outlier",
    [StepShape(1, 2048, 40_000), StepShape(1, 256, 4096), StepShape(1, 256, 60_000)],
)
def test_one_1gb_outlier_moves_the_window_by_at_most_5_percent(outlier):
    """RED on the first guard: the attention coefficient was the MAX pair
    slope, so one +1 GB observation at a small chunk took it from ~104 to
    ~2000 B/unit and the honest window from ~147k to ~26k tokens, forever."""
    guard = _calibrated_guard()
    before = _window(guard)
    a_before = guard.model.attention_bytes_per_unit
    guard.observe(outlier, guard.model.project(outlier) + 1_000_000_000)
    after = _window(guard)
    assert abs(after - before) <= 0.05 * before, (before, after)
    assert guard.model.attention_bytes_per_unit <= 1.10 * a_before


def test_the_attention_coefficient_is_capped_at_a_multiple_of_the_prior():
    guard = MemoryGuard(QWEN35_27B, floor_tokens=256)
    for x in range(2048, 40_000, 2048):
        s = StepShape(1, 2048, x)
        guard.observe(s, int(5000.0 * s.units))  # an absurd device
    assert (
        guard.model.attention_bytes_per_unit <= 4 * QWEN35_27B.attention_bytes_per_unit
    )


def test_a_backstop_lower_bound_never_enters_the_fit():
    """A system-pressure OOM records 'needed >= working set - active' (tens
    of GB). It becomes a decaying safety margin on that shape, never a slope:
    RED on the first guard (coefficient -> ~12,864 B/unit, window -> ~4k)."""
    guard = _calibrated_guard()
    before = _window(guard)
    a_before = guard.model.attention_bytes_per_unit
    shape = StepShape(1, 2048, 30_000)
    guard.record_backstop(shape, 25_000_000_000)
    assert guard.model.attention_bytes_per_unit == a_before
    assert _window(guard) == before
    # The plan at that shape now carries the extra margin ...
    t0, _ = guard.need(StepShape(1, 2048, 31_000), GOOSE_CEILING)
    assert t0 >= guard.model.project(StepShape(1, 2048, 31_000)) + 1e9
    # ... which decays with successful dispatches.
    for _ in range(200):
        guard.note_dispatch_ok(StepShape(1, 2048, 31_000))
    t1, _ = guard.need(StepShape(1, 2048, 31_000), GOOSE_CEILING)
    assert t1 == guard.model.project(StepShape(1, 2048, 31_000))
