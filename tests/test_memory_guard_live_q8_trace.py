# SPDX-License-Identifier: Apache-2.0
"""The attention coefficient on Gabee's LIVE goose traffic (review 10).

Gabee (M3 Max 64 GB, AREX-2 "pi" Q8: 24 query heads, head_dim 256, no fused
SDPA, 16 of 64 layers full attention), guard b06acd8ea, real goose Forge
traffic, 2026-10-14/15. After 313 observations the guard logged

  calibrate (step) shape=1x2046x95146 measured=12.45GB projected_before=8.53GB
      a=0.00->0.00

-- a 4.07 GB under-plan at 100k context, peak 51.2 GB over a 50.10 GB
ceiling (the Metal working set, 55.7 GB, is what kept it from an OOM).

``tests/fixtures/memguard_gabee_q8_calibrate_20261015.txt`` is every
``calibrate (step)`` line of that engine log. The INFO log records only the
observations that MOVED the coefficient, so the 12-48-token tail chunks
that end every goose turn are under-represented; they are added back here
with the law fitted to the logged ones (0.12 GB + slope x context, the
slope swept 1.5-4.5 kB/token), optionally with the real measurement
outliers (a few GB on a tail step). Truth for a 95k step is the live
measurement; for a 128k step it is that measurement extended along the
evidence (the mainstream chunk's own context slope).
"""

from __future__ import annotations

import random
import statistics
from pathlib import Path

import pytest

from rapid_mlx.memory_guard import MemoryGuard, PeakPrior, StepShape, kv_growth_step

GB = 1e9  # the log's unit (memory_guard._gb)
CEILING = int(50.10 * GB)  # the guard's ceiling on Gabee (log)
LIVE = StepShape(1, 2046, 95_146)
LIVE_MEASURED = 12.45 * GB
# The guard's analytic prior on this model, as the engine logged it.
PRIOR = PeakPrior(
    attention_bytes_per_unit=192.0,
    token_bytes=251_904.0,
    kv_copy_bytes_per_context=4096.0,
)
TAILS = [12, 13, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
FIXTURE = (
    Path(__file__).parent / "fixtures" / "memguard_gabee_q8_calibrate_20261015.txt"
)


def _logged():
    rows = []
    for line in FIXTURE.read_text().splitlines():
        if line.startswith("#"):
            continue
        _t, batch, chunk, context, measured, *_rest = line.split()
        rows.append((int(batch), int(chunk), int(context), float(measured)))
    return rows


def _trace(slope_kb: float, seed: int, outliers: bool):
    """The logged observations in order, each big chunk followed (as a turn's
    remainder) by a tail chunk the INFO log did not record."""
    rnd = random.Random(seed)
    out = []
    for batch, chunk, context, measured in _logged():
        out.append((StepShape(batch, chunk, context), measured))
        if chunk >= 1000 and rnd.random() < 0.8:
            tail = rnd.choice(TAILS)
            x = context + tail
            t = 0.12 + slope_kb * 1024 * x / GB + rnd.uniform(-0.02, 0.02)
            if outliers and rnd.random() < 0.08:
                t += rnd.uniform(4.0, 6.5)
            out.append((StepShape(1, tail, x), round(t, 2)))
    return out


def _evidence() -> float:
    """``a`` as the mainstream chunk's own context slope says: the median
    pair slope of the logged 2048-token steps, less one layer's copy."""
    steps = sorted((x, m * GB) for _b, c, x, m in _logged() if c == 2048)
    slopes = [
        (hi[1] - lo[1]) / (hi[0] - lo[0])
        for i, lo in enumerate(steps)
        for hi in steps[i + 1 :]
        if hi[0] - lo[0] >= 0.25 * hi[0]
    ]
    return (statistics.median(slopes) - PRIOR.kv_copy_bytes_per_context) / 2048


def _replay(slope_kb, seed, outliers, *, until_live=False):
    guard = MemoryGuard(PRIOR, floor_tokens=kv_growth_step())
    lows = []
    for shape, measured in _trace(slope_kb, seed, outliers):
        if until_live and shape == LIVE:
            return guard, lows
        guard.observe(shape, int(measured * GB))
        if guard.model.calibrated:
            lows.append(guard.model.attention_bytes_per_unit)
    return guard, lows


CASES = [
    pytest.param(
        slope,
        seed,
        outliers,
        id=f"tail{slope}kB-seed{seed}-{'outliers' if outliers else 'clean'}",
    )
    for slope in (1.5, 3.0, 4.5)
    for outliers in (False, True)
    for seed in range(5)
]


@pytest.mark.parametrize(("slope_kb", "seed", "outliers"), CASES)
def test_the_coefficient_never_falls_below_the_evidence(slope_kb, seed, outliers):
    evidence = _evidence()
    assert 40.0 < evidence < 60.0, evidence  # one bf16 score tensor: 48 B
    guard, lows = _replay(slope_kb, seed, outliers)
    # (b06acd8ea: 0.00 after 313 observations; cb6ecf545: as low as 4.7)
    assert min(lows) >= 0.8 * evidence, (min(lows), evidence)
    assert guard.model.attention_bytes_per_unit > 0


@pytest.mark.parametrize(("slope_kb", "seed", "outliers"), CASES)
def test_the_live_95k_step_and_a_128k_step_fit_in_the_plan(slope_kb, seed, outliers):
    guard, _lows = _replay(slope_kb, seed, outliers, until_live=True)
    transient, margin = guard.need(LIVE, CEILING)
    assert transient + margin >= LIVE_MEASURED, (
        (transient + margin) / GB,
        LIVE_MEASURED / GB,
    )
    # 128k: the live measurement extended along the evidence.
    far = StepShape(1, 2048, 128_000)
    truth = (
        LIVE_MEASURED
        + _evidence() * (far.units - LIVE.units)
        + PRIOR.kv_copy_bytes_per_context * (far.context - LIVE.context)
    )
    transient, margin = guard.need(far, CEILING)
    assert transient + margin >= truth, ((transient + margin) / GB, truth / GB)


@pytest.mark.parametrize(("slope_kb", "seed", "outliers"), CASES[::5])
def test_a_large_overshoot_at_the_mainstream_chunk_raises_the_coefficient(
    slope_kb, seed, outliers
):
    """The live event, replayed: the guard planned the 95k step and it came
    in 4 GB above the plan. That must RAISE the coefficient (b06acd8ea:
    0.00 -> 0.00), and the same step is then planned at least as high as it
    measured."""
    guard, _lows = _replay(slope_kb, seed, outliers, until_live=True)
    before = guard.model.attention_bytes_per_unit
    overshoot = guard.model.project(LIVE) + 4.07 * GB
    guard.observe(LIVE, int(overshoot))
    assert guard.model.attention_bytes_per_unit > before, (
        before,
        guard.model.attention_bytes_per_unit,
    )
    # and the same step is planned higher than before the overshoot
    assert guard.model.project(LIVE) > overshoot - 4.07 * GB


def test_tail_groups_just_above_the_copy_never_zero_the_coefficient():
    """A head_dim-256 model whose mainstream chunk measures ~48 B/unit, and
    many tail groups (below the floor chunk) whose slopes sit just above one
    layer's copy, with measurement outliers: the coefficient stays with the
    evidence (within the fit's noise), never near 0 (the least-squares fit used to take a tiny ``a``
    with a context term far over its cap)."""
    rnd = random.Random(11)
    guard = MemoryGuard(PRIOR, floor_tokens=kv_growth_step())
    g = PRIOR.kv_copy_bytes_per_context

    def steady(chunk, x):
        return 48.0 * chunk * x + g * x + 1.36e6 * chunk + 0.3e9

    for turn in range(60):
        x = 20_000 + turn * 1_500
        guard.observe(
            StepShape(1, 2048, x), int(steady(2048, x) * rnd.uniform(0.97, 1.03))
        )
        tail = 12 + turn % 20
        t = 0.12e9 + (g + rnd.uniform(50, 400)) * x
        if turn % 9 == 4:
            t += rnd.uniform(4.0, 6.5) * GB
        guard.observe(StepShape(1, tail, x + tail), int(t))
        if guard.model.calibrated:
            assert guard.model.attention_bytes_per_unit >= 0.7 * 48.0, (
                turn,
                guard.model.attention_bytes_per_unit,
            )
