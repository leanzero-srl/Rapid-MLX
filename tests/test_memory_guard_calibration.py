# SPDX-License-Identifier: Apache-2.0
"""Memory guard calibration on goose's live Q8 ladder (Gabee, 2026-10-14).

Gabee (64 GB M3 Max, Qwen3.5-27B-class Q8, ~30 GB weights), one growing
session, guard 8ab971457. Engine log, verbatim numbers:

  calibrate shape=1x1024x51447 measured=7.40GB projected_before=6.32GB a=46.00->103.39
  calibrate shape=1x512x56407  measured=6.02GB projected_before=5.41GB a=103.39->360.53
  calibrate shape=1x512x61815  measured=6.50GB projected_before=11.95GB a=360.53->332.72
  calibrate shape=1x512x62327  measured=2.75GB projected_before=11.16GB a=332.72->237.84
  calibrate shape=1x512x62839  measured=2.63GB projected_before=8.19GB a=237.84->103.39

and /v1/models max_model_len 207,900 -> 177,738 -> 102,913 -> 177,741 in ~2
minutes (goose compacts at 0.8x it: a dip to 103k compacts an 82k session
for nothing). On Q4 the window jittered 261,769-262,144 with a constant
coefficient.

The 6-7 GB steps are each turn's FIRST chunk after a prefix hit: the
fetched KV is still held by its prefix entry, so the first write grows every
layer by concatenation into NEW buffers -- a one-off copy of the cached KV
(~64 KiB/token here; ~3.5 GB at 56k), proportional to the context, not to
chunk x context. Charged to the attention term it moved the coefficient 3.5x
on one observation. The steady chunks of the same turns measured 2.6-2.75 GB.
"""

from __future__ import annotations

import random

import pytest

from rapid_mlx.memory_guard import (
    MemoryGuard,
    PeakModel,
    PeakPrior,
    StepShape,
)

GB = 1_000_000_000
# The goose 27B: 24 query heads, 4 KV heads, head_dim 256, bf16, 16 of 64
# layers full attention; GatedDeltaNet state 157 MB per sequence.
Q8_PRIOR = PeakPrior.from_dims(
    query_heads=24,
    score_itemsize=2,
    hidden_size=5120,
    mlp_intermediate=17408,
    kv_bytes_per_token=65_536,
    layer_kv_bytes_per_token=2 * 4 * 256 * 2,
)
KV_FIXED = 156_893_184
CEILING = int(50.1 * GB)
BASE = int(30.5 * GB)
NATIVE = 262_144


def _kv(tokens: int) -> int:
    return KV_FIXED + 65_536 * tokens


# The log's steps: (shape, measured bytes, first chunk of a turn after a hit)
LOG = [
    (StepShape(1, 1024, 51_447), int(7.40 * GB), True),
    (StepShape(1, 512, 56_407), int(6.02 * GB), True),
    (StepShape(1, 512, 61_815), int(6.50 * GB), True),
    (StepShape(1, 512, 62_327), int(2.75 * GB), False),
    (StepShape(1, 512, 62_839), int(2.63 * GB), False),
]


def _steady(shape: StepShape) -> float:
    # What the steady chunks of the same ladder measure (2.75/2.63 GB at
    # 62k with a 512 chunk): ~75 B per chunk x context + one layer's KV copy
    # + ~280 kB per prefill token.
    return 75.0 * shape.units + 4096 * shape.context + 280_000 * shape.tokens


def _ladder_guard(seed: int = 3) -> MemoryGuard:
    """The session up to 11:01: steady (non-first) chunks of earlier turns.
    Like the live ladder, the chunk shrank as the context grew (2048 ->
    1024 -> 512), so the 512 group holds only a few observations when the
    log's first 512 spike arrives -- the sparse group whose median pair
    slope that one spike dominated."""
    rng = random.Random(seed)
    guard = MemoryGuard(Q8_PRIOR, floor_tokens=256)
    for x in range(8_000, 30_000, 2_000):
        s = StepShape(1, 2048, x)
        guard.observe(s, int(_steady(s) * rng.uniform(0.97, 1.03)))
    for x in range(30_000, 50_000, 1_500):
        s = StepShape(1, 1024, x)
        guard.observe(s, int(_steady(s) * rng.uniform(0.97, 1.03)))
    for x in (30_000, 34_000, 38_000, 40_000):
        s = StepShape(1, 512, x)
        guard.observe(s, int(_steady(s) * rng.uniform(0.97, 1.03)))
    return guard


def _window(guard: MemoryGuard) -> int:
    return guard.max_context(base=BASE, kv_footprint=_kv, ceiling=CEILING, upper=NATIVE)


def test_the_first_chunk_copy_is_charged_as_such_and_never_fitted():
    """GREEN: with the copy charged explicitly (the scheduler knows the
    row's cached tokens), the log's steps move neither the coefficient nor
    the advertised window. RED on 8ab971457 (no such term: the 6-7 GB steps
    were fitted, coefficient 103 -> 360)."""
    guard = _ladder_guard()
    a0 = guard.model.attention_bytes_per_unit
    published = guard.published_window(_window(guard), NATIVE)
    for shape, measured, first in LOG:
        admit = _kv(shape.context - shape.chunk) if first else 0
        guard.observe(shape, measured, admit_bytes=admit)
        a = guard.model.attention_bytes_per_unit
        assert a0 / 1.25 <= a <= a0 * 1.25, (shape, a0, a)
        assert guard.published_window(_window(guard), NATIVE) == published
    # And the projection of a first chunk includes the copy.
    s = StepShape(1, 512, 70_000)
    assert guard.plan_transient(s, admit_bytes=_kv(69_488)) >= guard.model.project(
        s
    ) + _kv(69_488)


def test_one_observation_moves_a_coefficient_by_a_bounded_factor():
    """Even with no copy information, one observation can move a fitted
    coefficient by at most 1.25x. RED on 8ab971457: 103 -> 360 (3.5x) on
    the 1x512x56407 step."""
    guard = _ladder_guard()
    for shape, measured, _first in LOG:
        before = (
            guard.model.attention_bytes_per_unit,
            guard.model.context_bytes_per_token,
        )
        guard.observe(shape, measured)
        after = (
            guard.model.attention_bytes_per_unit,
            guard.model.context_bytes_per_token,
        )
        for b, a in zip(before, after, strict=True):
            assert b / 1.25 - 1e-6 <= a <= b * 1.25 + 1e-6, (shape, before, after)


def test_the_advertised_window_moves_only_on_a_confirmed_change():
    """A single model state never moves the published window; a change that
    persists across two model states does. RED on 8ab971457 (no
    confirmation: 207,900 -> 177,738 -> 102,913 -> 177,741)."""
    guard = _ladder_guard()
    published = guard.published_window(_window(guard), NATIVE)
    shape, measured, _ = LOG[1]
    guard.observe(shape, measured)  # one spike, no copy information
    assert guard.published_window(_window(guard), NATIVE) == published
    assert guard.published_window(_window(guard), NATIVE) == published  # re-asked
    # A real, persistent change: the device now costs 2x on every chunk.
    rng = random.Random(9)
    for x in range(52_000, 90_000, 1_536):
        s = StepShape(1, 512, x)
        guard.observe(s, int(2 * _steady(s) * rng.uniform(0.98, 1.02)))
        guard.published_window(_window(guard), NATIVE)
    assert guard.published_window(_window(guard), NATIVE) < 0.9 * published


def test_q4_jitter_around_the_native_window_publishes_the_native_window():
    """Q4: the computed window wandered 261,769-262,144 with a constant
    coefficient; within the band of the native window the native window is
    published, and small moves elsewhere do not flicker."""
    guard = MemoryGuard(Q8_PRIOR, floor_tokens=256)
    seen = {guard.published_window(c, NATIVE) for c in (262_144, 261_769, 262_001)}
    assert seen == {NATIVE}
    first = guard.published_window(180_000, NATIVE)
    # a different model state is required to move; jitter inside the band
    # never moves it.
    for c in (178_000, 181_500, 179_200, 176_000):
        assert guard.published_window(c, NATIVE) == first


def test_context_scaled_cost_is_separated_from_chunk_x_context():
    """Truth with a large per-context term (b = 30 kB per context token,
    7x one layer's KV copy) at two chunk sizes: the fit separates a from b.
    RED on 8ab971457 (b was the analytic 4 kB; the rest went into a)."""
    rng = random.Random(5)
    model = PeakModel(Q8_PRIOR)

    def truth(s: StepShape) -> float:
        return 75.0 * s.units + 30_000 * s.context + 250_000 * s.tokens

    for x in range(10_000, 90_000, 1_700):
        for chunk in (512, 1024):
            s = StepShape(1, chunk, x)
            model.observe(s, int(truth(s) * rng.uniform(0.98, 1.02)))
    assert model.attention_bytes_per_unit == pytest.approx(75.0, rel=0.15)
    assert model.context_bytes_per_token == pytest.approx(30_000, rel=0.25)
    far = StepShape(1, 256, 200_000)
    assert truth(far) <= 1.10 * model.project(far)
