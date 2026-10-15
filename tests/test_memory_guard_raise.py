# SPDX-License-Identifier: Apache-2.0
"""The overshoot raise (review 10) must not be a ratchet (review 11).

With 2,048-token chunks and OBS_PER_GROUP observations per group, no pair of
a group spans SLOPE_MIN_SPAN_FRACTION of the context above ~131k tokens: the
pair fit returns nothing there, so nothing pulled a raised coefficient back.
One +4 GB outlier at 140k moved it 48.2 -> 53.0 for good; six -> 64.2."""

from __future__ import annotations

import random

from rapid_mlx.memory_guard import MemoryGuard, PeakPrior, StepShape, kv_growth_step

GB = 1e9
A, G = 48.0, 4096.0
PRIOR = PeakPrior(192.0, 251_904.0, G, 65_536.0)


def _truth(shape, rnd):
    return (A * shape.units + G * shape.context + 1.36e6 * shape.tokens + 0.3e9) * (
        rnd.uniform(0.99, 1.01)
    )


def _long_context_guard():
    rnd = random.Random(5)
    guard = MemoryGuard(PRIOR, floor_tokens=kv_growth_step())
    x = 10_000
    while x < 140_000:
        s = StepShape(1, 2048, x)
        guard.observe(s, int(_truth(s, rnd)))
        x += 2048
    return guard, rnd


def _clean(guard, rnd, n=300):
    for _i in range(n):
        s = StepShape(1, 2048, 130_000 + rnd.randint(0, 12_000))
        guard.observe(s, int(_truth(s, rnd)))


def test_a_raise_decays_back_where_the_fit_is_silent():
    guard, rnd = _long_context_guard()
    model = guard.model
    assert model._fit() is None  # the precondition: no pair spans 25% here
    settled = model.attention_bytes_per_unit
    s = StepShape(1, 2048, 140_000)
    guard.observe(s, int(_truth(s, rnd) + 4 * GB))
    assert model.attention_bytes_per_unit > settled  # the raise
    _clean(guard, rnd)
    assert abs(model.attention_bytes_per_unit - settled) <= 0.01 * settled
    for _k in range(6):
        guard.observe(s, int(_truth(s, rnd) + 4 * GB))
    _clean(guard, rnd)
    assert abs(model.attention_bytes_per_unit - settled) <= 0.01 * settled


def test_the_raise_is_the_same_rule_at_zero():
    """From a = 0 and from a = 0.01 the same outlier moves the coefficient
    by the same relative bound (0 stays 0; it was 0 -> 5.5 vs 0.01 ->
    0.011)."""
    s = StepShape(1, 2048, 100_000)
    moved = []
    for a in (0.0, 0.01):
        model = MemoryGuard(PRIOR, floor_tokens=kv_growth_step()).model
        model._a_learned, model._b_learned = a, G
        model._a_fit = a
        model.observe(s, int(model.project(s) * 1.5))
        moved.append(model.attention_bytes_per_unit)
    assert moved[0] == 0.0, moved
    assert moved[1] <= 0.01 * 1.1 + 1e-12, moved
