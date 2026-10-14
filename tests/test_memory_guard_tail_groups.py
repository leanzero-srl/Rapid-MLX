# SPDX-License-Identifier: Apache-2.0
"""A turn's short tail chunks must not drag the attention coefficient down.

Live, Gabee 64 GB M3 Max, Q8 AREX-2 (24 query heads, head_dim 256, no fused
SDPA), b06acd8ea, the concurrent ladder at k=117 (19:29:56Z):

  reduce chunk 2048->1567 (projected step 9.28GB at context 111179)
      footprint=38.82GB margin=1.93GB ceiling=50.10GB
  STEP OVER CEILING: prefill shape=1x1567x111179 planned=48.10GB
      measured_peak=50.11GB (+2.01GB unplanned)

The guard calibrated a = 46-48 B per chunk x context on the 2048 ladder
(one bf16 score tensor of 24 heads is 48 B), then the fit fell to 21.6.
Every main turn and every side request ends in a 13-48-token tail chunk
(the message-boundary split). Such a chunk usually lands inside the KV
buffer's KVCache.step slack and writes in place: it does NOT pay the
growth copy (``g`` = one layer's KV per context token) the model charges
every chunk, so its group's slope (0.5-3 kB/token) is below ``g``. Clipped
to a = 0 those groups voted in the median across chunk sizes and halved
the coefficient; the 111k step, extrapolated from it, was planned short.

The truth below is that physics (attention 48 B/unit, ~1.36 MB per prefill
token, a fixed 0.3 GB, the growth copy only when the buffer grows) with a
+-6% run-to-run spread, replayed through the ladder's request shapes."""

from __future__ import annotations

import random

import pytest

from rapid_mlx.memory_guard import MemoryGuard, PeakPrior, StepShape

GB = 1_000_000_000
PRIOR = PeakPrior.from_dims(
    query_heads=24,
    score_itemsize=2,
    hidden_size=5120,
    mlp_intermediate=17408,
    kv_bytes_per_token=65_536,
    layer_kv_bytes_per_token=2 * 4 * 256 * 2,
)
CEILING = int(50.1 * GB)
STEP = 256  # mlx-lm KVCache.step
TARGET = StepShape(1, 1567, 111_179)


def _truth(chunk: int, context: int) -> float:
    start = context - chunk
    grows = (context + STEP - 1) // STEP > (start + STEP - 1) // STEP
    return (
        48.0 * chunk * context
        + 1.36e6 * chunk
        + 0.3e9
        + (4096 * context if grows else 0)
    )


def _ladder(seed: int) -> MemoryGuard:
    rng = random.Random(seed)
    guard = MemoryGuard(PRIOR, floor_tokens=STEP)

    def observe(chunk: int, context: int) -> None:
        measured = _truth(chunk, context) * rng.uniform(0.94, 1.06)
        guard.observe(StepShape(1, chunk, context), int(measured))

    def prefill(start: int, n: int, chunk: int) -> None:
        x = start
        while n > 0:
            c = min(chunk, n)
            x, n = x + c, n - c
            observe(c, x)

    context = 30_000
    for k in range(10, 118, 4):
        # main: a prefix hit, ~1-2k new tokens, then the boundary tail
        new = rng.randint(700, 2000)
        tail = rng.choice([13, 20, 23, 24, 29, 31, 48])
        chunk = rng.choice([2048, 2048, 1567, 1343, 1119, 1818])
        prefill(context, new - tail, chunk)
        prefill(context + new - tail, tail, chunk)
        context += new
        # side subagent: a miss (22.6k from scratch) or a hit (its tail)
        if rng.random() < 0.5:
            prefill(0, 22_592, rng.choice([2048, 2048, 1796, 1162, 622, 335, 256]))
            prefill(22_592, 31, 2048)
        else:
            observe(31, 22_623)
        if k in (26, 62):  # the big subagents, alone
            prefill(0, 57_000 if k == 26 else 65_000, rng.choice([2048, 1733, 1385]))
    return guard


@pytest.mark.parametrize("seed", range(40))
def test_the_ladders_111k_step_is_not_under_planned(seed):
    guard = _ladder(seed)
    planned = guard.plan_transient(TARGET)
    # The guard's own margin is its error bound: the step must fit in it.
    assert planned + guard.margin(planned, CEILING) >= _truth(1567, 111_179), (
        guard.model.attention_bytes_per_unit,
        planned / GB,
    )
