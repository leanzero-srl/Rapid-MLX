# SPDX-License-Identifier: Apache-2.0
"""Step-peak memory guard: a prefill chunk is dispatched only if it fits.

LeanZero fork, goose T12 (2026-10-14). The owner's goal: prefix caching is
invisible ("it should just work ... device aware or memory aware") and a GPU
out-of-memory is not possible on the text lane, for one long agent session,
on any Mac and model size.

Root cause of the Metal OOM goose hit (``[METAL] Command buffer execution
failed: Insufficient Memory``) on a 64 GB M3 Max with Qwen3.5-27B Q8 at a
22.6k-token context: head_dim-256 attention has no fused SDPA kernel when
``q_len < k_len`` on a non-NAX chip, so every prefill chunk materialises
score tensors of ``query heads x chunk x context x itemsize`` (2.4 GB at
2048 x 24.6k with 24 heads) and the MLX graph keeps several of them alive at
once. Nothing charged that transient: admission projected KV only, adaptive
prefill started at 32k tokens, and ``mx.set_memory_limit`` is a GC
threshold, never a refusal. So the guard must be ours.

What this module holds (pure Python, no MLX import, unit-testable):

* :class:`PeakModel` -- the per-step transient model. SELF-CALIBRATING: the
  scheduler resets MLX's peak counter before each prefill chunk and feeds the
  measured ``peak - active_before`` back here. It starts from a conservative
  analytic prior derived from the model config and never decays.
* :class:`MemoryGuard` -- the decision procedure run before EVERY dispatch
  (feasibility first; clear the allocator cache, evict prefix entries in the
  one order, take the LARGEST chunk that fits, defer the latest arrival or
  refuse), the honest max context, the admission check, the derived
  prefix-cache budget and allocator cache limit, and the counters
  ``/v1/status`` reports.

Every number is derived from the device (the scheduler's Metal cap), the
model config, or measurement. NUMERIC CONSTANTS ON THE GUARD PATH (owner:
"you should not build hard coded solutions"):

  constant                      value  class
  ----------------------------  -----  ----------------------------------------
  MARGIN_CEILING_RATIO          0.02   ratio of the device ceiling (cap gap)
  MARGIN_TRANSIENT_RATIO        0.10   measured (2x the 5% run-to-run spread)
  CACHE_LIMIT_FLOOR_RATIO       0.02   ratio of the device ceiling
  ALLOCATOR_CACHE_SHARE         0.25   ratio of the device ceiling (lz boot)
  DECODE_GROWTH_LAYERS          2.0    measured (2.12 layers, real allocator)
  WINDOW_BAND_RATIO             0.03   measured (Q4 window jitter)
  MAX_STEP_FACTOR               1.25   measured motive (Gabee 103->360 jump)
  MAX_PRIOR_MULTIPLE            4.0    ratio of the analytic prior
  PRIOR_SCORE_COPIES            4      measured (M4 Max SDPA peak ladder)
  PRIOR_HIDDEN_COPIES           8      ratio of model dims; prior only, the
  PRIOR_MLP_COPIES              3        linear term is measured after it
  REAPPLY_RATIO                 0.05   algorithm (setter hysteresis)
  SLOPE_MIN_SPAN_CHUNKS / _FRACTION, SLOPE_MIN_PAIRS, OBS_PER_GROUP,
  ROBUST_MIN_OBS                4 / 0.25, 3, 16, 4   algorithm (statistics)
  BACKSTOP_CONTEXT_FRACTION / _DECAY / _DROP_FRACTION
                                0.75 / 0.97 / 0.01   algorithm (decay)
  EVICT_UNUSED..EVICT_NEEDED    1..4   enumeration (eviction order)
  _PLAN_ATTEMPTS (scheduler)    64     algorithm bound: re-plans per step
                                       (each defers / refuses / hands over /
                                       evicts one thing); exhausting it is a
                                       defect signal (plan_exhausted)
  SYSTEM_PRESSURE_WARN /        2 / 4  the kernel's own memory-pressure enum
    _CRITICAL (memory_budget)          (kern.memorystatus_vm_pressure_level)
  idle-base window              OBS_PER_GROUP idle samples (statistics)
  floor chunk                   KVCache.step: an mlx-lm library constant
                                (256), read at runtime, not measured on the
                                device; the KV growth granularity. Operator
                                override: SchedulerConfig only (no CLI flag)
  normal (largest) chunk        the engine's prefill_step_size; the guard
                                derives the largest that fits, never more
  idle line / Metal trigger     1 - MARGIN_CEILING_RATIO under the guard
  KV growth step                KVCache.step (mlx-lm, read at runtime)
  per-context term cap          the model's whole KV per token (was a typed
                                "16 layers" -- removed)
  window version bucket         MARGIN_CEILING_RATIO x ceiling (was a typed
                                16 MiB -- removed)
  auto-cap headroom floor       1% of the device budget (memory_budget; was
                                a typed 512 MiB -- removed)
  prefix budget                 the guard's idle room unless the operator
                                chose one (the 20%-of-RAM heuristic and its
                                100 MiB floor only seed the cache); given
                                back at once when the KERNEL reports memory
                                pressure (other apps / engines), and a
                                session's superseded entries are dropped

Absolutes that remain, and why: the unknown-shape prior (64 query heads,
8192 hidden) when the model config cannot be read -- it governs only the
first chunks, then measurement replaces it; and the GUARD-OFF policies
(adaptive-prefill ladder 0.70/0.80/0.88 -> 1024/512/256, its 32,768-token
trigger, the 0.9 pressure fraction, memory_budget's 2/32 GiB cache clamps),
which the guard bypasses -- they exist only for RAPID_MLX_MEMORY_GUARD=0.
"""

from __future__ import annotations

import logging
import math
import os
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# A/B switch only. The guard is always on; ``RAPID_MLX_MEMORY_GUARD=0``
# (or off/false/no) restores the unguarded engine for a comparison run.
GUARD_ENV = "RAPID_MLX_MEMORY_GUARD"

# ratio: live score-tensor copies per query head in the prior. Measured on an
# M4 Max, mlx 0.32.2, head_dim 256, 24 query / 4 KV heads, bf16, causal mask
# (scratch sdpa_peak.py / sdpa_peak2.py, 2026-10-14): one SDPA fallback call
# holds 2.04-2.17 B per (chunk x context x head) = ~1 score tensor; 4 chained
# attention calls in one eval hold 6.19 B (~3.1 tensors, the command buffer
# keeps freed scores in flight); a 16-layer transformer-shaped graph holds
# 4.27-6.39 B. goose's measured 5-8 GB live at 2203 x 24.8k x 24 heads is
# 3.8-6.1 B. Four 2-byte copies (8 B) covers the worst of all of these. It is
# only the starting point: observations replace it after a few chunks.
PRIOR_SCORE_COPIES = 4
# ratio: activation copies of the residual stream per prefill token in the
# prior (residual, norm output, attention input/output, q/k/v projections of
# a couple of layers in flight). Order of magnitude only -- it governs the
# very first chunk; the linear term is measured after that.
PRIOR_HIDDEN_COPIES = 8
# ratio: MLP intermediates per token (gate, up, activation output).
PRIOR_MLP_COPIES = 3

# ratio: margin floor as a fraction of the ceiling. Covers allocations that
# happen between this check and the dispatch and are not part of the step
# (boundary-snapshot extraction, sampler/detokenizer arrays, Metal heap
# rounding). The ceiling itself is already the engine's Metal cap
# (0.90-0.97 of the recommended working set), so the physical OOM line sits
# a further 3-10% above it.
MARGIN_CEILING_RATIO = 0.02
# ratio: margin on the projected step transient = the peak model's own
# error bound. The synthetic ladder above varied 5% run-to-run for the same
# graph at a different context (4.27 vs 4.50 B); doubled.
MARGIN_TRANSIENT_RATIO = 0.10
# ratio: the allocator cache is never set below this fraction of the
# ceiling, so decode's per-step buffer churn is still served from the cache
# (a B=1 decode step's transient is tens of MB).
CACHE_LIMIT_FLOOR_RATIO = 0.02
# ratio: the allocator cache's ceiling share under the guard -- lz's boot
# share (memory_budget.compute_metal_cache_limit: 25% of the soft limit),
# without its 2 GiB floor / 32 GiB cap (absolutes).
ALLOCATOR_CACHE_SHARE = 0.25
# Re-apply the allocator cache limit / prefix budget only when it moves by
# more than this fraction (avoids a setter call per chunk while a ladder of
# observations nudges the model up).
REAPPLY_RATIO = 0.05

# Pair slopes are trusted only across a context span of at least this many
# chunks AND this fraction of the larger context, so measurement noise in the
# peak (a few hundred MB) cannot dominate the fitted attention coefficient.
# Consequence, kept on purpose (review 11): a group holds OBS_PER_GROUP
# consecutive chunks, so no pair qualifies once the context exceeds
# chunk x (OBS_PER_GROUP - 1) / SLOPE_MIN_SPAN_FRACTION (~123k tokens at 2,048):
# there the fit is silent, the coefficient keeps the value measured where
# pairs exist (at 25% spans the run-to-run noise is a ~10% slope error; at a
# shorter span it would dominate), and an overshoot raise decays back to it.
SLOPE_MIN_SPAN_CHUNKS = 4
SLOPE_MIN_SPAN_FRACTION = 0.25
# Observations kept per (batch, chunk) group for pair slopes, and per
# prefill-token count for the residual window: older ones age out, so the
# model follows the device instead of ratcheting on its worst moment.
OBS_PER_GROUP = 16
# A group calibrates only once this many qualifying pairs agree (median).
SLOPE_MIN_PAIRS = 3
# ratio: once calibrated, ONE observation moves a fitted coefficient by at
# most this factor up or down (Gabee Q8 2026-10-14: one first-chunk step took
# the attention coefficient 103 -> 360 and the window 207,900 -> 102,913).
MAX_STEP_FACTOR = 1.25
# ratio: the advertised window (max_model_len) ignores moves within this band
# and publishes the native window when within it of the native window (Q4
# jittered 261,769-262,144 with a constant coefficient); a move beyond it is
# published only when two different model states agree within the band.
WINDOW_BAND_RATIO = 0.03
# The residual window ignores its single largest value once it holds this
# many observations (one noisy step does not count; a repeated peak does).
ROBUST_MIN_OBS = 4
# ratio: the fitted attention coefficient never exceeds this multiple of the
# analytic prior (itself 4 score copies per query head, above every measured
# graph). 4x = 16 live score copies: a device beyond that is not a fit, it
# is a measurement defect (another lane allocating during the step).
MAX_PRIOR_MULTIPLE = 4.0
# E5 backstop: an OOM's lower bound becomes an extra margin on shapes at or
# above this fraction of the failed context, multiplied by BACKSTOP_DECAY per
# successful dispatch there (half-life ~23 chunks) and dropped once below
# BACKSTOP_DROP_FRACTION of its original size (~150 dispatches) -- a decaying
# safety term, never a fit.
BACKSTOP_CONTEXT_FRACTION = 0.75
BACKSTOP_DECAY = 0.97
BACKSTOP_DROP_FRACTION = 0.01
# A decode step at which mlx-lm grows the KV (a row's offset reaches the
# buffer's capacity; KVCache grows by KVCache.step tokens by concatenation)
# holds this many full-attention LAYERS of the whole context beyond the
# resident footprint: the layer being concatenated (old + new buffer) and
# the previous layer still referenced by the in-flight async step (mlx-lm's
# generation pipelines one step ahead).
# measured: real MLX allocator, session sim at 1/100 of Gabee, peak - resident
# on the growth step = 42.54 MB at 19,600 tokens and 43.58 MB at 20,100
# (one layer 20.07 / 20.58 MB): 2.12 layers each, of which 0.10 is the new
# KVCache.step block (charged separately); post-step +1.1 layer, freed on
# the next step. 605548ef6..f7df48332 never planned it: 1.5% over the
# ceiling at the window edge (0.4% at Gabee's 1/8 scale).
DECODE_GROWTH_LAYERS = 2.0  # measured: see above


def guard_enabled() -> bool:
    raw = os.environ.get(GUARD_ENV, "1").strip().lower()
    return raw not in ("0", "off", "false", "no")


def kv_growth_step() -> int:
    """Tokens mlx-lm's KVCache grows by when a write exceeds its buffer."""
    from mlx_lm.models.cache import KVCache

    return max(1, int(KVCache.step))


def _gb(value: float) -> float:
    return value / 1e9


@dataclass(frozen=True)
class StepShape:
    """One prefill dispatch: ``batch`` rows of ``chunk`` new tokens each,
    attending over ``context`` keys (the KV length after the chunk)."""

    batch: int
    chunk: int
    context: int

    @property
    def units(self) -> int:
        return self.batch * self.chunk * self.context

    @property
    def tokens(self) -> int:
        return self.batch * self.chunk


@dataclass(frozen=True)
class PeakPrior:
    """Analytic, conservative starting coefficients for :class:`PeakModel`."""

    # bytes per (batch x chunk x context): attention score tensors
    attention_bytes_per_unit: float
    # bytes per prefill token (batch x chunk): new KV + activations
    token_bytes: float
    # bytes per context token: one full-attention layer's KV growth copy
    # (mlx-lm KVCache grows by concatenation, so old + new buffer of the
    # layer being extended are live for a moment). Analytic, never learned.
    kv_copy_bytes_per_context: float
    # bytes per token of the whole KV/state (every layer): the learned
    # per-context term is capped at a whole copy of the KV. 0: unknown (the
    # cap is then MAX_PRIOR_MULTIPLE x one layer's copy).
    kv_bytes_per_token: float = 0.0

    @classmethod
    def from_dims(
        cls,
        *,
        query_heads: int,
        score_itemsize: int,
        hidden_size: int,
        mlp_intermediate: int,
        kv_bytes_per_token: int,
        layer_kv_bytes_per_token: int,
    ) -> PeakPrior:
        itemsize = max(2, int(score_itemsize))
        attention = max(1, int(query_heads)) * itemsize * PRIOR_SCORE_COPIES
        token = (
            max(0, int(kv_bytes_per_token))
            + max(0, int(hidden_size)) * itemsize * PRIOR_HIDDEN_COPIES
            + max(0, int(mlp_intermediate)) * itemsize * PRIOR_MLP_COPIES
        )
        return cls(
            attention_bytes_per_unit=float(attention),
            token_bytes=float(token),
            kv_copy_bytes_per_context=float(max(0, int(layer_kv_bytes_per_token))),
            kv_bytes_per_token=float(max(0, int(kv_bytes_per_token))),
        )


@dataclass
class _Obs:
    shape: StepShape
    transient: int


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


class PeakModel:
    """Projected transient bytes of one prefill dispatch.

    ``project(s) = a * s.units + g * s.context + linear(s.tokens)``

    * ``a`` (attention): the prior until one ``(batch, chunk)`` group holds
      at least ``SLOPE_MIN_PAIRS`` pairs of observations far enough apart in
      context; then the MEDIAN pair slope ``(T_j - T_i) / (X_j - X_i)`` less
      the analytic ``g``, per ``batch x chunk`` (within one group the linear
      terms cancel). Robust by construction: one noisy observation moves a
      median of many pairs by little, and only the last ``OBS_PER_GROUP``
      observations of each group count, so an outlier ages out. Capped at
      ``MAX_PRIOR_MULTIPLE`` x the analytic prior. (The first guard used the
      MAX pair slope and never forgot: one +1 GB observation at a small chunk
      took the coefficient 104 -> ~2000 B/unit for the life of the process.)
    * ``g``: one layer's KV growth copy, analytic.
    * ``linear``: model-free. For every observed ``L = batch x chunk`` the
      residual ``R = T - a*U - g*X`` (clipped at 0) of its last
      ``OBS_PER_GROUP`` observations, the largest after dropping the single
      largest once ``ROBUST_MIN_OBS`` exist (one outlier does not count; a
      repeated high peak does); monotone in ``L``; between observed sizes
      ``min(R(next larger L), R(next smaller L) * L / L_smaller)`` -- both
      valid upper bounds for any non-negative ``k*L + c``. No credit for a
      chunk smaller than any observed; linear above the largest. Before any
      observation: the prior ``k0 * L``.

    Only ``fit=True`` observations (a prefill that ran alone in its step)
    enter the slope; mixed steps (other rows decoding beside it) feed only
    the residual window. Backstop (OOM) lower bounds never enter the model:
    see :meth:`MemoryGuard.record_backstop`.
    """

    def __init__(
        self, prior: PeakPrior, *, fit_floor_tokens: int | None = None
    ) -> None:
        self.prior = prior
        # The smallest chunk whose context slope says anything about ``a``:
        # the guard's floor chunk. Honestly: that is mlx-lm's ``KVCache.step``
        # -- a library constant (256 today), read at runtime rather than
        # typed, NOT a device measurement. It is the right physics: it is the
        # granularity at which mlx-lm grows a KV buffer by concatenation, so
        # a smaller chunk usually writes inside the buffer's slack and pays
        # no growth copy, and its slope is not ``a*u + g`` (review 10: the
        # 12-48-token tails of every goose turn, plus a few measurement
        # outliers, drove ``a`` to 0.00 on Gabee -- a 2046 x 95,146 step
        # planned 4.07 GB short). The operator override
        # (``SchedulerConfig.adaptive_prefill_min_chunk_size``) is reachable
        # only programmatically: the CLI does not expose it.
        if fit_floor_tokens is None:
            try:
                fit_floor_tokens = kv_growth_step()
            except Exception:  # noqa: BLE001 -- no mlx-lm: no floor to read
                fit_floor_tokens = 1
        self.fit_floor_tokens = max(1, int(fit_floor_tokens))
        self._a_learned: float | None = None
        # The coefficient as the pair fit last set it (before any overshoot
        # raise): what a raise decays back to while the fit is silent.
        self._a_fit: float | None = None
        self._b_learned: float | None = None
        self._groups: dict[tuple[int, int], deque[_Obs]] = {}
        self._by_tokens: dict[int, deque[_Obs]] = {}
        self._envelope: list[tuple[int, float]] = []
        self.observations = 0
        self.worst_transient = 0
        self.worst_shape: StepShape | None = None

    # -- coefficients ---------------------------------------------------
    @property
    def calibrated(self) -> bool:
        return self._a_learned is not None

    @property
    def attention_bytes_per_unit(self) -> float:
        if self._a_learned is None:
            return self.prior.attention_bytes_per_unit
        return self._a_learned

    @property
    def context_bytes_per_token(self) -> float:
        """The per-context-token term ``b`` (bytes per KV token that scale
        with the context but not with the chunk): the analytic one-layer KV
        growth copy until two chunk sizes calibrate, then fitted."""
        if self._b_learned is None:
            return self.prior.kv_copy_bytes_per_context
        return self._b_learned

    def _residual(self, obs: _Obs, a: float, b: float | None = None) -> float:
        s = obs.shape
        if b is None:
            b = self.context_bytes_per_token
        return max(0.0, obs.transient - a * s.units - b * s.context)

    def _group_slopes(self) -> list[tuple[int, float]]:
        """``(batch x chunk, median dT/dX)`` of every calibrated group: within
        one group ``dT/dX = a * batch * chunk + b`` (the linear terms cancel)."""
        out: list[tuple[int, float]] = []
        for (batch, chunk), group in self._groups.items():
            if chunk < self.fit_floor_tokens:
                continue  # below the floor chunk: see ``fit_floor_tokens``
            obs = list(group)
            slopes: list[float] = []
            for i in range(len(obs)):
                for j in range(i + 1, len(obs)):
                    lo, hi = obs[i], obs[j]
                    if lo.shape.context > hi.shape.context:
                        lo, hi = hi, lo
                    span = hi.shape.context - lo.shape.context
                    need = max(
                        SLOPE_MIN_SPAN_CHUNKS * chunk,
                        SLOPE_MIN_SPAN_FRACTION * hi.shape.context,
                    )
                    if span <= 0 or span < need:
                        continue
                    slopes.append((hi.transient - lo.transient) / span)
            if len(slopes) >= SLOPE_MIN_PAIRS:
                out.append((batch * chunk, _median(slopes)))
        return out

    def _fit(self) -> tuple[float, float] | None:
        """Separate ``a`` (bytes per chunk x context) from ``b`` (bytes per
        context token) across chunk sizes: ``slope(u) = a*u + b``, least
        squares over the calibrated groups when two or more sizes exist,
        else ``b`` = the analytic layer copy. Clamped to ``[0, 4x prior]`` for
        ``a`` and ``[analytic, a whole copy of the KV]`` for ``b``."""
        g = self.prior.kv_copy_bytes_per_context
        a_cap = MAX_PRIOR_MULTIPLE * self.prior.attention_bytes_per_unit
        whole = self.prior.kv_bytes_per_token or MAX_PRIOR_MULTIPLE * g
        b_cap = max(g, whole)
        # A group whose slope is below one layer's growth copy did not pay
        # the copy on every chunk (a chunk inside the KV buffer's
        # KVCache.step slack writes in place): its slope is not
        # ``a*u + g`` and it says nothing about ``a``. Clipped to 0 it
        # VOTED 0 -- Gabee Q8 (b06acd8ea, 19:29Z): the 13-48-token tails of
        # each turn (slopes 0.5-3 kB < 4 kB) took ``a`` from ~46 to 21.6 and
        # a 1567 x 111,179 step was planned 2.01 GB short, over the ceiling.
        # Its observations still bound its own size (the residual window).
        groups = [(u, sl) for u, sl in self._group_slopes() if sl >= g]
        if not groups:
            return None
        sizes = {u for u, _s in groups}
        a = b = None
        if len(sizes) >= 2:
            n = len(groups)
            mu = sum(u for u, _s in groups) / n
            ms = sum(sl for _u, sl in groups) / n
            var = sum((u - mu) ** 2 for u, _s in groups)
            if var > 0:
                a = sum((u - mu) * (sl - ms) for u, sl in groups) / var
                b = ms - a * mu
            # A context term outside [one layer's copy, a whole copy of the
            # KV] is no separation: it is ``a`` leaking into ``b`` (a tiny
            # ``a`` with a huge, clamped ``b`` -- review 10). Fall through.
            if a is None or a < 0 or b is None or b < g or b > b_cap:
                a = b = None  # not separable on this data: fall through
        if a is None:
            # Not separable: b = the analytic copy, a = the MEDIAN over chunk
            # sizes (the first guard took the max, so one size's noise -- or
            # chunk-256's non-linearity in context -- set the coefficient).
            b = g
            a = _median([max(0.0, sl - g) / max(1, u) for u, sl in groups])
        return min(max(0.0, a), a_cap), min(max(g, b), b_cap)

    @staticmethod
    def _bounded(new: float, old: float | None) -> float:
        if old is None or old <= 0:
            return new
        return min(max(new, old / MAX_STEP_FACTOR), old * MAX_STEP_FACTOR)

    def _robust_residual(self, window: deque[_Obs], a: float) -> float:
        values = sorted(self._residual(o, a) for o in window)
        if len(values) >= ROBUST_MIN_OBS:
            return values[-2]
        return values[-1]

    def _rebuild(self) -> None:
        fitted = self._fit()
        if fitted is not None:
            a, b = fitted
            if self._a_learned is not None:
                # One observation moves a calibrated coefficient by a bounded
                # factor; a real change converges over a few observations.
                a = self._bounded(a, self._a_learned)
                b = self._bounded(b, self._b_learned)
            self._a_learned, self._b_learned = a, b
            self._a_fit = a
        elif (
            self._a_learned is not None
            and self._a_fit is not None
            and self._a_learned > self._a_fit
        ):
            # No group yields a pair spanning SLOPE_MIN_SPAN_FRACTION of the
            # context -- with OBS_PER_GROUP chunks of 2,048 that is every
            # step beyond ~131k tokens -- so the fit is silent and cannot
            # pull an overshoot raise back: it decays toward the last fitted
            # value by the raise's own bound per observation (review 11: one
            # +4 GB outlier at 140k held 48.2 -> 53.0 for good, six 64.2).
            self._a_learned = max(
                self._a_fit, self._a_learned / (1.0 + MARGIN_TRANSIENT_RATIO)
            )
        self._rebuild_envelope()

    def _rebuild_envelope(self) -> None:
        a = self.attention_bytes_per_unit
        running = 0.0
        env: list[tuple[int, float]] = []
        for tokens in sorted(self._by_tokens):
            running = max(running, self._robust_residual(self._by_tokens[tokens], a))
            env.append((tokens, running))
        self._envelope = env

    def _linear(self, tokens: int) -> float:
        env = self._envelope
        if not env:
            return self.prior.token_bytes * tokens
        lo: tuple[int, float] | None = None
        hi: tuple[int, float] | None = None
        for point in env:
            if point[0] <= tokens:
                lo = point
            if point[0] >= tokens:
                hi = point
                break
        candidates = []
        if hi is not None:
            candidates.append(hi[1])
        if lo is not None:
            candidates.append(lo[1] * tokens / max(1, lo[0]))
        return min(candidates)

    # -- public ---------------------------------------------------------
    def project(self, shape: StepShape) -> int:
        if shape.batch <= 0 or shape.chunk <= 0:
            return 0
        value = (
            self.attention_bytes_per_unit * shape.units
            + self.context_bytes_per_token * shape.context
            + self._linear(shape.tokens)
        )
        return int(math.ceil(value))

    def observe(self, shape: StepShape, transient: int, *, fit: bool = True) -> bool:
        """Record one measured dispatch. Returns True when the projection of
        some shape moved (a coefficient or the residual envelope)."""
        if shape.batch <= 0 or shape.chunk <= 0 or shape.context <= 0:
            return False
        transient = max(0, int(transient))
        self.observations += 1
        if transient > self.worst_transient:
            self.worst_transient = transient
            self.worst_shape = shape
        obs = _Obs(shape, transient)
        projected_before = self.project(shape)
        if fit:
            self._groups.setdefault(
                (shape.batch, shape.chunk), deque(maxlen=OBS_PER_GROUP)
            ).append(obs)
        self._by_tokens.setdefault(shape.tokens, deque(maxlen=OBS_PER_GROUP)).append(
            obs
        )
        before = (self._a_learned, self._b_learned, list(self._envelope))
        self._rebuild()
        self._raise_on_overshoot(shape, transient, projected_before, fit)
        return before != (self._a_learned, self._b_learned, self._envelope)

    def _raise_on_overshoot(
        self, shape: StepShape, transient: int, projected_before: int, fit: bool
    ) -> None:
        """A step at or above the floor chunk that measured beyond the
        model's own error bound (MARGIN_TRANSIENT_RATIO) over its plan RAISES
        ``a`` -- toward explaining the excess by attention, by at most that
        same error bound per observation, so repeated overshoots compound and
        a one-off moves it little. The pair medians follow only once a group
        holds enough pairs, and an under-plan is the one error that becomes
        an OOM (review 10: Gabee planned 8.53 GB, measured 12.45 GB, and the
        coefficient went 0.00 -> 0.00). If the overshoot was a one-off, the
        pair fit pulls it back one bounded step per observation -- or, where
        the fit is silent (long contexts), it decays back to the last fitted
        value (``_rebuild``)."""
        if not fit or self._a_learned is None or shape.chunk < self.fit_floor_tokens:
            return
        excess = transient - projected_before
        if excess <= MARGIN_TRANSIENT_RATIO * projected_before or shape.units <= 0:
            return
        a = self._a_learned
        a_cap = MAX_PRIOR_MULTIPLE * self.prior.attention_bytes_per_unit
        # One rule at every a, 0 included (review 11: 0 -> 5.5 while
        # 0.01 -> 0.011): a coefficient the evidence put at 0 is moved by the
        # pair fit, never by one step's excess.
        raised = min(a + excess / shape.units, a * (1.0 + MARGIN_TRANSIENT_RATIO))
        raised = min(raised, a_cap)
        if raised > a:
            self._a_learned = raised
            self._rebuild_envelope()

    def snapshot(self) -> dict[str, Any]:
        return {
            "source": "calibrated" if self.calibrated else "prior",
            "calibrated": self.calibrated,
            "observations": self.observations,
            "attention_bytes_per_unit": round(self.attention_bytes_per_unit, 3),
            "attention_bytes_per_unit_prior": round(
                self.prior.attention_bytes_per_unit, 3
            ),
            "attention_cap_multiple_of_prior": MAX_PRIOR_MULTIPLE,
            "context_bytes_per_token": round(self.context_bytes_per_token, 1),
            "kv_copy_bytes_per_context_prior": round(
                self.prior.kv_copy_bytes_per_context, 1
            ),
            "token_bytes_prior": round(self.prior.token_bytes, 1),
            "linear_bytes_by_tokens": {str(t): int(r) for t, r in self._envelope},
            "worst_transient_mb": round(self.worst_transient / 2**20, 1),
            "worst_shape": (
                None
                if self.worst_shape is None
                else {
                    "batch": self.worst_shape.batch,
                    "chunk": self.worst_shape.chunk,
                    "context": self.worst_shape.context,
                }
            ),
        }


@dataclass(frozen=True)
class RowPlan:
    """One prompt row the next dispatch may prefill."""

    request_id: str
    offset: int  # KV length already in the row's cache
    segment: int  # tokens left in the row's current segment
    chunk_cap: int = 0  # per-request chunk ceiling (0 = none), E5 backstop
    chunk_floor: int = 0  # per-request floor override (0 = guard floor)
    prompt_tokens: int = 0  # the request's full prompt, for the refusal
    # When the request arrived (its arrival_time): under pressure the LATEST
    # arrival yields (deferred), never the session that was running first.
    arrival: float = 0.0
    # Bytes of the row's cached KV its admission copies (copy-on-admit /
    # batched merge; the source stays alive until the copy is evaluated in
    # this very dispatch). Gabee Q8: +3-4 GB at 51-62k on the first chunk of
    # each turn. Context-proportional and one-off: charged to the plan as
    # such, never fitted as chunk x context.
    admit_bytes: int = 0
    # Pulled into the prompt batch by THIS dispatch (first chunk).
    admitting: bool = False


def step_shape(rows: Sequence[RowPlan], chunk: int) -> StepShape:
    """The dispatch shape mlx-lm will run with ``prefill_step_size=chunk``.

    Rows are right-padded to the longest chunk in the batch and the batch
    KV is aligned to the longest row, so the context is the longest row's
    offset plus the padded chunk.
    """
    active = [r for r in rows if r.segment > 0]
    if not active or chunk <= 0:
        return StepShape(0, 0, 0)
    per_row = [
        min(r.segment, chunk, r.chunk_cap if r.chunk_cap > 0 else chunk) for r in active
    ]
    c = max(per_row)
    x = max(r.offset for r in active) + c
    return StepShape(len(active), c, x)


@dataclass
class GuardProbe:
    """The live memory surfaces the guard reads and acts on (injected so the
    decision procedure is testable without Metal)."""

    active: Callable[[], int]
    cache: Callable[[], int]
    clear_cache: Callable[[], None]
    # Evict one prefix entry of eviction level <= ``max_level`` (the lowest
    # level first, LRU within a level; see EVICT_* below). Returns True when
    # one was evicted.
    evict_one: Callable[[int], bool]
    entry_count: Callable[[], int]
    # Bytes evicting every entry of level <= ``max_level`` would free (the
    # feasibility test runs BEFORE anything is evicted). ``None``: unknown.
    evictable: Callable[[int], int] | None = None


# The ONE prefix-cache eviction order every path uses (E1, the pressure
# evictor, E5, store-time room). What the NEXT turn of a live session needs is
# the entry stored at the END of this turn; during a turn the entry it was
# fetched from is redundant once copied into the live cache (mlx-lm merges /
# writes it into new buffers).
# 1: an entry on no live request's path (LRU) -- includes finished turns'
#    completion entries a re-rendered next prompt will not match.
EVICT_UNUSED = 1
# 2: an entry on a running request's path that it has already fetched
#    (key length <= its cached tokens): its bytes are duplicated in the live
#    cache, so evicting it frees memory and loses nothing for that session.
EVICT_REDUNDANT = 2
# 3: the HEAD of a session that finished a turn (the entry its next turn
#    fetches) other than the most recently used one (an abandoned session, a
#    finished side request; a head its OWN conversation has since diverged
#    from -- compaction, a re-rendered reply -- is level 1, it is dead
#    weight the line's next store drops): evicted -- cheapest to re-prefill
#    first, so a short side request's head goes before a long session's --
#    after the unused and redundant entries and BEFORE the chunk is reduced
#    (losing a 1.6k-token head costs one 1.6k re-prefill; a reduced chunk
#    slows every chunk of the turn). Without this stage the heads of
#    finished side requests filled the idle room and forced chunk
#    reductions on the main session (session sim, 2026-10-14).
EVICT_HEAD = 3
# 4: an entry a live session still needs: stored during this turn beyond what
#    the running request fetched, on a waiting request's path, or the most
#    recently used session head. Evicted only after the chunk is at the floor,
#    before refusing a prefill, or under hard pressure at the cap.
EVICT_NEEDED = 4


@dataclass
class GuardPlan:
    chunk: int
    shape: StepShape
    projected_transient: int
    margin: int
    footprint: int
    ceiling: int
    refuse: RowPlan | None = None
    needed: int = 0
    actions: list[str] = field(default_factory=list)
    admit_bytes: int = 0
    evicted: int = 0  # prefix entries this plan evicted


class MemoryGuard:
    """Decision procedure + bookkeeping for the step-peak guard."""

    def __init__(self, prior: PeakPrior, *, floor_tokens: int) -> None:
        self.model = PeakModel(prior, fit_floor_tokens=floor_tokens)
        # The floor: mlx-lm's KVCache grows in 256-token steps, so a chunk
        # below that buys no KV-growth relief, and each chunk re-reads every
        # weight once, so chunks this small already cost throughput. The
        # scheduler passes ``adaptive_prefill_min_chunk_size`` (256).
        self.floor_tokens = max(1, int(floor_tokens))
        self.counts: dict[str, int] = {
            "checks": 0,
            "clear_cache": 0,
            "evictions": 0,
            "chunk_halvings": 0,
            "chunks_reduced": 0,
            "refusals": 0,
            "admission_refusals": 0,
            "backstop_fired": 0,
            "backstop_requeued": 0,
            "calibrations": 0,
            "deferrals": 0,
            "store_skips": 0,
            # Of which end-of-turn (prompt + output) entries: the one a
            # session's next turn fetches. Mid-turn prompt/boundary entries
            # are skipped first under pressure, by design.
            "turn_store_skips": 0,
            # Prefix entries handed over to the request admitted from them
            # (no copy-on-admit) because the copy did not fit.
            "admission_handoffs": 0,
            # Decode steps whose KV growth would cross the ceiling even after
            # clearing the cache and evicting every entry: decoded anyway (a
            # decoding row is never aborted). Expected 0: the window and the
            # admission of every joiner include the growth.
            "decode_over_ceiling": 0,
            # Plans that evicted prefix entries and then deferred the request
            # anyway (feasibility is decided first, so expected 0).
            "evictions_then_deferred": 0,
            # Steps whose plan did not converge and dispatched unplanned
            # (a planner defect signal; expected 0).
            "plan_exhausted": 0,
            # Older entries of a session superseded by its newest
            # end-of-turn entry (dead weight, not pressure).
            "superseded": 0,
            # Entries of a conversation's dead branch (compaction, a turn
            # the template re-rendered) dropped when its next turn stored.
            "abandoned": 0,
            # Entries given back because the kernel reported machine-wide
            # memory pressure (other apps / engines need it).
            "system_pressure_evictions": 0,
        }
        self.last_chunk: int | None = None
        self.last_ceiling = 0
        self.last_margin = 0
        self.effective_prefix_budget: int | None = None
        self.configured_prefix_budget: int | None = None
        self.cache_limit: int | None = None
        # Last dispatch's live KV/state beyond the idle base + its transient:
        # reported, never fed back into a budget (the first guard derived the
        # prefix budget from the worst of these, which double-counted the live
        # KV and ratcheted prefix caching off for long sessions).
        self.last_step_footprint = 0
        # The engine's own per-step peak (every dispatched step), and how
        # many steps crossed the ceiling.
        self.last_step_peak = 0
        self.max_step_peak = 0
        self.steps_over_ceiling = 0
        # The advertised window and a pending (candidate, model version).
        self._published_window: int | None = None
        self._published_calibrated = False
        self._pending_window: tuple[int, Any] | None = None
        # E5: [context_threshold, extra_bytes, original] decaying margins.
        self._backstop: list[list[float]] = []

    # -- arithmetic -----------------------------------------------------
    def margin(self, transient: int, ceiling: int) -> int:
        return int(MARGIN_CEILING_RATIO * ceiling + MARGIN_TRANSIENT_RATIO * transient)

    def backstop_extra(self, shape: StepShape) -> int:
        extra = 0.0
        for threshold, value, _original in self._backstop:
            if shape.context >= threshold:
                extra = max(extra, value)
        return int(extra)

    def need(
        self, shape: StepShape, ceiling: int, *, backstop: bool = True
    ) -> tuple[int, int]:
        """(projected transient, margin) of one dispatch. The transient
        includes any live E5 backstop margin for this shape unless
        ``backstop=False`` (admission and the honest window are structural:
        a decaying OOM margin must not shrink the advertised context)."""
        transient = self.model.project(shape)
        if backstop:
            transient += self.backstop_extra(shape)
        return transient, self.margin(transient, ceiling)

    def record_step_peak(self, peak: int, ceiling: int) -> None:
        self.last_step_peak = max(0, int(peak))
        self.max_step_peak = max(self.max_step_peak, self.last_step_peak)
        if ceiling > 0 and self.last_step_peak > ceiling:
            self.steps_over_ceiling += 1

    def plan_transient(self, shape: StepShape, *, admit_bytes: int = 0) -> int:
        """The projected transient of one dispatch: the peak model, any live
        backstop margin, and the first-chunk copy of rows admitted with a
        shared prefix-cache hit."""
        transient, _margin = self.need(shape, max(1, self.last_ceiling))
        return transient + max(0, int(admit_bytes))

    def published_window(
        self, candidate: int, native: int | None = None, *, version: Any = None
    ) -> int:
        """The advertised context window (``max_model_len``): stable.

        Within ``WINDOW_BAND_RATIO`` of the native window it is the native
        window; a move within the band of the published value is ignored; a
        larger move is published only when two DIFFERENT model states
        (``version``, default the observation count) agree within the band.
        goose caches this per model and compacts at 0.8x it, so a dip that
        one noisy step causes must never reach it. Admission and refusals
        use the exact, unpublished figure.
        """
        candidate = max(0, int(candidate))
        if native and native > 0 and candidate >= native * (1 - WINDOW_BAND_RATIO):
            candidate = int(native)
        if version is None:
            version = self.model.observations
        published = self._published_window
        if published is None or (
            not self._published_calibrated and self.model.calibrated
        ):
            # The first figure, and the first CALIBRATED figure (the prior is
            # conservative), publish at once; after that only confirmed
            # decreases move it.
            if published != candidate and published is not None:
                logger.warning(
                    "[memory_guard] advertised window %d -> %d tokens (first "
                    "calibrated figure)",
                    published,
                    candidate,
                )
            self._published_window = candidate
            self._published_calibrated = self.model.calibrated
            self._pending_window = None
            return candidate
        if abs(candidate - published) <= WINDOW_BAND_RATIO * max(1, published):
            self._pending_window = None
            return published
        if candidate > published and self._published_calibrated:
            # Never RISES once a calibrated figure is out: goose caches the
            # first figure it reads (real Metal 124,440 -> 155,476 -> 140,512
            # -> 145,647 before this rule).
            self._pending_window = None
            return published
        pending = self._pending_window
        if (
            pending is not None
            and pending[1] != version
            and abs(candidate - pending[0]) <= WINDOW_BAND_RATIO * max(1, pending[0])
        ):
            logger.warning(
                "[memory_guard] advertised window %d -> %d tokens (confirmed by "
                "two model states)",
                published,
                candidate,
            )
            self._published_window = candidate
            self._published_calibrated = self.model.calibrated
            self._pending_window = None
            return candidate
        if pending is None or pending[1] != version:
            self._pending_window = (candidate, version)
        return published

    def record_backstop(self, shape: StepShape, lower_bound: int) -> int:
        """E5: a dispatch of ``shape`` hit a Metal OOM; it needed at least
        ``lower_bound`` bytes. The deficit over the projection becomes a
        decaying extra margin on shapes at or above
        ``BACKSTOP_CONTEXT_FRACTION`` of its context -- never a fit
        observation (the lower bound 'working set - active' can be tens of GB
        when other processes hold the GPU, and the first guard's slope took
        that as the device's attention cost). Returns the deficit recorded."""
        if shape.batch <= 0:
            return 0
        deficit = int(lower_bound) - self.model.project(shape)
        if deficit <= 0:
            return 0
        self._backstop.append(
            [
                float(BACKSTOP_CONTEXT_FRACTION * shape.context),
                float(deficit),
                float(deficit),
            ]
        )
        return deficit

    def note_dispatch_ok(self, shape: StepShape) -> None:
        """A dispatch completed: decay the backstop margins it was under."""
        if not self._backstop:
            return
        kept = []
        for threshold, value, original in self._backstop:
            if shape.context >= threshold:
                value *= BACKSTOP_DECAY
            if value >= BACKSTOP_DROP_FRACTION * original:
                kept.append([threshold, value, original])
        self._backstop = kept

    # -- E1: before every prefill chunk ---------------------------------
    def plan(
        self,
        rows: Sequence[RowPlan],
        *,
        configured_chunk: int,
        ceiling: int,
        probe: GuardProbe,
        extra_bytes: int = 0,
    ) -> GuardPlan:
        """``extra_bytes``: allocated in the same ``next()`` outside the
        prefill projection (the decoding rows' KV growth): charged like the
        admission copy, and never fitted."""
        self.counts["checks"] += 1
        self.last_ceiling = ceiling
        caps = [r.chunk_cap for r in rows if r.chunk_cap > 0]
        chunk = max(1, min([int(configured_chunk), *caps]))
        floors = [r.chunk_floor for r in rows if r.chunk_floor > 0]
        floor = max(1, min([self.floor_tokens, *floors]))
        floor = min(floor, chunk)
        actions: list[str] = []

        def footprint() -> int:
            return int(probe.active()) + int(probe.cache())

        admit = sum(max(0, int(r.admit_bytes)) for r in rows if r.segment > 0)
        admit += max(0, int(extra_bytes))

        def fits(c: int, fp: int) -> tuple[bool, StepShape, int, int]:
            shape = step_shape(rows, c)
            transient = self.plan_transient(shape, admit_bytes=admit)
            margin = self.margin(transient, ceiling)
            return fp + transient + margin <= ceiling, shape, transient, margin

        fp = footprint()
        ok, shape, transient, margin = fits(chunk, fp)
        evicted = 0

        def fits_plain(c: int, fp_: int) -> bool:
            shape_ = step_shape(rows, c)
            t, _m = self.need(shape_, ceiling, backstop=False)
            t += admit
            return fp_ + t + self.margin(t, ceiling) <= ceiling

        infeasible = False
        if not ok:
            # FEASIBILITY FIRST: if even the floor chunk does not fit with
            # the allocator cache and every evictable entry gone, nothing is
            # cleared, evicted or reduced -- the plan refuses at once and the
            # scheduler defers the latest arrival (Gabee Q8 12:22: E1 evicted
            # two entries, one the main session's 36k, then deferred anyway).
            reclaim = int(probe.cache())
            if probe.evictable is not None:
                reclaim += int(probe.evictable(EVICT_NEEDED))
            elif probe.entry_count() > 0:
                reclaim = fp  # unknown: do not pre-empt the evictions
            infeasible = not fits_plain(floor, fp - reclaim)
        if infeasible:
            chunk = floor
            ok, shape, transient, margin = fits(chunk, fp)
            actions.append(
                f"infeasible even at floor chunk {floor} with the allocator "
                "cache and every evictable entry gone: no eviction, no reduction"
            )
        if not ok and not infeasible and probe.cache() > 0:
            # (a) the allocator cache is reusable memory MLX holds; give it
            # back before touching anything that has value.
            before = fp
            probe.clear_cache()
            self.counts["clear_cache"] += 1
            fp = footprint()
            actions.append(f"clear_cache footprint {_gb(before):.2f}->{_gb(fp):.2f}GB")
            ok, shape, transient, margin = fits(chunk, fp)

        def evict(max_level: int) -> None:
            nonlocal fp, ok, shape, transient, margin, evicted
            while not ok and not infeasible and probe.entry_count() > 0:
                before = fp
                if not probe.evict_one(max_level):
                    break
                probe.clear_cache()
                self.counts["evictions"] += 1
                evicted += 1
                fp = footprint()
                actions.append(
                    f"evict prefix entry (level<={max_level}) footprint "
                    f"{_gb(before):.2f}->{_gb(fp):.2f}GB"
                )
                ok, shape, transient, margin = fits(chunk, fp)

        # (b) prefix entries no live session needs, then the entries a running
        # request already fetched (duplicated in its live cache). Session
        # heads (the entry a next turn fetches) are not among them.
        evict(EVICT_REDUNDANT)
        evict(EVICT_HEAD)
        # (c) the LARGEST chunk that fits, capped at the engine's normal
        # chunk (derived, not halved: a big Mac never reduces it; a small
        # one loses only what the headroom requires). Logged and counted
        # only when the chunk changes from the previous dispatch.
        start = chunk
        if not ok and not infeasible and chunk > floor:
            lo, hi = floor, chunk - 1
            best = floor
            while lo <= hi:
                mid = (lo + hi) // 2
                if fits(mid, fp)[0]:
                    best, lo = mid, mid + 1
                else:
                    hi = mid - 1
            chunk = best
            ok, shape, transient, margin = fits(chunk, fp)
        if chunk < start:
            self.counts["chunks_reduced"] += 1
            if chunk != self.last_chunk:
                self.counts["chunk_halvings"] += 1
                actions.append(
                    f"reduce chunk {start}->{chunk} (largest that fits; projected "
                    f"step {_gb(transient):.2f}GB at context {shape.context})"
                )
        # (b') entries a live session still needs, and session heads, go
        # before anyone is refused -- re-prefill cost order (cheapest first).
        evict(EVICT_NEEDED)
        if not ok and self.backstop_extra(shape):
            # Only a decaying E5 margin stands in the way at the floor: the
            # structural projection fits, so dispatch (a repeat OOM aborts
            # that request as before; the margin is never a reason to refuse).
            t_plain, m_plain = self.need(shape, ceiling, backstop=False)
            t_plain += admit
            m_plain = self.margin(t_plain, ceiling)
            if fp + t_plain + m_plain <= ceiling:
                ok = True
                actions.append(
                    "dispatch at the floor: only the decaying backstop margin "
                    f"({_gb(self.backstop_extra(shape)):.2f}GB) exceeds the ceiling"
                )

        plan = GuardPlan(
            chunk=chunk,
            shape=shape,
            projected_transient=transient,
            margin=margin,
            footprint=fp,
            ceiling=ceiling,
            actions=actions,
            admit_bytes=admit,
            evicted=evicted,
        )
        self.last_margin = margin
        if not ok:
            # (d) refuse BEFORE allocating. The scheduler decides: defer
            # (fits once the others finish) or refuse (cannot fit even
            # alone). The LATEST arrival yields -- a request admitted beside
            # a running one, never the session that was running first
            # (Gabee Q8 12:22: the main session was deferred behind a side
            # request because it had the longest context).
            active_rows = [r for r in rows if r.segment > 0]
            if active_rows:
                victim = max(
                    active_rows,
                    key=lambda r: (r.admitting, r.arrival, r.offset + r.segment),
                )
                plan.refuse = victim
                plan.needed = fp + transient + margin
                actions.append(
                    f"cannot dispatch request {victim.request_id[:12]} prompt="
                    f"{victim.prompt_tokens}: needs {_gb(plan.needed):.2f}GB "
                    f"> ceiling {_gb(ceiling):.2f}GB at floor chunk {chunk}"
                )
        for line in actions:
            logger.warning(
                "[memory_guard] %s | footprint=%.2fGB step=%.2fGB margin=%.2fGB "
                "ceiling=%.2fGB shape=%dx%dx%d",
                line,
                _gb(fp),
                _gb(transient),
                _gb(margin),
                _gb(ceiling),
                shape.batch,
                shape.chunk,
                shape.context,
            )
        self.last_chunk = chunk
        return plan

    def observe(
        self,
        shape: StepShape,
        transient: int,
        *,
        live_beyond_base: int = 0,
        source: str = "step",
        fit: bool = True,
        admit_bytes: int = 0,
    ) -> bool:
        """Feed one measured dispatch back. Returns True on a model change."""
        if shape.batch <= 0:
            return False
        projected = self.model.project(shape) + max(0, int(admit_bytes))
        a_before = self.model.attention_bytes_per_unit
        # A first chunk after a shared prefix hit carries the one-off copy of
        # the cached KV: take it out, and never fit the slope on such a step
        # (the copy estimate is an upper bound, so the remainder is a lower
        # one).
        if admit_bytes > 0:
            transient = max(0, int(transient) - int(admit_bytes))
            fit = False
        changed = self.model.observe(shape, transient, fit=fit)
        self.last_step_footprint = max(0, int(live_beyond_base)) + max(
            0, int(transient)
        )
        self.note_dispatch_ok(shape)
        if changed:
            self.counts["calibrations"] += 1
            # One line when the attention coefficient moves; residual-window
            # moves are frequent and visible in /v1/status.
            if self.model.attention_bytes_per_unit != a_before:
                logger.info(
                    "[memory_guard] calibrate (%s) shape=%dx%dx%d measured=%.2fGB "
                    "projected_before=%.2fGB a=%.2f->%.2fB/unit",
                    source,
                    shape.batch,
                    shape.chunk,
                    shape.context,
                    _gb(transient),
                    _gb(projected),
                    a_before,
                    self.model.attention_bytes_per_unit,
                )
        return changed

    # -- honest context / admission -------------------------------------
    def decode_growth(
        self,
        context: int,
        *,
        batch: int = 1,
        kv_footprint: Callable[[int], int] | None = None,
    ) -> int:
        """Bytes beyond the resident footprint of a decode step at which the
        KV grows (see ``DECODE_GROWTH_LAYERS``): that many full-attention
        layers of the grown buffer for every row of the decoding batch (rows
        are padded to the longest), plus the new ``KVCache.step`` block."""
        if context <= 0:
            return 0
        step = kv_growth_step()
        rows = max(1, int(batch))
        layer = self.model.prior.kv_copy_bytes_per_context
        grown = DECODE_GROWTH_LAYERS * layer * (int(context) + step) * rows
        if kv_footprint is not None:
            block = int(kv_footprint(int(context) + step)) - int(
                kv_footprint(int(context))
            )
            grown += max(0, block) * rows
        return int(math.ceil(grown))

    def single_sequence_need(
        self,
        context: int,
        *,
        base: int,
        kv_footprint: Callable[[int], int],
        ceiling: int,
    ) -> int:
        """Bytes one sequence of ``context`` tokens needs at its worst step:
        the idle base (weights), its live KV/state, and the larger of the
        floor chunk's prefill peak and a decode step that grows the KV, plus
        margin."""
        chunk = min(self.floor_tokens, max(1, context))
        shape = StepShape(1, chunk, max(1, context))
        transient, margin = self.need(shape, ceiling, backstop=False)
        decode = self.decode_growth(context, kv_footprint=kv_footprint)
        step = max(transient + margin, decode + self.margin(decode, ceiling))
        return int(base) + int(kv_footprint(context)) + step

    def max_context(
        self,
        *,
        base: int,
        kv_footprint: Callable[[int], int],
        ceiling: int,
        upper: int,
    ) -> int:
        """The largest context whose :meth:`single_sequence_need` fits
        under ``ceiling`` (0 when not even one token fits). Monotone
        non-decreasing in ``ceiling - base``."""
        if upper <= 0:
            return 0

        def ok(x: int) -> bool:
            return (
                self.single_sequence_need(
                    x, base=base, kv_footprint=kv_footprint, ceiling=ceiling
                )
                <= ceiling
            )

        if ok(upper):
            return upper
        if not ok(1):
            return 0
        lo, hi = 1, upper
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if ok(mid):
                lo = mid
            else:
                hi = mid - 1
        return lo

    # -- E2/E3 ----------------------------------------------------------
    @staticmethod
    def idle_room(*, ceiling: int, base: int, fraction: float) -> int:
        """Bytes the prefix entries (and the allocator cache) may hold while
        the engine is IDLE: ``fraction x ceiling - idle base``. ``fraction``
        is the pressure evictor's own trigger (under the guard 1 - the static
        margin), so an idle engine never sits above it and nothing is
        evicted between turns. Live KV is NOT charged here: during a turn
        the guard makes room on demand (E1 evicts the entry the turn was
        fetched from -- or the next turn takes it over -- then halves the
        chunk)."""
        return int(fraction * ceiling - base)

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "ceiling_mb": round(self.last_ceiling / 2**20, 1),
            "coefficients": {
                "source": "calibrated" if self.model.calibrated else "prior",
                "observations": self.model.observations,
                "attention_bytes_per_chunk_context": round(
                    self.model.attention_bytes_per_unit, 3
                ),
                "context_bytes_per_token": round(self.model.context_bytes_per_token, 1),
                "linear_bytes_by_tokens": {
                    str(t): int(r) for t, r in self.model._envelope
                },
            },
            "margin_mb": round(self.last_margin / 2**20, 1),
            "margin_ratios": {
                "ceiling": MARGIN_CEILING_RATIO,
                "step_transient": MARGIN_TRANSIENT_RATIO,
            },
            "floor_chunk_tokens": self.floor_tokens,
            "last_chunk_tokens": self.last_chunk,
            "peak_model": self.model.snapshot(),
            "last_step_footprint_mb": round(self.last_step_footprint / 2**20, 1),
            "last_step_peak_mb": round(self.last_step_peak / 2**20, 1),
            "max_step_peak_mb": round(self.max_step_peak / 2**20, 1),
            "steps_over_ceiling": self.steps_over_ceiling,
            "backstop_margins": [
                {"from_context": int(t), "extra_mb": round(v / 2**20, 1)}
                for t, v, _o in self._backstop
            ],
            "effective_prefix_budget_mb": (
                None
                if self.effective_prefix_budget is None
                else round(self.effective_prefix_budget / 2**20, 1)
            ),
            "configured_prefix_budget_mb": (
                None
                if self.configured_prefix_budget is None
                else round(self.configured_prefix_budget / 2**20, 1)
            ),
            "allocator_cache_limit_mb": (
                None if self.cache_limit is None else round(self.cache_limit / 2**20, 1)
            ),
            "actions": dict(self.counts),
            "refusals": self.counts["refusals"] + self.counts["admission_refusals"],
            "backstop_firings": self.counts["backstop_fired"],
        }


def is_metal_oom(error: BaseException) -> bool:
    """True for the Metal command-buffer out-of-memory failure."""
    text = str(error)
    lowered = text.lower()
    return (
        "insufficient memory" in lowered
        or "outofmemory" in lowered.replace(" ", "")
        or "kiogpucommandbuffercallbackerroroutofmemory" in lowered
    )
