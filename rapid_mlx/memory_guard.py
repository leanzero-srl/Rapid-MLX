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
* :class:`MemoryGuard` -- the decision procedure run before EVERY prefill
  chunk (clear the allocator cache, evict prefix entries oldest-first, halve
  the chunk to a floor, refuse), the honest max context, the admission
  check, the derived prefix-cache budget and allocator cache limit, and the
  counters ``/v1/status`` reports.

Every number is derived from the device (the scheduler's Metal cap), the
model config, or measurement. The ratios below carry their receipts.
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
# Re-apply the allocator cache limit / prefix budget only when it moves by
# more than this fraction (avoids a setter call per chunk while a ladder of
# observations nudges the model up).
REAPPLY_RATIO = 0.05

# Pair slopes are trusted only across a context span of at least this many
# chunks AND this fraction of the larger context, so measurement noise in the
# peak (a few hundred MB) cannot dominate the fitted attention coefficient.
SLOPE_MIN_SPAN_CHUNKS = 4
SLOPE_MIN_SPAN_FRACTION = 0.25
# Observations kept per (batch, chunk) group for pair slopes, and per
# prefill-token count for the residual window: older ones age out, so the
# model follows the device instead of ratcheting on its worst moment.
OBS_PER_GROUP = 16
# A group calibrates only once this many qualifying pairs agree (median).
SLOPE_MIN_PAIRS = 3
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
# successful dispatch there (half-life ~23 chunks) and dropped below
# BACKSTOP_FLOOR_RATIO of the ceiling -- a decaying safety term, never a fit.
BACKSTOP_CONTEXT_FRACTION = 0.75
BACKSTOP_DECAY = 0.97
BACKSTOP_FLOOR_RATIO = 0.001


def guard_enabled() -> bool:
    raw = os.environ.get(GUARD_ENV, "1").strip().lower()
    return raw not in ("0", "off", "false", "no")


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

    def __init__(self, prior: PeakPrior) -> None:
        self.prior = prior
        self._a_learned: float | None = None
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

    def _residual(self, obs: _Obs, a: float) -> float:
        s = obs.shape
        return max(
            0.0,
            obs.transient
            - a * s.units
            - self.prior.kv_copy_bytes_per_context * s.context,
        )

    def _fit_attention(self) -> float | None:
        g = self.prior.kv_copy_bytes_per_context
        cap = MAX_PRIOR_MULTIPLE * self.prior.attention_bytes_per_unit
        best: float | None = None
        for (batch, chunk), group in self._groups.items():
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
                    slope = (hi.transient - lo.transient) / span
                    slopes.append(max(0.0, slope - g) / max(1, batch * chunk))
            if len(slopes) < SLOPE_MIN_PAIRS:
                continue
            estimate = _median(slopes)
            best = estimate if best is None else max(best, estimate)
        if best is None:
            return None
        return min(best, cap)

    def _robust_residual(self, window: deque[_Obs], a: float) -> float:
        values = sorted(self._residual(o, a) for o in window)
        if len(values) >= ROBUST_MIN_OBS:
            return values[-2]
        return values[-1]

    def _rebuild(self) -> None:
        self._a_learned = self._fit_attention()
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
            + self.prior.kv_copy_bytes_per_context * shape.context
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
        if fit:
            self._groups.setdefault(
                (shape.batch, shape.chunk), deque(maxlen=OBS_PER_GROUP)
            ).append(obs)
        self._by_tokens.setdefault(shape.tokens, deque(maxlen=OBS_PER_GROUP)).append(
            obs
        )
        before = (self._a_learned, list(self._envelope))
        self._rebuild()
        return before != (self._a_learned, self._envelope)

    def snapshot(self) -> dict[str, Any]:
        return {
            "calibrated": self.calibrated,
            "observations": self.observations,
            "attention_bytes_per_unit": round(self.attention_bytes_per_unit, 3),
            "attention_bytes_per_unit_prior": round(
                self.prior.attention_bytes_per_unit, 3
            ),
            "attention_cap_multiple_of_prior": MAX_PRIOR_MULTIPLE,
            "kv_copy_bytes_per_context": round(self.prior.kv_copy_bytes_per_context, 1),
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
# 3: an entry a live session still needs: stored during this turn beyond what
#    the running request fetched (the next turn's base), or on a waiting
#    request's path. Evicted only before refusing a prefill or under hard
#    pressure at the cap.
EVICT_NEEDED = 3


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


class MemoryGuard:
    """Decision procedure + bookkeeping for the step-peak guard."""

    def __init__(self, prior: PeakPrior, *, floor_tokens: int) -> None:
        self.model = PeakModel(prior)
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
        # E5: [context_threshold, extra_bytes] decaying safety margins.
        self._backstop: list[list[float]] = []

    # -- arithmetic -----------------------------------------------------
    def margin(self, transient: int, ceiling: int) -> int:
        return int(MARGIN_CEILING_RATIO * ceiling + MARGIN_TRANSIENT_RATIO * transient)

    def backstop_extra(self, shape: StepShape) -> int:
        extra = 0.0
        for threshold, value in self._backstop:
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
            [float(BACKSTOP_CONTEXT_FRACTION * shape.context), float(deficit)]
        )
        return deficit

    def note_dispatch_ok(self, shape: StepShape) -> None:
        """A dispatch completed: decay the backstop margins it was under."""
        if not self._backstop:
            return
        floor = BACKSTOP_FLOOR_RATIO * max(1, self.last_ceiling)
        kept = []
        for threshold, value in self._backstop:
            if shape.context >= threshold:
                value *= BACKSTOP_DECAY
            if value >= floor:
                kept.append([threshold, value])
        self._backstop = kept

    # -- E1: before every prefill chunk ---------------------------------
    def plan(
        self,
        rows: Sequence[RowPlan],
        *,
        configured_chunk: int,
        ceiling: int,
        probe: GuardProbe,
    ) -> GuardPlan:
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

        def fits(c: int, fp: int) -> tuple[bool, StepShape, int, int]:
            shape = step_shape(rows, c)
            transient, margin = self.need(shape, ceiling)
            return fp + transient + margin <= ceiling, shape, transient, margin

        fp = footprint()
        ok, shape, transient, margin = fits(chunk, fp)
        if not ok and probe.cache() > 0:
            # (a) the allocator cache is reusable memory MLX holds; give it
            # back before touching anything that has value.
            before = fp
            probe.clear_cache()
            self.counts["clear_cache"] += 1
            fp = footprint()
            actions.append(f"clear_cache footprint {_gb(before):.2f}->{_gb(fp):.2f}GB")
            ok, shape, transient, margin = fits(chunk, fp)

        def evict(max_level: int) -> None:
            nonlocal fp, ok, shape, transient, margin
            while not ok and probe.entry_count() > 0:
                before = fp
                if not probe.evict_one(max_level):
                    break
                probe.clear_cache()
                self.counts["evictions"] += 1
                fp = footprint()
                actions.append(
                    f"evict prefix entry (level<={max_level}) footprint "
                    f"{_gb(before):.2f}->{_gb(fp):.2f}GB"
                )
                ok, shape, transient, margin = fits(chunk, fp)

        # (b) prefix entries no live session needs, then the entries a running
        # request already fetched (duplicated in its live cache). The entry
        # the next turn will fetch is not among them.
        evict(EVICT_REDUNDANT)
        # (c) halve the chunk down to the floor. Every chunk of a long
        # prefill under pressure re-derives the same halving from the
        # configured size; it is logged and counted only when the chunk
        # actually changes from the previous dispatch (no per-chunk spam).
        start = chunk
        halvings: list[str] = []
        while not ok and chunk > floor:
            new = max(floor, chunk // 2)
            halvings.append(
                f"halve chunk {chunk}->{new} (projected step "
                f"{_gb(transient):.2f}GB at context {shape.context})"
            )
            chunk = new
            ok, shape, transient, margin = fits(chunk, fp)
        if chunk < start:
            self.counts["chunks_reduced"] += 1
        if halvings and chunk != self.last_chunk:
            actions.extend(halvings)
            self.counts["chunk_halvings"] += len(halvings)
        # (b') entries a live session still needs go before anyone is refused.
        evict(EVICT_NEEDED)
        if not ok and self.backstop_extra(shape):
            # Only a decaying E5 margin stands in the way at the floor: the
            # structural projection fits, so dispatch (a repeat OOM aborts
            # that request as before; the margin is never a reason to refuse).
            t_plain, m_plain = self.need(shape, ceiling, backstop=False)
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
        )
        self.last_margin = margin
        if not ok:
            # (d) refuse BEFORE allocating: the row with the longest context.
            active_rows = [r for r in rows if r.segment > 0]
            if active_rows:
                # The scheduler decides: defer (fits once the others finish)
                # or refuse (cannot fit even alone).
                victim = max(active_rows, key=lambda r: r.offset + r.segment)
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
    ) -> bool:
        """Feed one measured dispatch back. Returns True on a model change."""
        if shape.batch <= 0:
            return False
        projected = self.model.project(shape)
        a_before = self.model.attention_bytes_per_unit
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
    def single_sequence_need(
        self,
        context: int,
        *,
        base: int,
        kv_footprint: Callable[[int], int],
        ceiling: int,
    ) -> int:
        """Bytes one sequence of ``context`` tokens needs at its last chunk:
        the idle base (weights), its live KV/state, and the floor chunk's
        step peak plus margin."""
        chunk = min(self.floor_tokens, max(1, context))
        shape = StepShape(1, chunk, max(1, context))
        transient, margin = self.need(shape, ceiling, backstop=False)
        return int(base) + int(kv_footprint(context)) + transient + margin

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
        is the pressure evictor's own trigger (0.9), so an idle engine never
        sits above it and nothing is evicted between turns. Live KV is NOT
        charged here: during a turn the guard makes room on demand (E1 evicts
        the entry the turn was fetched from, then halves the chunk)."""
        return int(fraction * ceiling - base)

    def snapshot(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "ceiling_mb": round(self.last_ceiling / 2**20, 1),
            "margin_mb": round(self.last_margin / 2**20, 1),
            "margin_ratios": {
                "ceiling": MARGIN_CEILING_RATIO,
                "step_transient": MARGIN_TRANSIENT_RATIO,
            },
            "floor_chunk_tokens": self.floor_tokens,
            "last_chunk_tokens": self.last_chunk,
            "peak_model": self.model.snapshot(),
            "last_step_footprint_mb": round(self.last_step_footprint / 2**20, 1),
            "backstop_margins": [
                {"from_context": int(t), "extra_mb": round(v / 2**20, 1)}
                for t, v in self._backstop
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
