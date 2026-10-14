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
# Observations kept per (batch, chunk) group for pair slopes.
OBS_PER_GROUP = 16


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


class PeakModel:
    """Projected transient bytes of one prefill dispatch.

    ``project(s) = a * s.units + g * s.context + linear(s.tokens)``

    * ``a`` (attention): the prior until two observations of the same
      ``(batch, chunk)`` lie far enough apart in context; then the WORST
      (largest) pair slope ``(T_j - T_i) / (X_j - X_i)`` less the analytic
      ``g``, per ``batch x chunk``. Within one group the linear terms cancel,
      so the slope is not inflated by activations. It never decays.
    * ``g``: one layer's KV growth copy, analytic.
    * ``linear``: model-free. For every observed ``L = batch x chunk`` the
      residual ``R = T - a*U - g*X`` (clipped at 0) is kept as a running max,
      made monotone in ``L``; between observed sizes the bound is
      ``min(R(next larger L), R(next smaller L) * L / L_smaller)`` -- both are
      valid upper bounds for any non-negative ``k*L + c``. Below the smallest
      observed size no credit is taken for a smaller chunk; above the largest
      it scales linearly. Before any observation: the prior ``k0 * L``.

    By construction every observation is covered: ``project(obs) >= T``.
    Projection cost is O(number of distinct observed L), a handful.
    """

    def __init__(self, prior: PeakPrior) -> None:
        self.prior = prior
        self._a_learned: float | None = None
        self._residual_by_tokens: dict[int, float] = {}
        self._groups: dict[tuple[int, int], deque[_Obs]] = {}
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

    def _rebuild_envelope(self) -> None:
        running = 0.0
        env: list[tuple[int, float]] = []
        for tokens in sorted(self._residual_by_tokens):
            running = max(running, self._residual_by_tokens[tokens])
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

    def observe(self, shape: StepShape, transient: int) -> bool:
        """Record one measured dispatch. Returns True when a coefficient moved."""
        if shape.batch <= 0 or shape.chunk <= 0 or shape.context <= 0:
            return False
        transient = max(0, int(transient))
        self.observations += 1
        if transient > self.worst_transient:
            self.worst_transient = transient
            self.worst_shape = shape
        obs = _Obs(shape, transient)
        key = (shape.batch, shape.chunk)
        group = self._groups.setdefault(key, deque(maxlen=OBS_PER_GROUP))
        changed = False

        # 1) attention coefficient from pair slopes within the group.
        best: float | None = None
        g = self.prior.kv_copy_bytes_per_context
        for other in group:
            lo, hi = (
                (other, obs) if other.shape.context < shape.context else (obs, other)
            )
            span = hi.shape.context - lo.shape.context
            need = max(
                SLOPE_MIN_SPAN_CHUNKS * shape.chunk,
                SLOPE_MIN_SPAN_FRACTION * hi.shape.context,
            )
            if span <= 0 or span < need:
                continue
            slope = (hi.transient - lo.transient) / span
            a_pair = max(0.0, slope - g) / max(1, shape.batch * shape.chunk)
            best = a_pair if best is None else max(best, a_pair)
        group.append(obs)
        if best is not None and (self._a_learned is None or best > self._a_learned):
            first = self._a_learned is None
            self._a_learned = best
            changed = True
            if first:
                # The prior (usually larger) is replaced: recompute every
                # residual from the stored observations so each one stays
                # covered under the new, smaller attention term.
                self._residual_by_tokens = {}
                for grp in self._groups.values():
                    for o in grp:
                        r = self._residual(o, best)
                        tok = o.shape.tokens
                        self._residual_by_tokens[tok] = max(
                            self._residual_by_tokens.get(tok, 0.0), r
                        )

        # 2) model-free linear residual for this size.
        r = self._residual(obs, self.attention_bytes_per_unit)
        tok = shape.tokens
        if r > self._residual_by_tokens.get(tok, -1.0):
            self._residual_by_tokens[tok] = r
            changed = True
        if changed:
            self._rebuild_envelope()
        return changed

    def snapshot(self) -> dict[str, Any]:
        return {
            "calibrated": self.calibrated,
            "observations": self.observations,
            "attention_bytes_per_unit": round(self.attention_bytes_per_unit, 3),
            "attention_bytes_per_unit_prior": round(
                self.prior.attention_bytes_per_unit, 3
            ),
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
    # Evict one prefix entry, oldest first; ``keep_newest`` refuses to take
    # the last remaining entry. Returns True when one was evicted.
    evict_one: Callable[[bool], bool]
    entry_count: Callable[[], int]


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
        }
        self.last_chunk: int | None = None
        self.last_ceiling = 0
        self.last_margin = 0
        self.effective_prefix_budget: int | None = None
        self.configured_prefix_budget: int | None = None
        self.cache_limit: int | None = None
        # Worst observed non-evictable footprint of one dispatch beyond the
        # idle base: the request's live KV/state at that moment + its
        # transient. Drives the derived prefix budget and cache limit (E2/E3).
        self.worst_step_footprint = 0

    # -- arithmetic -----------------------------------------------------
    def margin(self, transient: int, ceiling: int) -> int:
        return int(MARGIN_CEILING_RATIO * ceiling + MARGIN_TRANSIENT_RATIO * transient)

    def need(self, shape: StepShape, ceiling: int) -> tuple[int, int]:
        """(projected transient, margin) of one dispatch."""
        transient = self.model.project(shape)
        return transient, self.margin(transient, ceiling)

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

        def evict(keep_newest: bool) -> None:
            nonlocal fp, ok, shape, transient, margin
            while not ok and probe.entry_count() > (1 if keep_newest else 0):
                before = fp
                if not probe.evict_one(keep_newest):
                    break
                probe.clear_cache()
                self.counts["evictions"] += 1
                fp = footprint()
                actions.append(
                    f"evict prefix entry (oldest) footprint "
                    f"{_gb(before):.2f}->{_gb(fp):.2f}GB"
                )
                ok, shape, transient, margin = fits(chunk, fp)

        # (b) prefix entries yield oldest-first -- but the newest entry (the
        # session's own next-turn prefix) is kept until halving the chunk
        # has been tried: a smaller chunk costs some prefill throughput,
        # losing that entry costs the next turn a full re-prefill.
        evict(keep_newest=True)
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
        # (b') the last entry goes too before anyone is refused.
        evict(keep_newest=False)

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
                victim = max(active_rows, key=lambda r: r.offset + r.segment)
                plan.refuse = victim
                plan.needed = fp + transient + margin
                self.counts["refusals"] += 1
                actions.append(
                    f"REFUSE request {victim.request_id[:12]} prompt="
                    f"{victim.prompt_tokens} needs {_gb(plan.needed):.2f}GB "
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
    ) -> bool:
        """Feed one measured dispatch back. Returns True on a model change."""
        if shape.batch <= 0:
            return False
        projected = self.model.project(shape)
        changed = self.model.observe(shape, transient)
        footprint = max(0, int(live_beyond_base)) + max(0, int(transient))
        if footprint > self.worst_step_footprint:
            self.worst_step_footprint = footprint
            changed = True
        if changed:
            self.counts["calibrations"] += 1
            logger.info(
                "[memory_guard] calibrate (%s) shape=%dx%dx%d measured=%.2fGB "
                "projected_before=%.2fGB a=%.2fB/unit worst_step=%.2fGB",
                source,
                shape.batch,
                shape.chunk,
                shape.context,
                _gb(transient),
                _gb(projected),
                self.model.attention_bytes_per_unit,
                _gb(self.worst_step_footprint),
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
        transient, margin = self.need(shape, ceiling)
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
    def derived_headroom(self, *, ceiling: int, base: int) -> int:
        """Bytes left beside the idle base, the worst dispatch seen and the
        margin. Both the prefix budget and the allocator cache live here."""
        worst = self.worst_step_footprint
        return int(ceiling - base - worst - self.margin(worst, ceiling))

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
            "worst_step_footprint_mb": round(self.worst_step_footprint / 2**20, 1),
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
