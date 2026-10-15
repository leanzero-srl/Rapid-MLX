# SPDX-License-Identifier: Apache-2.0
"""An MTP decode step planned from the MODEL, not from a measurement.

Gabee T12 control run (2026-10-15, Qwen3.8-27B Q8 + mtp.safetensors, engine
e84498a4a): decode steps at 84-86k planned 45.2-46.4 GB and peaked
50.15-50.24 GB against a 50.10 GB ceiling, on CONSECUTIVE steps ~0.55 s
apart (4 in a ~546-token turn) -- not KV growth (once per 256 tokens).

The cost (independent review, measured on Metal with the 27B's shapes): an
MTP verify of S rows keeps S - 1 restore points -- the recurrent state and
the conv slice -- of every linear-attention (GatedDeltaNet) layer until the
round ends (spec_decode/mtp/cache_patch.py, ``gated_delta_verify_with_
states``). FIXED, independent of the context: 32 rows (a prompt-lookup
copy draft) = 31 x 48 layers x (48 x 128 x 128 x 4 B + 3 x 10,240 x 2 B)
= 4.77 GB (the +4.85 GB crossings); 4 rows (k = 3) = 0.46 GB (the
<= 0.4 GB the run's other decode steps showed). The copy-draft width is
adaptive, which is why the crossings were intermittent.

And why a plan of exactly that cost still crossed in the 1/100 sim: a
prefix entry stored mid-turn from a live cache with no slack SHARED its K/V
buffers (``mx.contiguous`` does not copy a whole contiguous buffer), so
the bytes the store's room check made room for were allocated at the live
cache's next write -- the first decode step of every turn (+60.5 MB at
7,100 tokens against 17.2 MB planned). RC6's slack hid it; a fixed cost
planned beside it crossed. The buffers stay shared and the guard plans
them until that write (tests/test_memory_guard_shared_store.py).
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from rapid_mlx.memory_guard import MemoryGuard, PeakPrior, kv_growth_step
from rapid_mlx.scheduler import (
    Scheduler,
    _held_rollback_bytes,
    _model_recurrent_state_bytes,
    _recurrent_state_bytes_per_row,
)
from tests.test_memory_guard_session import (  # noqa: F401
    GABEE_PHYSICAL,
    M4_MAX_128_CEILING,
    _auto_budget_scheduler,
    _Device,
    _no_metal,
    _session,
    cpu,
    llama,
)

GB = 10**9
CEILING = int(50.10 * GB)
FOOTPRINT = int(45.37 * GB)
CONTEXT = 83_666
LAYER = 4096.0  # one full-attention layer's K+V per token
PER_TOKEN = 73.0 * 1024
BIG_CEILING = int(0.9 * 115.4 * GB)
# Qwen3.8-27B (Gabee's config.json): 64 layers, one full-attention in four
# -> 48 GatedDeltaNet layers; 48 value / 16 key heads of 128; conv kernel 4.
LINEAR_LAYERS = 48
V_HEADS, K_HEADS, HEAD_DIM, KERNEL = 48, 16, 128, 4
CONV_DIM = 2 * K_HEADS * HEAD_DIM + V_HEADS * HEAD_DIM


def _guard() -> MemoryGuard:
    return MemoryGuard(
        PeakPrior(192.0, 251_904.0, LAYER, PER_TOKEN), floor_tokens=kv_growth_step()
    )


def _kv(context: int) -> int:
    return int(PER_TOKEN * context)


class _Model(SimpleNamespace):
    def named_modules(self):
        return self.modules


def _gdn():
    return SimpleNamespace(
        num_v_heads=V_HEADS,
        head_v_dim=HEAD_DIM,
        head_k_dim=HEAD_DIM,
        conv_dim=CONV_DIM,
        conv_kernel_size=KERNEL,
        norm=SimpleNamespace(weight=mx.zeros((1,), mx.bfloat16)),
    )


def _model_27b(wide: bool = True, lookup: int | None = 31):
    from rapid_mlx.spec_decode.mtp.prompt_lookup import PromptLookupPolicy

    model = _Model(modules=[(f"layers.{i}.linear_attn", _gdn()) for i in range(48)])
    if wide:
        model.mtp_wide_verify_rollback_supported = True
    if lookup is not None:
        model.mtp_prompt_lookup_supported = True
        model.mtp_prompt_lookup_policy = PromptLookupPolicy(
            enabled_by_default=True,
            enabled_under_sampling=True,
            min_ngram=8,
            max_ngram=64,
            max_tokens=lookup,
        )
    return model


def _scheduler(spec="mtp", *, max_k=3, model=None):
    s = object.__new__(Scheduler)
    s.config = SimpleNamespace(spec_decode=spec, mtp_max_k=max_k)
    s.model = model if model is not None else _model_27b()
    s.batch_generator = SimpleNamespace(_generation_batch=None)
    return s


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("RAPID_MLX_MTP_PROMPT_LOOKUP", raising=False)
    monkeypatch.delenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MAX_TOKENS", raising=False)


def test_the_27b_restore_point_is_read_from_the_model():
    per = _model_recurrent_state_bytes(_model_27b())
    state = V_HEADS * HEAD_DIM * HEAD_DIM * 4  # mlx-lm keeps it in float32
    conv = (KERNEL - 1) * CONV_DIM * 2  # bf16 activations
    assert per == LINEAR_LAYERS * (state + conv)


@pytest.mark.parametrize(
    ("max_k", "lookup", "gb"),
    [(3, 31, 4.77), (3, None, 0.46)],
    ids=["copy-draft-32-rows", "mtp-k3-4-rows"],
)
def test_the_verify_cost_is_the_measured_one(max_k, lookup, gb):
    """4.77 GB for a 32-row copy-draft verify (Gabee's +4.85 GB crossings,
    the review's 97.5 MB per layer on Metal), 0.46 GB at k = 3 (the run's
    <= 0.4 GB MTP decode steps)."""
    s = _scheduler(max_k=max_k, model=_model_27b(lookup=lookup))
    cost = s._memguard_verify_bytes_per_row(s._memguard_decode_write_rows())
    assert cost == pytest.approx(gb * GB, rel=0.01), cost / GB


def test_no_cost_off_mtp_or_without_the_wide_verify():
    assert _scheduler("none")._memguard_verify_bytes_per_row(1) == 0
    s = _scheduler(model=_model_27b(wide=False))
    assert s._memguard_verify_bytes_per_row(s._memguard_decode_write_rows()) == 0


def test_the_live_cache_gives_the_same_restore_point(cpu):
    """From the decoding cache's own arrays (exact): one row's conv + SSM
    state per recurrent layer."""
    from mlx_lm.models.cache import ArraysCache, KVCache

    rows = 2
    layer = ArraysCache(size=2)
    layer[0] = mx.zeros((rows, KERNEL - 1, 64), mx.bfloat16)
    layer[1] = mx.zeros((rows, 4, 16, 16), mx.float32)
    kv = KVCache()
    assert _recurrent_state_bytes_per_row([layer, kv]) == (
        (KERNEL - 1) * 64 * 2 + 4 * 16 * 16 * 4
    )


# -- the plan --------------------------------------------------------------


class _Batch(SimpleNamespace):
    def __len__(self) -> int:
        return len(self.uids)


def _decoding(s, *, capacity: int, idx: int, context: int, held=None):
    from mlx_lm.models.cache import ArraysCache, BatchKVCache

    attn = BatchKVCache(left_padding=[0])
    k = mx.zeros((1, 1, capacity, 8))
    attn.keys, attn.values = k, k
    attn._idx = idx
    attn.offset = mx.array([idx])
    gdn = ArraysCache(size=2)
    gdn.rollback_state = held
    request = SimpleNamespace(prompt_token_ids=[0] * context, num_output_tokens=0)
    s.batch_generator = SimpleNamespace(
        _generation_batch=_Batch(uids=[7], prompt_cache=[attn, gdn])
    )
    s._memguard_request_for_uid = lambda uid: request
    s._kv_footprint_fn = lambda: _kv
    # the model's restore point, not the toy cache's
    s._memguard_state_bytes_per_row = _model_recurrent_state_bytes(_model_27b())


def test_every_mtp_decode_step_plans_the_verify(cpu):
    guard = _guard()
    s = _scheduler(max_k=3)
    _decoding(s, capacity=CONTEXT + 200, idx=CONTEXT, context=CONTEXT)
    planned = s._memguard_decode_growth(guard)
    assert planned == pytest.approx(4.77 * GB, rel=0.01)
    assert s._memguard_decode_growth(guard) == guard.decode_verify(batch=1)
    # Gabee: footprint + verify + margin does not fit -> room is made
    assert FOOTPRINT + planned + guard.margin(planned, CEILING) > CEILING
    # 128 GB: it fits, no protective action
    assert FOOTPRINT + planned + guard.margin(planned, BIG_CEILING) < BIG_CEILING


def test_the_previous_rounds_restore_points_are_not_counted_twice(cpu):
    """The verify drops the last round's restore points as it writes its
    own: what is still held is subtracted."""
    guard = _guard()
    s = _scheduler(max_k=3)
    held = [(mx.zeros((1, 3, 64)), mx.zeros((1, 48, 128, 128)))]
    _decoding(s, capacity=CONTEXT + 200, idx=CONTEXT, context=CONTEXT, held=held)
    assert _held_rollback_bytes(s.batch_generator._generation_batch.prompt_cache) > 0
    assert s._memguard_decode_growth(guard) == guard.decode_verify(
        batch=1
    ) - _held_rollback_bytes(s.batch_generator._generation_batch.prompt_cache)


def test_an_mtp_verify_near_capacity_is_a_growth_step(cpu):
    """3 positions short of the buffer, a 4-row verify (k = 3): the KV
    grows; ``idx + 1 > capacity`` never saw it."""
    guard = _guard()
    s = _scheduler(max_k=3, model=_model_27b(lookup=None))
    _decoding(s, capacity=CONTEXT + 3, idx=CONTEXT, context=CONTEXT)
    planned = s._memguard_decode_growth(guard)
    assert planned == guard.decode_growth(CONTEXT, kv_footprint=_kv)
    assert planned >= 2 * LAYER * CONTEXT + guard.decode_verify(batch=1)


def test_a_plain_decode_is_planned_exactly_as_rc6(cpu):
    guard = _guard()
    s = _scheduler("none")
    _decoding(s, capacity=CONTEXT + 3, idx=CONTEXT, context=CONTEXT)
    assert s._memguard_decode_growth(guard) == 0
    assert guard.decode_verify_bytes_per_row == 0
    _decoding(s, capacity=CONTEXT, idx=CONTEXT, context=CONTEXT)
    rc6 = int(2.0 * LAYER * (CONTEXT + kv_growth_step())) + (
        _kv(CONTEXT + kv_growth_step()) - _kv(CONTEXT)
    )
    assert s._memguard_decode_growth(guard) == pytest.approx(rc6, abs=1)


def test_the_window_includes_the_verify_on_gabee_not_on_128():
    """The window's worst step is the larger of the floor-chunk prefill and
    the decode step; the verify joins the decode step. At Gabee's prior
    the floor-chunk prefill still binds (no window lost); a cost beyond it
    shrinks the window to what fits; 128 GB keeps its native window."""
    guard = _guard()
    base = int(31.8 * GB)

    def window(ceiling):
        return guard.max_context(
            base=base, kv_footprint=_kv, ceiling=ceiling, upper=262_144
        )

    before = window(CEILING)
    guard.decode_verify_bytes_per_row = int(4.77 * GB)
    assert window(CEILING) <= before
    guard.decode_verify_bytes_per_row = int(12 * GB)
    after = window(CEILING)
    assert after < before
    need = guard.single_sequence_need(
        after, base=base, kv_footprint=_kv, ceiling=CEILING
    )
    assert need <= CEILING
    assert need >= base + _kv(after) + guard.decode_verify(batch=1)
    guard.decode_verify_bytes_per_row = int(4.77 * GB)
    assert window(BIG_CEILING) == 262_144


# -- the whole path: the review's fixed-cost sim -----------------------------


def _fixed_verify(scheduler, nbytes: int) -> None:
    """Every decode step holds a FIXED ``nbytes`` (an MTP verify's restore
    points: context-independent), as a real allocation, and the guard
    plans exactly that (as it does from the model on an MTP engine)."""
    scheduler._memguard_verify_bytes_per_row = lambda writes: nbytes
    scheduler._memguard().decode_verify_bytes_per_row = nbytes
    real_step = scheduler.step

    def step():
        bg = scheduler.batch_generator
        if bg is not None and not getattr(bg, "_lz_fixed", False):
            real_next = bg.next

            def next_():
                gen = getattr(bg, "_generation_batch", None)
                prompting = len(getattr(bg, "_prompt_batch", ()) or ()) or len(
                    getattr(bg, "_unprocessed_sequences", ()) or ()
                )
                decoding = gen is not None and len(gen) > 0 and not prompting
                out = real_next()
                if decoding:
                    extra = mx.zeros((nbytes,), mx.uint8)
                    mx.eval(extra)
                    del extra
                return out

            bg.next = next_
            bg._lz_fixed = True
        return real_step()

    scheduler.step = step


def _actions(guard) -> dict[str, int]:
    rc6_bookkeeping = {
        "checks",
        "calibrations",
        "superseded",
        "abandoned",
        # the plain session's own RC6 counts (no MTP cost) on both arms
        "store_skips",
        "admission_handoffs",
        "chunk_halvings",
        "chunks_reduced",
    }
    return {k: v for k, v in guard.counts.items() if v and k not in rc6_bookkeeping}


@pytest.mark.parametrize("arm", ["gabee", "m4max128"])
def test_a_fixed_verify_cost_is_planned_without_a_crossing(llama, monkeypatch, arm):
    """The review's scenario: a FIXED 0.9 of the KV at 65% of the sim's
    "84k" (6,500 sim tokens = Gabee's window; 84k = 0.65 of it). RC6 with
    the cost unplanned: 20 crossings on Gabee, 12 on 128 GB; planned
    exactly but with the shared buffers of mid-turn stores unplanned: 5
    and 3. Now 0, the window shrinks by what the cost occupies and no more,
    and the 128 GB arm takes no protective action beyond the plain
    session's."""
    physical = GABEE_PHYSICAL if arm == "gabee" else int(M4_MAX_128_CEILING / 0.97) + 1
    device = _Device(physical=physical)
    s = _auto_budget_scheduler(llama, monkeypatch, device)
    device.install(monkeypatch, s)
    kv_fn = s._kv_footprint_fn()
    nbytes = int(0.9 * kv_fn(int(0.65 * 6500)))
    _fixed_verify(s, nbytes)
    shared = _count_shared(s)
    history, windows = _session(s, delta=500, native=65536)
    guard = s._memguard()
    assert shared["bytes"] > 0  # mid-turn stores shared live buffers
    assert device.ooms == 0
    # the conversation resumes from its entry every turn; its newest entry
    # is never the price of a step
    assert all(cached >= prev - 8 for (_t, prev, _c), (_t2, _n, cached) in zip(history, history[1:])), history
    assert guard.counts["newest_evictions"] == 0
    assert guard.steps_over_ceiling == 0, guard.steps_over_ceiling
    assert device.max_seen <= device.ceiling
    assert not _actions(guard), _actions(guard)
    if arm == "gabee":
        # RC6's plain window is 18,008; the cost occupies nbytes of KV
        lost = nbytes / (kv_fn(1000) / 1000)
        assert min(windows) >= 18_008 - 1.1 * lost, (windows, lost)
        assert len(history) >= 20, len(history)


def _knob_verify(s, nbytes: int, widest: int = 32, depth: int = 3) -> None:
    """An MTP verify of up to ``widest`` rows holding ``(rows - 1) x
    nbytes / (widest - 1)`` (restore points, context-independent), as a
    real allocation that honours the guard's verify cap -- the vendored
    generator's ``verify_rows_cap``."""
    per_row = nbytes // (widest - 1)
    s.config.mtp_max_k = depth
    s._memguard_decode_write_rows = lambda: widest
    s._memguard_verify_bytes_per_row = lambda writes: max(0, writes - 1) * per_row
    real_step = s.step

    def step():
        bg = s.batch_generator
        if bg is not None and not getattr(bg, "_lz_fixed", False):
            real_next = bg.next

            def next_():
                gen = getattr(bg, "_generation_batch", None)
                prompting = len(getattr(bg, "_prompt_batch", ()) or ()) or len(
                    getattr(bg, "_unprocessed_sequences", ()) or ()
                )
                decoding = gen is not None and len(gen) > 0 and not prompting
                rows = getattr(bg, "_lz_verify_rows_cap", None) or widest
                out = real_next()
                if decoding:
                    extra = mx.zeros(((rows - 1) * per_row,), mx.uint8)
                    mx.eval(extra)
                    del extra
                return out

            bg.next = next_
            bg._lz_fixed = True
        return real_step()

    s.step = step


@pytest.mark.parametrize("arm", ["gabee", "m4max128"])
def test_a_narrower_verify_keeps_the_conversations_entry(llama, monkeypatch, arm):
    """The same session with the verify a knob (32 rows at most, the MTP
    chain's 4 at least): where the decode step does not fit beside the
    session's entry the verify narrows; nothing crosses, no newest entry is
    evicted, every turn resumes from its entry, on both arms."""
    physical = GABEE_PHYSICAL if arm == "gabee" else int(M4_MAX_128_CEILING / 0.97) + 1
    device = _Device(physical=physical)
    s = _auto_budget_scheduler(llama, monkeypatch, device)
    device.install(monkeypatch, s)
    _knob_verify(s, int(0.9 * s._kv_footprint_fn()(int(0.65 * 6500))))
    history, windows = _session(s, delta=500, native=65536)
    guard = s._memguard()
    assert device.ooms == 0 and guard.steps_over_ceiling == 0
    assert device.max_seen <= device.ceiling
    assert guard.counts["newest_evictions"] == 0
    assert guard.counts["decode_over_ceiling"] == 0
    assert guard.counts["verify_narrowed"] > 0
    assert all(cached >= prev - 8 for (_t, prev, _c), (_t2, _n, cached) in zip(history, history[1:])), history
    if arm == "gabee":
        assert len(history) >= 20


def _count_shared(s) -> dict[str, int]:
    seen = {"bytes": 0}
    real = s._memguard_note_shared

    def note(tokens):
        before = set(getattr(s, "_memguard_shared", {}) or {})
        real(tokens)
        after = getattr(s, "_memguard_shared", {}) or {}
        seen["bytes"] += sum(r.nbytes for a, r in after.items() if a not in before)

    s._memguard_note_shared = note
    return seen


def test_the_shared_buffers_are_what_the_plan_was_missing(llama, monkeypatch):
    """Negative control: the same Gabee run with the shared buffers left out
    of the footprint, and a mid-turn store no longer leaving room for its
    own session's next decode step, crosses the ceiling (5 times at review
    time). Either one alone holds this run's decode steps."""
    device = _Device(physical=GABEE_PHYSICAL)
    s = _auto_budget_scheduler(llama, monkeypatch, device)
    device.install(monkeypatch, s)
    _fixed_verify(s, int(0.9 * s._kv_footprint_fn()(int(0.65 * 6500))))
    monkeypatch.setattr(Scheduler, "_memguard_shared_pending", lambda self: 0)
    monkeypatch.setattr(Scheduler, "_memguard_own_decode_floor", lambda self, r: 0)
    _session(s, delta=500, native=65536)
    assert s._memguard().steps_over_ceiling > 0


def test_the_scheduler_reads_the_restore_point_from_the_live_cache(cpu):
    """With a decoding cache, its own arrays decide (exact for the loaded
    weights and dtypes); the model's layers are the fallback before one."""
    from mlx_lm.models.cache import ArraysCache

    s = _scheduler(max_k=3, model=_model_27b(lookup=None))
    layer = ArraysCache(size=2)
    layer[0] = mx.zeros((1, KERNEL - 1, 64), mx.bfloat16)
    layer[1] = mx.zeros((1, 4, 16, 16), mx.float32)
    s.batch_generator = SimpleNamespace(
        _generation_batch=_Batch(uids=[7], prompt_cache=[layer])
    )
    live = (KERNEL - 1) * 64 * 2 + 4 * 16 * 16 * 4
    assert s._memguard_recurrent_state_bytes_per_row() == live
    fresh = _scheduler(max_k=3, model=_model_27b(lookup=None))
    assert fresh._memguard_recurrent_state_bytes_per_row() == (
        _model_recurrent_state_bytes(_model_27b())
    )
