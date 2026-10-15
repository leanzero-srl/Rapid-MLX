# SPDX-License-Identifier: Apache-2.0
"""Decode steps planned from MEASUREMENT (Gabee T12 control run, 2026-10-15).

Qwen3.8-27B Q8 + mtp.safetensors on Gabee's M3 Max (64 GB, ceiling 50.10 GB,
engine e84498a4a = the b06acd8ea guard). The first STEP OVER CEILING lines
fired in DECODE at B=1:

    [memory_guard] STEP OVER CEILING: decode shape=decode B=1 prompt_rows=0
    decoding_rows=1 planned=45.37GB measured_peak=50.22GB ceiling=50.10GB
    (+4.85GB unplanned; guard defect signal)

at the request scheduled with prompt_tokens=83666; the store line beside it
reads ``kv=5228.3MB ... per_token=73.0KB`` for 83,653 tokens (16
full-attention layers, 4 KV heads x head_dim 256 x bf16: 4,096 B per token
per layer). Seventeen such lines, every peak 50.15-50.24 GB whatever was
planned (45.20 .. 49.96 GB): MLX throttles at its memory limit (the
guard's ceiling), so +4.85 GB is a lower bound -- about 0.9 of the KV.

The vendored MTP verify writes ``mtp_max_k + 1`` rows a step (32 with a
prompt-lookup copy draft), so ``idx + 1 > capacity`` never saw the KV
growth coming, and the growth the guard did plan was 2 layers of 16.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest

from rapid_mlx.memory_guard import MemoryGuard, PeakPrior, kv_growth_step
from rapid_mlx.scheduler import Scheduler
from tests.test_memory_guard_session import (  # noqa: F401
    GABEE_PHYSICAL,
    M4_MAX_128_CEILING,
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

GB = 10**9
CEILING = int(50.10 * GB)  # Gabee: 90% of 55.7 GB
FOOTPRINT = int(45.37 * GB)  # planned (no growth, no per-step term)
PEAK = int(50.22 * GB)
CONTEXT = 83_666
LAYER = 4096.0  # one full-attention layer's K+V per token
PER_TOKEN = 73.0 * 1024  # the store line's per_token (KV + recurrent state)
# M4 Max 128 GB: 0.9 x its recommended working set (test_memory_guard_session)
BIG_CEILING = int(0.9 * 115.4 * GB)


def _guard() -> MemoryGuard:
    prior = PeakPrior(192.0, 251_904.0, LAYER, PER_TOKEN)
    return MemoryGuard(prior, floor_tokens=kv_growth_step())


def _kv(context: int) -> int:
    return int(PER_TOKEN * context)


def test_gabee_growth_step_over_the_ceiling_teaches_the_growth_term():
    guard = _guard()
    before = guard.decode_growth(CONTEXT, kv_footprint=_kv)
    # the 2-layer prior: 0.71 GB at 84k against the 4.85 GB measured
    assert before < (PEAK - FOOTPRINT) / 4, before / GB
    assert guard.observe_decode(
        PEAK - FOOTPRINT,
        context=CONTEXT,
        batch=1,
        planned=before,
        ceiling=CEILING,
        peak=PEAK,
        grew=True,
        kv_footprint=_kv,
    )
    planned = guard.decode_growth(CONTEXT, kv_footprint=_kv)
    assert planned >= PEAK - FOOTPRINT, planned / GB
    # capped at a whole KV (16 full-attention layers + the recurrent state)
    assert guard.decode_growth_layers <= PER_TOKEN / LAYER
    # the next growth at that context fits only with room made on Gabee ...
    margin = guard.margin(planned, CEILING)
    assert FOOTPRINT + planned + margin > CEILING
    # ... and needs none on a 128 GB Mac (no protective action there)
    assert FOOTPRINT + planned + guard.margin(planned, BIG_CEILING) < BIG_CEILING


def test_a_step_under_the_ceiling_teaches_nothing():
    """Below the ceiling an unplanned byte costs nothing: the same excess
    on a Mac with room changes no plan (owner: no action without duress)."""
    guard = _guard()
    for grew in (True, False):
        assert not guard.observe_decode(
            PEAK - FOOTPRINT,
            context=CONTEXT,
            batch=1,
            planned=0,
            ceiling=BIG_CEILING,
            peak=PEAK,
            grew=grew,
            kv_footprint=_kv,
        )
    assert guard.decode_growth_layers == 2.0
    assert guard.decode_step(CONTEXT) == 0


def test_an_overshooting_decode_step_teaches_the_every_step_term():
    guard = _guard()
    assert guard.decode_step(CONTEXT) == 0
    guard.observe_decode(
        PEAK - FOOTPRINT,
        context=CONTEXT,
        batch=1,
        planned=0,
        ceiling=CEILING,
        peak=PEAK,
    )
    assert guard.decode_step(CONTEXT) >= PEAK - FOOTPRINT - 1
    assert guard.decode_growth(CONTEXT, kv_footprint=_kv) >= guard.decode_step(CONTEXT)


def test_a_decode_step_under_the_ceiling_teaches_nothing():
    """The MTP decodes at 48-72k on the same run: peak - active <= 0.4 GB
    (the [Metal memory] samples, e.g. active=44.1GB peak=44.5GB at ~70k)."""
    guard = _guard()
    guard.observe_decode(
        int(0.4 * GB),
        context=70_000,
        batch=1,
        planned=0,
        ceiling=CEILING,
        peak=int(44.5 * GB),
    )
    assert guard.decode_step(CONTEXT) == 0


def test_the_window_includes_the_measured_decode_step():
    guard = _guard()
    base = int(31.8 * GB)  # weights (the boot line)
    before = guard.max_context(
        base=base, kv_footprint=_kv, ceiling=CEILING, upper=262_144
    )
    guard.observe_decode(
        PEAK - FOOTPRINT,
        context=CONTEXT,
        batch=1,
        planned=0,
        ceiling=CEILING,
        peak=PEAK,
    )
    after = guard.max_context(
        base=base, kv_footprint=_kv, ceiling=CEILING, upper=262_144
    )
    assert after < before
    need = guard.single_sequence_need(
        after, base=base, kv_footprint=_kv, ceiling=CEILING
    )
    assert need <= CEILING
    # 128 GB: the native window is untouched
    assert (
        guard.max_context(
            base=base, kv_footprint=_kv, ceiling=BIG_CEILING, upper=262_144
        )
        == 262_144
    )


# -- the scheduler's growth predicate ---------------------------------------


def _scheduler(spec_decode: str, *, max_k: int, lookup: int | None):
    from rapid_mlx.spec_decode.mtp.prompt_lookup import PromptLookupPolicy

    s = object.__new__(Scheduler)
    s.config = SimpleNamespace(spec_decode=spec_decode, mtp_max_k=max_k)
    model = SimpleNamespace()
    if lookup is not None:
        model.mtp_prompt_lookup_supported = True
        model.mtp_prompt_lookup_policy = PromptLookupPolicy(
            enabled_by_default=True,
            enabled_under_sampling=True,
            min_ngram=8,
            max_ngram=64,
            max_tokens=lookup,
        )
    s.model = model
    return s


class _Batch(SimpleNamespace):
    def __len__(self) -> int:
        return len(self.uids)


def _decoding(s: Scheduler, *, capacity: int, idx: int, context: int):
    from mlx_lm.models.cache import BatchKVCache

    layer = BatchKVCache(left_padding=[0])
    k = mx.zeros((1, 1, capacity, 8))
    layer.keys, layer.values = k, k
    layer._idx = idx
    layer.offset = mx.array([idx])
    request = SimpleNamespace(prompt_token_ids=[0] * context, num_output_tokens=0)
    s.batch_generator = SimpleNamespace(
        _generation_batch=_Batch(uids=[7], prompt_cache=[layer])
    )
    s._memguard_request_for_uid = lambda uid: request
    s._kv_footprint_fn = lambda: _kv


@pytest.mark.parametrize(
    ("spec", "max_k", "lookup", "rows"),
    [("none", 3, None, 1), ("mtp", 3, None, 4), ("mtp", 3, 31, 32)],
)
def test_a_step_writes_the_verify_width(spec, max_k, lookup, rows, monkeypatch):
    monkeypatch.delenv("RAPID_MLX_MTP_PROMPT_LOOKUP", raising=False)
    monkeypatch.delenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MAX_TOKENS", raising=False)
    s = _scheduler(spec, max_k=max_k, lookup=lookup)
    assert s._memguard_decode_write_rows() == rows


def test_an_mtp_verify_near_capacity_is_a_growth_step(cpu, monkeypatch):
    """Gabee's shape: 3 positions short of the buffer, a 4-row verify."""
    monkeypatch.delenv("RAPID_MLX_MTP_PROMPT_LOOKUP", raising=False)
    guard = _guard()
    s = _scheduler("mtp", max_k=3, lookup=None)
    _decoding(s, capacity=CONTEXT + 3, idx=CONTEXT, context=CONTEXT)
    # planned as a growth step (2 layers + the block before any crossing)
    assert s._memguard_decode_growth(guard) >= 2 * LAYER * CONTEXT
    assert s._memguard_last_decode_grows is True
    # plain decode at the same position does not grow there
    s.config.spec_decode = "none"
    assert s._memguard_decode_growth(guard) == 0
    assert s._memguard_last_decode_grows is False


# -- the whole path, through the real scheduler (the 1/100 session sim) -----


# Gabee's measured excess per KV byte at 84k: 4.85 GB of a 5.23 GB KV.
EXCESS_OF_KV = 4.85 / 5.23


def _mtp_like(scheduler):
    """Every decode-only step also holds EXCESS_OF_KV of its context's KV
    beyond the footprint (what Gabee's MTP decode measured), as a real
    allocation of the CPU allocator."""
    kv_fn = scheduler._kv_footprint_fn
    real_step = scheduler.step

    def step():
        bg = scheduler.batch_generator
        if bg is not None and not getattr(bg, "_lz_mtp_like", False):
            real_next = bg.next

            def next_():
                gen = getattr(bg, "_generation_batch", None)
                prompting = len(getattr(bg, "_prompt_batch", ()) or ()) or len(
                    getattr(bg, "_unprocessed_sequences", ()) or ()
                )
                decoding = gen is not None and len(gen) > 0 and not prompting
                out = real_next()
                lengths = scheduler._memguard_generation_lengths()
                fn = kv_fn()
                if decoding and lengths and fn is not None:
                    extra = mx.zeros((int(EXCESS_OF_KV * fn(max(lengths))),), mx.uint8)
                    mx.eval(extra)
                    del extra
                return out

            bg.next = next_
            bg._lz_mtp_like = True
        return real_step()

    scheduler.step = step


def _actions(guard) -> dict[str, int]:
    quiet = {"checks", "calibrations", "superseded", "abandoned"}
    return {k: v for k, v in guard.counts.items() if v and k not in quiet}


def test_an_mtp_like_decode_is_learned_and_then_planned_on_gabee(llama, monkeypatch):
    device = _Device(physical=GABEE_PHYSICAL)
    scheduler = _auto_budget_scheduler(llama, monkeypatch, device)
    device.install(monkeypatch, scheduler)
    _mtp_like(scheduler)
    history, _windows = _session(scheduler, delta=500, native=65536)
    guard = scheduler._memguard()
    assert len(history) >= 4, history
    assert device.ooms == 0
    # learned from the steps that overshot, then planned: the overshoots are
    # the learning steps, not every later decode step
    assert guard.decode_step_bytes_per_context > 0
    assert guard.decode_step_raises >= 1
    assert guard.steps_over_ceiling <= guard.decode_step_raises, (
        guard.steps_over_ceiling,
        guard.decode_step_raises,
    )


def test_an_mtp_like_decode_takes_no_protective_action_on_128_gb(llama, monkeypatch):
    """The same session on a 128 GB Mac up to the goose 27B's "116k" (8.85k
    sim tokens: real = sim x 13.1, test_the_guard_acts_only_under_duress),
    Gabee's 84k overshoot included: no duress there, so no protective
    action, no step over the ceiling, nothing learned."""
    device = _Device(physical=int(M4_MAX_128_CEILING / 0.97) + 1)
    scheduler = _auto_budget_scheduler(llama, monkeypatch, device)
    device.install(monkeypatch, scheduler)
    _mtp_like(scheduler)
    prompt = _new_tokens(0, 600)
    turns = 0
    while len(prompt) < 8_850:
        rid = f"turn-{turns}"
        outs = _drive(scheduler, [_request(rid, prompt, 64)])[rid]
        assert outs[-1].finished and not outs[-1].error, outs[-1].error
        prompt = prompt + _tokens(outs) + _new_tokens(turns + 1, 800)
        turns += 1
    guard = scheduler._memguard()
    assert turns >= 4, turns
    assert device.ooms == 0
    assert guard.steps_over_ceiling == 0
    assert not _actions(guard), _actions(guard)
    assert guard.decode_step_raises == 0
