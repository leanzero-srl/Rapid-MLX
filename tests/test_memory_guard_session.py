# SPDX-License-Identifier: Apache-2.0
"""Memory guard with the prefix cache ON: a long agent session keeps hitting.

Owner's goal (goose T12, 2026-10-14): prefix caching is INVISIBLE (every
turn of a long session re-prefills only its new tokens) AND a Metal OOM is
impossible. Live evidence on a 64 GB M3 Max (Q6 AREX-2, the app's argv):
cache hits up to 74k, then two turns with 0 prefix entries left (the
pressure evictor emptied the store) re-prefilled everything (361 s, 465 s
vs ~30 s on a hit), then an OOM every step from ~76k.

These tests run the REAL scheduler, the memory-aware prefix cache and
mlx-lm's BatchGenerator on a tiny Llama on the CPU device. The Metal
allocator is SIMULATED and its cost deliberately does NOT have the guard's
model form (a context**1.15 term, a per-step constant, and +-6% noise), so
calibration cannot pass by construction:

  active  = weights + the live batch caches' bytes + the prefix-cache ledger
  step    = (0.2*B*C*X**1.15 + 700*B*C + 30*X + 40 kB) * (1 +- 6%)

The engine's pressure evictor (engine_core runs it every 16 steps) is called
after every step here, which is stricter.
"""

from __future__ import annotations

import importlib
import math
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx  # noqa: E402

from rapid_mlx.memory_guard import MemoryGuard, PeakPrior  # noqa: E402
from rapid_mlx.request import Request, SamplingParams  # noqa: E402
from rapid_mlx.scheduler import Scheduler, SchedulerConfig  # noqa: E402

_GEN = importlib.import_module("mlx_lm.generate")
_REAL_PROMPT = _GEN.PromptProcessingBatch.prompt

OOM_TEXT = (
    "[METAL] Command buffer execution failed: Insufficient Memory "
    "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
)
MB = 1_000_000
BASE = 1 * MB  # simulated weights
CHUNK = 64
FLOOR = 8


@pytest.fixture(scope="module")
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture(autouse=True)
def _no_metal(monkeypatch, cpu):
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)


@pytest.fixture(scope="module")
def llama(cpu):
    from mlx_lm.models.llama import Model, ModelArgs

    mx.random.seed(0)
    model = Model(
        ModelArgs(
            model_type="llama",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=128,
            num_key_value_heads=2,
            max_position_embeddings=16384,
        )
    )
    mx.eval(model.parameters())
    return model


def _context_of(batch, n: int) -> int:
    best = 0
    for layer in batch.prompt_cache:
        idx = getattr(layer, "_idx", None)
        if idx is None:
            offset = getattr(layer, "offset", 0)
            idx = offset if isinstance(offset, int) else 0
        best = max(best, int(idx))
    return best + n


class _Device:
    """Simulated Metal allocator over the real scheduler's live objects."""

    def __init__(self, physical: int):
        self.physical = physical
        self.peak = BASE
        self.ooms = 0
        self.fail_next = 0
        self.steps = 0
        self.scheduler: Scheduler | None = None

    def active(self) -> int:
        s = self.scheduler
        live = 0
        bg = getattr(s, "batch_generator", None)
        if bg is not None:
            try:
                live = int(bg.prompt_cache_nbytes)
            except Exception:
                live = 0
        ledger = s.memory_aware_cache._current_memory if s.memory_aware_cache else 0
        return BASE + live + ledger

    def _transient(self, b: int, n: int, x: int) -> int:
        # NOT the guard's form: x**1.15 attention, a per-step constant, and
        # deterministic +-6% noise.
        self.steps += 1
        noise = 1.0 + 0.06 * math.sin(self.steps * 1.7)
        att = 1.6 * b * n * (x**1.15) / 8.0
        return int((att + 700 * b * n + 30 * x + 40_000) * noise)

    def install(self, monkeypatch, scheduler: Scheduler) -> None:
        self.scheduler = scheduler
        device = self

        def prompt(batch, tokens):
            if tokens:
                n = max(len(t) for t in tokens)
                b = len(tokens)
                x = _context_of(batch, n)
                before = device.active()
                transient = device._transient(b, n, x)
                device.peak = max(device.peak, before + transient)
                if device.fail_next > 0 or before + transient > device.physical:
                    device.fail_next = max(0, device.fail_next - 1)
                    device.ooms += 1
                    raise RuntimeError(OOM_TEXT)
            return _REAL_PROMPT(batch, tokens)

        monkeypatch.setattr(_GEN.PromptProcessingBatch, "prompt", prompt)
        monkeypatch.setattr(mx, "get_peak_memory", lambda: device.peak)

        def reset():
            device.peak = device.active()

        monkeypatch.setattr(mx, "reset_peak_memory", reset)
        scheduler._current_metal_active_bytes = device.active
        scheduler._current_metal_cache_bytes = lambda: 0
        scheduler._current_process_resident_bytes = device.active
        cap = int(device.physical * 0.97)
        scheduler._resolve_metal_cap_bytes = lambda: cap
        # A near-truth prior so these tests pin the eviction / budget /
        # deferral POLICY, not the calibration (tested separately).
        scheduler._memory_guard = MemoryGuard(
            PeakPrior(
                attention_bytes_per_unit=6.0,
                token_bytes=1500.0,
                kv_copy_bytes_per_context=64.0,
            ),
            floor_tokens=FLOOR,
        )


def _scheduler(model, **config) -> Scheduler:
    tokenizer = SimpleNamespace(
        encode=lambda text: list(text),
        decode=lambda ids, **_: " ".join(map(str, ids)),
        eos_token_id=10_000,
        eos_token_ids={10_000},
    )
    return Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            **{
                "max_num_seqs": 1,
                "spec_decode": "none",
                "prefill_step_size": CHUNK,
                "adaptive_prefill_min_chunk_size": FLOOR,
                "enable_prefix_cache": True,
                "use_memory_aware_cache": True,
                "use_paged_cache": False,
                "cache_memory_mb": 64,
                **config,
            }
        ),
    )


def _request(rid: str, prompt: list[int], max_tokens: int = 4) -> Request:
    return Request(
        request_id=rid,
        prompt=list(prompt),
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
    )


def _drive(scheduler: Scheduler, requests, steps: int = 4000, arrive=None):
    """Step to completion, running the pressure evictor after every step."""
    outputs: dict[str, list] = {}
    arrive = arrive or {}
    for r in requests:
        scheduler.add_request(r)
    for step in range(steps):
        for r in arrive.get(step, ()):
            scheduler.add_request(r)
        out = scheduler.step()
        scheduler.evict_prefix_cache_under_pressure()
        for o in out.outputs:
            outputs.setdefault(o.request_id, []).append(o)
        if step >= max(arrive, default=0) and not scheduler.has_requests():
            break
    return outputs


def _tokens(outs) -> list[int]:
    return [t for o in outs for t in (o.new_token_ids or [])]


def _new_tokens(turn: int, n: int) -> list[int]:
    return [(turn * 37 + i * 11) % 120 + 1 for i in range(n)]


def test_a_long_session_keeps_hitting_turn_after_turn(llama, monkeypatch):
    """RED on the branch head: the E3 budget derived from the live KV + step
    peak ratchets down below one entry around ~7.5k tokens here; from then
    on every turn re-prefills the whole prompt. GREEN: each turn re-prefills
    only the new tokens, with no OOM, to the end of the session."""
    device = _Device(physical=int(10.5 * MB))
    scheduler = _scheduler(llama)
    device.install(monkeypatch, scheduler)

    prompt = _new_tokens(0, 600)
    delta = 500
    history = []
    for turn in range(17):
        rid = f"turn-{turn}"
        # The route's admission check runs first (it also measures the idle
        # base, as in production).
        assert scheduler.device_memory_admission(len(prompt)) is None
        outs = _drive(scheduler, [_request(rid, prompt)])[rid]
        assert outs[-1].finished and not outs[-1].error, outs[-1].error
        cached = outs[-1].cached_tokens
        history.append((turn, len(prompt), cached))
        prompt = prompt + _tokens(outs) + _new_tokens(turn + 1, delta - 4)

    assert device.ooms == 0
    for turn, length, cached in history[1:]:
        reprefilled = length - cached
        assert reprefilled <= delta + 8, history
    assert history[-1][1] > 8000  # well past where the old E3 stopped storing


def test_a_refusal_while_another_request_runs_defers_instead(llama, monkeypatch):
    """B fits this Mac alone but not beside A: it waits for A, then runs.
    RED on the branch head: B is refused with context_length_exceeded."""
    device = _Device(physical=int(4.2 * MB))
    scheduler = _scheduler(llama, max_num_seqs=4, enable_prefix_cache=False)
    device.install(monkeypatch, scheduler)
    a = _request("a", _new_tokens(1, 3000), max_tokens=200)
    b = _request("b", _new_tokens(2, 3000), max_tokens=4)
    # B arrives while A decodes; B's prefill beside A's KV runs out of room.
    outputs = _drive(scheduler, [a], arrive={50: [b]})

    assert device.ooms == 0
    for rid in ("a", "b"):
        assert outputs[rid][-1].finished, rid
        assert not outputs[rid][-1].error, (rid, outputs[rid][-1].error)
        assert _tokens(outputs[rid]), rid
    guard = scheduler.get_stats()["memory_guard"]
    assert guard["actions"]["refusals"] == 0
    assert guard["actions"]["deferrals"] >= 1
    # B ran after A finished.
    assert outputs["a"][-1] is not None


def test_a_prompt_that_cannot_fit_even_alone_is_still_refused(llama, monkeypatch):
    # Negative control for the deferral: alone and still too big -> the
    # typed refusal, with n_ctx below the prompt.
    device = _Device(physical=int(2.0 * MB))
    scheduler = _scheduler(llama, enable_prefix_cache=False)
    device.install(monkeypatch, scheduler)
    outs = _drive(scheduler, [_request("r", _new_tokens(3, 3000))])["r"]
    assert outs[-1].error_kind == "context_exceeds_device_memory"
    detail = outs[-1].error_detail
    assert detail["n_ctx"] < detail["n_prompt_tokens"] == 3000
    assert device.ooms == 0


def test_backstop_requeue_keeps_the_entry_it_will_fetch_and_the_slope(
    llama, monkeypatch
):
    """E5 on turn 2 of a session: the OOM is synchronous (as on Gabee). The
    requeued request re-fetches turn 1's entry (re-prefills only the delta),
    and the recorded OOM does not move the attention slope or the honest
    window. RED on the branch head: every entry is evicted, cached_tokens is
    zeroed (full re-prefill), and the working-set lower bound poisons the
    slope."""
    device = _Device(physical=int(64 * MB))
    scheduler = _scheduler(llama)
    device.install(monkeypatch, scheduler)
    # A Metal device with a 60 MB working set: the backstop's lower bound
    # (working set - active) is far above any real step.
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 60 * MB}
    )
    first = _new_tokens(0, 1500)
    outs = _drive(scheduler, [_request("t0", first)])["t0"]
    assert not outs[-1].error
    guard = scheduler._memguard()
    slope_before = guard.model.attention_bytes_per_unit
    window_before = scheduler.memory_guard_max_context(16384)

    second = first + _tokens(outs) + _new_tokens(1, 300)
    device.fail_next = 1
    outs = _drive(scheduler, [_request("t1", second)])["t1"]

    assert device.ooms == 1
    assert outs[-1].finished and not outs[-1].error, outs[-1].error
    assert len(second) - outs[-1].cached_tokens <= 300 + 8
    assert guard.counts["backstop_fired"] == 1
    assert guard.model.attention_bytes_per_unit == slope_before
    assert scheduler.memory_guard_max_context(16384) >= 0.9 * window_before

