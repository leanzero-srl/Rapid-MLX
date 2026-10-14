# SPDX-License-Identifier: Apache-2.0
"""Memory guard with the prefix cache ON: a long agent session keeps hitting.

Owner's goal (goose T12, 2026-10-14): prefix caching is INVISIBLE (every
turn of a long session re-prefills only its new tokens) AND a Metal OOM is
impossible. Live evidence on a 64 GB M3 Max (Q6 AREX-2, the app's argv):
cache hits up to 74k, then two turns with 0 prefix entries left (the
pressure evictor emptied the store) re-prefilled everything (361 s, 465 s
vs ~30 s on a hit), then an OOM every step from ~76k.

These tests run the REAL scheduler, the memory-aware prefix cache and
mlx-lm's BatchGenerator on a tiny KV-heavy Llama on the CPU device, whose
allocator is REAL: resident memory is every buffer the engine actually
holds (live batch, prefix entries, a finished request's extracted cache,
admission copies, padded merges), not the engine's own ledger of them.
Only the weights (an offset) and a big model's attention scratch are
simulated; the scratch deliberately does NOT have the guard's model form,
so calibration cannot pass by construction:

  step = (0.2*B*C*X**1.15 + 700*B*C + 30*X + 40 kB) * (1 +- 6%)

Sizes are Gabee's 64 GB Mac at 1/100 (30 GB weights, 50.1 GB ceiling).
The engine's pressure evictor (engine_core runs it every 16 steps) is called
after every step here, which is stricter.
"""

from __future__ import annotations

import gc
import importlib
import math
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx  # noqa: E402

from rapid_mlx.memory_guard import (  # noqa: E402
    MARGIN_CEILING_RATIO,
    MemoryGuard,
    PeakPrior,
)
from rapid_mlx.request import Request, SamplingParams, SessionHint  # noqa: E402
from rapid_mlx.scheduler import Scheduler, SchedulerConfig  # noqa: E402

_GEN = importlib.import_module("mlx_lm.generate")
_REAL_PROMPT = _GEN.PromptProcessingBatch.prompt

OOM_TEXT = (
    "[METAL] Command buffer execution failed: Insufficient Memory "
    "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
)
MB = 1_000_000
# measured: Gabee (M3 Max 64 GB, Q8 AREX-2) 30.0 GB weights, 50.1 GB
# ceiling; here at 1/100 (the ceiling is 0.97 x physical).
BASE = 300 * MB
GABEE_PHYSICAL = int(501.0 / 0.97 * MB)
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
            # KV-heavy on purpose: 8 KB of KV per token (8 layers x K+V x
            # 2 heads x 64 x fp32) against ~0.5 KB of score per context token
            # per chunk row, so the session's KV dominates the real
            # allocator the way it does at 100k+ on a 30 GB model.
            hidden_size=128,
            num_hidden_layers=8,
            intermediate_size=256,
            num_attention_heads=2,
            rms_norm_eps=1e-6,
            vocab_size=128,
            num_key_value_heads=2,
            max_position_embeddings=65536,
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
    """A simulated Metal device whose RESIDENT memory is the real allocator.

    ``active`` is the CPU device's real ``mx.get_active_memory()`` above the
    reading at install, plus a simulated weights offset: every buffer the
    engine really holds counts -- the live batch, prefix entries, a finished
    request's extracted cache still referenced by its response, admission
    copies, padded merges -- whether or not the engine knows about it. Only
    the attention scratch of a big model is simulated, and its cost
    deliberately does NOT have the guard's model form (a context**1.15 term,
    a per-step constant, +-6% noise), so calibration cannot pass by
    construction::

        step = (0.2*B*C*X**1.15 + 700*B*C + 30*X + 40 kB) * (1 +- 6%)

    An allocation the guard did not plan is caught twice: before a prompt
    step (resident + the simulated scratch over ``physical`` raises the
    real Metal OOM text, as on Gabee) and after the fact (``max_seen``, the
    larger of the real peak and resident + simulated scratch, folded in at
    every peak reset and after every step; the tests assert it never
    crossed ``physical``).
    """

    def __init__(self, physical: int, scratch_scale: float = 1.0):
        self.physical = physical
        # Multiplies the simulated attention scratch. 600 = the REAL ratio
        # for the goose 27B: a 2048-token chunk at 116k is ~27.3 GB of
        # scratch (review 3) -> 273 MB at 1/100, which is what a 64-token
        # chunk at 8.85k tokens costs here at 600.
        self.scratch_scale = scratch_scale
        self.ooms = 0
        self.fail_next = 0
        self.steps = 0
        self.scratch_peak = 0
        self.max_seen = BASE
        self.scheduler: Scheduler | None = None

    def active(self) -> int:
        return BASE + max(0, self._real_active() - self._baseline)

    def peak(self) -> int:
        real = BASE + max(0, self._real_peak() - self._baseline)
        return max(real, self.scratch_peak)

    def fold(self) -> None:
        self.max_seen = max(self.max_seen, self.peak())

    def reset(self) -> None:
        self.fold()
        self._real_reset()
        self.scratch_peak = 0

    def _transient(self, b: int, n: int, x: int) -> int:
        self.steps += 1
        noise = 1.0 + 0.06 * math.sin(self.steps * 1.7)
        att = 1.6 * b * n * (x**1.15) / 8.0
        return int((att * self.scratch_scale + 700 * b * n + 30 * x + 40_000) * noise)

    def install(self, monkeypatch, scheduler: Scheduler, guard: bool = True) -> None:
        self.scheduler = scheduler
        scheduler._test_device = self
        self._real_active = mx.get_active_memory
        self._real_peak = mx.get_peak_memory
        self._real_reset = mx.reset_peak_memory
        gc.collect()
        mx.clear_cache()
        self._baseline = self._real_active()
        self._real_reset()
        device = self

        def prompt(batch, tokens):
            if tokens:
                n = max(len(t) for t in tokens)
                b = len(tokens)
                x = _context_of(batch, n)
                before = device.active()
                transient = device._transient(b, n, x)
                if device.fail_next > 0 or before + transient > device.physical:
                    device.fail_next = max(0, device.fail_next - 1)
                    device.ooms += 1
                    raise RuntimeError(OOM_TEXT)
                device.scratch_peak = max(device.scratch_peak, before + transient)
                return _REAL_PROMPT(batch, tokens)
            return _REAL_PROMPT(batch, tokens)

        monkeypatch.setattr(_GEN.PromptProcessingBatch, "prompt", prompt)
        monkeypatch.setattr(mx, "get_active_memory", device.active)
        monkeypatch.setattr(mx, "get_peak_memory", device.peak)
        monkeypatch.setattr(mx, "reset_peak_memory", device.reset)
        scheduler._current_metal_active_bytes = device.active
        scheduler._current_metal_cache_bytes = lambda: 0
        scheduler._current_process_resident_bytes = device.active
        cap = int(device.physical * 0.97)
        device.ceiling = cap
        scheduler._resolve_metal_cap_bytes = lambda: cap
        if not guard:
            # The A/B control arm: lz.10 exactly (RAPID_MLX_MEMORY_GUARD=0).
            monkeypatch.setenv("RAPID_MLX_MEMORY_GUARD", "0")
            scheduler._memory_guard = None
            scheduler._memory_guard_disabled = True
            return
        # A near-truth prior so these tests pin the eviction / budget /
        # deferral / handoff POLICY, not the calibration (tested separately).
        # The KV growth copy (one layer, mlx-lm KVCache grows by
        # concatenation) is the engine's own analytic term.
        scheduler._memory_guard = MemoryGuard(
            PeakPrior(
                # The simulated scratch costs 0.2 x context**0.15 x scale
                # bytes per (batch x chunk x context): <= 0.8 x scale up to
                # 10k tokens. 6.0 (scale 1) is this module's historical,
                # deliberately loose prior.
                attention_bytes_per_unit=(
                    6.0 if device.scratch_scale == 1.0 else 0.8 * device.scratch_scale
                ),
                token_bytes=1500.0,
                kv_copy_bytes_per_context=(
                    scheduler._memguard_prior().kv_copy_bytes_per_context
                ),
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


def _request(
    rid: str,
    prompt: list[int],
    max_tokens: int = 4,
    session: str | None = None,
    frame: int = 0,
    first_end: int = 0,
    turn_start: bool = False,
) -> Request:
    """``session``: the client's session id (goose's ``agent-session-id``
    header, which the chat route turns into the request's SessionHint with
    the frame measured from the chat template)."""
    request = Request(
        request_id=rid,
        prompt=list(prompt),
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
    )
    if session is not None or first_end:
        request.session_hint = SessionHint(
            session_id=session,
            frame=frame,
            first_end=first_end,
            turn_start=turn_start,
        )
    return request


def _drive(
    scheduler: Scheduler, requests, steps: int = 4000, arrive=None, admit_check=False
):
    """Step to completion, running the pressure evictor after every step."""
    outputs: dict[str, list] = {}
    arrive = arrive or {}
    for r in requests:
        scheduler.add_request(r)
    for step in range(steps):
        for r in arrive.get(step, ()):
            if admit_check:
                # The route's HTTP admission runs while the others are busy.
                assert scheduler.device_memory_admission(len(r.prompt)) is None
            scheduler.add_request(r)
        out = scheduler.step()
        scheduler.evict_prefix_cache_under_pressure()
        device = getattr(scheduler, "_test_device", None)
        if device is not None:
            device.fold()
        for o in out.outputs:
            outputs.setdefault(o.request_id, []).append(o)
        if step >= max(arrive, default=0) and not scheduler.has_requests():
            break
    return outputs


def _tokens(outs) -> list[int]:
    return [t for o in outs for t in (o.new_token_ids or [])]


def _new_tokens(turn: int, n: int) -> list[int]:
    return [(turn * 37 + i * 11) % 120 + 1 for i in range(n)]


def _session(scheduler, delta: int, native: int, max_tokens: int = 4):
    """One growing session, turn after turn, up to the EDGE of the honest
    window: a turn runs while its prompt + output fits the window."""
    prompt = _new_tokens(0, 600)
    history = []
    windows = []
    for turn in range(200):
        rid = f"turn-{turn}"
        # The route's admission check runs first (it also measures the idle
        # base, as in production).
        window = scheduler.memory_guard_max_context(native)
        if window and len(prompt) + max_tokens > window:
            break
        assert scheduler.device_memory_admission(len(prompt)) is None, turn
        outs = _drive(scheduler, [_request(rid, prompt, max_tokens)])[rid]
        assert outs[-1].finished and not outs[-1].error, (turn, outs[-1].error)
        history.append((turn, len(prompt), outs[-1].cached_tokens))
        windows.append(scheduler.memory_guard_max_context(native))
        prompt = prompt + _tokens(outs) + _new_tokens(turn + 1, delta - max_tokens)
    return history, windows


def _auto_budget_scheduler(llama, monkeypatch, device, **config) -> Scheduler:
    """A scheduler with NO operator prefix budget, on a Mac whose free memory
    after the weights load is what this device leaves (the cache sizes its
    starting budget from it, as on a real Mac: 20% of it, floored at 100 MiB)."""
    import rapid_mlx.memory_cache as memory_cache

    monkeypatch.setattr(
        memory_cache, "_get_available_memory", lambda: device.physical - BASE
    )
    return _scheduler(llama, cache_memory_mb=None, **config)


def test_a_long_session_keeps_hitting_turn_after_turn(llama, monkeypatch):
    """Every turn of a session re-prefills only its new tokens, up to the
    edge of the honest window, with every end-of-turn entry stored and
    every step -- prefill AND decode -- under the guard's own ceiling. The
    end-of-turn entry TAKES OVER the finished request's KV (no second copy),
    the next turn takes it over in turn when copying it would not fit, and
    the turn's own prompt entry is redundant once the end-of-turn entry is
    in. The decode step at which the KV grows (~2 layers of the context
    beyond the footprint) is in the plan and in the window.

    RED on 605548ef6 (same device): end-of-turn stores skipped from 8.1k,
    0 prefix entries left from 10.1k, every later turn a full re-prefill.
    RED on f7df48332: the window let the session reach 20.1k, where the
    decode growth step peaked at 508.65 MB over the 501 MB ceiling; and the
    starting heuristic (20% of free RAM, floored at 100 MiB) stayed an upper
    bound, so entries above it were refused ("Cache entry too large")."""
    device = _Device(physical=GABEE_PHYSICAL)
    # No operator budget (no --cache-memory-*): the guard derives it.
    scheduler = _auto_budget_scheduler(llama, monkeypatch, device)
    device.install(monkeypatch, scheduler)

    delta = 500
    history, windows = _session(scheduler, delta=delta, native=65536)

    guard = scheduler._memguard()
    assert device.ooms == 0
    assert device.max_seen <= device.ceiling, (device.max_seen, device.ceiling)
    assert guard.counts["decode_over_ceiling"] == 0
    assert guard.counts["turn_store_skips"] == 0
    # Near the window the next turn took its entry over instead of copying.
    assert guard.counts["admission_handoffs"] >= 1
    for _turn, length, cached in history[1:]:
        assert length - cached <= delta + 8, history
    last_length = history[-1][1]
    assert last_length + delta >= windows[-1], (last_length, windows[-1])
    # The advertised window never dips (goose compacts at 0.8x it).
    for before, after in zip(windows, windows[1:], strict=False):
        assert after >= 0.97 * before, windows


def _timeline(scheduler: Scheduler) -> dict[str, list[int]]:
    """Wrap ``scheduler.step`` to record the steps each request emitted on."""
    steps: dict[str, list[int]] = {}
    real_step = scheduler.step
    count = [0]

    def step():
        out = real_step()
        count[0] += 1
        for o in out.outputs:
            if o.new_token_ids:
                steps.setdefault(o.request_id, []).append(count[0])
        return out

    scheduler.step = step
    return steps


def _three_requests(llama, monkeypatch, physical: int):
    """A decodes a 6000-token context; B (6000) and C (2000) arrive while it
    does. Joining a row into the decoding batch allocates the merged batch
    buffers (every row padded to the longest) and, when the batch is one
    promoted singleton row, a copy of that row (measured on Metal: an 82 MB
    row joined by a 50-token row = +246 MB peak = 3 x 82)."""
    device = _Device(physical=physical)
    scheduler = _scheduler(llama, max_num_seqs=4, enable_prefix_cache=False)
    device.install(monkeypatch, scheduler)
    steps = _timeline(scheduler)
    a = _request("a", _new_tokens(1, 6000), max_tokens=300)
    b = _request("b", _new_tokens(2, 6000), max_tokens=60)
    c = _request("c", _new_tokens(3, 2000), max_tokens=20)
    outputs = _drive(scheduler, [a], arrive={120: [b], 140: [c]})
    for rid, n in (("a", 300), ("b", 60), ("c", 20)):
        assert outputs[rid][-1].finished, rid
        assert not outputs[rid][-1].error, (rid, outputs[rid][-1].error)
        assert len(_tokens(outputs[rid])) == n, rid
    assert device.ooms == 0
    assert device.max_seen <= device.ceiling, (device.max_seen, device.ceiling)
    guard = scheduler.get_stats()["memory_guard"]
    assert guard["actions"]["refusals"] == 0
    return steps, guard


def test_joining_the_decoding_batch_is_charged_and_deferred(llama, monkeypatch):
    """B and C fit this Mac alone, never merged beside A's 6000-token row:
    each waits and runs, and nothing is allocated over the device.
    RED on 605548ef6: B joins A unplanned (real peak 483 MB on a 440 MB
    device -- a Metal OOM) and B is refused."""
    steps, guard = _three_requests(llama, monkeypatch, physical=440 * MB)
    assert guard["actions"]["deferrals"] >= 1
    assert steps["b"][0] > steps["a"][-1]  # B ran after A


def test_a_deferred_request_stops_gating_once_it_decodes(llama, monkeypatch):
    """B waits for A; once B has its first token it is an ordinary decoding
    row, so C (which fits beside B) runs beside it instead of waiting for
    B to finish (review 2026-10-14, item 7). Negative control for the
    test above: here C does join a decoding batch."""
    steps, guard = _three_requests(llama, monkeypatch, physical=595 * MB)
    assert guard["actions"]["deferrals"] >= 1
    assert steps["b"][0] > steps["a"][-1]  # B still waited for A
    assert steps["c"][0] < steps["b"][-1]  # C ran beside B


def test_a_prompt_that_cannot_fit_even_alone_is_still_refused(llama, monkeypatch):
    # Negative control for the deferral: alone and still too big -> the
    # typed refusal, with n_ctx below the prompt.
    device = _Device(physical=320 * MB)
    scheduler = _scheduler(llama, enable_prefix_cache=False)
    device.install(monkeypatch, scheduler)
    outs = _drive(scheduler, [_request("r", _new_tokens(3, 3000))])["r"]
    assert outs[-1].error_kind == "context_exceeds_device_memory"
    detail = outs[-1].error_detail
    assert detail["n_ctx"] < detail["n_prompt_tokens"] == 3000
    assert device.ooms == 0
    assert device.max_seen <= device.ceiling


def test_backstop_requeue_keeps_the_entry_it_will_fetch_and_the_slope(
    llama, monkeypatch
):
    """E5 on turn 2 of a session: the OOM is synchronous (as on Gabee). The
    requeued request re-fetches turn 1's entry (re-prefills only the delta),
    and the recorded OOM does not move the attention slope or the honest
    window. RED on the branch head: every entry is evicted, cached_tokens is
    zeroed (full re-prefill), and the working-set lower bound poisons the
    slope."""
    device = _Device(physical=1000 * MB)
    scheduler = _scheduler(llama)
    device.install(monkeypatch, scheduler)
    # A Metal device with a 960 MB working set: the backstop's lower bound
    # (working set - active) is far above any real step.
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 960 * MB}
    )
    first = _new_tokens(0, 1500)
    outs = _drive(scheduler, [_request("t0", first)])["t0"]
    assert not outs[-1].error
    guard = scheduler._memguard()
    window_before = scheduler.memory_guard_max_context(16384)
    # What the model is fed, and the OOM's lower bound. (The slope itself
    # may still be converging -- one bounded 1.25x step per observation --
    # so comparing it before/after was flaky; what matters is that the
    # lower bound never becomes an observation.)
    fed: list[int] = []
    bounds: list[int] = []
    real_observe = guard.model.observe
    real_backstop = guard.record_backstop

    def observe(shape, transient, **kw):
        fed.append(int(transient))
        return real_observe(shape, transient, **kw)

    def record_backstop(shape, lower_bound):
        bounds.append(int(lower_bound))
        return real_backstop(shape, lower_bound)

    monkeypatch.setattr(guard.model, "observe", observe)
    monkeypatch.setattr(guard, "record_backstop", record_backstop)

    second = first + _tokens(outs) + _new_tokens(1, 300)
    device.fail_next = 1
    outs = _drive(scheduler, [_request("t1", second)])["t1"]

    assert device.ooms == 1
    assert outs[-1].finished and not outs[-1].error, outs[-1].error
    assert len(second) - outs[-1].cached_tokens <= 300 + 8
    assert guard.counts["backstop_fired"] == 1
    assert bounds and fed
    assert max(fed) < min(bounds), (max(fed), bounds)
    assert scheduler.memory_guard_max_context(16384) >= 0.9 * window_before


def test_max_model_len_is_the_same_idle_figure_under_load(llama, monkeypatch):
    """/v1/models max_model_len is the STABLE idle window: identical with 0
    and with 3 concurrent running requests (goose compacts at 0.8x it and
    caches it per model; load is handled by deferral, never by shrinking it).
    The peak model keeps observing the loaded steps (mixed batches feed its
    residual window) -- the published figure must not move on them."""
    device = _Device(physical=440 * MB)
    scheduler = _scheduler(llama, max_num_seqs=4)
    device.install(monkeypatch, scheduler)
    monkeypatch.setattr(mx.metal, "is_available", lambda: True)  # /v1/models path
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 460 * MB}
    )
    # Warm up once so the idle base is measured and the model calibrates.
    _drive(scheduler, [_request("warm", _new_tokens(9, 1500))])
    guard = scheduler._memguard()
    idle = scheduler.projected_memory_max_context(16384)
    assert idle and 0 < idle < 16384  # memory-bound, not the native window
    assert scheduler.get_stats()["memory_guard"]["max_context_idle"] == idle
    observed = guard.model.observations

    for i in range(3):
        scheduler.add_request(_request(f"c{i}", _new_tokens(10 + i, 1000), 200))
    for _ in range(60):  # all three prefilled, decoding
        scheduler.step()
        device.fold()
    assert len(scheduler.running) == 3
    assert guard.model.observations > observed  # the model kept learning
    assert device.active() > BASE + 20 * MB  # the load is real
    assert device.max_seen <= device.ceiling
    assert scheduler.projected_memory_max_context(16384) == idle
    assert scheduler.get_stats()["memory_guard"]["max_context_idle"] == idle


def test_the_idle_base_is_not_inflated_by_kv_a_finished_response_holds(
    llama, monkeypatch
):
    """The idle base (weights + fixed state) is the MINIMUM since load: an
    idle moment can still hold a finished request's KV (its responses / lazy
    extract copies outlive the step, more so when its store was skipped).
    Real Metal (review 2026-10-14): 35.26 GB recorded vs 30.00 GB of weights
    at 80k -> the prefix budget shrank under the session's entry and every
    turn re-prefilled. RED on 605548ef6: the base grows by the held bytes."""
    device = _Device(physical=GABEE_PHYSICAL)
    scheduler = _scheduler(llama, cache_memory_mb=4000)
    device.install(monkeypatch, scheduler)
    assert scheduler.device_memory_admission(600) is None  # measures the base
    base = scheduler._memguard_idle_base()

    held = mx.zeros((50 * MB // 4,), dtype=mx.float32)  # a response's KV
    mx.eval(held)
    _drive(scheduler, [_request("t", _new_tokens(0, 600))])
    assert not scheduler.running
    assert device.active() >= base + 50 * MB  # the held bytes are resident
    assert scheduler._memguard_idle_base() <= base + MB
    del held


def _pressure_tick(scheduler: Scheduler, footprint_of_cap: float) -> tuple[int, int]:
    """Two stored entries, then one pressure-evictor tick with the footprint
    at ``footprint_of_cap`` x the cap: (entries before, entries evicted)."""
    _drive(scheduler, [_request("a", _new_tokens(1, 300))])
    _drive(scheduler, [_request("b", _new_tokens(2, 300))])
    mac = scheduler.memory_aware_cache
    before = len(mac._entries)
    cap = 10**9
    scheduler._resolve_metal_cap_bytes = lambda: cap
    scheduler._metal_footprint_bytes = lambda reclaim_over=0: int(
        footprint_of_cap * cap
    )
    return before, scheduler.evict_prefix_cache_under_pressure()


def test_guard_off_restores_the_lz10_pressure_evictor(llama, monkeypatch):
    """RAPID_MLX_MEMORY_GUARD=0 is the A/B control: the pressure evictor is
    lz.10's exactly -- trigger at 0.9 x cap, plain LRU, the last entry too.
    RED on 605548ef6: the guard's ordered policy (keep the last entry) ran
    with the guard off."""
    monkeypatch.setenv("RAPID_MLX_MEMORY_GUARD", "0")
    scheduler = _scheduler(llama, cache_memory_mb=4000)
    before, evicted = _pressure_tick(scheduler, 0.95)
    assert scheduler._memguard() is None
    assert before >= 2
    assert evicted == before
    assert not scheduler.memory_aware_cache._entries


def test_guard_on_pressure_evictor_keeps_the_session_entry(llama, monkeypatch):
    # Negative control for the test above: guard on, the same footprint.
    # Below the guard's line (1 - its static margin) nothing is evicted;
    # above it, below the cap, everything but each session's HEAD (the
    # newest entry of a session that finished a turn): two sessions here.
    scheduler = _scheduler(llama, cache_memory_mb=4000)
    before, evicted = _pressure_tick(scheduler, 0.95)
    assert scheduler._memguard() is not None
    assert before >= 2 and evicted == 0
    scheduler = _scheduler(llama, cache_memory_mb=4000)
    before, evicted = _pressure_tick(scheduler, 0.985)
    left = list(scheduler.memory_aware_cache._entries)
    assert len(left) == 2 and evicted == before - 2
    firsts = {_new_tokens(1, 1)[0], _new_tokens(2, 1)[0]}
    assert {key[0] for key in left} == firsts


def test_an_operator_prefix_budget_still_caps_the_session(llama, monkeypatch):
    """Negative control for the derived budget, and the measured case for
    goose's ``--cache-memory-percent 0.35``: an operator budget stays an upper
    bound, so once the session's entry outgrows it (0.35 x the free memory
    after the weights, floored at 100 MiB = 105 MB here; the derived budget
    is 191 MB) it is refused and every later turn re-prefills in full --
    while the guard would have kept it to the window."""
    device = _Device(physical=GABEE_PHYSICAL)
    scheduler = _auto_budget_scheduler(
        llama, monkeypatch, device, cache_memory_percent=0.35
    )
    device.install(monkeypatch, scheduler)
    history, _windows = _session(scheduler, delta=500, native=65536)
    full = [length for _t, length, cached in history[1:] if length - cached > 508]
    assert full and min(full) < 14_000, history
    assert scheduler._memguard().configured_prefix_budget is not None


# A subagent's own system prompt (goose's subagent_system.md): every side
# request starts with it, the main session never does.
_SIDE_SYSTEM = [127 - ((i * 7) % 60) for i in range(320)]


def _side_prompt(turn: int, n: int) -> list[int]:
    # The subagent system prompt, then this request's own messages.
    tail = [126 - ((turn * 13 + i * 7) % 59) for i in range(max(0, n - 320))]
    return (_SIDE_SYSTEM + tail)[:n]


def _session_with_side_requests(scheduler, turns: int, side_tokens: int):
    """The main session grows turn after turn; a side request (a subagent,
    its own prompt) arrives three steps into every main turn."""
    prompt = _new_tokens(0, 600)
    history = []
    windows = []
    for turn in range(turns):
        main = _request(f"main-{turn}", prompt, 4, session="main")
        side = _request(
            f"side-{turn}", _side_prompt(turn, side_tokens), 8, session=f"side-{turn}"
        )
        assert scheduler.device_memory_admission(len(prompt)) is None
        outputs = _drive(scheduler, [main], arrive={3: [side]}, admit_check=True)
        windows.append(scheduler.memory_guard_max_context(65536))
        for rid in (main.request_id, side.request_id):
            assert outputs[rid][-1].finished, rid
            assert not outputs[rid][-1].error, (rid, outputs[rid][-1].error)
        history.append((turn, len(prompt), outputs[main.request_id][-1].cached_tokens))
        prompt = prompt + _tokens(outputs[main.request_id]) + _new_tokens(turn + 1, 496)
    return history, windows


def test_the_main_session_keeps_hitting_with_a_side_request_every_turn(
    llama, monkeypatch
):
    """Gabee Q8 12:22 (f7df48332): the main session's next turn (36k cached,
    handed over) was batched with a 21k side request; E1 evicted the main
    session's newest entry and the side request's, then DEFERRED the main
    turn anyway, dropping its handed-over KV -- a full 36k re-prefill
    (241.9 s vs 26.9 s). Now: feasibility is decided before anything is
    evicted, the latest arrival yields, a deferred request's KV is kept as
    an entry, and session heads are protected -- the main session hits on
    every turn, nothing is evicted for a dispatch that then defers, and
    every step stays under the ceiling. Side requests are ~1/7 of the device
    room each, as 21k of 50.1 GB on Gabee; each passes the HTTP admission
    while the session runs (it fits alone: accepted, then deferred)."""
    device = _Device(physical=GABEE_PHYSICAL)
    scheduler = _auto_budget_scheduler(llama, monkeypatch, device, max_num_seqs=4)
    device.install(monkeypatch, scheduler)

    history, windows = _session_with_side_requests(
        scheduler, turns=22, side_tokens=1600
    )

    guard = scheduler._memguard()
    # Mixed steps (a side request beside the session) stay out of the peak
    # model: the advertised window does not collapse (Gabee Q8 k=22: a 21k
    # side request was refused "maximum context here is 0 tokens").
    assert min(windows) >= 0.97 * windows[0], windows
    # Under pressure the side request waits; the session's chunk is never
    # reduced for it.
    assert guard.counts["deferrals"] >= 1
    assert guard.counts["chunks_reduced"] == 0
    for _turn, length, cached in history[1:]:
        assert length - cached <= 508, history
    assert guard.counts["evictions_then_deferred"] == 0
    assert guard.counts["decode_over_ceiling"] == 0
    assert device.ooms == 0
    assert device.max_seen <= device.ceiling, (device.max_seen, device.ceiling)
    # The engine's own per-step peak, as /v1/status reports it.
    status = scheduler.get_stats()["memory_guard"]
    assert status["steps_over_ceiling"] == 0
    assert 0 < status["max_step_peak_mb"] <= device.ceiling / 2**20


def test_a_refusal_carries_the_advertised_window_as_n_ctx(llama, monkeypatch):
    """ONE feasibility rule for the HTTP 400 and the dispatch: the prompt
    ALONE on the idle engine. Anything within the advertised max_model_len
    is accepted; a refusal's n_ctx IS that advertised figure (> 0, below the
    prompt) -- never one priced from what other requests hold now."""
    device = _Device(physical=GABEE_PHYSICAL)
    scheduler = _auto_budget_scheduler(llama, monkeypatch, device)
    device.install(monkeypatch, scheduler)
    _drive(scheduler, [_request("warm", _new_tokens(0, 600))])
    window = scheduler.memory_guard_max_context(65536)
    assert window and 0 < window < 65536
    assert scheduler.device_memory_admission(window) is None
    refusal = scheduler.device_memory_admission(window + 500)
    assert refusal is not None
    detail = refusal.error_object()
    assert detail["n_ctx"] == window
    # And while a long request runs, a prompt that fits alone is accepted.
    scheduler.add_request(_request("long", _new_tokens(1, int(0.6 * window)), 50))
    for _ in range(30):
        scheduler.step()
    assert scheduler.running
    assert scheduler.device_memory_admission(int(0.5 * window)) is None


# measured/published: the M4 Max 128 GB recommended working set is 115.4 GB,
# the M3 Ultra 96 GB 83.5 GB; goose's cap is 0.9 of it. Gabee (M3 Max 64 GB)
# measured a 50.1 GB ceiling. All at 1/100 like the rest of this module.
M4_MAX_128_CEILING = int(0.9 * 1154 * MB)
M3_ULTRA_96_CEILING = int(0.9 * 835 * MB)
GABEE_64_CEILING = int(501 * MB)
# The goose 27B session of the live ladders, at 1/100 of its bytes: 116k
# tokens x 62.5 KB of KV = 7.25 GB -> 72.5 MB = 8.85k tokens of this model's
# 8.19 KB; the subagent's 21k-token prompt -> 1.6k tokens.
SESSION_TOKENS = 8_850
SIDE_TOKENS = 1_600
# A goose subagent's own run (~60k tokens at 1/13.1), between main turns.
SUBAGENT_TOKENS = 4_600


# The guard's degradations (what a user feels): a reduced chunk, a deferred
# request, a re-prefill (a miss) of the session's turn. An eviction by
# itself is not one -- the guarded engine keeps far more cache than the
# unguarded one (whose budget is 20% of free RAM) and gives some back
# before a dispatch; it counts when the session then misses. (A hand-over
# costs nothing: the KV moves, nothing is lost.)
DEGRADATIONS = ("chunks_reduced", "deferrals")


def _one_arm(
    llama,
    monkeypatch,
    ceiling: int,
    side: int,
    guard: bool,
    turns: int,
    after: bool = False,
):
    """The session grows (+500 tokens a turn, a side request of ``side``
    tokens 3 steps into each turn) on a device with this ``ceiling``; the
    attention scratch at the REAL ratio of the goose 27B. Per turn: the
    step peak, an error, and the guard's degradations."""
    device = _Device(physical=int(ceiling / 0.97) + 1, scratch_scale=600.0)
    scheduler = _auto_budget_scheduler(llama, monkeypatch, device, max_num_seqs=4)
    device.install(monkeypatch, scheduler, guard=guard)
    prompt = _new_tokens(0, 600)
    turns_out = []
    last: dict[str, int] = {}
    for turn in range(turns):
        device.max_seen = 0
        main = _request(f"main-{turn}", prompt, 4, session="main")
        arrive = (
            {
                3: [
                    _request(
                        f"side-{turn}",
                        _side_prompt(turn, side),
                        8,
                        session=f"side-{turn}",
                    )
                ]
            }
            if side
            else {}
        )
        ooms = device.ooms
        if after:
            # goose's real pattern: the main turn ends in a tool call, its
            # subagent runs ALONE, then the main session's next turn.
            outputs = _drive(scheduler, [main])
            sub = _request(
                f"sub-{turn}", _side_prompt(turn, side), 8, session=f"sub-{turn}"
            )
            sub_out = _drive(scheduler, [sub])[sub.request_id][-1]
            if sub_out.error:
                outputs[main.request_id][-1].error = sub_out.error
        else:
            outputs = _drive(scheduler, [main], arrive=arrive)
        out = outputs[main.request_id][-1]
        g = scheduler._memguard()
        counts = {k: g.counts[k] for k in DEGRADATIONS} if g else {}
        counts["metal_pressure"] = getattr(
            scheduler, "num_prefix_cache_metal_pressure_evictions", 0
        )
        acted = {k: v - last.get(k, 0) for k, v in counts.items() if v > last.get(k, 0)}
        if turn and len(prompt) - out.cached_tokens > 508:
            acted["session_miss"] = 1
        last = counts
        turns_out.append(
            (len(prompt), device.max_seen, bool(out.error) or device.ooms > ooms, acted)
        )
        if out.error:
            break
        prompt = prompt + _tokens(outputs[main.request_id]) + _new_tokens(turn + 1, 496)
    return turns_out, device


@pytest.mark.parametrize(
    ("ceiling", "side", "duress"),
    [
        (M4_MAX_128_CEILING, 0, False),
        (M4_MAX_128_CEILING, SIDE_TOKENS, True),
        (M3_ULTRA_96_CEILING, 0, True),
        (M3_ULTRA_96_CEILING, SIDE_TOKENS, True),
        (GABEE_64_CEILING, 0, True),
        (GABEE_64_CEILING, SIDE_TOKENS, True),
        (M4_MAX_128_CEILING, -SUBAGENT_TOKENS, False),
        (GABEE_64_CEILING, -SUBAGENT_TOKENS, True),
    ],
    ids=[
        "128GB",
        "128GB+side",
        "96GB",
        "96GB+side",
        "64GB",
        "64GB+side",
        "128GB+subagent-after",
        "64GB+subagent-after",
    ],
)
def test_the_guard_acts_only_under_duress(llama, monkeypatch, ceiling, side, duress):
    """Owner rule: no degradation when there is no duress. Both arms on the
    SAME workload -- a session growing to "116k" (8.85k tokens at 1/100 of
    the goose 27B's bytes, 30 GB of weights), alone or with a "21k" side
    request every turn, the scratch at the real 2048 x 116k ratio:

    * guard OFF (lz.10) measures the duress: the turn its step peak first
      comes within the guard's own margins of the cap (or crosses / OOMs);
    * guard ON may degrade (reduce a chunk, defer, evict) only from that
      turn on -- never while the unguarded engine stays clear of the cap --
      and never crosses the cap itself.

    Measured boundary (sim; real tokens = sim x 13.1 -- 8.85k = 116k):
    guard-off's peak crosses the cap at 128 GB never to 9.1k alone (766 of
    1,039 MB), 8.6k with the side request; 96 GB never alone (it comes
    within the margin, 97.8%, at 9.1k), 5.6k with it; 64 GB 3.6k alone,
    2.1k with it (an OOM)."""
    turns = 18
    after = side < 0
    side = abs(side)
    with monkeypatch.context() as off_patch:
        off, _ = _one_arm(
            llama, off_patch, ceiling, side, guard=False, turns=turns, after=after
        )
    with monkeypatch.context() as on_patch:
        on, on_device = _one_arm(
            llama, on_patch, ceiling, side, guard=True, turns=turns, after=after
        )

    near = (1 - 2 * MARGIN_CEILING_RATIO) * ceiling
    duress_from = next(
        (i for i, (_n, peak, failed, _a) in enumerate(off) if failed or peak > near),
        None,
    )
    for turn, (length, _peak, failed, acted) in enumerate(on):
        assert not failed, (turn, length)
        if acted:
            assert duress_from is not None and turn >= duress_from, (
                turn,
                length,
                acted,
                duress_from,
            )
    assert on_device.max_seen <= on_device.ceiling
    if duress:
        assert duress_from is not None
    else:
        assert duress_from is None
        assert not any(acted for *_x, acted in on)


def test_a_handed_over_admission_is_handed_over_once(llama, monkeypatch):
    """Hybrid models keep a fixed per-sequence state (the goose 27B's
    GatedDeltaNet layers: 157 MB -> 1.57 MB at 1/100) that admission still
    copies after a hand-over. f7df48332..64c1f288b took that remaining
    charge for a reason to hand over AGAIN: Gabee Q8 k=18, the same request
    "takes over its prefix hit (35966 cached tokens, 0.16GB, 0 entries
    evicted)" 64 times in one step (admission_handoffs 1 -> 65), the plan
    loop ran out, and that step dispatched UNPLANNED: 48,375.8 MiB peak over
    the 47,775.8 MiB ceiling. Now: one hand-over per admitted request, the
    plan converges, every step under the ceiling."""
    device = _Device(physical=GABEE_PHYSICAL)
    scheduler = _auto_budget_scheduler(llama, monkeypatch, device, max_num_seqs=4)
    device.install(monkeypatch, scheduler)
    state = int(1.57 * MB)
    real_kv_fn = scheduler._kv_footprint_fn

    def kv_fn_with_state():
        fn = real_kv_fn()
        return None if fn is None else (lambda tokens: fn(tokens) + state)

    monkeypatch.setattr(scheduler, "_kv_footprint_fn", kv_fn_with_state)
    # The live case: the session's next turn is handed over while a side
    # request is DECODING beside it, so the plan does not fit right after.
    prompt = _new_tokens(0, 600)
    history = []
    for turn in range(22):
        side = _request(
            f"side-{turn}", _side_prompt(turn, 1600), 120, session=f"side-{turn}"
        )
        main = _request(f"main-{turn}", prompt, 4, session="main")
        outputs = _drive(scheduler, [side], arrive={30: [main]})
        for rid in (main.request_id, side.request_id):
            assert not outputs[rid][-1].error, (rid, outputs[rid][-1].error)
        history.append((turn, len(prompt), outputs[main.request_id][-1].cached_tokens))
        prompt = prompt + _tokens(outputs[main.request_id]) + _new_tokens(turn + 1, 496)

    counts = scheduler._memguard().counts
    assert counts["admission_handoffs"] >= 1
    # At most one per admission: a request deferred after its hand-over
    # keeps its KV as an entry and is handed over again when re-admitted.
    assert counts["admission_handoffs"] <= len(history) + counts["deferrals"]
    assert counts.get("plan_exhausted", 0) == 0
    status = scheduler.get_stats()["memory_guard"]
    assert status["steps_over_ceiling"] == 0
    assert device.max_seen <= device.ceiling


def test_an_abandoned_session_head_goes_before_a_new_session_slows(llama, monkeypatch):
    """Review 3 (real Metal, half Gabee scale): an abandoned 32k session's
    head stayed protected as the "costliest" while a NEW session's chunk was
    cut 2048 -> 1968 -> 1784 at 27k (control without it: 2048). That is
    exactly goose compaction: the pre-compaction head is never continued.
    A head is abandoned when something newer of the SAME session (goose's
    session id, the same rendered frame) diverges from it -- here the
    conversation after compaction -- and goes before any chunk is reduced.
    RED on 4ac506f7d."""
    device = _Device(physical=GABEE_PHYSICAL, scratch_scale=600.0)
    scheduler = _auto_budget_scheduler(llama, monkeypatch, device)
    device.install(monkeypatch, scheduler)
    # The pre-compaction session: the SAME system prompt as the new one,
    # then a conversation the new session never continues.
    abandoned = _new_tokens(0, 320) + [3 + (i * 5) % 97 for i in range(5680)]
    _drive(scheduler, [_request("abandoned", abandoned, 4, session="main", frame=320)])
    mac = scheduler.memory_aware_cache
    heads = [k for k in mac._entries if k[: len(abandoned)] == tuple(abandoned)]
    assert heads

    guard = scheduler._memguard()
    reduced_while_abandoned_kept = 0
    real_step = scheduler.step

    def step():
        before = guard.counts["chunks_reduced"]
        out = real_step()
        if guard.counts["chunks_reduced"] > before and any(
            k[: len(abandoned)] == tuple(abandoned) for k in mac._entries
        ):
            nonlocal reduced_while_abandoned_kept
            reduced_while_abandoned_kept += 1

        return out

    scheduler.step = step
    prompt = _new_tokens(0, 600)
    for turn in range(10):
        new = _request(f"new-{turn}", prompt, 4, session="main", frame=320)
        outs = _drive(scheduler, [new])[f"new-{turn}"]
        assert not outs[-1].error
        prompt = prompt + _tokens(outs) + _new_tokens(turn + 1, 496)
    assert reduced_while_abandoned_kept == 0
    assert device.max_seen <= device.ceiling


def test_a_session_keeps_only_its_newest_entries(llama, monkeypatch):
    """Review 3 (real Metal, 128 GB): with the guard the idle cache held 40
    GB vs 29 GB without, mostly OLDER entries of the same session that its
    newest entry strictly extends (dead weight). After every end-of-turn
    store they are superseded (a trimmable cache: the fetch rewinds a longer
    entry, so a strict prefix is dead weight); the newest entry still serves
    the next turn. RED on 4ac506f7d: every turn's entries accumulate."""
    device = _Device(physical=int(M4_MAX_128_CEILING / 0.97) + 1)
    scheduler = _auto_budget_scheduler(llama, monkeypatch, device)
    device.install(monkeypatch, scheduler)
    prompt = _new_tokens(0, 600)
    history = []
    for turn in range(10):
        outs = _drive(scheduler, [_request(f"t{turn}", prompt, 4)])[f"t{turn}"]
        history.append((turn, len(prompt), outs[-1].cached_tokens))
        prompt = prompt + _tokens(outs) + _new_tokens(turn + 1, 496)
    newest = max(scheduler.memory_aware_cache._entries, key=len)
    dead = [
        k
        for k in scheduler.memory_aware_cache._entries
        if k != newest and newest[: len(k)] == k
    ]
    assert dead == []
    assert scheduler._memguard().counts["superseded"] > 0
    for _turn, length, cached in history[1:]:
        assert length - cached <= 508


@pytest.mark.parametrize("branch_first", [True, False], ids=["branch", "control"])
def test_a_side_branch_does_not_supersede_the_main_line(
    llama, monkeypatch, branch_first
):
    """Review 4 (live, Gabee Q8, 244d8de5c): a side branch -- main's
    conversation so far plus one extra user message (a retry, a title or
    summary helper call, a compaction attempt) -- extended main's newest
    end-of-turn entry and stored its own; superseding then dropped MAIN's
    entry, and main's next turn (which extends main's entry, not the
    branch's) fell back to an older one: cached 34,334 of 55,921 instead of
    ~50.4k. On a TRIMMABLE cache (this Llama) that is harmless -- the fetch
    rewinds the branch's entry to the common prefix -- so superseding stays
    full there; this pins that (green on 244d8de5c too). The hybrid case,
    which lost the hit live, is the next test. Control: no branch."""
    device = _Device(physical=int(M4_MAX_128_CEILING / 0.97) + 1)
    scheduler = _auto_budget_scheduler(llama, monkeypatch, device, max_num_seqs=4)
    device.install(monkeypatch, scheduler)
    prompt = _new_tokens(0, 600)
    for turn in range(4):
        main = _request(f"main-{turn}", prompt, 4, session="main")
        outs = _drive(scheduler, [main])[f"main-{turn}"]
        line = prompt + _tokens(outs)
        if branch_first and turn == 2:
            branch = _request("branch", line + _new_tokens(50, 120), 8, session="main")
            assert not _drive(scheduler, [branch])["branch"][-1].error
        prompt = line + _new_tokens(turn + 1, 496)
        if turn >= 2:
            nxt = _drive(
                scheduler, [_request(f"next-{turn}", prompt, 4, session="main")]
            )
            cached = nxt[f"next-{turn}"][-1].cached_tokens
            assert cached >= len(line) - 8, (turn, len(line), cached)
            prompt = prompt + _tokens(nxt[f"next-{turn}"]) + _new_tokens(9, 64)
    assert scheduler._memguard().counts["superseded"] > 0


def _three_sessions(llama, monkeypatch):
    """Main ``a`` (its first turn), its subagent ``s`` (own session, runs
    alone), main's next turn, then a second conversation ``b``."""
    device = _Device(physical=int(M4_MAX_128_CEILING / 0.97) + 1)
    scheduler = _auto_budget_scheduler(llama, monkeypatch, device)
    device.install(monkeypatch, scheduler)
    main = _new_tokens(1, 900)
    outs = _drive(scheduler, [_request("a", main, 4, session="a")])["a"]
    _drive(scheduler, [_request("s", _side_prompt(2, 900), 4, session="s")])
    main = main + _tokens(outs) + _new_tokens(5, 200)
    _drive(scheduler, [_request("a2", main, 4, session="a")])
    _drive(scheduler, [_request("b", _new_tokens(7, 900), 4, session="b")])
    return scheduler


@pytest.mark.parametrize(("level", "left"), [(1, 3), (2, 2), (4, 1)])
def test_the_cache_yields_to_machine_memory_pressure(llama, monkeypatch, level, left):
    """Device-aware is system-aware: other apps and other MLX engines show
    up as the kernel's memory-pressure level. NORMAL (1): nothing given
    back (control). WARN (2): everything but each live conversation's
    newest head -- the finished subagent (outlived by its main session)
    goes. CRITICAL (4): all but the last entry."""
    import rapid_mlx.memory_budget as memory_budget

    scheduler = _three_sessions(llama, monkeypatch)
    mac = scheduler.memory_aware_cache
    assert len({key[0] for key in mac._entries}) == 3
    monkeypatch.setattr(memory_budget, "system_memory_pressure", lambda: level)
    scheduler._memguard_yield_to_system()
    firsts = {key[0] for key in mac._entries}
    assert len(firsts) == left
    if level == 2:
        assert firsts == {_new_tokens(1, 1)[0], _new_tokens(7, 1)[0]}
    if left < 3:
        assert scheduler._memguard().counts["system_pressure_evictions"] > 0
    else:
        assert scheduler._memguard().counts["system_pressure_evictions"] == 0


@pytest.mark.parametrize(
    "ceiling", [GABEE_64_CEILING, M4_MAX_128_CEILING], ids=["64GB", "128GB"]
)
def test_the_main_session_hits_after_its_subagent_runs_alone(
    llama, monkeypatch, ceiling
):
    """Review 4 (C1): goose's real pattern -- the main turn ends in a tool
    call, a subagent (its OWN system prompt, ~60k tokens) runs alone while
    the main session waits, then the main session's next turn. 244d8de5c
    protected no head while anything ran: E1 evicted the main session's head
    to fit the subagent and the main session missed 14 of 15 turns at 64 GB.
    The main session's head is the newest of ITS session, so it stays; the
    subagent gets the reduced chunk if anything has to give."""
    device = _Device(physical=int(ceiling / 0.97) + 1, scratch_scale=600.0)
    scheduler = _auto_budget_scheduler(llama, monkeypatch, device, max_num_seqs=4)
    device.install(monkeypatch, scheduler)
    prompt = _new_tokens(0, 600)
    misses = []
    for turn in range(16):
        main = _request(f"main-{turn}", prompt, 4, session="main")
        outs = _drive(scheduler, [main])[f"main-{turn}"]
        assert not outs[-1].error, (turn, outs[-1].error)
        if turn and len(prompt) - outs[-1].cached_tokens > 508:
            misses.append((turn, len(prompt), outs[-1].cached_tokens))
        sub = _request(
            f"sub-{turn}", _side_prompt(turn, SUBAGENT_TOKENS), 8, session=f"sub-{turn}"
        )
        assert not _drive(scheduler, [sub])[sub.request_id][-1].error, turn
        prompt = prompt + _tokens(outs) + _new_tokens(turn + 1, 496)
    assert misses == []
    assert device.max_seen <= device.ceiling
    assert scheduler._memguard().steps_over_ceiling == 0


def test_status_polls_do_not_move_the_idle_base(llama, monkeypatch):
    """Review 4 (C4): the idle base window was fed by every READER --
    /v1/status (goose polls it every 2 s) and /v1/models -- so 16 polls of
    one idle moment, with a finished response still holding 50 MB, raised
    the base by the full 50 MB. Only the engine going idle samples it."""
    device = _Device(physical=GABEE_PHYSICAL)
    scheduler = _scheduler(llama, cache_memory_mb=4000)
    device.install(monkeypatch, scheduler)
    assert scheduler.device_memory_admission(600) is None
    base = scheduler._memguard_idle_base()
    held = mx.zeros((50 * MB // 4,), dtype=mx.float32)
    mx.eval(held)
    _drive(scheduler, [_request("t", _new_tokens(0, 600))])
    assert not scheduler.running
    for _ in range(16):
        scheduler.get_stats()
    assert scheduler._memguard_idle_base() <= base + MB
    del held


def test_an_admission_spike_counts_in_its_own_step_peak(llama, monkeypatch):
    """Review 4 (C5): the step peak window was reset AFTER scheduling (the
    admission copy, E1's evictions): from idle, an allocation made while a
    request was admitted fell into no step at all -- the reset discarded it
    and /v1/status max_step_peak_mb never showed it. The window now opens
    at the start of step(). Control: the same request with no spike stays
    well below it."""
    spikes = []
    for spike in (True, False):
        device = _Device(physical=GABEE_PHYSICAL)
        scheduler = _scheduler(llama, cache_memory_mb=4000)
        device.install(monkeypatch, scheduler)
        real = scheduler._schedule_waiting

        once = [spike]

        def admit(*args, _real=real, _once=once, _s=scheduler, **kwargs):
            if _once[0] and _s.waiting:  # the admission from idle
                _once[0] = False
                held = mx.zeros((80 * MB // 4,), dtype=mx.float32)
                mx.eval(held)
                del held
            return _real(*args, **kwargs)

        scheduler._schedule_waiting = admit
        _drive(scheduler, [_request("t", _new_tokens(0, 600))])
        peak = scheduler.get_stats()["memory_guard"]["max_step_peak_mb"] * 2**20
        spikes.append(peak)
    with_spike, without = spikes
    assert with_spike >= BASE + 0.9 * 80 * MB, (with_spike, without)
    assert without < BASE + 0.5 * 80 * MB, (with_spike, without)


def test_machine_pressure_keeps_the_waiting_main_session_head(llama, monkeypatch):
    """Review 4 (C2) + review 5 (item 5): at WARN the cache keeps what live
    / queued requests need and every live conversation's newest head --
    while the subagent RUNS and after it FINISHES (244d8de5c..45e157bab kept
    only the most recently used finished session: the subagent, then the
    main session's head went), and the main session's next turn hits."""
    import rapid_mlx.memory_budget as memory_budget

    device = _Device(physical=int(M4_MAX_128_CEILING / 0.97) + 1)
    scheduler = _auto_budget_scheduler(llama, monkeypatch, device, max_num_seqs=4)
    device.install(monkeypatch, scheduler)
    main = _new_tokens(0, 1200)
    outs = _drive(scheduler, [_request("main", main, 4, session="main")])["main"]
    main_next = main + _tokens(outs)
    monkeypatch.setattr(memory_budget, "system_memory_pressure", lambda: 2)
    scheduler.add_request(_request("sub", _side_prompt(0, 900), 40, session="sub"))
    for _ in range(25):  # the subagent prefills and decodes under WARN
        scheduler.step()
        scheduler.evict_prefix_cache_under_pressure()
    assert scheduler.running
    entries = scheduler.memory_aware_cache._entries
    assert any(list(key) == main_next[: len(key)] for key in entries)
    while scheduler.has_requests():  # the subagent finishes, still WARN
        scheduler.step()
        scheduler.evict_prefix_cache_under_pressure()
    scheduler._memguard_yield_to_system()
    assert any(list(key) == main_next[: len(key)] for key in entries)
    nxt = _request("main-2", main_next + _new_tokens(5, 300), 4, session="main")
    outs = _drive(scheduler, [nxt])
    assert outs["main-2"][-1].cached_tokens >= len(main_next) - 8


def _tiny_hybrid():
    """A tiny Qwen3.5 (GatedDeltaNet + full attention, the goose 27B's
    layout: one full-attention layer in four) -- a non-trimmable cache."""
    from mlx_lm.models import qwen3_5

    text = {
        "model_type": "qwen3_5_text",
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "vocab_size": 1024,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 65536,
        "full_attention_interval": 4,
        "layer_types": ["linear_attention"] * 3
        + ["full_attention"]
        + ["linear_attention"] * 3
        + ["full_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 32,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 8,
        "linear_value_head_dim": 32,
        "attn_output_gate": True,
        "partial_rotary_factor": 0.25,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        "tie_word_embeddings": False,
    }
    mx.random.seed(0)
    model = qwen3_5.Model(
        qwen3_5.ModelArgs.from_dict(
            {"model_type": "qwen3_5", "text_config": text, "tie_word_embeddings": False}
        )
    )
    mx.eval(model.parameters())
    return model


class _Tokenizer(SimpleNamespace):
    pass


def _tiny_tokenizer(strips):
    """A test tokenizer. ``strips`` None: no chat template (unknown); True:
    the rendering drops a whole earlier TOOL LOOP's reasoning once a new
    user message follows; "final": only the final reply's (Rapid-MLX keeps
    tool-loop reasoning on the Qwen-family templates); False: keeps all of
    it (AREX-2)."""
    tok = _Tokenizer(
        encode=lambda text: list(text),
        decode=lambda ids, **_: "",
        eos_token_id=9999,
        eos_token_ids={9999},
    )
    if strips is not None:

        def render(messages, tokenize=False, **_):
            last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
            parts = []
            for i, m in enumerate(messages):
                text = m.get("content") or ""
                reasoning = m.get("reasoning_content")
                tool_row = bool(m.get("tool_calls"))
                dropped = i < last_user and (
                    strips is True or (strips == "final" and not tool_row)
                )
                if reasoning and not dropped:
                    text = f"<think>{reasoning}</think>{text}"
                parts.append(f"<{m['role']}>{text}")
            return "".join(parts)

        tok.apply_chat_template = render
    return tok


def _hybrid_scheduler(model, tokenizer, entries=24):
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            max_num_seqs=2,
            spec_decode="none",
            prefill_step_size=512,
            enable_prefix_cache=True,
            use_memory_aware_cache=True,
            use_paged_cache=False,
            hybrid_cache_entries=entries,
            cache_memory_mb=2048,
        ),
    )
    scheduler._memory_guard = MemoryGuard(
        PeakPrior(
            attention_bytes_per_unit=6.0,
            token_bytes=1500.0,
            kv_copy_bytes_per_context=(
                scheduler._memguard_prior().kv_copy_bytes_per_context
            ),
        ),
        floor_tokens=64,
    )
    scheduler._resolve_metal_cap_bytes = lambda: 10**12
    return scheduler


def _hybrid_tokens(seed, n):
    return [(seed * 37 + i * 11) % 900 + 1 for i in range(n)]


_GEN_PROMPT = [901, 902, 903]
# The main conversation's rendered frame and first message (what the chat
# route measures from the template): a session id + these.
_MAIN = {"session": "main", "frame": 300, "first_end": 1000}


def _hybrid_run(scheduler, rid, prompt, boundary, turn_start=False, **session):
    request = _request(rid, prompt, 4, turn_start=turn_start, **session)
    request.prefix_boundary = boundary
    outs = _drive(scheduler, [request])[rid]
    assert not outs[-1].error, outs[-1].error
    return outs[-1].cached_tokens, _tokens(outs)


def _hybrid_tool_loop(scheduler, user, rid, **session):
    """A user message, then a 4-request tool loop that keeps its reasoning
    (each prompt extends the previous end-of-turn entry). Returns the
    loop's next prompt, where the loop started, and the last request's
    prompt (the final reply's)."""
    session = session or _MAIN
    prompt = user + _GEN_PROMPT
    boundary = len(user)
    loop_start = len(prompt)
    last = prompt
    for k in range(4):
        last = prompt
        _cached, out = _hybrid_run(
            scheduler, f"{rid}-loop{k}", prompt, boundary, turn_start=k == 0, **session
        )
        nxt = prompt + out + [904] + _hybrid_tokens(10 + k, 700)
        boundary = len(nxt)
        prompt = nxt + _GEN_PROMPT
    return prompt, loop_start, last


def _strip_loop(prompt, user, loop_start, seed):
    """The new user message when the rendering drops the whole loop's
    reasoning (the first generated tokens) -- the rest re-rendered."""
    return (
        user + _GEN_PROMPT + [905] + prompt[loop_start + 4 :] + _hybrid_tokens(seed, 50)
    )


@pytest.mark.parametrize("strips", [True, None], ids=["probed", "observed"])
def test_a_new_user_message_on_a_hybrid_resumes_from_its_turn_boundary(
    monkeypatch, cpu, strips
):
    """Review 4 (C3), review 5 (items 2-4): a non-trimmable (hybrid) cache
    cannot rewind. When the rendering drops a whole tool loop's reasoning
    on a new user message, the prompt diverges right after the boundary
    its turn started at. A template probed to do that (rendered with a
    tool-call row and its response) keeps that ONE boundary per line from
    the start; an unknown one keeps it once the line was OBSERVED to rewrite
    (its first new user message pays once). Superseding still happens."""
    scheduler = _hybrid_scheduler(_tiny_hybrid(), _tiny_tokenizer(strips))
    first = _hybrid_tokens(1, 1500)
    prompt, loop_start, _last = _hybrid_tool_loop(scheduler, first, "a")
    assert scheduler._memguard().counts["superseded"] > 0  # the win stays
    new_user = _strip_loop(prompt, first, loop_start, 77)
    cached, out = _hybrid_run(
        scheduler, "new-user", new_user, len(new_user) - 3, turn_start=True, **_MAIN
    )
    if strips:
        assert cached >= len(first)
    second = new_user + out + [906] + _hybrid_tokens(78, 40)
    prompt, loop_start, _last = _hybrid_tool_loop(scheduler, second, "b")
    new_user = _strip_loop(prompt, second, loop_start, 79)
    cached, _out = _hybrid_run(
        scheduler, "new-user-2", new_user, len(new_user) - 3, turn_start=True, **_MAIN
    )
    assert cached >= len(second)


@pytest.mark.parametrize("strips", ["final", False], ids=["final-reply", "keeps"])
def test_a_template_that_keeps_tool_loops_keeps_superseding_boundaries(cpu, strips):
    """Negative controls. "final-reply" (the Qwen-family templates once
    Rapid-MLX re-renders tool-loop reasoning): a new user message drops only
    the final reply's reasoning and resumes from the NEWEST boundary, which
    superseding always keeps -- nothing is flagged, nothing extra kept.
    "keeps" (AREX-2, measured live): the prompt just extends. Review 5:
    the old detector marked lines on ordinary traffic and kept every
    boundary (24 entries / 445.7 MB vs 3 / 82.9 MB)."""
    scheduler = _hybrid_scheduler(_tiny_hybrid(), _tiny_tokenizer(strips))
    assert scheduler._memguard_template_rewrites_tool_loops() is False
    user = _hybrid_tokens(1, 1500)
    for turn in range(4):
        prompt, _loop_start, last = _hybrid_tool_loop(scheduler, user, f"t{turn}")
        if strips == "final":
            new_user = last + [905] + _hybrid_tokens(90 + turn, 50)
        else:
            new_user = prompt + [905] + _hybrid_tokens(90 + turn, 50)
        cached, out = _hybrid_run(
            scheduler,
            f"u{turn}",
            new_user,
            len(new_user) - 3,
            turn_start=True,
            **_MAIN,
        )
        assert cached >= len(last) - len(_GEN_PROMPT) - 8, (turn, cached, len(last))
        user = new_user + out + [906] + _hybrid_tokens(78 + turn, 40)
    assert not scheduler.__dict__.get("_memguard_rewriting_lines")
    assert scheduler._memguard().counts["superseded"] > 0
    assert len(scheduler.memory_aware_cache._entries) <= 4


def test_a_rewriting_template_keeps_one_boundary_per_conversation(cpu):
    """Review 5 (item 3): on a template that drops tool-loop reasoning the
    cache keeps the boundary each turn started at -- ONE per conversation,
    not every boundary: over four user turns the idle entry count stays
    flat."""
    scheduler = _hybrid_scheduler(_tiny_hybrid(), _tiny_tokenizer(True))
    user = _hybrid_tokens(1, 1500)
    counts = []
    for turn in range(4):
        prompt, loop_start, _last = _hybrid_tool_loop(scheduler, user, f"t{turn}")
        new_user = _strip_loop(prompt, user, loop_start, 77 + turn)
        cached, out = _hybrid_run(
            scheduler, f"u{turn}", new_user, len(new_user) - 3, turn_start=True, **_MAIN
        )
        assert cached >= len(user), (turn, cached, len(user))
        counts.append(len(scheduler.memory_aware_cache._entries))
        user = new_user + out + [906] + _hybrid_tokens(78 + turn, 40)
    assert max(counts) <= 4 and counts[-1] <= counts[1], counts


def test_a_side_branch_on_a_hybrid_does_not_supersede_the_main_line(cpu):
    """Review 4 (live, Gabee Q8 = a hybrid): the branch shape on a
    non-trimmable cache -- main's turn, a branch (main + one extra user
    message), then main's next turn must still resume from main's entry."""
    scheduler = _hybrid_scheduler(_tiny_hybrid(), _tiny_tokenizer(False))
    first = _hybrid_tokens(1, 1500)
    prompt, _loop_start, _last = _hybrid_tool_loop(scheduler, first, "a")
    _cached, out = _hybrid_run(scheduler, "main", prompt, len(prompt) - 3, **_MAIN)
    line = prompt + out
    branch = line + [907] + _hybrid_tokens(81, 120) + _GEN_PROMPT
    _hybrid_run(scheduler, "branch", branch, len(branch) - 3, **_MAIN)
    nxt = line + [904] + _hybrid_tokens(82, 300) + _GEN_PROMPT
    cached, _out = _hybrid_run(scheduler, "main-next", nxt, len(nxt) - 3, **_MAIN)
    assert cached >= len(line) - 8, (len(line), cached)
