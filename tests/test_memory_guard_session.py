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
        # Multiplies the simulated attention scratch: 30 puts a full-context
        # prefill step at ~3% of Gabee's ceiling at 1/100 (measured live:
        # 1.72 GB at 1x512x29,184 of a 50.1 GB ceiling).
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

    def install(self, monkeypatch, scheduler: Scheduler) -> None:
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
        # A near-truth prior so these tests pin the eviction / budget /
        # deferral / handoff POLICY, not the calibration (tested separately).
        # The KV growth copy (one layer, mlx-lm KVCache grows by
        # concatenation) is the engine's own analytic term.
        scheduler._memory_guard = MemoryGuard(
            PeakPrior(
                attention_bytes_per_unit=6.0 * device.scratch_scale,
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


def _request(rid: str, prompt: list[int], max_tokens: int = 4) -> Request:
    return Request(
        request_id=rid,
        prompt=list(prompt),
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
    )


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


def _side_prompt(turn: int, n: int) -> list[int]:
    # A different system prompt from the main session: no shared prefix.
    return [127 - ((turn * 13 + i * 7) % 60) for i in range(n)]


def _session_with_side_requests(scheduler, turns: int, side_tokens: int):
    """The main session grows turn after turn; a side request (a subagent,
    its own prompt) arrives three steps into every main turn."""
    prompt = _new_tokens(0, 600)
    history = []
    windows = []
    for turn in range(turns):
        main = _request(f"main-{turn}", prompt, 4)
        side = _request(f"side-{turn}", _side_prompt(turn, side_tokens), 8)
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


def _ladder_with_a_subagent(llama, monkeypatch, ceiling: int):
    device = _Device(physical=int(ceiling / 0.97) + 1, scratch_scale=30.0)
    scheduler = _auto_budget_scheduler(llama, monkeypatch, device, max_num_seqs=4)
    device.install(monkeypatch, scheduler)
    chunks: list[int] = []
    real_step = scheduler.step

    def step():
        out = real_step()
        bg = scheduler.batch_generator
        if bg is not None and len(bg._prompt_batch):
            chunks.append(int(bg.prefill_step_size))
        return out

    scheduler.step = step
    turns = (SESSION_TOKENS - 600) // 500 + 1
    history, _windows = _session_with_side_requests(
        scheduler, turns=turns, side_tokens=SIDE_TOKENS
    )
    assert history[-1][1] >= SESSION_TOKENS - 500
    for _turn, length, cached in history[1:]:
        assert length - cached <= 508, history
    assert device.ooms == 0
    assert device.max_seen <= device.ceiling
    return scheduler._memguard().counts, chunks, scheduler


@pytest.mark.parametrize(
    "ceiling", [M4_MAX_128_CEILING, M3_ULTRA_96_CEILING], ids=["128GB", "96GB"]
)
def test_a_big_mac_never_sees_the_guard_act(llama, monkeypatch, ceiling):
    """Owner rule: the chunk is reduced only under duress on weaker devices;
    a Mac with a lot of memory must see none of it. The live ladder (a
    session growing to 116k + a 21k subagent request every turn) at 128 GB
    and 96 GB proportions: zero chunk reductions, zero evictions forced by
    the guard (E1, pressure, store-room, join), zero deferrals, and every
    prefill dispatch at the engine's normal chunk."""
    counts, chunks, scheduler = _ladder_with_a_subagent(llama, monkeypatch, ceiling)
    assert counts["chunks_reduced"] == 0 and counts["chunk_halvings"] == 0
    assert counts["evictions"] == 0
    # (The cache keeping to its own budget -- capacity LRU, as without the
    # guard -- is not device pressure.)
    assert getattr(scheduler, "num_prefix_cache_metal_pressure_evictions", 0) == 0
    assert counts["deferrals"] == 0
    assert counts["admission_handoffs"] == 0
    assert chunks and set(chunks) == {CHUNK}


def test_the_same_ladder_at_64gb_makes_the_guard_act(llama, monkeypatch):
    # Negative control for the test above: Gabee's proportions. The subagent
    # cannot join the session's batch here: it waits (and the session still
    # hits every turn, under the ceiling).
    counts, _chunks, scheduler = _ladder_with_a_subagent(
        llama, monkeypatch, GABEE_64_CEILING
    )
    acted = (
        counts["deferrals"]
        + counts["evictions"]
        + counts["chunks_reduced"]
        + counts["admission_handoffs"]
        + scheduler.num_prefix_cache_pressure_evictions
    )
    assert acted > 0
    assert counts["evictions_then_deferred"] == 0
