# SPDX-License-Identifier: Apache-2.0
"""The scheduler-side memory tunables (lz.20): the free-memory floor held
against another process grabbing memory mid-run, ``--max-model-len``, the
soft reserve fraction and the context-aware need, on the simulated device
of the session tests (1/100 of the bytes: 1 MB here = 100 MB real).

The floor (``--memory-floor-gb``): a prefill step plans under
``min(engine cap, physical - others - floor)``, others re-read from the
machine every step. Owner rule 2026-10-30: the engine never pushes the Mac
below the floor, whatever other apps do. When only the floor is short the
step waits (bounded) instead of refusing a prompt that fits the engine.
"""

from __future__ import annotations

import pytest

import rapid_mlx.memory_budget as memory_budget
import rapid_mlx.scheduler as scheduler_mod
from tests.test_memory_guard_busy_machine import (  # noqa: F401
    MACHINE_128,
    _Machine,
    _scheduler,
)
from tests.test_memory_guard_session import (  # noqa: F401
    M4_MAX_128_CEILING,
    MB,
    SESSION_TOKENS,
    _drive,
    _new_tokens,
    _no_metal,
    _request,
    cpu,
    llama,
)

FLOOR = 100 * MB  # 10 GB real
_ENVS = (
    memory_budget.ENV_MEMORY_FLOOR_GB,
    memory_budget.ENV_MAX_MODEL_LEN,
    memory_budget.ENV_RESERVE_FRACTION,
    memory_budget.ENV_FLOOR_WAIT_S,
)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in _ENVS:
        monkeypatch.delenv(name, raising=False)
    # Re-read the machine every step and let the others' reference fall
    # within the test (production: 1 s cache, 120 s window).
    monkeypatch.setattr(scheduler_mod, "_MEMGUARD_SYSTEM_READ_S", 0.0)
    monkeypatch.setattr(scheduler_mod, "_MEMGUARD_OTHERS_WINDOW_S", 0.2)


def _grab_run(llama, monkeypatch, floor: int, grab: int = 900 * MB):
    """One long prompt; at step 2 another process grabs 900 MB (90 GB real)
    and gives it back at step 40. Returns the per-step lowest free memory
    while it held it, the machine, the scheduler and the outputs."""
    if floor:
        monkeypatch.setenv(memory_budget.ENV_MEMORY_FLOOR_GB, str(floor / 1e9))
    scheduler, device = _scheduler(llama, monkeypatch, M4_MAX_128_CEILING)
    machine = _Machine(monkeypatch, device, 50 * MB, MACHINE_128)
    real = scheduler.step
    count = {"n": 0}
    free_while_grabbed = []

    def step():
        n = count["n"] = count["n"] + 1
        if n == 2:
            machine.others = grab
        if n == 40:
            machine.others = 50 * MB
        out = real()
        engine = max(machine.engine(), device.peak())
        machine.settle(engine)
        if 2 <= n < 40:
            resident = max(0, machine.others - machine.swapped)
            free_while_grabbed.append(machine.physical - engine - resident)
        return out

    scheduler.step = step
    prompt = _new_tokens(0, SESSION_TOKENS)
    try:
        outs = _drive(scheduler, [_request("main", prompt, 4)])
    finally:
        scheduler.step = real
    return free_while_grabbed, machine, scheduler, device, outs


@pytest.mark.parametrize(
    "grab,action",
    # 90 GB real: the steps shrink to fit; 100 GB real: not even the weights
    # leave the floor, so the step waits until the memory comes back.
    [(900 * MB, "chunks_reduced"), (1000 * MB, "floor_holds")],
)
def test_the_floor_holds_when_another_process_grabs_memory(
    llama, monkeypatch, grab, action
):
    free, machine, scheduler, device, outs = _grab_run(llama, monkeypatch, FLOOR, grab)
    jitter = int(0.0011 * machine.physical)
    # The engine never takes the Mac below the floor. When the OTHER process
    # alone already did (1000 MB: weights + live KV + it leave less than the
    # floor), the engine cannot give back what it holds, but it never grows
    # either: free never drops below where the grab left it.
    assert free and min(free) >= min(FLOOR, free[0]) - jitter, (
        min(free) / MB, free[0] / MB, FLOOR / MB
    )
    assert machine.swapouts == 0
    counts = scheduler._memguard().counts
    assert counts[action] >= 1, dict(counts)
    # The request was never refused: it waited, then finished.
    assert outs["main"][-1].finished and not outs["main"][-1].error
    assert device.ooms == 0


def test_without_the_floor_the_same_grab_pushes_the_mac_into_swap(llama, monkeypatch):
    """The control arm: the test discriminates (the floor is what holds)."""
    free, machine, scheduler, _device, _outs = _grab_run(llama, monkeypatch, 0)
    assert machine.swapouts > 0 or min(free) < FLOOR
    assert scheduler._memguard().counts["floor_holds"] == 0


def test_a_step_held_past_the_wait_is_refused_not_run(llama, monkeypatch):
    monkeypatch.setenv(memory_budget.ENV_FLOOR_WAIT_S, "0")
    monkeypatch.setenv(memory_budget.ENV_MEMORY_FLOOR_GB, str(FLOOR / 1e9))
    scheduler, device = _scheduler(llama, monkeypatch, M4_MAX_128_CEILING)
    machine = _Machine(monkeypatch, device, 1000 * MB, MACHINE_128)
    outs = _drive(scheduler, [_request("main", _new_tokens(0, SESSION_TOKENS), 4)])
    assert outs["main"][-1].finished and outs["main"][-1].error
    assert machine.swapouts == 0


def test_the_status_names_the_floor_as_the_binding_cap(llama, monkeypatch):
    monkeypatch.setenv(memory_budget.ENV_MEMORY_FLOOR_GB, str(FLOOR / 1e9))
    scheduler, device = _scheduler(llama, monkeypatch, M4_MAX_128_CEILING)
    _Machine(monkeypatch, device, 1000 * MB, MACHINE_128)
    scheduler._memguard()
    block = scheduler.get_stats()["memory_budget"]
    assert block["step_binding"] == "memory_floor"
    assert block["effective_ceiling_bytes"] < block["limit_bytes"]
    assert block["memory_floor_bytes"] == FLOOR
    assert block["tunables"]["memory_floor_gb"] == pytest.approx(FLOOR / 1e9)
    assert block["physical_bytes"] == MACHINE_128


# ── --max-model-len ──────────────────────────────────────────────────────


def test_max_model_len_caps_admission_and_the_window(llama, monkeypatch):
    scheduler, _device = _scheduler(llama, monkeypatch, M4_MAX_128_CEILING)
    before = scheduler.memory_guard_max_context(scheduler._memguard_native_context())
    monkeypatch.setenv(memory_budget.ENV_MAX_MODEL_LEN, "3000")
    assert scheduler._memguard_native_context() == 3000
    refusal = scheduler.device_memory_admission(3001)
    assert refusal is not None and refusal.limit == "max_model_len"
    assert refusal.error_object()["n_ctx"] == 3000
    assert refusal.error_object()["lz_reason"] == "max_model_len"
    assert "--max-model-len" in refusal.message
    assert scheduler.device_memory_admission(2999) is None
    window = scheduler.memory_guard_max_context(3000, stable=False)
    assert window <= 3000 < (before or 10**9)


def test_the_need_grows_with_max_model_len(llama, monkeypatch):
    scheduler, _device = _scheduler(llama, monkeypatch, M4_MAX_128_CEILING)
    monkeypatch.setenv(memory_budget.ENV_MAX_MODEL_LEN, "2000")
    small = scheduler.memory_plan_need_bytes(M4_MAX_128_CEILING)
    monkeypatch.setenv(memory_budget.ENV_MAX_MODEL_LEN, "20000")
    big = scheduler.memory_plan_need_bytes(M4_MAX_128_CEILING)
    assert small and big and big > small
    kv = scheduler._kv_footprint_fn()
    assert big - small >= kv(20000) - kv(2000)


# ── the soft reserve fraction ────────────────────────────────────────────


def test_the_reserve_fraction_moves_the_optional_memory_ceiling(llama, monkeypatch):
    scheduler, device = _scheduler(llama, monkeypatch, M4_MAX_128_CEILING)
    _Machine(monkeypatch, device, 400 * MB, MACHINE_128)
    default = scheduler._memguard_cache_ceiling()
    monkeypatch.setenv(memory_budget.ENV_RESERVE_FRACTION, "0.3")
    wider = scheduler._memguard_cache_ceiling()
    assert default - wider == pytest.approx(0.22 * MACHINE_128, rel=0.01)


# ── what the floor counts as used (owner question 2026-10-30) ────────────

PAGE = 16384
GB = 10**9


def _pages(gb: float) -> int:
    return int(gb * GB / PAGE)


def test_file_cache_and_purgeable_are_not_counted_as_used():
    """The engine's own memory-mapped files (qwen4 PLE shards, np.memmap,
    read every step) are file-backed pages macOS frees on demand: they must
    not count as memory other apps hold, nor may purgeable pages."""
    base = dict(page=PAGE, wired=_pages(102.0), internal=_pages(16.5),
                purgeable=_pages(0.04), compressor=_pages(1.36))
    used, br = memory_budget.used_bytes_from_pages(**base, external=0)
    used_cache, br_cache = memory_budget.used_bytes_from_pages(
        **base, external=_pages(19.5), free=_pages(8.0), speculative=_pages(0.1)
    )
    assert used_cache == used  # 19.5 GB of file cache changes nothing
    assert used == pytest.approx((102.0 + 16.5 - 0.04 + 1.36) * GB, rel=1e-3)
    assert br_cache["file_backed_excluded_bytes"] == pytest.approx(19.5 * GB, rel=1e-3)
    more_purgeable, _ = memory_budget.used_bytes_from_pages(
        **{**base, "purgeable": _pages(5.04)}
    )
    assert used - more_purgeable == pytest.approx(5.0 * GB, rel=1e-3)


def test_inactive_anonymous_memory_still_counts():
    """An inactive anonymous page leaves RAM only by compression or swap:
    the anonymous count (active + inactive) is used whole."""
    used, br = memory_budget.used_bytes_from_pages(
        page=PAGE, wired=0, internal=_pages(10.0), purgeable=0, compressor=0
    )
    assert used == br["anonymous_bytes"] == _pages(10.0) * PAGE


def test_mule_floor_room_from_the_measured_breakdown():
    """Mule mid-Gauntlet (lz.18, 2026-10-30): used 119.5 GB, MLX 97.4 GB,
    so others = 22.4 GB; with a 7 GB floor (5% of 137.4) the RAM leaves the
    engine 108.0 GB -- below the 112.0 GB auto cap at the default working
    set, so a larger iogpu.wired_limit_mb buys no context there."""
    physical = 137_438_953_472
    used, _ = memory_budget.used_bytes_from_pages(
        page=PAGE, wired=_pages(102.0), internal=_pages(16.5),
        purgeable=_pages(0.04), compressor=_pages(1.36), external=_pages(19.5),
    )
    others = used - int(97.4 * GB)
    room = memory_budget.floor_room_bytes(physical=physical, others=others, floor=7 * GB)
    assert room == pytest.approx(108.0 * GB, abs=0.1 * GB)
    assert room < int(0.97 * 115.45 * GB)


def test_the_status_publishes_what_the_floor_counted(llama, monkeypatch):
    monkeypatch.setenv(memory_budget.ENV_MEMORY_FLOOR_GB, str(FLOOR / 1e9))
    scheduler, device = _scheduler(llama, monkeypatch, M4_MAX_128_CEILING)
    machine = _Machine(monkeypatch, device, 1000 * MB, MACHINE_128)
    breakdown = {"wired_bytes": 1, "anonymous_bytes": 2, "file_backed_excluded_bytes": 3}
    monkeypatch.setattr(
        memory_budget, "system_memory_used",
        lambda: memory_budget.MachineMemory(machine.read().used, 0, breakdown),
    )
    monkeypatch.setattr(scheduler, "_current_process_resident_bytes", lambda: 0)
    scheduler._memguard()
    block = scheduler.get_stats()["memory_budget"]["machine"]
    assert block["file_backed_excluded_bytes"] == 3
    assert block["anonymous_bytes"] == 2
    assert "engine_metal_bytes" in block and "others_rest_bytes" in block


# ── review 2026-10-30: one margin rule, the served window, floor fairness ─


def test_the_guard_margin_flag_moves_every_static_margin(llama, monkeypatch):
    """--memory-guard-margin-gb replaces the 2% static margin EVERYWHERE the
    scheduler keeps a gap under the ceiling (verify narrowing, footprint fit,
    store time, window bucket, the idle eviction line), not only in
    MemoryGuard.margin()."""
    scheduler, _device = _scheduler(llama, monkeypatch, M4_MAX_128_CEILING)
    guard = scheduler._memguard()
    ceiling = int(scheduler._memguard_ceiling())
    default = scheduler._memguard_static_margin(ceiling)
    assert default == int(0.02 * ceiling)
    assert scheduler._memguard_pressure_fraction() == pytest.approx(0.98, abs=1e-3)
    guard.margin_bytes_override = ceiling // 10
    assert guard.static_margin(ceiling) == ceiling // 10
    assert scheduler._memguard_static_margin(ceiling) == ceiling // 10
    assert scheduler._memguard_pressure_fraction() == pytest.approx(0.9, abs=1e-3)
    assert guard.margin(1000, ceiling) == ceiling // 10 + 100


def test_max_model_len_is_the_served_window_on_every_lane(monkeypatch):
    """/v1/models context_window and the context_length_exceeded check read
    get_model_max_context: with --max-model-len it is that window, for text
    and vision engines alike, not max_position_embeddings."""
    from types import SimpleNamespace

    from rapid_mlx.service.helpers import get_model_max_context

    model = SimpleNamespace(args=SimpleNamespace(max_position_embeddings=262_144))
    text = SimpleNamespace(_model=model, is_mllm=False)
    vision = SimpleNamespace(_model=model, is_mllm=True)
    assert get_model_max_context(text) == 262_144
    monkeypatch.setenv(memory_budget.ENV_MAX_MODEL_LEN, "65536")
    assert get_model_max_context(text) == 65_536
    assert get_model_max_context(vision) == 65_536


def test_a_new_floor_shortage_gets_a_whole_new_wait(llama, monkeypatch):
    """A request held, then admitted, then short again later gets the full
    --memory-floor-wait-s again, not what was left of the first wait."""
    monkeypatch.setenv(memory_budget.ENV_FLOOR_WAIT_S, "5")
    scheduler, _device = _scheduler(llama, monkeypatch, M4_MAX_128_CEILING)
    clock = {"t": 1000.0}
    import time as _time

    monkeypatch.setattr(_time, "monotonic", lambda: clock["t"])
    row = type("Row", (), {"request_id": "r1"})()
    assert scheduler._memguard_floor_hold(row)  # held at t=1000
    clock["t"] += 4.0  # not renewed for 4 s: admitted in between
    assert scheduler._memguard_floor_hold(row)  # a new wait, not refused
    clock["t"] += 1.0
    assert scheduler._memguard_floor_hold(row)
    results = []
    for _ in range(4):
        clock["t"] += 1.0
        results.append(scheduler._memguard_floor_hold(row))
    # renewed every second from t=1004: held at 1006..1008, refused at 1009
    assert results == [True, True, True, False]


def test_a_floor_shortage_beside_running_requests_defers_not_holds(llama, monkeypatch):
    """Only the floor is short for one of two decoding requests: that row
    waits in the queue and the other keeps decoding (its end is what frees
    memory) -- the batch is not frozen for a row the floor alone stops."""
    monkeypatch.setenv(memory_budget.ENV_MEMORY_FLOOR_GB, str(FLOOR / 1e9))
    scheduler, device = _scheduler(llama, monkeypatch, M4_MAX_128_CEILING)
    machine = _Machine(monkeypatch, device, 50 * MB, MACHINE_128)
    real = scheduler.step
    count = {"n": 0}

    def step():
        n = count["n"] = count["n"] + 1
        if n == 20:
            machine.others = 950 * MB
        if n == 60:
            machine.others = 50 * MB
        out = real()
        machine.settle(max(machine.engine(), device.peak()))
        return out

    scheduler.step = step
    try:
        outs = _drive(
            scheduler,
            [
                _request("side", _new_tokens(7, 2000), 400),
                _request("main", _new_tokens(50_000, 2000), 400),
            ],
        )
    finally:
        scheduler.step = real
    counts = scheduler._memguard().counts
    assert outs["side"][-1].finished and not outs["side"][-1].error
    assert outs["main"][-1].finished
    assert machine.swapouts == 0
    # the row short of the floor beside a running request was deferred
    # (queue), not held (whole batch)
    assert counts["floor_defers"] >= 1, dict(counts)
