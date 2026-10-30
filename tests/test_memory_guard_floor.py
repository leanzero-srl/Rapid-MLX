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
