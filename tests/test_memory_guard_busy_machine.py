# SPDX-License-Identifier: Apache-2.0
"""The guard plans under what the MACHINE leaves the engine, not under the
device alone (RC10 128 GB ladder, 2026-10-16).

M4 Max 128 GB, device ceiling 99.09 GB (0.9 x the 115.4 GB working set),
the engine started by hand beside the user's own apps. The guard's planning
took no actions -- yet Metal active reached 91.0 GB (peak 99.18), the
machine stood at 5% free with 46.8 GB of swap in use, and the kernel's
pressure LEVEL still said normal. The device ceiling assumed the engine owns
the Mac. Now the plan ceiling is

    min(device ceiling, physical - what the other processes use now
        - SYSTEM_RESERVE_RATIO x physical)

measured live (``kern.memorystatus_level``, the free share
``memory_pressure`` prints, re-read each second): the prefix budget and the
step plans follow it, and when the other processes grow into the reserve
the cache gives entries back. Feasibility (the advertised window, a refusal)
stays on the device: a busy machine defers or runs small, it never calls a
prompt too long. A big idle Mac and Gabee (the engine owns it) keep the
device ceiling.

At 1/100 of the bytes like the rest of the session tests; the simulated
device's memory is the CPU allocator's real bytes.
"""

from __future__ import annotations

import pytest

import rapid_mlx.memory_budget as memory_budget
from tests.test_memory_guard_session import (  # noqa: F401
    M4_MAX_128_CEILING,
    MB,
    SUBAGENT_TOKENS,
    _auto_budget_scheduler,
    _Device,
    _drive,
    _new_tokens,
    _no_metal,
    _request,
    _side_prompt,
    _tokens,
    cpu,
    llama,
)

MACHINE_128 = int(1374.4 * MB)  # 137.4 GB of RAM at 1/100


def _machine(monkeypatch, device, others: int, physical: int = MACHINE_128):
    """The rest of the machine holds ``others``; the kernel's free share is
    what the engine (its real bytes) and they leave."""
    import mlx.core as mx

    reading = {"others": others}

    def free() -> float:
        used = device.active() + mx.get_cache_memory() + reading["others"]
        return max(0.0, min(1.0, 1.0 - used / physical))

    monkeypatch.setattr(memory_budget, "system_physical_memory", lambda: physical)
    monkeypatch.setattr(memory_budget, "system_memory_free_fraction", free)
    return reading


def _ladder(llama, monkeypatch, others: int | None, turns: int = 24):
    """The 128 GB ladder: a session growing +500 tokens a turn, a fresh side
    request (a subagent's 60k-token run, its own prompt) beside every turn --
    every side entry stays cached while the budget has room, so the cache
    fills toward the ceiling it plans under."""
    device = _Device(physical=int(M4_MAX_128_CEILING / 0.97) + 1, scratch_scale=600.0)
    scheduler = _auto_budget_scheduler(llama, monkeypatch, device, max_num_seqs=4)
    device.install(monkeypatch, scheduler)
    reading = _machine(monkeypatch, device, others) if others is not None else None
    scheduler._memguard_others_read = None
    prompt = _new_tokens(0, 600)
    cached = []
    worst = 0
    for turn in range(turns):
        main = _request(f"main-{turn}", prompt, 4, session="main")
        side = _request(
            f"side-{turn}", _side_prompt(turn, SUBAGENT_TOKENS), 8, session=f"side-{turn}"
        )
        device.max_seen = 0
        outs = _drive(scheduler, [main], arrive={3: [side]})
        for rid in (main.request_id, side.request_id):
            assert outs[rid][-1].finished and not outs[rid][-1].error, rid
        worst = max(worst, device.max_seen)
        cached.append(outs[main.request_id][-1].cached_tokens)
        prompt = prompt + _tokens(outs[main.request_id]) + _new_tokens(turn + 1, 496)
    return scheduler, device, worst, cached, reading


def test_a_busy_128_gb_mac_stays_out_of_swap(llama, monkeypatch):
    """Others hold 40 GB (400 MB here). RED on 150a141ee: the engine plans
    under the 99.09 GB device ceiling and its cache grows past what the
    machine leaves it (engine + others > physical: swap). GREEN: the engine
    never takes more than physical - others - the reserve, and the session
    still resumes every turn."""
    others = 400 * MB
    scheduler, device, worst, cached, _ = _ladder(llama, monkeypatch, others)
    reserve = int(memory_budget.SYSTEM_RESERVE_RATIO * MACHINE_128)
    print("worst", worst / MB, "others", others / MB, "cached", cached)
    assert worst + others <= MACHINE_128 - reserve // 2, (worst, others)
    assert all(c > 0 for c in cached[1:]), cached
    snap = scheduler.get_stats()["memory_guard"]
    assert snap["plan_ceiling_bytes"] < snap["ceiling_bytes"]
    assert device.ooms == 0


def test_a_big_idle_mac_keeps_the_device_ceiling(llama, monkeypatch):
    """Others hold 5 GB: the machine leaves more than the device ceiling,
    so nothing changes -- zero busy-machine actions."""
    scheduler, device, worst, cached, _ = _ladder(llama, monkeypatch, 50 * MB, turns=6)
    snap = scheduler.get_stats()["memory_guard"]
    assert snap["plan_ceiling_bytes"] == snap["ceiling_bytes"]
    counts = scheduler._memguard().counts
    assert counts["busy_machine_evictions"] == 0
    assert counts["busy_machine_fallbacks"] == 0


def test_gabee_64_gb_keeps_its_ceiling_with_the_others_it_runs(llama, monkeypatch):
    """Gabee: 64 GB, ceiling 50.10 GB. Its plan ceiling moves only when the
    other processes use more than physical - 50.10 GB - the reserve
    (68.7 - 50.1 - 5.5 = 13.1 GB): with 8 GB of others, unchanged."""
    from tests.test_memory_guard_session import GABEE_64_CEILING

    physical = int(687.2 * MB)
    device = _Device(physical=int(GABEE_64_CEILING / 0.97) + 1, scratch_scale=600.0)
    scheduler = _auto_budget_scheduler(llama, monkeypatch, device, max_num_seqs=4)
    device.install(monkeypatch, scheduler)
    _machine(monkeypatch, device, 80 * MB, physical=physical)
    assert scheduler._memguard_plan_ceiling() == scheduler._memguard_ceiling()
    scheduler._memguard_others_read = None
    _machine(monkeypatch, device, 140 * MB, physical=physical)
    assert scheduler._memguard_plan_ceiling() < scheduler._memguard_ceiling()


def test_others_growing_into_the_reserve_take_entries_back(llama, monkeypatch):
    """Idle engine with a full cache; the other processes then grow by 300
    MB: the next pressure tick gives entries back until the engine fits
    what the machine leaves, and the prefix budget follows."""
    scheduler, device, _worst, _cached, reading = _ladder(
        llama, monkeypatch, 50 * MB, turns=6
    )
    mac = scheduler.memory_aware_cache
    before = len(mac._entries)
    held = sum(int(e.memory_bytes) for e in mac._entries.values())
    assert before >= 3 and held > 40 * MB, (before, held)
    budget_before = scheduler._memguard().effective_prefix_budget
    footprint = scheduler._metal_footprint_bytes()
    reserve = int(memory_budget.SYSTEM_RESERVE_RATIO * MACHINE_128)
    # the others grow until the machine leaves the engine 30 MB less than
    # it holds now
    reading["others"] = MACHINE_128 - reserve - (footprint - 30 * MB)
    scheduler._memguard_others_read = None
    scheduler.evict_prefix_cache_under_pressure()
    counts = scheduler._memguard().counts
    assert counts["busy_machine_evictions"] >= 1
    assert len(mac._entries) < before
    assert scheduler._metal_footprint_bytes() <= footprint - 30 * MB
    assert scheduler._memguard().effective_prefix_budget < budget_before


def test_a_lone_request_is_never_refused_for_the_others(llama, monkeypatch):
    """The machine leaves less than the request needs, but the device fits
    it: it runs (alone, under the device ceiling), never a too-long."""
    device = _Device(physical=int(M4_MAX_128_CEILING / 0.97) + 1, scratch_scale=600.0)
    scheduler = _auto_budget_scheduler(llama, monkeypatch, device, max_num_seqs=4)
    device.install(monkeypatch, scheduler)
    # others leave the engine barely its weights
    _machine(monkeypatch, device, MACHINE_128 - int(0.08 * MACHINE_128) - 320 * MB)
    scheduler._memguard_others_read = None
    prompt = _new_tokens(0, 3000)
    outs = _drive(scheduler, [_request("lone", prompt, 4)])
    assert outs["lone"][-1].finished and not outs["lone"][-1].error
    assert scheduler._memguard().counts["busy_machine_fallbacks"] >= 1
