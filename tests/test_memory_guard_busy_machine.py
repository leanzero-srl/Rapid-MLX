# SPDX-License-Identifier: Apache-2.0
"""The engine's OPTIONAL memory follows what the MACHINE leaves it (RC10 128
GB ladder, 2026-10-16; review of c976b2a3a).

M4 Max 128 GB, device ceiling 99.09 GB, the engine beside the user's own
apps: Metal active reached 91.0 GB (40.2 GB of it prefix entries), the
machine went 46.8 GB into swap, the kernel's pressure LEVEL said normal.

The other processes are measured as ``host_statistics64`` memory in use
(wired + anonymous - purgeable + compressor: Activity Monitor's "Memory
Used") less the engine's own Metal footprint, with hysteresis: the
reference rises at once and falls only to a 2-minute window's highest
reading, and only when nothing was swapped out in it (swapped-out pages stop
counting as used: an engine that swaps the others out must not read that as
room). ``kern.memorystatus_level`` -- c976b2a3a's signal -- counts anonymous
and swapped-out pages as free (75% "free" while 99 GB were in use) and moved
5 points on one GPU wire/unwire.

What follows the machine: the prefix budget, a mid-turn COPY of an entry,
entries kept beside a step (given back up to EVICT_HEAD before it is
planned, and on the pressure tick). What does not: the engine's necessary
memory -- weights, live KV, a step's transient and chunk, a turn's resume
point handed over -- which plans under the device ceiling exactly as on an
idle Mac (c976b2a3a planned steps under the lower ceiling: Gabee at 16 GB of
others evicted newest entries and missed turns).

The machine here is a model of the real signal: integer pages, every
engine byte counted whether wired or not (a jitter of the measured
+-0.15 GB), and an OS that swaps the OTHERS out when the engine and they
exceed physical RAM -- their swapped pages leave "used" and bump the
swap-out counter. At 1/100 of the bytes like the session tests.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

import rapid_mlx.memory_budget as memory_budget
from tests.test_memory_guard_session import (  # noqa: F401
    GABEE_64_CEILING,
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
MACHINE_64 = int(687.2 * MB)  # Gabee: 68.7 GB
PAGE = 16384


class _Machine:
    """The rest of the machine holds ``others`` bytes; the OS swaps them out
    (sticky: a swapped page comes back only when touched) whenever the
    engine and they exceed physical RAM."""

    def __init__(self, monkeypatch, device, others: int, physical: int):
        self.device = device
        self.others = others
        self.physical = physical
        self.swapped = 0
        self.swapouts = 0
        self.reads = 0
        monkeypatch.setattr(memory_budget, "system_physical_memory", lambda: physical)
        monkeypatch.setattr(memory_budget, "system_memory_used", self.read)
        device._machine = self

    def engine(self) -> int:
        return self.device.active() + mx.get_cache_memory()

    def settle(self, engine: int | None = None) -> None:
        engine = self.engine() if engine is None else engine
        resident = max(0, self.others - self.swapped)
        over = engine + resident - self.physical
        if over > 0:
            self.swapped += over
            self.swapouts += -(-over // PAGE)

    def read(self):
        self.reads += 1
        self.settle()
        # a GPU wire/unwire moves engine bytes between wired and anonymous:
        # both are "used"; only the measured +-0.15 GB jitter remains
        jitter = (1 if self.reads % 2 else -1) * int(0.0011 * self.physical)
        resident = max(0, self.others - self.swapped)
        pages = max(0, self.engine() + resident + jitter) // PAGE
        return memory_budget.MachineMemory(pages * PAGE, self.swapouts)


def _scheduler(llama, monkeypatch, ceiling):
    device = _Device(physical=int(ceiling / 0.97) + 1, scratch_scale=600.0)
    scheduler = _auto_budget_scheduler(llama, monkeypatch, device, max_num_seqs=4)
    device.install(monkeypatch, scheduler)
    return scheduler, device


def _drive_on(machine, scheduler, requests, **kw):
    """``_drive`` with the OS settling after every step (the peak counts)."""
    real = scheduler.step

    def step():
        out = real()
        if machine is not None:
            machine.settle(max(machine.engine(), machine.device.peak()))
        return out

    scheduler.step = step
    try:
        return _drive(scheduler, requests, **kw)
    finally:
        scheduler.step = real


def _ladder(llama, monkeypatch, others, *, ceiling=M4_MAX_128_CEILING,
            physical=MACHINE_128, turns=24, beside=True):
    """A session growing +500 tokens a turn; a subagent's 60k-token run
    beside it (``beside``) or alone after it, every turn -- every side entry
    stays cached while the budget has room."""
    scheduler, device = _scheduler(llama, monkeypatch, ceiling)
    machine = (
        _Machine(monkeypatch, device, others, physical) if others is not None else None
    )
    prompt = _new_tokens(0, 600)
    misses = []
    for turn in range(turns):
        main = _request(f"main-{turn}", prompt, 4, session="main")
        sub = _request(
            f"sub-{turn}", _side_prompt(turn, SUBAGENT_TOKENS), 8, session=f"sub-{turn}"
        )
        if beside:
            outs = _drive_on(machine, scheduler, [main], arrive={3: [sub]})
        else:
            outs = _drive_on(machine, scheduler, [main])
            outs.update(_drive_on(machine, scheduler, [sub]))
        for rid in (main.request_id, sub.request_id):
            assert outs[rid][-1].finished and not outs[rid][-1].error, rid
        if turn and len(prompt) - outs[main.request_id][-1].cached_tokens > 508:
            misses.append(turn)
        prompt = prompt + _tokens(outs[main.request_id]) + _new_tokens(turn + 1, 496)
    return scheduler, device, machine, misses


def test_a_busy_128_gb_mac_stays_out_of_swap(llama, monkeypatch):
    """Others hold 40 GB (400 MB here). RED on 150a141ee and c976b2a3a (the
    engine's cache grows toward the 99.09 GB device ceiling and the OS swaps
    the others out). GREEN: nothing is ever swapped out, the session
    resumes every turn, and the guard's own planning is untouched."""
    scheduler, device, machine, misses = _ladder(llama, monkeypatch, 400 * MB)
    assert machine.swapouts == 0, (machine.swapped / MB, device.max_seen / MB)
    assert misses == [], misses
    counts = scheduler._memguard().counts
    assert counts["busy_machine_evictions"] >= 1
    assert counts["newest_evictions"] == 0 and counts["chunks_reduced"] == 0
    snap = scheduler.get_stats()["memory_guard"]
    assert snap["cache_ceiling_bytes"] < snap["ceiling_bytes"]
    assert device.ooms == 0


def test_a_big_idle_mac_keeps_the_device_ceiling(llama, monkeypatch):
    """Others hold 5 GB: the machine leaves more than the device ceiling,
    so nothing changes -- zero busy-machine actions."""
    scheduler, _device, machine, misses = _ladder(llama, monkeypatch, 50 * MB, turns=8)
    snap = scheduler.get_stats()["memory_guard"]
    assert snap["cache_ceiling_bytes"] == snap["ceiling_bytes"]
    assert scheduler._memguard().counts["busy_machine_evictions"] == 0
    assert machine.swapouts == 0 and misses == []


@pytest.mark.parametrize("others", [120, 160])
def test_gabee_keeps_its_pass_beside_the_others_it_runs(llama, monkeypatch, others):
    """Gabee, 64 GB, ceiling 50.10 GB, main turn then its subagent ALONE
    (the reviewer's probe): with 12 GB of others (measured) and 16 GB (where
    c976b2a3a missed turns 24-25 with 3-6 newest evictions), the guard's
    degradations are exactly those of the machine-blind engine and the
    session misses nothing."""
    blind, _d, _m, blind_misses = _ladder(
        llama, monkeypatch, None, ceiling=GABEE_64_CEILING, physical=MACHINE_64,
        turns=26, beside=False,
    )
    scheduler, _d2, _m2, misses = _ladder(
        llama, monkeypatch, others * MB, ceiling=GABEE_64_CEILING,
        physical=MACHINE_64, turns=26, beside=False,
    )
    assert misses == blind_misses == []
    a, b = scheduler._memguard().counts, blind._memguard().counts
    for key in ("newest_evictions", "chunks_reduced", "deferrals", "turn_store_skips"):
        assert a[key] <= b[key], (key, a[key], b[key])


def test_the_reference_never_falls_while_the_machine_swaps(llama, monkeypatch):
    """The others get swapped out (their 'used' falls): the reference stays
    at its high-water mark, so the engine cannot read that as room; it
    falls only after a window with no swap-out."""
    import rapid_mlx.scheduler as sched

    scheduler, device = _scheduler(llama, monkeypatch, M4_MAX_128_CEILING)
    machine = _Machine(monkeypatch, device, 400 * MB, MACHINE_128)
    clock = {"t": 0.0}
    monkeypatch.setattr(sched, "_MEMGUARD_SYSTEM_READ_S", 0.0)
    import time as _time

    monkeypatch.setattr(_time, "monotonic", lambda: clock["t"])
    first = scheduler._memguard_others_bytes()
    assert first == pytest.approx(400 * MB, abs=0.003 * MACHINE_128)
    machine.swapped = 150 * MB  # the OS swaps 150 MB of them out, and keeps
    for step in range(1, 200):  # swapping for a while
        clock["t"] = step * 1.0
        machine.swapouts += 1
        assert scheduler._memguard_others_bytes() >= first - 0.003 * MACHINE_128
    # a whole window with no swap-out: the reference follows the real reading
    for step in range(200, 200 + 2 * int(sched._MEMGUARD_OTHERS_WINDOW_S)):
        clock["t"] = step * 1.0
        ref = scheduler._memguard_others_bytes()
    assert ref == pytest.approx(250 * MB, abs=0.003 * MACHINE_128)


def test_a_budget_shrink_evicts_in_the_guards_order(llama, monkeypatch):
    """The machine leaves less: the prefix budget shrinks through the
    guard's ONE order (the conversation's newest entry stays), not the
    cache's plain LRU."""
    scheduler, device, machine, _misses = _ladder(llama, monkeypatch, 50 * MB, turns=6)
    mac = scheduler.memory_aware_cache
    newest_main = max(
        (k for k, e in mac._entries.items() if e.owner and "main" in str(e.owner)),
        key=len,
    )
    footprint = scheduler._metal_footprint_bytes()
    reserve = int(memory_budget.SYSTEM_RESERVE_RATIO * MACHINE_128)
    before = len(mac._entries)
    # the user opens something big: the machine now leaves the engine 60 MB
    # less than it holds
    machine.others = MACHINE_128 - reserve - (footprint - 60 * MB)
    scheduler._memguard_others_read = None
    scheduler.evict_prefix_cache_under_pressure()
    assert newest_main in mac._entries
    assert len(mac._entries) < before
    assert scheduler._memguard().counts["busy_machine_evictions"] >= 1
