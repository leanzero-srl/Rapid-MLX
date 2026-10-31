# SPDX-License-Identifier: Apache-2.0
"""The free-memory floor's "others" reference follows SUSTAINED readings only
(lz.21, 2026-10-31).

lz.20 regression, measured on the workmule (M3 Ultra 137.4 GB, Jundot
Qwen3.8-Flash-Next, goose's growing-session replay): the floor ceiling is
``min(cap, physical - others - floor)`` and the others' reference rose at
once to any single reading. At turn 110 it read 40.95 GB (the idle machine
is 19.6 GB), the ceiling fell from 111.99 to 89.49 GB, the request did not
fit beside its own prefix hit and the session went cold at 167,508 tokens
(lz.19, without the floor: warm to 371,418). The status reads 30 s apart
around it saw the machine at 19.5-27 GB again within a minute, and a 1 s
status sampler on the confirming re-run never saw the others above 23 GB
while the reference still climbed: the high readings are moments the guard
alone caught, not memory another process held.

Rules tested here (memory_budget.OthersReference, Scheduler._memguard_
others_bytes): the engine's whole process -- max(phys_footprint, MLX active
+ cache), read with the page counts -- is out of "others"; a part rises only
to the lowest reading of the last ``sustain_s`` (15 s); at once only while
the machine swaps out; it falls after a quiet window to that window's
highest sustained reading. The negative control: another app really holding
30 GB for 30 s lowers the ceiling.
"""

from __future__ import annotations

import pytest

import rapid_mlx.memory_budget as memory_budget
from rapid_mlx.memory_budget import OthersReference, floor_room_bytes

GB = 10**9
PHYSICAL = 137_438_953_472  # workmule hw.memsize
CAP = 111_985_263_738  # lz.20's resolved limit on the mule (limit_bytes)
FLOOR = 7 * GB

# The lz.20 replay's /v1/status samples, 30 s apart (evidence-mule-2026-10-31-
# lz20-replay/mlimit3/samples.jsonl): used_bytes, engine_resident_bytes
# (phys_footprint), engine_metal_bytes (MLX active + cache), in GB. The turn
# 110 cold requeue is at index 23 (t=1793440636), where the guard's reference
# stood at 40.95 GB.
REPLAY = [
    (103.71, 81.57, 80.71), (104.97, 79.19, 78.13), (109.15, 79.62, 78.70),
    (110.12, 83.18, 82.23), (100.34, 80.42, 79.40), (105.75, 82.79, 81.43),
    (111.04, 87.61, 86.40), (113.84, 88.19, 86.85), (110.22, 89.84, 88.44),
    (107.56, 86.90, 85.46), (110.55, 87.21, 85.75), (120.57, 93.67, 85.06),
    (109.43, 89.72, 88.18), (114.06, 94.35, 89.02), (112.84, 88.82, 87.27),
    (107.08, 84.70, 83.14), (112.36, 92.63, 91.05), (110.83, 90.79, 87.46),
    (111.96, 92.19, 90.54), (116.61, 92.17, 89.73), (118.14, 95.00, 92.99),
    (119.44, 91.41, 88.66), (122.43, 88.80, 86.57), (120.10, 82.27, 80.49),
    (100.00, 82.27, 80.49), (100.10, 81.35, 79.43), (102.51, 82.72, 80.72),
    (104.92, 84.45, 81.91), (106.77, 83.70, 81.06), (106.80, 84.03, 82.11),
    (103.79, 85.35, 83.19), (105.13, 86.16, 81.93), (104.03, 85.86, 83.94),
    (105.08, 86.22, 84.30),
]
SPIKE_AT = 23
SPIKE_OTHERS = 40.95  # the guard's reference at the cold requeue


IDLE = min(CAP, PHYSICAL - int(19.6 * GB) - FLOOR)  # the idle mule: floor-bound, 110.84


def _parts(used, resid, metal):
    engine = max(resid, metal)
    return int((used - engine) * GB), int(max(0.0, resid - metal) * GB)


def _ceiling(ref):
    room = floor_room_bytes(physical=PHYSICAL, others=ref, floor=FLOOR)
    return CAP if room is None else min(CAP, room)


def _feed(ref, t0, seconds, rest, extra=0, swapouts=0, step=1.0):
    t = t0
    out = []
    while t < t0 + seconds:
        out.append(ref.update(t, rest, extra, swapouts))
        t += step
    return t, out


def _replay(spike_seconds):
    """The replay's readings at their real times (one every 30 s), and the
    turn-110 reading of SPIKE_OTHERS GB as a burst read every second for
    ``spike_seconds`` (rest = the reading, i.e. what lz.20 counted)."""
    ref = OthersReference()
    ceilings, refs = [], []
    for i, (used, resid, metal) in enumerate(REPLAY):
        t = 30.0 * i
        if i == SPIKE_AT:
            _t, out = _feed(ref, t, spike_seconds, int(SPIKE_OTHERS * GB))
        else:
            out = [ref.update(t, *_parts(used, resid, metal), 0)]
        refs += out
        ceilings += [_ceiling(v) for v in out]
    return ceilings, refs


@pytest.mark.parametrize("spike_seconds", [1.0, 5.0, 12.0])
def test_the_replays_spike_never_lowers_the_ceiling(spike_seconds):
    """RED on lz.20 (reference = max of any reading: ceiling 89.49 GB, below
    the 90.68 GB the turn-110 request needed at chunk 256). GREEN: the
    40.95 GB reading (held 1, 5 or 12 s) never becomes the reference. Read
    30 s apart, two high readings in a row count as sustained (nothing
    between them says otherwise), so the machine's real pressure just before
    the spike (samples 21-22: 1.3 GB of free pages, the compressor growing
    89k -> 412k pages) still lowers the ceiling, to 94.6 GB: more than turn
    110 needed. Outside that episode it stays at 102 GB or more, with a
    median of 105 GB or more (lz.19's whole cap: 103.9 GB)."""
    ceilings, refs = _replay(spike_seconds)
    assert max(refs) < 36.0 * GB, max(refs) / GB  # never the 40.95 spike
    assert min(ceilings) >= 94.0 * GB, min(ceilings) / GB  # > 90.68 needed
    calm = ceilings[:21] + ceilings[-6:]
    assert min(calm) >= 102.0 * GB, min(calm) / GB
    assert sorted(calm)[len(calm) // 2] >= 105.0 * GB


def test_lz20s_rule_on_the_same_sequence_drops_the_ceiling():
    """The control arm: the old rule (rise at once) on the same sequence
    reproduces the measured 89.49 GB ceiling -- the test discriminates."""
    ceilings = []
    ref = None
    for i, (used, resid, metal) in enumerate(REPLAY):
        reading = SPIKE_OTHERS * GB if i == SPIKE_AT else (used - metal) * GB
        ref = reading if ref is None or reading > ref else ref
        ceilings.append(_ceiling(int(ref)))
    assert min(ceilings) == pytest.approx(89.49 * GB, abs=0.1 * GB)


def test_another_app_holding_30_gb_for_30_s_lowers_the_ceiling():
    """Negative control: the idle mule's 19.6 GB, then an app takes 30 GB
    more and keeps it 30 s: the reference follows within the sustain span
    and the ceiling drops to physical - 49.6 - 7 = ~80.8 GB."""
    ref = OthersReference()
    t, _ = _feed(ref, 0.0, 60.0, int(19.6 * GB))
    assert _ceiling(ref.value) == IDLE
    t, out = _feed(ref, t, 30.0, int(49.6 * GB))
    assert _ceiling(out[-1]) == pytest.approx(PHYSICAL - 49.6 * GB - FLOOR, abs=0.2 * GB)
    # not before the sustain span: the first 14 s of the grab keep the cap
    assert all(_ceiling(v) == IDLE for v in out[:14])
    # it gives the memory back: the reference falls only after a quiet
    # window (>= half of OTHERS_WINDOW_S), then to the machine's real level
    t, out = _feed(ref, t, 30.0, int(19.6 * GB))
    assert _ceiling(out[-1]) < IDLE
    t, out = _feed(ref, t, memory_budget.OTHERS_WINDOW_S + 20, int(19.6 * GB))
    assert _ceiling(out[-1]) == IDLE


def test_a_burst_shorter_than_the_sustain_span_is_ignored():
    ref = OthersReference()
    t, _ = _feed(ref, 0.0, 60.0, int(19.6 * GB))
    t, out = _feed(ref, t, memory_budget.OTHERS_SUSTAIN_S - 2, int(49.6 * GB))
    t, out2 = _feed(ref, t, 60.0, int(19.6 * GB))
    assert all(_ceiling(v) == IDLE for v in out + out2)


def test_sparse_reads_still_see_a_sustained_grab():
    """The guard reads only while it plans: two readings 30 s apart, both
    high, are a sustained reading (nothing between them says otherwise)."""
    ref = OthersReference()
    _feed(ref, 0.0, 60.0, int(19.6 * GB), step=30.0)
    ref.update(90.0, int(49.6 * GB), 0, 0)
    assert ref.value == int(19.6 * GB)  # one reading: not yet
    ref.update(120.0, int(49.6 * GB), 0, 0)
    assert ref.value == int(49.6 * GB)


def test_swapping_raises_the_reference_at_once():
    """Real pressure: the machine swaps out -- no waiting for the span."""
    ref = OthersReference()
    t, _ = _feed(ref, 0.0, 60.0, int(19.6 * GB), swapouts=100)
    assert ref.update(t, int(49.6 * GB), 0, 101) == int(49.6 * GB)


def test_the_reference_never_falls_while_the_machine_swaps():
    ref = OthersReference()
    t, _ = _feed(ref, 0.0, 30.0, int(40 * GB), swapouts=5)
    so = 5
    for _ in range(300):
        so += 1
        ref.update(t, int(25 * GB), 0, so)
        t += 1.0
    assert ref.value == int(40 * GB)


def test_the_engines_own_memory_is_not_others():
    """The engine's phys_footprint - MLX active - cache (its memory outside
    MLX's tally) is a separate part: a transient of 8.6 GB (replay sample 11:
    footprint 93.67 vs Metal 85.06) does not move the reference; a lasting
    one (Python objects the engine keeps) does, as engine memory."""
    ref = OthersReference()
    t, _ = _feed(ref, 0.0, 60.0, int(19.6 * GB), extra=int(0.9 * GB))
    base = ref.value
    t, out = _feed(ref, t, 5.0, int(19.6 * GB), extra=int(8.6 * GB))
    assert max(out) == base
    t, out = _feed(ref, t, 30.0, int(19.6 * GB), extra=int(3.0 * GB))
    assert out[-1] == int(19.6 * GB) + int(3.0 * GB)
    assert ref.last["engine_extra_ref_bytes"] == int(3.0 * GB)
    assert ref.last["others_ref_bytes"] == int(19.6 * GB)


def test_the_scheduler_takes_the_whole_engine_process_out(monkeypatch):
    """Scheduler path: the reading's phys_footprint (read in the same call
    as the page counts) is the engine when it exceeds MLX active + cache."""
    import time as _time

    import rapid_mlx.scheduler as sched

    class _S:
        _current_metal_active_bytes = lambda self: int(80 * GB)  # noqa: E731
        _current_metal_cache_bytes = lambda self: int(1 * GB)  # noqa: E731

    s = _S()
    clock = {"t": 100.0}
    monkeypatch.setattr(_time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(memory_budget, "system_physical_memory", lambda: PHYSICAL)
    reading = {"r": memory_budget.MachineMemory(int(110 * GB), 0, None, int(90 * GB))}
    monkeypatch.setattr(memory_budget, "system_memory_used", lambda: reading["r"])
    first = sched.Scheduler._memguard_others_bytes(s)
    # used 110 - engine max(90, 81) = 20 GB others + 9 GB engine extra
    assert first == int(20 * GB) + int(9 * GB)
    # the engine frees 9 GB outside MLX (footprint 81): a 1 s moment where
    # the page counts still hold it is never read as another app
    for k in range(1, 4):
        clock["t"] += 1.0
        reading["r"] = memory_budget.MachineMemory(int(110 * GB), 0, None, int(81 * GB))
        assert sched.Scheduler._memguard_others_bytes(s) <= first
    # without a footprint (non-Darwin) the engine is its Metal footprint
    reading["r"] = memory_budget.MachineMemory(int(110 * GB), 0)
    s._memguard_others_read = None
    assert sched.Scheduler._memguard_others_bytes(s) == int(29 * GB)
