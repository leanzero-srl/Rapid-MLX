"""Review 12: one targeted, deterministic case per weakly-killed mutation (M3, M4, M18,
M21, M24, M25, M26). Unit-level, deterministic."""

import random
import types

from rapid_mlx.memory_guard import MemoryGuard, PeakPrior, StepShape, kv_growth_step
from rapid_mlx.scheduler import Scheduler

KA, KB, KO, KH = (48, 1), (48, 2), (48, 3), (40, 9)


def _sched():
    s = object.__new__(Scheduler)
    s.memory_aware_cache = types.SimpleNamespace(_entries={})
    s.running, s.waiting = {}, []
    s._memguard_lines = {}
    return s


def _line(s, owner, kind, first, last, turn, outcome="stop", continues=True):
    s._memguard_lines[owner] = {
        "first": first,
        "last": last,
        "turn": turn,
        "hint": types.SimpleNamespace(outcome=[outcome]),
        "rid": None,
        "prev": None,
        "continues": continues,
        "kind": kind,
    }
    s.memory_aware_cache._entries[f"e{owner}"] = types.SimpleNamespace(owner=owner)
    return owner


def test_m3_a_one_shot_call_of_another_kind_leaves_an_idle_chat_alone():
    s = _sched()
    a = _line(s, (("session", "A"), KA), KA, 1, 3, 1)
    _line(s, (("frame", 1, 2), KH), KH, 4, 4, 4, continues=False)  # a stranger's title
    assert a not in s._memguard_outlived_lines()


def test_m4_a_line_with_a_running_request_is_never_released():
    s = _sched()
    a = _line(s, (("session", "A"), KA), KA, 1, 5, 1)
    _line(s, (("session", "B"), KA), KA, 6, 9, 7)  # younger, same kind, user turn
    assert a in s._memguard_outlived_lines()  # control: A alone is left
    s.running = {"r": types.SimpleNamespace(_memguard_line_key=a)}
    assert a not in s._memguard_outlived_lines()


def test_m18_a_forgotten_main_still_releases_its_tool_ended_subagent():
    s = _sched()
    m = _line(s, (("session", "M"), KA), KA, 1, 10, 1)
    sub = _line(s, (("session", "S"), KB), KB, 3, 8, 3, outcome="tool_calls")
    assert sub not in s._memguard_outlived_lines()  # main is not left yet
    _line(s, (("session", "N"), KA), KA, 11, 13, 12)  # user leaves main
    assert sub in s._memguard_outlived_lines()
    del s.memory_aware_cache._entries[f"e{m}"]  # duress: main's entries go first
    assert sub in s._memguard_outlived_lines()


def test_m24_a_forgotten_new_window_still_releases_the_left_one():
    s = _sched()
    x = _line(s, (("session", "X"), KO), KO, 1, 4, 1)
    y = _line(s, (("session", "Y"), KO), KO, 5, 8, 6)
    assert x in s._memguard_outlived_lines()
    del s.memory_aware_cache._entries[f"e{y}"]
    assert x in s._memguard_outlived_lines()


G = 4096.0
PRIOR = PeakPrior(192.0, 251_904.0, G, 65_536.0)


def test_m21_a_context_term_over_the_whole_kv_falls_back_to_the_median():
    m = MemoryGuard(PRIOR, floor_tokens=kv_growth_step()).model
    # slopes per context token: chunk 256 -> 80,000, chunk 2048 -> 88,960:
    # least squares says a = 5, b = 78,720 > the whole KV (65,536).
    for _r in range(3):
        for x in range(20_000, 120_000, 6_000):
            for chunk, slope in ((256, 80_000.0), (2048, 88_960.0)):
                m.observe(StepShape(1, chunk, x), int(slope * x + 1e8))
    assert m.attention_bytes_per_unit > 40, m.attention_bytes_per_unit


def _truth(shape, rnd):
    return (48.0 * shape.units + G * shape.context + 1.36e6 * shape.tokens + 0.3e9) * (
        rnd.uniform(0.99, 1.01)
    )


def test_m25_three_outliers_beyond_the_span_limit_decay_back():
    rnd = random.Random(11)
    guard = MemoryGuard(PRIOR, floor_tokens=kv_growth_step())
    for x in range(10_000, 150_000, 2048):
        s = StepShape(1, 2048, x)
        guard.observe(s, int(_truth(s, rnd)))
    model = guard.model
    assert model._fit() is None
    settled = model.attention_bytes_per_unit
    peak = settled
    for x in (146_000, 148_000, 150_000):
        s = StepShape(1, 2048, x)
        guard.observe(s, int(_truth(s, rnd) + 3e9))
        peak = max(peak, model.attention_bytes_per_unit)
    assert peak > settled
    for _i in range(40):
        s = StepShape(1, 2048, 140_000 + rnd.randint(0, 10_000))
        guard.observe(s, int(_truth(s, rnd)))
    assert abs(model.attention_bytes_per_unit - settled) <= 0.01 * settled


def test_m26_repeated_overshoots_never_lift_a_zero_coefficient_in_one_step():
    guard = MemoryGuard(PRIOR, floor_tokens=kv_growth_step())
    model = guard.model
    model._a_learned, model._b_learned, model._a_fit = 0.0, G, 0.0
    for x in (60_000, 80_000, 100_000):
        s = StepShape(1, 2048, x)
        guard.observe(s, int(model.project(s) * 1.4), fit=True)
    assert model.attention_bytes_per_unit < 1.0, model.attention_bytes_per_unit


def _rec(seq, kind, outcome="stop", continues=True):
    return {
        "first": seq,
        "last": seq,
        "turn": seq,
        "hint": types.SimpleNamespace(outcome=[outcome]),
        "rid": None,
        "prev": None,
        "continues": continues,
        "kind": kind,
    }


def test_tombstones_are_bounded_by_the_conversations_held():
    """Review 12: one stale held line (an old one-shot request with no
    session id that ended on a tool call) kept EVERY later tombstone: 8,000
    of them after 8,000 short chats, 80 ms per call under the cache lock.
    Only a tombstone that decides a release is kept -- at most one per held
    conversation."""
    s = _sched()
    lines = s._memguard_lines
    seq = 1
    stale = (("frame", 1), (1, 1))
    lines[stale] = _rec(seq, (1, 1), "tool_calls", continues=False)
    s.memory_aware_cache._entries["stale"] = types.SimpleNamespace(owner=stale)
    held = [stale]
    for j in range(3):
        seq += 1
        owner = (("session", f"old{j}"), (48, 100 + j))
        lines[owner] = _rec(seq, (48, 100 + j), "tool_calls")
        s.memory_aware_cache._entries[f"old{j}"] = types.SimpleNamespace(owner=owner)
        held.append(owner)
    for i in range(2000):
        seq += 2
        owner = (("session", f"c{i}"), (48, i % 3))
        lines[owner] = _rec(seq, (48, i % 3))
        s.memory_aware_cache._entries[f"c{i}"] = types.SimpleNamespace(owner=owner)
        s._memguard_outlived_lines()
        del s.memory_aware_cache._entries[f"c{i}"]  # the count bound evicts it
        s._memguard_outlived_lines()
        assert len(s.__dict__.get("_memguard_forgotten", {})) <= len(held), i
