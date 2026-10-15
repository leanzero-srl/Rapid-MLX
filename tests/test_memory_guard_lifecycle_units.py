"""Review 12: one targeted, deterministic case per weakly-killed mutation (M3, M4, M18,
M21, M24, M25, M26). Unit-level, deterministic."""

import random
import time
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
    for i in range(3):  # Y's tombstone is kept: needed under the "left" rule
        assert x in s._memguard_outlived_lines(), i


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
    Only a tombstone that witnesses a recorded conversation under some rule
    is kept, and none of these witnesses anything for the held lines."""
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
    started = time.perf_counter()
    for i in range(8000):
        seq += 2
        owner = (("session", f"c{i}"), (48, i % 3))
        lines[owner] = _rec(seq, (48, i % 3))
        s.memory_aware_cache._entries[f"c{i}"] = types.SimpleNamespace(owner=owner)
        s._memguard_outlived_lines()
        del s.memory_aware_cache._entries[f"c{i}"]  # the count bound evicts it
        s._memguard_outlived_lines()
        assert len(s.__dict__.get("_memguard_forgotten", {})) <= len(held), i
    # 16,000 calls: the old growth cost 80 ms per call at the end
    assert time.perf_counter() - started < 8.0


# -- review 13: tombstones and history -------------------------------------

OW = (("session", "O"), KA)
M = (("session", "M"), KA)
S = (("session", "S"), KB)
Z = (("session", "Z"), KA)


def test_a_resumed_older_window_still_releases_the_finished_subagent():
    """Review 13 (the reviewer's repro): rule 1 released S through the older
    window OW, so main's tombstone decided nothing and was dropped; the user
    resumed OW and nothing released S again."""
    s = _sched()
    _line(s, OW, KA, 1, 5, 1)  # older window, used before
    _line(s, M, KA, 2, 7, 2)  # main: hands off, takes the result, stops
    _line(s, S, KB, 3, 6, 3, outcome="tool_calls")  # subagent ended on a tool call
    s._memguard_lines[OW].update(last=9, turn=9)  # the user goes back to OW
    _line(s, Z, KA, 10, 11, 11)  # then opens Z
    assert {S, OW, M} <= s._memguard_outlived_lines()
    del s.memory_aware_cache._entries[f"e{M}"]  # duress: main's entries go
    assert S in s._memguard_outlived_lines()
    del s.memory_aware_cache._entries[f"e{OW}"]  # then OW's
    assert S in s._memguard_outlived_lines()
    _line(s, OW, KA, 12, 13, 13)  # the user resumes OW (same session id)
    for i in range(5):  # any amount of further traffic
        q = (("session", f"Q{i}"), KA)
        _line(s, q, KA, 20 + 3 * i, 21 + 3 * i, 21 + 3 * i)
        assert S in s._memguard_outlived_lines(), i


def test_the_last_forgotten_of_two_witnesses_resumes():
    """Review 13, second variant: of two forgotten witnesses only the latest
    was kept, and the subagent was held again once that one resumed."""
    s = _sched()
    _line(s, M, KA, 1, 1, 1)
    _line(s, OW, KA, 2, 2, 2)
    _line(s, S, KB, 3, 4, 3, outcome="tool_calls")  # M's subagent, tool-ended
    s._memguard_lines[M].update(last=6, turn=6)  # the user types in M
    s._memguard_lines[OW].update(last=8, turn=8)  # and then in OW
    _line(s, Z, KA, 9, 10, 10)  # then opens Z
    assert S in s._memguard_outlived_lines()
    del s.memory_aware_cache._entries[f"e{M}"]
    del s.memory_aware_cache._entries[f"e{OW}"]
    assert S in s._memguard_outlived_lines()
    _line(s, OW, KA, 12, 13, 13)  # the user resumes OW
    for i in range(4):
        _line(s, (("session", f"Q{i}"), KA), KA, 20 + 3 * i, 21 + 3 * i, 21 + 3 * i)
    assert S in s._memguard_outlived_lines(), "pinned"


def test_a_resumed_main_alone_is_still_older_than_its_subagent():
    """The only evidence is the conversation that comes back: main typed
    after its tool-ended subagent, was left and forgotten, and resumes with
    a helper call first (no user turn of its own). It is the same
    conversation -- older than S, its user turn after S still counts."""
    s = _sched()
    _line(s, M, KA, 1, 7, 7)  # main: the user typed after S's last request
    _line(s, S, KB, 3, 6, 3, outcome="tool_calls")
    _line(s, Z, KA, 8, 9, 9)  # the user opens Z
    assert S in s._memguard_outlived_lines()
    del s.memory_aware_cache._entries[f"e{M}"]
    assert S in s._memguard_outlived_lines()
    # main comes back: a helper call of its session first (another frame,
    # one-shot), then its conversation frame
    _line(s, (("session", "M"), KH), KH, 10, 10, 10, continues=False)
    assert S in s._memguard_outlived_lines()
    _line(s, M, KA, 11, 11, 11, continues=False)  # first request after return
    assert S in s._memguard_outlived_lines()


def test_a_lost_frame_does_not_make_a_conversation_younger():
    """A conversation that changed frames (an extension enabled mid-session)
    loses its OLDER frame's entries: it still began before S."""
    s = _sched()
    w1 = _line(s, (("session", "W"), KA), KA, 1, 4, 1)
    _line(s, S, KB, 5, 6, 5, outcome="tool_calls")
    _line(s, (("session", "W"), KO), KO, 10, 12, 12)  # W's new frame, typed after S
    assert S in s._memguard_outlived_lines()
    del s.memory_aware_cache._entries[f"e{w1}"]  # the old frame's entries go
    assert S in s._memguard_outlived_lines()
    for i in range(3):
        _line(s, (("session", f"Q{i}"), KA), KA, 20 + 3 * i, 21 + 3 * i, 21 + 3 * i)
        assert S in s._memguard_outlived_lines(), i


def test_a_tombstone_is_kept_for_any_rule_not_only_the_deciding_one():
    """S (tool-ended subagent) is released by the chain through two parents
    that ran after it: P (recorded, released: the user left it for Z) and
    main (forgotten). P's tool then returns -- P is no longer released --
    and only main's tombstone still releases S."""
    s = _sched()
    _line(s, M, KA, 1, 7, 1)  # main took the result (a tool round), stopped
    p = _line(s, (("session", "P"), KA), KA, 2, 8, 2, outcome="tool_calls")
    _line(s, S, KB, 3, 6, 3, outcome="tool_calls")
    _line(s, Z, KA, 9, 10, 10)  # the user opens Z: main and P left
    assert {S, M, p} <= s._memguard_outlived_lines()
    del s.memory_aware_cache._entries[f"e{M}"]
    assert S in s._memguard_outlived_lines()
    s._memguard_lines[p].update(last=11, hint=types.SimpleNamespace(outcome=["stop"]))
    out = s._memguard_outlived_lines()
    assert p not in out  # P acted again: not left
    assert S in out


def test_tombstones_stay_bounded_beside_a_released_line_that_stays():
    """A released conversation whose entry stays (a shared prefix it stored
    keeps being hit) is witnessed by EVERY later conversation; the
    tombstones kept for it are bounded by what is recorded."""
    s = _sched()
    a = _line(s, (("session", "A"), KA), KA, 1, 2, 1)
    seq = 3
    for i in range(2000):
        owner = (("session", f"c{i}"), KA)
        _line(s, owner, KA, seq, seq + 1, seq + 1)
        seq += 2
        assert a in s._memguard_outlived_lines()
        del s.memory_aware_cache._entries[f"e{owner}"]
        assert a in s._memguard_outlived_lines()
        recorded = len({o[0] for o in s._memguard_lines})
        assert len(s._memguard_forgotten) <= 3 * recorded, i


def test_a_returning_conversation_turn_counts_from_its_first_request():
    """Main R hands off to subagent L (another kind), which ends on a tool
    call; R is forgotten. The user comes back to R and types: R's FIRST
    request after the return continues R (no predecessor to continue yet),
    so its user turn releases L at once, not one request later."""
    s = _sched()
    r = _line(s, (("session", "R"), KA), KA, 1, 2, 1)
    sub = _line(s, (("session", "L"), KB), KB, 3, 5, 3, outcome="tool_calls")
    assert sub not in s._memguard_outlived_lines()
    del s.memory_aware_cache._entries[f"e{r}"]  # R forgotten
    assert sub not in s._memguard_outlived_lines()
    _line(s, r, KA, 7, 7, 7, continues=False)  # back to R: first request
    assert sub in s._memguard_outlived_lines()


def test_an_older_window_of_the_builds_kind_comes_back():
    """Review 10's shape across a forgetting: a build X waits on its tool;
    the older window R of its own kind was forgotten; the user goes back to
    R and types. R is older than X -- not a new window the user left X for
    -- so X stays held."""
    s = _sched()
    r = _line(s, (("session", "R"), KA), KA, 1, 3, 3)
    x = _line(s, (("session", "X"), KA), KA, 4, 6, 4, outcome="tool_calls")
    assert x not in s._memguard_outlived_lines()
    del s.memory_aware_cache._entries[f"e{r}"]  # R forgotten
    assert x not in s._memguard_outlived_lines()
    _line(s, r, KA, 7, 7, 7, continues=False)  # back to R: the user types
    assert x not in s._memguard_outlived_lines()
    # control: a NEW window of its kind is the user leaving X
    _line(s, (("session", "N"), KA), KA, 8, 9, 9)
    assert x in s._memguard_outlived_lines()


def test_a_forgotten_main_keeps_releasing_its_finished_subagent():
    """A subagent that finished (its last request a reply) is released by
    its main, which ran after it; main is then forgotten. Main's tombstone
    is needed under rule 1 only -- and it is kept, call after call."""
    s = _sched()
    _line(s, M, KA, 1, 5, 1)  # main took the result after the subagent's end
    sub = _line(s, S, KB, 2, 3, 2)
    assert sub in s._memguard_outlived_lines()
    del s.memory_aware_cache._entries[f"e{M}"]
    for i in range(3):
        assert sub in s._memguard_outlived_lines(), i


def test_a_second_needed_tombstone_is_kept_while_there_is_room():
    """S (tool-ended subagent) is released through two forgotten parents:
    main (done) and P (forgotten while its tool ran). P's tool then returns:
    P is recorded again and no longer released, so only main's tombstone
    still releases S. The latest needed tombstone alone (P's) is not
    enough; the others are kept while they fit the bound."""
    s = _sched()
    _line(s, M, KA, 1, 7, 1)  # main took the result, stopped
    p = _line(s, (("session", "P"), KA), KA, 2, 8, 2, outcome="tool_calls")
    _line(s, S, KB, 3, 6, 3, outcome="tool_calls")
    _line(s, Z, KA, 9, 10, 10)  # the user opens Z
    assert S in s._memguard_outlived_lines()
    del s.memory_aware_cache._entries[f"e{M}"]
    del s.memory_aware_cache._entries[f"e{p}"]  # evicted while its tool ran
    assert S in s._memguard_outlived_lines()
    _line(s, p, KA, 11, 11, 0, continues=False)  # P's tool result comes back
    out = s._memguard_outlived_lines()
    assert p not in out
    assert S in out


def test_a_forgotten_session_keeps_only_its_own_kinds():
    """A session's helper call (another frame, one-shot) is not of the
    session's KIND. Once the session is forgotten its tombstone carries its
    conversation frames' kinds only: a pending one-shot line of the helper's
    kind is not released by it, before or after the forgetting."""
    s = _sched()
    x = _line(s, (("frame", 5, 1), KH), KH, 1, 1, 1, "tool_calls", continues=False)
    m = _line(s, M, KA, 2, 4, 4)
    mh = _line(s, (("session", "M"), KH), KH, 3, 3, 3, continues=False)
    assert x not in s._memguard_outlived_lines()
    del s.memory_aware_cache._entries[f"e{m}"]
    del s.memory_aware_cache._entries[f"e{mh}"]
    assert x not in s._memguard_outlived_lines()


def test_past_the_bound_the_most_recently_active_tombstone_stays():
    """Past the bound only the latest needed tombstone of a (conversation,
    rule) is sure to stay: the one most recently active, the likeliest to
    come back. Build X waits on its tool; two older windows of its kind are
    forgotten; the user goes back to the latest of them: it is still older
    than X, so X stays held."""
    s = _sched()
    r1 = _line(s, (("session", "R1"), KA), KA, 1, 2, 2)
    r2 = _line(s, (("session", "R2"), KA), KA, 3, 4, 4)
    x = _line(s, (("session", "X"), KA), KA, 5, 7, 5, outcome="tool_calls")
    assert x not in s._memguard_outlived_lines()
    del s.memory_aware_cache._entries[f"e{r1}"]
    s._memguard_outlived_lines()
    del s.memory_aware_cache._entries[f"e{r2}"]
    assert x not in s._memguard_outlived_lines()
    assert len(s._memguard_forgotten) == 1  # one recorded conversation
    _line(s, r2, KA, 8, 8, 8, continues=False)  # back to R2: the user types
    assert x not in s._memguard_outlived_lines()


def test_past_the_bound_every_rule_keeps_its_latest():
    """Past the bound each (conversation, rule) keeps its own latest needed
    tombstone. Subagent S is released by the chain through its forgotten
    main; an older window R of S's own kind, active later than main, is
    forgotten too. R is the latest tombstone older than S (its age rule),
    main the only chain witness: both stay, and S stays released."""
    s = _sched()
    r = _line(s, (("session", "R"), KB), KB, 1, 8, 1)
    _line(s, M, KA, 2, 7, 2)
    _line(s, S, KB, 3, 6, 3, outcome="tool_calls")
    del s.memory_aware_cache._entries[f"e{r}"]
    del s.memory_aware_cache._entries[f"e{M}"]
    for i in range(3):
        assert S in s._memguard_outlived_lines(), i
