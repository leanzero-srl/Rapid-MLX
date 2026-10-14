"""Eviction order: a conversation's dead branch goes before another
conversation's head.

Gabee 64 GB, Q8 AREX-2 (a hybrid: non-trimmable cache), the concurrent
ladder, b06acd8ea vs 45e157bab (2026-10-14). Each main turn stores its
message-boundary entry and its end-of-turn entry; the next main turn (a
fixed history, so the generated reply is not in it) resumes from the
boundary and has LEFT the end-of-turn entry behind: a dead branch, which
b06acd8ea itself drops at the line's next store ("dropped 1 its
conversation left behind"). Between the two, a ~22k side subagent (a new
session id each turn, the same content) stored its own boundary.

At main's admission the guard had to free memory (real duress: footprint
44.94 GB + a 5.85 GB step over a 50.10 GB ceiling). Both kinds of entry
ranked EVICT_HEAD and the shorter went first, so it evicted the side's two
entries and kept main's dead 72,260-token branch -- which the store dropped
seconds later anyway. The side then missed, re-prefilled 22.6k tokens at the
floor chunk, and that prefill had to evict main's 73,223-token boundary:
main's next turns cached 72,199 instead of 73,223.

A head a NEWER request of its own conversation (same session, same frame)
has diverged from keeps the level its live use gives it (unused), below
the heads of other conversations."""

import pytest

from rapid_mlx.memory_guard import EVICT_HEAD, EVICT_NEEDED, EVICT_UNUSED
from tests.test_memory_guard_session import (  # noqa: F401
    _GEN_PROMPT,
    _hybrid_run,
    _hybrid_scheduler,
    _hybrid_tokens,
    _no_metal,
    _request,
    _tiny_hybrid,
    _tiny_tokenizer,
    cpu,
)

FRAME = 300
MAIN = {"session": "ladder-main", "frame": FRAME}


def _side_prompt():
    # The side subagent's prompt: identical every turn (the ladder replays
    # the same six messages), shorter than main's entries.
    return _hybrid_tokens(300, 600) + _GEN_PROMPT


@pytest.fixture
def ladder(cpu):
    """One ladder step done, the next main turn admitted: main's boundary
    B and end-of-turn T, side-1's entries, and main-2 queued."""
    s = _hybrid_scheduler(_tiny_hybrid(), _tiny_tokenizer(False))
    history = _hybrid_tokens(1, 1500)
    m1 = history + _GEN_PROMPT
    _c, out = _hybrid_run(s, "main-1", m1, len(history), **MAIN)
    boundary = tuple(history)
    turn_end = tuple(m1 + out)
    side = _side_prompt()
    _hybrid_run(s, "side-1", side, len(side) - 3, session="side-1", frame=FRAME)
    entries = s.memory_aware_cache._entries
    assert boundary in entries and turn_end in entries
    side_keys = [k for k in entries if k[:20] == tuple(side[:20])]
    assert side_keys and all(len(k) < len(turn_end) for k in side_keys)
    # The fixed history's next main turn: the boundary, the HISTORY's reply
    # (not the generated one) and a tool result.
    m2 = list(boundary) + [904] + _hybrid_tokens(7, 200) + _GEN_PROMPT
    nxt = _request("main-2", m2, 4, **MAIN)
    nxt.prefix_boundary = len(m2) - 3
    s.add_request(nxt)
    return s, boundary, turn_end, side_keys, side, m2


def test_main_dead_branch_ranks_below_a_finished_subagents_head(ladder):
    s, _boundary, turn_end, side_keys, _side, _m2 = ladder
    with s.memory_aware_cache._lock:
        ranked = s._memguard_ranked_entries()
    levels = {k: level for k, level, _c in ranked}
    order = [k for k, _l, _c in ranked]
    assert levels[turn_end] == EVICT_UNUSED, levels[turn_end]
    for k in side_keys:
        assert levels[k] == EVICT_HEAD, levels[k]
        assert order.index(turn_end) < order.index(k)


def test_duress_at_main_admission_evicts_the_dead_branch_first(ladder):
    """The live k=57 decision, replayed: one eviction up to EVICT_HEAD takes
    main's dead branch, the side's boundary survives and its next request
    (a new session id, the same content) resumes from it; main's own
    boundary is untouched throughout."""
    s, boundary, turn_end, side_keys, side, m2 = ladder
    assert s._memguard_evict_ordered(EVICT_HEAD, reason="E1")
    entries = s.memory_aware_cache._entries
    assert turn_end not in entries
    assert all(k in entries for k in side_keys)
    assert boundary in entries
    # main-2 runs (resumes from its boundary), then side-2.
    from tests.test_memory_guard_session import _drive

    outs = _drive(s, [])
    assert outs["main-2"][-1].cached_tokens >= len(boundary), outs["main-2"][-1]
    cached, _o = _hybrid_run(
        s, "side-2", side, len(side) - 3, session="side-2", frame=FRAME
    )
    assert cached >= len(side) - 8, (cached, len(side))


def test_the_live_conversations_newest_heads_stay_needed(ladder):
    """Review 5 controls stay: main's boundary (what main-2 resumes from)
    is NEEDED, never ranked with the dead branch."""
    s, boundary, turn_end, _side_keys, _side, _m2 = ladder
    with s.memory_aware_cache._lock:
        levels = {k: lv for k, lv, _c in s._memguard_ranked_entries()}
    assert levels[boundary] == EVICT_NEEDED
    assert levels[turn_end] < levels[boundary]
