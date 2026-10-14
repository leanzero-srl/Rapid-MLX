"""Review 6 (an independent review of b06acd8ea, FIX-FIRST): conversation
lines are tracked per (session, frame), a line is abandoned when a younger
conversation starts a new user turn after its last activity, the hybrid
count bound evicts by recency within a level, and the store drops a
conversation's dead branches only when the client named the conversation.
Every case is RED on b06acd8ea; the no-pressure arms show zero loss."""

import pytest

from rapid_mlx.memory_guard import EVICT_HEAD, EVICT_NEEDED
from tests.test_memory_guard_session import (  # noqa: F401
    _GEN_PROMPT,
    _MAIN,
    _hybrid_run,
    _hybrid_scheduler,
    _hybrid_tokens,
    _hybrid_tool_loop,
    _no_metal,
    _strip_loop,
    _tiny_hybrid,
    _tiny_tokenizer,
    cpu,
)


def _levels(s) -> dict:
    with s.memory_aware_cache._lock:
        return {k: lv for k, lv, _c in s._memguard_ranked_entries()}


def test_a_helper_turn_keeps_mains_turn_boundary(cpu):
    """Item 4: on a template that rewrites the tool loop, main keeps the
    boundary its turn started at. A helper call under main's session id
    (title, MemoryReview, FactCheck: its own frame, ends in a user message)
    replaced it on b06acd8ea (keyed by the session only): main's boundary
    fell to UNUSED and duress took it -- the next user message cached 512
    of 1,500."""
    s = _hybrid_scheduler(_tiny_hybrid(), _tiny_tokenizer(True))
    first = _hybrid_tokens(1, 1500)
    prompt, loop_start, _last = _hybrid_tool_loop(s, first, "a")
    h = _hybrid_tokens(700, 200) + _hybrid_tokens(701, 600)
    _hybrid_run(
        s, "helper", h + _GEN_PROMPT, len(h), turn_start=True,
        session="main", frame=200,
    )
    assert _levels(s)[tuple(first)] == EVICT_NEEDED
    # Duress: everything below NEEDED goes.
    while s._memguard_evict_ordered(EVICT_HEAD, reason="E1"):
        pass
    new_user = _strip_loop(prompt, first, loop_start, 77)
    cached, _o = _hybrid_run(
        s, "new-user", new_user, len(new_user) - 3, turn_start=True, **_MAIN
    )
    assert cached >= len(first), cached


def test_a_helper_frame_is_outlived_once_main_runs_again(cpu):
    """Item 5: goose's summary (compaction) helper runs under MAIN's session
    id with its own frame and the whole conversation; main then continues
    from the compacted history. On b06acd8ea lines were tracked per session,
    so the helper's 3k head stayed NEEDED for the whole session. Main's own
    newest head stays NEEDED (the control)."""
    s = _hybrid_scheduler(_tiny_hybrid(), _tiny_tokenizer(False))
    frame = _hybrid_tokens(1, 300)
    main = frame + _hybrid_tokens(2, 1500)
    _c, out = _hybrid_run(s, "m0", main + _GEN_PROMPT, len(main), turn_start=True, **_MAIN)
    summary = _hybrid_tokens(7, 200) + main[300:] + out + _hybrid_tokens(8, 50)
    _hybrid_run(
        s, "summary", summary + _GEN_PROMPT, len(summary), turn_start=True,
        session="main", frame=200,
    )
    compacted = frame + _hybrid_tokens(9, 400)
    for t in range(2):
        _c, out = _hybrid_run(
            s, f"m{t + 1}", compacted + _GEN_PROMPT, len(compacted),
            turn_start=True, **_MAIN,
        )
        compacted = compacted + _GEN_PROMPT + out + _hybrid_tokens(20 + t, 100)
    levels = _levels(s)
    helper = [lv for k, lv in levels.items() if k[:200] == tuple(summary[:200])]
    mains = [lv for k, lv in levels.items() if k[:300] == tuple(frame) and len(k) > 700]
    assert helper and max(helper) < EVICT_NEEDED, helper
    assert EVICT_NEEDED in mains, mains
