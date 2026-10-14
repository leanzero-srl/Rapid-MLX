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
