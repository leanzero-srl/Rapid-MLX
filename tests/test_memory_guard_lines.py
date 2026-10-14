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
        s,
        "helper",
        h + _GEN_PROMPT,
        len(h),
        turn_start=True,
        session="main",
        frame=200,
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
    # main's first request carries one message; goose's summary helper is
    # a system prompt (the transcript) and ONE user message.
    _c, out = _hybrid_run(
        s, "m0", main + _GEN_PROMPT, len(main), turn_start=True, multi=False, **_MAIN
    )
    summary = _hybrid_tokens(7, 200) + main[300:] + out + _hybrid_tokens(8, 50)
    _hybrid_run(
        s,
        "summary",
        summary + _GEN_PROMPT,
        len(summary),
        turn_start=True,
        session="main",
        frame=200,
        multi=False,
    )
    compacted = frame + _hybrid_tokens(9, 400)
    for t in range(2):
        _c, out = _hybrid_run(
            s,
            f"m{t + 1}",
            compacted + _GEN_PROMPT,
            len(compacted),
            turn_start=True,
            **_MAIN,
        )
        compacted = compacted + _GEN_PROMPT + out + _hybrid_tokens(20 + t, 100)
    levels = _levels(s)
    helper = [lv for k, lv in levels.items() if k[:200] == tuple(summary[:200])]
    mains = [lv for k, lv in levels.items() if k[:300] == tuple(frame) and len(k) > 700]
    assert helper and max(helper) < EVICT_NEEDED, helper
    assert EVICT_NEEDED in mains, mains


# -- item 1: chat A finished, chat B opened (64 GB duress) ----------------

from tests.test_memory_guard_session import (  # noqa: E402,F401
    GABEE_PHYSICAL,
    _auto_budget_scheduler,
    _Device,
    _drive,
    _new_tokens,
    _request,
    _tokens,
    llama,
)


@pytest.mark.parametrize("mode", ["header", "noheader"])
def test_a_finished_chat_goes_before_the_new_chats_chunk_shrinks(
    llama, monkeypatch, mode
):
    """Item 1 (HIGH): the user finishes chat A and opens chat B (same system
    prompt and tools, its own session -- or, with no header, its own first
    message) and never returns. On b06acd8ea A's head stayed NEEDED forever:
    B's chunk was reduced 35 times while A was held (65 reductions vs 37).
    Once B starts its second user turn, A is abandoned: under duress it goes
    before any chunk reduction."""
    device = _Device(physical=GABEE_PHYSICAL, scratch_scale=600.0)
    scheduler = _auto_budget_scheduler(llama, monkeypatch, device)
    device.install(monkeypatch, scheduler)
    if mode == "header":
        a_kw, b_kw = (
            {"session": "chatA", "frame": 320},
            {"session": "chatB", "frame": 320},
        )
    else:
        a_kw, b_kw = {"first_end": 330}, {"first_end": 330}
    old = _new_tokens(0, 320) + [3 + (i * 5) % 97 for i in range(5680)]
    # Chat A's turn ended with a reply (the chat route reports "stop").
    _drive(
        scheduler,
        [_request("old", old, 4, turn_start=True, outcome=["stop"], **a_kw)],
    )
    mac = scheduler.memory_aware_cache
    guard = scheduler._memguard()
    held_while_reduced = 0
    real_step = scheduler.step

    def step():
        nonlocal held_while_reduced
        before = guard.counts["chunks_reduced"]
        out = real_step()
        if guard.counts["chunks_reduced"] > before and any(
            k[: len(old)] == tuple(old) for k in mac._entries
        ):
            held_while_reduced += 1
        return out

    scheduler.step = step
    prompt = _new_tokens(0, 600)
    turns = []
    for turn in range(10):
        rid = f"new-{turn}"
        outs = _drive(scheduler, [_request(rid, prompt, 4, turn_start=True, **b_kw)])[
            rid
        ]
        assert not outs[-1].error
        turns.append(outs[-1].cached_tokens)
        prompt = prompt + _tokens(outs) + _new_tokens(turn + 1, 496)
    # B's first turn may still yield to A (A could be waiting on a subagent);
    # from B's second user turn on, A goes first.
    assert held_while_reduced <= 1, held_while_reduced
    assert all(c > 0 for c in turns[1:]), turns


@pytest.mark.parametrize("main_outcome", [["tool_calls"], ["length"], None])
def test_main_waiting_on_a_subagent_stays_protected(cpu, main_outcome):
    """Review 7, item 1 (HIGH): main is mid-turn (its last completion is a
    tool call -- the subagent --, a length cut, or not reported) while a
    YOUNGER session runs requests that end in a user message: a subagent's
    task, then mid-task trailing user messages (goose: a tool image becomes
    a trailing user message; FINAL_OUTPUT / LENGTH_CUT continuations, goal
    nudges, steers). On 5ba8a8870 the second such request abandoned main
    and an eviction to EVICT_HEAD took every main entry. Main's head stays
    NEEDED, and duress up to EVICT_HEAD leaves it."""
    s = _hybrid_scheduler(_tiny_hybrid(), _tiny_tokenizer(False))
    first = _hybrid_tokens(1, 1500)
    prompt, _ls, _last = _hybrid_tool_loop(s, first, "main")
    _c, out = _hybrid_run(
        s, "main-final", prompt, len(prompt) - 3, outcome=main_outcome, **_MAIN
    )
    head = tuple(prompt + out)
    sub = {"session": "sub", "frame": 300}
    sp = _hybrid_tokens(300, 600) + _GEN_PROMPT
    b = len(sp) - 3
    for j in range(5):
        # every request but the tool rounds ends in a user message
        _c, o = _hybrid_run(s, f"sub{j}", sp, b, turn_start=j != 1, **sub)
        nxt = sp + o + [904] + _hybrid_tokens(400 + j, 300)
        b = len(nxt)
        sp = nxt + _GEN_PROMPT
        assert _levels(s)[head] == EVICT_NEEDED, j
    assert ("session", "main") not in {o[0] for o in s._memguard_outlived_lines()}
    while s._memguard_evict_ordered(EVICT_HEAD, reason="E1"):
        pass
    assert head in s.memory_aware_cache._entries


# -- item 2: the hybrid entry-count bound, NO memory pressure -------------


@pytest.mark.parametrize("cap", [8, 24])
@pytest.mark.parametrize("header", [True, False])
def test_the_count_bound_keeps_the_live_chat_after_finished_chats(cpu, cap, header):
    """Item 2 (HIGH, the owner's hard rule): the hybrid entry-count bound is
    not memory pressure (the cap here is 10**12). After cap/2+1 finished
    chats their NEEDED heads filled the bound and, shortest first, the LIVE
    chat's entries went: its next turn cached 0 of 811 on b06acd8ea (plain
    LRU on 45e157bab: 507). Within a level the bound now evicts LRU."""
    s = _hybrid_scheduler(_tiny_hybrid(), _tiny_tokenizer(False), entries=cap)
    frame = _hybrid_tokens(1, 300)
    for c in range(cap // 2 + 1):
        kw = {"session": f"old{c}", "frame": 300} if header else {"first_end": 2800}
        user = frame + _hybrid_tokens(50 + c, 2500)
        prompt, _s, _l = _hybrid_tool_loop(s, user, f"old{c}", **kw)
        _hybrid_run(s, f"old{c}-final", prompt, len(prompt) - 3, **kw)
    kw = {"session": "live", "frame": 300} if header else {"first_end": 500}
    user = frame + _hybrid_tokens(900, 200)
    prompt = user + _GEN_PROMPT
    _c, out = _hybrid_run(s, "live-0", prompt, len(user), turn_start=True, **kw)
    line = prompt + out
    nxt = line + [904] + _hybrid_tokens(901, 300) + _GEN_PROMPT
    cached, _o = _hybrid_run(s, "live-1", nxt, len(nxt) - 3, **kw)
    assert cached >= len(line) - 8, (cached, len(line))


# -- item 3: no-header twins, a big Mac with no pressure ------------------

from tests.test_memory_guard_session import M4_MAX_128_CEILING  # noqa: E402


@pytest.mark.parametrize("hint", ["frame", "none", "sessions"])
def test_twin_clients_without_a_session_id_keep_their_heads(llama, monkeypatch, hint):
    """Item 3 (MED): two clients with no session id that share the system
    prompt, tools and first message (best-of-N, parallel agents) are one
    line. On b06acd8ea each store dropped the other's head as "its
    conversation left behind" -- no pressure at all on 128 GB: both cached
    900 of 1,404-2,016 tokens every turn (9 abandoned). The store drops
    dead branches only of a conversation the client named. "none" (raw)
    and "sessions" (two session ids) are the controls."""
    device = _Device(physical=int(M4_MAX_128_CEILING / 0.97) + 1)
    s = _auto_budget_scheduler(llama, monkeypatch, device)
    device.install(monkeypatch, s)
    first = _new_tokens(0, 900)
    paths = {"A": first + _new_tokens(1, 300), "B": first + _new_tokens(2, 300)}
    rows = []
    for t in range(5):
        for name in ("A", "B"):
            kw = {
                "frame": {"first_end": 900},
                "none": {},
                "sessions": {"session": name, "frame": 900},
            }[hint]
            rid = f"{name}{t}"
            outs = _drive(s, [_request(rid, paths[name], 4, **kw)])[rid]
            rows.append((name, t, len(paths[name]), outs[-1].cached_tokens))
            paths[name] = (
                paths[name]
                + _tokens(outs)
                + _new_tokens(10 * t + (1 if name == "A" else 2), 200)
            )
    misses = [r for r in rows if r[1] > 0 and r[2] - r[3] > 210]
    assert not misses, misses
    guard = s._memguard()
    assert guard.counts["abandoned"] == 0
    assert guard.counts["evictions"] == 0 and guard.counts["chunks_reduced"] == 0
