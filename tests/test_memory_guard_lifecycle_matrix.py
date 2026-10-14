# SPDX-License-Identifier: Apache-2.0
"""The conversation-line lifecycle, as ONE matrix: every goose traffic shape
through the memory guard on BOTH device arms (review 8).

Three review rounds in a row, a lifecycle rule fixed one goose shape and
broke a sibling (6: a chat the user left was held forever; 7: a subagent's
trailing user message abandoned a live main, a title helper won the
admission race and demoted main; 8: anything but a clean "stop" -- a user
Stop, a disconnect, "cancelled", "abort", a helper's "length" or tool call,
the /v1/messages and /v1/responses routes -- held the left chat again).
This matrix drives all of them through the same rule and checks:

(a) a LIVE main -- mid-turn (its last completion a tool call / length cut),
    waiting on a subagent or a helper -- keeps its newest head at
    EVICT_NEEDED, and its next request hits;
(b) a chat the user LEFT is evictable (below NEEDED) once a younger chat
    completed a turn; on the duress arm its entries go before any chunk of
    the younger chat is reduced;
(c) on the no-pressure arm the guard takes no protective action at all;
(d) no line stays NEEDED forever without positive evidence: after a younger
    chat completes a turn and starts another, only that chat is NEEDED.

Arms: Gabee's 64 GB at 1/100 with the real attention-scratch ratio (duress),
and the same at 64x the memory (no pressure). The engine is the real
scheduler + memory-aware prefix cache + mlx-lm on the CPU device (see
test_memory_guard_session)."""

from __future__ import annotations

import itertools

import pytest

from rapid_mlx.memory_guard import EVICT_NEEDED
from tests.test_memory_guard_session import (  # noqa: F401
    GABEE_PHYSICAL,
    _auto_budget_scheduler,
    _Device,
    _drive,
    _no_metal,
    _request,
    _tokens,
    cpu,
    llama,
)

ARMS = {
    "duress64": GABEE_PHYSICAL,
    "nopressure": GABEE_PHYSICAL * 64,
}
FRAME = 48  # the rendered system/tools frame of each conversation
MAIN = 5  # goose's main-agent frame (system prompt + tools)
SUBAGENT = 6  # a subagent's frame (its own system prompt)
OTHER = 9  # another agent's frame (an async delegate, another tool set)
HELPER_FRAME = 40


def _seq(seed: int, n: int) -> list[int]:
    return [(seed * 37 + i * 11) % 120 + 1 for i in range(n)]


class _Conv:
    def __init__(self, name, sid, frame_seed, first_seed, first_len, header):
        self.name = name
        self.sid = sid if header else None
        self.frame = _seq(frame_seed, FRAME)
        self.tokens = self.frame + _seq(first_seed, first_len)
        self.first_end = len(self.tokens)
        self.header = header
        self.owner = None
        self.requests = 0
        self.last_prompt = None


class World:
    def __init__(self, llama, monkeypatch, arm: str):
        self.arm = arm
        self.device = _Device(physical=ARMS[arm], scratch_scale=600.0)
        # Two rows: a helper call can run beside a conversation's request.
        self.s = _auto_budget_scheduler(llama, monkeypatch, self.device, max_num_seqs=2)
        self.device.install(monkeypatch, self.s)
        self.guard = self.s._memguard()
        self.convs: dict[str, _Conv] = {}
        self.helpers: dict[str, object] = {}
        self.ids = itertools.count()
        self.left: set[str] = set()
        self.violations: list[str] = []
        # Conversation requests that did not resume from their own previous
        # request ((name, cached, needed)); on the no-pressure arm there must
        # be none (c).
        self.misses: list[tuple[str, int, int]] = []
        self.helper_prompts: dict[str, list[int]] = {}
        real = self.s.step

        def step():
            before = self.guard.counts["chunks_reduced"]
            out = real()
            if self.guard.counts["chunks_reduced"] > before:
                held = [n for n in self.left if self._entries(self.convs[n].owner)]
                if held:
                    self.violations.append(f"chunk reduced while {held} held")
            return out

        self.s.step = step

    # -- traffic ----------------------------------------------------------
    def chat(self, name, sid=None, *, size=1500, header=True, seed=None, kind=MAIN):
        """A conversation. ``kind``: its rendered system/tools frame -- every
        goose chat window (and every sequential benchmark task) renders the
        same one; a subagent renders its own (another system prompt)."""
        seed = seed if seed is not None else 10 + len(self.convs) * 7
        conv = _Conv(name, sid or name, kind, seed + 1, size, header)
        self.convs[name] = conv
        return conv

    def _run(self, request, conv=None):
        rid = request.request_id
        outs = _drive(self.s, [request])[rid]
        assert not outs[-1].error, outs[-1].error
        return outs[-1].cached_tokens, _tokens(outs)

    def turn(
        self, name, *, add=400, outcome="stop", user=True, multi=None, build=False
    ):
        """One request of a conversation: ``user`` = it ends in a user
        message; ``outcome`` = the finish_reason the route reports
        ("none" = the holder stayed empty: a disconnect, a user Stop, an
        error; "noholder" = a route that reports nothing)."""
        conv = self.convs[name]
        prompt = conv.tokens + _seq(100 + next(self.ids), add)
        multi = conv.requests > 0 if multi is None else multi
        holder = {"noholder": None, "none": [None]}.get(outcome, [outcome])
        kw = {"first_end": conv.first_end, "frame": FRAME}
        if conv.sid is not None:
            kw["session"] = conv.sid
        request = _request(
            f"{name}-{next(self.ids)}",
            prompt,
            4,
            turn_start=user,
            multi=multi,
            outcome=holder,
            **kw,
        )
        request.prefix_boundary = len(prompt)
        if build:
            return request, prompt
        before = len(conv.tokens) if conv.requests else 0
        cached, out = self._run(request)
        if before and not _hit(cached, len(prompt), before):
            self.misses.append((name, cached, before))
        conv.owner = request._memguard_line_key
        conv.requests += 1
        conv.last_prompt = prompt
        conv.tokens = prompt + out
        return cached, len(prompt)

    def helper(
        self,
        kind,
        of,
        *,
        outcome="stop",
        multi=False,
        size=300,
        repeat=False,
        build=False,
    ):
        """A helper call under ``of``'s session id: its own system prompt
        (frame), one user message (title, summary) or several (Recipe, MCP
        sampling). ``repeat``: exactly the previous call of this kind (a
        retitle with the same content)."""
        conv = self.convs[of]
        if repeat:
            prompt = list(self.helper_prompts[kind])
        else:
            frame = _seq(500 + len(self.helpers), HELPER_FRAME)
            body = _seq(600 + next(self.ids), size)
            if multi:
                body = body + [119] + _seq(700 + next(self.ids), 120)
            prompt = frame + body
        self.helper_prompts[kind] = prompt
        holder = {"noholder": None, "none": [None]}.get(outcome, [outcome])
        kw = {"frame": HELPER_FRAME, "first_end": HELPER_FRAME + size}
        if conv.sid is not None:
            kw["session"] = conv.sid
        request = _request(
            f"{kind}-{next(self.ids)}",
            prompt,
            4,
            turn_start=True,
            multi=multi,
            outcome=holder,
            **kw,
        )
        request.prefix_boundary = len(prompt)
        if build:
            return request
        self._run(request)
        self.helpers[kind] = request._memguard_line_key

    def concurrent(self, first, then, *, after_steps=1):
        """Run ``first`` and, ``after_steps`` steps later while it still
        runs, ``then`` beside it (built requests from turn/helper(build=True))."""
        outs = _drive(self.s, [first], arrive={after_steps: [then]})
        for r in (first, then):
            assert not outs[r.request_id][-1].error, outs[r.request_id][-1].error
        return outs

    # -- observations -----------------------------------------------------
    def _entries(self, owner):
        mac = self.s.memory_aware_cache
        return [
            k for k, e in mac._entries.items() if getattr(e, "owner", None) == owner
        ]

    def level(self, owner) -> int:
        with self.s.memory_aware_cache._lock:
            ranked = self.s._memguard_ranked_entries()
        mac = self.s.memory_aware_cache
        levels = [
            lv
            for k, lv, _c in ranked
            if getattr(mac._entries[k], "owner", None) == owner
        ]
        return max(levels) if levels else 0

    def needed(self, name) -> bool:
        return self.level(self.convs[name].owner) == EVICT_NEEDED

    def protected(self, name, why=""):
        """(a): the newest head is NEEDED (or already in no entry is not
        acceptable either: a live main must keep its head)."""
        owner = self.convs[name].owner
        assert self._entries(owner), (name, why, "lost every entry")
        assert self.needed(name), (name, why, self.level(owner))

    def helper_level(self, kind) -> int:
        return self.level(self.helpers[kind])

    def evictable(self, name, why=""):
        """(b): below NEEDED from here on; on the duress arm its entries go
        before any chunk reduction."""
        owner = self.convs[name].owner
        assert self.level(owner) < EVICT_NEEDED, (name, why, self.level(owner))
        self.left.add(name)

    def flush(self, header=True):
        """(d): a younger chat completes a turn and starts the next one; only
        it may then hold NEEDED heads -- and lines with POSITIVE evidence
        (their last completion a tool call or a length cut) of a kind the
        flush chat is not (with no session id, a helper of another system
        prompt that ended on "length" is its own line; the client owes it a
        continuation)."""
        self.chat("zflush", header=header, size=300, seed=97)
        self.turn("zflush", add=100, outcome="stop")
        self.turn("zflush", add=100, outcome="stop")
        records = self.s.__dict__.get("_memguard_lines", {})
        flush_record = records.get(self.convs["zflush"].owner)
        flush_kind = (
            flush_record.get("kind") if isinstance(flush_record, dict) else None
        )
        owners = {c.owner for c in self.convs.values() if c.name != "zflush"}
        owners |= set(self.helpers.values())
        for owner in owners:
            if owner is None or not self._entries(owner):
                continue
            record = records.get(owner)
            record = record if isinstance(record, dict) else {}
            holder = getattr(record.get("hint"), "outcome", None) or [None]
            if (
                holder[0] in ("tool_calls", "length")
                and record.get("kind") != flush_kind
            ):
                continue  # positive evidence, another kind
            assert self.level(owner) < EVICT_NEEDED, ("pinned", owner)

    # Counters that are bookkeeping, not protective actions.
    BOOKKEEPING = frozenset({"checks", "calibrations"})
    # Dead-entry cleanup: an entry the conversation itself has moved past
    # (superseded by its own newer entry, or a branch it diverged from). No
    # request is denied a hit by it -- asserted below through ``misses``.
    CLEANUP = frozenset({"superseded", "abandoned"})

    def finish(self):
        assert not self.violations, self.violations
        counts = {k: v for k, v in self.guard.counts.items() if isinstance(v, int)}
        actions = {
            k: v
            for k, v in counts.items()
            if v and k not in self.BOOKKEEPING | self.CLEANUP
        }
        if self.arm == "nopressure":
            # (c) zero protective actions, and the cleanup cost no hit.
            assert not actions, actions
            assert not self.misses, self.misses
        else:
            # The duress arm must actually be under duress, or its
            # invariants prove nothing.
            assert actions, ("no duress reached", counts)
        assert self.device.ooms == 0


# -- scenarios: (name, header, function) ------------------------------------

END_STATES = ["stop", "none", "cancelled", "abort", "error", "noholder"]
MID_STATES = ["tool_calls", "length"]


def _hit(cached, length, conv_tokens_before):
    # the request re-prefills at most its new tokens (+ the output tail)
    return cached >= conv_tokens_before - 8


def sc_main_waits_on_subagent(w, header, mid):
    """main's turn ends in a tool call (the subagent) or a length cut; the
    subagent (its own session) runs a task whose requests END IN USER
    MESSAGES mid-task (a tool image, a FINAL_OUTPUT / LENGTH_CUT
    continuation, a steer) and finishes; main resumes and hits."""
    w.chat("main", size=2500, header=header)
    w.turn("main", outcome="tool_calls")
    w.turn("main", user=False, outcome=mid)
    before = len(w.convs["main"].tokens)
    w.chat("sub", size=800, header=header, kind=SUBAGENT)
    for j, out in enumerate(["tool_calls", "tool_calls", "stop", "tool_calls", "stop"]):
        w.turn("sub", user=j != 1, outcome=out)
        w.protected("main", f"subagent step {j}")
    cached, length = w.turn("main", user=False, outcome="stop")
    assert _hit(cached, length, before), (cached, before)
    # main took the result and went on: the finished subagent is released
    w.evictable("sub", "finished subagent")
    w.flush(header)


def sc_main_waits_on_helpers(w, header, mid):
    """goose helpers under main's session id while main is mid-turn: the
    title (spawned before the reply loop, so admitted FIRST), PermissionCheck
    (answers with a tool call), FactCheck / MemoryReview (may end on
    "length"), Recipe and MCP sampling (several messages)."""
    w.chat("main", size=2500, header=header)
    w.helper("title", "main")  # wins the admission race
    w.turn("main", outcome=mid)
    w.helper("permission", "main", outcome="tool_calls")
    w.protected("main", "permission check")
    w.helper("factcheck", "main", outcome="length")
    w.protected("main", "factcheck")
    w.helper("recipe", "main", multi=True)
    w.protected("main", "recipe")
    w.helper("sampling", "main", multi=True, outcome="none")
    w.protected("main", "sampling")
    before = len(w.convs["main"].tokens)
    cached, length = w.turn("main", user=False, outcome="stop")
    assert _hit(cached, length, before), (cached, before)
    # main continued after its helpers: they are released (with no session
    # id each helper is its own line: those that ended idle; one ending on
    # a tool call / length cut keeps its positive evidence)
    idle = ("title", "recipe", "sampling")
    for kind in idle if not header else (*idle, "permission", "factcheck"):
        assert w.helper_level(kind) < EVICT_NEEDED, (kind, w.helper_level(kind))
    w.helper("title", "main", repeat=True)  # TurnEnd retitle, same content
    w.protected("main", "retitle")
    w.flush(header)


def sc_first_turn_then_multi_helper(w, header, end):
    """A new chat's first turn carries ONE message; a multi-message helper of
    the same session follows (review 8, item 3): main is not demoted by it."""
    w.chat("main", size=5600, header=header)
    w.turn("main", outcome=end)
    w.helper("sampling", "main", multi=True, size=3000)
    w.protected("main", "multi helper after a one-message first turn")
    w.flush(header)


def sc_user_leaves_chat(w, header, end):
    """chat A's turn ended -- a reply, a user Stop / disconnect / error (the
    holder stays empty), "cancelled", a guard refusal ("abort"), or a route
    that reports nothing -- with an end-of-turn helper that ended non-stop;
    the user opens chat B. A becomes evictable at B's first completed turn
    and stays so while B grows."""
    w.chat("A", size=5600, header=header)
    w.turn("A", outcome="tool_calls")
    w.turn("A", user=False, outcome=end)
    w.helper("factcheck", "A", outcome="length")
    w.chat("B", size=600, header=header)
    w.turn("B", add=200, outcome="stop")
    w.evictable("A", f"A ended with {end}")
    for _t in range(8):
        w.turn("B", add=500, outcome="stop")
        w.protected("B", "B grows")
    w.flush(header)


def sc_chat_abandoned_mid_tool_call(w, header, arg):
    """A ends mid-turn (a tool call or a length cut) and the user never comes
    back: a chat of the same kind starts its turns -- whether or not its
    route reports how they ended -- and A is released."""
    mid, b_out = arg
    w.chat("A", size=5600, header=header)
    w.turn("A", outcome=mid)
    w.chat("B", size=600, header=header)
    w.turn("B", add=200, outcome=b_out)
    w.evictable("A", "abandoned mid tool call")
    for _t in range(8):
        w.turn("B", add=500, outcome=b_out)
    w.flush(header)


def sc_one_request_chat_then_helper(w, header, helper_out):
    """Review 8's shape: chat A is ONE request (a plain reply), then an
    end-of-turn helper of A's session ends on "length" or a tool call
    (FactCheck, PermissionCheck) or is still unreported; the user opens chat
    B, whose route reports nothing. A is released at B's first turn."""
    w.chat("A", size=5600, header=header)
    w.turn("A", outcome="stop")
    w.helper("factcheck", "A", outcome=helper_out)
    w.chat("B", size=600, header=header)
    w.turn("B", add=200, outcome="noholder")
    w.evictable("A", f"helper {helper_out}")
    for _t in range(8):
        w.turn("B", add=500, outcome="noholder")
    w.flush(header)


def sc_mid_tool_call_beside_another_kind(w, header, mid):
    """The limit of what a request can tell: A is mid-turn (a tool call
    pending) and a session of ANOTHER kind (another system prompt: a
    subagent, a different agent) completes turns and starts new ones --
    including a FINAL_OUTPUT / goal / grind nudge after a reply, which looks
    exactly like a user's next turn. That is main waiting on its subagent,
    so A keeps its head (positive evidence: the pending tool call); a chat of
    A's own kind completing a turn releases it (the flush)."""
    w.chat("A", size=5600, header=header)
    w.turn("A", outcome=mid)
    w.chat("S", size=1500, header=header, kind=SUBAGENT)
    for _t in range(5):
        w.turn("S", add=600, outcome="stop")
        w.protected("A", "another kind resumed after a reply")
    w.flush(header)


def sc_concurrent_same_kind_workers(w, header, mid):
    """Two conversations of the same kind running side by side (swarm
    workers, two goose windows in use): W1 is mid-turn (its tool is running)
    while W2, which began before W1's latest request, takes user turns. W1
    keeps its head and hits when its tool result arrives."""
    w.chat("W1", size=1500, header=header)
    w.chat("W2", size=1500, header=header)
    w.turn("W1", outcome="tool_calls")
    w.turn("W2", outcome="tool_calls")
    w.turn("W1", user=False, outcome=mid)
    before = len(w.convs["W1"].tokens)
    for _t in range(3):
        w.turn("W2", add=300, outcome="stop")
        w.protected("W1", "a concurrent worker of the same kind")
    cached, length = w.turn("W1", user=False, outcome="stop")
    assert _hit(cached, length, before), (cached, before)
    w.flush(header)


def sc_younger_waits_on_its_tool(w, header, kind):
    """Review 9: two conversations side by side (swarm workers, goose async
    delegates, two windows); the YOUNGER one waits on its own long tool
    while the older one keeps going with tool rounds. The waiter keeps its
    head and hits when its result arrives (of the same kind or not)."""
    w.chat("W1", size=1500, header=header)
    w.chat("W2", size=2500, header=header, kind=MAIN if kind == "same" else OTHER)
    w.turn("W1", outcome="tool_calls")
    w.turn("W1", user=False, outcome="tool_calls")
    w.turn("W2", outcome="tool_calls")
    w.turn("W2", user=False, outcome="tool_calls")  # W2 waits on a long tool
    before = len(w.convs["W2"].tokens)
    for _t in range(6):
        w.turn("W1", user=False, add=600, outcome="tool_calls")
        w.protected("W2", "an older conversation's tool rounds")
    cached, length = w.turn("W2", user=False, outcome="stop")
    assert _hit(cached, length, before), (cached, before)
    w.flush(header)


def sc_left_for_another_kind(w, header, end):
    """The user leaves chat A (its end-of-turn FactCheck ended on "length")
    for a conversation of ANOTHER kind (a benchmark task with another system
    prompt): A is released once that conversation continues -- A's helper's
    outcome is not A's."""
    w.chat("A", size=5600, header=header)
    w.turn("A", outcome="tool_calls")
    w.turn("A", user=False, outcome=end)
    w.helper("factcheck", "A", outcome="length")
    w.chat("B", size=600, header=header, kind=OTHER)
    w.turn("B", add=200, outcome="stop")
    w.turn("B", add=200, outcome="stop")
    w.evictable("A", "left for another kind")
    for _t in range(8):
        w.turn("B", add=500, outcome="stop")
    w.flush(header)


def sc_recent_chat_beside_a_build(w, header, end):
    """Two windows: B (younger) started a build earlier and runs tool rounds;
    the user's LATEST turn was in A, which replied. B's tool rounds are no
    user turn: A stays the user's most recent chat and keeps its head."""
    w.chat("A", size=3000, header=header)
    w.turn("A", outcome="stop")
    w.chat("B", size=3000, header=header)
    w.turn("B", outcome="tool_calls")
    w.turn("A", add=300, outcome=end)
    before = len(w.convs["A"].tokens)
    for _t in range(6):
        w.turn("B", user=False, add=500, outcome="tool_calls")
        w.protected("A", "a build's tool rounds")
    cached, length = w.turn("A", add=300, outcome="stop")
    assert _hit(cached, length, before), (cached, before)
    w.flush(header)


def sc_helper_beside_a_younger_turn(w, header, end):
    """A replied; its end-of-turn helper (a retitle, same session) is RUNNING
    when a younger chat's user turn arrives beside it. While its helper
    runs, A is live: its head stays out of the younger chat's way, and A's
    next turn hits."""
    w.chat("A", size=5600, header=header)
    w.turn("A", outcome="tool_calls")
    w.turn("A", user=False, outcome=end)
    before = len(w.convs["A"].tokens)
    w.chat("B", size=1500, header=header)
    # a long helper call: the younger chat's whole prefill runs beside it
    title = w.helper("title", "A", size=4000, build=True)
    b_request, b_prompt = w.turn("B", add=200, outcome="stop", build=True)
    w.concurrent(title, b_request)
    conv = w.convs["B"]
    conv.owner = b_request._memguard_line_key
    conv.requests += 1
    conv.tokens = b_prompt + [0]  # its output is not reused below
    w.helpers["title"] = title._memguard_line_key
    cached, length = w.turn("A", add=300, outcome="stop")
    assert _hit(cached, length, before), (cached, before)
    w.flush(header)


def sc_two_chats_alternating(w, header, end):
    """The user alternates between two chats: each one being used is
    protected and hits; the other is evictable under duress only."""
    w.chat("A", size=3500, header=header)
    w.chat("B", size=3500, header=header)
    for t in range(3):
        for name in ("A", "B"):
            before = len(w.convs[name].tokens)
            cached, length = w.turn(name, add=600, outcome=end)
            w.protected(name, f"alternating {t}")
            if t and w.arm == "nopressure":
                assert _hit(cached, length, before), (name, t, cached, before)
    w.flush(header)


def sc_twins(w, header, end):
    """Two clients with NO session id that share the system prompt, tools and
    first message (best-of-N, parallel agents): one line. No pressure: both
    hit every turn."""
    w.chat("T1", size=3500, header=False, seed=33)
    w.chat("T2", size=3500, header=False, seed=33)
    for t in range(4):
        for name in ("T1", "T2"):
            before = len(w.convs[name].tokens)
            cached, length = w.turn(name, add=600, outcome=end)
            if t and w.arm == "nopressure":
                assert _hit(cached, length, before), (name, t, cached, before)
    w.flush(False)


SCENARIOS = []
for header in (True, False):
    tag = "hdr" if header else "nohdr"
    for mid in MID_STATES:
        SCENARIOS.append(
            (f"subagent-{mid}-{tag}", sc_main_waits_on_subagent, header, mid)
        )
        SCENARIOS.append(
            (f"helpers-{mid}-{tag}", sc_main_waits_on_helpers, header, mid)
        )
        for b_out in ("stop", "noholder"):
            SCENARIOS.append(
                (
                    f"abandoned-{mid}-b{b_out}-{tag}",
                    sc_chat_abandoned_mid_tool_call,
                    header,
                    (mid, b_out),
                )
            )
        SCENARIOS.append(
            (
                f"otherkind-{mid}-{tag}",
                sc_mid_tool_call_beside_another_kind,
                header,
                mid,
            )
        )
    for end in END_STATES:
        SCENARIOS.append((f"left-{end}-{tag}", sc_user_leaves_chat, header, end))
    for end in ("stop", "tool_calls"):
        SCENARIOS.append(
            (f"first-turn-{end}-{tag}", sc_first_turn_then_multi_helper, header, end)
        )
    for helper_out in ("length", "tool_calls", "none"):
        SCENARIOS.append(
            (
                f"onerequest-helper{helper_out}-{tag}",
                sc_one_request_chat_then_helper,
                header,
                helper_out,
            )
        )
    for mid in MID_STATES:
        SCENARIOS.append(
            (f"concurrent-{mid}-{tag}", sc_concurrent_same_kind_workers, header, mid)
        )
    for end in ("stop", "none"):
        SCENARIOS.append(
            (f"leftother-{end}-{tag}", sc_left_for_another_kind, header, end)
        )
        SCENARIOS.append(
            (f"recentbuild-{end}-{tag}", sc_recent_chat_beside_a_build, header, end)
        )
        if header:  # with no session id a helper is its own line, not A's
            SCENARIOS.append(
                (
                    f"helperbeside-{end}-{tag}",
                    sc_helper_beside_a_younger_turn,
                    header,
                    end,
                )
            )
    for kind in ("same", "other"):
        SCENARIOS.append(
            (f"youngerwaits-{kind}-{tag}", sc_younger_waits_on_its_tool, header, kind)
        )
    SCENARIOS.append((f"alternating-{tag}", sc_two_chats_alternating, header, "stop"))
SCENARIOS.append(("twins", sc_twins, False, "stop"))


@pytest.mark.parametrize("arm", list(ARMS))
@pytest.mark.parametrize(
    ("scenario", "header", "arg"),
    [pytest.param(fn, h, a, id=name) for name, fn, h, a in SCENARIOS],
)
def test_lifecycle(llama, monkeypatch, arm, scenario, header, arg):
    w = World(llama, monkeypatch, arm)
    scenario(w, header, arg)
    w.finish()
