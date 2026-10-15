# SPDX-License-Identifier: Apache-2.0
"""Review 13 end to end (the reviewer's repros, through the real scheduler on
the duress arm): older window O and main are the same kind; main's subagent
ends on a tool call and main takes the result; the user types in O, opens Z;
under duress main's and O's entries go; the user resumes O. A tombstone
pruning that kept only the witness that DECIDED a release dropped main's
(rule 1 had released the subagent through O), and when O came back -- as a
new, younger conversation -- the finished subagent stayed NEEDED through
three more windows: 165 "chunk reduced while ['sub'] held"."""

import pytest

from rapid_mlx.memory_guard import EVICT_NEEDED
from tests.test_memory_guard_lifecycle_matrix import (  # noqa: F401
    SUBAGENT,
    World,
    _no_metal,
    cpu,
    llama,
)


def _setup(w, ssize):
    w.chat("O", size=500)
    w.turn("O", add=100)
    w.turn("O", add=100)
    w.chat("main", size=700)
    w.turn("main", add=100, outcome="tool_calls")
    w.chat("sub", size=ssize, kind=SUBAGENT)
    w.turn("sub", add=100, outcome="tool_calls")
    w.turn("sub", user=False, add=300, outcome="tool_calls")
    w.turn("sub", user=False, add=300, outcome="tool_calls")  # its last
    w.turn("main", user=False, add=100, outcome="stop")
    w.turn("O", add=100)  # back to O
    w.chat("Z", size=400, seed=77)
    w.turn("Z", add=100)
    w.turn("Z", add=100)


def _ent(w, name):
    return len(w._entries(w.convs[name].owner))


def test_pin(llama, monkeypatch):
    w = World(llama, monkeypatch, "duress64")
    _setup(w, 3000)
    for name in ("O", "main", "sub"):
        w.evictable(name, "left")
    for _i in range(60):
        w.turn("Z", add=60)
        if _ent(w, "main") == 0 and _ent(w, "O") == 0:
            break
    assert _ent(w, "main") == 0 and _ent(w, "O") == 0
    w.left.discard("O")
    w.turn("O", add=100)  # the user resumes O
    w.turn("O", add=100)
    w.evictable("sub", "O resumed")
    for k in range(3):  # three more new windows, three turns each
        w.chat(f"N{k}", size=300, seed=200 + k)
        for _t in range(3):
            w.turn(f"N{k}", add=400)
        w.evictable("sub", f"window N{k}")
    for _t in range(25):
        w.turn("N2", add=300)
    assert not w.violations, w.violations[:2]
    w.flush()
    w.finish()


@pytest.mark.parametrize("ssize", [3000, 6000])
@pytest.mark.parametrize("zadd", [60, 120, 250])
def test_e2e(llama, monkeypatch, ssize, zadd):
    w = World(llama, monkeypatch, "duress64")
    _setup(w, ssize)
    for _i in range(60):
        w.turn("Z", add=zadd)
        if _ent(w, "main") == 0 and _ent(w, "O") == 0:
            break
    w.turn("O", add=100)  # the user resumes O
    w.turn("O", add=100)
    for _i in range(3):
        w.turn("Z", add=200)
    assert not (_ent(w, "sub") and w.level(w.convs["sub"].owner) == EVICT_NEEDED), (
        "dead sub pinned NEEDED"
    )
