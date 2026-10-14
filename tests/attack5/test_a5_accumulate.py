"""Review 5, attacks 3/4: the entries a hybrid conversation holds at idle
over user turns. 45e157bab kept EVERY message boundary on a rewriting
template (24 entries / 445.7 MB vs 3 / 82.9 MB). Now: one boundary per
conversation where the template rewrites tool loops, none extra elsewhere;
the idle count stays flat and every new user message resumes."""

import pytest

from tests.test_memory_guard_session import (  # noqa: F401
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


def _resident(s):
    e = s.memory_aware_cache._entries
    return len(e), sum(int(getattr(v, "memory_bytes", 0) or 0) for v in e.values())


@pytest.mark.parametrize("strips", [True, False], ids=["rewrites", "keeps"])
def test_idle_entries_over_user_turns(cpu, strips):
    s = _hybrid_scheduler(_tiny_hybrid(), _tiny_tokenizer(strips))
    user = _hybrid_tokens(1, 1500)
    rows = []
    for t in range(4):
        prompt, loop_start, _last = _hybrid_tool_loop(s, user, f"t{t}")
        nu = (
            _strip_loop(prompt, user, loop_start, 77 + t)
            if strips
            else prompt + _hybrid_tokens(90 + t, 50)
        )
        cached, out = _hybrid_run(s, f"u{t}", nu, len(nu) - 3, turn_start=True, **_MAIN)
        n, b = _resident(s)
        rows.append((t, len(nu), cached, n, round(b / 1e6, 1)))
        assert cached >= len(user), rows
        user = nu + out + [906] + _hybrid_tokens(78 + t, 40)
    counts = [r[3] for r in rows]
    assert max(counts) <= 4 and counts[-1] <= counts[1], rows
