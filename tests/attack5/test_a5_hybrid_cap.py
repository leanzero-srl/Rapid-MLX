"""Review 5, attack 2: the hybrid entry-count bound (production default 8,
goose passes 24) evicted plain LRU on every Mac, not only under duress:
main turn (a tool loop) -> a subagent's K-request tool loop (its own
session) -> main's next request. 45e157bab: main cached 512 of 4,643. The
bound now evicts in the guard's order (the subagent's dead entries first,
main's newest head last)."""

import pytest

from tests.test_memory_guard_session import (  # noqa: F401
    _GEN_PROMPT,
    _MAIN,
    _hybrid_run,
    _hybrid_scheduler,
    _hybrid_tokens,
    _hybrid_tool_loop,
    _no_metal,
    _tiny_hybrid,
    _tiny_tokenizer,
    cpu,
)


@pytest.mark.parametrize("cap", [8, 24])
@pytest.mark.parametrize("strips", [True, False], ids=["rewrites", "keeps"])
@pytest.mark.parametrize("k", [3, 6])
def test_main_survives_subagent_loop(cpu, strips, k, cap):
    s = _hybrid_scheduler(_tiny_hybrid(), _tiny_tokenizer(strips), entries=cap)
    main = _hybrid_tokens(1, 1500)
    prompt, _start, _last = _hybrid_tool_loop(s, main, "m")
    _cached, out = _hybrid_run(s, "m-last", prompt, len(prompt) - 3, **_MAIN)
    line = prompt + out
    sub = {"session": "sub", "frame": 300, "first_end": 900}
    sp = _hybrid_tokens(300, 1200) + _GEN_PROMPT
    b = 1200
    for j in range(k):
        _c, o = _hybrid_run(s, f"sub{j}", sp, b, turn_start=j == 0, **sub)
        nxt = sp + o + [904] + _hybrid_tokens(400 + j, 300)
        b = len(nxt)
        sp = nxt + _GEN_PROMPT
    nxt = line + [904] + _hybrid_tokens(500, 300) + _GEN_PROMPT
    cached, _o = _hybrid_run(s, "m-next", nxt, len(nxt) - 3, **_MAIN)
    assert cached >= len(line) - 8, (cached, len(line))
