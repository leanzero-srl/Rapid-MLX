"""Review 5, attacks 1+2 on a hybrid: a template that keeps reasoning
(Qwen3.8-Atlassian: the probe says so) with a tools-first prefix the main
session and its subagent share. 45e157bab: the subagent's first request
marked the shared "session" as rewriting history (a 512-token span, a fixed
absolute) and a subagent tool loop pushed main's entries out of the hybrid
count bound (plain LRU). Per-conversation identity + the bound in the
guard's eviction order: main's next turn resumes from its own entry."""

import pytest

from tests.test_memory_guard_session import (  # noqa: F401
    _GEN_PROMPT,
    _hybrid_run,
    _hybrid_scheduler,
    _hybrid_tokens,
    _hybrid_tool_loop,
    _no_metal,
    _tiny_hybrid,
    _tiny_tokenizer,
    cpu,
)

SHARED = 600
TOOLS = [((i * 5) % 880) + 7 for i in range(SHARED)]


def _hint(kind: str, session: str) -> dict:
    if kind == "header":
        return {"session": session, "frame": SHARED}
    return {"first_end": SHARED + (900 if session == "main" else 600)}


@pytest.mark.parametrize("kind", ["header", "frame"])
@pytest.mark.parametrize("cap", [8, 24])
@pytest.mark.parametrize("k", [3, 6])
def test_shared_tools_block_on_a_keeping_template(cpu, k, cap, kind):
    s = _hybrid_scheduler(_tiny_hybrid(), _tiny_tokenizer(False), entries=cap)
    main_hint = _hint(kind, "main")
    main = TOOLS + _hybrid_tokens(1, 900)
    prompt, _start, _last = _hybrid_tool_loop(s, main, "m", **main_hint)
    _cached, out = _hybrid_run(s, "m-last", prompt, len(prompt) - 3, **main_hint)
    line = prompt + out
    sub_hint = _hint(kind, "sub")
    sp = TOOLS + _hybrid_tokens(300, 600) + _GEN_PROMPT
    b = len(sp) - 3
    for j in range(k):
        _c, o = _hybrid_run(s, f"sub{j}", sp, b, **sub_hint)
        nxt = sp + o + [904] + _hybrid_tokens(400 + j, 300)
        b = len(nxt)
        sp = nxt + _GEN_PROMPT
    # (45e157bab marked the shared "session" here: _memguard_divergent_sessions)
    assert not s.__dict__.get("_memguard_rewriting_lines")
    assert not s.__dict__.get("_memguard_divergent_sessions")
    nxt = line + [904] + _hybrid_tokens(500, 300) + _GEN_PROMPT
    cached, _o = _hybrid_run(s, "m-next", nxt, len(nxt) - 3, **main_hint)
    assert cached >= len(line) - 8, (cached, len(line))
