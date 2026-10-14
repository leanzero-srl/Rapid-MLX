"""Review 5, attack 1: tools-first chat templates (Qwen3.5, Qwen3.8-27B-
Atlassian) render the reasoning-effort line and the sorted tool JSON BEFORE
goose's system prompt, so the main session and its subagent (which inherits
the session's extensions) share their first 539-577 tokens -- more than a
KV block. 45e157bab took that block as the session: main and subagent were
one "session" and the main session missed 15 of 15 turns at 64 GB.

goose sends ``agent-session-id`` on every chat request (a subagent has its
own); without it, the chat route's measured frame + first message tells the
conversations apart. Either way the main session hits on every turn."""

import pytest

from tests.test_memory_guard_session import (  # noqa: F401
    GABEE_64_CEILING,
    M4_MAX_128_CEILING,
    SUBAGENT_TOKENS,
    _auto_budget_scheduler,
    _Device,
    _drive,
    _new_tokens,
    _no_metal,
    _request,
    _side_prompt,
    _tokens,
    cpu,
    llama,
)

SHARED = 300  # > KVCache.step (256); goose's real tool blocks are thousands
TOOLS = [((i * 5) % 113) + 3 for i in range(SHARED)]
MAIN_FIRST = 600  # the main conversation's first message after the tools


def _hint(kind: str, session: str) -> dict:
    if kind == "header":
        return {"session": session, "frame": SHARED}
    # No header: the frame + first message the route measures.
    first = MAIN_FIRST if session == "main" else 320
    return {"first_end": SHARED + first}


@pytest.mark.parametrize("kind", ["header", "frame"])
@pytest.mark.parametrize(
    "ceiling", [GABEE_64_CEILING, M4_MAX_128_CEILING], ids=["64GB", "128GB"]
)
def test_main_hits_after_subagent_with_shared_tool_block(
    llama, monkeypatch, ceiling, kind
):
    device = _Device(physical=int(ceiling / 0.97) + 1, scratch_scale=600.0)
    scheduler = _auto_budget_scheduler(llama, monkeypatch, device, max_num_seqs=4)
    device.install(monkeypatch, scheduler)
    prompt = TOOLS + _new_tokens(0, MAIN_FIRST)
    misses = []
    for turn in range(16):
        main = _request(f"main-{turn}", prompt, 4, **_hint(kind, "main"))
        outs = _drive(scheduler, [main])[f"main-{turn}"]
        assert not outs[-1].error, (turn, outs[-1].error)
        if turn and len(prompt) - outs[-1].cached_tokens > 508:
            misses.append((turn, len(prompt), outs[-1].cached_tokens))
        sub = _request(
            f"sub-{turn}",
            TOOLS + _side_prompt(turn, SUBAGENT_TOKENS),
            8,
            **_hint(kind, f"sub-{turn}"),
        )
        assert not _drive(scheduler, [sub])[sub.request_id][-1].error, turn
        prompt = prompt + _tokens(outs) + _new_tokens(turn + 1, 496)
    assert misses == []
    assert device.max_seen <= device.ceiling
