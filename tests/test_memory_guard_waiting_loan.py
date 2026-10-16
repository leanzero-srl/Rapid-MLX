# SPDX-License-Identifier: Apache-2.0
"""A queued request's prefix-hit loan gives way before a running request is
refused (RC12 live ladder, Gabee, 2026-10-16).

k=110: the main's 160,332-token prompt (hit 146,457, 13,875 to prefill) went
infeasible at 159,423 tokens, 2.5 minutes into its prefill:

    infeasible even at floor chunk 256 with the allocator cache and every
    evictable entry gone | footprint=45.59GB step=3.18GB margin=1.32GB
    ceiling=50.10GB

-- refused, "needs 50.1 GB but this Mac can hold 50.1 GB". A side request
was WAITING with a prefix-hit loan of its 22.8k-token entry: a loan shares
the entry's buffers, so evicting the entry freed nothing. Larger prompts
(162k-171k) ran fine later: a packing failure, not a capacity limit. Now the
queued loans are released first (the side re-fetches at its admission, or
re-prefills ~22k tokens if its entry went), and the main runs.

At 1/100 of the bytes on the simulated Gabee device of the session tests.
"""

from __future__ import annotations

from tests.test_memory_guard_session import (  # noqa: F401
    GABEE_64_CEILING,
    _auto_budget_scheduler,
    _Device,
    _drive,
    _new_tokens,
    _no_metal,
    _request,
    _side_prompt,
    cpu,
    llama,
)


def _gabee(llama, monkeypatch):
    device = _Device(physical=int(GABEE_64_CEILING / 0.97) + 1, scratch_scale=600.0)
    scheduler = _auto_budget_scheduler(
        llama, monkeypatch, device, max_num_seqs=1, singleton_prefill_order="fifo"
    )
    device.install(monkeypatch, scheduler)
    return scheduler, device


def _main_beside_a_queued_loan(llama, monkeypatch, back_off: int):
    scheduler, device = _gabee(llama, monkeypatch)
    side_prompt = _side_prompt(0, 3000)
    _drive(scheduler, [_request("side-0", side_prompt, 4, session="side")])
    window = scheduler.memory_guard_max_context(65536)
    main = _request("main", _new_tokens(1, window - back_off), 4, session="main")
    side = _request("side-1", side_prompt, 4, session="side")
    outs = _drive(scheduler, [main], arrive={2: [side]})
    return scheduler, device, outs


def test_a_queued_loan_gives_way_before_a_running_main_is_refused(llama, monkeypatch):
    """RED on 02ec502ac: the main (its prompt inside the advertised window)
    is refused mid-prefill because the queued side's loan pins its entry.
    GREEN: the loan is released, the main finishes, the side re-fetches."""
    refused = []
    for back_off in (120, 240, 400):
        scheduler, device, outs = _main_beside_a_queued_loan(
            llama, monkeypatch, back_off
        )
        main_out = outs["main"][-1]
        side_out = outs["side-1"][-1]
        assert main_out.finished and side_out.finished
        refused.append(bool(main_out.error))
        assert not side_out.error
        assert device.ooms == 0
    assert not any(refused), refused
    assert scheduler._memguard().counts["refusals"] == 0


def test_a_close_refusal_says_by_how_much():
    """RC12: "needs 50.1 GB but this Mac can hold 50.1 GB". A close call
    shows the digits that tell them apart and the shortfall; a clear one
    keeps its one decimal."""
    from rapid_mlx.errors import ContextExceedsDeviceMemoryError

    close = ContextExceedsDeviceMemoryError(
        prompt_tokens=160_332,
        needed_bytes=50_103_000_000,
        available_bytes=50_100_000_000,
        max_context=160_331,
    ).message
    assert "needs 50.103 GB but this Mac can hold 50.100 GB (3 MB short)" in close
    clear = ContextExceedsDeviceMemoryError(
        prompt_tokens=200_000,
        needed_bytes=58_400_000_000,
        available_bytes=50_100_000_000,
        max_context=160_000,
    ).message
    assert "needs 58.4 GB but this Mac can hold 50.1 GB;" in clear
