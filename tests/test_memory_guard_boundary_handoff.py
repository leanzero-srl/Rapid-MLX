# SPDX-License-Identifier: Apache-2.0
"""A hybrid conversation whose message-boundary COPY does not fit keeps its
resume point anyway: the turn end trims the request's own KV back to the
boundary, attaches the recurrent state saved there, and hands it over.

RC8 live side ladder (Gabee, 2026-10-16, lz/memory-guard-rc8 d550069a4,
stock Qwen3.8-27B Q8 + MTP, ceiling 50.10 GB). The main conversation reused
its cache through 120,601 tokens, then at 05:34:10

    skipped storing a 9.22GB prefix entry: resident 43.08GB + entry + margin
    1.00GB > ceiling 50.10GB -- the conversation keeps NO entry

(chatcmpl-e91 had taken its 111,513-token hit over by admission handoff, so
the live row held the conversation's only KV). The next main step MISSed
(126,664 tokens, ``LCP unavailable ... non_trimmable=True``, 922 s to the
first token), and so did the one after it. Real capacity: Q8 weights, the
live KV at ~120k and a full COPY of it do not fit on a 64 GB Mac. The only
way out is never to hold two copies of the KV.

At the boundary only what cannot be trimmed is kept (the linear-attention
layers' recurrent + conv state, ~147 MB on the 27B); the full-attention KV is
not copied. KV positions below the boundary are written once, by the
prefill, and never again (decode, MTP verify and rollback only append past
the prompt and trim back to it), so at the turn end the request's own KV
trimmed to the boundary IS the boundary's KV, whatever the verifier left
past it.

The first half runs the REAL scheduler, mlx-lm ``BatchGenerator`` and the
vendored MTP verifier on a tiny Qwen3.5 model (GatedDeltaNet + attention,
random weights, MTP head injected) on the CPU device: the answers after a
handed-over boundary are the answers after a copied one. The second half
replays the RC8 numbers at 1/1000 scale against the guard's accounting.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import ArraysCache, KVCache  # noqa: E402

from rapid_mlx.request import Request, SamplingParams  # noqa: E402
from tests.test_lz_singleton_srpf import (  # noqa: E402,F401
    _no_metal,
    _run,
    _scheduler,
    cpu,
)

BOUNDARY_1, PROMPT_1 = 48, 61


@pytest.fixture(autouse=True)
def _calm_kernel(monkeypatch):
    """The laptop's own kernel pressure level must not reach these tests
    (the guard gives entries back at WARN)."""
    import rapid_mlx.memory_budget as memory_budget

    monkeypatch.setattr(memory_budget, "system_memory_pressure", lambda: 0)
BOUNDARY_2, PROMPT_2 = 72, 83


@pytest.fixture(scope="module")
def hybrid4(cpu):
    """Layers 0/2 GatedDeltaNet (ArraysCache), 1/3 attention (KVCache)."""
    from mlx_lm.models.qwen3_5 import TextModel

    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import inject_mtp_support
    from tests.test_lz_singleton_srpf import _tiny_args

    args = _tiny_args(2)
    object.__setattr__(args, "num_hidden_layers", 4)
    mx.random.seed(0)
    model = TextModel(args)
    assert inject_mtp_support(model, allow_random_init=True)
    mx.eval(model.parameters())
    assert [layer.is_linear for layer in model.layers] == [True, False, True, False]
    return model


def _turn(rid: str, prompt: list[int], boundary: int, max_tokens: int) -> Request:
    request = Request(
        request_id=rid,
        prompt=list(prompt),
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
    )
    request.prefix_boundary = boundary
    return request


def _conversation():
    turn_1 = [(7 * i + 3) % 120 + 1 for i in range(PROMPT_1)]
    # the next turn re-renders the reply: it shares turn 1's prompt only up
    # to the message boundary
    turn_2 = turn_1[:BOUNDARY_1] + [(11 * i + 5) % 120 + 1 for i in range(PROMPT_2 - BOUNDARY_1)]
    return turn_1, turn_2


def _cached_scheduler(model, *, refuse_copies: bool):
    s = _scheduler(
        model,
        enable_prefix_cache=True,
        use_memory_aware_cache=True,
        hybrid_cache_entries=8,
    )
    seen = {"copies_refused": 0, "handoffs": []}
    original = s._memguard_room_for_store

    def room(request, cache, **kwargs):
        has_kv = any(
            isinstance(layer, KVCache) and (layer.offset or 0) > 0
            for layer in (cache or ())
        )
        if refuse_copies and has_kv and not kwargs.get("turn_end"):
            # the copy of the live KV does not fit beside it (Gabee 05:34:10)
            seen["copies_refused"] += 1
            return False
        return original(request, cache, **kwargs)

    hand_over = getattr(s, "_memguard_hand_over_boundary", None)
    s._memguard_room_for_store = room
    if hand_over is None:  # d550069a4: no turn-end handoff to observe
        return s, seen

    def spy(request, kv, **kwargs):
        seen["handoffs"].append(
            {
                "emitted": len(request.prompt_token_ids) + request.num_output_tokens,
                "kv_offsets": [l.offset for l in kv if isinstance(l, KVCache)],
            }
        )
        return hand_over(request, kv, **kwargs)

    s._memguard_hand_over_boundary = spy
    return s, seen


def _two_turns(model, *, refuse_copies: bool, max_tokens: int = 12):
    turn_1, turn_2 = _conversation()
    s, seen = _cached_scheduler(model, refuse_copies=refuse_copies)
    _, tokens_1, _ = _run(s, {0: [_turn("t1", turn_1, BOUNDARY_1, max_tokens)]})
    entries_after_1 = sorted(len(k) for k in s.memory_aware_cache._entries)
    _, tokens_2, cached_2 = _run(s, {0: [_turn("t2", turn_2, BOUNDARY_2, 8)]})
    return tokens_1["t1"], tokens_2["t2"], cached_2["t2"], entries_after_1, seen, s


def test_a_boundary_that_cannot_be_copied_is_handed_over_at_the_turn_end(hybrid4):
    """RED on d550069a4: both copies refused, the MTP turn end publishes
    nothing, the next turn re-prefills all 83 tokens. GREEN: turn 2 hits the
    48-token boundary, and every answer is the answer after a COPIED
    boundary (and after no cache at all)."""
    copied_1, copied_2, copied_cached, copied_entries, _, _ = _two_turns(
        hybrid4, refuse_copies=False
    )
    assert BOUNDARY_1 in copied_entries and copied_cached == BOUNDARY_1

    tokens_1, tokens_2, cached_2, entries_1, seen, s = _two_turns(
        hybrid4, refuse_copies=True
    )
    assert seen["copies_refused"] >= 1
    assert entries_1 == [BOUNDARY_1], entries_1
    assert cached_2 == BOUNDARY_1
    assert tokens_1 == copied_1
    assert tokens_2 == copied_2

    _, cold = _run(
        _cached_scheduler(hybrid4, refuse_copies=False)[0],
        {0: [_turn("cold", _conversation()[1], 0, 8)]},
    )[:2]
    assert tokens_2 == cold["cold"]


def test_the_handed_over_kv_is_the_freshly_prefilled_boundary_kv(hybrid4):
    """The entry's arrays against a fresh prefill of exactly the boundary
    tokens: KV equal position by position, recurrent state equal. The turn
    end saw KV past the prompt (the verifier's lookahead included) and cut
    it to the boundary."""
    from mlx_lm.models.cache import make_prompt_cache

    turn_1, _ = _conversation()
    s, seen = _cached_scheduler(hybrid4, refuse_copies=True)
    _run(s, {0: [_turn("t1", turn_1, BOUNDARY_1, 12)]})
    (key,) = list(s.memory_aware_cache._entries)
    assert key == tuple(turn_1[:BOUNDARY_1])
    entry = s.memory_aware_cache._entries[key]
    assert entry.message_boundary
    # what the turn end trimmed: the whole prompt plus the decoded tokens
    assert seen["handoffs"], seen
    assert all(o >= PROMPT_1 for o in seen["handoffs"][-1]["kv_offsets"])

    fresh = make_prompt_cache(hybrid4)
    for start in range(0, BOUNDARY_1, 8):
        hybrid4(mx.array([turn_1[start : start + 8]]), cache=fresh)
        mx.eval([c.state for c in fresh])
    stored = entry.cache
    assert len(stored) == len(fresh)
    for got, want in zip(stored, fresh, strict=True):
        if isinstance(want, KVCache):
            assert got.offset == BOUNDARY_1
            assert mx.array_equal(got.state[0], want.state[0]).item()
            assert mx.array_equal(got.state[1], want.state[1]).item()
        else:
            assert isinstance(got, ArraysCache)
            for a, b in zip(got.cache, want.cache, strict=True):
                assert mx.allclose(a, b, rtol=1e-5, atol=1e-6).item()


def test_a_scheduler_side_stop_hands_over_too(hybrid4, monkeypatch):
    """A text stop / repetition abort ends the row before mlx-lm does
    (``_retire_scheduler_finished_uid``): the boundary is kept the same way."""
    turn_1, turn_2 = _conversation()
    s, _ = _cached_scheduler(hybrid4, refuse_copies=True)
    calls = {"n": 0}
    original = s._generation_fills_memory

    def stop_after_three(request):
        calls["n"] += 1
        return request.num_output_tokens >= 3 or original(request)

    monkeypatch.setattr(s, "_generation_fills_memory", stop_after_three)
    _run(s, {0: [_turn("t1", turn_1, BOUNDARY_1, 12)]})
    assert [len(k) for k in s.memory_aware_cache._entries] == [BOUNDARY_1]
    monkeypatch.setattr(s, "_generation_fills_memory", original)
    _, tokens_2, cached_2 = _run(s, {0: [_turn("t2", turn_2, BOUNDARY_2, 8)]})
    assert cached_2["t2"] == BOUNDARY_1
    _, copied_2, *_ = _two_turns(hybrid4, refuse_copies=False)
    assert tokens_2["t2"] == copied_2


def test_the_cut_is_exact_when_the_kv_runs_ahead_of_the_emitted_tokens(hybrid4):
    """The verifier can leave KV past the emitted tokens (accepted drafts not
    yet surfaced) and a recurrent state that has consumed them. Neither
    reaches the entry: the KV is cut to the boundary and the recurrent
    layers are the state kept there. Continuing from the entry gives the
    logits of a fresh prefill, bit for bit."""
    from mlx_lm.models.cache import make_prompt_cache

    turn_1, turn_2 = _conversation()
    s, _ = _cached_scheduler(hybrid4, refuse_copies=True)

    def prefill(tokens, cache):
        for start in range(0, len(tokens), 8):
            out = hybrid4(mx.array([tokens[start : start + 8]]), cache=cache)
            mx.eval(out, [c.state for c in cache])
        return out

    at_boundary = make_prompt_cache(hybrid4)
    prefill(turn_1[:BOUNDARY_1], at_boundary)
    kept = [
        layer if not isinstance(layer, KVCache) else None for layer in at_boundary
    ]
    # the finished row: the whole prompt, the reply, and 9 positions of
    # lookahead nobody emitted
    finished = make_prompt_cache(hybrid4)
    prefill(turn_1 + [5, 6, 7, 8, 9, 10, 11, 12, 13] * 3, finished)
    assert finished[1].offset == PROMPT_1 + 27

    request = Request(
        request_id="t1",
        prompt=list(turn_1),
        sampling_params=SamplingParams(max_tokens=4),
    )
    request.prompt_token_ids = list(turn_1)
    request._memguard_boundary_state = (BOUNDARY_1, kept)
    assert s._memguard_hand_over_boundary(request, finished)
    entry = s.memory_aware_cache._entries[tuple(turn_1[:BOUNDARY_1])]

    fresh = make_prompt_cache(hybrid4)
    prefill(turn_1[:BOUNDARY_1], fresh)
    tail = turn_2[BOUNDARY_1:]
    got = prefill(tail, entry.cache)
    want = prefill(tail, fresh)
    assert mx.array_equal(got, want).item()


# -- the RC8 numbers, 1/1000 scale ------------------------------------------

MB = 10**6
CEILING = int(50.10 * MB)
RC8_PROMPT, RC8_BOUNDARY, RC8_FETCHED = 120_601, 120_588, 111_513
RC8_REPLY = 48
KV_LAYERS = 9  # 9 x 0.965 MB of KV at 120.6k tokens + 0.15 MB recurrent: 8.83 MB


def _kv_rows(rows: int) -> KVCache:
    layer = KVCache()
    layer.keys = mx.zeros((1, 1, rows, 1), mx.float32)
    layer.values = mx.zeros((1, 1, rows, 1), mx.float32)
    layer.offset = rows
    mx.eval(layer.keys, layer.values)
    return layer


def _recurrent() -> ArraysCache:
    layer = ArraysCache(size=2)
    layer[0] = mx.zeros((1, 18_750), mx.float32)  # 0.075 MB conv state
    layer[1] = mx.zeros((1, 18_750), mx.float32)  # 0.075 MB SSM state
    mx.eval(layer[0], layer[1])
    return layer


def _rc8(resident: float = 43.08):
    """05:34:10.903: main chatcmpl-e91 (120,601 tokens) at its message
    boundary, its 111,513-token hit taken over by admission handoff (the
    live row is the conversation's only KV); the side's 22,848-token entry
    loaned to the waiting side request (its eviction frees nothing)."""
    import copy
    from dataclasses import replace

    from rapid_mlx.memory_guard import MemoryGuard, PeakPrior
    from tests.test_prompt_cache_snapshot import _make_scheduler_with_cache

    s = _make_scheduler_with_cache()
    s.memory_aware_cache._config = replace(
        s.memory_aware_cache._config, hybrid_reuse_max_entries=8
    )
    s._memory_guard = MemoryGuard(PeakPrior(100.0, 0.0, 0.0, 0.0), floor_tokens=256)
    s._memguard_ceiling = lambda: CEILING
    mac = s.memory_aware_cache
    base = int((resident - 2.16) * MB)  # weights + main's live row

    def active() -> int:
        return base + sum(int(e.memory_bytes) for e in mac._entries.values())

    s._current_metal_active_bytes = active
    s._current_metal_cache_bytes = lambda: 0
    side_tokens = [10**6 + i for i in range(22_863)]
    side_layer = _kv_rows(int(2.16 * MB) // 8)
    assert mac.store(side_tokens[:22_848], [side_layer], evict_prefixes=False)
    side = Request(
        request_id="46e",
        prompt="ignored",
        prompt_token_ids=side_tokens,
        sampling_params=SamplingParams(max_tokens=256),
    )
    side.cached_tokens = 22_848
    side.prompt_cache = copy.deepcopy(mac._entries[tuple(side_tokens[:22_848])].cache)
    side._memguard_line_key = (("session", "side"), (0, 0))
    s.waiting.append(side)
    main = Request(
        request_id="e91",
        prompt="ignored",
        prompt_token_ids=list(range(RC8_PROMPT)),
        sampling_params=SamplingParams(max_tokens=RC8_REPLY),
    )
    main.cached_tokens = RC8_FETCHED
    main.prefix_boundary = RC8_BOUNDARY
    main._memguard_line_key = (("session", "main"), (0, 0))
    s.running["e91"] = main
    s.requests["e91"] = main
    s.uid_to_request_id[52] = "e91"
    s.request_id_to_uid["e91"] = 52
    return s, main


def _rc8_boundary(s):
    snapshot = [_recurrent()] + [_kv_rows(RC8_BOUNDARY) for _ in range(KV_LAYERS)]
    bg = MagicMock()
    bg.extract_cache.return_value = {52: (snapshot, list(range(RC8_BOUNDARY)))}
    bg._generation_batch = SimpleNamespace(uids=[], __len__=lambda self: 0)
    s.batch_generator = bg
    s._extract_cache_states = MagicMock(return_value=[{"k": "v"}])
    s._reconstruct_cache_from_states = MagicMock(return_value=snapshot)
    s._snapshot_boundary_segments(
        [
            SimpleNamespace(
                uid=52,
                progress=(RC8_BOUNDARY - RC8_FETCHED, RC8_PROMPT - RC8_FETCHED),
                end_of_segment=True,
                end_of_prompt=False,
            )
        ]
    )
    return snapshot


def _rc8_turn_end(s, main, *, lookahead: int = 3):
    """The MTP turn end: the row's KV holds prompt + reply (+ lookahead);
    the generation batch is empty (the side waits)."""
    main.output_token_ids = list(range(RC8_PROMPT, RC8_PROMPT + RC8_REPLY))
    held = RC8_PROMPT + RC8_REPLY + lookahead
    finished = [_recurrent()] + [_kv_rows(held) for _ in range(KV_LAYERS)]
    s.batch_generator._generation_batch = []
    main._memguard_turn_end_kv = finished
    s._cleanup_finished({"e91"})


from unittest.mock import MagicMock  # noqa: E402


def test_rc8_the_boundary_copy_does_not_fit_and_the_turn_end_keeps_it(cpu, caplog):
    """RED on d550069a4: the copy is skipped (43.08 + 8.83 + 1.00 > 50.10; the live 9.22),
    the MTP turn end publishes nothing, the conversation has no entry and
    the next turn (126,664 tokens) re-prefills. GREEN: only the recurrent
    state is kept at the boundary; at the turn end the row's KV, cut to
    120,588 tokens, is handed over (one layer's copy at a time: 43 + 1.9 +
    1.0 fits) -- and nothing else was evicted for it."""
    import logging

    from rapid_mlx.memory_cache import estimate_kv_cache_memory

    s, main = _rc8()
    assert s._metal_footprint_bytes() == pytest.approx(43.08 * MB, rel=0.001)
    snapshot = _rc8_boundary(s)
    assert estimate_kv_cache_memory(snapshot) == pytest.approx(8.83 * MB, rel=0.01)
    mac = s.memory_aware_cache
    assert tuple(range(RC8_BOUNDARY)) not in mac._entries  # the copy: skipped
    counts = s._memguard().counts
    assert counts["store_skips"] == 1
    assert counts["orphaned_store_skips"] == 0  # not known before the turn end
    with caplog.at_level(logging.WARNING):
        _rc8_turn_end(s, main)
    entry = mac._entries.get(tuple(range(RC8_BOUNDARY)))
    assert entry is not None and entry.message_boundary
    assert all(
        layer.offset == RC8_BOUNDARY for layer in entry.cache if isinstance(layer, KVCache)
    )
    assert tuple(10**6 + i for i in range(22_848)) in mac._entries  # side: kept
    assert counts["orphaned_store_skips"] == 0
    assert counts["boundary_states_kept"] == 1 and counts["boundary_handoffs"] == 1
    assert "keeps NO entry" not in caplog.text
    # the next main turn resumes from it
    nxt = list(range(RC8_BOUNDARY)) + [7] * (126_664 - RC8_BOUNDARY)
    cache, remaining = mac.fetch(nxt)
    assert cache is not None and len(remaining) == 126_664 - RC8_BOUNDARY


def test_rc8_a_turn_that_really_ends_with_no_entry_is_counted_then(cpu, caplog):
    """No room even for the recurrent state (resident 49.5 GB): the skip is
    counted as an orphan at the turn end, once, with the loud log."""
    import logging

    s, main = _rc8(resident=49.5)
    _rc8_boundary(s)
    counts = s._memguard().counts
    assert counts["orphaned_store_skips"] == 0
    assert getattr(main, "_memguard_boundary_state", None) is None
    with caplog.at_level(logging.WARNING):
        main.output_token_ids = list(range(RC8_PROMPT, RC8_PROMPT + RC8_REPLY))
        s._cleanup_finished({"e91"})
    assert counts["orphaned_store_skips"] == 1
    assert "keeps NO entry" in caplog.text


def test_rc8_a_mid_prefill_skip_followed_by_a_kept_entry_is_no_orphan(cpu, caplog):
    """05:28:34 -> 05:28:39: the boundary store was skipped mid-prefill and
    the prompt entry was stored 4 s later; the warning had already said the
    conversation keeps NO entry. Now nothing is said or counted -- and the
    boundary is still handed over: a LONGER, non-trimmable prompt entry does
    not resume a next prompt that re-renders the reply (review of a3f5247ee:
    it cancelled the handoff, and that next turn re-prefilled 126,664)."""
    import logging

    s, main = _rc8()
    _rc8_boundary(s)
    # the prompt-end store fitted after all (something finished meanwhile)
    prompt_entry = [_recurrent()] + [_kv_rows(RC8_PROMPT) for _ in range(2)]
    assert s.memory_aware_cache.store(
        list(range(RC8_PROMPT)), prompt_entry, evict_prefixes=False
    )
    with caplog.at_level(logging.WARNING):
        _rc8_turn_end(s, main)
    counts = s._memguard().counts
    assert counts["orphaned_store_skips"] == 0
    assert "keeps NO entry" not in caplog.text
    assert tuple(range(RC8_BOUNDARY)) in s.memory_aware_cache._entries
    nxt = list(range(RC8_BOUNDARY)) + [7] * (126_664 - RC8_BOUNDARY)
    cache, remaining = s.memory_aware_cache.fetch(nxt)
    assert cache is not None and len(remaining) == 126_664 - RC8_BOUNDARY


def test_rc8_a_longer_prompt_entry_does_not_hide_an_orphan(cpu, caplog):
    """The turn ends with only its prompt-end entry (the handoff did not
    fit): that entry cannot resume a re-rendered next prompt, so the
    conversation IS orphaned and says so."""
    import logging

    s, main = _rc8(resident=47.5)
    _rc8_boundary(s)
    prompt_entry = [_recurrent()] + [_kv_rows(RC8_PROMPT) for _ in range(2)]
    assert s.memory_aware_cache.store(
        list(range(RC8_PROMPT)), prompt_entry, evict_prefixes=False
    )
    with caplog.at_level(logging.WARNING):
        _rc8_turn_end(s, main)
    assert tuple(range(RC8_BOUNDARY)) not in s.memory_aware_cache._entries
    assert s._memguard().counts["orphaned_store_skips"] == 1
    assert "keeps NO entry" in caplog.text


def test_rc8_128_gb_copies_as_today(cpu):
    """128 GB: the copy fits, the boundary entry is stored at once, no state
    is set aside and the turn end does nothing extra (zero actions)."""
    s, main = _rc8()
    s._memguard_ceiling = lambda: int(0.9 * 115.4 * MB)
    _rc8_boundary(s)
    assert tuple(range(RC8_BOUNDARY)) in s.memory_aware_cache._entries
    assert getattr(main, "_memguard_boundary_state", None) is None
    counts = s._memguard().counts
    assert counts["store_skips"] == 0 and counts["orphaned_store_skips"] == 0
    assert counts["boundary_states_kept"] == 0 and counts["boundary_handoffs"] == 0


def test_a_cancelled_turn_keeps_its_boundary_too(hybrid4):
    """The client goes away mid-decode (an abort): the row leaves the batch
    through ``_do_abort_request_impl``, and the boundary its copy could not
    hold is still kept from the row's own KV."""
    turn_1, turn_2 = _conversation()
    s, _ = _cached_scheduler(hybrid4, refuse_copies=True)
    s.add_request(_turn("t1", turn_1, BOUNDARY_1, 40))
    for _ in range(400):
        output = s.step()
        if any(o.request_id == "t1" and (o.new_token_ids or ()) for o in output.outputs):
            if s.requests["t1"].num_output_tokens >= 3:
                break
    assert s.requests["t1"].num_output_tokens >= 3
    assert s._do_abort_request_impl("t1")
    assert [len(k) for k in s.memory_aware_cache._entries] == [BOUNDARY_1]
    _, tokens_2, cached_2 = _run(s, {0: [_turn("t2", turn_2, BOUNDARY_2, 8)]})
    assert cached_2["t2"] == BOUNDARY_1
    _, copied_2, *_ = _two_turns(hybrid4, refuse_copies=False)
    assert tokens_2["t2"] == copied_2


def test_rc8_the_handoff_is_planned_at_its_real_transient(cpu):
    """Each layer is cut from the finished row's copy: for a moment that
    layer's full copy (0.97 MB) and its cut copy (0.97 MB) both exist. At
    47.5 MB resident one layer fits under the margin and two do not: the
    handoff is refused, honestly, rather than planned at one layer."""
    s, main = _rc8(resident=47.5)
    _rc8_boundary(s)
    assert getattr(main, "_memguard_boundary_state", None) is not None
    _rc8_turn_end(s, main)
    assert tuple(range(RC8_BOUNDARY)) not in s.memory_aware_cache._entries
    assert s._memguard().counts["orphaned_store_skips"] == 1


def test_the_cut_plan_prices_a_handover_and_a_copy(cpu):
    """Solo row: one layer's full copy + its cut copy. A batch still
    running keeps its buffers, so the cut copies all stay, plus one full
    copy in flight. A layout the turn end cannot rebuild is refused."""
    from rapid_mlx.scheduler import Scheduler

    kv = [_recurrent(), _kv_rows(1000), _kv_rows(1000)]
    full = 2 * 1000 * 4
    kept = (400, [kv[0], None, None])
    assert Scheduler._memguard_boundary_cut(kept, kv) == (
        1000,
        full + full * 400 // 1000,
        2 * (full * 400 // 1000) + full,
    )
    assert Scheduler._memguard_boundary_cut((1200, [kv[0], None, None]), kv) is None
    assert Scheduler._memguard_boundary_cut((400, [kv[0], None]), kv) is None
    rotating = [kv[0], ArraysCache(size=2), kv[2]]
    assert Scheduler._memguard_boundary_cut(kept, rotating) is None


def test_an_abort_mid_prefill_beside_other_prefilling_rows_is_planned_as_a_copy(cpu):
    """The row's buffers are shared with the other rows of its prompt batch
    (filter gathers them): cutting it is a COPY, not a handover."""
    from rapid_mlx.scheduler import Scheduler

    s = Scheduler.__new__(Scheduler)
    for stage, rows, alone in ((1, 2, False), (1, 1, True), (2, 1, True), (2, 3, False)):
        s.batch_generator = SimpleNamespace(
            _find_uids=lambda uids, stage=stage: {uids[0]: (stage, 0)},
            _prompt_batch=[object()] * (rows if stage == 1 else 0),
            _generation_batch=[object()] * (rows if stage == 2 else 0),
        )
        assert s._memguard_row_is_alone(9) is alone, (stage, rows)


def test_a_cancelled_parked_turn_keeps_its_boundary_too(hybrid4):
    """A row parked by SRPF after its boundary holds its own cache on the
    request; cancelling it then keeps the boundary from that cache."""
    turn_1, turn_2 = _conversation()
    s, _ = _cached_scheduler(hybrid4, refuse_copies=True)
    long_ = _turn("t1", turn_1, BOUNDARY_1, 4)
    s.add_request(long_)
    for _ in range(200):
        s.step()
        if getattr(long_, "_memguard_boundary_state", None) is not None:
            break
    assert getattr(long_, "_memguard_boundary_state", None) is not None
    # park it: a 3-token request takes the engine at the next chunk boundary
    s.add_request(_turn("short", [40, 41, 42], 0, 1))
    for _ in range(50):
        s.step()
        if long_.request_id not in s.request_id_to_uid and long_.prompt_cache:
            break
    assert long_.request_id not in s.request_id_to_uid and long_.prompt_cache
    assert s._do_abort_request_impl("t1")
    assert BOUNDARY_1 in [len(k) for k in s.memory_aware_cache._entries]
    while s.has_requests():
        s.step()
    _, tokens_2, cached_2 = _run(s, {0: [_turn("t2", turn_2, BOUNDARY_2, 8)]})
    assert cached_2["t2"] == BOUNDARY_1
    _, copied_2, *_ = _two_turns(hybrid4, refuse_copies=False)
    assert tokens_2["t2"] == copied_2


def test_rc8_an_error_abort_releases_the_kept_state_and_counts_the_orphan(cpu):
    """A generation error aborts the running rows (the batch is gone, its KV
    with it): the kept recurrent state is released at once and the
    conversation, left with no entry, is counted."""
    s, main = _rc8()
    _rc8_boundary(s)
    assert main._memguard_boundary_state is not None
    s._recover_from_generation_error()
    assert main._memguard_boundary_state is None
    assert s._memguard().counts["orphaned_store_skips"] == 1


def test_the_orphan_check_ignores_entries_past_the_message_boundary(cpu):
    """Directly: a skipped store, then only the 120,601-token prompt entry
    left -- longer than the 120,588-token boundary, so not a resume point
    for a re-rendered next prompt: orphaned. A boundary entry: not."""
    s, main = _rc8()
    main._memguard_store_skipped = True
    prompt_entry = [_recurrent()] + [_kv_rows(RC8_PROMPT)]
    assert s.memory_aware_cache.store(
        list(range(RC8_PROMPT)), prompt_entry, evict_prefixes=False
    )
    s._memguard_note_orphaned_turn(main)
    assert s._memguard().counts["orphaned_store_skips"] == 1
    main._memguard_store_skipped = True
    boundary_entry = [_recurrent()] + [_kv_rows(RC8_BOUNDARY)]
    assert s.memory_aware_cache.store(
        list(range(RC8_BOUNDARY)), boundary_entry, evict_prefixes=False
    )
    s._memguard_note_orphaned_turn(main)
    assert s._memguard().counts["orphaned_store_skips"] == 1
