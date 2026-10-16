# SPDX-License-Identifier: Apache-2.0
"""A conversation's next-turn entry is stored whenever anything cheaper can
give the room.

RC7 live ladder (Gabee, 2026-10-16, lz/memory-guard-rc7 354c3ac0e, stock
Qwen3.8-27B Q8 + MTP, ceiling 50.10 GB). Hybrid Qwen resumes each turn from
the previous turn's MESSAGE-BOUNDARY snapshot (the prompt-end entry is
superseded; no end-of-turn entry matches a re-rendered prompt). Step 14
(main chatcmpl-393, 87,412 tokens) took its 86,477-token hit over by
admission handoff -- the entry then lived only in the live batch -- and its
new boundary snapshot (87,375 tokens) was refused:

    02:50:30.505 skipped storing a 6.50GB prefix entry: resident 43.02GB +
                 entry + margin 1.00GB > ceiling 50.10GB

so step 15 MISSed (88,730 tokens re-prefilled, 567 s to the first token),
and so did step 16. A side request (chatcmpl-d50, 22,863 tokens) was
waiting with a loan of its own 22,848-token entry.

What the store had to find was 0.42 GB, and two things were wrong:

1. The 6.50 GB counted 0.62 GB of recurrent-state CHECKPOINTS twice: they
   ride on the layer by reference -- the very arrays the live request's
   checkpoint holders keep -- so storing them allocates nothing.
2. Only REDUNDANT entries could give room. Entries of other conversations
   that are cheaper to re-prefill (the side's 22k-token entries against
   the main's 87k), or the conversation's own older ones, now give way,
   cheapest first, only for real bytes (an
   entry a queued request's loan shares frees nothing and stays) and only
   when that makes the store fit. A skip that leaves the conversation with
   no entry is counted (``orphaned_store_skips``) and logged loudly.

Double count of the main's LIVE KV: none. The boundary entry is a copy of a
row that keeps prefilling and decoding, so live + entry are both real.

Replayed at 1/1000 scale (MB for GB) with real arrays.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache, KVCache

from rapid_mlx.hybrid_state_checkpoints import StateCheckpoints, attach_checkpoints
from rapid_mlx.memory_guard import MemoryGuard, PeakPrior
from rapid_mlx.request import Request, SamplingParams
from tests.test_memory_guard_session import _no_metal, cpu  # noqa: F401

MB = 10**6
CEILING = int(50.10 * MB)
MARGIN = int(0.02 * CEILING)
BOUNDARY, PROMPT, FETCHED = 87_375, 87_412, 86_477
SIDE_TOKENS = [10**6 + i for i in range(22_863)]
MAIN_LINE = (("session", "main"), (0, 0))


def _floats(nbytes: int) -> mx.array:
    a = mx.random.normal((max(1, nbytes // 4),))
    mx.eval(a)
    return a


def _kv(nbytes: int, offset: int) -> KVCache:
    layer = KVCache()
    rows = max(1, nbytes // (2 * 4 * 64))
    layer.keys = mx.random.normal((1, 1, rows, 64))
    layer.values = mx.random.normal((1, 1, rows, 64))
    layer.offset = rows
    mx.eval(layer.keys, layer.values)
    return layer


def _recurrent(state_bytes: int) -> ArraysCache:
    layer = ArraysCache(size=2)
    layer[0] = _floats(state_bytes // 2)
    layer[1] = _floats(state_bytes // 2)
    return layer


def _scheduler(resident_extra: int = 0):
    from tests.test_prompt_cache_snapshot import _make_scheduler_with_cache

    from dataclasses import replace

    s = _make_scheduler_with_cache()
    # hybrid entries are kept (the engine's --hybrid-reuse setting on goose)
    s.memory_aware_cache._config = replace(
        s.memory_aware_cache._config, hybrid_reuse_max_entries=8
    )
    s._memory_guard = MemoryGuard(
        PeakPrior(100.0, 0.0, 0.0, 0.0), floor_tokens=256
    )
    s._memguard_ceiling = lambda: CEILING
    mac = s.memory_aware_cache
    base = int(38.40 * MB) + resident_extra  # weights + the main's live row

    def active() -> int:
        return base + sum(
            int(e.memory_bytes)
            for k, e in mac._entries.items()
            if k[:3] != tuple(range(3))  # the main's new entry: checked separately
        )

    s._current_metal_active_bytes = active
    s._current_metal_cache_bytes = lambda: 0
    return s


def _other(s, tokens: list[int], nbytes: int, line: str):
    layer = _kv(nbytes, len(tokens))
    assert s.memory_aware_cache.store(tokens, [layer], evict_prefixes=False)
    entry = s.memory_aware_cache._entries[tuple(tokens)]
    entry.owner = (("session", line), (0, 0))
    return tuple(tokens)


def _trace(resident_extra: int = 0):
    """The state at 02:50:30.505: the main mid-prefill at its message
    boundary, the side waiting with a loan of its own entry, two more
    entries of other (side) conversations."""
    s = _scheduler(resident_extra)
    side_boundary = _other(s, SIDE_TOKENS[:22_848], int(2.16 * MB), "side-d50")
    older_side = _other(
        s, [2 * 10**6 + i for i in range(22_863)], int(1.40 * MB), "side-older"
    )
    helper = _other(s, [3 * 10**6 + i for i in range(1_376)], int(1.06 * MB), "helper")
    # the side request waits with its loan (a deepcopy: the same buffers)
    side = Request(
        request_id="d50",
        prompt="ignored",
        prompt_token_ids=list(SIDE_TOKENS),
        sampling_params=SamplingParams(max_tokens=256),
    )
    side.cached_tokens = 22_848
    side.prompt_cache = copy.deepcopy(
        s.memory_aware_cache._entries[side_boundary].cache
    )
    side._memguard_line_key = (("session", "side-d50"), (0, 0))
    s.waiting.append(side)
    # the main, prefilling: its boundary snapshot, checkpoints by reference
    main = Request(
        request_id="393",
        prompt="ignored",
        prompt_token_ids=list(range(PROMPT)),
        sampling_params=SamplingParams(max_tokens=48),
    )
    main.cached_tokens = FETCHED
    main.prefix_boundary = BOUNDARY
    main._memguard_line_key = MAIN_LINE
    s.running["393"] = main
    s.requests["393"] = main
    s.uid_to_request_id[36] = "393"
    s.request_id_to_uid["393"] = 36
    checkpoints = StateCheckpoints(
        [(80_000, (_floats(int(0.31 * MB)),)), (86_000, (_floats(int(0.31 * MB)),))]
    )
    s._hybrid_checkpoints[36] = [None, checkpoints]
    snapshot = [_kv(int(5.73 * MB), BOUNDARY), _recurrent(int(0.15 * MB))]
    attach_checkpoints(snapshot, [None, checkpoints])
    return s, snapshot, side_boundary, older_side, helper


def _boundary_store(s, snapshot):
    """Drive the real message-boundary snapshot path."""
    bg = MagicMock()
    bg.extract_cache.return_value = {36: (snapshot, list(range(BOUNDARY)))}
    s.batch_generator = bg
    s._extract_cache_states = MagicMock(return_value=[{"k": "v"}])
    s._reconstruct_cache_from_states = MagicMock(return_value=snapshot)
    s._snapshot_boundary_segments(
        [
            SimpleNamespace(
                uid=36,
                progress=(BOUNDARY - FETCHED, PROMPT - FETCHED),
                end_of_segment=True,
                end_of_prompt=False,
            )
        ]
    )
    return tuple(range(BOUNDARY)) in s.memory_aware_cache._entries


def test_the_rc7_boundary_store_is_kept(cpu):
    """43.02 GB resident, a 6.50 GB entry of which 0.62 GB are resident
    checkpoints: 43.02 + 5.88 + 1.00 = 49.90 <= 50.10. Stored, nothing
    evicted (354c3ac0e skipped it: every later turn MISSed)."""
    s, snapshot, side_boundary, older_side, helper = _trace()
    from rapid_mlx.memory_cache import estimate_kv_cache_memory

    assert estimate_kv_cache_memory(snapshot) == pytest.approx(6.50 * MB, rel=0.02)
    assert s._metal_footprint_bytes() == pytest.approx(43.02 * MB, rel=0.001)
    assert _boundary_store(s, snapshot)
    for key in (side_boundary, older_side, helper):
        assert key in s.memory_aware_cache._entries
    counts = s._memguard().counts
    assert counts["store_skips"] == 0


def test_cheaper_conversations_give_way_for_the_next_turn_entry(cpu):
    """1 GB more resident: the side's entries -- cheaper to re-prefill --
    give the room, cheapest first; the one the waiting side's loan shares
    frees nothing and stays."""
    s, snapshot, side_boundary, older_side, helper = _trace(resident_extra=MB)
    with s.memory_aware_cache._lock:
        assert s._memguard_freeable_bytes()[side_boundary] == 0
    assert _boundary_store(s, snapshot)
    entries = s.memory_aware_cache._entries
    assert side_boundary in entries  # loan-held: evicting it gives nothing
    assert helper not in entries  # the cheapest: 1,376 tokens
    assert older_side in entries  # not needed: one eviction made the room
    assert s._memguard().counts["orphaned_store_skips"] == 0


def test_a_costlier_conversation_never_gives_way(cpu):
    """An entry of another conversation LONGER than the one being stored is
    not traded for it, even when it alone would complete the room (2.80 MB
    short; the cheaper ones give 2.46): the skip happens, loudly, and is
    counted -- and nothing is evicted for a store that does not happen."""
    s, snapshot, side_boundary, older_side, helper = _trace(resident_extra=MB)
    big = _other(s, [4 * 10**6 + i for i in range(90_000)], int(2.0 * MB), "other-main")
    assert not _boundary_store(s, snapshot)
    entries = s.memory_aware_cache._entries
    assert all(k in entries for k in (big, side_boundary, older_side, helper))
    counts = s._memguard().counts
    assert counts["store_skips"] == 1 and counts["orphaned_store_skips"] == 1


def test_nothing_is_evicted_when_it_would_not_make_room(cpu):
    """Feasibility first: if every cheaper entry together cannot make the
    room, none is evicted."""
    s, snapshot, side_boundary, older_side, helper = _trace(resident_extra=6 * MB)
    assert not _boundary_store(s, snapshot)
    entries = s.memory_aware_cache._entries
    assert all(k in entries for k in (side_boundary, older_side, helper))


def test_128_gb_stores_without_any_action(cpu):
    s, snapshot, side_boundary, older_side, helper = _trace()
    s._memguard_ceiling = lambda: int(0.9 * 115.4 * MB)
    assert _boundary_store(s, snapshot)
    assert all(
        k in s.memory_aware_cache._entries for k in (side_boundary, older_side, helper)
    )
    counts = s._memguard().counts
    assert counts["store_skips"] == 0 and counts["orphaned_store_skips"] == 0


def test_a_prefilling_row_reserves_its_own_narrowest_verify(cpu):
    """A message-boundary snapshot is taken mid-prefill: the decode that
    follows needs at least the narrowest verify's restore points (27B: 3
    rows x 153.9 MB), reserved by the store so it is not stored-then-evicted."""
    from tests.test_memory_guard_mtp_decode import _model_27b

    s = _scheduler()
    s.config.spec_decode = "mtp"
    s.config.mtp_max_k = 3
    s.model = _model_27b()
    main = Request(
        request_id="393",
        prompt="ignored",
        prompt_token_ids=list(range(PROMPT)),
        sampling_params=SamplingParams(max_tokens=48),
    )
    s.running["393"] = main
    s.request_id_to_uid["393"] = 36
    s.batch_generator = SimpleNamespace(_generation_batch=None)
    assert s._memguard_own_decode_floor(main) == 3 * 153_944_064
    s.config.spec_decode = "none"
    assert s._memguard_own_decode_floor(main) == 0
