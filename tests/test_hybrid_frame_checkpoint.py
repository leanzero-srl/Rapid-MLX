# SPDX-License-Identifier: Apache-2.0
"""A new conversation resumes a hybrid entry at the end of the shared frame.

Measured 2026-10-31 (Gabee, Qwen3.5-9B 4-bit, engine lz.19): a second goose
chat with a byte-identical 21,860-token prefix (tools + system prompt) logged
``LCP snapped to checkpoint: shared=21860 entry_len=34055 resumed_at=16384
remaining=5917`` -- 11.6 s of re-prefill. Two causes, two fixes:

1. The prefill recorded recurrent checkpoints only at stride multiples, never
   at the frame end, so even the best entry stopped up to a stride short of
   it. The scheduler now splits the prefill at ``SessionHint.frame`` and
   records a pinned checkpoint there; thinning keeps it.
2. The LCP fetch compared only the two sorted neighbours and snapped the one
   with the longest shared prefix, even when another entry with the same
   shared prefix kept a deeper checkpoint. It now keeps the deepest resume.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.requires_mlx
ArraysCache = pytest.importorskip("mlx_lm.models.cache").ArraysCache

from rapid_mlx.hybrid_state_checkpoints import (  # noqa: E402
    CHECKPOINT_ATTR,
    StateCheckpoints,
    collect_checkpoints,
    layer_checkpoints,
    record_checkpoints,
)


class _Array:
    def __init__(self, tag: int, shape=(1, 4, 4)):
        self.tag = tag
        self.shape = shape
        self.dtype = SimpleNamespace(size=2)


class _RecurrentLayer(ArraysCache):
    def __init__(self, tag: int):
        super().__init__(2)
        self.cache = [_Array(tag), _Array(tag, (1, 2, 2))]


class _KVLayer:
    def __init__(self):
        self.offset = 0

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n):
        return n


def _live_cache(tag: int = 0):
    return [_KVLayer(), _RecurrentLayer(tag), _RecurrentLayer(tag)]


class TestPinnedFrameCheckpoint:
    def test_pin_skips_the_stride_gate(self):
        holder = StateCheckpoints()
        for pos in (2048, 4096):
            holder = holder.with_checkpoint(
                pos, (_Array(pos),), max_count=4, stride=2048
            )
        # 4600 is 504 tokens past the newest: a chunk would be refused...
        assert (
            holder.with_checkpoint(4600, (_Array(1),), max_count=4, stride=2048)
            is holder
        )
        # ... the frame end is recorded and pinned.
        pinned = holder.with_checkpoint(
            4600, (_Array(1),), max_count=4, stride=2048, pin=True
        )
        assert pinned.positions == (2048, 4096, 4600)
        assert pinned.pinned == frozenset({4600})

    def test_thinning_keeps_the_frame_through_a_growing_conversation(self):
        """The motivating shape: frame at 21,696 (tile-aligned 21,713),
        then the conversation grows to 34k with chunk checkpoints."""
        holder = StateCheckpoints()
        for pos in range(2048, 21696, 2048):
            holder = holder.with_checkpoint(
                pos, (_Array(pos),), max_count=4, stride=2048
            )
        holder = holder.with_checkpoint(
            21696, (_Array(21696),), max_count=4, stride=2048, pin=True
        )
        assert 21696 in holder.positions
        for pos in range(23744, 34055, 2048):
            holder = holder.with_checkpoint(
                pos, (_Array(pos),), max_count=4, stride=2048
            )
            assert 21696 in holder.positions, pos
            assert holder.positions[-1] == pos  # the newest is still kept
            assert len(holder) == 4  # the bound is unchanged
        assert holder.newest_at_or_below(21860) == 21696

    def test_pin_on_an_existing_position_and_truncation_keep_the_pin(self):
        holder = StateCheckpoints([(2048, (_Array(1),)), (4096, (_Array(2),))])
        again = holder.with_checkpoint(
            4096, (_Array(9),), max_count=4, stride=2048, pin=True
        )
        assert again.positions == (2048, 4096)
        assert again.arrays_at(4096)[0].tag == 2  # same state, kept
        assert again.pinned == frozenset({4096})
        assert again.with_checkpoint(4096, (), max_count=4, stride=1, pin=True) is again
        assert again.truncated(5000).pinned == frozenset({4096})
        assert again.truncated(3000).pinned == frozenset()

    def test_pins_never_exceed_the_bound(self):
        holder = StateCheckpoints()
        for pos in (100, 200, 300, 400):
            holder = holder.with_checkpoint(
                pos, (_Array(pos),), max_count=2, stride=1, pin=True
            )
        assert len(holder) == 2
        assert holder.positions[-1] == 400
        assert holder.pinned <= set(holder.positions)

    def test_record_checkpoints_pins_across_every_recurrent_layer(self):
        cache = _live_cache()
        holders = collect_checkpoints(cache)
        assert record_checkpoints(cache, holders, 2048, max_count=4, stride=2048)
        assert not record_checkpoints(cache, holders, 2500, max_count=4, stride=2048)
        assert record_checkpoints(
            cache, holders, 2500, max_count=4, stride=2048, pin=True
        )
        assert not record_checkpoints(
            cache, holders, 2500, max_count=4, stride=2048, pin=True
        )
        assert holders[0] is None
        for h in holders[1:]:
            assert h.positions == (2048, 2500)
            assert h.pinned == frozenset({2500})


@pytest.mark.requires_mlx
class TestDeepestResumeSelection:
    """Entries share ``PREFIX`` tokens with the request; the request then
    diverges with token 50. ``after`` is the divergent token of an entry:
    below 50 it sorts before the request, above 50 after it."""

    PREFIX = 5000

    @classmethod
    def _tokens(cls, after: int, extra: int) -> list[int]:
        return list(range(1000, 1000 + cls.PREFIX)) + [after] * extra

    @staticmethod
    def _entry(length: int, checkpoints: tuple[int, ...]):
        import mlx.core as mx
        from mlx_lm.models.cache import ArraysCache, KVCache

        kv = KVCache()
        kv.keys = mx.zeros((1, 1, length, 4))
        kv.values = mx.zeros((1, 1, length, 4))
        kv.offset = length
        rec = ArraysCache(2)
        rec.cache = [mx.full((1, 3, 4), length), mx.full((1, 2, 2), length)]
        holder = StateCheckpoints(
            (pos, (mx.full((1, 3, 4), pos), mx.full((1, 2, 2), pos)))
            for pos in checkpoints
        )
        setattr(rec, CHECKPOINT_ATTR, holder)
        return [kv, rec]

    @staticmethod
    def _cache():
        from rapid_mlx.memory_cache import MemoryAwarePrefixCache, MemoryCacheConfig

        config = MemoryCacheConfig(
            max_memory_mb=256, max_entries=16, hybrid_reuse_max_entries=8
        )
        return MemoryAwarePrefixCache(MagicMock(), config)

    def _store(self, cache, after: int, extra: int, checkpoints: tuple[int, ...]):
        tokens = self._tokens(after, extra)
        assert cache.store(
            tokens, self._entry(len(tokens), checkpoints), evict_prefixes=False
        )

    def _fetch(self, cache):
        request = self._tokens(50, 20)
        result, remaining = cache.fetch(request)
        assert result is not None
        return len(request) - len(remaining), result

    def test_tied_neighbours_resume_at_the_deeper_checkpoint(self):
        """Old rule: the first neighbour on a tied shared prefix (idx-1,
        here the long entry thinned to 2048) won."""
        cache = self._cache()
        self._store(cache, after=10, extra=3000, checkpoints=(2048, 8000))
        self._store(cache, after=90, extra=30, checkpoints=(2048, 4096, 4992))
        resumed, result = self._fetch(cache)
        assert resumed == 4992
        assert result[0].offset == 4992
        assert result[1].cache[0][0, 0, 0].item() == 4992
        assert cache._last_match_type == "lcp"

    def test_walks_past_the_neighbour_to_a_deeper_entry(self):
        """Old rule: only the two sorted neighbours were looked at; the
        frame-checkpointed entry sorts two places away."""
        cache = self._cache()
        self._store(cache, after=60, extra=4000, checkpoints=(2048, 9000))
        self._store(cache, after=70, extra=6000, checkpoints=(4096, 11000))
        self._store(cache, after=80, extra=10, checkpoints=(2048, 4096, 4992))
        resumed, _ = self._fetch(cache)
        assert resumed == 4992

    def test_a_shorter_shared_prefix_never_wins(self):
        cache = self._cache()
        self._store(cache, after=60, extra=100, checkpoints=(2048, 4096))
        # Shares only 3000 tokens with the request, its checkpoints cannot
        # be deeper than that.
        other = list(range(1000, 4000)) + [7] * 3000
        assert cache.store(other, self._entry(len(other), (2048, 2900)))
        resumed, _ = self._fetch(cache)
        assert resumed == 4096

    def test_no_usable_checkpoint_anywhere_is_still_a_miss(self):
        cache = self._cache()
        self._store(cache, after=60, extra=100, checkpoints=(6000,))
        self._store(cache, after=70, extra=100, checkpoints=())
        request = self._tokens(50, 20)
        result, remaining = cache.fetch(request)
        assert result is None
        assert remaining == request

    def test_common_prefix_length(self):
        from rapid_mlx.memory_cache import _common_prefix_length

        a = tuple(range(3000))
        assert _common_prefix_length(a, a) == 3000
        assert _common_prefix_length(a, a[:1500]) == 1500
        b = a[:1234] + (-1,) + a[1235:]
        assert _common_prefix_length(a, b) == 1234
        assert _common_prefix_length(a, (5,)) == 0
        assert _common_prefix_length((), a) == 0


class TestSchedulerFrameSplit:
    @staticmethod
    def _scheduler(hybrid_entries: int = 8, recurrent: bool = True, tile: int = 1):
        pytest.importorskip("mlx")
        from rapid_mlx.scheduler import Scheduler, SchedulerConfig

        tokenizer = MagicMock()
        tokenizer.encode = lambda x: list(range(len(x.split())))
        config = SchedulerConfig(
            enable_prefix_cache=True,
            use_memory_aware_cache=True,
            hybrid_cache_entries=hybrid_entries,
        )
        scheduler = Scheduler(MagicMock(), tokenizer, config)
        scheduler.batch_generator = MagicMock()
        scheduler._model_recurrent_state = recurrent
        scheduler._prefill_tile_rows_cached = tile
        return scheduler

    @staticmethod
    def _request(frame: int, cached: int = 0, length: int = 22157):
        from rapid_mlx.request import Request, SamplingParams, SessionHint

        request = Request(
            request_id="req-frame",
            prompt="ignored",
            prompt_token_ids=list(range(length)),
            sampling_params=SamplingParams(max_tokens=4),
        )
        request.cached_tokens = cached
        request.session_hint = SessionHint(frame=frame, first_end=frame + 400)
        return request

    def test_cold_prompt_splits_at_the_frame(self):
        scheduler = self._scheduler()
        request = self._request(frame=21713)
        tokens = request.prompt_token_ids
        assert scheduler._frame_checkpoint_split(request, tokens) == 21713
        assert request._frame_checkpoint == 21713

    def test_split_is_local_to_the_reused_prefix_and_tile_aligned(self):
        scheduler = self._scheduler(tile=32)
        # 13,521 frame tokens + 421 after it: split unaligned, the halves
        # cost 423 + 14 tiles against 436 unsplit, so the split rounds down.
        request = self._request(frame=21713, cached=8192, length=22134)
        tokens = request.prompt_token_ids[8192:]
        local = scheduler._frame_checkpoint_split(request, tokens)
        # 21713 - 8192 = 13521 -> 13504 (a free tile multiple)
        assert local == 13504
        assert request._frame_checkpoint == 8192 + 13504

    def test_no_split_when_the_frame_is_already_reused_or_absent(self):
        scheduler = self._scheduler()
        request = self._request(frame=21713, cached=21860)
        assert (
            scheduler._frame_checkpoint_split(request, request.prompt_token_ids[21860:])
            is None
        )
        assert request._frame_checkpoint == 0
        request = self._request(frame=0)
        assert (
            scheduler._frame_checkpoint_split(request, request.prompt_token_ids) is None
        )

    def test_no_split_without_recurrent_state_or_hybrid_entries(self):
        for scheduler in (
            self._scheduler(recurrent=False),
            self._scheduler(hybrid_entries=0),
        ):
            request = self._request(frame=21713)
            assert (
                scheduler._frame_checkpoint_split(request, request.prompt_token_ids)
                is None
            )

    def test_prefill_segments(self):
        from rapid_mlx.scheduler import Scheduler

        tokens = list(range(10))
        assert Scheduler._prefill_segments(tokens, (None, None)) is None
        assert Scheduler._prefill_segments(tokens, (0, 10)) is None
        assert Scheduler._prefill_segments(tokens, (7, None)) == [
            list(range(7)),
            [7, 8, 9],
        ]
        assert Scheduler._prefill_segments(tokens, (7, 3, 7)) == [
            [0, 1, 2],
            [3, 4, 5, 6],
            [7, 8, 9],
        ]

    def test_frame_segment_end_records_a_pinned_checkpoint(self, monkeypatch):
        monkeypatch.setenv("RAPID_MLX_HYBRID_CHECKPOINT_STRIDE", "2048")
        monkeypatch.setenv("RAPID_MLX_HYBRID_CHECKPOINT_MAX", "4")
        scheduler = self._scheduler()
        request = self._request(frame=21713)
        assert scheduler._frame_checkpoint_split(request, request.prompt_token_ids)
        scheduler.requests[request.request_id] = request
        scheduler.uid_to_request_id[3] = request.request_id
        scheduler.batch_generator.extract_cache.side_effect = lambda uids: {
            uids[0]: (_live_cache(uids[0]), [])
        }

        def chunk(processed):
            return SimpleNamespace(
                uid=3,
                progress=(processed, 22157),
                end_of_segment=processed == 21713,
                end_of_prompt=False,
            )

        for processed in (*range(2048, 21713, 2048), 21713, 22144):
            scheduler._record_hybrid_checkpoints([chunk(processed)])

        holder = scheduler._hybrid_checkpoints[3][1]
        assert 21713 in holder.positions
        assert holder.pinned == frozenset({21713})
        stored = _live_cache(9)
        scheduler._attach_hybrid_checkpoints(3, stored, length=22142)
        assert layer_checkpoints(stored[1]).newest_at_or_below(21860) == 21713
