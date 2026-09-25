# SPDX-License-Identifier: Apache-2.0
"""LeanZero fork: an omitted max_tokens runs until the model stops or the window
is full, and memory — not a typed default — bounds decode growth.

goose Q-65 (2026-09-25): its OpenAI-compatible client sends no max_tokens on
purpose, so the serve default (32768) silently capped every answer and refused
any prompt within 32768 tokens of the window.
"""

from __future__ import annotations

from types import SimpleNamespace

from rapid_mlx.routes.responses import (
    _resolve_context_safe_implicit_responses_max_tokens as responses_budget,
)
from rapid_mlx.scheduler import Scheduler


def _engine(window):
    return SimpleNamespace(
        _model=SimpleNamespace(args=SimpleNamespace(max_position_embeddings=window))
    )


def test_responses_omitted_budget_is_the_room_not_the_default():
    assert responses_budget(_engine(262_144), 1_000, 32_768) == 261_144
    assert responses_budget(_engine(131_072), 100_581, 32_768) == 30_491
    assert responses_budget(_engine(262_144), None, 32_768) == 32_768, "no count"


def test_responses_model_without_a_window_keeps_the_resolved_default():
    no_window = SimpleNamespace(_model=SimpleNamespace(args=SimpleNamespace()))
    assert responses_budget(no_window, 1_000, 32_768) == 32_768


class _Fake:
    """The attributes ``_generation_fills_memory`` reads, measured by hand."""

    def __init__(self, cap, active_readings, running):
        self.cap = cap
        self.readings = list(active_readings)
        self.running = {r.request_id: r for r in running}
        self.evictions = 0

    def _resolve_metal_cap_bytes(self):
        return self.cap

    def _current_metal_active_bytes(self):
        return self.readings.pop(0)

    def evict_prefix_cache_under_pressure(self):
        self.evictions += 1
        return 0

    def release_paged_cache_blocks_under_pressure(self):
        self.evictions += 1
        return 0


def _req(rid, n):
    return SimpleNamespace(request_id=rid, num_output_tokens=n)


def _fills(fake, request):
    return Scheduler._generation_fills_memory(fake, request)


def test_the_longest_generation_stops_when_memory_is_full_after_eviction():
    short, long_ = _req("a", 10), _req("b", 9_000)
    fake = _Fake(100, [100, 100], [short, long_])
    assert _fills(fake, long_)
    assert fake.evictions == 2, "the caches are asked to give memory back first"


def test_only_the_longest_generation_is_stopped():
    short, long_ = _req("a", 10), _req("b", 9_000)
    fake = _Fake(100, [120], [short, long_])
    assert not _fills(fake, short)
    assert fake.evictions == 0


def test_no_stop_below_the_cap_or_when_eviction_frees_enough_or_with_no_cap():
    long_ = _req("b", 9_000)
    assert not _fills(_Fake(100, [99], [long_]), long_)
    assert not _fills(_Fake(100, [100, 60], [long_]), long_), "eviction freed it"
    assert not _fills(_Fake(0, [10**12], [long_]), long_), "no admission cap configured"
