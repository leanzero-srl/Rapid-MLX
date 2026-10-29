# SPDX-License-Identifier: Apache-2.0
"""A conversation whose turn end cannot rebuild its boundary keeps its
resume point by the mid-turn COPY (workmule Flash-Next, 2026-10-29).

Jundot/Qwen3.8-Flash-Next-oQ4e-mtp (qwen4_exp: hybrid layers that are not
plain ``ArraysCache``), app argv, an idle 128 GB Studio, the real Gauntlet
session replayed and grown. From 213k tokens on every boundary copy was
skipped:

    skipped storing a 6.41GB prefix entry: resident 92.30GB + entry +
    margin 1.94GB > ceiling 96.91GB

96.91 GB is the MACHINE ceiling (physical - others - 8 % reserve; the
kernel/OS alone wired ~25 GB), the device ceiling is 103.90 GB. The copy was
treated as optional because "the turn end hands the boundary over" -- but
for this layout ``_memguard_keep_boundary_state`` returns without keeping
anything, so the turn end stored nothing either. The hit stayed at 200,785
tokens and every later turn re-prefilled 14-24k tokens (TTFT 88-162 s).

And where the device ceiling itself is tight, the store-room ladder refused
to evict the conversation's own older resume point -- the very entry the new
boundary supersedes -- and skipped the store instead.

At 1/1000 of the bytes, on the RC8 fixture of the hand-over tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache

import rapid_mlx.memory_budget as memory_budget
from tests.test_memory_guard_boundary_handoff import (
    CEILING,
    KV_LAYERS,
    MB,
    RC8_BOUNDARY,
    RC8_FETCHED,
    RC8_PROMPT,
    _kv_rows,
    _rc8,
)


class _HybridState(ArraysCache):
    """A non-trimmable layer of a type the turn end cannot rebuild (as
    qwen4_exp's): not ``ArraysCache`` itself."""


def _hybrid() -> _HybridState:
    layer = _HybridState(size=2)
    layer[0] = mx.zeros((1, 18_750), mx.float32)
    layer[1] = mx.zeros((1, 18_750), mx.float32)
    mx.eval(layer[0], layer[1])
    return layer


def _boundary(s):
    snapshot = [_hybrid()] + [_kv_rows(RC8_BOUNDARY) for _ in range(KV_LAYERS)]
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


def test_layout_is_not_keepable():
    s, _main = _rc8(resident=39.5)
    assert not s._memguard_boundary_keepable([_hybrid(), _kv_rows(8)])
    plain = ArraysCache(size=2)
    plain[0] = mx.zeros((1, 4), mx.float32)
    assert s._memguard_boundary_keepable([plain, _kv_rows(8)])


def test_copy_is_stored_under_the_device_ceiling_when_the_machine_ceiling_is_lower():
    """REGRESSION (RED before the fix): the machine leaves the engine 48 MB,
    the device ceiling is 50.10 MB, main's 8.83 MB boundary copy fits beside
    39.5 MB resident only under the device ceiling. Its turn end cannot hand
    the boundary over (hybrid layout), so the copy must be made."""
    s, main = _rc8(resident=39.5)
    mac = s.memory_aware_cache
    physical = 100 * MB
    others = physical - int(memory_budget.SYSTEM_RESERVE_RATIO * physical) - 48 * MB
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(memory_budget, "system_physical_memory", lambda: physical)
        monkey.setattr(
            memory_budget,
            "system_memory_used",
            lambda: memory_budget.MachineMemory(
                others + s._current_metal_active_bytes(), 0
            ),
        )
        s._memguard_others_read = None
        assert s._memguard_cache_ceiling() == pytest.approx(48 * MB, abs=MB // 100)
        assert s._memguard_cache_ceiling() < CEILING
        _boundary(s)
    finally:
        monkey.undo()
    assert tuple(range(RC8_BOUNDARY)) in mac._entries
    assert s._memory_guard.counts["store_skips"] == 0


def test_older_own_resume_point_is_evicted_before_the_store_is_skipped():
    """REGRESSION (RED before the fix): the device ceiling is tight; main's
    own older 111,513-token entry (its hit, its resume point until now) must
    give way to the 120,588-token boundary that supersedes it."""
    s, main = _rc8(resident=39.5)
    mac = s.memory_aware_cache
    old_key = list(range(RC8_FETCHED))
    old = [_hybrid()] + [_kv_rows(RC8_FETCHED) for _ in range(KV_LAYERS)]
    assert mac.store(
        old_key,
        old,
        evict_prefixes=False,
        message_boundary=True,
        **s._memguard_owner(main),
    )
    resident = s._current_metal_active_bytes()
    copy = 8.83 * MB
    margin = 0.02 * CEILING
    assert resident + copy + margin > CEILING  # does not fit beside the old one
    _boundary(s)
    assert tuple(range(RC8_BOUNDARY)) in mac._entries
    assert tuple(old_key) not in mac._entries
    assert s._memory_guard.counts["store_skips"] == 0
