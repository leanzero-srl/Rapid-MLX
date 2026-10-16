# SPDX-License-Identifier: Apache-2.0
"""An abort of a REQUEUED, waiting request plans its boundary handoff as the
copy it is (RC9 review, 2026-10-16).

``_memguard_requeue`` keeps the row's KV as a ``prompt[:held]`` entry and the
request re-fetches it as a loan (a deepcopy shares the entry's buffers), and
the boundary state survives the requeue. The abort branch for a waiting
request handed it over with ``solo=True`` -- planned as a cut nobody else
references -- although the source buffers stay alive in the held entry: on the
RC8 1/1000 fixture the footprint went 43.99 -> 52.83 MB over a 50.10 ceiling,
with two copies of the session (about +8.8 GB at full scale)."""

from __future__ import annotations

import copy

from tests.test_lz_singleton_srpf import _no_metal, cpu  # noqa: F401
from tests.test_memory_guard_boundary_handoff import (
    CEILING,
    KV_LAYERS,
    MB,
    RC8_BOUNDARY,
    _kv_rows,
    _rc8,
    _rc8_boundary,
    _recurrent,
)


def test_an_aborted_requeued_request_stays_under_the_ceiling(cpu):  # noqa: F811
    s, main = _rc8()
    _rc8_boundary(s)
    assert main._memguard_boundary_state is not None
    del s.running["e91"]
    del s.request_id_to_uid["e91"]
    del s.uid_to_request_id[52]
    held = RC8_BOUNDARY + 5
    mac = s.memory_aware_cache
    assert mac.store(
        list(range(held)),
        [_recurrent()] + [_kv_rows(held) for _ in range(KV_LAYERS)],
        evict_prefixes=False,
    )
    main.prompt_cache = copy.deepcopy(mac._entries[tuple(range(held))].cache)
    main._srpf_parked = False
    s.waiting.append(main)
    s._current_metal_active_bytes = lambda: int(33.0 * MB) + sum(
        int(e.memory_bytes) for e in mac._entries.values()
    )
    s._do_abort_request_impl("e91")
    assert s._metal_footprint_bytes() <= CEILING
