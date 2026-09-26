"""LeanZero Q-110: a prefix-cache entry holds exactly the bytes it is charged for.

mlx-lm's ``extract_cache`` hands one row of a live batch over as a view; a
view keeps every row of the batch buffer alive while the byte ledger charges
one row. Measured on the Studio (27B Q8, decode batched): 28.8 GB held beyond
the idle engine against 6.95 GB the cache reported, then out-of-memory on
every request.
"""

import gc

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx  # noqa: E402
from mlx_lm.models.cache import ArraysCache, KVCache  # noqa: E402

from rapid_mlx.hybrid_state_checkpoints import (  # noqa: E402
    StateCheckpoints,
    record_checkpoints,
)
from rapid_mlx.memory_cache import (  # noqa: E402
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
    estimate_kv_cache_memory,
)

ROWS = 5
MB = 1 << 20


def _settle() -> int:
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    return mx.get_active_memory()


def _row_views_of_a_batch():
    """One row of a 5-row batch, the way extract_cache returns it."""
    keys = mx.random.normal((ROWS, 4, 1024, 128))
    values = mx.random.normal((ROWS, 4, 1024, 128))
    conv = mx.random.normal((ROWS, 3, 1024))
    ssm = mx.random.normal((ROWS, 8, 128, 128))
    mx.eval(keys, values, conv, ssm)
    kv = KVCache()
    kv.keys, kv.values, kv.offset = keys[1:2], values[1:2], 1024
    rec = ArraysCache(2)
    rec.cache = [conv[1:2], ssm[1:2]]
    return [kv, rec]


def test_a_stored_entry_holds_only_its_own_rows():
    cache = MemoryAwarePrefixCache(
        model=object(),
        config=MemoryCacheConfig(
            max_memory_mb=1024, max_entries=10, hybrid_reuse_max_entries=4
        ),
    )
    before = _settle()
    entry = _row_views_of_a_batch()
    charged = estimate_kv_cache_memory(entry)
    assert cache.store(list(range(1024)), entry)
    del entry
    held = _settle() - before
    # The batch buffer is five rows; the entry is charged one. It must hold one.
    assert held <= charged * 1.01, (held // MB, charged // MB)
    assert len(cache) == 1
    assert cache.get_stats()["current_memory_bytes"] == charged


def test_a_recurrent_checkpoint_holds_only_its_own_row():
    before = _settle()
    layers = _row_views_of_a_batch()
    holders = [None, None]
    assert record_checkpoints(layers, holders, 2048, max_count=4, stride=1)
    del layers
    held = _settle() - before
    charged = holders[1].nbytes
    assert isinstance(holders[1], StateCheckpoints)
    assert held <= charged * 1.01, (held // MB, charged // MB)
