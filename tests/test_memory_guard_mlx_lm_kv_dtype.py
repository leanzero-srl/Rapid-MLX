# SPDX-License-Identifier: Apache-2.0
"""The memory guard prices an mlx-lm model's KV in the dtype it really caches.

Live defect, 2026-10-19 (Mac Studio M3 Ultra 96 GB, goose app, engine
7108304a, ``lmstudio-community/granite-4.2-30b-MLX-8bit``; RUN-ISSUES I-29):
both Granite 4.2 30B Forge builds died at the guard's window, 69,888 of the
model's 131,072 tokens ("This prompt (70280 tokens) needs 75.4 GB but this
Mac can hold 75.1 GB"), while the engine measured 256 KB of KV per token
(``cache_store_bytes per_token=256.0KB``) and 49.5 GB resident at ~73k.

Cause: an mlx-lm model exposes its dims on ``.args`` (no ``.config``), so the
scheduler's config path resolves 0 and ``_kv_footprint_terms`` falls back to
reading the dims off the model -- but priced them with the config dtype probe,
and mlx-lm's ``ModelArgs`` carries no ``dtype`` key: fp32 fallback, 524,288
B/token, every mlx-lm model's KV doubled in the guard (the boot line's
``token=786432 B`` = 524,288 + the activation prior). The I-17 fix had moved
every OTHER KV term to the loaded parameters' itemsize; this one was missed.

Each test fails on 7108304a.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

FIXTURE = Path(__file__).parent / "fixtures" / "granite-4.2-30b"
# 64 layers x 2 (K, V) x 8 KV heads x 128 x bf16.
GRANITE_KV_PER_TOKEN = 64 * 2 * 8 * 128 * 2


def _stub_scheduler(model):
    from rapid_mlx.scheduler import Scheduler, SchedulerConfig

    s = Scheduler.__new__(Scheduler)
    s.model = model
    s.config = SchedulerConfig()
    s._kv_bytes_per_token_resolved = False
    s._kv_bytes_per_token = 0
    s._kv_fixed_baseline_bytes = 0
    s._kv_sliding_slot_bytes = 0
    s._kv_sliding_window = 0
    s._kv_quant_live_disabled = True
    return s


def _granite(**overrides):
    from mlx_lm.models import granite

    config = json.loads((FIXTURE / "config.json").read_text()) | overrides
    model = granite.Model(granite.ModelArgs.from_dict(config))
    model.set_dtype(mx.bfloat16)  # the converted checkpoint's activation dtype
    return model


def test_granite_42_30b_kv_is_priced_in_bf16():
    s = _stub_scheduler(_granite())  # the real config, weights never materialised
    assert getattr(s.model, "config", None) is None  # the mlx-lm fallback path
    per_tok, fixed, slot, window = s._kv_footprint_terms()
    assert per_tok == GRANITE_KV_PER_TOKEN  # 7108304a: 524,288
    assert (fixed, slot, window) == (0, 0, 0)
    assert s._memguard_prior().kv_bytes_per_token == GRANITE_KV_PER_TOKEN


def test_granite_42_30b_window_on_the_studio_is_near_native():
    """The live numbers of the refused run: idle base 29.65 GB, static
    ceiling 75.1 GB. The window was 69,888 (73,216 at boot)."""
    from rapid_mlx.memory_guard import MemoryGuard, kv_growth_step

    s = _stub_scheduler(_granite())
    guard = MemoryGuard(s._memguard_prior(), floor_tokens=kv_growth_step())
    kv_fn = s._kv_footprint_fn()
    window = guard.max_context(
        base=int(29.65e9), kv_footprint=kv_fn, ceiling=int(75.1e9), upper=131_072
    )
    assert window > 120_000
    # The prompt the coordinator re-checks live, and the refused 82,082 one.
    for prompt in (82_082, 90_000):
        need = guard.single_sequence_need(
            prompt, base=int(29.65e9), kv_footprint=kv_fn, ceiling=int(75.1e9)
        )
        assert need < 75.1e9


def test_the_estimate_equals_the_cache_a_real_forward_holds():
    """Random-weight forward of a 2-layer Granite (the real attention shape:
    8 KV heads x 128): the guard's per-token figure is the KV it holds."""
    from mlx_lm.models.cache import KVCache, make_prompt_cache

    mx.random.seed(0)
    model = _granite(num_hidden_layers=2, intermediate_size=256, vocab_size=512)
    mx.eval(model.parameters())
    cache = make_prompt_cache(model)
    tokens = 21
    logits = model(mx.arange(tokens).reshape(1, tokens), cache=cache)
    mx.eval(logits, [c.state for c in cache])
    measured = sum(
        c.keys[..., : c.offset, :].nbytes + c.values[..., : c.offset, :].nbytes
        for c in cache
        if isinstance(c, KVCache)
    ) // tokens
    per_tok, *_ = _stub_scheduler(model)._kv_footprint_terms()
    assert per_tok == measured == 2 * 2 * 8 * 128 * 2
