# SPDX-License-Identifier: Apache-2.0
"""The memory guard sizes a latent-attention (MLA) model's KV from what its cache
really stores (RUN-ISSUES I-7, 2026-10-18).

mlx-lm's absorbed MLA (deepseek_v3 and the vendored xing4_0) caches ONE head of
``kv_lora_rank`` latent + ``qk_rope_head_dim`` rope values per layer and token. The
uniform ``2 x kv_heads x head_dim`` formula over-stated Xing4.0-29B-A4B 12.4x and capped
a 96 GB Studio at 35,328 tokens, so goose could not compact the Forge task under it.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx

from rapid_mlx.kv_estimation import latent_attention_bytes_per_layer
from rapid_mlx.scheduler import Scheduler, SchedulerConfig

from tests.test_xing4_0_model import TINY, _tiny_model


def _measured_bytes_per_token(model, tokens: int = 37) -> float:
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    model(mx.array([list(range(1, tokens + 1))]), cache=cache)
    mx.eval([c.state for c in cache])
    total = 0
    for c in cache:
        keys, values = c.keys, c.values
        total += keys[..., : c.offset, :].nbytes + values[..., : c.offset, :].nbytes
    return total / tokens


def _scheduler(model):
    return Scheduler(
        model=model,
        tokenizer=SimpleNamespace(encode=list, decode=lambda ids, **_: ""),
        config=SchedulerConfig(enable_prefix_cache=False, spec_decode="none"),
    )


def test_xing_latent_estimate_matches_the_real_cache():
    _m, model = _tiny_model()
    itemsize = next(
        v.dtype.size for _k, v in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(model.parameters())
        if mx.issubdtype(v.dtype, mx.floating)
    )
    est = latent_attention_bytes_per_layer(model, itemsize)
    assert est is not None
    layers, per_layer = est
    assert layers == TINY["num_hidden_layers"]
    assert per_layer == (TINY["kv_lora_rank"] + TINY["qk_rope_head_dim"]) * itemsize
    measured = _measured_bytes_per_token(model)
    assert abs(layers * per_layer - measured) <= 0.05 * measured


def test_scheduler_and_guard_prior_use_the_latent_size():
    _m, model = _tiny_model()
    sched = _scheduler(model)
    per_tok, fixed, sliding, _window = sched._kv_footprint_terms()
    measured = _measured_bytes_per_token(model)
    assert abs(per_tok - measured) <= 0.05 * measured  # was 2 x 4 heads x 16 x 3 layers x 4 B = 1,536
    assert fixed == 0 and sliding == 0
    prior = sched._memguard_prior()
    assert prior.kv_copy_bytes_per_context == per_tok / TINY["num_hidden_layers"]


def test_full_kv_mla_and_stub_models_keep_the_uniform_estimate():
    # deepseek_v2-style: kv_lora_rank but per-head K/V cached (no embed_q absorption).
    v2_layer = SimpleNamespace(self_attn=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64))
    assert latent_attention_bytes_per_layer(SimpleNamespace(model=SimpleNamespace(layers=[v2_layer] * 4)), 2) is None
    # a mixed stack (one non-latent layer) is not latent
    latent = SimpleNamespace(self_attn=SimpleNamespace(kv_lora_rank=16, qk_rope_head_dim=8, embed_q=object()))
    dense = SimpleNamespace(self_attn=SimpleNamespace(n_heads=4))
    assert latent_attention_bytes_per_layer(SimpleNamespace(layers=[latent, dense]), 2) is None
    # MagicMock / no layers: unchanged accounting
    assert latent_attention_bytes_per_layer(MagicMock(), 2) is None
    assert latent_attention_bytes_per_layer(SimpleNamespace(), 2) is None
