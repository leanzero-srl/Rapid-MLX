# SPDX-License-Identifier: Apache-2.0
"""Vendored Mellum (``mellum``) registration and a tiny forward pass.

The full-model parity proof against transformers' ``MellumForCausalLM`` runs
when torch + transformers>=5.8 are importable (``RAPID_MLX_MELLUM_REFERENCE=1``);
it is skipped elsewhere.
"""

import os
import sys

import pytest

mx = pytest.importorskip("mlx.core")


def _tiny_args():
    from rapid_mlx.models.mellum import ModelArgs

    return ModelArgs(
        model_type="mellum",
        hidden_size=64,
        num_hidden_layers=4,
        intermediate_size=128,
        num_attention_heads=4,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        rms_norm_eps=1e-6,
        vocab_size=97,
        num_key_value_heads=2,
        head_dim=16,
        tie_word_embeddings=False,
        max_position_embeddings=4096,
        norm_topk_prob=True,
        sliding_window=8,
        layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        rope_parameters={
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": 500000.0,
                "factor": 16.0,
                "original_max_position_embeddings": 256,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "attention_factor": 1.2772588722239782,
            },
            "sliding_attention": {"rope_type": "default", "rope_theta": 500000.0},
        },
    )


def test_register_installs_mellum_for_mlx_lm_lookup():
    from rapid_mlx.utils.tokenizer import _register_vendored_archs

    _register_vendored_archs()
    import importlib

    mod = importlib.import_module("mlx_lm.models.mellum")
    assert hasattr(mod, "Model") and hasattr(mod, "ModelArgs")


def test_tiny_forward_and_cached_decode_match():
    from rapid_mlx.models.mellum import Model

    mx.random.seed(0)
    model = Model(_tiny_args())
    tokens = mx.array([[1, 5, 9, 2, 7, 3, 4, 8, 6, 11, 12, 13]])
    full = model(tokens)
    assert full.shape == (1, 12, 97)
    cache = model.make_cache()
    assert type(cache[0]).__name__ == "RotatingKVCache"
    assert type(cache[3]).__name__ == "KVCache"
    step = model(tokens[:, :10], cache=cache)
    for i in range(10, 12):
        step = model(tokens[:, i : i + 1], cache=cache)
        diff = mx.abs(step[0, -1] - full[0, i]).max().item()
        assert diff < 1e-3, (i, diff)


@pytest.mark.skipif(
    not os.environ.get("RAPID_MLX_MELLUM_REFERENCE"),
    reason="needs torch + transformers with MellumForCausalLM",
)
def test_parity_with_transformers_reference():
    import numpy as np
    import torch
    from transformers import MellumConfig, MellumForCausalLM

    from rapid_mlx.models.mellum import Model

    a = _tiny_args()
    cfg = MellumConfig(
        hidden_size=a.hidden_size,
        num_hidden_layers=a.num_hidden_layers,
        intermediate_size=a.intermediate_size,
        num_attention_heads=a.num_attention_heads,
        num_experts=a.num_experts,
        num_experts_per_tok=a.num_experts_per_tok,
        moe_intermediate_size=a.moe_intermediate_size,
        rms_norm_eps=a.rms_norm_eps,
        vocab_size=a.vocab_size,
        num_key_value_heads=a.num_key_value_heads,
        head_dim=a.head_dim,
        tie_word_embeddings=False,
        max_position_embeddings=a.max_position_embeddings,
        norm_topk_prob=True,
        sliding_window=a.sliding_window,
        layer_types=a.layer_types,
        rope_parameters=a.rope_parameters,
        mlp_layer_types=["sparse"] * a.num_hidden_layers,
    )
    torch.manual_seed(0)
    ref = MellumForCausalLM(cfg).float().eval()
    # transformers 5 holds the experts fused in memory (``experts.gate_up_proj``
    # [E, 2I, H], ``experts.down_proj`` [E, H, I]); the published checkpoint
    # stores them per expert, which ``sanitize`` stacks. Split the fused copy
    # into that per-expert layout so the test runs the same sanitize path.
    sd = {}
    inter = a.moe_intermediate_size
    for k, v in ref.state_dict().items():
        v = v.detach().numpy()
        if k.endswith("mlp.experts.gate_up_proj"):
            pre = k[: -len("gate_up_proj")]
            for e in range(a.num_experts):
                sd[f"{pre}{e}.gate_proj.weight"] = mx.array(v[e, :inter])
                sd[f"{pre}{e}.up_proj.weight"] = mx.array(v[e, inter:])
        elif k.endswith("mlp.experts.down_proj"):
            pre = k[: -len("down_proj")]
            for e in range(a.num_experts):
                sd[f"{pre}{e}.down_proj.weight"] = mx.array(v[e])
        else:
            sd[k] = mx.array(v)
    model = Model(a)
    sd = model.sanitize(sd)
    model.load_weights(list(sd.items()), strict=True)
    ids = list(range(1, 40))
    with torch.no_grad():
        r = ref(torch.tensor([ids])).logits[0].numpy()
    m = np.array(model(mx.array([ids]))[0].astype(mx.float32))
    print(f"mellum parity: max|dlogit|={np.abs(r - m).max():.3e} scale={np.abs(r).max():.3f} top1={(r.argmax(-1) == m.argmax(-1)).sum()}/{len(ids)}")
    assert np.abs(r - m).max() < 1e-3
    assert (r.argmax(-1) == m.argmax(-1)).all()
