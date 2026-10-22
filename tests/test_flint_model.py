# SPDX-License-Identifier: Apache-2.0
"""Recursive Flint (``rapid_mlx/models/flint.py``) against an independent
reference written from the checkpoint's own description (``config.json``,
``flint_sglang/configuration_flint.py`` + ``model.py`` 0.1.4):

* sliding layers attend to the current token and the ``sliding_window - 1``
  before it, with RoPE (neox halves); layers in ``no_rope_layer_indices`` have
  no positional encoding at all;
* every head has a learned sink logit in the softmax denominator (no value);
* q/k RMSNorm per head before RoPE; GQA;
* MoE: sigmoid scores (fp32), top-k on score + optional selection-only bias,
  weights = chosen scores renormalised (+1e-20), plus an ungated shared
  expert; dense SwiGLU on ``mlp_only_layers``.

The reference is plain numpy in float64 over an HF-layout state dict (one
tensor per expert), so the test also covers ``sanitize`` (expert stacking,
bias aliasing) and a strict load. Everything runs on the CPU.
"""

import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from rapid_mlx.models import flint  # noqa: E402

LAYER_TYPES = [
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "full_attention",
]

CONFIG = {
    "model_type": "flint",
    "hidden_size": 64,
    "num_hidden_layers": len(LAYER_TYPES),
    "intermediate_size": 96,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "num_experts": 8,
    "num_experts_per_tok": 3,
    "moe_intermediate_size": 24,
    "shared_expert_intermediate_size": 32,
    "mlp_only_layers": [0],
    "layer_types": LAYER_TYPES,
    "no_rope_layer_indices": [
        i for i, t in enumerate(LAYER_TYPES) if t == "full_attention"
    ],
    "sliding_window": 5,
    "use_sliding_window": True,
    "rms_norm_eps": 1e-6,
    "vocab_size": 101,
    "rope_theta": 10000.0,
    "norm_topk_prob": True,
    "router_score_function": "sigmoid",
    "softmax_type": "learnable",
    "hidden_act": "silu",
    "tie_word_embeddings": False,
}


def _hf_state(cfg, seed=0, expert_bias=False):
    rng = np.random.default_rng(seed)
    H, V = cfg["hidden_size"], cfg["vocab_size"]
    nh, nkv, hd = (
        cfg["num_attention_heads"],
        cfg["num_key_value_heads"],
        cfg["head_dim"],
    )
    E = cfg["num_experts"]

    def w(*shape, scale=0.2):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def norm(n):
        return (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)

    s = {
        "model.embed_tokens.weight": w(V, H, scale=1.0),
        "model.norm.weight": norm(H),
        "lm_head.weight": w(V, H),
    }
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}"
        s[f"{p}.input_layernorm.weight"] = norm(H)
        s[f"{p}.post_attention_layernorm.weight"] = norm(H)
        s[f"{p}.self_attn.q_proj.weight"] = w(nh * hd, H)
        s[f"{p}.self_attn.k_proj.weight"] = w(nkv * hd, H)
        s[f"{p}.self_attn.v_proj.weight"] = w(nkv * hd, H)
        s[f"{p}.self_attn.o_proj.weight"] = w(H, nh * hd)
        s[f"{p}.self_attn.q_norm.weight"] = norm(hd)
        s[f"{p}.self_attn.k_norm.weight"] = norm(hd)
        # Large sinks so a sink error cannot hide under the tolerance.
        s[f"{p}.self_attn.softmax_offset"] = (rng.standard_normal(nh) * 2.0).astype(
            np.float32
        )
        if i in cfg["mlp_only_layers"]:
            inter = cfg["intermediate_size"]
            s[f"{p}.mlp.gate_proj.weight"] = w(inter, H)
            s[f"{p}.mlp.up_proj.weight"] = w(inter, H)
            s[f"{p}.mlp.down_proj.weight"] = w(H, inter)
        else:
            inter = cfg["moe_intermediate_size"]
            S = cfg["shared_expert_intermediate_size"]
            s[f"{p}.mlp.gate.weight"] = w(E, H, scale=0.5)
            if expert_bias:
                s[f"{p}.mlp.gate.expert_bias"] = (rng.standard_normal(E) * 0.3).astype(
                    np.float32
                )
            for e in range(E):
                s[f"{p}.mlp.experts.{e}.gate_proj.weight"] = w(inter, H)
                s[f"{p}.mlp.experts.{e}.up_proj.weight"] = w(inter, H)
                s[f"{p}.mlp.experts.{e}.down_proj.weight"] = w(H, inter)
            s[f"{p}.mlp.shared_expert.gate_proj.weight"] = w(S, H)
            s[f"{p}.mlp.shared_expert.up_proj.weight"] = w(S, H)
            s[f"{p}.mlp.shared_expert.down_proj.weight"] = w(H, S)
    return s


# ---------------------------------------------------------------- reference


def _rms(x, wt, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * wt


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _rope(x, pos, theta):
    # x: (heads, T, D); neox rotation (first half / second half).
    D = x.shape[-1]
    inv = theta ** (-np.arange(0, D, 2, dtype=np.float64) / D)
    ang = pos[:, None] * inv[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., : D // 2], x[..., D // 2 :]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def reference_logits(state, cfg, tokens):
    st = {k: v.astype(np.float64) for k, v in state.items()}
    eps = cfg["rms_norm_eps"]
    nh, nkv, hd = (
        cfg["num_attention_heads"],
        cfg["num_key_value_heads"],
        cfg["head_dim"],
    )
    T = len(tokens)
    pos = np.arange(T, dtype=np.float64)
    x = st["model.embed_tokens.weight"][np.array(tokens)]
    for i, kind in enumerate(cfg["layer_types"]):
        p = f"model.layers.{i}"
        h = _rms(x, st[f"{p}.input_layernorm.weight"], eps)
        q = (
            (h @ st[f"{p}.self_attn.q_proj.weight"].T)
            .reshape(T, nh, hd)
            .transpose(1, 0, 2)
        )
        k = (
            (h @ st[f"{p}.self_attn.k_proj.weight"].T)
            .reshape(T, nkv, hd)
            .transpose(1, 0, 2)
        )
        v = (
            (h @ st[f"{p}.self_attn.v_proj.weight"].T)
            .reshape(T, nkv, hd)
            .transpose(1, 0, 2)
        )
        q = _rms(q, st[f"{p}.self_attn.q_norm.weight"], eps)
        k = _rms(k, st[f"{p}.self_attn.k_norm.weight"], eps)
        if i not in cfg["no_rope_layer_indices"]:
            q = _rope(q, pos, cfg["rope_theta"])
            k = _rope(k, pos, cfg["rope_theta"])
        sink = st[f"{p}.self_attn.softmax_offset"]
        out = np.zeros((nh, T, hd))
        for head in range(nh):
            kv = head // (nh // nkv)
            for t in range(T):
                lo = (
                    0
                    if kind == "full_attention"
                    else max(0, t - cfg["sliding_window"] + 1)
                )
                sc = q[head, t] @ k[kv, lo : t + 1].T * hd**-0.5
                m = max(sc.max(), sink[head])
                e = np.exp(sc - m)
                den = e.sum() + np.exp(sink[head] - m)
                out[head, t] = (e / den) @ v[kv, lo : t + 1]
        attn = (
            out.transpose(1, 0, 2).reshape(T, nh * hd)
            @ st[f"{p}.self_attn.o_proj.weight"].T
        )
        x = x + attn
        h = _rms(x, st[f"{p}.post_attention_layernorm.weight"], eps)
        if i in cfg["mlp_only_layers"]:
            y = (
                _silu(h @ st[f"{p}.mlp.gate_proj.weight"].T)
                * (h @ st[f"{p}.mlp.up_proj.weight"].T)
            ) @ st[f"{p}.mlp.down_proj.weight"].T
        else:
            scores = 1.0 / (1.0 + np.exp(-(h @ st[f"{p}.mlp.gate.weight"].T)))
            bias = st.get(f"{p}.mlp.gate.expert_bias", np.zeros(cfg["num_experts"]))
            y = np.zeros_like(h)
            for t in range(T):
                top = np.argsort(-(scores[t] + bias))[: cfg["num_experts_per_tok"]]
                wts = scores[t, top] / (scores[t, top].sum() + 1e-20)
                for wt, e in zip(wts, top):
                    ep = f"{p}.mlp.experts.{e}"
                    y[t] += wt * (
                        (
                            _silu(h[t] @ st[f"{ep}.gate_proj.weight"].T)
                            * (h[t] @ st[f"{ep}.up_proj.weight"].T)
                        )
                        @ st[f"{ep}.down_proj.weight"].T
                    )
            sp = f"{p}.mlp.shared_expert"
            y += (
                _silu(h @ st[f"{sp}.gate_proj.weight"].T)
                * (h @ st[f"{sp}.up_proj.weight"].T)
            ) @ st[f"{sp}.down_proj.weight"].T
        x = x + y
    x = _rms(x, st["model.norm.weight"], eps)
    return x @ st["lm_head.weight"].T


# ---------------------------------------------------------------- model side


def _model(state, cfg=CONFIG):
    model = flint.Model(flint.ModelArgs.from_dict(cfg))
    weights = model.sanitize({k: mx.array(v) for k, v in state.items()})
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model


TOKENS = [
    int(t) for t in np.random.default_rng(7).integers(0, CONFIG["vocab_size"], 40)
]


@pytest.mark.parametrize("expert_bias", [False, True])
def test_full_sequence_matches_the_reference(expert_bias):
    state = _hf_state(CONFIG, seed=1, expert_bias=expert_bias)
    ref = reference_logits(state, CONFIG, TOKENS)
    got = np.array(_model(state)(mx.array([TOKENS]))[0].astype(mx.float32))
    scale = np.abs(ref).max()
    err = np.abs(got - ref).max()
    assert err < 3e-5 * max(1.0, scale), (err, scale)
    assert (got.argmax(-1) == ref.argmax(-1)).all()


def test_the_reference_is_sensitive_to_each_rule():
    """Negative controls: each architectural rule moves the logits, so the
    parity above cannot pass by accident."""
    state = _hf_state(CONFIG, seed=1)
    base = reference_logits(state, CONFIG, TOKENS)
    variants = {
        "no sinks": dict(
            state,
            **{
                k: np.full_like(v, -1e9)
                for k, v in state.items()
                if k.endswith("softmax_offset")
            },
        ),
    }
    for name, st in variants.items():
        assert np.abs(reference_logits(st, CONFIG, TOKENS) - base).max() > 1e-3, name
    for name, cfg in {
        "rope on every layer": dict(CONFIG, no_rope_layer_indices=[]),
        "window excludes the current token": dict(CONFIG, sliding_window=4),
        "no sliding": dict(CONFIG, layer_types=["full_attention"] * len(LAYER_TYPES)),
    }.items():
        assert np.abs(reference_logits(state, cfg, TOKENS) - base).max() > 1e-3, name
        got = np.array(_model(state, cfg)(mx.array([TOKENS]))[0])
        assert np.abs(got - base).max() > 1e-3, name


def _decode_logits(model, tokens, chunks):
    cache = model.make_cache()
    out = []
    pos = 0
    for n in chunks:
        piece = tokens[pos : pos + n]
        logits = model(mx.array([piece]), cache=cache)
        mx.eval(logits)
        out.append(np.array(logits[0]))
        pos += n
    assert pos == len(tokens)
    return np.concatenate(out, axis=0)


@pytest.mark.parametrize(
    "chunks",
    [
        [1] * 40,  # token by token, far past the window
        [12] + [1] * 28,  # prefill longer than the window, then decode
        [3, 7, 2, 11, 1, 1, 9, 6],  # chunk sizes across window boundaries
    ],
)
def test_cached_prefill_and_decode_match_the_reference(chunks):
    state = _hf_state(CONFIG, seed=2)
    ref = reference_logits(state, CONFIG, TOKENS)
    got = _decode_logits(_model(state), TOKENS, chunks)
    assert np.abs(got - ref).max() < 3e-5 * max(1.0, np.abs(ref).max())


def test_cache_layout_and_kv_attributes():
    model = _model(_hf_state(CONFIG, seed=3))
    caches = model.make_cache()
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    for kind, c, layer in zip(LAYER_TYPES, caches, model.layers):
        if kind == "sliding_attention":
            assert (
                isinstance(c, RotatingKVCache)
                and c.max_size == CONFIG["sliding_window"]
            )
            assert layer.self_attn.is_sliding is True
        else:
            assert type(c) is KVCache
            assert layer.self_attn.is_sliding is False
        assert layer.self_attn.n_kv_heads == CONFIG["num_key_value_heads"]
        assert layer.self_attn.head_dim == CONFIG["head_dim"]
    from rapid_mlx.kv_estimation import attention_module_kv_bytes

    growth, slot = attention_module_kv_bytes(model, 2)
    per_layer = 2 * CONFIG["num_key_value_heads"] * CONFIG["head_dim"] * 2
    assert growth == per_layer * LAYER_TYPES.count("full_attention")
    assert slot == per_layer * LAYER_TYPES.count("sliding_attention")


def test_the_real_config_prices_only_full_attention_as_growing():
    """RecursiveSI/flint: 14 full-attention layers x 2 x 4 KV heads x 128 x
    bf16 = 28 KiB per token (the checkpoint README's own figure); the 35
    sliding layers are window-bounded."""
    from rapid_mlx.kv_estimation import estimate_kv_footprint

    real = json.loads(
        (Path(__file__).parent / "fixtures" / "flint" / "config.json").read_text()
    )
    per_layer = 2 * 4 * 128 * 2
    est = estimate_kv_footprint(
        real,
        dtype_bytes=2,
        uniform_per_token_bytes=49 * per_layer,
        base_num_layers=49,
        base_kv_heads=4,
        base_head_dim=128,
    )
    assert est.per_token_growth_bytes == 28 * 1024 == 14 * per_layer
    assert est.fixed_baseline_bytes == 0
    assert est.sliding_window == 513
    assert est.sliding_slot_bytes == 35 * per_layer


# ---------------------------------------------------------------- quantized


def _write_safetensors(path, tensors):
    mx.save_safetensors(str(path), tensors, metadata={"format": "mlx"})


def test_recursive_int4_layout_loads(tmp_path):
    """Recursive's MLX INT4 export: uint8-packed weights, an 8-bit embedding
    the config does not list, an unquantized router. Loads through the
    engine's override and matches the same quantized model built in memory."""
    from mlx_lm.utils import load_model

    from rapid_mlx.utils.tokenizer import (
        _flint_quantization_override,
        _register_vendored_archs,
    )

    _register_vendored_archs()
    qcfg = dict(CONFIG, moe_intermediate_size=32)  # every input dim % 32 == 0
    state = _hf_state(qcfg, seed=4)
    model = _model(state, qcfg)
    import mlx.nn as nn

    def predicate(path, module):
        if path.endswith("mlp.gate"):
            return False
        if path == "model.embed_tokens":
            return {"group_size": 32, "bits": 8}
        return hasattr(module, "to_quantized")

    nn.quantize(model, group_size=32, bits=4, class_predicate=predicate)
    mx.eval(model.parameters())
    flat = dict(nn.utils.tree_flatten(model.parameters()))
    on_disk = {
        k: (v.view(mx.uint8) if v.dtype == mx.uint32 else v) for k, v in flat.items()
    }
    assert on_disk["model.embed_tokens.weight"].dtype == mx.uint8
    _write_safetensors(tmp_path / "model-00000.safetensors", on_disk)
    cfg = dict(qcfg, quantization={"bits": 4, "group_size": 32})
    (tmp_path / "config.json").write_text(json.dumps(cfg))

    overlay = _flint_quantization_override(tmp_path)
    assert overlay["quantization"]["model.embed_tokens"] == {
        "group_size": 32,
        "bits": 8,
    }
    assert "lm_head" not in overlay["quantization"]  # 4-bit = the default

    loaded, _ = load_model(tmp_path, model_config=overlay)
    want = np.array(model(mx.array([TOKENS]))[0])
    got = np.array(loaded(mx.array([TOKENS]))[0])
    assert np.array_equal(got, want)


def test_override_leaves_other_models_alone(tmp_path):
    from rapid_mlx.utils.tokenizer import _flint_quantization_override

    (tmp_path / "config.json").write_text(
        json.dumps(
            {"model_type": "qwen3", "quantization": {"bits": 4, "group_size": 64}}
        )
    )
    header = json.dumps(
        {
            "model.embed_tokens.weight": {
                "dtype": "U8",
                "shape": [10, 64],
                "data_offsets": [0, 640],
            },
            "model.embed_tokens.scales": {
                "dtype": "BF16",
                "shape": [10, 1],
                "data_offsets": [640, 660],
            },
        }
    ).encode()
    (tmp_path / "m.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\0" * 660
    )
    assert _flint_quantization_override(tmp_path) is None


def test_registered_as_the_mlx_lm_flint_module():
    import sys

    from rapid_mlx.utils.tokenizer import (
        _RAPID_OWNED_RUNTIME_MODEL_TYPES,
        _VENDORED_MODEL_TYPES,
        _register_vendored_archs,
    )

    _register_vendored_archs()
    assert sys.modules["mlx_lm.models.flint"] is flint
    assert "flint" in _VENDORED_MODEL_TYPES
    assert "flint" in _RAPID_OWNED_RUNTIME_MODEL_TYPES


def test_the_trained_context_bounds_the_served_window():
    """Without ``max_position_embeddings`` in ModelArgs the engine knew no
    native window: /v1/models said context_window null and the guard
    published 768,512 on a 128 GB Mac (live, 2026-10-22)."""
    from types import SimpleNamespace

    from rapid_mlx.service.helpers import get_model_max_context

    real = json.loads(
        (Path(__file__).parent / "fixtures" / "flint" / "config.json").read_text()
    )
    args = flint.ModelArgs.from_dict(real)
    assert args.max_position_embeddings == 262144
    model = SimpleNamespace(args=args)
    assert get_model_max_context(SimpleNamespace(_model=model, model=model)) == 262144
