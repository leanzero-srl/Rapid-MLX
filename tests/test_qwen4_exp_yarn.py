"""Opt-in YaRN rope scaling for qwen4_exp (Qwen3.8-Flash-Next past 262k).

The golden frequencies and attention factor were produced by Hugging Face
transformers 5.15.1 ``ROPE_INIT_FUNCTIONS["yarn"]`` (torch, CPU) on the model
card's recipe: rope_type yarn, rope_theta 1e7, partial_rotary_factor 0.25,
head_dim 256, factor 4, original_max_position_embeddings 262144.
"""

import json
from dataclasses import asdict

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
pytestmark = pytest.mark.requires_mlx

import rapid_mlx.models.qwen4_exp as qwen4_exp
from rapid_mlx.models.qwen4_exp import (
    Model,
    ModelArgs,
    QSAAttention,
    TextModelArgs,
    apply_qwen4_exp_rope,
)
from rapid_mlx.models.qwen4_exp_rope import (
    ROPE_SCALING_ENV,
    parse_rope_scaling_override,
    resolve_rope_scaling,
    yarn_inverse_frequency,
)

CARD = {
    "mrope_interleaved": True,
    "mrope_section": [11, 11, 10],
    "rope_type": "yarn",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
    "factor": 4.0,
    "original_max_position_embeddings": 262144,
}
HF_ATTENTION_FACTOR = 1.138629436111989
HF_INV_FREQ = [
    1.0,
    0.6042963862419128,
    0.36517414450645447,
    0.22067339718341827,
    0.1333521455526352,
    0.08058421313762665,
    0.04869675263762474,
    0.02942727319896221,
    0.017782794311642647,
    0.010746078565716743,
    0.006493816617876291,
    0.003924189601093531,
    0.0023713738191872835,
    0.0014330126577988267,
    0.0008659643353894353,
    0.00047423981595784426,
    0.00025693507632240653,
    0.00013734975073020905,
    7.217386882985011e-05,
    3.707224823301658e-05,
    1.844922371674329e-05,
    8.759770025790203e-06,
    3.849816039291909e-06,
    2.3264301489689387e-06,
    1.405853367941745e-06,
    8.495520660289912e-07,
    5.133812237545499e-07,
    3.102344408034696e-07,
    1.8747356023141037e-07,
    1.1328959459433463e-07,
    6.846049416253663e-08,
    4.1370427084075345e-08,
]


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    # setenv first so teardown restores the pre-test state even when a test
    # (the CLI one) writes os.environ directly.
    monkeypatch.setenv(ROPE_SCALING_ENV, "")
    monkeypatch.delenv(ROPE_SCALING_ENV)


def _args(**overrides):
    values = {
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "vocab_size": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 3,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 3,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 4,
        "shared_expert_intermediate_size": 4,
        "hc_count": 4,
        "hc_lowrank": 3,
        "layer_types": ["linear_attention", "full_attention"],
        "indexer_n_heads": 2,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 4,
        "indexer_budget": 8,
        "indexer_compress_ratio": 2,
        "ple_layer_ids": [],
        "eos_token_id": 31,
    }
    values.update(overrides)
    return TextModelArgs(**values)


def _reference(x, positions, inverse_frequency, attention_factor, rotary_dim):
    """float64 HF rotate-half rope with cos/sin scaled by attention_factor."""
    angles = positions[:, None].astype(np.float64) * np.asarray(inverse_frequency)
    emb = np.concatenate([angles, angles], axis=-1)
    rot = x[..., :rotary_dim].astype(np.float64)
    half = rotary_dim // 2
    rotated_half = np.concatenate([-rot[..., half:], rot[..., :half]], axis=-1)
    out = (rot * np.cos(emb) + rotated_half * np.sin(emb)) * attention_factor
    return np.concatenate([out, x[..., rotary_dim:]], axis=-1)


def test_default_checkpoint_rope_stays_unscaled():
    args = _args()
    assert args.rope_scaling is None
    assert args.max_position_embeddings == 262144
    attention = QSAAttention(args)
    assert attention.rope_yarn is None and attention.indexer.rope_yarn is None
    # The resolved scaling is a plain attribute, never a loadable weight.
    assert not any("yarn" in name for name, _ in _flat(attention.parameters()))


def _flat(tree):
    from mlx.utils import tree_flatten

    return tree_flatten(tree)


def test_card_recipe_matches_hugging_face_yarn():
    scaling = resolve_rope_scaling(CARD, None, 262144)
    assert scaling.factor == 4.0
    assert scaling.original_max_position_embeddings == 262144
    assert scaling.max_position_embeddings == 1048576
    assert scaling.attention_factor == pytest.approx(HF_ATTENTION_FACTOR, abs=1e-15)
    inverse = np.asarray(yarn_inverse_frequency(64, 1e7, scaling))
    np.testing.assert_allclose(inverse, HF_INV_FREQ, rtol=1e-6, atol=0)


def test_text_args_parse_the_card_and_extend_the_window():
    args = _args(head_dim=256, indexer_head_dim=256, rope_parameters=dict(CARD))
    assert args.rope_scaling["rope_type"] == "yarn"
    assert args.max_position_embeddings == 1048576
    # The MTP head rebuilds its args from vars(trunk args): idempotent.
    params = dict(vars(args))
    params.update(num_hidden_layers=1, layer_types=["full_attention"])
    again = TextModelArgs.from_dict(params)
    assert again.rope_scaling == args.rope_scaling
    assert again.max_position_embeddings == 1048576


@pytest.mark.parametrize("metal", [True, False])
@pytest.mark.parametrize("ndim", [2, 3])
def test_scaled_rope_matches_reference(monkeypatch, metal, ndim):
    if metal and not mx.metal.is_available():
        pytest.skip("no Metal")
    if not metal:
        qwen4_exp._qwen4_exp_rope_kernel.cache_clear()
        monkeypatch.setattr(qwen4_exp.mx.metal, "is_available", lambda: False)
    scaling = resolve_rope_scaling(CARD, None, 262144)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 2, 4, 256)).astype(np.float32)
    positions = np.array([0, 1, 37, 4097], dtype=np.int64)
    pos = mx.array(positions)[None]
    if ndim == 3:
        pos = mx.broadcast_to(pos[None], (3, 1, 4))
    out = apply_qwen4_exp_rope(
        mx.array(x),
        pos,
        rotary_dim=64,
        base=1e7,
        mrope_section=(11, 11, 10),
        rope_scaling=scaling,
    )
    expected = _reference(x, positions, HF_INV_FREQ, HF_ATTENTION_FACTOR, 64)
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-4, atol=1e-3)
    if not metal:
        monkeypatch.undo()
        qwen4_exp._qwen4_exp_rope_kernel.cache_clear()


def test_unscaled_path_builds_the_original_kernel(monkeypatch):
    calls = []
    real = qwen4_exp._qwen4_exp_rope_kernel

    def spy(*key):
        calls.append(key)
        return real(*key)

    monkeypatch.setattr(qwen4_exp, "_qwen4_exp_rope_kernel", spy)
    x = mx.ones((1, 2, 3, 256))
    apply_qwen4_exp_rope(x, mx.arange(3)[None], rotary_dim=64, base=1e7)
    scaling = resolve_rope_scaling(CARD, None, 262144)
    apply_qwen4_exp_rope(
        x, mx.arange(3)[None], rotary_dim=64, base=1e7, rope_scaling=scaling
    )
    assert calls == [(64, 2), (64, 2, scaling.attention_factor)]


def test_operator_override_forms(monkeypatch):
    assert parse_rope_scaling_override(None) is None
    assert parse_rope_scaling_override("") is None
    assert parse_rope_scaling_override("off") == {}
    assert parse_rope_scaling_override("yarn") == {"rope_type": "yarn"}
    assert parse_rope_scaling_override("yarn:2") == {"rope_type": "yarn", "factor": 2.0}
    with pytest.raises(ValueError):
        parse_rope_scaling_override("linear:2")
    with pytest.raises(ValueError):
        parse_rope_scaling_override("[1]")

    monkeypatch.setenv(ROPE_SCALING_ENV, "yarn")
    args = _args()
    assert args.rope_scaling["factor"] == 4.0
    assert args.rope_scaling["original_max_position_embeddings"] == 262144
    assert args.max_position_embeddings == 1048576
    attention = QSAAttention(args)
    assert attention.rope_yarn == attention.indexer.rope_yarn
    assert attention.rope_yarn.attention_factor == pytest.approx(HF_ATTENTION_FACTOR)

    monkeypatch.setenv(ROPE_SCALING_ENV, "yarn:2")
    assert _args().max_position_embeddings == 524288

    monkeypatch.setenv(
        ROPE_SCALING_ENV, json.dumps({"rope_type": "yarn", "factor": 2.0})
    )
    assert _args().rope_scaling["factor"] == 2.0

    monkeypatch.setenv(ROPE_SCALING_ENV, "off")
    disabled = _args(head_dim=256, indexer_head_dim=256, rope_parameters=dict(CARD))
    assert disabled.rope_scaling is None
    assert disabled.max_position_embeddings == 262144


def test_unsupported_rope_types_fail_loudly():
    with pytest.raises(ValueError):
        resolve_rope_scaling({"rope_type": "linear", "factor": 2.0}, None, 262144)
    with pytest.raises(ValueError):
        resolve_rope_scaling({**CARD, "truncate": False}, None, 262144)


def test_model_advertises_the_extended_window_only_when_scaled(monkeypatch):
    plain = Model(
        ModelArgs.from_dict({"model_type": "qwen4_exp", "text_config": asdict(_args())})
    )
    assert not hasattr(plain.args, "max_position_embeddings")
    monkeypatch.setenv(ROPE_SCALING_ENV, "yarn")
    scaled = Model(
        ModelArgs.from_dict({"model_type": "qwen4_exp", "text_config": asdict(_args())})
    )
    assert scaled.args.max_position_embeddings == 1048576


def test_cli_flag_sets_the_env(monkeypatch):
    import argparse

    from rapid_mlx import cli

    parser = argparse.ArgumentParser()
    cli._add_qwen4_rope_args(parser)
    args = parser.parse_args(["--qwen4-rope-scaling", "yarn:4"])
    cli._apply_qwen4_rope_env(args)
    import os

    assert os.environ[ROPE_SCALING_ENV] == "yarn:4"
    bad = parser.parse_args(["--qwen4-rope-scaling", "dynamic"])
    with pytest.raises(SystemExit):
        cli._apply_qwen4_rope_env(bad)
    unset = parser.parse_args([])
    monkeypatch.delenv(ROPE_SCALING_ENV)
    cli._apply_qwen4_rope_env(unset)
    assert ROPE_SCALING_ENV not in os.environ
