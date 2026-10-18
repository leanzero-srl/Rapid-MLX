# SPDX-License-Identifier: Apache-2.0
"""Tests for the vendored Xing4.0 (``xing4_0``) port (mlx-lm PR #1901).

Pins the contract that:

1. The vendored module registers into mlx-lm's importlib lookup and marks
   ``xing4_0`` for the tokenizer fallback (``_register_vendored_archs``).
2. Layers below ``first_k_dense_replace`` are dense, the rest MoE with the
   float32 router (``Fp32RouterGate``).
3. A tiny config runs: incremental (KV cache) decode reproduces the
   full-prefill logits.
4. The router really multiplies in float32 (bf16 activations, fp32 logits).
5. The hyper-connection matches the reference torch code (copied verbatim
   from modeling_xing4_0.py, as PR #1901's own test does) in fp32 and bf16,
   and its mixing matrix is doubly stochastic.
6. ``sanitize`` drops the MTP layer and stacks per-expert tensors.
7. The vendored tokenizer loads a checkpoint folder that has only
   ``tokenizer.model`` (no remote code) and renders the chat template.
8. FULL-MODEL parity against the checkpoint's own ``modeling_xing4_0.py``
   (random tiny config, fp32, HF state dict -> ``sanitize`` -> strict
   ``load_weights``): runs only when ``RAPID_MLX_XING_REFERENCE`` points at a
   folder holding the reference ``modeling_xing4_0.py`` +
   ``configuration_xing4_0.py`` (needs torch + transformers). Measured
   2026-10-18 on the workhorse (mlx 0.32.4, mlx-lm 0.31.3, transformers
   5.15.1, torch 2.14.1), 96 random tokens: max |logit diff| 1.49e-06 at a
   logit scale of 1.56, top-1 agreement 96/96 (12 tokens: 2.91e-06).
9. The engine loader (``load_model_with_fallback``) serves a tiny on-disk
   ``xing4_0`` checkpoint through the vendored model and tokenizer while
   planted checkpoint code (``tokenization_xing4_0.py`` & co.) raises.
"""

import importlib
import json
import os
import sys

import mlx.core as mx
import numpy as np
import pytest

from rapid_mlx.utils import tokenizer as tok_mod

TINY = dict(
    model_type="xing4_0",
    vocab_size=97,
    hidden_size=64,
    intermediate_size=96,
    moe_intermediate_size=32,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=4,
    n_shared_experts=1,
    n_routed_experts=8,
    routed_scaling_factor=2.0,
    kv_lora_rank=16,
    q_lora_rank=24,
    qk_rope_head_dim=8,
    v_head_dim=16,
    qk_nope_head_dim=16,
    topk_method="noaux_tc",
    scoring_func="sigmoid",
    norm_topk_prob=True,
    n_group=1,
    topk_group=1,
    num_experts_per_tok=2,
    moe_layer_freq=1,
    first_k_dense_replace=1,
    max_position_embeddings=4096,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    rope_scaling={
        "type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 1024,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    },
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    mhc_h_res_clamp_min=-30.0,
    mhc_h_res_clamp_max=30.0,
)


def _mod():
    tok_mod._register_vendored_archs()
    return importlib.import_module("mlx_lm.models.xing4_0")


def _tiny_model(seed=0):
    m = _mod()
    mx.random.seed(seed)
    model = m.Model(m.ModelArgs.from_dict(TINY))
    # Random, non-trivial hyper-connection and router parameters (zeros/ones
    # by default would hide mixing and routing mistakes).
    params = {}
    for name, value in _flat(model.parameters()):
        if name.endswith(("hc_fn", "hc_base")):
            params[name] = mx.random.normal(value.shape) * 0.3
        elif name.endswith("hc_scale"):
            params[name] = mx.array([0.7, 1.3, 2.1])
        elif name.endswith("e_score_correction_bias"):
            params[name] = mx.random.normal(value.shape) * 0.1
    model.load_weights(list(params.items()), strict=False)
    mx.eval(model.parameters())
    return m, model


def _flat(tree, prefix=""):
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _flat(v, f"{prefix}{k}.")
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            yield from _flat(v, f"{prefix}{i}.")
    else:
        yield prefix[:-1], tree


def test_registers_into_mlx_lm():
    m = _mod()
    assert sys.modules["mlx_lm.models.xing4_0"] is m
    assert "xing4_0" in tok_mod._VENDORED_MODEL_TYPES


def test_dense_then_moe_layers_with_fp32_router():
    m, model = _tiny_model()
    from mlx_lm.models.deepseek_v3 import DeepseekV3MLP, DeepseekV3MoE

    assert isinstance(model.layers[0].mlp, DeepseekV3MLP)
    for layer in model.layers[1:]:
        assert isinstance(layer.mlp, DeepseekV3MoE)
        assert isinstance(layer.mlp.gate, m.Fp32RouterGate)


def test_incremental_decode_matches_full_prefill():
    from mlx_lm.models.cache import make_prompt_cache

    with mx.stream(mx.cpu):
        _, model = _tiny_model()
        tokens = mx.array([[5, 17, 3, 42, 8, 1, 60, 33]])
        full = model(tokens)
        cache = make_prompt_cache(model)
        step = [model(tokens[:, :3], cache=cache)]
        for i in range(3, tokens.shape[1]):
            step.append(model(tokens[:, i : i + 1], cache=cache))
        inc = mx.concatenate(step, axis=1)
        mx.eval(full, inc)
    err = np.abs(np.array(full) - np.array(inc)).max()
    assert err < 1e-4, err


def test_router_multiplies_in_float32():
    m, model = _tiny_model()
    gate = model.layers[1].mlp.gate
    x = (mx.random.normal((1, 5, TINY["hidden_size"])) * 3).astype(mx.bfloat16)
    with mx.stream(mx.cpu):
        inds, scores = gate(x)
        logits = (
            np.array(x.astype(mx.float32)) @ np.array(gate.weight.astype(mx.float32)).T
        )
        mx.eval(inds, scores)
    sig = 1 / (1 + np.exp(-logits))
    pick = np.argsort(-(sig + np.array(gate.e_score_correction_bias)), axis=-1)[..., :2]
    assert np.array_equal(np.sort(np.array(inds), -1), np.sort(pick, -1))
    assert scores.dtype == mx.float32


def test_sanitize_drops_mtp_and_stacks_experts():
    m, model = _tiny_model()
    h, e, i = (
        TINY["hidden_size"],
        TINY["n_routed_experts"],
        TINY["moe_intermediate_size"],
    )
    w = {}
    for x in range(e):
        for proj, shape in (
            ("gate_proj", (i, h)),
            ("up_proj", (i, h)),
            ("down_proj", (h, i)),
        ):
            w[f"model.layers.1.mlp.experts.{x}.{proj}.weight"] = mx.zeros(shape)
    w["model.layers.3.eh_proj.weight"] = mx.zeros((h, 2 * h))  # the MTP layer
    w["model.layers.3.self_attn.q_a_proj.weight"] = mx.zeros((24, h))
    out = model.sanitize(w)
    assert not any(k.startswith("model.layers.3.") for k in out)
    assert out["model.layers.1.mlp.switch_mlp.gate_proj.weight"].shape == (e, i, h)
    assert out["model.layers.1.mlp.switch_mlp.down_proj.weight"].shape == (e, h, i)


def test_hyper_connection_matches_reference():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as fnn
    from torch import nn

    m = _mod()

    # Verbatim from XingChen-AGI/Xing4.0-29B-A4B modeling_xing4_0.py.
    class RefUnweightedRMSNorm(nn.Module):
        def __init__(self, eps: float = 1.0e-6):
            super().__init__()
            self.eps = eps

        def forward(self, x):
            return (
                x.float()
                * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
            ).to(x.dtype)

    class RefHyperConnection(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.hc_mult = config.hc_mult
            self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
            self.hc_eps = config.hc_eps
            self.input_norm = RefUnweightedRMSNorm(eps=config.rms_norm_eps)
            mix = (2 + self.hc_mult) * self.hc_mult
            self.hc_fn = nn.Parameter(
                torch.empty(mix, self.hc_mult * config.hidden_size)
            )
            self.hc_base = nn.Parameter(torch.empty(mix))
            self.hc_scale = nn.Parameter(torch.empty(3))
            self.mhc_h_res_clamp_max = config.mhc_h_res_clamp_max
            self.mhc_h_res_clamp_min = config.mhc_h_res_clamp_min

        def forward(self, hidden_streams):
            ori_dtype = hidden_streams.dtype
            hc = self.hc_mult
            flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
            pre_w, post_w, comb_w = (
                fnn.linear(flat.to(ori_dtype), self.hc_fn.to(ori_dtype))
                .float()
                .split([hc, hc, hc * hc], dim=-1)
            )
            pre_b, post_b, comb_b = self.hc_base.split([hc, hc, hc * hc])
            pre_scale, post_scale, comb_scale = self.hc_scale.unbind(0)
            pre = torch.sigmoid(pre_w * pre_scale + pre_b)
            post = 2 * torch.sigmoid(post_w * post_scale + post_b)
            comb_logits = comb_w.view(
                *comb_w.shape[:-1], hc, hc
            ) * comb_scale + comb_b.view(hc, hc)
            comb_logits = torch.clamp(
                comb_logits, min=self.mhc_h_res_clamp_min, max=self.mhc_h_res_clamp_max
            )
            comb_max = comb_logits.amax(dim=-1, keepdim=True)
            comb = torch.exp(comb_logits - comb_max)
            for _ in range(self.hc_sinkhorn_iters):
                comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
                comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
            collapsed = (pre.unsqueeze(-1).to(ori_dtype) * hidden_streams).sum(dim=2)
            return post.to(ori_dtype), comb.to(ori_dtype), collapsed.to(ori_dtype)

    args = m.ModelArgs.from_dict(TINY)
    for dtype, mx_dtype, tol in (
        (torch.float32, mx.float32, 1e-5),
        (torch.bfloat16, mx.bfloat16, 2e-2),
    ):
        torch.manual_seed(0)
        ref = RefHyperConnection(args)
        with torch.no_grad():
            ref.hc_fn.normal_(0, 0.05)
            ref.hc_base.normal_(0, 0.5)
            ref.hc_scale.copy_(torch.tensor([0.7, 1.3, 2.1]))
        x = torch.randn(2, 5, args.hc_mult, args.hidden_size)
        with torch.no_grad():
            want = ref(x.to(dtype))
        hc = m.HyperConnection(args)
        hc.load_weights(
            [
                (n, mx.array(getattr(ref, n).detach().numpy()))
                for n in ("hc_fn", "hc_base", "hc_scale")
            ]
        )
        with mx.stream(mx.cpu):
            got = hc(mx.array(x.numpy()).astype(mx_dtype))
            mx.eval(got)
        for name, g, w_ in zip(("post", "comb", "collapsed"), got, want):
            g = np.array(g.astype(mx.float32))
            w_ = w_.float().numpy()
            rel = np.abs(g - w_).max() / max(np.abs(w_).max(), 1e-6)
            assert rel < tol, (str(dtype), name, rel)
    _, comb, _ = got
    comb = np.array(comb.astype(mx.float32))
    assert (
        np.abs(comb.sum(-1) - 1).max() < 5e-2 and np.abs(comb.sum(-2) - 1).max() < 5e-2
    )


def test_vendored_tokenizer_loads_a_tokenizer_model_only_folder(tmp_path):
    spm = pytest.importorskip("sentencepiece")
    specials = [
        "<_start>",
        "<_end>",
        "<_pad>",
        "<_user>",
        "<_bot>",
        "<_system>",
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
    ]
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "\n".join(
            ["hello world, the quick brown fox jumps", "def f(x): return x + 1"] * 50
        )
    )
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(tmp_path / "tokenizer"),
        vocab_size=60,
        hard_vocab_limit=False,
        user_defined_symbols=specials,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "xing4_0"}))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "Xing4_0Tokenizer",
                "auto_map": {
                    "AutoTokenizer": ["tokenization_xing4_0.Xing4_0Tokenizer", None]
                },
                "additional_special_tokens": specials,
                "add_bos_token": False,
                "add_eos_token": False,
                "bos_token": "<_start>",
                "eos_token": "<_end>",
                "pad_token": "<_pad>",
                "unk_token": "<unk>",
            }
        )
    )
    (tmp_path / "chat_template.jinja").write_text(
        "{%- for m in messages %}<_user>{{ m.content }}{%- endfor %}"
        "{%- if add_generation_prompt %}<_bot>{{ '</think>' if (enable_thinking is defined and not "
        "enable_thinking) else '<think>\\n' }}{%- endif %}"
    )
    # A planted remote-code file must never be executed.
    (tmp_path / "tokenization_xing4_0.py").write_text(
        "raise RuntimeError('remote code ran')\n"
    )

    assert tok_mod._xing4_0_sentencepiece_checkpoint(tmp_path)
    from rapid_mlx.models.xing4_0_tokenizer import (
        Xing4_0Tokenizer,
        load_xing4_0_tokenizer,
    )

    tok = load_xing4_0_tokenizer(tmp_path)
    assert isinstance(tok, Xing4_0Tokenizer)
    ids = tok.encode("<_user>hello world<tool_call>")
    assert tok.convert_ids_to_tokens(ids[0]) == "<_user>"
    assert tok.convert_ids_to_tokens(ids[-1]) == "<tool_call>"
    assert tok.decode(ids) == "<_user>hello world<tool_call>"
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True
    )
    assert rendered == "<_user>hi<_bot><think>\n"
    assert tok.eos_token == "<_end>"


def test_loader_serves_a_tiny_checkpoint_without_running_its_code(tmp_path):
    """End to end through the engine loader: a ``xing4_0`` folder with only
    ``tokenizer.model`` (and planted checkpoint code that raises) loads the
    vendored model + tokenizer and runs a forward pass."""
    spm = pytest.importorskip("sentencepiece")
    specials = [
        "<_start>",
        "<_end>",
        "<_pad>",
        "<_user>",
        "<_bot>",
        "<_system>",
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
    ]
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(["hello world, the quick brown fox jumps"] * 60))
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(tmp_path / "tokenizer"),
        vocab_size=TINY["vocab_size"],
        hard_vocab_limit=False,
        user_defined_symbols=specials,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
    )
    _, model = _tiny_model()
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(_flat(model.parameters()))
    )
    config = dict(
        TINY,
        architectures=["Xing4_0ForCausalLM"],
        auto_map={"AutoConfig": "configuration_xing4_0.Xing4_0Config"},
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "Xing4_0Tokenizer",
                "auto_map": {
                    "AutoTokenizer": ["tokenization_xing4_0.Xing4_0Tokenizer", None]
                },
                "additional_special_tokens": specials,
                "add_bos_token": False,
                "bos_token": "<_start>",
                "eos_token": "<_end>",
                "pad_token": "<_pad>",
            }
        )
    )
    (tmp_path / "chat_template.jinja").write_text(
        "{%- for m in messages %}<_user>{{ m.content }}{%- endfor %}<_bot><think>\n"
    )
    for f in (
        "tokenization_xing4_0.py",
        "configuration_xing4_0.py",
        "modeling_xing4_0.py",
    ):
        (tmp_path / f).write_text("raise RuntimeError('checkpoint code ran')\n")

    loaded_model, tokenizer = tok_mod.load_model_with_fallback(str(tmp_path))
    from rapid_mlx.models.xing4_0_tokenizer import Xing4_0Tokenizer

    assert type(loaded_model).__module__ in (
        "mlx_lm.models.xing4_0",
        "rapid_mlx.models.xing4_0",
    )
    base = getattr(tokenizer, "_tokenizer", tokenizer)
    assert isinstance(base, Xing4_0Tokenizer)
    ids = base.encode("<_user>hello world")
    with mx.stream(mx.cpu):
        logits = loaded_model(mx.array([ids]))
        mx.eval(logits)
    assert logits.shape == (1, len(ids), TINY["vocab_size"])


@pytest.mark.skipif(
    not os.environ.get("RAPID_MLX_XING_REFERENCE"),
    reason="out-of-band reference parity",
)
def test_full_model_parity(tmp_path):
    """Random tiny config, fp32, reference transformers model vs this port.

    Measured 2026-10-18 (workhorse): see the assertion tolerances; the
    observed max |logit diff| is printed with ``-s``.
    """
    torch = pytest.importorskip("torch")
    import shutil

    # The reference uses a package-relative import of its configuration.
    ref_dir = os.environ["RAPID_MLX_XING_REFERENCE"]
    pkg = tmp_path / "xing_ref"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for f in ("modeling_xing4_0.py", "configuration_xing4_0.py"):
        shutil.copy(os.path.join(ref_dir, f), pkg / f)
    sys.path.insert(0, str(tmp_path))
    try:
        rm = importlib.import_module("xing_ref.modeling_xing4_0")
        Xing4_0Config = importlib.import_module(
            "xing_ref.configuration_xing4_0"
        ).Xing4_0Config
    finally:
        sys.path.remove(str(tmp_path))

    cfg_kwargs = {k: v for k, v in TINY.items() if k != "model_type"}
    hf_cfg = Xing4_0Config(**cfg_kwargs)
    hf_cfg._attn_implementation = "eager"
    # The reference's _init_weights reads ``module.fn`` (the attribute is
    # ``hc_fn``) and raises on random init; every parameter is set below.
    rm.Xing4_0PreTrainedModel._init_weights = lambda self, module: None
    torch.manual_seed(0)
    ref = rm.Xing4_0ForCausalLM(hf_cfg).float().eval()
    with torch.no_grad():
        for name, p in ref.named_parameters():
            p.copy_(torch.randn_like(p) * (0.3 if "hc_" in name else 0.05))
            if name.endswith("hc_scale"):
                p.copy_(torch.tensor([0.7, 1.3, 2.1]))
            if "layernorm" in name or name.endswith("norm.weight"):
                p.copy_(1 + torch.randn_like(p) * 0.05)
        for name, b in ref.named_buffers():
            if name.endswith("e_score_correction_bias"):
                b.copy_(torch.randn_like(b) * 0.1)
    state = {
        k: mx.array(v.detach().float().numpy()) for k, v in ref.state_dict().items()
    }

    m = _mod()
    model = m.Model(m.ModelArgs.from_dict(TINY))
    weights = model.sanitize(dict(state))
    model.load_weights(list(weights.items()), strict=True)

    tokens = np.random.default_rng(0).integers(0, TINY["vocab_size"], (1, 96)).tolist()
    with torch.no_grad():
        want = ref(torch.tensor(tokens)).logits.float().numpy()
    with mx.stream(mx.cpu):
        got = model(mx.array(tokens))
        mx.eval(got)
    got = np.array(got)
    diff = np.abs(got - want).max()
    print(
        f"xing4_0 full-model parity: max|dlogit|={diff:.3e} scale={np.abs(want).max():.3e} "
        f"top1 agree={(got.argmax(-1) == want.argmax(-1)).mean():.3f}"
    )
    assert diff / np.abs(want).max() < 1e-4
    assert (got.argmax(-1) == want.argmax(-1)).all()
