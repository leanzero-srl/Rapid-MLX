# SPDX-License-Identifier: MIT
# Copyright © 2026 Apple Inc. (mlx-lm, MIT); vendored by LeanZero for Rapid-MLX.
"""
Vendored XingChen-AGI Xing4.0 (``xing4_0``) model class from mlx-lm PR #1901.

**Why vendor:** stock mlx-lm (0.31.3 pinned here, 0.32.2 latest) has no
``xing4_0`` module, so ``Xing4.0-29B-A4B`` (31.2B total / ~4B active, 64
routed experts + 1 shared, 4 per token, Apache-2.0) cannot load at all:
``ValueError: Model type xing4_0 not supported``. ml-explore/mlx-lm PR #1901
(``Add xing4_0 model support``, open since 2026-09-19) adds it as a thin layer
over mlx-lm's own DeepSeek-V3 MLA attention + noaux_tc MoE, plus the model's
manifold-constrained hyper-connections (``hc_mult`` = 4 residual streams mixed
by a Sinkhorn-normalised matrix around every attention and FFN sublayer).

**Upstream:** https://github.com/ml-explore/mlx-lm/pull/1901
**Vendored from:** TokenAI-zer/mlx-lm@26f5528e73124fbc4f9874ad77bf3aac5d52d815
**File:** the PR's ``mlx_lm/models/xing4_0.py`` (sha256 2e0d1d17a7724a91…)
verbatim between the ``BEGIN/END PR #1901`` markers, with its two
package-relative imports (``.base``, ``.deepseek_v3``) rewritten to absolute
``mlx_lm.models.*`` paths (both exist in the pinned ``mlx-lm>=0.31.3``).
**Reference implementation it ports:**
https://huggingface.co/XingChen-AGI/Xing4.0-29B-A4B/blob/main/modeling_xing4_0.py
(Apache-2.0). Numerical parity against that file: see
``tests/test_xing4_0_model.py`` (hyper-connection, CI-safe) and the
out-of-band full-model numbers recorded there.

**LeanZero addition (after END):** ``Model`` routes in float32. The
reference computes the router logits in float32
(``F.linear(hidden.float(), weight.float())``); mlx-lm's DeepSeek-V3
``MoEGate`` multiplies in the activation dtype (bf16), which flips near-tie
expert choices. Same parameters, so every conversion loads unchanged.

**Registration:** ``rapid_mlx.utils.tokenizer._register_vendored_archs``
installs this module as ``sys.modules["mlx_lm.models.xing4_0"]`` unless
mlx-lm ships its own. The tokenizer (a SentencePiece ``tokenizer.model``
behind the checkpoint's remote-code ``Xing4_0Tokenizer``; no
``tokenizer.json``) is loaded by ``rapid_mlx.models.xing4_0_tokenizer``, so
no checkpoint code runs.

**Sync policy:** when mlx-lm ships native ``xing4_0`` (PR #1901 merged),
diff the PR section against the merged file, then delete this file
(preferred) or cherry-pick fixes.
"""
# ---- BEGIN PR #1901 (verbatim but for the two imports) ----

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.deepseek_v3 import DeepseekV3Attention, DeepseekV3MLP, DeepseekV3MoE
from mlx_lm.models.deepseek_v3 import Model as DeepseekV3LM
from mlx_lm.models.deepseek_v3 import ModelArgs as DeepseekV3Args


@dataclass
class ModelArgs(DeepseekV3Args):
    model_type: str = "xing4_0"
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    mhc_h_res_clamp_min: float = -30.0
    mhc_h_res_clamp_max: float = 30.0


class HyperConnection(nn.Module):
    """Mixes the ``hc_mult`` parallel residual streams around a sublayer.

    Returns the per-stream output gate, the doubly stochastic stream mixing
    matrix, and the single stream the sublayer actually runs on.
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        self.clamp_min = config.mhc_h_res_clamp_min
        self.clamp_max = config.mhc_h_res_clamp_max

        mix = (2 + self.hc_mult) * self.hc_mult
        self.hc_fn = mx.zeros((mix, self.hc_mult * config.hidden_size))
        self.hc_base = mx.zeros((mix,))
        self.hc_scale = mx.ones((3,))

    def __call__(self, x: mx.array):
        hc = self.hc_mult
        z = mx.fast.rms_norm(
            mx.flatten(x.astype(mx.float32), -2, -1), None, self.norm_eps
        )
        mixes = (z.astype(x.dtype) @ self.hc_fn.T).astype(mx.float32)
        base = self.hc_base.astype(mx.float32)
        scale = self.hc_scale.astype(mx.float32)

        pre_w, post_w, comb_w = mx.split(mixes, [hc, 2 * hc], axis=-1)
        pre_b, post_b, comb_b = mx.split(base, [hc, 2 * hc])

        pre = mx.sigmoid(pre_w * scale[0] + pre_b)
        post = 2 * mx.sigmoid(post_w * scale[1] + post_b)

        comb = mx.unflatten(comb_w, -1, (hc, hc)) * scale[2] + comb_b.reshape(hc, hc)
        comb = mx.clip(comb, self.clamp_min, self.clamp_max)
        comb = mx.exp(comb - comb.max(axis=-1, keepdims=True))
        for _ in range(self.sinkhorn_iters):
            comb = comb / (comb.sum(axis=-1, keepdims=True) + self.hc_eps)
            comb = comb / (comb.sum(axis=-2, keepdims=True) + self.hc_eps)

        collapsed = (pre[..., None].astype(x.dtype) * x).sum(axis=2)
        return post.astype(x.dtype), comb.astype(x.dtype), collapsed


class Xing4_0DecoderLayer(nn.Module):
    def __init__(self, config: ModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = DeepseekV3Attention(config)
        self.mlp = (
            DeepseekV3MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekV3MLP(config)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        post, comb, collapsed = self.attn_hc(x)
        r = self.self_attn(self.input_layernorm(collapsed), mask, cache)
        x = post[..., None] * r[..., None, :] + comb @ x

        post, comb, collapsed = self.ffn_hc(x)
        r = self.mlp(self.post_attention_layernorm(collapsed))
        return post[..., None] * r[..., None, :] + comb @ x


class Xing4_0Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Xing4_0DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.embed_tokens(x)

        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_attention_mask(h, cache[0], return_array=True)

        # Every layer runs on hc_mult parallel residual streams
        h = mx.contiguous(
            mx.broadcast_to(
                mx.expand_dims(h, -2), (*h.shape[:-1], self.hc_mult, h.shape[-1])
            )
        )

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h.mean(axis=-2))


class Model(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.args = config
        self.model_type = config.model_type
        self.model = Xing4_0Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
    ):
        out = self.model(inputs, cache)
        return self.lm_head(out)

    def sanitize(self, weights):
        # Same expert stacking and MLA absorption as DeepSeek V3
        weights = DeepseekV3LM.sanitize(self, weights)

        # Remove the multi-token prediction layers
        def keep(k):
            if not k.startswith("model.layers."):
                return True
            return int(k.split(".")[2]) < self.args.num_hidden_layers

        return {k: v for k, v in weights.items() if keep(k)}

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        def predicate(k):
            return not k.endswith(("e_score_correction_bias", "hc_scale"))

        return predicate
# ---- END PR #1901 ----

# ---- LeanZero addition: float32 router (see the module docstring) ----
from mlx_lm.models.deepseek_v3 import MoEGate, group_expert_select  # noqa: E402


class Fp32RouterGate(MoEGate):
    """DeepSeek-V3 noaux_tc gate with the router matmul in float32."""

    def __call__(self, x):
        return group_expert_select(
            x.astype(mx.float32) @ self.weight.astype(mx.float32).T,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


_PRModel = Model


class Model(_PRModel):  # noqa: F811
    def __init__(self, config: ModelArgs):
        super().__init__(config)
        for layer in self.model.layers:
            if isinstance(layer.mlp, DeepseekV3MoE):
                layer.mlp.gate = Fp32RouterGate(config)
