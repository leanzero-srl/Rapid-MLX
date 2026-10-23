# Copyright © 2026 Apple Inc.
# SPDX-License-Identifier: MIT
"""Recursive Flint (``model_type: flint``, ``FlintForCausalLM``).

Vendored from ml-explore/mlx-lm PR #1998 "Recursive Flint model support"
(pcuenca/mlx-lm@8531254e, open at vendoring time, 2026-10-22). Stock mlx-lm
(0.31.3 pinned, 0.32.2 latest) has no ``flint`` module, so a Flint checkpoint
could not load at all. ``rapid_mlx.utils.tokenizer._register_vendored_archs``
installs this module as ``mlx_lm.models.flint`` unless mlx-lm ships its own.

Architecture (RecursiveSI/flint config + the checkpoint's SGLang reference,
``flint_sglang/model.py`` 0.1.4): 39.8B total / ~2.5B active decoder-only MoE.

* Attention: GQA (32 q / 4 kv heads, head_dim 128) with q/k RMSNorm and a
  learned per-head attention SINK (``softmax_offset``: one logit added to the
  softmax denominator, no value). Layers mix ``sliding_attention`` (window
  ``sliding_window`` = 513 INCLUDING the current token, RoPE) and
  ``full_attention`` (listed in ``no_rope_layer_indices``: no positional
  encoding at all).
* MLP: dense SwiGLU on ``mlp_only_layers``; elsewhere 256 routed experts,
  top-6 by SIGMOID score, weights renormalised to sum to one, plus an ungated
  shared expert.

Differences from the PR, each matching the checkpoint's reference:

* RoPE follows ``no_rope_layer_indices`` (the PR keys it on the layer type;
  the shipped config makes both the same, a future one need not).
* Routing in float32 with plain ops (the PR's Metal path calls
  ``mlx_lm.kernels.expert_routing``, absent from mlx-lm 0.31.3), the
  reference's ``+ 1e-20`` in the renormalisation, an optional router
  ``expert_bias`` that affects selection only, and the weighted expert sum in
  float32 (mlx-vlm 0.7.10's Flint does the same).
* The output projection runs on float32 hidden states (the reference serves
  with ``--enable-fp32-lm-head``).
* ``sanitize`` also reads Recursive's own MLX INT4 export (RecursiveSI/flint-
  int4): its packed weights are ``uint8``, mlx expects ``uint32`` with the
  same bit layout, so they are viewed, not converted.
* Attention modules expose ``n_kv_heads`` / ``head_dim`` / ``is_sliding`` so
  the engine's KV estimator prices only the full-attention layers as growing
  with context (``kv_estimation.attention_module_kv_bytes``).
"""

from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.switch_layers import SwitchGLU

_SLIDING = "sliding_attention"

# Marks Rapid's copy (``_register_vendored_archs`` installs it over any native
# ``mlx_lm.models.flint``).
_RAPID_FLINT = True


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    mlp_only_layers: list[int] | None = None
    layer_types: list[str] | None = None
    no_rope_layer_indices: list[int] | None = None
    sliding_window: int | None = 513
    use_sliding_window: bool = True
    # Read by the engine's context window (``get_model_max_context``) and the
    # memory guard's native bound (``_memguard_native_context``). Without it
    # a 128 GB Mac advertised a 768,512-token window for a 262,144-token model.
    max_position_embeddings: int = 262144
    rope_theta: float = 10000.0
    rope_parameters: dict[str, Any] | None = None
    norm_topk_prob: bool = True
    router_score_function: str = "sigmoid"
    softmax_type: str = "learnable"
    hidden_act: str = "silu"
    attention_bias: bool = False
    tie_word_embeddings: bool = False

    def __post_init__(self):
        if self.rope_parameters and self.rope_parameters.get("rope_theta") is not None:
            self.rope_theta = self.rope_parameters["rope_theta"]
        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries but "
                f"num_hidden_layers is {self.num_hidden_layers}"
            )
        if set(self.layer_types) - {"full_attention", _SLIDING}:
            raise ValueError(f"unsupported Flint layer types {set(self.layer_types)}")
        self.mlp_only_layers = list(self.mlp_only_layers or [])
        self.no_rope_layer_indices = list(self.no_rope_layer_indices or [])
        if not self.use_sliding_window:
            self.sliding_window = None
        if _SLIDING in self.layer_types and not self.sliding_window:
            raise ValueError("sliding_attention layers need a sliding_window")
        if self.router_score_function != "sigmoid":
            raise ValueError("Flint routes with sigmoid scores")
        if self.softmax_type != "learnable":
            raise ValueError("Flint needs learned attention sinks")
        if self.hidden_act != "silu":
            raise ValueError("Flint uses SiLU (SwiGLU) MLPs")


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.scale = head_dim**-0.5
        self.is_sliding = args.layer_types[layer_idx] == _SLIDING
        bias = args.attention_bias

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=bias)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.softmax_offset = mx.zeros((n_heads,))

        if layer_idx in args.no_rope_layer_indices:
            self.rope = None
        else:
            self.rope = nn.RoPE(head_dim, traditional=False, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if self.rope is not None:
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
            sinks=self.softmax_offset.astype(queries.dtype),
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class SparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob

        self.gate = nn.Linear(dim, args.num_experts, bias=False)
        # Selection-only correction bias; zero when the checkpoint has none
        # (``sanitize`` fills it), as in the reference.
        self.expert_bias = mx.zeros((args.num_experts,))
        self.switch_mlp = SwitchGLU(dim, args.moe_intermediate_size, args.num_experts)
        self.shared_expert = (
            MLP(dim, args.shared_expert_intermediate_size)
            if args.shared_expert_intermediate_size > 0
            else None
        )

    def route(self, x: mx.array):
        # The reference casts the router's output to float32 before the sigmoid
        # (``torch.sigmoid(gating_output.float())``).
        scores = mx.sigmoid(self.gate(x).astype(mx.float32))
        choice = scores + self.expert_bias.astype(mx.float32)
        k = self.top_k
        inds = mx.stop_gradient(mx.argpartition(-choice, kth=k - 1, axis=-1)[..., :k])
        weights = mx.take_along_axis(scores, inds, axis=-1)
        if self.norm_topk_prob:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        return inds, weights

    def __call__(self, x: mx.array) -> mx.array:
        inds, weights = self.route(x)
        y = self.switch_mlp(x, inds)
        y = (y.astype(mx.float32) * weights[..., None]).sum(axis=-2).astype(x.dtype)
        if self.shared_expert is not None:
            y = y + self.shared_expert(x)
        return y


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.use_sliding = args.layer_types[layer_idx] == _SLIDING
        self.self_attn = Attention(args, layer_idx)
        if layer_idx in args.mlp_only_layers:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)
        else:
            self.mlp = SparseMoeBlock(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class FlintModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        types = args.layer_types
        self.fa_idx = (
            types.index("full_attention") if "full_attention" in types else None
        )
        self.swa_idx = types.index(_SLIDING) if _SLIDING in types else None

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = swa_mask = None
        if self.fa_idx is not None:
            fa_mask = create_attention_mask(h, cache[self.fa_idx])
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.args.sliding_window
            )

        for layer, c in zip(self.layers, cache):
            mask = swa_mask if layer.use_sliding else fa_mask
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = FlintModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        # The reference serves with an fp32 output projection.
        out = out.astype(mx.float32)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        weights = {
            (k[len("language_model.") :] if k.startswith("language_model.") else k): v
            for k, v in weights.items()
        }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        # RecursiveSI/flint-int4 packs quantized weights as uint8; mlx's
        # quantized layers read the same bits as uint32.
        for k, v in list(weights.items()):
            if (
                k.endswith(".weight")
                and v.dtype == mx.uint8
                and f"{k[: -len('.weight')]}.scales" in weights
                and v.shape[-1] % 4 == 0
            ):
                weights[k] = v.view(mx.uint32)
        for layer_idx in range(self.args.num_hidden_layers):
            if layer_idx in self.args.mlp_only_layers:
                continue
            prefix = f"model.layers.{layer_idx}.mlp"
            for n in ["up_proj", "down_proj", "gate_proj"]:
                for suffix in ("weight", "scales", "biases"):
                    if f"{prefix}.experts.0.{n}.{suffix}" in weights:
                        to_join = [
                            weights.pop(f"{prefix}.experts.{e}.{n}.{suffix}")
                            for e in range(self.args.num_experts)
                        ]
                        weights[f"{prefix}.switch_mlp.{n}.{suffix}"] = mx.stack(to_join)
            for alias in ("gate.e_score_correction_bias", "gate.expert_bias"):
                if f"{prefix}.{alias}" in weights:
                    weights[f"{prefix}.expert_bias"] = weights.pop(f"{prefix}.{alias}")
            if (
                f"{prefix}.gate.weight" in weights
                and f"{prefix}.expert_bias" not in weights
            ):
                weights[f"{prefix}.expert_bias"] = mx.zeros((self.args.num_experts,))
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                if layer.use_sliding
                else KVCache()
            )
            for layer in self.layers
        ]

    @property
    def quant_predicate(self):
        # The router stays in full precision (Recursive's INT4 export and the
        # PR keep ``mlp.gate`` unquantized); so do the selection bias and sinks.
        def predicate(path, _):
            return not path.endswith("mlp.gate")

        return predicate
