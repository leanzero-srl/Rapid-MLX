"""Opt-in YaRN rope scaling for Qwen4-Exp (Qwen3.8-Flash-Next).

The checkpoint trains at 262,144 positions.  Qwen's model card extends it to
~1M with static YaRN by replacing ``text_config.rope_parameters`` with::

    {"mrope_interleaved": true, "mrope_section": [11, 11, 10],
     "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
     "factor": 4.0, "original_max_position_embeddings": 262144}

Only the full-attention (QSA) layers and their indexer rotate; the Gated
DeltaNet layers carry no positions.  YaRN changes two things there:

1. the inverse frequencies of the 64 rotary dims (NTK-by-parts ramp between
   extrapolation and ``1/factor`` interpolation, beta_fast=32 / beta_slow=1);
2. an attention factor (``0.1 * ln(factor) + 1`` unless the config names one)
   that multiplies cos and sin, i.e. the rotary half of q and k.

The frequencies come from mlx-lm's ``YarnRoPE`` (same NTK-by-parts maths as
Hugging Face transformers ``_compute_yarn_parameters`` with ``truncate=True``);
the attention factor follows transformers' rule exactly, including an explicit
``attention_factor`` and the ``mscale``/``mscale_all_dim`` pair.

Scaling is OPT-IN.  It turns on when the checkpoint's ``rope_parameters`` (or a
legacy ``rope_scaling`` block) says ``yarn``, or when the operator sets
``RAPID_MLX_QWEN4_ROPE_SCALING`` (the CLI's ``--qwen4-rope-scaling``).  With
neither, nothing here runs and the default rope path is unchanged.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from functools import cache
from typing import Any

import mlx.core as mx

ROPE_SCALING_ENV = "RAPID_MLX_QWEN4_ROPE_SCALING"
_OFF = {"0", "off", "none", "default", "false", "no"}
_DEFAULT_TYPES = {None, "default"}


@dataclass(frozen=True)
class YarnScaling:
    """Resolved, hashable YaRN parameters (a plain attribute, never a weight)."""

    factor: float
    original_max_position_embeddings: int
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    attention_factor: float = 1.0

    @property
    def max_position_embeddings(self) -> int:
        return int(self.factor * self.original_max_position_embeddings)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rope_type": "yarn",
            "factor": self.factor,
            "original_max_position_embeddings": self.original_max_position_embeddings,
            "beta_fast": self.beta_fast,
            "beta_slow": self.beta_slow,
            "attention_factor": self.attention_factor,
        }


def _get_mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def parse_rope_scaling_override(text: str | None) -> dict[str, Any] | None:
    """Parse the operator override: JSON object, ``yarn[:factor]``, or off.

    Returns ``{}`` for an explicit off (disable even a checkpoint's yarn),
    ``None`` when no override is set, else the override dict.
    """

    if text is None:
        return None
    value = text.strip()
    if not value:
        return None
    if value.lower() in _OFF:
        return {}
    if value.startswith("{"):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("--qwen4-rope-scaling JSON must be an object")
        return parsed
    kind, _, factor = value.partition(":")
    if kind.strip().lower() != "yarn":
        raise ValueError(
            f"--qwen4-rope-scaling accepts yarn[:factor], a JSON object or off, got {text!r}"
        )
    override: dict[str, Any] = {"rope_type": "yarn"}
    if factor.strip():
        override["factor"] = float(factor)
    return override


def resolve_rope_scaling(
    rope_parameters: dict[str, Any] | None,
    rope_scaling: dict[str, Any] | None,
    max_position_embeddings: int,
    env: str | None = None,
) -> YarnScaling | None:
    """Resolve the effective rope scaling, or ``None`` for the default rope.

    Order: an explicit ``rope_scaling`` dict (legacy HF field, or the resolved
    dict this function's caller stored) wins over ``rope_parameters``; the
    operator override (env / CLI) is merged on top, and ``off`` disables both.
    Resolution is idempotent: feeding the stored dict back gives the same
    result, so the MTP head built from ``vars(args)`` matches the trunk.
    """

    if env is None:
        env = os.environ.get(ROPE_SCALING_ENV)
    override = parse_rope_scaling_override(env)
    if override == {}:
        return None
    if rope_scaling:
        source = dict(rope_scaling)
    else:
        source = dict(rope_parameters or {})
    if override:
        source.update(override)
    kind = source.get("rope_type", source.get("type"))
    if kind in _DEFAULT_TYPES:
        return None
    if kind != "yarn":
        raise ValueError(f"Qwen4-Exp supports rope_type default or yarn, not {kind!r}")
    if source.get("truncate", True) is False:
        raise ValueError("Qwen4-Exp YaRN supports truncate=true only")
    factor = float(source.get("factor") or 4.0)
    original = int(
        source.get("original_max_position_embeddings") or max_position_embeddings
    )
    if factor <= 1.0 or original <= 0:
        raise ValueError("YaRN needs factor > 1 and a positive original context")
    attention_factor = source.get("attention_factor")
    if attention_factor is None:
        mscale = source.get("mscale")
        mscale_all_dim = source.get("mscale_all_dim")
        if mscale and mscale_all_dim:
            attention_factor = _get_mscale(factor, mscale) / _get_mscale(
                factor, mscale_all_dim
            )
        else:
            attention_factor = _get_mscale(factor)
    return YarnScaling(
        factor=factor,
        original_max_position_embeddings=original,
        beta_fast=float(source.get("beta_fast") or 32.0),
        beta_slow=float(source.get("beta_slow") or 1.0),
        attention_factor=float(attention_factor),
    )


@cache
def yarn_inverse_frequency(
    rotary_dim: int, base: float, scaling: YarnScaling
) -> mx.array:
    """YaRN inverse frequencies for ``rotary_dim`` dims, via mlx-lm's YarnRoPE."""

    from mlx_lm.models.rope_utils import YarnRoPE

    rope = YarnRoPE(
        rotary_dim,
        traditional=False,
        max_position_embeddings=scaling.max_position_embeddings,
        base=base,
        scaling_factor=scaling.factor,
        original_max_position_embeddings=scaling.original_max_position_embeddings,
        beta_fast=scaling.beta_fast,
        beta_slow=scaling.beta_slow,
    )
    inverse = (1.0 / rope._freqs).astype(mx.float32)
    mx.eval(inverse)
    return inverse
