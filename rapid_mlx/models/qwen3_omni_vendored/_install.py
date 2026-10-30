# SPDX-License-Identifier: Apache-2.0
"""Serve Qwen3-Omni through the vendored mlx-vlm 0.7.11 model code.

mlx-vlm 0.7.1 (the pin) answers every image or video request on qwen3_omni_moe with garbage
("!!!!", "median.median...") and faults the GPU when audio and images share a prompt: its deepstack
residuals are indexed against the whole prompt while prefill runs in chunks (upstream
Blaizzy/mlx-vlm #2099). 0.7.11 aligns the residuals with token positions and slices them per
chunk. Measured 2026-10-30 on workmule (Qwen3-Omni-30B-A3B-Instruct-4bit): 0.7.1 frames ->
"!!!!", 0.7.11 frames -> a description of the frames; audio correct on both. Bumping the whole
mlx-vlm pin would change every other model's code too, so only this model's package moves.
"""

import importlib
import sys

_PUBLIC = "mlx_vlm.models.qwen3_omni_moe"
_SUBMODULES = (
    "config",
    "audio",
    "vision",
    "language",
    "thinker",
    "talker",
    "code2wav",
    "omni_utils",
    "processing_qwen3_omni_moe",
    "qwen3_omni_moe",
)


def install() -> bool:
    """Alias the vendored package over mlx-vlm's qwen3_omni_moe. True when it is in place."""
    pkg = importlib.import_module(__package__)
    if sys.modules.get(_PUBLIC) is pkg:
        return True
    parent = importlib.import_module("mlx_vlm.models")
    for name in _SUBMODULES:
        mod = importlib.import_module(f"{__package__}.{name}")
        sys.modules[f"{_PUBLIC}.{name}"] = mod
        setattr(pkg, name, mod)
    sys.modules[_PUBLIC] = pkg
    # ``import mlx_vlm.models.qwen3_omni_moe`` resolves through the parent package's attribute
    # once mlx-vlm has imported its own copy, so the attribute is replaced too.
    setattr(parent, "qwen3_omni_moe", pkg)
    return True
