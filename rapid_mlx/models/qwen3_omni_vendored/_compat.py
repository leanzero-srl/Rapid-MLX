# SPDX-License-Identifier: Apache-2.0
"""Two helpers the vendored mlx-vlm 0.7.11 qwen3_omni_moe needs that mlx-vlm 0.7.1 lacks."""

try:
    from mlx_vlm.models.base import cache_position_offset
except ImportError:  # mlx-vlm < 0.7.11
    import mlx.core as mx

    def cache_position_offset(cache) -> int:
        """Processed prompt columns, independent of the cache's storage cursor (0.7.11 copy)."""
        if cache is None:
            return 0
        offset = cache.offset
        if not isinstance(offset, mx.array):
            return int(offset)
        shared_offset = getattr(cache, "_offset", None)
        if shared_offset is None:
            shared_offset = getattr(cache, "_idx", None)
        if shared_offset is not None:
            return int(shared_offset)
        return int(offset.max().item()) if offset.size else 0


try:
    from mlx_vlm.models.switch_layers import EXPERT_WEIGHT_SUFFIXES
except ImportError:  # mlx-vlm < 0.7.11
    EXPERT_WEIGHT_SUFFIXES = ("weight", "scales", "biases")
