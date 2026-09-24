# SPDX-License-Identifier: Apache-2.0
"""D-METAL-CAP prices a quantized LIVE KV cache at its stored bytes.

``--kv-cache-dtype int8/int4`` stores the full-attention KV as packed values plus one
scale and one bias per group (``QuantizedBatchKVCache``). The admission projection used
to price it at the activation dtype, over-estimating an int8 request ~1.9x and an int4
request ~3.6x. These tests pin the re-pricing and every case that must keep bf16.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from rapid_mlx.scheduler import Scheduler, SchedulerConfig

# Qwen3.x-27B shape: 16 full-attention layers of 64 (the rest GatedDeltaNet), 4 KV heads x 256.
LAYER_TYPES = (["linear_attention"] * 3 + ["full_attention"]) * 16
BF16_GROWTH = 2 * 16 * 4 * 256 * 2


def _sched(bits: int | None, *, live_disabled: bool = False, group: int = 64, turbo=None):
    config = SchedulerConfig(
        max_num_seqs=8,
        max_concurrent_requests=64,
        enable_prefix_cache=False,
        use_memory_aware_cache=False,
        use_paged_cache=False,
        gpu_memory_utilization=0.5,
        metal_cap_kv_bytes_per_token=0,
    )
    model = MagicMock()
    model.config = SimpleNamespace(
        num_hidden_layers=64,
        num_key_value_heads=4,
        head_dim=256,
        layer_types=LAYER_TYPES,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        dtype="bfloat16",
    )
    sched = Scheduler(model=model, tokenizer=MagicMock(), config=config)
    if bits is not None:
        sched.config.kv_cache_quantization = True
        sched.config.kv_cache_quantization_bits = bits
        sched.config.kv_cache_turboquant = turbo
        sched._kv_quant_live_disabled = live_disabled
        sched._kv_quant_group_size = group
    sched._kv_bytes_per_token_resolved = False
    return sched


def test_bf16_growth_is_the_full_attention_kv():
    assert _sched(None)._resolve_kv_bytes_per_token() == BF16_GROWTH


@pytest.mark.parametrize(
    "bits, per_element",
    [(8, 1 + 2 * 2 / 64), (4, 0.5 + 2 * 2 / 64)],
)
def test_quantized_live_cache_is_priced_at_stored_bytes(bits, per_element):
    expected = BF16_GROWTH * per_element / 2
    assert _sched(bits)._resolve_kv_bytes_per_token() == pytest.approx(expected, abs=1)


def test_int8_and_int4_ratios_match_the_packed_layout():
    assert _sched(8)._resolve_kv_bytes_per_token() == 34_816  # 272 of 512 bytes per head-token
    assert _sched(4)._resolve_kv_bytes_per_token() == 18_432  # 144 of 512


def test_disabled_live_path_keeps_the_bf16_price():
    assert _sched(8, live_disabled=True)._resolve_kv_bytes_per_token() == BF16_GROWTH


def test_turboquant_keeps_its_operator_knob():
    assert _sched(8, turbo="v4")._resolve_kv_bytes_per_token() == BF16_GROWTH


def test_smaller_group_costs_more_scale_bytes():
    assert _sched(8, group=32)._resolve_kv_bytes_per_token() > _sched(8)._resolve_kv_bytes_per_token()


def test_operator_override_still_wins():
    sched = _sched(8)
    sched.config.metal_cap_kv_bytes_per_token = 12_345
    assert sched._resolve_kv_bytes_per_token() == 12_345


def test_recurrent_baseline_is_not_repriced():
    bf16 = _sched(None)
    int4 = _sched(4)
    bf16._resolve_kv_bytes_per_token()
    int4._resolve_kv_bytes_per_token()
    assert int4._resolve_kv_fixed_baseline_bytes() == bf16._resolve_kv_fixed_baseline_bytes()
