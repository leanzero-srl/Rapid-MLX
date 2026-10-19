# SPDX-License-Identifier: Apache-2.0
"""NVIDIA Nemotron 3.5 Bolt (``nemotron_h_puzzle``) served from a local folder.

LeanZero fork, 2026-10-19 (port of raullenchai/Rapid-MLX#4679 onto the RC13
line). The goose app mounts a FOLDER
(``~/.goose/models/rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit``), which
no hub alias names, so the profile must come from the checkpoint itself, and
the memory guard must price what the model really caches.

Defects this file pins (each test fails on 7108304a):

1. The folder got NO profile (``detect_model_config`` returned None): no tool
   parser (calls stream as content) and no reasoning parser (the primed
   thought streams as the answer).
2. The guard priced Bolt as 88 full-attention layers in fp32: 180,224 B/token
   with no recurrent state, against a real 8,192 B/token (8 attention blocks
   x 2 KV heads x 128 x bf16) plus 40 Mamba2 states of 4,255,744 B. The idle
   window on the Mac Studio fell to 116,480 of 262,144 tokens.

Fixtures are the checkpoint's real ``config.json`` and ``chat_template.jinja``
(md5 5177dc45876f506d1328c176b09687af).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from rapid_mlx import model_auto_config as auto_config_mod
from rapid_mlx.model_auto_config import (
    config_declares_linear_attention,
    detect_model_config,
)

FIXTURE = Path(__file__).parent / "fixtures" / "nemotron-3.5-bolt"
FOLDER = "rapid-mlx/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-MLX-4bit"

# 8 attention blocks x 2 (K, V) x 2 KV heads x 128 x bf16.
BOLT_KV_PER_TOKEN = 8 * 2 * 2 * 128 * 2
# 40 Mamba2 blocks x (conv ring (4 - 1) x conv_dim 10,240 x bf16
#                    + SSM state 128 heads x 64 x 128 x fp32).
BOLT_STATE = 40 * ((4 - 1) * (128 * 64 + 2 * 8 * 128) * 2 + 128 * 64 * 128 * 4)


@pytest.fixture(autouse=True)
def _fresh_resolution_log():
    auto_config_mod._reset_resolution_log_cache()
    yield
    auto_config_mod._reset_resolution_log_cache()


@pytest.fixture
def bolt_folder(tmp_path):
    folder = tmp_path / FOLDER
    folder.mkdir(parents=True)
    for name in ("config.json", "chat_template.jinja"):
        shutil.copy(FIXTURE / name, folder / name)
    return folder


def test_the_real_folder_gets_the_nemotron_profile_from_its_template(bolt_folder):
    profile = detect_model_config(str(bolt_folder))
    assert profile is not None
    assert profile.tool_call_parser == "nemotron"
    assert profile.reasoning_parser == "qwen3"
    assert profile.is_hybrid is True
    assert profile.is_hybrid_explicit is True
    assert profile.is_moe is True
    assert profile.supports_spec_decode is False


def test_the_real_config_declares_recurrence_before_any_weight_loads():
    config = json.loads((FIXTURE / "config.json").read_text())
    assert config_declares_linear_attention(config) is True


def test_a_puzzle_folder_without_the_tool_wire_gets_no_tool_parser(bolt_folder):
    # The parser follows the template: the same architecture with a template
    # that never shows the XML call wire must not claim the nemotron parser.
    (bolt_folder / "chat_template.jinja").write_text(
        "{%- for m in messages %}{{ m.content }}{% endfor %}"
        "{%- if add_generation_prompt %}{%- if enable_thinking %}<think>\n"
        "{%- else %}<think></think>{%- endif %}{%- endif %}"
    )
    profile = detect_model_config(str(bolt_folder))
    assert profile.tool_call_parser is None
    assert profile.reasoning_parser == "qwen3"
    assert profile.is_hybrid is True


# ---------------------------------------------------------------- memory guard

mx = pytest.importorskip("mlx.core")


def _stub_scheduler(model):
    from rapid_mlx.scheduler import Scheduler, SchedulerConfig

    s = Scheduler.__new__(Scheduler)
    s.model = model
    s.config = SchedulerConfig()
    s._kv_bytes_per_token_resolved = False
    s._kv_bytes_per_token = 0
    s._kv_fixed_baseline_bytes = 0
    s._kv_sliding_slot_bytes = 0
    s._kv_sliding_window = 0
    s._kv_quant_live_disabled = True
    return s


def _real_bolt_model():
    """The real config's module tree, unmaterialised (MLX arrays are lazy)."""
    from rapid_mlx.models import nemotron_h_puzzle as puzzle

    config = json.loads((FIXTURE / "config.json").read_text())
    model = puzzle.Model(puzzle.ModelArgs.from_dict(config))
    model.set_dtype(mx.bfloat16)  # the converted checkpoint's activation dtype
    return model


def test_the_guard_prices_bolt_by_what_each_block_caches():
    s = _stub_scheduler(_real_bolt_model())
    per_tok, fixed, slot, window = s._kv_footprint_terms()
    assert per_tok == BOLT_KV_PER_TOKEN
    assert fixed == BOLT_STATE
    assert (slot, window) == (0, 0)
    prior = s._memguard_prior()
    assert prior.kv_bytes_per_token == BOLT_KV_PER_TOKEN
    assert prior.kv_copy_bytes_per_context == 2 * 2 * 128 * 2  # one attention layer


def test_bolt_keeps_its_native_window_on_the_studio():
    from rapid_mlx.memory_guard import MemoryGuard, kv_growth_step

    s = _stub_scheduler(_real_bolt_model())
    guard = MemoryGuard(s._memguard_prior(), floor_tokens=kv_growth_step())
    # Mac Studio M3 Ultra 96 GB: static ceiling 75.1 GB; Bolt 4-bit weights
    # 43.3 GB (idle base with runtime buffers ~44 GB).
    window = guard.max_context(
        base=int(44e9),
        kv_footprint=s._kv_footprint_fn(),
        ceiling=int(75.1e9),
        upper=262_144,
    )
    assert window == 262_144  # 7108304a: 116,480


def _tiny_puzzle(dtype):
    from rapid_mlx.models import nemotron_h_puzzle as puzzle

    config = dict(
        model_type="nemotron_h_puzzle",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=4,
        max_position_embeddings=256,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        attention_bias=False,
        mamba_num_heads=2,
        mamba_head_dim=16,
        mamba_proj_bias=False,
        ssm_state_size=32,
        conv_kernel=4,
        n_groups=1,
        mlp_bias=False,
        layer_norm_epsilon=1e-5,
        use_bias=False,
        use_conv_bias=True,
        moe_latent_size=16,
        moe_shared_expert_intermediate_size=48,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        n_routed_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        routed_scaling_factor=5.0,
        time_step_limit=[0.0, {"__float__": "Infinity"}],
        block_configs=[
            {"block_type": "mamba"},
            {
                "block_type": "moe",
                "n_routed_experts": 4,
                "moe_intermediate_size": 16,
                "num_experts_per_tok": 2,
            },
            {"block_type": "attention"},
            {"block_type": "mamba"},
        ],
    )
    mx.random.seed(0)
    model = puzzle.Model(puzzle.ModelArgs.from_dict(config))
    model.set_dtype(dtype)
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_the_estimate_equals_the_cache_a_real_forward_holds(dtype):
    """Random-weight forward on a tiny Puzzle stack (mamba, moe, attention,
    mamba): the estimate is the bytes the live caches hold, exactly."""
    from mlx_lm.models.cache import ArraysCache, KVCache

    model = _tiny_puzzle(getattr(mx, dtype))
    cache = model.make_cache()
    tokens = 37
    logits = model(mx.arange(tokens).reshape(1, tokens) % 64, cache=cache)
    mx.eval(logits, [c.state for c in cache])
    assert bool(mx.all(mx.isfinite(logits.astype(mx.float32))))

    kv = [c for c in cache if isinstance(c, KVCache)]
    states = [c for c in cache if isinstance(c, ArraysCache)]
    assert len(kv) == 1 and len(states) == 2
    measured_per_token = sum(
        c.keys[..., : c.offset, :].nbytes + c.values[..., : c.offset, :].nbytes
        for c in kv
    ) // tokens
    measured_state = sum(a.nbytes for c in states for a in c.state if a is not None)

    per_tok, fixed, _slot, _window = _stub_scheduler(model)._kv_footprint_terms()
    assert per_tok == measured_per_token
    assert fixed == measured_state
