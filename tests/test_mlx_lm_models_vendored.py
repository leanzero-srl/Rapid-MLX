# SPDX-License-Identifier: Apache-2.0
"""Contract coverage for ``rapid_mlx.mlx_lm_models`` (vendored mimo_v2).

The import contract other tools build on: after ``import rapid_mlx.mlx_lm_models``
``importlib.import_module("mlx_lm.models.mimo_v2")`` resolves to the vendored
module (unless the installed mlx-lm ships its own), and the single engine's load
path registers it without that import.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import importlib
import importlib.util
import subprocess
import sys
from types import SimpleNamespace

# Same tiny config as the upstream PR's tests/test_models.py entry.
TINY = {
    "model_type": "mimo_v2",
    "vocab_size": 1000,
    "hidden_size": 256,
    "intermediate_size": 512,
    "moe_intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 96,
    "v_head_dim": 64,
    "swa_num_attention_heads": 4,
    "swa_num_key_value_heads": 2,
    "swa_head_dim": 96,
    "swa_v_head_dim": 64,
    "swa_rope_theta": 1000.0,
    "sliding_window_size": 32,
    "add_full_attention_sink_bias": False,
    "add_swa_attention_sink_bias": True,
    "hybrid_layer_pattern": [0, 1, 1, 0],
    "moe_layer_freq": [0, 1, 1, 1],
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "topk_method": "noaux_tc",
    "partial_rotary_factor": 0.334,
    "attention_bias": False,
    "attention_value_scale": 0.707,
    "layernorm_epsilon": 1e-5,
    "max_position_embeddings": 1000,
    "rope_theta": 1000.0,
}

_NATIVE = importlib.util.find_spec("mlx_lm.models.mimo_v2") is not None and (
    "mlx_lm.models.mimo_v2" not in sys.modules
    or not sys.modules["mlx_lm.models.mimo_v2"].__name__.startswith("rapid_mlx.")
)


def _fresh(code: str) -> str:
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


@pytest.mark.skipif(_NATIVE, reason="installed mlx-lm ships mimo_v2 natively")
def test_import_contract_resolves_vendored_module() -> None:
    line = _fresh(
        "import importlib, rapid_mlx.mlx_lm_models as m;"
        "mod = importlib.import_module('mlx_lm.models.mimo_v2');"
        "import sys;"
        "heavy = [k for k in sys.modules if k.startswith(('rapid_mlx.server',"
        " 'rapid_mlx.engine', 'rapid_mlx.scheduler', 'rapid_mlx.api'))];"
        "print(mod.__name__, m.REGISTERED['mimo_v2'], len(heavy))"
    )
    assert line == "rapid_mlx.mlx_lm_models.mimo_v2 vendored 0"


@pytest.mark.skipif(_NATIVE, reason="installed mlx-lm ships mimo_v2 natively")
def test_engine_load_path_registers_without_explicit_import() -> None:
    line = _fresh(
        "import importlib;"
        "from rapid_mlx.utils.tokenizer import _register_vendored_archs;"
        "_register_vendored_archs();"
        "print(importlib.import_module('mlx_lm.models.mimo_v2').__name__)"
    )
    assert line == "rapid_mlx.mlx_lm_models.mimo_v2"


def test_register_is_idempotent_and_defers_to_native(monkeypatch) -> None:
    import rapid_mlx.mlx_lm_models as m

    assert m.register() == m.register()
    monkeypatch.delitem(sys.modules, "mlx_lm.models.mimo_v2", raising=False)
    monkeypatch.setattr(m, "REGISTERED", {})
    monkeypatch.setattr(
        m.importlib.util, "find_spec", lambda name: SimpleNamespace(name=name)
    )
    assert m.register() == {"mimo_v2": "native"}
    assert "mlx_lm.models.mimo_v2" not in sys.modules


def test_tiny_model_forward_cache_and_pipeline_split() -> None:
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    from rapid_mlx.mlx_lm_models import mimo_v2

    model = mimo_v2.Model(mimo_v2.ModelArgs.from_dict(TINY))
    cache = model.make_cache()
    assert [type(c) for c in cache] == [
        KVCache,
        RotatingKVCache,
        RotatingKVCache,
        KVCache,
    ]
    logits = model(mx.array([[1, 2, 3, 4, 5]]), cache=cache)
    assert logits.shape == (1, 5, TINY["vocab_size"])
    assert bool(mx.isfinite(logits).all())

    group = SimpleNamespace(rank=lambda: 0, size=lambda: 2)
    inner = mimo_v2.Model(mimo_v2.ModelArgs.from_dict(TINY)).model
    inner.pipeline(group, [1, 3])  # rank 0 holds the LAST layers
    assert (inner.start_idx, inner.end_idx) == (3, 4)
    assert len(inner.pipeline_layers) == 1

    inner = mimo_v2.Model(mimo_v2.ModelArgs.from_dict(TINY)).model
    inner.pipeline(SimpleNamespace(rank=lambda: 1, size=lambda: 2))
    assert (inner.start_idx, inner.end_idx) == (0, 2)
    assert inner.layers[2:] == []

    inner = mimo_v2.Model(mimo_v2.ModelArgs.from_dict(TINY)).model
    with pytest.raises(ValueError):
        inner.pipeline(group, [2, 3])
