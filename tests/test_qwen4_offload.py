"""Model-free tests for the qwen4_exp PLE / routed-expert offload.

Synthetic checkpoints in the real tensor layout are written to tmp_path; the
offloaded modules must reproduce the resident modules bit for bit.
"""

from __future__ import annotations

import types

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_lm.models.switch_layers import SwitchGLU

from rapid_mlx.models import qwen4_offload as qo
from rapid_mlx.models.qwen4_exp import ShardedEmbedding, SparseMoeBlock

PLE_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{}"


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith(qo.ENV_PREFIX):
            monkeypatch.delenv(key, raising=False)
    qo._INDEX = None
    qo._PLE_STORE = None
    qo._MANAGER = None
    yield
    qo._INDEX = None
    qo._PLE_STORE = None
    qo._MANAGER = None


def _ple_checkpoint(tmp_path, parts=4, rows=96, dims=160):
    mx.random.seed(0)
    tensors = {}
    for shard in range(parts):
        w = mx.random.normal((rows, dims)).astype(mx.bfloat16)
        q, s, b = mx.quantize(w, group_size=32, bits=4)
        name = PLE_PREFIX.format(shard)
        tensors[name + ".weight"] = q
        tensors[name + ".scales"] = s
        tensors[name + ".biases"] = b
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
    return tensors


def test_ple_rows_bit_identical(tmp_path, monkeypatch):
    parts, rows, dims = 4, 96, 160
    tensors = _ple_checkpoint(tmp_path, parts, rows, dims)
    monkeypatch.setenv("RAPID_MLX_QWEN4_OFFLOAD_DIR", str(tmp_path))

    resident = ShardedEmbedding(parts * rows, dims, parts)
    nn.quantize(resident, group_size=32, bits=4)
    resident.load_weights(
        [
            (f"shards.{i}.{leaf}", tensors[PLE_PREFIX.format(i) + "." + leaf])
            for i in range(parts)
            for leaf in ("weight", "scales", "biases")
        ]
    )
    offloaded = qo.OffloadedShardedEmbedding(rows, dims, parts, group_size=32, bits=4)
    rng = np.random.default_rng(1)
    for shape in [(1, 1, 16), (2, 7, 16), (1, 300, 16)]:
        ids = mx.array(rng.integers(0, parts * rows, size=shape), dtype=mx.int64)
        ref = resident(ids)
        got = offloaded(ids)
        assert got.shape == ref.shape and got.dtype == ref.dtype
        assert mx.array_equal(got, ref).item()


def _expert_checkpoint(tmp_path, layers=2, experts=16, hidden=128, inter=64):
    mx.random.seed(3)
    tensors = {}
    for layer in range(layers):
        prefix = f"model.language_model.layers.{layer}.mlp.experts"
        gate_up = mx.random.normal((experts, 2 * inter, hidden)).astype(mx.bfloat16) * 0.05
        down = mx.random.normal((experts, hidden, inter)).astype(mx.bfloat16) * 0.05
        for leaf, w in (("gate_up_proj", gate_up), ("down_proj", down)):
            q, s, b = mx.quantize(w, group_size=64, bits=4)
            tensors[f"{prefix}.{leaf}"] = q
            tensors[f"{prefix}.{leaf}.scales"] = s
            tensors[f"{prefix}.{leaf}.biases"] = b
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
    return tensors


def _resident_switch(tensors, layer, experts, hidden, inter, fused):
    prefix = f"model.language_model.layers.{layer}.mlp.experts"
    sw = SwitchGLU(hidden, inter, experts)
    sw.set_dtype(mx.bfloat16)
    nn.quantize(sw, group_size=64, bits=4)
    weights = []
    for leaf in ("", ".scales", ".biases"):
        attr = {"": "weight", ".scales": "scales", ".biases": "biases"}[leaf]
        gu = tensors[f"{prefix}.gate_up_proj{leaf}"]
        weights.append((f"gate_proj.{attr}", gu[:, :inter]))
        weights.append((f"up_proj.{attr}", gu[:, inter:]))
        weights.append((f"down_proj.{attr}", tensors[f"{prefix}.down_proj{leaf}"]))
    sw.load_weights(weights)
    mx.eval(sw.parameters())
    if fused:
        from rapid_mlx.moe_fusion import fuse_gate_up

        holder = nn.Module()
        holder.sw = sw
        fuse_gate_up(holder)
    return sw


def _placeholder_switch(hidden, inter, fused):
    sw = SwitchGLU(hidden, inter, 1)
    sw.set_dtype(mx.bfloat16)
    nn.quantize(sw, group_size=64, bits=4)
    mx.eval(sw.parameters())
    if fused:
        from rapid_mlx.moe_fusion import fuse_gate_up

        holder = nn.Module()
        holder.sw = sw
        fuse_gate_up(holder)
    return sw


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("policy", ["lru", "lfu"])
def test_expert_slots_bit_identical(tmp_path, monkeypatch, fused, policy):
    layers, experts, hidden, inter, top_k, slots = 2, 16, 128, 64, 3, 5
    tensors = _expert_checkpoint(tmp_path, layers, experts, hidden, inter)
    monkeypatch.setenv("RAPID_MLX_QWEN4_OFFLOAD_DIR", str(tmp_path))
    monkeypatch.setenv("RAPID_MLX_QWEN4_EXPERT_SLOTS", str(slots))
    monkeypatch.setenv("RAPID_MLX_QWEN4_EXPERT_POLICY", policy)
    cfg = qo.offload_config()
    manager = qo.ExpertOffloadManager(layers, experts, cfg)
    resident, blocks = {}, {}
    for layer in range(layers):
        resident[layer] = types.SimpleNamespace(
            switch_mlp=_resident_switch(tensors, layer, experts, hidden, inter, fused)
        )
        blocks[layer] = types.SimpleNamespace(switch_mlp=_placeholder_switch(hidden, inter, fused))
        blocks[layer]._routed = lambda x, i, target_verify=False, b=blocks[layer]: SparseMoeBlock._routed(
            b, x, i, target_verify=target_verify
        )
        manager.register(layer, blocks[layer])
    rng = np.random.default_rng(7)

    def check(shape, target_verify=False):
        x = (mx.random.normal((*shape, hidden)) * 0.5).astype(mx.bfloat16)
        idx = np.stack(
            [rng.choice(experts, size=top_k, replace=False) for _ in range(int(np.prod(shape)))]
        ).reshape(*shape, top_k)
        indices = mx.array(idx.astype(np.uint32))
        for layer in range(layers):
            ref = SparseMoeBlock._routed(resident[layer], x, indices, target_verify=target_verify)
            got = manager.routed(blocks[layer], layer, x, indices, target_verify)
            assert mx.array_equal(got, ref).item(), (shape, layer)

    for _ in range(12):  # decode steps: churn through the cache
        check((1, 1))
    check((1, 4), target_verify=True)  # MTP verify shape (union of experts)
    check((2, 1))  # batched decode
    check((1, 64))  # prefill: needs > slots experts -> grouped path
    assert manager.grouped_calls > 0
    stats = manager.stats()
    assert stats["slots_per_layer"] == slots
    assert stats["misses"] > 0 and stats["hits"] > 0
    for ls in manager.layers.values():
        resident_experts = ls.expert_in[ls.expert_in >= 0]
        assert np.all(ls.slot_of[resident_experts] == np.flatnonzero(ls.expert_in >= 0))
        assert (ls.slot_of >= 0).sum() == resident_experts.size <= slots


def test_budget_to_slots(tmp_path, monkeypatch):
    _expert_checkpoint(tmp_path, 1, 16, 128, 64)
    monkeypatch.setenv("RAPID_MLX_QWEN4_OFFLOAD_DIR", str(tmp_path))
    active = mx.get_active_memory()
    per_expert = qo.CheckpointIndex(tmp_path)
    eb = sum(
        per_expert[f"model.language_model.layers.0.mlp.experts.{leaf}"].nbytes
        for leaf, _ in qo._EXPERT_LEAVES
    ) // 16
    monkeypatch.setenv("RAPID_MLX_QWEN4_EXPERT_RESERVE_GB", "0.000001")
    budget = (active + 1000 + 40 * eb) / 1e9
    monkeypatch.setenv("RAPID_MLX_QWEN4_EXPERT_BUDGET_GB", repr(budget))
    manager = qo.ExpertOffloadManager(1, 16, qo.offload_config())
    assert manager._decide_slots(eb) == 16  # clamped to num_experts
    monkeypatch.setenv("RAPID_MLX_QWEN4_EXPERT_BUDGET_GB", repr((active + 1000) / 1e9))
    manager = qo.ExpertOffloadManager(1, 16, qo.offload_config())
    with pytest.raises(RuntimeError):
        manager._decide_slots(eb)


def test_sanitize_stubs_and_drops(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_QWEN4_PLE_OFFLOAD", "1")
    monkeypatch.setenv("RAPID_MLX_QWEN4_EXPERT_SLOTS", "8")
    w = {
        "language_model.model.layers.0.mlp.experts.gate_up_proj": mx.zeros((512, 4, 2), dtype=mx.uint32),
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_3.weight": mx.zeros((2, 2)),
        "language_model.model.layers.0.mlp.gate.weight": mx.zeros((512, 2)),
    }
    out = qo.sanitize_weights(w)
    assert out["language_model.model.layers.0.mlp.experts.gate_up_proj"].shape == (1, 4, 2)
    assert not any("ngram_embedding.shard_" in k for k in out)
    assert out["language_model.model.layers.0.mlp.gate.weight"].shape == (512, 2)


def test_default_off_is_noop():
    cfg = qo.offload_config()
    assert not cfg.any
    w = {"language_model.model.layers.0.mlp.experts.down_proj": mx.zeros((512, 2))}
    assert qo.sanitize_weights(w) is w


MLXVLM_PLE_PREFIX = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.{}"


@pytest.mark.parametrize(
    "prefix,dtype",
    [
        (PLE_PREFIX, mx.bfloat16),
        # mlx-vlm / oMLX spelling (Jundot/Qwen3.8-Flash-Next-oQ4e-mtp, 2026-10-29)
        (MLXVLM_PLE_PREFIX, mx.bfloat16),
        (MLXVLM_PLE_PREFIX, mx.float16),
    ],
)
def test_ple_rows_bit_identical_for_every_checkpoint_spelling(tmp_path, monkeypatch, prefix, dtype):
    parts, rows, dims = 4, 96, 160
    mx.random.seed(0)
    tensors = {}
    for shard in range(parts):
        q, s, b = mx.quantize(mx.random.normal((rows, dims)).astype(dtype), group_size=32, bits=4)
        for leaf, value in (("weight", q), ("scales", s), ("biases", b)):
            tensors[f"{prefix.format(shard)}.{leaf}"] = value
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
    monkeypatch.setenv("RAPID_MLX_QWEN4_OFFLOAD_DIR", str(tmp_path))

    resident = ShardedEmbedding(parts * rows, dims, parts)
    nn.quantize(resident, group_size=32, bits=4)
    resident.load_weights(
        [
            (f"shards.{i}.{leaf}", tensors[f"{prefix.format(i)}.{leaf}"])
            for i in range(parts)
            for leaf in ("weight", "scales", "biases")
        ]
    )
    # Constructed with a WRONG layout on purpose: attach reads bits/group from the tensors.
    offloaded = qo.OffloadedShardedEmbedding(rows, dims, parts, group_size=64, bits=8)
    ids = mx.array(np.random.default_rng(1).integers(0, parts * rows, size=(2, 9, 16)), dtype=mx.int64)
    ref, got = resident(ids), offloaded(ids)
    assert (offloaded.bits, offloaded.group_size) == (4, 32)
    assert got.dtype == ref.dtype == dtype
    assert mx.array_equal(got, ref).item()


def test_a_checkpoint_without_ple_tables_is_named(tmp_path):
    mx.save_safetensors(str(tmp_path / "model.safetensors"), {"x.weight": mx.zeros((1,))})
    with pytest.raises(KeyError, match="shards.0.weight"):
        qo.ple_shard_prefix(qo.CheckpointIndex(tmp_path), 1, 0)


def test_sanitize_drops_both_ple_spellings(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_QWEN4_PLE_OFFLOAD", "1")
    w = {
        "model.language_model.layers.2.ple.ple_embedding.ngram_embedding.shard_0.weight": mx.zeros((2, 2)),
        "language_model.model.layers.2.ple.ple_embedding.ngram_embedding.shards.127.scales": mx.zeros((2, 2)),
        "language_model.model.layers.2.ple.key_proj.weight": mx.zeros((2, 2)),
    }
    assert list(qo.sanitize_weights(w)) == ["language_model.model.layers.2.ple.key_proj.weight"]
