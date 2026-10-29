"""The real tensor names and packed shapes of Jundot/Qwen3.8-Flash-Next-oQ4e-mtp (oMLX, imatrix,
mixed 4/8-bit; fixture = its index + shard headers, 2026-10-29). Model-free: shapes only."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rapid_mlx.models import qwen4_offload as qo
from rapid_mlx.models.qwen4_exp_vision import VISION_KEY_PREFIXES
from rapid_mlx.spec_decode.mtp import qwen4_exp_inject as inject

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "jundot_qwen38_flash_next_oq4e_keys.json").read_text()
)
TENSORS = FIXTURE["tensors"]


def _shape(key):
    return SimpleNamespace(shape=tuple(TENSORS[key][1]))


def test_every_ple_shard_resolves_under_the_mlxvlm_spelling():
    index = SimpleNamespace(locs=TENSORS, model_dir="jundot")
    prefixes = [qo.ple_shard_prefix(index, 1, shard) for shard in range(128)]
    assert prefixes[0] == "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0"
    assert len(set(prefixes)) == 128


def test_the_ple_quant_layout_reads_as_4bit_group_32():
    first = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0"
    rows, words = TENSORS[first + ".weight"][1]
    store = SimpleNamespace(
        rows_per_shard=rows,
        words_per_row=words,
        groups_per_row=TENSORS[first + ".scales"][1][1],
    )
    emb = qo.OffloadedShardedEmbedding(rows, 160, 128, group_size=64, bits=8)
    emb.attach(store)
    assert (emb.bits, emb.group_size) == (4, 32)
    assert TENSORS[first + ".scales"][0] == "BF16"


def test_sanitize_drops_every_ple_table_and_keeps_the_rest(monkeypatch):
    monkeypatch.setenv("RAPID_MLX_QWEN4_PLE_OFFLOAD", "1")
    out = qo.sanitize_weights({key: None for key in TENSORS})
    dropped = set(TENSORS) - set(out)
    assert len(dropped) == 128 * 3
    assert all(".ngram_embedding.shards." in key for key in dropped)
    assert "language_model.model.layers.1.ple.key_proj.weight" in out


def test_the_tower_ships_under_a_prefix_the_loader_reads():
    tower = [key for key in TENSORS if key.startswith(VISION_KEY_PREFIXES)]
    assert len(tower) == 333 and all(TENSORS[k][0] == "BF16" for k in tower)


@pytest.mark.parametrize("module", sorted(FIXTURE["mtp_quantization"]))
def test_each_listed_mtp_module_gets_the_checkpoints_bits(module):
    spec = FIXTURE["mtp_quantization"][module]
    path = module.removeprefix("mtp.")
    packed = TENSORS[f"{module}.weight"][1]
    in_dim = packed[-1] * 32 // spec["bits"]
    checkpoint = {f"{path}.weight": _shape(f"{module}.weight"), f"{path}.scales": _shape(f"{module}.scales")}
    got = inject.checkpoint_quant_spec(checkpoint, path, in_dim)
    assert got == {"group_size": spec["group_size"], "bits": spec["bits"]}


def test_the_mtp_head_ships_split_experts_and_an_unquantized_router():
    """The oQ4e head stores experts already split as switch_mlp.{gate,up,down}_proj (4-bit g64) and
    its router mtp.layers.0.mlp.gate in BF16. The old uniform predicate quantized that router 8-bit,
    so the strict contract refused the head; the checkpoint now decides per tensor."""
    raw = {key: _shape(key) for key in TENSORS if key.startswith("mtp.")}
    sanitized = inject._sanitize_mtp_weights(raw)
    assert set(sanitized) == {key.removeprefix("mtp.") for key in raw}
    assert TENSORS["mtp.layers.0.mlp.gate.weight"][0] == "BF16"
    assert inject.checkpoint_quant_spec(sanitized, "layers.0.mlp.gate", 2560) is False
    assert inject.checkpoint_quant_spec(sanitized, "layers.0.mlp.switch_mlp.gate_proj", 2560) == {
        "group_size": 64,
        "bits": 4,
    }


def test_the_folded_fp8_ple_scale_is_dropped_and_an_unfolded_one_refused():
    """The oQ4e build carries ...ngram_embedding.weight_scale = 1.0 (BF16 [1]), the official FP8
    checkpoint's PLE scale already folded; strict load_weights refused it as an unknown parameter."""
    import mlx.core as mx

    from rapid_mlx.models.qwen4_exp import TextModel

    key = "model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
    assert TENSORS["language_model." + key] == ["BF16", [1]]
    fake = SimpleNamespace(args=SimpleNamespace(tie_word_embeddings=False))
    out = TextModel.sanitize(fake, {key: mx.ones((1,), dtype=mx.bfloat16), "model.norm.weight": mx.ones((2,))})
    assert list(out) == ["model.norm.weight"]
    with pytest.raises(ValueError, match="unfolded FP8 PLE scale"):
        TextModel.sanitize(fake, {key: mx.full((1,), 0.5, dtype=mx.bfloat16)})


def test_the_official_preprocessor_pixel_limit_is_capped_on_the_text_lane(tmp_path, monkeypatch):
    """Qwen's preprocessor_config.json (shipped by the oQ4e build) allows 16,777,216 px per image:
    ~340 GB of encode scores. The text lane caps it at the measured 1,003,520 px unless told."""
    from rapid_mlx.models import qwen4_exp_vision as v

    (tmp_path / "preprocessor_config.json").write_text(json.dumps({
        "size": {"longest_edge": 16777216, "shortest_edge": 65536},
        "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
    }))
    assert v._image_processor(tmp_path).max_pixels == v.TEXT_LANE_MAX_PIXELS == 1_003_520
    monkeypatch.setenv("RAPID_MLX_TEXT_LANE_MAX_PIXELS", "4000000")
    assert v._image_processor(tmp_path).max_pixels == 4_000_000
