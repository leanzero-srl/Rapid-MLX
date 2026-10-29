"""Image input for qwen4_exp on the single engine's TEXT lane.

The text lane keeps MTP, prompt lookup, the PLE offload, continuous batching
and the prefix cache; images reach it as slot ids above the vocabulary
(``qwen4_exp_vision.TextLaneVision``).  The reference is the fork's model fed
merged embeddings + the full multimodal RoPE table (``MRopePositions``), the
path the pipeline split proved equal to mlx-vlm's qwen4_exp.  The text lane
must match it bit for bit: full prefill, chunked prefill (chunks that cut an
image in half), decode after the image, a padded batch, and a batch formed by
merging rows that prefilled alone.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_lm")
pytest.importorskip("mlx_vlm")
from PIL import Image  # noqa: E402

from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models.cache import CacheList  # noqa: E402
from mlx_lm.utils import load_model, quantize_model, save_model  # noqa: E402

from rapid_mlx.models.qwen4_exp import Model, ModelArgs, TextModelArgs  # noqa: E402
from rapid_mlx.models.qwen4_exp_cache import QSAIndexCache  # noqa: E402
from rapid_mlx.models.qwen4_exp_vision import (  # noqa: E402
    IMAGE_ID_BASE,
    IMAGE_ID_STRIDE,
    ImageSpan,
    MRopePositions,
    SpanRopePositions,
    TextLaneVision,
    load_vision_tower,
    merge_image_features,
    rope_position,
)
from rapid_mlx.utils.tokenizer import _register_vendored_archs  # noqa: E402

IMAGE, VIDEO, START, END, EOS = 250, 251, 252, 253, 255


def _tiny_config() -> dict:
    args = TextModelArgs(
        hidden_size=64,
        num_hidden_layers=8,
        vocab_size=256,
        max_position_embeddings=4096,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        rope_parameters={
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [3, 3, 2],
            "mrope_interleaved": True,
            "rope_type": "default",
        },
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        hc_count=4,
        hc_lowrank=64,
        layer_types=["linear_attention"] * 3
        + ["full_attention"]
        + ["linear_attention"] * 3
        + ["full_attention"],
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=64,
        indexer_budget=8,
        indexer_compress_ratio=2,
        ple_layer_ids=[2],
        ple_embed_dim=64,
        heads_per_ngram=1,
        ngram_vocab_size_base=257,
        make_ngram_vocab_size_divisible_by=64,
        split_ngram_parts=4,
        eos_token_id=EOS,
    )
    text = asdict(args)
    text["model_type"] = "qwen4_exp_text"
    return {
        "model_type": "qwen4_exp",
        "image_token_id": IMAGE,
        "video_token_id": VIDEO,
        "vision_start_token_id": START,
        "vision_end_token_id": END,
        "text_config": text,
        "vision_config": {
            "model_type": "qwen4_exp",
            "depth": 2,
            "hidden_size": 64,
            "num_heads": 2,
            "intermediate_size": 128,
            "out_hidden_size": 64,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "in_channels": 3,
            "num_position_embeddings": 64,
            "hidden_act": "gelu_pytorch_tanh",
            "deepstack_visual_indexes": [],
        },
    }


@pytest.fixture(scope="module")
def vision_checkpoint(tmp_path_factory) -> Path:
    from mlx_vlm.models.qwen4_exp import VisionConfig, VisionModel

    path = tmp_path_factory.mktemp("qwen4_text_lane_vision_ckpt")
    mx.random.seed(20261029)
    config = _tiny_config()
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=config["text_config"]))
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    model, config = quantize_model(model, config, group_size=64, bits=4)
    mx.eval(model.parameters())
    save_model(path, model, donate_model=False)

    tower = VisionModel(VisionConfig.from_dict(config["vision_config"]))
    tower.set_dtype(mx.bfloat16)
    quantized: list[str] = []

    def predicate(module_path, module):
        if hasattr(module, "to_quantized") and module.weight.shape[-1] % 64 == 0:
            quantized.append(module_path)
            return True
        return False

    nn.quantize(tower, group_size=64, bits=4, class_predicate=predicate)
    mx.eval(tower.parameters())
    visual = {}
    for key, value in tree_flatten(tower.parameters()):
        if key == "patch_embed.proj.weight":
            value = value.transpose(0, 4, 1, 2, 3)
        visual[f"model.visual.{key}"] = value
    mx.save_safetensors(str(path / "model-visual.safetensors"), visual)
    index = json.loads((path / "model.safetensors.index.json").read_text())
    index["weight_map"].update({key: "model-visual.safetensors" for key in visual})
    (path / "model.safetensors.index.json").write_text(json.dumps(index))
    for module_path in quantized:
        config["quantization"][f"vision_tower.{module_path}"] = {
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
        }
    (path / "config.json").write_text(json.dumps(config, indent=2))
    return path


@pytest.fixture(scope="module")
def loaded(vision_checkpoint):
    _register_vendored_archs()
    model, _ = load_model(vision_checkpoint)
    tower = load_vision_tower(vision_checkpoint)
    return model, tower


def _png(tmp_path: Path, height: int, width: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    pixels = (rng.random((height, width, 3)) * 255).astype(np.uint8)
    path = tmp_path / f"img-{height}x{width}-{seed}.png"
    Image.fromarray(pixels).save(path)
    return path


def _grid(tower, path: Path) -> list[int]:
    with Image.open(path) as opened:
        processed = tower.image_processor(images=[opened.convert("RGB")])
    return [int(v) for v in processed["image_grid_thw"][0]], processed["pixel_values"]


def _reference_row(model, tower, ids_expanded, images):
    """(embeddings [1, L, H], (table, delta)) for one row, pipeline-style."""
    embeddings = model.language_model.model.embed_tokens(mx.array([ids_expanded]))
    if not images:
        return embeddings, ([list(range(len(ids_expanded)))] * 3, 0)
    grids, pixels = zip(*(_grid(tower, path) for path in images))
    features = tower.encode(mx.array(np.concatenate(pixels)), list(grids))
    merged = merge_image_features(embeddings, mx.array(ids_expanded), features, IMAGE)
    return merged, tower.rope_positions(ids_expanded, list(grids))


def _expanded_reference_ids(tower, ids, images):
    out = []
    image_iter = iter(images)
    for token in ids:
        if token == IMAGE:
            grid, _ = _grid(tower, next(image_iter))
            out.extend([IMAGE] * (grid[0] * grid[1] * grid[2] // 4))
        else:
            out.append(token)
    return out


def _run(model, tokens, embeddings, rope, prefix_step, decode, cache):
    """Logits of every step: chunked prefill of all but the last token, then
    greedy decode.  ``embeddings``/``rope`` None = the text-lane path."""
    logits = []
    prefix = tokens[:, :-1]
    for offset in range(0, prefix.shape[1], prefix_step):
        out = model(
            prefix[:, offset : offset + prefix_step],
            cache=cache,
            input_embeddings=None
            if embeddings is None
            else embeddings[:, :-1][:, offset : offset + prefix_step],
            rope_positions=rope,
        )
        logits.append(out)
        mx.eval(out, [layer.state for layer in cache])
    current = tokens[:, -1:]
    current_embeddings = None if embeddings is None else embeddings[:, -1:]
    for _ in range(decode):
        out = model(
            current, cache=cache, input_embeddings=current_embeddings, rope_positions=rope
        )
        current_embeddings = None
        logits.append(out)
        current = mx.argmax(out[:, -1, :], axis=-1).astype(mx.int32)[:, None]
        mx.eval(current, [layer.state for layer in cache])
    return [np.array(item.astype(mx.float32)) for item in logits]


def _attach(model, tower):
    vision = TextLaneVision(tower)
    model.language_model.model.text_lane_vision = vision
    return vision


def _detach(model):
    model.language_model.model.text_lane_vision = None


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------


def test_span_positions_equal_qwen3_5_get_rope_index(loaded, tmp_path):
    model, tower = loaded
    images = [_png(tmp_path, 64, 96, 1), _png(tmp_path, 128, 64, 2)]
    ids = [9, 31, START, IMAGE, END, 18, 42, START, IMAGE, END, 5, 60, 3]
    expanded = _expanded_reference_ids(tower, ids, images)
    grids = [_grid(tower, path)[0] for path in images]
    table, delta = tower.rope_positions(expanded, grids)

    vision = TextLaneVision(tower)
    slot_ids, _ = vision.expand_prompt(ids, [str(path) for path in images])
    spans: list[ImageSpan] = []
    from rapid_mlx.models.qwen4_exp_vision import delta_before

    for position, token in enumerate(slot_ids):
        if token >= IMAGE_ID_BASE and (token - IMAGE_ID_BASE) % IMAGE_ID_STRIDE == 0:
            entry = vision._by_slot[(token - IMAGE_ID_BASE) // IMAGE_ID_STRIDE]
            h, w = entry.grid_thw[1], entry.grid_thw[2]
            spans.append(
                ImageSpan(position, h * w, h, w, position + delta_before(spans, position))
            )
    host = [rope_position(spans, p) for p in range(len(expanded) + 6)]
    assert [list(axis) for axis in zip(*host[: len(expanded)])] == table
    # decode continues at logical + delta on all three axes
    for p in range(len(expanded), len(expanded) + 6):
        assert host[p] == (p + delta,) * 3
    device = SpanRopePositions([spans]).at(mx.arange(len(expanded) + 6)[None])
    assert device[:, 0, :].T.tolist() == [list(item) for item in host]
    assert SpanRopePositions([spans]).at_row(0, mx.array([[3, len(expanded) + 2]])).tolist() == [
        [[host[3][axis], host[len(expanded) + 2][axis]]] for axis in range(3)
    ]


def test_same_image_same_ids_and_a_different_image_never_shares_them(loaded, tmp_path):
    _, tower = loaded
    vision = TextLaneVision(tower)
    first = _png(tmp_path, 64, 64, 5)
    copy = tmp_path / "copy.png"
    copy.write_bytes(first.read_bytes())
    other = _png(tmp_path, 64, 64, 6)
    ids = [7, START, IMAGE, END, 8]
    a, _ = vision.expand_prompt(ids, [str(first)])
    b, _ = vision.expand_prompt(ids, [str(copy)])
    c, _ = vision.expand_prompt(ids, [str(other)])
    assert a == b
    assert a != c and len(a) == len(c)
    assert vision.images_registered == 2
    with pytest.raises(ValueError, match="placeholders"):
        vision.expand_prompt(ids, [])


def test_prefix_boundary_moves_into_the_expanded_space(loaded, tmp_path):
    _, tower = loaded
    vision = TextLaneVision(tower)
    image = _png(tmp_path, 64, 96, 9)
    ids = [7, START, IMAGE, END, 8, 9, 10]
    expanded, boundary = vision.expand_prompt(ids, [str(image)], prefix_boundary=5)
    count = len(expanded) - len(ids) + 1
    assert boundary == 5 + count - 1
    assert expanded[boundary - 1] == 8
    _, before = vision.expand_prompt(ids, [str(image)], prefix_boundary=2)
    assert before == 2


# ---------------------------------------------------------------------------
# Decoder equality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prefix_step", [1000, 5, 3])
def test_text_lane_image_row_matches_the_reference(loaded, tmp_path, prefix_step):
    model, tower = loaded
    images = [_png(tmp_path, 64, 96, 11), _png(tmp_path, 128, 64, 12)]
    ids = [9, 31, 77, START, IMAGE, END, 18, 42, START, IMAGE, END, 5, 60, 61, 3]
    expanded = _expanded_reference_ids(tower, ids, images)
    embeddings, table = _reference_row(model, tower, expanded, images)
    rope = MRopePositions.from_rows([table])
    _detach(model)
    want = _run(
        model,
        mx.array([expanded]),
        embeddings,
        rope,
        prefix_step,
        8,
        model.make_cache(),
    )
    vision = _attach(model, tower)
    try:
        slot_ids, _ = vision.expand_prompt(ids, [str(path) for path in images])
        got = _run(
            model, mx.array([slot_ids]), None, None, prefix_step, 8, model.make_cache()
        )
    finally:
        _detach(model)
    assert len(got) == len(want)
    for step, (a, b) in enumerate(zip(got, want)):
        assert np.array_equal(a, b), f"step {step} differs"
    assert vision.images_encoded == 2


def test_text_only_rows_are_untouched_by_an_attached_tower(loaded, tmp_path):
    model, tower = loaded
    tokens = mx.array([[9, 31, 77, 18, 42, 5, 60, 61, 62, 3]])
    _detach(model)
    want = _run(model, tokens, None, None, 4, 6, model.make_cache())
    vision = _attach(model, tower)
    try:
        vision.expand_prompt([1, START, IMAGE, END], [str(_png(tmp_path, 64, 64, 3))])
        got = _run(model, tokens, None, None, 4, 6, model.make_cache())
    finally:
        _detach(model)
    for a, b in zip(got, want):
        assert np.array_equal(a, b)


def _left_padded(rows):
    width = max(map(len, rows))
    padding = [width - len(row) for row in rows]
    return padding, mx.array([[0] * pad + row for pad, row in zip(padding, rows)])


def _batch_cache(model, padding):
    from mlx_lm.models.cache import BatchKVCache

    caches = []
    for layer in model.layers:
        if layer.is_linear:
            from rapid_mlx.models.qwen4_exp_cache import Qwen4ExpStateCache

            caches.append(
                Qwen4ExpStateCache(
                    size=4 if layer.ple is not None else 2, left_padding=padding
                )
            )
        else:
            caches.append(
                CacheList(
                    BatchKVCache(padding),
                    QSAIndexCache(
                        layer.self_attn.indexer.compress_ratio, left_padding=padding
                    ),
                )
            )
    return caches


def test_padded_batch_with_one_image_row_matches_the_reference(loaded, tmp_path):
    model, tower = loaded
    image = _png(tmp_path, 96, 64, 21)
    image_ids = [9, START, IMAGE, END, 18, 42, 3]
    text_ids = [5, 60, 61, 3]
    expanded = _expanded_reference_ids(tower, image_ids, [image])
    padding, tokens = _left_padded([expanded, text_ids])
    row0, table0 = _reference_row(model, tower, expanded, [image])
    row1, table1 = _reference_row(model, tower, text_ids, [])
    pad1 = model.language_model.model.embed_tokens(tokens[1:2, : padding[1]])
    embeddings = mx.concatenate(
        [row0, mx.concatenate([pad1, row1], axis=1)], axis=0
    )
    rope = MRopePositions.from_rows([table0, table1])
    _detach(model)
    want = _run(model, tokens, embeddings, rope, 1000, 6, _batch_cache(model, padding))

    vision = _attach(model, tower)
    try:
        slot_ids, _ = vision.expand_prompt(image_ids, [str(image)])
        padding2, tokens2 = _left_padded([slot_ids, text_ids])
        assert padding2 == padding
        got = _run(model, tokens2, None, None, 1000, 6, _batch_cache(model, padding))
    finally:
        _detach(model)
    for step, (a, b) in enumerate(zip(got, want)):
        real = [slice(padding[0], None), slice(padding[1], None)] if step == 0 else None
        if real is None:
            assert np.array_equal(a, b), f"decode step {step} differs"
        else:
            for row, cut in enumerate(real):
                assert np.array_equal(a[row, cut], b[row, cut]), f"row {row} prefill"


def test_rows_that_prefilled_alone_decode_alike_once_merged(loaded, tmp_path):
    """Continuous batching: each row prefills on its own cache, then joins."""
    model, tower = loaded
    image = _png(tmp_path, 64, 128, 31)
    vision = _attach(model, tower)
    try:
        slot_ids, _ = vision.expand_prompt([9, START, IMAGE, END, 18, 3], [str(image)])
        text_ids = [5, 60, 61, 62, 63, 3]

        def alone(ids, steps):
            cache = model.make_cache()
            model(mx.array([ids[:-1]]), cache=cache)
            current = mx.array([[ids[-1]]])
            out = []
            for _ in range(steps):
                logits = model(current, cache=cache)
                current = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)[
                    :, None
                ]
                mx.eval(current)
                out.append(np.array(logits[0, -1].astype(mx.float32)))
            return out

        want_image, want_text = alone(slot_ids, 5), alone(text_ids, 5)

        caches = []
        for ids in (slot_ids, text_ids):
            cache = model.make_cache()
            model(mx.array([ids[:-1]]), cache=cache)
            mx.eval([layer.state for layer in cache])
            caches.append(cache)
        assert any(caches[0][3][1].rope_spans) and not any(caches[1][3][1].rope_spans)
        merged = [type(a).merge([a, b]) for a, b in zip(*caches)]
        current = mx.array([[slot_ids[-1]], [text_ids[-1]]])
        for step in range(5):
            logits = model(current, cache=merged)
            current = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)[:, None]
            mx.eval(current)
            got = np.array(logits[:, -1].astype(mx.float32))
            np.testing.assert_allclose(got[0], want_image[step], atol=2e-2)
            np.testing.assert_allclose(got[1], want_text[step], atol=2e-2)
            assert int(got[0].argmax()) == int(want_image[step].argmax())
    finally:
        _detach(model)


# ---------------------------------------------------------------------------
# Cache lifecycle
# ---------------------------------------------------------------------------


def _qsa_with_spans():
    cache = QSAIndexCache(2)
    cache._offsets = [40]
    cache._compressed_counts = [20]
    cache.rope_spans = [[ImageSpan(3, 12, 3, 4, 3), ImageSpan(20, 6, 2, 3, 15)]]
    return cache


def test_qsa_cache_spans_survive_state_extract_merge_filter():
    cache = _qsa_with_spans()
    clone = QSAIndexCache.from_state(cache.state, cache.meta_state)
    assert clone.rope_spans == cache.rope_spans
    plain = QSAIndexCache(2)
    plain._offsets = [9]
    plain._compressed_counts = [4]
    assert "rope_spans" not in plain.meta_state
    merged = QSAIndexCache.merge([cache, plain])
    assert merged.rope_spans == [cache.rope_spans[0], []]
    assert merged.extract(0).rope_spans == cache.rope_spans
    merged.filter([1])
    assert merged.rope_spans == [[]]


def test_trim_drops_only_spans_that_start_past_the_new_end():
    cache = QSAIndexCache(2)
    cache._offsets = [22]
    cache._compressed_counts = [11]
    cache.rope_spans = [[ImageSpan(3, 12, 3, 4, 3), ImageSpan(21, 6, 2, 3, 15)]]
    assert cache.trim(1) == 1
    assert [span.start for span in cache.rope_spans[0]] == [3]


def test_mtp_head_embeds_image_ids_as_the_image_token(loaded, tmp_path):
    model, tower = loaded
    vision = _attach(model, tower)
    try:
        ids = mx.array([[IMAGE_ID_BASE + 5, 7, IMAGE_ID_BASE + IMAGE_ID_STRIDE]])
        assert vision.map_ids(ids).tolist() == [[IMAGE, 7, IMAGE]]
    finally:
        _detach(model)
