"""Image input for qwen4_exp: multimodal RoPE, the merged embeddings, and the split.

A tiny random-init checkpoint carries every Qwen4-Exp layer kind AND a
(quantized) Qwen3-VL vision tower in the checkpoint's own key layout
(``model.visual.*``, PyTorch conv layout).  The single-process reference runs
the fork's model directly — ``input_embeddings`` + ``rope_positions`` — and the
2/3-rank pipelines (``mlx.launch --backend ring`` on 127.0.0.1) must match it
bit for bit: logits and greedy tokens.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("mlx_lm")
pytest.importorskip("mlx_vlm")
pytestmark = pytest.mark.requires_mlx

from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.generate import _make_cache  # noqa: E402
from mlx_lm.utils import (  # noqa: E402
    load_model,
    quantize_model,
    save_model,
)

from rapid_mlx.distributed import pipeline_qwen4 as pipe  # noqa: E402
from rapid_mlx.models import qwen4_exp  # noqa: E402
from rapid_mlx.models.qwen4_exp import Model, ModelArgs, TextModelArgs  # noqa: E402
from rapid_mlx.models.qwen4_exp_vision import (  # noqa: E402
    MRopePositions,
    load_vision_tower,
    merge_image_features,
)
from rapid_mlx.utils.tokenizer import _register_vendored_archs  # noqa: E402

from .test_pipeline_qwen4 import DECODE_TOKENS, _launch  # noqa: E402

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

    path = tmp_path_factory.mktemp("qwen4_vision_ckpt")
    mx.random.seed(20260924)
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
            value = value.transpose(0, 4, 1, 2, 3)  # the checkpoint's torch layout
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
    # mlx-lm's save_config drops vision_config; write the checkpoint's own shape.
    (path / "config.json").write_text(json.dumps(config, indent=2))
    return path


def _images(vision, sizes, seed):
    rng = np.random.default_rng(seed)
    out = []
    for height, width in sizes:
        pixels = (rng.random((height, width, 3)) * 255).astype(np.uint8)
        processed = vision.image_processor([Image.fromarray(pixels)])
        out.append(
            pipe.ImageInput(
                processed["pixel_values"], processed["image_grid_thw"].tolist()
            )
        )
    return out


def _image_row(text_before, image: pipe.ImageInput, text_after, merge=2):
    tokens = int(np.prod(image.grid_thw[0])) // merge**2
    return text_before + [START] + [IMAGE] * tokens + [END] + text_after


def _two_images(vision):
    """Two images in one row, as a chat with two attachments renders."""
    first, second = _images(vision, [(64, 96), (128, 64)], 7)
    ids = _image_row([9, 31, 77], first, [18, 42]) + _image_row(
        [], second, [5, 60, 61, 62, 63, 64, 3]
    )
    both = pipe.ImageInput(
        np.concatenate([first.pixel_values, second.pixel_values]),
        first.grid_thw + second.grid_thw,
    )
    return ids, both


def _reference(model_dir, rows, images, max_tokens, prefill_step=None):
    """The fork's single-process model fed merged embeddings + RoPE positions."""
    _register_vendored_archs()
    model, _ = load_model(model_dir)
    vision = load_vision_tower(model_dir)
    width = max(map(len, rows))
    padding = [width - len(row) for row in rows]
    tokens = mx.array([[0] * pad + row for pad, row in zip(padding, rows)])
    embedded = []
    tables = []
    for index, (row_ids, pad, image) in enumerate(zip(rows, padding, images)):
        row = model.language_model.model.embed_tokens(tokens[index][None])
        if image is not None:
            features = vision.encode(mx.array(image.pixel_values), image.grid_thw)
            real = merge_image_features(
                row[:, pad:], mx.array(row_ids), features, IMAGE
            )
            row = mx.concatenate([row[:, :pad], real], axis=1)
            tables.append(vision.rope_positions(row_ids, image.grid_thw))
        else:
            tables.append(([list(range(len(row_ids)))] * 3, 0))
        embedded.append(row)
    embeddings = mx.concatenate(embedded, axis=0)
    rope = MRopePositions.from_rows(tables)

    def fresh_cache():
        return (
            model.make_cache() if len(rows) == 1 else _make_cache(model, padding, None)
        )

    logits = model(
        tokens, cache=fresh_cache(), input_embeddings=embeddings, rope_positions=rope
    )
    mx.eval(logits)
    cache = fresh_cache()
    prefix = tokens[:, :-1]
    step = prefill_step or prefix.shape[1]
    for offset in range(0, prefix.shape[1], step):
        model(
            prefix[:, offset : offset + step],
            cache=cache,
            input_embeddings=embeddings[:, :-1][:, offset : offset + step],
            rope_positions=rope,
        )
        mx.eval([layer.state for layer in cache])
    current, current_embeddings = tokens[:, -1:], embeddings[:, -1:]
    generated = []
    for _ in range(max_tokens):
        out = model(
            current,
            cache=cache,
            input_embeddings=current_embeddings,
            rope_positions=rope,
        )
        current_embeddings = None
        next_tokens = mx.argmax(out[:, -1, :], axis=-1).astype(mx.int32)
        mx.eval(next_tokens, [layer.state for layer in cache])
        generated.append(next_tokens.tolist())
        current = next_tokens[:, None]
    return np.array(logits.astype(mx.float32)), np.array(generated).T


# ---------------------------------------------------------------------------
# RoPE and positions
# ---------------------------------------------------------------------------


def test_equal_axes_rotate_bit_identically_to_text_positions():
    x = mx.random.normal((2, 3, 5, 64)).astype(mx.bfloat16)
    text = mx.array([[0, 1, 2, 3, 4], [7, 8, 9, 10, 11]], dtype=mx.int64)
    plain = qwen4_exp.apply_qwen4_exp_rope(x, text, rotary_dim=16, base=1e7)
    multimodal = qwen4_exp.apply_qwen4_exp_rope(
        x,
        mx.broadcast_to(text[None], (3, 2, 5)),
        rotary_dim=16,
        base=1e7,
        mrope_section=(3, 3, 2),
    )
    assert mx.array_equal(plain, multimodal).item()


def test_mrope_kernel_matches_the_pure_path_and_the_interleaving(monkeypatch):
    x = mx.random.normal((1, 2, 4, 64)).astype(mx.float32)
    positions = mx.array(
        [[[3, 3, 3, 3]], [[0, 0, 1, 1]], [[0, 1, 0, 1]]], dtype=mx.int64
    )
    kernel = qwen4_exp.apply_qwen4_exp_rope(
        x, positions, rotary_dim=16, base=1e7, mrope_section=(3, 3, 2)
    )
    monkeypatch.setattr(qwen4_exp, "_qwen4_exp_rope_kernel", lambda *_args: None)
    pure = qwen4_exp.apply_qwen4_exp_rope(
        x, positions, rotary_dim=16, base=1e7, mrope_section=(3, 3, 2)
    )
    np.testing.assert_allclose(np.array(kernel), np.array(pure), atol=1e-5)
    # Qwen3-VL interleaving over 8 frequencies with section (3, 3, 2):
    # h at 1, 4, 7; w at 2, 5; t everywhere else.
    selector = qwen4_exp._interleaved_mrope_selector((3, 3, 2), 8).tolist()
    assert selector == [0, 1, 2, 0, 1, 2, 0, 1]
    with pytest.raises(ValueError, match="mrope_section"):
        qwen4_exp.apply_qwen4_exp_rope(x, positions, rotary_dim=16, base=1e7)


def test_mrope_positions_extend_past_the_prompt_by_the_delta():
    rope = MRopePositions.from_rows(
        [([[0, 1, 1, 1, 3], [0, 1, 1, 2, 3], [0, 1, 2, 1, 3]], -1), ([[0, 1]] * 3, 0)]
    )
    logical = mx.array([[3, 4, 5, 6], [-2, -1, 0, 1]])
    got = rope.at(logical).tolist()
    assert [axis[0] for axis in got] == [[1, 3, 4, 5], [2, 3, 4, 5], [1, 3, 4, 5]]
    # Text row: logical positions unchanged, padding included.
    assert all(axis[1] == [-2, -1, 0, 1] for axis in got)
    assert rope.at_row(0, mx.array([[2, 9]])).tolist() == [[[1, 8]], [[1, 8]], [[2, 8]]]


def test_merge_refuses_a_feature_count_mismatch():
    embeddings = mx.zeros((1, 4, 8))
    ids = mx.array([1, IMAGE, IMAGE, 2])
    merged = merge_image_features(embeddings, ids, mx.ones((2, 8)), IMAGE)
    assert merged[0, :, 0].tolist() == [0.0, 1.0, 1.0, 0.0]
    with pytest.raises(ValueError, match="do not match"):
        merge_image_features(embeddings, ids, mx.ones((3, 8)), IMAGE)


# ---------------------------------------------------------------------------
# Loading and planning
# ---------------------------------------------------------------------------


def test_vision_tower_loads_quantized_from_the_checkpoint_layout(vision_checkpoint):
    vision = load_vision_tower(vision_checkpoint)
    assert isinstance(vision.model.blocks[0].attn.qkv, nn.QuantizedLinear)
    assert vision.mrope_section == (3, 3, 2)
    (image,) = _images(vision, [(64, 96)], 1)
    assert image.grid_thw == [[1, 4, 6]]
    features = vision.encode(mx.array(image.pixel_values), image.grid_thw)
    assert features.shape == (6, 64)
    table, delta = vision.rope_positions(_image_row([1, 2], image, [3]), image.grid_thw)
    # t, h, w of the 2x3 merged grid after "1 2 <start>", then text resumes
    # at max + 1: the grid spans 3 positions for 6 tokens, so delta = -3.
    assert table[1][3:9] == [3, 3, 3, 4, 4, 4]
    assert table[2][3:9] == [3, 4, 5, 3, 4, 5]
    assert delta == -3


def test_rank0_plan_carries_the_tower(vision_checkpoint):
    args = pipe.load_text_args(vision_checkpoint)
    ckpt = pipe.read_checkpoint_bytes(vision_checkpoint, args.num_hidden_layers)
    cost = pipe.vision_cost(vision_checkpoint, ckpt)
    vision = load_vision_tower(vision_checkpoint)
    assert cost.weight_bytes == vision.parameter_bytes
    nodes = [pipe.NodeBudget(f"n{i}", 2**34, 2**32, "test") for i in range(2)]
    text = pipe.plan_pipeline(
        args, ckpt, nodes, context=256, batch=1, prefill_step=256, starts=[0, 4]
    )
    both = pipe.plan_pipeline(
        args,
        ckpt,
        nodes,
        context=256,
        batch=1,
        prefill_step=256,
        starts=[0, 4],
        vision=cost,
    )
    assert (
        both.stages[0].weight_bytes == text.stages[0].weight_bytes + cost.weight_bytes
    )
    assert both.stages[0].workspace_bytes == max(
        text.stages[0].workspace_bytes, cost.workspace_bytes
    )
    assert both.stages[1].total_bytes == text.stages[1].total_bytes
    as_json = pipe.plan_json(both, pipe.wire_bytes_per_token(args, 2, 2))
    assert as_json["vision"]["weight_bytes"] == cost.weight_bytes


# ---------------------------------------------------------------------------
# Single process vs the split
# ---------------------------------------------------------------------------


def _vision_npz(path: Path, images) -> Path:
    arrays = {}
    for row, image in enumerate(images):
        if image is not None:
            arrays[f"pixel_values_{row}"] = image.pixel_values
            arrays[f"grid_thw_{row}"] = np.array(image.grid_thw)
    np.savez(path, **arrays)
    return path


def _cases(vision):
    first, second = _images(vision, [(64, 96), (128, 64)], 3)
    single = [_image_row([9, 31, 77, 5, 200], first, [18, 42, 11, 64, 3])]
    ids, both = _two_images(vision)
    return {
        "one-image": (single, [first]),
        "two-images-one-row": ([ids], [both]),
        # A batch: an image row beside a longer text row, and two different
        # images left-padded against each other.
        "image-beside-text": (
            [
                single[0],
                [12, 34, 56, 78, 90, 11, 22, 44, 13, 24, 35, 46, 57, 68, 79]
                + [80, 91, 102, 113, 124, 7, 8, 9, 10, 14, 15],
            ],
            [first, None],
        ),
        "two-image-rows": (
            [single[0], _image_row([100, 101], second, [102, 103, 104])],
            [first, second],
        ),
    }


@pytest.mark.parametrize(
    ("case", "ranks", "split", "prefill_step"),
    [
        ("one-image", 2, "3", None),
        ("one-image", 2, "3", 4),  # a prefill chunk boundary inside the image
        ("two-images-one-row", 2, "1", 5),
        ("image-beside-text", 2, "3", None),
        ("two-image-rows", 3, "2,5", 4),
    ],
    ids=["one", "one-chunked", "two-in-a-row", "beside-text", "two-rows-3-ranks"],
)
def test_image_pipeline_matches_single_process(
    vision_checkpoint, tmp_path, case, ranks, split, prefill_step
):
    vision = load_vision_tower(vision_checkpoint)
    rows, images = _cases(vision)[case]
    ref_logits, ref_tokens = _reference(
        vision_checkpoint, rows, images, DECODE_TOKENS, prefill_step
    )
    dump = tmp_path / "pipeline.npz"
    extra = ("--vision-inputs", str(_vision_npz(tmp_path / "vision.npz", images)))
    if prefill_step is not None:
        extra += ("--prefill-step", str(prefill_step))
    stdout = _launch(vision_checkpoint, ranks, split, dump, rows, extra).stdout
    result = np.load(dump)
    padding = [max(map(len, rows)) - len(row) for row in rows]
    diffs = [
        float(np.max(np.abs(result["logits"][row, pad:] - ref_logits[row, pad:])))
        for row, pad in enumerate(padding)
    ]
    print(
        f"{case} ranks={ranks} split={result['starts'].tolist()} max|dlogits|={diffs}"
    )
    assert max(diffs) == 0.0, stdout
    np.testing.assert_array_equal(result["tokens"], ref_tokens)


def test_the_image_changes_the_answer(vision_checkpoint):
    """Guard against a merge that silently drops the features."""
    vision = load_vision_tower(vision_checkpoint)
    first, other = _images(vision, [(64, 96), (64, 96)], 11)
    row = [_image_row([9, 31, 77, 5, 200], first, [18, 42, 11, 64, 3])]
    logits_a, _ = _reference(vision_checkpoint, row, [first], 1)
    logits_b, _ = _reference(vision_checkpoint, row, [other], 1)
    assert float(np.max(np.abs(logits_a[0, -1] - logits_b[0, -1]))) > 0.0


# ---------------------------------------------------------------------------
# The OpenAI server: image content parts end to end over two ring ranks
# ---------------------------------------------------------------------------

VISION_TEMPLATE = (
    "{% for m in messages %}{% if m['content'] is string %}{{ m['content'] }}"
    "{% else %}{% for c in m['content'] %}{% if c['type'] == 'image' %}"
    "<|vision_start|><|image_pad|><|vision_end|>{% else %}{{ c['text'] }}"
    "{% endif %}{% endfor %}{% endif %} {% endfor %}"
)
SPECIALS = {
    "<|image_pad|>": IMAGE,
    "<|video_pad|>": VIDEO,
    "<|vision_start|>": START,
    "<|vision_end|>": END,
}


@pytest.fixture(scope="module")
def vision_server_checkpoint(vision_checkpoint, tmp_path_factory) -> Path:
    import shutil

    from tokenizers import Tokenizer, models, pre_tokenizers

    path = tmp_path_factory.mktemp("qwen4_vision_serve")
    for item in vision_checkpoint.iterdir():
        shutil.copy(item, path / item.name)
    by_id = {value: key for key, value in SPECIALS.items()}
    vocab = {by_id.get(i, f"w{i}"): i for i in range(256)}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="w1"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.add_special_tokens(list(SPECIALS))
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "eos_token": "w255",
                "unk_token": "w1",
                "chat_template": VISION_TEMPLATE,
            }
        )
    )
    return path


@pytest.fixture(scope="module")
def vision_server(vision_server_checkpoint):
    import os
    import signal

    from .test_pipeline_qwen4_serve import _Server

    running = _Server(vision_server_checkpoint)
    yield running
    if running.process.poll() is None:
        os.kill(running.ready[0]["pid"], signal.SIGTERM)
        running.process.wait(timeout=60)


def _png_data_url(height: int, width: int, seed: int) -> str:
    import base64
    import io

    rng = np.random.default_rng(seed)
    pixels = (rng.random((height, width, 3)) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _image_chat(url: str, text: str = "w9 w31 w77", **extra) -> dict:
    return {
        "model": "tiny-flash-pipeline",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": text},
                ],
            }
        ],
        "temperature": 0,
        **extra,
    }


def test_server_advertises_vision_like_the_single_engine(vision_server):
    (entry,) = vision_server.get("/v1/models")["data"]
    assert entry["modality"] == "image"
    assert entry["capabilities"][:2] == ["text", "vision"]
    assert vision_server.ready[0]["vision"] is True


def test_image_chat_is_answered_as_the_single_process_answers_it(
    vision_server, vision_server_checkpoint
):
    from mlx_lm.utils import load_tokenizer

    from rapid_mlx.utils.chat_template import apply_chat_template

    url = _png_data_url(64, 96, 5)
    body = _image_chat(url, max_tokens=24)
    answer = json.load(vision_server.post(body))
    again = json.load(vision_server.post(body))
    assert answer["choices"][0]["message"] == again["choices"][0]["message"]

    tokenizer = load_tokenizer(vision_server_checkpoint)
    vision = load_vision_tower(vision_server_checkpoint)
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "w9 w31 w77"}],
        }
    ]
    prompt = apply_chat_template(tokenizer, messages, model_name="tiny-flash-pipeline")
    from rapid_mlx.distributed.pipeline_qwen4_serve import _load_image

    processed = vision.processor(tokenizer)(
        text=[prompt], images=[_load_image(url)], return_tensors="np"
    )
    ids = processed["input_ids"][0].tolist()
    assert ids.count(IMAGE) == 6 and ids.count(START) == 1
    assert answer["usage"]["prompt_tokens"] == len(ids)
    image = pipe.ImageInput(
        processed["pixel_values"], processed["image_grid_thw"].tolist()
    )
    _, tokens = _reference(vision_server_checkpoint, [ids], [image], 24)
    expected = []
    for token in tokens[0].tolist():
        expected.append(token)
        if token == EOS:
            break
    # The server counts the EOS it stops on and never prints it; its streaming
    # detokenizer drops special tokens, as the single engine's does.
    assert answer["usage"]["completion_tokens"] == len(expected)
    words = (answer["choices"][0]["message"]["content"] or "").split()
    shown = [token for token in expected if token != EOS]
    assert words == tokenizer.decode(shown, skip_special_tokens=True).split()


def test_images_that_cannot_be_read_are_a_named_400(vision_server):
    import urllib.error

    for url, needle in [
        ("data:image/png;base64,bm90IGFuIGltYWdl", "image"),
        ("file:///etc/hosts", "Cannot process image"),
    ]:
        with pytest.raises(urllib.error.HTTPError) as refusal:
            vision_server.post(_image_chat(url))
        assert refusal.value.code == 400
        assert needle in json.load(refusal.value)["error"]["message"], url
