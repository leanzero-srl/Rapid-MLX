"""Unit tests for rapid_mlx.mlx_lm_models.mimo_v2_media: token layout, markers, the
feature store across prefill chunk boundaries, the omni-checkpoint loader shims.

The template/tokenizer test needs a MiMo tokenizer folder: MIMO_TOKENIZER_DIR (any omni
build's folder; skipped when unset)."""

import os
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from rapid_mlx.mlx_lm_models import mimo_v2_media as M

VOCAB = 152576


def manifest(*items):
    return {"items": list(items), "ship": {}, "evict": []}


def test_markers_are_deterministic_unused_rows_and_differ_per_image():
    a = M.markers("img-a", 500, VOCAB)
    assert (a == M.markers("img-a", 500, VOCAB)).all()
    assert a.min() >= M.MARKER_BASE and a.max() < VOCAB
    assert not (a == M.markers("img-b", 500, VOCAB)).all()


def test_expand_replaces_each_placeholder_in_order_and_refuses_a_mismatch():
    m = manifest(("image", "k1", 3), ("video", "k2", 2))
    toks = [1, M.VISION_START, M.IMAGE_PAD, M.VISION_END, 2, M.VISION_START, M.VIDEO_PAD, M.VISION_END]
    out = M.expand(toks, m, VOCAB, require_all=True)
    assert len(out) == len(toks) + 2 + 1
    assert out[2:5] == list(M.markers("k1", 3, VOCAB))
    with pytest.raises(ValueError):
        M.expand([M.VIDEO_PAD], manifest(("image", "k1", 3)), VOCAB, require_all=True)
    with pytest.raises(ValueError):
        M.expand([1, 2], m, VOCAB, require_all=True)
    # the system-prefix render sees fewer placeholders: allowed when not required
    assert M.expand([1], m, VOCAB, require_all=False) == [1]


@pytest.mark.parametrize("chunk", [1, 2, 3, 5, 7, 64])
def test_substitute_finds_the_right_rows_in_any_prefill_chunking(chunk):
    if chunk == 1:
        chunk = 2  # decode-sized chunks are never substituted (sampled tokens)
    st = M.MediaStore(VOCAB)
    fa = mx.arange(6 * 4, dtype=mx.float32).reshape(6, 4) + 1000
    fb = mx.arange(5 * 4, dtype=mx.float32).reshape(5, 4) + 2000
    st.put("a", fa)
    st.put("b", fb)
    prompt = M.expand(
        [7, M.VISION_START, M.IMAGE_PAD, M.VISION_END, 8, M.VISION_START, M.IMAGE_PAD,
         M.VISION_END, 9],
        manifest(("image", "a", 6), ("image", "b", 5)), VOCAB, require_all=True)
    want = np.zeros((len(prompt), 4), np.float32)
    want[2:8] = np.array(fa)
    want[11:16] = np.array(fb)
    got = []
    for s in range(0, len(prompt), chunk):
        ids = mx.array([prompt[s : s + chunk]])
        h = mx.zeros((1, ids.shape[1], 4))
        got.append(np.array(st.substitute(ids, h))[0] if ids.shape[1] >= 2 else np.zeros((1, 4)))
    got = np.concatenate(got)
    assert np.array_equal(got, want)


def test_substitute_handles_left_padded_batches_and_a_run_that_fills_a_chunk():
    st = M.MediaStore(VOCAB)
    f = mx.random.normal((300, 8))
    st.put("x", f)
    mk = [int(v) for v in M.markers("x", 300, VOCAB)]
    ids = mx.array([[0, 0, M.VISION_START] + mk[:61], mk[100:164]])
    h = st.substitute(ids, mx.zeros((2, 64, 8)))
    assert np.allclose(np.array(h[0, 3:]), np.array(f[:61]))
    assert np.allclose(np.array(h[1]), np.array(f[100:164]))


def test_a_marker_run_without_features_fails_loudly():
    st = M.MediaStore(VOCAB)
    st.put("x", mx.zeros((4, 2)))
    with pytest.raises(RuntimeError):
        st.substitute(mx.array([[5] + [int(v) for v in M.markers("gone", 4, VOCAB)]]), mx.zeros((1, 5, 2)))


def test_wire_round_trip_is_bit_exact():
    f = mx.random.normal((7, 16)).astype(mx.bfloat16)
    g = M.feats_from_wire(M.feats_to_wire(f))
    assert g.dtype == mx.bfloat16 and mx.array_equal(f, g).item()


def test_smart_resize_matches_xiaomi_bounds():
    assert M.smart_resize(600, 800, 32, 8192, 8388608) == (608, 800)
    h, w = M.smart_resize(4000, 6000, 32, 8192, 8388608)
    assert h * w <= 8388608 and h % 32 == 0 and w % 32 == 0
    h, w = M.smart_resize(40, 60, 32, 8192, 8388608)
    assert h * w >= 8192


def test_patch_order_is_qwen2vl_merge_grouped():
    c = M.MediaConfig()
    frames = np.arange(2 * 3 * 64 * 64, dtype=np.float32).reshape(2, 3, 64, 64)
    p, grid = M.patchify(frames, c)
    assert grid == (1, 4, 4) and p.shape == (16, 3 * 2 * 16 * 16)
    # 2nd patch of the first merge unit is the one to the RIGHT of patch 0 (row 0, col 1)
    assert p[1].reshape(3, 2, 16, 16)[0, 0, 0, 0] == frames[0, 0, 0, 16]
    # 3rd patch is BELOW patch 0 (row 1, col 0)
    assert p[2].reshape(3, 2, 16, 16)[0, 0, 0, 0] == frames[0, 0, 16, 0]


def test_quantization_keys_and_weights_lose_the_language_model_prefix():
    import rapid_mlx.mlx_lm_models as reg

    reg.register()
    from mlx_lm.models import mimo_v2

    params = {"quantization": {"group_size": 64, "bits": 4,
                               "language_model.model.layers.0.self_attn.q_proj": {"bits": 8}}}
    try:
        mimo_v2.ModelArgs.from_dict(params)
    except TypeError:
        pass  # incomplete args: only the in-place key rewrite matters here
    assert "model.layers.0.self_attn.q_proj" in params["quantization"]


@pytest.mark.skipif(not os.environ.get("MIMO_TOKENIZER_DIR"), reason="MIMO_TOKENIZER_DIR unset")
def test_template_renders_an_image_part_as_one_placeholder_then_expands():
    from mlx_lm.utils import load_tokenizer  # what mlx_lm.server renders with

    tok = load_tokenizer(Path(os.environ["MIMO_TOKENIZER_DIR"]))
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "What is in cell R1C2?"},
        {"type": "text", "text": "<|vision_start|><|image_pad|><|vision_end|>"}]}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    assert ids.count(M.IMAGE_PAD) == 1
    i = ids.index(M.IMAGE_PAD)
    assert ids[i - 1] == M.VISION_START and ids[i + 1] == M.VISION_END
    # the template's own image branch renders the same tokens
    native = tok.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": "What is in cell R1C2?"},
                                      {"type": "image_url", "image_url": {"url": "x"}}]}],
        add_generation_prompt=True, tokenize=True)
    if isinstance(native, dict):
        native = native["input_ids"]
    assert native == ids
    out = M.expand(ids, manifest(("image", "k", 40)), VOCAB, require_all=True)
    assert len(out) == len(ids) + 39
    vt = [M.VIDEO_START] + tok.encode("00:03", add_special_tokens=False)
    assert tok.encode("<|mimo_video_start|>00:03", add_special_tokens=False) == vt
