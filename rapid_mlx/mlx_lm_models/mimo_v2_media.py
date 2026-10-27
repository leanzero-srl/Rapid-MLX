# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2.6 media input (images, video) for the mlx_lm ``mimo_v2`` text model.

The language model is the vendored ``mimo_v2`` (text only). This module adds what an
omni checkpoint (e.g. EigenLabs/MiMo-V2.6-Flash-MOPD-MLX-4bit-mtp: ``language_model.*``
+ ``vision_tower.*`` + ``audio_encoder.*``) needs to read images and video:

* ``VisionModel``: the MiMo ViT (``vision_model_type: mimovl``), adapted from mlx-vlm
  (MIT, ``mlx_vlm/models/mimo_v2/vision.py`` v0.7.10). One change: the band-window
  layers attend in query blocks over a key band instead of building an ``[H, n, n]``
  mask, so a full-size screenshot (up to 32k patches) does not materialise gigabytes of
  mask. The maths is the HF reference's (``modeling_mimo_v2.py`` MiMoVisionTransformer):
  window ``|i - j| <= visual_token_window_size`` in the block's token order, the sink
  logit added to key 0, column-ordered blocks for ``vit_window_attn_types == 1``.
* preprocessing as Xiaomi's processor (preprocessor_config.json: Qwen2VLImageProcessor,
  patch 16, merge 2, temporal 2, min/max pixels 8192 / 8388608), smart_resize to
  multiples of 32, bicubic, Qwen2-VL patch order. Video: SGLang's layout
  (``<|mimo_video_start|>`` + per frame pair ``"MM:SS" <|vision_start|> pads
  <|vision_end|>`` + ``<|mimo_video_end|>``), 1 fps, 8..3600 frames, even count.
* the MARKER scheme: the language model positions are plain 1-D (HF modeling,
  SGLang and vLLM all use ``arange``; no M-RoPE), so an image is N embedding rows that
  replace N placeholder tokens. Each placeholder becomes a marker id from the unused
  vocab rows past the last special token, drawn from the image's content hash. The
  prompt cache keys on token ids, so two different images never share a cached prefix,
  and the model finds which features a marker run needs from the ids alone, whatever
  prefill chunk or cache hit it arrives in.
* ``MediaStore``: per-process features by key; ``substitute`` swaps a chunk's marker
  rows for the stored features before layer 0.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import subprocess
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

IMAGE_PAD = 151655
VIDEO_PAD = 151656
AUDIO_PAD = 151669
VISION_START = 151652
VISION_END = 151653
VIDEO_START = 151670
VIDEO_END = 151671
# First marker id: past every special token (the last is <|mimo_audio_end|> 151674).
MARKER_BASE = 151680

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# ---------------------------------------------------------------------------
# Vision tower
# ---------------------------------------------------------------------------


@dataclass
class VisionConfig:
    depth: int = 28
    hidden_size: int = 1280
    intermediate_size: int = 4608
    out_hidden_size: int = 4096
    num_heads: int = 32
    num_key_value_heads: int = 8
    qk_channels: int = 64
    in_chans: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    rms_norm_eps: float = 1e-6
    use_sink: bool = False
    visual_token_window_size: int = -1
    fullatt_block_indexes: list = field(default_factory=list)
    vit_window_attn_types: Optional[list] = None

    @classmethod
    def from_dict(cls, d: dict) -> "VisionConfig":
        names = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in names})


def _rotate_half(x):
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _rope(q, k, cos, sin):
    dtype = q.dtype
    cos = mx.expand_dims(cos, -2).astype(mx.float32)
    sin = mx.expand_dims(sin, -2).astype(mx.float32)
    q, k = q.astype(mx.float32), k.astype(mx.float32)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q.astype(dtype), k.astype(dtype)


class PatchEmbed(nn.Module):
    def __init__(self, c: VisionConfig):
        super().__init__()
        self.patch_dim = c.in_chans * c.temporal_patch_size * c.patch_size**2
        self.proj = nn.Linear(self.patch_dim, c.hidden_size, bias=False)

    def __call__(self, x):
        return self.proj(x.reshape(-1, self.patch_dim))


class VisionMLP(nn.Module):
    def __init__(self, c: VisionConfig):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# Query block of the band attention; keys per block = BLOCK + 2 * window.
BAND_BLOCK = 128
# Blocks attended per SDPA call (bounds the [blocks, H, BLOCK, BLOCK + 2W] mask).
BAND_GROUP = 64


class VisionAttention(nn.Module):
    def __init__(self, c: VisionConfig, use_sinks: bool, window: int):
        super().__init__()
        self.num_heads = c.num_heads
        self.num_kv_heads = c.num_key_value_heads
        self.head_dim = c.qk_channels
        self.scale = self.head_dim**-0.5
        self.window = window
        self.qkv = nn.Linear(
            c.hidden_size, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        )
        self.proj = nn.Linear(self.num_heads * self.head_dim, c.hidden_size)
        self.sinks = mx.zeros((self.num_heads,)) if use_sinks else None

    def _sink_mask(self, n, dtype):
        if self.sinks is None:
            return None
        col0 = (mx.arange(n) == 0)[None, None, None, :]
        return mx.where(
            col0, self.sinks.astype(dtype)[None, :, None, None], mx.array(0, dtype)
        )

    def _full(self, q, k, v):
        # q [n, H, d], k/v [n, H, d] (kv already repeated)
        n = q.shape[0]
        qt, kt, vt = (t.transpose(1, 0, 2)[None] for t in (q, k, v))
        out = mx.fast.scaled_dot_product_attention(
            qt, kt, vt, scale=self.scale, mask=self._sink_mask(n, q.dtype)
        )
        return out[0].transpose(1, 0, 2)

    def _band(self, q, k, v):
        n, H, d = q.shape
        W, B = self.window, BAND_BLOCK
        nb = -(-n // B)
        span = B + 2 * W
        q = mx.pad(q, [(0, nb * B - n), (0, 0), (0, 0)])
        k = mx.pad(k, [(W, nb * B - n + W), (0, 0), (0, 0)])
        v = mx.pad(v, [(W, nb * B - n + W), (0, 0), (0, 0)])
        outs = []
        r = mx.arange(B)[:, None]
        s = mx.arange(span)[None, :]
        for g0 in range(0, nb, BAND_GROUP):
            g1 = min(nb, g0 + BAND_GROUP)
            qb = mx.stack([q[b * B : (b + 1) * B] for b in range(g0, g1)])
            kb = mx.stack([k[b * B : b * B + span] for b in range(g0, g1)])
            vb = mx.stack([v[b * B : b * B + span] for b in range(g0, g1)])
            base = mx.arange(g0, g1)[:, None, None] * B
            i = base + r  # query index
            j = base - W + s  # key index
            ok = (mx.abs(i - j) <= W) & (j >= 0) & (j < n)
            mask = mx.where(ok, mx.array(0, q.dtype), mx.array(-mx.inf, q.dtype))[
                :, None
            ]
            if self.sinks is not None:
                sink = mx.where(
                    (j == 0)[:, None],
                    self.sinks.astype(q.dtype)[None, :, None, None],
                    mx.array(0, q.dtype),
                )
                mask = mask + sink
            out = mx.fast.scaled_dot_product_attention(
                qb.transpose(0, 2, 1, 3),
                kb.transpose(0, 2, 1, 3),
                vb.transpose(0, 2, 1, 3),
                scale=self.scale,
                mask=mask,
            )
            outs.append(out.transpose(0, 2, 1, 3).reshape(-1, H, d))
        return mx.concatenate(outs, axis=0)[:n]

    def __call__(self, x, cos, sin, full_attn, cu_seqlens):
        n = x.shape[0]
        qkv = self.qkv(x)
        qd = self.num_heads * self.head_dim
        kd = self.num_kv_heads * self.head_dim
        q = qkv[:, :qd].reshape(n, self.num_heads, self.head_dim)
        k = qkv[:, qd : qd + kd].reshape(n, self.num_kv_heads, self.head_dim)
        v = qkv[:, qd + kd :].reshape(n, self.num_kv_heads, self.head_dim)
        q, k = _rope(q, k, cos, sin)
        rep = self.num_heads // self.num_kv_heads
        if rep > 1:
            k = mx.repeat(k, rep, axis=1)
            v = mx.repeat(v, rep, axis=1)
        band = not full_attn and self.window > 0
        chunks = []
        for a, b in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            fn = self._band if band else self._full
            chunks.append(fn(q[a:b], k[a:b], v[a:b]))
        out = mx.concatenate(chunks, axis=0).reshape(n, -1)
        return self.proj(out)


class VisionBlock(nn.Module):
    def __init__(self, c: VisionConfig, use_sinks: bool, window: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.norm2 = nn.RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.attn = VisionAttention(c, use_sinks, window)
        self.mlp = VisionMLP(c)

    def __call__(self, x, cos, sin, full_attn, cu_seqlens):
        x = x + self.attn(self.norm1(x), cos, sin, full_attn, cu_seqlens)
        return x + self.mlp(self.norm2(x))


class PatchMerger(nn.Module):
    def __init__(self, c: VisionConfig):
        super().__init__()
        self.hidden = c.hidden_size * c.spatial_merge_size**2
        self.ln_q = nn.LayerNorm(c.hidden_size, eps=1e-6, bias=False)
        self.mlp = [
            nn.Linear(self.hidden, self.hidden, bias=False),
            nn.GELU(),
            nn.Linear(self.hidden, c.out_hidden_size, bias=False),
        ]

    def __call__(self, x):
        x = self.ln_q(x).reshape(-1, self.hidden)
        for layer in self.mlp:
            x = layer(x)
        return x


class VisionModel(nn.Module):
    def __init__(self, c: VisionConfig):
        super().__init__()
        self.config = c
        self.m = c.spatial_merge_size
        self.unit = self.m**2
        self.full = set(c.fullatt_block_indexes)
        self.types = c.vit_window_attn_types or [-1] * c.depth
        self.patch_embed = PatchEmbed(c)
        dim = c.qk_channels // 2
        self._inv_freq = 1.0 / (10000.0 ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
        self.blocks = [
            VisionBlock(
                c,
                use_sinks=c.use_sink and i not in self.full,
                window=c.visual_token_window_size,
            )
            for i in range(c.depth)
        ]
        self.merger = PatchMerger(c)

    @staticmethod
    def sanitize(weights):
        out = {}
        for k, v in weights.items():
            if k.endswith("patch_embed.proj.weight") and v.ndim == 5:
                v = v.reshape(v.shape[0], -1)
            out[k] = v
        return out

    def _apply_index(self, x, index):
        x = x.reshape(-1, self.unit, *x.shape[1:])
        return x[index].reshape(-1, *x.shape[2:])

    def _col_index(self, grid):
        out, off = [], 0
        for t, h, w in grid:
            gh, gw = h // self.m, w // self.m
            idx = mx.arange(t * gh * gw).reshape(t, gh, gw)
            out.append(idx.transpose(0, 2, 1).reshape(-1) + off)
            off += t * gh * gw
        return mx.concatenate(out, axis=0)

    def _rot(self, grid):
        m = self.m
        ids = []
        for t, h, w in grid:
            hp = mx.broadcast_to(mx.arange(h)[:, None], (h, w))
            hp = hp.reshape(h // m, m, w // m, m).transpose(0, 2, 1, 3).reshape(-1)
            wp = mx.broadcast_to(mx.arange(w)[None, :], (h, w))
            wp = wp.reshape(h // m, m, w // m, m).transpose(0, 2, 1, 3).reshape(-1)
            ids.append(mx.tile(mx.stack([hp, wp], axis=-1), (t, 1)))
        ids = mx.concatenate(ids, axis=0)
        mg = int(max(max(h, w) for _, h, w in grid))
        table = mx.outer(mx.arange(mg).astype(mx.float32), self._inv_freq)
        return table[ids].reshape(ids.shape[0], -1)

    def __call__(self, pixel_values, grid, return_layers=False):
        grid = [tuple(int(v) for v in row) for row in grid]
        dtype = self.patch_embed.proj.weight.dtype
        if dtype not in (mx.float32, mx.float16, mx.bfloat16):
            dtype = mx.bfloat16
        x = self.patch_embed(pixel_values.astype(dtype))
        rot = self._rot(grid)
        emb = mx.concatenate([rot, rot], axis=-1)
        row = (mx.cos(emb), mx.sin(emb))
        col_index = self._col_index(grid)
        rev = mx.argsort(col_index)
        cemb = self._apply_index(emb, col_index)
        col = (mx.cos(cemb), mx.sin(cemb))
        cu, tot = [0], 0
        for t, h, w in grid:
            for _ in range(t):
                tot += h * w
                cu.append(tot)
        layers = []
        T = self.types
        for i, blk in enumerate(self.blocks):
            if T[i] == 1 and (i == 0 or T[i - 1] != 1):
                x = self._apply_index(x, col_index)
            elif i > 0 and T[i] != 1 and T[i - 1] == 1:
                x = self._apply_index(x, rev)
            cos, sin = col if T[i] == 1 else row
            x = blk(x, cos, sin, i in self.full, cu)
            if return_layers:
                # every layer in ROW order, comparable with the reference
                layers.append(self._apply_index(x, rev) if T[i] == 1 else x)
        out = self.merger(x)
        return (out, layers) if return_layers else out


def _index(model_dir: Path) -> dict:
    p = model_dir / "model.safetensors.index.json"
    return json.loads(p.read_text())["weight_map"] if p.exists() else {}


def has_vision(model_dir) -> bool:
    model_dir = Path(model_dir)
    try:
        cfg = json.loads((model_dir / "config.json").read_text())
    except Exception:
        return False
    if cfg.get("model_type") != "mimo_v2" or not cfg.get("vision_config"):
        return False
    return any(k.startswith("vision_tower.") for k in _index(model_dir))


def load_vision(model_dir, dtype=mx.bfloat16, bits: int = 0, weights_file=None):
    """The vision tower from an omni checkpoint (``vision_tower.*``), or from a single
    safetensors file holding those keys (``weights_file``, tests)."""
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    vc = VisionConfig.from_dict(cfg["vision_config"])
    model = VisionModel(vc)
    files = (
        [Path(weights_file)]
        if weights_file
        else sorted(
            {model_dir / f for k, f in _index(model_dir).items() if k.startswith("vision_tower.")}
        )
    )
    weights = {}
    for f in files:
        for k, v in mx.load(str(f)).items():
            if k.startswith("vision_tower."):
                weights[k[len("vision_tower.") :]] = v
    weights = VisionModel.sanitize(weights)
    if dtype is not None:
        weights = {k: v.astype(dtype) for k, v in weights.items()}
    model.load_weights(list(weights.items()), strict=True)
    if bits:
        nn.quantize(
            model,
            group_size=64,
            bits=bits,
            class_predicate=lambda p, m: isinstance(m, nn.Linear)
            and m.weight.shape[-1] % 64 == 0,
        )
    mx.eval(model.parameters())
    return model, cfg


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


@dataclass
class MediaConfig:
    patch_size: int = 16
    temporal_patch_size: int = 2
    merge_size: int = 2
    min_pixels: int = 8192
    max_pixels: int = 8388608
    mean: tuple = IMAGENET_MEAN
    std: tuple = IMAGENET_STD
    video_fps: float = 1.0
    video_min_frames: int = 8
    video_max_frames: int = 3600
    video_total_pixels: int = 268435456
    # Local cap on one video's tokens (Xiaomi's serving allows ~131k; the split's
    # context is ~256k for everything). GOOSE_MIMO_VIDEO_MAX_TOKENS.
    video_max_tokens: int = 16384

    @classmethod
    def from_model_dir(cls, model_dir) -> "MediaConfig":
        """Xiaomi's serving values live in config.json `processor_config` (what SGLang reads:
        image/video min/max pixels, fps, max frames, total video pixels). A converted build's
        preprocessor_config.json may differ from upstream's (EigenLabs ships CLIP stats and
        3136 / 12845056 pixels where XiaomiMiMo's file has ImageNet stats and 8192 / 8388608),
        so only patch/merge/temporal sizes come from it. The normalisation default is the
        upstream file's ImageNet; SGLang switched to CLIP (PR #40888) and vLLM keeps ImageNet:
        GOOSE_MIMO_IMAGE_NORM=clip|imagenet decides until a live colour A/B settles it."""
        c = cls()
        p = Path(model_dir) / "preprocessor_config.json"
        if p.exists():
            d = json.loads(p.read_text())
            c.patch_size = d.get("patch_size", c.patch_size)
            c.temporal_patch_size = d.get("temporal_patch_size", c.temporal_patch_size)
            c.merge_size = d.get("merge_size", c.merge_size)
        try:
            pc = json.loads((Path(model_dir) / "config.json").read_text()).get("processor_config") or {}
        except Exception:
            pc = {}
        c.min_pixels = pc.get("image_min_pixels") or c.min_pixels
        c.max_pixels = pc.get("image_max_pixels") or c.max_pixels
        c.video_fps = pc.get("fps") or c.video_fps
        c.video_max_frames = pc.get("max_frames") or c.video_max_frames
        c.video_min_frames = pc.get("min_frames") or c.video_min_frames
        c.video_total_pixels = pc.get("video_total_max_pixels") or c.video_total_pixels
        norm = os.environ.get("GOOSE_MIMO_IMAGE_NORM", "").lower()
        if norm == "clip":
            c.mean, c.std = CLIP_MEAN, CLIP_STD
        elif norm == "imagenet":
            c.mean, c.std = IMAGENET_MEAN, IMAGENET_STD
        if os.environ.get("GOOSE_MIMO_IMAGE_MAX_PIXELS"):
            c.max_pixels = int(os.environ["GOOSE_MIMO_IMAGE_MAX_PIXELS"])
        if os.environ.get("GOOSE_MIMO_VIDEO_MAX_TOKENS"):
            c.video_max_tokens = int(os.environ["GOOSE_MIMO_VIDEO_MAX_TOKENS"])
        return c

    @property
    def factor(self) -> int:
        return self.patch_size * self.merge_size


def smart_resize(height, width, factor, min_pixels, max_pixels):
    """Xiaomi/SGLang/Qwen2-VL smart_resize (sglang processors/mimo_v2.py)."""
    if min(height, width) < factor:
        scale = factor / min(height, width)
        height, width = int(round(height * scale)), int(round(width * scale))
    elif max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h = round(height / factor) * factor
    w = round(width / factor) * factor
    if h * w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h = math.floor(height / beta / factor) * factor
        w = math.floor(width / beta / factor) * factor
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return int(h), int(w)


def patchify(frames: np.ndarray, c: MediaConfig):
    """[T, C, H, W] (T a multiple of temporal_patch_size) -> ([N, C*tp*ps*ps], (t, h, w))
    in Qwen2-VL order (merge units contiguous)."""
    T, C, H, W = frames.shape
    tp, ps, m = c.temporal_patch_size, c.patch_size, c.merge_size
    gt, gh, gw = T // tp, H // ps, W // ps
    x = frames.reshape(gt, tp, C, gh // m, m, ps, gw // m, m, ps)
    x = x.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return np.ascontiguousarray(x.reshape(gt * gh * gw, C * tp * ps * ps)), (gt, gh, gw)


def _normalise(img, c: MediaConfig, size):
    from PIL import Image

    img = img.convert("RGB")
    if img.size != (size[1], size[0]):
        img = img.resize((size[1], size[0]), Image.BICUBIC)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - np.array(c.mean, np.float32)) / np.array(c.std, np.float32)
    return a.transpose(2, 0, 1)


def image_patches(img, c: MediaConfig):
    """PIL image -> (patches, grid (1, gh, gw))."""
    from PIL import Image

    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, rgba)
    w, h = img.size
    size = smart_resize(h, w, c.factor, c.min_pixels, c.max_pixels)
    a = _normalise(img, c, size)
    frames = np.repeat(a[None], c.temporal_patch_size, axis=0)
    return patchify(frames, c)


def tokens_for(grid, c: MediaConfig) -> int:
    t, h, w = grid
    return t * h * w // (c.merge_size**2)


def decode_data_url(url: str) -> bytes:
    if url.startswith("data:"):
        return base64.b64decode(url.split(",", 1)[1])
    if url.startswith("file://"):
        return Path(url[7:]).read_bytes()
    if url.startswith(("http://", "https://")):
        import urllib.request

        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read()
    # bare base64
    return base64.b64decode(url)


def format_timestamp(ts: float) -> str:
    return f"{int(ts // 60):02d}:{int(ts % 60):02d}"


def _ffprobe(path):
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_read_packets",
            "-show_entries", "format=duration", "-of", "json", path,
        ],
        capture_output=True, check=True, timeout=120,
    )
    d = json.loads(out.stdout)
    s = d["streams"][0]
    num, den = (s.get("avg_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    frames = int(s.get("nb_read_packets") or 0)
    dur = float(d.get("format", {}).get("duration") or (frames / fps if fps else 0))
    return int(s["width"]), int(s["height"]), fps or (frames / dur if dur else 1.0), frames, dur


def video_pairs(data: bytes, c: MediaConfig):
    """Video bytes -> list of (timestamp_seconds, patches, grid (1, gh, gw)), one per
    frame pair. Sampling: SGLang's smart_nframes (fps 1, 8..3600, even), linspace."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".bin") as f:
        f.write(data)
        f.flush()
        W, H, fps, total, dur = _ffprobe(f.name)
        total = total or max(1, int(round(dur * fps)))
        n = int(round(total / fps * c.video_fps)) if fps else c.video_min_frames
        n = max(c.video_min_frames, min(c.video_max_frames, n, total))
        n = max(2, n - n % 2)
        idx = np.unique(np.linspace(0, total - 1, num=n).astype(np.int64))
        if len(idx) % 2:
            idx = np.append(idx, idx[-1])
        per_frame = max(
            c.min_pixels, min(c.max_pixels, c.video_total_pixels * c.temporal_patch_size // len(idx))
        )
        # local token cap: pairs * tokens_per_pair <= video_max_tokens
        pairs = len(idx) // 2
        cap = c.video_max_tokens * (c.factor**2) // max(1, pairs)
        per_frame = max(c.min_pixels, min(per_frame, cap))
        h, w = smart_resize(H, W, c.factor, c.min_pixels, per_frame)
        sel = "+".join(f"eq(n\\,{int(i)})" for i in sorted(set(idx.tolist())))
        raw = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", f.name, "-vf",
                f"select='{sel}',scale={w}:{h}:flags=bicubic", "-vsync", "0",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
            ],
            capture_output=True, check=True, timeout=600,
        ).stdout
    got = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    uniq = sorted(set(idx.tolist()))
    by_index = {i: got[min(k, len(got) - 1)] for k, i in enumerate(uniq)}
    frames = np.stack([by_index[int(i)] for i in idx]).astype(np.float32) / 255.0
    frames = (frames - np.array(c.mean, np.float32)) / np.array(c.std, np.float32)
    frames = frames.transpose(0, 3, 1, 2)
    out = []
    for p in range(0, len(idx), 2):
        patches, grid = patchify(frames[p : p + 2], c)
        out.append((float(idx[p]) / (fps or 1.0), patches, grid))
    return out


# ---------------------------------------------------------------------------
# Markers and the feature store
# ---------------------------------------------------------------------------


def markers(key: str, n: int, vocab_size: int) -> np.ndarray:
    span = vocab_size - MARKER_BASE
    if span < 64:
        raise ValueError(f"vocab {vocab_size} leaves no marker rows past {MARKER_BASE}")
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
    return (MARKER_BASE + np.random.default_rng(seed).integers(0, span, n)).astype(np.int64)


def media_key(kind: str, patches: np.ndarray, grid, c: MediaConfig) -> str:
    h = hashlib.sha256()
    h.update(kind.encode())
    h.update(np.asarray(grid, np.int64).tobytes())
    h.update(np.asarray(c.mean + c.std, np.float32).tobytes())
    h.update(patches.astype(np.float16).tobytes())
    return h.hexdigest()[:32]


def feats_to_wire(f: mx.array) -> tuple:
    f = f.astype(mx.bfloat16)
    return (bytes(np.array(f.view(mx.uint16))), tuple(f.shape))


def feats_from_wire(w: tuple) -> mx.array:
    data, shape = w
    return mx.array(np.frombuffer(data, np.uint16).reshape(shape)).view(mx.bfloat16)


class MediaStore:
    """Features by media key on the process that runs layer 0."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.items: "OrderedDict[str, tuple[np.ndarray, mx.array]]" = OrderedDict()
        self.lock = threading.Lock()

    def put(self, key, feats: mx.array):
        with self.lock:
            self.items[key] = (markers(key, feats.shape[0], self.vocab_size), feats)
            self.items.move_to_end(key)

    def has(self, key) -> bool:
        return key in self.items

    def evict(self, keys):
        with self.lock:
            for k in keys:
                self.items.pop(k, None)

    def _locate(self, run: np.ndarray, at_start: bool, at_end: bool):
        L = len(run)
        for key, (mk, feats) in reversed(self.items.items()):
            N = len(mk)
            if L > N:
                continue
            if not at_start:  # preceded by a non-marker: the run is this image's start
                if np.array_equal(mk[:L], run):
                    return feats, 0
            elif not at_end:  # followed by a non-marker: its end
                if np.array_equal(mk[N - L :], run):
                    return feats, N - L
            else:
                for o in np.flatnonzero(mk[: N - L + 1] == run[0]):
                    if np.array_equal(mk[o : o + L], run):
                        return feats, int(o)
        return None, None

    def substitute(self, inputs: mx.array, h: mx.array) -> mx.array:
        """Replace marker rows of a prefill chunk with their features."""
        if not self.items or inputs.ndim != 2 or inputs.shape[1] < 2:
            return h
        ids = np.array(inputs)
        hit = ids >= MARKER_BASE
        if not hit.any():
            return h
        rows, cols, vals = [], [], []
        B, L = ids.shape
        for b in range(B):
            m = hit[b]
            if not m.any():
                continue
            edges = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(np.int8), [0]])))
            for s, e in zip(edges[0::2], edges[1::2]):
                feats, off = self._locate(ids[b, s:e], at_start=(s == 0), at_end=(e == L))
                if feats is None:
                    raise RuntimeError(
                        f"mimo media: a run of {e - s} marker tokens has no stored features "
                        "(the image was evicted or never shipped)"
                    )
                rows.append(np.full(e - s, b))
                cols.append(np.arange(s, e))
                vals.append(feats[off : off + (e - s)])
        rows = mx.array(np.concatenate(rows))
        cols = mx.array(np.concatenate(cols))
        h[rows, cols] = mx.concatenate(vals, axis=0).astype(h.dtype)
        return h


STORE: Optional[MediaStore] = None


def store(vocab_size: int) -> MediaStore:
    global STORE
    if STORE is None:
        STORE = MediaStore(vocab_size)
    return STORE


# ---------------------------------------------------------------------------
# Request side: content parts -> text layout + manifest; token expansion
# ---------------------------------------------------------------------------

MEDIA_KEY = "_goose_media"


def part_kind(part) -> Optional[str]:
    if not isinstance(part, dict):
        return None
    t = part.get("type")
    if t in ("image_url", "image") or "image_url" in part:
        return "image"
    if t in ("video_url", "video") or "video_url" in part:
        return "video"
    if t in ("input_audio", "audio_url", "audio") or "input_audio" in part or "audio_url" in part:
        return "audio"
    return None


def part_url(part) -> str:
    for k in ("image_url", "video_url", "audio_url"):
        v = part.get(k)
        if isinstance(v, dict):
            return v.get("url", "")
        if isinstance(v, str):
            return v
    v = part.get("image") or part.get("video")
    return v if isinstance(v, str) else ""


class MediaFrontend:
    """Runs where requests arrive (rank 0): decodes media, encodes it with the vision
    tower, rewrites each media part as text in the model's layout, and returns the
    manifest the token expansion and the layer-0 store need.

    ``peer`` mirrors what the layer-0 process holds (it may be this process): features
    are shipped only for keys it lacks, evictions are explicit, so both sides agree
    whatever order requests interleave in.
    """

    def __init__(self, model_dir, vocab_size: int, local_is_layer0: bool,
                 peer_budget_bytes: int = 512 << 20, cache_bytes: int = 1 << 30):
        self.model_dir = Path(model_dir)
        self.vocab_size = vocab_size
        self.local_is_layer0 = local_is_layer0
        self.cfg = MediaConfig.from_model_dir(model_dir)
        self.vision = None
        self.cache: "OrderedDict[str, mx.array]" = OrderedDict()
        self.cache_bytes = cache_bytes
        self.peer: "OrderedDict[str, int]" = OrderedDict()
        self.peer_budget = peer_budget_bytes

    def _vision(self):
        if self.vision is None:
            bits = int(os.environ.get("GOOSE_MIMO_VISION_BITS", "0") or 0)
            self.vision, _ = load_vision(self.model_dir, bits=bits)
        return self.vision

    def _encode(self, key, patches, grid) -> mx.array:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        feats = self._vision()(mx.array(patches), [grid]).astype(mx.bfloat16)
        mx.eval(feats)
        self.cache[key] = feats
        while sum(v.nbytes for v in self.cache.values()) > self.cache_bytes and len(self.cache) > 1:
            self.cache.popitem(last=False)
        return feats

    def prepare(self, messages: list) -> Optional[dict]:
        items, texts = [], []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            new = []
            for part in content:
                kind = part_kind(part)
                if kind is None:
                    new.append(part)
                    continue
                if kind == "audio":
                    raise ValueError("mimo media: audio input is not wired on this engine yet")
                data = decode_data_url(part_url(part))
                if kind == "image":
                    from PIL import Image

                    patches, grid = image_patches(Image.open(io.BytesIO(data)), self.cfg)
                    key = media_key("image", patches, grid, self.cfg)
                    feats = self._encode(key, patches, grid)
                    items.append(("image", key, int(feats.shape[0]), feats))
                    new.append({"type": "text",
                                "text": "<|vision_start|><|image_pad|><|vision_end|>"})
                else:
                    text = "<|mimo_video_start|>"
                    for ts, patches, grid in video_pairs(data, self.cfg):
                        key = media_key("video", patches, grid, self.cfg)
                        feats = self._encode(key, patches, grid)
                        items.append(("video", key, int(feats.shape[0]), feats))
                        text += format_timestamp(ts) + "<|vision_start|><|video_pad|><|vision_end|>"
                    text += "<|mimo_video_end|>"
                    new.append({"type": "text", "text": text})
            msg["content"] = new
        if not items:
            return None
        manifest = {"items": [(k, key, n) for k, key, n, _ in items], "ship": {}, "evict": []}
        if self.local_is_layer0:
            st = store(self.vocab_size)
            for _, key, _, feats in items:
                st.put(key, feats)
            needed = {key for _, key, _, _ in items}
            # keep the store bounded the same way as a peer's
            self._account(manifest, needed, items, ship=False)
            st.evict(manifest["evict"])
        else:
            needed = {key for _, key, _, _ in items}
            self._account(manifest, needed, items, ship=True)
        return manifest

    def _account(self, manifest, needed, items, ship):
        for _, key, _, feats in items:
            if key in self.peer:
                self.peer.move_to_end(key)
                continue
            if ship:
                manifest["ship"][key] = feats_to_wire(feats)
            self.peer[key] = feats.nbytes
        while sum(self.peer.values()) > self.peer_budget:
            victim = next((k for k in self.peer if k not in needed), None)
            if victim is None:
                break
            self.peer.pop(victim)
            manifest["evict"].append(victim)


def apply_manifest(manifest: dict, vocab_size: int, is_layer0: bool, shipped_locally: bool):
    """On the layer-0 process: store what was shipped, drop what was evicted."""
    if not is_layer0 or shipped_locally:
        return
    st = store(vocab_size)
    st.evict(manifest.get("evict", []))
    for key, wire in manifest.get("ship", {}).items():
        st.put(key, feats_from_wire(wire))
    missing = [key for _, key, _ in manifest["items"] if not st.has(key)]
    if missing:
        raise RuntimeError(f"mimo media: features missing for {len(missing)} item(s)")


def expand(tokens: list, manifest: dict, vocab_size: int, require_all: bool) -> list:
    """Each IMAGE_PAD / VIDEO_PAD placeholder -> its item's marker ids, in order."""
    pad_kind = {IMAGE_PAD: "image", VIDEO_PAD: "video", AUDIO_PAD: "audio"}
    items = manifest["items"]
    out, i = [], 0
    for t in tokens:
        kind = pad_kind.get(int(t))
        if kind is None:
            out.append(t)
            continue
        if i >= len(items) or items[i][0] != kind:
            raise ValueError(
                f"mimo media: placeholder {i} is {kind} but the request's media list does not "
                "match the rendered prompt"
            )
        _, key, n = items[i]
        out.extend(int(x) for x in markers(key, n, vocab_size))
        i += 1
    if require_all and i != len(items):
        raise ValueError(f"mimo media: {len(items)} media items, {i} placeholders rendered")
    return out
