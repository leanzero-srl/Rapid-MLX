# SPDX-License-Identifier: Apache-2.0
"""Image input for the Qwen4-Exp text decoder in :mod:`.qwen4_exp`.

Everything architectural is reused from mlx-vlm's ``qwen4_exp`` package (the
same package the single-node MLLM lane serves Flash-Next with): the vision
tower is its Qwen3-VL ViT, the preprocessing and ``<|image_pad|>`` expansion
are its numpy ``Qwen3VLProcessor``, and the multimodal RoPE table is Qwen3.5's
``get_rope_index``.  What lives here is the glue the text decoder needs:

* :func:`load_vision_tower` builds the tower from the checkpoint's
  ``vision_config`` and loads only the ``model.visual.*`` tensors, quantized
  per the checkpoint's own per-module quantization entries;
* :func:`merge_image_features` puts the tower's output rows where the prompt
  carries image tokens, before layer 0 (the decoder itself never sees pixels);
* :class:`MRopePositions` maps a sequence's logical token positions to the
  (t, h, w) RoPE positions the attention and QSA-indexer layers rotate by —
  image tokens carry 2-D grid positions, and every token after an image is
  shifted by the image's ``rope_delta``.  It is plain integers, so a pipeline
  rank that never sees the image still rotates exactly like rank 0.

The single engine's TEXT lane serves images too (:class:`TextLaneVision`), so
an image request keeps MTP, prompt lookup, the PLE offload, continuous
batching and the prefix cache.  There the image reaches the decoder as token
ids: each image's ``<|image_pad|>`` becomes one id per image token, drawn
from a range above the vocabulary (:data:`IMAGE_ID_BASE`) that names the
image (its content-hash slot) and the token's index inside it.  The prefix
cache keys on those ids, so the same picture hits and a different one never
does; the decoder maps them back to ``image_token_id`` for the embedding and
the PLE tables, writes the tower's features over those rows, and keeps the
image layout per batch row in the QSA caches (:class:`ImageSpan`), from which
:class:`SpanRopePositions` derives every (t, h, w) RoPE position.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

VISION_KEY_PREFIXES = ("model.visual.", "vision_tower.")
# measured: the tower's encode peak over the float32 attention scores of one
# image (heads x patches^2 x 4 B) — 1.243 at 3,920 patches (the processor's
# max_pixels, 1,003,520 px) on Qwen3.8-Flash-Next-4bit, M4 Max, 2026-09-24;
# 1.139 GiB absolute, and every smaller image peaked lower (1.02-1.10 GiB).
VISION_ENCODE_PEAK_MULTIPLE = 1.25


def checkpoint_config(model_dir: Path) -> dict[str, Any]:
    return json.loads((Path(model_dir) / "config.json").read_text())


def declares_vision(config: dict[str, Any]) -> bool:
    return bool(config.get("vision_config")) and not config.get(
        "language_model_only", False
    )


def is_vision_key(key: str) -> bool:
    return key.startswith(VISION_KEY_PREFIXES)


def _vision_shards(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map: dict[str, str] = json.loads(index.read_text())["weight_map"]
        names = sorted({name for key, name in weight_map.items() if is_vision_key(key)})
    else:
        names = sorted(path.name for path in model_dir.glob("model*.safetensors"))
    return [model_dir / name for name in names]


@dataclass
class VisionTower:
    """The loaded tower plus the processor that feeds it and the token ids."""

    model: nn.Module
    image_processor: Any
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    spatial_merge_size: int
    mrope_section: tuple[int, ...]

    @property
    def parameter_bytes(self) -> int:
        from mlx.utils import tree_flatten

        return sum(value.nbytes for _, value in tree_flatten(self.model.parameters()))

    def processor(self, tokenizer) -> Any:
        """mlx-vlm's Qwen3VLProcessor over THIS server's tokenizer.

        Its ``__call__`` expands each ``<|image_pad|>`` into the image's token
        count and tokenizes, so the ids are the serving tokenizer's own.
        """
        from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

        hf_tokenizer = getattr(tokenizer, "_tokenizer", tokenizer)
        return Qwen3VLProcessor(
            image_processor=self.image_processor,
            tokenizer=hf_tokenizer,
            video_processor=None,
            chat_template=getattr(hf_tokenizer, "chat_template", None),
        )

    def encode(self, pixel_values: mx.array, grid_thw: list[list[int]]) -> mx.array:
        """``[image tokens, hidden]`` for every image, in prompt order.

        One image per forward: the tower's attention transient grows with the
        square of an image's patches, and the planner reserves one image's.
        """
        dtype = self.model.patch_embed.proj.weight.dtype
        features = []
        start = 0
        for grid in grid_thw:
            patches = grid[0] * grid[1] * grid[2]
            out, _ = self.model(
                pixel_values[start : start + patches].astype(dtype),
                mx.array([grid], dtype=mx.int32),
            )
            mx.eval(out)
            features.append(out)
            start += patches
        if start != pixel_values.shape[0]:
            raise ValueError(
                f"pixel rows {pixel_values.shape[0]} do not match the grids' {start}"
            )
        return mx.concatenate(features, axis=0)

    def rope_positions(
        self, ids: list[int], grid_thw: list[list[int]]
    ) -> tuple[list[list[int]], int]:
        """Qwen3.5's ``get_rope_index`` for one unpadded row: ``(3 x L table, delta)``."""
        from mlx_vlm.models.qwen3_5.language import LanguageModel as Qwen35Language

        owner = SimpleNamespace(
            config=SimpleNamespace(
                vision_config=SimpleNamespace(
                    spatial_merge_size=self.spatial_merge_size
                ),
                image_token_id=self.image_token_id,
                video_token_id=self.video_token_id,
                vision_start_token_id=self.vision_start_token_id,
            )
        )
        positions, deltas = Qwen35Language.get_rope_index(
            owner,
            mx.array([ids], dtype=mx.int32),
            mx.array(grid_thw, dtype=mx.int32) if grid_thw else None,
        )
        if positions.ndim == 2:
            positions = mx.broadcast_to(positions[None], (3, *positions.shape))
        return positions[:, 0, :].tolist(), int(deltas.reshape(-1)[0].item())


def load_vision_tower(model_dir: Path) -> VisionTower | None:
    """The checkpoint's vision tower, or None when it declares none.

    A checkpoint that declares a vision config but ships no ``model.visual``
    tensors is refused by name — it would otherwise serve images it cannot see.
    """
    require_vision_runtime()
    from mlx_vlm.models.qwen4_exp import VisionConfig, VisionModel

    model_dir = Path(model_dir)
    config = checkpoint_config(model_dir)
    if not declares_vision(config):
        return None
    vision_config = VisionConfig.from_dict(config["vision_config"])
    tower = VisionModel(vision_config)
    weights: dict[str, mx.array] = {}
    for shard in _vision_shards(model_dir):
        for key, value in mx.load(str(shard)).items():
            for prefix in VISION_KEY_PREFIXES:
                if key.startswith(prefix):
                    weights[key[len(prefix) :]] = value
    if not weights:
        raise ValueError(
            f"{model_dir} declares a vision_config but ships no "
            f"{' / '.join(VISION_KEY_PREFIXES)} tensors"
        )
    weights = tower.sanitize(weights)
    quantization = config.get("quantization") or config.get("quantization_config")
    if quantization:

        def predicate(path: str, module: nn.Module):
            entry = quantization.get(f"vision_tower.{path}")
            if isinstance(entry, dict):
                return entry
            return hasattr(module, "to_quantized") and f"{path}.scales" in weights

        nn.quantize(
            tower,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=predicate,
        )
    tower.load_weights(list(weights.items()), strict=True)
    mx.eval(tower.parameters())
    tower.eval()

    image_processor = _image_processor(model_dir)
    declared = (
        vision_config.patch_size,
        vision_config.temporal_patch_size,
        vision_config.spatial_merge_size,
    )
    preprocessed = (
        image_processor.patch_size,
        image_processor.temporal_patch_size,
        image_processor.merge_size,
    )
    if declared != preprocessed:
        raise ValueError(
            f"image processor (patch, temporal, merge) {preprocessed} does not match "
            f"the checkpoint's vision_config {declared}"
        )
    text_config = config.get("text_config", config)
    section = tuple(
        (text_config.get("rope_parameters") or {}).get("mrope_section") or ()
    )
    if not section:
        raise ValueError(
            f"{model_dir} declares vision but its text_config has no mrope_section"
        )
    return VisionTower(
        model=tower,
        image_processor=image_processor,
        image_token_id=int(config["image_token_id"]),
        video_token_id=int(config["video_token_id"]),
        vision_start_token_id=int(config["vision_start_token_id"]),
        spatial_merge_size=int(vision_config.spatial_merge_size),
        mrope_section=section,
    )


def require_vision_runtime() -> None:
    """mlx-vlm carries the tower, the processor and the RoPE index; say so by name."""
    try:
        import mlx_vlm  # noqa: F401
    except ImportError as missing:
        raise RuntimeError(
            "this checkpoint declares a vision tower but mlx-vlm is not installed in "
            "this interpreter (pip install mlx-vlm==0.7.1), or serve it --text-only"
        ) from missing


def _image_processor(model_dir: Path):
    require_vision_runtime()
    from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import (
        Qwen3VLImageProcessor,
        _qwen_vl_image_kwargs,
    )

    return Qwen3VLImageProcessor(
        **_qwen_vl_image_kwargs(str(model_dir), default_patch_size=16)
    )


def vision_workspace_bytes(model_dir: Path) -> int:
    """Encode transient of the largest image the processor will produce.

    ``encode`` runs one image per forward, and the processor resizes every
    image to at most ``max_pixels``, so this bounds every request.
    """
    config = checkpoint_config(model_dir)
    processor = _image_processor(model_dir)
    patches = processor.max_pixels // processor.patch_size**2
    heads = int(config["vision_config"]["num_heads"])
    return int(heads * patches * patches * 4 * VISION_ENCODE_PEAK_MULTIPLE)


def merge_image_features(
    embeddings: mx.array, ids: mx.array, features: mx.array, image_token_id: int
) -> mx.array:
    """Scatter ``features`` rows onto the image-token positions of ``embeddings``.

    ``embeddings`` is ``[1, L, hidden]`` for the unpadded row whose ids are
    ``ids`` ``[L]``; the count of image tokens must equal the feature rows.
    """
    mask = ids == image_token_id
    count = int(mask.sum().item())
    if count != features.shape[0]:
        raise ValueError(
            f"image features and image tokens do not match: {count} tokens, "
            f"{features.shape[0]} feature rows"
        )
    index = mx.cumsum(mask.astype(mx.int32)) - 1
    gathered = features.astype(embeddings.dtype)[mx.maximum(index, 0)]
    return mx.where(mask[None, :, None], gathered[None], embeddings)


@dataclass(frozen=True)
class MRopePositions:
    """Logical token position -> multimodal RoPE position, per batch row.

    ``table[:, b, p]`` is the (t, h, w) position of row ``b``'s logical token
    ``p`` for ``p < lengths[b]`` (the prompt); every later position ``p``
    rotates at ``p + deltas[b]`` on all three axes, which is how Qwen3-VL
    decodes after an image.  A row without images carries ``table = p`` and
    ``delta = 0``.  Logical positions are the ones the caches already use:
    0 at the row's first real token, negative across left padding.
    """

    table: mx.array
    lengths: mx.array
    deltas: mx.array

    @classmethod
    def from_rows(cls, rows: list[tuple[list[list[int]], int]]) -> MRopePositions:
        width = max(len(table[0]) for table, _ in rows)
        padded = [
            [axis + [0] * (width - len(axis)) for axis in table] for table, _ in rows
        ]
        return cls(
            table=mx.array(padded, dtype=mx.int64).transpose(1, 0, 2),
            lengths=mx.array([len(table[0]) for table, _ in rows], dtype=mx.int64),
            deltas=mx.array([delta for _, delta in rows], dtype=mx.int64),
        )

    def at(self, logical: mx.array) -> mx.array:
        """``[B, L]`` logical positions -> ``[3, B, L]`` RoPE positions."""
        return self._lookup(self.table, self.lengths, self.deltas, logical)

    def at_row(self, row: int, logical: mx.array) -> mx.array:
        """``[1, N]`` logical positions of one row -> ``[3, 1, N]``."""
        return self._lookup(
            self.table[:, row : row + 1],
            self.lengths[row : row + 1],
            self.deltas[row : row + 1],
            logical,
        )

    @staticmethod
    def _lookup(table, lengths, deltas, logical):
        logical = logical.astype(mx.int64)
        batch, length = logical.shape
        index = mx.clip(logical, 0, table.shape[2] - 1)
        gathered = mx.take_along_axis(
            table, mx.broadcast_to(index[None], (3, batch, length)), axis=2
        )
        inside = (logical >= 0) & (logical < lengths[:, None])
        shifted = logical + deltas[:, None]
        return mx.where(inside[None], gathered, shifted[None])


# ---------------------------------------------------------------------------
# Text-lane image input (single engine)
# ---------------------------------------------------------------------------

# Image-token ids live above every vocabulary this family ships (248,320):
# ``IMAGE_ID_BASE + slot * IMAGE_ID_STRIDE + k`` is token ``k`` of the image
# whose content hash owns ``slot``.  int32 holds 130,048 slots of 16,384
# tokens (the processor's max_pixels gives <= 980 tokens per image).
IMAGE_ID_BASE = 1 << 24
IMAGE_ID_STRIDE = 1 << 14
IMAGE_ID_MAX_SLOTS = ((1 << 31) - IMAGE_ID_BASE) // IMAGE_ID_STRIDE
# Encoded features kept for prefix-cache misses on recent images
# (Flash: <= 980 x 2560 bf16 = 5 MB each).
FEATURE_CACHE_ENTRIES = 32


@dataclass(frozen=True)
class ImageSpan:
    """One image inside a sequence: logical ``start``, ``count`` tokens,
    the merged grid ``height`` x ``width`` and its rotary origin ``origin``
    (= ``start`` + the delta of every earlier image)."""

    start: int
    count: int
    height: int
    width: int
    origin: int

    @property
    def end(self) -> int:
        return self.start + self.count

    @property
    def delta(self) -> int:
        # Qwen3-VL ``get_rope_index`` for a one-frame image: the next text
        # token rotates at origin + max(h, w), i.e. its logical position
        # shifted by max(h, w) - h * w.
        return max(self.height, self.width) - self.count

    def to_json(self) -> list[int]:
        return [self.start, self.count, self.height, self.width, self.origin]

    @classmethod
    def from_json(cls, value) -> ImageSpan:
        return cls(*(int(item) for item in value))


def delta_before(spans: list[ImageSpan], position: int) -> int:
    """RoPE shift of a text token at logical ``position``."""
    return sum(span.delta for span in spans if span.end <= position)


def rope_position(spans: list[ImageSpan], position: int) -> tuple[int, int, int]:
    """(t, h, w) RoPE position of one logical position (host reference)."""
    for span in spans:
        if span.start <= position < span.end:
            k = position - span.start
            return (
                span.origin,
                span.origin + k // span.width,
                span.origin + k % span.width,
            )
    shifted = position + delta_before(spans, position)
    return (shifted, shifted, shifted)


def drop_spans_from(spans: list[ImageSpan], offset: int) -> list[ImageSpan]:
    """Spans still owned by a sequence trimmed back to ``offset`` tokens."""
    return [span for span in spans if span.start < offset]


class SpanRopePositions:
    """``rope_positions`` for :mod:`.qwen4_exp` built from per-row image spans.

    Same contract as :class:`MRopePositions` (``at`` / ``at_row``), but it
    covers every logical position, prompt and decode alike, from a handful of
    integers per image rather than a prompt-length table.
    """

    def __init__(self, rows: list[list[ImageSpan]]):
        self.rows = [list(row) for row in rows]
        width = max((len(row) for row in self.rows), default=0)
        width = max(width, 1)
        # Padding spans start beyond any position and hold no tokens: never
        # inside, never "after" (end = start), delta 0.
        never = 1 << 40

        def column(getter, pad):
            return mx.array(
                [
                    [getter(span) for span in row] + [pad] * (width - len(row))
                    for row in self.rows
                ],
                dtype=mx.int64,
            )

        self.start = column(lambda s: s.start, never)
        self.count = column(lambda s: s.count, 0)
        self.width = column(lambda s: s.width, 1)
        self.origin = column(lambda s: s.origin, 0)
        self.delta = column(lambda s: s.delta, 0)

    def at(self, logical: mx.array) -> mx.array:
        """``[B, L]`` logical positions -> ``[3, B, L]`` RoPE positions."""
        return self._lookup(
            logical, self.start, self.count, self.width, self.origin, self.delta
        )

    def at_row(self, row: int, logical: mx.array) -> mx.array:
        """``[1, N]`` logical positions of one row -> ``[3, 1, N]``."""
        pick = slice(row, row + 1)
        return self._lookup(
            logical,
            self.start[pick],
            self.count[pick],
            self.width[pick],
            self.origin[pick],
            self.delta[pick],
        )

    @staticmethod
    def _lookup(logical, start, count, width, origin, delta):
        p = logical.astype(mx.int64)[..., None]  # [B, L, 1]
        start = start[:, None, :]
        end = start + count[:, None, :]
        inside = (p >= start) & (p < end)
        after = (p >= end) & (count[:, None, :] > 0)
        shifted = p[..., 0] + (after.astype(mx.int64) * delta[:, None, :]).sum(-1)
        k = mx.maximum(p - start, 0)
        w = width[:, None, :]
        o = origin[:, None, :]
        hit = inside.astype(mx.int64)
        any_inside = inside.any(axis=-1)
        t_pos = (hit * o).sum(-1)
        h_pos = (hit * (o + k // w)).sum(-1)
        w_pos = (hit * (o + k % w)).sum(-1)
        return mx.stack(
            [
                mx.where(any_inside, t_pos, shifted),
                mx.where(any_inside, h_pos, shifted),
                mx.where(any_inside, w_pos, shifted),
            ]
        )


@dataclass
class _RegisteredImage:
    digest: str
    slot: int
    data: bytes  # the encoded file; pixels are re-derived when encoding
    grid_thw: tuple[int, int, int]  # merged (t, h, w)

    @property
    def merged_hw(self) -> tuple[int, int]:
        return (self.grid_thw[1], self.grid_thw[2])


class TextLaneVision:
    """Image input for a Qwen4-Exp checkpoint on the engine's text lane.

    Host side (any thread, numpy only): :meth:`expand_prompt` loads each
    image, preprocesses it and replaces the prompt's ``<|image_pad|>`` ids by
    the image's slot ids.  Model side (the MLX worker): :meth:`prepare`
    turns a forward's ids into embeddings with the image features written in,
    the ids the PLE tables read, and the batch's ``rope_positions``.
    """

    def __init__(self, tower: VisionTower):
        self.tower = tower
        self.merge = tower.spatial_merge_size
        self._lock = threading.Lock()
        self._by_digest: dict[str, _RegisteredImage] = {}
        self._by_slot: dict[int, _RegisteredImage] = {}
        self._features: OrderedDict[int, mx.array] = OrderedDict()
        self.images_registered = 0
        self.images_encoded = 0
        self.image_requests = 0

    # -- host side ---------------------------------------------------------

    def _preprocess(self, data: bytes) -> dict[str, Any]:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(data)) as opened:
            picture = opened.convert("RGB")
        return self.tower.image_processor(images=[picture])

    @property
    def used(self) -> bool:
        return bool(self._by_slot)

    def register(self, path: str) -> _RegisteredImage:
        """Slot for the image file at ``path`` (content-addressed)."""
        data = Path(path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        with self._lock:
            known = self._by_digest.get(digest)
            if known is not None:
                return known
        processed = self._preprocess(data)
        grid = tuple(int(v) for v in processed["image_grid_thw"][0])
        if grid[0] != 1:
            raise ValueError(f"an image must preprocess to one frame, got {grid}")
        merged = (grid[0], grid[1] // self.merge, grid[2] // self.merge)
        if merged[1] * merged[2] >= IMAGE_ID_STRIDE:
            raise ValueError(
                f"image of {merged[1]}x{merged[2]} tokens exceeds {IMAGE_ID_STRIDE}"
            )
        with self._lock:
            known = self._by_digest.get(digest)
            if known is not None:
                return known
            slot = len(self._by_slot)
            if slot >= IMAGE_ID_MAX_SLOTS:
                raise ValueError("this server has registered too many images")
            entry = _RegisteredImage(
                digest=digest, slot=slot, data=data, grid_thw=merged
            )
            self._by_digest[digest] = entry
            self._by_slot[slot] = entry
            self.images_registered += 1
        return entry

    def expand_prompt(
        self, ids: list[int], image_paths: list[str], prefix_boundary: int = 0
    ) -> tuple[list[int], int]:
        """Replace the j-th ``<|image_pad|>`` by image j's slot ids.

        Returns the expanded ids and ``prefix_boundary`` moved into the
        expanded token space.  A pad/image count mismatch is a ValueError.
        """
        pad = self.tower.image_token_id
        pads = [index for index, token in enumerate(ids) if token == pad]
        if len(pads) != len(image_paths):
            raise ValueError(
                f"the prompt holds {len(pads)} image placeholders for "
                f"{len(image_paths)} images"
            )
        entries = [self.register(path) for path in image_paths]
        out: list[int] = []
        boundary = prefix_boundary
        cursor = 0
        for index, entry in zip(pads, entries):
            out.extend(ids[cursor:index])
            count = entry.grid_thw[1] * entry.grid_thw[2]
            base = IMAGE_ID_BASE + entry.slot * IMAGE_ID_STRIDE
            out.extend(range(base, base + count))
            if index < prefix_boundary:
                boundary += count - 1
            cursor = index + 1
        out.extend(ids[cursor:])
        with self._lock:
            self.image_requests += 1
        return out, boundary

    # -- model side (MLX worker) --------------------------------------------

    def features(self, slot: int) -> mx.array:
        cached = self._features.get(slot)
        if cached is not None:
            self._features.move_to_end(slot)
            return cached
        entry = self._by_slot[slot]
        pixels = self._preprocess(entry.data)["pixel_values"]
        grid = [1, entry.grid_thw[1] * self.merge, entry.grid_thw[2] * self.merge]
        out = self.tower.encode(mx.array(pixels), [grid])
        mx.eval(out)
        self._features[slot] = out
        while len(self._features) > FEATURE_CACHE_ENTRIES:
            self._features.popitem(last=False)
        mx.clear_cache()
        self.images_encoded += 1
        return out

    @staticmethod
    def qsa_caches(cache) -> list[Any]:
        from .qwen4_exp_cache import QSAIndexCache

        found = []
        for layer_cache in cache or ():
            subs = getattr(layer_cache, "caches", None)
            if subs is None:
                continue
            for sub in subs:
                if isinstance(sub, QSAIndexCache):
                    found.append(sub)
        return found

    def map_ids(self, ids: mx.array) -> mx.array:
        """Image slot ids -> ``image_token_id`` (embedding / PLE / MTP)."""
        return mx.where(
            ids >= IMAGE_ID_BASE,
            mx.array(self.tower.image_token_id, dtype=ids.dtype),
            ids,
        )

    def prepare(self, embed_tokens, inputs: mx.array, cache):
        """``(ids, input_embeddings or None, rope_positions or None, commit)``.

        ``commit()`` writes the batch's updated image spans into every QSA
        cache; the caller runs it after the forward succeeds.
        """
        qsa = self.qsa_caches(cache)
        if not qsa:
            return inputs, None, None, None
        batch, length = inputs.shape[0], inputs.shape[1]
        first = qsa[0]
        first._ensure_batch(batch)
        rows = [list(row) for row in first._rope_spans]
        found: list[tuple[int, int, int, int]] = []  # row, column, slot, k
        if length > 1 and self.used:
            host = inputs.tolist()
            for row, tokens in enumerate(host):
                for column, token in enumerate(tokens):
                    if token >= IMAGE_ID_BASE:
                        local = token - IMAGE_ID_BASE
                        found.append(
                            (row, column, local // IMAGE_ID_STRIDE, local % IMAGE_ID_STRIDE)
                        )
        if not found:
            if not any(rows):
                return inputs, None, None, None
            return inputs, None, SpanRopePositions(rows), None

        offsets = list(first._offsets)
        starts = [start for start, _ in first.valid_spans(length)]
        for row, column, slot, k in found:
            entry = self._by_slot.get(slot)
            if entry is None:
                raise ValueError(f"image slot {slot} is not registered")
            height, width = entry.grid_thw[1], entry.grid_thw[2]
            if k >= height * width:
                raise ValueError(f"image token {k} is outside slot {slot}")
            position = offsets[row] + column - starts[row]
            span_start = position - k
            if any(span.start == span_start for span in rows[row]):
                continue
            span = ImageSpan(
                start=span_start,
                count=height * width,
                height=height,
                width=width,
                origin=span_start + delta_before(rows[row], span_start),
            )
            rows[row] = sorted([*rows[row], span], key=lambda item: item.start)

        ids = self.map_ids(inputs)
        embeddings = embed_tokens(ids)
        slots = sorted({slot for _, _, slot, _ in found})
        base = {}
        feature_rows = []
        total = 0
        for slot in slots:
            features = self.features(slot)
            base[slot] = total
            total += features.shape[0]
            feature_rows.append(features)
        table = mx.concatenate(feature_rows, axis=0).astype(embeddings.dtype)
        flat_index = mx.array([row * length + column for row, column, _, _ in found])
        feature_index = mx.array([base[slot] + k for _, _, slot, k in found])
        hidden = embeddings.shape[-1]
        flat = embeddings.reshape(batch * length, hidden)
        flat[flat_index] = table[feature_index]
        embeddings = flat.reshape(batch, length, hidden)

        def commit():
            for qsa_cache in qsa:
                qsa_cache._rope_spans = [list(row) for row in rows]

        return ids, embeddings, SpanRopePositions(rows), commit

    def stats(self) -> dict[str, int]:
        return {
            "images_registered": self.images_registered,
            "images_encoded": self.images_encoded,
            "image_requests": self.image_requests,
            "features_cached": len(self._features),
        }
