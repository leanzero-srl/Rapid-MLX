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
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import mlx.nn as nn

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
            "this interpreter (pip install mlx-vlm==0.7.1), or serve it --no-vision"
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
