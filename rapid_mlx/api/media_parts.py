# SPDX-License-Identifier: Apache-2.0
"""Audio and video content parts on the MLLM chat lane.

Video: every ``video`` / ``video_url`` part becomes a short text label plus one ``image_url``
part per sampled frame, BEFORE the chat template renders. The template then emits one image
placeholder per frame and the scheduler's image list carries exactly those frames, so prompt and
pixels can never disagree. Native video paths were measured broken for Qwen3-Omni on mlx-vlm 0.7.1
and 0.7.11 (2026-10-30, goose omni-media PROGRESS.md); frames work on every vision model.

Audio: every ``input_audio`` part ({"data": base64, "format": "wav"}) is written to a private
temp file and replaced by the Hugging Face marker ``{"type": "audio"}``, which omni chat
templates (Qwen3-Omni, Gemma 4) render as the model's audio placeholder. The file paths travel
to ``mlx_vlm.prepare_inputs(audio=...)`` in prompt order.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import tempfile
from typing import Any

logger = logging.getLogger(__name__)

AUDIO_MARKER_TYPE = "audio"

# Frames sampled per second of video and the cap per video part. One frame per second up to 16
# keeps a 16 s clip under ~4k vision tokens on Qwen-family processors.
VIDEO_FPS = float(os.environ.get("RAPID_MLX_VIDEO_FRAME_FPS", "1.0"))
VIDEO_MAX_FRAMES = int(os.environ.get("RAPID_MLX_VIDEO_MAX_FRAMES", "16"))

_AUDIO_SUFFIX = {
    "wav": ".wav",
    "mp3": ".mp3",
    "flac": ".flac",
    "m4a": ".m4a",
    "ogg": ".ogg",
}


def is_audio_marker(part: Any) -> bool:
    """A bare ``{"type": "audio"}`` part: the rendered placeholder of an extracted clip."""
    return (
        isinstance(part, dict)
        and part.get("type") == AUDIO_MARKER_TYPE
        and part.get("audio") is None
        and part.get("url") is None
    )


def _part_dict(part: Any) -> Any:
    if hasattr(part, "model_dump"):
        return part.model_dump(exclude_none=True)
    return part


def _video_source(part: dict) -> Any:
    if part.get("type") == "video_url":
        value = part.get("video_url")
        if isinstance(value, dict):
            return value.get("url")
        return value
    return part.get("video", part.get("url"))


def _write_audio(data: str, fmt: str) -> str:
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError("input_audio.data is not valid base64") from e
    if not raw:
        raise ValueError("input_audio.data is empty")
    fd, path = tempfile.mkstemp(
        prefix="rapid-mlx-audio-",
        suffix=_AUDIO_SUFFIX.get(fmt.lower(), "." + fmt.lower()),
    )
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return path


def _frames_for(source: Any, fps: float, max_frames: int) -> list[str]:
    """The sampled frames as JPEG data URLs (the image loader refuses arbitrary local paths)."""
    import io

    from PIL import Image

    from ..models.mllm import extract_video_frames_smart, process_video_input

    video_path = process_video_input(source)
    frames = extract_video_frames_smart(video_path, fps=fps, max_frames=max_frames)
    if not frames:
        raise ValueError("the video has no decodable frames")
    urls = []
    for frame in frames:
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, "JPEG", quality=90)
        urls.append(
            "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        )
    return urls


def expand_media_parts(
    messages: list[dict],
    *,
    fps: float | None = None,
    max_frames: int | None = None,
) -> tuple[list[dict], list[str]]:
    """Messages with video parts expanded to frames and audio parts extracted.

    Returns the new messages and the audio file paths in prompt order. Messages without media
    come back unchanged (same objects).
    """
    fps = VIDEO_FPS if fps is None else fps
    max_frames = VIDEO_MAX_FRAMES if max_frames is None else max_frames
    audios: list[str] = []
    out: list[dict] = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            out.append(msg)
            continue
        changed = False
        parts: list[Any] = []
        for raw in content:
            part = _part_dict(raw)
            kind = part.get("type") if isinstance(part, dict) else None
            if kind in ("video", "video_url"):
                frames = _frames_for(_video_source(part), fps, max_frames)
                spacing = 1.0 / fps if fps > 0 else 0.0
                parts.append(
                    {
                        "type": "text",
                        "text": (
                            f"[Video: {len(frames)} frames in order, about one every "
                            f"{spacing:g} s]"
                        ),
                    }
                )
                parts.extend(
                    {"type": "image_url", "image_url": {"url": p}} for p in frames
                )
                changed = True
            elif kind == "input_audio":
                value = part.get("input_audio") or {}
                audios.append(
                    _write_audio(value.get("data", ""), value.get("format", "wav"))
                )
                parts.append({"type": AUDIO_MARKER_TYPE})
                changed = True
            else:
                parts.append(raw)
        out.append({**msg, "content": parts} if changed else msg)
    if audios:
        logger.info("media: %d audio clip(s) extracted for the processor", len(audios))
    return out, audios


def model_hears_audio(config: dict | None) -> bool:
    """Whether a checkpoint's config declares an audio encoder."""
    if not isinstance(config, dict):
        return False
    if config.get("audio_config"):
        return True
    thinker = config.get("thinker_config")
    return isinstance(thinker, dict) and bool(thinker.get("audio_config"))
