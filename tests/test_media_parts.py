# SPDX-License-Identifier: Apache-2.0
"""Audio/video parts on the MLLM chat lane (rapid_mlx/api/media_parts.py)."""

import base64
import os

import pytest

from rapid_mlx.api import media_parts
from rapid_mlx.api.media_parts import (
    expand_media_parts,
    is_audio_marker,
    model_hears_audio,
)
from rapid_mlx.api.utils import extract_multimodal_content

WAV = base64.b64encode(b"RIFF\x24\x00\x00\x00WAVEfmt ").decode()


def test_input_audio_becomes_a_file_and_a_template_marker():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is said?"},
                {"type": "input_audio", "input_audio": {"data": WAV, "format": "wav"}},
            ],
        }
    ]
    out, audios = expand_media_parts(messages)
    assert len(audios) == 1
    try:
        assert audios[0].endswith(".wav")
        with open(audios[0], "rb") as f:
            assert f.read().startswith(b"RIFF")
        assert out[0]["content"][1] == {"type": "audio"}
        assert is_audio_marker(out[0]["content"][1])
    finally:
        os.unlink(audios[0])


def test_messages_without_media_come_back_unchanged():
    messages = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}, {"role": "assistant", "content": "x"}]
    out, audios = expand_media_parts(messages)
    assert audios == []
    assert out[0] is messages[0] and out[1] is messages[1]


def test_bad_base64_audio_is_a_client_error():
    with pytest.raises(ValueError, match="base64"):
        expand_media_parts(
            [{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "@@", "format": "wav"}}]}]
        )


def test_a_video_part_becomes_a_label_and_one_image_part_per_frame(monkeypatch):
    monkeypatch.setattr(media_parts, "_frames_for", lambda source, fps, n: ["/tmp/f0.jpg", "/tmp/f1.jpg", "/tmp/f2.jpg"])
    out, audios = expand_media_parts(
        [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}}]}],
        fps=1.0,
    )
    parts = out[0]["content"]
    assert audios == []
    assert parts[0]["type"] == "text" and "3 frames" in parts[0]["text"]
    assert [p["image_url"]["url"] for p in parts[1:]] == ["/tmp/f0.jpg", "/tmp/f1.jpg", "/tmp/f2.jpg"]


def test_the_extractor_skips_the_audio_marker_and_counts_frames_as_images(monkeypatch):
    monkeypatch.setattr(media_parts, "_frames_for", lambda source, fps, n: ["/tmp/f0.jpg"])
    out, audios = expand_media_parts(
        [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
                    {"type": "input_audio", "input_audio": {"data": WAV, "format": "wav"}},
                ],
            }
        ]
    )
    try:
        _, images, videos = extract_multimodal_content(out)
        assert images == ["/tmp/f0.jpg"] and videos == []
    finally:
        for a in audios:
            os.unlink(a)


@pytest.mark.parametrize(
    "config,hears",
    [
        ({"audio_config": {"d_model": 1}}, True),
        ({"thinker_config": {"audio_config": {"d_model": 1}}}, True),
        ({"vision_config": {}}, False),
        (None, False),
    ],
)
def test_model_hears_audio_reads_the_audio_encoder(config, hears):
    assert model_hears_audio(config) is hears
