# SPDX-License-Identifier: Apache-2.0
"""Image parts on a text-only engine: a text notice instead of a 400.

LeanZero fork. An agent client (goose) can hand the model a ``read_image``
tool; the model screenshots something, calls the tool, and the client sends
the picture back as an image content part. On a ``--text-only`` engine the
content validator used to answer that with HTTP 400
``image_input_unsupported``, and the client treated the 400 as fatal: one
image ended a 6.75-hour benchmark session with no score.

Policy ``notice`` (the default) swaps each well-formed image part for a TEXT
part that says the image could not be viewed, so the turn goes through and
the model can check the thing another way. Policy ``reject`` keeps the old
400 untouched (``serve --text-only-image-policy reject``).

Invariants the callers rely on:

* The swap runs at the route boundary, BEFORE the content validator, the
  chat-template render and therefore before prefix-cache keying.
* The notice is a pure function of the part (mime type, byte size, source):
  no timestamp, no counter, no request id. Two identical requests render
  identically, so the prefix cache keeps hitting across turns that replay
  the same image.
* A malformed image part is left alone, so the validator still rejects it
  with the same shape error as before.
* Only images. Video and audio keep their existing 400s: video comes out of
  a different branch of the validator, and audio is rejected even on a
  vision engine (there is no chat audio lane), so neither is the same code
  path as the text-only image rejection.
* Never silent: every swap is counted (``/v1/status``
  ``text_only_images_replaced``) and each request that swaps logs one
  WARNING with its count.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from .utils import (
    IMAGE_CONTENT_TYPES,
    _content_part_to_dict,
    _validate_content_part_payload,
)

logger = logging.getLogger(__name__)

TEXT_ONLY_IMAGE_POLICIES = ("notice", "reject")
DEFAULT_TEXT_ONLY_IMAGE_POLICY = "notice"

# A remote source longer than this is cut in the notice. The cut is
# deterministic (same URL, same notice) so the prefix cache is unaffected.
_MAX_SOURCE_CHARS = 200

_lock = threading.Lock()
_images_replaced = 0
_requests_with_replacements = 0


# ---------------------------------------------------------------------------
# Policy + counters
# ---------------------------------------------------------------------------


def text_only_image_policy() -> str:
    """The configured policy, falling back to the default on a bad value."""
    from ..config import get_config

    policy = getattr(get_config(), "text_only_image_policy", None)
    if policy in TEXT_ONLY_IMAGE_POLICIES:
        return policy
    return DEFAULT_TEXT_ONLY_IMAGE_POLICY


def should_replace_images(engine) -> bool:
    """True when this engine cannot see images and the policy is ``notice``."""
    if getattr(engine, "is_mllm", False):
        return False
    return text_only_image_policy() == "notice"


def record_text_only_image_replacements(count: int, *, route: str) -> None:
    """Count one request's swaps and log a single WARNING for it."""
    global _images_replaced, _requests_with_replacements
    if count <= 0:
        return
    with _lock:
        _images_replaced += count
        _requests_with_replacements += 1
    logger.warning(
        "Text-only engine: replaced %d image part(s) with a text notice on %s "
        "(--text-only-image-policy notice; 'reject' restores the HTTP 400).",
        count,
        route,
    )


def text_only_image_stats() -> dict:
    with _lock:
        return {
            "images_replaced": _images_replaced,
            "requests": _requests_with_replacements,
        }


def reset_text_only_image_stats() -> None:
    global _images_replaced, _requests_with_replacements
    with _lock:
        _images_replaced = 0
        _requests_with_replacements = 0


# ---------------------------------------------------------------------------
# The notice
# ---------------------------------------------------------------------------


def _base64_size(data: str) -> int | None:
    payload = "".join(data.split())
    if not payload:
        return None
    padding = len(payload) - len(payload.rstrip("="))
    size = (len(payload) * 3) // 4 - padding
    return size if size > 0 else None


def _describe_url(url: str) -> list[str]:
    """Mime / size / source facts from an image URL, data: URLs included."""
    if url.startswith("data:"):
        header, sep, payload = url[5:].partition(",")
        mime = header.split(";", 1)[0].strip() or None
        facts = [mime or "unknown type"]
        if sep and ";base64" in header:
            size = _base64_size(payload)
            if size is not None:
                facts.append(f"{size} bytes")
        return facts
    source = url if len(url) <= _MAX_SOURCE_CHARS else url[:_MAX_SOURCE_CHARS] + "..."
    return [f"from {source}"]


def text_only_image_notice(facts: list[str]) -> str:
    detail = f" ({', '.join(facts)})" if facts else ""
    return (
        "[image not shown: this model is serving text-only, so the attached "
        f"image{detail} could not be viewed. Describe or check it another way.]"
    )


def _openai_image_facts(item: dict) -> list[str]:
    item_type = item.get("type")
    if item_type == "image_url" or item_type == "input_image":
        value = item.get("image_url")
        url = value.get("url") if isinstance(value, dict) else value
    else:  # "image"
        url = item.get("image", item.get("url"))
    return _describe_url(url) if isinstance(url, str) and url else []


def _anthropic_image_facts(source: Any) -> list[str]:
    if not isinstance(source, dict):
        return []
    src_type = source.get("type")
    if src_type == "base64":
        facts = [source.get("media_type") or "unknown type"]
        data = source.get("data")
        if isinstance(data, str):
            size = _base64_size(data)
            if size is not None:
                facts.append(f"{size} bytes")
        return facts
    if src_type == "url" and isinstance(source.get("url"), str):
        return _describe_url(source["url"])
    if src_type == "file" and isinstance(source.get("file_id"), str):
        return [f"file {source['file_id']}"]
    return []


# ---------------------------------------------------------------------------
# OpenAI-shape messages (chat completions, and Responses after conversion)
# ---------------------------------------------------------------------------


def _text_part_like(original, text: str):
    """A text part of the same kind (model or dict) as ``original``."""
    if hasattr(original, "model_dump"):
        try:
            return type(original)(type="text", text=text)
        except Exception:  # pragma: no cover - defensive; dict always works
            pass
    return {"type": "text", "text": text}


def replace_images_in_openai_messages(messages: list) -> int:
    """Swap well-formed image parts for a text notice, in place.

    Covers every role, including ``role:"tool"`` replies, so a mixed
    text+image tool result keeps its text and gains the notice. Returns the
    number of parts swapped.
    """
    replaced = 0
    for msg in messages or []:
        content = (
            msg.get("content")
            if isinstance(msg, dict)
            else getattr(msg, "content", None)
        )
        if not isinstance(content, list):
            continue
        new_content = None
        for idx, raw_item in enumerate(content):
            try:
                item = _content_part_to_dict(raw_item)
            except ValueError:
                continue
            if item.get("type") not in IMAGE_CONTENT_TYPES:
                continue
            try:
                _validate_content_part_payload(item)
            except ValueError:
                # Malformed: leave it for the validator's own 400.
                continue
            if new_content is None:
                new_content = list(content)
            new_content[idx] = _text_part_like(
                raw_item, text_only_image_notice(_openai_image_facts(item))
            )
            replaced += 1
        if new_content is not None:
            if isinstance(msg, dict):
                msg["content"] = new_content
            else:
                msg.content = new_content
    return replaced


# ---------------------------------------------------------------------------
# Responses API: structured function_call_output lists
# ---------------------------------------------------------------------------


def replace_images_in_responses_input(responses_request) -> int:
    """Swap ``input_image`` entries inside ``function_call_output`` lists.

    The adapter JSON-encodes a list output into the tool message verbatim, so
    an image there would reach a text-only prompt as a base64 blob. Message
    items need no handling here: their ``input_image`` parts become
    ``image_url`` parts on the converted chat request, which
    :func:`replace_images_in_openai_messages` covers.
    """
    items = getattr(responses_request, "input", None)
    if not isinstance(items, list):
        return 0
    replaced = 0
    for item in items:
        if getattr(item, "type", None) != "function_call_output":
            continue
        output = getattr(item, "output", None)
        if not isinstance(output, list):
            continue
        new_output = None
        for idx, part in enumerate(output):
            if (
                not isinstance(part, dict)
                or part.get("type") not in IMAGE_CONTENT_TYPES
            ):
                continue
            if new_output is None:
                new_output = list(output)
            new_output[idx] = {
                "type": "input_text",
                "text": text_only_image_notice(_openai_image_facts(part)),
            }
            replaced += 1
        if new_output is not None:
            item.output = new_output
    return replaced


# ---------------------------------------------------------------------------
# Anthropic /v1/messages
# ---------------------------------------------------------------------------


def _block_type(block) -> str:
    if isinstance(block, dict):
        return block.get("type", "") or ""
    return getattr(block, "type", "") or ""


def _block_source(block):
    if isinstance(block, dict):
        return block.get("source")
    return getattr(block, "source", None)


def replace_images_in_anthropic_messages(messages: list) -> int:
    """Swap Anthropic ``image`` blocks for text blocks, in place.

    Top-level ``image`` blocks would otherwise hit the text-only 400 guard;
    ``image`` items nested in a ``tool_result`` were silently dropped by the
    adapter, which only keeps text. Both become a text notice. ``document``
    blocks are not images and keep their existing 400.
    """
    from .anthropic_models import AnthropicContentBlock

    replaced = 0
    for msg in messages or []:
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        new_content = None
        for idx, block in enumerate(content):
            btype = _block_type(block)
            if btype == "image":
                notice = text_only_image_notice(
                    _anthropic_image_facts(_block_source(block))
                )
                if new_content is None:
                    new_content = list(content)
                new_content[idx] = (
                    {"type": "text", "text": notice}
                    if isinstance(block, dict)
                    else AnthropicContentBlock(type="text", text=notice)
                )
                replaced += 1
            elif btype == "tool_result":
                inner = (
                    block.get("content")
                    if isinstance(block, dict)
                    else getattr(block, "content", None)
                )
                if not isinstance(inner, list):
                    continue
                new_inner = None
                for j, item in enumerate(inner):
                    if _block_type(item) != "image":
                        continue
                    if new_inner is None:
                        new_inner = list(inner)
                    new_inner[j] = {
                        "type": "text",
                        "text": text_only_image_notice(
                            _anthropic_image_facts(_block_source(item))
                        ),
                    }
                    replaced += 1
                if new_inner is not None:
                    if isinstance(block, dict):
                        block["content"] = new_inner
                    else:
                        block.content = new_inner
        if new_content is not None:
            msg.content = new_content
    return replaced
