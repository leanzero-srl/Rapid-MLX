# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible server over the qwen4_exp pipeline split.

Every rank runs :func:`serve`.  Rank 0 also runs an HTTP thread (FastAPI +
uvicorn) and owns the scheduling; the generation loop stays on each rank's
main thread because MLX work and the collectives must be issued in the same
order on every rank.  Rank 0 hands each batch to the other ranks as three
``all_sum`` broadcasts (an int header, the left-padded token ids, the
sampling floats); every decode step then closes with one ``all_sum`` that
carries the sampled tokens, the memory-guard stop flag and rank 0's control
word (end the batch).

What the HTTP side reuses from the single engine, unchanged:
``utils.chat_template.apply_chat_template`` (the same rendering, tool-loop
think handling included), ``api.tool_calling.convert_tools_for_template`` and
``engine.batched._normalize_tool_call_arguments_for_template``, and the
route layer's ``service.postprocessor.StreamingPostProcessor`` configured with
the parsers the checkpoint's own chat template declares (the parameterized
XML tool contract -> ``qwen3_coder_xml``; ``<think>`` + ``enable_thinking``
-> ``deepseek_r1``) — the same rule goose-sidecar's ``model_parsers.rs``
applies when it launches the single engine.

Scheduling: a batch is formed from whatever is queued when the previous batch
ends (up to ``--max-batch``, 2 proven bit-exact against single-process
batches); requests arriving mid-batch wait for the next one.  A finished,
EOS'd or cancelled row stops receiving tokens; the batch ends one decode
step after its last live row finishes (the control word rides the next
step's collective).

Prefix cache (Q-75): every rank snapshots ITS OWN layers' caches at a stable
prompt boundary and restores them for a later prompt that starts with the same
tokens.  The layers are hybrid (GDN/PLE recurrent state cannot be trimmed), so
only an exact stored prefix is reusable, and every rank must reuse the SAME
prefix length or the collectives stop pairing.  So rank 0 alone decides — which
entry to restore, where to snapshot, what to evict — and the batch header
carries the decision; the other ranks hold snapshots by the id rank 0 assigned
and never decide anything.  The boundary is the single engine's
(``BatchedEngine._compute_prefix_boundary``, the ``rapid_mlx_transient_tail``
extension included).  Bytes: the cache lives inside each rank's planned KV
budget — at every admission it is trimmed to that budget minus the batch's
own reservation, the new snapshot pre-charged — so it never holds memory the
plan did not.  A request the cache acts on (restores or snapshots) runs as a
batch of one: rows are left-padded to one width, so a restored prefix cannot
share a batch.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import queue
import signal
import socket
import sys
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
from fastapi import Request

from . import pipeline_qwen4 as pipe

_CMD_SHUTDOWN = 1
_CMD_BATCH = 2
_TOOL_XML_MARKERS = (
    "tool_calls",
    "arguments",
    "<tool_call>",
    "</tool_call>",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
)
_THINK_MARKERS = ("enable_thinking", "<think>", "</think>")


# ---------------------------------------------------------------------------
# rank-shared batch execution
# ---------------------------------------------------------------------------


@dataclass
class _Row:
    ids: list[int]
    max_tokens: int
    temperature: float
    top_p: float
    # Rank 0 only: the row's preprocessed images.  They never cross ranks —
    # rank 0 merges the tower's features and shares the RoPE table instead.
    images: Any = None
    # The prefix-cache directive, identical on every rank (rank 0 decides, the
    # batch header carries it; a batch of more than one row carries none):
    # restore entry ``reuse_id`` holding the first ``cached`` tokens, and
    # snapshot entry ``store_id`` after the first ``store_at`` tokens.
    reuse_id: int = 0
    cached: int = 0
    store_id: int = 0
    store_at: int = 0
    # Rank 0 only: the stable-prefix boundary the HTTP side computed (0 = the
    # request asks the cache for nothing: images, or no boundary).
    boundary: int = 0


# The header words after the rows: the one-row directive, then the evictions.
_DIRECTIVE_FIELDS = len(("reuse_id", "cached", "store_id", "store_at", "evictions"))


def _broadcast_batch(
    group, rows: list[_Row] | None, max_batch: int, evict: list[int] | None = None
) -> tuple[list[_Row], list[int]] | None:
    """Rank 0's batch and prefix-cache evictions, identical on every rank; None = shut down."""
    rank0 = group.rank() == 0
    header = [0] * (3 + 2 * max_batch + _DIRECTIVE_FIELDS)
    directive = 3 + 2 * max_batch
    if rank0:
        if rows is None:
            header[0] = _CMD_SHUTDOWN
        else:
            header[0] = _CMD_BATCH
            header[1] = len(rows)
            header[2] = max(len(row.ids) for row in rows)
            for index, row in enumerate(rows):
                header[3 + index] = len(row.ids)
                header[3 + max_batch + index] = row.max_tokens
            if len(rows) == 1:
                row = rows[0]
                header[directive : directive + 4] = [
                    row.reuse_id,
                    row.cached,
                    row.store_id,
                    row.store_at,
                ]
            header[directive + 4] = len(evict or [])
    agreed = mx.distributed.all_sum(mx.array(header, dtype=mx.int32), group=group)
    header = agreed.tolist()
    if header[0] == _CMD_SHUTDOWN:
        return None
    batch, width = header[1], header[2]
    ids = [0] * (batch * width)
    floats = [0.0] * (2 * batch)
    if rank0:
        for index, row in enumerate(rows):
            offset = index * width + width - len(row.ids)
            ids[offset : offset + len(row.ids)] = row.ids
            floats[2 * index] = row.temperature
            floats[2 * index + 1] = row.top_p
    ids = mx.distributed.all_sum(mx.array(ids, dtype=mx.int32), group=group).tolist()
    floats = mx.distributed.all_sum(
        mx.array(floats, dtype=mx.float32), group=group
    ).tolist()
    evictions = header[directive + 4]
    if evictions:
        dropped = list(evict) if rank0 else [0] * evictions
        evict = mx.distributed.all_sum(
            mx.array(dropped, dtype=mx.int32), group=group
        ).tolist()
    else:
        evict = []
    result = []
    for index in range(batch):
        length = header[3 + index]
        row_ids = ids[index * width + width - length : (index + 1) * width]
        result.append(
            _Row(
                ids=row_ids,
                max_tokens=header[3 + max_batch + index],
                temperature=floats[2 * index],
                top_p=floats[2 * index + 1],
            )
        )
    if batch == 1:
        row = result[0]
        row.reuse_id, row.cached, row.store_id, row.store_at = header[
            directive : directive + 4
        ]
    return result, evict


def _sample(logits: mx.array, rows: list[_Row]) -> mx.array:
    from mlx_lm.sample_utils import make_sampler

    picked = []
    for index, row in enumerate(rows):
        line = logits[index : index + 1].astype(mx.float32)
        if row.temperature <= 0:
            picked.append(mx.argmax(line, axis=-1))
        else:
            logprobs = line - mx.logsumexp(line, axis=-1, keepdims=True)
            sampler = make_sampler(temp=row.temperature, top_p=row.top_p)
            picked.append(sampler(logprobs))
    return mx.concatenate(picked).astype(mx.int32)


def _step(
    stage, out, cache, rows, guard, control: int, *, sample: bool
) -> tuple[list[int], int]:
    """One step's collective: tokens, the guard's stop flag, rank 0's control."""
    batch = len(rows)
    reason = guard.check() if guard is not None else None
    if stage.is_last and sample:
        tokens = _sample(out[:, -1, :], rows)
    else:
        tokens = mx.depends(mx.zeros((batch,), dtype=mx.int32), out)
    payload = mx.concatenate(
        [
            tokens,
            mx.array(
                [1 if reason else 0, control if stage.is_first else 0], dtype=mx.int32
            ),
        ]
    )
    if stage.size > 1:
        payload = mx.distributed.all_sum(payload, group=stage.group)
    mx.eval(payload, [layer_cache.state for layer_cache in cache])
    values = payload.tolist()
    if values[batch]:
        raise pipe.PipelineMemoryStopError(
            reason or "a peer rank's memory guard tripped"
        )
    return values[:batch], values[batch + 1]


def prefill_chunks(
    start: int, end: int, step: int, split: int = 0
) -> list[tuple[int, int]]:
    """The ``[a, b)`` prefill ranges from ``start`` to ``end``, ``step`` tokens each.

    When ``start < split <= end`` one range ends exactly at ``split`` (where
    the prefix snapshot is taken); otherwise the ranges are the plain chunks.
    """
    ranges: list[tuple[int, int]] = []
    edges = [split] if start < split < end else []
    offset = start
    for stop in [*edges, end]:
        while offset < stop:
            ranges.append((offset, min(offset + step, stop)))
            offset = ranges[-1][1]
    return ranges


def _held_bytes(value, seen: set[int] | None = None) -> int:
    """Bytes of every MLX array a cache object holds (whole buffers, not views)."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, mx.array):
        return value.nbytes
    if isinstance(value, (list, tuple)):
        return sum(_held_bytes(item, seen) for item in value)
    if isinstance(value, dict):
        return sum(_held_bytes(item, seen) for item in value.values())
    if hasattr(value, "__dict__"):
        return sum(_held_bytes(item, seen) for item in vars(value).values())
    return 0


class _PrefixStore:
    """One rank's prefix snapshots, by the id rank 0 assigned.  Decides nothing."""

    def __init__(self):
        self.entries: dict[int, list[Any]] = {}

    def take(self, entry_id: int) -> list[Any]:
        if entry_id not in self.entries:
            raise RuntimeError(
                f"prefix cache: rank 0 restores entry {entry_id}, which this rank "
                f"does not hold (held: {sorted(self.entries)}) — the ranks diverged"
            )
        # Generation mutates offsets and writes into the KV buffers: the
        # restored copy must not alias the stored entry (the single engine's
        # MemoryAwarePrefixCache.fetch copies for the same reason).
        return copy.deepcopy(self.entries[entry_id])

    def put(self, entry_id: int, cache: list[Any]) -> int:
        self.entries[entry_id] = copy.deepcopy(cache)
        return _held_bytes(self.entries[entry_id])

    def drop(self, entry_ids: list[int]) -> None:
        for entry_id in entry_ids:
            self.entries.pop(entry_id, None)


def _agree_bytes(stage, local: int) -> list[int]:
    """Every rank's measured bytes, rank-indexed (KiB over the wire: int32)."""
    kib = [0] * stage.size
    kib[stage.rank] = -(-local // 1024)
    if stage.size > 1:
        kib = mx.distributed.all_sum(
            mx.array(kib, dtype=mx.int32), group=stage.group
        ).tolist()
    return [value * 1024 for value in kib]


def run_batch(
    stage,
    guard,
    rows: list[_Row],
    prefill_step: int,
    on_tokens=None,
    control_fn=None,
    store: _PrefixStore | None = None,
    evict: list[int] | None = None,
    on_stored=None,
) -> None:
    """Prefill + decode one batch on this rank.

    ``on_tokens(step, tokens)`` (rank 0) receives each step's sampled tokens;
    ``control_fn()`` (rank 0) returns 1 to end the batch at the next step.
    A one-row batch may carry a prefix-cache directive: restore ``reuse_id``
    and prefill from ``cached``; snapshot ``store_id`` after ``store_at``
    tokens, then ``on_stored(store_id, bytes_per_rank)`` (rank 0).
    ``evict`` entries are dropped after the restore copied its entry.
    """
    width = max(len(row.ids) for row in rows)
    padded = [[0] * (width - len(row.ids)) + row.ids for row in rows]
    tokens = mx.array(padded, dtype=mx.int32)
    padding = [width - len(row.ids) for row in rows]
    embeddings, rope = pipe.prepare_multimodal(
        stage,
        [row.ids for row in rows],
        [row.images for row in rows] if stage.is_first else None,
    )
    head = rows[0]
    directed = len(rows) == 1 and (head.reuse_id or head.store_id)
    if directed and (store is None or embeddings is not None or rope is not None):
        raise RuntimeError(
            "prefix cache: a directive reached a rank without a store, or a row "
            "with images (rank 0 never directs either)"
        )
    if len(rows) == 1 and head.reuse_id:
        cache, start = store.take(head.reuse_id), head.cached
    else:
        cache, start = stage.make_cache(padding if len(rows) > 1 else None), 0
    if store is not None and evict:
        store.drop(evict)
    split = head.store_at if len(rows) == 1 and head.store_id else 0
    for offset, stop in prefill_chunks(start, width - 1, prefill_step, split):
        out = stage.forward(
            tokens[:, offset:stop],
            cache,
            logits=None,
            embeddings=None if embeddings is None else embeddings[:, offset:stop],
            rope_positions=rope,
        )
        _step(stage, out, cache, rows, guard, 0, sample=False)
        if split and stop == split:
            measured = _agree_bytes(stage, store.put(head.store_id, cache))
            if on_stored is not None:
                on_stored(head.store_id, measured)
    current = tokens[:, -1:]
    current_embeddings = None if embeddings is None else embeddings[:, -1:]
    for step in range(max(row.max_tokens for row in rows)):
        control = control_fn() if (control_fn is not None and stage.is_first) else 0
        out = stage.forward(
            current,
            cache,
            logits="last",
            embeddings=current_embeddings,
            rope_positions=rope,
        )
        current_embeddings = None
        sampled, ended = _step(stage, out, cache, rows, guard, control, sample=True)
        if ended:
            return
        if on_tokens is not None:
            on_tokens(step, sampled)
        current = mx.array(sampled, dtype=mx.int32)[:, None]


# ---------------------------------------------------------------------------
# rank 0: jobs, HTTP
# ---------------------------------------------------------------------------


@dataclass
class _Job:
    row: _Row
    loop: asyncio.AbstractEventLoop
    events: asyncio.Queue
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:24])
    cancelled: bool = False
    finished: bool = False
    produced: int = 0

    def push(self, item) -> None:
        self.loop.call_soon_threadsafe(self.events.put_nowait, item)


class _KvBudget:
    """Admission by what a batch would hold on EVERY rank, in the planner's own terms.

    The split was planned for ``slots`` full-context sequences (the planner's
    ``batch``): each rank's budget for runtime state is its planned KV/recurrent
    state plus prefill workspace at that shape.  A candidate batch reserves, on
    each rank, the same two terms at ``rows`` x the longest row's
    prompt + max_tokens (+1 for the step that ends the batch), rounded to the
    caches' allocation steps by ``layer_state_bytes``.  Rows are left-padded to
    one width, so a batch costs rows x its longest row, never the sum.
    """

    def __init__(self, plan, prefill_step: int):
        self.plan = plan
        self.prefill_step = prefill_step
        self.args = plan.args
        self.budgets = [
            stage.state_bytes + stage.workspace_bytes for stage in plan.stages
        ]
        self.slots = plan.batch

    def reserve(self, lengths: list[int]) -> list[int]:
        size = len(self.plan.stages)
        return [
            (lambda planned: planned.state_bytes + planned.workspace_bytes)(
                pipe._stage_plan(
                    self.args,
                    self.plan.checkpoint,
                    stage.node,
                    stage.rank,
                    size,
                    stage.start,
                    stage.end,
                    max(lengths),
                    len(lengths),
                    min(self.prefill_step, max(lengths)),
                )
            )
            for stage in self.plan.stages
        ]

    def fits(self, lengths: list[int]) -> bool:
        return all(
            need <= budget for need, budget in zip(self.reserve(lengths), self.budgets)
        )

    def entry_bytes(self, tokens: int) -> list[int]:
        """Each rank's bound on one snapshot of ``tokens`` tokens (one sequence).

        The planner's own state formula at ``tokens`` plus one KV allocation
        step: a snapshot keeps its KV buffers whole, and a buffer runs at most
        one step past the tokens it holds.
        """
        size = len(self.plan.stages)
        span = tokens + pipe.KVCache.step
        return [
            pipe._stage_plan(
                self.args,
                self.plan.checkpoint,
                stage.node,
                stage.rank,
                size,
                stage.start,
                stage.end,
                span,
                1,
                min(self.prefill_step, span),
            ).state_bytes
            for stage in self.plan.stages
        ]


@dataclass
class _Entry:
    key: tuple[int, ...]
    bytes: list[int]


class _PrefixIndex:
    """Rank 0's prefix-cache decisions: what to restore, snapshot and evict.

    Bounded by bytes alone, per rank: at every admission the entries plus the
    snapshot this batch will take must fit each rank's planned KV budget minus
    the batch's own reservation (``_KvBudget``) — the cache holds only the
    budget live requests leave idle, and yields it to them first.  LRU order,
    refreshed on every hit.  No entry count bound: the count can never evict
    before the bytes do.
    """

    def __init__(self, kv: _KvBudget):
        self.kv = kv
        self.entries: OrderedDict[int, _Entry] = OrderedDict()
        self.next_id = 1
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0
        self.stores = 0
        self.evicted = 0
        self.skipped: dict[str, int] = {}

    def _skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def lookup(self, ids: list[int]) -> tuple[int, int]:
        """The longest stored exact prefix that still leaves a token to feed."""
        best_id, best = 0, 0
        for entry_id, entry in self.entries.items():
            length = len(entry.key)
            if best < length < len(ids) and tuple(ids[:length]) == entry.key:
                best_id, best = entry_id, length
        return best_id, best

    def acts_on(self, row: _Row) -> bool:
        return row.images is None and (row.boundary > 0 or self.lookup(row.ids)[1] > 0)

    def admit(self, rows: list[_Row], lengths: list[int]) -> list[int]:
        """Set the directive on a one-row batch; return the entries to evict."""
        with self.lock:
            room = [
                budget - need
                for budget, need in zip(self.kv.budgets, self.kv.reserve(lengths))
            ]
            new = [0] * len(room)
            if len(rows) == 1 and rows[0].images is None:
                row = rows[0]
                row.reuse_id, row.cached = self.lookup(row.ids)
                if row.reuse_id:
                    self.hits += 1
                    self.tokens_saved += row.cached
                    self.entries.move_to_end(row.reuse_id)
                else:
                    self.misses += 1
                if row.boundary <= row.cached:
                    if row.boundary:
                        self._skip("boundary_within_restored_prefix")
                elif row.boundary >= len(row.ids):
                    self._skip("boundary_past_prompt")
                elif any(
                    entry.key == tuple(row.ids[: row.boundary])
                    for entry in self.entries.values()
                ):
                    self._skip("already_stored")
                else:
                    new = self.kv.entry_bytes(row.boundary)
                    row.store_id, row.store_at = self.next_id, row.boundary
                    self.next_id += 1
            evict = []

            def held() -> list[int]:
                return [
                    sum(entry.bytes[rank] for entry in self.entries.values())
                    for rank in range(len(room))
                ]

            def over(extra: list[int]) -> bool:
                return any(h + e > r for h, e, r in zip(held(), extra, room))

            while self.entries and over(new):
                entry_id, _ = self.entries.popitem(last=False)
                evict.append(entry_id)
                self.evicted += 1
            if any(new) and over(new):
                # Even an empty cache cannot hold this snapshot beside the batch.
                self._skip("no_room_beside_the_batch")
                rows[0].store_id = rows[0].store_at = 0
            return evict

    def stored(self, entry_id: int, key: tuple[int, ...], measured: list[int]) -> None:
        with self.lock:
            self.entries[entry_id] = _Entry(key, measured)
            self.stores += 1

    def status(self) -> dict[str, Any]:
        with self.lock:
            held = [
                sum(entry.bytes[rank] for entry in self.entries.values())
                for rank in range(len(self.kv.budgets))
            ]
            return {
                "enabled": True,
                "entries": len(self.entries),
                "entry_tokens": [len(entry.key) for entry in self.entries.values()],
                "bytes": held,
                "limit_bytes": list(self.kv.budgets),
                "hits": self.hits,
                "misses": self.misses,
                "tokens_saved": self.tokens_saved,
                "stored": self.stores,
                "evicted": self.evicted,
                "skipped": dict(self.skipped),
            }


def _reservation_length(row: _Row) -> int:
    return len(row.ids) + row.max_tokens + 1


@dataclass
class _State:
    served: str
    context: int
    max_batch: int
    kv: _KvBudget | None = None
    prefix: _PrefixIndex | None = None
    jobs: queue.Queue = field(default_factory=queue.Queue)
    held: object = None
    active: list = field(default_factory=list)
    reserved: list = field(default_factory=list)
    steps: int = 0
    admission_open: bool = True
    admission_reason: str | None = None
    shutting_down: bool = False
    eos_ids: frozenset = frozenset()


def _template_parsers(tokenizer) -> tuple[str | None, str | None]:
    template = getattr(tokenizer, "chat_template", None) or ""
    if not isinstance(template, str):
        return None, None
    tool = "qwen3_coder_xml" if all(m in template for m in _TOOL_XML_MARKERS) else None
    reasoning = (
        "deepseek_r1" if tool and all(m in template for m in _THINK_MARKERS) else None
    )
    return tool, reasoning


_IMAGE_PART_TYPES = ("image_url", "input_image", "image")


def _image_source(part: dict) -> Any:
    """The OpenAI/Responses image reference inside one content part."""
    for key in ("image_url", "image", "url"):
        if part.get(key):
            return part[key]
    raise ValueError(f"image content part carries no image: {sorted(part)}")


def _split_images(messages: list[dict], vision: bool) -> tuple[list[dict], list]:
    """Messages for the chat template, plus the image references in order.

    Text parts of a text-only message are joined (the template's plain-string
    path).  An image part becomes ``{"type": "image"}``, which the checkpoint's
    template renders as one ``<|vision_start|><|image_pad|><|vision_end|>``.
    Without a vision tower an image is refused by name.
    """
    rendered, images = [], []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            parts, texts, has_image = [], [], False
            for part in content:
                kind = part.get("type") if isinstance(part, dict) else None
                if kind == "text":
                    parts.append({"type": "text", "text": part.get("text", "")})
                    texts.append(part.get("text", ""))
                elif kind in _IMAGE_PART_TYPES:
                    if not vision:
                        raise ValueError(
                            f"content part type {kind!r} is not supported: this "
                            "split serves text only (the checkpoint declares no "
                            "vision tower, or the server runs --no-vision)"
                        )
                    images.append(_image_source(part))
                    parts.append({"type": "image"})
                    has_image = True
                else:
                    raise ValueError(f"content part type {kind!r} is not supported")
            message = {**message, "content": parts if has_image else "".join(texts)}
        rendered.append(message)
    return rendered, images


def _load_image(source: Any):
    """A PIL image from a data URL / base64, http(s) URL, file:// URL or path.

    Reuses the single engine's own resolver (``models.mllm.process_image_input``):
    size caps, the SSRF guard on remote URLs, and local paths only inside
    ``RAPID_MLX_MEDIA_ROOT``.  ``file://`` is that same local-path branch.
    """
    from urllib.parse import unquote, urlparse

    from PIL import Image

    from ..models.mllm import process_image_input

    if isinstance(source, dict):
        source = source.get("url") or source.get("image_url") or ""
    if isinstance(source, str) and source.startswith("file://"):
        source = unquote(urlparse(source).path)
    path = process_image_input(source)
    with Image.open(path) as image:
        return image.convert("RGB")


def _merge_tool_call_deltas(deltas: list[dict]) -> list[dict]:
    """Fold the postprocessor's streaming tool-call deltas into whole calls."""
    merged: dict[int, dict] = {}
    for delta in deltas:
        call = merged.setdefault(
            delta.get("index", 0),
            {
                "id": None,
                "type": "function",
                "function": {"name": None, "arguments": ""},
            },
        )
        call["id"] = call["id"] or delta.get("id")
        function = delta.get("function") or {}
        call["function"]["name"] = call["function"]["name"] or function.get("name")
        call["function"]["arguments"] += function.get("arguments") or ""
    return [merged[index] for index in sorted(merged)]


class _BoundaryRenderer:
    """What ``BatchedEngine._compute_prefix_boundary`` reads from its engine.

    The boundary rule is the single engine's, reused as is; this server
    renders prompts with the same shared ``apply_chat_template`` the engine's
    ``_apply_chat_template`` ends in, so the two agree token for token.
    """

    def __init__(self, tokenizer, model_name: str):
        self.tokenizer = tokenizer
        self.model_name = model_name

    def _apply_chat_template(
        self,
        messages,
        tools=None,
        num_images: int = 0,
        enable_thinking=None,
        add_generation_prompt: bool = True,
        chat_template_kwargs=None,
    ) -> str:
        from ..engine.batched import _normalize_tool_call_arguments_for_template
        from ..utils.chat_template import apply_chat_template

        return apply_chat_template(
            self.tokenizer,
            _normalize_tool_call_arguments_for_template(messages),
            tools=tools,
            enable_thinking=enable_thinking,
            model_name=self.model_name,
            add_generation_prompt=add_generation_prompt,
            chat_template_kwargs=chat_template_kwargs,
        )


def _build_app(state: _State, tokenizer, eos_ids: set[int], vision=None):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse

    from ..api.models import REQUEST_EXTENSIONS
    from ..api.tool_calling import convert_tools_for_template
    from ..config.server_config import ServerConfig
    from ..engine.base import GenerationOutput
    from ..engine.batched import (
        BatchedEngine,
        _normalize_tool_call_arguments_for_template,
    )
    from ..service.helpers import _should_start_in_thinking
    from ..service.postprocessor import StreamingPostProcessor
    from ..utils.chat_template import apply_chat_template

    tool_parser, reasoning_parser = _template_parsers(tokenizer)
    cfg = ServerConfig()
    cfg.enable_auto_tool_choice = tool_parser is not None
    cfg.tool_call_parser = tool_parser
    cfg.reasoning_parser_name = reasoning_parser
    cfg.engine = type(
        "_TokenizerHolder", (), {"tokenizer": tokenizer, "_tokenizer": tokenizer}
    )()
    app = FastAPI()
    vision_processor = vision.processor(tokenizer) if vision is not None else None
    capabilities = ["text"]
    if vision is not None:
        capabilities.append("vision")
    if tool_parser is not None:
        capabilities.append("tools")

    boundary_renderer = _BoundaryRenderer(tokenizer, state.served)

    def error(status: int, message: str, kind: str) -> JSONResponse:
        return JSONResponse(
            status_code=status, content={"error": {"message": message, "type": kind}}
        )

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": state.served,
                    "object": "model",
                    "owned_by": "rapid-mlx-pipeline",
                    # The single engine's /v1/models shape (routes/models.py
                    # _detect_capabilities): text -> vision -> tools.
                    "modality": "image" if vision is not None else "text",
                    "capabilities": capabilities,
                    "context_window": state.context,
                    "tool_call_parser": tool_parser,
                    "reasoning_parser": reasoning_parser,
                    # The single engine's declaration: the transient-tail
                    # field moves the prefix snapshot, so it is declared only
                    # while there is a prefix cache to move it in.
                    "request_extensions": list(REQUEST_EXTENSIONS)
                    if state.prefix is not None
                    else [],
                }
            ],
        }

    @app.get("/v1/status")
    async def status():
        running = sum(1 for job in state.active if not job.finished)
        kv = state.kv
        reserved = list(state.reserved)
        if kv is not None and reserved:
            share = max(r / b for r, b in zip(reserved, kv.budgets))
        else:
            share = 0.0
        return {
            "num_running": running,
            "num_waiting": state.jobs.qsize() + (1 if state.held is not None else 0),
            "slots": kv.slots if kv is not None else None,
            "slots_in_use": math.ceil(share * kv.slots - 1e-9)
            if kv is not None
            else None,
            "sequences_in_flight": len(state.active),
            "kv_reserved_bytes": reserved,
            "kv_budget_bytes": kv.budgets if kv is not None else None,
            "prefix_cache": state.prefix.status()
            if state.prefix is not None
            else {"enabled": False, "reason": "--no-prefix-cache"},
            "status": "ok",
        }

    @app.get("/goose/progress")
    async def progress():
        return {
            "steps": state.steps,
            "inflight": len(state.active) + state.jobs.qsize(),
            "admission_open": state.admission_open,
            "active": mx.get_active_memory(),
            "peak": mx.get_peak_memory(),
        }

    @app.post("/goose/admission")
    async def admission(request: Request):
        body = await request.json()
        state.admission_open = bool(body.get("open"))
        state.admission_reason = body.get("reason")
        return {"admission_open": state.admission_open}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        if not state.admission_open or state.shutting_down:
            return error(
                503,
                "the pipeline engine is not admitting new requests: "
                + str(state.admission_reason or "shutting down"),
                "server_busy",
            )
        body = await request.json()
        model = body.get("model")
        if model not in (None, state.served):
            return error(
                404,
                f"model '{model}' is not served here; this engine serves '{state.served}'",
                "model_not_found",
            )
        tools = body.get("tools") or None
        if tools and tool_parser is None:
            return error(
                400,
                "tools are not supported for this checkpoint: its chat template does not "
                "declare the parameterized XML tool contract the pipeline server parses",
                "tools_unsupported",
            )
        try:
            messages, image_sources = _split_images(
                list(body.get("messages") or []), vision is not None
            )
            messages = _normalize_tool_call_arguments_for_template(messages)
            pictures = [_load_image(source) for source in image_sources]
        except Exception as refusal:  # noqa: BLE001 - every cause is the client's input
            return error(400, str(refusal), "invalid_request_error")
        kwargs = dict(body.get("chat_template_kwargs") or {})
        enable_thinking = kwargs.pop("enable_thinking", body.get("enable_thinking"))
        prompt = apply_chat_template(
            tokenizer,
            messages,
            tools=convert_tools_for_template(tools) if tools else None,
            enable_thinking=enable_thinking,
            model_name=state.served,
            chat_template_kwargs=kwargs or None,
        )
        images = None
        if pictures:
            try:
                processed = vision_processor(
                    text=[prompt], images=pictures, return_tensors="np"
                )
            except Exception as refusal:  # noqa: BLE001 - malformed image input
                return error(
                    400, f"image preprocessing: {refusal}", "invalid_request_error"
                )
            ids = processed["input_ids"][0].tolist()
            images = pipe.ImageInput(
                processed["pixel_values"], processed["image_grid_thw"].tolist()
            )
        else:
            ids = tokenizer.encode(prompt)
        boundary = 0
        if state.prefix is not None and images is None:
            stable = BatchedEngine._stable_messages_before_transient_tail(
                messages, None, body.get("rapid_mlx_transient_tail")
            )
            boundary = BatchedEngine._compute_prefix_boundary(
                boundary_renderer,
                messages,
                tools,
                stable_messages=stable,
                generation_prompt=ids,
                enable_thinking=enable_thinking,
                chat_template_kwargs=kwargs or None,
            )
        budget = state.context - len(ids) - 1
        requested = body.get("max_completion_tokens") or body.get("max_tokens")
        if budget < 1:
            return error(
                400,
                f"prompt is {len(ids)} tokens; the split was planned for a {state.context}-token context",
                "context_length_exceeded",
            )
        max_tokens = min(int(requested), budget) if requested else budget
        temperature = float(body.get("temperature") or 0.0)
        top_p = float(body.get("top_p") or 1.0)
        stops = body.get("stop") or []
        stops = [stops] if isinstance(stops, str) else list(stops)
        loop = asyncio.get_running_loop()
        job = _Job(
            _Row(ids, max_tokens, temperature, top_p, images, boundary=boundary),
            loop,
            asyncio.Queue(),
        )
        state.jobs.put(job)
        created = int(time.time())
        processor = StreamingPostProcessor(
            cfg,
            tools_requested=bool(tools),
            enable_thinking=enable_thinking,
            request=body,
        )
        processor.reset()
        if processor.reasoning_parser is not None and _should_start_in_thinking(
            getattr(tokenizer, "chat_template", "") or "",
            enable_thinking,
            tools_requested=bool(tools),
        ):
            # The template opens <think> in the generation prompt, so the
            # model never emits the opener.  deepseek_r1's streaming path
            # flips a tagless stream to content after 64 characters unless it
            # is told the prompt primed thinking (its own hook, set by
            # configure_request on the distill variant).  Measured on Flash:
            # without it every thought past 64 chars streamed as content.
            processor.reasoning_parser._prompt_primed_thinking = True

        async def generate():
            """Yield (events, finish_reason, completion_tokens) as tokens arrive."""
            detok = tokenizer.detokenizer
            text = ""
            finish = "length"
            completion = 0
            while True:
                item = await job.events.get()
                if item[0] == "error":
                    raise RuntimeError(item[1])
                if item[0] == "done":
                    finish = item[1]
                    break
                token = item[1]
                completion += 1
                if token in eos_ids:
                    finish = "stop"
                    job.finished = True
                    break
                detok.add_token(token)
                piece = detok.last_segment
                if stops:
                    candidate = text + piece
                    hit = min(
                        (
                            candidate.find(s, max(0, len(text) - len(s)))
                            for s in stops
                            if s in candidate
                        ),
                        default=-1,
                    )
                    if hit >= 0:
                        piece = candidate[:hit][len(text) :]
                        text += piece
                        finish = "stop"
                        job.finished = True
                        if piece:
                            yield (
                                processor.process_chunk(
                                    GenerationOutput(
                                        text=text,
                                        new_text=piece,
                                        prompt_tokens=len(ids),
                                        completion_tokens=completion,
                                        finished=False,
                                        finish_reason=None,
                                    )
                                ),
                                None,
                                completion,
                            )
                        break
                text += piece
                if piece:
                    yield (
                        processor.process_chunk(
                            GenerationOutput(
                                text=text,
                                new_text=piece,
                                prompt_tokens=len(ids),
                                completion_tokens=completion,
                                finished=False,
                                finish_reason=None,
                            )
                        ),
                        None,
                        completion,
                    )
            detok.finalize()
            tail = detok.last_segment
            if tail and finish != "stop":
                text += tail
                yield (
                    processor.process_chunk(
                        GenerationOutput(
                            text=text,
                            new_text=tail,
                            prompt_tokens=len(ids),
                            completion_tokens=completion,
                            finished=False,
                            finish_reason=None,
                        )
                    ),
                    None,
                    completion,
                )
            terminal = processor.process_chunk(
                GenerationOutput(
                    text="",
                    new_text="",
                    prompt_tokens=len(ids),
                    completion_tokens=completion,
                    finished=True,
                    finish_reason=finish,
                )
            )
            terminal.extend(processor.finalize())
            yield terminal, finish, completion

        def delta_of(event) -> dict:
            delta: dict[str, Any] = {}
            if event.content:
                delta["content"] = event.content
            if event.reasoning:
                delta["reasoning_content"] = event.reasoning
            if event.tool_calls:
                delta["tool_calls"] = event.tool_calls
            return delta

        if body.get("stream"):

            async def sse():
                def chunk(delta, finish=None, usage=None):
                    payload = {
                        "id": f"chatcmpl-{job.id}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": state.served,
                        "choices": [
                            {"index": 0, "delta": delta, "finish_reason": finish}
                        ],
                    }
                    if usage is not None:
                        payload["usage"] = usage
                    return f"data: {json.dumps(payload)}\n\n"

                yield chunk({"role": "assistant"})
                finish = "stop"
                completion = 0
                try:
                    async for events, final, completion in generate():
                        for event in events:
                            if event.finish_reason:
                                finish = event.finish_reason
                            delta = delta_of(event)
                            if delta:
                                yield chunk(delta)
                        if final is not None and finish == "stop":
                            finish = final
                except RuntimeError as failure:
                    yield f"data: {json.dumps({'error': {'message': str(failure), 'type': 'pipeline_error'}})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                finally:
                    job.cancelled = True
                usage = {
                    "prompt_tokens": len(ids),
                    "completion_tokens": completion,
                    "total_tokens": len(ids) + completion,
                    "prompt_tokens_details": {"cached_tokens": job.row.cached},
                }
                yield chunk({}, finish, usage)
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        content, reasoning, calls = [], [], []
        finish = "stop"
        completion = 0
        try:
            async for events, final, completion in generate():
                for event in events:
                    if event.finish_reason:
                        finish = event.finish_reason
                    if event.content:
                        content.append(event.content)
                    if event.reasoning:
                        reasoning.append(event.reasoning)
                    if event.tool_calls:
                        calls.extend(event.tool_calls)
                if final is not None and finish == "stop":
                    finish = final
        except RuntimeError as failure:
            return error(500, str(failure), "pipeline_error")
        finally:
            job.cancelled = True
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content).strip() or None,
        }
        if reasoning:
            message["reasoning_content"] = "".join(reasoning).strip()
        if calls:
            message["tool_calls"] = _merge_tool_call_deltas(calls)
            finish = "tool_calls"
        return {
            "id": f"chatcmpl-{job.id}",
            "object": "chat.completion",
            "created": created,
            "model": state.served,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": len(ids),
                "completion_tokens": completion,
                "total_tokens": len(ids) + completion,
                "prompt_tokens_details": {"cached_tokens": job.row.cached},
            },
        }

    return app


def _run_jobs(
    stage,
    guard,
    state: _State,
    batch: list[_Job],
    prefill_step: int,
    store: _PrefixStore | None = None,
    evict: list[int] | None = None,
) -> None:
    def on_tokens(step: int, tokens: list[int]) -> None:
        state.steps += 1
        for job, token in zip(batch, tokens):
            if job.finished or job.cancelled:
                continue
            job.produced += 1
            job.push(("token", token))
            if token in state.eos_ids:
                job.finished = True
                job.push(("done", "stop"))
            elif job.produced >= job.row.max_tokens:
                job.finished = True
                job.push(("done", "length"))

    def control() -> int:
        return int(all(job.finished or job.cancelled for job in batch))

    def on_stored(entry_id: int, measured: list[int]) -> None:
        row = batch[0].row
        state.prefix.stored(entry_id, tuple(row.ids[: row.store_at]), measured)

    try:
        run_batch(
            stage,
            guard,
            [job.row for job in batch],
            prefill_step,
            on_tokens,
            control,
            store=store,
            evict=evict,
            on_stored=on_stored if state.prefix is not None else None,
        )
    except pipe.PipelineMemoryStopError as stop:
        for job in batch:
            job.push(("error", f"memory guard stopped the pipeline: {stop}"))
        raise
    for job in batch:
        if not job.finished:
            job.finished = True
            job.push(("done", "length"))


def _rank0_loop(
    stage,
    guard,
    state: _State,
    prefill_step: int,
    wake: _Wake,
    store: _PrefixStore | None = None,
) -> None:
    group = stage.group
    prefix = state.prefix
    while True:
        if state.held is not None:
            first, state.held = state.held, None
        else:
            first = state.jobs.get()
        if first is None:
            wake.ring()
            _broadcast_batch(group, None, state.max_batch)
            return
        if first.cancelled:
            continue
        batch = [first]
        lengths = [_reservation_length(first.row)]
        # A request the prefix cache acts on runs alone: a restored prefix
        # cannot share a left-padded batch.
        alone = prefix is not None and prefix.acts_on(first.row)
        while len(batch) < state.max_batch and not alone:
            try:
                extra = state.jobs.get_nowait()
            except queue.Empty:
                break
            if extra is None:
                state.jobs.put(None)
                break
            if extra.cancelled:
                continue
            candidate = [*lengths, _reservation_length(extra.row)]
            if (state.kv is not None and not state.kv.fits(candidate)) or (
                prefix is not None and prefix.acts_on(extra.row)
            ):
                # First come, first served: the request that does not fit
                # beside this batch (by KV, or because the prefix cache acts
                # on it) heads the next one, and nothing behind it jumps the
                # line.
                state.held = extra
                break
            batch.append(extra)
            lengths = candidate
        evict = (
            prefix.admit([job.row for job in batch], lengths)
            if prefix is not None
            else []
        )
        state.active = batch
        state.reserved = state.kv.reserve(lengths) if state.kv is not None else []
        wake.ring()
        _broadcast_batch(group, [job.row for job in batch], state.max_batch, evict)
        _run_jobs(stage, guard, state, batch, prefill_step, store, evict)
        state.active = []
        state.reserved = []


def _rank0_host() -> str:
    """Rank 0's address as the launcher told every rank (JACCL coordinator or ring host 0)."""
    coordinator = os.environ.get("MLX_JACCL_COORDINATOR")
    if coordinator:
        return coordinator.rsplit(":", 1)[0]
    hostfile = os.environ.get("MLX_HOSTFILE")
    if hostfile:
        text = Path(hostfile).read_text() if Path(hostfile).is_file() else hostfile
        return json.loads(text)[0][0].rsplit(":", 1)[0]
    raise RuntimeError(
        "no MLX_JACCL_COORDINATOR or MLX_HOSTFILE: rank 0's address is unknown"
    )


class _Wake:
    """A kernel-blocking doorbell in front of every batch command.

    Waiting for the next batch inside the header ``all_sum`` busy-polls the
    transport: an idle worker rank burned a full core (measured over JACCL on
    the M3 Ultra: 60.2 CPU-s in 60 s idle).  Rank 0 now rings one byte per
    worker over a plain TCP socket right before it issues the collective, and
    each worker parks in ``recv`` — no clock, no polling.  A closed socket
    (rank 0 gone) ends the worker.
    """

    def __init__(self, group):
        self.rank = group.rank()
        self.peers: list[socket.socket] = []
        self.link: socket.socket | None = None
        if group.size() == 1:
            return
        host = _rank0_host()
        if self.rank == 0:
            server = socket.create_server((host, 0))
            port = server.getsockname()[1]
        else:
            server, port = None, 0
        agreed = mx.distributed.all_sum(mx.array([port], dtype=mx.int32), group=group)
        port = int(agreed.item())
        if self.rank == 0:
            for _ in range(group.size() - 1):
                peer, _ = server.accept()
                peer.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.peers.append(peer)
            server.close()
        else:
            self.link = socket.create_connection((host, port))
            self.link.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def ring(self) -> None:
        for peer in self.peers:
            peer.sendall(b"\x01")

    def wait(self) -> bool:
        if self.link is None:
            return True
        return self.link.recv(1) == b"\x01"


def _worker_loop(
    stage,
    guard,
    max_batch: int,
    prefill_step: int,
    wake: _Wake,
    store: _PrefixStore | None = None,
) -> None:
    while True:
        if not wake.wait():
            return
        batch = _broadcast_batch(stage.group, None, max_batch)
        if batch is None:
            return
        rows, evict = batch
        run_batch(stage, guard, rows, prefill_step, store=store, evict=evict)


def serve(options, emit=None) -> int:
    """Run one rank of the server (``mlx.distributed.init`` already reachable)."""
    emit = emit or (
        lambda tag, payload: print(f"PIPELINE_{tag} {json.dumps(payload)}", flush=True)
    )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    group = mx.distributed.init(strict=True)
    emit(
        "RANK_GROUP",
        {"rank": group.rank(), "size": group.size(), "mlx": mx.__version__},
    )
    model_dir = Path(options.model).expanduser()
    prefill_step = options.prefill_step or pipe.default_prefill_step()
    stage, plan, guard = pipe.load_stage(
        model_dir,
        group,
        context=options.context,
        batch=options.slots,
        prefill_step=prefill_step,
        starts=pipe._parse_starts(options.split),
        vision=not options.no_vision,
        log=lambda line: print(line, flush=True),
    )
    emit("RANK_CAPS", {**stage.limits, "planned": plan.stages[stage.rank].total_bytes})
    # Kernel compilation and first-touch paging happen here, before anything
    # is advertised: a readiness probe that succeeds means a request can run.
    warm = [_Row(ids=[0] * 8, max_tokens=2, temperature=0.0, top_p=1.0)]
    run_batch(stage, guard, warm, prefill_step)
    wake = _Wake(group)
    context = options.context or plan.context
    store = None if options.no_prefix_cache else _PrefixStore()
    if not stage.is_first:
        emit(
            "READY",
            {
                "rank": stage.rank,
                "pid": os.getpid(),
                "layers": [stage.start, stage.end],
            },
        )
        _worker_loop(stage, guard, options.max_batch, prefill_step, wake, store)
        return 0

    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(model_dir)
    kv = _KvBudget(plan, prefill_step)
    state = _State(
        served=options.served_model_name,
        context=context,
        max_batch=options.max_batch,
        kv=kv,
        prefix=None if store is None else _PrefixIndex(kv),
        eos_ids=frozenset(tokenizer.eos_token_ids),
    )

    import uvicorn

    app = _build_app(state, tokenizer, set(tokenizer.eos_token_ids), stage.vision)
    server = uvicorn.Server(
        uvicorn.Config(app, host=options.host, port=options.port, log_level="warning")
    )
    http_started = threading.Event()
    http_loop: list[asyncio.AbstractEventLoop] = []

    @app.on_event("startup")
    async def _capture_loop() -> None:
        http_loop.append(asyncio.get_running_loop())
        http_started.set()

    threading.Thread(target=server.run, name="pipeline-http", daemon=True).start()
    http_started.wait()

    def request_shutdown() -> None:
        state.jobs.put(None)

    def shutdown(*_):
        # A signal handler must not take the job queue's lock: the main
        # thread may hold it inside ``get()`` when the signal lands (measured:
        # the handler's put() deadlocked rank 0 and left rank 1 waiting in
        # its header all_sum).  The HTTP loop's thread does the put instead.
        state.shutting_down = True
        for job in list(state.active):
            job.cancelled = True
        http_loop[0].call_soon_threadsafe(request_shutdown)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    emit(
        "READY",
        {
            "rank": 0,
            "pid": os.getpid(),
            "layers": [stage.start, stage.end],
            "port": options.port,
            "served": state.served,
            "context": context,
            "starts": plan.starts,
            "vision": stage.vision is not None,
            "prefix_cache": store is not None,
        },
    )
    try:
        _rank0_loop(stage, guard, state, prefill_step, wake, store)
    finally:
        server.should_exit = True
    return 0


def add_arguments(parser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument(
        "--slots",
        type=int,
        default=2,
        help="full-context sequences the split is planned (and KV-budgeted) for",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=2,
        help="most rows one batch may carry; the KV budget decides how many do",
    )
    parser.add_argument("--prefill-step", type=int)
    parser.add_argument(
        "--split", help="pinned starts for ranks 1..N-1 (the preflighted plan)"
    )
    parser.add_argument(
        "--no-prefix-cache",
        action="store_true",
        help="keep no prefix cache: every request prefills its whole prompt",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="serve text only: rank 0 does not load the checkpoint's vision tower",
    )


if __name__ == "__main__":
    sys.exit(pipe.main(["serve", *sys.argv[1:]]))
