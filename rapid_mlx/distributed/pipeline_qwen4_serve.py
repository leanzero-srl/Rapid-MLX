# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible server over the qwen4_exp pipeline split.

Every rank runs :func:`serve`.  Rank 0 also runs an HTTP thread (FastAPI +
uvicorn) and owns the scheduling; the generation loop stays on each rank's
main thread because MLX work and the collectives must be issued in the same
order on every rank.  Rank 0 hands each change to the other ranks as a PLAN
(``_broadcast_plan``: an int header, then the joining row's token ids and
sampling floats); every decode step and prefill chunk closes with one
``all_sum`` that carries the sampled tokens, the memory-guard stop flag and
rank 0's control words (a plan opens the next tick; a prefill chunk runs
after this decode step).

What the HTTP side reuses from the single engine, unchanged:
``utils.chat_template.apply_chat_template`` (the same rendering, tool-loop
think handling included), ``api.tool_calling.convert_tools_for_template`` and
``engine.batched._normalize_tool_call_arguments_for_template``, and the
route layer's ``service.postprocessor.StreamingPostProcessor`` configured with
the parsers the checkpoint's own chat template declares (the parameterized
XML tool contract -> ``qwen3_coder_xml``; ``<think>`` + ``enable_thinking``
-> ``deepseek_r1``) — the same rule goose-sidecar's ``model_parsers.rs``
applies when it launches the single engine.

Scheduling is continuous (Q-134).  Every rank runs the same TICK: one decode
step for the running rows, then — when rank 0 grants it — one prefill chunk
of the ONE request that is joining.  A queued request joins the moment a row
slot is free (``--max-batch``) and its KV fits every rank's planned budget
beside the running rows (``_KvBudget``: rows x the longest reservation); it
prefills in its own cache, ``--prefill-step`` tokens a chunk (the chunk
edges a solo run uses), samples its first token from its last chunk, and is
merged into the running batch by extracting each row's caches and merging
them again (mlx-lm's own batching path; ``filter`` is not used — measured,
``QSAIndexCache.filter`` desyncs from ``BatchKVCache.filter`` when the
surviving row was the padded one).  A finished or cancelled row leaves the
same way at the next plan.  While both have work, the joining prefill and
the running decode share the pipeline's time equally (``_PREFILL_SHARE``,
rank 0's measured step and chunk times): the share decides only WHEN a
chunk runs — never whether a request is admitted, nor where its prompt is
cut.  A plan is broadcast only when the previous collective announced one,
so a steady decode pays no extra collective.  Before Q-134 a batch formed
only when the previous one ended and a request the prefix cache acted on
ran alone: one long generation froze every other request (a 69-token
canary waited 308 s beside a free slot and half an unreserved KV budget).

Prefix cache (Q-75): every rank snapshots ITS OWN layers' caches at a stable
prompt boundary and restores them for a later prompt that starts with the same
tokens.  The layers are hybrid (GDN/PLE recurrent state cannot be trimmed), so
only an exact stored prefix is reusable, and every rank must reuse the SAME
prefix length or the collectives stop pairing.  So rank 0 alone decides — which
entry to restore, where to snapshot, what to evict — and the plan that admits
the request carries the decision; the other ranks hold snapshots by the id rank 0 assigned
and never decide anything.  The boundary is the single engine's
(``BatchedEngine._compute_prefix_boundary``, the ``rapid_mlx_transient_tail``
extension included).  Bytes: the cache lives inside each rank's planned KV
budget — at every admission it is trimmed to that budget minus the batch's
own reservation, the new snapshot pre-charged — so it never holds memory the
plan did not.  The cache acts on a request while it prefills in its own
cache, before it joins the batch, so a restored prefix never shares a
padded prefill.
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
_CMD_PLAN = 2
# Rank 0's words on every step's collective: [a plan opens the next tick,
# a prefill chunk of the joining row runs after this decode step].
_CONTROL_WORDS = 2
# The running rows' decode and a joining row's prefill split the pipeline's
# time equally while both have work.  A policy ratio (Q-134): the owner weighs
# the running row's decode rate and the queued request's wait alike.  Q-103's
# single-engine fair prefill shrank the chunks instead (50-60 tokens a row)
# and cost decode -67%; here the chunk stays the planned --prefill-step and
# the share only spaces the chunks out between decode steps.
_PREFILL_SHARE = 0.5
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
    # plan that admits the row carries it): restore entry ``reuse_id`` holding
    # the first ``cached`` tokens, and snapshot entry ``store_id`` after the
    # first ``store_at`` tokens.
    reuse_id: int = 0
    cached: int = 0
    store_id: int = 0
    store_at: int = 0
    # Rank 0 only: the stable-prefix boundary the HTTP side computed (0 = the
    # request asks the cache for nothing: images, or no boundary).
    boundary: int = 0


@dataclass
class _Plan:
    """What changes at the start of a tick, decided by rank 0 alone."""

    # Running-row indices (batch order) that leave: finished or cancelled.
    leave: list[int] = field(default_factory=list)
    # The joining row leaves before it joined (its request was cancelled).
    abort: bool = False
    # The request that starts prefilling (at most one prefills at a time).
    joiner: _Row | None = None
    # Prefix-cache entries every rank drops once the joiner restored its own.
    evict: list[int] = field(default_factory=list)


def _all_sum(group, value: mx.array) -> mx.array:
    """Every collective of the tick loop (one seam, so a test can replay them)."""
    if group is None or group.size() == 1:
        return value
    return mx.distributed.all_sum(value, group=group)


# The plan header after [cmd, abort, one leave flag per slot]: the joiner's
# fields (length 0 = no joiner), then the eviction count.
_PLAN_TAIL = len(
    ("length", "max_tokens", "reuse_id", "cached", "store_id", "store_at", "evictions")
)


def _broadcast_plan(
    group, plan: _Plan | None, max_batch: int, *, deciding: bool
) -> _Plan | None:
    """Rank 0's plan, identical on every rank; None = shut down.

    ``deciding`` is rank 0 (``plan`` None there means shut down); every other
    rank passes None and learns the plan from the collectives alone.
    """
    tail = 2 + max_batch
    header = [0] * (tail + _PLAN_TAIL)
    if deciding:
        if plan is None:
            header[0] = _CMD_SHUTDOWN
        else:
            header[0] = _CMD_PLAN
            header[1] = int(plan.abort)
            for index in plan.leave:
                header[2 + index] = 1
            row = plan.joiner
            if row is not None:
                header[tail : tail + 6] = [
                    len(row.ids),
                    row.max_tokens,
                    row.reuse_id,
                    row.cached,
                    row.store_id,
                    row.store_at,
                ]
            header[tail + 6] = len(plan.evict)
    header = _all_sum(group, mx.array(header, dtype=mx.int32)).tolist()
    if header[0] == _CMD_SHUTDOWN:
        return None
    if header[0] != _CMD_PLAN:
        raise RuntimeError(
            f"pipeline plan: unknown command {header[0]} — the ranks diverged"
        )
    length = header[tail]
    joiner = None
    if length:
        ids = plan.joiner.ids if deciding else [0] * length
        floats = (
            [plan.joiner.temperature, plan.joiner.top_p] if deciding else [0.0, 0.0]
        )
        ids = _all_sum(group, mx.array(ids, dtype=mx.int32)).tolist()
        floats = _all_sum(group, mx.array(floats, dtype=mx.float32)).tolist()
        if deciding:
            joiner = plan.joiner
        else:
            joiner = _Row(
                ids=ids,
                max_tokens=header[tail + 1],
                temperature=floats[0],
                top_p=floats[1],
            )
            joiner.reuse_id, joiner.cached, joiner.store_id, joiner.store_at = header[
                tail + 2 : tail + 6
            ]
    evictions = header[tail + 6]
    evict = []
    if evictions:
        dropped = list(plan.evict) if deciding else [0] * evictions
        evict = _all_sum(group, mx.array(dropped, dtype=mx.int32)).tolist()
    return _Plan(
        leave=[index for index in range(max_batch) if header[2 + index]],
        abort=bool(header[1]),
        joiner=joiner,
        evict=evict,
    )


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
    stage, out, cache, rows, guard, words: list[int] | None, *, sample: bool
) -> tuple[list[int], list[int]]:
    """One step's collective: tokens, the guard's stop flag, rank 0's words.

    ``words`` are rank 0's ``_CONTROL_WORDS`` (None elsewhere: zeros).
    """
    batch = len(rows)
    reason = guard.check() if guard is not None else None
    if stage.is_last and sample:
        tokens = _sample(out[:, -1, :], rows)
    else:
        tokens = mx.depends(mx.zeros((batch,), dtype=mx.int32), out)
    control = (
        list(words) if stage.is_first and words is not None else [0] * _CONTROL_WORDS
    )
    payload = mx.concatenate(
        [tokens, mx.array([1 if reason else 0, *control], dtype=mx.int32)]
    )
    payload = _all_sum(stage.group, payload)
    mx.eval(payload, [layer_cache.state for layer_cache in cache])
    values = payload.tolist()
    if values[batch]:
        raise pipe.PipelineMemoryStopError(
            reason or "a peer rank's memory guard tripped"
        )
    return values[:batch], values[batch + 1 :]


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
    kib = _all_sum(stage.group, mx.array(kib, dtype=mx.int32)).tolist()
    return [value * 1024 for value in kib]


def _batch_rope(ropes: list[Any]):
    """One batch's RoPE table from its rows' own (None = a text row); None if all text.

    A text row carries a zero-length table, so every position it asks for
    rotates at ``position + 0`` — exactly what ``MRopePositions`` does past a
    row's prompt.
    """
    if all(rope is None for rope in ropes):
        return None
    from ..models.qwen4_exp_vision import MRopePositions

    width = max(rope.table.shape[2] for rope in ropes if rope is not None)
    tables, lengths, deltas = [], [], []
    for rope in ropes:
        if rope is None:
            tables.append(mx.zeros((3, 1, width), dtype=mx.int64))
            lengths.append(mx.zeros((1,), dtype=mx.int64))
            deltas.append(mx.zeros((1,), dtype=mx.int64))
            continue
        pad = width - rope.table.shape[2]
        tables.append(
            mx.pad(rope.table, [(0, 0), (0, 0), (0, pad)]) if pad else rope.table
        )
        lengths.append(rope.lengths)
        deltas.append(rope.deltas)
    return MRopePositions(
        table=mx.concatenate(tables, axis=1),
        lengths=mx.concatenate(lengths),
        deltas=mx.concatenate(deltas),
    )


@dataclass
class _Joining:
    """The one row still prefilling, in its own cache, before it joins the batch."""

    row: _Row
    cache: list[Any]
    tokens: mx.array
    embeddings: Any
    rope: Any
    ranges: list[tuple[int, int]]


class _Engine:
    """One rank's rows: the running batch and at most one row still prefilling.

    Every rank holds the same rows in the same order and applies the same
    plans, so every forward and every collective pairs across ranks; what a
    plan holds is rank 0's decision alone (``_Scheduler``).
    """

    def __init__(self, stage, guard, prefill_step: int, store=None, on_stored=None):
        self.stage = stage
        self.guard = guard
        self.prefill_step = prefill_step
        self.store = store
        # Rank 0: ``on_stored(row, bytes_per_rank)`` once every rank holds a snapshot.
        self.on_stored = on_stored
        self.rows: list[_Row] = []
        self.cache: list[Any] | None = None
        self.current: list[int] = []
        self.ropes: list[Any] = []
        self.rope = None
        self.joining: _Joining | None = None

    @property
    def idle(self) -> bool:
        return not self.rows and self.joining is None

    @property
    def last_chunk(self) -> bool:
        return self.joining is not None and len(self.joining.ranges) == 1

    def apply(self, plan: _Plan) -> None:
        if plan.abort:
            self.joining = None
        if plan.leave:
            leaving = set(plan.leave)
            self._regroup([i for i in range(len(self.rows)) if i not in leaving])
        if plan.joiner is not None:
            self._start(plan.joiner)
        if self.store is not None and plan.evict:
            # After the restore copied its entry: an evicted entry may be the
            # one this joiner restores from.
            self.store.drop(plan.evict)

    def _start(self, row: _Row) -> None:
        if self.joining is not None:
            raise RuntimeError(
                "pipeline plan: a second row started prefilling beside the first "
                "— rank 0 admits one at a time"
            )
        embeddings, rope = pipe.prepare_multimodal(
            self.stage,
            [row.ids],
            [row.images] if self.stage.is_first else None,
        )
        directed = row.reuse_id or row.store_id
        if directed and (
            self.store is None or embeddings is not None or rope is not None
        ):
            raise RuntimeError(
                "prefix cache: a directive reached a rank without a store, or a row "
                "with images (rank 0 never directs either)"
            )
        if row.reuse_id:
            cache, start = self.store.take(row.reuse_id), row.cached
        else:
            cache, start = self.stage.make_cache(), 0
        self.joining = _Joining(
            row=row,
            cache=cache,
            tokens=mx.array([row.ids], dtype=mx.int32),
            embeddings=embeddings,
            rope=rope,
            ranges=prefill_chunks(
                start,
                len(row.ids),
                self.prefill_step,
                row.store_at if row.store_id else 0,
            ),
        )

    def decode(self, words: list[int] | None) -> tuple[list[int], list[int]]:
        """One decode step of every running row; its tokens and rank 0's words."""
        out = self.stage.forward(
            mx.array(self.current, dtype=mx.int32)[:, None],
            self.cache,
            logits="last",
            rope_positions=self.rope,
        )
        sampled, control = _step(
            self.stage, out, self.cache, self.rows, self.guard, words, sample=True
        )
        self.current = sampled
        return sampled, control

    def prefill(self, words: list[int] | None) -> tuple[int | None, list[int]]:
        """The joining row's next chunk; its first token when the prompt is done.

        The last chunk ends at the prompt's last token and samples from it; the
        row then joins the running batch with that token as its next input.
        """
        joining = self.joining
        start, stop = joining.ranges.pop(0)
        last = not joining.ranges
        out = self.stage.forward(
            joining.tokens[:, start:stop],
            joining.cache,
            logits="last" if last else None,
            embeddings=None
            if joining.embeddings is None
            else joining.embeddings[:, start:stop],
            rope_positions=joining.rope,
        )
        sampled, control = _step(
            self.stage,
            out,
            joining.cache,
            [joining.row],
            self.guard,
            words,
            sample=last,
        )
        row = joining.row
        if row.store_id and stop == row.store_at:
            measured = _agree_bytes(
                self.stage, self.store.put(row.store_id, joining.cache)
            )
            if self.on_stored is not None:
                self.on_stored(row, measured)
        if not last:
            return None, control
        self._regroup(list(range(len(self.rows))), joining, sampled[0])
        return sampled[0], control

    def _regroup(
        self, keep: list[int], joining: _Joining | None = None, first: int = 0
    ) -> None:
        """Rebuild the running batch from the kept rows (+ the joined row), layer by layer.

        Each kept row's cache is extracted and the rows are merged again (a lone
        row keeps its own, unbatched cache).  One layer at a time, evaluated
        before the next, so the rebuild never holds more than one layer twice.
        """
        count = len(self.rows)
        rebuilt = self.cache if self.cache is not None else list(joining.cache)
        for index in range(len(rebuilt)):
            parts: list[Any] = []
            if self.cache is not None:
                layer = self.cache[index]
                if count == 1:
                    parts = [layer] if keep == [0] else []
                else:
                    parts = [layer.extract(row) for row in keep]
            if joining is not None:
                parts.append(joining.cache[index])
            if not parts:
                merged = None
            elif len(parts) == 1:
                merged = parts[0]
            else:
                merged = type(parts[0]).merge(parts)
            if merged is not None:
                mx.eval(merged.state)
            rebuilt[index] = merged
        self.rows = [self.rows[i] for i in keep]
        self.current = [self.current[i] for i in keep]
        self.ropes = [self.ropes[i] for i in keep]
        if joining is not None:
            self.rows.append(joining.row)
            self.current.append(first)
            self.ropes.append(joining.rope)
            self.joining = None
        self.cache = rebuilt if self.rows else None
        self.rope = _batch_rope(self.ropes) if self.rows else None


def _warm(engine: _Engine) -> None:
    """Kernel compilation and first-touch paging, before anything is advertised.

    The same path a request takes — prefill, sample, join, decode, leave —
    identical on every rank, so no plan is broadcast.
    """
    engine.apply(
        _Plan(joiner=_Row(ids=[0] * 8, max_tokens=2, temperature=0.0, top_p=1.0))
    )
    while engine.joining is not None:
        engine.prefill(None)
    engine.decode(None)
    engine.apply(_Plan(leave=[0]))


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

    def admit(self, row: _Row, lengths: list[int]) -> list[int]:
        """Set the joining row's directive; return the entries to evict.

        ``lengths`` are the reservations of every row the batch will hold once
        the row joins, its own included.
        """
        with self.lock:
            room = [
                budget - need
                for budget, need in zip(self.kv.budgets, self.kv.reserve(lengths))
            ]
            new = [0] * len(room)
            if row.images is None:
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
                row.store_id = row.store_at = 0
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
    # Further names the served model answers to (``--served-model-alias``),
    # listed after ``served`` on /v1/models.
    aliases: tuple[str, ...] = ()
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


def _refusal_fields(processor, served: str) -> dict[str, Any]:
    """The finishing choice's ``refused_tool_calls``, when the parser refused any.

    goose Q-133: Flash called ``bash`` on the split while the request declared
    ``shell``; the parser refuses an undeclared name (never executable), so the
    call went out as content and goose showed the XML as the finished reply.
    The refusal stands — this only says it happened: each refused call's name
    rides the final choice (streamed and not), and the server log names it.
    """
    refused = processor.refused_tool_calls()
    if not refused:
        return {}
    print(
        "PIPELINE_TOOL_CALL_REFUSED "
        + json.dumps({"model": served, "refused_tool_calls": refused}),
        flush=True,
    )
    return {"refused_tool_calls": refused}


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


def _served_names(state: _State) -> list[str]:
    """Every name the served model answers to, the served name first."""
    return [state.served, *(name for name in state.aliases if name != state.served)]


def _not_served(state: _State, model: str) -> str:
    also = [name for name in state.aliases if name != state.served]
    names = f" (also answering to {', '.join(repr(n) for n in also)})" if also else ""
    return f"model '{model}' is not served here; this engine serves '{state.served}'{names}"


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
                    "id": name,
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
                for name in _served_names(state)
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
        if model is not None and model not in _served_names(state):
            return error(404, _not_served(state, model), "model_not_found")
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
                def chunk(delta, finish=None, usage=None, refusal=None):
                    payload = {
                        "id": f"chatcmpl-{job.id}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": state.served,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": finish,
                                **(refusal or {}),
                            }
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
                yield chunk({}, finish, usage, _refusal_fields(processor, state.served))
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
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish,
                    **_refusal_fields(processor, state.served),
                }
            ],
            "usage": {
                "prompt_tokens": len(ids),
                "completion_tokens": completion,
                "total_tokens": len(ids) + completion,
                "prompt_tokens_details": {"cached_tokens": job.row.cached},
            },
        }

    return app


class _Scheduler:
    """Rank 0's decisions: which rows leave, which queued request joins, when its chunks run.

    Admission is by slots and memory alone: the oldest live queued request
    joins when no other request is prefilling, a row slot is free
    (``max_batch``) and ``_KvBudget`` fits every row the batch would hold on
    every rank.  First come, first served: a request that does not fit waits
    at the head and nothing behind it jumps the line.  The joining prefill's
    chunks are spaced by ``_PREFILL_SHARE`` of the measured pipeline time.
    """

    def __init__(self, state: _State, engine: _Engine):
        self.state = state
        self.engine = engine
        self.running: list[_Job] = []
        self.joining: _Job | None = None
        # The request the plan being broadcast admits (joining once applied).
        self.admitted: _Job | None = None
        self.stopping = False
        self.fitted: dict[tuple[int, ...], bool] = {}
        # Seconds of pipeline time the joining prefill may still spend before
        # the running rows' decode has had its share (both measured here).
        self.credit = 0.0
        if state.prefix is not None:
            engine.on_stored = self._stored

    def _stored(self, row: _Row, measured: list[int]) -> None:
        self.state.prefix.stored(row.store_id, tuple(row.ids[: row.store_at]), measured)

    def _head(self, block: bool) -> _Job | None:
        """The oldest live queued request (it stays held until it joins)."""
        state = self.state
        while not self.stopping:
            if state.held is not None:
                if not state.held.cancelled:
                    return state.held
                state.held = None
            try:
                job = state.jobs.get() if block else state.jobs.get_nowait()
            except queue.Empty:
                return None
            if job is None:
                self.stopping = True
            else:
                state.held = job
        return None

    def _fits(self, job: _Job, beside: list[_Job]) -> bool:
        if len(beside) + 1 > self.state.max_batch:
            return False
        kv = self.state.kv
        if kv is None:
            return True
        # Asked before every decode step while a request waits: the same
        # question until the rows change, and each plan clears the memo.
        lengths = tuple(_reservation_length(other.row) for other in [*beside, job])
        if lengths not in self.fitted:
            self.fitted[lengths] = kv.fits(list(lengths))
        return self.fitted[lengths]

    def _leaving(self) -> list[int]:
        return [
            index
            for index, job in enumerate(self.running)
            if job.finished or job.cancelled
        ]

    def wants_plan(self, joined: bool) -> bool:
        """Whether the next tick opens with a plan (``joined``: the joiner joins this tick)."""
        if self.stopping or self.state.shutting_down or self._leaving():
            return True
        if self.joining is not None:
            if self.joining.cancelled:
                return True
            if not joined:
                return False
        beside = [*self.running, *([self.joining] if self.joining else [])]
        head = self._head(block=False)
        return self.stopping or (head is not None and self._fits(head, beside))

    def plan(self, block: bool) -> _Plan | None:
        """The next tick's plan; None = shut down.  ``block``: the engine is idle."""
        state = self.state
        leave = self._leaving()
        abort = self.joining is not None and self.joining.cancelled
        survivors = [job for i, job in enumerate(self.running) if i not in leave]
        joiner, evict = None, []
        while not (self.stopping or state.shutting_down):
            if (self.joining is not None and not abort) or len(
                survivors
            ) >= state.max_batch:
                break
            head = self._head(block=block and not survivors)
            if head is None:
                break
            if self._fits(head, survivors):
                state.held = None
                self.admitted = head
                joiner = head.row
                if state.prefix is not None:
                    evict = state.prefix.admit(
                        head.row,
                        [_reservation_length(job.row) for job in [*survivors, head]],
                    )
                break
            if not survivors:
                # Alone it still does not fit: the plan cannot hold it at all
                # (a lone row at <= context always fits the plan's first slot,
                # so this names a planner defect rather than waiting forever).
                state.held = None
                needed = state.kv.reserve([_reservation_length(head.row)])
                head.finished = True
                head.push(
                    (
                        "error",
                        f"the pipeline's KV budget {state.kv.budgets} cannot hold "
                        f"this request alone ({needed} bytes per rank)",
                    )
                )
                continue
            break
        if self.stopping or state.shutting_down:
            return None
        return _Plan(leave=leave, abort=abort, joiner=joiner, evict=evict)

    def applied(self, plan: _Plan) -> None:
        self.fitted.clear()
        leaving = set(plan.leave)
        self.running = [job for i, job in enumerate(self.running) if i not in leaving]
        if plan.abort:
            self.joining = None
        if plan.joiner is not None:
            self.joining, self.admitted = self.admitted, None
            self.credit = 0.0
        self._publish()

    def _publish(self) -> None:
        active = [*self.running, *([self.joining] if self.joining else [])]
        kv = self.state.kv
        self.state.reserved = (
            kv.reserve([_reservation_length(job.row) for job in active])
            if kv is not None and active
            else []
        )
        self.state.active = active

    def _give(self, job: _Job, token: int) -> None:
        if job.finished or job.cancelled:
            return
        job.produced += 1
        job.push(("token", token))
        if token in self.state.eos_ids:
            job.finished = True
            job.push(("done", "stop"))
        elif job.produced >= job.row.max_tokens:
            job.finished = True
            job.push(("done", "length"))

    def decode_words(self) -> list[int]:
        chunk = self.joining is not None and self.credit >= 0
        # When a chunk follows, its own collective carries the plan word.
        plan = False if chunk else self.wants_plan(joined=False)
        return [int(plan), int(chunk)]

    def decoded(self, tokens: list[int], seconds: float) -> None:
        self.state.steps += 1
        for job, token in zip(self.running, tokens):
            self._give(job, token)
        if self.joining is not None:
            self.credit += seconds * _PREFILL_SHARE / (1 - _PREFILL_SHARE)

    def chunk_words(self, last: bool) -> list[int]:
        return [int(self.wants_plan(joined=last)), 0]

    def chunked(self, first: int | None, seconds: float) -> None:
        self.credit -= seconds
        if first is None:
            return
        job, self.joining = self.joining, None
        self.running.append(job)
        self._give(job, first)
        self._publish()

    def tell(self, message: str) -> None:
        """Every request the engine holds or has queued ends with ``message``."""
        held = [self.state.held] if self.state.held is not None else []
        for job in [*self.running, *([self.joining] if self.joining else []), *held]:
            if not job.finished:
                job.finished = True
                job.push(("error", message))


def _ticks(
    engine: _Engine,
    group,
    max_batch: int,
    wake: _Wake,
    scheduler: _Scheduler | None = None,
) -> None:
    """The tick loop every rank runs; only rank 0 passes its ``scheduler``.

    Every branch below depends only on what every rank holds (the rows, the
    joining row, the collectives' words), so the forwards and collectives
    pair across ranks.
    """
    deciding = scheduler is not None
    pending = True
    while True:
        if pending:
            plan = None
            if engine.idle:
                if deciding:
                    plan = scheduler.plan(block=True)
                    wake.ring()
                elif not wake.wait():
                    return
            elif deciding:
                plan = scheduler.plan(block=False)
            plan = _broadcast_plan(group, plan, max_batch, deciding=deciding)
            if plan is None:
                return
            engine.apply(plan)
            if deciding:
                scheduler.applied(plan)
            pending = engine.idle
            if pending:
                continue
        chunk = engine.joining is not None
        if engine.rows:
            words = scheduler.decode_words() if deciding else None
            began = time.perf_counter()
            tokens, words = engine.decode(words)
            if deciding:
                scheduler.decoded(tokens, time.perf_counter() - began)
            pending, chunk = bool(words[0]), chunk and bool(words[1])
        if chunk:
            words = scheduler.chunk_words(engine.last_chunk) if deciding else None
            began = time.perf_counter()
            first, words = engine.prefill(words)
            if deciding:
                scheduler.chunked(first, time.perf_counter() - began)
            pending = bool(words[0])
        pending = pending or engine.idle


def _rank0_loop(state: _State, engine: _Engine, group, wake: _Wake) -> None:
    scheduler = _Scheduler(state, engine)
    try:
        _ticks(engine, group, state.max_batch, wake, scheduler)
    except pipe.PipelineMemoryStopError as stop:
        scheduler.tell(f"memory guard stopped the pipeline: {stop}")
        raise
    scheduler.tell("the pipeline engine is shutting down")


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
        attention_scores_bytes=options.attention_scores_bytes,
        log=lambda line: print(line, flush=True),
    )
    emit("RANK_CAPS", {**stage.limits, "planned": plan.stages[stage.rank].total_bytes})
    store = None if options.no_prefix_cache else _PrefixStore()
    engine = _Engine(stage, guard, prefill_step, store)
    # A readiness probe that succeeds means a request can run.
    _warm(engine)
    wake = _Wake(group)
    context = options.context or plan.context
    if not stage.is_first:
        emit(
            "READY",
            {
                "rank": stage.rank,
                "pid": os.getpid(),
                "layers": [stage.start, stage.end],
            },
        )
        _ticks(engine, group, options.max_batch, wake)
        return 0

    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(model_dir)
    kv = _KvBudget(plan, prefill_step)
    state = _State(
        served=options.served_model_name,
        aliases=tuple(options.served_model_alias or ()),
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
            "aliases": list(state.aliases),
            "context": context,
            "starts": plan.starts,
            "vision": stage.vision is not None,
            "prefix_cache": store is not None,
        },
    )
    try:
        _rank0_loop(state, engine, group, wake)
    finally:
        server.should_exit = True
    return 0


def add_arguments(parser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument(
        "--served-model-alias",
        dest="served_model_alias",
        action="append",
        default=None,
        metavar="NAME",
        help="Another name the served model answers to (repeatable); listed after "
        "--served-model-name on /v1/models. Any other name is refused.",
    )
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
        "--attention-scores-bytes",
        type=int,
        default=0,
        help="this rank's prefill attention scores at --prefill-step that the "
        "planner's workspace model leaves out, as the requester charged them; "
        "the MLX buffer cache holds the budget less the plan less these",
    )
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
