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
of ONE prefilling request (the one with the fewest prompt tokens left, Q-145).  A queued request joins the moment every
rank's planned budget has room for what it needs beside the rows already
there (``_KvBudget``, goose Q-160: its own prompt and max_tokens, what the
padded batch will hold until its last step, and what the running rows
will still grow into); it
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

Admission is by memory, never by a row count (goose Q-160).  The batch is
one merged cache with every row left-padded to the longest, so a row costs
the batch's width for as long as it is in it; but a row also LEAVES by its
horizon (prompt + max_tokens + 1), so a 600-token helper call with 256
tokens to write, joining two chats 45k tokens in, holds three 45.3k-token
rows for 256 steps and is gone long before the chats could grow into the
room it used.  ``_KvBudget.held`` prices exactly that: the largest of rows
alive x their width at each row's last step, prefilling rows at their
prompts in their own caches, the transient of one decode step or one solo
chunk.  A request is admitted when that fits with it prefilling, with every
prefilling row joined at once (so its own join never waits and stalls the
shorter-first order), and with any prefilling row grown to its horizon
beside the others (so no join waits for good once the batch has drained);
its last chunk — the merge — runs only once the batch it makes still fits,
prefix-cache entries yielding first.  The plan header's per-row flags are
sized by the most one-token rows the budgets hold, derived and agreed on
every rank, which admission never reaches.  Before Q-160 ``--max-batch``
(goose passed the planned slots, 2) capped the rows and every row was
priced at the longest horizon for its whole life: two chats that reserve the
whole context (goose sends them no max_tokens) filled both slots, and a
helper call waited for a whole multi-minute decode to leave.

Prefill order is shortest-remaining-prefill-first at chunk granularity
(Q-145).  Prefill is compute-bound, so the chunks still run one at a time;
what changed is WHICH request's chunk runs.  Several requests may be
prefilling at once, each in its own cache and each holding a row slot and
its KV reservation; every rank runs the next chunk of the prefilling row
with the fewest prompt tokens left (ties: the earlier admitted), a rule
over state every rank holds identically (the rows the plans admitted, the
ranges their chunks have consumed), so no extra word crosses the ranks.
Rank 0 alone decides admission: of the queued requests it picks the one
first in ``_Scheduler._head`` and admits it only when it would be that
rule's pick — fewer tokens left than every prefilling row — so a short
request (goose's title, fact checker, compaction) is prefilled between two
chunks of a long one and joins the running batch the moment its last chunk
samples.  The starvation bound, in prefill tokens (the pipeline's own
prefill rate cancels, so no clock), limits the delay JUMPERS cause: a
request's ``waited`` counts only the tokens of LATER arrivals prefilled
ahead of it.  Its SLACK is its own prefill less that count and less what
the later rows already admitted will still prefill ahead of it, and a
later arrival may go ahead of it only by taking at most half the slack it
finds.  So no stream of later requests stretches a request past twice its
own prefill, a long prompt barely shorter than an earlier one keeps
first-come order instead of spending that one's whole slack, and a short
request is held back only when an earlier one's slack is down to twice
its size.  The queueing a request would have had first come first
served never ages it — counting that too (fork 419306f70) protected every
long request before it started under goose Q-145 26d's steady load, and
canaries 3 and 4 waited 353 s and 297 s behind FIFO prefills.  Before
Q-145 one request prefilled at a time, first come first served: on the
Flash split three ~39k-token prompts and a 13-token canary arriving
together got their first tokens at 115 / 232 / 347 s, the canary last.

Prefix cache (Q-75): every rank snapshots ITS OWN layers' caches at a stable
prompt boundary and restores them for a later prompt that starts with the same
tokens.  The layers are hybrid (GDN/PLE recurrent state cannot be trimmed), so
only an exact stored prefix is reusable, and every rank must reuse the SAME
prefix length or the collectives stop pairing.  So rank 0 alone decides — which
entry to restore, where to snapshot, what to evict — and the plan that admits
the request carries the decision; the other ranks hold snapshots by the id rank 0 assigned
and never decide anything.  The boundary is the single engine's
(``BatchedEngine._compute_prefix_boundary``, the ``rapid_mlx_transient_tail``
extension included, on the last user or tool message — ``_on_tool``, so goose
keeps its turn-context block joined to the tool results instead of posting it
as a user turn of its own, goose Q-94/Q-143).  Bytes: the cache lives inside each rank's planned KV
budget — at every admission it is trimmed to that budget minus the batch's
own reservation, the new snapshot pre-charged — so it never holds memory the
plan did not.  The cache acts on a request while it prefills in its own
cache, before it joins the batch, so a restored prefix never shares a
padded prefill.  A snapshot owns exactly the bytes it is charged for
(``_own_bytes``, the single engine's lz.8 / goose Q-110): a stored view would
keep its whole source buffer alive beside the budget.

Sampling (goose Q-159): a request's own temperature / top_p / top_k / min_p
win; a field it leaves out (or sends null) takes the single engine's chain —
``--default-*`` (the operator's per-model profile), then the checkpoint's
``generation_config.json``, then the engine fallback (``_SamplingDefaults``).
The plan carries all four to the sampling rank.  Before Q-159 a request with no
temperature decoded greedy.  A missing or unreadable ``generation_config.json``
is named on /v1/status (``sampling_defaults.generation_config_error``) and in
the log (``GENERATION_CONFIG_UNREAD``); penalties in force are reported
``applied: false`` — the pipeline's sampler keeps no per-row token history.

Tool calls (the single engine's lz.7, goose Q-85): the last rank — the one
that samples — holds a tool request's decode to the XML tool-call skeleton
(``xml_tool_close_guard``) at the positions where the template admits no free
text, so ``</parameter>`` residue never becomes an argument's payload.  The
plan carries whether a row declared tools; the guard reads the row's prompt and
the tokens every rank learns from the step's collective.
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
from mlx_lm.models.cache import CacheList

from . import pipeline_qwen4 as pipe
from .pipeline_stream import Markers, StreamRelay, StreamWatch

_CMD_SHUTDOWN = 1
_CMD_PLAN = 2
# Rank 0's words on every step's collective: [a plan opens the next tick,
# a prefill chunk (of the row ``_Engine.target`` names) runs after this
# decode step].
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
# sampling defaults (goose Q-159)
# ---------------------------------------------------------------------------

# The sampling fields the single engine resolves through its chain
# (service/helpers.py ``_resolve_*``), in utils/generation_config.py's order.
_SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
)
# What the sampling rank applies (``_sample``).  The penalties need a per-row
# token history the pipeline's sampler does not keep: one in force is reported
# ``applied: false`` on /v1/status, never dropped unsaid.
_APPLIED_KEYS = frozenset(("temperature", "top_p", "top_k", "min_p"))


@dataclass
class _SamplingDefaults:
    """What a request that names no sampling field samples with — the single
    engine's chain below the request (service/helpers.py ``_cascade``): the
    operator's ``--default-*`` flags (goose's per-model profile), then the
    checkpoint's ``generation_config.json`` (filtered by the single engine's own
    ``load_generation_config_sampling``), then the engine fallback
    (``_FALLBACK_TEMPERATURE`` / ``_FALLBACK_TOP_P``; the rest stay off).  The
    alias catalog's layer never applies here: ``resolve_profile`` matches alias
    names and HF ids, and the split serves a local checkpoint directory.

    Before goose Q-159 the server read ``body.get("temperature") or 0.0``: every
    request goose sent (it names no sampling field) decoded GREEDY, where the
    single engine sampled the checkpoint's own temperature 1.0 / top_k 20 /
    top_p 0.95 — Qwen's card warns greedy decoding repeats endlessly.  The
    single engine skips a missing or unreadable ``generation_config.json``
    silently; here ``error`` names it on /v1/status and in the rank's log.
    """

    profile: dict = field(default_factory=dict)
    generation_config: dict = field(default_factory=dict)
    path: str | None = None
    error: str | None = "no checkpoint directory was read for generation_config.json"
    # Sampling keys the file carries that the single engine's filter drops.
    ignored: list = field(default_factory=list)

    @classmethod
    def load(cls, model_dir: Path, profile: dict) -> _SamplingDefaults:
        from ..utils.generation_config import load_generation_config_sampling

        path = Path(model_dir).expanduser() / "generation_config.json"
        profile = {k: v for k, v in profile.items() if v is not None}
        try:
            raw = json.loads(path.read_text())
        except FileNotFoundError:
            return cls(profile, {}, str(path), f"{path} is absent", [])
        except (OSError, ValueError) as exc:
            return cls(profile, {}, str(path), f"{path} is unreadable: {exc}", [])
        if not isinstance(raw, dict):
            return cls(profile, {}, str(path), f"{path} holds no JSON object", [])
        values = load_generation_config_sampling(str(path.parent))
        ignored = [
            f"{key}={raw[key]!r}"
            for key in _SAMPLING_KEYS
            if key in raw and key not in values
        ]
        return cls(profile, values, str(path), None, ignored)

    @staticmethod
    def profile_of(options) -> dict:
        return {key: getattr(options, f"default_{key}", None) for key in _SAMPLING_KEYS}

    def report(self) -> dict[str, Any]:
        from ..service.helpers import _FALLBACK_TEMPERATURE, _FALLBACK_TOP_P

        return {
            "profile": dict(self.profile),
            "generation_config": dict(self.generation_config),
            "generation_config_path": self.path,
            "generation_config_error": self.error,
            "generation_config_ignored": list(self.ignored),
            "engine_fallback": {
                "temperature": _FALLBACK_TEMPERATURE,
                "top_p": _FALLBACK_TOP_P,
            },
            "unapplied": sorted(
                key
                for key in {*self.profile, *self.generation_config}
                if key not in _APPLIED_KEYS
            ),
        }

    def resolve(self, body: dict) -> dict[str, dict[str, Any]]:
        """Every sampling field this request runs with and the layer it came
        from (``request`` / ``profile`` / ``generation_config`` /
        ``engine_fallback`` / ``unset``).  A request's null is no value, as on
        the single engine (its pydantic field is None either way).  A value of
        the wrong type is a ValueError naming the field."""
        from ..service.helpers import _FALLBACK_TEMPERATURE, _FALLBACK_TOP_P

        fallback = {"temperature": _FALLBACK_TEMPERATURE, "top_p": _FALLBACK_TOP_P}
        resolved = {}
        for key in _SAMPLING_KEYS:
            if body.get(key) is not None:
                value, source = body[key], "request"
            elif key in self.profile:
                value, source = self.profile[key], "profile"
            elif key in self.generation_config:
                value, source = self.generation_config[key], "generation_config"
            elif key in fallback:
                value, source = fallback[key], "engine_fallback"
            else:
                value, source = None, "unset"
            if value is not None:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError(
                        f"{key} must be a finite number, not {value!r} ({source})"
                    )
                if key == "top_k":
                    if value != int(value):
                        raise ValueError(f"top_k must be a whole number, not {value}")
                    value = int(value)
                else:
                    value = float(value)
            entry = {"value": value, "from": source}
            if key not in _APPLIED_KEYS and value is not None:
                entry["applied"] = False
            resolved[key] = entry
        return resolved


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
    # Identical on every rank (the plan carries it): the request declared
    # tools, so the rank that samples holds its decode to the XML tool-call
    # skeleton (``xml_tool_close_guard``, the single engine's lz.7, goose Q-85).
    tools: bool = False
    # Identical on every rank (the plan carries them with temperature and
    # top_p): mlx-lm's own off values (0 = no top-k, 0.0 = no min-p).
    top_k: int = 0
    min_p: float = 0.0
    # Rank 0 only: every sampling field and the layer it came from
    # (``_SamplingDefaults.resolve``), for /v1/status.
    sampling: Any = field(default=None, compare=False, repr=False)
    # The sampling rank only: that guard and the token history it reads
    # (the prompt, then every sampled token).
    close_guard: Any = field(default=None, compare=False, repr=False)
    history: Any = field(default=None, compare=False, repr=False)
    # Every rank: what the row's cache will not give back at ``store_at``
    # (``_boundary_record``), until it leaves and its cache becomes the entry.
    record: Any = field(default=None, compare=False, repr=False)


@dataclass
class _Plan:
    """What changes at the start of a tick, decided by rank 0 alone."""

    # Running-row indices (batch order) that leave: finished or cancelled.
    leave: list[int] = field(default_factory=list)
    # Prefilling-row indices (admission order) that leave before they joined
    # (their requests were cancelled).
    abort: list[int] = field(default_factory=list)
    # The request that starts prefilling (appended to the prefilling rows).
    joiner: _Row | None = None
    # Prefix-cache entries every rank drops once the joiner restored its own.
    evict: list[int] = field(default_factory=list)


def _all_sum(group, value: mx.array) -> mx.array:
    """Every collective of the tick loop (one seam, so a test can replay them)."""
    if group is None or group.size() == 1:
        return value
    return mx.distributed.all_sum(value, group=group)


# The plan header after [cmd, one leave flag per slot, one abort flag per
# slot]: the joiner's fields (length 0 = no joiner), then the eviction count.
# Running and prefilling rows share the slots, so neither list outgrows them.
_PLAN_TAIL = len(
    (
        "length",
        "max_tokens",
        "reuse_id",
        "cached",
        "store_id",
        "store_at",
        "tools",
        "evictions",
    )
)


def _broadcast_plan(
    group, plan: _Plan | None, max_batch: int, *, deciding: bool
) -> _Plan | None:
    """Rank 0's plan, identical on every rank; None = shut down.

    ``deciding`` is rank 0 (``plan`` None there means shut down); every other
    rank passes None and learns the plan from the collectives alone.
    """
    tail = 1 + 2 * max_batch
    header = [0] * (tail + _PLAN_TAIL)
    if deciding:
        if plan is None:
            header[0] = _CMD_SHUTDOWN
        else:
            header[0] = _CMD_PLAN
            for index in plan.leave:
                header[1 + index] = 1
            for index in plan.abort:
                header[1 + max_batch + index] = 1
            row = plan.joiner
            if row is not None:
                header[tail : tail + 7] = [
                    len(row.ids),
                    row.max_tokens,
                    row.reuse_id,
                    row.cached,
                    row.store_id,
                    row.store_at,
                    int(row.tools),
                ]
            header[tail + 7] = len(plan.evict)
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
            [
                plan.joiner.temperature,
                plan.joiner.top_p,
                plan.joiner.top_k,
                plan.joiner.min_p,
            ]
            if deciding
            else [0.0] * 4
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
                top_k=int(round(floats[2])),
                min_p=floats[3],
            )
            joiner.reuse_id, joiner.cached, joiner.store_id, joiner.store_at = header[
                tail + 2 : tail + 6
            ]
            joiner.tools = bool(header[tail + 6])
    evictions = header[tail + 7]
    evict = []
    if evictions:
        dropped = list(plan.evict) if deciding else [0] * evictions
        evict = _all_sum(group, mx.array(dropped, dtype=mx.int32)).tolist()
    return _Plan(
        leave=[index for index in range(max_batch) if header[1 + index]],
        abort=[index for index in range(max_batch) if header[1 + max_batch + index]],
        joiner=joiner,
        evict=evict,
    )


def _sample(logits: mx.array, rows: list[_Row]) -> mx.array:
    from mlx_lm.sample_utils import make_sampler

    picked = []
    for index, row in enumerate(rows):
        line = logits[index : index + 1].astype(mx.float32)
        if row.close_guard is not None:
            line = row.close_guard(row.history, line)
        if row.temperature <= 0:
            picked.append(mx.argmax(line, axis=-1))
        else:
            logprobs = line - mx.logsumexp(line, axis=-1, keepdims=True)
            sampler = make_sampler(
                temp=row.temperature, top_p=row.top_p, top_k=row.top_k, min_p=row.min_p
            )
            picked.append(sampler(logprobs))
    return mx.concatenate(picked).astype(mx.int32)


def _extend_history(row: _Row, token: int) -> None:
    """A guarded row's history gains the token every rank just learned."""
    if row.close_guard is not None:
        row.history = mx.concatenate([row.history, mx.array([token], dtype=mx.int32)])


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


def _own_bytes(value, owned: dict[int, Any] | None = None, arrays=None):
    """``value`` with every MLX array it holds replaced, in place, by one that
    owns exactly its bytes — evaluated now, so the buffers it viewed are free.

    The single engine's lz.8 (goose Q-110) on this store's own layers: an
    array that is a VIEW keeps its whole source buffer alive while
    ``_held_bytes`` (shape x dtype) charges the view.  Measured here, CPU,
    the tiny qwen4_exp, 512-token prefill chunks: a full-attention layer's
    ``QSAIndexCache`` raw ring (``raw_keys[:, a:b, :]``) freed 458,752 bytes
    against 66,048 charged, so an entry held 1.35x what the plan budgeted.
    The walk is ``_held_bytes``'s own, so what is owned is what is charged;
    an array reached twice is copied once.
    """
    top = owned is None
    # id -> (the original, what replaces it); the original is kept so its id
    # cannot be reused by a copy made later in the same walk.
    owned = {} if owned is None else owned
    arrays = [] if arrays is None else arrays
    if id(value) in owned:
        return owned[id(value)][1]
    if isinstance(value, mx.array):
        result = mx.contiguous(value)
        arrays.append(result)
        owned[id(value)] = (value, result)
    else:
        owned[id(value)] = (value, value)
        if isinstance(value, list):
            value[:] = [_own_bytes(item, owned, arrays) for item in value]
            result = value
        elif isinstance(value, tuple):
            items = [_own_bytes(item, owned, arrays) for item in value]
            result = type(value)(*items) if hasattr(value, "_fields") else tuple(items)
            owned[id(value)] = (value, result)
        elif isinstance(value, dict):
            for key in list(value):
                value[key] = _own_bytes(value[key], owned, arrays)
            result = value
        elif hasattr(value, "__dict__"):
            fields = vars(value)
            for key in list(fields):
                fields[key] = _own_bytes(fields[key], owned, arrays)
            result = value
        else:
            result = value
    if top and arrays:
        mx.eval(arrays)
    return result


def _boundary_record(cache: list[Any]) -> list[Any]:
    """What a row's cache cannot give back later, taken at its snapshot boundary.

    goose Q-179: the prefix snapshot is no longer a copy of the whole cache
    beside the running row; the row's own cache BECOMES the entry when it
    leaves (``_adopted``).  What the row keeps writing over after the
    boundary is only what is small: a recurrent layer's state (GDN, the PLE
    n-gram history — a constant per sequence) and a QSA layer's raw-key ring
    (``compress_ratio`` keys) with its counters.  The attention KV and the
    QSA compressed keys are append-only, so the positions before the
    boundary are still in the row's cache when it leaves.
    """
    record = []
    for layer in cache:
        if isinstance(layer, CacheList):
            kv, qsa = layer.caches
            record.append(
                (
                    int(kv.offset),
                    _own_bytes(copy.deepcopy(qsa.raw_ring)),
                    qsa._offsets[0],
                    qsa._compressed_counts[0],
                )
            )
        else:
            record.append(_own_bytes(copy.deepcopy(layer)))
    return record


def _adopted(layer: Any, record: Any, boundary: int) -> Any:
    """One layer of a leaving row's cache, cut back to its snapshot boundary.

    ``layer`` is the row's own (a lone row's cache, or ``extract`` of its
    batch row); ``record`` what ``_boundary_record`` kept of it.  The result
    is exactly the layer a prefill of the first ``boundary`` tokens leaves:
    the same types ``make_cache`` builds, owning only its own bytes.
    """
    if not isinstance(record, tuple):
        return record
    offset, raw_ring, qsa_offset, compressed = record
    kv, qsa = layer.caches
    if offset != boundary or qsa_offset != boundary or int(kv.offset) < boundary:
        raise RuntimeError(
            f"prefix cache: a row leaves at KV offset {kv.offset} with its record at "
            f"{offset} / QSA {qsa_offset} for a snapshot at {boundary} — the caches "
            "diverged from the plan"
        )
    entry_kv = type(kv)()
    entry_kv.keys = mx.contiguous(kv.keys[..., :boundary, :])
    entry_kv.values = mx.contiguous(kv.values[..., :boundary, :])
    entry_kv.offset = boundary
    entry_qsa = type(qsa)(qsa.compress_ratio)
    entry_qsa.raw_ring = raw_ring
    if compressed:
        entry_qsa.compressed_keys = mx.contiguous(qsa.compressed_keys[:, :compressed])
    entry_qsa._offsets = [boundary]
    entry_qsa._compressed_counts = [compressed]
    entry_qsa._pending_left_padding = [0]
    return CacheList(entry_kv, entry_qsa)


class _PrefixStore:
    """One rank's prefix snapshots, by the id rank 0 assigned.  Decides nothing."""

    def __init__(self):
        self.entries: dict[int, list[Any]] = {}

    def take(self, entry_id: int, move: bool = False) -> list[Any]:
        """The entry's cache for a restoring row; ``move``: rank 0 evicts it
        in the same plan, so the row takes the entry itself (goose Q-179: one
        copy, never the entry and its restore side by side)."""
        if entry_id not in self.entries:
            raise RuntimeError(
                f"prefix cache: rank 0 restores entry {entry_id}, which this rank "
                f"does not hold (held: {sorted(self.entries)}) — the ranks diverged"
            )
        if move:
            return self.entries.pop(entry_id)
        # Generation mutates offsets and writes into the KV buffers: the
        # restored copy must not alias the stored entry (the single engine's
        # MemoryAwarePrefixCache.fetch copies for the same reason).
        return copy.deepcopy(self.entries[entry_id])

    def put(self, entry_id: int, cache: list[Any]) -> int:
        self.entries[entry_id] = _own_bytes(copy.deepcopy(cache))
        return _held_bytes(self.entries[entry_id])

    def adopt(self, entry_id: int, cache: list[Any]) -> int:
        """A leaving row's cache, already cut to its boundary, becomes the entry."""
        self.entries[entry_id] = _own_bytes(cache)
        return _held_bytes(self.entries[entry_id])

    def drop(self, entry_ids: list[int]) -> None:
        for entry_id in entry_ids:
            self.entries.pop(entry_id, None)


def _agree_rows(stage, local: int) -> int:
    """The plan header's row count, proven equal on every rank (its shape must pair)."""
    counts = [0] * stage.size
    counts[stage.rank] = local
    counts = _all_sum(stage.group, mx.array(counts, dtype=mx.int32)).tolist()
    if len(set(counts)) != 1:
        raise RuntimeError(
            f"pipeline plan header: the ranks derive {counts} rows from their "
            "KV budgets — the plans diverged"
        )
    return local


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
    """A row still prefilling, in its own cache, before it joins the batch."""

    row: _Row
    cache: list[Any]
    tokens: mx.array
    embeddings: Any
    rope: Any
    ranges: list[tuple[int, int]]

    @property
    def left(self) -> int:
        """Prompt tokens this row still has to prefill."""
        return sum(stop - start for start, stop in self.ranges)


def _shortest(prefilling: list[_Joining]) -> int | None:
    """The prefilling row whose chunk runs next: fewest tokens left, then the earliest admitted."""
    if not prefilling:
        return None
    return min(range(len(prefilling)), key=lambda i: (prefilling[i].left, i))


class _Engine:
    """One rank's rows: the running batch and the rows still prefilling.

    Every rank holds the same rows in the same order and applies the same
    plans, so every forward and every collective pairs across ranks; what a
    plan holds is rank 0's decision alone (``_Scheduler``).  Which prefilling
    row's chunk runs is not in any plan: ``target`` derives it from the rows
    and their remaining ranges, which every rank holds identically.
    """

    def __init__(
        self,
        stage,
        guard,
        prefill_step: int,
        store=None,
        on_stored=None,
        close_guard=None,
    ):
        self.stage = stage
        self.guard = guard
        self.prefill_step = prefill_step
        self.store = store
        # Rank 0: ``on_stored(row, bytes_per_rank)`` once every rank holds a snapshot.
        self.on_stored = on_stored
        # The sampling rank: the XML skeleton rules (``XmlCloseGuardSpec``) a
        # tool row decodes under; None elsewhere or for another wire.
        self.close_guard = close_guard
        self.rows: list[_Row] = []
        self.cache: list[Any] | None = None
        self.current: list[int] = []
        self.ropes: list[Any] = []
        self.rope = None
        self.prefilling: list[_Joining] = []

    @property
    def idle(self) -> bool:
        return not self.rows and not self.prefilling

    @property
    def target(self) -> int | None:
        """The index (in ``prefilling``) of the row whose chunk ``prefill`` runs next."""
        return _shortest(self.prefilling)

    @property
    def joining(self) -> _Joining | None:
        """The prefilling row whose chunk runs next (None: nothing is prefilling)."""
        target = self.target
        return None if target is None else self.prefilling[target]

    def apply(self, plan: _Plan) -> None:
        if plan.abort:
            aborted = set(plan.abort)
            self.prefilling = [
                joining
                for index, joining in enumerate(self.prefilling)
                if index not in aborted
            ]
        evict = frozenset(plan.evict)
        if plan.leave:
            leaving = set(plan.leave)
            self._regroup(
                [i for i in range(len(self.rows)) if i not in leaving], discard=evict
            )
        if plan.joiner is not None:
            self._start(plan.joiner, evict)
        if self.store is not None and plan.evict:
            # After the restore took its entry: an evicted entry may be the
            # one this joiner restores from.
            self.store.drop(plan.evict)

    def _start(self, row: _Row, evict: frozenset = frozenset()) -> None:
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
            cache = self.store.take(row.reuse_id, move=row.reuse_id in evict)
            start = row.cached
        else:
            cache, start = self.stage.make_cache(), 0
        if self.close_guard is not None and row.tools:
            from ..xml_tool_close_guard import XmlToolCloseGuard

            row.close_guard = XmlToolCloseGuard(self.close_guard)
            # The whole prompt: whether a call or a <think> is open is read
            # from the last opener, which may be in the generation prompt.
            row.history = mx.array(row.ids, dtype=mx.int32)
        joining = _Joining(
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
        self.prefilling.append(joining)

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
        for row, token in zip(self.rows, sampled):
            _extend_history(row, token)
        return sampled, control

    def prefill(self, words: list[int] | None) -> tuple[int | None, list[int]]:
        """The target row's next chunk; its first token when the prompt is done.

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
            # goose Q-179: only what the row will write over; its cache
            # becomes the entry when it leaves (``_regroup``).
            row.record = _boundary_record(joining.cache)
        if not last:
            return None, control
        _extend_history(row, sampled[0])
        self._regroup(list(range(len(self.rows))), joining, sampled[0])
        return sampled[0], control

    def _regroup(
        self,
        keep: list[int],
        joining: _Joining | None = None,
        first: int = 0,
        discard: frozenset = frozenset(),
    ) -> None:
        """Rebuild the running batch from the kept rows (+ the joined row), layer by layer.

        Each kept row's cache is extracted and the rows are merged again (a lone
        row keeps its own, unbatched cache).  One layer at a time, evaluated
        before the next, so the rebuild never holds more than one layer twice.

        A leaving row that took a prefix-cache record gives its cache, cut back
        to the boundary, to the store as that entry (goose Q-179) — unless rank
        0's plan evicts the entry in the same plan (``discard``).
        """
        count = len(self.rows)
        adopting = {
            i: []
            for i in range(count)
            if i not in keep
            and self.rows[i].store_id
            and self.rows[i].store_id not in discard
        }
        for i in adopting:
            if self.rows[i].record is None or self.store is None:
                raise RuntimeError(
                    f"prefix cache: row {i} leaves directed to store entry "
                    f"{self.rows[i].store_id} with no boundary record on this rank"
                )
        rebuilt = self.cache if self.cache is not None else list(joining.cache)
        for index in range(len(rebuilt)):
            parts: list[Any] = []
            if self.cache is not None:
                layer = self.cache[index]
                for i, entry in adopting.items():
                    row = self.rows[i]
                    own = layer if count == 1 else layer.extract(i)
                    entry.append(_adopted(own, row.record[index], row.store_at))
                    mx.eval(entry[-1].state)
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
        for i, entry in adopting.items():
            row = self.rows[i]
            measured = _agree_bytes(self.stage, self.store.adopt(row.store_id, entry))
            if self.on_stored is not None:
                self.on_stored(row, measured)
        for i in range(count):
            if i not in keep:
                self.rows[i].record = None
        self.rows = [self.rows[i] for i in keep]
        self.current = [self.current[i] for i in keep]
        self.ropes = [self.ropes[i] for i in keep]
        if joining is not None:
            self.rows.append(joining.row)
            self.current.append(first)
            self.ropes.append(joining.rope)
            self.prefilling = [row for row in self.prefilling if row is not joining]
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
    while engine.prefilling:
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
    # Rank 0's arrival order (set when the scheduler takes the job off the queue).
    seq: int = 0
    # Prompt tokens of LATER arrivals prefilled ahead of this one: the delay
    # the shortest-first order cost it, in the pipeline's own prefill time
    # (``_Scheduler._slack``).  Never the tokens of earlier arrivals.
    waited: int = 0
    # Prompt tokens it had to prefill when admitted (after its prefix restore).
    prefill: int = 0
    # Rank 0's HTTP side, a streamed chat answer only: what its client has and
    # has not been sent (``pipeline_stream.StreamWatch``, goose Q-146) — the
    # ``stream`` block of its /v1/status row.
    stream: Any = field(default=None, compare=False, repr=False)

    def push(self, item) -> None:
        self.loop.call_soon_threadsafe(self.events.put_nowait, item)


class _KvBudget:
    """Admission by what the rows will hold on EVERY rank, priced in the planner's own terms.

    The split was planned for ``slots`` full-context sequences (the planner's
    ``batch``): each rank's budget for runtime state is its planned KV/recurrent
    state plus prefill workspace at that shape.  What rows hold is priced per
    rank with the planner's one-sequence formulas (``_stage_plan`` at batch 1,
    rounded to the caches' allocation steps by ``layer_state_bytes``) over the
    shapes the engine really builds:

    * the running batch is ONE merged cache, every row left-padded to the
      longest (mlx-lm's ``BatchKVCache.merge``), so it holds rows x its
      longest row's tokens — the padding is memory, not bookkeeping;
    * a prefilling row holds its own cache, its prompt at most, until its
      last chunk merges it into the batch;
    * a prefill chunk runs alone (one row over its own cache) and a decode
      step runs the batch; they never overlap, so the transient is the
      larger of the two.  A join or a departure also rebuilds a batch of two
      rows or more one layer at a time (``_Engine._regroup``: the rows'
      copies of one layer beside the old layer) — a transient the planner
      does not budget, so a batch within the planned ``slots`` is priced as
      the plan was (its promise of ``slots`` full-context rows holds; the
      pipeline has always run that way), and a batch of MORE rows than the
      plan was made for — the room Q-160 opens — is admitted only with room
      for that copy too.

    Widths are rounded as the planner rounds them (``layer_state_bytes``: up
    to the caches' allocation step), the rounding the plan's own budget was
    made with.  A merged buffer can run up to one step past that (a merge
    leaves it exact, the next token adds a whole step) — under one step per
    row, the same gap the planner and the rows x longest reservation had.

    A running row grows one token a step and has left by its horizon (prompt
    + max_tokens + 1), so what the rows hold before anything else joins
    (``held``) peaks at some row's LAST step — between two departures the
    width only grows — and is the largest of rows alive x their width there.
    Before goose Q-160 every row was priced at the longest horizon for its
    whole life (rows x longest) and ``--max-batch`` capped the rows at the
    planned slots: two chats that reserve the whole context (goose sends no
    max_tokens) filled both, and a 600-token helper call waited for a
    multi-minute decode to end though it would have left long before the
    long rows grew into the room it used.
    """

    def __init__(self, plan, prefill_step: int):
        self.plan = plan
        self.prefill_step = prefill_step
        self.args = plan.args
        self.budgets = [
            stage.state_bytes + stage.workspace_bytes for stage in plan.stages
        ]
        self.slots = plan.batch
        # (width, chunk tokens) -> each rank's (state, workspace, largest
        # layer's state) for ONE row.
        # Emptied at every plan (``_Scheduler.applied``), so it holds only the
        # widths priced since the rows last changed.
        self.priced: dict[tuple[int, int], list[tuple[int, int, int]]] = {}

    def _price(self, width: int, tokens: int) -> list[tuple[int, int, int]]:
        """Each rank's (state, workspace, largest layer's state) for one row
        ``width`` tokens long running ``tokens`` at once."""
        key = (width, tokens)
        if key not in self.priced:
            size = len(self.plan.stages)
            planned = [
                pipe._stage_plan(
                    self.args,
                    self.plan.checkpoint,
                    stage.node,
                    stage.rank,
                    size,
                    stage.start,
                    stage.end,
                    width,
                    1,
                    tokens,
                )
                for stage in self.plan.stages
            ]
            act = self.plan.checkpoint.activation_bytes
            self.priced[key] = [
                (
                    planned_stage.state_bytes,
                    planned_stage.workspace_bytes,
                    max(
                        pipe.layer_state_bytes(self.args, index, width, 1, act)
                        for index in range(stage.start, stage.end)
                    ),
                )
                for stage, planned_stage in zip(self.plan.stages, planned)
            ]
        return self.priced[key]

    def held(
        self, batch: list[tuple[int, int]], prefilling: list[tuple[int, int]]
    ) -> list[int]:
        """The most each rank holds until a row joins or is admitted.

        ``batch``: the running rows as (tokens in their cache, horizon);
        ``prefilling``: the rows prefilling in their own caches, as (prompt
        tokens, horizon) — priced at their whole prompt for as long as they
        wait, since the batch decides when they may join.
        """
        size = len(self.budgets)
        solo = [0] * size
        chunk = [0] * size
        for length, _ in prefilling:
            for rank, (state, workspace, _) in enumerate(
                self._price(length, min(self.prefill_step, length))
            ):
                solo[rank] += state
                chunk[rank] = max(chunk[rank], workspace)
        peak = [solo[rank] + chunk[rank] for rank in range(size)]
        left = [max(horizon - length, 0) for length, horizon in batch]
        for last in set(left):
            alive = [
                length + last
                for (length, _), remaining in zip(batch, left)
                if remaining >= last
            ]
            rows = len(alive)
            for rank, (state, workspace, layer) in enumerate(
                self._price(max(alive), 1)
            ):
                merge = rows * layer if rows > self.slots else 0
                transient = max(rows * workspace, merge, chunk[rank])
                peak[rank] = max(peak[rank], rows * state + solo[rank] + transient)
        return peak

    def admission(
        self, batch: list[tuple[int, int]], prefilling: list[tuple[int, int]]
    ) -> list[int]:
        """What each rank must have room for to admit the last of ``prefilling``.

        The largest of: what the rows then hold (``held``); what they would
        hold had every prefilling row joined the batch now, so an admitted
        row is never one whose own join must wait for a departure (it would
        be the shortest prefill and stall every other one behind it); and,
        for each prefilling row, what it holds grown to its horizon beside
        the others' prompts, so a join is never held for good once the
        batch has drained.
        """
        need = self.held(batch, prefilling)
        needs = [need, self.held([*batch, *prefilling], [])]
        for index, row in enumerate(prefilling):
            needs.append(self.held([row], prefilling[:index] + prefilling[index + 1 :]))
        return [max(values) for values in zip(*needs)]

    def fits(self, need: list[int]) -> bool:
        return all(n <= budget for n, budget in zip(need, self.budgets))

    def rows(self) -> int:
        """The most rows the budgets could hold, each one token long (the wire's row count).

        Every row holds at least that, so admission by ``fits`` never
        reaches it; it only sizes the plan header's per-row flags.
        """
        least = self._price(1, 1)
        return min(
            budget // state for budget, (state, _, _) in zip(self.budgets, least)
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

    def record_bytes(self) -> list[int]:
        """Each rank's bound on one row's boundary record (``_boundary_record``):
        every recurrent layer's state and every QSA layer's raw-key ring — what
        a directed snapshot holds beside its row until the row leaves."""
        act = self.plan.checkpoint.activation_bytes
        ring = int(self.args.indexer_compress_ratio) * int(self.args.indexer_head_dim)
        return [
            sum(
                pipe.layer_state_bytes(self.args, index, 1, 1, act)
                if self.args.layer_types[index] == "linear_attention"
                else ring * act
                for index in range(stage.start, stage.end)
            )
            for stage in self.plan.stages
        ]


@dataclass
class _Entry:
    key: tuple[int, ...]
    bytes: list[int]


class _PrefixIndex:
    """Rank 0's prefix-cache decisions: what to restore, snapshot and evict.

    Bounded by bytes alone, per rank: at every admission the entries plus the
    snapshots in flight must fit each rank's planned KV budget minus what the
    rows need then (``_KvBudget.admission``), and before a prefilling row's
    last chunk merges it into the batch they must fit beside what the batch
    will hold (``yield_to``) — the cache holds only the budget live requests
    leave idle, and yields it to them first.  LRU order, refreshed on every
    hit.  No entry count bound: the count can never evict before the bytes do.

    One copy (goose Q-179).  A directed snapshot no longer copies the row's
    cache beside it: the row keeps a small boundary record (charged at
    ``record_bytes`` while it runs) and its own cache becomes the entry when
    it leaves (charged at ``entry_bytes`` from the plan it leaves in —
    ``leaving`` — and measured once every rank holds it).  A restore whose
    entry is evicted in the same plan moves the entry into the row.  E2E #5b
    (Flash, two rows each reserving the whole 262k context): the copy beside
    the batch stopped fitting rank 1's 0.91 GB of room at 53,409 tokens
    (entry 0.98 GB), every later call re-read 60-62k tokens, ~4 min each.
    """

    def __init__(self, kv: _KvBudget):
        self.kv = kv
        self.entries: OrderedDict[int, _Entry] = OrderedDict()
        # Snapshots directed but not yet stored: charged at their record's
        # bound while their rows run, at the entry's bound from the plan the
        # row leaves in (``leaving``), until ``stored`` or ``forget``.
        self.pending: dict[int, list[int]] = {}
        # The pending snapshots whose rows leave in the plan being made.
        self.adopting: set[int] = set()
        # Bumped whenever ``lookup`` could answer differently.
        self.version = 0
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

    def _held(self, pending_only: bool = False) -> list[int]:
        """Each rank's bytes: the pending snapshots' bounds, and the entries unless ``pending_only``."""
        entries = [] if pending_only else list(self.entries.values())
        return [
            sum(entry.bytes[rank] for entry in entries)
            + sum(bound[rank] for bound in self.pending.values())
            for rank in range(len(self.kv.budgets))
        ]

    def room(self, need: list[int]) -> str:
        """Beside rows that need ``need``: "fits", "evict" (dropping entries makes it fit) or "full"."""
        with self.lock:
            if self.kv.fits([n + h for n, h in zip(need, self._held())]):
                return "fits"
            if self.kv.fits([n + h for n, h in zip(need, self._held(True))]):
                return "evict"
            return "full"

    def yield_to(self, need: list[int]) -> list[int]:
        """Evict entries, oldest first, until they fit beside rows that need ``need``."""
        with self.lock:
            evict = []
            while self.entries and not self.kv.fits(
                [n + h for n, h in zip(need, self._held())]
            ):
                entry_id, _ = self.entries.popitem(last=False)
                evict.append(entry_id)
                self.evicted += 1
            if evict:
                self.version += 1
            return evict

    def admit(self, row: _Row, need: list[int]) -> list[int]:
        """Set the joining row's directive; return the entries to evict.

        ``need`` is what each rank must keep for the rows once the row is
        admitted, its own included (``_KvBudget.admission``).
        """
        with self.lock:
            room = [budget - n for budget, n in zip(self.kv.budgets, need)]
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
                    new = self.kv.record_bytes()
                    row.store_id, row.store_at = self.next_id, row.boundary
                    self.next_id += 1
            evict = []

            def over(extra: list[int]) -> bool:
                return any(h + e > r for h, e, r in zip(self._held(), extra, room))

            while self.entries and over(new):
                entry_id, _ = self.entries.popitem(last=False)
                evict.append(entry_id)
                self.evicted += 1
            if any(new) and over(new):
                # Even an empty cache cannot hold this row's record beside the batch.
                self._skip("no_room_beside_the_batch")
                row.store_id = row.store_at = 0
            elif row.store_id:
                self.pending[row.store_id] = new
            if evict:
                self.version += 1
            return evict

    def leaving(self, rows: list[_Row]) -> None:
        """Rows that leave in the plan being made: each one's snapshot is its
        cache, cut back to the boundary, from this plan on (goose Q-179)."""
        with self.lock:
            for row in rows:
                if row.store_id in self.pending:
                    self.pending[row.store_id] = self.kv.entry_bytes(row.store_at)
                    self.adopting.add(row.store_id)

    def settle(self, need: list[int]) -> list[int]:
        """Once the plan's rows need ``need``: entries go, oldest first, until
        the cache fits beside them; if the snapshots taken in this plan still
        do not, they are not kept (their ids ride the plan's evictions, so no
        rank adopts them).  Returns the ids every rank drops."""
        with self.lock:
            evict = []
            while self.entries and not self.kv.fits(
                [n + h for n, h in zip(need, self._held())]
            ):
                entry_id, _ = self.entries.popitem(last=False)
                evict.append(entry_id)
                self.evicted += 1
            for entry_id in sorted(self.adopting):
                if self.kv.fits([n + h for n, h in zip(need, self._held())]):
                    break
                self.pending.pop(entry_id)
                evict.append(entry_id)
                self._skip("no_room_after_the_row")
            self.adopting.clear()
            if evict:
                self.version += 1
            return evict

    def stored(self, entry_id: int, key: tuple[int, ...], measured: list[int]) -> None:
        with self.lock:
            self.pending.pop(entry_id, None)
            self.entries[entry_id] = _Entry(key, measured)
            self.stores += 1
            self.version += 1

    def forget(self, entry_id: int) -> None:
        """A directed snapshot that will never be taken (its row was aborted)."""
        with self.lock:
            self.pending.pop(entry_id, None)
            self.adopting.discard(entry_id)

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
    # The rows the plan header can name (``_KvBudget.rows`` on every rank):
    # running and prefilling rows together.  Memory admits fewer.
    max_batch: int
    # Further names the served model answers to (``--served-model-alias``),
    # listed after ``served`` on /v1/models.
    aliases: tuple[str, ...] = ()
    kv: _KvBudget | None = None
    prefix: _PrefixIndex | None = None
    jobs: queue.Queue = field(default_factory=queue.Queue)
    # Rank 0's queued requests, taken off ``jobs`` in arrival order and not
    # yet admitted.  Replaced whole, never mutated, so the HTTP thread can
    # read it without a lock.
    waiting: list = field(default_factory=list)
    active: list = field(default_factory=list)
    reserved: list = field(default_factory=list)
    steps: int = 0
    admission_open: bool = True
    admission_reason: str | None = None
    shutting_down: bool = False
    eos_ids: frozenset = frozenset()
    # Rank 0's sampling layers under each request's own fields (goose Q-159).
    sampling: _SamplingDefaults = field(default_factory=_SamplingDefaults)
    # The logits' width (the checkpoint's vocab_size): a top_k at or past it
    # raises in the sampling rank's sampler and ends every rank (goose Q-164's
    # class), so rank 0 refuses it first.  0 only where no model is loaded (an
    # app built for a test), which checks nothing against it.
    vocab: int = 0
    # serve's ``emit`` (goose's rank log); None prints ``PIPELINE_<tag>`` lines.
    emit: Any = field(default=None, compare=False, repr=False)
    # The last answer the engine ended itself (goose Q-161): the request, why
    # (``tool_call_repeated`` / ``text_cycle``) and the words.  Its row leaves
    # /v1/status with it; this stays.
    last_engine_stop: Any = None


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


def _unhonoured(body: dict) -> str | None:
    """Why this request asks for something the split cannot do, or None.

    goose Q-177 (the tensor split, 2026-09-27): a ``top_logprobs`` above 11
    dropped the connection with no reply.  Here such fields were ignored — an
    answer that looked like what was asked and was not — and a malformed
    ``stop`` raised inside the stream, after its headers, which drops the
    connection the same way.  Each is now a named 400 before anything runs.
    """
    n = body.get("n")
    if n is not None and n != 1:
        return f"n={n!r}: the pipeline split writes one choice per request"
    if body.get("logprobs") or body.get("top_logprobs") not in (None, 0):
        return (
            "logprobs / top_logprobs: the pipeline split returns no log-probabilities "
            "(its sampling rank shares only the sampled token)"
        )
    if body.get("logit_bias"):
        return "logit_bias: the pipeline split's sampler applies no per-token bias"
    if body.get("seed") is not None:
        return (
            "seed: the pipeline split samples from one process-wide random state, so "
            "a per-request seed cannot be honoured"
        )
    response_format = body.get("response_format")
    if isinstance(response_format, dict) and response_format.get("type") not in (
        None,
        "text",
    ):
        return (
            f"response_format {response_format.get('type')!r}: the pipeline split does "
            "not constrain its output to a format"
        )
    stop = body.get("stop")
    if stop is not None and not (
        (isinstance(stop, str) and stop)
        or (
            isinstance(stop, list)
            and all(isinstance(item, str) and item for item in stop)
        )
    ):
        return f"stop must be a non-empty string or a list of them, not {stop!r}"
    for key in ("max_completion_tokens", "max_tokens"):
        value = body.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 1
        ):
            return f"{key} must be a whole number of at least 1, not {value!r}"
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return "messages must be a non-empty list"
    if not all(isinstance(message, dict) for message in messages):
        return "every message must be an object"
    tools = body.get("tools")
    if tools is not None and not (
        isinstance(tools, list)
        and all(
            isinstance(tool, dict)
            and isinstance(tool.get("function"), dict)
            and isinstance(tool["function"].get("name"), str)
            for tool in tools
        )
    ):
        return "tools must be a list of {type: function, function: {name, parameters}}"
    return None


def _sampling_refusal(sampling: dict, vocab: int) -> str | None:
    """Why the resolved sampling would raise in the sampling rank, or None.

    mlx_lm's sampler raises on a top_k that is not below the logits' width and
    on a min_p outside [0, 1] (measured, mlx_lm 0.31.3), inside the tick loop
    every rank runs — so one request would end the pair (goose Q-164's class).
    The layer that supplied the value is named with it.
    """
    top_k, top_p, min_p = (sampling[key] for key in ("top_k", "top_p", "min_p"))
    temperature = sampling["temperature"]
    if top_k["value"] is not None and (
        top_k["value"] < 0 or (vocab and top_k["value"] >= vocab)
    ):
        return (
            f"top_k {top_k['value']} ({top_k['from']}) must be 0 (off) or below the "
            f"vocabulary's {vocab} tokens"
        )
    for key, entry in (("top_p", top_p), ("min_p", min_p)):
        if entry["value"] is not None and not 0.0 <= entry["value"] <= 1.0:
            return f"{key} {entry['value']} ({entry['from']}) must be within [0, 1]"
    if temperature["value"] is not None and temperature["value"] < 0:
        return (
            f"temperature {temperature['value']} ({temperature['from']}) must not be "
            "negative"
        )
    return None


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
    say = state.emit or (
        lambda tag, payload: print(f"PIPELINE_{tag} {json.dumps(payload)}", flush=True)
    )
    # goose Q-178 (pipeline_stream.py): the markers that move an answer between
    # reasoning, text and a call, and the parser a streamed call is read by.
    markers = Markers.of(tokenizer)
    call_reader = None
    if tool_parser == "qwen3_coder_xml":
        from ..tool_parsers import ToolParserManager

        call_reader = ToolParserManager.get_tool_parser(tool_parser)(tokenizer)

    def record_stop(stop: dict) -> None:
        state.last_engine_stop = stop

    def unstreamed(body: dict, tools) -> str | None:
        """Why the relay leaves this request's calls to the post-processor, or None."""
        if not tools:
            return "the request declares no tools"
        if call_reader is None:
            return (
                f"the tool parser {tool_parser} has no streamer on the pipeline split"
            )
        if markers.call_start is None or markers.call_end is None:
            return "the checkpoint's <tool_call> / </tool_call> are not single tokens"
        if body.get("parallel_tool_calls") is False:
            return (
                "parallel_tool_calls is false: the post-processor keeps the first call"
            )
        if not call_reader._declared_tool_names(body):
            return "tool_choice leaves no tool executable"
        return None

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
                    # The single engine's declaration (REQUEST_EXTENSIONS: the
                    # tail and its _on_tool form): the transient-tail field
                    # moves the prefix snapshot, so it is declared only while
                    # there is a prefix cache to move it in.
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
            "num_waiting": state.jobs.qsize() + len(state.waiting),
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
            "sampling_defaults": state.sampling.report(),
            # The last answer the engine ended itself (goose Q-161): its row
            # has left the table with it.
            "last_engine_stop": state.last_engine_stop,
            "status": "ok",
        }

    @app.get("/goose/progress")
    async def progress():
        return {
            "steps": state.steps,
            "inflight": len(state.active) + state.jobs.qsize() + len(state.waiting),
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
        if not isinstance(body, dict):
            return error(
                400, "the request body must be a JSON object", "invalid_request_error"
            )
        refusal = _unhonoured(body)
        if refusal is not None:
            return error(400, refusal, "invalid_request_error")
        model = body.get("model")
        if model is not None and model not in _served_names(state):
            return error(404, _not_served(state, model), "model_not_found")
        transient_tail = body.get("rapid_mlx_transient_tail")
        if transient_tail is not None and not isinstance(transient_tail, str):
            return error(
                400,
                "rapid_mlx_transient_tail must be a string: the exact trailing text "
                "of the last user or tool message",
                "invalid_request_error",
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
        try:
            sampling = state.sampling.resolve(body)
        except ValueError as refusal:
            return error(400, str(refusal), "invalid_request_error")
        refusal = _sampling_refusal(sampling, state.vocab)
        if refusal is not None:
            return error(400, refusal, "invalid_request_error")
        kwargs = dict(body.get("chat_template_kwargs") or {})
        enable_thinking = kwargs.pop("enable_thinking", body.get("enable_thinking"))
        try:
            prompt = apply_chat_template(
                tokenizer,
                messages,
                tools=convert_tools_for_template(tools) if tools else None,
                enable_thinking=enable_thinking,
                model_name=state.served,
                chat_template_kwargs=kwargs or None,
            )
        except Exception as refusal:  # noqa: BLE001 - the template reads only the request
            return error(
                400,
                f"the chat template cannot render this request: "
                f"{type(refusal).__name__}: {refusal}",
                "invalid_request_error",
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
        if not ids:
            # An empty row would join no plan on any rank and wait forever.
            return error(400, "the rendered prompt is empty", "invalid_request_error")
        boundary = 0
        if state.prefix is not None and images is None:
            stable = BatchedEngine._stable_messages_before_transient_tail(
                messages, None, transient_tail
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
        values = {key: entry["value"] for key, entry in sampling.items()}
        stops = body.get("stop") or []
        stops = [stops] if isinstance(stops, str) else list(stops)
        loop = asyncio.get_running_loop()
        job = _Job(
            _Row(
                ids,
                max_tokens,
                values["temperature"],
                values["top_p"],
                images,
                boundary=boundary,
                tools=bool(tools),
                top_k=values["top_k"] or 0,
                min_p=values["min_p"] or 0.0,
                sampling=sampling,
            ),
            loop,
            asyncio.Queue(),
        )
        created = int(time.time())
        processor = StreamingPostProcessor(
            cfg,
            tools_requested=bool(tools),
            enable_thinking=enable_thinking,
            request=body,
        )
        processor.reset()
        primed = processor.reasoning_parser is not None and _should_start_in_thinking(
            getattr(tokenizer, "chat_template", "") or "",
            enable_thinking,
            tools_requested=bool(tools),
        )
        if primed:
            # The template opens <think> in the generation prompt, so the
            # model never emits the opener.  deepseek_r1's streaming path
            # flips a tagless stream to content after 64 characters unless it
            # is told the prompt primed thinking (its own hook, set by
            # configure_request on the distill variant).  Measured on Flash:
            # without it every thought past 64 chars streamed as content.
            processor.reasoning_parser._prompt_primed_thinking = True
        relay = None
        if body.get("stream"):
            # goose Q-178 (pipeline_stream.py): the relay streams the calls the
            # post-processor would hold to the end, reports what the client
            # has not been sent, and ends an answer that repeats itself.
            job.stream = StreamWatch(say, job.id)
            relay = StreamRelay(
                job.stream,
                markers,
                reasoning=primed,
                parser=call_reader,
                request=body,
                owns=unstreamed(body, tools),
                cycles=bool(tools),
                say=say,
                record_stop=record_stop,
            )
        # Queued only once everything the answer's reader needs exists: a job
        # nobody reads would run to its horizon.
        state.jobs.put(job)
        fed = ""

        def events_of(events) -> list[tuple[dict, str | None]]:
            items = []
            for event in events:
                delta = delta_of(event)
                if relay is not None and delta.get("tool_calls"):
                    delta["tool_calls"] = relay.remap(delta["tool_calls"])
                items.append((delta, event.finish_reason))
            return items

        def post(piece: str, completion: int) -> list[tuple[dict, str | None]]:
            """One piece through the post-processor: (delta, finish_reason) per event."""
            nonlocal fed
            fed += piece
            return events_of(
                processor.process_chunk(
                    GenerationOutput(
                        text=fed,
                        new_text=piece,
                        prompt_tokens=len(ids),
                        completion_tokens=completion,
                        finished=False,
                        finish_reason=None,
                    )
                )
            )

        def through(actions, completion: int) -> list[tuple[dict, str | None]]:
            """The relay's actions, in order: text for the post-processor, calls it sends."""
            items = []
            for kind, value in actions:
                if kind == "call":
                    items.append(({"tool_calls": [value]}, None))
                elif value:
                    items.extend(post(value, completion))
            return items

        def pieces(token, piece: str, completion: int):
            if relay is None:
                return post(piece, completion) if piece else []
            return through(relay.take(token, piece), completion)

        async def generate():
            """Yield ([(delta, finish_reason)], finish_reason, completion_tokens) as tokens arrive."""
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
                hit = False
                if stops:
                    candidate = text + piece
                    at = min(
                        (
                            candidate.find(s, max(0, len(text) - len(s)))
                            for s in stops
                            if s in candidate
                        ),
                        default=-1,
                    )
                    if at >= 0:
                        piece = candidate[:at][len(text) :]
                        hit = True
                        finish = "stop"
                        job.finished = True
                text += piece
                # A stop sequence cut this token's text: its marker never ran.
                items = pieces(None if hit else token, piece, completion)
                if items:
                    yield items, None, completion
                if relay is not None and relay.stop is not None:
                    # goose Q-161: the engine ended the answer; every rank's
                    # row leaves at the next plan.
                    finish = "stop"
                    job.finished = True
                    break
                if hit:
                    break
            detok.finalize()
            tail = detok.last_segment
            if tail and finish != "stop":
                text += tail
                items = pieces(None, tail, completion)
                if items:
                    yield items, None, completion
            if relay is not None:
                items = through(relay.finish(), completion)
                if items:
                    yield items, None, completion
                if relay.calls_sent and finish == "stop":
                    finish = "tool_calls"
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
            yield events_of(terminal), finish, completion

        def delta_of(event) -> dict:
            delta: dict[str, Any] = {}
            if event.content:
                delta["content"] = event.content
            if event.reasoning:
                delta["reasoning_content"] = event.reasoning
            if event.tool_calls:
                delta["tool_calls"] = event.tool_calls
            return delta

        def failure_text(failure: BaseException) -> str:
            if isinstance(failure, RuntimeError):
                return str(failure)
            return f"{type(failure).__name__}: {failure}"

        if body.get("stream"):
            watch = job.stream

            async def sse():
                def chunk(delta, finish=None, usage=None, refusal=None):
                    watch.sent(delta)
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
                    async for items, final, completion in generate():
                        for delta, reason in items:
                            if reason:
                                finish = reason
                            if delta:
                                yield chunk(delta)
                        watch.settle()
                        if final is not None and finish == "stop":
                            finish = final
                except Exception as failure:  # noqa: BLE001 - said on the stream (goose Q-177: never a dropped connection)
                    if not isinstance(failure, RuntimeError):
                        say(
                            "RANK_STREAM_FAILED",
                            {
                                "request_id": job.id,
                                "error": failure_text(failure),
                                "generated_chars": watch.generated_chars,
                                "tail": watch.tail,
                            },
                        )
                    error_payload = {
                        "error": {
                            "message": failure_text(failure),
                            "type": "pipeline_error",
                        }
                    }
                    yield f"data: {json.dumps(error_payload)}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                finally:
                    job.cancelled = True
                    watch.end()
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
            async for items, final, completion in generate():
                for delta, reason in items:
                    if reason:
                        finish = reason
                    if delta.get("content"):
                        content.append(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning.append(delta["reasoning_content"])
                    if delta.get("tool_calls"):
                        calls.extend(delta["tool_calls"])
                if final is not None and finish == "stop":
                    finish = final
        except Exception as failure:  # noqa: BLE001 - a named 500 (goose Q-177)
            return error(500, failure_text(failure), "pipeline_error")
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
    """Rank 0's decisions: which rows leave, which queued request is admitted, when chunks run.

    Admission is by memory, in ``_head``'s order (Q-145): the queued request
    first in that order is admitted when every rank has room for what the
    rows would then need (``_KvBudget.admission``: its own prompt and
    max_tokens, beside what the running rows will still grow into — goose
    Q-160) and — while other rows are prefilling — the request has fewer
    prompt tokens left than every one of them, so it is exactly the row
    ``_shortest`` runs next on every rank.  ``max_batch`` (the rows the plan
    header can name, derived from the same budgets) never binds before the
    memory does.  The order only lets a request go ahead of an earlier one
    while it takes at most half that one's ``_slack``.  A request that cannot
    be admitted waits at the head of the order and nothing behind it jumps
    it.  Chunks are spaced by ``_PREFILL_SHARE`` of the measured pipeline
    time, and a prefilling row's last chunk — the one that merges it into
    the batch — runs only once the batch it makes fits (``_join_room``).
    """

    def __init__(self, state: _State, engine: _Engine):
        self.state = state
        self.engine = engine
        self.running: list[_Job] = []
        # The engine's prefilling rows' jobs, in the engine's order.
        self.prefilling: list[_Job] = []
        # The request the plan being broadcast admits (prefilling once applied).
        self.admitted: _Job | None = None
        self.stopping = False
        # The target's last chunk is held this step: its join would not fit.
        self.holding = False
        # Seconds of pipeline time the prefills may still spend before the
        # running rows' decode has had its share (both measured here).
        self.credit = 0.0
        self.arrivals = 0
        # A queued job's prompt tokens left, by job id: (prefix-cache version, tokens).
        self.lefts: dict[str, tuple[int, int]] = {}
        if state.prefix is not None:
            engine.on_stored = self._stored

    def _stored(self, row: _Row, measured: list[int]) -> None:
        self.state.prefix.stored(row.store_id, tuple(row.ids[: row.store_at]), measured)

    def _collect(self, block: bool) -> None:
        """Take every queued job off ``jobs``; ``block``: until one live job waits."""
        state = self.state
        waiting = [job for job in state.waiting if not job.cancelled]
        for job in state.waiting:
            if job.cancelled:
                self.lefts.pop(job.id, None)
        while not self.stopping:
            try:
                job = (
                    state.jobs.get()
                    if block and not waiting
                    else state.jobs.get_nowait()
                )
            except queue.Empty:
                break
            if job is None:
                self.stopping = True
            elif not job.cancelled:
                self.arrivals += 1
                job.seq = self.arrivals
                waiting.append(job)
        if len(waiting) != len(state.waiting):
            state.waiting = waiting

    def _queued_left(self, job: _Job) -> int:
        """Prompt tokens a queued job would prefill if admitted now (after its prefix restore)."""
        prefix = self.state.prefix
        if prefix is None or job.row.images is not None:
            return len(job.row.ids)
        memo = self.lefts.get(job.id)
        if memo is None or memo[0] != prefix.version:
            with prefix.lock:
                _, cached = prefix.lookup(job.row.ids)
            memo = (prefix.version, len(job.row.ids) - cached)
            self.lefts[job.id] = memo
        return memo[1]

    @staticmethod
    def _slack(
        job: _Job, left: int, prefilling: list[tuple[_Job, int]], queued: bool
    ) -> int:
        """Prompt tokens later arrivals may still prefill ahead of ``job``.

        Its own prefill (``left`` while it is queued; while it prefills, what
        it had to prefill when admitted), less what later arrivals already
        prefilled ahead of it (``waited``) and what the ones already admitted
        will still prefill ahead of it: every one of them while ``job`` is
        queued, the ones with fewer tokens left than its ``left`` while it
        is prefilling itself.  Spent in full, the jumpers doubled its prefill.
        """
        ahead = sum(
            other_left
            for other, other_left in prefilling
            if other.seq > job.seq and (queued or other_left < left)
        )
        own = left if queued else job.prefill
        return own - job.waited - ahead

    def _head(self, prefilling: list[tuple[_Job, int]] | None = None) -> _Job | None:
        """The queued request admitted next: fewest tokens left, then the oldest.

        A request goes ahead of an earlier arrival (queued, or prefilling in
        ``prefilling`` as (job, tokens left)) only when it takes at most HALF
        of that one's ``_slack``: a jump never leaves less slack than it took,
        so the next short request still finds room — a later long prompt one
        percent shorter never spends a long one's whole slack and blocks
        every canary behind it (goose Q-145, the 26d replay) — and a stream
        of jumpers stops before the slack runs out, so none stretches a
        request past twice its own prefill.  The oldest queued request is
        never held by a queued one, so while nothing prefills a head exists.
        """
        waiting = [job for job in self.state.waiting if not job.cancelled]
        if not waiting:
            return None
        if prefilling is None:
            prefilling = self._prefilling_lefts()
        lefts = {job.id: self._queued_left(job) for job in waiting}
        earlier = [
            (job, self._slack(job, lefts[job.id], prefilling, queued=True))
            for job in waiting
        ] + [
            (job, self._slack(job, left, prefilling, queued=False))
            for job, left in prefilling
        ]
        for job in sorted(waiting, key=lambda job: (lefts[job.id], job.seq)):
            left = lefts[job.id]
            if all(
                2 * left <= slack for other, slack in earlier if other.seq < job.seq
            ):
                return job
        return None

    def _take(self, job: _Job) -> None:
        self.state.waiting = [other for other in self.state.waiting if other is not job]
        self.lefts.pop(job.id, None)

    @staticmethod
    def _decoding(jobs: list[_Job]) -> list[tuple[int, int]]:
        """Running rows as ``_KvBudget`` prices them: (cache tokens after the step in flight, horizon)."""
        return [
            (len(job.row.ids) + job.produced, _reservation_length(job.row))
            for job in jobs
        ]

    @staticmethod
    def _prompts(jobs: list[_Job]) -> list[tuple[int, int]]:
        """Prefilling rows as ``_KvBudget`` prices them: (prompt tokens, horizon)."""
        return [(len(job.row.ids), _reservation_length(job.row)) for job in jobs]

    def _admission(
        self, job: _Job, running: list[_Job], prefilling: list[_Job]
    ) -> list[int]:
        return self.state.kv.admission(
            self._decoding(running), self._prompts([*prefilling, job])
        )

    def _fits(self, job: _Job, running: list[_Job], prefilling: list[_Job]) -> bool:
        if len(running) + len(prefilling) + 1 > self.state.max_batch:
            return False
        kv = self.state.kv
        return kv is None or kv.fits(self._admission(job, running, prefilling))

    def _join_room(self, running: list[_Job], pairs: list[tuple[_Job, _Joining]]):
        """Whether the next chunk may run: "fits", "evict" or "full", with the join's need.

        ``pairs`` are the prefilling rows (job, the engine's row).  Only the
        target's LAST chunk changes what the rows hold — it moves the row
        from its own cache into the padded batch — so any other chunk fits.
        """
        kv = self.state.kv
        target = _shortest([joining for _, joining in pairs])
        if kv is None or target is None or len(pairs[target][1].ranges) != 1:
            return "fits", None
        job = pairs[target][0]
        others = [other for other, _ in pairs if other is not job]
        need = kv.held(self._decoding([*running, job]), self._prompts(others))
        if not kv.fits(need):
            return "full", need
        if self.state.prefix is None:
            return "fits", need
        return self.state.prefix.room(need), need

    def _admissible(
        self, running: list[_Job], prefilling: list[tuple[_Job, int]]
    ) -> _Job | None:
        """The head, if it is admitted beside ``running`` and ``prefilling`` (job, tokens left)."""
        head = self._head(prefilling)
        if head is None or not self._fits(
            head, running, [job for job, _ in prefilling]
        ):
            return None
        if prefilling and self._queued_left(head) >= min(
            left for _, left in prefilling
        ):
            return None
        return head

    def _prefilling_lefts(self) -> list[tuple[_Job, int]]:
        return [
            (job, joining.left)
            for job, joining in zip(self.prefilling, self.engine.prefilling)
        ]

    def _leaving(self) -> list[int]:
        return [
            index
            for index, job in enumerate(self.running)
            if job.finished or job.cancelled
        ]

    def _aborting(self) -> list[int]:
        return [index for index, job in enumerate(self.prefilling) if job.cancelled]

    def wants_plan(self, chunk: bool) -> bool:
        """Whether the next tick opens with a plan (``chunk``: the target's chunk runs this step)."""
        if self.stopping or self.state.shutting_down or self._leaving():
            return True
        if self._aborting():
            return True
        self._collect(block=False)
        if self.stopping:
            return True
        running = list(self.running)
        prefilling = self._prefilling_lefts()
        if chunk:
            # The state after this step's chunk: the target has that many
            # tokens fewer left, or it joined the running batch.
            target = self.engine.target
            start, stop = self.engine.prefilling[target].ranges[0]
            job, left = prefilling[target]
            if left - (stop - start):
                prefilling[target] = (job, left - (stop - start))
            else:
                del prefilling[target]
                running.append(job)
        return self._admissible(running, prefilling) is not None

    def plan(self, block: bool) -> _Plan | None:
        """The next tick's plan; None = shut down.  ``block``: the engine is idle."""
        state = self.state
        leave = self._leaving()
        abort = self._aborting()
        survivors = [job for i, job in enumerate(self.running) if i not in leave]
        pairs = [
            pair
            for i, pair in enumerate(zip(self.prefilling, self.engine.prefilling))
            if i not in abort
        ]
        prefilling = [(job, joining.left) for job, joining in pairs]
        joiner, evict = None, []
        needs = []
        if state.prefix is not None:
            # goose Q-179: a leaving row's cache becomes its snapshot in this plan.
            state.prefix.leaving([self.running[i].row for i in leave])
            needs.append(
                state.kv.held(
                    self._decoding(survivors),
                    self._prompts([job for job, _ in prefilling]),
                )
            )
        if state.prefix is not None and survivors:
            # The target's merge waits on prefix entries: they go first.
            room, need = self._join_room(survivors, pairs)
            if room == "evict":
                evict = state.prefix.yield_to(need)
            if need is not None:
                needs.append(need)
        self._collect(block=block and not survivors and not prefilling)
        while not (self.stopping or state.shutting_down):
            head = self._admissible(survivors, prefilling)
            if head is not None:
                self._take(head)
                self.admitted = head
                joiner = head.row
                if state.prefix is not None:
                    need = self._admission(
                        head, survivors, [job for job, _ in prefilling]
                    )
                    needs.append(need)
                    evict += state.prefix.admit(head.row, need)
                break
            head = self._head(prefilling)
            if head is None or survivors or prefilling:
                break
            # Alone it still does not fit: the plan cannot hold it at all
            # (a lone row at <= context always fits the plan's first slot,
            # so this names a planner defect rather than waiting forever).
            self._take(head)
            needed = self._admission(head, [], [])
            head.finished = True
            head.push(
                (
                    "error",
                    f"the pipeline's KV budget {state.kv.budgets} cannot hold "
                    f"this request alone ({needed} bytes per rank)",
                )
            )
            self._collect(block=block)
        if self.stopping or state.shutting_down:
            return None
        if needs:
            evict += state.prefix.settle([max(values) for values in zip(*needs)])
        return _Plan(leave=leave, abort=abort, joiner=joiner, evict=evict)

    def applied(self, plan: _Plan) -> None:
        if self.state.kv is not None:
            self.state.kv.priced.clear()
        leaving = set(plan.leave)
        self.running = [job for i, job in enumerate(self.running) if i not in leaving]
        if plan.abort:
            aborted = set(plan.abort)
            for index in aborted:
                row = self.prefilling[index].row
                if self.state.prefix is not None and row.store_id:
                    self.state.prefix.forget(row.store_id)
            self.prefilling = [
                job for i, job in enumerate(self.prefilling) if i not in aborted
            ]
        if plan.joiner is not None:
            if not self.prefilling:
                self.credit = 0.0
            # The engine applied the plan first: its last prefilling row is
            # the joiner, its ranges whole (after the prefix restore).
            self.admitted.prefill = self.engine.prefilling[-1].left
            self.prefilling.append(self.admitted)
            self.admitted = None
        self._publish()

    def _publish(self) -> None:
        active = [*self.running, *self.prefilling]
        kv = self.state.kv
        self.state.reserved = (
            kv.held(self._decoding(self.running), self._prompts(self.prefilling))
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
        room, _ = self._join_room(
            self.running, list(zip(self.prefilling, self.engine.prefilling))
        )
        # A held join is no prefill work to share time with.
        self.holding = room != "fits"
        chunk = bool(self.prefilling) and self.credit >= 0 and not self.holding
        # When a chunk follows, its own collective carries the plan word.
        plan = False if chunk else room == "evict" or self.wants_plan(chunk=False)
        return [int(plan), int(chunk)]

    def decoded(self, tokens: list[int], seconds: float) -> None:
        self.state.steps += 1
        for job, token in zip(self.running, tokens):
            self._give(job, token)
        if self.prefilling and not self.holding:
            self.credit += seconds * _PREFILL_SHARE / (1 - _PREFILL_SHARE)

    def chunk_words(self) -> list[int]:
        return [int(self.wants_plan(chunk=True)), 0]

    def chunked(
        self, index: int, tokens: int, first: int | None, seconds: float
    ) -> None:
        """Prefilling row ``index`` ran a chunk of ``tokens`` tokens (``first``: it joined)."""
        self.credit -= seconds
        job = self.prefilling[index]
        # A JUMP: a later arrival's tokens went ahead of every earlier request
        # still waiting for its own.  An earlier arrival's chunk is the order
        # a later request would have had anyway, so it ages nothing (goose
        # Q-145 26d: counting it too made every long request protected before
        # it started, and the canaries queued FIFO again).
        for other in [*self.state.waiting, *self.prefilling]:
            if other.seq < job.seq:
                other.waited += tokens
        if first is None:
            return
        self.prefilling = [other for other in self.prefilling if other is not job]
        self.running.append(job)
        self._give(job, first)
        self._publish()

    def tell(self, message: str) -> None:
        """Every request the engine holds or has queued ends with ``message``."""
        admitted = [self.admitted] if self.admitted is not None else []
        for job in [*self.running, *self.prefilling, *admitted, *self.state.waiting]:
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
    prefilling rows and their ranges, the collectives' words), so the
    forwards and collectives pair across ranks.
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
        chunk = bool(engine.prefilling)
        if engine.rows:
            words = scheduler.decode_words() if deciding else None
            began = time.perf_counter()
            tokens, words = engine.decode(words)
            if deciding:
                scheduler.decoded(tokens, time.perf_counter() - began)
            pending, chunk = bool(words[0]), chunk and bool(words[1])
        if chunk:
            if deciding:
                target = engine.target
                start, stop = engine.prefilling[target].ranges[0]
                words = scheduler.chunk_words()
            else:
                words = None
            began = time.perf_counter()
            first, words = engine.prefill(words)
            if deciding:
                scheduler.chunked(
                    target, stop - start, first, time.perf_counter() - began
                )
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


def _xml_close_guard_spec(model_dir: Path, rank: int, log):
    """The sampling rank's XML tool-call skeleton rules, or None — said either way.

    The single engine's lz.7 (goose Q-85), derived from this checkpoint's own
    tokenizer and template exactly as ``Scheduler._xml_tool_close_guard`` does,
    with the same opt-out variable.  Only the last rank samples, so only it
    loads the tokenizer for this.
    """
    from ..xml_tool_close_guard import OPT_OUT_ENV, xml_close_guard_spec

    opt_out = os.environ.get(OPT_OUT_ENV, "").strip().lower()
    if opt_out in ("0", "off", "false", "no"):
        log(
            f"[pipeline] rank {rank}: xml tool-call skeleton guard disabled by "
            f"{OPT_OUT_ENV}={opt_out}"
        )
        return None
    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(model_dir)
    spec = xml_close_guard_spec(tokenizer, tokenizer.eos_token_ids)
    log(
        f"[pipeline] rank {rank}: xml tool-call skeleton guard "
        + (
            f"armed ({len(spec.rules)} rules)"
            if spec is not None
            else "not armed: this checkpoint's wire is not the parameterised XML "
            "tool call (or its markers are not single tokens)"
        )
    )
    return spec


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
    close_guard = (
        _xml_close_guard_spec(
            model_dir, stage.rank, lambda line: print(line, flush=True)
        )
        if stage.is_last
        else None
    )
    engine = _Engine(stage, guard, prefill_step, store, close_guard=close_guard)
    # A readiness probe that succeeds means a request can run.
    _warm(engine)
    kv = _KvBudget(plan, prefill_step)
    rows = _agree_rows(stage, kv.rows())
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
        _ticks(engine, group, rows, wake)
        return 0

    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(model_dir)
    sampling = _SamplingDefaults.load(model_dir, _SamplingDefaults.profile_of(options))
    emit("SAMPLING_DEFAULTS", sampling.report())
    if sampling.error is not None:
        emit(
            "GENERATION_CONFIG_UNREAD",
            {"error": sampling.error, "in_force": sampling.report()["engine_fallback"]},
        )
    state = _State(
        served=options.served_model_name,
        aliases=tuple(options.served_model_alias or ()),
        context=context,
        max_batch=rows,
        kv=kv,
        prefix=None if store is None else _PrefixIndex(kv),
        eos_ids=frozenset(tokenizer.eos_token_ids),
        sampling=sampling,
        vocab=plan.args.vocab_size,
        emit=emit,
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
        help="full-context sequences the split is planned (and KV-budgeted) for; "
        "requests are admitted by what each one needs of that budget (its prompt "
        "and max_tokens), not by this count",
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
    for key in _SAMPLING_KEYS:
        parser.add_argument(
            f"--default-{key.replace('_', '-')}",
            type=int if key == "top_k" else float,
            default=None,
            help=f"{key} for a request that sets none, above the checkpoint's "
            "generation_config.json (the single engine's flag of the same name)",
        )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="serve text only: rank 0 does not load the checkpoint's vision tower",
    )


if __name__ == "__main__":
    sys.exit(pipe.main(["serve", *sys.argv[1:]]))
