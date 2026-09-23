# SPDX-License-Identifier: Apache-2.0
"""Pipeline-parallel split of the qwen4_exp text decoder over MLX distributed.

Each rank holds one contiguous range of decoder layers.  Rank 0 also holds the
token embedding; the last rank also holds the final hyper-connection mixer and
``lm_head``.  The only activation that crosses a rank boundary is the
four-stream hyper-connection state ``[batch, tokens, hc_count * hidden]`` that
``Qwen4ExpTextModel`` threads through its layer loop; every cache
(GDN/PLE state, main KV, QSA index keys) stays on the rank that owns the layer.

The PLE n-gram tables are NOT a model-level table: the checkpoint stores them
under the decoder layers named by ``ple_layer_ids`` (Qwen3.8-Flash-Next:
layer 2, about 32 GB at 4-bit), so they travel with their layer and the
planner accounts for them as that layer's bytes.  PLE needs the raw token ids;
every rank already holds them because every rank runs the same driver loop and
the sampled tokens are broadcast each step.

Nothing here is imported by the single-node serving path.  Text only: the
vision tower (``model.visual.*``) and the MTP head (``mtp.*``) are dropped by
the model's own ``sanitize`` and are reported as excluded bytes.  Serving
images would need the vision tower on rank 0 and its merged embeddings fed in
place of ``embed_tokens``; that is not implemented.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import inspect
import json
import os
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx

from .. import _mlx_compat as _mlx_compat

_mlx_compat.install()

from mlx_lm.models.cache import KVCache  # noqa: E402

from ..models.qwen4_exp import Model, TextModelArgs  # noqa: E402
from ..models.qwen4_exp_cache import QSAIndexCache  # noqa: E402

# ratio: the MTPLX run measured a stable ceiling at 75% of RAM for MLX
# allocations on 96-128 GB Apple silicon (mlx-jaccl-cluster skill, guardrail 3).
MEMORY_LIMIT_RATIO = 0.75
# ratio: the same MTPLX receipt measured 60% of RAM as the safe wired ceiling;
# exo's unbounded wiring is what panicked the 96 GB M3 Ultra.
WIRED_LIMIT_RATIO = 0.60
# ratio: policy, not yet measured — 10% of the memory the kernel reports as
# free stays free for the OS and for transients the workspace model misses.
AVAILABLE_HEADROOM_RATIO = 0.90
# Inside DecoderLayer._combine the incoming stream, the residual copy and the
# combined output are alive together: three stream-width buffers.
STREAM_LIVE_BUFFERS = 3

# kern.memorystatus_vm_pressure_level values published by xnu
# (osfmk/kern/kern_memorystatus... kVMPressureNormal/Warning/Critical).
_PRESSURE_WARN = 2
_PRESSURE_CRITICAL = 4

_DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "I32": 4, "I64": 8, "U8": 1}
_MX_DTYPE_CODES = {mx.bfloat16: 1, mx.float16: 2, mx.float32: 3}
_MX_DTYPES_BY_CODE = {code: dtype for dtype, code in _MX_DTYPE_CODES.items()}
_LAYER_KEY = re.compile(r"^language_model\.model\.layers\.(\d+)\.")


class PipelineDoesNotFitError(RuntimeError):
    """A rank's slice plus runtime state exceeds that node's budget."""


class PipelineMemoryStopError(RuntimeError):
    """A rank's guard tripped; every rank stops on the same step."""


# ---------------------------------------------------------------------------
# Memory measurement (macOS kernel counters, never "available" from psutil).
# ---------------------------------------------------------------------------


def _sysctl_int(name: str) -> int:
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    value = ctypes.c_uint64(0)
    size = ctypes.c_size_t(ctypes.sizeof(value))
    if libc.sysctlbyname(
        name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0
    ):
        errno = ctypes.get_errno()
        raise OSError(errno, f"sysctlbyname({name}) failed: {os.strerror(errno)}")
    if size.value == 4:
        return int(value.value & 0xFFFFFFFF)
    return int(value.value)


@dataclass(frozen=True)
class NodeMemory:
    total_bytes: int
    available_bytes: int
    free_percent: int
    pressure_level: int

    @property
    def budget_bytes(self) -> int:
        return min(
            int(self.available_bytes * AVAILABLE_HEADROOM_RATIO),
            int(self.total_bytes * MEMORY_LIMIT_RATIO),
        )


class _VMStatistics64(ctypes.Structure):
    """``struct vm_statistics64`` from <mach/vm_statistics.h>."""

    _fields_ = [
        ("free_count", ctypes.c_uint32),
        ("active_count", ctypes.c_uint32),
        ("inactive_count", ctypes.c_uint32),
        ("wire_count", ctypes.c_uint32),
        ("zero_fill_count", ctypes.c_uint64),
        ("reactivations", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("pageouts", ctypes.c_uint64),
        ("faults", ctypes.c_uint64),
        ("cow_faults", ctypes.c_uint64),
        ("lookups", ctypes.c_uint64),
        ("hits", ctypes.c_uint64),
        ("purges", ctypes.c_uint64),
        ("purgeable_count", ctypes.c_uint32),
        ("speculative_count", ctypes.c_uint32),
        ("decompressions", ctypes.c_uint64),
        ("compressions", ctypes.c_uint64),
        ("swapins", ctypes.c_uint64),
        ("swapouts", ctypes.c_uint64),
        ("compressor_page_count", ctypes.c_uint32),
        ("throttled_count", ctypes.c_uint32),
        ("external_page_count", ctypes.c_uint32),
        ("internal_page_count", ctypes.c_uint32),
        ("total_uncompressed_pages_in_compressor", ctypes.c_uint64),
    ]


_HOST_VM_INFO64 = 4


def _host_vm_statistics() -> _VMStatistics64:
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    libc.mach_host_self.restype = ctypes.c_uint32
    stats = _VMStatistics64()
    count = ctypes.c_uint32(ctypes.sizeof(stats) // ctypes.sizeof(ctypes.c_int32))
    result = libc.host_statistics64(
        libc.mach_host_self(), _HOST_VM_INFO64, ctypes.byref(stats), ctypes.byref(count)
    )
    if result != 0:
        raise OSError(f"host_statistics64 returned kern_return_t {result}")
    return stats


def measure_node_memory() -> NodeMemory:
    """Memory a new allocation can take without pushing anything to swap.

    (free - speculative) + file-backed + purgeable pages, from
    host_statistics64: truly-free pages plus file cache and purgeable memory
    the kernel reclaims on demand.  The same measure goose-sidecar's
    memory.rs uses.  Neither psutil/sysinfo "available" (counts file cache as
    used: read 0 GB on a machine 88% free) nor ``kern.memorystatus_level``
    (stayed at 46-47% while the real figure moved 41.4 -> 28.0 GiB, measured
    by goose-sidecar 2026-09-23).
    """
    total = _sysctl_int("hw.memsize")
    page = _sysctl_int("hw.pagesize")
    stats = _host_vm_statistics()
    truly_free = max(0, stats.free_count - stats.speculative_count)
    pages = truly_free + stats.external_page_count + stats.purgeable_count
    available = min(total, pages * page)
    return NodeMemory(
        total_bytes=total,
        available_bytes=available,
        free_percent=round(100 * available / total),
        pressure_level=_sysctl_int("kern.memorystatus_vm_pressure_level"),
    )


# ---------------------------------------------------------------------------
# Checkpoint accounting from index + safetensors headers (no tensor is read).
# ---------------------------------------------------------------------------


@dataclass
class CheckpointBytes:
    layer_bytes: list[int]
    head_bytes: int
    tail_bytes: int
    excluded_bytes: dict[str, int]
    activation_bytes: int

    @property
    def text_bytes(self) -> int:
        return sum(self.layer_bytes) + self.head_bytes + self.tail_bytes


def _normalized_key(key: str) -> str | None:
    """Map a checkpoint key onto the loaded module path (Model.sanitize)."""
    if key.startswith("mtp."):
        return None
    if key.startswith("model.visual") or key.startswith("vision_tower"):
        return None
    if key.startswith("model.language_model"):
        return key.replace("model.language_model", "language_model.model", 1)
    if key.startswith("language_model."):
        return key
    return f"language_model.{key}"


def _read_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(length))


def read_checkpoint_bytes(model_dir: Path, num_layers: int) -> CheckpointBytes:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        weight_map: dict[str, str] = json.loads(index.read_text())["weight_map"]
        files = sorted(set(weight_map.values()))
    else:
        files = sorted(path.name for path in model_dir.glob("model*.safetensors"))
        weight_map = {}
    if not files:
        raise FileNotFoundError(f"no safetensors shards in {model_dir}")
    layer_bytes = [0] * num_layers
    head = tail = 0
    excluded = {"mtp": 0, "vision": 0}
    embed_dtypes: dict[str, str] = {}
    for name in files:
        header = _read_header(model_dir / name)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if weight_map and weight_map.get(key) != name:
                raise ValueError(f"{key} is in {name} but the index says otherwise")
            start, end = meta["data_offsets"]
            nbytes = end - start
            mapped = _normalized_key(key)
            if mapped is None:
                excluded["mtp" if key.startswith("mtp.") else "vision"] += nbytes
                continue
            layer = _LAYER_KEY.match(mapped)
            if layer is not None:
                layer_bytes[int(layer.group(1))] += nbytes
            elif mapped.startswith("language_model.model.embed_tokens."):
                head += nbytes
                embed_dtypes[mapped.rsplit(".", 1)[1]] = meta["dtype"]
            elif mapped.startswith(
                "language_model.model.hyper_connection_mixer."
            ) or mapped.startswith("language_model.lm_head."):
                tail += nbytes
            else:
                raise ValueError(f"checkpoint tensor {key} has no pipeline owner")
    # A quantized embedding dequantizes to its scales' dtype; that is the
    # dtype of the hyper-connection stream and of every KV/state buffer.
    stream_dtype = embed_dtypes.get("scales", embed_dtypes.get("weight"))
    if stream_dtype not in _DTYPE_BYTES:
        raise ValueError(f"cannot derive the activation dtype from {embed_dtypes}")
    return CheckpointBytes(
        layer_bytes=layer_bytes,
        head_bytes=head,
        tail_bytes=tail,
        excluded_bytes=excluded,
        activation_bytes=_DTYPE_BYTES[stream_dtype],
    )


def load_text_args(model_dir: Path) -> TextModelArgs:
    config = json.loads((model_dir / "config.json").read_text())
    return TextModelArgs.from_dict(config.get("text_config", config))


# ---------------------------------------------------------------------------
# Runtime state and workspace model.
# ---------------------------------------------------------------------------


def _round_up(value: int, step: int) -> int:
    return -(-value // step) * step


def layer_state_bytes(
    args: TextModelArgs, index: int, context: int, batch: int, act: int
) -> int:
    """Cache bytes one decoder layer holds for ``batch`` sequences."""
    kind = args.layer_types[index]
    if kind == "linear_attention":
        key_dim = args.linear_num_key_heads * args.linear_key_head_dim
        value_dim = args.linear_num_value_heads * args.linear_value_head_dim
        conv = (args.linear_conv_kernel_dim - 1) * (2 * key_dim + value_dim) * act
        # gated_delta.py allocates the recurrent state in float32.
        recurrent = (
            args.linear_num_value_heads
            * args.linear_value_head_dim
            * args.linear_key_head_dim
            * 4
        )
        per_sequence = conv + recurrent
        if index + 1 in args.ple_layer_ids:
            hc_width = args.hc_count * args.hidden_size
            ple_conv = (args.ple_conv_kernel_size - 1) * args.ngram_size
            per_sequence += ple_conv * hc_width * act + (args.ngram_size - 1) * 8
        return batch * per_sequence
    kv = 2 * args.num_key_value_heads * args.head_dim * act
    kv *= _round_up(context, KVCache.step)
    ratio = int(args.indexer_compress_ratio)
    index_dim = int(args.indexer_head_dim)
    compressed = _round_up(-(-context // ratio), QSAIndexCache.step) * index_dim * act
    ring = ratio * index_dim * act
    return batch * (kv + compressed + ring)


def workspace_bytes(
    args: TextModelArgs,
    layers: range,
    *,
    is_last: bool,
    context: int,
    batch: int,
    prefill_step: int,
    act: int,
) -> int:
    """Modeled (unmeasured) peak transient for one prefill chunk on a stage."""
    tokens = min(prefill_step, context)
    hc_width = args.hc_count * args.hidden_size
    stream = batch * tokens * hc_width * act * STREAM_LIVE_BUFFERS
    widths = []
    has_attention = False
    for index in layers:
        if args.layer_types[index] == "linear_attention":
            key_dim = args.linear_num_key_heads * args.linear_key_head_dim
            value_dim = args.linear_num_value_heads * args.linear_value_head_dim
            widths.append(2 * key_dim + 2 * value_dim + 2 * args.linear_num_value_heads)
        else:
            has_attention = True
            widths.append(
                2 * args.num_attention_heads * args.head_dim
                + 2 * args.num_key_value_heads * args.head_dim
                + (int(args.indexer_n_heads) + 1) * int(args.indexer_head_dim)
            )
        widths.append(
            args.num_experts_per_tok
            * max(3 * args.moe_intermediate_size, args.hidden_size)
        )
    projections = batch * tokens * max(widths) * 4
    # The QSA dense fallback materializes a boolean selection and an additive
    # mask over the physical KV length for the chunk's queries.
    attention = batch * tokens * context * (1 + act) if has_attention else 0
    logits = batch * args.vocab_size * 4 if is_last else 0
    return stream + projections + attention + logits


# ---------------------------------------------------------------------------
# The split.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeBudget:
    name: str
    total_bytes: int
    budget_bytes: int
    source: str


@dataclass
class StagePlan:
    rank: int
    node: NodeBudget
    start: int
    end: int
    weight_bytes: int
    state_bytes: int
    workspace_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.weight_bytes + self.state_bytes + self.workspace_bytes

    @property
    def utilization(self) -> float:
        return self.total_bytes / self.node.budget_bytes


@dataclass
class PipelinePlan:
    stages: list[StagePlan]
    context: int
    batch: int
    prefill_step: int
    checkpoint: CheckpointBytes
    max_context: int | None = field(default=None)

    @property
    def starts(self) -> list[int]:
        return [stage.start for stage in self.stages]


def _stage_plan(
    args: TextModelArgs,
    ckpt: CheckpointBytes,
    node: NodeBudget,
    rank: int,
    size: int,
    start: int,
    end: int,
    context: int,
    batch: int,
    prefill_step: int,
) -> StagePlan:
    weights = sum(ckpt.layer_bytes[start:end])
    if rank == 0:
        weights += ckpt.head_bytes
    if rank == size - 1:
        weights += ckpt.tail_bytes
    act = ckpt.activation_bytes
    state = sum(
        layer_state_bytes(args, index, context, batch, act)
        for index in range(start, end)
    )
    workspace = workspace_bytes(
        args,
        range(start, end),
        is_last=rank == size - 1,
        context=context,
        batch=batch,
        prefill_step=prefill_step,
        act=act,
    )
    return StagePlan(rank, node, start, end, weights, state, workspace)


def _stages_for_starts(args, ckpt, nodes, starts, context, batch, prefill_step):
    bounds = [*starts, args.num_hidden_layers]
    return [
        _stage_plan(
            args,
            ckpt,
            node,
            rank,
            len(nodes),
            bounds[rank],
            bounds[rank + 1],
            context,
            batch,
            prefill_step,
        )
        for rank, node in enumerate(nodes)
    ]


def _balanced_starts(args, ckpt, nodes, context, batch, prefill_step) -> list[int]:
    """Contiguous split minimizing the worst rank's cost / budget.

    Equal utilization is the same thing as layer bytes proportional to each
    node's usable memory, but it also charges the embedding, lm_head, KV and
    workspace to the rank that actually holds them.
    """
    layers = args.num_hidden_layers
    size = len(nodes)
    if size > layers:
        raise PipelineDoesNotFitError(f"{size} ranks cannot split {layers} layers")
    inf = float("inf")
    best = [[inf] * (layers + 1) for _ in range(size)]
    choice = [[0] * (layers + 1) for _ in range(size)]
    for end in range(1, layers + 1):
        best[0][end] = _stage_plan(
            args, ckpt, nodes[0], 0, size, 0, end, context, batch, prefill_step
        ).utilization
    for rank in range(1, size):
        for end in range(rank + 1, layers + 1):
            for start in range(rank, end):
                if best[rank - 1][start] == inf:
                    continue
                cost = _stage_plan(
                    args,
                    ckpt,
                    nodes[rank],
                    rank,
                    size,
                    start,
                    end,
                    context,
                    batch,
                    prefill_step,
                ).utilization
                worst = max(best[rank - 1][start], cost)
                if worst < best[rank][end]:
                    best[rank][end] = worst
                    choice[rank][end] = start
    starts = [0] * size
    end = layers
    for rank in range(size - 1, 0, -1):
        starts[rank] = choice[rank][end]
        end = starts[rank]
    return starts


def _fits(stages: list[StagePlan]) -> bool:
    return all(stage.total_bytes <= stage.node.budget_bytes for stage in stages)


def plan_pipeline(
    args: TextModelArgs,
    ckpt: CheckpointBytes,
    nodes: list[NodeBudget],
    *,
    context: int,
    batch: int,
    prefill_step: int,
    starts: list[int] | None = None,
) -> PipelinePlan:
    if starts is None:
        starts = _balanced_starts(args, ckpt, nodes, context, batch, prefill_step)
    if len(starts) != len(nodes) or starts[0] != 0 or starts != sorted(set(starts)):
        raise ValueError(f"invalid split starts {starts} for {len(nodes)} ranks")
    if starts[-1] >= args.num_hidden_layers:
        raise ValueError(f"split {starts} leaves the last rank without layers")
    stages = _stages_for_starts(args, ckpt, nodes, starts, context, batch, prefill_step)
    plan = PipelinePlan(stages, context, batch, prefill_step, ckpt)
    # Context ceiling on this split: the largest context every rank fits.
    low, high = 0, args.max_position_embeddings
    while low < high:
        middle = (low + high + 1) // 2
        trial = _stages_for_starts(
            args, ckpt, nodes, starts, middle, batch, min(prefill_step, middle)
        )
        if _fits(trial):
            low = middle
        else:
            high = middle - 1
    plan.max_context = low or None
    return plan


def require_fit(plan: PipelinePlan) -> None:
    if _fits(plan.stages):
        return
    raise PipelineDoesNotFitError(
        "pipeline does not fit — refusing to load:\n" + format_plan(plan)
    )


def _gib(value: int) -> str:
    return f"{value / 2**30:,.2f} GiB"


def format_plan(plan: PipelinePlan) -> str:
    ckpt = plan.checkpoint
    lines = [
        f"checkpoint text bytes {_gib(ckpt.text_bytes)} "
        f"(excluded: mtp {_gib(ckpt.excluded_bytes['mtp'])}, "
        f"vision {_gib(ckpt.excluded_bytes['vision'])}); "
        f"embed {_gib(ckpt.head_bytes)}, mixer+lm_head {_gib(ckpt.tail_bytes)}",
        f"context {plan.context:,} tokens, batch {plan.batch}, "
        f"prefill chunk {plan.prefill_step}",
    ]
    for stage in plan.stages:
        node = stage.node
        source = node.source
        verdict = "fits" if stage.total_bytes <= node.budget_bytes else "DOES NOT FIT"
        lines.append(
            f"rank {stage.rank} {node.name}: layers [{stage.start}, {stage.end}) "
            f"= {stage.end - stage.start} layers | weights {_gib(stage.weight_bytes)} "
            f"+ state {_gib(stage.state_bytes)} + workspace(modeled) "
            f"{_gib(stage.workspace_bytes)} = {_gib(stage.total_bytes)} "
            f"of budget {_gib(node.budget_bytes)} ({source}; RAM "
            f"{_gib(node.total_bytes)}) -> {stage.utilization:.0%} {verdict}"
        )
    if plan.max_context is None:
        lines.append("no context length fits this split")
    else:
        lines.append(f"largest context that fits this split: {plan.max_context:,}")
    return "\n".join(lines)


def wire_bytes_per_token(args: TextModelArgs, act: int, ranks: int) -> dict[str, int]:
    """Bytes crossing every rank boundary, per token of one sequence."""
    stream = args.hc_count * args.hidden_size * act
    return {
        "stream_per_hop": stream,
        "hops": ranks - 1,
        # One int32 token plus the shared stop flag, all_sum'd each step.
        "token_broadcast": 4,
        "total_per_token": stream * (ranks - 1) + 4,
    }


# ---------------------------------------------------------------------------
# Guardrails.
# ---------------------------------------------------------------------------


def apply_memory_guardrails(node: NodeMemory, planned_bytes: int) -> dict[str, int]:
    """Bound MLX on this rank as ratios of this node's RAM.

    ``set_memory_limit`` is a guideline in MLX 0.32 (the allocator waits and
    reclaims cache first), so the hard stop is :class:`MemoryGuard`; the wired
    limit is the one that prevents the exo-style wired-memory panic.
    """
    budget = node.budget_bytes
    if planned_bytes > budget:
        raise PipelineDoesNotFitError(
            f"this rank plans {_gib(planned_bytes)} against a budget of "
            f"{_gib(budget)} ({node.free_percent}% of {_gib(node.total_bytes)} free)"
        )
    memory_limit = int(node.total_bytes * MEMORY_LIMIT_RATIO)
    system_wired = int(mx.device_info()["max_recommended_working_set_size"])
    wired_limit = min(int(node.total_bytes * WIRED_LIMIT_RATIO), system_wired)
    cache_limit = max(0, memory_limit - planned_bytes)
    mx.set_memory_limit(memory_limit)
    mx.set_wired_limit(wired_limit)
    mx.set_cache_limit(cache_limit)
    return {
        "memory_limit": memory_limit,
        "wired_limit": wired_limit,
        "cache_limit": cache_limit,
        "budget": budget,
    }


class MemoryGuard:
    """Per-step check; a trip is broadcast so every rank stops together."""

    def __init__(self, budget_bytes: int, rank: int):
        self.budget_bytes = budget_bytes
        self.rank = rank
        self.warned = False

    def check(self) -> str | None:
        pressure = _sysctl_int("kern.memorystatus_vm_pressure_level")
        if pressure >= _PRESSURE_CRITICAL:
            return f"rank {self.rank}: kernel memory pressure CRITICAL"
        if pressure >= _PRESSURE_WARN and not self.warned:
            self.warned = True
            print(
                f"[pipeline] rank {self.rank}: kernel memory pressure WARN",
                file=sys.stderr,
                flush=True,
            )
        active = mx.get_active_memory()
        if active > self.budget_bytes:
            return (
                f"rank {self.rank}: MLX active memory {_gib(active)} exceeds the "
                f"rank budget {_gib(self.budget_bytes)}"
            )
        return None


# ---------------------------------------------------------------------------
# The stage.
# ---------------------------------------------------------------------------


def slice_model(model: Model, rank: int, size: int, start: int, end: int) -> None:
    """Drop every module this rank does not own (before weights materialize)."""
    text = model.language_model
    if text.args.tie_word_embeddings and size > 1:
        raise NotImplementedError(
            "tied embeddings would need embed_tokens on the first and last rank"
        )
    inner = text.model
    inner.layers = inner.layers[start:end]
    if rank != 0:
        del inner["embed_tokens"]
    if rank != size - 1:
        del inner["hyper_connection_mixer"]
        del text["lm_head"]


class PipelineStage:
    def __init__(
        self,
        model: Model,
        group: Any,
        start: int,
        end: int,
        wire_dtype: mx.Dtype,
    ):
        self.model = model
        self.group = group
        self.rank = group.rank() if group is not None else 0
        self.size = group.size() if group is not None else 1
        self.start = start
        self.end = end
        self.wire_dtype = wire_dtype
        self.args = model.language_model.args

    @property
    def is_first(self) -> bool:
        return self.rank == 0

    @property
    def is_last(self) -> bool:
        return self.rank == self.size - 1

    def make_cache(self, left_padding: list[int] | None = None) -> list[Any]:
        if left_padding is None:
            return self.model.make_cache()
        from mlx_lm.generate import _make_cache

        return _make_cache(self.model, left_padding, None)

    def forward(self, inputs: mx.array, cache: list[Any], logits: str | None):
        """One pipeline stage of ``Qwen4ExpTextModel.__call__``.

        Returns the send handle on non-last ranks; on the last rank the
        logits (``"all"`` positions or the ``"last"`` one), or the stream
        itself when ``logits`` is None (prefill chunks whose logits nobody
        reads).
        """
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        inner = self.model.language_model.model
        batch, length = inputs.shape
        if self.is_first:
            hidden = inner.embed_tokens(inputs)
            hidden = mx.tile(hidden, (1, 1, self.args.hc_count))
        else:
            hidden = mx.distributed.recv(
                (batch, length, self.args.hc_count * self.args.hidden_size),
                self.wire_dtype,
                self.rank - 1,
                group=self.group,
            )
        layers = inner.layers
        linear_index = next(
            (index for index, layer in enumerate(layers) if layer.is_linear), None
        )
        linear_mask = (
            None
            if linear_index is None
            else create_ssm_mask(hidden, cache[linear_index])
        )
        attention_index = next(
            (index for index, layer in enumerate(layers) if not layer.is_linear), None
        )
        attention_cache = (
            None
            if attention_index is None or cache[attention_index] is None
            else cache[attention_index][0]
        )
        attention_mask = create_attention_mask(hidden, attention_cache)
        for layer, layer_cache in zip(layers, cache):
            hidden = layer(
                hidden,
                input_ids=inputs,
                mask=linear_mask if layer.is_linear else attention_mask,
                cache=layer_cache,
            )
        if not self.is_last:
            if hidden.dtype != self.wire_dtype:
                raise TypeError(
                    f"rank {self.rank} stream is {hidden.dtype}, the wire "
                    f"contract is {self.wire_dtype}"
                )
            return mx.distributed.send(hidden, self.rank + 1, group=self.group)
        if logits is None:
            return hidden
        if logits == "last":
            hidden = hidden[:, -1:, :]
        return self.model.language_model.lm_head(inner.hyper_connection_mixer(hidden))


def _step_sync(
    stage: PipelineStage,
    out: mx.array,
    cache: list[Any],
    batch: int,
    guard: MemoryGuard | None,
    *,
    sample: bool,
) -> list[int]:
    """Close one pipeline step on every rank with a single collective.

    The payload is ``[next tokens..., stop flag]``.  Non-owners contribute
    zeros that depend on this rank's own work, so a rank never enters the
    all_sum before its send/recv have been issued (the ordering that would
    otherwise deadlock a blocking ring send).
    """
    reason = guard.check() if guard is not None else None
    if stage.is_last and sample:
        tokens = mx.argmax(out[:, -1, :], axis=-1).astype(mx.int32)
    else:
        tokens = mx.depends(mx.zeros((batch,), dtype=mx.int32), out)
    payload = mx.concatenate([tokens, mx.array([1 if reason else 0], dtype=mx.int32)])
    if stage.size > 1:
        payload = mx.distributed.all_sum(payload, group=stage.group)
    mx.eval(payload, [layer_cache.state for layer_cache in cache])
    values = payload.tolist()
    if values[-1]:
        raise PipelineMemoryStopError(reason or "a peer rank's memory guard tripped")
    return values[:batch]


def _left_pad(prompts: list[list[int]], pad_id: int) -> tuple[mx.array, list[int]]:
    width = max(len(prompt) for prompt in prompts)
    padding = [width - len(prompt) for prompt in prompts]
    rows = [[pad_id] * pad + prompt for pad, prompt in zip(padding, prompts)]
    return mx.array(rows, dtype=mx.int32), padding


def pipeline_prefill_logits(
    stage: PipelineStage,
    prompts: list[list[int]],
    *,
    pad_id: int = 0,
    guard: MemoryGuard | None = None,
) -> mx.array | None:
    """Full-prompt logits from one un-chunked forward (last rank only)."""
    tokens, padding = _left_pad(prompts, pad_id)
    cache = stage.make_cache(padding if len(prompts) > 1 else None)
    out = stage.forward(tokens, cache, logits="all")
    if stage.is_last:
        mx.eval(out)
    _step_sync(stage, out, cache, len(prompts), guard, sample=False)
    return out if stage.is_last else None


def default_prefill_step() -> int:
    from mlx_lm.generate import generate_step

    return int(inspect.signature(generate_step).parameters["prefill_step_size"].default)


def pipeline_generate(
    stage: PipelineStage,
    prompts: list[list[int]],
    max_tokens: int,
    *,
    pad_id: int = 0,
    eos_ids: tuple[int, ...] = (),
    prefill_step: int | None = None,
    guard: MemoryGuard | None = None,
) -> list[list[int]]:
    """Greedy decode, identical token stream on every rank.

    Mirrors mlx-lm's ``generate_step``: all prompt tokens but the last are
    prefilled in chunks with their logits never evaluated; the last prompt
    token is the first decode step.
    """
    step = prefill_step or default_prefill_step()
    tokens, padding = _left_pad(prompts, pad_id)
    batch = len(prompts)
    cache = stage.make_cache(padding if batch > 1 else None)
    prefix = tokens[:, :-1]
    for offset in range(0, prefix.shape[1], step):
        out = stage.forward(prefix[:, offset : offset + step], cache, logits=None)
        _step_sync(stage, out, cache, batch, guard, sample=False)
    current = tokens[:, -1:]
    generated: list[list[int]] = [[] for _ in range(batch)]
    finished = [False] * batch
    for _ in range(max_tokens):
        out = stage.forward(current, cache, logits="last")
        next_tokens = _step_sync(stage, out, cache, batch, guard, sample=True)
        for row, token in enumerate(next_tokens):
            generated[row].append(token)
            finished[row] = finished[row] or token in eos_ids
        if eos_ids and all(finished):
            break
        current = mx.array(next_tokens, dtype=mx.int32)[:, None]
    return generated


# ---------------------------------------------------------------------------
# Loading one stage.
# ---------------------------------------------------------------------------


def _exchange_wire_dtype(model: Model, group: Any) -> mx.Dtype:
    """Rank 0 measures the embedding's output dtype and broadcasts it."""
    code = 0
    if group.rank() == 0:
        probe = model.language_model.model.embed_tokens(mx.array([[0]]))
        code = _MX_DTYPE_CODES[probe.dtype]
    agreed = mx.distributed.all_sum(mx.array([code], dtype=mx.int32), group=group)
    return _MX_DTYPES_BY_CODE[int(agreed.item())]


def _gather_node_budgets(node: NodeMemory, ckpt_total: int, group: Any):
    """Every rank learns every rank's measured budget and checkpoint size."""
    mib = 2**20
    local = mx.array(
        [node.total_bytes // mib, node.budget_bytes // mib, ckpt_total // mib],
        dtype=mx.int32,
    )
    rows = mx.distributed.all_gather(local[None], group=group).tolist()
    sizes = {row[2] for row in rows}
    if len(sizes) != 1:
        raise RuntimeError(
            f"ranks see different checkpoints (text MiB per rank: {[r[2] for r in rows]})"
        )
    return [
        NodeBudget(f"rank{rank}", row[0] * mib, row[1] * mib, "measured free")
        for rank, row in enumerate(rows)
    ]


def load_stage(
    model_dir: Path,
    group: Any,
    *,
    context: int | None,
    batch: int,
    prefill_step: int | None = None,
    starts: list[int] | None = None,
    log=print,
) -> tuple[PipelineStage, PipelinePlan, MemoryGuard]:
    from mlx_lm.utils import load_model

    from ..utils.tokenizer import _register_vendored_archs

    args = load_text_args(model_dir)
    ckpt = read_checkpoint_bytes(model_dir, args.num_hidden_layers)
    node = measure_node_memory()
    nodes = _gather_node_budgets(node, ckpt.text_bytes, group)
    step = prefill_step or default_prefill_step()
    plan = plan_pipeline(
        args,
        ckpt,
        nodes,
        context=context or args.max_position_embeddings,
        batch=batch,
        prefill_step=step,
        starts=starts,
    )
    rank = group.rank()
    if rank == 0:
        log(format_plan(plan))
    require_fit(plan)
    stage_plan = plan.stages[rank]
    limits = apply_memory_guardrails(node, stage_plan.total_bytes)
    log(f"[pipeline] rank {rank} guardrails " + json.dumps(limits))

    _register_vendored_archs()
    model, _ = load_model(model_dir, lazy=True)
    slice_model(model, rank, group.size(), stage_plan.start, stage_plan.end)
    mx.eval(model.parameters())
    wire_dtype = _exchange_wire_dtype(model, group)
    stage = PipelineStage(model, group, stage_plan.start, stage_plan.end, wire_dtype)
    return stage, plan, MemoryGuard(node.budget_bytes, rank)


# ---------------------------------------------------------------------------
# CLI: ``plan`` (dry run, headers only) and ``run`` (under mlx.launch).
# ---------------------------------------------------------------------------


def _parse_node(text: str) -> NodeBudget:
    parts = text.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("--node NAME:RAM_GIB[:FREE_GIB]")
    total = int(float(parts[1]) * 2**30)
    if len(parts) == 3:
        free = int(float(parts[2]) * 2**30)
        memory = NodeMemory(total, free, round(100 * free / total), 1)
        return NodeBudget(parts[0], total, memory.budget_bytes, "free given")
    return NodeBudget(
        parts[0], total, int(total * MEMORY_LIMIT_RATIO), "RAM cap, free not measured"
    )


def _parse_starts(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [0, *(int(item) for item in text.split(","))]


def _cmd_plan(options) -> int:
    model_dir = Path(options.model).expanduser()
    args = load_text_args(model_dir)
    ckpt = read_checkpoint_bytes(model_dir, args.num_hidden_layers)
    nodes = options.node
    plan = plan_pipeline(
        args,
        ckpt,
        nodes,
        context=options.context or args.max_position_embeddings,
        batch=options.batch,
        prefill_step=options.prefill_step or default_prefill_step(),
        starts=_parse_starts(options.split),
    )
    print(format_plan(plan))
    wire = wire_bytes_per_token(args, ckpt.activation_bytes, len(nodes))
    print(
        "wire per token per sequence: "
        f"{wire['stream_per_hop']:,} B stream x {wire['hops']} hop(s) + "
        f"{wire['token_broadcast']} B token = {wire['total_per_token']:,} B "
        "(prefill: the same per prompt token; decode: per generated token)"
    )
    kinds = [args.layer_types[i] for i in range(args.num_hidden_layers)]
    for stage in plan.stages:
        span = kinds[stage.start : stage.end]
        ple = [i for i in args.ple_layer_ids if stage.start < i <= stage.end]
        print(
            f"rank {stage.rank}: {span.count('linear_attention')} GDN + "
            f"{span.count('qwen_sparse_attention')} QSA layers"
            + (f", PLE n-gram layer id(s) {ple}" if ple else "")
        )
    return 0 if _fits(plan.stages) else 2


def _cmd_run(options) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    group = mx.distributed.init(strict=True)
    model_dir = Path(options.model).expanduser()
    if options.prompt_ids:
        prompts = [
            [int(token) for token in row.split(",")]
            for row in options.prompt_ids.split(";")
        ]
        tokenizer = None
    else:
        from mlx_lm.utils import load_tokenizer

        tokenizer = load_tokenizer(model_dir)
        prompts = [tokenizer.encode(text) for text in options.prompt]
    stage, plan, guard = load_stage(
        model_dir,
        group,
        context=options.context,
        batch=len(prompts),
        prefill_step=options.prefill_step,
        starts=_parse_starts(options.split),
        log=lambda line: print(line, flush=True),
    )
    if options.guard_limit_gib is not None:
        guard.budget_bytes = min(
            guard.budget_bytes, int(options.guard_limit_gib * 2**30)
        )
    eos_ids: tuple[int, ...] = ()
    if tokenizer is not None and not options.ignore_eos:
        eos_ids = tuple(tokenizer.eos_token_ids)
    logits = None
    if options.dump:
        logits = pipeline_prefill_logits(stage, prompts, guard=guard)
    tokens = pipeline_generate(
        stage,
        prompts,
        options.max_tokens,
        eos_ids=eos_ids,
        prefill_step=options.prefill_step,
        guard=guard,
    )
    if options.dump and stage.is_last:
        import numpy as np

        np.savez(
            options.dump,
            logits=np.array(logits.astype(mx.float32)),
            tokens=np.array(tokens, dtype=np.int64),
            starts=np.array(plan.starts),
            wire_dtype=str(stage.wire_dtype),
        )
    if stage.is_first:
        for row, generated in enumerate(tokens):
            text = tokenizer.decode(generated) if tokenizer is not None else generated
            print(f"[pipeline] row {row}: {text}", flush=True)
    # mlx.launch tears the other ranks down as soon as one exits; nobody
    # leaves until every rank has finished its own output.
    mx.eval(mx.distributed.all_sum(mx.array([1], dtype=mx.int32), group=group))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rapid_mlx.distributed.pipeline_qwen4"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="dry-run split from index/headers only")
    plan.add_argument("--model", required=True)
    plan.add_argument("--node", type=_parse_node, action="append", required=True)
    plan.add_argument("--context", type=int)
    plan.add_argument("--batch", type=int, default=1)
    plan.add_argument("--prefill-step", type=int)
    plan.add_argument("--split", help="forced starts for ranks 1..N-1, e.g. 20")

    run = commands.add_parser("run", help="run one rank (launch with mlx.launch)")
    run.add_argument("--model", required=True)
    run.add_argument("--prompt", action="append", default=[])
    run.add_argument("--prompt-ids", help="token ids: '1,2,3;4,5' (rows by ';')")
    run.add_argument("--max-tokens", type=int, required=True)
    run.add_argument("--context", type=int)
    run.add_argument("--prefill-step", type=int)
    run.add_argument("--split")
    run.add_argument("--ignore-eos", action="store_true")
    run.add_argument("--dump", help="last rank writes prefill logits + tokens (npz)")
    run.add_argument(
        "--guard-limit-gib",
        type=float,
        help="lower the per-step memory stop threshold (it can never be raised)",
    )

    options = parser.parse_args(argv)
    if options.command == "plan":
        return _cmd_plan(options)
    if not options.prompt and not options.prompt_ids:
        parser.error("run needs --prompt or --prompt-ids")
    return _cmd_run(options)


if __name__ == "__main__":
    sys.exit(main())
