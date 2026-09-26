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

Nothing here is imported by the single-node serving path.  The MTP head
(``mtp.*``) is dropped by the model's own ``sanitize``.  Image input: the vision
tower (``model.visual.*``, :mod:`..models.qwen4_exp_vision`) is loaded on rank
0 only and counted in rank 0's plan; rank 0 merges its features into the token
embeddings before layer 0, and every rank receives the batch's multimodal RoPE
table (plain integers, one all_sum) so the attention layers it owns rotate by
the image's (t, h, w) positions.  Nothing else crosses a rank boundary.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import dataclasses
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

# ratio: the stand-in ceiling for a node whose GPU working-set ceiling was not
# given (a dry ``plan --node NAME:RAM[:FREE]``): the MTPLX run measured a stable
# ceiling at 75% of RAM for MLX allocations on 96-128 GB Apple silicon.  A
# measured node always carries Metal's own ceiling instead (NodeMemory).
MEMORY_LIMIT_RATIO = 0.75
# measured: the share of RAM that stays available (host_statistics64) under a
# rank's full budget = the highest kernel-WARN point measured + the load drift.
# 2026-09-24, goose's compaction (memory_pressure -l warn, incompressible pages)
# reached kernel WARN at 9.3 GiB available on the M4 Max 128 GB (7.3% of RAM)
# and at 3.3-4.0 GiB on the M3 Ultra 96 GB (3.4-4.2%); the ranks' re-measure at
# load reads up to 0.6% of RAM below goose's preflight (goose carries 2%):
# 7.3% + 2% = 9.3%.  The first loosening (7%, from the M3 Ultra alone) sat BELOW
# the M4 Max's WARN point, and a live Flash split planned rank 0 at 96% of it
# served under kernel WARN (7.1 GiB available) — backed off to this.  It
# replaced the 21% floor ("lowest share seen NORMAL", not where WARN starts).
AVAILABLE_MARGIN_RATIO = 0.093
# measured: one decoder layer's forward peaks at 49x the chunk's HC-stream
# bytes above its resident weights — 1.96 GiB (GDN) / 1.94 GiB (QSA) for a
# 2,101-token chunk of the real 4-bit checkpoint, 2026-09-24; the MoE block is
# 1.53-1.59 GiB of it.  The earlier term-by-term guess modeled 0.33 GiB.
LAYER_TRANSIENT_STREAM_MULTIPLE = 49

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
    # Metal's recommended working-set ceiling for this GPU
    # (``max_recommended_working_set_size``): the most MLX may hold on the node.
    ceiling_bytes: int

    @property
    def budget_bytes(self) -> int:
        return node_budget_bytes(
            self.total_bytes, self.available_bytes, self.ceiling_bytes
        )


def node_budget_bytes(
    total_bytes: int, available_bytes: int, ceiling_bytes: int
) -> int:
    """min(available − RAM × AVAILABLE_MARGIN_RATIO, the GPU ceiling), never below 0."""
    return max(
        0,
        min(
            available_bytes - int(total_bytes * AVAILABLE_MARGIN_RATIO),
            ceiling_bytes,
        ),
    )


def gpu_ceiling_bytes() -> int:
    return int(mx.device_info()["max_recommended_working_set_size"])


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
        ceiling_bytes=gpu_ceiling_bytes(),
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


@dataclass(frozen=True)
class VisionCost:
    """What serving images adds to rank 0: the tower and its encode transient.

    The encode runs before the prefill and its buffers are released first
    (``mx.clear_cache``), so the transient is the larger of the two, not the
    sum.
    """

    weight_bytes: int
    workspace_bytes: int


def vision_cost(model_dir: Path, ckpt: CheckpointBytes) -> VisionCost | None:
    """None when the checkpoint declares no vision tower."""
    from ..models.qwen4_exp_vision import (
        checkpoint_config,
        declares_vision,
        vision_workspace_bytes,
    )

    if not declares_vision(checkpoint_config(model_dir)):
        return None
    if not ckpt.excluded_bytes["vision"]:
        raise ValueError(f"{model_dir} declares a vision_config but ships no tower")
    return VisionCost(
        weight_bytes=ckpt.excluded_bytes["vision"],
        workspace_bytes=vision_workspace_bytes(model_dir),
    )


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
    """Peak transient of one prefill chunk on a stage (measured multiple)."""
    tokens = min(prefill_step, context)
    stream_bytes = args.hc_count * args.hidden_size * act
    layer = batch * tokens * stream_bytes * LAYER_TRANSIENT_STREAM_MULTIPLE
    has_attention = any(
        args.layer_types[index] != "linear_attention" for index in layers
    )
    # The QSA dense fallback materializes a boolean selection and an additive
    # mask over the physical KV length for the chunk's queries; it grows with
    # context, which the 2,101-token measurement barely exercised.
    attention = batch * tokens * context * (1 + act) if has_attention else 0
    logits = batch * args.vocab_size * 4 if is_last else 0
    return layer + attention + logits


# ---------------------------------------------------------------------------
# The split.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeBudget:
    name: str
    total_bytes: int
    budget_bytes: int
    source: str
    # The figures the budget was built from, when they were given (plan JSON).
    available_bytes: int | None = None
    ceiling_bytes: int | None = None


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
        # A node already below its available margin has no budget at all: every
        # split that gives it layers is infinitely over, and the plan still
        # prints (and refuses) with the numbers instead of crashing.
        if self.node.budget_bytes <= 0:
            return float("inf")
        return self.total_bytes / self.node.budget_bytes


@dataclass
class PipelinePlan:
    stages: list[StagePlan]
    context: int
    batch: int
    prefill_step: int
    checkpoint: CheckpointBytes
    max_context: int | None = field(default=None)
    args: TextModelArgs | None = field(default=None)
    vision: VisionCost | None = field(default=None)

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
    vision: VisionCost | None = None,
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
    if rank == 0 and vision is not None:
        weights += vision.weight_bytes
        workspace = max(workspace, vision.workspace_bytes)
    return StagePlan(rank, node, start, end, weights, state, workspace)


def _stages_for_starts(
    args, ckpt, nodes, starts, context, batch, prefill_step, vision=None
):
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
            vision,
        )
        for rank, node in enumerate(nodes)
    ]


def _balanced_starts(
    args, ckpt, nodes, context, batch, prefill_step, vision=None
) -> list[int]:
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
            args, ckpt, nodes[0], 0, size, 0, end, context, batch, prefill_step, vision
        ).utilization
    for rank in range(1, size):
        for end in range(rank + 1, layers + 1):
            for start in range(rank, end):
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
                    vision,
                ).utilization
                worst = max(best[rank - 1][start], cost)
                # choice 0 is never a legal start for rank >= 1: it marks
                # "unset", so an all-infinite row (a node below its available
                # margin) still yields a split to print and refuse.
                if worst < best[rank][end] or choice[rank][end] == 0:
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
    vision: VisionCost | None = None,
) -> PipelinePlan:
    if starts is None:
        starts = _balanced_starts(
            args, ckpt, nodes, context, batch, prefill_step, vision
        )
    if len(starts) != len(nodes) or starts[0] != 0 or starts != sorted(set(starts)):
        raise ValueError(f"invalid split starts {starts} for {len(nodes)} ranks")
    if starts[-1] >= args.num_hidden_layers:
        raise ValueError(f"split {starts} leaves the last rank without layers")
    stages = _stages_for_starts(
        args, ckpt, nodes, starts, context, batch, prefill_step, vision
    )
    plan = PipelinePlan(
        stages, context, batch, prefill_step, ckpt, args=args, vision=vision
    )
    # Context ceiling on this split: the largest context every rank fits.
    low, high = 0, args.max_position_embeddings
    while low < high:
        middle = (low + high + 1) // 2
        trial = _stages_for_starts(
            args, ckpt, nodes, starts, middle, batch, min(prefill_step, middle), vision
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
    vision = (
        f"vision tower {_gib(plan.vision.weight_bytes)} + encode "
        f"{_gib(plan.vision.workspace_bytes)} on rank 0"
        if plan.vision is not None
        else f"vision {_gib(ckpt.excluded_bytes['vision'])} not loaded"
    )
    lines = [
        f"checkpoint text bytes {_gib(ckpt.text_bytes)} "
        f"(excluded: mtp {_gib(ckpt.excluded_bytes['mtp'])}; {vision}); "
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


def apply_memory_guardrails(
    node: NodeMemory, planned_bytes: int, attention_scores_bytes: int = 0
) -> dict[str, int]:
    """Bound MLX on this rank at this GPU's own working-set ceiling.

    ``set_memory_limit`` is a guideline in MLX 0.32 (the allocator waits and
    reclaims cache first), so the hard stop is :class:`MemoryGuard`; the wired
    limit is Metal's recommended working set — never more — which is what
    prevents the exo-style wired-memory panic.

    The buffer cache holds what the measured budget leaves beside the plan:
    ``budget − planned − attention_scores_bytes``, where the last term is the
    prefill's dense attention scores this planner's workspace model leaves out
    and the requester charged beside it (goose: ``PipelineAttention::scores_bytes``
    at ``--prefill-step``; 0 when the requester charged none).  Freed buffers
    stay resident until the cache limit, so a limit of the GPU ceiling less the
    plan let the cache grow past the budget: goose Q-127, rank 0 of the Flash
    split, cache 0 → 49.5 GB beside ~60 GB active against a 81.4 GB budget, the
    kernel at WARN and one request refused.
    """
    budget = node.budget_bytes
    charged = planned_bytes + attention_scores_bytes
    if charged > budget:
        raise PipelineDoesNotFitError(
            f"this rank plans {_gib(planned_bytes)} + attention scores "
            f"{_gib(attention_scores_bytes)} against a budget of "
            f"{_gib(budget)} = min(available {_gib(node.available_bytes)} − RAM × "
            f"{AVAILABLE_MARGIN_RATIO:.2f}, GPU ceiling {_gib(node.ceiling_bytes)})"
        )
    memory_limit = node.ceiling_bytes
    wired_limit = node.ceiling_bytes
    cache_limit = budget - charged
    mx.set_memory_limit(memory_limit)
    mx.set_wired_limit(wired_limit)
    mx.set_cache_limit(cache_limit)
    return {
        "memory_limit": memory_limit,
        "wired_limit": wired_limit,
        "cache_limit": cache_limit,
        "budget": budget,
        "attention_scores": attention_scores_bytes,
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


def receive_stream(shape, dtype, src: int, group: Any) -> mx.array:
    """Receive and land the stream BEFORE any GPU work depends on it.

    A lazy recv fused into the first layer's graph makes a Metal command
    buffer wait on the transfer's event; if the upstream rank needs more than
    the macOS GPU watchdog allows (measured: 3 s passes, 6 s fails with
    kIOGPUCommandBufferCallbackErrorTimeout) the downstream rank dies.  That
    killed rank 1 on the real model when rank 0 ran under memory pressure.
    Waiting on the host instead has no such limit.
    """
    hidden = mx.distributed.recv(shape, dtype, src, group=group)
    mx.eval(hidden)
    return hidden


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
        self.vision = None

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

    def forward(
        self,
        inputs: mx.array,
        cache: list[Any],
        logits: str | None,
        *,
        embeddings: mx.array | None = None,
        rope_positions: Any | None = None,
    ):
        """One pipeline stage of ``Qwen4ExpTextModel.__call__``.

        Returns the send handle on non-last ranks; on the last rank the
        logits (``"all"`` positions or the ``"last"`` one), or the stream
        itself when ``logits`` is None (prefill chunks whose logits nobody
        reads).  ``embeddings`` (rank 0) replaces ``embed_tokens(inputs)`` —
        the image-merged rows; ``rope_positions`` (every rank) is the batch's
        :class:`MRopePositions` when any row carries an image.
        """
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        inner = self.model.language_model.model
        batch, length = inputs.shape
        if self.is_first:
            hidden = inner.embed_tokens(inputs) if embeddings is None else embeddings
            hidden = mx.tile(hidden, (1, 1, self.args.hc_count))
        else:
            hidden = receive_stream(
                (batch, length, self.args.hc_count * self.args.hidden_size),
                self.wire_dtype,
                self.rank - 1,
                self.group,
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
                rope_positions=rope_positions,
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


@dataclass
class ImageInput:
    """One row's preprocessed images (rank 0 only): pixels + (t, h, w) grids."""

    pixel_values: Any
    grid_thw: list[list[int]]


def exchange_rope_positions(
    stage: PipelineStage,
    rows: list[tuple[list[list[int]], int]] | None,
    lengths: list[int],
):
    """Rank 0's multimodal RoPE table, identical on every rank; None if text-only.

    Two all_sums: a flag, then (when set) the ``[3, B, W]`` table and the
    per-row deltas as int32.  Every rank calls this once per batch, images or
    not, so the collective order never depends on the request.
    """
    from ..models.qwen4_exp_vision import MRopePositions

    batch, width = len(lengths), max(lengths)
    local = stage.is_first and rows is not None
    flag = mx.array([1 if local else 0], dtype=mx.int32)
    if stage.size > 1:
        flag = mx.distributed.all_sum(flag, group=stage.group)
    if not int(flag.item()):
        return None
    table = [[[0] * width for _ in range(batch)] for _ in range(3)]
    deltas = [0] * batch
    if local:
        for row, (positions, delta) in enumerate(rows):
            if len(positions[0]) != lengths[row]:
                raise ValueError(
                    f"row {row}: RoPE table covers {len(positions[0])} tokens, "
                    f"the row has {lengths[row]}"
                )
            for axis in range(3):
                table[axis][row][: lengths[row]] = positions[axis]
            deltas[row] = delta
    flat = mx.array(
        [value for axis in table for row in axis for value in row] + deltas,
        dtype=mx.int32,
    )
    if stage.size > 1:
        flat = mx.distributed.all_sum(flat, group=stage.group)
    values = flat.tolist()

    def axis_row(axis: int, row: int) -> list[int]:
        start = (axis * batch + row) * width
        return values[start : start + lengths[row]]

    return MRopePositions.from_rows(
        [
            (
                [axis_row(axis, row) for axis in range(3)],
                values[3 * batch * width + row],
            )
            for row in range(batch)
        ]
    )


def prepare_multimodal(
    stage: PipelineStage,
    rows_ids: list[list[int]],
    images: list[ImageInput | None] | None,
):
    """``(embeddings, rope_positions)`` for one left-padded batch.

    ``embeddings`` is rank 0's ``[B, W, hidden]`` with image features merged
    (None on other ranks or for a text-only batch).  Called by every rank.
    """
    from ..models.qwen4_exp_vision import merge_image_features

    lengths = [len(ids) for ids in rows_ids]
    width = max(lengths)
    vision = stage.vision
    has_images = bool(images) and any(item is not None for item in images)
    if has_images and (not stage.is_first or vision is None):
        raise ValueError("image input needs the vision tower on rank 0")
    tables = None
    if has_images:
        tables = [
            vision.rope_positions(ids, item.grid_thw)
            if item is not None
            else ([list(range(len(ids)))] * 3, 0)
            for ids, item in zip(rows_ids, images)
        ]
    rope = exchange_rope_positions(stage, tables, lengths)
    if not has_images:
        return None, rope
    embed = stage.model.language_model.model.embed_tokens
    merged = []
    for ids, item in zip(rows_ids, images):
        pad = width - len(ids)
        tokens = mx.array([[0] * pad + ids], dtype=mx.int32)
        row = embed(tokens)
        if item is not None:
            features = vision.encode(mx.array(item.pixel_values), item.grid_thw)
            real = merge_image_features(
                row[:, pad:], mx.array(ids), features, vision.image_token_id
            )
            row = mx.concatenate([row[:, :pad], real], axis=1) if pad else real
        merged.append(row)
    embeddings = mx.concatenate(merged, axis=0)
    mx.eval(embeddings)
    # The encode transient is planned as the larger of it and the prefill's,
    # not their sum: hand its buffers back before the prefill allocates.
    mx.clear_cache()
    return embeddings, rope


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
    images: list[ImageInput | None] | None = None,
) -> mx.array | None:
    """Full-prompt logits from one un-chunked forward (last rank only)."""
    tokens, padding = _left_pad(prompts, pad_id)
    embeddings, rope = prepare_multimodal(stage, prompts, images)
    cache = stage.make_cache(padding if len(prompts) > 1 else None)
    out = stage.forward(
        tokens, cache, logits="all", embeddings=embeddings, rope_positions=rope
    )
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
    images: list[ImageInput | None] | None = None,
) -> list[list[int]]:
    """Greedy decode, identical token stream on every rank.

    Mirrors mlx-lm's ``generate_step``: all prompt tokens but the last are
    prefilled in chunks with their logits never evaluated; the last prompt
    token is the first decode step.
    """
    step = prefill_step or default_prefill_step()
    tokens, padding = _left_pad(prompts, pad_id)
    batch = len(prompts)
    embeddings, rope = prepare_multimodal(stage, prompts, images)
    cache = stage.make_cache(padding if batch > 1 else None)
    prefix = tokens[:, :-1]
    for offset in range(0, prefix.shape[1], step):
        out = stage.forward(
            prefix[:, offset : offset + step],
            cache,
            logits=None,
            embeddings=None
            if embeddings is None
            else embeddings[:, :-1][:, offset : offset + step],
            rope_positions=rope,
        )
        _step_sync(stage, out, cache, batch, guard, sample=False)
    current = tokens[:, -1:]
    current_embeddings = None if embeddings is None else embeddings[:, -1:]
    generated: list[list[int]] = [[] for _ in range(batch)]
    finished = [False] * batch
    for _ in range(max_tokens):
        out = stage.forward(
            current,
            cache,
            logits="last",
            embeddings=current_embeddings,
            rope_positions=rope,
        )
        current_embeddings = None
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


def _agree_vision_cost(local: VisionCost | None, group: Any) -> VisionCost | None:
    """Rank 0's VisionCost on every rank (MiB-exact ints, like the node budgets)."""
    # int32 would overflow a >2 GiB figure, so the figures travel as KiB, and
    # every rank (rank 0 included) plans with the same rounded-up values.
    kib = [0, 0, 0]
    if group.rank() == 0 and local is not None:
        kib = [1, -(-local.weight_bytes // 1024), -(-local.workspace_bytes // 1024)]
    agreed = mx.array(kib, dtype=mx.int32)
    if group.size() > 1:
        agreed = mx.distributed.all_sum(agreed, group=group)
    present, weight, workspace = agreed.tolist()
    if not present:
        return None
    return VisionCost(weight_bytes=weight * 1024, workspace_bytes=workspace * 1024)


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


def truncate_layers(
    args: TextModelArgs, ckpt: CheckpointBytes, layer_limit: int
) -> tuple[TextModelArgs, CheckpointBytes]:
    """The first ``layer_limit`` decoder layers plus the embedding and head.

    A verification device only: the truncated network is not the model, but
    its single-process and split forwards must agree exactly, which lets real
    weights prove the split on a machine that cannot hold all of them.
    """
    if not 0 < layer_limit <= args.num_hidden_layers:
        raise ValueError(
            f"layer_limit {layer_limit} outside 1..{args.num_hidden_layers}"
        )
    kept = dataclasses.replace(
        args,
        num_hidden_layers=layer_limit,
        layer_types=list(args.layer_types[:layer_limit]),
        ple_layer_ids=[i for i in args.ple_layer_ids if i <= layer_limit],
    )
    return kept, dataclasses.replace(ckpt, layer_bytes=ckpt.layer_bytes[:layer_limit])


def load_stage(
    model_dir: Path,
    group: Any,
    *,
    context: int | None,
    batch: int,
    prefill_step: int | None = None,
    starts: list[int] | None = None,
    layer_limit: int | None = None,
    vision: bool = True,
    attention_scores_bytes: int = 0,
    log=print,
) -> tuple[PipelineStage, PipelinePlan, MemoryGuard]:
    from mlx_lm.utils import load_model

    from ..models.qwen4_exp_vision import load_vision_tower
    from ..utils.tokenizer import _register_vendored_archs

    args = load_text_args(model_dir)
    ckpt = read_checkpoint_bytes(model_dir, args.num_hidden_layers)
    if layer_limit is not None:
        args, ckpt = truncate_layers(args, ckpt, layer_limit)
    # Only rank 0 holds the tower, so only rank 0 needs mlx-vlm: it measures the
    # cost and every rank plans with rank 0's figures.
    local = vision_cost(model_dir, ckpt) if vision and group.rank() == 0 else None
    cost = _agree_vision_cost(local, group) if vision else None
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
        vision=cost,
    )
    rank = group.rank()
    if rank == 0:
        log(format_plan(plan))
    require_fit(plan)
    stage_plan = plan.stages[rank]
    limits = apply_memory_guardrails(
        node, stage_plan.total_bytes, attention_scores_bytes
    )
    log(f"[pipeline] rank {rank} guardrails " + json.dumps(limits))

    _register_vendored_archs()
    model, _ = load_model(model_dir, lazy=True)
    slice_model(model, rank, group.size(), stage_plan.start, stage_plan.end)
    mx.eval(model.parameters())
    wire_dtype = _exchange_wire_dtype(model, group)
    stage = PipelineStage(model, group, stage_plan.start, stage_plan.end, wire_dtype)
    if rank == 0 and cost is not None:
        stage.vision = load_vision_tower(model_dir)
    stage.limits = limits
    return stage, plan, MemoryGuard(node.budget_bytes, rank)


# ---------------------------------------------------------------------------
# CLI: ``plan`` (dry run, headers only) and ``run`` (under mlx.launch).
# ---------------------------------------------------------------------------


def _parse_node(text: str) -> NodeBudget:
    parts = text.split(":")
    if len(parts) not in (2, 3, 4):
        raise argparse.ArgumentTypeError("--node NAME:RAM_GIB[:FREE_GIB[:CEILING_GIB]]")
    total = int(float(parts[1]) * 2**30)
    if len(parts) == 2:
        return NodeBudget(
            parts[0],
            total,
            int(total * MEMORY_LIMIT_RATIO),
            "RAM cap, free not measured",
        )
    free = int(float(parts[2]) * 2**30)
    if len(parts) == 4:
        ceiling, source = int(float(parts[3]) * 2**30), "free + GPU ceiling given"
    else:
        ceiling, source = int(total * MEMORY_LIMIT_RATIO), "free given, ceiling RAM cap"
    return NodeBudget(
        parts[0],
        total,
        node_budget_bytes(total, free, ceiling),
        source,
        available_bytes=free,
        ceiling_bytes=ceiling,
    )


def _parse_starts(text: str | None) -> list[int] | None:
    if not text:
        return None
    return [0, *(int(item) for item in text.split(","))]


def plan_json(plan: PipelinePlan, wire: dict[str, int]) -> dict[str, Any]:
    """The plan as data — what goose's preflight reads (no text parsing)."""
    ckpt = plan.checkpoint
    return {
        "context": plan.context,
        "batch": plan.batch,
        "prefill_step": plan.prefill_step,
        "max_context": plan.max_context,
        "starts": plan.starts,
        "slots": plan.batch,
        "fits": _fits(plan.stages),
        "vision": None
        if plan.vision is None
        else {
            "rank": 0,
            "weight_bytes": plan.vision.weight_bytes,
            "workspace_bytes": plan.vision.workspace_bytes,
        },
        "checkpoint": {
            "text_bytes": ckpt.text_bytes,
            "head_bytes": ckpt.head_bytes,
            "tail_bytes": ckpt.tail_bytes,
            "excluded_bytes": ckpt.excluded_bytes,
            "layer_bytes": ckpt.layer_bytes,
        },
        "ratios": {
            "available_margin": AVAILABLE_MARGIN_RATIO,
            "layer_transient_stream_multiple": LAYER_TRANSIENT_STREAM_MULTIPLE,
        },
        "wire": wire,
        "stages": [
            {
                "rank": stage.rank,
                "node": stage.node.name,
                "layer_start": stage.start,
                "layer_end": stage.end,
                "weight_bytes": stage.weight_bytes,
                "state_bytes": stage.state_bytes,
                "workspace_bytes": stage.workspace_bytes,
                "total_bytes": stage.total_bytes,
                "budget_bytes": stage.node.budget_bytes,
                "available_bytes": stage.node.available_bytes,
                "ceiling_bytes": stage.node.ceiling_bytes,
                "ram_bytes": stage.node.total_bytes,
                "budget_source": stage.node.source,
                "fits": stage.total_bytes <= stage.node.budget_bytes,
            }
            for stage in plan.stages
        ],
    }


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
        vision=None if options.no_vision else vision_cost(model_dir, ckpt),
    )
    wire = wire_bytes_per_token(args, ckpt.activation_bytes, len(nodes))
    if options.json:
        print(json.dumps(plan_json(plan, wire)))
        return 0 if _fits(plan.stages) else 2
    print(format_plan(plan))
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
        vision=not options.no_vision,
        log=lambda line: print(line, flush=True),
    )
    images = None
    if stage.is_first and options.vision_inputs:
        images = _load_vision_inputs(Path(options.vision_inputs), len(prompts))
    elif stage.is_first and options.image:
        prompts[0], first = chat_image_prompt(
            stage.vision, tokenizer, options.prompt[0], options.image
        )
        images = [first] + [None] * (len(prompts) - 1)
    if options.image:
        prompts = _agree_prompts(stage, prompts)
    if options.guard_limit_gib is not None:
        guard.budget_bytes = min(
            guard.budget_bytes, int(options.guard_limit_gib * 2**30)
        )
    eos_ids: tuple[int, ...] = ()
    if tokenizer is not None and not options.ignore_eos:
        eos_ids = tuple(tokenizer.eos_token_ids)
    logits = None
    if options.dump:
        logits = pipeline_prefill_logits(stage, prompts, guard=guard, images=images)
    tokens = pipeline_generate(
        stage,
        prompts,
        options.max_tokens,
        eos_ids=eos_ids,
        prefill_step=options.prefill_step,
        guard=guard,
        images=images,
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


def _load_vision_inputs(path: Path, rows: int) -> list[ImageInput | None]:
    """``pixel_values_<row>`` / ``grid_thw_<row>`` arrays per row (npz)."""
    import numpy as np

    data = np.load(path)
    return [
        ImageInput(data[f"pixel_values_{row}"], data[f"grid_thw_{row}"].tolist())
        if f"pixel_values_{row}" in data
        else None
        for row in range(rows)
    ]


def chat_image_prompt(vision, tokenizer, text: str, image_paths: list[str]):
    """One user turn — the images, then ``text`` — rendered and expanded.

    The same path the server takes: the checkpoint's chat template places one
    ``<|vision_start|><|image_pad|><|vision_end|>`` per image, and mlx-vlm's
    processor resizes each image and expands its pad into the image's tokens.
    """
    from PIL import Image

    from ..utils.chat_template import apply_chat_template

    if vision is None:
        raise ValueError("--image needs a checkpoint with a vision tower")
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in image_paths]
            + [{"type": "text", "text": text}],
        }
    ]
    prompt = apply_chat_template(
        tokenizer, messages, enable_thinking=False, model_name="pipeline"
    )
    images = [Image.open(path).convert("RGB") for path in image_paths]
    out = vision.processor(tokenizer)(text=[prompt], images=images, return_tensors="np")
    return (
        out["input_ids"][0].tolist(),
        ImageInput(out["pixel_values"], out["image_grid_thw"].tolist()),
    )


def _agree_prompts(stage: PipelineStage, prompts: list[list[int]]) -> list[list[int]]:
    """Rank 0's prompt ids on every rank (image expansion happens on rank 0)."""
    if stage.size == 1:
        return prompts
    lengths = mx.distributed.all_sum(
        mx.array(
            [len(p) for p in prompts] if stage.is_first else [0] * len(prompts),
            dtype=mx.int32,
        ),
        group=stage.group,
    ).tolist()
    flat = (
        [token for p in prompts for token in p]
        if stage.is_first
        else [0] * sum(lengths)
    )
    flat = mx.distributed.all_sum(mx.array(flat, dtype=mx.int32), group=stage.group)
    values = flat.tolist()
    agreed, start = [], 0
    for length in lengths:
        agreed.append(values[start : start + length])
        start += length
    return agreed


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
    plan.add_argument("--json", action="store_true", help="machine-readable plan")
    plan.add_argument(
        "--no-vision",
        action="store_true",
        help="plan text only (the vision tower is not loaded on rank 0)",
    )

    from . import pipeline_qwen4_serve

    serve = commands.add_parser("serve", help="OpenAI server; one rank per process")
    pipeline_qwen4_serve.add_arguments(serve)

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
        "--image",
        action="append",
        default=[],
        help="attach an image to the first --prompt (rendered as a chat turn)",
    )
    run.add_argument(
        "--vision-inputs",
        help="npz of pixel_values_<row>/grid_thw_<row> for --prompt-ids rows",
    )
    run.add_argument("--no-vision", action="store_true")
    run.add_argument(
        "--guard-limit-gib",
        type=float,
        help="lower the per-step memory stop threshold (it can never be raised)",
    )

    options = parser.parse_args(argv)
    if options.command == "plan":
        return _cmd_plan(options)
    if options.command == "serve":
        code = pipeline_qwen4_serve.serve(options)
        # Every rank has left the collective loop; do not let interpreter
        # teardown wait on HTTP/executor threads.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    if not options.prompt and not options.prompt_ids:
        parser.error("run needs --prompt or --prompt-ids")
    return _cmd_run(options)


if __name__ == "__main__":
    sys.exit(main())
