"""Qwen3.8 Flash-Next (qwen4_exp) weight offload: PLE n-gram tables and routed experts.

Two independent, default-OFF levers (both driven by environment variables so the
model constructor, the loader's ``sanitize`` and the server all see the same
switch; ``rapid-mlx serve`` sets them from CLI flags):

* ``RAPID_MLX_QWEN4_PLE_OFFLOAD=1`` keeps the 32 GB of PLE n-gram embedding
  shards out of Metal. Rows are gathered on the CPU from read-only memory maps
  of the checkpoint (``~16`` rows of 100 bytes per token), uploaded, and
  dequantized on the GPU with exactly the arithmetic ``QuantizedEmbedding``
  uses, so the result is bit-identical.

* ``RAPID_MLX_QWEN4_EXPERT_BUDGET_GB=<GB>`` keeps only as many routed experts per
  layer resident as fit in a total resident budget (decimal GB, measured with
  ``mx.get_active_memory`` after load). Each MoE layer owns a slot stack
  ``[C, ...]`` that the unmodified ``SwitchGLU``/``gather_qmm`` path computes
  on; a CPU table maps expert id -> slot. Missing experts are ``pread`` from the
  checkpoint straight into the Metal buffer of a victim slot (zero copy, unified
  memory). The router output is synchronised once per layer to know the
  experts; this is the cost of exact on-demand loading. When one call needs
  more distinct experts than there are slots (long prefill chunks), the call is
  computed in groups and every (token, k) row is taken from the group that
  holds its expert, so the routed tensor is the same one the resident model
  would have produced.

Other knobs: ``RAPID_MLX_QWEN4_OFFLOAD_DIR`` (checkpoint dir; required when a
lever is on), ``..._EXPERT_POLICY`` (``lru`` | ``lfu``), ``..._EXPERT_SLOTS``
(explicit slots per layer, overrides the budget), ``..._EXPERT_RESERVE_GB``
(headroom left for KV/activations inside the budget, default 2),
``..._EXPERT_STATS_IN`` (routing counts .npy used to seed the resident set),
``..._EXPERT_STATS_OUT`` (where to write routing counts), ``..._IO_THREADS``
(parallel preads, default 8).
"""

from __future__ import annotations

import json
import logging
import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

ENV_PREFIX = "RAPID_MLX_QWEN4_"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(ENV_PREFIX + name)
    return default if value is None or value == "" else value


@dataclass(frozen=True)
class OffloadConfig:
    ple: bool
    expert_budget_gb: float | None
    expert_slots: int | None
    model_dir: str | None
    policy: str
    reserve_gb: float
    stats_in: str | None
    stats_out: str | None
    io_threads: int

    @property
    def experts(self) -> bool:
        return self.expert_budget_gb is not None or self.expert_slots is not None

    @property
    def any(self) -> bool:
        return self.ple or self.experts


def offload_config() -> OffloadConfig:
    budget = _env("EXPERT_BUDGET_GB")
    slots = _env("EXPERT_SLOTS")
    policy = (_env("EXPERT_POLICY", "lru") or "lru").lower()
    if policy not in ("lru", "lfu"):
        raise ValueError(f"{ENV_PREFIX}EXPERT_POLICY must be lru or lfu, got {policy!r}")
    return OffloadConfig(
        ple=_env("PLE_OFFLOAD", "0") in ("1", "true", "on", "yes"),
        expert_budget_gb=float(budget) if budget is not None else None,
        expert_slots=int(slots) if slots is not None else None,
        model_dir=_env("OFFLOAD_DIR"),
        policy=policy,
        reserve_gb=float(_env("EXPERT_RESERVE_GB", "2") or 2),
        stats_in=_env("EXPERT_STATS_IN"),
        stats_out=_env("EXPERT_STATS_OUT"),
        io_threads=max(1, int(_env("IO_THREADS", "8") or 8)),
    )


# ---------------------------------------------------------------------------
# safetensors location index (headers only, no tensor data is read)


_DTYPES = {"U32": np.uint32, "BF16": np.uint16, "F16": np.float16, "F32": np.float32}


@dataclass(frozen=True)
class TensorLoc:
    path: str
    offset: int  # absolute byte offset of the tensor data in the file
    nbytes: int
    shape: tuple[int, ...]
    dtype: str


class CheckpointIndex:
    def __init__(self, model_dir: str | os.PathLike[str]):
        self.model_dir = Path(model_dir)
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.exists():
            weight_map = json.loads(index_path.read_text())["weight_map"]
            files = sorted(set(weight_map.values()))
        else:
            files = sorted(p.name for p in self.model_dir.glob("*.safetensors"))
        self.locs: dict[str, TensorLoc] = {}
        for name in files:
            path = self.model_dir / name
            with open(path, "rb") as handle:
                (header_len,) = struct.unpack("<Q", handle.read(8))
                header = json.loads(handle.read(header_len))
            base = 8 + header_len
            for key, meta in header.items():
                if key == "__metadata__":
                    continue
                start, end = meta["data_offsets"]
                self.locs[key] = TensorLoc(
                    str(path), base + start, end - start, tuple(meta["shape"]), meta["dtype"]
                )
        self._fds: dict[str, int] = {}
        self._lock = threading.Lock()

    def __getitem__(self, key: str) -> TensorLoc:
        return self.locs[key]

    def fd(self, path: str) -> int:
        with self._lock:
            fd = self._fds.get(path)
            if fd is None:
                fd = os.open(path, os.O_RDONLY)
                self._fds[path] = fd
            return fd

    def memmap(self, key: str) -> np.ndarray:
        loc = self.locs[key]
        return np.memmap(
            loc.path, dtype=_DTYPES[loc.dtype], mode="r", offset=loc.offset, shape=loc.shape
        )


def _read_exact(fd: int, buffers: list[memoryview], offset: int) -> None:
    want = sum(b.nbytes for b in buffers)
    got = os.preadv(fd, buffers, offset)
    if got == want:
        return
    # Short read (rare): finish byte-wise from the flattened view.
    flat = [b.cast("B") for b in buffers]
    done = got
    while done < want:
        # locate buffer/position for `done`
        pos = done
        for buf in flat:
            if pos < buf.nbytes:
                n = os.preadv(fd, [buf[pos:]], offset + done)
                if n <= 0:
                    raise OSError(f"short read at {offset + done}")
                done += n
                break
            pos -= buf.nbytes


# ---------------------------------------------------------------------------
# PLE


PLE_KEY_RE_PREFIX = "model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding.shard_{shard}"


class PLEOffloadStore:
    """CPU-side gather of quantized PLE rows from the checkpoint."""

    def __init__(self, index: CheckpointIndex, layer: int, parts: int, threads: int = 8):
        self.parts = parts
        self.w: list[np.ndarray] = []
        self.s: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for shard in range(parts):
            prefix = PLE_KEY_RE_PREFIX.format(layer=layer, shard=shard)
            self.w.append(index.memmap(prefix + ".weight"))
            self.s.append(index.memmap(prefix + ".scales"))
            self.b.append(index.memmap(prefix + ".biases"))
        self.rows_per_shard = self.w[0].shape[0]
        self.threads = threads
        self._pool = ThreadPoolExecutor(max_workers=threads) if threads > 1 else None
        self.rows_gathered = 0
        self.gather_seconds = 0.0

    def gather(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rows for flat global ids -> (weight u32 [n,W], scales u16 [n,G], biases u16 [n,G])."""
        t0 = time.perf_counter()
        ids = ids.astype(np.int64, copy=False)
        n = ids.shape[0]
        out_w = np.empty((n, self.w[0].shape[1]), dtype=np.uint32)
        out_s = np.empty((n, self.s[0].shape[1]), dtype=np.uint16)
        out_b = np.empty((n, self.b[0].shape[1]), dtype=np.uint16)
        shard = ids // self.rows_per_shard
        local = ids - shard * self.rows_per_shard
        order = np.argsort(shard, kind="stable")
        bounds = np.searchsorted(shard[order], np.arange(self.parts + 1))

        def work(sh: int) -> None:
            sel = order[bounds[sh] : bounds[sh + 1]]
            if sel.size == 0:
                return
            rows = local[sel]
            out_w[sel] = self.w[sh][rows]
            out_s[sel] = self.s[sh][rows]
            out_b[sel] = self.b[sh][rows]

        used = [sh for sh in range(self.parts) if bounds[sh + 1] > bounds[sh]]
        if self._pool is not None and len(used) > 1:
            list(self._pool.map(work, used))
        else:
            for sh in used:
                work(sh)
        self.rows_gathered += n
        self.gather_seconds += time.perf_counter() - t0
        return out_w, out_s, out_b


class OffloadedShardedEmbedding(nn.Module):
    """Drop-in for ``ShardedEmbedding`` whose tables stay on disk."""

    def __init__(self, rows_per_shard: int, dims: int, parts: int, *, group_size: int, bits: int):
        super().__init__()
        self.rows_per_shard = rows_per_shard
        self.dims = dims
        self.parts = parts
        self.group_size = group_size
        self.bits = bits
        self.store: PLEOffloadStore | None = None  # attached lazily (needs the checkpoint dir)
        self.ple_layer = 1

    def attach(self, store: PLEOffloadStore) -> None:
        if store.rows_per_shard != self.rows_per_shard:
            raise ValueError("PLE shard geometry mismatch")
        self.store = store

    def __call__(self, indices: mx.array) -> mx.array:
        if self.store is None:
            self.attach(_shared_ple_store(self.ple_layer, self.parts))
        assert self.store is not None
        ids = np.asarray(indices.astype(mx.int64)).reshape(-1)
        uniq, inverse = np.unique(ids, return_inverse=True)
        w, s, b = self.store.gather(uniq)
        rows = mx.dequantize(
            mx.array(w),
            mx.array(s).view(mx.bfloat16),
            mx.array(b).view(mx.bfloat16),
            group_size=self.group_size,
            bits=self.bits,
        )
        out = mx.take(rows, mx.array(inverse.reshape(-1).astype(np.int32)), axis=0)
        return out.reshape(*indices.shape, rows.shape[-1])


_INDEX: CheckpointIndex | None = None
_PLE_STORE: PLEOffloadStore | None = None


def _checkpoint_index() -> CheckpointIndex:
    global _INDEX
    if _INDEX is None:
        cfg = offload_config()
        if not cfg.model_dir:
            raise RuntimeError(
                f"{ENV_PREFIX}OFFLOAD_DIR must name the local checkpoint directory "
                "when PLE or expert offload is enabled"
            )
        _INDEX = CheckpointIndex(cfg.model_dir)
    return _INDEX


def _shared_ple_store(layer: int, parts: int) -> PLEOffloadStore:
    global _PLE_STORE
    if _PLE_STORE is None:
        _PLE_STORE = PLEOffloadStore(
            _checkpoint_index(), layer, parts, threads=offload_config().io_threads
        )
    return _PLE_STORE


# ---------------------------------------------------------------------------
# routed experts

# leaf in the checkpoint -> list of (module attr path, row split) destinations.
# gate_up_proj rows [0, I) are gate_proj, [I, 2I) up_proj (see sanitize()).
_EXPERT_LEAVES = (
    ("gate_up_proj", "weight"),
    ("gate_up_proj.scales", "scales"),
    ("gate_up_proj.biases", "biases"),
    ("down_proj", "weight"),
    ("down_proj.scales", "scales"),
    ("down_proj.biases", "biases"),
)


class LayerSlots:
    """Resident slot stack for one MoE layer's routed experts.

    Works on the fused layout (``switch_mlp.gate_up_proj``, the checkpoint's own
    per-expert ``[2I, ...]`` block, one read per leaf) and on the split
    ``gate_proj``/``up_proj`` layout (one ``preadv`` into two buffers).
    """

    def __init__(
        self,
        layer: int,
        num_experts: int,
        slots: int,
        switch_mlp: Any,
        index: CheckpointIndex,
        prefix: str,
    ):
        self.layer = layer
        self.num_experts = num_experts
        self.slots = slots
        self.slot_of = np.full(num_experts, -1, dtype=np.int64)
        self.expert_in = np.full(slots, -1, dtype=np.int64)
        self.last_use = np.zeros(slots, dtype=np.int64)
        fused = getattr(switch_mlp, "gate_up_proj", None) is not None
        self.fused = fused
        # (fd, base offset, bytes per expert, [numpy views of slot stacks])
        self.reads: list[tuple[int, int, int, list[np.ndarray]]] = []
        self.expert_bytes = 0
        keep: list[mx.array] = []
        assign: list[tuple[Any, str, mx.array]] = []
        for leaf, attr in _EXPERT_LEAVES:
            loc = index[f"{prefix}.{leaf}"]
            if loc.shape[0] != num_experts:
                raise ValueError(f"{prefix}.{leaf}: unexpected expert count {loc.shape}")
            per_expert = loc.nbytes // num_experts
            self.expert_bytes += per_expert
            dtype = mx.uint32 if loc.dtype == "U32" else mx.uint16
            if leaf.startswith("gate_up_proj") and not fused:
                half = loc.shape[1] // 2
                tail = (half, *loc.shape[2:])
                stacks = [mx.zeros((slots, *tail), dtype=dtype) for _ in range(2)]
                owners = [switch_mlp.gate_proj, switch_mlp.up_proj]
            elif leaf.startswith("gate_up_proj"):
                stacks = [mx.zeros((slots, *loc.shape[1:]), dtype=dtype)]
                owners = [switch_mlp.gate_up_proj]
            else:
                stacks = [mx.zeros((slots, *loc.shape[1:]), dtype=dtype)]
                owners = [switch_mlp.down_proj]
            mx.eval(stacks)
            views = [np.asarray(memoryview(st)) for st in stacks]
            self.reads.append((index.fd(loc.path), loc.offset, per_expert, views))
            for owner, st in zip(owners, stacks):
                value = st.view(mx.bfloat16) if dtype == mx.uint16 else st
                assign.append((owner, attr, value))
                keep.append(value)
        mx.eval(keep)
        for owner, attr, value in assign:
            current = owner[attr]
            if tuple(current.shape[1:]) != tuple(value.shape[1:]) or current.dtype != value.dtype:
                raise ValueError(
                    f"layer {layer} {attr}: slot stack {value.shape}/{value.dtype} does not "
                    f"match module {current.shape}/{current.dtype}"
                )
            setattr(owner, attr, value)
        self.resident_bytes = self.expert_bytes * slots

    def load(self, expert: int, slot: int) -> None:
        for fd, base, per_expert, views in self.reads:
            bufs = [memoryview(v[slot]).cast("B") for v in views]
            _read_exact(fd, bufs, base + expert * per_expert)


class ExpertOffloadManager:
    """Owns all layers' slot stacks, the eviction policy and the routing stats."""

    def __init__(self, num_layers: int, num_experts: int, cfg: OffloadConfig):
        self.cfg = cfg
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.layers: dict[int, LayerSlots] = {}
        self.modules: dict[int, Any] = {}
        self.ready = False
        self.tick = 0
        self.hits = 0
        self.misses = 0
        self.grouped_calls = 0
        self.load_seconds = 0.0
        self.counts = np.zeros((num_layers, num_experts), dtype=np.int64)
        self.freq = np.zeros((num_layers, num_experts), dtype=np.float64)
        self._pool = ThreadPoolExecutor(max_workers=cfg.io_threads)
        self._calls_since_dump = 0
        self.slots = 0

    def register(self, layer: int, moe_block: Any) -> None:
        self.modules[layer] = moe_block

    # -- setup -------------------------------------------------------------
    def _decide_slots(self, expert_bytes: int) -> int:
        if self.cfg.expert_slots is not None:
            return max(1, min(self.num_experts, self.cfg.expert_slots))
        active = mx.get_active_memory()
        budget = float(self.cfg.expert_budget_gb or 0) * 1e9
        avail = budget - active - self.cfg.reserve_gb * 1e9
        slots = int(avail // (expert_bytes * self.num_layers))
        if slots < 32:
            raise RuntimeError(
                f"expert budget {self.cfg.expert_budget_gb} GB leaves {slots} slots/layer "
                f"(resident before experts {active / 1e9:.2f} GB, reserve {self.cfg.reserve_gb} GB)"
            )
        return min(self.num_experts, slots)

    def setup(self) -> None:
        if self.ready:
            return
        t0 = time.perf_counter()
        index = _checkpoint_index()
        prefix = "model.language_model.layers.{}.mlp.experts"
        first = prefix.format(min(self.modules))
        expert_bytes = sum(index[f"{first}.{leaf}"].nbytes for leaf, _ in _EXPERT_LEAVES) // self.num_experts
        self.slots = self._decide_slots(expert_bytes)
        seed = None
        if self.cfg.stats_in and Path(self.cfg.stats_in).exists():
            seed = np.load(self.cfg.stats_in)
            self.freq[:] = seed
        for layer, block in sorted(self.modules.items()):
            ls = LayerSlots(layer, self.num_experts, self.slots, block.switch_mlp, index, prefix.format(layer))
            if seed is not None:
                order = np.argsort(-seed[layer], kind="stable")[: self.slots]
            else:
                order = np.arange(self.slots)
            jobs = []
            for slot, expert in enumerate(order):
                ls.slot_of[expert] = slot
                ls.expert_in[slot] = expert
                jobs.append(self._pool.submit(ls.load, int(expert), slot))
            for job in jobs:
                job.result()
            self.layers[layer] = ls
        self.ready = True
        logger.info(
            "[qwen4-offload] experts: %d/%d slots per layer x %d layers = %.2f GB resident, "
            "policy=%s seed=%s, filled in %.1fs; active now %.2f GB",
            self.slots,
            self.num_experts,
            len(self.layers),
            sum(ls.resident_bytes for ls in self.layers.values()) / 1e9,
            self.cfg.policy,
            bool(seed is not None),
            time.perf_counter() - t0,
            mx.get_active_memory() / 1e9,
        )

    # -- mapping -----------------------------------------------------------
    def _victims(self, ls: LayerSlots, count: int, protected: np.ndarray) -> np.ndarray:
        free = np.flatnonzero(ls.expert_in < 0)
        if free.size >= count:
            return free[:count]
        candidates = np.ones(ls.slots, dtype=bool)
        candidates[protected] = False
        cand = np.flatnonzero(candidates)
        if self.cfg.policy == "lfu":
            key = self.freq[ls.layer][ls.expert_in[cand]]
            key = key + 1e-9 * ls.last_use[cand]  # ties -> older first
        else:
            key = ls.last_use[cand]
        pick = cand[np.argsort(key, kind="stable")[:count]]
        return pick

    def ensure(self, layer: int, experts: np.ndarray) -> None:
        """Make every expert in ``experts`` resident (len(experts) <= slots)."""
        ls = self.layers[layer]
        slots = ls.slot_of[experts]
        missing = experts[slots < 0]
        self.hits += int(experts.size - missing.size)
        self.misses += int(missing.size)
        if missing.size:
            t0 = time.perf_counter()
            protected = slots[slots >= 0]
            victims = self._victims(ls, int(missing.size), protected)
            for expert, slot in zip(missing, victims):
                old = ls.expert_in[slot]
                if old >= 0:
                    ls.slot_of[old] = -1
                ls.expert_in[slot] = expert
                ls.slot_of[expert] = slot
            jobs = [self._pool.submit(ls.load, int(e), int(s)) for e, s in zip(missing, victims)]
            for job in jobs:
                job.result()
            self.load_seconds += time.perf_counter() - t0
        self.tick += 1
        ls.last_use[ls.slot_of[experts]] = self.tick

    def record(self, layer: int, flat: np.ndarray) -> None:
        np.add.at(self.counts[layer], flat, 1)
        if self.cfg.policy == "lfu":
            self.freq[layer] *= 0.999
            np.add.at(self.freq[layer], flat, 1.0)
        if self.cfg.stats_out:
            self._calls_since_dump += 1
            if self._calls_since_dump >= 4096:
                self.dump_stats()

    def dump_stats(self) -> None:
        if self.cfg.stats_out:
            np.save(self.cfg.stats_out, self.counts)
            self._calls_since_dump = 0

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "slots_per_layer": self.slots,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else None,
            "grouped_calls": self.grouped_calls,
            "load_seconds": round(self.load_seconds, 3),
            "resident_expert_gb": round(
                sum(ls.resident_bytes for ls in self.layers.values()) / 1e9, 3
            ),
        }

    def reset_counters(self) -> None:
        self.hits = self.misses = self.grouped_calls = 0
        self.load_seconds = 0.0

    # -- forward -----------------------------------------------------------
    def routed(self, block: Any, layer: int, x: mx.array, indices: mx.array, target_verify: bool) -> mx.array:
        """Routed expert outputs ``[..., k, D]`` for ``indices`` (expert ids)."""
        if not self.ready:
            self.setup()
        ls = self.layers[layer]
        idx = np.asarray(indices).astype(np.int64)
        flat = idx.reshape(-1)
        self.record(layer, flat)
        need = np.unique(flat)
        if need.size <= ls.slots:
            self.ensure(layer, need)
            slot_idx = mx.array(ls.slot_of[idx].astype(np.uint32))
            return block._routed(x, slot_idx, target_verify=target_verify)
        # More distinct experts than slots: compute in groups, each (token, k)
        # row taken from the group that holds its expert.
        self.grouped_calls += 1
        resident = need[ls.slot_of[need] >= 0]
        rest = need[ls.slot_of[need] < 0]
        ordered = np.concatenate([resident, rest])
        out = None
        for start in range(0, ordered.size, ls.slots):
            group = ordered[start : start + ls.slots]
            self.ensure(layer, group)
            in_group = np.isin(idx, group)
            fallback = ls.slot_of[group[0]]
            slots = np.where(in_group, ls.slot_of[np.where(in_group, idx, group[0])], fallback)
            part = block._routed(x, mx.array(slots.astype(np.uint32)), target_verify=target_verify)
            mask = mx.array(in_group)[..., None]
            out = part if out is None else mx.where(mask, part, out)
            mx.eval(out)  # slots are reused by the next group
        assert out is not None
        return out


_MANAGER: ExpertOffloadManager | None = None


def expert_manager() -> ExpertOffloadManager | None:
    return _MANAGER


def install_into_text_model(text_model: Any, args: Any) -> None:
    """Called from ``Qwen4ExpTextModel.__init__``: swap in offloaded modules."""
    global _MANAGER
    cfg = offload_config()
    if not cfg.any:
        return
    from mlx_lm.models.switch_layers import SwitchGLU

    if cfg.experts:
        _MANAGER = ExpertOffloadManager(len(text_model.layers), args.num_experts, cfg)
        for i, layer in enumerate(text_model.layers):
            # One placeholder expert: the loader's sanitize() hands in [1, ...]
            # zeros, the real slot stacks are allocated on first use.
            layer.mlp.switch_mlp = SwitchGLU(args.hidden_size, args.moe_intermediate_size, 1)
            layer.mlp._offload_layer = i
            layer.mlp._offload = _MANAGER
            _MANAGER.register(i, layer.mlp)
    if cfg.ple:
        for i, layer in enumerate(text_model.layers):
            if getattr(layer, "ple", None) is None:
                continue
            ngram = layer.ple.ple_embedding
            old = ngram.ngram_embedding
            dims = old.shards[0].weight.shape[-1]
            emb = OffloadedShardedEmbedding(
                old.rows_per_shard, dims, len(old.shards), group_size=32, bits=4
            )
            emb.ple_layer = i
            ngram.ngram_embedding = emb
    logger.info("[qwen4-offload] installed: %s", cfg)


def sanitize_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Drop / stub offloaded tensors before ``load_weights`` (keys pre-sanitize)."""
    cfg = offload_config()
    if not cfg.any:
        return weights
    out = {}
    for key, value in weights.items():
        if cfg.ple and ".ple.ple_embedding.ngram_embedding.shard_" in key:
            continue
        if (
            cfg.experts
            and key.startswith("language_model.model.layers.")
            and (".mlp.experts.gate_up_proj" in key or ".mlp.experts.down_proj" in key)
        ):
            value = mx.zeros((1, *value.shape[1:]), dtype=value.dtype)
        out[key] = value
    return out


def status() -> dict[str, Any] | None:
    """Counters for /v1/status (None when both levers are off)."""
    cfg = offload_config()
    if not cfg.any:
        return None
    out: dict[str, Any] = {"ple": cfg.ple, "experts": cfg.experts, "policy": cfg.policy}
    if _MANAGER is not None:
        out["expert"] = _MANAGER.stats()
    if _PLE_STORE is not None:
        out["ple_rows_gathered"] = _PLE_STORE.rows_gathered
        out["ple_gather_seconds"] = round(_PLE_STORE.gather_seconds, 3)
    return out
