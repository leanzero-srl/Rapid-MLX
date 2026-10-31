"""Per-model Metal memory budgeting (#2858).

The engine historically exposed one global knob — ``--gpu-memory-utilization``,
default ``0.90`` — that had to be tuned per model by hand: a 12B checkpoint
runs fine at ``0.75`` while a 20B MoE with an ~11.2 GB Metal footprint needs
``0.95`` on the same 16 GB Mac.  Worse, when the cap was too low the model
still loaded and reported healthy while the D-METAL-CAP admission gate
deterministically rejected every request with HTTP 503.

This module closes that loop with two pieces, both pure and unit-testable:

* :func:`plan_metal_limit` — resolve the effective utilization for one loaded
  model.  When the operator passed an explicit ``--gpu-memory-utilization``
  the value is honored verbatim (advanced override).  In auto mode (the new
  default) the limit is sized to the model actually loaded: the MEASURED
  weight footprint (``mx.get_active_memory()`` right after load — no disk
  heuristics) plus a runtime headroom, clamped to
  ``[AUTO_UTILIZATION_FLOOR, AUTO_UTILIZATION_CEILING]`` of the device's
  recommended working-set budget.  The floor keeps small models byte-identical
  to the historical ``0.90`` default; the ceiling leaves the OS a margin the
  same way vLLM's ``gpu_memory_utilization`` never goes to 1.0.

* :func:`format_preflight_error` — the actionable admission-impossible
  message required by #2858: required vs available memory plus concrete
  remediations.  Raised (wrapped in :class:`MetalPreflightError`) by
  ``Scheduler.preflight_metal_admission`` when even a modest request could
  never be admitted under the resolved cap, so startup fails BEFORE the
  server reports the model ready instead of serving deterministic 503s.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

# Auto-mode bounds. The floor matches the historical global default so any
# model that fit comfortably before resolves to a byte-identical limit; the
# ceiling reserves ~3% of the device budget for allocator slack so auto mode
# never plans right up to the working-set edge.
AUTO_UTILIZATION_FLOOR = 0.90
AUTO_UTILIZATION_CEILING = 0.97

# The knob's absolute maximum: an explicit --gpu-memory-utilization may go
# all the way to 1.0, past the auto ceiling. Advice to raise the knob is
# only impossible (and therefore suppressed) at or beyond this value; a
# tiny slack absorbs float representation of "1.0".
MAX_UTILIZATION = 1.0 - 1e-9

# Runtime headroom charged on top of the measured weight footprint in auto
# mode: KV cache for in-flight requests, activation workspace, Metal heap
# fragmentation. Fractional so big models reserve proportionally more, with a
# floor so tiny models still get a workable slice.
_AUTO_HEADROOM_FRACTION = 0.08
# ratio: of the device budget -- 512 MiB (the old absolute floor) on a 64 GB
# Mac's 51.5 GB working set; it binds only where 8% of the weights is less.
_AUTO_HEADROOM_MIN_FRACTION = 0.01


# ── Process-wide utilization ratchet (codex round 2 BLOCKING #2) ──
# In multi-model mode each BatchedEngine resolves its own budget, but the
# Metal allocator and ``mx.get_active_memory()`` are process-wide: when a
# second model auto-raises the limit, the FIRST model's scheduler must not
# keep enforcing its earlier, lower cap against an ``active`` figure that
# now includes the new model — that would reject every request to the
# already-resident model. Every resolution notes the utilization here;
# schedulers read the floor (with a generation counter so their cached cap
# invalidates on each upward ratchet) and enforce
# ``max(own configured utilization, process floor)``. The floor only ever
# rises, mirroring the allocation limit itself.
_process_floor_lock = threading.Lock()
_process_utilization_floor: float = 0.0
_process_floor_generation: int = 0


def note_resolved_utilization(utilization: float) -> None:
    """Record a resolved per-engine utilization into the process floor."""
    global _process_utilization_floor, _process_floor_generation
    with _process_floor_lock:
        if utilization > _process_utilization_floor:
            _process_utilization_floor = utilization
            _process_floor_generation += 1


def process_utilization_floor() -> tuple[float, int]:
    """Return ``(floor, generation)`` for cache-invalidating readers."""
    with _process_floor_lock:
        return _process_utilization_floor, _process_floor_generation


def ratchet_utilization_and_apply(
    utilization: float,
    apply: Callable[[float], None] | None = None,
) -> tuple[float, int]:
    """Ratchet the floor and apply the effective value atomically.

    Concurrent model loads each publish a resolved utilization and then
    install the corresponding Metal allocation limit. Doing those as two
    separate lock acquisitions leaves a window (codex round 4 BLOCKING #1)
    where a loader holding an older, lower floor applies its limit LAST —
    schedulers would then enforce the newer higher cap while Metal holds
    the stale lower allocation limit. Holding the lock across both the
    ratchet and the ``apply`` callback serializes the setter calls in
    floor order, so the last limit installed always reflects the highest
    published utilization.

    ``apply`` receives the effective (post-ratchet) utilization —
    ``max(utilization, floor)`` — and must contain its own failures if a
    setter error should not propagate (the callers do; see codex round 1
    BLOCKING #2). Returns ``(effective_utilization, generation)``.
    """
    global _process_utilization_floor, _process_floor_generation
    with _process_floor_lock:
        if utilization > _process_utilization_floor:
            _process_utilization_floor = utilization
            _process_floor_generation += 1
        effective = max(utilization, _process_utilization_floor)
        if apply is not None:
            apply(effective)
        return effective, _process_floor_generation


# ── Operator tunables (LeanZero fork, v0.14.3-lz.20) ──
# Every number that decides how much memory the engine may take is a tunable:
# a ``rapid-mlx serve`` flag that the CLI exports as the env var below (so a
# worker process sees the same value), read here on each call. Unset = the
# historical constant above, so an engine started without the flags resolves
# byte-identically to lz.19.
ENV_METAL_LIMIT_GB = "RAPID_MLX_METAL_LIMIT_GB"
ENV_AUTO_FLOOR = "RAPID_MLX_AUTO_UTILIZATION_FLOOR"
ENV_AUTO_CEILING = "RAPID_MLX_AUTO_UTILIZATION_CEILING"
ENV_AUTO_HEADROOM = "RAPID_MLX_AUTO_HEADROOM_FRACTION"
ENV_MEMORY_FLOOR_GB = "RAPID_MLX_MEMORY_FLOOR_GB"
ENV_RESERVE_FRACTION = "RAPID_MLX_SYSTEM_RESERVE_FRACTION"
ENV_MAX_MODEL_LEN = "RAPID_MLX_MAX_MODEL_LEN"
ENV_GUARD_MARGIN_GB = "RAPID_MLX_MEMORY_GUARD_MARGIN_GB"
ENV_FLOOR_WAIT_S = "RAPID_MLX_MEMORY_FLOOR_WAIT_S"

# How long a prefill step may wait for other apps to give memory back before
# the free-memory floor refuses it (the floor never lets a step through).
DEFAULT_FLOOR_WAIT_S = 60.0


def _env_float(name: str) -> float | None:
    import os

    raw = os.environ.get(name, "").strip()
    if not raw or raw.lower() == "auto":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


@dataclass(frozen=True)
class MemoryTunables:
    """The operator's memory tunables in force (``None`` = not set)."""

    metal_limit_bytes: int | None
    auto_floor: float
    auto_ceiling: float
    auto_headroom_fraction: float
    memory_floor_bytes: int
    reserve_fraction: float
    max_model_len: int | None
    guard_margin_bytes: int | None
    floor_wait_s: float

    def snapshot(self) -> dict:
        return {
            "metal_limit_gb": (
                None if self.metal_limit_bytes is None else self.metal_limit_bytes / 1e9
            ),
            "auto_utilization_floor": self.auto_floor,
            "auto_utilization_ceiling": self.auto_ceiling,
            "auto_headroom_fraction": self.auto_headroom_fraction,
            "memory_floor_gb": self.memory_floor_bytes / 1e9,
            "system_reserve_fraction": self.reserve_fraction,
            "max_model_len": self.max_model_len,
            "memory_guard_margin_gb": (
                None if self.guard_margin_bytes is None else self.guard_margin_bytes / 1e9
            ),
            "memory_floor_wait_s": self.floor_wait_s,
        }


def tunables() -> MemoryTunables:
    """The tunables in force now (env, set from the serve flags)."""
    limit = _env_float(ENV_METAL_LIMIT_GB)
    floor = _env_float(ENV_AUTO_FLOOR)
    ceiling = _env_float(ENV_AUTO_CEILING)
    headroom = _env_float(ENV_AUTO_HEADROOM)
    mem_floor = _env_float(ENV_MEMORY_FLOOR_GB)
    reserve = _env_float(ENV_RESERVE_FRACTION)
    mml = _env_float(ENV_MAX_MODEL_LEN)
    margin = _env_float(ENV_GUARD_MARGIN_GB)
    wait = _env_float(ENV_FLOOR_WAIT_S)
    return MemoryTunables(
        metal_limit_bytes=int(limit * 1e9) if limit and limit > 0 else None,
        auto_floor=AUTO_UTILIZATION_FLOOR if floor is None else floor,
        auto_ceiling=AUTO_UTILIZATION_CEILING if ceiling is None else ceiling,
        auto_headroom_fraction=(
            _AUTO_HEADROOM_FRACTION if headroom is None else max(0.0, headroom)
        ),
        memory_floor_bytes=int(max(0.0, mem_floor or 0.0) * 1e9),
        reserve_fraction=(
            _SYSTEM_RESERVE_RATIO_DEFAULT if reserve is None else max(0.0, reserve)
        ),
        max_model_len=int(mml) if mml and mml > 0 else None,
        guard_margin_bytes=int(margin * 1e9) if margin is not None and margin >= 0 else None,
        floor_wait_s=DEFAULT_FLOOR_WAIT_S if wait is None else max(0.0, wait),
    )


def validate_tunables(
    t: MemoryTunables, *, physical_bytes: int, utilization_set: bool = False
) -> str | None:
    """One line naming what is impossible, or ``None`` when the set is valid."""
    if t.metal_limit_bytes is not None and utilization_set:
        return (
            "--metal-limit-gb and --gpu-memory-utilization both set: choose one "
            "(an absolute cap in GB, or a fraction of the GPU working set)"
        )
    if not (0.1 <= t.auto_floor <= 1.0):
        return f"--auto-utilization-floor {t.auto_floor:g} must be in 0.1..1.0"
    if not (0.1 <= t.auto_ceiling <= 1.0):
        return f"--auto-utilization-ceiling {t.auto_ceiling:g} must be in 0.1..1.0"
    if t.auto_floor > t.auto_ceiling:
        return (
            f"--auto-utilization-floor {t.auto_floor:g} is above "
            f"--auto-utilization-ceiling {t.auto_ceiling:g}"
        )
    if not (0.0 <= t.reserve_fraction <= 0.5):
        return f"--system-reserve-fraction {t.reserve_fraction:g} must be in 0..0.5"
    if physical_bytes > 0:
        if t.memory_floor_bytes >= physical_bytes:
            return (
                f"--memory-floor-gb {t.memory_floor_bytes / 1e9:.1f} is not below "
                f"this Mac's {physical_bytes / 1e9:.1f} GB of RAM"
            )
        if (
            t.metal_limit_bytes is not None
            and t.metal_limit_bytes + t.memory_floor_bytes > physical_bytes
        ):
            return (
                f"--metal-limit-gb {t.metal_limit_bytes / 1e9:.1f} + --memory-floor-gb "
                f"{t.memory_floor_bytes / 1e9:.1f} = "
                f"{(t.metal_limit_bytes + t.memory_floor_bytes) / 1e9:.1f} GB is more "
                f"than this Mac's {physical_bytes / 1e9:.1f} GB of RAM"
            )
    return None


def floor_room_bytes(*, physical: int, others: int | None, floor: int) -> int | None:
    """What the free-memory floor leaves the engine (its whole footprint):
    ``physical - others - floor``; ``None`` when the floor is off or the
    machine is unmeasured."""
    if floor <= 0 or physical <= 0 or others is None:
        return None
    return max(0, int(physical) - int(others) - int(floor))


# The others' reference (the free-memory floor's input, ``Scheduler.
# _memguard_others_bytes``): it RISES only to a reading held for this long
# (the lowest reading since the newest sample at least this old), and FALLS
# to the highest such sustained reading of the window, only after a window
# with no swap-out. lz.20 rose at once to any single reading: on the mule's
# growing-session replay (2026-10-31, Qwen3.8-Flash-Next, turn 110) one
# 40.95 GB reading -- the idle machine is 19.6 GB -- dropped the ceiling from
# 111.99 to 89.49 GB for ~2 minutes and the session went cold at 167k tokens
# (lz.19: 371k). Another app really holding 30 GB for 30 s still lowers it.
OTHERS_SUSTAIN_S = 15.0
OTHERS_WINDOW_S = 120.0


class OthersReference:
    """Hysteresis over the machine readings, per part: ``rest`` (memory in
    use less this engine's whole process -- max(phys_footprint, MLX active +
    cache) -- i.e. wired + anonymous of OTHER processes + the compressor)
    and ``engine_extra`` (the engine's own memory outside MLX's tally:
    phys_footprint - active - cache, Python objects and driver buffers the
    Metal plan does not count). The reference is the sum of both filtered
    parts: what the engine's Metal plan must leave room for.

    A part RISES only to a SUSTAINED reading: the minimum over every sample
    since the newest one at least ``sustain_s`` old (none that old yet: no
    rise) -- one spike, or a burst shorter than ``sustain_s``, never lowers
    the ceiling. Exception: when the machine swapped out within that span the
    pressure is real and the part rises to the current reading at once (the
    engine must not read others' swapped-out pages as room). A part FALLS
    only when nothing was swapped out over a window spanning at least half
    of ``window_s``, and then only to the highest sustained reading of that
    window."""

    __slots__ = ("sustain_s", "window_s", "samples", "rest", "extra", "last")

    def __init__(self, sustain_s: float = OTHERS_SUSTAIN_S, window_s: float = OTHERS_WINDOW_S):
        from collections import deque

        self.sustain_s = float(sustain_s)
        self.window_s = float(window_s)
        # (t, rest, extra, swapouts, sustained_rest, sustained_extra)
        self.samples: deque = deque()
        self.rest: int | None = None
        self.extra: int | None = None
        self.last: dict | None = None  # the newest sample's parts (status)

    @property
    def value(self) -> int | None:
        if self.rest is None:
            return None
        return int(self.rest) + int(self.extra or 0)

    def _sustained(self, now: float, index: int) -> int | None:
        anchor = None
        for i in range(len(self.samples) - 1, -1, -1):
            if self.samples[i][0] <= now - self.sustain_s:
                anchor = i
                break
        if anchor is None:
            return None
        return min(s[index] for s in list(self.samples)[anchor:])

    def update(self, now: float, rest: int, extra: int, swapouts: int) -> int:
        rest, extra = max(0, int(rest)), max(0, int(extra))
        self.samples.append((now, rest, extra, int(swapouts), None, None))
        keep = max(self.window_s, self.sustain_s) + 1.0
        # keep one sample older than the sustain span (its anchor)
        while len(self.samples) > 2 and now - self.samples[1][0] > keep:
            self.samples.popleft()
        s_rest = self._sustained(now, 1)
        s_extra = self._sustained(now, 2)
        t, r, e, so, _a, _b = self.samples[-1]
        self.samples[-1] = (t, r, e, so, s_rest, s_extra)
        if self.rest is None:  # the first reading: nothing to filter against
            self.rest, self.extra = rest, extra
        else:
            swapping = int(swapouts) > self._anchor_swapouts(now)
            window = [x for x in self.samples if now - x[0] <= self.window_s]
            quiet = (
                now - window[0][0] >= 0.5 * self.window_s
                and window[-1][3] <= window[0][3]
            )
            self.rest = self._step(self.rest, rest, s_rest, swapping, quiet, window, 4)
            self.extra = self._step(self.extra, extra, s_extra, swapping, quiet, window, 5)
        self.last = {
            "others_instant_bytes": rest,
            "engine_extra_instant_bytes": extra,
            "others_sustained_bytes": s_rest,
            "engine_extra_sustained_bytes": s_extra,
            "others_ref_bytes": self.rest,
            "engine_extra_ref_bytes": self.extra,
        }
        return self.value

    def _anchor_swapouts(self, now: float) -> int:
        """The swap-out counter at the start of the sustain span."""
        anchor = self.samples[0][3]
        for x in self.samples:
            if x[0] <= now - self.sustain_s:
                anchor = x[3]
            else:
                break
        return anchor

    @staticmethod
    def _step(ref, current, sustained, swapping, quiet, window, index):
        if swapping and current > ref:
            return current
        if sustained is not None and sustained > ref:
            return sustained
        if current < ref and quiet:
            # the highest SUSTAINED reading of the window (a spike in it does
            # not hold the reference up); no sustained reading yet: the raw max
            highs = [x[index] for x in window if x[index] is not None]
            target = max(highs) if highs else max(x[index - 3] for x in window)
            return min(ref, max(target, current))
        return ref


class MetalPreflightError(RuntimeError):
    """The resolved Metal cap can never admit a request for this model.

    Raised during engine startup — before the server reports the model
    ready — so the operator sees one actionable message instead of a
    healthy-looking model whose every request returns HTTP 503.
    """


@dataclass(frozen=True)
class MetalBudgetPlan:
    """The resolved Metal allocation budget for one loaded model."""

    weights_bytes: int
    device_budget_bytes: int
    requested_utilization: float | None
    resolved_utilization: float
    limit_bytes: int
    mode: str  # "manual" (operator override) or "auto"
    # "auto-weights" (weights + headroom), "auto-context" (the context-aware
    # need), "manual-utilization" or "manual-gb".
    source: str = "auto-weights"
    need_bytes: int = 0
    # "engine" (the plan's own figure), "auto_ceiling" (the need was above
    # the auto ceiling), or "macos_working_set" (a GB cap or the need was
    # above what the macOS GPU working set allows).
    binding: str = "engine"


def plan_metal_limit(
    *,
    weights_bytes: int,
    device_budget_bytes: int,
    requested_utilization: float | None = None,
    need_bytes: int = 0,
    limit_bytes: int | None = None,
    floor: float = AUTO_UTILIZATION_FLOOR,
    ceiling: float = AUTO_UTILIZATION_CEILING,
    headroom_fraction: float = _AUTO_HEADROOM_FRACTION,
) -> MetalBudgetPlan:
    """Resolve the Metal allocation limit for one loaded model.

    Args:
        weights_bytes: Measured Metal footprint of the loaded weights
            (``mx.get_active_memory()`` right after load). ``<= 0`` means
            "no measurement available" and auto mode falls back to the
            historical floor.
        device_budget_bytes: The device's recommended working-set size
            (``max_recommended_working_set_size``). Must be positive.
        requested_utilization: The operator's explicit
            ``--gpu-memory-utilization``, or ``None`` for auto.

    Returns:
        A :class:`MetalBudgetPlan`. ``resolved_utilization`` is the value the
        engine must feed to BOTH ``mx.set_memory_limit`` and the scheduler's
        D-METAL-CAP admission gate — the two enforcement points must always
        agree on the same cap.
    """
    if device_budget_bytes <= 0:
        raise ValueError("device_budget_bytes must be positive")

    binding = "engine"
    need = 0
    if limit_bytes is not None and limit_bytes > 0:
        # An absolute cap in GB (``--metal-limit-gb``): the macOS GPU working
        # set is the hard edge above it (utilization 1.0).
        wanted = limit_bytes / device_budget_bytes
        resolved = min(wanted, 1.0)
        if wanted > 1.0:
            binding = "macos_working_set"
        mode, source = "manual", "manual-gb"
    elif requested_utilization is not None:
        resolved = float(requested_utilization)
        mode, source = "manual", "manual-utilization"
    elif weights_bytes <= 0 and need_bytes <= 0:
        resolved = floor
        mode, source = "auto", "auto-weights"
    else:
        headroom = max(
            int(weights_bytes * headroom_fraction),
            int(device_budget_bytes * _AUTO_HEADROOM_MIN_FRACTION),
        )
        weights_need = weights_bytes + headroom if weights_bytes > 0 else 0
        # The context-aware need (weights + KV of the served context + one
        # step's peak + margin + prefix budget, the scheduler's own figures)
        # when it is the larger: a 72 GB model served at 360k needs more
        # than its weights + 8% (workmule 2026-10-30: 103.9 GB floor while
        # 112 GB was allowed).
        need = max(weights_need, int(need_bytes))
        source = "auto-context" if int(need_bytes) > weights_need else "auto-weights"
        needed = need / device_budget_bytes
        resolved = min(max(needed, floor), ceiling)
        if needed > ceiling:
            # The need is above what auto may take: the auto ceiling binds
            # (or, at a ceiling of 1.0, the macOS working set itself).
            binding = "macos_working_set" if ceiling >= 1.0 else "auto_ceiling"
        mode = "auto"

    return MetalBudgetPlan(
        weights_bytes=max(0, int(weights_bytes)),
        device_budget_bytes=int(device_budget_bytes),
        requested_utilization=requested_utilization,
        resolved_utilization=resolved,
        limit_bytes=int(device_budget_bytes * resolved),
        mode=mode,
        source=source,
        need_bytes=int(need),
        binding=binding,
    )


def format_preflight_error(
    *,
    required_bytes: int,
    active_bytes: int,
    min_kv_bytes: int,
    cap_bytes: int,
    utilization: float,
    device_budget_bytes: int,
) -> str:
    """Build the actionable admission-impossible startup message (#2858).

    The remediation list is tailored to what can still help (codex rounds
    1 and 3): "increase --gpu-memory-utilization" is suggested while the
    enforced utilization sits below 1.0 — an explicit override can legally
    go all the way there, even past the auto ceiling. Only at a full 1.0
    is the knob truly exhausted, and the honest advice becomes that this
    Mac does not have the memory for this configuration.
    """
    if utilization < MAX_UTILIZATION:
        remediation = (
            "Increase --gpu-memory-utilization, reduce context length or "
            "concurrency, close memory-heavy apps, or choose a smaller model."
        )
    else:
        remediation = (
            "This Mac does not have enough unified memory for this "
            "configuration even at the maximum Metal budget — reduce "
            "context length or concurrency, close memory-heavy apps, "
            "retry after in-flight requests on other models finish, or "
            "choose a smaller model."
        )
    return (
        f"This model needs approximately {required_bytes / 1e9:.1f} GB of "
        f"Metal memory for the current configuration (weights and runtime "
        f"{active_bytes / 1e9:.1f} GB + minimum KV cache "
        f"{min_kv_bytes / 1e9:.1f} GB), but the current limit is "
        f"{cap_bytes / 1e9:.1f} GB "
        f"(gpu_memory_utilization={utilization:g} of the "
        f"{device_budget_bytes / 1e9:.1f} GB Metal working-set budget). "
        f"{remediation}"
    )


def compute_metal_cache_limit(soft_limit_bytes: int) -> int:
    """Pick a Metal free-cache size that scales with the device's working set.

    The free cache holds memory that was freed by Python objects but not yet
    returned to the GPU. A larger cache speeds up subsequent allocations
    (KV cache churn, prefix cache moves) but caps the budget that inference
    can grow into under load.

    Old behavior (hardcoded 32 GB) was sized for big machines: comfortable on
    M3 Ultra 256GB (15% of soft limit), but allowed cache to grow to ~50% of
    the soft limit on M2 Max 96GB, leaving insufficient room for a 35B model
    + accumulated prefix cache + transient prefill allocations. Small machines
    hit memory pressure → macOS paging → catastrophic slowdown.

    Scale to 25% of the soft allocation limit, capped at 32 GiB (no change for
    big machines), floored at 2 GiB (avoid degenerate cache on small machines).
    Clamp to soft_limit to preserve MLX's implicit cache ≤ memory invariant on
    pathologically tiny devices.
    """
    cache = max(
        2 * 1024 * 1024 * 1024,
        min(32 * 1024 * 1024 * 1024, soft_limit_bytes // 4),
    )
    return min(cache, soft_limit_bytes) if soft_limit_bytes > 0 else cache


# The kernel's memory-pressure levels (Darwin dispatch / memorystatus enum,
# not tunables): 1 normal, 2 warn, 4 critical.
SYSTEM_PRESSURE_WARN = 2
SYSTEM_PRESSURE_CRITICAL = 4
_LIBC = None


def system_memory_pressure() -> int:
    """The machine's memory-pressure level as the kernel reports it
    (``kern.memorystatus_vm_pressure_level``): what the REST of the machine
    needs -- other apps, other MLX engines -- as one measured signal. 0 when
    unavailable (not Darwin)."""
    try:
        import ctypes
        import ctypes.util
        import sys

        if sys.platform != "darwin":
            return 0
        global _LIBC
        if _LIBC is None:
            _LIBC = ctypes.CDLL(ctypes.util.find_library("c"))
        libc = _LIBC
        value = ctypes.c_int(0)
        size = ctypes.c_size_t(ctypes.sizeof(value))
        if libc.sysctlbyname(
            b"kern.memorystatus_vm_pressure_level",
            ctypes.byref(value),
            ctypes.byref(size),
            None,
            0,
        ):
            return 0
        return int(value.value)
    except Exception:  # noqa: BLE001
        return 0


# LeanZero memory guard: the share of the machine's RAM kept FREE for
# everything that is not the engine (other apps, other engines, the OS's own
# file cache). The engine's OPTIONAL memory (its prefix cache, copies of
# entries) stays under physical RAM - what the other processes use - this
# reserve. A ratio of physical RAM, the headroom auto mode charges on top of
# the weights.
SYSTEM_RESERVE_RATIO = _AUTO_HEADROOM_FRACTION
# The default of ``--system-reserve-fraction`` (receipt: RC10, a busy 128 GB
# laptop -- the engine took 91 GB, 40.2 GB of it prefix entries, and the
# machine went 46.8 GB into swap).
_SYSTEM_RESERVE_RATIO_DEFAULT = SYSTEM_RESERVE_RATIO


def _sysctl_int(name: bytes, ctype_name: str = "c_int") -> int | None:
    try:
        import ctypes
        import ctypes.util
        import sys

        if sys.platform != "darwin":
            return None
        global _LIBC
        if _LIBC is None:
            _LIBC = ctypes.CDLL(ctypes.util.find_library("c"))
        value = getattr(ctypes, ctype_name)(0)
        size = ctypes.c_size_t(ctypes.sizeof(value))
        if _LIBC.sysctlbyname(name, ctypes.byref(value), ctypes.byref(size), None, 0):
            return None
        return int(value.value)
    except Exception:  # noqa: BLE001
        return None


class MachineMemory:
    """One reading of the machine's memory (``host_statistics64``)."""

    __slots__ = ("used", "swapouts", "breakdown", "engine_footprint")

    def __init__(
        self,
        used: int,
        swapouts: int,
        breakdown: dict | None = None,
        engine_footprint: int | None = None,
    ):
        self.used = int(used)  # resident bytes in use, Activity Monitor's
        self.swapouts = int(swapouts)  # pages swapped out since boot
        # bytes per vm_stat class: what counts and what is left out (status)
        self.breakdown = breakdown
        # this process's phys_footprint (task_info), read in the SAME call as
        # the page counts so "used - engine" never mixes two moments; None
        # when unknown (then the engine is its Metal footprint alone)
        self.engine_footprint = (
            int(engine_footprint) if engine_footprint and engine_footprint > 0 else None
        )


def used_bytes_from_pages(
    *,
    page: int,
    wired: int,
    internal: int,
    purgeable: int,
    compressor: int,
    external: int = 0,
    free: int = 0,
    speculative: int = 0,
) -> tuple[int, dict]:
    """Memory that is really unavailable, from ``host_statistics64`` page
    counts: wired + anonymous (internal, active AND inactive: an inactive
    anonymous page only leaves RAM by being compressed or swapped) -
    purgeable (volatile, the kernel drops it on demand) + the pages the
    compressor occupies. NOT counted: file-backed (external) pages -- the
    file cache, including the engine's own memory-mapped files such as the
    qwen4 PLE shards (np.memmap, read-only, never wired), which macOS frees
    on demand -- nor free or speculative pages. Measured on the mule
    2026-10-30 mid-Gauntlet: used 119.5 GB = wired 102.0 + anonymous 16.5 +
    compressor 1.4, with 19.5 GB of file-backed pages left out.

    Returns ``(used_bytes, breakdown)``; the breakdown is published in
    /v1/status ``memory_budget.machine`` so the floor's input is auditable."""
    pages = int(wired) + int(internal) - int(purgeable) + int(compressor)
    breakdown = {
        "wired_bytes": int(wired) * page,
        "anonymous_bytes": int(internal) * page,
        "compressor_bytes": int(compressor) * page,
        "purgeable_excluded_bytes": int(purgeable) * page,
        "file_backed_excluded_bytes": int(external) * page,
        "free_bytes": int(free) * page,
        "speculative_excluded_bytes": int(speculative) * page,
    }
    return max(0, pages) * page, breakdown


_VM_STATISTICS64 = None


def system_memory_used() -> MachineMemory | None:
    """The machine's resident memory in use, as Activity Monitor's "Memory
    Used": wired + internal (anonymous) - purgeable + compressor pages
    (``host_statistics64(HOST_VM_INFO64)``), plus the swap-out counter.
    ``None`` when unavailable (not Darwin).

    NOT ``kern.memorystatus_level``: that "free percentage" counts anonymous
    and swapped-out pages as available -- on the 128 GB laptop it read 75%
    free (~34 GB used) while ``top`` said 99 GB used with 27 GB in swap, and
    an engine buffer the GPU wires or unwires moved it by 5 points (review of
    c976b2a3a). Wired + anonymous stays put across a wire/unwire (+-0.15 GB).
    Pages already swapped out are not resident and do not count here either:
    the guard's reference therefore never falls while the machine swaps
    (``Scheduler._memguard_others_ref``)."""
    try:
        import ctypes
        import ctypes.util
        import sys

        if sys.platform != "darwin":
            return None
        global _LIBC, _VM_STATISTICS64
        if _LIBC is None:
            _LIBC = ctypes.CDLL(ctypes.util.find_library("c"))
        if _VM_STATISTICS64 is None:
            u32, u64 = ctypes.c_uint32, ctypes.c_uint64

            class _VM(ctypes.Structure):  # <mach/vm_statistics.h>
                _fields_ = [
                    ("free_count", u32), ("active_count", u32),
                    ("inactive_count", u32), ("wire_count", u32),
                    ("zero_fill_count", u64), ("reactivations", u64),
                    ("pageins", u64), ("pageouts", u64), ("faults", u64),
                    ("cow_faults", u64), ("lookups", u64), ("hits", u64),
                    ("purges", u64), ("purgeable_count", u32),
                    ("speculative_count", u32), ("decompressions", u64),
                    ("compressions", u64), ("swapins", u64), ("swapouts", u64),
                    ("compressor_page_count", u32), ("throttled_count", u32),
                    ("external_page_count", u32), ("internal_page_count", u32),
                    ("total_uncompressed_pages_in_compressor", u64),
                ]

            _LIBC.mach_host_self.restype = ctypes.c_uint32
            _VM_STATISTICS64 = _VM
        vm = _VM_STATISTICS64()
        count = ctypes.c_uint32(ctypes.sizeof(vm) // 4)
        if _LIBC.host_statistics64(
            _LIBC.mach_host_self(), 4, ctypes.byref(vm), ctypes.byref(count)
        ):
            return None
        page = _sysctl_int(b"hw.pagesize", "c_uint64") or 0
        if page <= 0:
            return None
        used, breakdown = used_bytes_from_pages(
            page=page,
            wired=vm.wire_count,
            internal=vm.internal_page_count,
            purgeable=vm.purgeable_count,
            compressor=vm.compressor_page_count,
            external=vm.external_page_count,
            free=vm.free_count,
            speculative=vm.speculative_count,
        )
        footprint = None
        try:
            from .runtime.process_memory import get_phys_footprint

            footprint = get_phys_footprint() or None
        except Exception:  # noqa: BLE001
            footprint = None
        return MachineMemory(used, vm.swapouts, breakdown, footprint)
    except Exception:  # noqa: BLE001
        return None


def system_physical_memory() -> int:
    """The machine's physical RAM in bytes (``hw.memsize``); 0 if unknown."""
    return int(_sysctl_int(b"hw.memsize", "c_uint64") or 0)
