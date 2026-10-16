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


def plan_metal_limit(
    *,
    weights_bytes: int,
    device_budget_bytes: int,
    requested_utilization: float | None = None,
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

    if requested_utilization is not None:
        resolved = float(requested_utilization)
        mode = "manual"
    elif weights_bytes <= 0:
        resolved = AUTO_UTILIZATION_FLOOR
        mode = "auto"
    else:
        headroom = max(
            int(weights_bytes * _AUTO_HEADROOM_FRACTION),
            int(device_budget_bytes * _AUTO_HEADROOM_MIN_FRACTION),
        )
        needed = (weights_bytes + headroom) / device_budget_bytes
        resolved = min(max(needed, AUTO_UTILIZATION_FLOOR), AUTO_UTILIZATION_CEILING)
        mode = "auto"

    return MetalBudgetPlan(
        weights_bytes=max(0, int(weights_bytes)),
        device_budget_bytes=int(device_budget_bytes),
        requested_utilization=requested_utilization,
        resolved_utilization=resolved,
        limit_bytes=int(device_budget_bytes * resolved),
        mode=mode,
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
# file cache). The engine's planning ceiling is what the machine leaves it:
# physical RAM - what the other processes use now - this reserve. A ratio of
# physical RAM, the same headroom auto mode charges on top of the weights.
SYSTEM_RESERVE_RATIO = _AUTO_HEADROOM_FRACTION


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


def system_memory_free_fraction() -> float | None:
    """The share of the machine's RAM free or reclaimable now, as the
    kernel measures it (``kern.memorystatus_level``: what ``memory_pressure``
    prints as "System-wide memory free percentage"). ``None`` when
    unavailable (not Darwin). Finer than the pressure LEVEL, which stayed at
    1 (normal) on a 128 GB laptop at 5% free with 46.8 GB of swap in use
    (RC10 ladder, 2026-10-16)."""
    level = _sysctl_int(b"kern.memorystatus_level")
    if level is None or not 0 <= level <= 100:
        return None
    return level / 100.0


def system_physical_memory() -> int:
    """The machine's physical RAM in bytes (``hw.memsize``); 0 if unknown."""
    return int(_sysctl_int(b"hw.memsize", "c_uint64") or 0)
