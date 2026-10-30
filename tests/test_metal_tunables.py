# SPDX-License-Identifier: Apache-2.0
"""Memory tunables and the context-aware auto Metal cap (lz.20).

Receipt (workmule, M5 Max 128 GiB, 2026-10-30, live Gauntlet on lz.18):
physical 137.44 GB, iogpu.wired_limit_mb 0 -> MLX working set 115.45 GB,
Jundot Flash-Next oQ4e (~72 GB of weights with PLE offload) served a 360k
window, yet the guard planned under 103.9 GB (0.90 x 115.45): auto mode sized
the cap from the weights + 8% only, which clamped UP to the 0.90 floor, and
the 0.97 ceiling (112.0 GB) was never used. ~8 GB the engine was allowed to
take, plus ~22 GB macOS kept for itself, sat unused.

Each tunable is proven to REACH the engine and change what it does, not just
to be stored. Devices: the fleet's three RAM sizes, with the measured MLX
working sets (128 GiB workmule/laptop 115.45 GB, 96 GiB workhorse 83.49 GB;
64 GiB Gabee 55.67 GB, the 50.10 GB guard ceiling of RC12 / 0.90).
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import rapid_mlx.memory_budget as mb
from rapid_mlx.memory_budget import (
    AUTO_UTILIZATION_CEILING,
    AUTO_UTILIZATION_FLOOR,
    floor_room_bytes,
    plan_metal_limit,
    tunables,
    validate_tunables,
)
from rapid_mlx.memory_guard import MemoryGuard, PeakPrior

GB = 1_000_000_000

# (name, physical bytes, MLX max_recommended_working_set_size)
DEVICES = {
    "128": (137_438_953_472, 115_448_725_504),  # workmule / laptop, measured
    "96": (103_079_215_104, 83_494_174_720),  # workhorse, measured 2026-10-30
    "64": (68_719_476_736, 55_666_666_666),  # Gabee: 50.10 GB ceiling / 0.90
}

_ENVS = (
    mb.ENV_METAL_LIMIT_GB,
    mb.ENV_AUTO_FLOOR,
    mb.ENV_AUTO_CEILING,
    mb.ENV_AUTO_HEADROOM,
    mb.ENV_MEMORY_FLOOR_GB,
    mb.ENV_RESERVE_FRACTION,
    mb.ENV_MAX_MODEL_LEN,
    mb.ENV_GUARD_MARGIN_GB,
    mb.ENV_FLOOR_WAIT_S,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENVS:
        monkeypatch.delenv(name, raising=False)
    with mb._process_floor_lock:
        saved = (mb._process_utilization_floor, mb._process_floor_generation)
        mb._process_utilization_floor = 0.0
        mb._process_floor_generation += 1
    yield
    with mb._process_floor_lock:
        mb._process_utilization_floor = saved[0]
        mb._process_floor_generation += 1


def _flash_next_need(working_set: int, context: int = 360_000) -> int:
    """The scheduler's own need for Flash-Next at ``context``: the guard's
    single-sequence need (idle base 72 GB, KV/state, the floor chunk's step
    peak at the analytic prior the live engine reported -- attention 192
    B/(chunk*context), token 107,776 B, kv copy 2,048 B/context token -- and
    the margin) plus the cache's configured prefix budget (16.86 GB live)."""
    guard = MemoryGuard(
        PeakPrior(
            attention_bytes_per_unit=192.0,
            token_bytes=107_776.0,
            kv_copy_bytes_per_context=2048.0,
            kv_bytes_per_token=40_960.0,
        ),
        floor_tokens=256,
    )
    need = guard.single_sequence_need(
        context,
        base=72 * GB,
        kv_footprint=lambda t: 40_960 * t,
        ceiling=int(working_set * AUTO_UTILIZATION_CEILING),
    )
    return need + 16_863_260_000


# ── the regression: Flash-Next 360k on the workmule ──────────────────────


def test_flash_next_360k_on_the_workmule_resolves_the_ceiling():
    _phys, ws = DEVICES["128"]
    before = plan_metal_limit(weights_bytes=72 * GB, device_budget_bytes=ws)
    # lz.19: weights + 8% = 77.8 GB -> clamped UP to the floor: 103.9 GB.
    assert before.resolved_utilization == AUTO_UTILIZATION_FLOOR
    assert round(before.limit_bytes / GB, 1) == 103.9
    after = plan_metal_limit(
        weights_bytes=72 * GB, device_budget_bytes=ws, need_bytes=_flash_next_need(ws)
    )
    assert after.resolved_utilization == AUTO_UTILIZATION_CEILING
    assert after.source == "auto-context"
    assert after.binding == "auto_ceiling"
    assert round(after.limit_bytes / GB, 1) == 112.0


@pytest.mark.parametrize("device", sorted(DEVICES))
def test_small_models_stay_byte_identical_on_every_fleet_size(device):
    """A 5 GB model whose whole served context fits under 0.90 resolves
    exactly the lz.19 limit on 64, 96 and 128 GB Macs."""
    _phys, ws = DEVICES[device]
    old = plan_metal_limit(weights_bytes=5 * GB, device_budget_bytes=ws)
    new = plan_metal_limit(
        weights_bytes=5 * GB, device_budget_bytes=ws, need_bytes=int(0.5 * ws)
    )
    assert new.resolved_utilization == old.resolved_utilization == 0.90
    assert new.limit_bytes == old.limit_bytes


@pytest.mark.parametrize("device", sorted(DEVICES))
def test_a_need_between_floor_and_ceiling_is_met_exactly(device):
    _phys, ws = DEVICES[device]
    need = int(0.93 * ws)
    plan = plan_metal_limit(weights_bytes=int(0.5 * ws), device_budget_bytes=ws, need_bytes=need)
    assert plan.resolved_utilization == pytest.approx(0.93)
    assert plan.binding == "engine"


@pytest.mark.parametrize("device", sorted(DEVICES))
def test_the_effective_ceiling_is_the_min_of_the_three_caps(device):
    """effective = min(engine cap, working-set cap, physical - others - floor)
    on each fleet size, each layer binding in turn."""
    phys, ws = DEVICES[device]
    floor = 4 * GB
    # an engine cap of 0.97 x ws, idle Mac: the engine cap binds
    cap = plan_metal_limit(weights_bytes=1, device_budget_bytes=ws, need_bytes=ws).limit_bytes
    room = floor_room_bytes(physical=phys, others=2 * GB, floor=floor)
    assert min(cap, room) == cap
    # other apps take most of the RAM: the floor binds
    busy = phys - cap // 2
    room = floor_room_bytes(physical=phys, others=busy, floor=floor)
    assert min(cap, room) == phys - busy - floor < cap
    # a GB cap above the working set: macOS binds at 1.0 x ws
    manual = plan_metal_limit(weights_bytes=1, device_budget_bytes=ws, limit_bytes=phys)
    assert manual.limit_bytes == ws and manual.binding == "macos_working_set"
    # floor off: no third cap
    assert floor_room_bytes(physical=phys, others=busy, floor=0) is None


def test_a_gb_cap_below_the_working_set_is_honoured():
    _phys, ws = DEVICES["128"]
    plan = plan_metal_limit(weights_bytes=72 * GB, device_budget_bytes=ws, limit_bytes=110 * GB)
    assert plan.source == "manual-gb" and plan.binding == "engine"
    assert plan.limit_bytes == pytest.approx(110 * GB, abs=1)


def test_auto_bounds_and_headroom_are_tunable():
    _phys, ws = DEVICES["96"]
    plan = plan_metal_limit(weights_bytes=1 * GB, device_budget_bytes=ws, floor=0.5)
    assert plan.resolved_utilization == 0.5
    plan = plan_metal_limit(
        weights_bytes=int(0.95 * ws), device_budget_bytes=ws, ceiling=0.99
    )
    assert plan.resolved_utilization == 0.99  # 0.95 x 1.08 = 1.026 -> 0.99
    plan = plan_metal_limit(
        weights_bytes=int(0.8 * ws), device_budget_bytes=ws, headroom_fraction=0.15
    )
    assert plan.resolved_utilization == pytest.approx(0.92)


# ── tunables: defaults, env, validation, CLI ─────────────────────────────


def test_unset_tunables_are_todays_constants():
    t = tunables()
    assert (t.auto_floor, t.auto_ceiling, t.auto_headroom_fraction) == (0.90, 0.97, 0.08)
    assert t.reserve_fraction == mb.SYSTEM_RESERVE_RATIO == 0.08
    assert t.metal_limit_bytes is None and t.memory_floor_bytes == 0
    assert t.max_model_len is None and t.guard_margin_bytes is None
    assert t.floor_wait_s == mb.DEFAULT_FLOOR_WAIT_S


@pytest.mark.parametrize(
    "env,problem",
    [
        ({mb.ENV_METAL_LIMIT_GB: "130", mb.ENV_MEMORY_FLOOR_GB: "10"}, "more than this Mac"),
        ({mb.ENV_MEMORY_FLOOR_GB: "200"}, "not below"),
        ({mb.ENV_AUTO_FLOOR: "0.98", mb.ENV_AUTO_CEILING: "0.95"}, "is above"),
        ({mb.ENV_RESERVE_FRACTION: "0.9"}, "0..0.5"),
        ({mb.ENV_AUTO_CEILING: "1.5"}, "0.1..1.0"),
    ],
)
def test_impossible_combinations_are_rejected(monkeypatch, env, problem):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    message = validate_tunables(tunables(), physical_bytes=DEVICES["128"][0])
    assert message and problem in message


def test_a_gb_cap_and_a_utilization_are_exclusive(monkeypatch):
    monkeypatch.setenv(mb.ENV_METAL_LIMIT_GB, "100")
    message = validate_tunables(tunables(), physical_bytes=0, utilization_set=True)
    assert "--gpu-memory-utilization" in message


def test_a_valid_inference_box_set_passes(monkeypatch):
    monkeypatch.setenv(mb.ENV_METAL_LIMIT_GB, "112")
    monkeypatch.setenv(mb.ENV_MEMORY_FLOOR_GB, "4")
    assert validate_tunables(tunables(), physical_bytes=DEVICES["128"][0]) is None


def _serve_args(*extra: str) -> argparse.Namespace:
    from rapid_mlx.cli import build_parser

    return build_parser().parse_args(["serve", "some/model", *extra])


def test_every_flag_reaches_the_env_the_engine_reads(monkeypatch):
    from rapid_mlx.cli import _apply_memory_tunable_args

    monkeypatch.setattr(mb, "system_physical_memory", lambda: DEVICES["128"][0])
    args = _serve_args(
        "--metal-limit-gb", "112",
        "--auto-utilization-floor", "0.8",
        "--auto-utilization-ceiling", "0.99",
        "--auto-headroom-fraction", "0.1",
        "--memory-floor-gb", "4",
        "--memory-floor-wait-s", "30",
        "--system-reserve-fraction", "0.05",
        "--max-model-len", "360000",
        "--memory-guard-margin-gb", "1.5",
    )
    _apply_memory_tunable_args(args)
    t = tunables()
    assert t.metal_limit_bytes == 112 * GB
    assert (t.auto_floor, t.auto_ceiling, t.auto_headroom_fraction) == (0.8, 0.99, 0.1)
    assert t.memory_floor_bytes == 4 * GB and t.floor_wait_s == 30
    assert t.reserve_fraction == 0.05 and t.max_model_len == 360_000
    assert t.guard_margin_bytes == int(1.5 * GB)


def test_an_impossible_flag_set_exits_2(monkeypatch, capsys):
    from rapid_mlx.cli import _apply_memory_tunable_args

    monkeypatch.setattr(mb, "system_physical_memory", lambda: DEVICES["64"][0])
    args = _serve_args("--metal-limit-gb", "64", "--memory-floor-gb", "8")
    with pytest.raises(SystemExit) as exc:
        _apply_memory_tunable_args(args)
    assert exc.value.code == 2
    assert "--memory-floor-gb" in capsys.readouterr().err


# ── the engine resolves the tunables (BatchedEngine wiring) ──────────────


def _bare_engine():
    from rapid_mlx.engine.batched import BatchedEngine

    engine = BatchedEngine.__new__(BatchedEngine)
    engine._gpu_memory_utilization = None
    engine._model_load_executor = None
    engine._engine = None
    return engine


def _patched_mx(ws: int, active: int):
    return patch.multiple(
        "mlx.core",
        device_info=MagicMock(return_value={"max_recommended_working_set_size": ws}),
        clear_cache=MagicMock(),
        get_active_memory=MagicMock(return_value=active),
        set_memory_limit=MagicMock(),
        set_cache_limit=MagicMock(),
    ), patch("mlx.core.metal.is_available", return_value=True)


def test_metal_limit_gb_sets_the_metal_limit(monkeypatch):
    import mlx.core as mx

    _phys, ws = DEVICES["128"]
    monkeypatch.setenv(mb.ENV_METAL_LIMIT_GB, "110")
    engine = _bare_engine()
    core, metal = _patched_mx(ws, 72 * GB)
    with core, metal:
        resolved = engine._resolve_and_set_metal_limits()
        assert resolved == pytest.approx(110 * GB / ws)
        mx.set_memory_limit.assert_called_once_with(int(ws * resolved))
    assert engine.metal_budget_snapshot()["source"] == "manual-gb"


def test_auto_floor_tunable_changes_the_installed_limit(monkeypatch):
    import mlx.core as mx

    _phys, ws = DEVICES["96"]
    monkeypatch.setenv(mb.ENV_AUTO_FLOOR, "0.75")
    engine = _bare_engine()
    core, metal = _patched_mx(ws, 5 * GB)
    with core, metal:
        assert engine._resolve_and_set_metal_limits() == 0.75
        mx.set_memory_limit.assert_called_once_with(int(ws * 0.75))


class _SyncExecutor:
    def submit(self, fn, *a, **kw):
        result = fn(*a, **kw)
        return SimpleNamespace(result=lambda: result)


def test_the_context_pass_raises_the_limit_to_the_need(monkeypatch):
    """After the scheduler exists, auto mode re-plans with its need and
    raises the installed limit (Flash-Next 360k: 103.9 -> 112.0 GB)."""
    import mlx.core as mx

    _phys, ws = DEVICES["128"]
    need = _flash_next_need(ws)
    engine = _bare_engine()
    engine._model_load_executor = _SyncExecutor()
    seen = {}

    def need_fn(ceiling):
        seen["ceiling"] = ceiling
        return need

    sched = SimpleNamespace(
        memory_plan_need_bytes=need_fn,
        config=SimpleNamespace(gpu_memory_utilization=0.90),
    )
    engine._engine = SimpleNamespace(engine=SimpleNamespace(scheduler=sched))
    core, metal = _patched_mx(ws, 72 * GB)
    with core, metal:
        assert engine._resolve_and_set_metal_limits() == AUTO_UTILIZATION_FLOOR
        engine._gpu_memory_utilization = AUTO_UTILIZATION_FLOOR
        engine._raise_metal_budget_for_context()
        assert mx.set_memory_limit.call_args_list[-1].args[0] == int(ws * 0.97)
    assert engine._gpu_memory_utilization == 0.97
    assert sched.config.gpu_memory_utilization == 0.97
    assert seen["ceiling"] == int(ws * 0.97)
    snap = engine.metal_budget_snapshot()
    assert snap["source"] == "auto-context" and snap["plan_need_bytes"] == need


def test_the_context_pass_never_touches_a_manual_cap(monkeypatch):
    import mlx.core as mx

    _phys, ws = DEVICES["128"]
    monkeypatch.setenv(mb.ENV_METAL_LIMIT_GB, "100")
    engine = _bare_engine()
    engine._model_load_executor = _SyncExecutor()
    engine._engine = SimpleNamespace(
        engine=SimpleNamespace(
            scheduler=SimpleNamespace(memory_plan_need_bytes=lambda c: 10**15)
        )
    )
    core, metal = _patched_mx(ws, 72 * GB)
    with core, metal:
        engine._resolve_and_set_metal_limits()
        calls = len(mx.set_memory_limit.call_args_list)
        engine._raise_metal_budget_for_context()
        assert len(mx.set_memory_limit.call_args_list) == calls


# ── the guard margin ─────────────────────────────────────────────────────


def test_the_guard_margin_override_replaces_the_ceiling_share():
    guard = MemoryGuard(
        PeakPrior(attention_bytes_per_unit=1.0, token_bytes=1.0, kv_copy_bytes_per_context=1.0),
        floor_tokens=256,
    )
    ceiling = 100 * GB
    assert guard.margin(0, ceiling) == 2 * GB  # 2% of the ceiling
    guard.margin_bytes_override = int(0.5 * GB)
    assert guard.margin(0, ceiling) == int(0.5 * GB)
    assert guard.margin(10 * GB, ceiling) == int(0.5 * GB + 1 * GB)
