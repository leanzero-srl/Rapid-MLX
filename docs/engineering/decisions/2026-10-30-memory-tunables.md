# Engine memory tunables and the context-aware auto Metal cap (v0.14.3-lz.20)

## Why
Workmule (M5 Max, 128 GiB), live Gauntlet on lz.18, 2026-10-30, read-only `/v1/status`:
guard ceiling 103.9 GB, margin 2.0 GB, Metal active 94.4 GB, peak 102.1 GB; `hw.memsize` 137.44 GB,
`iogpu.wired_limit_mb` 0, MLX `max_recommended_working_set_size` 115.45 GB. Auto mode sized the cap as
weights + 8% (~78 GB for Flash-Next's ~72 GB) and clamped it UP to the 0.90 floor: 103.9 GB. The KV of the
served window, the step peak and the prefix budget were never part of it, so the 0.97 ceiling (112.0 GB) was
never reached. Cap layers, top down: physical 137.44 -> macOS GPU working set 115.45 -> engine cap 103.9 ->
guard step ceiling 103.9 - 2% margin.

## What changed
1. Auto mode, second pass: after the scheduler exists, `Scheduler.memory_plan_need_bytes` gives
   weights + KV/state at `--max-model-len` (else the model's window) + the floor chunk's step peak + margin
   (the guard's own `single_sequence_need`) + the prefix budget (operator's, else the cache's configured
   default). The cap is `clamp(need / working set, floor, ceiling)`; raise only (the ratchet).
2. Every number is a tunable. Flag -> env (the CLI exports it; workers read it):

| Flag | Env | Default | Receipt for the default |
|---|---|---|---|
| `--metal-limit-gb` | RAPID_MLX_METAL_LIMIT_GB | auto | #2858: auto sizing replaced the hand-tuned global knob |
| `--auto-utilization-floor` | RAPID_MLX_AUTO_UTILIZATION_FLOOR | 0.90 | the historical global default; small models stay byte-identical |
| `--auto-utilization-ceiling` | RAPID_MLX_AUTO_UTILIZATION_CEILING | 0.97 | #2858: 3% allocator slack, never plan to the working-set edge |
| `--auto-headroom-fraction` | RAPID_MLX_AUTO_HEADROOM_FRACTION | 0.08 | #2858: KV + activations + fragmentation on top of weights |
| `--memory-floor-gb` | RAPID_MLX_MEMORY_FLOOR_GB | 0 (off) | lz.19 had no hard floor; goose sets one per Mac |
| `--memory-floor-wait-s` | RAPID_MLX_MEMORY_FLOOR_WAIT_S | 60 | a step waits this long for other apps before it is refused |
| `--system-reserve-fraction` | RAPID_MLX_SYSTEM_RESERVE_FRACTION | 0.08 | RC10: 128 GB laptop, 40.2 GB of prefix entries, 46.8 GB swap |
| `--max-model-len` | RAPID_MLX_MAX_MODEL_LEN | model window | the context auto sizes for; prompts above it are refused |
| `--memory-guard-margin-gb` | RAPID_MLX_MEMORY_GUARD_MARGIN_GB | 2% of ceiling | MARGIN_CEILING_RATIO (the transient 10% share stays) |
| `--cache-memory-mb` (existing) | RAPID_MLX_PREFIX_CACHE_MAX_BYTES | guard idle room | E3 |
| `--prefill-step-size` (existing) | - | 2048 | mlx-lm |
| `--max-concurrent-requests` (existing) | - | engine default | - |

3. The free-memory floor: every prefill step plans under `min(engine cap, physical - others - floor)`,
   others = `host_statistics64` used - the engine's Metal footprint (hysteresis: rises at once, falls slowly),
   re-read at most every second. Order of actions: allocator cache, evictions, smaller chunk; if only the floor
   is short, the step WAITS (no dispatch, `floor_holds` counter) up to `--memory-floor-wait-s`, then the request
   is refused like any prompt that does not fit. The engine cannot give back what it already holds: when another
   process alone takes the Mac below the floor, the engine stops growing (tests/test_memory_guard_floor.py).
4. `/v1/status` `memory_budget`: physical, working set, `iogpu.wired_limit_mb`, plan (mode, source, binding,
   need), enforced limit, step ceiling and its binding, others, free, active/cache/peak, KV bytes/token, the
   window at the ceiling, and the tunables in force.
5. Validation at start (exit 2, one line): GB cap + floor > physical, floor >= physical, auto floor > ceiling,
   fractions out of range, `--metal-limit-gb` with `--gpu-memory-utilization`.

## Measured
See /Volumes/AI-workhorse/runs/2026-10-30-flash-fixes-verify/metal-limit/ (CAUSE.md, VIGIL.md, NOTES).

## The macOS layer
`sysctl iogpu.wired_limit_mb` (0 = macOS default: measured 115.45 of 137.44 GB on 128 GiB (84%), 83.49 of 103.08 on 96 GiB and 55.66 of 68.72 on 64 GiB (81%)) is the hard edge
above every engine number; only an admin can raise it, and it resets at reboot.
