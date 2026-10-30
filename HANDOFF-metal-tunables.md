# HANDOFF: engine memory tunables + context-aware auto Metal cap (2026-10-30 ~19:40, agent hit 350k guard)

Run dir (CAUSE, VIGIL, INTERFACE, measure.sh): /Volumes/AI-workhorse/runs/2026-10-30-flash-fixes-verify/metal-limit/
Owner spec (coordinator messages): REAL tunables per Mac (goose Settings), dynamic free-memory floor, macOS wired-limit
button via admin prompt + opt-in LaunchDaemon, live panel, validation, tests per tunable, independent reviewer, readiness
gate + bench parity rows + launcher enforcement. Final report <= 5 lines with numbers.

## DONE
- Step 1 (read-only, mule live lz.18 :8291): guard ceiling 103.9 GB = 0.90 x 115.45 GB working set; margin 2.0 GB;
  active 94-96 GB, peak 102.12 GB; hw.memsize 137.44 GB; iogpu.wired_limit_mb 0. CAUSE.md has the chain + the
  "others" finding (24.5 GB = GPU-driver wired + anon; the "unused 25-30 GB" is mostly PLE memmap FILE CACHE).
- Fork branch lz/metal-tunables (this worktree), pushed, tip 3ed994de9 (on lz.19 = 813e97e0b):
  memory_budget.py tunables()/validate_tunables()/floor_room_bytes(), plan_metal_limit(need_bytes, limit_bytes,
  floor, ceiling, headroom); batched.py second pass _raise_metal_budget_for_context + metal_budget_snapshot;
  scheduler.py memory_plan_need_bytes, memory_budget_status, _memguard_floor_ceiling + floor hold
  (_memguard_floor_hold/_memguard_take_hold), --max-model-len in _memguard_native_context + admission;
  memory_guard.py margin_bytes_override + floor_holds counter; errors.py limit="max_model_len";
  routes/health.py /v1/status memory_budget; cli.py flags (_MEMORY_TUNABLE_FLAGS). Decision record:
  docs/engineering/decisions/2026-10-30-memory-tunables.md (each default with its receipt).
- Tests: tests/test_metal_tunables.py (28: Flash-Next 360k regression 103.9 -> 112.0, 64/96/128 per device,
  CLI->env->engine per flag, validation) + tests/test_memory_guard_floor.py (8: floor under a 900/1000 MB grab
  incl. control arm, wait->refuse, max_model_len, need grows, reserve fraction). Memory suite 632 passed.
  Test venv: ../Rapid-MLX/.venv/bin/python -m pytest -q -p no:cacheprovider <files>.
- Live workhorse boot (Llama-3.2-1B, flags --max-model-len 65536 --memory-floor-gb 8 --memory-guard-margin-gb 1):
  memory_budget populated, floor binds (others 40.7 GB), 80,035-token prompt refused naming --max-model-len.
- loop-state 2caf694: memory_tunables_gate.py (15 rows, cross-checks /v1/status memory_budget) wired into
  bench_dispatch.mjs + start_bench.mjs (exit 6), test_memory_tunables_gate.py 8 pass. `emit --engine-port P` prints rows.
- Device specs measured: mule/laptop 137.44 GB, ws 115.45, wired 0; workhorse 103.08, ws 83.49; Gabee 68.72, ws 55.66.

## OPEN (do in order)
1. Goose lane: a fresh general-purpose agent (spawned by this agent) builds the goose Settings tunables, branch
   lz/memory-tunables in a goose worktree, merges to goose main itself; told: default floor = round(0.05 x physical GB)
   (8% would cap the mule at 101.9 < 103.9), Inference box preset floor 4 GB, Workstation 8%. Check
   `git -C ~/Projects/goose log origin/main --oneline -5` for its merge; if absent, read its worktree HANDOFF.md.
2. Mule measurement when the Gauntlet engine (:8291, lz.18) is gone (`ssh workmule 'pgrep -fl "rapid-mlx serve"'`):
   via node-run on workmule, long timeout:
   `~/Projects/machine-sync/bin/node-run --host workmule --timeout 10800 -- 'zsh /private/tmp/gverify/measure.sh 3ed994de9 409600'`
   (boots lz.19 and lz.20 with the app argv -> *.models.json/*.status.json in /private/tmp/gverify/out/mlimit;
   then REPLAY session1.json growth on lz.20 to 409,600 -> out/mlimit-lz20/grow.txt). Report ceiling (lz.19 103.9 ->
   lz.20 memory_budget.limit_bytes), max admitted (/v1/models max_model_len + max_context_at_ceiling), peak GB
   (grow.txt peak), idle others_bytes. Copy results to the run dir (scp from workhorse). Before: fix1/lz18 replay
   (peak 103.26, 372,016 no refusal).
3. Tag v0.14.3-lz.20 at the final fork tip (git -c user.name=leanzero.srl -c user.email=office@leanzero.net tag; push),
   pin in goose crates/goose-sidecar/src/engine.rs ENGINE_LAUNCHER (currently pins lz.18; add the doc line
   "v0.14.3-lz.20 (fork lz/metal-tunables) = lz.19 + memory tunables ..."), run `cargo test -p goose-sidecar`, commit,
   push. NO AI/Co-Authored trailers.
4. Skills: goose-mlx-inference SKILL.md gate step 3 + 5 and references/readiness-gate.md step 5: device-spec check
   first (`sysctl hw.memsize iogpu.wired_limit_mb`, MLX working set, engine resolved limit = /v1/status
   memory_budget.limit_bytes, `memory_pressure`), the fleet table above, every cap layer (physical -> iogpu working
   set -> engine cap (auto 0.90..0.97 / --metal-limit-gb) -> floor room physical-others-floor -> guard margin),
   flag unused RAM, each tunable's goose config key (from the goose lane) + engine flag + Settings location, and the
   procedure: read spec, set every tunable on purpose, write values in the parity file (`memory_tunables_gate.py
   emit`), verify /v1/status. goose-benchmark-iteration SKILL.md: add a "memory tunables (local)" row to GATES IN
   CODE and the parity rows. Both skills are symlinks into ~/Projects/goose/local-edition/skills (commit there).
5. Owner recommendation (wired limit): mule 122000 MB = 127.9 GB working set:
   `sudo sysctl iogpu.wired_limit_mb=122000` (resets at reboot). It adds nothing while "others" ~24 GB: the RAM binds
   at ~109 GB with a 4 GB floor. Measure idle others in step 2 before recommending.

## TRAPS
- The guard's "others" counts GPU-driver wired memory the engine causes: the floor is honest about real free memory
  but binds earlier than Activity Monitor suggests.
- lz.19 tag exists (cache reclaim) but goose main pins lz.18.
- Do not touch the mule while the Gauntlet engine runs (read-only GETs only).

## GOOSE LANE STATE (agent returned at its own 350k guard, 2026-10-30)
Worktree ~/Projects/goose-memory-tunables, branch lz/memory-tunables, NOT merged/pushed. Commits d2e9f6bed3 (sidecar argv
per tunable), 5a553eac88 (ACP fields + memory_budget + wired-limit read/apply/LaunchDaemon), 51bab82038 (desktop Memory
tab, 16 locales), 2e0451c552 (SDK regen), 39c7168fe2 (its HANDOFF.md: exact commands). Config keys under
mlx_engine.memory. lz.20-only flags sent only when the launcher pins >= v0.14.3-lz.20 (else mount fails naming it).
Open there: re-run clippy + regen acp-schema/SDK after the 5% commit, UI vitest + full test:run (Memory tab may break
tab-list tests), the live CDP check (one Goose app per Mac), the independent reviewer, rebase + merge + push.
Known unrelated failure: key_connection::only_a_refused_loopback_connect (verify on clean main).
