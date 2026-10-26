# HANDOFF: Qwen3.8-Flash-Next-4bit memory offload (branch lz/qwen4-expert-offload)

## FINAL STATE (agent 4, 2026-10-26 ~20:55) - EXPERT OFFLOAD UNFINISHED, STOPPED BY OWNER DECISION
- Expert offload code stays on the branch, default OFF, UNFINISHED. Do not resume without a new owner ask.
- e9c76e8e5 was tested: it was not enough. The engine's gate+up fusion ran after the fill, copied the slot
  stacks (+42 GB idle, 105.8 GB at a 66 GB budget) and misses loaded into dead buffers. Fixed in 3c8421965
  (fuse before the fill + bound-stack guard). After it: 66 GB budget -> 65.22 GB idle with MTP, no refusals.
- Parity MTP off: ref/e73/e70/e66/e60 x 6 prompts x 2 reps = 60 outputs, all shas identical.
- Speed: see docs/engineering/performance/2026-10-26-qwen4-ple-expert-offload.md. MTP on, e66: 75-90% of
  PLE-only; the floor is the per-layer router sync (all-resident e73 is already -15-25%).
- --qwen4-expert-async tried and reverted (slower, not bit-identical). LRU vs LFU, prompt lookup off/on and
  prefetch were NOT run.
- Workmule cleanup done: ~/models-local/Qwen3.8-Flash-Next-4bit, ~/q4bench, the prefix_cache dir deleted.

Written 2026-10-26 ~19:30 by the first agent. It hit the 182k context guard right after loading the
`goose-mlx-inference` + `node-dispatch` skills (their text alone is ~150k tokens). NO CODE WRITTEN YET.
Next agent: do NOT load those two skills whole again; the facts you need from them are below.

## Goal
Run Qwen3.8-Flash-Next-4bit (model_type qwen4_exp) on the owner's MacBook (M4 Max 128 GiB, Metal
recommendedMaxWorkingSetSize 115,448,725,504 B) with ~66-67 GB resident (weights + KV); the rest read
from the local internal SSD. Full model does NOT fit today (148 GB process peak upstream); don't re-prove.

## Done so far
- Worktree /Users/workhorse/Projects/Rapid-MLX-wt-offload, branch lz/qwen4-expert-offload @ 03fa3b38f
  (= tag v0.14.3-lz.14, package `rapid_mlx`). Clean.
- Model copy to workmule's INTERNAL disk started: node-run job `workhorse-20261026-192939-54e9`
  (heavy slot on workmule): rsync ~/.goose/models/rapid-mlx/Qwen3.8-Flash-Next-4bit -> ~/models-local/.
  Check: `~/Projects/machine-sync/bin/node-jobs status workhorse-20261026-192939-54e9` / `log ... -n 5`.
  98 GiB over the Thunderbolt NFS mount. MUST be deleted when the work is done (report it).

## STATE after agent 2 (2026-10-26 ~20:00) - CODE BUILT, UNIT-TESTED, NOT YET RUN ON THE REAL MODEL
Agent 2 also hit the 160k guard (the system prompt alone is ~100k: keep EVERY tool output tiny).
- Commits on lz/qwen4-expert-offload: fe3a3494d (module + CLI + tests) and the next one (status endpoint +
  this file). Pushed. Workmule worktree exists: ~/Projects/Rapid-MLX-wt-offload (git pull there).
- rapid_mlx/models/qwen4_offload.py (env-driven, default off):
  * PLE: OffloadedShardedEmbedding replaces layer-1 ngram_embedding; np.memmap of the 128x3 shard tensors,
    unique ids -> threaded fancy-index gather -> mx.dequantize(gs32,b4) on GPU -> take(inverse). Bit-exact
    vs QuantizedEmbedding (unit test). sanitize drops `...ngram_embedding.shard_*` keys.
  * Experts: install_into_text_model() (called in Qwen4ExpTextModel.__init__, main model only, NOT the MTP
    head) swaps each layer's switch_mlp for SwitchGLU(...,1); sanitize stubs main-model expert tensors to
    zeros [1,...]. On first MoE call ExpertOffloadManager.setup() picks C slots/layer from
    budget - mx.get_active_memory() - reserve (2 GB default), allocates [C,...] u32/u16 stacks (bf16 via
    .view, shares buffer), fills them by os.preadv straight into the Metal buffer (np.asarray(memoryview(arr))
    is writable; gather_qmm sees CPU writes - verified on workmule). Works with fused gate_up (moe_fusion)
    and split layout. SparseMoeBlock.__call__ now: offload.routed(...) or self._routed(...) (refactor only).
    Per layer: np.asarray(indices) sync -> ensure() loads misses (LRU or decayed-LFU victims, never evicting
    experts needed by the same call) -> slot ids -> unmodified SwitchGLU/gather_qmm. >C distinct experts in
    one call (prefill) -> grouped passes, mx.where row select, mx.eval between groups (exact).
  * Stats: hits/misses/hit_rate/grouped_calls/load_seconds, counts [48,512] -> STATS_OUT .npy (every 4096
    calls), STATS_IN seeds top-C per layer. /v1/status has "qwen4_offload".
- Env: RAPID_MLX_QWEN4_{PLE_OFFLOAD=1, EXPERT_BUDGET_GB, EXPERT_SLOTS, EXPERT_POLICY=lru|lfu,
  EXPERT_RESERVE_GB, EXPERT_STATS_IN, EXPERT_STATS_OUT, IO_THREADS=8, OFFLOAD_DIR=<ckpt dir>}.
  CLI (serve only): --qwen4-ple-offload --qwen4-expert-budget-gb --qwen4-expert-slots --qwen4-expert-policy
  --qwen4-expert-reserve-gb --qwen4-expert-stats-in/out (cli.py _add_qwen4_offload_args /
  _apply_qwen4_offload_env, sets OFFLOAD_DIR from the model dir).
- Tests: tests/test_qwen4_offload.py, 8 pass (run: PYTHONPATH=$PWD ~/Projects/Rapid-MLX/.venv/bin/python
  -m pytest -q tests/test_qwen4_offload.py -p no:cacheprovider).
- Workmule python with mlx 0.32.4 / mlx_lm 0.31.3 / lz.14 deps: ~/.cache/uv/archive-v0/7bolWjpGXF64QzWe/bin/python
  (run fork code with PYTHONPATH=~/Projects/Rapid-MLX-wt-offload).
- Model copy on workmule was at 54/98 GiB at 19:58 (job workhorse-20261026-192939-54e9).

## STATE after agent 3 (2026-10-26 ~20:10 EET) - REAL-MODEL RUNS STARTED
Model copy on workmule ~/models-local/Qwen3.8-Flash-Next-4bit DONE and VERIFIED (all weight/config SHA256 match
SHA256SUMS.txt; README.md differs from the sums file but equals the SSD source = edited after sums). DELETE at end.
Commits: 6286abc86 bench client, run script (+/health/ready, MTP k=2), 0f.. gather threading fix, bench cold
prompts, slot-fill-at-load hook (tokenizer.py, both load paths: e9c76e8e5 is the one qwen4_exp uses).

### Tools (in repo)
- rapid_mlx/bench/qwen4_offload_run.sh <label> <outdir> <prompts> [serve args]: starts OUR server on 8197
  (app argv + MTP k=2), waits /health/ready, runs the client, kills its own pid. env MTP=0, REPEAT=N, MTP_K.
- rapid_mlx/bench/qwen4_offload_bench.py: prompts code,prose,json,zh,long(7840 tok),long1,long2 (cold PLE),
  copy (768 tok copy-heavy). One line per prompt: sha, ttft, prefill/decode tok/s, hit_rate, ple rows/s, active GB.
- Launch (copy exactly; single quotes keep $HOME remote; ~ in double quotes expands LOCALLY = trap):
  R='$HOME/Projects/Rapid-MLX-wt-offload/rapid_mlx/bench/qwen4_offload_run.sh'
  ~/Projects/machine-sync/bin/node-run --host workmule --heavy --need-mem 80 --max-rss-pct 92 --no-wait \
    --timeout 2400 -- "$R e66 ~/q4bench code,prose,json,zh,long,copy --qwen4-ple-offload --disable-prefix-cache --qwen4-expert-budget-gb 66"
  Results: workmule ~/q4bench/<label>.json + <label>.server.log. Wait loop: node-jobs status <id> | awk '/ status /'.
  TRAPS: node-run default kills a job at 50% RAM RSS (64 GB) -> always --max-rss-pct 92; --need-mem >~93 queues
  forever (use --no-wait); git pull on workmule after every push (ssh workmule 'cd ~/Projects/Rapid-MLX-wt-offload && git pull -q').

### Measured (workmule M5 Max 128 GB, MTP k=2 on, --disable-prefix-cache unless noted)
| config | idle active GB | decode tok/s code/prose/json/zh/long/copy | long prefill (7840 tok) |
| fully resident (no offload) | 104.25 (peak 108.1 w/ 8k ctx) | 65.7/52.3/75.7/54.0/51.7/114.2 | 7.18 s |
| PLE offload only (ref) | 72.25 | 64.3/51.2/63.2/55.9/50.7/118.8 | 6.95 s (PLE gather 2.0 s cold, 0.23 s warm) |
- FULL MODEL FITS a 128 GB Mac today (104 GB active, peak 108 GB, Metal ceiling 103.9 GB per guard is the
  limit for context!). The "148 GB peak" premise in this file is NOT reproduced on lz.14 + workmule.
- PLE offload: -32 GB, decode ~= resident AFTER the gather-threading fix (before it: 36 tok/s, -35%, cold page
  faults serial for decode-size gathers). 32 IO threads no better than 8 (cold 2.3 s vs 2.0 s) -> keep 8.
- MTP k=3 is rejected by the engine for Flash-Next ("supports 1-2"); goose would pass 3 but only when
  mtp.safetensors exists, which this checkpoint lacks (MTP is in the shards) -> goose serves it WITHOUT MTP. Report.
- DETERMINISM: MTP OFF greedy is run-to-run deterministic (prose/json/zh rep0==rep1). MTP ON is NOT (prose,
  json, zh differ between reps of the same server and between fresh servers; code/long/copy stable) - pre-
  existing engine behaviour, independent of offload. PLE offload vs resident: identical shas wherever the
  reference itself is stable (code 634481b917e88300, copy e241cf727b736e1b; det runs prose/long matched exactly).
  => exact parity must be tested with MTP=0 (bench env MTP=0); MTP-on runs report match-rate only.
  Persistent prefix cache (~/.cache/rapid-mlx/prefix_cache/--Users--workmule--models-local--Qwen3.8...) makes
  fresh servers "prefill" 8k in 0.25 s: always --disable-prefix-cache for benches; delete that dir at the end.
- Expert offload 66 GB: setup OK (448/512 slots/layer, 59.45 GB experts, filled 6.4 s, active 63.75 GB, warmup
  hit rate 0.887) BUT every request was refused by the memory guard (step=59.54 GB learned during warmup,
  because slots were allocated on the first MoE call). Fixed in e9c76e8e5 (fill at load). NOT YET RE-RUN.
  Check the server log: "[qwen4-offload] experts:" must come BEFORE "Hybrid model: running full request warmup";
  if MTP head is injected later than the hook, budget overshoots by ~1.5 GB (reserve 2 GB covers it) - check.
- RSS of an expert-offload server is ~60 GB (CPU-written slot pages count as RSS) - fine with --max-rss-pct 92.

## NEXT (in order)
1. Re-run e66 (command above). If refused again, grep server log for "memory_guard" lines (step=, idle base).
2. MTP=0 parity: ref0 (PLE only, MTP=0) vs e73/e70/e66/e60 MTP=0 on code,prose,json,zh,long,copy -> shas must
   match exactly. Then MTP-on perf runs at 60/66/70/73 (tok/s, per-prompt hit_rate, load_s), LRU vs LFU at 66.
3. Prompt lookup off vs on at 66: RAPID_MLX_MTP_PROMPT_LOOKUP=0 env prefix on the copy prompt (default on;
   resident copy decode 114-119 tok/s with lookup on).
4. Perf doc docs/engineering/performance/2026-10-26-qwen4-ple-expert-offload.md (tables above + new), goose notes
   worktree (see BUILD), push. Prefetch only if time. Cleanup on workmule: ~/models-local copy, ~/q4bench,
   prefix_cache dir above.

## Measured facts (safetensors headers, 2026-10-26)
- Model dir (SSD): /Volumes/AI-workhorse/models/goose/rapid-mlx/Qwen3.8-Flash-Next-4bit, 98 GiB, 28 shards.
- 48 layers, hidden 2560, 512 routed experts top-10, moe_intermediate 640, shared expert 640, 12 full-attn
  layers (every 4th) + GatedDeltaNet, 2 KV heads x 256 (~25 KB/token KV), 4-bit affine gs64.
- Bytes: routed experts 67.95 GB (per layer gate_up_proj U32[512,1280,320] + down_proj U32[512,2560,80]
  + bf16 scales/biases = 1.416 GB/layer, ~2.77 MB/expert); PLE n-gram tables 32.00 GB; attn/other 1.95;
  mtp head 1.47 (own 512 experts); embed+lm_head 0.72; vision 0.45; shared expert 0.13.
- PLE = `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{0..127}.{weight
  U32[2500012,20], scales BF16[2500012,5], biases BF16[2500012,5]}` + ngram_heads offsets/vocab_sizes;
  config ngram_size 3, heads_per_ngram 8, ngram_vocab_size_base 20,000,000, split_ngram_parts 128.
  ~16 rows/token gathered (~100 B each). Only layer 1 has PLE -> ids known from input tokens at step start.
- Code: rapid_mlx/models/qwen4_exp.py (2137 lines): `ShardedEmbedding` ~1540, PLE module ~1580-1720,
  sanitize renames ~2057-2067; MoE `SparseMoeBlock` ~376 (SwitchGLU gate_up fused, `target_verify` flag
  used by MTP verify).
- Prior art in fork: rapid_mlx/models/deepseek_v41_native/load.py `engram_ssd_offload` (rows from
  checkpoint mmap, PREFETCHED while preceding layers run; doc docs/engineering/performance/
  2026-09-11-deepseek-v41-engram-offload.md: -61 GB peak, -1.0% decode; sync-only was -44%).
  Upstream disk-stream MoE spike: rapid_mlx/disk_stream_patch.py, expert_cache.py (byte-budget LRU),
  offset_reader.py (safetensors pread), registry.py adapters, cli flags --disk-stream /
  --disk-stream-cache-gb (cli.py ~2912, 5693, 6542, 11862) - one expert at a time via quantized_matmul.
  Judge vs a slot-tensor + gather_qmm design (per layer resident [C,...] slot stack, GPU expert->slot
  remap table, misses pread into slots); measure.
- Web prior art to cite (search yourself): github.com/Anemll/flash-moe (+ danveloper original: SSD
  streaming, OS page cache, prefetch backfired on Apple Silicon), ssd-moe deepseek-v4-flash-mlx (LRU,
  ~5 tok/s on 48 GB), Apple "LLM in a flash", mlx-expert-sniper, SwiftLM. VERIFY (don't repeat) the
  llama.cpp claim: Gemma-3n per-layer embedding kept on CPU via
  `--override-tensor per_layer_token_embd.weight=CPU`, ~25% RAM saved, no noticeable slowdown.

## Hardware rules
- Dev/bench host `workmule` (ssh alias; M5 Max 128 GB, idle, 604 GiB free internal). ~/.goose/models
  there is NFS of the SSD over TB -> bench from ~/models-local copy only.
- Heavy jobs via ~/Projects/machine-sync/bin/node-run --host workmule [--heavy|--light] --timeout N --
  '<cmd>' (a hook DENIES raw heavy ssh). node-jobs status|log|cancel <id>. One heavy job per Mac in
  practice for benches (don't overlap with the copy for timing runs).
- Workhorse (96 GB, busy renders): no benching. Laptop (`ssh laptop`): ONLY the final test, only when
  node-probe shows it idle; it has ~34 GiB free disk, so a local copy needs ~100 GB freed by owner: if
  still true at the end, report instead of testing.
- Bulky outputs -> /Volumes/AI-workhorse/runs or experiments (run ~/Projects/machine-sync/bin/ai-disk-guard
  --need-gb N --out <dir> first). Never touch port 8090 or any engine you did not start; never pkill by
  pattern; never touch other Rapid-MLX worktrees.

## Correctness references
(a) PLE: bit-exact row parity resident-gather vs disk-gather on random + real token ids; plus the
PLE-offload-only run on the mule with all experts resident (~72-74 GB) = end-to-end greedy reference.
(b) experts: greedy output of expert-offload at 66/70 GB budgets token-identical (SHA) to (a), several
prompts (code, prose, JSON, Chinese, one long-context), MTP ON as goose serves it.
App argv (goose build_serve_command) for qwen4_exp: `rapid-mlx serve <dir> --text-only
--max-concurrent-requests 8 --enable-prefix-cache --hybrid-cache-entries 24` + MTP
`--speculative-config '{"method":"mtp","model":"<dir>","num_speculative_tokens":3}'` when mtp.safetensors
exists; lz.14 is Guarded => NO --cache-memory-percent. Run fork code with
`PYTHONPATH=<worktree> <venv python> -m rapid_mlx.cli serve ...` (uvx caches wheels by mtime - trap).
Engine logs are UTC.

## BUILD (every behaviour a tunable CLI flag, default OFF)
1. PLE offload: 128 ngram shards out of Metal; mmap safetensors region on local disk; gather + dequant
   rows on CPU (or small staging buffer) -> [tokens, dims] to GPU; prefetch overlapped with layer 0.
   Flags: on/off, optional row-cache cap. Measure MLX active/peak, load time, decode tok/s (short + ~8k
   prompt), parity.
2. Routed-expert offload to `--resident-budget-gb` (unset default; laptop target 66): dense/attn/shared/
   router/KV resident; experts LRU and/or frequency (tunable), optionally seeded from routing stats on a
   calibration set (record stats file); cold experts pread; try next-layer prefetch (layer L+1 router on
   layer L hidden), keep only if faster. MTP verify/multi-token = union of experts. Measure tok/s + hit
   rate at 60/66/70/73 GB and fully cached; prompt-lookup copy-draft ON vs OFF
   (RAPID_MLX_MTP_PROMPT_LOOKUP=0/1) on a copy-heavy prompt.
3. Model-free unit tests (row parity, budget accounting, slot remap) + perf doc
   docs/engineering/performance/2026-10-xx-qwen4-ple-expert-offload.md with tables. Commit + push
   (`git push -u origin HEAD:lz/qwen4-expert-offload`).
Also: dated facts file ~/Projects/goose/local-edition/skills/goose-mlx-inference/references/
qwen4-flash-offload.md, committed ONLY in own goose worktree:
`cd ~/Projects/goose && git worktree add --no-track -b flash-offload-notes ~/Projects/goose-wt-offload
origin/main`, push. Do NOT bump goose's engine pin or tag a release.

## Commit rules
Git identity repo-local leanzero.srl <office@leanzero.net>. Both repos PUBLIC: commit-msg hook REJECTS
any Co-Authored-By / AI-credit line - omit it. Commit every change. At ~160k context: refresh this file,
commit, push, stop.

## Report back (to coordinator)
Flags, commits, branch; resident GB and tok/s per config; hit rates; parity hashes; what the laptop
needs; honest confidence; is tok/s usable; the model copy on workmule to delete.
