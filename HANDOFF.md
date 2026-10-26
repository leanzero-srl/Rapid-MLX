# HANDOFF: Qwen3.8-Flash-Next-4bit memory offload (branch lz/qwen4-expert-offload)

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
