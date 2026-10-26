# Qwen3.8-Flash-Next-4bit (qwen4_exp): PLE and routed-expert offload

Date: 2026-10-26. Branch `lz/qwen4-expert-offload` (base `v0.14.3-lz.14`).

Status: PLE offload qualified (opt-in). ROUTED-EXPERT OFFLOAD IS UNFINISHED:
work stopped by owner decision on 2026-10-26 ~20:50; the code stays on the
branch, default OFF; LRU vs LFU and prompt-lookup off/on were never run.
PLE offload: -32 GB at resident decode speed, bit-exact.
Routed-expert offload works and is bit-exact at every budget tested, but costs
about a third of decode speed, mostly from the per-layer router sync, not from
disk reads. Both are off by default.

## What changes

- `--qwen4-ple-offload`: the 128 layer-1 n-gram embedding shards (32.00 GB of
  4-bit rows) stay in the checkpoint. Each step gathers the ~16 rows/token it
  needs from an `np.memmap` (8 threads), dequantizes them on the GPU with the
  same `mx.dequantize` as `QuantizedEmbedding`. Unit test: bit-exact rows.
- `--qwen4-expert-budget-gb N` (or `--qwen4-expert-slots C`): each of the 48 MoE
  layers keeps a resident slot stack of C experts (out of 512, ~2.77 MB each)
  inside the same `SwitchGLU`/`gather_qmm`. Per layer the router ids are read on
  the CPU, missing experts are `preadv`'d straight into the Metal buffer of an
  evicted slot (`--qwen4-expert-policy lru|lfu`), and ids are remapped to slots.
  A call that needs more distinct experts than slots (prefill) runs in exact
  grouped passes. C is chosen as `(budget - resident_before - reserve) / bytes`;
  the slot fill happens at load (before the memory guard learns its baseline).
- Stats in `/v1/status` -> `qwen4_offload`: hits, misses, hit_rate, load_seconds,
  calls, sync_seconds (time blocked on the router read), cpu_seconds.

## Bugs found and fixed on the real model

1. Slots allocated on the first MoE call: the memory guard booked the 59 GB fill
   as a per-step transient and refused every request. Fill moved to load
   (`e9c76e8e5`).
2. The engine fuses MoE gate+up AFTER load (`engine/batched.py`); fusion
   concatenates copies, so the stacks were duplicated (+42 GB: idle active
   105.8 GB at a 66 GB budget) and expert loads went into the dead split
   buffers (silently wrong output after the first eviction). Now the hook fuses
   the stub-sized modules before the fill and a guard checks on the first MoE
   call that the bound stacks are still the module's arrays (`3c8421965`).

## Environment

workmule: Apple M5 Max, 128 GB, model on the internal SSD (98 GiB copy).
Prompts: code, prose, json, zh (256 new tokens each, json stops at 170), long
(7,840-token prompt), copy (768 new tokens, copy-heavy). Greedy, temperature 0,
`--disable-prefix-cache`, REPEAT=2 (rep0 cold, rep1 warm). App argv
(`--max-concurrent-requests 8 --hybrid-cache-entries 24`).
NOISE: from ~20:20 an unrelated Remotion render shared workmule (load avg 10);
single reps vary by up to 40%. Compare the better of the two reps, or rows from
the same chain.

## Parity (MTP off; MTP-on greedy is run-to-run nondeterministic even resident)

SHA-256 prefix of the generated text, identical in both reps of every config:

| prompt | PLE-only ref | e73 | e70 | e66 | e60 |
|---|---|---|---|---|---|
| code | 634481b917e88300 | = | = | = | = |
| prose | e5ea8cc87b033280 | = | = | = | = |
| json | 0c03e044c8406910 | = | = | = | = |
| zh | d5077af736836c88 | = | = | = | = |
| long | 93572ea03dae06b4 | = | = | = | = |
| copy | e241cf727b736e1b | = | = | = | = |

5 configs x 6 prompts x 2 reps = 60 outputs, 0 mismatches, including e60 at a 94.5% hit rate on
the long prompt (11k misses).

## MTP off: memory and decode tok/s (better rep)

| config | idle active GB | slots/layer | code | prose | json | zh | long | copy | hit rate (decode prompts) |
|---|---|---|---|---|---|---|---|---|---|
| PLE only (ref) | 70.78 | 512 (resident) | 43.0 | 43.2 | 43.2 | 43.2 | 39.2 | 40.8 | - |
| e73 | 70.78 | 512 | 33.4 | 33.5 | 33.5 | 33.4 | 31.2 | 32.2 | 1.000 |
| e70 | 67.86 | 490 | 32.2 | 31.3 | 31.6 | 31.8 | 28.2 | 31.1 | 0.995-0.999 (long 0.992) |
| e66 | 63.75 | 459 | 27.9 | 28.3 | 30.1 | 30.1 | 29.7 | 27.7 | 0.987-0.994 (long 0.977) |
| e60 | 57.78 | 414 | 28.0 | 27.6 | 25.4 | 30.3 | 29.2 | 30.2 | 0.976-0.984 (long 0.945) |

Where the time goes (e73: every expert resident, 0 misses): 48 router syncs per
token cost ~23% (33.4 vs 43.0 tok/s, ~146 us per layer of lost GPU pipelining).
Disk loads are small: e66 code 1.4 s cold / 0.5 s warm over 256 tokens; e60 2.3
/ 0.8 s. CPU bookkeeping is ~10 us/call.

Rejected: `mx.async_eval` of each MoE block right after its sync (to overlap
graph building with expert compute): slower (29.9 vs 33.0 tok/s at e73) and the
long-prompt output changed (7fbeae49 vs 93572ea0). Reverted (`18e0f6369`).

## MTP on (k=2, as served): decode tok/s and hit rates

Better of two reps; hit rate from the warm rep. Same chain, same noise.

| config | idle active GB | code | prose | json | zh | long | copy | hit rate (long) |
|---|---|---|---|---|---|---|---|---|
| PLE only (ref) | 72.25 | 63.3 | 51.7 | 73.7 | 55.9 | 51.1 | 119.6 | - |
| e73 (512 slots) | 72.25 | 53.5 | 43.0 | 52.7 | 46.5 | 41.5 | 106.3 | 1.000 |
| e70 (490) | 69.33 | 48.7 | 41.4 | 49.3 | 44.9 | 43.0 | 104.9 | 0.996-0.999 (0.993) |
| e66 (459) | 65.22 | 46.6 | 39.1 | 50.5 | 43.0 | 41.3 | 96.8 | 0.988-0.992 (0.978) |
| e60 (414) | 59.24 | 39.5 | 32.8 | 42.9 | 29.3 | 35.0 | 82.1 | 0.974-0.977 (0.944) |

The MTP head (1.47 GB) loads after the slot fill and lands inside the 2 GB
reserve (66 GB budget -> 65.22 GB idle). LRU vs LFU at 66 GB: NOT RUN (stopped).

## Prompt lookup off vs on (e66, copy prompt)

NOT RUN (stopped by owner decision). Resident reference with lookup on: copy 119.6 tok/s.

## Verdict

- PLE offload: use it. -32 GB (104 -> 72 GB) at resident decode speed, exact.
- Expert offload (UNFINISHED): exact at 60-73 GB, but decode drops to ~75-90% of
  the PLE-only reference at 66 GB with MTP on (code 46.6 vs 63.3, prose 39.1 vs
  51.7), ~65-70% with MTP off. The floor is the per-layer router sync (all experts
  resident still loses ~15-25%), so a bigger cache cannot recover it; removing the
  sync needs a different design (e.g. speculative routing with a per-step miss
  check and re-run, viable only at much higher per-step hit rates). Next-layer
  prefetch was not tried: miss loads are only ~5-15% of decode time.
- The full model fits a 128 GB Mac resident (104 GB active), so the offload is
  only needed for long contexts on 128 GB or for smaller Macs.

## Prior art

- flash-moe (Qwen3.5-397B on a 48 GB MacBook, C/Metal): streams the K active
  experts per token with parallel `pread`, relies on the OS page cache (~71% hit),
  reports mmap'd experts 5x slower from cold page faults.
- ssd-moe/deepseek-v4-flash-mlx: active experts streamed into a device-side LRU
  in unified memory (the closest design to this one).
- This fork: `deepseek_v41_native` Engram SSD offload (2026-09-11 doc): row
  gather from the checkpoint mmap, prefetched; -61 GB at -1.0% decode. The PLE
  offload here is the same idea and lands at the same cost.
- Not verified: a llama.cpp recipe keeping Gemma-3n `per_layer_token_embd` on the
  CPU with `--override-tensor`; the general `-ot` / `--n-cpu-moe` flags exist,
  the Gemma-3n-specific claim was not found in a primary source.
