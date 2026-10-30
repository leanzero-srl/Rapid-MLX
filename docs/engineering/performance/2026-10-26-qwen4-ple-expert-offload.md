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

NOT RUN at e66 (expert offload stopped by owner decision). Resident reference with lookup on: copy 119.6
tok/s. Lookup off vs on with PLE offload at 8k-64k: see "Long context" below.

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

## Long context (8k / 32k / 64k): PLE offload and prompt lookup (2026-10-27)

workmule (M5 Max 128 GB, nothing else running), branch @450bd19e5, models on
the internal disk, expert offload OFF, `--disable-prefix-cache`, app argv.
Three frozen prompts per length (built with the Flash tokenizer from this repo):
code_edit = fork .py files with one target file
(`rapid_mlx/bench/qwen4_offload_bench.py`) in the middle, "rename `_offload`
to `_read_status`, return the FULL file" (2,226 output tokens); doc_quote =
docs/*.md, "summarise ~400 words, quote 8 sentences verbatim" (~600-950);
free_chat = 53/155/301-message history, "explain in ~500 words, refer back"
(~460-800). Greedy, thinking off, streamed. N=3 at 8k, N=2 at 32k/64k (median).
Decode tok/s = (tokens-1)/(last-first content chunk). Raw data, prompts, client
and server logs: `/Volumes/AI-workhorse/runs/2026-10-26-longctx-ple-pld/`.
Flash MTP k=2, 27B MTP k=3 (as goose serves it). PLD =
`RAPID_MLX_MTP_PROMPT_LOOKUP` set explicitly in every config.

Decode tok/s, code_edit / doc_quote / free_chat at 8k, 32k, 64k:

| config | code_edit | doc_quote | free_chat |
|---|---|---|---|
| Flash resident, MTP off | 38.4 / 35.4 / 31.1 | 38.7 / 35.7 / 27.9 | 38.8 / 35.6 / 31.6 |
| Flash PLE, MTP off | 38.8 / 35.7 / 31.8 | 39.0 / 35.6 / 32.1 | 39.0 / 35.3 / 32.0 |
| Flash resident, MTP + PLD on | 107.1 / 87.0 / REFUSED | 50.0 / 43.7 / REFUSED | 45.8 / 41.7 / REFUSED |
| Flash PLE, MTP, PLD off | 57.2 / 50.0 / 43.6 | 47.0 / 44.8 / 38.4 | 45.8 / 42.7 / 37.7 |
| Flash PLE, MTP, PLD on | 105.9 / 87.0 / 72.2 | 49.0 / 44.7 / 38.5 | 46.0 / 42.7 / 38.0 |
| 27B-Atlassian-Q8, MTP, PLD off | 36.3 / 35.0 / 31.9 | 30.2 / 26.1 / 22.2 | 27.0 / 22.7 / 21.2 |
| 27B-Atlassian-Q8, MTP, PLD on | 157.5 / 140.1 / 119.5 | 30.1 / 26.0 / 22.2 | 27.1 / 22.5 / 20.3 |

(Flash resident MTP-off doc_quote 64k: reps 31.9 and 24.0, the low one is noise.)

TTFT (prefill tok/s), median over workloads:

| config | 8k | 32k | 64k |
|---|---|---|---|
| Flash resident | 5.0 s (1,555) | 37 s (870-920) | 103 s (630) |
| Flash PLE | 4.9 s (1,585) | 24.5 s (1,320-1,350) | 58 s (1,125) |
| 27B-Atlassian-Q8 | 8.1 s (955) | 37 s (870) | 85 s (767) |

Memory (MLX active idle / peak during the L block; server RSS):

| config | active | peak 8k / 32k / 64k | RSS GB |
|---|---|---|---|
| Flash resident, MTP off | 102.8 | 105.9 / 106.2 / 106.2 | 14.8-15.1 |
| Flash resident, MTP on | 104.2 | 107.3 / 107.5 / refused | 16.8-16.9 |
| Flash PLE, MTP off | 70.8 | 73.9 / 74.7 / 76.5 | 31.6 / 39.0 / 44.4 |
| Flash PLE, MTP on | 72.2 | 75.3 / 76.1 / 77.9 | 29.5-45.5 |
| 27B, MTP, PLD off / on | 31.8 | 35.1 / 36.8 / 39.1 (on: 37.5 / 39.3 / 41.6) | 30.4-30.7 |

RSS does not count the resident Flash weights (Metal buffers); for the PLE
configs it grows with the page cache of the memmapped n-gram rows the prompts
touch (25 GB idle, 44 GB after 64k prompts). That is reclaimable file cache,
not wired memory.

PLD counters per request (drafted / accepted, from `/v1/status`
`mtp_prompt_lookup`): Flash code_edit ~1,905 / 1,894 (99 %) at every length;
27B code_edit 2,187 / 2,097 (96 %). doc_quote: Flash 8-13 drafted (50-80 %
accepted), 27B 34-138 (46-57 %). free_chat: 0-5 drafted.

### Findings

1. PLD is the long-context lever for copy-heavy output, and it keeps paying at
   64k. Flash PLE + MTP: code_edit 1.85x / 1.74x / 1.66x over PLD off at 8k /
   32k / 64k (105.9 vs 57.2, 87.0 vs 50.0, 72.2 vs 43.6 tok/s). 27B: 4.3x /
   4.0x / 3.7x (157.5 vs 36.3 ... 119.5 vs 31.9). The ratio shrinks with
   length, likely because every step's attention cost grows with the context
   while the tokens per step stay the same; acceptance does not drop (it is flat at 99 % / 96 %).
2. PLD neither helps nor hurts summarise/quote or chat: within +-3 % at every
   length on both models (27B free_chat 64k 20.3 vs 21.2 is the largest gap,
   inside the run-to-run spread of these MTP rows). It drafts almost nothing
   there.
3. Fully resident Flash with MTP is REFUSED at 64k by the memory guard: "needs
   112.2 GB but this Mac can hold 112.0 GB; the maximum context here is 62,062
   tokens". MTP off it is admitted at 64k, but the guard halves the prefill
   chunk 102 times per 64k request (19 at 32k), so prefill drops to 630 tok/s.
   With PLE offload there are no halvings: prefill 1.8x faster at 64k (58 vs
   103 s TTFT) and 1.5x at 32k, with ~34 GB to spare (peak 77.9 GB). Admission is learned state:
   a fresh resident server with `--prefill-step-size 128` refused 64k MTP off
   too (max 63,996 tokens).
4. PLE offload decode = resident decode at every length (MTP off 31.8 vs 31.1
   at 64k code_edit).

### Identity verdicts

- code_edit: all 47 successful outputs in every config, both models, every
  length is byte-identical (sha 5aea2712, 2,226 tokens): resident vs PLE, MTP
  off vs on, PLD off vs on, all reps. It is the correct edit except the known
  streaming defect that drops the space before `[` / `"[` (4 lines; the same in
  every config, see the 2026-10-26 PLD run notes).
- (a) PLE offload vs resident, MTP off: 8k identical on all three workloads.
  At 32k/64k doc_quote and free_chat differ (from char ~200-270), each config
  stable across its own reps. Cause: the memory guard shrinks the resident
  server's prefill chunks (see finding 3), PLE runs ~1,500-2,048-token chunks,
  and chunk size changes the numerics. Control with a fixed
  `--prefill-step-size 128` on both (0 halvings): 32k doc_quote and free_chat
  byte-identical (b3198082, 82f6af62), and 48k too (010052e8, a7fbe959).
  64k and 60k could not be controlled: a fresh resident server with step 128
  refuses them (60k: "maximum context here is 51,200 tokens"); PLE alone runs
  them. Verdict: PLE offload is output-identical to resident up to 48k
  measured; no evidence of a PLE-caused difference at any length.
- (b) PLD on vs off: not testable byte-for-byte outside code_edit, because MTP
  is not repeatable. Repeat control (same config, same server): doc_quote and
  free_chat reps first differ at char 26-3,147; PLD on vs off first differ at
  char 26-1,791, the same range. Both models. So PLD adds no divergence
  beyond MTP's own; on the one workload where MTP is stable (code_edit), PLD
  on == off.

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
