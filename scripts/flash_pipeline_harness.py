#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint harness for the qwen4_exp pipeline split.

    prompts  build the fixed prompt set (token ids) from the checkpoint's tokenizer
    ref      single-process greedy reference through the fork's normal loader
    pipe     one pipeline rank (run under mlx.launch); optional soak afterwards
    compare  token-by-token diff of two result directories

Every mode writes JSON artefacts; nothing is downloaded.  Greedy decode on
both sides feeds the last prompt token as the first decode step (mlx-lm's
generate_step shape), so every decode logit comes from a one-row lm_head in
both paths.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

from rapid_mlx.distributed import pipeline_qwen4 as pipe

TOP_K = 5
REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _node_snapshot() -> dict:
    node = pipe.measure_node_memory()
    return {
        "time": time.time(),
        "available_gib": round(node.available_bytes / 2**30, 3),
        "free_percent": node.free_percent,
        "pressure_level": node.pressure_level,
        "memorystatus_level": pipe._sysctl_int("kern.memorystatus_level"),
        "mlx_active_gib": round(mx.get_active_memory() / 2**30, 3),
        "mlx_peak_gib": round(mx.get_peak_memory() / 2**30, 3),
        "mlx_cache_gib": round(mx.get_cache_memory() / 2**30, 3),
    }


def _trace_row(logits_row: mx.array) -> dict:
    """Top-k ids, logprobs and the raw top-1/top-2 logit margin of one row."""
    row = logits_row.astype(mx.float32)
    logprobs = row - mx.logsumexp(row, axis=-1, keepdims=True)
    order = mx.argsort(-row)[:TOP_K]
    top_logits = row[order]
    mx.eval(order, top_logits, logprobs)
    ids = order.tolist()
    return {
        "top_ids": ids,
        "top_logprobs": [round(v, 6) for v in logprobs[order].tolist()],
        "margin": float(top_logits[0].item() - top_logits[1].item()),
    }


def _load_prompts(path: Path) -> list[dict]:
    return json.loads(path.read_text())["prompts"]


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------


def cmd_prompts(options) -> int:
    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(Path(options.model).expanduser())

    def chat(text: str) -> list[int]:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=True,
        )
        return list(ids["input_ids"] if isinstance(ids, dict) else ids)

    readme = (REPO_ROOT / "README.md").read_text()
    document_ids = tokenizer.encode(readme)[: options.long_tokens]
    document = tokenizer.decode(document_ids)
    prompts = [
        {
            "name": "chat-capital",
            "kind": "chat",
            "ids": chat(
                "What is the capital of Australia, and why was it chosen over "
                "Sydney and Melbourne?"
            ),
        },
        {
            "name": "chat-long-2k",
            "kind": "chat",
            "ids": chat(
                "Here is a project README:\n\n"
                + document
                + "\n\nSummarize what this project does in five short bullet points."
            ),
        },
        {
            "name": "raw-code",
            "kind": "raw",
            "ids": tokenizer.encode(
                "def fibonacci(n: int) -> int:\n"
                '    """Return the n-th Fibonacci number iteratively."""\n'
            ),
        },
        {
            "name": "chat-tcp-udp",
            "kind": "chat",
            "ids": chat(
                "Explain the difference between TCP and UDP in three sentences."
            ),
        },
        {
            "name": "raw-history",
            "kind": "raw",
            "ids": tokenizer.encode("The history of the Roman Empire began"),
        },
    ]
    out = Path(options.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": str(options.model),
                "eos_token_ids": sorted(tokenizer.eos_token_ids),
                "prompts": prompts,
            },
            indent=1,
        )
    )
    for prompt in prompts:
        print(f"{prompt['name']}: {len(prompt['ids'])} tokens")
    return 0


# ---------------------------------------------------------------------------
# ref: single process, the fork's normal loader
# ---------------------------------------------------------------------------


def _greedy_single(model, ids: list[int], max_tokens: int, prefill_step: int):
    cache = model.make_cache()
    tokens = mx.array([ids], dtype=mx.int32)
    start = time.perf_counter()
    prefix = tokens[:, :-1]
    for offset in range(0, prefix.shape[1], prefill_step):
        model(prefix[:, offset : offset + prefill_step], cache=cache)
        mx.eval([layer.state for layer in cache])
    current = tokens[:, -1:]
    generated, trace = [], []
    first = None
    for _ in range(max_tokens):
        logits = model(current, cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
        mx.eval(token, logits, [layer.state for layer in cache])
        if first is None:
            first = time.perf_counter()
        value = int(token.item())
        generated.append(value)
        trace.append(_trace_row(logits[0, -1]))
        current = token[:, None]
    end = time.perf_counter()
    return (
        generated,
        trace,
        {
            "prompt_tokens": len(ids),
            "ttft_s": first - start,
            "prefill_tok_s": len(ids) / (first - start),
            "decode_tok_s": (max_tokens - 1) / (end - first)
            if max_tokens > 1
            else None,
        },
    )


def cmd_ref(options) -> int:
    from rapid_mlx.utils.tokenizer import load_model_with_fallback

    model_dir = Path(options.model).expanduser()
    out = Path(options.out)
    out.mkdir(parents=True, exist_ok=True)
    before = _node_snapshot()
    args = pipe.load_text_args(model_dir)
    ckpt = pipe.read_checkpoint_bytes(model_dir, args.num_hidden_layers)
    if options.layer_limit:
        args, ckpt = pipe.truncate_layers(args, ckpt, options.layer_limit)
    node = pipe.measure_node_memory()
    one = [pipe.NodeBudget("single", node.total_bytes, node.budget_bytes, "measured")]
    plan = pipe.plan_pipeline(
        args,
        ckpt,
        one,
        context=options.context,
        batch=1,
        prefill_step=options.prefill_step,
    )
    print(pipe.format_plan(plan), flush=True)
    need = plan.stages[0].total_bytes
    usable = node.available_bytes - int(node.total_bytes * pipe.PRESSURE_FLOOR_RATIO)
    if need > usable:
        print(
            f"REFUSED: single-node reference needs {need / 2**30:.2f} GiB, "
            f"measured available - pressure floor = {usable / 2**30:.2f} GiB",
            flush=True,
        )
        return 3
    # The fork's own serving path wires up to the device's recommended
    # working set (mllm_batch_generator.py); the reference does the same.
    mx.set_wired_limit(int(mx.device_info()["max_recommended_working_set_size"]))
    load_start = time.perf_counter()
    if options.layer_limit:
        from mlx_lm.utils import load_model, load_tokenizer

        from rapid_mlx.utils.tokenizer import _register_vendored_archs

        _register_vendored_archs()
        model, _ = load_model(model_dir, lazy=True)
        pipe.slice_model(model, 0, 1, 0, options.layer_limit)
        tokenizer = load_tokenizer(model_dir)
    else:
        model, tokenizer = load_model_with_fallback(str(model_dir))
    mx.eval(model.parameters())
    load_s = time.perf_counter() - load_start
    after_load = _node_snapshot()
    prompts = _load_prompts(Path(options.prompts))
    # Kernel compilation and first-touch paging are not the model's speed.
    _greedy_single(model, prompts[0]["ids"], 4, options.prefill_step)
    results = []
    for prompt in prompts:
        generated, trace, timing = _greedy_single(
            model, prompt["ids"], options.max_tokens, options.prefill_step
        )
        results.append(
            {"name": prompt["name"], "tokens": generated, "trace": trace, **timing}
        )
        print(
            f"{prompt['name']}: {timing['prompt_tokens']} prompt tok, "
            f"prefill {timing['prefill_tok_s']:.1f} tok/s, decode "
            f"{timing['decode_tok_s']:.2f} tok/s | {tokenizer.decode(generated)[:120]!r}",
            flush=True,
        )
    (out / "results.json").write_text(
        json.dumps(
            {
                "mode": "single",
                "load_s": load_s,
                "memory_before": before,
                "memory_after_load": after_load,
                "memory_end": _node_snapshot(),
                "results": results,
            },
            indent=1,
        )
    )
    return 0


# ---------------------------------------------------------------------------
# pipe: one rank under mlx.launch
# ---------------------------------------------------------------------------


def _greedy_pipe(
    stage, guard, rows: list[list[int]], max_tokens: int, prefill_step: int
):
    """Batch of rows; the last rank traces every step of every row."""
    batch = len(rows)
    tokens, padding = pipe._left_pad(rows, 0)
    cache = stage.make_cache(padding if batch > 1 else None)
    start = time.perf_counter()
    prefix = tokens[:, :-1]
    for offset in range(0, prefix.shape[1], prefill_step):
        out = stage.forward(
            prefix[:, offset : offset + prefill_step], cache, logits=None
        )
        pipe._step_sync(stage, out, cache, batch, guard, sample=False)
    current = tokens[:, -1:]
    generated = [[] for _ in range(batch)]
    traces = [[] for _ in range(batch)]
    first = None
    for _ in range(max_tokens):
        out = stage.forward(current, cache, logits="last")
        next_tokens = pipe._step_sync(stage, out, cache, batch, guard, sample=True)
        if first is None:
            first = time.perf_counter()
        if stage.is_last:
            for row in range(batch):
                traces[row].append(_trace_row(out[row, -1]))
        for row, token in enumerate(next_tokens):
            generated[row].append(token)
        current = mx.array(next_tokens, dtype=mx.int32)[:, None]
    end = time.perf_counter()
    width = tokens.shape[1]
    return (
        generated,
        traces,
        {
            "prompt_tokens": [len(row) for row in rows],
            "ttft_s": first - start,
            "prefill_tok_s": batch * width / (first - start),
            "decode_tok_s": batch * (max_tokens - 1) / (end - first)
            if max_tokens > 1
            else None,
        },
    )


def _agree(group, value: int) -> int:
    """Rank 0's value, identical on every rank."""
    contribution = value if group.rank() == 0 else 0
    agreed = mx.distributed.all_sum(
        mx.array([contribution], dtype=mx.int32), group=group
    )
    return int(agreed.item())


def cmd_pipe(options) -> int:
    group = mx.distributed.init(strict=True)
    rank = group.rank()
    out = Path(options.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"rank{rank}.log"
    log_file = log_path.open("a")

    def log(line: str) -> None:
        stamped = f"{time.strftime('%H:%M:%S')} rank{rank} {line}"
        print(stamped, flush=True)
        log_file.write(stamped + "\n")
        log_file.flush()

    model_dir = Path(options.model).expanduser()
    prompts = _load_prompts(Path(options.prompts))
    before = _node_snapshot()
    log(f"before load {json.dumps(before)}")
    load_start = time.perf_counter()
    stage, plan, guard = pipe.load_stage(
        model_dir,
        group,
        context=options.context,
        batch=2 if options.soak_minutes or options.pairs else 1,
        prefill_step=options.prefill_step,
        starts=pipe._parse_starts(options.split),
        layer_limit=options.layer_limit,
        log=log,
    )
    load_s = time.perf_counter() - load_start
    after_load = _node_snapshot()
    log(
        f"loaded layers [{stage.start}, {stage.end}) in {load_s:.1f}s {json.dumps(after_load)}"
    )
    _greedy_pipe(stage, guard, [prompts[0]["ids"]], 4, options.prefill_step)
    results = []
    for prompt in prompts:
        generated, traces, timing = _greedy_pipe(
            stage, guard, [prompt["ids"]], options.max_tokens, options.prefill_step
        )
        results.append(
            {
                "name": prompt["name"],
                "tokens": generated[0],
                "trace": traces[0],
                **timing,
            }
        )
        log(
            f"{prompt['name']}: prefill {timing['prefill_tok_s']:.1f} tok/s decode "
            f"{timing['decode_tok_s']:.2f} tok/s peak {mx.get_peak_memory() / 2**30:.2f} GiB"
        )
    pairs = []
    for first, second in _pairs(prompts) if options.pairs else []:
        rows = [prompts[first], prompts[second]]
        generated, traces, _ = _greedy_pipe(
            stage,
            guard,
            [row["ids"] for row in rows],
            options.pairs,
            options.prefill_step,
        )
        pairs.append(
            {
                "names": [row["name"] for row in rows],
                "tokens": generated,
                "traces": traces if stage.is_last else None,
            }
        )
    summary = {
        "mode": "pipeline",
        "rank": rank,
        "size": group.size(),
        "layers": [stage.start, stage.end],
        "starts": plan.starts,
        "wire_dtype": str(stage.wire_dtype),
        "plan": pipe.format_plan(plan),
        "guard_budget_gib": round(guard.budget_bytes / 2**30, 3),
        "load_s": load_s,
        "memory_before": before,
        "memory_after_load": after_load,
        "memory_end": _node_snapshot(),
        "pairs": pairs,
        "results": results
        if stage.is_last
        else [
            {key: value for key, value in item.items() if key != "trace"}
            for item in results
        ],
    }
    (out / f"rank{rank}.json").write_text(json.dumps(summary, indent=1))

    if options.soak_minutes:
        _soak(stage, guard, group, prompts, results, options, out, log)

    mx.eval(mx.distributed.all_sum(mx.array([1], dtype=mx.int32), group=group))
    log("done")
    return 0


def _soak(stage, guard, group, prompts, baseline, options, out, log) -> None:
    """Alternate single requests and two-request batches; watch memory."""
    rank = group.rank()
    expected = {item["name"]: item["tokens"] for item in baseline}
    deadline = time.time() + options.soak_minutes * 60
    events = (out / f"soak-rank{rank}.jsonl").open("a")
    iteration = 0
    mismatches = 0
    while _agree(group, int(time.time() < deadline)):
        first = prompts[iteration % len(prompts)]
        if iteration % 2 == 0:
            rows = [first]
        else:
            rows = [first, prompts[(iteration + 1) % len(prompts)]]
        started = time.time()
        generated, _, timing = _greedy_pipe(
            stage,
            guard,
            [row["ids"] for row in rows],
            options.soak_tokens,
            options.prefill_step,
        )
        record = {
            "iteration": iteration,
            "batch": len(rows),
            "names": [row["name"] for row in rows],
            "seconds": round(time.time() - started, 3),
            "decode_tok_s": timing["decode_tok_s"],
            **_node_snapshot(),
        }
        if len(rows) == 1:
            same = generated[0] == expected[first["name"]][: options.soak_tokens]
            record["matches_first_run"] = same
            mismatches += int(not same)
        else:
            record["batch_rows_match_single"] = [
                gen == expected[row["name"]][: options.soak_tokens]
                for gen, row in zip(generated, rows)
            ]
        events.write(json.dumps(record) + "\n")
        events.flush()
        if rank == 0 and iteration % 5 == 0:
            log(f"soak {json.dumps(record)}")
        iteration += 1
    log(
        f"soak finished: {iteration} iterations, {mismatches} single-request mismatches"
    )


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def _results(directory: Path) -> tuple[dict, dict]:
    single = directory / "results.json"
    if single.exists():
        data = json.loads(single.read_text())
        return data, {item["name"]: item for item in data["results"]}
    ranks = sorted(directory.glob("rank*.json"), key=lambda p: int(p.stem[4:]))
    data = json.loads(ranks[-1].read_text())
    return data, {item["name"]: item for item in data["results"]}


def cmd_compare(options) -> int:
    _, ref = _results(Path(options.ref))
    _, other = _results(Path(options.pipe))
    report = []
    for name, left in ref.items():
        right = other[name]
        diverge = next(
            (
                i
                for i, (a, b) in enumerate(zip(left["tokens"], right["tokens"]))
                if a != b
            ),
            None,
        )
        entry = {
            "name": name,
            "identical": diverge is None
            and len(left["tokens"]) == len(right["tokens"]),
            "tokens": len(left["tokens"]),
            "max_abs_logprob_diff_before_divergence": 0.0,
        }
        limit = diverge if diverge is not None else len(left["tokens"])
        worst = 0.0
        for step in range(limit):
            a, b = left["trace"][step], right["trace"][step]
            if a["top_ids"][0] == b["top_ids"][0]:
                worst = max(worst, abs(a["top_logprobs"][0] - b["top_logprobs"][0]))
        entry["max_abs_logprob_diff_before_divergence"] = worst
        entry["min_margin_ref"] = min(step["margin"] for step in left["trace"])
        if diverge is not None:
            a, b = left["trace"][diverge], right["trace"][diverge]
            entry.update(
                {
                    "first_divergence": diverge,
                    "ref_top": list(zip(a["top_ids"], a["top_logprobs"])),
                    "pipe_top": list(zip(b["top_ids"], b["top_logprobs"])),
                    "ref_margin": a["margin"],
                    "pipe_margin": b["margin"],
                }
            )
        entry["speed"] = {
            "ref_prefill_tok_s": left["prefill_tok_s"],
            "pipe_prefill_tok_s": right["prefill_tok_s"],
            "ref_decode_tok_s": left["decode_tok_s"],
            "pipe_decode_tok_s": right["decode_tok_s"],
        }
        report.append(entry)
        print(json.dumps(entry))
    if options.out:
        Path(options.out).write_text(json.dumps(report, indent=1))
    return 0 if all(item["identical"] for item in report) else 1


def _pairs(prompts: list[dict]) -> list[tuple[int, int]]:
    """Every neighbouring pair, as the soak's two-request batches formed them."""
    return [(index, (index + 1) % len(prompts)) for index in range(len(prompts))]


def _load_truncatable(model_dir: Path, layer_limit: int | None):
    from mlx_lm.utils import load_model

    from rapid_mlx.utils.tokenizer import _register_vendored_archs

    _register_vendored_archs()
    model, _ = load_model(model_dir, lazy=True)
    if layer_limit:
        pipe.slice_model(model, 0, 1, 0, layer_limit)
    return model


def _greedy_rows_model(
    model, rows: list[list[int]], max_tokens: int, prefill_step: int
):
    """Single-process greedy through ``model(...)`` itself; left-padded batch
    caches come from mlx-lm's own ``_make_cache`` (the BatchGenerator seam)."""
    from mlx_lm.generate import _make_cache

    batch = len(rows)
    tokens, padding = pipe._left_pad(rows, 0)
    cache = model.make_cache() if batch == 1 else _make_cache(model, padding, None)
    prefix = tokens[:, :-1]
    for offset in range(0, prefix.shape[1], prefill_step):
        model(prefix[:, offset : offset + prefill_step], cache=cache)
        mx.eval([layer.state for layer in cache])
    current = tokens[:, -1:]
    generated = [[] for _ in range(batch)]
    traces = [[] for _ in range(batch)]
    for _ in range(max_tokens):
        logits = model(current, cache=cache)
        next_tokens = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
        mx.eval(next_tokens, logits, [layer.state for layer in cache])
        for row, token in enumerate(next_tokens.tolist()):
            generated[row].append(token)
            traces[row].append(_trace_row(logits[row, -1]))
        current = next_tokens[:, None]
    return generated, traces


def _first_divergence(left: list[int], right: list[int]) -> int | None:
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)


def cmd_batchcheck(options) -> int:
    """Single-process: does a two-request batch reproduce single requests?"""
    model_dir = Path(options.model).expanduser()
    prompts = _load_prompts(Path(options.prompts))
    model = _load_truncatable(model_dir, options.layer_limit)
    mx.eval(model.parameters())
    singles = {}
    for prompt in prompts:
        tokens, traces = _greedy_rows_model(
            model, [prompt["ids"]], options.max_tokens, options.prefill_step
        )
        singles[prompt["name"]] = (tokens[0], traces[0])
    report = []
    for first, second in _pairs(prompts):
        rows = [prompts[first], prompts[second]]
        tokens, traces = _greedy_rows_model(
            model,
            [row["ids"] for row in rows],
            options.max_tokens,
            options.prefill_step,
        )
        for row, generated, trace in zip(rows, tokens, traces):
            single_tokens, single_trace = singles[row["name"]]
            step = _first_divergence(single_tokens, generated)
            entry = {
                "pair": [prompts[first]["name"], prompts[second]["name"]],
                "row": row["name"],
                "matches_single": step is None,
                "tokens": generated,
                "trace": trace,
            }
            if step is not None:
                entry.update(
                    {
                        "first_divergence": step,
                        "single_margin": single_trace[step]["margin"],
                        "batched_margin": trace[step]["margin"],
                        "single_top2": single_trace[step]["top_ids"][:2],
                        "batched_top2": trace[step]["top_ids"][:2],
                        "max_abs_top1_logprob_diff_before": max(
                            [
                                abs(
                                    single_trace[i]["top_logprobs"][0]
                                    - trace[i]["top_logprobs"][0]
                                )
                                for i in range(step)
                            ]
                            or [0.0]
                        ),
                    }
                )
            report.append(entry)
            print(
                json.dumps(
                    {k: v for k, v in entry.items() if k not in ("tokens", "trace")}
                ),
                flush=True,
            )
    out = Path(options.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "batchcheck.json").write_text(json.dumps(report, indent=1))
    matched = sum(entry["matches_single"] for entry in report)
    print(f"batched rows matching single: {matched}/{len(report)}", flush=True)
    if options.pipe_dir:
        ranks = sorted(
            Path(options.pipe_dir).glob("rank*.json"), key=lambda p: int(p.stem[4:])
        )
        piped = json.loads(ranks[-1].read_text())["pairs"]
        same_tokens = same_traces = total = 0
        for index, pair in enumerate(piped):
            for row in range(2):
                ours = report[2 * index + row]
                total += 1
                same_tokens += pair["tokens"][row] == ours["tokens"]
                same_traces += pair["traces"][row] == ours["trace"]
        print(
            f"pipeline batched vs single-process batched: tokens identical "
            f"{same_tokens}/{total}, top-5 traces identical {same_traces}/{total}",
            flush=True,
        )
    return 0


def cmd_stream(options) -> int:
    """Teacher-forced full-model reference holding one layer's weights at a time.

    Replays ``Qwen4ExpTextModel.__call__`` over each prompt + the pipeline's
    own generated tokens as one prefill, loading a layer's tensors (lazy
    safetensors reads), running it for every sequence, then dropping it.
    """
    import gc

    from mlx_lm.generate import _make_cache
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    model_dir = Path(options.model).expanduser()
    prompts = {p["name"]: p["ids"] for p in _load_prompts(Path(options.prompts))}
    ranks = sorted(
        Path(options.pipe_dir).glob("rank*.json"), key=lambda p: int(p.stem[4:])
    )
    source = json.loads(ranks[-1].read_text())
    groups = [
        {"rows": [(r["name"], r["tokens"], r["trace"])]} for r in source["results"]
    ]
    for pair in source.get("pairs") or []:
        groups.append(
            {"rows": list(zip(pair["names"], pair["tokens"], pair["traces"]))}
        )

    model = _load_truncatable(model_dir, None)
    text = model.language_model
    inner = text.model
    args = text.args
    for group in groups:
        sequences = [prompts[name] + tokens[:-1] for name, tokens, _ in group["rows"]]
        tokens, padding = pipe._left_pad(sequences, 0)
        group["tokens"] = tokens
        group["padding"] = padding
        group["cache"] = (
            model.make_cache()
            if len(sequences) == 1
            else _make_cache(model, padding, None)
        )
    peaks = []

    def release(label: str) -> None:
        gc.collect()
        mx.clear_cache()
        peaks.append((label, round(mx.get_peak_memory() / 2**30, 3)))
        mx.reset_peak_memory()

    mx.eval(inner.embed_tokens.parameters())
    first_linear = next(i for i, layer in enumerate(inner.layers) if layer.is_linear)
    first_attention = next(
        i for i, layer in enumerate(inner.layers) if not layer.is_linear
    )
    for group in groups:
        hidden = inner.embed_tokens(group["tokens"])
        hidden = mx.tile(hidden, (1, 1, args.hc_count))
        cache = group["cache"]
        group["linear_mask"] = create_ssm_mask(hidden, cache[first_linear])
        group["attention_mask"] = create_attention_mask(
            hidden, cache[first_attention][0]
        )
        mx.eval(hidden)
        group["hidden"] = hidden
    del inner["embed_tokens"]
    release("embed")
    started = time.perf_counter()
    for index in range(args.num_hidden_layers):
        layer = inner.layers[index]
        mx.eval(layer.parameters())
        for group in groups:
            group["hidden"] = layer(
                group["hidden"],
                input_ids=group["tokens"],
                mask=group["linear_mask"]
                if layer.is_linear
                else group["attention_mask"],
                cache=group["cache"][index],
            )
            mx.eval(group["hidden"])
            group["cache"][index] = None
        inner.layers[index] = None
        del layer
        release(f"layer{index}")
        print(
            f"layer {index} done, peak {peaks[-1][1]} GiB, "
            f"{time.perf_counter() - started:.0f}s",
            flush=True,
        )
    mx.eval(inner.hyper_connection_mixer.parameters(), text.lm_head.parameters())
    report = []
    for group in groups:
        for row, (name, generated, trace) in enumerate(group["rows"]):
            start = group["padding"][row] + len(prompts[name]) - 1
            positions = mx.arange(start, start + len(generated))
            logits = text.lm_head(
                inner.hyper_connection_mixer(group["hidden"][row : row + 1, positions])
            )[0]
            mx.eval(logits)
            exact = 0
            worst = 0.0
            disagreements = []
            for step, token in enumerate(generated):
                ours = _trace_row(logits[step])
                row_logprobs = logits[step].astype(mx.float32)
                row_logprobs = row_logprobs - mx.logsumexp(row_logprobs)
                theirs = mx.array(trace[step]["top_ids"])
                diff = mx.max(
                    mx.abs(row_logprobs[theirs] - mx.array(trace[step]["top_logprobs"]))
                ).item()
                worst = max(worst, diff)
                if ours["top_ids"][0] == token:
                    exact += 1
                else:
                    disagreements.append(
                        {
                            "step": step,
                            "pipeline_token": token,
                            "stream_top2": ours["top_ids"][:2],
                            "stream_margin": ours["margin"],
                            "pipeline_margin": trace[step]["margin"],
                        }
                    )
            entry = {
                "row": name,
                "batch": len(group["rows"]),
                "argmax_matches": exact,
                "steps": len(generated),
                "max_abs_logprob_diff_top5": round(worst, 6),
                "disagreements": disagreements,
            }
            report.append(entry)
            print(json.dumps(entry), flush=True)
    out = Path(options.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stream.json").write_text(
        json.dumps({"report": report, "peaks_gib": peaks}, indent=1)
    )
    print(f"max per-layer peak {max(p for _, p in peaks)} GiB", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="flash_pipeline_harness")
    commands = parser.add_subparsers(dest="command", required=True)

    prompts = commands.add_parser("prompts")
    prompts.add_argument("--model", required=True)
    prompts.add_argument("--out", required=True)
    prompts.add_argument("--long-tokens", type=int, default=1900)

    for name in ("ref", "pipe"):
        sub = commands.add_parser(name)
        sub.add_argument("--model", required=True)
        sub.add_argument("--prompts", required=True)
        sub.add_argument("--out", required=True)
        sub.add_argument("--max-tokens", type=int, default=128)
        sub.add_argument("--context", type=int, default=8192)
        sub.add_argument(
            "--prefill-step", type=int, default=pipe.default_prefill_step()
        )
        sub.add_argument(
            "--layer-limit",
            type=int,
            help="verification only: first N decoder layers + embed + head",
        )
        if name == "pipe":
            sub.add_argument("--split")
            sub.add_argument("--soak-minutes", type=float, default=0)
            sub.add_argument("--soak-tokens", type=int, default=64)
            sub.add_argument(
                "--pairs",
                type=int,
                default=0,
                help="also run every neighbouring prompt pair as a batch of 2",
            )

    check = commands.add_parser("batchcheck")
    check.add_argument("--model", required=True)
    check.add_argument("--prompts", required=True)
    check.add_argument("--out", required=True)
    check.add_argument("--max-tokens", type=int, default=64)
    check.add_argument("--prefill-step", type=int, default=pipe.default_prefill_step())
    check.add_argument("--layer-limit", type=int)
    check.add_argument(
        "--pipe-dir", help="pipeline run with --pairs to compare against"
    )

    stream = commands.add_parser("stream")
    stream.add_argument("--model", required=True)
    stream.add_argument("--prompts", required=True)
    stream.add_argument("--pipe-dir", required=True)
    stream.add_argument("--out", required=True)

    compare = commands.add_parser("compare")
    compare.add_argument("--ref", required=True)
    compare.add_argument("--pipe", required=True)
    compare.add_argument("--out")

    options = parser.parse_args(argv)
    return {
        "prompts": cmd_prompts,
        "ref": cmd_ref,
        "pipe": cmd_pipe,
        "compare": cmd_compare,
        "batchcheck": cmd_batchcheck,
        "stream": cmd_stream,
    }[options.command](options)


if __name__ == "__main__":
    sys.exit(main())
