# SPDX-License-Identifier: Apache-2.0
"""What continuous admission costs the running row (Q-134), in process, one rank.

A random-init qwen4_exp (every layer kind, sized by --hidden/--layers) runs
the pipeline server's own tick loop on a singleton MLX group — no rank is
launched.  One long request decodes; a second request with a ``--prompt``-
token prompt is queued mid-generation.  Reported, from the running row's own
token timestamps:

  alone     the running row's decode rate before the second request arrives
  joining   its rate while the second request prefills beside it
  batched   its rate once both decode in one batch
  ttft      the second request's time to first token, and its prefill rate
            beside the running row vs. the same prompt on an idle engine

Before Q-134 the second request waited for the running row to END.

    python scripts/bench_pipeline_admission.py --hidden 1024 --layers 8 --prompt 4096
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import asdict, dataclass

import mlx.core as mx

from rapid_mlx.distributed import pipeline_qwen4 as pipe
from rapid_mlx.distributed import pipeline_qwen4_serve as serve
from rapid_mlx.models.qwen4_exp import Model, ModelArgs, TextModelArgs


def _config(hidden: int, layers: int) -> dict:
    heads = hidden // 128
    pattern = (["linear_attention"] * 3 + ["full_attention"]) * (layers // 4)
    args = TextModelArgs(
        hidden_size=hidden,
        num_hidden_layers=len(pattern),
        vocab_size=32_000,
        max_position_embeddings=65_536,
        num_attention_heads=heads,
        num_key_value_heads=max(1, heads // 4),
        head_dim=128,
        linear_num_key_heads=max(1, heads // 2),
        linear_num_value_heads=heads,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_experts=16,
        num_experts_per_tok=4,
        moe_intermediate_size=hidden,
        shared_expert_intermediate_size=hidden,
        hc_count=4,
        hc_lowrank=64,
        layer_types=pattern,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=64,
        indexer_budget=64,
        indexer_compress_ratio=4,
        ple_layer_ids=[2],
        ple_embed_dim=64,
        heads_per_ngram=1,
        ngram_vocab_size_base=32_001,
        make_ngram_vocab_size_divisible_by=64,
        split_ngram_parts=4,
        eos_token_id=31_999,
    )
    config = asdict(args)
    config["model_type"] = "qwen4_exp_text"
    return config


@dataclass(eq=False)
class _Timed(serve._Job):
    def __post_init__(self):
        self.stamps: list[float] = []

    def push(self, item) -> None:
        if item[0] == "token":
            self.stamps.append(time.perf_counter())


def _rate(stamps: list[float], start: float, end: float) -> tuple[float, int]:
    inside = [t for t in stamps if start <= t <= end]
    if len(inside) < 2:
        return float("nan"), len(inside)
    return (len(inside) - 1) / (inside[-1] - inside[0]), len(inside)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--prompt", type=int, default=4096)
    parser.add_argument("--prefill-step", type=int, default=512)
    parser.add_argument("--before", type=int, default=64)
    options = parser.parse_args()

    group = mx.distributed.init(strict=False)
    mx.random.seed(20260926)
    text = _config(options.hidden, options.layers)
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=text))
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    stage = pipe.PipelineStage(model, group, 0, text["num_hidden_layers"], mx.bfloat16)
    engine = serve._Engine(stage, None, options.prefill_step)
    serve._warm(engine)
    rng = [((i * 7919) % 31_000) + 1 for i in range(options.prompt)]

    # The same prompt on an idle engine: its prefill rate alone.
    solo = _Timed(serve._Row(rng, 2, 0.0, 1.0), None, None)
    state = serve._State(served="bench", context=65_536, max_batch=2)
    loop = threading.Thread(
        target=serve._rank0_loop,
        args=(state, engine, group, serve._Wake(group)),
        daemon=True,
    )
    loop.start()
    began = time.perf_counter()
    state.jobs.put(solo)
    while not solo.stamps:
        time.sleep(0.001)
    solo_ttft = solo.stamps[0] - began
    while state.active:
        time.sleep(0.001)

    running = _Timed(serve._Row([5, 6, 7, 8], 60_000, 0.0, 1.0), None, None)
    state.jobs.put(running)
    while len(running.stamps) < options.before:
        time.sleep(0.001)
    joiner = _Timed(serve._Row(list(reversed(rng)), 64, 0.0, 1.0), None, None)
    queued = time.perf_counter()
    state.jobs.put(joiner)
    while len(joiner.stamps) < 64:
        time.sleep(0.001)
    running.cancelled = True
    joiner.cancelled = True
    while state.active:
        time.sleep(0.001)
    state.jobs.put(None)
    loop.join()

    first = joiner.stamps[0]
    alone, n_alone = _rate(running.stamps, running.stamps[0], queued)
    joining, n_joining = _rate(running.stamps, queued, first)
    batched, n_batched = _rate(running.stamps, first, joiner.stamps[-1])
    ttft = first - queued
    print(
        f"model hidden={options.hidden} layers={text['num_hidden_layers']} "
        f"prompt={options.prompt} prefill_step={options.prefill_step} "
        f"share={serve._PREFILL_SHARE}"
    )
    print(f"alone    {alone:8.1f} tok/s  ({n_alone} tokens)")
    print(
        f"joining  {joining:8.1f} tok/s  ({n_joining} tokens)  "
        f"{joining / alone - 1:+.0%} vs alone"
    )
    print(
        f"batched  {batched:8.1f} tok/s  ({n_batched} tokens)  "
        f"{batched / alone - 1:+.0%} vs alone"
    )
    print(
        f"ttft     {ttft:8.3f} s beside the running row "
        f"({options.prompt / ttft:.0f} prompt tok/s) vs {solo_ttft:.3f} s alone "
        f"({options.prompt / solo_ttft:.0f} tok/s)"
    )


if __name__ == "__main__":
    main()
