# SPDX-License-Identifier: Apache-2.0
"""LeanZero Q-103: shortest-remaining-prefill-first on the one-row (MTP) engine.

goose Q-103: under three streams of unique 13k-token prompts a 1-token request
waited up to 161 s, because the vendored MTP verifier decodes one request at a
time and admission kept that request alone for its whole life, prefill
included. ``singleton_prefill_order="srpf"`` (the default) lets a waiting
request with fewer prompt tokens left take the engine at the prefilling row's
next chunk boundary, parks the prefilling row with its cache, and resumes it
where it stopped — the pipeline runner's order (goose Q-145/Q-160) on an engine
that holds one row.

Everything here runs the REAL scheduler, mlx-lm ``BatchGenerator`` and the
vendored MTP verifier on a tiny Qwen3.5 model (GatedDeltaNet + attention, MTP
head injected with random weights) on the CPU device. No GPU is touched and no
server is started. The memory probes are stubbed (CPU has no Metal cap); the
engine clock is a counter so one engine step costs one unit.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx  # noqa: E402

from rapid_mlx.cli import build_parser  # noqa: E402
from rapid_mlx.request import Request, RequestStatus, SamplingParams  # noqa: E402
from rapid_mlx.scheduler import Scheduler, SchedulerConfig  # noqa: E402

CHUNK = 8
LONG = 64


def _tiny_args(full_attention_interval: int):
    from mlx_lm.models.qwen3_5 import TextModelArgs

    args = TextModelArgs(
        model_type="qwen3_5",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        linear_num_value_heads=2,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        tie_word_embeddings=False,
        attention_bias=False,
        head_dim=16,
        full_attention_interval=full_attention_interval,
        num_experts=0,
        num_experts_per_tok=0,
        decoder_sparse_step=0,
        shared_expert_intermediate_size=0,
        moe_intermediate_size=0,
        norm_topk_prob=True,
    )
    object.__setattr__(args, "mtp_num_hidden_layers", 1)
    return args


@pytest.fixture(scope="module")
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture(autouse=True)
def _no_metal(monkeypatch, cpu):
    # mlx-lm's BatchGenerator reads the Metal working-set size when Metal is
    # available; on the CPU device that key is absent.
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)


def _model(full_attention_interval: int):
    from mlx_lm.models.qwen3_5 import TextModel

    from rapid_mlx.spec_decode.mtp.qwen3_5_inject import inject_mtp_support

    mx.random.seed(0)
    model = TextModel(_tiny_args(full_attention_interval))
    assert inject_mtp_support(model, allow_random_init=True)
    mx.eval(model.parameters())
    return model


@pytest.fixture(scope="module")
def hybrid(cpu):
    """Layer 0 GatedDeltaNet (ArraysCache), layer 1 attention (KVCache)."""
    return _model(2)


@pytest.fixture(scope="module")
def llama(cpu):
    """Plain attention, no MTP: the one-row engine by ``max_num_seqs=1``."""
    from mlx_lm.models.llama import Model, ModelArgs

    mx.random.seed(0)
    model = Model(
        ModelArgs(
            model_type="llama",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=128,
            num_key_value_heads=2,
        )
    )
    mx.eval(model.parameters())
    return model


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.5
        return self.now


def _scheduler(model, *, cap: int = 10**12, active: int = 0, **config) -> Scheduler:
    if not hasattr(model, "mtp_forward"):
        config = {"max_num_seqs": 1, "spec_decode": "none", **config}
    tokenizer = SimpleNamespace(
        encode=lambda text: list(text),
        decode=lambda ids, **_: " ".join(map(str, ids)),
        eos_token_id=127,
        eos_token_ids={127},
    )
    scheduler = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            **{
                "max_num_seqs": 4,
                "prefill_step_size": CHUNK,
                "enable_prefix_cache": False,
                "spec_decode": "mtp",
                **config,
            }
        ),
    )
    scheduler._resolve_metal_cap_bytes = lambda: cap
    scheduler._current_metal_active_bytes = lambda: active
    scheduler._srpf_clock = _Clock()
    return scheduler


def _request(rid: str, tokens: int, max_tokens: int, first: int = 1) -> Request:
    return Request(
        request_id=rid,
        prompt=list(range(first, first + tokens)),
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
    )


def _run(scheduler: Scheduler, arrivals: dict[int, list[Request]], steps: int = 400):
    """Step the engine; ``arrivals[step]`` are added before that step.

    Returns (first-token step, tokens, cached_tokens last reported) per request.
    """
    first: dict[str, int] = {}
    tokens: dict[str, list[int]] = {}
    cached: dict[str, int] = {}
    last_arrival = max(arrivals, default=0)
    for step in range(steps):
        for request in arrivals.get(step, ()):
            scheduler.add_request(request)
        output = scheduler.step()
        for out in output.outputs:
            first.setdefault(out.request_id, step)
            tokens.setdefault(out.request_id, []).extend(out.new_token_ids or [])
            cached[out.request_id] = out.cached_tokens
        if step >= last_arrival and not scheduler.has_requests():
            break
    return first, tokens, cached


def _solo(model, request: Request) -> tuple[int, list[int]]:
    first, tokens, _ = _run(_scheduler(model), {0: [request]})
    return first[request.request_id], tokens[request.request_id]


def _copy(request: Request) -> Request:
    return _request(
        request.request_id,
        len(request.prompt),
        request.sampling_params.max_tokens,
        first=request.prompt[0],
    )


@pytest.mark.parametrize("family", ["hybrid", "llama"])
def test_a_short_request_arriving_during_a_long_prefill_gets_its_first_token_within_a_few_chunk_steps(
    family, request
):
    """RED on lz.9: the 3-token request waited for the 64-token prompt's whole
    prefill and decode (first token 13 steps after it arrived). GREEN: it takes
    the engine at the next chunk boundary, and both answers are the tokens
    each request produces alone — the parked hybrid cache resumed exactly."""
    model = request.getfixturevalue(family)
    long_ = _request("long", LONG, 4)
    short = _request("short", 3, 1, first=40)
    long_solo_first, long_solo = _solo(model, _copy(long_))
    _, short_solo = _solo(model, _copy(short))

    scheduler = _scheduler(model)
    arrival = 2
    first, tokens, _ = _run(scheduler, {0: [long_], arrival: [short]})

    assert first["short"] - arrival <= 3, first
    assert tokens["short"] == short_solo
    assert tokens["long"] == long_solo
    stats = scheduler.get_stats()["singleton_prefill_order"]
    assert stats["active"] is True
    assert stats["parks"] == 1, stats
    assert stats["parks_cache_dropped"] == 0
    # The long prompt resumed from its parked cache: it re-prefilled nothing,
    # so it lost only the steps the short request ran.
    assert first["long"] <= long_solo_first + (first["short"] - arrival) + 1


def test_the_fifo_order_is_the_lz9_behaviour(hybrid):
    """Control: ``singleton_prefill_order="fifo"`` keeps lz.9's order, so the
    measurement above is the order and nothing else."""
    scheduler = _scheduler(hybrid, singleton_prefill_order="fifo")
    first, _, _ = _run(
        scheduler, {0: [_request("long", LONG, 4)], 2: [_request("short", 3, 1, 40)]}
    )
    assert first["short"] > first["long"]
    assert scheduler.get_stats()["singleton_prefill_order"]["parks"] == 0


def _stream(scheduler: Scheduler, steps: int, every: int = 1):
    arrivals = {0: [_request("long", LONG, 2)]}
    for index, step in enumerate(range(1, steps, every)):
        arrivals[step] = [_request(f"short-{index}", 3, 2, first=40 + index % 50)]
    return _run(scheduler, arrivals, steps=steps + 200)


def test_a_stream_of_short_requests_never_starves_the_long_prefill(hybrid):
    """A 3-token request arrives at EVERY step. Jumpers go ahead of the long
    prompt only while each takes at most half of its remaining slack, so the
    long prompt's first token lands within its own prefill plus that slack —
    twice its solo time — plus the one jumper already running. Negative
    control: with the slack unbounded the long prompt never gets the engine
    while the stream lasts."""
    solo_first, _ = _solo(hybrid, _request("long", LONG, 2))
    stream_steps = 12 * solo_first

    scheduler = _scheduler(hybrid)
    first, _, _ = _stream(scheduler, stream_steps)
    jumped = [
        rid for rid, step in first.items() if rid != "long" and step < first["long"]
    ]
    assert jumped, "no short request went ahead of the long prefill"
    one_jumper = 4  # prefill chunk, kickoff, first token, second token
    assert first["long"] <= 2 * solo_first + one_jumper, (first["long"], solo_first)

    control = _scheduler(hybrid)
    control._srpf_slack = lambda *args, **kwargs: math.inf
    first_control, _, _ = _stream(control, stream_steps)
    starved = first_control.get("long", math.inf)
    assert starved > stream_steps, (starved, stream_steps)


def test_a_later_prompt_barely_shorter_keeps_first_come_order(hybrid):
    """Taking at most HALF the slack: a 60-token prompt arriving while a
    64-token one prefills does not jump it (2 x 60 > its slack), while an
    8-token one does."""
    scheduler = _scheduler(hybrid)
    first, _, _ = _run(
        scheduler,
        {0: [_request("first", LONG, 1)], 1: [_request("second", 60, 1, first=3)]},
    )
    assert first["first"] < first["second"]
    assert scheduler.get_stats()["singleton_prefill_order"]["parks"] == 0

    scheduler = _scheduler(hybrid)
    first, _, _ = _run(
        scheduler,
        {0: [_request("first", LONG, 1)], 1: [_request("small", 8, 1, first=3)]},
    )
    assert first["small"] < first["first"]


def test_parks_nest_and_every_answer_is_its_solo_answer(hybrid):
    """A 24-token request jumps the 64-token prefill and is itself jumped by a
    3-token one while it prefills: two rows parked at once, each resumed from
    its own cache."""
    long_ = _request("long", LONG, 3)
    mid = _request("mid", 24, 3, first=20)
    short = _request("short", 3, 2, first=90)
    solos = {r.request_id: _solo(hybrid, _copy(r))[1] for r in (long_, mid, short)}

    scheduler = _scheduler(hybrid)
    first, tokens, _ = _run(scheduler, {0: [long_], 2: [mid], 3: [short]})
    assert tokens == solos
    assert scheduler.get_stats()["singleton_prefill_order"]["parks"] == 2
    # The short request got the engine first. When it left, "mid" (16 tokens
    # left) could not go ahead of "long" (48 left): the jumpers had spent
    # long's slack (64 - 40 waited - 16 mid still to prefill = 8 < 2 x 16),
    # so long resumed first — the bound, not a tie-break.
    assert first["short"] < first["long"] < first["mid"], first


def test_a_park_is_refused_unless_the_measured_memory_holds_both(hybrid):
    """Q-110: nothing opens unless the memory holds it. The need is priced
    from the live row's own cache bytes (parked copy + the jumper at its
    horizon); a cap it does not fit under — or no cap at all — keeps FIFO."""
    for cap, active, reason in (
        (10**12, 10**12 - 1, "kv_budget"),
        (0, 0, "no_memory_cap"),
    ):
        scheduler = _scheduler(hybrid, cap=cap, active=active)
        first, _, _ = _run(
            scheduler,
            {0: [_request("long", LONG, 2)], 2: [_request("short", 3, 1, 40)]},
        )
        stats = scheduler.get_stats()["singleton_prefill_order"]
        assert first["short"] > first["long"], (reason, first)
        assert stats["parks"] == 0
        assert stats["refused"].get(reason, 0) >= 1, stats


def test_the_memory_need_is_measured_from_the_live_row(hybrid):
    """The 27B's config-derived KV projection is 0 (goose d1b58f6ed); the park
    prices from buffers: attention layers per token, recurrent state fixed."""
    scheduler = _scheduler(hybrid)
    scheduler.add_request(_request("long", LONG, 2))
    scheduler.step()
    scheduler.step()
    live = scheduler.batch_generator._prompt_batch.prompt_cache
    fixed, per_token, total = Scheduler._srpf_cache_bytes(live)
    assert fixed > 0 and per_token > 0
    assert total == sum(layer.nbytes for layer in live)
    kv = [layer for layer in live if isinstance(getattr(layer, "offset", None), int)]
    assert per_token == pytest.approx(sum(layer.nbytes / layer.offset for layer in kv))


def test_a_parked_request_reports_only_prefix_cache_reuse_as_cached(hybrid):
    scheduler = _scheduler(hybrid)
    _, _, cached = _run(
        scheduler, {0: [_request("long", LONG, 2)], 2: [_request("short", 3, 1, 40)]}
    )
    assert scheduler.get_stats()["singleton_prefill_order"]["parks"] == 1
    assert cached["long"] == 0


def test_a_parked_request_can_be_aborted_and_frees_its_cache(hybrid):
    scheduler = _scheduler(hybrid)
    long_ = _request("long", LONG, 2)
    scheduler.add_request(long_)
    scheduler.step()
    scheduler.step()
    scheduler.add_request(_request("short", 3, 4, first=40))
    scheduler.step()
    assert long_._srpf_parked and long_.prompt_cache
    status = {row["request_id"]: row for row in scheduler.get_running_requests_info()}
    assert status["long"]["phase"] == "parked"
    assert status["long"]["prefilled_tokens"] == 2 * CHUNK

    assert scheduler.abort_request("long")
    scheduler.step()
    assert long_.status == RequestStatus.FINISHED_CANCELLED
    assert long_.prompt_cache is None
    assert all(r is not long_ for r in scheduler.waiting)


def test_the_serve_cli_exposes_the_order():
    args = build_parser().parse_args(["serve", "model"])
    assert args.singleton_prefill_order == "srpf"
    args = build_parser().parse_args(
        ["serve", "model", "--singleton-prefill-order", "fifo"]
    )
    assert args.singleton_prefill_order == "fifo"
    with pytest.raises(ValueError, match="singleton_prefill_order"):
        SchedulerConfig(singleton_prefill_order="lifo")
