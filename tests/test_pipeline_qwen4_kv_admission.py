"""Admission by memory, not by a row count, on the pipeline split (goose Q-160), in process.

Nothing here launches a rank: the tiny float32 qwen4_exp of the Q-134 tests
runs rank 0's real tick loop on a singleton group against a REAL
``_KvBudget`` planned for it, or the scheduler runs alone over the 14:39
budgets (``_FakeKv``) and the 26d replay (``_Drive``).

Q-160: the Flash split ran ``--slots 2 --max-batch 2`` (goose passed the
planned slots as the batch cap) and ``_KvBudget`` priced every row at the
longest horizon for its whole life (rows x longest).  goose's chats send no
max_tokens, so each reserves the whole context: with two chats decoding, a
helper call (fact checker, title, compaction summary: a few hundred prompt
tokens, a few hundred out) waited for a whole multi-minute decode to leave,
though it needed a few hundred tokens' KV and would have left long before
the chats could grow into the room it used.
"""

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
pytest.importorskip("fastapi")
pytestmark = pytest.mark.requires_mlx

from rapid_mlx.distributed import pipeline_qwen4 as pipe  # noqa: E402
from rapid_mlx.distributed import pipeline_qwen4_serve as serve  # noqa: E402

from .test_pipeline_qwen4_continuous import (  # noqa: E402, F401 - fixtures
    LONG,
    PREFILL_STEP,
    SHORT,
    THIRD,
    _apply,
    _chunk,
    _FakeKv,
    _job,
    _run_rank0,
    _solo,
    _start_fake,
    _wait,
    group,
    stage,
)
from .test_pipeline_qwen4_srpf import _Drive  # noqa: E402

CONTEXT = 512


def _plan(stage, context: int, slots: int):
    """The planner's split of the tiny model on one rank: its KV budget is what is under test."""
    args = stage.args
    ckpt = pipe.CheckpointBytes(
        layer_bytes=[0] * args.num_hidden_layers,
        head_bytes=0,
        tail_bytes=0,
        excluded_bytes={},
        activation_bytes=4,  # the float32 stage's caches
    )
    node = pipe.NodeBudget("cpu", 2**36, 2**35, "test")
    return pipe.plan_pipeline(
        args,
        ckpt,
        [node],
        context=context,
        batch=slots,
        prefill_step=PREFILL_STEP,
        starts=[0],
    )


def _chat(ids) -> serve._Job:
    """A request with no max_tokens: the HTTP side grants the context's room."""
    return _job(ids, CONTEXT - len(ids) - 1)


def test_a_short_request_joins_two_long_rows_within_the_measured_budget(
    stage, group, monkeypatch
):
    """Two chats decoding hold the plan's two slots; a short request arrives.

    It is admitted at the next plan, prefilled in one chunk and answers its
    first token within a few decode steps while both chats keep decoding;
    the bytes every cache really holds stay inside the planned budget at
    every step; and a worker rank replaying rank 0's collectives runs the
    same forwards (the three-row batch, the derived plan header).

    Red on f8a2461a5: ``--max-batch 2`` held the short request until a chat
    left (the chats here run 450+ tokens), and at a larger cap the rows x
    longest reservation refused it the same way.
    """
    kv = serve._KvBudget(_plan(stage, CONTEXT, slots=2), PREFILL_STEP)
    state = serve._State(served="tiny", context=CONTEXT, max_batch=kv.rows(), kv=kv)
    engine = serve._Engine(stage, None, PREFILL_STEP)

    measured: list[tuple[int, list[int]]] = []

    def held_now() -> int:
        caches = [] if engine.cache is None else [engine.cache]
        caches += [joining.cache for joining in engine.prefilling]
        return sum(serve._held_bytes(cache) for cache in caches)

    for name in ("decode", "prefill"):
        real = getattr(engine, name)

        def measuring(words, real=real):
            result = real(words)
            measured.append((held_now(), list(state.reserved)))
            return result

        setattr(engine, name, measuring)

    recorded: list = []
    forwards: list = []
    real_forward = stage.forward

    def logged_forward(inputs, cache, logits, **kwargs):
        forwards.append((tuple(inputs.shape), logits, inputs.tolist()))
        return real_forward(inputs, cache, logits, **kwargs)

    def record(group_, value):
        mx.eval(value)
        recorded.append(value)
        return value

    monkeypatch.setattr(stage, "forward", logged_forward)
    monkeypatch.setattr(serve, "_all_sum", record)

    loop = _run_rank0(state, engine, group)
    chats = [_chat(LONG), _chat(THIRD)]
    for chat in chats:
        state.jobs.put(chat)
    _wait(lambda: all(chat.produced >= 8 for chat in chats), "the chats never decoded")
    arrived = state.steps
    short = _job(SHORT, 4)
    state.jobs.put(short)
    _wait(lambda: short.produced >= 1, "the short request never answered")
    first_after = state.steps - arrived
    _wait(lambda: short.finished, "the short request never finished")
    assert not any(chat.finished for chat in chats)
    for chat in chats:
        chat.cancelled = True
    _wait(lambda: not state.active, "the cancelled chats never left")
    state.jobs.put(None)
    loop.join(timeout=60)
    assert not loop.is_alive()

    # Seen at the next step's words, admitted by the plan after it, its one
    # chunk riding the step after that: its first token within three steps.
    assert first_after <= 3, first_after
    assert any(shape[0] == 3 for shape, _, _ in forwards), "never a three-row batch"
    (budget,) = kv.budgets
    most = max(held for held, _ in measured)
    assert most <= budget, (most, budget)
    # What the rows really held never ran past what was reserved for them.
    assert all(held <= max(reserved, default=0) for held, reserved in measured)

    rank0_forwards, forwards[:] = list(forwards), []
    replay = iter(recorded)

    def replayed(group_, value):
        expected = next(replay)
        assert expected.shape == value.shape, "a collective would not pair"
        return expected

    monkeypatch.setattr(serve, "_all_sum", replayed)
    worker = serve._Engine(stage, None, PREFILL_STEP)
    serve._ticks(worker, group, kv.rows(), serve._Wake(group))
    assert next(replay, None) is None
    assert forwards == rank0_forwards
    monkeypatch.undo()
    assert short.tokens() == _solo(stage, SHORT, 4)
    for chat, ids in zip(chats, (LONG, THIRD)):
        got = chat.tokens()
        assert got == _solo(stage, ids, len(got))


def _flash(context: int = 131_072) -> serve._Scheduler:
    kv = _FakeKv(context)
    state = serve._State(served="flash", context=context, max_batch=kv.rows(), kv=kv)
    return serve._Scheduler(state, serve._Engine(None, None, kv.prefill_step))


def _decoding(scheduler, prompt: int, produced: int) -> serve._Job:
    """A running chat that sent no max_tokens, ``produced`` tokens in."""
    job = _job([1] * prompt, scheduler.state.context - prompt - 1)
    job.produced = produced
    return job


def test_a_helper_call_joins_two_chats_at_flash_scale():
    """The 14:39 budgets: two chats 44k and 43k tokens in, a 600-token helper
    call with 256 tokens to write.  It is admitted at once and its one chunk
    rides the next decode step; it holds the padded batch's width for 256
    steps and leaves long before the chats could grow into that room."""
    scheduler = _flash()
    state = scheduler.state
    chats = [_decoding(scheduler, 39_263, 5_000), _decoding(scheduler, 38_880, 4_000)]
    scheduler.running = list(chats)
    scheduler._publish()
    helper = _job([1] * 600, 256)
    state.jobs.put(helper)
    assert scheduler.decode_words() == [1, 0]  # the step announces the plan
    plan = scheduler.plan(block=False)
    assert plan.joiner is helper.row and plan.leave == []
    _apply(scheduler, plan)
    assert scheduler.decode_words() == [0, 1]  # its one chunk rides the next step
    scheduler.decoded([1, 1], 0.123)
    assert _chunk(scheduler) == 5
    assert scheduler.running == [*chats, helper] and helper.produced == 1
    assert all(r <= b for r, b in zip(state.reserved, state.kv.budgets))


def test_the_last_chunk_waits_until_the_batch_it_makes_fits():
    """A prefilling row's merge pads it to the batch's width: if the rows grew
    past what that holds, its last chunk waits — decode goes on, the wait
    earns the prefill no time share — and runs at the step a row leaves."""
    scheduler = _flash()
    state = scheduler.state
    # Three chats 125k tokens into a 131k context: three rows fit, a fourth
    # of their width does not.
    chats = [_decoding(scheduler, 1_000, 124_000 + i) for i in range(3)]
    scheduler.running = list(chats)
    helper = _job([1] * 600, 256)
    scheduler.prefilling = [helper]
    _start_fake(scheduler.engine, helper.row)
    scheduler._publish()
    assert all(r <= b for r, b in zip(state.reserved, state.kv.budgets))
    for _ in range(3):
        assert scheduler.decode_words() == [0, 0]
        assert scheduler.holding
        scheduler.decoded([1, 1, 1], 0.123)
        assert scheduler.credit == 0.0
    chats[0].finished = True
    assert scheduler.decode_words() == [1, 0]
    scheduler.decoded([1, 1, 1], 0.123)
    plan = scheduler.plan(block=False)
    assert plan.leave == [0] and plan.joiner is None
    _apply(scheduler, plan)
    assert scheduler.decode_words() == [0, 1]
    assert _chunk(scheduler) == 5
    assert scheduler.running == [chats[1], chats[2], helper]


def test_prefix_entries_yield_to_a_join_before_the_join_waits():
    """The prefix cache holds only what the rows leave idle: an entry that
    stands between a prefilling row and its merge is evicted by a plan, and
    the chunk runs at the next step."""
    scheduler = _flash()
    state = scheduler.state
    state.prefix = serve._PrefixIndex(state.kv)
    chats = [_decoding(scheduler, 1_000, 100_000 + i) for i in range(3)]
    scheduler.running = list(chats)
    helper = _job([1] * 600, 256)
    scheduler.prefilling = [helper]
    _start_fake(scheduler.engine, helper.row)
    join = state.kv.held(scheduler._decoding([*chats, helper]), scheduler._prompts([]))
    assert state.kv.fits(join)
    # One entry holding all the room the join leaves, and a byte more.
    room = [b - n + 1 for b, n in zip(state.kv.budgets, join)]
    state.prefix.entries[7] = serve._Entry((1, 2, 3), room)
    assert scheduler.decode_words() == [1, 0]
    plan = scheduler.plan(block=False)
    assert plan.evict == [7] and plan.joiner is None
    _apply(scheduler, plan)
    assert not state.prefix.entries
    assert scheduler.decode_words() == [0, 1]


def test_a_canary_beside_a_decoding_chat_and_a_long_prefill_is_answered_in_a_chunk():
    """The Q-145 residual, replayed at the 26d rates: one chat decoding
    (5,000 tokens to write), a 39k-token prompt prefilling beside it, and a
    27-token canary arriving 20 s later.  Admitted by memory, the canary is
    prefilled at the next chunk boundary; with the cap goose passed
    (``--max-batch 2``) it waited until a row LEFT — here the long prompt's
    whole prefill and its 37 tokens (negative control)."""

    def replay(max_batch: int) -> _Drive:
        drive = _Drive(max_batch=max_batch)
        drive.at(0.0, "chat", 3_000, 5_000)
        drive.at(10.0, "long", 39_263, 37)
        drive.at(30.0, "canary", 27, 1)
        drive.run(until=lambda d: d.answered("canary"))
        return drive

    canary = replay(_FakeKv(131_072).rows()).first("canary")
    chunk_s = _Drive.PREFILL_STEP / _Drive.PREFILL_TOKENS_PER_S
    assert canary["chunk_steps"] <= 2, canary
    assert canary["first"] - canary["arrived"] <= 2 * chunk_s, canary

    capped = replay(2)
    left = capped.first("long")["done"]
    assert left is not None and capped.first("canary")["first"] >= left
    assert capped.first("canary")["first"] - canary["arrived"] > 10 * chunk_s


def test_the_26d_load_answers_every_canary_between_two_chunks_admitted_by_memory():
    """LOAD-2026-09-26d's arrivals with admission by memory instead of the
    two slots: every canary is still prefilled within three chunk-steps of
    its arrival, and every worker's prompts are answered."""
    drive = _Drive(max_batch=_FakeKv(131_072).rows())
    workers = {
        "w1": [(39_263, 37), (39_016, 37), (39_263, 44)],
        "w2": [(38_880, 37), (39_259, 52), (38_880, 44)],
        "w0": [(39_015, 39), (38_849, 31), (39_015, 44)],
    }
    for name, prompts in workers.items():
        (prompt, output), *rest = prompts
        drive.at(0.0, name, prompt, output, rest)
    for index, when in enumerate((0.0, 63.3, 124.8, 538.2)):
        drive.at(when, f"canary{index + 1}", 27, 1)
    drive.run(
        until=lambda d: (
            all(d.answered(f"canary{i}") for i in (1, 2, 3, 4))
            and all(d.answered(name, 1) for name in workers)
        )
    )
    steps = [drive.first(f"canary{i}")["chunk_steps"] for i in (1, 2, 3, 4)]
    assert all(step <= 3 for step in steps), steps
