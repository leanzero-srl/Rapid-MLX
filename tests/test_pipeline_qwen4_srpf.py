"""Shortest-remaining-prefill-first on the pipeline split (Q-145), in process, one rank.

Nothing here launches a rank: the tiny float32 qwen4_exp of the Q-134 tests
runs rank 0's real tick loop on a singleton group (or no engine at all, for
the scheduler walk-throughs).

Q-145, measured live on 2026-09-26 (LOAD-2026-09-26c, Flash pipeline, slots
2): three ~39k-token prompts and a 13-token canary arrived together; first
tokens came at 115 / 232 / 347 s, the canary's at 347 s — one prefill at a
time, first come first served, and a short request waited behind every
earlier long one.  /v1/status at 22:14 read ``slots 2, slots_in_use 1,
num_waiting 2`` with the one row in prefill at 361 tok/s.
"""

import threading

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
pytest.importorskip("fastapi")
pytestmark = pytest.mark.requires_mlx

from rapid_mlx.distributed import pipeline_qwen4_serve as serve  # noqa: E402

from .test_pipeline_qwen4_continuous import (  # noqa: E402, F401 - fixtures
    LONG,
    PREFILL_STEP,
    SHORT,
    THIRD,
    _apply,
    _chunk,
    _Collected,
    _FakeKv,
    _job,
    _run_rank0,
    _solo,
    _state,
    _wait,
    group,
    stage,
)

# 162 tokens: eleven 16-token chunks.
LONGER = LONG * 3


class _Firsts:
    """The order in which named requests got their first token."""

    def __init__(self):
        self.names: list[str] = []
        self.lock = threading.Lock()

    def job(self, name: str, ids, max_tokens: int) -> _Collected:
        firsts = self

        class Named(_Collected):
            def push(self, item) -> None:
                super().push(item)
                if item[0] == "token" and len(self.tokens()) == 1:
                    with firsts.lock:
                        firsts.names.append(name)

        return Named(serve._Row(list(ids), max_tokens, 0.0, 1.0), None, None)


def _stop(state, loop) -> None:
    _wait(lambda: not state.active, "the rows never left")
    state.jobs.put(None)
    loop.join(timeout=60)
    assert not loop.is_alive()


def _hook_chunks(engine, after):
    """Call ``after(prompt_length_of_the_row, chunk_index)`` once each chunk's collective returned."""
    real = engine.prefill
    count = [0]

    def prefill(words):
        length = len(engine.joining.row.ids)
        result = real(words)
        count[0] += 1
        after(length, count[0])
        return result

    engine.prefill = prefill


def test_a_short_request_is_prefilled_between_the_chunks_of_a_long_one(stage, group):
    """Red before Q-145: the short request's first token came after the long one's."""
    state = _state(max_batch=2)
    engine = serve._Engine(stage, None, PREFILL_STEP)
    firsts = _Firsts()
    long_job = firsts.job("long", LONGER, 4)
    short = firsts.job("short", SHORT, 3)
    chunks: list[int] = []

    def after(length: int, index: int) -> None:
        chunks.append(length)
        if index == 1:
            state.jobs.put(short)  # it arrives while the long prompt prefills

    _hook_chunks(engine, after)
    loop = _run_rank0(state, engine, group)
    state.jobs.put(long_job)
    _wait(lambda: long_job.finished and short.finished, "the requests never finished")
    _stop(state, loop)

    assert firsts.names == ["short", "long"]
    # One chunk of the short prompt, between the long prompt's chunks: the
    # long one was seen at the first chunk's boundary after it arrived.
    assert chunks.count(len(SHORT)) == 1
    position = chunks.index(len(SHORT))
    assert 1 <= position <= 2 and chunks[position + 1] == len(LONGER)
    assert chunks.count(len(LONGER)) == 11
    # The suspended prefill resumed in its own cache: both answer as if alone.
    assert short.tokens() == _solo(stage, SHORT, 3)
    assert long_job.tokens() == _solo(stage, LONGER, 4)


def test_a_stream_of_short_requests_cannot_starve_a_long_prefill(stage, group):
    """Every chunk brings a new short request; aging still lets the long one through.

    Without aging (``_protected`` never true) the long prompt gets one chunk
    and then waits for the harness to stop offering short requests.
    """
    state = _state(max_batch=3)
    engine = serve._Engine(stage, None, PREFILL_STEP)
    firsts = _Firsts()
    long_job = firsts.job("long", LONGER, 2)
    offered: list[_Collected] = []
    offer_bound = 60  # the harness's own bound on the stream, never the engine's

    def after(length: int, index: int) -> None:
        if long_job.produced == 0 and len(offered) < offer_bound:
            short = firsts.job(f"short{len(offered)}", SHORT, 1)
            offered.append(short)
            state.jobs.put(short)

    _hook_chunks(engine, after)
    loop = _run_rank0(state, engine, group)
    state.jobs.put(long_job)
    _wait(
        lambda: long_job.finished and all(short.finished for short in offered),
        "the requests never finished",
    )
    _stop(state, loop)

    ahead = firsts.names.index("long")
    assert len(offered) < offer_bound, "the long prompt waited out the whole stream"
    # Protected once the tokens prefilled for others reach its own tokens left
    # (<= 162 - 16 after its first chunk): at most that many 12-token requests
    # go ahead of it, and every later one waits.
    assert 1 <= ahead <= -(-(len(LONGER) - PREFILL_STEP) // len(SHORT))
    assert len(offered) > ahead
    assert long_job.tokens() == _solo(stage, LONGER, 2)


def test_the_26c_arrival_pattern_prefills_the_canary_between_two_chunks():
    """LOAD-2026-09-26c walked through the scheduler at Flash scale (no engine work)."""
    context = 131_072
    state = serve._State(served="flash", context=context, max_batch=2)
    state.kv = _FakeKv(context)
    scheduler = serve._Scheduler(state, serve._Engine(None, None, 2048))

    def request(prompt: int, max_tokens: int) -> _Collected:
        return _job([1] * prompt, max_tokens)

    first = request(39_263, 64)
    others = [request(38_880, 64), request(39_015, 64)]
    canary = request(13, 8)

    # The loop takes the first arrival alone (the engine was idle).
    state.jobs.put(first)
    plan = scheduler.plan(block=True)
    assert plan.joiner is first.row
    _apply(scheduler, plan)
    # The rest arrive during its first chunk, after that chunk's words were sent.
    assert not scheduler.wants_plan(chunk=True)
    for job in [*others, canary]:
        state.jobs.put(job)
    assert _chunk(scheduler) is None
    # The next chunk's collective announces the plan that admits the canary:
    # 13 tokens left against 35,167 once that chunk ran (before Q-145 it
    # queued behind all three).
    assert scheduler.wants_plan(chunk=True)
    assert _chunk(scheduler) is None
    plan = scheduler.plan(block=False)
    assert plan.joiner is canary.row
    _apply(scheduler, plan)
    assert scheduler.engine.target == 1  # every rank runs the canary's chunk next
    assert _chunk(scheduler) == 5  # its only chunk samples its first token
    assert scheduler.running == [canary] and canary.produced == 1
    assert first.produced == 0

    # The two long ones wait: each has more left than the suspended first, and
    # both slots are held until the canary leaves.
    assert not scheduler.wants_plan(chunk=False)
    canary.finished = True
    assert scheduler.wants_plan(chunk=False)
    plan = scheduler.plan(block=False)
    assert plan.leave == [0] and plan.joiner is None
    _apply(scheduler, plan)
    while _chunk(scheduler) is None:
        assert not scheduler.wants_plan(chunk=False)
    assert scheduler.running == [first]
    # The first joined the running batch; the shorter long one is next.
    plan = scheduler.plan(block=False)
    assert plan.joiner is others[0].row


def test_a_protected_prefill_is_not_jumped():
    """Aging: once a row has waited its own prefill, a later short request waits for it."""
    state = serve._State(served="t", context=100_000, max_batch=4)
    scheduler = serve._Scheduler(state, serve._Engine(None, None, 100))
    long_job = _job([1] * 1_000, 4)
    state.jobs.put(long_job)
    _apply(scheduler, scheduler.plan(block=True))
    _chunk(scheduler)  # 900 left
    long_job.waited = 899
    state.jobs.put(_job([2] * 10, 4))
    assert scheduler.wants_plan(chunk=False)  # not yet protected: 899 < 900
    long_job.waited = 900
    assert not scheduler.wants_plan(chunk=False)
    assert scheduler.plan(block=False).joiner is None
    # A protected queued request goes before a shorter later one.
    waiting = _job([3] * 500, 4)
    later = _job([4] * 20, 4)
    scheduler2 = serve._Scheduler(
        serve._State(served="t", context=100_000, max_batch=4),
        serve._Engine(None, None, 100),
    )
    for job in (waiting, later):
        scheduler2.state.jobs.put(job)
    scheduler2._collect(block=False)
    assert scheduler2._head() is later
    waiting.waited = 500
    assert scheduler2._head() is waiting


def test_a_worker_rank_replays_interleaved_prefills_and_an_abort(
    stage, group, monkeypatch
):
    """Rank 0's collectives, replayed into the worker path: the same forwards in the same order.

    The worker never sees rank 0's queue; it learns admissions and aborts from
    the plans and derives which prefilling row runs next from the rows'
    ranges alone.  The run below interleaves three prefilling rows and
    aborts a suspended one.
    """
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
    state = _state(max_batch=3)
    engine = serve._Engine(stage, None, PREFILL_STEP)
    long_job, short, third = _job(LONGER, 8), _job(SHORT, 30), _job(THIRD, 12)
    order: list[int] = []

    def after(length: int, index: int) -> None:
        order.append(length)
        if index == 1:
            state.jobs.put(short)
            state.jobs.put(third)
        if length == len(THIRD) and order.count(len(THIRD)) == 1:
            long_job.cancelled = True  # suspended behind THIRD's prefill

    _hook_chunks(engine, after)
    loop = _run_rank0(state, engine, group)
    state.jobs.put(long_job)
    _wait(lambda: short.finished and third.finished, "rows")
    _stop(state, loop)
    rank0_forwards, forwards[:] = list(forwards), []

    # The run did what it claims: SHORT and THIRD prefilled between LONGER's
    # chunks, LONGER never finished its prompt, and the survivors are exact.
    first_short = order.index(len(SHORT))
    assert order[0] == len(LONGER) and first_short <= 2
    assert order.index(len(THIRD)) > first_short
    assert order.count(len(LONGER)) < 11 and not long_job.tokens()

    replay = iter(recorded)

    def replayed(group_, value):
        expected = next(replay)
        assert expected.shape == value.shape, "a collective would not pair"
        return expected

    monkeypatch.setattr(serve, "_all_sum", replayed)
    worker = serve._Engine(stage, None, PREFILL_STEP)
    serve._ticks(worker, group, 3, serve._Wake(group))
    assert next(replay, None) is None
    assert forwards == rank0_forwards
    assert not worker.prefilling and not worker.rows
    monkeypatch.undo()
    assert short.tokens() == _solo(stage, SHORT, 30)
    assert third.tokens() == _solo(stage, THIRD, 12)


class _BytesKv:
    """Reserves rows x longest; a snapshot costs its tokens on rank 0, double on rank 1."""

    def __init__(self, budgets: list[int]):
        self.budgets = budgets

    def reserve(self, lengths: list[int]) -> list[int]:
        return [len(lengths) * max(lengths)] * len(self.budgets)

    def entry_bytes(self, tokens: int) -> list[int]:
        return [tokens, 2 * tokens]


def test_a_pending_snapshot_is_charged_until_it_is_stored_or_forgotten():
    """Rows prefill interleaved, so a directed snapshot can still be untaken at the next admission."""
    index = serve._PrefixIndex(_BytesKv([100, 100]))
    first = serve._Row([1] * 30, 4, 0.0, 1.0, boundary=20)
    assert index.admit(first, [30]) == []
    assert first.store_at == 20 and index.pending == {first.store_id: [20, 40]}
    # Beside it, the batch leaves room 40 per rank: the pending [20, 40] plus
    # a second [20, 40] would not fit rank 1, so the second is not directed.
    second = serve._Row([2] * 30, 4, 0.0, 1.0, boundary=20)
    index.admit(second, [30, 30])
    assert second.store_id == 0
    assert index.status()["skipped"] == {"no_room_beside_the_batch": 1}
    # The first row was aborted before its snapshot: the charge goes.
    index.forget(first.store_id)
    third = serve._Row([3] * 30, 4, 0.0, 1.0, boundary=20)
    index.admit(third, [30, 30])
    assert third.store_id and index.pending == {third.store_id: [20, 40]}
    version = index.version
    index.stored(third.store_id, tuple(third.ids[:20]), [20, 40])
    assert index.pending == {} and index.version == version + 1
