"""Continuous admission on the pipeline split (Q-134), in process, one rank.

Nothing here launches a rank: a singleton MLX group runs the same tick loop
every rank runs, over a random-init float32 qwen4_exp with every layer kind
(float32 so a row's tokens do not depend on its batch — measured: in bfloat16
the same prompt twice in one batch drifts from a solo run at step 3, logits
differing by 0.008, a numeric batch-size effect; in float32 it is 4e-7 and
the greedy tokens agree).

Q-134, measured live on 2026-09-26 14:39: rank 0 read ``running 1 waiting 4
slots 2 in_use 1``, half of each rank's KV budget unreserved, and a 69-token
canary queued 308 s — a batch formed only when the previous one ended.
"""

import threading
import time
from dataclasses import dataclass

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
pytest.importorskip("fastapi")
pytestmark = pytest.mark.requires_mlx

from rapid_mlx.distributed import pipeline_qwen4 as pipe  # noqa: E402
from rapid_mlx.distributed import pipeline_qwen4_serve as serve  # noqa: E402
from rapid_mlx.models.qwen4_exp import Model, ModelArgs  # noqa: E402

from .test_pipeline_qwen4 import _tiny_text_config  # noqa: E402

PREFILL_STEP = 16
LONG = [7, 19, 3, 88, 42, 254, 17, 5, 200, 61, 9, 33, 250, 14, 71, 8, 99, 123] * 3
SHORT = [12, 34, 56, 78, 90, 11, 22, 254, 44, 13, 24, 35]
THIRD = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4, 3]


@pytest.fixture(scope="module")
def group():
    return mx.distributed.init(strict=False)


@pytest.fixture(scope="module")
def stage(group):
    mx.random.seed(20260926)
    text_config = _tiny_text_config()
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=text_config))
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return pipe.PipelineStage(
        model, group, 0, text_config["num_hidden_layers"], mx.float32
    )


def _row(ids, max_tokens=64, **directive) -> serve._Row:
    return serve._Row(list(ids), max_tokens, 0.0, 1.0, **directive)


def _join(engine: serve._Engine, ids) -> int:
    """Prefill ``ids`` alone and return its first token (the row then runs)."""
    engine.apply(serve._Plan(joiner=_row(ids)))
    first = None
    while engine.joining is not None:
        first, _ = engine.prefill(None)
    return first


def _solo(stage, ids, tokens: int) -> list[int]:
    engine = serve._Engine(stage, None, PREFILL_STEP)
    out = [_join(engine, ids)]
    while len(out) < tokens:
        out.append(engine.decode(None)[0][0])
    return out


def test_rows_that_join_and_leave_mid_decode_answer_as_if_each_ran_alone(stage):
    engine = serve._Engine(stage, None, PREFILL_STEP)
    streams = {"long": [_join(engine, LONG)], "short": [], "third": []}
    names = ["long"]

    def decode() -> None:
        tokens, _ = engine.decode(None)
        for name, token in zip(names, tokens):
            streams[name].append(token)

    for _ in range(10):
        decode()
    # SHORT prefills in its own cache, one chunk per decode step of LONG.
    engine.apply(serve._Plan(joiner=_row(SHORT)))
    while engine.joining is not None:
        decode()
        first, _ = engine.prefill(None)
        if first is not None:
            streams["short"].append(first)
            names.append("short")
    for _ in range(15):
        decode()
    engine.apply(serve._Plan(leave=[0]))  # LONG leaves; SHORT was the padded row
    names.remove("long")
    for _ in range(10):
        decode()
    engine.apply(serve._Plan(joiner=_row(THIRD)))
    while engine.joining is not None:
        decode()
        first, _ = engine.prefill(None)
        if first is not None:
            streams["third"].append(first)
            names.append("third")
    for _ in range(12):
        decode()

    for name, ids in (("long", LONG), ("short", SHORT), ("third", THIRD)):
        got = streams[name]
        assert len(got) > 10
        assert got == _solo(stage, ids, len(got)), name


@dataclass(eq=False)
class _Collected(serve._Job):
    """A request whose events land in a list (the HTTP side is not under test)."""

    def __post_init__(self):
        self.seen: list = []
        self.lock = threading.Lock()

    def push(self, item) -> None:
        with self.lock:
            self.seen.append(item)

    def tokens(self) -> list[int]:
        with self.lock:
            return [value for kind, value in self.seen if kind == "token"]


def _job(ids, max_tokens: int) -> _Collected:
    return _Collected(_row(ids, max_tokens), None, None)


def _state(max_batch=2, kv=None) -> serve._State:
    return serve._State(served="tiny", context=4096, max_batch=max_batch, kv=kv)


def _wait(predicate, what: str) -> None:
    # A hang guard for the harness only; the engine has no clock.
    deadline = time.monotonic() + 120
    while not predicate():
        assert time.monotonic() < deadline, what
        time.sleep(0.005)


def _run_rank0(state, engine, group) -> threading.Thread:
    thread = threading.Thread(
        target=serve._rank0_loop,
        args=(state, engine, group, serve._Wake(group)),
        daemon=True,
    )
    thread.start()
    return thread


def test_a_queued_request_joins_a_running_generation_between_decode_steps(stage, group):
    state = _state()
    engine = serve._Engine(stage, None, PREFILL_STEP)
    loop = _run_rank0(state, engine, group)
    long_job = _job(LONG, 3000)
    state.jobs.put(long_job)
    _wait(lambda: long_job.produced >= 20, "the long row never decoded")
    canary = _job(SHORT, 6)
    state.jobs.put(canary)
    _wait(lambda: canary.finished, "the canary never finished")
    # The canary was admitted, prefilled and answered while the long row ran.
    assert not long_job.finished
    assert 20 <= long_job.produced < 3000
    assert canary.seen[-1] == ("done", "length")
    assert canary.tokens() == _solo(stage, SHORT, 6)
    long_job.cancelled = True
    _wait(lambda: not state.active, "the cancelled row never left")
    state.jobs.put(None)
    loop.join(timeout=60)
    assert not loop.is_alive()
    got = long_job.tokens()
    assert got == _solo(stage, LONG, len(got))


class _FakeKv:
    """The 14:39 budgets; every reservation of a full context costs one slot."""

    budgets = [7_465_986_080, 9_720_686_592]
    slots = 2

    def __init__(self, context: int):
        self.context = context

    def reserve(self, lengths: list[int]) -> list[int]:
        rows = len(lengths) * max(lengths) / self.context
        return [int(rows * budget / self.slots) for budget in self.budgets]

    def fits(self, lengths: list[int]) -> bool:
        return all(n <= b for n, b in zip(self.reserve(lengths), self.budgets))


def _start_fake(engine: serve._Engine, row: serve._Row) -> None:
    """What ``_Engine._start`` leaves behind, without a stage: the scheduler reads the ranges only."""
    engine.prefilling.append(
        serve._Joining(
            row,
            None,
            None,
            None,
            None,
            serve.prefill_chunks(
                row.cached,
                len(row.ids),
                engine.prefill_step,
                row.store_at if row.store_id else 0,
            ),
        )
    )


def _apply(scheduler: serve._Scheduler, plan: serve._Plan) -> None:
    """``_ticks``' plan step on rank 0, the engine's rows faked (no forwards)."""
    engine = scheduler.engine
    if plan.abort:
        engine.prefilling = [
            joining
            for index, joining in enumerate(engine.prefilling)
            if index not in plan.abort
        ]
    if plan.joiner is not None:
        _start_fake(engine, plan.joiner)
    scheduler.applied(plan)


def _chunk(scheduler: serve._Scheduler, seconds: float = 0.0) -> int | None:
    """``_ticks``' chunk step on rank 0: the target's next chunk; its first token if it joined."""
    engine = scheduler.engine
    target = engine.target
    joining = engine.prefilling[target]
    start, stop = joining.ranges.pop(0)
    first = None
    if not joining.ranges:
        engine.prefilling = [row for row in engine.prefilling if row is not joining]
        first = 5
    scheduler.chunked(target, stop - start, first, seconds)
    return first


def test_the_1439_queue_admits_the_canary_first_beside_the_running_row():
    """The 14:39 status walked through the scheduler (no engine work)."""
    context = 131_072
    state = serve._State(served="flash", context=context, max_batch=2)
    state.kv = _FakeKv(context)

    def default_request(prompt: int) -> _Collected:
        # No max_tokens: the HTTP side grants context - prompt - 1.
        return _job([1] * prompt, context - prompt - 1)

    running = default_request(6_896)
    running.produced = 12_351
    waiting = [default_request(n) for n in (39_194, 39_186, 39_132)]
    canary = default_request(69)
    for job in [*waiting, canary]:
        state.jobs.put(job)
    scheduler = serve._Scheduler(state, serve._Engine(None, None, 2048))
    scheduler.running = [running]
    scheduler._publish()
    assert state.reserved == [3_732_993_040, 4_860_343_296]  # "kv_reserved" at 14:39

    # A slot is free and two full-context reservations fit: the next decode
    # step's collective announces a plan, and the plan admits the request
    # with the fewest prompt tokens left — the canary (Q-145), not the oldest.
    assert scheduler.wants_plan(chunk=False)
    plan = scheduler.plan(block=False)
    assert plan.joiner is canary.row and plan.leave == []
    _apply(scheduler, plan)
    assert state.reserved == _FakeKv.budgets

    # Both slots are held (the running row and the prefilling canary), so
    # nothing else is admitted; the canary's one chunk samples its first
    # token and it joins the running batch at once.
    assert not scheduler.wants_plan(chunk=False)
    assert not scheduler.wants_plan(chunk=True)
    assert _chunk(scheduler) == 5
    assert scheduler.running == [running, canary]
    assert canary.produced == 1
    running.finished = True
    assert scheduler.wants_plan(chunk=False)
    plan = scheduler.plan(block=False)
    # The long three are ordered by prompt tokens left: 39,132 first.
    assert plan.leave == [0] and plan.joiner is waiting[2].row
    assert state.waiting == [waiting[0], waiting[1]] and state.jobs.qsize() == 0


def test_a_request_that_does_not_fit_waits_at_the_head_until_a_row_leaves():
    context = 1_000
    state = serve._State(served="t", context=context, max_batch=4)
    state.kv = _FakeKv(context)
    running = [_job([1] * 10, 989), _job([1] * 10, 989)]  # two full slots
    short = _job([2] * 5, 3)  # fits by length, but a third row is 3 x longest
    state.jobs.put(short)
    scheduler = serve._Scheduler(state, serve._Engine(None, None, PREFILL_STEP))
    scheduler.running = list(running)
    assert not scheduler.wants_plan(chunk=False)
    assert state.waiting == [short]  # held at the head, never dropped
    running[1].cancelled = True
    assert scheduler.wants_plan(chunk=False)
    plan = scheduler.plan(block=False)
    assert plan.leave == [1] and plan.joiner is short.row


def test_the_prefill_share_spaces_chunks_by_measured_time():
    scheduler = serve._Scheduler(_state(), serve._Engine(None, None, 2))
    scheduler.running = [_job(SHORT, 99)]
    prefilling = _job(LONG, 9)
    scheduler.prefilling = [prefilling]
    _start_fake(scheduler.engine, prefilling.row)
    scheduler.credit = 0.0
    # The first chunk runs at once: a short request is answered after one step.
    assert scheduler.decode_words() == [0, 1]
    scheduler.decoded([1], seconds=0.25)
    _chunk(scheduler, seconds=1.0)  # a chunk took four decode steps
    runs = []
    for _ in range(8):
        words = scheduler.decode_words()
        runs.append(words[1])
        scheduler.decoded([1], seconds=0.25)
        if words[1]:
            _chunk(scheduler, seconds=1.0)
    # Equal shares: a 1 s chunk runs with every four 0.25 s decode steps
    # (credit -0.75 -> -0.5 -> -0.25 -> 0.0, then the chunk rides the step).
    assert runs == [0, 0, 0, 1, 0, 0, 0, 1]


def test_a_worker_rank_takes_every_decision_from_the_collectives(
    stage, group, monkeypatch
):
    """Replay rank 0's collectives into the worker code path: same forwards, same order."""
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
    state = _state()
    loop = _run_rank0(state, serve._Engine(stage, None, PREFILL_STEP), group)
    first = _job(LONG, 200)
    state.jobs.put(first)
    _wait(lambda: first.produced >= 5, "the first row never decoded")
    second, third = _job(SHORT, 30), _job(THIRD, 25)
    state.jobs.put(second)
    state.jobs.put(third)
    _wait(lambda: first.finished and second.finished and third.finished, "rows")
    _wait(lambda: not state.active, "the rows never left")
    state.jobs.put(None)
    loop.join(timeout=60)
    assert not loop.is_alive()
    rank0_forwards, forwards[:] = list(forwards), []

    replay = iter(recorded)

    def replayed(group_, value):
        expected = next(replay)
        assert expected.shape == value.shape, "a collective would not pair"
        return expected

    monkeypatch.setattr(serve, "_all_sum", replayed)
    serve._ticks(serve._Engine(stage, None, PREFILL_STEP), group, 2, serve._Wake(group))
    assert next(replay, None) is None
    assert forwards == rank0_forwards
    # The replay covered joins between decode steps, not only a lone row.
    assert any(shape[0] == 2 for shape, _, _ in rank0_forwards)


def test_a_batch_rope_rotates_each_row_as_its_own_table_does():
    from rapid_mlx.models.qwen4_exp_vision import MRopePositions

    image_row = MRopePositions.from_rows(
        [([[0, 1, 1, 1, 4], [0, 1, 1, 2, 4], [0, 1, 2, 1, 4]], -2)]
    )
    batch = serve._batch_rope([None, image_row, None])
    logical = mx.array([[0, 3, 9], [0, 3, 9], [2, 5, 6]])
    at = batch.at(logical)
    assert at[:, 0].tolist() == [[0, 3, 9]] * 3
    assert at[:, 2].tolist() == [[2, 5, 6]] * 3
    assert at[:, 1].tolist() == image_row.at(logical[1:2])[:, 0].tolist()
    assert serve._batch_rope([None, None]) is None
