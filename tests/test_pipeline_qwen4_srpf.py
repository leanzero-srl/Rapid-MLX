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
    # A 12-token request goes ahead only while it takes at most half the
    # long prompt's slack (162 less what went ahead): at most 12 of them,
    # and every later one waits.
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


def test_a_prefill_whose_slack_is_spent_is_not_jumped():
    """A later request goes ahead only while it fits in the earlier one's slack."""
    state = serve._State(served="t", context=100_000, max_batch=4)
    scheduler = serve._Scheduler(state, serve._Engine(None, None, 100))
    long_job = _job([1] * 1_000, 4)
    state.jobs.put(long_job)
    _apply(scheduler, scheduler.plan(block=True))
    _chunk(scheduler)  # 900 left
    assert (
        long_job.waited == 0 and long_job.prefill == 1_000
    )  # its own chunk is no wait
    long_job.waited = 980
    state.jobs.put(_job([2] * 10, 4))
    assert scheduler.wants_plan(chunk=False)  # 10 is half of 1,000 - 980
    long_job.waited = 981
    assert not scheduler.wants_plan(chunk=False)
    assert scheduler.plan(block=False).joiner is None
    # A queued request whose slack is spent goes before a shorter later one.
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
    waiting.waited = 461  # 20 is more than half of 500 - 461
    assert scheduler2._head() is waiting


def test_only_later_arrivals_age_a_request():
    """An earlier arrival's chunk is the order a request had anyway; a later one's is a jump."""
    state = serve._State(served="t", context=100_000, max_batch=4)
    scheduler = serve._Scheduler(state, serve._Engine(None, None, 100))
    first, second, third = _job([1] * 300, 4), _job([2] * 400, 4), _job([3] * 50, 4)
    for job in (first, second):
        state.jobs.put(job)
    _apply(scheduler, scheduler.plan(block=True))
    assert scheduler.prefilling == [first]
    _chunk(scheduler)
    assert second.waited == 0  # first arrived before it
    state.jobs.put(third)
    _apply(scheduler, scheduler.plan(block=False))
    assert scheduler.prefilling == [first, third]
    assert _chunk(scheduler) == 5  # third's one chunk: it joined
    assert (first.waited, second.waited, third.waited) == (50, 50, 0)


class _Drive:
    """Rank 0's tick loop over the scheduler alone, the engine's rows faked and
    time taken from the rates goose's load runs measured on the Flash split.

    It mirrors ``_ticks``: a plan when the previous collective announced one,
    a decode step while rows run, the target's chunk when the prefill share
    allows.  Requests arrive at their times (seen by the next collective's
    words, as on the live loop) and a finished load worker sends its next
    prompt at once, as goose's load.py does.
    """

    PREFILL_STEP = 2048
    PREFILL_TOKENS_PER_S = 333.0  # 26d: 39k-token prompts' first tokens 117.55 s apart
    DECODE_S = 0.123  # 26d: a load row's 37 tokens took 4.55 s after its first
    EOS = 2

    def __init__(self, context=131_072, max_batch=2):
        self.state = serve._State(served="flash", context=context, max_batch=max_batch)
        self.state.kv = _FakeKv(context)
        self.state.eos_ids = frozenset({self.EOS})
        self.engine = serve._Engine(None, None, self.PREFILL_STEP)
        self.scheduler = serve._Scheduler(self.state, self.engine)
        self.now = 0.0
        self.chunk_ran = (-1.0, -1.0)
        self.chunks = 0
        self.prefilled = 0
        self.arrivals: list = []
        self.log: list[dict] = []
        self.outputs: dict[int, int] = {}

    def at(self, when: float, name: str, prompt: int, output: int, then=None):
        """``name`` arrives at ``when``: ``prompt`` tokens, ``output`` generated
        (an EOS ends it; ``output`` 1 = max_tokens 1, the canary), ``then``
        (``[(prompt, output), ...]``) its worker's next prompts."""
        self.arrivals.append((when, len(self.arrivals), name, prompt, output, then))

    def _arrive(self) -> None:
        self.arrivals.sort(key=lambda a: a[:2])
        while self.arrivals and self.arrivals[0][0] <= self.now:
            when, _, name, prompt, output, then = self.arrivals.pop(0)
            # Queued while the last chunk ran: that chunk is part of its wait.
            during = self.chunk_ran[0] <= when < self.chunk_ran[1]
            context = self.state.context
            max_tokens = 1 if output == 1 else context - prompt - 1
            job = _Collected(serve._Row([1] * prompt, max_tokens, 0.0, 1.0), None, None)
            record = {
                "name": name,
                "prompt": prompt,
                "arrived": when,
                "chunks_at": self.chunks - during,
                "prefilled_at": self.prefilled,
                "first": None,
                "done": None,
            }
            self.log.append(record)
            self.outputs[id(job)] = output

            def push(item, record=record, then=then, name=name):
                if item[0] == "token" and record["first"] is None:
                    record["first"] = self.now
                    record["chunk_steps"] = self.chunks - record["chunks_at"]
                    record["tokens_before"] = self.prefilled - record["prefilled_at"]
                if item[0] == "done":
                    record["done"] = self.now
                    if then:
                        (prompt, output), *rest = then
                        self.at(self.now, name, prompt, output, rest)

            job.push = push
            self.state.jobs.put(job)

    def _idle(self) -> bool:
        return not self.scheduler.running and not self.engine.prefilling

    def run(self, until) -> None:
        scheduler, engine = self.scheduler, self.engine
        pending = True
        while not until(self):
            self._arrive()
            if pending:
                if (
                    self._idle()
                    and not self.state.jobs.qsize()
                    and not self.state.waiting
                ):
                    assert self.arrivals, "nothing left to arrive and nothing running"
                    self.now = min(a[0] for a in self.arrivals)
                    continue
                _apply(scheduler, scheduler.plan(block=False))
                pending = self._idle()
                if pending:
                    continue
            chunk = bool(engine.prefilling)
            if scheduler.running:
                words = scheduler.decode_words()
                self.now += self.DECODE_S
                tokens = [
                    self.EOS if job.produced + 1 >= self.outputs[id(job)] else 1
                    for job in scheduler.running
                ]
                scheduler.decoded(tokens, self.DECODE_S)
                pending, chunk = bool(words[0]), chunk and bool(words[1])
            if chunk:
                words = scheduler.chunk_words()
                start, stop = engine.prefilling[engine.target].ranges[0]
                seconds = (stop - start) / self.PREFILL_TOKENS_PER_S
                self.chunk_ran = (self.now, self.now + seconds)
                self.now += seconds
                self.chunks += 1
                self.prefilled += stop - start
                _chunk(scheduler, seconds)
                pending = bool(words[0])
            pending = pending or self._idle()

    def first(self, name: str, nth: int = 0) -> dict | None:
        rows = [r for r in self.log if r["name"] == name]
        return rows[nth] if nth < len(rows) else None

    def answered(self, name: str, nth: int = 0) -> bool:
        row = self.first(name, nth)
        return row is not None and row["first"] is not None


def test_the_26d_steady_load_admits_every_canary_between_two_chunks():
    """LOAD-2026-09-26d replayed: three load workers x ~39k-token prompts and a
    1-token canary at +0 / +63.3 / +124.8 / +538.2 s.

    Live on fork 419306f70 the canaries' first tokens came after 3.32 / 1.48 /
    353.41 / 297.16 s: once the workers' second prompts queued, every long
    request had 'waited' through the earlier long prefills it was behind
    anyway, was protected before it started, and nothing could jump it —
    /v1/status during canary 3 read prefill 20,224/39,016, waiting [27,
    39,259, 38,849] beside a free slot.  Counting only jumpers' tokens, each
    canary is seen at the next chunk boundary and prefilled in one chunk.
    """
    drive = _Drive()
    # Prompt sizes: 26c's three (the same load.py prompts), then the three
    # /v1/status read during canary 3; outputs from 26d's requests.csv.
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

    canaries = [drive.first(f"canary{i}") for i in (1, 2, 3, 4)]
    waits = [round(c["first"] - c["arrived"], 1) for c in canaries]
    steps = [c["chunk_steps"] for c in canaries]
    # A canary queued mid-chunk waits out that chunk, is seen by the next
    # chunk's words, and prefills in one chunk of its own: at most three
    # chunk-steps, <= ~18.5 s at the measured ~6.2 s per 2,048-token chunk.
    # This replay: 0.1 / 10.7 / 4.2 / 11.3 s (1 / 3 / 1 / 3 steps).  On
    # 419306f70 the same replay gives canaries 3 and 4 236.9 / 311.3 s (40 /
    # 52 steps); live they took 353.41 / 297.16 s.
    assert all(step <= 3 for step in steps), (steps, waits)
    chunk_s = drive.PREFILL_STEP / drive.PREFILL_TOKENS_PER_S
    assert all(wait <= 3 * chunk_s for wait in waits), waits
    # The long prompts still flow: every worker's first two prompts answered
    # (the run stops there), in first-come order — each barely shorter than
    # another, so none takes half an earlier one's slack.
    longs = sorted(
        (r["first"], r["arrived"])
        for r in drive.log
        if not r["name"].startswith("canary") and r["first"]
    )
    assert len(longs) == 6
    assert [arrived for _, arrived in longs] == sorted(arrived for _, arrived in longs)


def _endless_stream(drive: _Drive, long_prompt: int, short_prompt: int) -> dict:
    """The long prompt at 0, then a short one every quarter chunk — far more
    than are served — until the long prompt's first token."""
    drive.at(0.0, "long", long_prompt, 2)
    chunk_s = drive.PREFILL_STEP / drive.PREFILL_TOKENS_PER_S
    for index in range(4 * long_prompt // short_prompt):
        drive.at(0.5 + index * chunk_s / 4, f"short{index}", short_prompt, 1)
    drive.run(until=lambda d: d.answered("long"))
    return drive.first("long")


def test_an_endless_stream_of_short_requests_stretches_a_long_prefill_at_most_twofold(
    monkeypatch,
):
    """Short requests arriving faster than they are served: the long prompt's
    first token still comes within twice its solo prefill (in prefill tokens,
    what a compute-bound prefill's time is), after the stream took all the
    slack it may."""
    long_prompt, short_prompt = 39_016, 1_024
    drive = _Drive(max_batch=8)
    long_row = _endless_stream(drive, long_prompt, short_prompt)
    answered = [r for r in drive.log if r["name"] != "long" and r["first"] is not None]
    assert long_row["tokens_before"] == long_prompt + len(answered) * short_prompt
    assert long_row["tokens_before"] <= 2 * long_prompt
    # The stream jumped it until the slack left was under twice a short one ...
    assert long_prompt - len(answered) * short_prompt < 2 * short_prompt
    # ... and the shorts still queued then waited for it.
    assert any(r["first"] is None for r in drive.log if r["name"] != "long")

    # Negative control: with no slack bound the same stream holds the long
    # prompt until the harness stops offering (every short goes first).
    monkeypatch.setattr(
        serve._Scheduler, "_slack", staticmethod(lambda *_, **__: 1 << 60)
    )
    control = _endless_stream(_Drive(max_batch=8), long_prompt, short_prompt)
    assert control["tokens_before"] > 2 * long_prompt


def test_a_long_prompt_jumps_an_earlier_one_only_with_half_its_slack():
    """A later long prompt one percent shorter keeps first-come order (it
    would spend the earlier one's whole slack and hold every canary behind
    it); one under half the earlier one's prefill goes first; both leave
    the earlier one within twice its own prefill after the running row."""
    for later_prompt, later_first in ((38_849, False), (12_000, True)):
        drive = _Drive(max_batch=2)
        drive.at(0.0, "running", 30_000, 2)
        drive.at(1.0, "earlier", 39_259, 2)
        drive.at(2.0, "later", later_prompt, 2)
        drive.at(3.0, "canary", 27, 1)
        drive.run(until=lambda d: d.answered("earlier") and d.answered("later"))

        earlier, later, canary = (
            drive.first(n) for n in ("earlier", "later", "canary")
        )
        assert (later["first"] < earlier["first"]) is later_first, later_prompt
        assert canary["chunk_steps"] <= 3
        # From the earlier one's arrival: the running prompt's rest (earlier
        # arrivals are no jump), then jumpers, then its own prefill.
        running_rest = 30_000 - drive.PREFILL_STEP
        assert earlier["tokens_before"] <= running_rest + 2 * 39_259


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
    assert index.admit(first, index.kv.reserve([30])) == []
    assert first.store_at == 20 and index.pending == {first.store_id: [20, 40]}
    # Beside it, the batch leaves room 40 per rank: the pending [20, 40] plus
    # a second [20, 40] would not fit rank 1, so the second is not directed.
    second = serve._Row([2] * 30, 4, 0.0, 1.0, boundary=20)
    index.admit(second, index.kv.reserve([30, 30]))
    assert second.store_id == 0
    assert index.status()["skipped"] == {"no_room_beside_the_batch": 1}
    # The first row was aborted before its snapshot: the charge goes.
    index.forget(first.store_id)
    third = serve._Row([3] * 30, 4, 0.0, 1.0, boundary=20)
    index.admit(third, index.kv.reserve([30, 30]))
    assert third.store_id and index.pending == {third.store_id: [20, 40]}
    version = index.version
    index.stored(third.store_id, tuple(third.ids[:20]), [20, 40])
    assert index.pending == {} and index.version == version + 1
