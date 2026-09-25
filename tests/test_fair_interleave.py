"""LeanZero Q-103: a short request is not parked behind long prefills.

Two mechanisms, both measured on a 27B Qwen3.8 MTP engine (the leanzero
fork's ledger row Q-103): the ordinary vendored MTP verifier used to hold
admission at one running request for that request's whole life, prefill
included, so a 1-token request waited behind every long prompt queued ahead
of it; and mlx-lm prompts one ``prefill_step_size`` chunk per row per step,
padded, so a step shared by several rows (or a decoding row) lasted as long
as all of their chunks together.
"""

from collections import deque
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx  # noqa: E402

from rapid_mlx.scheduler import (  # noqa: E402
    Scheduler,
    SchedulerConfig,
    _install_mtp_vendored,
)


class _Rows(list):
    """A prompt batch: ``len`` is its rows, plus the chunk attribute."""

    def __init__(self, n=0, prefill_step_size=2048):
        super().__init__(object() for _ in range(n))
        self.prefill_step_size = prefill_step_size


class _Gen:
    """``GenerationBatch._step`` shape: emit ``_next_tokens``, stash the next."""

    def __init__(self):
        self.uids = []
        self.tokens = [[]]
        self.logits_processors = []
        self.prompt_cache = []
        self.max_tokens = [4096]
        self._next_tokens = mx.array([500], dtype=mx.uint32)
        self._next_logprobs = [mx.array([0.0])]
        self.orig_step_calls = 0

    def _step(self):
        self.orig_step_calls += 1
        current = [int(self._next_tokens[i].item()) for i in range(len(self.uids))]
        self._next_tokens = mx.array([999] * len(self.uids), dtype=mx.uint32)
        return current, self._next_logprobs

    def next(self):
        return []


class _StubModel:
    mtp_forward = object()
    make_mtp_cache = object()
    mtp = object()


@pytest.fixture
def fake_mtp(monkeypatch):
    from rapid_mlx.spec_decode.mtp import generator as gen_mod

    constructed = []

    class _FakeGen:
        def __init__(self):
            self.n = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.n += 1
            return (1000 + self.n, mx.array([0.0]), False)

        def close(self):
            pass

    def _fake(*args, **kwargs):
        constructed.append(kwargs)
        return _FakeGen()

    monkeypatch.setattr(gen_mod, "mtp_generate_step", _fake)
    return constructed


def _install(*, yield_to_contention, prompt_rows=0, queued=0, uids=(7,)):
    gb = _Gen()
    gb.uids = list(uids)
    batch_gen = SimpleNamespace(
        _generation_batch=gb,
        completion_batch_size=32,
        _prompt_batch=_Rows(prompt_rows),
        _unprocessed_sequences=deque(object() for _ in range(queued)),
    )
    greedy = SimpleNamespace(sampling_params=SimpleNamespace(temperature=0.0))
    assert _install_mtp_vendored(
        batch_gen,
        model=_StubModel(),
        requests={f"req-{u}": greedy for u in (1, 2, 7)},
        uid_to_request_id={u: f"req-{u}" for u in (1, 2, 7)},
        yield_to_contention=yield_to_contention,
    )
    return batch_gen, gb


# --- the MTP admission gate -------------------------------------------------


def _mtp_scheduler(batch_generator, **config):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SchedulerConfig(
        spec_decode="mtp", max_num_seqs=8, max_concurrent_requests=8, **config
    )
    scheduler.spec_decode_runtime_attempted = True
    scheduler.spec_decode_runtime_method = "mtp"
    scheduler.batch_generator = batch_generator
    scheduler.running = {}
    return scheduler


def test_admission_stays_open_until_a_lone_request_owns_the_verifier():
    bg = SimpleNamespace(
        _mtp_vendored_yields_to_contention=True, _mtp_vendored_admission_owner=None
    )
    scheduler = _mtp_scheduler(bg)
    assert scheduler._max_running_sequences() == 8

    bg._mtp_vendored_admission_owner = 7
    assert scheduler._max_running_sequences() == 1


def test_admission_keeps_batch_one_when_the_verifier_cannot_yield():
    bg = SimpleNamespace(
        _mtp_vendored_yields_to_contention=False, _mtp_vendored_admission_owner=None
    )
    assert _mtp_scheduler(bg)._max_running_sequences() == 1

    bg._mtp_vendored_yields_to_contention = True
    off = _mtp_scheduler(bg, mtp_yield_to_contention=False)
    assert off._max_running_sequences() == 1


def test_first_decode_beside_a_prefill_runs_plain_and_takes_no_lock(fake_mtp):
    batch_gen, gb = _install(yield_to_contention=True, prompt_rows=1)
    assert batch_gen._mtp_vendored_yields_to_contention is True

    tokens, _ = gb._step()

    assert tokens == [500]
    assert gb.orig_step_calls == 1
    assert fake_mtp == []
    assert batch_gen.completion_batch_size == 32
    assert getattr(batch_gen, "_mtp_vendored_admission_owner", None) is None
    assert batch_gen._mtp_vendored_stats["ft_contended"] == 1

    # The prefill finished and left: this request still never primes MTP
    # from a mid-stream placeholder.
    batch_gen._prompt_batch = _Rows(0)
    gb._step()
    assert gb.orig_step_calls == 2
    assert fake_mtp == []


def test_a_queued_arrival_counts_as_contention(fake_mtp):
    batch_gen, gb = _install(yield_to_contention=True, queued=1)
    gb._step()
    assert fake_mtp == []
    assert batch_gen.completion_batch_size == 32


def test_a_lone_first_decode_still_primes_mtp_and_locks_admission(fake_mtp):
    batch_gen, gb = _install(yield_to_contention=True)

    tokens, _ = gb._step()

    assert tokens == [500]
    assert gb.orig_step_calls == 0
    assert len(fake_mtp) == 1
    assert batch_gen.completion_batch_size == 1
    assert batch_gen._mtp_vendored_admission_owner == 7


def test_rows_that_decoded_together_stay_plain_when_one_is_left(fake_mtp):
    batch_gen, gb = _install(yield_to_contention=True, uids=(1, 2))
    gb._step()
    assert gb.orig_step_calls == 1

    gb.uids = [2]
    gb._next_tokens = mx.array([999], dtype=mx.uint32)
    gb._step()
    assert gb.orig_step_calls == 2
    assert fake_mtp == []
    assert batch_gen.completion_batch_size == 32


def test_without_yield_the_batch_one_contract_is_unchanged(fake_mtp):
    batch_gen, gb = _install(yield_to_contention=False, prompt_rows=1)
    assert batch_gen._mtp_vendored_yields_to_contention is False
    gb._step()
    assert len(fake_mtp) == 1
    assert batch_gen.completion_batch_size == 1


# --- the shared-step bound --------------------------------------------------


class _Model:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        if self.fail:
            raise RuntimeError("no metal here")
        return x


def _fair_scheduler(*, prompt_rows, decoding=0, queued=0, chunk=2048, ratio=16.0):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SchedulerConfig(
        prefill_step_size=2048, fair_prefill_step_ratio=ratio
    )
    scheduler.model = _Model()
    scheduler.batch_generator = SimpleNamespace(
        prefill_step_size=chunk,
        prefill_batch_size=8,
        _prompt_batch=_Rows(prompt_rows, chunk),
        _generation_batch=_Rows(decoding),
        _unprocessed_sequences=deque(object() for _ in range(queued)),
        _prompt_tokens_counter=0,
    )
    return scheduler


def test_a_lone_prefill_keeps_the_configured_chunk():
    scheduler = _fair_scheduler(prompt_rows=1)
    scheduler._fair_forward_s = 0.1
    scheduler._fair_seconds_per_token = 0.004

    assert scheduler._apply_fair_prefill_size() is None
    assert scheduler.batch_generator.prefill_step_size == 2048
    assert scheduler.model.calls == 0


def test_a_shared_step_is_sized_from_the_measured_forward_and_token_cost():
    scheduler = _fair_scheduler(prompt_rows=3)
    scheduler._fair_forward_s = 0.1
    scheduler._fair_seconds_per_token = 0.004

    # 16 forwards of 0.1 s = 1.6 s; 3 rows at 0.004 s/token -> 133 per row.
    assert scheduler._apply_fair_prefill_size() == 133
    bg = scheduler.batch_generator
    assert bg.prefill_step_size == 133
    assert bg._prompt_batch.prefill_step_size == 133
    assert scheduler._fair_bounded_steps == 1


def test_a_decoding_row_or_a_queued_arrival_makes_a_step_shared():
    decoding = _fair_scheduler(prompt_rows=1, decoding=1)
    queued = _fair_scheduler(prompt_rows=1, queued=1)
    for scheduler in (decoding, queued):
        scheduler._fair_forward_s = 0.1
        scheduler._fair_seconds_per_token = 0.004
    assert decoding._apply_fair_prefill_size() == 400
    assert queued._apply_fair_prefill_size() == 200


def test_the_bound_only_ever_lowers_the_memory_guarded_chunk():
    scheduler = _fair_scheduler(prompt_rows=1, decoding=1, chunk=256)
    scheduler._fair_forward_s = 0.1
    scheduler._fair_seconds_per_token = 0.004
    assert scheduler._apply_fair_prefill_size() == 256
    assert scheduler.batch_generator.prefill_step_size == 256
    assert scheduler._fair_bounded_steps == 0


def test_the_forward_is_measured_once_on_the_first_shared_step():
    scheduler = _fair_scheduler(prompt_rows=2)
    scheduler._apply_fair_prefill_size()
    assert scheduler._fair_forward_s is not None
    calls = scheduler.model.calls
    scheduler._apply_fair_prefill_size()
    assert scheduler.model.calls == calls


def test_an_unmeasurable_forward_is_reported_and_leaves_the_chunk_alone():
    scheduler = _fair_scheduler(prompt_rows=2)
    scheduler.model = _Model(fail=True)
    scheduler._fair_seconds_per_token = 0.004

    assert scheduler._apply_fair_prefill_size() is None
    assert scheduler._fair_forward_error == "RuntimeError: no metal here"
    assert scheduler.batch_generator.prefill_step_size == 2048


def test_no_token_cost_yet_leaves_the_chunk_alone():
    scheduler = _fair_scheduler(prompt_rows=2)
    scheduler._fair_forward_s = 0.1
    assert scheduler._apply_fair_prefill_size() is None
    assert scheduler.batch_generator.prefill_step_size == 2048


def test_ratio_zero_disables_the_bound():
    scheduler = _fair_scheduler(prompt_rows=3, ratio=0)
    scheduler._fair_forward_s = 0.1
    scheduler._fair_seconds_per_token = 0.004
    assert scheduler._apply_fair_prefill_size() is None
    assert scheduler.batch_generator.prefill_step_size == 2048


def test_token_cost_is_read_only_from_unpadded_full_steps():
    scheduler = _fair_scheduler(prompt_rows=2, chunk=100)
    bg = scheduler.batch_generator

    bg._prompt_tokens_counter = 200
    scheduler._record_fair_prefill_step(1.0, 0, 100)
    assert scheduler._fair_seconds_per_token == pytest.approx(0.005)

    # A remainder row was padded to the chunk: its wall time is not per token.
    bg._prompt_tokens_counter = 350
    scheduler._record_fair_prefill_step(9.0, 200, 100)
    assert scheduler._fair_seconds_per_token == pytest.approx(0.005)

    # A decode-only step prompted nothing.
    scheduler._record_fair_prefill_step(0.1, 350, 100)
    assert scheduler._fair_seconds_per_token == pytest.approx(0.005)


@pytest.mark.parametrize("ratio", [-1, float("nan"), float("inf"), True])
def test_the_ratio_must_be_a_finite_non_negative_number(ratio):
    with pytest.raises(ValueError, match="fair_prefill_step_ratio"):
        SchedulerConfig(fair_prefill_step_ratio=ratio)
