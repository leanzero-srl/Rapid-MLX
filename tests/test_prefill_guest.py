"""LeanZero Q-103: one short request may enter between a long prompt's prefill chunks.

Under the ordinary batch-1 MTP verifier a 1-token request queued behind long
prompts waited 146-162 s on the Studio (3 x 17k prompts). A guest is let in
only while the one running request is still prefilling; decode stays batch-1.
"""

from collections import deque
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

from rapid_mlx.scheduler import Scheduler, SchedulerConfig  # noqa: E402


def _req(rid, tokens, uid=None):
    return SimpleNamespace(
        request_id=rid,
        remaining_tokens=list(range(tokens)),
        prompt_token_ids=[],
        batch_uid=uid,
    )


def _scheduler(*, host_left=10_000, host_where="prompt", owner=None, **config):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SchedulerConfig(
        spec_decode="mtp", prefill_step_size=2048, completion_batch_size=32, **config
    )
    scheduler.spec_decode_runtime_attempted = True
    scheduler.spec_decode_runtime_method = "mtp"
    host = _req("host", 17_000, uid=0)
    scheduler.running = {"host": host}
    prompt = SimpleNamespace(uids=[], prefill_step_size=2048, prompt_cache=[])
    processing, unprocessed, generating = [], deque(), []
    if host_where == "prompt":
        prompt.uids = [0]
        processing = [[[list(range(host_left - 1)), [7]], 17_000 - host_left, 17_000]]
    elif host_where == "queued":
        unprocessed.append((0, [list(range(host_left - 1)), [7]]))
    else:
        generating = [0]
    scheduler.batch_generator = SimpleNamespace(
        prefill_step_size=2048,
        completion_batch_size=32,
        _prompt_batch=prompt,
        _currently_processing=processing,
        _unprocessed_sequences=unprocessed,
        _generation_batch=SimpleNamespace(uids=generating, prompt_cache=[]),
        _mtp_vendored_admission_owner=owner,
    )
    scheduler.waiting = deque()
    return scheduler


def test_a_short_request_is_let_in_beside_a_prefilling_long_prompt():
    scheduler = _scheduler()
    scheduler.waiting.extend([_req("long-b", 17_000), _req("canary", 28)])
    assert scheduler._prefill_guest_index() == 1


def test_a_long_request_is_never_a_guest():
    scheduler = _scheduler()
    scheduler.waiting.extend([_req("long-b", 17_000), _req("helper", 2049)])
    assert scheduler._prefill_guest_index() is None


@pytest.mark.parametrize(
    "setup",
    [
        dict(host_where="decode"),  # the host decodes: no handoff, it finishes first
        dict(owner=0),  # the host holds the MTP verifier
        dict(host_left=30),  # the host could finish its prefill beside the guest
        dict(mtp_prefill_guests=False),
    ],
)
def test_no_guest_when_it_could_batch_decode_or_is_off(setup):
    scheduler = _scheduler(**setup)
    scheduler.waiting.append(_req("canary", 28))
    assert scheduler._prefill_guest_index() is None


def test_one_guest_at_a_time():
    scheduler = _scheduler()
    scheduler._prefill_guest_rid = "canary-1"
    scheduler.waiting.append(_req("canary-2", 28))
    assert scheduler._prefill_guest_index() is None


def test_a_host_not_yet_started_can_take_a_guest():
    scheduler = _scheduler(host_where="queued")
    scheduler.waiting.append(_req("canary", 28))
    assert scheduler._prefill_guest_index() == 0


def test_a_cache_hit_counts_only_the_tail_left_to_prefill():
    scheduler = _scheduler()
    warm = _req("warm", 30)
    warm.prompt_token_ids = list(range(40_000))
    scheduler.waiting.append(warm)
    assert scheduler._prefill_guest_index() == 0


def _with_guest(scheduler, where, first_segment=27):
    guest = _req("canary", 28, uid=1)
    scheduler.running["canary"] = guest
    scheduler._prefill_guest_rid = "canary"
    bg = scheduler.batch_generator
    if where == "prompt":
        bg._prompt_batch.uids.append(1)
        bg._currently_processing.append([[list(range(first_segment)), [9]], 0, 28])
    elif where == "decode":
        bg._generation_batch.uids.append(1)
    return bg


def test_a_guest_in_prefill_sizes_the_shared_step_to_its_prompt():
    scheduler = _scheduler()
    bg = _with_guest(scheduler, "prompt")
    scheduler._apply_prefill_guest()
    assert bg.prefill_step_size == 27
    assert bg._prompt_batch.prefill_step_size == 27


def test_a_guest_on_plain_decode_holds_the_host_at_its_chunk_boundary():
    scheduler = _scheduler()
    bg = _with_guest(scheduler, "decode")
    scheduler._apply_prefill_guest()
    assert bg.completion_batch_size == 1
    # The step that ends the guest re-opens mlx-lm's prompt call inside
    # next(): the host gets one token, not a full chunk.
    assert bg.prefill_step_size == 1

    del scheduler.running["canary"]
    scheduler._apply_prefill_guest()
    assert bg.completion_batch_size == 32
    assert scheduler._prefill_guest_rid is None


def test_a_guest_under_mtp_leaves_the_verifier_lock_alone():
    scheduler = _scheduler()
    bg = _with_guest(scheduler, "decode")
    bg._mtp_vendored_admission_owner = 1
    bg.completion_batch_size = 1  # the verifier's own lock
    scheduler._apply_prefill_guest()
    assert scheduler._prefill_guest_paused_host is False
    del scheduler.running["canary"]
    scheduler._apply_prefill_guest()
    assert bg.completion_batch_size == 1  # released by the verifier, not here


def test_a_one_row_batched_cache_goes_back_to_its_singleton_form():
    import mlx.core as mx
    from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache

    from rapid_mlx.singleton_cache_fastpath import demote_single_row

    kv = BatchKVCache([0, 5])
    keys = mx.random.normal((2, 4, 12, 8))
    kv.update_and_fetch(keys, keys)
    kv.filter([0])
    rec = ArraysCache(2)
    rec.cache = [mx.zeros((1, 3, 8)), mx.zeros((1, 2, 4, 4))]
    rec.left_padding = mx.array([0])

    single = demote_single_row([kv, rec])

    assert type(single[0]) is KVCache
    assert single[0].offset == 12
    assert mx.array_equal(single[0].keys, keys[0:1]).item()
    assert hasattr(single[0], "filter") and hasattr(single[0], "extract")
    assert single[1] is rec and rec.left_padding is None


def test_a_two_row_or_unknown_batched_cache_stays_batched():
    import mlx.core as mx
    from mlx_lm.models.cache import BatchKVCache

    from rapid_mlx.singleton_cache_fastpath import demote_single_row

    kv = BatchKVCache([0, 0])
    keys = mx.random.normal((2, 4, 3, 8))
    kv.update_and_fetch(keys, keys)
    assert demote_single_row([kv]) is None
    assert demote_single_row([object()]) is None
