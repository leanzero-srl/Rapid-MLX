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
    prompt = SimpleNamespace(uids=[], prefill_step_size=2048)
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
        _generation_batch=SimpleNamespace(uids=generating),
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
