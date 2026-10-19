# SPDX-License-Identifier: Apache-2.0
"""The repetition guard's hard stop reaches LONG exact loops (RUN-ISSUES I-24).

Live defect, 2026-10-19 03:4x (Gabee, Forge 2.0 through the goose app,
Gemma 4 12B): one reply looped a ~240-token paragraph 38 times (9,250 tokens,
max_tokens 219,782 = the window minus the prompt) at 15 tok/s, up to ~4 h on a
single reply. The guard only checked periods <= 128 inside the last 768
tokens, so it never saw a paragraph-sized loop.

Periods 129..1024 are now checked too, needing FOUR exact adjacent copies
(the short rule needs three). The short rule's results are unchanged (parity
against the shipped cd53a8247 function, vendored below), the long rule judges
a loop only at its primitive period, and it reads a bounded window.
"""

import random
from collections.abc import Sequence
from math import ceil
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

from rapid_mlx.repetition_guard import (  # noqa: E402
    DEFAULT_MAX_LONG_PERIOD_TOKENS,
    RepetitionMatch,
    detect_long_repeated_token_suffix,
    detect_repeated_token_suffix,
)
from rapid_mlx.request import Request, RequestStatus, SamplingParams  # noqa: E402
from rapid_mlx.scheduler import Scheduler, SchedulerConfig  # noqa: E402

# The I-24 shape: a paragraph of 240 distinct tokens.
PARAGRAPH = [10_000 + i for i in range(240)]


# ---- the shipped short rule, verbatim from v0.14.3-lz.12 (cd53a8247) --------


def _old_detect(
    token_ids: Sequence[int],
    *,
    min_period_tokens: int = 1,
    max_period_tokens: int = 128,
    required_repeats: int = 3,
    min_repeated_tokens: int = 72,
    min_short_period_tokens: int = 256,
    max_window_tokens: int = 768,
) -> RepetitionMatch | None:
    """Return an exact periodic-suffix match, or ``None``.

    The defaults require at least three adjacent copies and at least 72 repeated
    tokens.  The effective repeat count is adaptive: a 12-token sentence needs
    six copies, while a 61-token paragraph needs only three.  Periods below six
    tokens use the higher ``min_short_period_tokens`` threshold, tolerating
    intentional stutters while still catching the sustained one-token CJK loop
    that previously ran for 10k+ tokens and exhausted Metal resource handles.
    Work is bounded to the most recent 768 tokens and callers are expected to
    invoke it only every few decode steps.
    """

    if required_repeats < 2 or min_period_tokens < 1:
        raise ValueError("invalid repetition guard thresholds")

    tokens = token_ids[-max_window_tokens:]
    max_period = min(max_period_tokens, len(tokens) // required_repeats)
    for period in range(min_period_tokens, max_period + 1):
        repeated_floor = min_short_period_tokens if period < 6 else min_repeated_tokens
        repeats = max(required_repeats, ceil(repeated_floor / period))
        repeated_tokens = period * repeats
        if repeated_tokens > len(tokens):
            continue
        suffix = tokens[-repeated_tokens:]
        pattern = suffix[:period]
        # Evaluate a loop at its primitive period.  Otherwise 200 copies of a
        # single token would masquerade as a 6-token pattern and inherit the
        # lower long-period threshold.
        if any(
            period % smaller == 0 and pattern == pattern[:smaller] * (period // smaller)
            for smaller in range(1, period)
        ):
            continue
        if all(
            suffix[offset : offset + period] == pattern
            for offset in range(period, repeated_tokens, period)
        ):
            return RepetitionMatch(period_tokens=period, repeats=repeats)
    return None


def test_the_live_paragraph_loop_is_stopped_after_four_copies():
    assert _old_detect(PARAGRAPH * 38) is None, "the shipped guard never saw it"
    assert detect_repeated_token_suffix([1, 2, 3] + PARAGRAPH * 3) is None
    match = detect_repeated_token_suffix([1, 2, 3] + PARAGRAPH * 4)
    assert match == RepetitionMatch(period_tokens=240, repeats=4)


def test_long_periods_up_to_the_configured_maximum():
    for period in (129, 512, 1024):
        pattern = [50_000 + i for i in range(period)]
        match = detect_repeated_token_suffix([7] + pattern * 4)
        assert match is not None and match.period_tokens == period, period
    too_long = [60_000 + i for i in range(1025)]
    assert detect_repeated_token_suffix(too_long * 4) is None


def test_the_short_rule_alone_when_the_maximum_is_128_or_0():
    for limit in (0, 128):
        assert detect_repeated_token_suffix(PARAGRAPH * 8, max_long_period_tokens=limit) is None


def test_a_near_repeat_is_not_a_loop():
    copy = list(PARAGRAPH)
    copy[117] += 1
    assert detect_repeated_token_suffix(PARAGRAPH * 2 + copy + PARAGRAPH) is None


def test_a_long_period_is_judged_at_its_primitive_period():
    unit = [70_000 + i for i in range(64)]
    # 64 x 16: every multiple of 64 in 129..1024 repeats too; none is primitive.
    assert detect_long_repeated_token_suffix(unit * 16) is None
    # Two different halves make 128+72 a primitive 200-token period.
    halves = [80_000 + i for i in range(128)] + [90_000 + i for i in range(72)]
    match = detect_long_repeated_token_suffix(halves * 4)
    assert match == RepetitionMatch(period_tokens=200, repeats=4)


def _sequences():
    rng = random.Random(1019)
    out: list[list[int]] = []
    for _ in range(400):
        alphabet = rng.choice([2, 3, 5, 50, 30_000])
        out.append([rng.randrange(alphabet) for _ in range(rng.randrange(0, 2_000))])
    for period in (1, 2, 5, 6, 11, 12, 24, 61, 100, 128, 129, 240, 400):
        pattern = [rng.randrange(30_000) for _ in range(period)]
        for copies in (1, 2, 3, 4, 5, 6, 12):
            for prefix in ([], [rng.randrange(30_000) for _ in range(37)]):
                seq = prefix + pattern * copies
                out.append(seq)
                if seq:
                    tweaked = list(seq)
                    tweaked[rng.randrange(len(tweaked))] += 1
                    out.append(tweaked)
                    out.append(seq + pattern[: rng.randrange(1, period + 1)])
    return out


def test_the_short_rule_is_byte_identical_to_the_shipped_one():
    for seq in _sequences():
        old = _old_detect(seq)
        assert detect_repeated_token_suffix(seq, max_long_period_tokens=0) == old
        new = detect_repeated_token_suffix(seq)
        if old is not None:
            assert new == old, seq[-20:]
        elif new is not None:
            assert new.period_tokens > 128, new


def test_the_long_rule_reads_a_bounded_window():
    class _Bounded(Sequence):
        def __init__(self, size):
            self.size = size
            self.slices = []

        def __len__(self):
            return self.size

        def __getitem__(self, key):
            assert isinstance(key, slice) and key.stop is None, key
            self.slices.append(key.start)
            return list(range(-key.start))

    history = _Bounded(1_000_000)
    assert detect_repeated_token_suffix(history) is None
    assert history.slices == [-768, -DEFAULT_MAX_LONG_PERIOD_TOKENS * 4]


def test_a_long_diverse_stream_has_no_false_positive():
    tokens: list[int] = []
    for i in range(20_000):
        tokens.append((i * i + 31 * i + (i % 97) * 13) % 32_749)
        if len(tokens) % 8 == 0:
            assert detect_repeated_token_suffix(tokens) is None


def _scheduler(**config) -> Scheduler:
    tokenizer = MagicMock()
    tokenizer.encode = lambda text: list(range(len(text.split())))
    tokenizer.decode = lambda tokens, **_kwargs: " ".join(map(str, tokens))
    scheduler = Scheduler(
        MagicMock(), tokenizer, SchedulerConfig(max_num_seqs=2, **config)
    )
    scheduler.batch_generator = MagicMock()
    scheduler.batch_generator.remove.return_value = {}
    return scheduler


def _drive_paragraph_loop(**config):
    scheduler = _scheduler(**config)
    req = Request("paragraph", "prompt", SamplingParams(max_tokens=219_782))
    req.status = RequestStatus.RUNNING
    req.has_tools = True
    # The response lands on the fourth copy (960 tokens, a multiple of 8).
    req.output_token_ids = PARAGRAPH * 3 + PARAGRAPH[:-1]
    scheduler.running[req.request_id] = req
    scheduler.uid_to_request_id[1] = req.request_id
    response = MagicMock(uid=1, token=PARAGRAPH[-1], finish_reason=None, logprobs=None)
    del response.prompt_cache
    outputs, finished = scheduler._process_batch_responses([response])
    return scheduler, req, outputs[0], finished


def test_the_scheduler_stops_a_paragraph_loop_on_a_tool_request():
    scheduler, req, output, finished = _drive_paragraph_loop()
    assert finished == {req.request_id}
    assert output.finish_reason == "abort"
    assert output.error_kind == "repetition"
    assert "period_tokens=240" in output.error
    assert scheduler.num_repetition_loop_stops == 1


def test_the_serve_setting_turns_the_long_rule_off():
    scheduler, req, output, finished = _drive_paragraph_loop(
        repetition_guard_max_period=128
    )
    assert finished == set()
    assert output.finished is False
    with pytest.raises(ValueError):
        SchedulerConfig(repetition_guard_max_period=-1)


def test_the_cli_default_is_the_guards_default():
    from rapid_mlx.cli import REPETITION_GUARD_MAX_PERIOD_DEFAULT
    from rapid_mlx.mllm_scheduler import MLLMSchedulerConfig

    assert REPETITION_GUARD_MAX_PERIOD_DEFAULT == DEFAULT_MAX_LONG_PERIOD_TOKENS
    assert SchedulerConfig().repetition_guard_max_period == DEFAULT_MAX_LONG_PERIOD_TOKENS
    assert MLLMSchedulerConfig().repetition_guard_max_period == DEFAULT_MAX_LONG_PERIOD_TOKENS
