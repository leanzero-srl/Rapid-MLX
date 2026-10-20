# SPDX-License-Identifier: Apache-2.0
"""The long-period repetition stop is for agent (tool-bearing) requests on the
multimodal lane too (RUN-ISSUES I-24 review): the text lane gates its whole stop
on ``has_tools``; the MLLM lane keeps its short rule on every request (it guards
the Metal-handle exhaustion of a sustained short loop) and adds the long rule
only when tools were sent."""

from __future__ import annotations

import pytest

pytest.importorskip("mlx")

import random
from types import SimpleNamespace

from rapid_mlx.mllm_scheduler import MLLMRequest, MLLMScheduler, MLLMSchedulerConfig
from rapid_mlx.runtime.model_performance import ModelPerformanceLedger


class _Detokenizer:
    last_segment = "x"
    text = "x"

    def reset(self) -> None:
        pass

    def add_token(self, _token: int) -> None:
        pass

    def finalize(self) -> None:
        pass


class _Tokenizer:
    def __init__(self) -> None:
        self.detokenizer = _Detokenizer()


class _BatchGenerator:
    def __init__(self) -> None:
        self.removed: list[int] = []

    def remove(self, uids) -> None:
        self.removed.extend(uids)


def _scheduler() -> MLLMScheduler:
    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    scheduler.config = MLLMSchedulerConfig()
    scheduler.processor = SimpleNamespace(tokenizer=_Tokenizer())
    scheduler.uid_to_request_id = {}
    scheduler.running = {}
    scheduler._detokenizer_pool = {}
    scheduler.total_prompt_tokens = 0
    scheduler.total_completion_tokens = 0
    scheduler.num_requests_processed = 0
    scheduler.num_repetition_loop_stops = 0
    scheduler.batch_generator = _BatchGenerator()
    scheduler.performance = ModelPerformanceLedger("mllm-test")
    return scheduler


def _paragraph_loop() -> list[int]:
    """A 240-token paragraph repeated four times (the I-24 Gemma shape), ending one
    token short so the last response completes the fourth copy on an 8-token boundary."""
    rng = random.Random(24)
    block = [rng.randrange(1000, 9000) for _ in range(240)]
    return (block * 4)[:-1]


def _finish_one_more_token(has_tools: bool):
    scheduler = _scheduler()
    tokens = _paragraph_loop()
    request = MLLMRequest(request_id="req-loop", prompt="hello")
    request.has_tools = has_tools
    request.output_tokens = list(tokens)
    request.num_output_tokens = len(tokens)
    scheduler.uid_to_request_id[3] = request.request_id
    scheduler.running[request.request_id] = request
    # The 960th token (the block's last) closes the fourth adjacent copy.
    response = SimpleNamespace(
        uid=3,
        token=tokens[239],
        prompt_tokens=10,
        finish_reason=None,
        token_is_stop_token=False,
        logprobs=None,
    )
    scheduler._process_batch_responses([response])
    return scheduler, request


def test_a_long_loop_without_tools_is_not_stopped_on_the_multimodal_lane():
    scheduler, request = _finish_one_more_token(has_tools=False)
    assert request.num_output_tokens == 960
    assert scheduler.num_repetition_loop_stops == 0
    assert scheduler.batch_generator.removed == []


def test_a_long_loop_with_tools_is_stopped_on_the_multimodal_lane():
    scheduler, request = _finish_one_more_token(has_tools=True)
    assert request.num_output_tokens == 960
    assert scheduler.num_repetition_loop_stops == 1
    assert scheduler.batch_generator.removed == [3]


def test_the_request_carries_has_tools_from_add_request():
    request = MLLMRequest(request_id="r", prompt="p", has_tools=True)
    assert request.has_tools is True
    assert MLLMRequest(request_id="r2", prompt="p").has_tools is False
