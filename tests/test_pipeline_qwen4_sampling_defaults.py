# SPDX-License-Identifier: Apache-2.0
"""A request that names no sampling field samples the checkpoint's own defaults
on the pipeline split, as on the single engine (goose Q-159).

Measured 2026-09-26 on goose's split: goose sends no temperature, and the
pipeline server read ``body.get("temperature") or 0.0`` — greedy — where the
single engine applies the checkpoint's ``generation_config.json`` (Qwen3.8:
temperature 1.0, top_k 20, top_p 0.95).  On the tensor split the same greedy
default wrote one answer of 54 identical tool calls over 40 minutes.

In process, CPU, no ranks: the server's own HTTP app over a word-level
tokenizer, a stand-in batch loop that keeps each admitted row, the plan's
collectives replayed into a worker, and ``_sample`` over a recording sampler.
"""

import argparse
import json
import threading
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
pytest.importorskip("fastapi")
pytestmark = pytest.mark.requires_mlx

from fastapi.testclient import TestClient  # noqa: E402

from rapid_mlx.distributed import pipeline_qwen4_serve as serve  # noqa: E402
from rapid_mlx.service.helpers import (  # noqa: E402
    _FALLBACK_TEMPERATURE,
    _FALLBACK_TOP_P,
)

SERVED = "tiny-pipeline"
# ~/.goose/models/Mihai-LeanZero/Qwen3.8-27B-Atlassian-Q8-mlx/generation_config.json
QWEN38_GENERATION_CONFIG = {
    "bos_token_id": 248044,
    "do_sample": True,
    "eos_token_id": [248046, 248044],
    "pad_token_id": 248044,
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95,
}
TEMPLATE = "{% for m in messages %}{{ m['content'] }} {% endfor %}"


def _checkpoint(path: Path, generation_config=None) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {f"t{chr(97 + i // 26)}{chr(97 + i % 26)}": i for i in range(64)}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="taa"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "eos_token": "tab",
                "unk_token": "taa",
                "chat_template": TEMPLATE,
            }
        )
    )
    if generation_config is not None:
        (path / "generation_config.json").write_text(
            generation_config
            if isinstance(generation_config, str)
            else json.dumps(generation_config)
        )
    return path


def _options(*flags):
    parser = argparse.ArgumentParser()
    serve.add_arguments(parser)
    return parser.parse_args(
        [
            "--model",
            "/m",
            "--served-model-name",
            SERVED,
            "--port",
            "1",
            "--context",
            "64",
        ]
        + list(flags)
    )


class _Server:
    """Rank 0's app with a stand-in batch loop that keeps every admitted row."""

    def __init__(self, checkpoint: Path, *flags):
        from mlx_lm.utils import load_tokenizer

        tokenizer = load_tokenizer(checkpoint)
        self.sampling = serve._SamplingDefaults.load(
            checkpoint, serve._SamplingDefaults.profile_of(_options(*flags))
        )
        self.state = serve._State(
            served=SERVED, context=4096, max_batch=1, sampling=self.sampling
        )
        self.rows = []
        self.client = TestClient(
            serve._build_app(self.state, tokenizer, set(tokenizer.eos_token_ids))
        )
        self.loop = threading.Thread(target=self._batch_loop, daemon=True)
        self.loop.start()

    def _batch_loop(self):
        while (job := self.state.jobs.get()) is not None:
            self.rows.append(job.row)
            job.push(("done", "length"))

    def row(self, **fields):
        body = {
            "model": SERVED,
            "messages": [{"role": "user", "content": "tac tad"}],
            "max_tokens": 1,
            **fields,
        }
        answer = self.client.post("/v1/chat/completions", json=body)
        assert answer.status_code == 200, answer.text
        return self.rows[-1]

    def close(self):
        self.state.jobs.put(None)
        self.loop.join(timeout=10)


@pytest.fixture
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture
def server(cpu, tmp_path):
    running = _Server(_checkpoint(tmp_path, QWEN38_GENERATION_CONFIG))
    yield running
    running.close()


def _sampled(row):
    return (row.temperature, row.top_p, row.top_k, row.min_p)


def test_a_request_with_no_sampling_field_samples_the_checkpoints_defaults(server):
    row = server.row()
    assert _sampled(row) == (1.0, 0.95, 20, 0.0), "never greedy (was 0.0, 1.0)"
    assert {key: entry["from"] for key, entry in row.sampling.items()} == {
        "temperature": "generation_config",
        "top_p": "generation_config",
        "top_k": "generation_config",
        "min_p": "unset",
        "repetition_penalty": "unset",
        "presence_penalty": "unset",
        "frequency_penalty": "unset",
    }


def test_the_requests_own_value_wins_and_a_null_is_no_value(server):
    row = server.row(temperature=0.2, top_k=5)
    assert _sampled(row) == (0.2, 0.95, 5, 0.0)
    assert row.sampling["temperature"] == {"value": 0.2, "from": "request"}
    assert _sampled(server.row(temperature=0)) == (0.0, 0.95, 20, 0.0), (
        "an explicit 0 is the client's greedy, not an absence"
    )
    assert _sampled(server.row(temperature=None, top_p=None)) == (1.0, 0.95, 20, 0.0)


def test_the_operators_profile_sits_between_the_request_and_the_checkpoint(
    cpu, tmp_path
):
    running = _Server(
        _checkpoint(tmp_path, QWEN38_GENERATION_CONFIG),
        "--default-top-p",
        "0.8",
        "--default-min-p",
        "0.05",
    )
    try:
        row = running.row()
        assert _sampled(row) == (1.0, 0.8, 20, 0.05)
        assert row.sampling["top_p"]["from"] == "profile"
        assert _sampled(running.row(top_p=0.5)) == (1.0, 0.5, 20, 0.05)
    finally:
        running.close()


@pytest.mark.parametrize(
    ("content", "said"),
    [(None, "is absent"), ("{not json", "is unreadable"), ("[1, 2]", "no JSON object")],
)
def test_a_missing_or_unreadable_config_is_named_and_never_greedy(
    cpu, tmp_path, content, said
):
    running = _Server(_checkpoint(tmp_path, content))
    try:
        status = running.client.get("/v1/status").json()["sampling_defaults"]
        assert said in status["generation_config_error"], status
        row = running.row()
        assert _sampled(row) == (_FALLBACK_TEMPERATURE, _FALLBACK_TOP_P, 0, 0.0)
        assert row.sampling["temperature"]["from"] == "engine_fallback"
    finally:
        running.close()


def test_the_status_names_the_layers_and_a_penalty_the_sampler_cannot_apply(
    cpu, tmp_path
):
    config = {**QWEN38_GENERATION_CONFIG, "repetition_penalty": 1.05, "top_k": 20.5}
    running = _Server(
        _checkpoint(tmp_path, config), "--default-presence-penalty", "1.5"
    )
    try:
        status = running.client.get("/v1/status").json()["sampling_defaults"]
        assert status["generation_config"] == {
            "temperature": 1.0,
            "top_p": 0.95,
            "repetition_penalty": 1.05,
        }
        assert status["generation_config_error"] is None
        assert status["generation_config_ignored"] == ["top_k=20.5"], (
            "the single engine's own filter drops a fractional top_k"
        )
        assert status["profile"] == {"presence_penalty": 1.5}
        assert status["unapplied"] == ["presence_penalty", "repetition_penalty"]
        row = running.row()
        assert row.sampling["presence_penalty"] == {
            "value": 1.5,
            "from": "profile",
            "applied": False,
        }
    finally:
        running.close()


def test_a_sampling_field_of_the_wrong_type_is_a_400_naming_it(server):
    answer = server.client.post(
        "/v1/chat/completions",
        json={
            "model": SERVED,
            "messages": [{"role": "user", "content": "tac"}],
            "temperature": "hot",
        },
    )
    assert answer.status_code == 400
    assert "temperature must be a finite number" in answer.json()["error"]["message"]


def test_the_plan_carries_top_k_and_min_p_to_the_sampling_rank(cpu, monkeypatch):
    recorded = []

    def record(group_, value):
        mx.eval(value)
        recorded.append(value)
        return value

    monkeypatch.setattr(serve, "_all_sum", record)
    joiner = serve._Row([5, 6, 7], 9, 1.0, 0.95, top_k=20, min_p=0.05)
    assert serve._broadcast_plan(None, serve._Plan(joiner=joiner), 2, deciding=True)
    replay = iter(recorded)
    monkeypatch.setattr(serve, "_all_sum", lambda group_, value: next(replay))
    got = serve._broadcast_plan(None, None, 2, deciding=False).joiner
    assert next(replay, None) is None, "a collective would not pair"
    assert got.temperature == 1.0 and got.top_k == 20
    assert got.top_p == pytest.approx(0.95) and got.min_p == pytest.approx(0.05)


def test_the_sampling_rank_hands_every_field_to_mlx_lms_sampler(cpu, monkeypatch):
    import mlx_lm.sample_utils

    made = []
    real = mlx_lm.sample_utils.make_sampler

    def recording(**kwargs):
        made.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(mlx_lm.sample_utils, "make_sampler", recording)
    rows = [serve._Row([1], 4, 1.0, 0.95, top_k=20, min_p=0.05)]
    serve._sample(mx.zeros((1, 32)), rows)
    assert made == [{"temp": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.05}]
