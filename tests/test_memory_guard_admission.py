# SPDX-License-Identifier: Apache-2.0
"""Memory guard: admission refusal shape, the honest window, the SSE error.

goose maps an HTTP 400 whose body is ``{"error": {"code":
"context_length_exceeded", ...}}`` (or one carrying ``n_prompt_tokens >
n_ctx``) to ContextLengthExceeded, then COMPACTS the conversation and resends
(goose-providers/src/http_status.rs). So a prompt this Mac cannot prefill is
refused at admission -- before the SSE 200 is committed -- with exactly that
shape plus ``lz_reason``, and ``/v1/models``' ``max_model_len`` is the honest
window goose compacts against.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import mlx.core as mx  # noqa: E402
from fastapi import HTTPException  # noqa: E402

from rapid_mlx.memory_guard import StepShape  # noqa: E402
from rapid_mlx.request import InferenceAbortedError  # noqa: E402
from rapid_mlx.scheduler import Scheduler, SchedulerConfig  # noqa: E402
from rapid_mlx.service.helpers import (  # noqa: E402
    _disconnect_guard,
    enforce_context_length,
)

BASE = 1_000_000_000  # simulated weights
CEILING = 1_060_000_000  # the engine's Metal cap
WINDOW = 4096


@pytest.fixture(scope="module")
def llama():
    from mlx_lm.models.llama import Model, ModelArgs

    mx.random.seed(0)
    model = Model(
        ModelArgs(
            model_type="llama",
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=128,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=128,
            num_key_value_heads=2,
            max_position_embeddings=WINDOW,
        )
    )
    mx.eval(model.parameters())
    return model


def _engine(llama, *, base=BASE, ceiling=CEILING):
    scheduler = Scheduler(
        model=llama,
        tokenizer=SimpleNamespace(encode=list, decode=lambda ids, **_: ""),
        config=SchedulerConfig(enable_prefix_cache=False, spec_decode="none"),
    )
    scheduler._current_metal_active_bytes = lambda: base
    scheduler._current_metal_cache_bytes = lambda: 0
    scheduler._resolve_metal_cap_bytes = lambda: ceiling
    engine = SimpleNamespace(_model=llama, scheduler=scheduler, is_mllm=False)
    return engine, scheduler


def test_a_prompt_this_mac_cannot_prefill_gets_the_exact_400_body(llama):
    from rapid_mlx.middleware.exception_handlers import _http_error_response

    engine, scheduler = _engine(llama)
    guard = scheduler._memguard()
    max_ctx = scheduler.memory_guard_max_context()
    assert 0 < max_ctx < 3000 < WINDOW
    prompt = 3000
    needed = guard.single_sequence_need(
        prompt,
        base=BASE,
        kv_footprint=scheduler._kv_footprint_fn(),
        ceiling=CEILING,
    )

    with pytest.raises(HTTPException) as info:
        enforce_context_length(engine, prompt, max_tokens=None)
    response = _http_error_response(info.value)

    assert response.status_code == 400
    assert json.loads(response.body) == {
        "error": {
            "message": (
                f"This prompt ({prompt} tokens) needs {needed / 1e9:.1f} GB but "
                f"this Mac can hold {CEILING / 1e9:.1f} GB; the maximum context "
                f"here is {max_ctx} tokens."
            ),
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "n_prompt_tokens": prompt,
            "n_ctx": max_ctx,
            "lz_reason": "context_exceeds_device_memory",
        }
    }
    assert guard.counts["admission_refusals"] == 1


def test_a_feasible_prompt_passes(llama):
    engine, scheduler = _engine(llama)
    max_ctx = scheduler.memory_guard_max_context()
    enforce_context_length(engine, max_ctx, max_tokens=None)
    enforce_context_length(engine, 100, max_tokens=50)
    assert scheduler._memguard().counts["admission_refusals"] == 0


def test_the_prefix_cache_is_not_charged_at_admission(llama):
    """Prefix entries are evictable: a full prefix cache must not turn a
    prompt that fits beside the weights into a refusal."""
    engine, scheduler = _engine(llama, base=BASE + 50_000_000)
    scheduler._cache_self_pressure_current_bytes = lambda: 50_000_000
    enforce_context_length(engine, 1000, max_tokens=None)


def test_the_window_over_the_native_limit_keeps_the_native_error(llama):
    # Negative control: past the model's own window the existing
    # context_length_exceeded (without lz_reason) still fires first.
    engine, _ = _engine(llama, ceiling=10**12)
    with pytest.raises(HTTPException) as info:
        enforce_context_length(engine, WINDOW + 1, max_tokens=None)
    assert "lz_reason" not in info.value.detail["error"]


def test_advertised_window_is_the_honest_one_and_tracks_the_peak_model(llama):
    """``max_model_len`` (/v1/models) = min(KV-only, guard); it shrinks when
    the peak model learns a larger step peak."""
    _, scheduler = _engine(llama)
    honest = scheduler.memory_guard_max_context(WINDOW)
    advertised = scheduler.projected_memory_max_context(WINDOW)
    if mx.metal.is_available():
        # The KV-only figure on this Mac is the whole native window; the
        # advertised one is the guard's.
        assert advertised == honest < WINDOW
    guard = scheduler._memguard()
    guard.observe(StepShape(1, guard.floor_tokens, 1000), 30_000_000)
    # The exact figure follows the model at once ...
    assert scheduler.memory_guard_max_context(WINDOW, stable=False) < honest
    # ... the advertised one only once a second model state confirms it.
    assert scheduler.memory_guard_max_context(WINDOW) == honest
    guard.observe(StepShape(1, guard.floor_tokens, 1100), 31_000_000)
    assert scheduler.memory_guard_max_context(WINDOW) < honest


def test_mid_stream_refusal_becomes_an_sse_error_chunk_with_the_same_object():
    detail = {
        "message": "This prompt (90000 tokens) needs 61.2 GB but this Mac can "
        "hold 50.1 GB; the maximum context here is 61000 tokens.",
        "type": "invalid_request_error",
        "code": "context_length_exceeded",
        "n_prompt_tokens": 90000,
        "n_ctx": 61000,
        "lz_reason": "context_exceeds_device_memory",
    }

    class Connected:
        async def is_disconnected(self):
            return False

    async def stream():
        yield 'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        raise InferenceAbortedError(
            detail["message"],
            error_kind="context_exceeds_device_memory",
            detail=detail,
        )

    async def collect():
        return [
            chunk
            async for chunk in _disconnect_guard(
                stream(), Connected(), keepalive_seconds=0
            )
        ]

    chunks = asyncio.run(collect())
    errors = [
        json.loads(line[6:])
        for chunk in chunks
        for line in chunk.splitlines()
        if line.startswith("data: {") and '"error"' in line
    ]
    assert errors == [{"error": detail}]
    assert chunks[-1] == "data: [DONE]\n\n"


# The text config of Qwen3.5/3.6-27B (the goose text lane's model family), as
# published (HF config.json ``text_config``); weights are never loaded.
QWEN_27B_TEXT = {
    "dtype": "bfloat16",
    "full_attention_interval": 4,
    "head_dim": 256,
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "layer_types": [
        "full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(64)
    ],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_value_head_dim": 128,
    "max_position_embeddings": 262144,
    "model_type": "qwen3_5_text",
    "num_attention_heads": 24,
    "num_hidden_layers": 64,
    "num_key_value_heads": 4,
}


def test_prior_and_kv_terms_for_the_goose_27b():
    """64 KiB/token of KV (16 full-attention layers x 2 x 4 heads x 256 x
    bf16), the GatedDeltaNet state as a fixed baseline, and a prior of four
    bf16 score copies per query head -- read from the loaded parameters'
    dtype, not the KV probe's fp32 fallback."""
    model = SimpleNamespace(
        args=SimpleNamespace(text_config=QWEN_27B_TEXT),
        parameters=lambda: {
            "norm": {"weight": mx.zeros((1,), dtype=mx.bfloat16)},
            "q": {"weight": mx.zeros((1,), dtype=mx.uint32)},
        },
    )
    scheduler = Scheduler(
        model=model,
        tokenizer=SimpleNamespace(encode=list, decode=lambda ids, **_: ""),
        config=SchedulerConfig(enable_prefix_cache=False, spec_decode="none"),
    )
    per_tok, fixed, sliding, _window = scheduler._kv_footprint_terms()
    assert per_tok == 16 * 2 * 4 * 256 * 2 == 65_536
    assert fixed > 0 and sliding == 0
    prior = scheduler._memguard_prior()
    assert prior.attention_bytes_per_unit == 24 * 2 * 4
    assert prior.kv_copy_bytes_per_context == 2 * 4 * 256 * 2
