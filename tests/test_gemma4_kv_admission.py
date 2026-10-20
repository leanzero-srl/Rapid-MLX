# SPDX-License-Identifier: Apache-2.0
"""RUN-ISSUES I-17: D-METAL-CAP refused a Gemma 4 12B agent turn mid-run.

"This request needs approximately 42.6 GB of Metal memory on top of 38.0 GB
already in use (reserved KV 0.0 GB + projected KV 42.6 GB), but the current
limit is 75.1 GB" -- for a ~60k-token conversation whose real KV is ~1.9 GB.
Three faults multiplied:

* Gemma 4's full-attention layers cache 1 global KV head x 512, not the
  config's 8 x 256 sliding shape (4x), and the inner ``TextConfig`` carries no
  ``dtype`` so the probe fell back to fp32 (2x): 131,072 B/token priced against
  16,384 measured on the loaded model, 655,360 vs 327,680 per sliding slot.
* The untokenized prompt string was counted as its UTF-8 bytes (~5x).
* The refusal reached goose as "Internal error during streaming", which it
  retried 3x into the same refusal and then refused the run.
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

from rapid_mlx.errors import BackpressureError  # noqa: E402
from rapid_mlx.kv_estimation import rotating_cache_slots  # noqa: E402
from rapid_mlx.scheduler import Scheduler, SchedulerConfig  # noqa: E402
from rapid_mlx.service.helpers import (  # noqa: E402
    _disconnect_guard,
    _raise_backpressure_503,
)

# Gemma 4 12B (lmstudio-community/gemma-4-12B-it-MLX-8bit): 48 layers, every
# 6th full attention; the TextConfig mlx-vlm builds has no dtype key.
LAYERS = 48
LAYER_TYPES = [
    "full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(LAYERS)
]
MEASURED_GROWTH = 8 * 2 * 1 * 512 * 2  # 16,384 B/token: cache nbytes on the real model
MEASURED_SLOT = 40 * 2 * 8 * 256 * 2  # 327,680 B per sliding slot


def _attn(sliding: bool, *, shared: bool = False):
    return SimpleNamespace(
        n_kv_heads=8 if sliding else 1,
        head_dim=256 if sliding else 512,
        is_sliding=sliding,
        is_kv_shared_layer=shared,
    )


class _GemmaLike:
    """The attention contract of mlx-vlm's Gemma 4 text stack, bf16 weights."""

    def __init__(self, *, shared_tail: int = 0, config_dtype: str | None = None):
        self.config = SimpleNamespace(
            model_type="gemma4_text",
            num_hidden_layers=LAYERS,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=256,
            hidden_size=3840,
            layer_types=list(LAYER_TYPES),
            sliding_window=1024,
            num_global_key_value_heads=1,
            global_head_dim=512,
        )
        if config_dtype is not None:
            self.config.dtype = config_dtype
        layers = []
        for i, kind in enumerate(LAYER_TYPES):
            shared = i >= LAYERS - shared_tail
            layers.append(SimpleNamespace(self_attn=_attn(kind != "full_attention", shared=shared)))
        self.model = SimpleNamespace(layers=layers)
        self._params = {"norm": mx.zeros((4,), dtype=mx.bfloat16)}

    def parameters(self):
        return self._params


class _DenseNoDtype:
    """A dense model whose mlx-lm ModelArgs dropped the HF ``dtype`` key."""

    def __init__(self):
        self.config = SimpleNamespace(
            model_type="llama",
            num_hidden_layers=32,
            num_key_value_heads=8,
            head_dim=128,
            num_attention_heads=32,
            hidden_size=4096,
        )
        self.model = SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace())] * 32)
        self._params = {"norm": mx.zeros((4,), dtype=mx.bfloat16)}

    def parameters(self):
        return self._params


def _sched(model, tokenizer=None):
    config = SchedulerConfig(
        max_num_seqs=8,
        max_concurrent_requests=64,
        enable_prefix_cache=False,
        use_memory_aware_cache=False,
        use_paged_cache=False,
        gpu_memory_utilization=0.78,
        metal_cap_kv_bytes_per_token=0,
    )
    sched = Scheduler(
        model=model,
        tokenizer=tokenizer or SimpleNamespace(encode=lambda s: None, decode=lambda ids, **_: ""),
        config=config,
    )
    sched._kv_bytes_per_token_resolved = False
    return sched


def _request(prompt="", *, tokens=0, max_tokens=32768):
    return SimpleNamespace(
        request_id="r-i17",
        num_prompt_tokens=tokens,
        prompt_token_ids=None,
        prompt=prompt,
        sampling_params=SimpleNamespace(max_tokens=max_tokens),
    )


def test_gemma4_full_layers_are_priced_at_their_own_kv_shape():
    sched = _sched(_GemmaLike())
    assert sched._resolve_kv_bytes_per_token() == MEASURED_GROWTH
    assert sched._kv_sliding_slot_bytes == MEASURED_SLOT
    assert sched._kv_sliding_window == 1024


def test_gemma4_refused_turn_projects_its_real_footprint():
    # The I-17 turn: ~60k prompt tokens + the default 32768 max_tokens.
    sched = _sched(_GemmaLike())
    projected = sched._estimate_request_kv_bytes(_request(tokens=60_000))
    slots = rotating_cache_slots(1024, 92_768)  # the window plus its rotation step
    assert projected == MEASURED_GROWTH * 92_768 + MEASURED_SLOT * slots
    assert projected < 2.0e9


def test_kv_shared_borrower_layers_allocate_nothing():
    # Gemma 3n / E-series: the tail layers reuse an earlier layer's KV.
    sched = _sched(_GemmaLike(shared_tail=12))
    assert sched._resolve_kv_bytes_per_token() == 6 * 2 * 1 * 512 * 2
    assert sched._kv_sliding_slot_bytes == 30 * 2 * 8 * 256 * 2


def test_element_size_comes_from_the_loaded_weights_when_the_config_has_none():
    sched = _sched(_DenseNoDtype())
    assert sched._resolve_kv_bytes_per_token() == 2 * 32 * 8 * 128 * 2


def test_untokenized_prompt_is_counted_by_the_engine_tokenizer():
    words = "tok " * 5000
    tokenizer = SimpleNamespace(encode=lambda s: s.split(), decode=lambda ids, **_: "")
    sched = _sched(_GemmaLike(), tokenizer)
    footprint = sched._kv_footprint_fn()
    assert sched._estimate_request_kv_bytes(_request(words, max_tokens=0)) == footprint(5000)


def test_byte_count_stays_the_fallback_without_a_usable_tokenizer():
    def boom(_s):
        raise RuntimeError("no tokenizer")

    sched = _sched(_GemmaLike(), SimpleNamespace(encode=boom, decode=lambda ids, **_: ""))
    footprint = sched._kv_footprint_fn()
    assert sched._estimate_request_kv_bytes(_request("é" * 100, max_tokens=0)) == footprint(200)


def _cap_sched(*, active, cap, reserved=0, idle=None):
    sched = _sched(_GemmaLike())
    sched._resolve_metal_cap_bytes = lambda: cap
    sched._metal_footprint_bytes = lambda reclaim_over=0: active
    sched._sum_in_flight_kv_bytes = lambda: reserved
    if idle is not None:
        sched._memguard = lambda: object()
        sched._memguard_idle_base = lambda **_: idle
    else:
        sched._memguard = lambda: None
    return sched


def test_a_request_that_never_fits_is_refused_as_context_length_exceeded():
    sched = _cap_sched(active=20_000_000_000, cap=22_000_000_000)
    with pytest.raises(BackpressureError) as raised:
        sched._enforce_metal_cap_at_admission(_request(tokens=200_000, max_tokens=4096))
    body = raised.value.error_object
    assert body["code"] == "context_length_exceeded"
    assert body["lz_reason"] == "metal_cap_exceeds_device"
    assert body["n_prompt_tokens"] == 200_000
    fn = sched._kv_footprint_fn()
    n_ctx = body["n_ctx"]
    assert 0 < n_ctx < 200_000
    assert 20_000_000_000 + fn(n_ctx + 4096) < 22_000_000_000
    assert 20_000_000_000 + fn(n_ctx + 1 + 4096) >= 22_000_000_000


def test_a_request_that_fits_once_others_drain_stays_retryable():
    sched = _cap_sched(active=20_000_000_000, cap=22_000_000_000, reserved=1_500_000_000)
    with pytest.raises(BackpressureError) as raised:
        sched._enforce_metal_cap_at_admission(_request(tokens=40_000, max_tokens=4096))
    assert raised.value.error_object is None


def test_a_request_that_fits_the_idle_engine_stays_retryable():
    # Prefix entries hold memory now, but the request fits weights + its own KV.
    sched = _cap_sched(active=21_500_000_000, cap=22_000_000_000, idle=14_000_000_000)
    with pytest.raises(BackpressureError) as raised:
        sched._enforce_metal_cap_at_admission(_request(tokens=40_000, max_tokens=4096))
    assert raised.value.error_object is None


def _sse_errors(exc):
    class Connected:
        async def is_disconnected(self):
            return False

    async def stream():
        yield 'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        raise exc

    async def collect():
        return [c async for c in _disconnect_guard(stream(), Connected(), keepalive_seconds=0)]

    chunks = asyncio.run(collect())
    assert chunks[-1] == "data: [DONE]\n\n"
    return [
        json.loads(line[6:])["error"]
        for chunk in chunks
        for line in chunk.splitlines()
        if line.startswith("data: {") and '"error"' in line
    ]


def test_mid_stream_backpressure_is_a_retryable_rate_limit_not_an_internal_error():
    message = "This request needs approximately 3.1 GB ... (D-METAL-CAP)."
    [error] = _sse_errors(BackpressureError(message))
    assert error["code"] == "rate_limit_exceeded"
    assert error["message"] == message
    assert error["lz_reason"] == "metal_cap_backpressure"
    assert "Internal error during streaming" not in json.dumps(error)


def test_mid_stream_backpressure_that_never_fits_carries_the_context_object():
    body = {
        "message": "This prompt (200000 tokens) needs 30.0 GB but this Mac can hold 22.0 GB",
        "type": "invalid_request_error",
        "code": "context_length_exceeded",
        "n_prompt_tokens": 200_000,
        "n_ctx": 80_000,
        "lz_reason": "metal_cap_exceeds_device",
    }
    assert _sse_errors(BackpressureError("cap", error_object=body)) == [body]


def test_pre_stream_backpressure_that_never_fits_is_http_400():
    body = {"message": "m", "type": "invalid_request_error", "code": "context_length_exceeded"}
    with pytest.raises(HTTPException) as raised:
        _raise_backpressure_503(BackpressureError("cap", error_object=body))
    assert raised.value.status_code == 400
    assert raised.value.detail == {"error": body}


def test_pre_stream_load_shedding_is_still_503_with_retry_after():
    with pytest.raises(HTTPException) as raised:
        _raise_backpressure_503(BackpressureError("busy"))
    assert raised.value.status_code == 503
    assert raised.value.headers == {"Retry-After": "1"}


def test_a_waiting_prompt_is_tokenized_once_across_admissions():
    calls = []

    def encode(s):
        calls.append(len(s))
        return s.split()

    sched = _sched(_GemmaLike(), SimpleNamespace(encode=encode, decode=lambda ids, **_: ""))
    request = _request("tok " * 3000, max_tokens=0)
    first = sched._estimate_request_kv_bytes(request)
    assert sched._estimate_request_kv_bytes(request) == first
    assert calls == [len("tok " * 3000)]


def test_the_metal_cap_refusal_names_the_prompt_room_not_a_context_window():
    """Review of goose I-9 (2026-10-20): this refusal's number is the largest
    PROMPT that fits beside the request's own max_tokens reservation. Worded as
    "the maximum context here is N tokens", a client sized its whole session's
    window to it. The memory guard's refusal (a real window) keeps that wording."""
    sched = _cap_sched(active=20_000_000_000, cap=22_000_000_000)
    with pytest.raises(BackpressureError) as raised:
        sched._enforce_metal_cap_at_admission(_request(tokens=200_000, max_tokens=4096))
    body = raised.value.error_object
    n_ctx = body["n_ctx"]
    assert body["max_tokens_reserved"] == 4096
    assert body["code"] == "context_length_exceeded"
    assert body["lz_reason"] == "metal_cap_exceeds_device"
    assert "maximum context" not in body["message"]
    assert body["message"].startswith(
        "This prompt (200000 tokens) plus this request's max_tokens reservation of "
        "4096 tokens needs "
    )
    assert body["message"].endswith(
        f"; the largest prompt that fits beside that reservation is {n_ctx} tokens."
    )

    from rapid_mlx.errors import ContextExceedsDeviceMemoryError

    guard = ContextExceedsDeviceMemoryError(
        prompt_tokens=82_684,
        needed_bytes=131_800_000_000,
        available_bytes=75_100_000_000,
        max_context=35_328,
    )
    assert guard.message.endswith("the maximum context here is 35328 tokens.")
    assert "max_tokens_reserved" not in guard.error_object()
