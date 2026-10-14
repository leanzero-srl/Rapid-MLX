# SPDX-License-Identifier: Apache-2.0
"""Memory guard E1 / E5 on the REAL scheduler and mlx-lm ``BatchGenerator``.

A tiny Llama runs on the CPU device. The Metal allocator is SIMULATED: each
prefill chunk costs ``A_TRUE x batch x chunk x context + K_TRUE x batch x
chunk`` of transient memory beyond the active bytes (the head_dim-256 score
tensors of goose's OOM, scaled down), the KV grows with every token, and a
chunk whose peak passes the physical line raises the exact Metal error goose
saw. No GPU is touched and no server is started.

On the unpatched lz.10 scheduler the long prompt dies with that error at a
long-context chunk; with the guard it is served in smaller chunks (E1), a
prompt that cannot fit even at the floor chunk is refused with the typed
error before anything is allocated (E1-d), and an OOM the guard did not
foresee requeues the request once and it completes (E5).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("mlx")
pytestmark = pytest.mark.requires_mlx

import importlib  # noqa: E402

import mlx.core as mx  # noqa: E402

_GEN = importlib.import_module("mlx_lm.generate")
_REAL_PROMPT = _GEN.PromptProcessingBatch.prompt

from rapid_mlx.request import Request, SamplingParams  # noqa: E402
from rapid_mlx.scheduler import Scheduler, SchedulerConfig  # noqa: E402

OOM_TEXT = (
    "[METAL] Command buffer execution failed: Insufficient Memory "
    "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
)
CHUNK = 64
FLOOR = 8
PROMPT = 256
BASE = 1_000_000  # simulated weights
KV_PER_TOKEN = 512
A_TRUE = 48.0  # bytes per (batch x chunk x context)
K_TRUE = 2048.0  # bytes per prefill token


@pytest.fixture(scope="module")
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.fixture(autouse=True)
def _no_metal(monkeypatch, cpu):
    # mlx-lm's BatchGenerator reads the Metal working-set size when Metal is
    # available; on the CPU device that key is absent.
    monkeypatch.setattr(mx.metal, "is_available", lambda: False)


@pytest.fixture(scope="module")
def llama(cpu):
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
            max_position_embeddings=4096,
        )
    )
    mx.eval(model.parameters())
    return model


class _Device:
    """Simulated Metal: active bytes, a peak counter, a physical OOM line."""

    def __init__(self, physical: int, a_true: float = A_TRUE):
        self.physical = physical
        self.a_true = a_true
        self.active = BASE
        self.peak = BASE
        self.ooms = 0
        self.chunks: list[tuple[int, int, int]] = []
        self.fail_next: int = 0  # force N OOMs regardless of size

    def install(self, monkeypatch, scheduler: Scheduler) -> None:
        device = self
        real_prompt = _REAL_PROMPT

        def prompt(batch, tokens):
            if tokens:
                n = max(len(t) for t in tokens)
                b = len(tokens)
                x = max(len(t) for t in batch.tokens) + n
                transient = int(device.a_true * b * n * x + K_TRUE * b * n)
                device.peak = max(device.peak, device.active + transient)
                device.chunks.append((b, n, x))
                if device.fail_next > 0 or device.active + transient > device.physical:
                    device.fail_next = max(0, device.fail_next - 1)
                    device.ooms += 1
                    raise RuntimeError(OOM_TEXT)
                device.active += KV_PER_TOKEN * b * n
            return real_prompt(batch, tokens)

        monkeypatch.setattr(_GEN.PromptProcessingBatch, "prompt", prompt)
        monkeypatch.setattr(mx, "get_peak_memory", lambda: device.peak)

        def reset():
            device.peak = device.active

        monkeypatch.setattr(mx, "reset_peak_memory", reset)
        scheduler._current_metal_active_bytes = lambda: device.active
        scheduler._current_metal_cache_bytes = lambda: 0
        scheduler._current_process_resident_bytes = lambda: device.active
        # The engine's Metal cap sits below the physical line (0.90-0.97 of
        # the working set in production).
        cap = int(device.physical * 0.97)
        scheduler._resolve_metal_cap_bytes = lambda: cap

    def finish(self, request: Request) -> None:
        # The finished request's KV is freed.
        self.active = BASE


def _scheduler(model, **config) -> Scheduler:
    tokenizer = SimpleNamespace(
        encode=lambda text: list(text),
        decode=lambda ids, **_: " ".join(map(str, ids)),
        eos_token_id=127,
        eos_token_ids={127},
    )
    return Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            **{
                "max_num_seqs": 1,
                "spec_decode": "none",
                "prefill_step_size": CHUNK,
                "adaptive_prefill_min_chunk_size": FLOOR,
                "enable_prefix_cache": False,
                **config,
            }
        ),
    )


def _request(rid: str, tokens: int, max_tokens: int = 4, first: int = 1) -> Request:
    return Request(
        request_id=rid,
        prompt=[(first + i) % 120 + 1 for i in range(tokens)],
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0),
    )


def _run(scheduler: Scheduler, requests, steps: int = 400):
    outputs: dict[str, list] = {}
    for r in requests:
        scheduler.add_request(r)
    for _ in range(steps):
        out = scheduler.step()
        for o in out.outputs:
            outputs.setdefault(o.request_id, []).append(o)
        if not scheduler.has_requests():
            break
    return outputs


def _tokens(outs) -> list[int]:
    return [t for o in outs for t in (o.new_token_ids or [])]


def _errors(outs) -> list:
    return [o for o in outs if o.error]


def test_long_prompt_is_served_in_smaller_chunks_instead_of_ooming(llama, monkeypatch):
    """RED on lz.10: the 64-token chunk at a ~200-token context peaks past the
    physical line and the request dies with the Metal OOM. GREEN: the guard
    halves the chunk before dispatch; no OOM, the answer is the solo answer."""
    # Reference answer with ample memory.
    roomy = _scheduler(llama)
    _Device(physical=10**12).install(monkeypatch, roomy)
    ref = _run(roomy, [_request("r", PROMPT)])["r"]
    assert not _errors(ref)

    # Physical line: weights + the prompt's KV + 600 kB of transient.
    device = _Device(physical=BASE + KV_PER_TOKEN * PROMPT + 600_000)
    scheduler = _scheduler(llama)
    device.install(monkeypatch, scheduler)
    outs = _run(scheduler, [_request("r", PROMPT)])["r"]

    assert device.ooms == 0, device.chunks
    assert not _errors(outs), [o.error for o in outs]
    assert outs[-1].finished
    assert len(_tokens(outs)) == len(_tokens(ref))
    guard = scheduler.get_stats()["memory_guard"]
    assert guard["actions"]["chunk_halvings"] >= 1
    assert max(n for _b, n, _x in device.chunks) == CHUNK  # early chunks full size
    assert min(n for _b, n, _x in device.chunks) < CHUNK
    assert guard["backstop_firings"] == 0


def test_a_prompt_that_cannot_fit_at_the_floor_is_refused_with_the_typed_error(
    llama, monkeypatch
):
    """Even an 8-token chunk at a ~256-token context does not fit: refused
    BEFORE allocating with the typed error (no OOM raised at all)."""
    device = _Device(physical=BASE + KV_PER_TOKEN * PROMPT + 60_000)
    scheduler = _scheduler(llama)
    device.install(monkeypatch, scheduler)
    outs = _run(scheduler, [_request("r", PROMPT)])["r"]

    assert device.ooms == 0
    terminal = outs[-1]
    assert terminal.finished and terminal.error
    assert terminal.error_kind == "context_exceeds_device_memory"
    detail = terminal.error_detail
    assert detail["code"] == "context_length_exceeded"
    assert detail["type"] == "invalid_request_error"
    assert detail["lz_reason"] == "context_exceeds_device_memory"
    assert detail["n_prompt_tokens"] == PROMPT
    # The honest window here is 0: the first KV write alone allocates a
    # KVCache.step block for every layer (131 kB), over the 60 kB left.
    assert 0 <= detail["n_ctx"] < PROMPT
    assert detail["message"].startswith(f"This prompt ({PROMPT} tokens) needs ")
    assert scheduler.get_stats()["memory_guard"]["actions"]["refusals"] == 1


def test_negative_control_roomy_device_takes_no_guard_action(llama, monkeypatch):
    device = _Device(physical=10**12)
    scheduler = _scheduler(llama)
    device.install(monkeypatch, scheduler)
    outs = _run(scheduler, [_request("r", PROMPT)])["r"]
    assert not _errors(outs)
    actions = scheduler.get_stats()["memory_guard"]["actions"]
    assert actions["chunk_halvings"] == 0 and actions["refusals"] == 0
    assert actions["checks"] >= PROMPT // CHUNK
    # Every chunk at the configured size (the last is the prompt's remainder).
    assert [n for _b, n, _x in device.chunks][:-1] == [CHUNK] * (len(device.chunks) - 1)


def test_backstop_requeues_a_tokenless_request_once_and_it_completes(
    llama, monkeypatch
):
    """E5: the guard's model under-projects (the true attention cost is 10x the
    prior), the first long-context chunk OOMs. The request had emitted no
    token: it is requeued once with a halved chunk cap, the guard learns the
    observation, and the retry completes."""
    device = _Device(physical=BASE + KV_PER_TOKEN * PROMPT + 1_500_000, a_true=200.0)
    scheduler = _scheduler(llama)
    device.install(monkeypatch, scheduler)
    outs = _run(scheduler, [_request("r", PROMPT)])["r"]

    guard = scheduler.get_stats()["memory_guard"]
    assert guard["backstop_firings"] >= 1
    assert guard["actions"]["backstop_requeued"] == 1
    assert not _errors(outs), [o.error for o in outs]
    assert outs[-1].finished and len(_tokens(outs)) > 0
    assert guard["peak_model"]["observations"] > 0


def test_backstop_requeues_only_once(llama, monkeypatch):
    """A second OOM on the same request aborts it as before (no loop)."""
    device = _Device(physical=10**12)
    scheduler = _scheduler(llama)
    device.install(monkeypatch, scheduler)
    device.fail_next = 2
    outs = _run(scheduler, [_request("r", PROMPT)])["r"]
    guard = scheduler.get_stats()["memory_guard"]
    assert guard["actions"]["backstop_requeued"] == 1
    assert guard["backstop_firings"] == 2
    assert outs[-1].finished and outs[-1].error
    assert "Insufficient Memory" in outs[-1].error


def test_backstop_aborts_a_request_that_already_emitted_tokens(llama, monkeypatch):
    """A decoding request (tokens already streamed) cannot be replayed
    transparently: it aborts as today, while the tokenless prefilling request
    beside it is requeued and completes."""
    device = _Device(physical=10**12)
    scheduler = _scheduler(llama, max_num_seqs=4)
    device.install(monkeypatch, scheduler)
    decoding = _request("decoding", 8, max_tokens=60, first=50)
    scheduler.add_request(decoding)
    outputs: dict[str, list] = {}
    for _ in range(4):  # let it prefill and emit a few tokens
        for o in scheduler.step().outputs:
            outputs.setdefault(o.request_id, []).append(o)
    assert _tokens(outputs.get("decoding", []))
    device.fail_next = 1
    rest = _run(scheduler, [_request("long", PROMPT)])
    for rid, outs in rest.items():
        outputs.setdefault(rid, []).extend(outs)

    assert outputs["decoding"][-1].error and "Insufficient Memory" in (
        outputs["decoding"][-1].error
    )
    assert not _errors(outputs["long"])
    assert outputs["long"][-1].finished
    guard = scheduler.get_stats()["memory_guard"]
    assert guard["actions"]["backstop_requeued"] == 1
