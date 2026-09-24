# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible server over the qwen4_exp pipeline split.

Every rank runs :func:`serve`.  Rank 0 also runs an HTTP thread (FastAPI +
uvicorn) and owns the scheduling; the generation loop stays on each rank's
main thread because MLX work and the collectives must be issued in the same
order on every rank.  Rank 0 hands each batch to the other ranks as three
``all_sum`` broadcasts (an int header, the left-padded token ids, the
sampling floats); every decode step then closes with one ``all_sum`` that
carries the sampled tokens, the memory-guard stop flag and rank 0's control
word (end the batch).

What the HTTP side reuses from the single engine, unchanged:
``utils.chat_template.apply_chat_template`` (the same rendering, tool-loop
think handling included), ``api.tool_calling.convert_tools_for_template`` and
``engine.batched._normalize_tool_call_arguments_for_template``, and the
route layer's ``service.postprocessor.StreamingPostProcessor`` configured with
the parsers the checkpoint's own chat template declares (the parameterized
XML tool contract -> ``qwen3_coder_xml``; ``<think>`` + ``enable_thinking``
-> ``deepseek_r1``) — the same rule goose-sidecar's ``model_parsers.rs``
applies when it launches the single engine.

Scheduling: a batch is formed from whatever is queued when the previous batch
ends (up to ``--max-batch``, 2 proven bit-exact against single-process
batches); requests arriving mid-batch wait for the next one.  A finished,
EOS'd or cancelled row stops receiving tokens; the batch ends one decode
step after its last live row finishes (the control word rides the next
step's collective).
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
from fastapi import Request

from . import pipeline_qwen4 as pipe

_CMD_SHUTDOWN = 1
_CMD_BATCH = 2
_TOOL_XML_MARKERS = (
    "tool_calls",
    "arguments",
    "<tool_call>",
    "</tool_call>",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
)
_THINK_MARKERS = ("enable_thinking", "<think>", "</think>")


# ---------------------------------------------------------------------------
# rank-shared batch execution
# ---------------------------------------------------------------------------


@dataclass
class _Row:
    ids: list[int]
    max_tokens: int
    temperature: float
    top_p: float


def _broadcast_batch(
    group, rows: list[_Row] | None, max_batch: int
) -> list[_Row] | None:
    """Rank 0's batch, identical on every rank; None means shut down."""
    rank0 = group.rank() == 0
    header = [0] * (3 + 2 * max_batch)
    if rank0:
        if rows is None:
            header[0] = _CMD_SHUTDOWN
        else:
            header[0] = _CMD_BATCH
            header[1] = len(rows)
            header[2] = max(len(row.ids) for row in rows)
            for index, row in enumerate(rows):
                header[3 + index] = len(row.ids)
                header[3 + max_batch + index] = row.max_tokens
    agreed = mx.distributed.all_sum(mx.array(header, dtype=mx.int32), group=group)
    header = agreed.tolist()
    if header[0] == _CMD_SHUTDOWN:
        return None
    batch, width = header[1], header[2]
    ids = [0] * (batch * width)
    floats = [0.0] * (2 * batch)
    if rank0:
        for index, row in enumerate(rows):
            offset = index * width + width - len(row.ids)
            ids[offset : offset + len(row.ids)] = row.ids
            floats[2 * index] = row.temperature
            floats[2 * index + 1] = row.top_p
    ids = mx.distributed.all_sum(mx.array(ids, dtype=mx.int32), group=group).tolist()
    floats = mx.distributed.all_sum(
        mx.array(floats, dtype=mx.float32), group=group
    ).tolist()
    result = []
    for index in range(batch):
        length = header[3 + index]
        row_ids = ids[index * width + width - length : (index + 1) * width]
        result.append(
            _Row(
                ids=row_ids,
                max_tokens=header[3 + max_batch + index],
                temperature=floats[2 * index],
                top_p=floats[2 * index + 1],
            )
        )
    return result


def _sample(logits: mx.array, rows: list[_Row]) -> mx.array:
    from mlx_lm.sample_utils import make_sampler

    picked = []
    for index, row in enumerate(rows):
        line = logits[index : index + 1].astype(mx.float32)
        if row.temperature <= 0:
            picked.append(mx.argmax(line, axis=-1))
        else:
            logprobs = line - mx.logsumexp(line, axis=-1, keepdims=True)
            sampler = make_sampler(temp=row.temperature, top_p=row.top_p)
            picked.append(sampler(logprobs))
    return mx.concatenate(picked).astype(mx.int32)


def _step(
    stage, out, cache, rows, guard, control: int, *, sample: bool
) -> tuple[list[int], int]:
    """One step's collective: tokens, the guard's stop flag, rank 0's control."""
    batch = len(rows)
    reason = guard.check() if guard is not None else None
    if stage.is_last and sample:
        tokens = _sample(out[:, -1, :], rows)
    else:
        tokens = mx.depends(mx.zeros((batch,), dtype=mx.int32), out)
    payload = mx.concatenate(
        [
            tokens,
            mx.array(
                [1 if reason else 0, control if stage.is_first else 0], dtype=mx.int32
            ),
        ]
    )
    if stage.size > 1:
        payload = mx.distributed.all_sum(payload, group=stage.group)
    mx.eval(payload, [layer_cache.state for layer_cache in cache])
    values = payload.tolist()
    if values[batch]:
        raise pipe.PipelineMemoryStopError(
            reason or "a peer rank's memory guard tripped"
        )
    return values[:batch], values[batch + 1]


def run_batch(
    stage, guard, rows: list[_Row], prefill_step: int, on_tokens=None, control_fn=None
) -> None:
    """Prefill + decode one batch on this rank.

    ``on_tokens(step, tokens)`` (rank 0) receives each step's sampled tokens;
    ``control_fn()`` (rank 0) returns 1 to end the batch at the next step.
    """
    width = max(len(row.ids) for row in rows)
    padded = [[0] * (width - len(row.ids)) + row.ids for row in rows]
    tokens = mx.array(padded, dtype=mx.int32)
    padding = [width - len(row.ids) for row in rows]
    cache = stage.make_cache(padding if len(rows) > 1 else None)
    prefix = tokens[:, :-1]
    for offset in range(0, prefix.shape[1], prefill_step):
        out = stage.forward(
            prefix[:, offset : offset + prefill_step], cache, logits=None
        )
        _step(stage, out, cache, rows, guard, 0, sample=False)
    current = tokens[:, -1:]
    for step in range(max(row.max_tokens for row in rows)):
        control = control_fn() if (control_fn is not None and stage.is_first) else 0
        out = stage.forward(current, cache, logits="last")
        sampled, ended = _step(stage, out, cache, rows, guard, control, sample=True)
        if ended:
            return
        if on_tokens is not None:
            on_tokens(step, sampled)
        current = mx.array(sampled, dtype=mx.int32)[:, None]


# ---------------------------------------------------------------------------
# rank 0: jobs, HTTP
# ---------------------------------------------------------------------------


@dataclass
class _Job:
    row: _Row
    loop: asyncio.AbstractEventLoop
    events: asyncio.Queue
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:24])
    cancelled: bool = False
    finished: bool = False
    produced: int = 0

    def push(self, item) -> None:
        self.loop.call_soon_threadsafe(self.events.put_nowait, item)


@dataclass
class _State:
    served: str
    context: int
    max_batch: int
    jobs: queue.Queue = field(default_factory=queue.Queue)
    active: list = field(default_factory=list)
    steps: int = 0
    admission_open: bool = True
    admission_reason: str | None = None
    shutting_down: bool = False
    eos_ids: frozenset = frozenset()


def _template_parsers(tokenizer) -> tuple[str | None, str | None]:
    template = getattr(tokenizer, "chat_template", None) or ""
    if not isinstance(template, str):
        return None, None
    tool = "qwen3_coder_xml" if all(m in template for m in _TOOL_XML_MARKERS) else None
    reasoning = (
        "deepseek_r1" if tool and all(m in template for m in _THINK_MARKERS) else None
    )
    return tool, reasoning


def _text_only(messages: list[dict]) -> list[dict]:
    flattened = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                kind = part.get("type") if isinstance(part, dict) else None
                if kind == "text":
                    parts.append(part.get("text", ""))
                else:
                    raise ValueError(
                        f"content part type {kind!r} is not supported: the pipeline "
                        "split serves text only (the vision tower is not loaded)"
                    )
            message = {**message, "content": "".join(parts)}
        flattened.append(message)
    return flattened


def _merge_tool_call_deltas(deltas: list[dict]) -> list[dict]:
    """Fold the postprocessor's streaming tool-call deltas into whole calls."""
    merged: dict[int, dict] = {}
    for delta in deltas:
        call = merged.setdefault(
            delta.get("index", 0),
            {
                "id": None,
                "type": "function",
                "function": {"name": None, "arguments": ""},
            },
        )
        call["id"] = call["id"] or delta.get("id")
        function = delta.get("function") or {}
        call["function"]["name"] = call["function"]["name"] or function.get("name")
        call["function"]["arguments"] += function.get("arguments") or ""
    return [merged[index] for index in sorted(merged)]


def _build_app(state: _State, tokenizer, eos_ids: set[int]):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse

    from ..api.tool_calling import convert_tools_for_template
    from ..config.server_config import ServerConfig
    from ..engine.base import GenerationOutput
    from ..engine.batched import _normalize_tool_call_arguments_for_template
    from ..service.helpers import _should_start_in_thinking
    from ..service.postprocessor import StreamingPostProcessor
    from ..utils.chat_template import apply_chat_template

    tool_parser, reasoning_parser = _template_parsers(tokenizer)
    cfg = ServerConfig()
    cfg.enable_auto_tool_choice = tool_parser is not None
    cfg.tool_call_parser = tool_parser
    cfg.reasoning_parser_name = reasoning_parser
    cfg.engine = type(
        "_TokenizerHolder", (), {"tokenizer": tokenizer, "_tokenizer": tokenizer}
    )()
    app = FastAPI()

    def error(status: int, message: str, kind: str) -> JSONResponse:
        return JSONResponse(
            status_code=status, content={"error": {"message": message, "type": kind}}
        )

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [
                {
                    "id": state.served,
                    "object": "model",
                    "owned_by": "rapid-mlx-pipeline",
                    "context_window": state.context,
                    "tool_call_parser": tool_parser,
                    "reasoning_parser": reasoning_parser,
                }
            ],
        }

    @app.get("/v1/status")
    async def status():
        running = sum(1 for job in state.active if not job.finished)
        return {
            "num_running": running,
            "num_waiting": state.jobs.qsize(),
            "status": "ok",
        }

    @app.get("/goose/progress")
    async def progress():
        return {
            "steps": state.steps,
            "inflight": len(state.active) + state.jobs.qsize(),
            "admission_open": state.admission_open,
            "active": mx.get_active_memory(),
            "peak": mx.get_peak_memory(),
        }

    @app.post("/goose/admission")
    async def admission(request: Request):
        body = await request.json()
        state.admission_open = bool(body.get("open"))
        state.admission_reason = body.get("reason")
        return {"admission_open": state.admission_open}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        if not state.admission_open or state.shutting_down:
            return error(
                503,
                "the pipeline engine is not admitting new requests: "
                + str(state.admission_reason or "shutting down"),
                "server_busy",
            )
        body = await request.json()
        model = body.get("model")
        if model not in (None, state.served):
            return error(
                404,
                f"model '{model}' is not served here; this engine serves '{state.served}'",
                "model_not_found",
            )
        tools = body.get("tools") or None
        if tools and tool_parser is None:
            return error(
                400,
                "tools are not supported for this checkpoint: its chat template does not "
                "declare the parameterized XML tool contract the pipeline server parses",
                "tools_unsupported",
            )
        try:
            messages = _normalize_tool_call_arguments_for_template(
                _text_only(list(body.get("messages") or []))
            )
        except ValueError as refusal:
            return error(400, str(refusal), "invalid_request_error")
        kwargs = dict(body.get("chat_template_kwargs") or {})
        enable_thinking = kwargs.pop("enable_thinking", body.get("enable_thinking"))
        prompt = apply_chat_template(
            tokenizer,
            messages,
            tools=convert_tools_for_template(tools) if tools else None,
            enable_thinking=enable_thinking,
            model_name=state.served,
            chat_template_kwargs=kwargs or None,
        )
        ids = tokenizer.encode(prompt)
        budget = state.context - len(ids) - 1
        requested = body.get("max_completion_tokens") or body.get("max_tokens")
        if budget < 1:
            return error(
                400,
                f"prompt is {len(ids)} tokens; the split was planned for a {state.context}-token context",
                "context_length_exceeded",
            )
        max_tokens = min(int(requested), budget) if requested else budget
        temperature = float(body.get("temperature") or 0.0)
        top_p = float(body.get("top_p") or 1.0)
        stops = body.get("stop") or []
        stops = [stops] if isinstance(stops, str) else list(stops)
        loop = asyncio.get_running_loop()
        job = _Job(_Row(ids, max_tokens, temperature, top_p), loop, asyncio.Queue())
        state.jobs.put(job)
        created = int(time.time())
        processor = StreamingPostProcessor(
            cfg,
            tools_requested=bool(tools),
            enable_thinking=enable_thinking,
            request=body,
        )
        processor.reset()
        if processor.reasoning_parser is not None and _should_start_in_thinking(
            getattr(tokenizer, "chat_template", "") or "",
            enable_thinking,
            tools_requested=bool(tools),
        ):
            # The template opens <think> in the generation prompt, so the
            # model never emits the opener.  deepseek_r1's streaming path
            # flips a tagless stream to content after 64 characters unless it
            # is told the prompt primed thinking (its own hook, set by
            # configure_request on the distill variant).  Measured on Flash:
            # without it every thought past 64 chars streamed as content.
            processor.reasoning_parser._prompt_primed_thinking = True

        async def generate():
            """Yield (events, finish_reason, completion_tokens) as tokens arrive."""
            detok = tokenizer.detokenizer
            text = ""
            finish = "length"
            completion = 0
            while True:
                item = await job.events.get()
                if item[0] == "error":
                    raise RuntimeError(item[1])
                if item[0] == "done":
                    finish = item[1]
                    break
                token = item[1]
                completion += 1
                if token in eos_ids:
                    finish = "stop"
                    job.finished = True
                    break
                detok.add_token(token)
                piece = detok.last_segment
                if stops:
                    candidate = text + piece
                    hit = min(
                        (
                            candidate.find(s, max(0, len(text) - len(s)))
                            for s in stops
                            if s in candidate
                        ),
                        default=-1,
                    )
                    if hit >= 0:
                        piece = candidate[:hit][len(text) :]
                        text += piece
                        finish = "stop"
                        job.finished = True
                        if piece:
                            yield (
                                processor.process_chunk(
                                    GenerationOutput(
                                        text=text,
                                        new_text=piece,
                                        prompt_tokens=len(ids),
                                        completion_tokens=completion,
                                        finished=False,
                                        finish_reason=None,
                                    )
                                ),
                                None,
                                completion,
                            )
                        break
                text += piece
                if piece:
                    yield (
                        processor.process_chunk(
                            GenerationOutput(
                                text=text,
                                new_text=piece,
                                prompt_tokens=len(ids),
                                completion_tokens=completion,
                                finished=False,
                                finish_reason=None,
                            )
                        ),
                        None,
                        completion,
                    )
            detok.finalize()
            tail = detok.last_segment
            if tail and finish != "stop":
                text += tail
                yield (
                    processor.process_chunk(
                        GenerationOutput(
                            text=text,
                            new_text=tail,
                            prompt_tokens=len(ids),
                            completion_tokens=completion,
                            finished=False,
                            finish_reason=None,
                        )
                    ),
                    None,
                    completion,
                )
            terminal = processor.process_chunk(
                GenerationOutput(
                    text="",
                    new_text="",
                    prompt_tokens=len(ids),
                    completion_tokens=completion,
                    finished=True,
                    finish_reason=finish,
                )
            )
            terminal.extend(processor.finalize())
            yield terminal, finish, completion

        def delta_of(event) -> dict:
            delta: dict[str, Any] = {}
            if event.content:
                delta["content"] = event.content
            if event.reasoning:
                delta["reasoning_content"] = event.reasoning
            if event.tool_calls:
                delta["tool_calls"] = event.tool_calls
            return delta

        if body.get("stream"):

            async def sse():
                def chunk(delta, finish=None, usage=None):
                    payload = {
                        "id": f"chatcmpl-{job.id}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": state.served,
                        "choices": [
                            {"index": 0, "delta": delta, "finish_reason": finish}
                        ],
                    }
                    if usage is not None:
                        payload["usage"] = usage
                    return f"data: {json.dumps(payload)}\n\n"

                yield chunk({"role": "assistant"})
                finish = "stop"
                completion = 0
                try:
                    async for events, final, completion in generate():
                        for event in events:
                            if event.finish_reason:
                                finish = event.finish_reason
                            delta = delta_of(event)
                            if delta:
                                yield chunk(delta)
                        if final is not None and finish == "stop":
                            finish = final
                except RuntimeError as failure:
                    yield f"data: {json.dumps({'error': {'message': str(failure), 'type': 'pipeline_error'}})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                finally:
                    job.cancelled = True
                usage = {
                    "prompt_tokens": len(ids),
                    "completion_tokens": completion,
                    "total_tokens": len(ids) + completion,
                }
                yield chunk({}, finish, usage)
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        content, reasoning, calls = [], [], []
        finish = "stop"
        completion = 0
        try:
            async for events, final, completion in generate():
                for event in events:
                    if event.finish_reason:
                        finish = event.finish_reason
                    if event.content:
                        content.append(event.content)
                    if event.reasoning:
                        reasoning.append(event.reasoning)
                    if event.tool_calls:
                        calls.extend(event.tool_calls)
                if final is not None and finish == "stop":
                    finish = final
        except RuntimeError as failure:
            return error(500, str(failure), "pipeline_error")
        finally:
            job.cancelled = True
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content).strip() or None,
        }
        if reasoning:
            message["reasoning_content"] = "".join(reasoning).strip()
        if calls:
            message["tool_calls"] = _merge_tool_call_deltas(calls)
            finish = "tool_calls"
        return {
            "id": f"chatcmpl-{job.id}",
            "object": "chat.completion",
            "created": created,
            "model": state.served,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": len(ids),
                "completion_tokens": completion,
                "total_tokens": len(ids) + completion,
            },
        }

    return app


def _run_jobs(
    stage, guard, state: _State, batch: list[_Job], prefill_step: int
) -> None:
    def on_tokens(step: int, tokens: list[int]) -> None:
        state.steps += 1
        for job, token in zip(batch, tokens):
            if job.finished or job.cancelled:
                continue
            job.produced += 1
            job.push(("token", token))
            if token in state.eos_ids:
                job.finished = True
                job.push(("done", "stop"))
            elif job.produced >= job.row.max_tokens:
                job.finished = True
                job.push(("done", "length"))

    def control() -> int:
        return int(all(job.finished or job.cancelled for job in batch))

    try:
        run_batch(
            stage, guard, [job.row for job in batch], prefill_step, on_tokens, control
        )
    except pipe.PipelineMemoryStopError as stop:
        for job in batch:
            job.push(("error", f"memory guard stopped the pipeline: {stop}"))
        raise
    for job in batch:
        if not job.finished:
            job.finished = True
            job.push(("done", "length"))


def _rank0_loop(stage, guard, state: _State, prefill_step: int) -> None:
    group = stage.group
    while True:
        first = state.jobs.get()
        if first is None:
            _broadcast_batch(group, None, state.max_batch)
            return
        batch = [first]
        while len(batch) < state.max_batch:
            try:
                extra = state.jobs.get_nowait()
            except queue.Empty:
                break
            if extra is None:
                state.jobs.put(None)
                break
            batch.append(extra)
        batch = [job for job in batch if not job.cancelled]
        if not batch:
            continue
        state.active = batch
        _broadcast_batch(group, [job.row for job in batch], state.max_batch)
        _run_jobs(stage, guard, state, batch, prefill_step)
        state.active = []


def _worker_loop(stage, guard, max_batch: int, prefill_step: int) -> None:
    while True:
        rows = _broadcast_batch(stage.group, None, max_batch)
        if rows is None:
            return
        run_batch(stage, guard, rows, prefill_step)


def serve(options, emit=None) -> int:
    """Run one rank of the server (``mlx.distributed.init`` already reachable)."""
    emit = emit or (
        lambda tag, payload: print(f"PIPELINE_{tag} {json.dumps(payload)}", flush=True)
    )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    group = mx.distributed.init(strict=True)
    emit(
        "RANK_GROUP",
        {"rank": group.rank(), "size": group.size(), "mlx": mx.__version__},
    )
    model_dir = Path(options.model).expanduser()
    prefill_step = options.prefill_step or pipe.default_prefill_step()
    stage, plan, guard = pipe.load_stage(
        model_dir,
        group,
        context=options.context,
        batch=options.max_batch,
        prefill_step=prefill_step,
        starts=pipe._parse_starts(options.split),
        log=lambda line: print(line, flush=True),
    )
    emit("RANK_CAPS", {**stage.limits, "planned": plan.stages[stage.rank].total_bytes})
    # Kernel compilation and first-touch paging happen here, before anything
    # is advertised: a readiness probe that succeeds means a request can run.
    warm = [_Row(ids=[0] * 8, max_tokens=2, temperature=0.0, top_p=1.0)]
    run_batch(stage, guard, warm, prefill_step)
    context = options.context or plan.context
    if not stage.is_first:
        emit(
            "READY",
            {
                "rank": stage.rank,
                "pid": os.getpid(),
                "layers": [stage.start, stage.end],
            },
        )
        _worker_loop(stage, guard, options.max_batch, prefill_step)
        return 0

    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(model_dir)
    state = _State(
        served=options.served_model_name,
        context=context,
        max_batch=options.max_batch,
        eos_ids=frozenset(tokenizer.eos_token_ids),
    )

    import uvicorn

    app = _build_app(state, tokenizer, set(tokenizer.eos_token_ids))
    server = uvicorn.Server(
        uvicorn.Config(app, host=options.host, port=options.port, log_level="warning")
    )
    http_started = threading.Event()
    http_loop: list[asyncio.AbstractEventLoop] = []

    @app.on_event("startup")
    async def _capture_loop() -> None:
        http_loop.append(asyncio.get_running_loop())
        http_started.set()

    threading.Thread(target=server.run, name="pipeline-http", daemon=True).start()
    http_started.wait()

    def request_shutdown() -> None:
        state.jobs.put(None)

    def shutdown(*_):
        # A signal handler must not take the job queue's lock: the main
        # thread may hold it inside ``get()`` when the signal lands (measured:
        # the handler's put() deadlocked rank 0 and left rank 1 waiting in
        # its header all_sum).  The HTTP loop's thread does the put instead.
        state.shutting_down = True
        for job in list(state.active):
            job.cancelled = True
        http_loop[0].call_soon_threadsafe(request_shutdown)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    emit(
        "READY",
        {
            "rank": 0,
            "pid": os.getpid(),
            "layers": [stage.start, stage.end],
            "port": options.port,
            "served": state.served,
            "context": context,
            "starts": plan.starts,
        },
    )
    try:
        _rank0_loop(stage, guard, state, prefill_step)
    finally:
        server.should_exit = True
    return 0


def add_arguments(parser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--max-batch", type=int, default=2)
    parser.add_argument("--prefill-step", type=int)
    parser.add_argument(
        "--split", help="pinned starts for ranks 1..N-1 (the preflighted plan)"
    )


if __name__ == "__main__":
    sys.exit(pipe.main(["serve", *sys.argv[1:]]))
