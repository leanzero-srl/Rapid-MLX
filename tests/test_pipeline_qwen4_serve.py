"""The pipeline split's OpenAI server, end to end over two local ring ranks.

A random-init 4-bit qwen4_exp (every layer kind) gets a word-level tokenizer
whose chat template declares the XML tool and think contracts, so the server
wires the same parsers it wires for Qwen3.8-Flash-Next.
"""

import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
pytest.importorskip("fastapi")
pytestmark = pytest.mark.requires_mlx

from mlx_lm.utils import quantize_model, save_config, save_model  # noqa: E402

from rapid_mlx.models.qwen4_exp import Model, ModelArgs  # noqa: E402

from .test_pipeline_qwen4 import _free_port_block, _tiny_text_config  # noqa: E402

SERVED = "tiny-flash-pipeline"
TEMPLATE = (
    "{# contracts: tool_calls arguments <tool_call></tool_call> <function=</function>"
    " <parameter=</parameter> enable_thinking <think></think> #}"
    "{% for m in messages %}{{ m['content'] }} {% endfor %}"
)


def _tokenizer_files(path: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {f"w{i}": i for i in range(256)}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="w1"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "eos_token": "w255",
                "unk_token": "w1",
                "chat_template": TEMPLATE,
            }
        )
    )


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("qwen4_pipeline_serve")
    mx.random.seed(20260924)
    text_config = _tiny_text_config()
    config = {"model_type": "qwen4_exp", "text_config": text_config}
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=text_config))
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    model, config = quantize_model(model, config, group_size=64, bits=4)
    mx.eval(model.parameters())
    save_model(path, model, donate_model=False)
    save_config(config, path / "config.json")
    _tokenizer_files(path)
    return path


class _Server:
    def __init__(self, checkpoint: Path):
        launcher = Path(sys.executable).parent / "mlx.launch"
        if not launcher.exists():
            pytest.skip("mlx.launch is not installed next to this interpreter")
        self.port = _free_port_block(1)
        self.url = f"http://127.0.0.1:{self.port}"
        self.process = subprocess.Popen(
            [
                str(launcher),
                "--backend",
                "ring",
                "-n",
                "2",
                "--hosts",
                "127.0.0.1",
                "--starting-port",
                str(_free_port_block(2)),
                "--",
                sys.executable,
                "-m",
                "rapid_mlx.distributed.pipeline_qwen4",
                "serve",
                "--model",
                str(checkpoint),
                "--served-model-name",
                SERVED,
                "--port",
                str(self.port),
                "--context",
                "512",
                "--split",
                "4",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.lines: list[str] = []
        self.ready: dict[int, dict] = {}
        # A hang guard for the harness only; the engine has no clock.
        deadline = time.monotonic() + 300
        while len(self.ready) < 2:
            assert time.monotonic() < deadline, "\n".join(self.lines)
            readable, _, _ = select.select([self.process.stdout], [], [], 1)
            if not readable:
                assert self.process.poll() is None, "\n".join(self.lines)
                continue
            line = self.process.stdout.readline()
            self.lines.append(line.rstrip())
            if line.startswith("PIPELINE_READY "):
                payload = json.loads(line.split(" ", 1)[1])
                self.ready[payload["rank"]] = payload
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        for line in self.process.stdout:
            self.lines.append(line.rstrip())

    def post(self, body: dict):
        request = urllib.request.Request(
            self.url + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
        )
        return urllib.request.urlopen(request, timeout=120)

    def get(self, path: str) -> dict:
        return json.load(urllib.request.urlopen(self.url + path, timeout=30))


@pytest.fixture(scope="module")
def server(checkpoint):
    running = _Server(checkpoint)
    yield running
    if running.process.poll() is None:
        os.kill(running.ready[0]["pid"], signal.SIGTERM)
        running.process.wait(timeout=60)


def _chat(text: str, **extra) -> dict:
    return {"model": SERVED, "messages": [{"role": "user", "content": text}], **extra}


def test_models_lists_only_the_served_id_with_its_parsers(server):
    models = server.get("/v1/models")["data"]
    assert [model["id"] for model in models] == [SERVED]
    assert models[0]["context_window"] == 512
    assert models[0]["tool_call_parser"] == "qwen3_coder_xml"
    assert models[0]["reasoning_parser"] == "deepseek_r1"


def test_another_model_name_is_a_404_never_a_load(server):
    with pytest.raises(urllib.error.HTTPError) as refusal:
        server.post({**_chat("w5 w6"), "model": "some-other-model"})
    assert refusal.value.code == 404
    assert SERVED in json.load(refusal.value)["error"]["message"]


def test_non_streaming_answer_is_deterministic_and_counts_usage(server):
    first = json.load(server.post(_chat("w5 w6 w7", max_tokens=12)))
    second = json.load(server.post(_chat("w5 w6 w7", max_tokens=12)))
    assert first["choices"][0]["message"] == second["choices"][0]["message"]
    usage = first["usage"]
    assert usage["completion_tokens"] <= 12
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert first["model"] == SERVED


def test_stream_has_role_deltas_finish_usage_and_done(server):
    raw = server.post(_chat("w9 w10", max_tokens=10, stream=True)).read().decode()
    frames = [line[6:] for line in raw.splitlines() if line.startswith("data: ")]
    assert frames[-1] == "[DONE]"
    chunks = [json.loads(frame) for frame in frames[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    final = chunks[-1]
    assert final["choices"][0]["finish_reason"] in ("stop", "length")
    assert final["usage"]["completion_tokens"] >= 1
    assert all(chunk["model"] == SERVED for chunk in chunks)


def test_streamed_and_non_streamed_text_agree(server):
    whole = json.load(server.post(_chat("w20 w21 w22", max_tokens=16)))["choices"][0]
    raw = server.post(_chat("w20 w21 w22", max_tokens=16, stream=True)).read().decode()
    pieces = {"content": "", "reasoning_content": ""}
    for line in raw.splitlines():
        if line.startswith("data: {"):
            delta = json.loads(line[6:])["choices"][0]["delta"]
            for key in pieces:
                pieces[key] += delta.get(key) or ""
    message = whole["message"]
    assert pieces["content"] == (message.get("content") or "")
    assert pieces["reasoning_content"] == (message.get("reasoning_content") or "")


def test_queued_requests_are_batched_and_a_disconnect_cancels(server):
    before = server.get("/goose/progress")["steps"]
    results: dict[str, dict] = {}

    def run(key: str, body: dict) -> None:
        results[key] = json.load(server.post(body))

    long_request = threading.Thread(
        target=run, args=("long", _chat("w3", max_tokens=160))
    )
    long_request.start()
    while server.get("/v1/status")["num_running"] == 0:
        time.sleep(0.05)
    pair = [
        threading.Thread(target=run, args=(key, _chat(text, max_tokens=40)))
        for key, text in (("a", "w30 w31"), ("b", "w40 w41 w42 w43"))
    ]
    for thread in pair:
        thread.start()
    while server.get("/v1/status")["num_waiting"] < 2:
        time.sleep(0.05)
    for thread in [long_request, *pair]:
        thread.join()
    steps = server.get("/goose/progress")["steps"] - before
    produced = sum(results[k]["usage"]["completion_tokens"] for k in ("a", "b"))
    # Sequential service would take one step per token of each answer; the
    # pair shared its decode steps.
    assert steps - results["long"]["usage"]["completion_tokens"] < produced

    import http.client

    connection = http.client.HTTPConnection("127.0.0.1", server.port)
    connection.request(
        "POST",
        "/v1/chat/completions",
        json.dumps(_chat("w7", max_tokens=400, stream=True)),
        {"content-type": "application/json"},
    )
    response = connection.getresponse()
    for _ in range(4):
        response.fp.readline()
    start = server.get("/goose/progress")["steps"]
    connection.sock.close()
    connection.close()
    answer = json.load(server.post(_chat("w8", max_tokens=4)))
    assert answer["usage"]["completion_tokens"] >= 1
    assert server.get("/goose/progress")["steps"] - start < 400


def test_image_parts_are_refused_by_name(server):
    body = {
        "model": SERVED,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "x"}}],
            }
        ],
    }
    with pytest.raises(urllib.error.HTTPError) as refusal:
        server.post(body)
    assert refusal.value.code == 400
    assert "text only" in json.load(refusal.value)["error"]["message"]


def test_sigterm_to_rank0_stops_every_rank(checkpoint):
    running = _Server(checkpoint)
    os.kill(running.ready[0]["pid"], signal.SIGTERM)
    assert running.process.wait(timeout=60) is not None
    for rank in (0, 1):
        pid = running.ready[rank]["pid"]
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
