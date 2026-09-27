"""The pipeline split streams a tool call while it is written, says what it withholds,
and ends an answer that repeats itself (goose Q-178: Q-141, Q-146, Q-161 on this
runner), and refuses by name what it cannot honour (goose Q-177, Q-164).

goose E2E #5b (3.0.58, Flash on the pipeline split, 2026-09-27): after "Let me
write VENDORED.md." goose's forming panel read "1 tool call forming — Received
so far: 1 chars of arguments · write" for five minutes while /v1/status said
4,624 tokens generated.  The post-processor sends a qwen3_coder_xml call's
header and ``{``, then — once a string value arrives unquoted
(``_legacy_raw_stream``) — nothing until the answer ends.

Nothing here launches a rank or loads a model: rank 0's HTTP app runs over a
character-level tokenizer whose template declares the XML tool and think
contracts (so the server wires the parsers it wires for Qwen3.8-Flash-Next),
a stand-in batch loop hands each request the tokens of a scripted answer, and
the relay and the post-processor are the real ones.  One test serves the app
with uvicorn on a free local port, to read the stream while it is written.
"""

import asyncio
import json
import queue
import random
import socket
import string
import threading
import time
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
pytest.importorskip("fastapi")
pytestmark = pytest.mark.requires_mlx

from fastapi.testclient import TestClient  # noqa: E402

from rapid_mlx.distributed import pipeline_qwen4_serve as serve  # noqa: E402
from rapid_mlx.distributed import pipeline_stream as ps  # noqa: E402
from rapid_mlx.tool_parsers import ToolParserManager  # noqa: E402

SERVED = "tiny-flash-pipeline"
TEMPLATE = (
    "{# contracts: tool_calls arguments <tool_call></tool_call> <function=</function>"
    " <parameter=</parameter> enable_thinking <think></think> #}"
    "{% for m in messages %}{{ m['content'] }} {% endfor %}"
)
SPECIAL = ["<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>"]
FIXTURES = Path(__file__).parent / "fixtures"
Q133 = json.loads((FIXTURES / "q133_split_flash_undeclared_bash.json").read_text())
CHARS = sorted(
    set(string.printable) | set("ünïcödéè’—…\U0001f9a2") | set("".join(Q133["pieces"]))
)


def schema(name, **props):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": props},
        },
    }


TOOLS = [
    schema(
        "shell",
        command={"type": "string"},
        timeout={"type": "integer"},
        ratio={"type": "number"},
        force={"type": "boolean"},
        env={"type": "object"},
        paths={"type": "array"},
        note={},
    ),
    schema("write", path={"type": "string"}, content={"type": "string"}),
]
LONG = "".join(
    f'line {i}: echo "quoted" \\ back ünïcödé \U0001f9a2\ttab </param <parameter=x> </function>\n'
    for i in range(300)
)


def call(name, *params):
    body = "".join(
        f"<parameter={key}>\n{value}\n</parameter>\n" for key, value in params
    )
    return f"\n<function={name}>\n{body}</function>\n"


# ---------------------------------------------------------------------------
# the relay alone, against the post-processor's own parser
# ---------------------------------------------------------------------------

# goose's tensor fixtures (launch.rs the_tool_stream_sends_exactly_what_mlx_lms_parser_reads),
# read by THIS server's parser, plus the shapes where it reads differently from
# mlx_lm's: a value that is a JSON string, a value holding a literal
# </parameter>, an undeclared parameter after a declared one, a typed value the
# parser keeps as text.
EXACT = {
    "shell": (call("shell", ("command", "cd /work && ls -la")), TOOLS),
    "long_write": (call("write", ("path", "/tmp/x.py"), ("content", LONG)), TOOLS),
    "typed": (
        call(
            "shell",
            ("command", "echo a command long enough to stream"),
            ("timeout", "30"),
            ("ratio", "2.5"),
            ("force", "true"),
            ("env", '{"A": 1}'),
            ("paths", '["a", "b"]'),
        ),
        TOOLS,
    ),
    "typed_first": (
        call(
            "shell",
            ("timeout", "7"),
            ("command", "echo the string after a typed value"),
        ),
        TOOLS,
    ),
    "null_word": (call("shell", ("command", "null")), TOOLS),
    "null_upper": (call("shell", ("command", "NULL")), TOOLS),
    "untyped_note": (
        call("shell", ("note", '{"looks": "like json but the schema has no type"}')),
        TOOLS,
    ),
    "no_newlines": (
        "<function=shell><parameter=command>ls -la /very/long/path/somewhere</parameter></function>",
        TOOLS,
    ),
    "two_newlines": (
        "<function=shell><parameter=command>\n\nkeeps one newline each side\n\n</parameter></function>",
        TOOLS,
    ),
    "no_parameters": ("\n<function=shell>\n</function>\n", TOOLS),
    "short": (call("shell", ("command", "ls")), TOOLS),
    "indented": (
        call(
            "write", ("path", "a.py"), ("content", "    indented first line\n\tx = 1")
        ),
        TOOLS,
    ),
    "json_string_value": (
        call("shell", ("command", '"echo decoded from a JSON string"')),
        TOOLS,
    ),
    "literal_close_in_value": (
        call(
            "write",
            ("path", "doc.md"),
            ("content", "a call ends with </parameter>\nand the doc goes on after it"),
        ),
        TOOLS,
    ),
    "undeclared_parameter_after": (
        "\n<function=shell>\n<parameter=command>\necho a long enough command here\n</parameter>\n"
        "<parameter=bogus>\nx\n</parameter>\n</function>\n",
        TOOLS,
    ),
    "typed_value_kept_as_text": (
        call("shell", ("command", "echo sleep then stop"), ("timeout", "soon")),
        TOOLS,
    ),
}
# Read differently from what was already sent, or not read at all: the client
# must hold arguments it cannot run.
REFUSED = {
    "cut_mid_value": (
        call("write", ("path", "/tmp/y"), ("content", LONG))[:-2000],
        TOOLS,
    ),
    "written_twice": (
        call(
            "shell",
            ("command", "echo the first of two values"),
            ("command", "echo the second"),
        ),
        TOOLS,
    ),
}


def chunkings(text):
    yield "whole", [text]
    yield "chars", list(text)
    rng = random.Random(178)
    for seed in range(3):
        pieces, i = [], 0
        while i < len(text):
            n = rng.randint(1, 9)
            pieces.append(text[i : i + n])
            i += n
        yield f"random{seed}", pieces


MARKERS = ps.Markers(think_start=1, think_end=2, call_start=3, call_end=4)


class _Said:
    def __init__(self):
        self.lines = []

    def __call__(self, tag, payload):
        self.lines.append((tag, payload))

    def tags(self, tag):
        return [payload for said, payload in self.lines if said == tag]


def _relay(tools=TOOLS, request=None, **options):
    said = _Said()
    body = request or {"tools": tools}
    watch = ps.StreamWatch(said, "req-1")
    parser = ToolParserManager.get_tool_parser("qwen3_coder_xml")(None)
    relay = ps.StreamRelay(
        watch,
        MARKERS,
        reasoning=options.pop("reasoning", False),
        parser=parser,
        request=body,
        say=said,
        **options,
    )
    return relay, watch, said


def _run(relay, pieces, closed=True):
    """The call's pieces between its two marker tokens; (client calls, text for the post-processor)."""
    actions = relay.take(3, "<tool_call>")
    for piece in pieces:
        actions += relay.take(None, piece)
    actions += relay.take(4, "</tool_call>") if closed else relay.finish()
    calls: dict[int, dict] = {}
    text = ""
    for kind, value in actions:
        if kind == "text":
            text += value
            continue
        entry = calls.setdefault(
            value["index"], {"name": None, "arguments": "", "frames": 0}
        )
        function = value.get("function") or {}
        entry["name"] = entry["name"] or function.get("name")
        entry["arguments"] += function.get("arguments") or ""
        entry["frames"] += 1
    return calls, text


def _parsed(text, tools):
    parser = ToolParserManager.get_tool_parser("qwen3_coder_xml")(None)
    result = parser.extract_tool_calls(
        f"<tool_call>{text}</tool_call>", {"tools": tools}
    )
    assert result.tools_called and len(result.tool_calls) == 1
    return result.tool_calls[0]


@pytest.mark.parametrize("case", sorted(EXACT))
def test_the_relay_sends_exactly_what_the_parser_reads(case):
    text, tools = EXACT[case]
    whole = _parsed(text, tools)
    for how, pieces in chunkings(text):
        relay, watch, said = _relay(tools)
        calls, rest = _run(relay, pieces)
        assert rest == "", (case, how, rest)
        assert list(calls) == [0], (case, how, calls)
        assert calls[0]["name"] == whole["name"], (case, how)
        assert calls[0]["arguments"] == whole["arguments"], (case, how)
        assert said.tags("RANK_TOOL_CALL_UNPARSED") == [], (case, how)
        if case == "long_write" and how != "whole":
            # The value went out while it was written: many frames, and the
            # last one (sent when the call closed) carries at most the held tail.
            assert calls[0]["frames"] > 100, (how, calls[0]["frames"])


@pytest.mark.parametrize("case", sorted(REFUSED))
def test_what_the_parser_reads_otherwise_is_named_and_unrunnable(case):
    text, tools = REFUSED[case]
    for how, pieces in chunkings(text):
        relay, watch, said = _relay(tools)
        calls, _ = _run(relay, pieces, closed=case != "cut_mid_value")
        (unparsed,) = said.tags("RANK_TOOL_CALL_UNPARSED")
        assert unparsed["why"], (case, how)
        with pytest.raises(json.JSONDecodeError):
            json.loads(calls[0]["arguments"])


def test_a_long_value_is_sent_while_it_is_written():
    text, tools = EXACT["long_write"]
    relay, watch, _ = _relay(tools)
    relay.take(3, "<tool_call>")
    sent = ""
    for piece in list(text[: len(text) // 2]):
        for kind, value in relay.take(None, piece):
            if kind == "call":
                sent += value["function"].get("arguments") or ""
    # Half the call is written: all of it but the held tail is out.
    written = text[: len(text) // 2]
    value_so_far = written.split("<parameter=content>\n", 1)[1]
    head, value_sent = sent.split('"content": "', 1)
    assert head == '{"path": "/tmp/x.py", '
    value_sent = json.loads(f'"{value_sent}"')
    assert value_so_far.startswith(value_sent)
    assert len(value_so_far) - len(value_sent) <= ps.ToolCallStream.HOLD
    report = watch.report()
    assert report["parser_state"] == "tool"
    assert report["tool_call"]["name"] == "write"
    assert report["tool_call"]["phase"] == "value"
    assert report["tool_call"]["parameter"] == "content"
    assert report["tool_call"]["string_value"] is True
    assert report["withholding"] is None and watch.withholding() is None


def test_an_undeclared_name_and_an_unread_call_go_to_the_post_processor_whole():
    undeclared = call("bash", ("cmd", "rm -rf build && make -j8 all install"))
    relay, watch, _ = _relay()
    relay.take(3, "<tool_call>")
    for piece in undeclared[:-4]:
        relay.take(None, piece)
    report = watch.report()["tool_call"]
    assert (
        report["streamed"] is False and "did not declare" in report["why_not_streamed"]
    )
    assert watch.withholding()[0] == "tool_not_streamed"
    relay, watch, _ = _relay()
    calls, text = _run(relay, list(undeclared))
    assert calls == {}
    assert text == f"<tool_call>{undeclared}</tool_call>"

    as_json = '\n{"name": "shell", "arguments": {"command": "ls"}}\n'
    relay, watch, _ = _relay()
    calls, text = _run(relay, list(as_json))
    assert calls == {} and text == f"<tool_call>{as_json}</tool_call>"


def test_a_marker_after_a_held_space_splits_at_the_marker():
    """BPE holds a lone space until the next token: ' <tool_call>' arrives as one
    piece with the marker's id (measured on the Qwen3.8-Flash tokenizer)."""
    relay, _, _ = _relay()
    assert relay.take(None, "Let me write.") == [("text", "Let me write.")]
    assert relay.take(3, " <tool_call>") == [("text", " ")]
    assert relay.state == "tool"


def test_reasoning_first_then_the_call_streams():
    relay, watch, _ = _relay(reasoning=True)
    assert relay.take(None, "plan the <tool_call> in my head") == [
        ("text", "plan the <tool_call> in my head")
    ]
    assert watch.report()["parser_state"] == "reasoning"
    assert relay.take(2, " </think>") == [("text", " "), ("text", "</think>")]
    calls, _ = _run(relay, list(EXACT["shell"][0]))
    assert calls[0]["arguments"] == _parsed(*EXACT["shell"])["arguments"]


def test_the_status_names_a_typed_value_it_holds_and_the_withheld_span():
    relay, watch, said = _relay()
    relay.take(3, "<tool_call>")
    for piece in ["\n<function=shell>\n", "<parameter=paths>\n", '["a", ']:
        relay.take(None, piece)
    watch.settle()
    assert watch.withholding()[0] == "tool_typed_value"
    (enter,) = said.tags("RANK_WITHHELD")
    assert enter["event"] == "enter" and enter["mode"] == "tool_typed_value"
    relay.take(None, '"b"]\n</parameter>\n<parameter=command>\necho now a string\n')
    watch.settle()
    leave = said.tags("RANK_WITHHELD")[-1]
    assert leave["event"] == "leave" and '["a", "b"]' in leave["tail"]


def test_the_relay_steps_aside_where_the_post_processor_decides():
    parser = ToolParserManager.get_tool_parser("qwen3_coder_xml")(None)
    assert parser._declared_tool_names({"tools": TOOLS, "tool_choice": "none"}) == set()
    relay, watch, _ = _relay(owns="parallel_tool_calls is false")
    calls, text = _run(relay, list(EXACT["shell"][0]))
    assert calls == {} and text.startswith("<tool_call>\n<function=shell>")
    assert watch.report()["tool_call"] is None  # the call closed
    assert watch.unstreamed == "parallel_tool_calls is false"


# ---------------------------------------------------------------------------
# rank 0's HTTP app
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer_dir(tmp_path_factory) -> Path:
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

    path = tmp_path_factory.mktemp("stream_parity_tokenizer")
    vocab = {"<unk>": 0, **{c: i + 1 for i, c in enumerate(CHARS)}}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        Regex(r"[\s\S]"), behavior="isolated"
    )
    tokenizer.decoder = decoders.Fuse()
    tokenizer.add_special_tokens(SPECIAL)
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "eos_token": "<|im_end|>",
                "unk_token": "<unk>",
                "chat_template": TEMPLATE,
            }
        )
    )
    return path


class _Server:
    """Rank 0's app over a stand-in batch loop that writes each request a scripted answer.

    The loop gives a token only once the handler has read the one before
    (``_give``'s place in the tick loop), and stops when the job is finished
    or cancelled — so an answer the engine ends leaves tokens ungiven.
    """

    def __init__(self, tokenizer_dir: Path):
        from mlx_lm.utils import load_tokenizer

        self.tokenizer = load_tokenizer(tokenizer_dir)
        self.said = _Said()
        self.state = serve._State(
            served=SERVED,
            context=1 << 20,
            max_batch=4,
            vocab=len(CHARS) + 1 + len(SPECIAL),
            emit=self.said,
        )
        self.app = serve._build_app(
            self.state, self.tokenizer, set(self.tokenizer.eos_token_ids)
        )
        self.client = TestClient(self.app)
        self.scripts: queue.Queue = queue.Queue()
        self.jobs = []
        self.given: dict[str, int] = {}
        self.loop = threading.Thread(target=self._batch_loop, daemon=True)
        self.loop.start()

    def script(self, answer: str, pause_at: int | None = None, gate=None) -> list[int]:
        tokens = [*self.tokenizer.encode(answer, add_special_tokens=False)]
        tokens.append(next(iter(self.tokenizer.eos_token_ids)))
        self.scripts.put((tokens, pause_at, gate))
        return tokens

    def _batch_loop(self):
        self.gate_timed_out = False
        while (job := self.state.jobs.get()) is not None:
            self.jobs.append(job)
            tokens, pause_at, gate = self.scripts.get_nowait()
            given = 0
            for index, token in enumerate(tokens):
                if index == pause_at and not gate.wait(timeout=30):
                    # A harness guard, never the engine's: the client did not
                    # read the arguments while the call was half written.
                    self.gate_timed_out = True
                if job.finished or job.cancelled:
                    break
                job.produced += 1
                given += 1
                asyncio.run_coroutine_threadsafe(
                    job.events.put(("token", token)), job.loop
                ).result(timeout=30)
                deadline = time.monotonic() + 30
                while job.events.qsize() and not (job.finished or job.cancelled):
                    assert time.monotonic() < deadline
                    time.sleep(0.0005)
            self.given[job.id] = given

    def close(self):
        self.state.jobs.put(None)
        self.loop.join(timeout=10)


@pytest.fixture
def server(tokenizer_dir):
    running = _Server(tokenizer_dir)
    yield running
    running.close()


def _body(tools=TOOLS, **extra):
    return {
        "model": SERVED,
        "messages": [{"role": "user", "content": "go"}],
        "tools": tools,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
        **extra,
    }


def _frames(lines):
    frames = []
    for line in lines:
        if line.startswith("data: ") and line != "data: [DONE]":
            frames.append(json.loads(line[len("data: ") :]))
    return frames


def _assemble(frames):
    content, calls, finish, refused, errors = "", {}, None, None, []
    for frame in frames:
        if "error" in frame:
            errors.append(frame["error"])
            continue
        choice = frame["choices"][0]
        delta = choice["delta"]
        content += delta.get("content") or ""
        for part in delta.get("tool_calls") or []:
            entry = calls.setdefault(
                part["index"], {"id": None, "name": None, "arguments": ""}
            )
            entry["id"] = entry["id"] or part.get("id")
            function = part.get("function") or {}
            entry["name"] = entry["name"] or function.get("name")
            entry["arguments"] += function.get("arguments") or ""
        finish = choice.get("finish_reason") or finish
        refused = choice.get("refused_tool_calls") or refused
    return content, calls, finish, refused, errors


def _stream(server, body):
    with server.client.stream("POST", "/v1/chat/completions", json=body) as response:
        assert response.status_code == 200, response.read()
        return _frames(list(response.iter_lines()))


WRITE = call(
    "write",
    ("path", "VENDORED.md"),
    ("content", "# Vendored\n\n" + "| pkg | sha256 |\n|---|---|\n" * 60),
)


def test_the_write_reaches_the_client_as_the_post_processor_would_send_it_whole(server):
    answer = f"I have all hashes. Let me write VENDORED.md.\n\n<tool_call>{WRITE}</tool_call>"
    server.script(answer)
    streamed = _stream(server, _body())
    server.script(answer)
    whole = server.client.post(
        "/v1/chat/completions", json={**_body(), "stream": False}
    ).json()
    content, calls, finish, _, errors = _assemble(streamed)
    assert errors == []
    message = whole["choices"][0]["message"]
    assert [(c["name"], json.loads(c["arguments"])) for c in calls.values()] == [
        (c["function"]["name"], json.loads(c["function"]["arguments"]))
        for c in message["tool_calls"]
    ]
    assert calls[0]["arguments"] == message["tool_calls"][0]["function"]["arguments"]
    assert (
        content.strip()
        == message["content"]
        == "I have all hashes. Let me write VENDORED.md."
    )
    assert finish == whole["choices"][0]["finish_reason"] == "tool_calls"
    argument_frames = [
        f
        for f in streamed
        if f.get("choices") and f["choices"][0]["delta"].get("tool_calls")
    ]
    assert len(argument_frames) > 100


def test_without_the_relay_the_arguments_wait_for_the_end(server, monkeypatch):
    """NEGATIVE CONTROL (E2E #5b's forming panel): the post-processor alone sends
    the header and ``{``, then nothing of the call until the answer ends."""
    real = serve.StreamRelay

    def off(*args, **kwargs):
        kwargs["owns"] = "negative control"
        return real(*args, **kwargs)

    monkeypatch.setattr(serve, "StreamRelay", off)
    server.script(f"Let me write VENDORED.md.\n\n<tool_call>{WRITE}</tool_call>")
    streamed = _stream(server, _body())
    argument_frames = [
        f["choices"][0]["delta"]["tool_calls"][0]["function"].get("arguments")
        for f in streamed
        if f.get("choices") and f["choices"][0]["delta"].get("tool_calls")
    ]
    assert argument_frames[0] == "{" and len(argument_frames) == 2
    _, calls, _, _, _ = _assemble(streamed)
    assert json.loads(calls[0]["arguments"])["path"] == "VENDORED.md"


def test_the_client_reads_the_arguments_while_the_generation_waits(tokenizer_dir):
    """Served for real (uvicorn, a free local port): the generation stops halfway
    through the call's value until the client has read part of it."""
    import httpx
    import uvicorn

    running = _Server(tokenizer_dir)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    http = uvicorn.Server(
        uvicorn.Config(running.app, host="127.0.0.1", port=port, log_level="warning")
    )
    threading.Thread(target=http.run, daemon=True).start()
    try:
        deadline = time.monotonic() + 30
        while not http.started:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        gate = threading.Event()
        tokens = running.script(
            f"<tool_call>{WRITE}</tool_call>", pause_at=len(WRITE) // 2, gate=gate
        )
        read_before_the_gate = ""
        with httpx.stream(
            "POST",
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json=_body(),
            timeout=60,
        ) as response:
            frames = []
            for line in response.iter_lines():
                frames += _frames([line])
                if not gate.is_set():
                    _, calls, _, _, _ = _assemble(frames)
                    arguments = calls.get(0, {}).get("arguments", "")
                    if len(arguments) > 300:
                        read_before_the_gate = arguments
                        # A status poll while the call is half written.
                        job = running.jobs[-1]
                        report = job.stream.report()
                        gate.set()
        assert not running.gate_timed_out
        assert len(tokens) > len(WRITE) // 2
        assert read_before_the_gate.startswith(
            '{"path": "VENDORED.md", "content": "# Vendored'
        )
        assert (
            report["parser_state"] == "tool"
            and report["tool_call"]["parameter"] == "content"
        )
        assert report["generated_chars"] > report["sent_chars"] > 300
        assert "<parameter=content>" in report["tail"]
        _, calls, finish, _, _ = _assemble(frames)
        assert (
            json.loads(calls[0]["arguments"])["path"] == "VENDORED.md"
            and finish == "tool_calls"
        )
    finally:
        http.should_exit = True
        running.close()


KICKOFF = call(
    "write",
    ("path", "/w/notes/kickoff.md"),
    ("content", "# Kickoff\n\n## Decisions\n- Confluence is out of scope for now\n"),
)
MKDIR = call("shell", ("command", "mkdir -p /w/notes"))


def test_a_call_written_again_word_for_word_ends_the_answer_unsent(server):
    """E2E #3e's shape: the same write + mkdir pair over and over."""
    pair = f"<tool_call>{KICKOFF}</tool_call>\n<tool_call>{MKDIR}</tool_call>\n"
    tokens = server.script(pair * 6)
    streamed = _stream(server, _body())
    _, calls, finish, _, errors = _assemble(streamed)
    assert errors == []
    assert [c["name"] for c in calls.values()] == ["write", "shell"]
    assert finish == "tool_calls"
    (stop,) = server.said.tags("RANK_TOOL_CALL_REPEATED")
    assert (stop["repeat_of"], stop["name"]) == (1, "write")
    status = server.client.get("/v1/status").json()
    assert status["last_engine_stop"]["reason"] == "tool_call_repeated"
    assert server.given[server.jobs[-1].id] < len(tokens) // 2


def test_a_call_that_differs_from_an_earlier_one_is_sent(server):
    again = KICKOFF.replace("out of scope for now", "out of scope until Q3")
    server.script(f"<tool_call>{KICKOFF}</tool_call>\n<tool_call>{again}</tool_call>")
    _, calls, finish, _, _ = _assemble(_stream(server, _body()))
    assert [json.loads(c["arguments"])["content"] for c in calls.values()] == [
        "# Kickoff\n\n## Decisions\n- Confluence is out of scope for now\n",
        "# Kickoff\n\n## Decisions\n- Confluence is out of scope until Q3\n",
    ]
    assert finish == "tool_calls"
    assert server.said.tags("RANK_TOOL_CALL_REPEATED") == []


PANEL = (
    "!\n\n</parameter>\n</function>\n!\n</parameter>\n</function>\n!\n</parameter>\n!\n</function>\n"
    "!\n</parameter>\n!\n</function>\n!\n</parameter>\n</function>\n!\n</parameter>\n</function>\n"
)
CYCLE = "!\n</function>\n"
RUNAWAY = PANEL + CYCLE * ((3251 - len(PANEL)) // len(CYCLE))
PASSING = (
    "Wrote notes/kickoff.md.\n- TBD\n- TBD\nDecisions, actions and open questions are in it; "
    "the two TBD lines are the owners Aoife has not named yet.\n"
)


def test_text_written_over_and_over_beside_the_calls_ends_the_answer(server):
    """E2E #3f's words: one write, then the checkpoint's `!` and closing tags."""
    tokens = server.script(f"<tool_call>{KICKOFF}</tool_call>{RUNAWAY}")
    content, calls, finish, _, _ = _assemble(_stream(server, _body()))
    assert json.loads(calls[0]["arguments"])["path"] == "/w/notes/kickoff.md"
    assert finish == "tool_calls"
    (stop,) = server.said.tags("RANK_TEXT_CYCLE")
    assert stop["copies"] >= 2 and "\n" in stop["unit"]
    assert len(content) < 200, content
    assert server.given[server.jobs[-1].id] < len(tokens) // 4
    assert (
        server.client.get("/v1/status").json()["last_engine_stop"]["reason"]
        == "text_cycle"
    )


def test_a_line_said_twice_in_passing_runs_to_its_end(server):
    server.script(PASSING)
    content, calls, finish, _, _ = _assemble(_stream(server, _body()))
    assert content == PASSING and calls == {} and finish == "stop"
    assert server.said.tags("RANK_TEXT_CYCLE") == []


def _q133_pieces():
    return Q133["pieces"], Q133["tools"]


def test_the_recorded_flash_stream_still_refuses_an_undeclared_name(server):
    """goose Q-133's recording: the call names `bash`, the request declared
    `shell`.  It stays text, and the final choice names the refusal."""
    pieces, tools = _q133_pieces()
    answer = "".join(pieces)
    after_think = answer.split("</think>", 1)[1]
    server.script(after_think)
    content, calls, _, refused, _ = _assemble(_stream(server, _body(tools=tools)))
    assert calls == {}
    assert "<function=bash>" in content
    assert refused == [{"name": "bash"}]


def test_the_recorded_flash_stream_naming_shell_is_one_streamed_call(server):
    pieces, tools = _q133_pieces()
    after_think = (
        "".join(pieces)
        .split("</think>", 1)[1]
        .replace("<function=bash>", "<function=shell>")
    )
    server.script(after_think)
    streamed = _stream(server, _body(tools=tools))
    content, calls, finish, refused, _ = _assemble(streamed)
    assert [c["name"] for c in calls.values()] == ["shell"]
    assert json.loads(calls[0]["arguments"])["command"].startswith(
        "cd /Users/mihaiperdum"
    )
    assert "<function=" not in content and refused is None and finish == "tool_calls"


# ---------------------------------------------------------------------------
# goose Q-177 / Q-164: what the split cannot honour is a named 400
# ---------------------------------------------------------------------------

UNSUPPORTED, INVALID = "unsupported_parameter", "invalid_value"
# case: (fields, words the message holds, the field it names, its code)
REFUSALS = {
    "n": ({"n": 2}, "n=2", "n", UNSUPPORTED),
    "logprobs": ({"logprobs": True}, "log-probabilities", "logprobs", UNSUPPORTED),
    "top_logprobs": (
        {"top_logprobs": 12},
        "log-probabilities",
        "top_logprobs",
        UNSUPPORTED,
    ),
    "logit_bias": (
        {"logit_bias": {"5": 10}},
        "per-token bias",
        "logit_bias",
        UNSUPPORTED,
    ),
    "seed": ({"seed": 7}, "seed=7", "seed", UNSUPPORTED),
    "response_format": (
        {"response_format": {"type": "json_object"}},
        "response_format type 'json_object'",
        "response_format",
        UNSUPPORTED,
    ),
    "stop_not_text": ({"stop": [123]}, "stop[0] must be a string", "stop", INVALID),
    "stop_empty": ({"stop": ""}, "stop[0] is empty", "stop", INVALID),
    "max_tokens_zero": ({"max_tokens": 0}, "max_tokens must be", "max_tokens", INVALID),
    "max_tokens_text": (
        {"max_tokens": "5"},
        "max_tokens must be",
        "max_tokens",
        INVALID,
    ),
    "max_tokens_bool": (
        {"max_tokens": True},
        "max_tokens must be",
        "max_tokens",
        INVALID,
    ),
    "no_messages": ({"messages": []}, "messages must be", "messages", INVALID),
    "tools_shape": (
        {"tools": [{"type": "function"}]},
        "tools must be",
        "tools",
        INVALID,
    ),
    "stream_options": (
        {"stream_options": "usage"},
        "stream_options must be",
        "stream_options",
        INVALID,
    ),
    "top_k_past_vocab": ({"top_k": 10_000}, "top_k 10000 (request)", "top_k", INVALID),
    "top_k_negative": ({"top_k": -1}, "top_k -1 (request)", "top_k", INVALID),
    "min_p": ({"min_p": 1.5}, "min_p 1.5 (request)", "min_p", INVALID),
    "top_p": ({"top_p": 2.0}, "top_p 2.0 (request)", "top_p", INVALID),
    "temperature": (
        {"temperature": -0.5},
        "temperature -0.5 (request)",
        "temperature",
        INVALID,
    ),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_a_field_the_split_cannot_honour_is_a_named_400(server, case):
    extra, words, param, code = REFUSALS[case]
    response = server.client.post("/v1/chat/completions", json={**_body(), **extra})
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert words in error["message"], error
    assert (error["param"], error["code"]) == (param, code), error
    assert server.state.jobs.qsize() == 0 and server.jobs == []


def test_the_same_fields_within_range_are_served(server):
    server.script("ok")
    body = {
        **_body(),
        "stream": False,
        "n": 1,
        "logprobs": False,
        "top_k": 20,
        "min_p": 0.05,
        "top_p": 0.95,
        "temperature": 0.7,
        "stop": ["never"],
        "max_tokens": 5,
        "response_format": {"type": "text"},
    }
    response = server.client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    row = server.jobs[-1].row
    assert (row.top_k, row.max_tokens) == (20, 5)


def test_a_failure_inside_the_stream_is_said_on_it_not_a_dropped_connection(
    server, monkeypatch
):
    real = ps.StreamRelay.take

    def broken(self, token, piece):
        if "#" in piece:
            raise KeyError("the relay met a piece it cannot read")
        return real(self, token, piece)

    monkeypatch.setattr(ps.StreamRelay, "take", broken)
    server.script("fine so far, then # and more")
    frames = _stream(server, _body())
    _, _, _, _, errors = _assemble(frames)
    assert errors and "KeyError" in errors[0]["message"]
    (failed,) = server.said.tags("RANK_STREAM_FAILED")
    assert "fine so far" in failed["tail"]
    assert server.jobs[-1].cancelled


def test_building_the_app_encodes_nothing(tokenizer_dir):
    """The relay's markers are read by the first streamed request: goose's
    render test replaces ``encode`` to capture the prompt, and an app built
    over such a tokenizer must still build."""
    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(tokenizer_dir)

    def refuse(*args, **kwargs):
        raise AssertionError("the app encoded while it was built")

    tokenizer.encode = refuse
    state = serve._State(served=SERVED, context=4096, max_batch=1)
    serve._build_app(state, tokenizer, set(tokenizer.eos_token_ids))
