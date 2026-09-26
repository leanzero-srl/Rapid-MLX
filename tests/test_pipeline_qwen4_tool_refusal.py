# SPDX-License-Identifier: Apache-2.0
"""goose Q-133: a refused tool call must not pass for the model's answer.

Measured on goose 3.0.49, 2026-09-26: Flash on the pipeline split ended a
turn by calling ``bash`` while the request declared ``shell``. The
qwen3_coder_xml parser refuses an undeclared name (never executable), so the
whole call streamed as content; goose showed the XML as the finished reply,
the command never ran, and the turn ended. Earlier calls in the same turn
named ``shell`` and parsed — the failure was the name, not the framing or the
chunking.

The refusal stands. These tests pin that it is now SAID: the post-processor
names every refused call, and the split's final choice carries
``refused_tool_calls``. Parser and state machine only — no model, no ranks.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")

from rapid_mlx.config.server_config import ServerConfig  # noqa: E402
from rapid_mlx.distributed.pipeline_qwen4_serve import _refusal_fields  # noqa: E402
from rapid_mlx.engine.base import GenerationOutput  # noqa: E402
from rapid_mlx.service.postprocessor import StreamingPostProcessor  # noqa: E402
from rapid_mlx.tool_parsers import ToolParserManager  # noqa: E402

_FIXTURE = json.loads(
    (
        Path(__file__).parent / "fixtures" / "q133_split_flash_undeclared_bash.json"
    ).read_text()
)
_PIECES: list[str] = _FIXTURE["pieces"]
_TOOLS: list[dict] = _FIXTURE["tools"]
_COMMAND = (
    "cd /Users/mihaiperdum/goose-builds/quality/RU-2026-09-26-5-split-pipeline-flash/work"
    " && cp /tmp/VENDORED.md ./VENDORED.md && wc -l VENDORED.md && grep -c '| `' VENDORED.md"
)


def _replay(pieces, tools=_TOOLS, tool_choice=None):
    """Feed ``pieces`` through the split's post-processor exactly as serve does."""
    cfg = ServerConfig()
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "qwen3_coder_xml"
    cfg.reasoning_parser_name = "deepseek_r1"
    cfg.engine = type("_TokenizerHolder", (), {"tokenizer": None, "_tokenizer": None})()
    request = {"model": "m", "messages": [], "tools": tools, "stream": True}
    if tool_choice is not None:
        request["tool_choice"] = tool_choice
    processor = StreamingPostProcessor(
        cfg, tools_requested=bool(tools), enable_thinking=None, request=request
    )
    processor.reset()
    # The Flash template opens <think> in the generation prompt (serve sets this).
    processor.reasoning_parser._prompt_primed_thinking = True

    content, reasoning, calls = [], [], []

    def take(events):
        for event in events:
            content.append(event.content or "")
            reasoning.append(event.reasoning or "")
            calls.extend(event.tool_calls or [])

    text = ""
    for count, piece in enumerate(pieces, start=1):
        text += piece
        take(
            processor.process_chunk(
                GenerationOutput(
                    text=text,
                    new_text=piece,
                    prompt_tokens=1,
                    completion_tokens=count,
                    finished=False,
                    finish_reason=None,
                )
            )
        )
    terminal = processor.process_chunk(
        GenerationOutput(
            text="",
            new_text="",
            prompt_tokens=1,
            completion_tokens=len(pieces),
            finished=True,
            finish_reason="stop",
        )
    )
    terminal.extend(processor.finalize())
    take(terminal)
    return "".join(content), "".join(reasoning), calls, processor


def _split_at_think_close(pieces):
    close = pieces.index("</think>")
    return pieces[: close + 1], pieces[close + 1 :]


def test_the_recorded_split_stream_refuses_bash_and_says_so():
    content, reasoning, calls, processor = _replay(_PIECES)

    assert calls == [], f"an undeclared name became executable: {calls}"
    assert content.strip().startswith("<tool_call>\n<function=bash>"), content
    assert content.count("<function=bash>") == 1, content
    assert "place the file and commit" in reasoning
    assert processor.refused_tool_calls() == [{"name": "bash"}]
    assert _refusal_fields(processor, "m") == {"refused_tool_calls": [{"name": "bash"}]}


@pytest.mark.parametrize("chunking", ["characters", "one_content_chunk"])
def test_chunk_boundaries_do_not_change_the_verdict(chunking):
    thinking, answer = _split_at_think_close(_PIECES)
    joined = "".join(answer)
    rechunked = list(joined) if chunking == "characters" else [joined]

    content, _, calls, processor = _replay(thinking + rechunked)

    assert calls == []
    assert "<function=bash>" in content
    assert processor.refused_tool_calls() == [{"name": "bash"}]


def test_the_same_stream_naming_shell_is_one_call_and_no_refusal():
    """Positive control: only the name differs, and the call parses."""
    name_at = _PIECES.index("=b")
    assert _PIECES[name_at - 1 : name_at + 2] == ["function", "=b", "ash"]
    pieces = [*_PIECES[:name_at], "=", "shell", *_PIECES[name_at + 2 :]]
    assert "".join(pieces).count("<function=shell>") == 1

    content, _, calls, processor = _replay(pieces)

    names = [c["function"]["name"] for c in calls if c.get("function", {}).get("name")]
    arguments = "".join(c.get("function", {}).get("arguments") or "" for c in calls)
    assert names == ["shell"]
    assert json.loads(arguments) == {"command": _COMMAND}
    assert "<function=" not in content
    assert processor.refused_tool_calls() == []
    assert _refusal_fields(processor, "m") == {}


def test_a_call_promoted_out_of_reasoning_is_reported_like_any_content():
    """The think parser promotes a framed call out of reasoning into content.

    What is reported is exactly what the content carried: the promoted block
    reached the client as text, so its refusal is named too.
    """
    thinking, _ = _split_at_think_close(_PIECES)
    scratch = "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>\n"
    pieces = thinking[:-1] + [scratch, "</think>", "\n\nDone."]

    content, _, calls, processor = _replay(pieces)

    assert calls == []
    assert "<function=bash>" in content
    assert processor.refused_tool_calls() == [{"name": "bash"}]


def test_prose_about_the_wire_in_content_is_not_reported():
    thinking, _ = _split_at_think_close(_PIECES)
    pieces = thinking + ["\n\nA call looks like `<function=bash></function>`."]

    content, _, calls, processor = _replay(pieces)

    assert calls == []
    assert "<function=bash>" in content
    assert processor.refused_tool_calls() == []


def test_the_parser_reports_nothing_a_request_did_not_ask_for():
    parser = ToolParserManager.get_tool_parser("qwen3_coder_xml")(None)
    block = "<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"
    request = {"tools": _TOOLS}

    assert parser.refused_tool_calls(block, request) == [{"name": "bash"}]
    # tool_choice none, and no tools at all: nothing was offered, nothing refused.
    assert parser.refused_tool_calls(block, {**request, "tool_choice": "none"}) == []
    assert parser.refused_tool_calls(block, {"tools": []}) == []
    assert parser.refused_tool_calls(block, None) == []
    # Prose that names the wire without its framing is not a call.
    assert (
        parser.refused_tool_calls("use <function=bash></function> like so", request)
        == []
    )
    # A declared name is admitted, so it is not refused.
    admitted = block.replace("bash", "shell")
    assert parser.refused_tool_calls(admitted, request) == []
    assert parser.extract_tool_calls(admitted, request).tools_called
