# SPDX-License-Identifier: Apache-2.0
"""IBM Granite 4.2 (dense) serves with its own wire, from a local folder.

Live defect, 2026-10-19 (Mac Studio, goose app, engine b20ec1b7, local
folder ``lmstudio-community/granite-4.2-30b-MLX-8bit``). Two faults:

1. Auto-detect matched the NAME ``granite[-_]?4`` and gave the folder the
   Granite 4.0-H profile: ``tool_call_parser=hermes, reasoning_parser=None,
   is_hybrid=True``. Granite 4.2 is a different model: ``GraniteForCausalLM``
   (``model_type: granite``, plain attention), a template that primes
   ``<think>\\n`` behind ``enable_thinking`` (default on) and instructs the
   qwen3-coder XML call wire. The thought streamed as the answer and a text
   reply ran 10k+ steps without a stop.
2. The documented override, ``--tool-call-parser qwen3_coder_xml``, was
   silently dropped by ``rapid-mlx serve`` unless ``--enable-auto-tool-choice``
   was ALSO passed (only an auto-detected parser switched it on). With no
   parser bound, a non-streaming call was still found by the generic fallback
   scan, but the STREAMED call reached goose as plain content with
   ``finish_reason=stop`` and zero ``tool_calls`` deltas.

The fixtures are the checkpoint's real ``config.json``,
``chat_template.jinja`` (md5 21ecc7833299bf0786fc44117fec262c) and
``tokenizer_config.json``.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from unittest import mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx import model_auto_config as auto_config_mod
from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.model_auto_config import detect_model_config
from rapid_mlx.routes.chat import router as chat_router
from rapid_mlx.tool_parsers import ToolParserManager

FIXTURE = Path(__file__).parent / "fixtures" / "granite-4.2-30b"
FOLDER = "lmstudio-community/granite-4.2-30b-MLX-8bit"

# The exact streamed content the engine sent for a ``write`` call
# (b20ec1b7, explicit --tool-call-parser qwen3_coder_xml, no
# --enable-auto-tool-choice).
GRANITE_WRITE = (
    "\n<tool_call>\n<function=write>\n<parameter=path>\nsrc/a.js\n</parameter>\n"
    "<parameter=content>\nexport const a = 1;\n</parameter>\n</function>\n"
    "</tool_call>\n"
)


@pytest.fixture(autouse=True)
def _fresh_resolution_log():
    auto_config_mod._reset_resolution_log_cache()
    yield
    auto_config_mod._reset_resolution_log_cache()


def _checkpoint(tmp_path: Path, name: str = FOLDER, **config_changes) -> str:
    folder = tmp_path / name
    folder.mkdir(parents=True)
    for f in ("chat_template.jinja", "tokenizer_config.json"):
        shutil.copy(FIXTURE / f, folder / f)
    config = json.loads((FIXTURE / "config.json").read_text())
    config.update(config_changes)
    (folder / "config.json").write_text(json.dumps(config))
    return str(folder)


# ---------------------------------------------------------------------------
# 1. The profile comes from the checkpoint, not the 4.0-H name rule.
# ---------------------------------------------------------------------------


def test_the_real_granite42_folder_gets_its_own_profile(tmp_path):
    cfg = detect_model_config(_checkpoint(tmp_path))

    assert cfg is not None
    assert cfg.tool_call_parser == "qwen3_coder_xml"
    assert cfg.reasoning_parser == "qwen3"
    assert cfg.is_hybrid is False
    assert cfg.is_hybrid_explicit is True
    assert cfg.supports_spec_decode is True


def test_it_matches_ibms_own_granite42_alias(tmp_path):
    """The hub alias for IBM's own conversion names the same profile."""
    alias = detect_model_config("ibm-granite/granite-4.2-30b-q8-mlx")
    local = detect_model_config(_checkpoint(tmp_path))

    assert alias is not None and local is not None
    for field in ("tool_call_parser", "reasoning_parser", "is_hybrid"):
        assert getattr(local, field) == getattr(alias, field), field


def test_a_repack_whose_name_says_nothing_still_resolves(tmp_path):
    cfg = detect_model_config(_checkpoint(tmp_path, "models/my-repack"))

    assert cfg is not None
    assert (cfg.tool_call_parser, cfg.reasoning_parser, cfg.is_hybrid) == (
        "qwen3_coder_xml",
        "qwen3",
        False,
    )


def test_granite_4_0_h_keeps_the_hybrid_profile(tmp_path):
    """Negative control: a granitemoehybrid snapshot under the same name rule."""
    path = _checkpoint(
        tmp_path,
        "mlx-community/granite-4.0-h-tiny-4bit",
        model_type="granitemoehybrid",
        architectures=["GraniteMoeHybridForCausalLM"],
        layer_types=["mamba", "mamba", "attention"],
    )
    # 4.0-H templates carry the hermes JSON wire and no <think>.
    Path(path, "chat_template.jinja").write_text(
        "{%- for m in messages %}{{ m.content }}{%- endfor %}"
        "<tool_call>{json}</tool_call>"
    )
    Path(path, "tokenizer_config.json").unlink()

    cfg = detect_model_config(path)

    assert cfg is not None
    assert cfg.tool_call_parser == "hermes"
    assert cfg.reasoning_parser is None
    assert cfg.is_hybrid is True
    assert cfg.supports_spec_decode is False


def test_a_hub_name_with_no_local_metadata_keeps_the_hybrid_profile():
    cfg = detect_model_config("mlx-community/granite-4.0-h-small-4bit")

    assert cfg is not None
    assert (cfg.tool_call_parser, cfg.reasoning_parser, cfg.is_hybrid) == (
        "hermes",
        None,
        True,
    )


# ---------------------------------------------------------------------------
# 2. The chosen parsers read what the real template renders.
# ---------------------------------------------------------------------------


def _render(**kwargs) -> str:
    import jinja2

    env = jinja2.Environment(
        loader=jinja2.BaseLoader(),
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"],
    )
    env.filters["tojson"] = lambda v, ensure_ascii=True, **_: json.dumps(
        v, ensure_ascii=ensure_ascii
    )
    template = env.from_string((FIXTURE / "chat_template.jinja").read_text())
    return template.render(**kwargs)


def test_the_template_primes_think_and_renders_the_qwen3_coder_wire():
    prompt = _render(
        messages=[{"role": "user", "content": "go"}], add_generation_prompt=True
    )
    assert prompt.endswith("<|im_start|>assistant\n<think>\n")
    off = _render(
        messages=[{"role": "user", "content": "go"}],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert off.endswith("<|im_start|>assistant\n<think></think>")

    history = _render(
        messages=[
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "write",
                            "arguments": {"path": "src/a.js", "content": "x"},
                        }
                    }
                ],
            },
        ]
    )
    rendered_call = history[
        history.index("<tool_call>") : history.index(
            "<|im_end|>", history.index("<tool_call>")
        )
    ]
    parser = ToolParserManager.get_tool_parser("qwen3_coder_xml")(None)
    result = parser.extract_tool_calls(rendered_call, _write_request())
    assert result.tools_called
    assert [c["name"] for c in result.tool_calls] == ["write"]
    assert json.loads(result.tool_calls[0]["arguments"]) == {
        "path": "src/a.js",
        "content": "x",
    }


# ---------------------------------------------------------------------------
# 3. The streamed call: tool_calls deltas, never content.
# ---------------------------------------------------------------------------


def _write_request() -> dict:
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "description": "write a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                },
            }
        ]
    }


class _Engine:
    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = None

    def __init__(self, deltas: list[str]):
        self._deltas = deltas

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def stream_chat(self, messages, **kwargs):
        text = ""
        for i, delta in enumerate(self._deltas):
            text += delta
            last = i == len(self._deltas) - 1
            yield GenerationOutput(
                text=text,
                new_text=delta,
                prompt_tokens=4,
                completion_tokens=i + 1,
                finished=last,
                finish_reason="stop" if last else None,
                channel=None,
            )


def _deltas(text: str, size: int) -> list[str]:
    # <tool_call> / </tool_call> are single tokens in Granite's vocabulary.
    out, i = [], 0
    while i < len(text):
        for tok in ("<tool_call>", "</tool_call>"):
            if text.startswith(tok, i):
                out.append(tok)
                i += len(tok)
                break
        else:
            nxt = min(
                [
                    p
                    for p in (text.find("<tool_call>", i), text.find("</tool_call>", i))
                    if p != -1
                ]
                or [len(text)]
            )
            end = min(i + size, nxt)
            out.append(text[i:end])
            i = end
    return out


def _stream(size: int, *, parser: str | None) -> dict:
    cfg = reset_config()
    cfg.engine = _Engine(_deltas(GRANITE_WRITE, size))
    cfg.model_name = "granite"
    cfg.model_registry = None
    cfg.enable_auto_tool_choice = parser is not None
    cfg.tool_call_parser = parser
    cfg.reasoning_parser_name = "qwen3"
    app = FastAPI()
    app.include_router(chat_router)
    resp = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "granite",
            "stream": True,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "write src/a.js"}],
            **_write_request(),
        },
    )
    assert resp.status_code == 200, resp.text
    content, finish, calls = "", [], {}
    for line in resp.text.splitlines():
        if not line.startswith("data:") or "[DONE]" in line:
            continue
        for choice in json.loads(line[5:]).get("choices", []):
            delta = choice.get("delta") or {}
            content += delta.get("content") or ""
            for tc in delta.get("tool_calls") or []:
                call = calls.setdefault(tc["index"], {"name": "", "arguments": ""})
                fn = tc.get("function") or {}
                call["name"] += fn.get("name") or ""
                call["arguments"] += fn.get("arguments") or ""
            if choice.get("finish_reason"):
                finish.append(choice["finish_reason"])
    reset_config()
    return {
        "content": content,
        "calls": [calls[i] for i in sorted(calls)],
        "finish": finish,
    }


@pytest.mark.parametrize("size", [1, 4, 100000])
def test_the_granite_write_call_streams_as_a_tool_call(size):
    """The leading newline before <tool_call> is not the fault: with the
    parser bound, the exact live wire streams one write call."""
    out = _stream(size, parser="qwen3_coder_xml")

    assert [c["name"] for c in out["calls"]] == ["write"], out
    assert json.loads(out["calls"][0]["arguments"]) == {
        "path": "src/a.js",
        "content": "export const a = 1;",
    }
    assert out["finish"][-1] == "tool_calls", out
    assert "<tool_call>" not in out["content"]
    assert "<function=" not in out["content"]


def test_with_no_parser_bound_the_call_is_content():
    """The live symptom, for the record: what an unbound parser streams."""
    out = _stream(100000, parser=None)

    assert out["calls"] == []
    assert "<function=write>" in out["content"]


# ---------------------------------------------------------------------------
# 4. ``rapid-mlx serve``: an explicit --tool-call-parser binds the parser.
# ---------------------------------------------------------------------------


class _StopError(Exception):
    pass


def _serve_until_scheduler(argv: list[str], monkeypatch) -> dict:
    pytest.importorskip("mlx")
    import rapid_mlx.cli as cli
    import rapid_mlx.engine_core  # noqa: F401  (bind the real class first)
    from rapid_mlx import server

    for name in (
        "_tool_call_parser",
        "_enable_auto_tool_choice",
        "_reasoning_parser",
        "_reasoning_parser_name",
        "_enable_tool_logits_bias",
    ):
        monkeypatch.setattr(server, name, getattr(server, name, None), raising=False)

    def _fake_scheduler_config(*args, **kwargs):
        raise _StopError

    with (
        mock.patch.object(cli, "_check_disk_space", lambda *a, **k: None),
        mock.patch.object(cli, "_check_memory_capacity", lambda *a, **k: None),
        mock.patch.object(cli, "_ensure_model_downloaded", lambda *a, **k: None),
        mock.patch.object(
            cli, "_gather_kv_cache_dtype_inputs", lambda *a, **k: ({}, None)
        ),
        mock.patch(
            "rapid_mlx._version_check.prompt_upgrade_if_available",
            return_value=False,
        ),
        mock.patch(
            "rapid_mlx.utils.tokenizer.load_model_with_fallback",
            return_value=(object(), object()),
        ),
        mock.patch("rapid_mlx.scheduler.SchedulerConfig", _fake_scheduler_config),
        # Another test may already have started server.app; adding a
        # middleware then raises. No middleware is under test here.
        mock.patch.object(server.app, "add_middleware", lambda *a, **k: None),
        mock.patch.object(sys, "argv", ["rapid-mlx", *argv]),
        mock.patch.object(sys.stdin, "isatty", return_value=False),
        pytest.raises(_StopError),
    ):
        cli.main()
    return {
        "tool_call_parser": server._tool_call_parser,
        "enable_auto_tool_choice": server._enable_auto_tool_choice,
        "reasoning_parser_name": getattr(server, "_reasoning_parser_name", None),
    }


def test_an_explicit_tool_call_parser_is_bound(tmp_path, monkeypatch):
    """The documented override (no --enable-auto-tool-choice) must bind."""
    path = _checkpoint(tmp_path)
    got = _serve_until_scheduler(
        ["serve", path, "--port", "8197", "--tool-call-parser", "qwen3_coder_xml"],
        monkeypatch,
    )
    assert got["tool_call_parser"] == "qwen3_coder_xml"
    assert got["enable_auto_tool_choice"] is True


def test_the_app_argv_alone_binds_granites_parsers(tmp_path, monkeypatch):
    """The goose app passes no parser flags: auto-detect alone must do it."""
    path = _checkpoint(tmp_path)
    got = _serve_until_scheduler(
        [
            "serve",
            path,
            "--port",
            "8197",
            "--enable-prefix-cache",
            "--hybrid-cache-entries",
            "24",
            "--max-concurrent-requests",
            "8",
        ],
        monkeypatch,
    )
    assert got["tool_call_parser"] == "qwen3_coder_xml"
    assert got["enable_auto_tool_choice"] is True
    assert got["reasoning_parser_name"] == "qwen3"


def test_no_tool_call_parser_still_opts_out(tmp_path, monkeypatch):
    path = _checkpoint(tmp_path)
    got = _serve_until_scheduler(
        ["serve", path, "--port", "8197", "--no-tool-call-parser"], monkeypatch
    )
    assert got["tool_call_parser"] is None
    assert got["enable_auto_tool_choice"] is False
