# SPDX-License-Identifier: Apache-2.0
"""A local folder whose template renders the qwen3-coder XML wire gets parsers.

Live defect, 2026-10-19 05:3x (RUN-ISSUES I-27, laptop, goose app):
abenzerps/Nex-N2.5-mini-MLX-8bit (``qwen3_5_moe``) booted with
``tool_call_parser=None, reasoning_parser=None``. Its template writes the
``<tool_call>\\n<function=..>\\n<parameter=..>`` wire, but entirely in
``{{- '...' }}`` print statements: the structural detector reads only
template-text literals, saw none, and no alias names the folder. Explicit
``--tool-call-parser qwen3_coder_xml --reasoning-parser qwen3`` flags made
calls parse, but the thought still streamed as content: the template primes
``<think>`` on ``reasoning_effort``, not ``enable_thinking``, so when tools
auto-disable thinking the model still thinks while a qwen3 parser treats the
reply as content.

Rule now (metadata fallback, every family no other rule claims): a template
that writes the wire gets ``qwen3_coder_xml``; a generation prompt that stays
primed with ``enable_thinking=False`` gets ``deepseek_r1_distill`` (the LFM2.5
contract), one primed only behind the switch gets ``qwen3``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rapid_mlx.config import reset_config
from rapid_mlx.engine.base import GenerationOutput
from rapid_mlx.model_auto_config import detect_model_config
from rapid_mlx.routes.chat import router as chat_router

# Nex-N2.5-mini's shape: print statements only, the wire in history tool calls,
# thinking primed on ``reasoning_effort`` (``none`` closes it), no enable_thinking.
WIRE_TOOLS = (
    "{%- if tools %}{{- '<|im_start|>system\\n# Tools\\n<tools>' }}"
    "{%- for tool in tools %}{{- '\\n' + (tool | tojson) }}{%- endfor %}"
    "{{- '\\n</tools>\\nCall a function in this format: <tool_call>\\n"
    "<function=example>\\n<parameter=p>\\nvalue\\n</parameter>\\n</function>\\n"
    "</tool_call><|im_end|>\\n' }}{%- endif %}"
)
HISTORY = (
    "{%- for message in messages %}"
    "{%- if message.role == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' + (message.content or '') }}"
    "{%- for tc in message.tool_calls or [] %}"
    "{{- '<tool_call>\\n<function=' + tc.function.name + '>\\n' }}"
    "{%- for k, v in tc.function.arguments | items %}"
    "{{- '<parameter=' + k + '>\\n' + (v | string) + '\\n</parameter>\\n' }}"
    "{%- endfor %}{{- '</function>\\n</tool_call>' }}{%- endfor %}"
    "{{- '<|im_end|>\\n' }}"
    "{%- else %}{{- '<|im_start|>' + message.role + '\\n' + message.content + "
    "'<|im_end|>\\n' }}{%- endif %}{%- endfor %}"
)
PRIMED_ON_EFFORT = (
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
    "{%- if reasoning_effort == 'none' %}{{- '<think>\\n\\n</think>\\n\\n' }}"
    "{%- else %}{{- '<think>' }}{%- endif %}{%- endif %}"
)
PRIMED_ON_SWITCH = (
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}"
    "{{- '<think>\\n\\n</think>\\n\\n' }}{%- else %}{{- '<think>\\n' }}"
    "{%- endif %}{%- endif %}"
)
NEX_TEMPLATE = WIRE_TOOLS + HISTORY + PRIMED_ON_EFFORT


@pytest.fixture
def tmp_path(tmp_path_factory) -> Path:
    # pytest names ``tmp_path`` after the test, and a test name holding a
    # family word (``qwen3_5_moe``) would let the NAME fallback answer first.
    # The live folder was ``~/.goose/models/abenzerps/Nex-N2.5-mini-MLX-8bit``.
    return tmp_path_factory.mktemp("models")


def _folder(tmp_path: Path, model_type: str, template: str, name: str) -> str:
    folder = tmp_path / "someone" / name
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "num_experts": 256,
                "text_config": {"model_type": f"{model_type}_text"},
            }
        )
    )
    (folder / "chat_template.jinja").write_text(template)
    (folder / "tokenizer_config.json").write_text(json.dumps({}))
    return str(folder)


def test_a_qwen3_5_moe_folder_with_the_wire_gets_both_parsers(tmp_path):
    profile = detect_model_config(
        _folder(tmp_path, "qwen3_5_moe", NEX_TEMPLATE, "Nex-mini-MLX-8bit")
    )
    assert profile is not None
    assert profile.tool_call_parser == "qwen3_coder_xml"
    assert profile.reasoning_parser == "deepseek_r1_distill"
    assert profile.is_hybrid is True and profile.is_moe is True


def test_the_rule_follows_the_template_not_the_family(tmp_path):
    profile = detect_model_config(
        _folder(tmp_path, "brand_new_arch", NEX_TEMPLATE, "Fresh-Arch-4bit")
    )
    assert profile is not None
    assert profile.tool_call_parser == "qwen3_coder_xml"
    assert profile.reasoning_parser == "deepseek_r1_distill"


def test_thinking_behind_enable_thinking_gets_qwen3(tmp_path):
    template = WIRE_TOOLS + HISTORY + PRIMED_ON_SWITCH
    profile = detect_model_config(
        _folder(tmp_path, "qwen3_5_moe", template, "Switch-mini-4bit")
    )
    assert profile is not None
    assert profile.tool_call_parser == "qwen3_coder_xml"
    assert profile.reasoning_parser == "qwen3"


def test_no_wire_no_parser(tmp_path):
    template = HISTORY.replace("<tool_call>", "[call]") + PRIMED_ON_SWITCH
    profile = detect_model_config(
        _folder(tmp_path, "brand_new_arch", template, "Plain-4bit")
    )
    assert profile is None or profile.tool_call_parser is None


class _Tokenizer:
    chat_template = NEX_TEMPLATE


class _Engine:
    preserve_native_tool_format = False
    is_mllm = False
    supports_guided_generation = False
    tokenizer = _Tokenizer()

    def __init__(self, text: str):
        self.text = text

    def build_prompt(self, messages, tools=None, enable_thinking=None):
        return "PROMPT"

    async def stream_chat(self, messages, **kwargs):
        # Special tokens arrive whole (one token each on this family); the
        # rest in 7-character pieces.
        pieces: list[str] = []
        rest = self.text
        specials = ("<tool_call>", "</tool_call>", "</think>")
        while rest:
            hit = next((t for t in specials if rest.startswith(t)), None)
            if hit:
                pieces.append(hit)
                rest = rest[len(hit) :]
                continue
            stop = min([rest.find(t) for t in specials if rest.find(t) > 0] + [7])
            pieces.append(rest[:stop])
            rest = rest[stop:]
        so_far = ""
        for i, piece in enumerate(pieces):
            so_far += piece
            last = i == len(pieces) - 1
            yield GenerationOutput(
                text=so_far,
                new_text=piece,
                prompt_tokens=4,
                completion_tokens=i + 1,
                finished=last,
                finish_reason="stop" if last else None,
            )


REPLY = (
    "We need to create the file first.</think>\n\nCreating it.\n"
    "<tool_call>\n<function=write>\n<parameter=path>\na.txt\n</parameter>\n"
    "<parameter=content>\nhi\n</parameter>\n</function>\n</tool_call>"
)


@pytest.mark.parametrize("chat_template_kwargs", [None, {"enable_thinking": False}])
def test_the_detected_parsers_split_a_nex_reply(chat_template_kwargs):
    """The same model output through the route with the detected parsers:
    the thought is reasoning, the call is a call, the prose is content —
    with no thinking flag (the released goose) and with thinking switched off
    (tools auto-disable it)."""
    cfg = reset_config()
    cfg.engine = _Engine(REPLY)
    cfg.model_name = "test-model"
    cfg.model_registry = None
    cfg.enable_auto_tool_choice = True
    cfg.tool_call_parser = "qwen3_coder_xml"
    cfg.reasoning_parser_name = "deepseek_r1_distill"
    app = FastAPI()
    app.include_router(chat_router)
    body = {
        "model": "test-model",
        "stream": True,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "go"}],
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
                    },
                },
            }
        ],
    }
    if chat_template_kwargs is not None:
        body["chat_template_kwargs"] = chat_template_kwargs
    resp = TestClient(app).post("/v1/chat/completions", json=body)
    assert resp.status_code == 200, resp.text
    content = reasoning = ""
    calls: dict[int, dict] = {}
    for line in resp.text.splitlines():
        if not line.startswith("data:") or "[DONE]" in line:
            continue
        for choice in json.loads(line[5:]).get("choices", []):
            delta = choice.get("delta") or {}
            content += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or ""
            for tc in delta.get("tool_calls") or []:
                call = calls.setdefault(tc["index"], {"name": "", "arguments": ""})
                call["name"] += (tc.get("function") or {}).get("name") or ""
                call["arguments"] += (tc.get("function") or {}).get("arguments") or ""
    assert reasoning.strip() == "We need to create the file first."
    assert "We need" not in content
    assert content.strip() == "Creating it."
    assert [c["name"] for c in calls.values()] == ["write"]
    assert json.loads(calls[0]["arguments"]) == {"path": "a.txt", "content": "hi"}
