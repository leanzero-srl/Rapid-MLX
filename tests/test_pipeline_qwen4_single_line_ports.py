# SPDX-License-Identifier: Apache-2.0
"""The single engine's fixes the pipeline split had missed (goose Q-144).

This line branched from the single engine at v0.14.3-lz.2; two fixes made
after it touch code the pipeline server runs in its own shape:

* lz.7 (goose Q-85): a tool request's decode is held to the XML tool-call
  skeleton, so ``</parameter>`` residue never becomes an argument's payload.
  Here the last rank samples, so the guard lives there, armed per row by the
  plan's ``tools`` flag.
* lz.8 (goose Q-110): a prefix-cache entry owns exactly the bytes it is
  charged for.  Here that is ``_PrefixStore.put``: a stored VIEW (the
  QSA index cache's raw ring is a slice of the chunk's raw keys) kept its
  whole source buffer alive beside the plan's budget.

In process, CPU-safe, one rank: nothing launches ``serve`` or a rank.
"""

import copy
import gc
import json

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
pytest.importorskip("fastapi")
pytestmark = pytest.mark.requires_mlx

from rapid_mlx.distributed import pipeline_qwen4 as pipe  # noqa: E402
from rapid_mlx.distributed import pipeline_qwen4_serve as serve  # noqa: E402
from rapid_mlx.models.qwen4_exp import Model, ModelArgs  # noqa: E402
from rapid_mlx.tool_parsers.qwen3coder_tool_parser import (  # noqa: E402
    Qwen3CoderToolParser,
)
from rapid_mlx.xml_tool_close_guard import OPT_OUT_ENV  # noqa: E402

from .test_pipeline_qwen4 import _tiny_text_config  # noqa: E402
from .test_xml_tool_close_guard import TOOLS, PieceTokenizer, _spec  # noqa: E402

PREFILL_STEP = 16
# The residue shape Qwen3.8-27B-Atlassian emitted on the single engine (Q-85):
# the value closes, the model writes "!", then closes again.
RESIDUE = (
    "<tool_call>\n<function=shell>\n<parameter=command>\nls\n</parameter>\n"
    "!\n</parameter>\n</function>\n</tool_call>"
)
CLEAN = (
    "<tool_call>\n<function=shell>\n<parameter=command>\nls\n</parameter>\n"
    "</function>\n</tool_call>"
)
PROMPT = "<|im_start|>user\nlist the files<|im_end|>\n<|im_start|>assistant\n"


@pytest.fixture(scope="module")
def group():
    return mx.distributed.init(strict=False)


@pytest.fixture(scope="module")
def model():
    mx.random.seed(20260926)
    text_config = _tiny_text_config()
    built = Model(ModelArgs(model_type="qwen4_exp", text_config=text_config))
    # float32: the CPU's gather_mm takes nothing else.
    built.set_dtype(mx.float32)
    mx.eval(built.parameters())
    return built


@pytest.fixture(scope="module")
def stage(group, model):
    layers = model.language_model.args.num_hidden_layers
    return pipe.PipelineStage(model, group, 0, layers, mx.float32)


def _scripted(stage, monkeypatch, tok, text: str) -> None:
    """The model prefers ``text``'s next token at every sampled step, then ``</``.

    A bias far above the tiny random model's own logits, so what is sampled
    is the script unless the guard masks it.
    """
    script = tok.encode(text)
    fallback = tok._id("</")
    real_forward = stage.forward
    sampled = [0]

    def forward(inputs, cache, logits, **kwargs):
        out = real_forward(inputs, cache, logits, **kwargs)
        if logits != "last":
            return out
        step = sampled[0]
        sampled[0] += 1
        bias = [0.0] * out.shape[-1]
        bias[fallback] = 500.0
        if step < len(script):
            bias[script[step]] = 1000.0
        return out + mx.array(bias, dtype=out.dtype)

    monkeypatch.setattr(stage, "forward", forward)


def _generate(engine, tok, tools: bool, steps: int) -> str:
    row = serve._Row(tok.encode(PROMPT), 256, 0.0, 1.0, tools=tools)
    engine.apply(serve._Plan(joiner=row))
    first = None
    while engine.joining is not None:
        first, _ = engine.prefill(None)
    tokens = [first]
    while len(tokens) < steps and not tok.decode(tokens).endswith("</tool_call>"):
        tokens.append(engine.decode(None)[0][0])
    engine.apply(serve._Plan(leave=[0]))
    return tok.decode(tokens)


def _command(text: str) -> str:
    result = Qwen3CoderToolParser(None).extract_tool_calls(text, {"tools": TOOLS})
    assert result.tools_called, text
    call = result.tool_calls[0]
    arguments = call["arguments"] if isinstance(call, dict) else call.function.arguments
    return json.loads(arguments)["command"]


def test_without_the_guard_the_residue_is_payload(stage, monkeypatch):
    """Negative control: the same scripted model, a row that declared no tools."""
    tok = PieceTokenizer()
    _scripted(stage, monkeypatch, tok, RESIDUE)
    engine = serve._Engine(stage, None, PREFILL_STEP, close_guard=_spec(tok))
    text = _generate(engine, tok, tools=False, steps=len(tok.encode(RESIDUE)))
    assert text == RESIDUE
    assert _command(text) == "ls\n</parameter>\n!"


def test_the_sampling_rank_holds_a_tool_row_to_the_skeleton(stage, monkeypatch):
    tok = PieceTokenizer()
    _scripted(stage, monkeypatch, tok, RESIDUE)
    engine = serve._Engine(stage, None, PREFILL_STEP, close_guard=_spec(tok))
    text = _generate(engine, tok, tools=True, steps=len(tok.encode(RESIDUE)))
    assert text == CLEAN
    assert _command(text) == "ls"


def test_a_tool_row_without_rules_decodes_as_before(stage, monkeypatch):
    """No spec on this rank (not the last one, or another wire): nothing changes."""
    tok = PieceTokenizer()
    _scripted(stage, monkeypatch, tok, RESIDUE)
    engine = serve._Engine(stage, None, PREFILL_STEP)
    text = _generate(engine, tok, tools=True, steps=len(tok.encode(RESIDUE)))
    assert text == RESIDUE


def _worker_plan(monkeypatch, plan: serve._Plan) -> serve._Plan:
    """Rank 0 broadcasts ``plan``; its collectives replayed into a worker rank."""
    recorded = []

    def record(group_, value):
        mx.eval(value)
        recorded.append(value)
        return value

    monkeypatch.setattr(serve, "_all_sum", record)
    assert serve._broadcast_plan(None, plan, 2, deciding=True) is not None
    replay = iter(recorded)
    monkeypatch.setattr(serve, "_all_sum", lambda group_, value: next(replay))
    got = serve._broadcast_plan(None, None, 2, deciding=False)
    assert next(replay, None) is None, "a collective would not pair"
    return got


def test_the_plan_carries_the_tools_flag_to_every_rank(monkeypatch):
    """Replay rank 0's collectives into a worker: the joiner's flag arrives."""
    for tools in (True, False):
        joiner = serve._Row([5, 6, 7], 9, 0.0, 1.0, store_id=3, store_at=2, tools=tools)
        got = _worker_plan(monkeypatch, serve._Plan(joiner=joiner, evict=[4]))
        assert got.joiner.tools is tools
        assert (got.joiner.ids, got.joiner.store_id, got.joiner.store_at) == (
            [5, 6, 7],
            3,
            2,
        )
        assert got.evict == [4]


class _EosPieceTokenizer(PieceTokenizer):
    @property
    def eos_token_ids(self):
        return [self._id("<|im_end|>")]


def test_the_last_rank_arms_the_guard_from_the_checkpoints_tokenizer(
    monkeypatch, tmp_path
):
    import mlx_lm.utils

    lines: list[str] = []
    monkeypatch.delenv(OPT_OUT_ENV, raising=False)
    monkeypatch.setattr(
        mlx_lm.utils, "load_tokenizer", lambda path: _EosPieceTokenizer()
    )
    spec = serve._xml_close_guard_spec(tmp_path, 3, lines.append)
    assert spec is not None and spec.rules
    assert lines == [
        f"[pipeline] rank 3: xml tool-call skeleton guard armed ({len(spec.rules)} rules)"
    ]

    lines.clear()
    monkeypatch.setattr(
        mlx_lm.utils,
        "load_tokenizer",
        lambda path: _EosPieceTokenizer(chat_template=None),
    )
    assert serve._xml_close_guard_spec(tmp_path, 3, lines.append) is None
    assert "not armed" in lines[0]

    lines.clear()
    monkeypatch.setenv(OPT_OUT_ENV, "0")

    def never(path):
        raise AssertionError("an opted-out rank loads no tokenizer")

    monkeypatch.setattr(mlx_lm.utils, "load_tokenizer", never)
    assert serve._xml_close_guard_spec(tmp_path, 3, lines.append) is None
    assert lines == [
        f"[pipeline] rank 3: xml tool-call skeleton guard disabled by {OPT_OUT_ENV}=0"
    ]


def _settle() -> int:
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    return mx.get_active_memory()


def test_a_prefix_snapshot_holds_only_the_bytes_it_is_charged(model):
    """512-token chunks, snapshot at 512: the QSA raw ring is a view of the chunk."""
    tokens = mx.array([[(i * 7) % 250 for i in range(1024)]], dtype=mx.int32)

    def snapshot(store_with) -> tuple[int, int]:
        base = _settle()
        cache = model.make_cache()
        entry = charged = None
        for start, stop in serve.prefill_chunks(0, 1024, 512, 512):
            model(tokens[:, start:stop], cache=cache)
            mx.eval([layer.state for layer in cache])
            if stop == 512:
                entry, charged = store_with(cache)
        del cache
        with_entry = _settle() - base
        del entry
        return with_entry - (_settle() - base), charged

    def unowned(cache):
        # The store before this change: a deep copy shares every buffer.
        entry = copy.deepcopy(cache)
        return entry, serve._held_bytes(entry)

    def stored(cache):
        store = serve._PrefixStore()
        charged = store.put(1, cache)
        return store, charged

    held, charged = snapshot(unowned)
    assert held > charged * 1.2, (held, charged)  # the defect, reproduced
    held, charged = snapshot(stored)
    # Allocator pages (16 KiB) round each recurrent layer's state up a little.
    assert held <= charged * 1.05, (held, charged)
