"""Each row runs its own tool-call guard for its whole answer on the pipeline split (goose Q-161).

On the tensor split (mlx_lm 0.31.3) a tool request that joined a batch after a
tool-less row left read the departed row's empty processor list: the lz.7 XML
skeleton guard ran on its first token and never again, and E2E #3f's answer
left its call and wrote ``!\\n</parameter>\\n</function>\\n`` over and over.  The
pipeline's sampler (``_sample``) applies ``row.close_guard`` per row, and a
row's guard travels with the row object through every join and departure
(``_regroup``) — this pins that: the guarded row's every token passes through
its own guard, whoever joins or leaves around it, and no other row's does.

In process, one rank, the tiny float32 qwen4_exp of the Q-134 tests; the guard
is a stand-in that forces one token, so a step it missed is visible.
"""

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
pytest.importorskip("fastapi")
pytestmark = pytest.mark.requires_mlx

from rapid_mlx import xml_tool_close_guard  # noqa: E402
from rapid_mlx.distributed import pipeline_qwen4_serve as serve  # noqa: E402

from .test_pipeline_qwen4_continuous import (  # noqa: E402, F401 - fixtures
    LONG,
    PREFILL_STEP,
    SHORT,
    THIRD,
    _solo,
    group,
    stage,
)

FORCED = 42


class _Forcing:
    """A guard that allows one token only, and counts the steps it saw."""

    made: list = []

    def __init__(self, spec):
        self.calls = 0
        _Forcing.made.append(self)

    def __call__(self, history, logits):
        self.calls += 1
        keep = mx.arange(logits.shape[-1]) == FORCED
        return mx.where(keep, logits, mx.array(-mx.inf, dtype=logits.dtype))


def _prefill(engine, streams, names, name, row):
    """``row`` prefills in its own cache, a decode step of the batch per chunk."""
    engine.apply(serve._Plan(joiner=row))
    while engine.joining is not None:
        if engine.rows:
            tokens, _ = engine.decode(None)
            for each, token in zip(names, tokens):
                streams[each].append(token)
        first, _ = engine.prefill(None)
        if first is not None:
            streams[name].append(first)
            names.append(name)


def _decode(engine, streams, names, steps):
    for _ in range(steps):
        tokens, _ = engine.decode(None)
        for name, token in zip(names, tokens):
            streams[name].append(token)


def test_a_tool_row_runs_its_guard_on_every_token_whoever_joins_or_leaves(
    stage, monkeypatch
):
    _Forcing.made = []
    monkeypatch.setattr(xml_tool_close_guard, "XmlToolCloseGuard", _Forcing)
    engine = serve._Engine(stage, None, PREFILL_STEP, close_guard=object())
    streams = {"plain": [], "tool": [], "third": []}
    names: list[str] = []

    # A tool-less row runs, then leaves before the tool row arrives (the
    # tensor split's #3f order: the title request, then the agent call).
    _prefill(engine, streams, names, "plain", serve._Row(list(SHORT), 64, 0.0, 1.0))
    _decode(engine, streams, names, 5)
    engine.apply(serve._Plan(leave=[0]))
    names.remove("plain")
    _prefill(
        engine,
        streams,
        names,
        "tool",
        serve._Row(list(LONG), 200, 0.0, 1.0, tools=True),
    )
    _decode(engine, streams, names, 6)
    # A tool-less row prefills beside it and joins the batch, then leaves it:
    # the tool row is extracted from the merged batch and runs on alone.
    _prefill(engine, streams, names, "third", serve._Row(list(THIRD), 64, 0.0, 1.0))
    _decode(engine, streams, names, 8)
    engine.apply(serve._Plan(leave=[names.index("third")]))
    names.remove("third")
    _decode(engine, streams, names, 6)

    (guard,) = _Forcing.made
    assert streams["tool"] == [FORCED] * len(streams["tool"])
    assert guard.calls == len(streams["tool"]) > 20
    # No other row sampled through it: theirs answer as each did alone.
    assert streams["plain"] == _solo(stage, SHORT, len(streams["plain"]))
    assert streams["third"] == _solo(stage, THIRD, len(streams["third"]))
