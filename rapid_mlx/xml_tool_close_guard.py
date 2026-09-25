# SPDX-License-Identifier: Apache-2.0
"""Keep a Qwen3-Coder XML tool call on its template's skeleton where no free text is legal.

The wire (``chat_template.jinja``) renders a call as::

    <tool_call>\\n<function=NAME>\\n<parameter=KEY>\\nVALUE\\n</parameter>\\n...</function>\\n</tool_call>

and a turn that calls tools ends right after its last ``</tool_call>``
(another call is joined with ``\\n``; text goes BEFORE the calls). So at
three positions the template admits no free text at all:

* after ``</parameter>\\n`` inside a call: ``<parameter`` or ``</function>``;
* after ``<tool_call>``: ``\\n<function``;
* after ``</tool_call>``: ``\\n<tool_call>`` or the end of the turn.

A model can leave the skeleton exactly there. Measured on
Qwen3.8-27B-Atlassian-Q8 (a LoRA merge of Qwen3.8-27B) replaying goose
session 20260925_39: after the LAST parameter's ``</parameter>\\n`` the top
candidates were ``}`` (p≈0.61), ``]`` (0.17), ``!`` (0.11) and only then
``</`` (0.05) — greedy decoding leaves the grammar too; between two
parameters the same model put p≈1.0 on ``<``. It then re-closed
(``!\\n</parameter>\\n</function>``), and because a value legitimately ends
at the LAST ``</parameter>`` before the next sibling
(``tool_call_scan.segment_by_next_opener``, omlx#2507), the residue became
payload: shell commands ending in ``</parameter>\\n!``, files with junk
lines, a ``write`` that never got its ``path``. Held to the first position
alone, the same model closed the call cleanly and then emitted ``!`` where
the turn ends, continuing into invented tool results — hence the other two.

The parser cannot tell residue from a value that really contains
``</parameter>`` (the same bytes on this wire). The decoder can: at these
positions it masks every token the template does not allow. Nothing is
rewritten after the fact and nothing is dropped; the model chooses among the
legal continuations itself.

The guard is a pure function of the token history it is handed, so it serves
the ordinary decode path and MTP verification (``mtp_apply``) alike and owns
no state for the verifier to snapshot. Its work stays on device: no host
synchronisation joins the decode loop. It is inert inside an open
``<think>`` block, where a drafted call is not a call.

Cost of the rule, stated: inside a ``<tool_call>`` a value can no longer hold
``</parameter>`` + newline followed by anything but ``<parameter`` /
``</function>`` — on this wire such a value was already indistinguishable
from a closed parameter plus stray text. An inline ``</parameter>``
(``print("</parameter>")``) is untouched.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

CALL_OPEN = "<tool_call>"
CALL_CLOSE = "</tool_call>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
TEMPLATE_MARKERS = (
    CALL_OPEN,
    CALL_CLOSE,
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
)
OPT_OUT_ENV = "RAPID_MLX_XML_CLOSE_GUARD"


@dataclass(frozen=True)
class SkeletonRule:
    """At ``window`` (the last tokens emitted), only ``allowed`` may come next.

    ``inside_call`` rules are armed only while a ``<tool_call>`` is open: their
    window is ordinary text that prose may also contain.
    """

    window: tuple[int, ...]
    allowed: frozenset[int]
    inside_call: bool


@dataclass
class XmlCloseGuardSpec:
    """The skeleton rules for one tokenizer."""

    rules: tuple[SkeletonRule, ...]
    open_id: int
    close_id: int
    think_ids: tuple[int, int] | None
    _masks: dict[int, mx.array] = field(default_factory=dict, repr=False)

    def allowed_masks(self, vocab_size: int) -> mx.array:
        """``(len(rules), vocab_size)`` bool rows, built once per vocab width."""
        masks = self._masks.get(vocab_size)
        if masks is None:
            vocab = mx.arange(vocab_size)
            masks = mx.stack(
                [
                    mx.any(
                        vocab[:, None] == mx.array(sorted(rule.allowed))[None, :],
                        axis=1,
                    )
                    for rule in self.rules
                ]
            )
            mx.eval(masks)
            self._masks[vocab_size] = masks
        return masks


def _template_text(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, dict):
        return "\n".join(t for t in template.values() if isinstance(t, str))
    return template if isinstance(template, str) else ""


def _encode(tokenizer: Any, text: str) -> tuple[int, ...]:
    return tuple(int(i) for i in tokenizer.encode(text, add_special_tokens=False))


def _single_token_id(tokenizer: Any, text: str) -> int | None:
    ids = _encode(tokenizer, text)
    if len(ids) != 1 or tokenizer.decode(list(ids)) != text:
        return None
    return ids[0]


def _rules_for(
    trigger: tuple[int, ...],
    continuations: Iterable[tuple[int, ...]],
    inside_call: bool,
) -> list[SkeletonRule]:
    """One rule per prefix of the continuations: the trie walked token by token."""
    trie: dict[tuple[int, ...], set[int]] = {}
    for tail in continuations:
        for depth in range(len(tail)):
            trie.setdefault(tail[:depth], set()).add(tail[depth])
    return [
        SkeletonRule(trigger + prefix, frozenset(allowed), inside_call)
        for prefix, allowed in sorted(trie.items(), key=lambda item: len(item[0]))
    ]


def _continuations(
    tokenizer: Any, trigger_text: str, texts: Iterable[str]
) -> tuple[tuple[int, ...], list[tuple[int, ...]]] | None:
    """``trigger_text`` and each continuation, as the tokenizer splits them in context."""
    trigger = _encode(tokenizer, trigger_text)
    tails = []
    for text in texts:
        full = _encode(tokenizer, trigger_text + text)
        # The trigger must tokenize the same way whatever follows it, or the
        # window would not be where the model's tokens put it.
        if full[: len(trigger)] != trigger or len(full) == len(trigger):
            logger.warning(
                "xml close guard: %r does not tokenize as %r + continuation; "
                "guard not armed for this tokenizer",
                trigger_text + text,
                trigger_text,
            )
            return None
        tails.append(full[len(trigger) :])
    return trigger, tails


def xml_close_guard_spec(
    tokenizer: Any, end_ids: Iterable[int]
) -> XmlCloseGuardSpec | None:
    """The skeleton rules for ``tokenizer``, or ``None`` when its wire is not this one.

    Applies only when the chat template renders the parameterised XML tool
    call (every marker in ``TEMPLATE_MARKERS``) and the tokenizer carries the
    ``<tool_call>`` wrapper as single tokens. ``end_ids`` are the ids that end
    a turn for this engine (its stop tokens). Every other id is the
    tokenizer's own encoding of the template text — none is assumed.
    """
    end_ids = frozenset(int(i) for i in end_ids)
    if (
        tokenizer is None
        or not callable(getattr(tokenizer, "encode", None))
        or not end_ids
    ):
        return None
    template = _template_text(tokenizer)
    if not all(marker in template for marker in TEMPLATE_MARKERS):
        return None
    open_id = _single_token_id(tokenizer, CALL_OPEN)
    close_id = _single_token_id(tokenizer, CALL_CLOSE)
    if open_id is None or close_id is None or open_id == close_id:
        return None
    think_open = _single_token_id(tokenizer, THINK_OPEN)
    think_close = _single_token_id(tokenizer, THINK_CLOSE)
    think_ids = (
        (think_open, think_close)
        if think_open is not None and think_close is not None
        else None
    )

    rules: list[SkeletonRule] = []
    after_close = _continuations(
        tokenizer, "</parameter>\n", ("<parameter", "</function>")
    )
    after_open = _continuations(tokenizer, CALL_OPEN, ("\n<function",))
    next_call = _continuations(tokenizer, CALL_CLOSE, ("\n" + CALL_OPEN,))
    if after_close is None or after_open is None or next_call is None:
        return None
    rules += _rules_for(after_close[0], after_close[1], inside_call=True)
    rules += _rules_for(after_open[0], after_open[1], inside_call=False)
    rules += _rules_for(
        next_call[0],
        next_call[1] + [(end_id,) for end_id in sorted(end_ids)],
        inside_call=False,
    )
    return XmlCloseGuardSpec(
        rules=tuple(rules), open_id=open_id, close_id=close_id, think_ids=think_ids
    )


class XmlToolCloseGuard:
    """Logits processor enforcing ``XmlCloseGuardSpec`` on one request."""

    def __init__(self, spec: XmlCloseGuardSpec):
        self.spec = spec
        self._windows = [mx.array(rule.window, dtype=mx.int32) for rule in spec.rules]
        self._inside_call = mx.array([rule.inside_call for rule in spec.rules])

    @staticmethod
    def _last(history: mx.array, positions: mx.array, token_id: int) -> mx.array:
        return mx.max(mx.where(history == token_id, positions, -1))

    def _apply(self, tokens: mx.array, logits: mx.array) -> mx.array:
        history = tokens.reshape(-1)
        n = history.shape[0]
        active = [
            row for row, window in enumerate(self._windows) if window.shape[0] <= n
        ]
        if not active:
            return logits
        tail = history.astype(mx.int32)
        matched = mx.stack(
            [
                mx.all(tail[-self._windows[row].shape[0] :] == self._windows[row])
                for row in active
            ]
        )
        positions = mx.arange(n)
        call_open = self._last(history, positions, self.spec.open_id) > self._last(
            history, positions, self.spec.close_id
        )
        rows = mx.array(active)
        armed = matched & (call_open | ~self._inside_call[rows])
        if self.spec.think_ids is not None:
            think_open, think_close = self.spec.think_ids
            thinking = self._last(history, positions, think_open) > self._last(
                history, positions, think_close
            )
            armed = armed & ~thinking
        masks = self.spec.allowed_masks(logits.shape[-1])[rows]
        allowed = mx.any(armed[:, None] & masks, axis=0)
        return mx.where(mx.any(armed), mx.where(allowed, logits, -mx.inf), logits)

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        return self._apply(tokens, logits)

    def mtp_apply(
        self,
        tokens: mx.array,
        _tentative_token_ids: mx.array,
        logits: mx.array,
    ) -> mx.array:
        """``tokens`` already carries the tentative prefix (the verifier contract)."""
        return self._apply(tokens, logits)

    def mtp_snapshot_state(self) -> None:
        return None

    def mtp_restore_state(self, _state: None) -> None:
        return None
