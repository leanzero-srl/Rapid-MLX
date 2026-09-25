# SPDX-License-Identifier: Apache-2.0
"""Keep a Qwen3-Coder XML tool call on its own grammar after a parameter closes.

The wire (``chat_template.jinja``) renders every parameter as
``<parameter=KEY>\\nVALUE\\n</parameter>\\n`` and the ONLY things that may
follow that newline are the next ``<parameter=`` or ``</function>``. A model
can leave that grammar at exactly this point. Measured on
Qwen3.8-27B-Atlassian-Q8 (a LoRA merge of Qwen3.8-27B) with the goose agent
prompt: after the LAST parameter's ``</parameter>\\n`` the top candidates were
``}`` (p≈0.61), ``]`` (0.17), ``!`` (0.11) and only then ``</`` (0.05) —
greedy decoding leaves the grammar too. The model then re-closes
(``!\\n</parameter>\\n</function>``), and because a value legitimately ends
at the LAST ``</parameter>`` before the next sibling
(``tool_call_scan.segment_by_next_opener``, omlx#2507), the residue became
payload: shell commands ending in ``</parameter>\\n!`` ("syntax error near
unexpected token"), files with junk lines, a ``write`` that never got its
``path``. Between two parameters the same model put p≈1.0 on ``<``.

The parser cannot tell residue from a value that really contains
``</parameter>`` — on this wire the two are the same bytes. The decoder can:
at the one position where the template allows only two continuations, mask
everything else. Nothing is rewritten after the fact, nothing is dropped;
the model is held to the format its template defines, and chooses between
the two legal continuations itself.

The guard is a pure function of the token history it is handed, so it is
safe on both the ordinary decode path and MTP verification (``mtp_apply``);
it owns no mutable state for the verifier to snapshot. All of its work stays
on device: no host synchronisation is added to the decode loop.

Scope and cost of the rule: a value may no longer contain ``</parameter>``
immediately followed by a newline and then anything other than
``<parameter`` / ``</function>`` inside a ``<tool_call>`` — on this wire such
a value was already indistinguishable from a closed parameter plus stray
text. ``</parameter>`` inline (``print("</parameter>")``) is untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

PARAMETER_CLOSE = "</parameter>\n"
LEGAL_CONTINUATIONS = ("<parameter", "</function>")
CALL_OPEN = "<tool_call>"
CALL_CLOSE = "</tool_call>"
TEMPLATE_MARKERS = (
    CALL_OPEN,
    CALL_CLOSE,
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
)
OPT_OUT_ENV = "RAPID_MLX_XML_CLOSE_GUARD"


@dataclass
class XmlCloseGuardSpec:
    """Token-level form of the rule, derived from one tokenizer.

    ``states`` maps each prefix already emitted since the close (as token
    ids) to the ids the template allows next.
    """

    trigger: tuple[int, ...]
    states: tuple[tuple[tuple[int, ...], frozenset[int]], ...]
    open_id: int
    close_id: int
    _masks: dict[int, mx.array] = field(default_factory=dict, repr=False)

    def allowed_masks(self, vocab_size: int) -> mx.array:
        """``(len(states), vocab_size)`` bool rows, built once per vocab width."""
        masks = self._masks.get(vocab_size)
        if masks is None:
            vocab = mx.arange(vocab_size)
            masks = mx.stack(
                [
                    mx.any(vocab[:, None] == mx.array(sorted(allowed))[None, :], axis=1)
                    for _prefix, allowed in self.states
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


def _single_token_id(tokenizer: Any, text: str) -> int | None:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1 or tokenizer.decode(ids) != text:
        return None
    return int(ids[0])


def xml_close_guard_spec(tokenizer: Any) -> XmlCloseGuardSpec | None:
    """The guard's rule for ``tokenizer``, or ``None`` when its wire is not this one.

    Applies only when the chat template renders the parameterised XML tool
    call (every marker in ``TEMPLATE_MARKERS``) and the tokenizer carries the
    ``<tool_call>`` wrapper as single tokens (the guard locates an open call
    by them). The trigger and the legal continuations are the tokenizer's own
    encodings of the template text, so no id is assumed.
    """
    if tokenizer is None or not callable(getattr(tokenizer, "encode", None)):
        return None
    template = _template_text(tokenizer)
    if not all(marker in template for marker in TEMPLATE_MARKERS):
        return None
    open_id = _single_token_id(tokenizer, CALL_OPEN)
    close_id = _single_token_id(tokenizer, CALL_CLOSE)
    if open_id is None or close_id is None or open_id == close_id:
        return None
    trigger = tuple(tokenizer.encode(PARAMETER_CLOSE, add_special_tokens=False))
    if not trigger:
        return None
    trie: dict[tuple[int, ...], set[int]] = {}
    for continuation in LEGAL_CONTINUATIONS:
        full = tuple(
            tokenizer.encode(PARAMETER_CLOSE + continuation, add_special_tokens=False)
        )
        # The close must tokenize the same way whatever follows it, or the
        # trigger would not be where the model's tokens put it.
        if full[: len(trigger)] != trigger or len(full) == len(trigger):
            logger.warning(
                "xml close guard: %r does not tokenize as %r + continuation; "
                "guard not armed for this tokenizer",
                PARAMETER_CLOSE + continuation,
                PARAMETER_CLOSE,
            )
            return None
        tail = full[len(trigger) :]
        for depth in range(len(tail)):
            trie.setdefault(tail[:depth], set()).add(tail[depth])
    states = tuple(
        (prefix, frozenset(allowed))
        for prefix, allowed in sorted(trie.items(), key=lambda item: len(item[0]))
    )
    return XmlCloseGuardSpec(
        trigger=trigger, states=states, open_id=open_id, close_id=close_id
    )


class XmlToolCloseGuard:
    """Logits processor enforcing ``XmlCloseGuardSpec`` on one request."""

    def __init__(self, spec: XmlCloseGuardSpec):
        self.spec = spec
        self._windows = [
            mx.array(spec.trigger + prefix, dtype=mx.int32)
            for prefix, _allowed in spec.states
        ]

    def _apply(self, tokens: mx.array, logits: mx.array) -> mx.array:
        history = tokens.reshape(-1)
        n = history.shape[0]
        active = [
            (row, window)
            for row, window in enumerate(self._windows)
            if window.shape[0] <= n
        ]
        if not active:
            return logits
        matched = mx.stack(
            [
                mx.all(history[-window.shape[0] :].astype(mx.int32) == window)
                for _row, window in active
            ]
        )
        positions = mx.arange(n)
        last_open = mx.max(mx.where(history == self.spec.open_id, positions, -1))
        last_close = mx.max(mx.where(history == self.spec.close_id, positions, -1))
        inside_call = last_open > last_close
        masks = self.spec.allowed_masks(logits.shape[-1])[
            mx.array([row for row, _window in active])
        ]
        allowed = mx.any(matched[:, None] & masks, axis=0)
        guarded = mx.where(allowed, logits, -mx.inf)
        return mx.where(mx.any(matched) & inside_call, guarded, logits)

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
