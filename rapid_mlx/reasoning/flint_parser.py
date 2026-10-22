# SPDX-License-Identifier: Apache-2.0
"""Reasoning parser for Recursive Flint (``model_type: flint``).

Flint decides on each turn whether to reason. Its chat template has NO
thinking switch and never primes ``<think>``: the generation prompt ends with
``<|assistant|>``. When the model reasons it writes ``<think>...</think>``
itself, first, and always closes it before a tool call (the checkpoint's own
``flint_sglang/reasoning_parser.py``: "never assumes a turn starts in
reasoning"; ``thinking`` / ``enable_thinking`` have no effect).

So, unlike Qwen3 / DeepSeek-R1:

* untagged output is CONTENT from the first character, streaming and not
  (the base parsers' no-tag default streams it as reasoning and corrects at
  the end, or switches only after 64 characters);
* ``enable_thinking`` / a prompt-priming guess never turns untagged output
  into reasoning (the source scan sees ``<think>`` in this template, which
  only renders HISTORY reasoning);
* a stream that never opened ``<think>`` gets no finalize correction: its
  bytes already went out as content.

A turn that opens ``<think>`` follows the shared ``<think>`` handling
(reasoning until ``</think>``, a ``<tool_call>`` written inside the thought is
promoted to content for the tool parser).
"""

from .base import DeltaMessage
from .deepseek_r1_parser import DeepSeekR1ReasoningParser


class FlintReasoningParser(DeepSeekR1ReasoningParser):
    """``<think>...</think>`` only where Flint writes it."""

    # Untagged output never starts inside reasoning, whatever the request
    # asked for: no rendered-priming probe is needed to say so.
    prompt_priming_rendered = False

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._prompt_primed_thinking = False
        self._prompt_closed_thinking = True

    def configure_request(
        self,
        *,
        enable_thinking: bool | None = None,
        prompt_thinking_active: bool | None = None,
        prompt_thinking_closed: bool | None = None,
    ) -> None:
        del enable_thinking, prompt_thinking_active, prompt_thinking_closed
        self.reset_state()
        self._prompt_primed_thinking = False
        self._prompt_closed_thinking = True

    def reset_state(self):
        super().reset_state()
        self._prompt_primed_thinking = False
        self._prompt_closed_thinking = True

    def extract_reasoning(
        self,
        model_output: str,
        enable_thinking: bool | None = None,
        prompt_thinking_active: bool | None = None,
    ) -> tuple[str | None, str | None]:
        del enable_thinking, prompt_thinking_active
        if self.start_token not in model_output and self.end_token not in model_output:
            return None, model_output
        return super().extract_reasoning(model_output, enable_thinking=None)

    def is_open_in_think(self, accumulated_text: str) -> bool:
        # Only an explicit opener can leave a turn inside reasoning.
        return super().is_open_in_think(accumulated_text)

    def finalize_streaming(
        self,
        accumulated_text: str,
        *,
        matched_stop: str | None = None,
        prompt_thinking_active: bool = False,
        finish_reason: str | None = None,
    ) -> DeltaMessage | None:
        del matched_stop, prompt_thinking_active, finish_reason
        # No tag was ever seen: every byte already streamed as content.
        # A tagged turn needs no correction either (the base parsers return
        # None once a tag was seen).
        return None
