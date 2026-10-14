# SPDX-License-Identifier: Apache-2.0
"""
GLM-4.7 tool call parser for rapid-mlx.

Handles GLM-4.7-Flash style tool calling format.
Based on vLLM's glm47_moe_tool_parser.py
"""

import json
import logging
import re
import uuid
from collections.abc import Sequence
from typing import Any

from ..tool_call_scan import inside_code, may_surface_undeclared, warn_undeclared_call
from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

logger = logging.getLogger(__name__)


def generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


@ToolParserManager.register_module(["glm47", "glm4"])
class Glm47ToolParser(ToolParser):
    """
    Tool call parser for GLM-4.7 and GLM-4.7-Flash models.

    Supports GLM-4.7 tool call format:
    <tool_call>function_name
    <arg_key>param1</arg_key><arg_value>value1</arg_value>
    <arg_key>param2</arg_key><arg_value>value2</arg_value>
    </tool_call>

    Used when --enable-auto-tool-choice --tool-call-parser glm47 are set.
    """

    # GLM-4.7 chat templates render `tool_calls` natively via the same
    # <tool_call>…</tool_call> markup the parser emits, so feed previous-turn
    # tool calls back to the model in their native form instead of converting
    # them to a synthetic <function=…> string.
    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("glm_named_tool_call",)

    # Match entire tool call block
    TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    # Match function name and optional arguments
    # GLM47 format: <tool_call>func_name\n<arg_key>...</arg_key>...
    FUNC_DETAIL_PATTERN = re.compile(
        r"<tool_call>\s*([^\n<]+?)(?:\n|\s*)(<arg_key>.*?)?</tool_call>", re.DOTALL
    )

    # Match individual argument key-value pairs
    ARG_PATTERN = re.compile(
        r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL
    )

    # The end of a call's arguments. Left over AFTER a later call, before the
    # next ``<tool_call>``, it is a close that only an earlier value running
    # over that call accounts for: the later "call" is that value's text.
    ORPHAN_CLOSE = re.compile(r"</arg_value>\s*</tool_call>")

    # Streaming: how many calls already reached the wire this turn.
    _emitted = 0
    _deferred_after_first = False

    def _deserialize(self, value: str) -> Any:
        """Convert string value to appropriate Python type.

        Uses json.loads for type coercion, falls back to raw string.
        """
        value = value.strip()

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def _get_tool_names(self, request: dict[str, Any] | None) -> set[str]:
        """Extract valid tool names from the request."""
        if not request:
            return set()
        tools = request.get("tools") or []
        return {
            t.get("function", {}).get("name", "") for t in tools if isinstance(t, dict)
        }

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete GLM-4.7 model response.
        """
        tool_calls = []

        # Strip think tags using the base class method (handles both
        # full <think>...</think> and implicit ...</think> patterns)
        cleaned_text = self.strip_think_tags(model_output)

        # Get valid tool names for validation
        valid_names = self._get_tool_names(request)
        surface_undeclared = may_surface_undeclared(request)

        # Find all tool call blocks
        matches = list(self.FUNC_DETAIL_PATTERN.finditer(cleaned_text))

        for position, found in enumerate(matches):
            match = found.groups()
            func_name = match[0].strip() if match[0] else ""
            args_section = match[1] if len(match) > 1 and match[1] else ""

            if not func_name:
                continue

            if position > 0:
                # A block followed by an orphan ``</arg_value></tool_call>``
                # (before the next opener) is an earlier value's text: the
                # safer reading never makes it a call, declared or not.
                limit = (
                    matches[position + 1].start()
                    if position + 1 < len(matches)
                    else len(cleaned_text)
                )
                if self.ORPHAN_CLOSE.search(cleaned_text, found.end(), limit):
                    logger.warning(
                        "glm47: %r is followed by a close it cannot account "
                        "for, so it is an earlier value's text; not a call",
                        func_name,
                    )
                    continue

            # Validate tool name against available tools if provided. A
            # well-formed call to a name the request did not offer still
            # reaches the client, which answers it with "tool not found"
            # (the rule every XML path applies since 26452cbe3; review item
            # 4): folding it into text ended an agent's turn on a "final
            # answer". ``tool_choice`` none / a named choice keep the refusal,
            # and an example inside markdown code is text.
            if valid_names and func_name not in valid_names:
                if not surface_undeclared or inside_code(cleaned_text, found.start()):
                    continue
                warn_undeclared_call(func_name, valid_names, "glm47")

            # Parse arguments
            arguments = {}
            if args_section:
                arg_matches = self.ARG_PATTERN.findall(args_section)
                for arg_key, arg_value in arg_matches:
                    key = arg_key.strip()
                    value = self._deserialize(arg_value)
                    if key:
                        arguments[key] = value

            tool_calls.append(
                {
                    "id": generate_tool_id(),
                    "name": func_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                }
            )

        # When tool calls are found, don't return reasoning text as content
        # GLM often outputs thinking/reasoning before tool calls without <think> tags
        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=None,
            )
        else:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=cleaned_text
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Extract tool calls from streaming GLM-4.7 model output.
        """
        if not previous_text:
            self._emitted = 0
            self._deferred_after_first = False

        # Skip thinking content in streaming
        if "<think>" in current_text and "</think>" not in current_text:
            return None

        # Once <tool_call> is detected, buffer everything until it closes.
        # Do NOT emit content deltas here, because if tool calls are found
        # the non-streaming path sets content=None (reasoning before the
        # tag should not leak as regular content).
        if "<tool_call>" in current_text:
            if self._deferred_after_first:
                return None
            # A close that completed in THIS delta, counted on the
            # accumulated text: a tokenizer can split ``</tool_call>`` across
            # deltas, and then no single delta contains it.
            if current_text.count("</tool_call>") > previous_text.count("</tool_call>"):
                result = self.extract_tool_calls(current_text, request)
                if result.tools_called:
                    # Only the FIRST call is emitted as it closes; whether a
                    # later block is a call or an earlier value's text is only
                    # known from what follows it (``ORPHAN_CLOSE``), so the
                    # rest is read once, from the complete output, by
                    # ``finalize_legacy_raw_stream``. This also stops the old
                    # re-emission of every call from index 0 on each close,
                    # which streamed two calls as ``readread`` with doubled
                    # arguments.
                    self._deferred_after_first = True
                    self._emitted = 1
                    return {"tool_calls": [self._wire_call(0, result.tool_calls[0])]}
            return None

        # No tool call detected yet; emit content delta.
        # Only strip think tags if they're actually present (avoid .strip()
        # on normal deltas which would eat inter-word spaces).
        if "</think>" in delta_text:
            clean_delta = self.strip_think_tags(delta_text)
            if clean_delta:
                return {"content": clean_delta}
            return None
        if delta_text:
            return {"content": delta_text}
        return None

    @staticmethod
    def _wire_call(index: int, tc: dict[str, Any]) -> dict[str, Any]:
        return {
            "index": index,
            "id": tc["id"],
            "type": "function",
            "function": {"name": tc["name"], "arguments": tc["arguments"]},
        }

    def finalize_legacy_raw_stream(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Every call after the first, read once from the complete output."""
        if not self._deferred_after_first:
            return None
        self._deferred_after_first = False
        result = self.extract_tool_calls(model_output, request)
        if not result.tools_called:
            return None
        later = result.tool_calls[self._emitted :]
        if not later:
            return None
        first = self._emitted
        self._emitted += len(later)
        return {
            "tool_calls": [self._wire_call(first + i, tc) for i, tc in enumerate(later)]
        }

    _CALL_HEADER = re.compile(r"<tool_call>\s*([^\n<]+?)\s*(?:\n|<arg_key>)")

    def holds_call_in_progress(
        self, text: str, request: dict[str, Any] | None = None
    ) -> bool:
        """An open ``<tool_call>NAME`` whose name may reach the client: a call
        being written, not markup to release (postprocessor #1359 budget)."""
        start = text.rfind("<tool_call>")
        if start < 0 or "</tool_call>" in text[start:]:
            return False
        header = self._CALL_HEADER.match(text, start)
        if header is None or inside_code(text, start):
            return False
        name = header.group(1).strip()
        return name in self._get_tool_names(request) or may_surface_undeclared(request)
