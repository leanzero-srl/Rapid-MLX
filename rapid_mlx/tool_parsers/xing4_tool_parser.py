# SPDX-License-Identifier: Apache-2.0
"""
Xing4.0 tool call parser for rapid-mlx.

XingChen-AGI Xing4.0 (``xing4_0``) chat template renders a call as::

    <tool_call>NAME<param_key>K1</param_key><param_value>V1</param_value>
    <param_key>K2</param_key><param_value>V2</param_value>...</tool_call>

(one line, no separators). A string value is written RAW (no escaping); any
other value as JSON (``v | tojson if v is not string``). It is GLM-4.7's shape
with other tag names, so streaming follows ``Glm47ToolParser`` (the first call
reaches the wire at its close, later calls from the complete output).

Two deliberate differences from the GLM parser, both about VALUES:

* A value ends at a ``</param_value>`` that is followed by the next
  ``<param_key>`` or by the call's ``</tool_call>``, not at the first
  ``</tool_call>`` / ``</param_value>`` in the text. A file whose own text
  contains ``</tool_call>`` (docs about this very wire) therefore arrives
  whole. Residual, the same as every unescaped XML wire: a value that itself
  contains ``</param_value><param_key>`` or ``</param_value></tool_call>``
  ends there.
* A value whose declared schema type is a string comes back byte-identical
  (no ``strip()``, no ``json.loads``): ``tests/test_tool_call_value_fidelity``.
  Other values are JSON-decoded, and kept raw when they are not JSON.
"""

import json
import logging
import re
from typing import Any

from ..api.tool_calling import _get_tool_param_config, _schema_type
from ..tool_call_scan import inside_code, may_surface_undeclared, warn_undeclared_call
from .abstract_tool_parser import ExtractedToolCallInformation, ToolParserManager
from .glm47_tool_parser import Glm47ToolParser, generate_tool_id

logger = logging.getLogger(__name__)

_STRING_TYPES = frozenset({"string", "str", "text", "varchar", "char", "enum"})

_OPEN = "<tool_call>"
_CLOSE = "</tool_call>"
_KEY_OPEN = "<param_key>"
_KEY_CLOSE = "</param_key>"
_VAL_OPEN = "<param_value>"
_VAL_CLOSE = "</param_value>"
# A value's real end: its close followed by the next key or the call's close.
_VALUE_END = re.compile(r"</param_value>\s*(?=<param_key>|</tool_call>)")


def _scan_call(text: str, start: int) -> tuple[str, list[tuple[str, str]], int] | None:
    """Read one call that opens at ``start``; None when it is not complete.

    Returns (name, [(key, raw value)], end index after ``</tool_call>``).
    """
    pos = start + len(_OPEN)
    k = text.find(_KEY_OPEN, pos)
    c = text.find(_CLOSE, pos)
    if c < 0:
        return None
    if k < 0 or c < k:  # a call with no arguments
        name = text[pos:c].strip()
        return (name, [], c + len(_CLOSE)) if name and "<" not in name else None
    name = text[pos:k].strip()
    if not name or "<" in name or "\n" in name:
        return None
    params: list[tuple[str, str]] = []
    pos = k
    while True:
        if not text.startswith(_KEY_OPEN, pos):
            return None
        kc = text.find(_KEY_CLOSE, pos)
        if kc < 0:
            return None
        key = text[pos + len(_KEY_OPEN) : kc].strip()
        pos = kc + len(_KEY_CLOSE)
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if not text.startswith(_VAL_OPEN, pos):
            return None
        vstart = pos + len(_VAL_OPEN)
        end = _VALUE_END.search(text, vstart)
        if end is None:
            return None
        params.append((key, text[vstart : end.start()]))
        pos = end.end()
        if text.startswith(_CLOSE, pos):
            return name, params, pos + len(_CLOSE)
        if not text.startswith(_KEY_OPEN, pos):
            return None


@ToolParserManager.register_module(["xing4", "xing4_0", "xing"])
class Xing4ToolParser(Glm47ToolParser):
    """Tool call parser for XingChen-AGI Xing4.0 (``<param_key>`` pairs)."""

    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("xing_param_tool_call",)
    _CALL_HEADER = re.compile(
        r"<tool_call>\s*([^\n<]+?)\s*(?:<param_key>|</tool_call>)"
    )

    def _value(
        self, func_name: str, key: str, raw: str, request: dict[str, Any] | None
    ) -> Any:
        schema = _get_tool_param_config(func_name, request).get(key)
        if _schema_type(schema) in _STRING_TYPES:
            return raw
        try:
            return json.loads(raw.strip())
        except (json.JSONDecodeError, ValueError):
            return raw

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        cleaned_text = self.strip_think_tags(model_output)
        valid_names = self._get_tool_names(request)
        surface_undeclared = may_surface_undeclared(request)

        tool_calls = []
        pos = 0
        while True:
            start = cleaned_text.find(_OPEN, pos)
            if start < 0:
                break
            found = _scan_call(cleaned_text, start)
            if found is None:
                pos = start + len(_OPEN)
                continue
            name, params, end = found
            pos = end
            if valid_names and name not in valid_names:
                if not surface_undeclared or inside_code(cleaned_text, start):
                    continue
                warn_undeclared_call(name, valid_names, "xing4")
            arguments = {}
            for key, raw in params:
                if key:
                    arguments[key] = self._value(name, key, raw, request)
            tool_calls.append(
                {
                    "id": generate_tool_id(),
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                }
            )

        if tool_calls:
            return ExtractedToolCallInformation(
                tools_called=True, tool_calls=tool_calls, content=None
            )
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=cleaned_text
        )
