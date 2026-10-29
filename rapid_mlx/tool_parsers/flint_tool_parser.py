# SPDX-License-Identifier: Apache-2.0
"""
Recursive Flint tool call parser for rapid-mlx.

Flint's chat template (RecursiveSI/flint ``chat_template.jinja``) renders a
call as::

    <tool_call>NAME<arg_key>K1</arg_key><arg_value>V1</arg_value>
    <arg_key>K2</arg_key><arg_value>V2</arg_value>...</tool_call>

(one line, no separators; ``v | tojson if v is not string else v``: a string
value is written RAW, any other value as JSON). That is GLM-4.7's tag set,
but ``glm47`` strips every value and ``json.loads`` it, so a file lost its
trailing newline and a string ``"42"`` came back as the number 42. The reading
rules are Xing4.0's (same wire, other tag names): a value ends at a
``</arg_value>`` followed by the next ``<arg_key>`` or the call's
``</tool_call>``, and a string-typed value comes back byte-identical.

Residual, from the format itself (Flint's README says so too): a value that
contains ``</arg_value><arg_key>`` or ``</arg_value></tool_call>`` ends there.
After the call the model ends its turn with ``<|observation|>`` (an EOS id).
"""

import re

from .abstract_tool_parser import ToolParserManager
from .xing4_tool_parser import Xing4ToolParser, _Wire

_FLINT_WIRE = _Wire("arg_key", "arg_value")


@ToolParserManager.register_module(["flint"])
class FlintToolParser(Xing4ToolParser):
    """Tool call parser for Recursive Flint (``<arg_key>`` pairs, raw strings)."""

    _WIRE = _FLINT_WIRE
    _PARSER_LABEL = "flint"
    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("flint_arg_tool_call",)
    # The route's default tool-use system suffix ("Do NOT think out loud ...")
    # fights Flint's protocol: with it, Flint wrote write calls with no
    # content argument. See ``routes.chat._auto_tool_use_suffix``.
    AUTO_TOOL_USE_SUFFIX = False
    _CALL_HEADER = re.compile(r"<tool_call>\s*([^\n<]+?)\s*(?:<arg_key>|</tool_call>)")

    def extract_tool_calls(self, model_output, request=None):
        """As Xing4.0's, but narration BEFORE the calls stays content.

        Flint's template renders an assistant row as ``content`` followed by
        its ``tool_calls`` (reasoning lives in ``<think>``, never bare), so
        text ahead of the first call is the model's visible narration. The
        streaming path already sends it as content; the whole-response path
        now agrees instead of dropping it (GLM's rule, written for models that
        reason without tags).
        """
        result = super().extract_tool_calls(model_output, request)
        if result.tools_called:
            cleaned = self.strip_think_tags(model_output)
            first = cleaned.find(self._WIRE.call_open)
            narration = cleaned[:first].strip() if first > 0 else ""
            result.content = narration or None
        return result
