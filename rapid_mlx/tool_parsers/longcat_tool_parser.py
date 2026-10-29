# SPDX-License-Identifier: Apache-2.0
"""
Meituan LongCat tool call parser for rapid-mlx.

LongCat's chat template (``LongCat-Flash-Lite``, ``model_type
longcat_flash_ngram``; ``tokenizer_config.json`` says ``tool_parser_type:
longcat``) renders a call as::

    <longcat_tool_call>NAME
    <longcat_arg_key>K1</longcat_arg_key>
    <longcat_arg_value>V1</longcat_arg_value>
    ...
    </longcat_tool_call>

``v if v is string else v | tojson``: a string value is written RAW, any other
value as JSON. Every tag is one special token. Upstream mlx-lm reads the same
wire in ``mlx_lm/tool_parsers/longcat.py``; it strips every value, which
drops a file's trailing newline. Here the reading rules are Xing4.0's / Flint's
(GLM-4.7's shape with other tag names): a value ends at a
``</longcat_arg_value>`` followed by the next ``<longcat_arg_key>`` or the
call's ``</longcat_tool_call>``, a string-typed value comes back
byte-identical, other values are JSON-decoded (kept raw when not JSON).
Narration before the first call stays content (the template renders an
assistant row as content, then its ``tool_calls``).

Before this parser the engine had none for LongCat: the call came back as
content and ``tool_calls`` was empty (2026-10-29, LongCat-Flash-Lite-4bit).
"""

import re

from .abstract_tool_parser import ToolParserManager
from .flint_tool_parser import FlintToolParser
from .xing4_tool_parser import _Wire

_LONGCAT_WIRE = _Wire("longcat_arg_key", "longcat_arg_value", "longcat_tool_call")


@ToolParserManager.register_module(["longcat"])
class LongCatToolParser(FlintToolParser):
    """Tool call parser for Meituan LongCat (``<longcat_tool_call>`` wire)."""

    _WIRE = _LONGCAT_WIRE
    _PARSER_LABEL = "longcat"
    _CALL_OPEN = _LONGCAT_WIRE.call_open
    _CALL_CLOSE = _LONGCAT_WIRE.call_close
    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("longcat_arg_tool_call",)
    # Flint switched the route's tool-use system suffix off for its own
    # protocol; LongCat's template carries its own tool instructions, so
    # keep the route default.
    AUTO_TOOL_USE_SUFFIX = True
    _CALL_HEADER = re.compile(
        r"<longcat_tool_call>\s*([^\n<]+?)\s*"
        r"(?:\n|<longcat_arg_key>|</longcat_tool_call>)"
    )
