# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-Coder XML tool call parser for rapid-mlx.

Ported from vLLM upstream (vllm/tool_parsers/qwen3coder_tool_parser.py).

Format:
  <tool_call>                           <- optional wrapper (framing only)
  <function=NAME>                       <- REQUIRED, defines the tool call
  <parameter=KEY>VALUE</parameter>
  </function>                           <- REQUIRED
  </tool_call>                          <- optional wrapper (framing only)

The ``<tool_call>...</tool_call>`` wrapper is OPTIONAL framing. What
structurally defines a tool call is the ``<function=NAME>...</function>``
XML block — so the streaming state machine anchors on ``<function=``
throughout and treats the wrapper as a prefix to strip from content
(issue #978). Anchoring on the wrapper would leak whole tool calls as
raw content when a fine-tune's tokenizer omits the wrapper as a special
token but the model still emits well-formed ``<function=...>`` bodies
(observed with ``Shiftedx/qwopus3.6-35b-a3b-coder-mxfp4-mlx``).

Similar to Seed-OSS but without the seed: namespace prefix.
"""

import ast
import json
import logging
import re
import uuid
from collections.abc import Sequence
from typing import Any

from ..api.tool_calling import _decode_json_like, _schema_type
from ..tool_call_scan import (
    CallShape,
    find_value_close,
    inside_code,
    scan_marked_parameters,
    top_level_calls,
    trim_wrapping_newlines,
    warn_undeclared_call,
)
from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)

logger = logging.getLogger(__name__)


def _generate_tool_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a request field from either its wire dict or Pydantic model."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _get_arguments_config(func_name: str, tools: list[Any] | None) -> dict:
    """Extract argument config from tools list for type conversion."""
    if tools is None:
        return {}
    for tool in tools:
        func = _field(tool, "function", {})
        if _field(func, "name") == func_name:
            params = _field(func, "parameters", {})
            if isinstance(params, dict) and "properties" in params:
                return params["properties"]
            return {}
    return {}


def _is_string_param(param_name: str, param_config: dict) -> bool:
    """Whether ``param_name`` is explicitly string-typed per the tool schema.

    Unknown / un-configured / typeless params return False so they stay on
    the buffer-then-emit-once path. Non-streaming ``_convert_param_value``
    routes those through ``_decode_json_like()`` which may parse a JSON-
    looking value into an object — streaming it as a raw string would
    break stream/non-stream parity for those cases.
    """
    if param_name not in param_config:
        return False
    param_type = _schema_type(param_config[param_name])
    if param_type is None:
        return False
    return param_type in ("string", "str", "text", "varchar", "char", "enum")


def _convert_param_value(
    param_value: str, param_name: str, param_config: dict, func_name: str
) -> Any:
    """Convert parameter value based on its type in the schema.

    The TYPED scalar branches match against a whitespace-trimmed copy, never
    against the value that gets returned. Values used to arrive here already
    ``.strip()``-ed; now that only the wire's wrapping newline is removed
    (#3401), a model that pads a scalar -- ``<parameter=flag> true </parameter>``
    -- must still resolve to the scalar rather than to ``False``.

    A padded ``null`` is the keyword for every parameter EXCEPT a string-typed
    one, and that boundary is ``_is_string_param`` rather than a per-branch
    test, because it has to hold for the shapes that never reach the type
    dispatch at all: an undeclared parameter, a schema with no ``type`` key,
    and ``{"type": ["null"]}`` / null-only ``anyOf`` / ``oneOf`` (for which
    ``_schema_type`` returns ``None``). v0.14.1 resolved all of them to
    ``None`` because the value arrived pre-stripped.

    The string case is the deliberate exception, twice over. Its padding is
    payload under this wire's contract, and it is also the only set of values
    that reaches ``_close_string_increment``, where this function can be
    applied a second time to an already-decoded value (a pre-existing defect
    left untouched here). Trimming there would turn the string ``" null "``
    into ``None`` in the streamed arguments but not the non-streamed ones.
    """
    keyword = param_value.strip()
    if param_value.lower() == "null":
        return None
    if keyword.lower() == "null" and not _is_string_param(param_name, param_config):
        return None

    if param_name not in param_config:
        return _decode_json_like(param_value)

    cfg = param_config[param_name]
    param_type = _schema_type(cfg)
    if param_type is None:
        return _decode_json_like(param_value)

    if param_type in ("string", "str", "text", "varchar", "char", "enum"):
        try:
            decoded = json.loads(param_value)
        except (json.JSONDecodeError, TypeError):
            decoded = None
        if isinstance(decoded, str):
            return decoded
        return param_value

    if param_type.startswith(("int", "uint", "long", "short", "unsigned")):
        try:
            return int(keyword)
        except (ValueError, TypeError):
            return param_value
    elif param_type.startswith(("num", "float", "double")):
        try:
            return float(keyword)
        except (ValueError, TypeError):
            return param_value
    elif param_type in ("boolean", "bool", "binary"):
        return keyword.lower() == "true"
    else:
        if param_type in ("object", "array", "arr") or param_type.startswith(
            ("dict", "list")
        ):
            decoded = _decode_json_like(param_value)
            if decoded is not param_value:
                return decoded
        try:
            return ast.literal_eval(param_value)
        except (ValueError, SyntaxError):
            return param_value


@ToolParserManager.register_module(["qwen3_coder_xml"])
class Qwen3CoderToolParser(ToolParser):
    """
    Tool call parser for Qwen3-family models using named XML format.

    Supports the XML-based tool call format with <tool_call>/<function=...>
    tags and type conversion from tool schema. The parser name is retained for
    CLI compatibility; routing is based on the checkpoint's wire format, which
    is also used by Qwen3.8 27B.

    Used when --enable-auto-tool-choice --tool-call-parser qwen3_coder_xml are set.
    """

    SUPPORTS_NATIVE_TOOL_FORMAT = True
    EXPECTED_WIRE_FORMATS = ("qwen3_coder_xml_named", "tool_call_xml_body")

    # Grammar-CAPABLE (#558 "做全"): overrides ``structure_info`` below to emit
    # the Qwen3-Coder XML arg body. Only ``<tool_call>``/``</tool_call>`` are
    # single special tokens on the Qwen3-Coder tokenizer (verified on
    # ``mlx-community/Qwen3-Coder-Next-4bit``: ids 151657 / 151658); the inner
    # ``<function=`` / ``<parameter=`` / ``</parameter>`` / ``</function>``
    # markers are ordinary MULTI-token text, emitted as byte literals by the
    # grammar builder.
    _GRAMMAR_SENTINELS = ("<tool_call>", "</tool_call>")

    SUPPORTS_GRAMMAR: bool = True

    def structure_info(self):
        """Grammar-constraint wire triple for the Qwen3-Coder XML tool call (#558).

        Extends #558 grammar constraint from the JSON-body families
        (hermes / qwen / harmony) to the Qwen3-Coder XML wire. The model emits
        (verified byte-for-byte against this model's ``chat_template.jinja``)::

            <tool_call>
            <function=NAME>
            <parameter=KEY>
            VALUE
            </parameter>
            ...
            </function>
            </tool_call>

        The ARGUMENTS are an XML body, NOT a JSON object, so we return a
        ``StructureInfo`` with ``arg_style="xml"``: the builder emits one
        ``<parameter=KEY>\\nVALUE\\n</parameter>`` block per schema property, each
        VALUE constrained per its sub-schema (JSON strings for free-form strings,
        ``%json`` for scalars / objects / arrays, an alternation for enums) — see
        ``rapid_mlx.api.tool_grammar._emit_xml_arg_body``. Those surface forms are
        exactly what ``_parse_xml_function_call`` / ``_convert_param_value``
        round-trip back into JSON ``arguments``.

        ``<tool_call>`` / ``</tool_call>`` are single special tokens on the
        Qwen3-Coder tokenizer, declared as ``sentinels`` (rendered as Lark
        special-token refs); the trigger is ``<tool_call>``. The inner
        ``<function=`` / ``<parameter=`` markers are ordinary text (byte
        literals).

        As on hermes / qwen / harmony, OPT OUT (return ``None`` -> free-form
        fallback) unless the tokenizer proves both sentinels are single special
        tokens — a special-token sentinel on a tokenizer that encodes
        ``<tool_call>`` as multi-token text would build an unenforceable grammar.
        Grammar constraint is a best-effort opt-in, never a hard requirement.
        """
        from rapid_mlx.api.tool_grammar import (  # pragma: no cover - grammar opt-in
            StructureInfo,
            are_single_special_tokens,
        )

        if not are_single_special_tokens(self.model_tokenizer, self._GRAMMAR_SENTINELS):
            return None

        def _info(name: str):
            # The tool NAME is a bare identifier inside the ``<function=NAME>``
            # header (NOT a JSON string — Qwen3-Coder does not quote it), emitted
            # raw as a byte literal by the builder. ``begin`` starts with the
            # ``<tool_call>`` trigger (builder invariant).
            begin = f"<tool_call>\n<function={name}>\n"
            end = "</function>\n</tool_call>"
            return StructureInfo(
                begin=begin,
                end=end,
                trigger="<tool_call>",
                sentinels=self._GRAMMAR_SENTINELS,
                arg_style="xml",
            )

        return _info

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)

        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.tool_call_prefix = "<function="
        self.function_end_token = "</function>"
        self.parameter_prefix = "<parameter="
        self.parameter_end_token = "</parameter>"

        self.tool_call_complete_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>", re.DOTALL
        )
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL
        )
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )

        # Token IDs for streaming (graceful fallback if tokenizer absent)
        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        self._reset_streaming_state()

    def _reset_streaming_state(self):
        self.current_tool_index = 0
        self.is_tool_call_started = False
        self.header_sent = False
        self._current_tool_id = None
        self.current_function_name = None
        self.param_count = 0
        self.in_param = False
        self.in_function = False
        self.accumulated_text = ""
        self.json_started = False
        self.json_closed = False
        self.accumulated_params = {}
        self._streaming_request = None
        self._pending_tool_start: int | None = None
        self._pending_tool_wrapped = False
        self.prev_tool_call_arr = []
        self.in_param_emitted_chars = 0
        self.in_param_opened = False
        self.in_param_name: str | None = None
        self._legacy_raw_stream = False
        self._legacy_raw_param_count = 0
        # Where the first call this stream emitted began (its wrapper
        # included). Everything before it already streamed as content; the
        # legacy-raw finalizer returns the content after it.
        self._first_call_start: int | None = None

    def flush_held_content(self, full_text: str) -> str:
        """Text withheld after an opener that never became a call.

        Called by the postprocessor at stream end when no call reached the
        wire: ``Hmm <tool_call> is the opener.``, a stream cut mid-header, a
        header followed by a ``{`` that never closed. The buffered parser
        returns those bytes as content; the stream used to drop them.
        """
        start = self._pending_tool_start
        if start is None or self.header_sent or self._first_call_start is not None:
            return ""
        return full_text[start:] if start < len(full_text) else ""

    def _emit_string_increment(self, param_name: str, value_text: str) -> str:
        """Return a JSON fragment for the safe (already-final) portion of an
        in-flight string param value, or "" if nothing new can be flushed.

        We withhold the last ``len("</parameter>")`` chars of unread tail so
        a partial close tag (e.g. ``</par`` straddling a chunk boundary)
        cannot leak into an emitted JSON fragment.
        """
        keep_back = len(self.parameter_end_token)
        safe_end = len(value_text) - keep_back
        if safe_end <= self.in_param_emitted_chars:
            return ""
        safe = value_text[self.in_param_emitted_chars : safe_end]
        if not safe:
            return ""
        inner = json.dumps(safe, ensure_ascii=False)[1:-1]
        self.in_param_emitted_chars = safe_end
        if not self.in_param_opened:
            self.in_param_opened = True
            prefix = "" if self.param_count == 0 else ", "
            return f'{prefix}"{param_name}": "{inner}'
        return inner

    @staticmethod
    def _decoded_json_string_prefix(value_text: str) -> str:
        """Decode the complete portion of an in-flight JSON string.

        Token boundaries may split an escape (including ``\\uXXXX``), so only
        the prefix consisting of complete JSON characters is returned.
        """
        text = value_text.lstrip()
        if not text.startswith('"'):
            return ""
        i = 1
        safe_end = i
        while i < len(text):
            char = text[i]
            if char == '"':
                break
            if char != "\\":
                i += 1
                safe_end = i
                continue
            if i + 1 >= len(text):
                break
            escape = text[i + 1]
            if escape == "u":
                if i + 6 > len(text):
                    break
                digits = text[i + 2 : i + 6]
                if any(c not in "0123456789abcdefABCDEF" for c in digits):
                    break
                codepoint = int(digits, 16)
                if 0xD800 <= codepoint <= 0xDBFF:
                    # A high surrogate is not independently emit-safe: JSON
                    # decoding combines it with a following low surrogate.
                    if i + 12 > len(text) or text[i + 6 : i + 8] != "\\u":
                        break
                    low_digits = text[i + 8 : i + 12]
                    if any(
                        c not in "0123456789abcdefABCDEF" for c in low_digits
                    ) or not (0xDC00 <= int(low_digits, 16) <= 0xDFFF):
                        break
                    i += 12
                    safe_end = i
                    continue
                i += 6
            elif escape in '"\\/bfnrt':
                i += 2
            else:
                break
            safe_end = i
        encoded = text[1:safe_end]
        try:
            return json.loads(f'"{encoded}"')
        except json.JSONDecodeError:
            return ""

    def _emit_decoded_string_increment(
        self, param_name: str, decoded_value: str
    ) -> str:
        """Emit newly decoded characters from an in-flight JSON string."""
        if len(decoded_value) <= self.in_param_emitted_chars:
            return ""
        safe = decoded_value[self.in_param_emitted_chars :]
        inner = json.dumps(safe, ensure_ascii=False)[1:-1]
        self.in_param_emitted_chars = len(decoded_value)
        if not self.in_param_opened:
            self.in_param_opened = True
            prefix = "" if self.param_count == 0 else ", "
            return f'{prefix}"{param_name}": "{inner}'
        return inner

    def _close_string_increment(
        self, param_name: str, full_value: str, param_config: dict
    ) -> str:
        """Emit the closing fragment for an in-flight string param now that
        ``</parameter>`` has arrived. Handles both the long-string case
        (opener already emitted; emit tail + closing quote) and the short-
        string case (opener never emitted; emit the whole ``"name": "value"``).
        """
        if not self.in_param_opened:
            converted = _convert_param_value(
                full_value,
                param_name,
                param_config,
                self.current_function_name or "",
            )
            serialized = json.dumps(converted, ensure_ascii=False)
            prefix = "" if self.param_count == 0 else ", "
            return f'{prefix}"{param_name}": {serialized}'
        tail = full_value[self.in_param_emitted_chars :]
        inner = json.dumps(tail, ensure_ascii=False)[1:-1]
        return f'{inner}"'

    def finalize_legacy_raw_stream(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> dict | None:
        """Return the un-emitted JSON suffix for a deferred raw parameter."""
        if not self._legacy_raw_stream:
            return None
        result = self._extract_tool_calls(model_output, request, warn_undeclared=False)
        if not result.tools_called or not result.tool_calls:
            return None
        if self.current_tool_index >= len(result.tool_calls):
            return None
        current = result.tool_calls[self.current_tool_index]
        arguments = json.loads(current["arguments"])
        remaining = list(arguments.items())[self._legacy_raw_param_count :]
        prefix = ", " if self._legacy_raw_param_count else ""
        suffix = prefix + ", ".join(
            f"{json.dumps(name)}: {json.dumps(value, ensure_ascii=False)}"
            for name, value in remaining
        )
        suffix += "}"
        self._legacy_raw_stream = False
        tool_calls = [
            {
                "index": self.current_tool_index,
                "function": {"arguments": suffix},
            }
        ]
        declared = self._declared_tool_names(request)
        for index, call in enumerate(
            result.tool_calls[self.current_tool_index + 1 :],
            start=self.current_tool_index + 1,
        ):
            if call["name"] not in declared:
                warn_undeclared_call(
                    call["name"], declared, "qwen3_coder_xml stream finalize"
                )
            tool_calls.append(
                {
                    "index": index,
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments"],
                    },
                }
            )
        out: dict[str, Any] = {"tool_calls": tool_calls}
        # Text after the first call that is not a call (a span kept as text,
        # prose between calls) was deferred with everything else; it belongs
        # on the wire as content, exactly as the buffered parser returns it.
        start = self._first_call_start
        if start is not None and result.content:
            already = model_output[:start]
            if result.content.startswith(already):
                tail = result.content[len(already) :]
                if tail.strip():
                    out["content"] = tail
        return out

    def _parse_xml_function_call(
        self, function_call_str: str, tools: list[Any] | None
    ) -> dict | None:
        """Parse a single function call from XML and return a tool call dict."""
        try:
            end_index = function_call_str.index(">")
        except ValueError:
            return None
        function_name = function_call_str[:end_index]
        param_config = _get_arguments_config(function_name, tools)
        parameters = function_call_str[end_index + 1 :]
        param_dict = {}
        # JSON-quoted values are exact and a second call's markup inside the
        # body is that value's text (``scan_marked_parameters``): neither can
        # contribute a parameter, here or in the recovery pass below.
        scanned = scan_marked_parameters(
            parameters,
            r"<parameter=([^>]+)>",
            self.parameter_end_token,
            valid_names=set(param_config) or None,
            call_close=self.function_end_token,
            call_header=self.tool_call_prefix,
        )
        parsed, protected = scanned if scanned is not None else ([], [])
        for p_name, p_value in parsed:
            if f"{self.parameter_end_token}\n" in p_value:
                # Kept as sent (the wire cannot tell payload from residue), but
                # said: an off-grammar re-close lands here, and so does a value
                # that quotes the format. xml_tool_close_guard is what stops
                # the former at decode time.
                logger.warning(
                    "qwen3_coder_xml: %s.%s carries an interior %r; kept as sent: %r",
                    function_name,
                    p_name,
                    self.parameter_end_token,
                    p_value[-80:],
                )
            param_dict[p_name] = _convert_param_value(
                p_value, p_name, param_config, function_name
            )
        # Preserve the upstream recovery behavior for malformed free-form
        # output that omitted one close tag: the positional scanner correctly
        # protects complete JSON-string payloads, while this fallback recovers
        # later declared parameters that would otherwise be swallowed by the
        # unterminated predecessor. Never overwrite a value the safe scan found.
        for match in self.tool_call_parameter_regex.finditer(parameters):
            if any(lo <= match.start() < hi for lo, hi in protected):
                continue
            match_text = match.group(1)
            try:
                idx = match_text.index(">")
            except ValueError:
                continue
            p_name = match_text[:idx]
            if p_name in param_dict or p_name not in param_config:
                continue
            p_value = trim_wrapping_newlines(str(match_text[idx + 1 :]))
            param_dict[p_name] = _convert_param_value(
                p_value, p_name, param_config, function_name
            )
        return {
            "id": _generate_tool_id(),
            "name": function_name,
            "arguments": json.dumps(param_dict, ensure_ascii=False),
        }

    def _get_function_calls(self, model_output: str) -> list[str]:
        return [
            body for body, _, _, _, _ in self._function_call_candidates(model_output)
        ]

    def _function_call_candidates(
        self, model_output: str, declared: set[str] | None = None
    ) -> list[tuple[str, int, int, bool, bool]]:
        """Return ``(body, span_start, span_end, is_wrapped, is_closed)``.

        ``is_closed`` is True only when the function's own top-level
        ``</function>`` was found — not for the EOS recoveries below that
        accept a call missing its inner closes.

        With ``declared`` given, nesting is decided by the shared structural
        reading (``tool_call_scan.top_level_calls``), the same one the
        nemotron parser and the scanner fallback use: a ``<function=`` that
        the reading puts inside an earlier call's value is not a candidate,
        and that earlier call's body runs to its last close. Before, this
        parser took every header after a first-close as a call, so a value
        that quoted ``</parameter></function>…<function=shell>…`` produced a
        DECLARED ``shell`` (review of 26452cbe3, items 5b/5c).
        """
        tops = None
        if declared is not None:
            tops = {
                top.start: top
                for top in top_level_calls(
                    model_output,
                    r"<function=([^>]+)>",
                    self.function_end_token,
                    declared,
                    wrapper=(self.tool_call_start_token, self.tool_call_end_token),
                )
            }
        candidates: list[tuple[str, int, int, bool, bool]] = []
        for start in self._function_start_positions(model_output):
            close = self._top_level_function_close(model_output, start)
            body_start = start + len(self.tool_call_prefix)
            if tops is not None:
                top = tops.get(start)
                if top is None:
                    continue
                if isinstance(top.shape, CallShape) and top.body_end >= 0:
                    # The shared reading's end: the structural close, read on
                    # the template's grid (a value that quotes a closer
                    # mid-line keeps it), or the last close before the next
                    # real call when the value runs on.
                    close = top.body_end
            if close == -1:
                body = model_output[body_start:]
                complete_params = (
                    self.parameter_prefix in body
                    and body.rstrip().endswith(self.parameter_end_token)
                )
                wrapper_start = model_output.rfind(self.tool_call_start_token, 0, start)
                wrapper_close = model_output.rfind(self.tool_call_end_token, 0, start)
                wrapped_zero_arg = (
                    wrapper_start > wrapper_close
                    and body.find(">") >= 0
                    and not body[body.find(">") + 1 :].strip()
                )
                next_wrapper = model_output.find(self.tool_call_start_token, body_start)
                recovery_end = next_wrapper if next_wrapper >= 0 else len(model_output)
                trailing_wrapper = model_output.rfind(
                    self.tool_call_end_token, body_start, recovery_end
                )
                if trailing_wrapper >= 0:
                    # EOS recovery for a malformed call missing inner closes.
                    # Use the final wrapper closer so literal closer text in a
                    # raw value remains payload (#1515).
                    body = model_output[body_start:trailing_wrapper]
                    function_end = trailing_wrapper
                else:
                    if not (complete_params or wrapped_zero_arg):
                        continue
                    function_end = len(model_output)
            else:
                body = model_output[body_start:close]
                function_end = close + len(self.function_end_token)

            span_start = start
            wrapper_start = model_output.rfind(self.tool_call_start_token, 0, start)
            wrapper_close_before = model_output.rfind(
                self.tool_call_end_token, 0, start
            )
            is_wrapped = wrapper_start > wrapper_close_before
            if (
                wrapper_start >= 0
                and not model_output[
                    wrapper_start + len(self.tool_call_start_token) : start
                ].strip()
            ):
                span_start = wrapper_start

            span_end = function_end
            wrapper_end = model_output.find(self.tool_call_end_token, function_end)
            if wrapper_end >= 0 and not model_output[function_end:wrapper_end].strip():
                span_end = wrapper_end + len(self.tool_call_end_token)
            candidates.append((body, span_start, span_end, is_wrapped, close != -1))
        return candidates

    def _content_without_admitted_calls(
        self, model_output: str, admitted_spans: list[tuple[int, int]]
    ) -> str:
        """Remove admitted functions without leaving half of a shared wrapper."""
        ranges: list[tuple[int, int]] = []
        for span_start, span_end in admitted_spans:
            function_start = model_output.find(
                self.tool_call_prefix, span_start, span_end
            )
            function_close = self._top_level_function_close(
                model_output, function_start
            )
            function_end = (
                function_close + len(self.function_end_token)
                if function_close >= 0
                else span_end
            )
            # An admitted call whose value runs on (the shared structural
            # reading) ends at its LAST close in the span, past the first.
            span_close = model_output.rfind(
                self.function_end_token, function_start, span_end
            )
            if span_close >= 0:
                function_end = max(
                    function_end, span_close + len(self.function_end_token)
                )
            owns_wrapper = model_output[span_start:].startswith(
                self.tool_call_start_token
            )
            wrapper_close = model_output.find(self.tool_call_end_token, function_end)
            if owns_wrapper and (function_close < 0 or wrapper_close < 0):
                function_start = span_start
            ranges.append((function_start, function_end))

        ranges.sort()
        merged_ranges: list[tuple[int, int]] = []
        for start, end in ranges:
            if merged_ranges and start <= merged_ranges[-1][1]:
                merged_ranges[-1] = (
                    merged_ranges[-1][0],
                    max(end, merged_ranges[-1][1]),
                )
            else:
                merged_ranges.append((start, end))
        ranges = merged_ranges

        # A wrapper may frame more than one function. Strip its delimiters only
        # when every non-whitespace byte inside it is already being removed;
        # otherwise both delimiters belong to the rejected content sibling.
        search_from = 0
        range_index = 0
        wrapper_ranges: list[tuple[int, int]] = []
        while True:
            wrapper_start = model_output.find(self.tool_call_start_token, search_from)
            if wrapper_start < 0:
                break
            inner_start = wrapper_start + len(self.tool_call_start_token)
            wrapper_close = model_output.find(self.tool_call_end_token, inner_start)
            if wrapper_close < 0:
                break
            while range_index < len(ranges) and ranges[range_index][1] <= inner_start:
                range_index += 1
            cursor = inner_start
            residual = False
            candidate_index = range_index
            while (
                candidate_index < len(ranges)
                and ranges[candidate_index][0] < wrapper_close
            ):
                start, end = ranges[candidate_index]
                if model_output[cursor : max(cursor, start)].strip():
                    residual = True
                    break
                cursor = max(cursor, min(end, wrapper_close))
                candidate_index += 1
            if not residual and model_output[cursor:wrapper_close].strip():
                residual = True
            if not residual:
                wrapper_ranges.append(
                    (
                        wrapper_start,
                        wrapper_close + len(self.tool_call_end_token),
                    )
                )
            search_from = wrapper_close + len(self.tool_call_end_token)

        content_parts = []
        cursor = 0
        for start, end in sorted([*ranges, *wrapper_ranges]):
            if start > cursor:
                content_parts.append(model_output[cursor:start])
            cursor = max(cursor, end)
        content_parts.append(model_output[cursor:])
        return "".join(content_parts)

    def _closes_canonically(self, text: str, function_start: int) -> bool:
        """Whether the call at ``function_start`` closed on the template's grid.

        The template writes every raw value as ``VALUE\n</parameter>``; the
        decode guard (``xml_tool_close_guard``) holds the model to that, so a
        line-start ``</parameter>`` IS the close on this wire. A raw value
        that this parser closed at a MID-line ``</parameter>`` is the one
        shape it cannot read for certain: the same bytes are a value that
        merely contains the marker (an agent writing about this format). A
        JSON-quoted value is exact either way. False also for a call that has
        not closed.
        """
        prefix_len = len(self.tool_call_prefix)
        param_open_len = len(self.parameter_prefix)
        param_close_len = len(self.parameter_end_token)
        j = function_start + prefix_len
        while True:
            next_param = text.find(self.parameter_prefix, j)
            next_close = text.find(self.function_end_token, j)
            if next_close == -1:
                return False
            if next_param == -1 or next_param > next_close:
                return True
            header_end = text.find(">", next_param + param_open_len)
            if header_end == -1:
                return False
            value_start = header_end + 1
            quoted = text[value_start:].lstrip().startswith('"')
            close = self._find_parameter_close(text, value_start)
            if close == -1:
                return False
            if not quoted and not text[:close].endswith("\n"):
                return False
            j = close + param_close_len

    def _undeclared_after_ambiguous_close(
        self, text: str, earlier_starts: list[int], declared: set[str]
    ) -> bool:
        """Whether an undeclared call could be payload of the call before it.

        Declared names decide nesting: an undeclared opener is a sibling only
        if the nearest DECLARED call before it closed unambiguously (see
        ``_closes_canonically``). Otherwise its bytes may be that call's
        value, split by a literal marker, and surfacing them would fabricate
        a sibling — the property ``tool_call_scan`` keeps for the other XML
        paths. No declared call before it: nothing to be nested in.
        """
        for start in reversed(earlier_starts):
            name_start = start + len(self.tool_call_prefix)
            name_end = text.find(">", name_start)
            if name_end >= 0 and text[name_start:name_end] in declared:
                return not self._closes_canonically(text, start)
        return False

    def _is_wrapped_at(self, text: str, function_start: int) -> bool:
        """Whether ``<function=`` at ``function_start`` sits inside an open
        ``<tool_call>`` wrapper — the same test ``_function_call_candidates``
        applies, so the streaming and buffered paths agree."""
        return text.rfind(self.tool_call_start_token, 0, function_start) > text.rfind(
            self.tool_call_end_token, 0, function_start
        )

    def _call_framing_start(self, text: str, function_start: int) -> int:
        """Start of the call at ``function_start``, including its wrapper.

        The ``<tool_call>`` wrapper is part of the call when only whitespace
        separates it from ``<function=``.
        """
        wrapper = text.rfind(self.tool_call_start_token, 0, function_start)
        if (
            wrapper >= 0
            and not text[
                wrapper + len(self.tool_call_start_token) : function_start
            ].strip()
        ):
            return wrapper
        return function_start

    @staticmethod
    def _named_tool_choice(request: dict[str, Any] | None) -> str | None:
        if not isinstance(request, dict):
            return None
        choice = request.get("tool_choice")
        if isinstance(choice, dict) or choice is not None:
            function = _field(choice, "function")
            selected = _field(function, "name") or _field(choice, "name")
            if isinstance(selected, str) and selected:
                return selected
        return None

    @classmethod
    def _declared_tool_names(cls, request: dict[str, Any] | None) -> set[str]:
        """Return executable tool names offered by this request."""
        if not isinstance(request, dict) or request.get("tool_choice") == "none":
            return set()
        names: set[str] = set()
        for tool in request.get("tools") or []:
            function = _field(tool, "function")
            name = _field(function, "name") or _field(tool, "name")
            if isinstance(name, str) and name:
                names.add(name)
        selected = cls._named_tool_choice(request)
        if selected:
            return names.intersection({selected})
        return names

    @classmethod
    def _may_surface_undeclared(cls, request: dict[str, Any] | None) -> bool:
        """Whether a call to a name the request did not offer may still surface.

        The model calling a tool it was not given is ITS mistake, and the
        client is the one that has to answer it ("tool not found") so the
        model can recover — a cloud API or vLLM passes such a call through.
        Turning it into answer text instead ends an agent's turn on what
        looks like a final answer, which is how a 3-hour benchmark build died
        after three hallucinated ``read`` calls (AREX-2, 2026-10-14). The
        client's tool list is the authority on what can execute; a name it
        never offered cannot run there.

        Two cases keep the old refusal, because they are contracts about
        WHICH call may appear rather than a tool list: ``tool_choice="none"``
        (no executable tool at all — ``_declared_tool_names`` is empty) and a
        NAMED ``tool_choice`` (only that function may be called; the
        forced-choice filters downstream would drop any other name anyway).
        """
        return bool(cls._declared_tool_names(request)) and (
            cls._named_tool_choice(request) is None
        )

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        return self._extract_tool_calls(model_output, request, warn_undeclared=True)

    def _extract_tool_calls(
        self,
        model_output: str,
        request: dict[str, Any] | None,
        *,
        warn_undeclared: bool,
    ) -> ExtractedToolCallInformation:
        """``extract_tool_calls``; the legacy-raw stream finalizer passes
        ``warn_undeclared=False`` and warns only for the calls it emits."""
        if self.tool_call_prefix not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            declared = self._declared_tool_names(request)
            candidates = self._function_call_candidates(model_output, declared)
            if not candidates:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            tools = request.get("tools") if isinstance(request, dict) else None
            selected = self._named_tool_choice(request)
            surface_undeclared = self._may_surface_undeclared(request)

            tool_calls = []
            accepted_spans: list[tuple[int, int]] = []
            earlier_starts: list[int] = []
            for fc_str, span_start, span_end, is_wrapped, is_closed in candidates:
                function_start = model_output.find(self.tool_call_prefix, span_start)
                previous_starts = list(earlier_starts)
                earlier_starts.append(function_start)
                candidate_name = fc_str.split(">", 1)[0]
                if (
                    not is_wrapped
                    and self.parameter_prefix not in fc_str
                    and candidate_name != selected
                ):
                    # Wrapper-less calls are supported for #978 models, but a
                    # zero-argument bare span is indistinguishable from prose
                    # documenting the wire format. Require either canonical
                    # framing or parameter structure before model text can
                    # become executable data.
                    continue
                tc = self._parse_xml_function_call(fc_str, tools)
                if not tc:
                    continue
                if tc.get("name") not in declared:
                    # A name the caller did not offer. Candidates are already
                    # TOP-LEVEL (``_function_start_positions`` skips any
                    # ``<function=`` inside a parameter value), so this is a
                    # call the model made, not marker text in another call's
                    # argument. Pass it to the client, which answers it with
                    # "tool not found" — but only in the canonical shape: its
                    # own ``</function>`` arrived (no EOS recovery of a
                    # truncated body) and it sits inside a ``<tool_call>``
                    # wrapper. The wrapper is a special token the model emits
                    # only when it decides to call a tool; a bare
                    # ``<function=example>`` with parameters is just as often
                    # prose documenting the format, and examples use made-up
                    # names, so without the wrapper it stays text. Under
                    # tool_choice="none" or a named tool_choice it stays text
                    # too (see ``_may_surface_undeclared``).
                    if not (surface_undeclared and is_closed and is_wrapped):
                        continue
                    if inside_code(model_output, span_start):
                        # A wrapped example inside a markdown code fence (or
                        # inline code) is the model SHOWING the format.
                        continue
                    if self._undeclared_after_ambiguous_close(
                        model_output, previous_starts, declared
                    ):
                        logger.warning(
                            "[TOOL-CALL-UNDECLARED] qwen3_coder_xml: %r follows "
                            "a declared call whose raw value closed mid-line, "
                            "so it may be that value's text; kept as text",
                            tc["name"],
                        )
                        continue
                    if warn_undeclared:
                        warn_undeclared_call(
                            tc["name"], declared, "qwen3_coder_xml extract_tool_calls"
                        )
                tool_calls.append(tc)
                accepted_spans.append((span_start, span_end))

            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            content = self._content_without_admitted_calls(model_output, accepted_spans)

            return ExtractedToolCallInformation(
                tools_called=len(tool_calls) > 0,
                tool_calls=tool_calls,
                content=content if content else None,
            )
        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    # --- streaming helpers -----------------------------------------------
    #
    # The streaming state machine is anchored on the ``<function=`` /
    # ``</function>`` pair, NOT on the optional ``<tool_call>`` wrapper.
    # These helpers make that decoupling explicit so the wrapper token
    # only appears in the content-before strip logic (where it must, to
    # avoid emitting wrapper framing as user-visible content).

    def _first_opener_pos(self, text: str) -> int:
        """Position of the earliest tool-call framing character in ``text``.

        A tool call may be introduced by either the ``<tool_call>`` wrapper
        or the bare ``<function=`` prefix; both must be stripped from any
        content emitted before the call. Returns ``len(text)`` when no
        opener is present so callers can use it as an unconditional slice
        endpoint.
        """
        tc = text.find(self.tool_call_start_token)
        fn = text.find(self.tool_call_prefix)
        if tc == -1 and fn == -1:
            return len(text)
        if tc == -1:
            return fn
        if fn == -1:
            return tc
        return min(tc, fn)

    def _has_new_opener(self, delta_text: str, delta_token_ids: Sequence[int]) -> bool:
        """True when this delta introduces the first-ever tool-call opener.

        Accepts the wrapper token via string OR token-id (tokenizers that
        expose ``<tool_call>`` as a special token), and the bare
        ``<function=`` prefix via string. The two openers are equivalent
        as far as the state machine is concerned — either triggers the
        transition out of content-only mode.
        """
        if (
            self.tool_call_start_token_id is not None
            and self.tool_call_start_token_id in delta_token_ids
        ):
            return True
        return (
            self.tool_call_start_token in delta_text
            or self.tool_call_prefix in delta_text
        )

    def _top_level_function_close(self, text: str, start: int) -> int:
        """Return the position of the top-level ``</function>`` that closes
        the tool opened at ``start`` — i.e. the first ``</function>`` after
        ``start`` that is NOT inside a ``<parameter=…>…</parameter>`` value.

        Returns ``-1`` when the tool hasn't closed yet in the buffer. A
        ``</function>`` embedded in a user's ``code`` parameter (XML
        code samples are the canonical example) MUST NOT be treated as
        the tool boundary; otherwise streaming truncates mid-argument
        (codex review on #978).
        """
        prefix_len = len(self.tool_call_prefix)
        param_open_len = len(self.parameter_prefix)
        param_close_len = len(self.parameter_end_token)
        j = start + prefix_len
        n = len(text)
        while j < n:
            next_param = text.find(self.parameter_prefix, j)
            next_close = text.find(self.function_end_token, j)
            if next_close == -1:
                return -1
            if next_param != -1 and next_param < next_close:
                header_end = text.find(">", next_param + param_open_len)
                if header_end == -1:
                    return -1
                pclose = self._find_parameter_close(text, header_end + 1)
                if pclose == -1:
                    return -1
                j = pclose + param_close_len
                continue
            return next_close
        return -1

    def _find_parameter_close(self, text: str, value_start: int) -> int:
        """Find the structural parameter close, skipping JSON-string payload.

        Constrained XML strings are JSON encoded (#1542), so delimiter-looking
        text inside the quotes is data. Legacy raw model output keeps the old
        first-close behavior.
        """
        i = value_start
        # The XML template permits formatting whitespace between the parameter
        # opener and its JSON-string value.  Treat that whitespace as wire, not
        # as part of a legacy raw string.  In particular, token streaming can
        # expose the whitespace one delta before the opening quote; failing to
        # skip it makes the incremental path escape both JSON wrapper quotes
        # into the user value (``/tmp/x`` becomes ``\"/tmp/x\"``).
        # The rule lives in ``tool_call_scan.find_value_close`` so every XML
        # function-call path reads a value's end the same way.
        return find_value_close(text, i, self.parameter_end_token)[0]

    def _top_level_function_close_count(
        self, text: str, top_level_starts: list[int]
    ) -> int:
        """Count ``</function>`` tokens that structurally close a top-level
        ``<function=…>`` opener from ``top_level_starts``.

        Uses ``_top_level_function_close`` per start so a ``</function>``
        inside a ``<parameter=…>…</parameter>`` value (e.g. a user's
        ``code`` argument containing XML) never counts as a tool close.
        """
        return sum(
            1
            for start in top_level_starts
            if self._top_level_function_close(text, start) != -1
        )

    def _function_start_positions(self, text: str) -> list[int]:
        """Positions of TOP-LEVEL ``<function=`` openers in ``text``.

        Skips ``<function=`` substrings that appear inside a
        ``<parameter=…>…</parameter>`` value — those are user data (e.g.
        a ``code`` parameter containing XML), not structural tool-call
        boundaries. Function tags don't nest in Qwen3-Coder XML, so a
        top-level scan alternates between (a) looking for the next
        function opener while skipping parameter-value spans and (b)
        recording found openers. Both the tool-index slicing AND the
        "any more tools?" counter rely on this — using a naive
        ``str.count(...)`` for either would let a bogus in-value
        ``<function=…>`` corrupt streaming state (codex review on #978).

        Incomplete parameter tails (opener without matching
        ``</parameter>``) terminate the scan: everything after an
        unclosed value is potentially user data, so we conservatively
        refuse to promote further ``<function=`` occurrences until the
        value closes.
        """
        positions: list[int] = []
        i = 0
        n = len(text)
        prefix_len = len(self.tool_call_prefix)
        param_open_len = len(self.parameter_prefix)
        param_close_len = len(self.parameter_end_token)
        while i < n:
            next_func = text.find(self.tool_call_prefix, i)
            next_param = text.find(self.parameter_prefix, i)
            if next_func == -1:
                return positions
            if next_param != -1 and next_param < next_func:
                header_end = text.find(">", next_param + param_open_len)
                if header_end == -1:
                    return positions
                close_pos = self._find_parameter_close(text, header_end + 1)
                if close_pos == -1:
                    return positions
                i = close_pos + param_close_len
                continue
            if not self._header_is_prose(text, next_func):
                positions.append(next_func)
            i = next_func + prefix_len
        return positions

    def _header_is_prose(self, text: str, function_start: int) -> bool:
        """Whether the ``<function=NAME>`` at ``function_start`` is prose.

        A call's header is followed (after whitespace) by its first
        ``<parameter=``, its ``</function>`` or a JSON body. Anything else —
        ``I'll use <function=shell> to list the files`` — is text naming the
        format. Read as a call it took the parameters of the REAL call after
        it (a ``shell`` carrying ``write``'s arguments), or withheld the
        prose to the end of the stream and lost it. False while what follows
        has not arrived yet.
        """
        header_end = text.find(">", function_start + len(self.tool_call_prefix))
        if header_end == -1:
            return False
        rest = text[header_end + 1 : header_end + 1 + 4096].lstrip()[:16]
        if not rest or rest.startswith("{"):
            return False  # not here yet / a JSON body (read as it always was)
        for marker in (self.parameter_prefix, self.function_end_token):
            if rest.startswith(marker) or marker.startswith(rest):
                return False
        return True

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
        if not previous_text:
            self._reset_streaming_state()
            self._streaming_request = request
        elif request is not None and self._streaming_request is None:
            self._streaming_request = request

        if not delta_text:
            return None

        declared = self._declared_tool_names(
            request if request is not None else self._streaming_request
        )
        if not declared:
            # No executable tool exists for this request. Bypass the XML state
            # machine entirely so protocol examples stream byte-for-byte and
            # tool_choice=none cannot be overturned by model-authored markup.
            return {"content": delta_text}

        if self._legacy_raw_stream:
            # An escaping-free raw XML string cannot distinguish a literal
            # complete close sequence from structure at an ordinary chunk
            # boundary. Do not make an irreversible streaming decision;
            # StreamingPostProcessor.finalize() runs the last-closer-aware
            # non-streaming parser over the complete output (#1515).
            self.accumulated_text = current_text
            return None

        delta_token_ids = delta_token_ids or []
        self.accumulated_text = current_text

        # Check if we need to advance to next tool. The tool boundary is
        # ``</function>`` — that is the invariant close, whether or not
        # a wrapping ``</tool_call>`` follows. Both the close-count and
        # the "any more tools?" check must ignore ``<function=`` /
        # ``</function>`` substrings inside parameter values — a
        # ``code`` parameter carrying XML code would otherwise trip the
        # advance loop into targeting a bogus next block (codex review
        # on #978).
        if self.json_closed and not self.in_function:
            top_level_starts = self._function_start_positions(current_text)
            tool_count = len(top_level_starts)
            tool_ends = self._top_level_function_close_count(
                current_text, top_level_starts
            )
            if tool_ends > self.current_tool_index:
                self.current_tool_index += 1
                self.header_sent = False
                self.param_count = 0
                self.json_started = False
                self.json_closed = False
                self.accumulated_params = {}
                if self.current_tool_index >= tool_count:
                    self.is_tool_call_started = False
                else:
                    return None

        # Handle content before tool calls. Either opener (wrapper or bare
        # ``<function=``) transitions us out of content-only mode; the
        # content-before-strip position is whichever opener appears first
        # in ``delta_text`` so wrapper framing never leaks to the client.
        if not self.is_tool_call_started:
            if self._has_new_opener(delta_text, delta_token_ids):
                self.is_tool_call_started = True
                opener_pos = self._first_opener_pos(delta_text)
                self._pending_tool_start = len(previous_text) + opener_pos
                wrapper_start = current_text.find(
                    self.tool_call_start_token, self._pending_tool_start
                )
                function_start = current_text.find(
                    self.tool_call_prefix, self._pending_tool_start
                )
                self._pending_tool_wrapped = wrapper_start >= 0 and (
                    function_start < 0 or wrapper_start <= function_start
                )
                header_start = current_text.find(
                    self.tool_call_prefix, self._pending_tool_start
                )
                if header_start >= 0:
                    name_start = header_start + len(self.tool_call_prefix)
                    header_end = current_text.find(">", name_start)
                    if header_end >= 0:
                        candidate_name = current_text[name_start:header_end]
                        if candidate_name not in declared and not (
                            self._pending_tool_wrapped
                            and self._may_surface_undeclared(
                                request
                                if request is not None
                                else self._streaming_request
                            )
                            and not inside_code(current_text, self._pending_tool_start)
                        ):
                            saved_request = self._streaming_request
                            self._reset_streaming_state()
                            self._streaming_request = saved_request
                            return {"content": delta_text}
                        candidate_text = current_text[header_start:]
                        if self._header_is_prose(current_text, header_start) or (
                            not self._pending_tool_wrapped
                            and self.function_end_token in candidate_text
                            and self.parameter_prefix not in candidate_text
                        ):
                            saved_request = self._streaming_request
                            self._reset_streaming_state()
                            self._streaming_request = saved_request
                            return {"content": delta_text}
                content_before = (
                    delta_text[:opener_pos] if opener_pos < len(delta_text) else ""
                )
                if content_before:
                    return {"content": content_before}
                # Fall through to header parsing below instead of returning
                # None — the function header may already be in current_text.
            else:
                # Suppress the trailing-wrapper-close whitespace event so
                # a stream that ends with just ``</tool_call>\n`` doesn't
                # emit an empty tail. ``</function>`` is the actual tool
                # close; ``</tool_call>`` may follow as optional framing.
                trailing = current_text.rstrip()
                if delta_text.strip() == "" and (
                    trailing.endswith(self.tool_call_end_token)
                    or trailing.endswith(self.function_end_token)
                ):
                    return None
                return {"content": delta_text}

        # Find current tool call portion. Slice from the current
        # ``<function=`` opener to the matching top-level ``</function>``
        # close — this is the wrapper-agnostic tool-call block. Both
        # ends use the parameter-aware scanners so a user-visible
        # ``<function=…>`` OR ``</function>`` embedded in a parameter
        # value can't corrupt the slice.
        if not self.header_sent and self._pending_tool_start is not None:
            pending = current_text.find(self.tool_call_prefix, self._pending_tool_start)
            if pending >= 0 and self._header_is_prose(current_text, pending):
                # The header completed as prose (``use <function=shell> to …``):
                # release everything withheld since the opener as text.
                rejected = current_text[self._pending_tool_start :]
                saved_request = self._streaming_request
                self._reset_streaming_state()
                self._streaming_request = saved_request
                return {"content": rejected}
        function_starts = self._function_start_positions(current_text)
        if self.current_tool_index >= len(function_starts):
            return None

        tool_start_idx = function_starts[self.current_tool_index]
        func_close_idx = self._top_level_function_close(current_text, tool_start_idx)
        if func_close_idx == -1:
            tool_text = current_text[tool_start_idx:]
        else:
            tool_text = current_text[
                tool_start_idx : func_close_idx + len(self.function_end_token)
            ]

        # Parse function header
        if not self.header_sent:
            if self.tool_call_prefix in tool_text:
                func_start = tool_text.find(self.tool_call_prefix) + len(
                    self.tool_call_prefix
                )
                func_end = tool_text.find(">", func_start)
                if func_end != -1:
                    self.current_function_name = tool_text[func_start:func_end]
                    undeclared = self.current_function_name not in declared
                    if undeclared and not (
                        self._is_wrapped_at(current_text, tool_start_idx)
                        and self._may_surface_undeclared(
                            request if request is not None else self._streaming_request
                        )
                        and not inside_code(
                            current_text,
                            self._call_framing_start(current_text, tool_start_idx),
                        )
                        and not self._undeclared_after_ambiguous_close(
                            current_text,
                            function_starts[: self.current_tool_index],
                            declared,
                        )
                    ):
                        start = self._pending_tool_start
                        rejected = (
                            current_text[start:] if start is not None else delta_text
                        )
                        saved_request = self._streaming_request
                        self._reset_streaming_state()
                        self._streaming_request = saved_request
                        return {"content": rejected}
                    if undeclared:
                        # A wrapped, top-level call to a name the request did
                        # not offer streams exactly like a declared one: its
                        # header now, its raw values deferred to the
                        # end-of-stream parse (``_legacy_raw_stream``). It
                        # used to be HELD until its ``</function>``, which
                        # left nothing on the wire, so the postprocessor's
                        # #1359 budget released any call over 64 KB as text
                        # (review of 26452cbe3, item 2) — the budget decided
                        # correctness. Cut off mid-call it is now a call with
                        # broken arguments, the same as a cut-off declared
                        # call: the client answers it with a tool error and
                        # the agent goes on, where text ended its turn.
                        warn_undeclared_call(
                            self.current_function_name,
                            declared,
                            "qwen3_coder_xml stream",
                        )
                    if (
                        not self._pending_tool_wrapped
                        and self.parameter_prefix not in tool_text
                        and self.current_function_name
                        != self._named_tool_choice(
                            request if request is not None else self._streaming_request
                        )
                    ):
                        if self.function_end_token not in tool_text:
                            # Wait for a possible first parameter before
                            # emitting an irreversible tool-call header.
                            return None
                        start = self._pending_tool_start
                        rejected = (
                            current_text[start:] if start is not None else delta_text
                        )
                        saved_request = self._streaming_request
                        self._reset_streaming_state()
                        self._streaming_request = saved_request
                        return {"content": rejected}
                    first_param = tool_text.find(self.parameter_prefix, func_end)
                    # Also when the whole call arrived in one delta: a raw
                    # value's close is only certain from what FOLLOWS the call
                    # (a value that quotes a whole call closes structurally at
                    # the quoted close), so its arguments come from the
                    # end-of-stream parse like any other raw call.
                    if first_param >= 0:
                        name_end = tool_text.find(">", first_param)
                        if name_end >= 0:
                            param_name = tool_text[
                                first_param + len(self.parameter_prefix) : name_end
                            ]
                            visible = tool_text[name_end + 1 :].lstrip()
                            request_tools = (
                                self._streaming_request.get("tools")
                                if isinstance(self._streaming_request, dict)
                                else None
                            )
                            config = _get_arguments_config(
                                self.current_function_name, request_tools
                            )
                            if (
                                visible
                                and not visible.startswith('"')
                                and (_is_string_param(param_name, config) or not config)
                            ):
                                self._legacy_raw_stream = True
                                self._legacy_raw_param_count = 0
                    self._current_tool_id = _generate_tool_id()
                    self.header_sent = True
                    self.in_function = True
                    if self._first_call_start is None:
                        self._first_call_start = self._call_framing_start(
                            current_text, tool_start_idx
                        )

                    # If the function body is already complete, emit the full
                    # tool call in one chunk to prevent header-only output
                    # when coarse deltas or max_tokens truncation leave no
                    # further parser calls.
                    if func_close_idx != -1 and not self._legacy_raw_stream:
                        tools = None
                        if request and isinstance(request, dict):
                            tools = request.get("tools")
                        fc = tool_text[func_start : -len(self.function_end_token)]
                        parsed = self._parse_xml_function_call(fc, tools)
                        args = parsed["arguments"] if parsed else "{}"
                        self.json_started = True
                        self.json_closed = True
                        self.in_function = False
                        self.accumulated_params = {}
                        self.prev_tool_call_arr.append(
                            {"name": self.current_function_name, "arguments": args}
                        )
                        return {
                            "tool_calls": [
                                {
                                    "index": self.current_tool_index,
                                    "id": self._current_tool_id,
                                    "type": "function",
                                    "function": {
                                        "name": self.current_function_name,
                                        "arguments": args,
                                    },
                                }
                            ]
                        }

                    self.prev_tool_call_arr.append(
                        {"name": self.current_function_name, "arguments": "{}"}
                    )
                    self.json_started = True
                    return {
                        "tool_calls": [
                            {
                                "index": self.current_tool_index,
                                "id": self._current_tool_id,
                                "type": "function",
                                "function": {
                                    "name": self.current_function_name,
                                    "arguments": "{",
                                },
                            }
                        ]
                    }
            return None

        # Handle function body
        if self.in_function:
            if not self.json_started:
                self.json_started = True
                return {
                    "tool_calls": [
                        {
                            "index": self.current_tool_index,
                            "function": {"arguments": "{"},
                        }
                    ]
                }

            # Parameter start positions, found STRUCTURALLY: each value is
            # skipped by its own close (``_find_parameter_close``, JSON-string
            # aware), so a ``<parameter=`` inside a quoted value is that
            # value's text. A plain ``find()`` counted it as a parameter and
            # the streamed arguments ended ``... end, }`` — invalid JSON for
            # a value the buffered parser reads exactly.
            param_starts = []
            si = 0
            while True:
                si = tool_text.find(self.parameter_prefix, si)
                if si == -1:
                    break
                param_starts.append(si)
                name_end = tool_text.find(">", si + len(self.parameter_prefix))
                if name_end == -1:
                    break
                value_close = self._find_parameter_close(tool_text, name_end + 1)
                if value_close == -1:
                    break
                si = value_close + len(self.parameter_end_token)

            tools = None
            if self._streaming_request:
                tools = (
                    self._streaming_request.get("tools")
                    if isinstance(self._streaming_request, dict)
                    else None
                )
            param_config = _get_arguments_config(
                self.current_function_name or "", tools
            )

            json_fragments = []

            # In-flight string param from a prior call: drain whatever's
            # now available (close it if </parameter> has arrived, else
            # emit another safe slice). Runs BEFORE the complete-param
            # loop so same-chunk trailing params after the close get
            # picked up in the same call.
            if (
                self.in_param
                and self.in_param_name is not None
                and self.param_count < len(param_starts)
            ):
                param_idx = param_starts[self.param_count]
                param_start = param_idx + len(self.parameter_prefix)
                remaining = tool_text[param_start:]
                if ">" in remaining:
                    name_end = remaining.find(">")
                    value_start = param_start + name_end + 1
                    value_text = tool_text[value_start:]
                    if value_text.startswith("\n"):
                        value_text = value_text[1:]

                    end_idx = self._find_parameter_close(value_text, 0)
                    json_string_pending = value_text.lstrip().startswith('"')
                    if end_idx == -1 and not json_string_pending:
                        # Defensive fallback: model emitted next-param-prefix,
                        # </function>, or </tool_call> without a </parameter>.
                        # Use any of those as the close to avoid hanging
                        # forever in incremental mode (mirrors the existing
                        # complete-param fallback path below).
                        nxt = value_text.find(self.parameter_prefix)
                        fe = value_text.find(self.function_end_token)
                        te = value_text.find(self.tool_call_end_token)
                        candidates = [c for c in (nxt, fe, te) if c != -1]
                        if candidates:
                            end_idx = min(candidates)
                    if end_idx != -1:
                        pv = value_text[:end_idx]
                        if pv.endswith("\n"):
                            pv = pv[:-1]
                        self.accumulated_params[self.in_param_name] = pv
                        close_value = (
                            _convert_param_value(
                                pv,
                                self.in_param_name,
                                param_config,
                                self.current_function_name or "",
                            )
                            if json_string_pending
                            else pv
                        )
                        frag = self._close_string_increment(
                            self.in_param_name, close_value, param_config
                        )
                        if frag:
                            json_fragments.append(frag)
                        self.param_count += 1
                        self.in_param = False
                        self.in_param_name = None
                        self.in_param_emitted_chars = 0
                        self.in_param_opened = False
                    else:
                        frag = (
                            self._emit_decoded_string_increment(
                                self.in_param_name,
                                self._decoded_json_string_prefix(value_text),
                            )
                            if json_string_pending
                            else self._emit_string_increment(
                                self.in_param_name, value_text
                            )
                        )
                        if frag:
                            json_fragments.append(frag)

            # Process complete parameters
            while not self.in_param and self.param_count < len(param_starts):
                param_idx = param_starts[self.param_count]
                param_start = param_idx + len(self.parameter_prefix)
                remaining = tool_text[param_start:]

                if ">" not in remaining:
                    break

                name_end = remaining.find(">")
                current_param_name = remaining[:name_end]
                value_start = param_start + name_end + 1
                value_text = tool_text[value_start:]
                if value_text.startswith("\n"):
                    value_text = value_text[1:]

                if not value_text.strip():
                    break
                is_string = _is_string_param(current_param_name, param_config)
                if not value_text.lstrip().startswith('"') and (
                    is_string or not param_config
                ):
                    # Raw XML has no escaping rule, so its first apparent close
                    # may be payload. Freeze only the un-emitted suffix and let
                    # EOS parsing select the final structural closer.
                    self._legacy_raw_stream = True
                    self._legacy_raw_param_count = self.param_count
                    return None

                param_end_idx = self._find_parameter_close(value_text, 0)
                json_string_pending = value_text.lstrip().startswith('"')
                if param_end_idx == -1 and not json_string_pending:
                    # Try next parameter or function end as delimiter
                    next_param = value_text.find(self.parameter_prefix)
                    func_end = value_text.find(self.function_end_token)
                    if next_param != -1 and (func_end == -1 or next_param < func_end):
                        param_end_idx = next_param
                    elif func_end != -1:
                        param_end_idx = func_end
                    else:
                        tool_end_in_val = value_text.find(self.tool_call_end_token)
                        if tool_end_in_val != -1:
                            param_end_idx = tool_end_in_val
                        else:
                            # Param not yet closed. For string params, emit
                            # incrementally; for non-string types we can't
                            # emit partial JSON (half an int isn't valid),
                            # so fall through to the existing break path.
                            if _is_string_param(current_param_name, param_config):
                                frag = self._emit_string_increment(
                                    current_param_name, value_text
                                )
                                if frag:
                                    json_fragments.append(frag)
                                self.in_param = True
                                self.in_param_name = current_param_name
                            break

                if param_end_idx == -1 and json_string_pending:
                    if _is_string_param(current_param_name, param_config):
                        frag = self._emit_decoded_string_increment(
                            current_param_name,
                            self._decoded_json_string_prefix(value_text),
                        )
                        if frag:
                            json_fragments.append(frag)
                        self.in_param = True
                        self.in_param_name = current_param_name
                    break

                if param_end_idx == -1:
                    break

                pv = value_text[:param_end_idx]
                if pv.endswith("\n"):
                    pv = pv[:-1]

                self.accumulated_params[current_param_name] = pv

                converted = _convert_param_value(
                    pv,
                    current_param_name,
                    param_config,
                    self.current_function_name or "",
                )
                serialized = json.dumps(converted, ensure_ascii=False)

                if self.param_count == 0:
                    frag = f'"{current_param_name}": {serialized}'
                else:
                    frag = f', "{current_param_name}": {serialized}'
                self.param_count += 1
                json_fragments.append(frag)

            # If the function body is now complete, fold the closing ``}``
            # into the same delta so a chunk that batches the last param +
            # </function> emits a self-contained, JSON-valid arguments
            # document instead of leaving the close stranded for a later
            # call (which may never come if the stream ended).
            close_pending = (
                not self.in_param and not self.json_closed and func_close_idx != -1
            )
            if close_pending:
                self.json_closed = True
                tools = None
                if self._streaming_request:
                    tools = (
                        self._streaming_request.get("tools")
                        if isinstance(self._streaming_request, dict)
                        else None
                    )
                func_start = tool_text.find(self.tool_call_prefix) + len(
                    self.tool_call_prefix
                )
                func_content_end = len(tool_text) - len(self.function_end_token)
                if func_content_end != -1:
                    fc = tool_text[func_start:func_content_end]
                    try:
                        parsed = self._parse_xml_function_call(fc, tools)
                        if parsed and self.current_tool_index < len(
                            self.prev_tool_call_arr
                        ):
                            self.prev_tool_call_arr[self.current_tool_index][
                                "arguments"
                            ] = parsed["arguments"]
                    except Exception:
                        pass
                self.in_function = False
                self.accumulated_params = {}
                json_fragments.append("}")

            if json_fragments:
                combined = "".join(json_fragments)
                return {
                    "tool_calls": [
                        {
                            "index": self.current_tool_index,
                            "function": {"arguments": combined},
                        }
                    ]
                }

        return None
