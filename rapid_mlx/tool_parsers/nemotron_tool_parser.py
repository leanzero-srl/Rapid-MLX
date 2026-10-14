# SPDX-License-Identifier: Apache-2.0
"""
Nemotron tool call parser for rapid-mlx.

Handles NVIDIA Nemotron models' tool calling format:
- <tool_call><function=name><parameter=p>v</parameter></function></tool_call>

Supports Nemotron-3-Nano-30B-A3B and similar models.
"""

import json
import logging
import re
import uuid
from collections.abc import Sequence
from typing import Any

from ..tool_call_scan import (
    declared_tool_names as _declared_tool_names,
)
from ..tool_call_scan import (
    inside_code,
    may_surface_undeclared,
    named_tool_choice,
    split_marked_calls,
    split_marked_parameters,
    warn_undeclared_call,
)
from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
    declared_parameter_names,
)

logger = logging.getLogger(__name__)


def generate_tool_id() -> str:
    """Generate a unique tool call ID."""
    return f"call_{uuid.uuid4().hex[:8]}"


@ToolParserManager.register_module(["nemotron", "nemotron3"])
class NemotronToolParser(ToolParser):
    """
    Tool call parser for NVIDIA Nemotron models.

    Supports Nemotron's tool call format:
    <tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>

    Also supports JSON arguments:
    <tool_call><function=get_weather>{"city": "Paris"}</function></tool_call>

    Used when --enable-auto-tool-choice --tool-call-parser nemotron are set.
    """

    EXPECTED_WIRE_FORMATS = ("tool_call_xml_body",)

    # Nemotron's tool template accepts structured assistant calls and tool
    # results, rendering its native function/parameter XML transcript.
    SUPPORTS_NATIVE_TOOL_FORMAT = True

    # How much of ``current_text`` this turn has already been accounted for on
    # the wire — forwarded as prose, consumed into an emitted call, or released
    # after a refusal. The refusal release below sends only what lies past it.
    #
    # A watermark rather than a "released once" boolean: a turn can hold more
    # than one refused block, and a boolean drops every block after the first.
    # Class-level default so an instance used without a preceding ``reset()``
    # still has the attribute.
    _content_upto = 0
    _stream_started = False
    # Deferred streaming (tools declared): the first call's header has been
    # sent and everything after it waits for the end-of-stream parse.
    _deferred = False
    _first_call_name: str | None = None

    # Pattern for Nemotron-style with parameters.
    #
    # The load-bearing signature of a call is ``<function=NAME>...</function>``;
    # the ``<tool_call>``/``</tool_call>`` wrapper is treated as optional /
    # decorative so that observed degradations still parse:
    #   (a) a missing/truncated ``</tool_call>``,
    #   (b) a bare ``<function=..>..</function>`` with no wrapper at all,
    #   (d) stray text between ``</function>`` and ``</tool_call>``,
    #   (e) prose between ``<tool_call>`` and ``<function=``.
    # Prose without ``<function=..>..</function>`` still never matches, so
    # this cannot manufacture a tool call out of plain text.
    TOOL_CALL_PATTERN = re.compile(
        r"<function=([^>]+)>(.*?)</function>",
        re.DOTALL,
    )

    # Residual bare wrapper tags left behind after the function bodies have
    # been stripped from ``content``; removed so they don't leak as text.
    RESIDUAL_WRAPPER_PATTERN = re.compile(r"</?tool_call>")

    # Pattern to extract parameters
    PARAM_PATTERN = re.compile(
        r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
        re.DOTALL,
    )

    def reset(self) -> None:
        """Clear per-request state.

        The base reset owns the tool-id counters; the content watermark is
        just as per-request. Relying on the ``not previous_text`` branch alone
        is not enough — the postprocessor can forward a new turn's opening
        prose through its own fast path before this parser sees a delta, so
        the first call can arrive with ``previous_text`` already non-empty and
        a stale watermark from the turn before.
        """
        super().reset()
        self._content_upto = 0
        self._stream_started = False
        self._deferred = False
        self._first_call_name = None

    def _visible_content_between(
        self,
        text: str,
        start: int,
        end: int,
        request: dict[str, Any] | None,
    ) -> str:
        """Project a raw range onto the non-tool content stream.

        Valid function spans are executable calls and must stay off the content
        channel. Refused spans are deliberately retained as prose. When at
        least one valid call exists, the decorative outer wrappers are removed
        too, matching :meth:`extract_tool_calls`.

        Keeping source offsets here is important: a single model delta can
        finish a refused block and a later valid call. A plain byte watermark
        advanced for the valid call used to swallow the refused range.
        """
        if end <= start:
            return ""

        valid_spans = [
            (span_start, span_end)
            for _name, _body, span_start, span_end in self._scan_calls(text, request)
        ]
        removed = list(valid_spans)
        if valid_spans:
            removed.extend(
                (match.start(), match.end())
                for match in self.RESIDUAL_WRAPPER_PATTERN.finditer(text)
            )
        removed.sort()

        parts: list[str] = []
        cursor = start
        removed_before = False
        for span_start, span_end in removed:
            if span_end <= start or span_start >= end:
                continue
            clipped_start = max(start, span_start)
            clipped_end = min(end, span_end)
            if cursor < clipped_start:
                gap = text[cursor:clipped_start]
                # Pretty-printed wire uses newlines between the decorative
                # wrapper and function body. If both neighboring spans are
                # removed, that formatting is markup too, not assistant text.
                if not (removed_before and gap.isspace()):
                    parts.append(gap)
            cursor = max(cursor, clipped_end)
            removed_before = True
        if cursor < end:
            parts.append(text[cursor:end])
        return "".join(parts)

    @staticmethod
    def _scan_calls(
        text: str, request: dict[str, Any] | None
    ) -> list[tuple[str, str, int, int]]:
        """The calls in ``text``, by the shared literal-safe scanner.

        Declared names decide nesting. A call to an undeclared name is kept
        only when it is top-level (outside every declared call's span — the
        scanner's rule) AND framed by the ``<tool_call>`` wrapper. This format
        tolerates a missing wrapper for DECLARED calls, but a bare
        ``<function=example>…</function>`` with a made-up name is just as
        likely prose documenting the format, so without the wrapper it stays
        text — the same rule the qwen3_coder_xml parser applies.
        """
        declared = _declared_tool_names(request)
        return split_marked_calls(
            text,
            r"<function=([^>]+)>",
            "</function>",
            valid_names=declared,
            surface_undeclared=True,
            wrapper="<tool_call>",
        )

    @staticmethod
    def _pending_markup_index(text: str) -> int | None:
        """Return where recognized complete/partial markup begins, if any.

        A bare ``<`` in assistant prose (for example ``2 < 3``) is content,
        not sufficient evidence of a tool-call opener. Only suppress a suffix
        that is already a known marker or can still grow into one.
        """
        markers = ("<function=", "<tool_call>", "</function>", "</tool_call>")
        candidates: list[int] = []
        for marker in markers:
            full = text.find(marker)
            if full != -1:
                candidates.append(full)
            for prefix_len in range(1, len(marker)):
                if text.endswith(marker[:prefix_len]):
                    candidates.append(len(text) - prefix_len)
        return min(candidates) if candidates else None

    @staticmethod
    def _safe_accounting_boundary(text: str) -> int:
        """Return the raw offset safe to account after a close event.

        Plain trailing text is safe through the end. If another markup opener
        has started, stop at the latest complete close so its partial bytes can
        still be released or parsed on a later delta.
        """
        latest_close = 0
        for tag in ("</function>", "</tool_call>"):
            idx = text.rfind(tag)
            if idx != -1:
                latest_close = max(latest_close, idx + len(tag))
        if latest_close == 0:
            return 0
        tail = text[latest_close:]
        pending = NemotronToolParser._pending_markup_index(tail)
        return len(text) if pending is None else latest_close + pending

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from Nemotron model output.
        """
        return self._extract(model_output, request, warn_undeclared=True)

    def _extract(
        self,
        model_output: str,
        request: dict[str, Any] | None,
        *,
        warn_undeclared: bool,
    ) -> ExtractedToolCallInformation:
        """``extract_tool_calls`` with the undeclared-call warning optional.

        The streaming path re-parses the whole accumulated text on every
        close tag, so it warns only for the calls it newly emits rather than
        once per re-parse.
        """
        if "<tool_call>" not in model_output and "<function=" not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        tool_calls = []
        cleaned_text = model_output

        # Positional scan, not ``TOOL_CALL_PATTERN.findall``. The compiled
        # non-greedy patterns above stop at the FIRST closing marker, so a
        # literal ``</function>`` or ``</parameter>`` inside an argument
        # truncated the call and dropped the rest in silence — the defect
        # ``rapid_mlx/api/tool_calling.py`` had, fixed there, and still live
        # here because this is a second implementation of the same format.
        # Both now share ``rapid_mlx/tool_call_scan``; the patterns are kept
        # only as the format's documentation.
        # Declared tool names decide nesting, same as the ``api/tool_calling``
        # path. This is the second implementation of this wire format, so a
        # rule applied only there leaves the other door open: argument text
        # containing ``</function><function=delete_everything>`` would still
        # fabricate a sibling call here. A well-formed TOP-LEVEL call to an
        # undeclared name is surfaced, as there, so the client can answer it
        # with "tool not found" instead of the call silently becoming text.
        declared = _declared_tool_names(request)
        matches = self._scan_calls(model_output, request)
        for func_name, content, _span_start, _span_end in matches:
            func_name = func_name.strip()

            # Try to parse content as JSON first
            content = content.strip()
            if content.startswith("{"):
                try:
                    json.loads(content)
                    tool_calls.append(
                        {
                            "id": generate_tool_id(),
                            "name": func_name,
                            "arguments": content,
                        }
                    )
                    continue
                except json.JSONDecodeError:
                    pass

            # Parse parameter tags
            params = split_marked_parameters(
                content,
                r"<parameter=([^>]+)>",
                "</parameter>",
                valid_names=declared_parameter_names(func_name, request),
                reject_undeclared_siblings=True,
                call_close="</function>",
                call_header="<function=",
            )
            if params is None:
                continue
            if params:
                arguments = {}
                for param_name, param_value in params:
                    # Try to parse value as JSON (for nested objects)
                    try:
                        arguments[param_name] = json.loads(param_value)
                    except json.JSONDecodeError:
                        arguments[param_name] = param_value

                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name,
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    }
                )
            elif content:
                # Raw content without parameter tags
                tool_calls.append(
                    {
                        "id": generate_tool_id(),
                        "name": func_name,
                        "arguments": content,
                    }
                )

        # Clean the text: drop the function bodies and any residual bare
        # <tool_call>/</tool_call> wrapper tags so they don't leak as content.
        if matches:
            # Excise the exact spans the scan identified. A second regex pass
            # would truncate at a different point than the parse did whenever
            # a value holds a literal marker, leaving a tail of the call in
            # the content the user sees.
            for _n, _b, span_start, span_end in matches:
                cleaned_text = cleaned_text.replace(
                    model_output[span_start:span_end], "", 1
                )
            cleaned_text = self.RESIDUAL_WRAPPER_PATTERN.sub("", cleaned_text).strip()

        if tool_calls:
            if warn_undeclared and declared is not None:
                for tc in tool_calls:
                    if tc["name"] not in declared:
                        warn_undeclared_call(tc["name"], declared, "nemotron")
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=cleaned_text if cleaned_text else None,
            )
        else:
            # Diagnostic: a tool-call marker was present but nothing parsed
            # out — i.e. an as-yet-unhandled wire variant. Emit only a
            # STRUCTURAL SUMMARY of the shape, never the raw payload: this is
            # the normal degraded-wire path, and model_output can carry user
            # prompts, tool arguments, or credentials. The counts below are
            # enough to triage the unhandled variant without leaking content.
            has_tool_call_marker = "<tool_call>" in model_output
            function_tag_count = model_output.count("<function=")
            logger.warning(
                "nemotron tool parser: tool-call marker present but no tool "
                "call extracted (possible unhandled variant); "
                "<tool_call> present=%s, %d <function= tags, 0 parseable",
                has_tool_call_marker,
                function_tag_count,
            )
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    _HEADER = re.compile(r"<function=([^>]+)>")
    _WHITESPACE = re.compile(r"\s*")

    def _first_call_header(
        self, text: str, request: dict[str, Any] | None
    ) -> tuple[int, str] | None:
        """``(span_start, name)`` of the first header that is a call, or None.

        A header is a call once something other than another header follows
        it, and its name is declared — or it is a wrapped, top-level call to
        an undeclared name that may surface (not under ``tool_choice`` none or
        a named choice, not inside markdown code). The FIRST call is always
        top-level: nothing before it can hold it as a value.
        """
        declared = _declared_tool_names(request) or frozenset()
        forced = named_tool_choice(request)
        if forced is not None:
            declared = declared & {forced}
        for match in self._HEADER.finditer(text, self._content_upto):
            after = self._WHITESPACE.match(text, match.end()).end()
            rest = text[after : after + len("<parameter=")]
            # A call's header is followed by its first parameter, its close
            # or a JSON body; anything else (``use <function=shell> to …``)
            # is prose naming the format, never a call — sending its header
            # made a turn end in a phantom ``shell`` (review of 7a1d93272).
            if not rest.startswith(("<parameter=", "</function>", "{")):
                if not rest or any(
                    marker.startswith(rest) for marker in ("<parameter=", "</function>")
                ):
                    return None  # what follows the header is not here yet
                continue
            name = match.group(1).strip()
            before = text[: match.start()].rstrip()
            wrapped = before.endswith("<tool_call>")
            span_start = len(before) - len("<tool_call>") if wrapped else match.start()
            if name in declared:
                return span_start, name
            if (
                wrapped
                and may_surface_undeclared(request)
                and not inside_code(text, span_start)
            ):
                return span_start, name
        return None

    def finalize_legacy_raw_stream(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """The calls (and text) a deferred stream withheld, from the full output.

        Called by the postprocessor once the engine stream ends. The first
        call's header is already on the wire; this returns its arguments,
        every later call whole, and the text after the first call that is not
        a call — all from ONE parse of the complete output, so streaming
        reads exactly what ``extract_tool_calls`` reads.
        """
        if not self._deferred:
            return None
        self._deferred = False
        result = self._extract(model_output, request, warn_undeclared=False)
        calls = list(result.tool_calls) if result.tools_called else []
        if not calls:
            # Cut off before the first call closed: its header stands alone,
            # as a cut-off call does on every streaming path.
            return None
        tool_calls: list[dict[str, Any]] = []
        if calls[0]["name"] == self._first_call_name:
            tool_calls.append(
                {"index": 0, "function": {"arguments": calls.pop(0)["arguments"]}}
            )
        else:
            logger.warning(
                "nemotron stream finalize: the full parse's first call is %r, "
                "not the %r whose header streamed",
                calls[0]["name"],
                self._first_call_name,
            )
        declared = _declared_tool_names(request)
        for index, tc in enumerate(calls, start=1):
            if declared is not None and tc["name"] not in declared:
                warn_undeclared_call(tc["name"], declared, "nemotron")
            tool_calls.append(
                {
                    "index": index,
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
            )
        self.current_tool_id = len(tool_calls) - 1
        out: dict[str, Any] = {"tool_calls": tool_calls}
        content = self._visible_content_between(
            model_output, self._content_upto, len(model_output), request
        )
        self._content_upto = len(model_output)
        if content.strip():
            out["content"] = content
        return out

    def flush_held_content(self, full_text: str) -> str:
        """Text withheld after a marker that never became a call.

        The postprocessor calls this at stream end when no call reached the
        wire. Nemotron withholds from the first ``<function=`` / ``<tool_call>``
        until a close; a stream that ends first (prose that merely mentions
        the format, a cut-off) used to lose those bytes. The buffered parser
        returns them as content, and so does the stream now.
        """
        if self._deferred or self._content_upto >= len(full_text):
            return ""
        held = full_text[self._content_upto :]
        self._content_upto = len(full_text)
        return held

    @staticmethod
    def _close_tag_count(text: str) -> int:
        """Number of completed tool-call close tags present in ``text``.

        A call may legitimately close on either ``</function>`` (a truncated
        variant that never emits the wrapper) or ``</tool_call>``. Used by the
        streaming path to detect that a NEW call finished in the latest delta
        (count went up) rather than re-parsing on every chunk.
        """
        return text.count("</function>") + text.count("</tool_call>")

    @staticmethod
    def _clean_trailing_content(current_text: str) -> str | None:
        """The plain-content tail of ``current_text``, or ``None`` if none.

        "Tail" = everything after the last COMPLETE tool-call close tag
        (``</function>`` / ``</tool_call>``). It is content-safe only if it
        contains no ``<`` at all — the moment a ``<`` appears we are (possibly)
        building the next call and must suppress, so no tag (complete or a
        partial fragment like ``"<fun"`` / ``"</fun"``) can ever leak into
        user-visible content.

        Returns:
          * ``None``  — still inside markup (no call closed yet, or a new
            ``<`` has started after the last close): suppress.
          * ``""``    — a call has closed and nothing (yet) follows it.
          * ``str``   — the safe trailing content after the last close.
        """
        end = 0
        for tag in ("</function>", "</tool_call>"):
            idx = current_text.rfind(tag)
            if idx != -1:
                end = max(end, idx + len(tag))
        if end == 0:
            # No close tag yet → we are still inside the (first) call's markup.
            return None
        tail = current_text[end:]
        pending = NemotronToolParser._pending_markup_index(tail)
        return None if pending is not None else tail

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
        Extract tool calls from streaming Nemotron model output.
        """
        if not self._stream_started or not previous_text:
            # The postprocessor's plain-text fast path can forward an opening
            # prefix without invoking this parser. On the first parser call of
            # a reset turn, that prefix is therefore already on the wire and
            # ``previous_text`` is non-empty. Start the watermark there rather
            # than replaying it. ``not previous_text`` also covers callers that
            # reuse an instance without honoring reset().
            self._content_upto = len(previous_text)
            self._stream_started = True
        if "<tool_call>" not in current_text and "<function=" not in current_text:
            # Ordinary prose, forwarded verbatim — already on the wire, so a
            # later refusal must not send it again.
            self._content_upto = len(current_text)
            return {"content": delta_text}

        if self._deferred:
            return None
        if (
            _declared_tool_names(request) is not None
            and isinstance(request, dict)
            and request.get("tool_choice") != "none"
        ):
            # Tools declared: the first call's header goes out as soon as it
            # is one, and everything after it is read ONCE, from the complete
            # output, by ``finalize_legacy_raw_stream`` — the model
            # qwen3_coder_xml's legacy-raw stream uses. A raw value has no
            # escaping, so whether a call closed, ran on into the next one,
            # or holds another call's markup as text is only known from the
            # bytes that come AFTER it (review of 26452cbe3, item 5): emitting
            # each call as its close arrived streamed ``write {"content":
            # "foo"}`` plus a fabricated ``shell`` for a value the buffered
            # parser reads as one ``write``. And because the header is on the
            # wire, a long call is no longer withheld whole, where the #1359
            # budget turned anything over 64 KB into text (item 2).
            first = self._first_call_header(current_text, request)
            if first is None:
                return self._stream_before_first_call(
                    previous_text, current_text, delta_text, request
                )
            span_start, name = first
            content = self._visible_content_between(
                current_text, self._content_upto, span_start, request
            )
            self._content_upto = span_start
            self._deferred = True
            self._first_call_name = name
            self.current_tool_id = 0
            declared = _declared_tool_names(request)
            if declared is not None and name not in declared:
                warn_undeclared_call(name, declared, "nemotron stream")
            out: dict[str, Any] = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": generate_tool_id(),
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    }
                ]
            }
            if content:
                out["content"] = content
            return out

        return self._stream_before_first_call(
            previous_text, current_text, delta_text, request
        )

    def _stream_before_first_call(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        request: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Close-driven streaming: prose, refused blocks, and (with no tools
        declared, or under ``tool_choice`` none) calls emitted as they close."""

        # Trigger from the COMPLETION STATE of current_text, NOT from a close
        # tag appearing inside a single delta_text. We fire only when a NEW
        # close tag finished in this delta — i.e. the close-tag count in
        # current_text ticked up versus previous_text.
        #
        # Counting the delta (rather than testing membership in delta_text) is
        # what makes a close tag split across chunks work: the tokenizer can
        # emit "</fun" then "ction>", so no single delta ever contains the
        # whole "</function>" — but the accumulated current_text does once both
        # fragments arrive, and only then does the count go up. Gating on the
        # *increase* also means we never re-parse current_text on the many
        # trailing deltas after a call has closed (avoiding O(n^2) re-parsing
        # and repeated fail-open WARNINGs on a trailing unparseable marker).
        #
        # extract_tool_calls re-parses current_text and returns ALL complete
        # calls; we de-dupe against the number already streamed (tracked in
        # current_tool_id, which reset() zeroes per request) so each completed
        # call is emitted exactly once even when </function> and </tool_call>
        # arrive in separate deltas (each bumps the count → one re-parse each,
        # but the second finds nothing new to emit).
        if self._close_tag_count(current_text) > self._close_tag_count(previous_text):
            # ``request`` matters here, not just to the non-streaming caller:
            # ``extract_tool_calls`` derives its declared-name gate from it, so
            # omitting it let a name the caller never offered through on the
            # streaming path while the same text was correctly refused when
            # buffered. Agents stream, so the gate was off where it counts.
            result = self._extract(current_text, request, warn_undeclared=False)
            safe_end = self._safe_accounting_boundary(current_text)
            unsent_content = self._visible_content_between(
                current_text, self._content_upto, safe_end, request
            )
            # The source-offset projection includes safe trailing assistant
            # text from this same delta while excluding executable spans.
            if result.tools_called:
                already_emitted = self.current_tool_id + 1
                total = len(result.tool_calls)
                if total > already_emitted:
                    new_calls = result.tool_calls[already_emitted:]
                    self.current_tool_id = total - 1
                    declared = _declared_tool_names(request)
                    for tc in new_calls:
                        if declared is not None and tc["name"] not in declared:
                            warn_undeclared_call(tc["name"], declared, "nemotron")
                    out: dict[str, Any] = {
                        "tool_calls": [
                            {
                                "index": already_emitted + i,
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc["arguments"],
                                },
                            }
                            for i, tc in enumerate(new_calls)
                        ]
                    }
                    if unsent_content:
                        out["content"] = unsent_content
                    # Account only through the latest complete/safe boundary.
                    # A partial following opener still belongs to a future
                    # block and must remain releasable.
                    self._content_upto = safe_end
                    return out
            elif safe_end > self._content_upto:
                # The block CLOSED and is not a call — the declared-name gate
                # refused it. Non-streaming answers that with
                # ``content=model_output``: text the caller never authorised as
                # a tool is still the model's answer and belongs on the wire.
                #
                # Streaming had no equivalent. Every delta of the block was
                # withheld (``None``), and the postprocessor buffers those only
                # until a closing tag arrives — at which point it drops the
                # buffer unemitted (``_tool_suppressed_buffer = ""``) because
                # the parser "made progress". Its #1359 release is byte-budget
                # driven, so a short refused call never trips it and the user
                # gets an EMPTY response instead of the prose.
                #
                # Returning the accumulated text hands the postprocessor the
                # content it is about to discard, so nothing is duplicated.
                # Once per turn: ``</function>`` and ``</tool_call>`` each bump
                # the close count, and the second must not re-send it.
                self._content_upto = safe_end
                if unsent_content:
                    return {"content": unsent_content}
            # Close tag but no NEW call to emit (e.g. the second of </function>
            # + </tool_call> for a call already streamed). Still surface any
            # trailing content that rode in on this delta.
            if safe_end > self._content_upto:
                # The projection is a view of the raw range after the watermark,
                # so a refusal released above cannot be sent twice.
                self._content_upto = safe_end
                if unsent_content:
                    return {"content": unsent_content}
            return None

        # No new call closed in this delta. If we are past all tool-call markup
        # (a call has closed and no new "<" has started since), the delta is
        # trailing assistant content and must pass through instead of being
        # silently dropped. _clean_trailing_content being non-None guarantees no
        # partial or complete tag can leak, so we never emit "<function=",
        # "</function>", or a fragment like "</fun" as user-visible content. We
        # Emit only the source range after the watermark, so already-streamed
        # trailing content is not re-sent while previously withheld partial
        # opener bytes can be recovered when they become ordinary prose.
        if self._clean_trailing_content(current_text) is not None:
            # Release the whole unaccounted visible range, not only this delta.
            # A prior suffix such as ``<fun`` may have been withheld while it
            # could still become ``<function=``; once a later byte turns it
            # into ordinary prose (``<funx``), those earlier bytes belong on
            # the wire too.
            # Everything after the watermark is already known to be outside
            # completed tool spans on this no-new-close path. Slice directly
            # instead of reparsing the full accumulated response for every
            # prose delta (which would make long streams quadratic).
            content = current_text[self._content_upto :]
            self._content_upto = len(current_text)
            return {"content": content} if content else None

        return None
