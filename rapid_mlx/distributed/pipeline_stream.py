# SPDX-License-Identifier: Apache-2.0
"""A streamed chat answer on the pipeline split, as its client receives it (goose Q-178).

goose E2E #5b (3.0.58, Flash on the pipeline split, 2026-09-27): after "Let me
write VENDORED.md." goose's forming panel read "1 tool call forming — Received
so far: 1 chars of arguments · write" for five minutes while the engine had
generated 4,624 tokens.  The route layer's ``StreamingPostProcessor`` sends a
qwen3_coder_xml call's header and ``{``, then — the first time a string value
arrives unquoted (``_legacy_raw_stream``, upstream #1515) — nothing at all
until the answer ends: not the call's arguments, not a later call, not the text
written after it.  A big file and a runaway looked identical, and nobody could
read a word of either.  The tensor split met the same class in mlx_lm's handler
(goose Q-141, Q-146, Q-161); this module is that fix, adapted to this server's
own parser.

``StreamRelay`` sits between the tick loop's tokens and the post-processor, on
rank 0's HTTP side only (nothing here crosses the ranks):

* Q-141: a call the model opens with the ``<tool_call>`` token and names with a
  tool the request declared is streamed by the relay as OpenAI ``tool_calls``
  deltas while it is written (``ToolCallStream``), and never reaches the
  post-processor.  What the relay sends is exact by construction: only what is
  already certain of the parser's own serialization, and when the call closes
  the parser (``extract_tool_calls``, the one ``finalize`` runs) reads the whole
  call — if what was sent is a prefix of its arguments the rest is sent, else
  nothing more is: the client holds unterminated arguments and fails the call
  loudly.  Every other call (an undeclared name, a call the relay cannot read,
  a request whose ``tool_choice`` / ``parallel_tool_calls`` the post-processor
  enforces) goes to the post-processor unchanged; its indices are renumbered
  onto the one sequence the client sees.
* Q-146: ``StreamWatch`` is the request's ``stream`` report (/v1/status) — the
  parser state, what was generated / sent / generated since the last frame, the
  withholding mode, the call being written, the last words — and one
  ``RANK_WITHHELD`` line per span the answer withholds text in.
* Q-161: a call written word for word again in the same answer is held while it
  is still the start of an earlier call, sent the moment it differs, and ends
  the answer unsent if it closes identical (``RANK_TOOL_CALL_REPEATED``); text
  outside the calls that has become one span written over and over ends the
  answer too (``RANK_TEXT_CYCLE``, ``verbatim_cycle``).  Either names itself in
  ``last_engine_stop``.

The parser this mirrors is ``tool_parsers/qwen3coder_tool_parser.py``, not
mlx_lm's: a value ends at the LAST ``</parameter>`` before the next declared
parameter opens (``tool_call_scan.split_marked_parameters``), one wrapping
``\\n`` per side is markup (``trim_wrapping_newlines``), a string value that is
a JSON string is decoded, the exact word ``null`` is None.  Where the streamer
cannot be certain of the parser's reading it stops sending and leaves the call
to the verdict at its close.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

# policy: the reader's window — goose gate 7's "read the last 2000-4000
# characters" (2026-08-30), its smaller end: what one poll of /v1/status or one
# withheld line shows of the words being written.  It bounds a report only.
READER_TAIL_CHARS = 2000

# policy: "mostly" — the share of an answer's text outside its calls that one
# span, written back to back, must cover before the engine ends the answer
# (goose Q-161, the tensor split's CYCLE_SHARE).  E2E #3f: after one ``write``
# the text outside it was 3,251 chars of ``!\n</parameter>\n</function>\n`` and
# then ``!\n</function>\n`` over and over.  A share of the answer's own text.
CYCLE_SHARE = 0.5

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
THINK_START = "<think>"
THINK_END = "</think>"


def verbatim_cycle(text: str) -> tuple[str, int] | None:
    """(unit, copies) when ``text`` ends in one span written back to back at
    least twice — a span holding a line break and something besides whitespace
    — whose copies cover more than CYCLE_SHARE of ``text``; else None.  The
    shortest such span.  goose's rank_stream_watch.py, verbatim."""
    backwards = text[::-1]
    n = len(backwards)
    # z[p]: how far ``text`` read backwards from its end agrees with itself
    # read from p further back (the Z-function of the reversed text).
    z = [0] * n
    left = right = 0
    for i in range(1, n):
        if i < right:
            z[i] = min(right - i, z[i - left])
        while i + z[i] < n and backwards[z[i]] == backwards[i + z[i]]:
            z[i] += 1
        if i + z[i] > right:
            left, right = i, i + z[i]
    for period in range(1, n // 2 + 1):
        copies = (z[period] + period) // period
        if copies < 2 or copies * period <= CYCLE_SHARE * n:
            continue
        unit = text[n - period :]
        if "\n" in unit and unit.strip():
            return unit, copies
    return None


def delta_chars(delta: dict) -> int:
    """The content characters one delta carries to the client: its text, its
    reasoning, and each tool call's name and argument text."""
    chars = len(delta.get("content") or "") + len(delta.get("reasoning_content") or "")
    for call in delta.get("tool_calls") or []:
        function = call.get("function") or {}
        chars += len(function.get("name") or "") + len(function.get("arguments") or "")
    return chars


class ToolCallStream:
    """One call: the text between ``<tool_call>`` and ``</tool_call>``, fed as
    it is generated, read as the qwen3_coder_xml parser reads it.

    ``convert(value, key, config, name)`` is the parser's own value conversion,
    ``arguments_config(name, tools)`` its schema lookup and ``is_string(key,
    config)`` its string-type test."""

    FUNCTION_OPEN = "<function="
    PARAM_OPEN = "<parameter="
    PARAM_CLOSE = "</parameter>"
    FUNCTION_CLOSE = "</function>"
    # A string value's last characters may still be its wrapping "\n" and the
    # start of its close tag: that many stay unsent until the close is seen.
    HOLD = len("\n</parameter>")
    NULL = "null"

    def __init__(self, convert, arguments_config, is_string):
        self._convert = convert
        self._arguments_config = arguments_config
        self._is_string = is_string
        self.text = ""
        self.name = None
        self.sent = ""
        self.broken = None
        self._config = {}
        self._cursor = 0
        self._phase = "head"
        self._key = None
        self._value_start = 0
        self._string = False
        self._value_sent = 0
        self._keys = []
        # Where the markup the streamer waits on starts (``stray``).
        self._markup = 0

    def feed(self, piece: str, tools) -> tuple[str | None, str]:
        """Append generated text; (name when the call just opened else None, fragment)."""
        self.text += piece
        opened = None
        fragments = []
        while self.broken is None:
            step = self._step(tools)
            if step is None:
                break
            name, fragment = step
            if name is not None:
                opened = name
            fragments.append(fragment)
        fragment = "".join(fragments)
        self.sent += fragment
        return opened, fragment

    def close(self, parse) -> tuple[str | None, str | None]:
        """The call ended.  ``parse(text)`` is the parser's (name, arguments
        JSON) or raises its refusal.  Returns (the rest of the arguments, None)
        when what was sent is a prefix of them; (None, why) when it is not."""
        try:
            name, whole = parse(self.text)
        except (
            Exception
        ) as refusal:  # the parser's verdict is the call's: said, never swallowed
            return (
                None,
                f"the parser refused the call: {type(refusal).__name__}: {refusal}",
            )
        if name != self.name:
            return None, f"the parser named {name!r}, the stream opened {self.name!r}"
        if not whole.startswith(self.sent):
            return (
                None,
                self.broken
                or "what was streamed is not a prefix of the parsed arguments",
            )
        rest = whole[len(self.sent) :]
        self.sent = whole
        return rest, None

    def position(self) -> dict[str, Any]:
        """Where the streamer is reading (Q-146): the phase (head: the function
        header; between: waiting for a parameter or ``</function>``; key: a
        parameter's name; value: its value), the open parameter, whether its
        value streams, and the streamer's verdict when it stopped reading."""
        value = self._phase == "value"
        return {
            "name": self.name,
            "phase": self._phase,
            "parameter": self._key if value else None,
            "string_value": self._string if value else None,
            "broken": self.broken,
            "sent_chars": len(self.sent),
        }

    def typed_value_open(self) -> str | None:
        """The open parameter whose value is sent only when it closes, or None."""
        if self.broken is None and self._phase == "value" and not self._string:
            return self._key
        return None

    def stray(self) -> str | None:
        """Text where the streamer waits for the qwen3_coder frame (``<function=``
        at the head, ``<parameter=`` or ``</function>`` between parameters) that
        is not that frame — of which the streamer reads, and sends, nothing.
        None while the text is the frame or a prefix of it."""
        if self.broken is not None or self._phase not in ("head", "between"):
            return None
        waiting = self.text[self._markup :].lstrip()
        if self._phase == "head":
            frames = (self.FUNCTION_OPEN,)
        else:
            frames = (self.PARAM_OPEN, self.FUNCTION_CLOSE)
        if not waiting or any(
            f.startswith(waiting) or waiting.startswith(f) for f in frames
        ):
            return None
        return waiting

    def _step(self, tools):
        if self._phase == "head":
            return self._head(tools)
        if self._phase == "between":
            return self._between()
        if self._phase == "key":
            return self._read_key()
        return self._value()

    def _head(self, tools):
        opener = self.text.find(self.FUNCTION_OPEN)
        if opener < 0:
            return None
        start = opener + len(self.FUNCTION_OPEN)
        end = self.text.find(">", start)
        if end < 0:
            return None
        self.name = self.text[start:end]
        self._config = self._arguments_config(self.name, tools)
        self._cursor = end + 1
        self._markup = self._cursor
        self._phase = "between"
        return self.name, "{"

    def _between(self):
        opener = self.text.find(self.PARAM_OPEN, self._cursor)
        if opener < 0:
            self._cursor = max(self._cursor, len(self.text) - len(self.PARAM_OPEN) + 1)
            return None
        self._cursor = opener + len(self.PARAM_OPEN)
        self._phase = "key"
        return None, ""

    def _read_key(self):
        end = self.text.find(">", self._cursor)
        if end < 0:
            return None
        written = self.text[self._cursor : end]
        if self.PARAM_CLOSE[:-1] in written:
            self.broken = "a parameter header closed without its name's '>'"
            return None
        key = written.strip()
        if self._config and key not in self._config:
            # The parser reads an undeclared opener as a sibling of nothing: it
            # is text inside the value before it, or a value of its own.
            self.broken = f"parameter {key!r} is not in the tool's schema"
            return None
        if key in self._keys:
            self.broken = f"parameter {key!r} written twice"
            return None
        self._key = key
        self._value_start = end + 1
        self._value_sent = 0
        self._string = self._is_string(key, self._config)
        self._phase = "value"
        return None, ""

    def _key_prefix(self) -> str:
        separator = ", " if self._keys else ""
        return f"{separator}{json.dumps(self._key, ensure_ascii=False)}: "

    def _ends_here(self, after: int) -> bool | None:
        """Whether the value's first ``</parameter>`` (ending at ``after``) is
        where the parser ends it: True when what follows is the next declared
        parameter or ``</function>``, None while that is not yet written, False
        when it is anything else (the parser reads on to a later close)."""
        waiting = self.text[after:].lstrip()
        if not waiting:
            return None
        if self.FUNCTION_CLOSE.startswith(waiting) or self.PARAM_OPEN.startswith(
            waiting
        ):
            return None
        if waiting.startswith(self.FUNCTION_CLOSE):
            return True
        if waiting.startswith(self.PARAM_OPEN):
            end = waiting.find(">", len(self.PARAM_OPEN))
            if end < 0:
                return None
            key = waiting[len(self.PARAM_OPEN) : end].strip()
            return not self._config or key in self._config
        return False

    def _value(self):
        close = self.text.find(self.PARAM_CLOSE, max(self._value_start, self._cursor))
        if close < 0:
            self._cursor = max(
                self._value_start, len(self.text) - len(self.PARAM_CLOSE) + 1
            )
            return self._string_increment(len(self.text) - self.HOLD)
        after = close + len(self.PARAM_CLOSE)
        ends = self._ends_here(after)
        if ends is None:
            # The value runs at least to this close; its last "\n" may be markup.
            limit = close - 1 if self.text[close - 1] == "\n" else close
            return self._string_increment(limit)
        if ends is False:
            self.broken = (
                f"parameter {self._key!r}: text after its </parameter> is neither the "
                "next parameter nor </function>, so the parser reads the value to a later close"
            )
            return None
        value = self.text[self._value_start : close]
        if value.startswith("\n"):
            value = value[1:]
        if value.endswith("\n"):
            value = value[:-1]
        if self._string and self._value_sent:
            fragment = (
                json.dumps(value[self._value_sent :], ensure_ascii=False)[1:-1] + '"'
            )
        else:
            try:
                converted = self._convert(value, self._key, self._config, self.name)
            except (
                Exception
            ) as refusal:  # the parser refuses the same value when the call ends
                self.broken = (
                    f"parameter {self._key!r}: {type(refusal).__name__}: {refusal}"
                )
                return None
            fragment = self._key_prefix() + json.dumps(converted, ensure_ascii=False)
        self._keys.append(self._key)
        self._cursor = after
        self._markup = after
        self._phase = "between"
        return None, fragment

    def _string_increment(self, limit: int):
        """The certain part of an open string value up to ``limit``: nothing
        until it is more than the word "null" (exactly that word is None), none
        of it when its first visible character is a quote (the parser decodes
        a value that is a JSON string), else all of it."""
        if not self._string:
            return None
        lead = 1 if self.text.startswith("\n", self._value_start) else 0
        start = self._value_start + lead
        certain = self.text[start:limit] if limit > start else ""
        if not self._value_sent:
            visible = certain.lstrip()
            if not visible:
                return None
            if visible.startswith('"'):
                self._string = False
                return None
        if len(certain) <= (self._value_sent or len(self.NULL)):
            return None
        piece = json.dumps(certain[self._value_sent :], ensure_ascii=False)[1:-1]
        prefix = "" if self._value_sent else self._key_prefix() + '"'
        self._value_sent = len(certain)
        return None, prefix + piece


class StreamWatch:
    """One streamed chat request's ``stream`` report (goose Q-146) and its
    withheld-span lines.  ``say(tag, payload)`` prints a line (serve's
    ``emit``)."""

    def __init__(self, say, request_id: str):
        self._say = say
        self.request_id = request_id
        self.state = None
        self.generated_chars = 0
        self.sent_chars = 0
        self.since_sent_chars = 0
        self.content_frames = 0
        self.call_chars = 0
        self.tail = ""
        # The relay's reading of the call being written, or why none is.
        self.streamer = None
        self.unstreamed = None
        self.holding = None
        self.episode = None
        # Why the engine ended this answer itself (Q-161), else None.
        self.stop = None

    def take(self, state: str, piece: str) -> None:
        if state == "tool" and self.state != "tool":
            self.call_chars = 0
        self.state = state
        self.tail = (self.tail + piece)[-READER_TAIL_CHARS:]
        self.generated_chars += len(piece)
        self.since_sent_chars += len(piece)
        if state == "tool":
            self.call_chars += len(piece)
        if self.episode is not None:
            self.episode["withheld_chars"] += len(piece)

    def sent(self, delta: dict) -> None:
        chars = delta_chars(delta)
        if chars:
            self.sent_chars += chars
            self.since_sent_chars = 0
            self.content_frames += 1

    def withholding(self) -> tuple[str, str] | None:
        """(mode, reason) while the answer is in a state that withholds its text."""
        if self.state != "tool":
            return None
        if self.streamer is None:
            return (
                "tool_not_streamed",
                self.unstreamed or "no tool-call streamer reads this call",
            )
        if self.holding is not None:
            return ("tool_repeat_held", self.holding)
        if self.streamer.broken is not None:
            return (
                "tool_broken",
                f"the streamer stopped reading the call: {self.streamer.broken}; the rest "
                "is sent when the call closes, if the parser reads what was sent",
            )
        if self.streamer.stray() is not None:
            return (
                "tool_unread",
                f"the call's text at its {self.streamer.position()['phase']} is not the "
                "qwen3_coder frame (<function=NAME>, <parameter=KEY>), so the streamer sends none of it",
            )
        typed = self.streamer.typed_value_open()
        if typed is not None:
            return (
                "tool_typed_value",
                f"parameter {typed!r} is not a streamed string: its value is sent whole "
                "when </parameter> closes it",
            )
        return None

    def settle(self) -> None:
        mode = self.withholding()
        current = None
        if self.episode is not None:
            current = (self.episode["mode"], self.episode["reason"])
        if mode == current:
            return
        if self.episode is not None:
            self._leave("left")
        if mode is not None:
            self.episode = {
                "mode": mode[0],
                "reason": mode[1],
                "withheld_chars": self.since_sent_chars,
            }
            self._say(
                "RANK_WITHHELD",
                {
                    "request_id": self.request_id,
                    "event": "enter",
                    "mode": mode[0],
                    "reason": mode[1],
                    "generated_chars": self.generated_chars,
                    "sent_chars": self.sent_chars,
                    "since_sent_chars": self.since_sent_chars,
                },
            )

    def end(self) -> None:
        self.settle()
        if self.episode is not None:
            self._leave("request ended")

    def _leave(self, how: str) -> None:
        episode, self.episode = self.episode, None
        self._say(
            "RANK_WITHHELD",
            {
                "request_id": self.request_id,
                "event": "leave",
                "how": how,
                "mode": episode["mode"],
                "reason": episode["reason"],
                "withheld_chars": episode["withheld_chars"],
                "since_sent_chars": self.since_sent_chars,
                "generated_chars": self.generated_chars,
                "sent_chars": self.sent_chars,
                "tail": self.tail,
            },
        )

    def report(self) -> dict[str, Any]:
        """The ``stream`` block of the request's /v1/status row."""
        call = None
        if self.state == "tool":
            call = {
                "streamed": self.streamer is not None,
                "call_chars": self.call_chars,
            }
            if self.streamer is not None:
                call.update(self.streamer.position())
            else:
                call["why_not_streamed"] = self.unstreamed
        episode = self.episode
        return {
            "parser_state": self.state,
            "generated_chars": self.generated_chars,
            "sent_chars": self.sent_chars,
            "since_sent_chars": self.since_sent_chars,
            "content_frames": self.content_frames,
            "withholding": None if episode is None else dict(episode),
            "tool_call": call,
            "stop": self.stop,
            "tail": self.tail,
            "tail_window_chars": READER_TAIL_CHARS,
        }


class Markers:
    """The token ids that move an answer between reasoning, text and a call —
    the checkpoint's own single-token markers, or None when one is not."""

    def __init__(self, think_start, think_end, call_start, call_end):
        self.think_start = think_start
        self.think_end = think_end
        self.call_start = call_start
        self.call_end = call_end

    @classmethod
    def of(cls, tokenizer) -> Markers:
        from ..xml_tool_close_guard import _single_token_id

        if not callable(getattr(tokenizer, "encode", None)):
            # xml_close_guard_spec's own test: no encoder, no single tokens.
            return cls(None, None, None, None)
        return cls(
            *(
                _single_token_id(tokenizer, text)
                for text in (THINK_START, THINK_END, TOOL_CALL_START, TOOL_CALL_END)
            )
        )


class _Call:
    """One call the relay reads: its streamer, its id, its client index (None
    while held), and the pieces it holds back from the post-processor until
    its name says who owns it."""

    def __init__(self, stream: ToolCallStream):
        self.stream = stream
        self.id = f"call_{uuid.uuid4().hex[:8]}"
        self.index = None
        self.owned = None
        self.pieces: list[str] = []
        self.held: list[dict] = []


class StreamRelay:
    """One streamed chat answer between the tick loop and the post-processor.

    ``take(token, piece)`` returns what the answer does with one generated
    piece, in order: ``("text", s)`` for the post-processor, ``("call", d)`` for
    a tool-call delta the relay sends itself.  ``remap(deltas)`` renumbers the
    post-processor's own calls onto the client's one sequence.  ``stop`` is set
    when the engine ends the answer itself (Q-161); the piece that tripped it is
    not handed on.

    ``owns`` is None when the relay streams calls; else why it does not (the
    post-processor then gets every piece, and the relay only reads the state).
    ``cycles``: the text-cycle stop reads this answer (a tool request's).
    """

    def __init__(
        self,
        watch: StreamWatch,
        markers: Markers,
        *,
        reasoning: bool,
        parser=None,
        request: dict | None = None,
        owns: str | None = None,
        cycles: bool = False,
        say=None,
        record_stop=None,
    ):
        self.watch = watch
        self.markers = markers
        self.state = "reasoning" if reasoning else "normal"
        self.parser = parser
        self.request = request or {}
        self.tools = self.request.get("tools")
        self.owns = owns
        self.cycles = cycles
        self._say = say or (lambda tag, payload: None)
        self._record_stop = record_stop or (lambda stop: None)
        self.declared = set()
        if owns is None:
            self.declared = parser._declared_tool_names(self.request)
        self.call: _Call | None = None
        # The text of every call this answer closed, in order.
        self.closed: list[str] = []
        self.outside = ""
        self.next_index = 0
        self.upstream_index: dict[int, int] = {}
        self.calls_sent = 0
        self.stop = None
        watch.unstreamed = owns

    # ------------------------------------------------------------------ pieces

    def take(self, token: int | None, piece: str) -> list[tuple[str, Any]]:
        out: list[tuple[str, Any]] = []
        marker = self._marker(token)
        before, framing = piece, ""
        if marker is not None and piece.endswith(marker[1]):
            # A held space reaches the client with the next token's text: what
            # precedes the marker belongs to the state before it.
            before, framing = piece[: -len(marker[1])], marker[1]
        if before:
            self._text(before, out)
            if self.stop is not None:
                return out
        if marker is None:
            return out
        kind = marker[0]
        if kind == "call_start" and self.state == "normal":
            self.state = "tool"
            self.watch.take("tool", framing)
            self._open(framing, out)
        elif kind == "call_end" and self.state == "tool":
            self.watch.take("tool", framing)
            self._close(framing, out, closed=True)
            self.state = "normal"
            self.watch.state = "normal"
        elif kind == "think_end" and self.state == "reasoning":
            self._text(framing, out)
            self.state = "normal"
            self.watch.state = "normal"
        else:
            if kind == "think_start" and self.state == "normal":
                self.state = "reasoning"
            # A marker where the state admits none is text of that state.
            self._text(framing, out)
        return out

    def finish(self) -> list[tuple[str, Any]]:
        """The answer ended: a call still open is read on what was written —
        as the post-processor reads a call the answer cut, with no close."""
        out: list[tuple[str, Any]] = []
        if self.call is not None:
            self._close("", out, closed=False)
            self.state = "normal"
        return out

    def _marker(self, token: int | None):
        m = self.markers
        for kind, token_id, text in (
            ("call_start", m.call_start, TOOL_CALL_START),
            ("call_end", m.call_end, TOOL_CALL_END),
            ("think_start", m.think_start, THINK_START),
            ("think_end", m.think_end, THINK_END),
        ):
            if token is not None and token_id is not None and token == token_id:
                return kind, text
        return None

    def _text(self, text: str, out) -> None:
        self.watch.take(self.state, text)
        if self.state == "tool":
            self._feed(text, out)
            return
        out.append(("text", text))
        if self.state == "normal" and self.cycles and self._cycled(text):
            out.pop()

    # ------------------------------------------------------------------- calls

    def _open(self, framing: str, out) -> None:
        if self.owns is not None:
            out.append(("text", framing))
            return
        from ..tool_parsers.qwen3coder_tool_parser import (
            _convert_param_value,
            _get_arguments_config,
            _is_string_param,
        )

        self.call = _Call(
            ToolCallStream(
                _convert_param_value, _get_arguments_config, _is_string_param
            )
        )
        self.call.pieces.append(framing)
        self.watch.streamer = self.call.stream

    def _handoff(self, call: _Call, why: str, out) -> None:
        """The call is the post-processor's: every piece it held goes there."""
        call.owned = False
        self.watch.streamer = None
        self.watch.unstreamed = why
        out.extend(("text", piece) for piece in call.pieces)
        call.pieces = []

    def _feed(self, text: str, out) -> None:
        call = self.call
        if call is None or call.owned is False:
            out.append(("text", text))
            return
        if call.owned is None:
            call.pieces.append(text)
        opened, fragment = call.stream.feed(text, self.tools)
        if call.owned is None:
            if opened is not None:
                if opened not in self.declared:
                    self._handoff(
                        call,
                        f"the call names {opened!r}, which this request did not declare: "
                        "the post-processor refuses it as text",
                        out,
                    )
                    return
                call.owned = True
                call.pieces = []
            elif call.stream.stray() is not None:
                self._handoff(
                    call,
                    "the call is not the qwen3_coder frame (<function=NAME>): the "
                    "post-processor reads it when the answer ends",
                    out,
                )
                return
            else:
                return
        frame = None
        if opened is not None:
            frame = {
                "id": call.id,
                "type": "function",
                "function": {"name": opened, "arguments": fragment},
            }
        elif fragment:
            frame = {"function": {"arguments": fragment}}
        if call.index is not None:
            if frame is not None:
                self._send(call.index, frame, out)
            return
        if frame is not None:
            call.held.append(frame)
        position = self._repeat_of(call.stream.text, whole=False)
        if position is None:
            self._release(call, out)
        else:
            self.watch.holding = (
                f"the call so far is word for word the start of call {position} of this "
                "answer: it is sent the moment it differs, and ends the answer unsent if "
                "it closes identical"
            )

    def _close(self, framing: str, out, *, closed: bool) -> None:
        call, self.call = self.call, None
        if call is None:
            out.append(("text", framing))
            return
        self.watch.streamer = None
        self.watch.holding = None
        if call.owned is not True:
            if call.owned is None:
                self._handoff(
                    call,
                    "the call closed before it named a tool: the post-processor reads it",
                    out,
                )
            out.append(("text", framing))
            self.watch.unstreamed = self.owns
            return
        stream = call.stream
        position = self._repeat_of(stream.text, whole=True)
        self.closed.append(stream.text)
        if position is not None:
            self._stop(
                "tool_call_repeated",
                {
                    "name": stream.name,
                    "repeat_of": position,
                    "calls": len(self.closed) - 1,
                    "chars": len(stream.text),
                    "generated_chars": self.watch.generated_chars,
                    "tail": self.watch.tail,
                },
            )
            return
        if call.index is None:
            self._release(call, out)
        rest, why = stream.close(
            lambda text: self._parse(text + (TOOL_CALL_END if closed else ""))
        )
        if why is not None:
            self._say(
                "RANK_TOOL_CALL_UNPARSED",
                {
                    "request_id": self.watch.request_id,
                    "name": stream.name,
                    "why": why,
                    "chars": len(stream.text),
                    "sent_chars": len(stream.sent),
                },
            )
            return
        if rest:
            self._send(call.index, {"function": {"arguments": rest}}, out)

    def _parse(self, text: str) -> tuple[str, str]:
        result = self.parser.extract_tool_calls(TOOL_CALL_START + text, self.request)
        if not result.tools_called:
            raise ValueError("it reads no call from the block")
        if len(result.tool_calls) != 1:
            raise ValueError(f"it reads {len(result.tool_calls)} calls from one block")
        parsed = result.tool_calls[0]
        return parsed["name"], parsed["arguments"]

    def _repeat_of(self, text: str, whole: bool) -> int | None:
        for position, earlier in enumerate(self.closed, start=1):
            if earlier == text if whole else earlier.startswith(text):
                return position
        return None

    def _release(self, call: _Call, out) -> None:
        call.index = self.next_index
        self.next_index += 1
        self.calls_sent += 1
        self.watch.holding = None
        for frame in call.held:
            self._send(call.index, frame, out)
        call.held = []

    def _send(self, index: int, frame: dict, out) -> None:
        out.append(("call", {"index": index, **frame}))

    def remap(self, calls: list[dict]) -> list[dict]:
        """The post-processor's own calls, on the client's index sequence."""
        mapped = []
        for call in calls:
            upstream = call.get("index", 0)
            if upstream not in self.upstream_index:
                self.upstream_index[upstream] = self.next_index
                self.next_index += 1
                self.calls_sent += 1
            mapped.append({**call, "index": self.upstream_index[upstream]})
        return mapped

    # ------------------------------------------------------------------- stops

    def _cycled(self, text: str) -> bool:
        """True when the answer's text outside its calls has become one span
        written over and over (Q-161, E2E #3f): the engine names it and the
        answer ends.  A span holds a line break, so the text is read when one
        arrives."""
        self.outside += text
        if "\n" not in text:
            return False
        cycle = verbatim_cycle(self.outside)
        if cycle is None:
            return False
        unit, copies = cycle
        self._stop(
            "text_cycle",
            {
                "unit": unit,
                "copies": copies,
                "outside_chars": len(self.outside),
                "calls": len(self.closed),
                "generated_chars": self.watch.generated_chars,
                "tail": self.watch.tail,
            },
        )
        return True

    def _stop(self, reason: str, detail: dict) -> None:
        stop = {"request_id": self.watch.request_id, "reason": reason, **detail}
        self.stop = stop
        self.watch.stop = stop
        self._record_stop(stop)
        self._say(f"RANK_{reason.upper()}", stop)
