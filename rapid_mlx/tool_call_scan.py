# SPDX-License-Identifier: Apache-2.0
"""Literal-safe scanning for marker-delimited tool-call wire formats.

XML-ish tool-call formats — Nemotron's ``<function=…><parameter=…>``,
Qwen 3.6's ``<tool_call>`` body, MiniCPM's ``<param name=…>`` — carry no
escaping layer. A value is whatever sits between two markers, which makes
the obvious implementation wrong in the same way every time::

    re.findall(r"<parameter=(\\w+)>(.*?)</parameter>", text, re.DOTALL)

The non-greedy ``(.*?)`` stops at the FIRST ``</parameter>``. When that
marker also occurs *inside* a value — an agent writing documentation about
tool calling, a diff of a parser, a shell heredoc — the match ends early,
the remaining JSON is truncated, and the call is dropped with no error and
no log line. That is jundot/omlx#2507; the silence is omlx#2420.

The resolution these formats actually use is positional: an element ends at
the LAST closing marker before the next sibling opens. This module is that
rule, written once.

It lives at package top level with no imports beyond ``re`` on purpose.
``rapid_mlx/api/tool_calling.py`` and the parsers under
``rapid_mlx/tool_parsers/`` both need it, and those two packages already
import each other (``api/__init__`` pulls in ``tool_calling``;
``tool_parsers/__init__`` eagerly loads ``qwen3coder`` which imports back
into ``api``). Anything shared between them has to sit outside both, or
the cycle closes.

Before this existed the rule was reimplemented per format, which is how
the same defect could be fixed in ``tool_calling.py`` and still be live in
``nemotron_tool_parser.py``.

Which call a ``<function=…>`` header belongs to is STRUCTURE, read once here
(``call_shape`` / ``top_level_calls``) for qwen3_coder_xml, nemotron and the
scanner fallback alike: each value is skipped by its own close (a JSON
string by its closing quote; a raw value that opened on the template's grid
by its first line-start close), and a header after the call's close is a
sibling unless the bytes say the value may still be running. Where the text
format genuinely cannot tell, the safer reading wins — one call keeps the
bytes as its value, and no extra call is made.
"""

from __future__ import annotations

import bisect
import logging
import re
from collections.abc import Iterable
from typing import Any

__all__ = [
    "CallShape",
    "TopCall",
    "top_level_calls",
    "HEADER_FIRST",
    "call_shape",
    "cut_call_name",
    "declared_tool_names",
    "find_value_close",
    "inside_code",
    "may_surface_undeclared",
    "named_tool_choice",
    "scan_marked_parameters",
    "segment_by_next_opener",
    "split_marked_calls",
    "split_marked_parameters",
    "trailing_cut_call",
    "trim_wrapping_newlines",
    "warn_undeclared_call",
]

_logger = logging.getLogger(__name__)


def warn_undeclared_call(name: str, declared: Iterable[str] | None, where: str) -> None:
    """Say, loudly, that a call to a tool the request never offered is passing.

    A well-formed top-level call to an undeclared name is the MODEL's mistake,
    and the client is the party that has to answer it ("tool not found") so
    the model can recover. Folding it into text instead ends the agent turn
    with a meaningless final answer and leaves no trace of why. Passing it
    on is right; passing it on silently is not, so every such call is logged
    once, at the point it is emitted, under one greppable tag.
    """
    _logger.warning(
        "[TOOL-CALL-UNDECLARED] %s: the model called %r, which the request did "
        "not offer (declared: %s); passing it to the client as a tool call so "
        "the client can answer it with a tool error",
        where,
        name,
        sorted(declared or ()),
    )


def trim_wrapping_newlines(value: str) -> str:
    r"""Remove the ONE newline the wire puts on each side of a value.

    The Qwen3-Coder chat template renders a parameter as
    ``<parameter=NAME>\n`` + value + ``\n</parameter>\n``, so exactly one
    newline per side is markup and everything else -- including the
    indentation of the FIRST line -- is payload. Stripping all surrounding
    whitespace instead silently de-indented the opening line of every value,
    which for a coding agent's ``new_string`` means the edit no longer
    compiles or, worse, changes scope while still parsing (#3401).

    This matches the two mature engines that serve this wire format: vLLM's
    ``_trim_wrapping_newlines`` (``vllm/parser/qwen3.py``) and SGLang's
    ``qwen3_coder_detector`` both drop one leading and one trailing ``\n``
    and nothing else, in their streaming and non-streaming paths alike.
    ``\r\n`` is deliberately NOT treated as a two-byte wrapper. The template
    frames with ``\n``, so under LF framing a payload that ends in ``\r``
    arrives as ``"\ntext\r\n"`` -- indistinguishable from a CRLF-framed
    ``"text"``. Consuming both bytes would delete a payload byte to tidy up
    markup that this wire does not actually emit. Removing one ``\n`` leaves
    ``"text\r"``, which is correct under LF framing and merely leaves a
    visible stray ``\r`` under the CRLF framing nothing here produces. When
    the two readings cannot be told apart, keep the byte.

    Same lesson as ``_decode_json_like`` in ``api/tool_calling.py``: on this
    wire, whitespace around a value is only safe to remove where the format
    says that whitespace is markup.
    """
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def declared_tool_names(request: dict[str, Any] | None) -> frozenset[str] | None:
    """Tool names the request actually declared, or ``None`` if it declared none.

    ``None`` rather than an empty set, because the two mean different things
    to the scanners: an empty set would reject every opener, while ``None``
    selects the position-only rules. A request with no tools cannot execute
    anything anyway, so there is nothing to protect there and no reason to
    change how its text is parsed.
    """
    if not isinstance(request, dict):
        return None
    tools = request.get("tools")
    if not isinstance(tools, list):
        return None
    names = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.add(function["name"])
    return frozenset(names) or None


def _next_sibling(
    text: str,
    openers: list[re.Match[str]],
    index: int,
    closer: str,
    valid_names: frozenset[str] | set[str] | None = None,
) -> int:
    """Index of the next opener that is a real SIBLING of ``openers[index]``.

    Two filters:

    * a ``closer`` must appear between the two openers — an element cannot
      begin before the previous one ends;
    * with ``valid_names`` given, the opener's name must be one the request
      declared.

    The name check resolves what syntax alone cannot. In
    ``<parameter=a>before </parameter> text <parameter=fake> after
    </parameter>`` the closer really is there before the second opener, so
    a position-only reading fabricates ``fake``. Nothing in these
    escaping-free formats distinguishes that from two real elements; only
    the schema does. Concretely, the case this exists for::

        <parameter=code>if x: print("</parameter><parameter=evil>")</parameter>

    An agent writing code or documentation ABOUT tool calling emits exactly
    that. With the filter, ``code`` keeps its whole value. Without it the
    value is truncated at ``print("`` and a phantom ``evil`` is handed to
    the tool as though the model had asked for it.

    Repeated names are NOT filtered. Two real ``<parameter=body>`` elements
    are the caller's to resolve — dict assignment makes it last-value-wins,
    which is the established behaviour — and suppressing the second opener
    instead merges its wire markup into the first value, which corrupts
    both.

    Returns ``len(openers)`` when no sibling follows.
    """
    for j in range(index + 1, len(openers)):
        if valid_names is not None and openers[j].group(1).strip() not in valid_names:
            continue  # not a declared name — literal text inside the value
        if closer in text[openers[index].end() : openers[j].start()]:
            return j
    return len(openers)


def segment_by_next_opener(
    text: str,
    openers: list[re.Match[str]],
    index: int,
    closer: str,
    valid_names: frozenset[str] | set[str] | None = None,
) -> tuple[str, int] | None:
    """``(body, end_offset)`` for ``openers[index]``, or ``None`` if unclosed.

    Bounded by the next *sibling* opener, then trimmed at the LAST ``closer``
    in that span. The two rules cover the two ways these escaping-free
    formats can be misread:

    * ending at the first ``closer`` truncates a value that legitimately
      contains one (jundot/omlx#2507);
    * ending at the next *textual* opener truncates a value that contains
      one and fabricates an element.

    ``None`` for an element with no ``closer`` at all: a truncated or
    malformed emission is rejected rather than silently swallowing the rest
    of the buffer as its value. The regexes this replaced also required the
    closing marker to match, so accepting them would be a behaviour change,
    not a fix.
    """
    sibling = _next_sibling(text, openers, index, closer, valid_names)
    end = openers[sibling].start() if sibling < len(openers) else len(text)
    body = text[openers[index].end() : end]
    cut = body.rfind(closer)
    if cut == -1:
        return None
    return body[:cut], openers[index].end() + cut + len(closer)


# ---------------------------------------------------------------------------
# Structure: where a call ends, and whether a later opener is its sibling.
# ---------------------------------------------------------------------------

PARAMETER_OPEN = "<parameter="
PARAMETER_CLOSE = "</parameter>"


def _json_string_end(text: str, quote: int) -> int:
    """Offset just past the JSON string opened at ``text[quote]``, or -1."""
    i = quote + 1
    escaped = False
    n = len(text)
    while i < n:
        char = text[i]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return i + 1
        i += 1
    return -1


def find_value_close(
    text: str, value_start: int, close: str = PARAMETER_CLOSE
) -> tuple[int, bool]:
    """``(offset of the close marker that ends this value, or -1; quoted)``.

    The structural rule both XML function-call families share (it began as
    ``Qwen3CoderToolParser._find_parameter_close``). A grammar-constrained
    value is a JSON string (#1542), so marker text inside its quotes is data
    and the close is the first marker after the closing quote. A raw value
    has no escaping at all; structurally it closes at its FIRST close
    marker, and whether that close can be trusted is a separate question
    (``CallShape.canonical`` / ``CallShape.consistent``). -1 when the value
    has not closed — including a JSON string still open, which on a stream
    means "wait", never "fall back to the raw rule".
    """
    i = value_start
    n = len(text)
    while i < n and text[i].isspace():
        i += 1
    if i >= n or text[i] != '"':
        return text.find(close, i), False
    end = _json_string_end(text, i)
    if end == -1:
        return -1, True
    return text.find(close, end), True


class CallShape:
    """How one call reads STRUCTURALLY: its values skipped by their own close.

    ``end``         offset just past the call's first top-level closer.
    ``free``        some value is raw text (or the body is not parameters at
                    all), so its bytes could legitimately run past ``end``.
    ``canonical``   every raw value closed at a line start, as the template
                    writes it (``VALUE\\n</parameter>``). A mid-line close may
                    be a value quoting the marker.
    ``consistent``  every raw value that OPENED on the template's grid
                    (``<parameter=K>\\n``) also closed on it. Such a value
                    closes at its first LINE-START ``</parameter>`` — the
                    template writes every close so, and
                    ``xml_tool_close_guard`` holds the model to it — so a
                    marker it quotes mid-line (``grep "</parameter>" f``) is
                    payload. Only when another line-start marker comes first
                    does the read fall back to the first close, and the call
                    is then inconsistent: the value may still be running.
    """

    __slots__ = ("end", "free", "canonical", "consistent", "stray")

    def __init__(
        self,
        end: int,
        free: bool,
        canonical: bool,
        consistent: bool,
        stray: bool = False,
    ):
        self.end = end
        self.free = free
        self.canonical = canonical
        self.consistent = consistent
        # A ``</parameter>`` OUTSIDE every value of this call (between its
        # parameters, or in a body that has none). A well-formed call never
        # has one; a header that is really text inside an earlier value does
        # (``… <function=tree> ok\n</parameter>\n</function>`` — the close is
        # the enclosing value's). Such a header is never a sibling.
        self.stray = stray

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"CallShape(end={self.end}, free={self.free}, "
            f"canonical={self.canonical}, consistent={self.consistent}, "
            f"stray={self.stray})"
        )


# A header that meets ANOTHER header before any parameter or close of its own
# (``use <function=NAME> to …`` in prose ahead of a real call) is not a call.
HEADER_FIRST = "header-first"


def call_shape(
    text: str,
    body_start: int,
    closer: str,
    *,
    header: str | None = None,
    param_open: str = PARAMETER_OPEN,
    param_close: str = PARAMETER_CLOSE,
) -> CallShape | str | None:
    """Read the call whose header ends at ``body_start`` structurally.

    Returns a ``CallShape``, ``HEADER_FIRST`` when another ``header`` arrives
    before this call has a parameter or a close, or ``None`` when it never
    closes in ``text``.
    """
    j = body_start
    free = False
    canonical = True
    consistent = True
    stray = False
    saw_param = False
    while True:
        next_close = text.find(closer, j)
        next_param = text.find(param_open, j)
        if header is not None and not saw_param:
            # Only BEFORE the first parameter: a header met after one is
            # inside a value or after it, and the loop below decides that.
            # (Checking it on every pass read a ``write`` whose content shows
            # a call as "header first", dropped the write and promoted the
            # example call to the top level — review of 7a1d93272.)
            next_header = text.find(header, j)
            if next_header != -1 and all(
                pos == -1 or next_header < pos for pos in (next_close, next_param)
            ):
                return HEADER_FIRST
        if next_close == -1:
            return None
        if next_param == -1 or next_param > next_close:
            gap = text[j:next_close]
            if not saw_param and gap.strip():
                free = True
            if _stray_close(gap, param_close, saw_param):
                stray = True
            return CallShape(
                next_close + len(closer), free, canonical, consistent, stray
            )
        if _stray_close(text[j:next_param], param_close, saw_param):
            stray = True
        saw_param = True
        name_end = text.find(">", next_param + len(param_open))
        if name_end == -1:
            return None
        value_start = name_end + 1
        close, quoted = find_value_close(text, value_start, param_close)
        if close == -1:
            return None
        if not quoted:
            free = True
            if not (close > 0 and text[close - 1] == "\n"):
                # qwen3_coder_xml's ``_closes_canonically``: the FIRST close
                # was mid-line, so a call after this one may be its text.
                canonical = False
                if text.startswith("\n", value_start):
                    grid = _grid_close(
                        text, value_start, param_close, (param_open, closer, header)
                    )
                    if grid is None:
                        consistent = False
                    else:
                        close = grid
        j = close + len(param_close)


def _stray_close(gap: str, param_close: str, saw_param: bool) -> bool:
    """A ``param_close`` outside every value that only an ENCLOSING value
    accounts for: in a body with no parameter at all, or on the template's
    grid (line start). A compact value that quotes the marker mid-line
    (``grep "</parameter>" x</parameter>``) leaves one mid-line after its
    first close — that is its own text, not a stray (review of 975a84f7b)."""
    if param_close not in gap:
        return False
    return not saw_param or ("\n" + param_close) in gap


def _grid_close(
    text: str, value_start: int, param_close: str, stops: tuple[str | None, ...]
) -> int | None:
    """The first LINE-START ``param_close`` of a value that opened on the
    grid, unless another line-start marker of the wire comes first."""
    close = text.find("\n" + param_close, value_start)
    if close == -1:
        return None
    for stop in stops:
        if stop and -1 < text.find("\n" + stop, value_start, close) < close:
            return None
    return close + 1


_FENCE = re.compile(r"^[ ]{0,3}(`{3,}|~{3,})(.*)$", re.MULTILINE)


def inside_code(text: str, pos: int) -> bool:
    """Whether ``pos`` sits inside markdown code: an open fence, or inline.

    A model that SHOWS the wire format — "the call looks like this:" — puts
    it in a code block, and examples use made-up names. A call is never
    quoted that way. Fences follow CommonMark's shape (up to three spaces of
    indent, three or more backticks or tildes; a closing fence uses the same
    character, is at least as long and carries no info string). Inline code
    is an odd number of backticks earlier on the same line.
    """
    fence: str | None = None
    for match in _FENCE.finditer(text, 0, pos):
        run = match.group(1)
        if fence is None:
            fence = run
        elif (
            run[0] == fence[0] and len(run) >= len(fence) and not match.group(2).strip()
        ):
            fence = None
    if fence is not None:
        return True
    line_start = text.rfind("\n", 0, pos) + 1
    return text.count("`", line_start, pos) % 2 == 1


_TRAILING_CUT = re.compile(
    r"(?P<wrapper><tool_call>\s*)?<function=(?P<name>[^>\s]*)(?P<closed>>)?(?P<rest>.*)",
    re.DOTALL,
)


def trailing_cut_call(text: str) -> tuple[int, str | None, bool] | None:
    """A call the stream was CUT inside: ``(start, name, wrapped)``, or None.

    ``text`` ends inside call markup that never closed — at ``<tool_call>``,
    inside ``<function=NA``, at or after ``<function=NAME>`` (its first
    parameter opened, even a value begun) — with nothing after it but that
    markup. ``name`` is None when the header never completed. ``start`` is
    where the markup begins (the wrapper, when there is one). A trailing
    region that holds a close, prose after the header, or that sits inside
    markdown code is not a cut call.

    The rule it serves (review of c88275f9c): a cut-off call is a call with
    broken arguments when its name is known and it may be called, and
    NOTHING otherwise — never raw markup in the answer.
    """
    wrapper = text.rfind("<tool_call>")
    header = text.rfind("<function=")
    if wrapper == -1 and header == -1:
        return None
    if header != -1 and (wrapper == -1 or header > wrapper):
        if wrapper != -1 and not text[wrapper + len("<tool_call>") : header].strip():
            start = wrapper
        else:
            start = header
    else:
        start = wrapper
    tail = text[start:]
    if "</function>" in tail or "</tool_call>" in tail or inside_code(text, start):
        return None
    if not tail[len("<tool_call>") :].strip() and tail.startswith("<tool_call>"):
        return start, None, True
    match = _TRAILING_CUT.fullmatch(tail)
    if match is None:
        return None
    wrapped = match.group("wrapper") is not None
    if not match.group("closed"):
        return start, None, wrapped  # the header never completed
    rest = match.group("rest").lstrip()
    if rest and not (rest.startswith("<parameter=") or "<parameter=".startswith(rest)):
        return None  # prose after the header: text, not a cut call
    return start, match.group("name").strip() or None, wrapped


def cut_call_name(text: str, request: Any) -> tuple[int, str | None] | None:
    """``(start, callable name or None)`` for a call the output was cut inside.

    None when ``text`` does not end inside call markup, or when the request
    passes text through verbatim (no tools, ``tool_choice`` none). The name
    is returned only for a WRAPPED header whose name may be called: declared
    (and the forced one, under a named ``tool_choice``) or, when undeclared
    calls may surface, any name. Otherwise the markup is to be dropped — a
    bare ``<function=NAME>`` at the end is indistinguishable from prose, and
    a header that never completed names nothing.
    """
    if request is None or _field(request, "tool_choice") == "none":
        return None
    tools = _field(request, "tools")
    if not isinstance(tools, list) or not tools:
        return None
    cut = trailing_cut_call(text)
    if cut is None:
        return None
    start, name, wrapped = cut
    if name is None or not wrapped:
        return start, None
    declared: set[str] = set()
    for tool in tools:
        function = _field(tool, "function")
        tool_name = _field(function, "name") if function is not None else None
        tool_name = tool_name or _field(tool, "name")
        if isinstance(tool_name, str):
            declared.add(tool_name)
    forced = named_tool_choice(request)
    if forced is not None:
        return start, name if name == forced else None
    if name in declared or may_surface_undeclared(request):
        return start, name
    return start, None


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def named_tool_choice(request: Any) -> str | None:
    """The one function a named ``tool_choice`` forces, or ``None``."""
    choice = _field(request, "tool_choice") if request is not None else None
    if choice is None or isinstance(choice, str):
        return None
    function = _field(choice, "function")
    selected = _field(function, "name") if function is not None else None
    selected = selected or _field(choice, "name")
    return selected if isinstance(selected, str) and selected else None


def may_surface_undeclared(request: Any) -> bool:
    """Whether a call to a name the request did not offer may reach the client.

    The model calling a tool it was not given is ITS mistake, and the client
    is the one that answers it ("tool not found") so the model can recover;
    a cloud API or vLLM passes such a call through. Two cases keep the
    refusal, because they are contracts about WHICH call may appear rather
    than a tool list: ``tool_choice="none"`` and a NAMED ``tool_choice``. A
    request with no tools at all has nothing to answer the call with.
    """
    if request is None or _field(request, "tool_choice") == "none":
        return False
    if named_tool_choice(request) is not None:
        return False
    tools = _field(request, "tools")
    if not isinstance(tools, list):
        return False
    for tool in tools:
        function = _field(tool, "function")
        name = _field(function, "name") if function is not None else None
        if isinstance(name or _field(tool, "name"), str):
            return True
    return False


def _end_of_parameters(param_close: str, closer: str) -> re.Pattern[str]:
    return re.compile(re.escape(param_close) + r"\s*" + re.escape(closer))


def split_marked_calls(
    text: str,
    opener: str,
    closer: str,
    outer: str | None = None,
    valid_names: frozenset[str] | set[str] | None = None,
    *,
    surface_undeclared: bool = False,
    header: str | None = None,
    wrapper: str | None = None,
    param_open: str = PARAMETER_OPEN,
    param_close: str = PARAMETER_CLOSE,
    wrapper_pair: tuple[str, str] | None = None,
) -> list[tuple[str, str, int, int]]:
    """``(name, body, span_start, span_end)`` for each call in ``text``.

    ``opener`` must capture the call name in group 1. ``outer``, when given,
    is an enclosing marker (e.g. ``</tool_call>`` around ``</function>``)
    that is absorbed into the span so callers can excise the whole
    invocation from surrounding prose in one step — a second, differently
    truncating regex pass over the same text is how tag fragments leak into
    user-visible content.

    Without ``valid_names`` the position-only rules apply (nothing declared,
    nothing can execute). With them, STRUCTURE decides nesting, read the way
    ``qwen3_coder_xml`` reads it (``call_shape``): an opener inside another
    call's values is that value; an opener after the call's structural close
    is a sibling, unless the bytes say the value may still be running —

    * the call has a raw value that opened on the template's grid and closed
      mid-line (``CallShape.consistent``), or
    * the candidate's own reading leaves an orphan ``</parameter>…</function>``
      behind it, a close that only a value running over the candidate can
      account for.

    In both the safer reading wins: ONE call keeps the bytes as its value and
    no extra call is made. Before this, declared names alone decided nesting,
    so a declared value ran to the last ``</function>`` and swallowed a
    following call to an undeclared name — and that call's parameters landed
    in the declared one (``shell {"command": "rm -rf ~"}``, review of
    26452cbe3).

    ``header`` (a regex capturing the name) names the structural unit when
    ``opener`` is narrower — ``<tool_call>\\s*<function=…>`` for the scanner
    fallback, which must still see a bare ``<function=…>`` as a boundary even
    though it never emits one.

    ``surface_undeclared`` also returns TOP-LEVEL calls to names the request
    did not declare, when they are closed, framed by ``wrapper`` (if given),
    outside markdown code (``inside_code``), and not right after a declared
    raw value that closed mid-line (those bytes may be that value). The
    result is in text order.
    """
    if valid_names is None:
        return [
            call
            for call, _region in _split_declared_calls(
                text, opener, closer, outer, valid_names
            )
            if call is not None
        ]

    unit_pattern = header or opener
    tops = top_level_calls(
        text,
        unit_pattern,
        closer,
        valid_names,
        param_open=param_open,
        param_close=param_close,
        wrapper=wrapper_pair,
    )
    if header is None:
        span_starts: dict[int, int] = {top.body_start: top.start for top in tops}
    else:
        span_starts = {
            match.end(): match.start() for match in re.finditer(opener, text, re.DOTALL)
        }

    calls: list[tuple[str, str, int, int]] = []
    last_declared: TopCall | None = None
    for index, top in enumerate(tops):
        span_start = span_starts.get(top.body_start)
        if index + 1 < len(tops):
            nxt = tops[index + 1]
            region_end = span_starts.get(nxt.body_start, nxt.start)
        else:
            region_end = len(text)
        after_declared = last_declared
        if top.declared:
            last_declared = top
        if span_start is None or top.body_end < 0:
            continue
        span_end = top.body_end + len(closer)
        if outer:
            tail = re.match(r"\s*" + re.escape(outer), text[span_end:region_end])
            if not tail:
                continue
            span_end += tail.end()
        if not top.declared:
            if not surface_undeclared or not isinstance(top.shape, CallShape):
                continue
            if wrapper is not None and not text[:span_start].rstrip().endswith(wrapper):
                continue
            if inside_code(text, span_start):
                continue
            if after_declared is not None and not after_declared.canonical:
                _logger.warning(
                    "[TOOL-CALL-UNDECLARED] %r follows a declared call whose "
                    "raw value closed mid-line, so it may be that value's "
                    "text; kept as text",
                    top.name,
                )
                continue
        calls.append(
            (top.name, text[top.body_start : top.body_end], span_start, span_end)
        )
    return calls


class TopCall:
    """One top-level call as ``top_level_calls`` reads it.

    ``start``/``body_start``  the header's span (the body follows it).
    ``body_end``  where the body ends (the offset of its closer), or -1 when
                  it never closes. The structural close, unless the value runs
                  on (see ``top_level_calls``) — then the LAST closer before
                  the next top-level call, which keeps a value that quotes a
                  closer whole (jundot/omlx#2507).
    ``shape``     its ``CallShape`` (``None`` when unclosed).
    """

    __slots__ = ("start", "body_start", "body_end", "name", "declared", "shape")

    def __init__(self, start, body_start, body_end, name, declared, shape):
        self.start = start
        self.body_start = body_start
        self.body_end = body_end
        self.name = name
        self.declared = declared
        self.shape = shape

    @property
    def canonical(self) -> bool:
        """Closed on the template's grid (False for an unclosed call)."""
        return isinstance(self.shape, CallShape) and self.shape.canonical

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"TopCall({self.name!r}, {self.start}, {self.body_end}, {self.shape})"


def top_level_calls(
    text: str,
    header: str,
    closer: str,
    valid_names: frozenset[str] | set[str],
    *,
    param_open: str = PARAMETER_OPEN,
    param_close: str = PARAMETER_CLOSE,
    wrapper: tuple[str, str] | None = None,
) -> list[TopCall]:
    """The calls that are top-level in ``text``, by STRUCTURE, in text order.

    ``header`` is a regex capturing the call name (``<function=([^>]+)>``).
    Each header is read with ``call_shape``. A header inside a call's values
    is that value. A header after the call's structural close is a sibling,
    unless the bytes say the call's value may still be running:

    * a raw value of the call opened on the template's grid and closed
      mid-line (``CallShape.consistent``), or
    * the candidate, read on its own, leaves an orphan
      ``</parameter>…</function>`` behind it before the next header — a close
      that only a value running over the candidate accounts for.

    In both cases the safer reading wins: the candidate is the call's text,
    and no extra call exists. A declared call that never closes keeps the
    position-only rule (a later DECLARED header after a closer is a sibling).
    A header that meets another header first is prose (``HEADER_FIRST``).
    An undeclared header that never closes is not a call at all.

    Names do not decide nesting here — before this, they alone did, and a
    declared value ran to the last ``</function>`` and took the parameters of
    the call after it (review of 26452cbe3).
    """
    units = list(re.finditer(header, text, re.DOTALL))
    if not units:
        return []
    header_literal = _header_literal(header)
    shape_cache: dict[int, CallShape | str | None] = {}

    def shape_of(k: int) -> CallShape | str | None:
        # Lazy: a header that lies inside an earlier call's value is never
        # read, so a value quoting thousands of headers costs one read.
        if k not in shape_cache:
            shape_cache[k] = call_shape(
                text,
                units[k].end(),
                closer,
                header=header_literal,
                param_open=param_open,
                param_close=param_close,
            )
        return shape_cache[k]

    names = [unit.group(1).strip() for unit in units]
    declared = [name in valid_names for name in names]
    orphan = _end_of_parameters(param_close, closer)
    # The close a running value leaves behind is its call's REAL close: for a
    # call framed by ``wrapper`` that is ``</parameter></function></tool_call>``.
    # Prose after real calls that merely quotes ``</parameter></function>``
    # ("values end with …") is not one — read as one, it merged a real
    # parallel call into the call before it (review of 975a84f7b).
    wrapped_orphan = (
        re.compile(
            re.escape(param_close)
            + r"\s*"
            + re.escape(closer)
            + r"\s*"
            + re.escape(wrapper[1])
        )
        if wrapper
        else None
    )
    starts = [unit.start() for unit in units]

    def orphan_for(k: int) -> re.Pattern[str]:
        if wrapped_orphan is not None and text[: starts[k]].rstrip().endswith(
            wrapper[0]
        ):
            return wrapped_orphan
        return orphan

    def first_unit_at(pos: int) -> int:
        return bisect.bisect_left(starts, pos)

    def free(k: int) -> bool:
        shape = shape_of(k)
        return isinstance(shape, CallShape) and shape.free

    def sibling(cur: int, k: int) -> bool:
        cur_shape = shape_of(cur)
        if not isinstance(cur_shape, CallShape):
            return declared[k] and closer in text[units[cur].end() : starts[k]]
        if starts[k] < cur_shape.end:
            return False
        k_shape = shape_of(k)
        if not isinstance(k_shape, CallShape) and not declared[k]:
            return False
        if cur_shape.free:
            if not cur_shape.consistent:
                return False
            if isinstance(k_shape, CallShape):
                if k_shape.stray:
                    return False
                nxt = first_unit_at(k_shape.end)
                limit = starts[nxt] if nxt < len(starts) else len(text)
                if orphan_for(cur).search(text, k_shape.end, limit):
                    return False
        return True

    picked: list[int] = []
    # A header judged to be TEXT inside a running value takes the headers it
    # structurally contains with it: they are text too, never siblings.
    skip_until = 0
    for k in range(len(units)):
        if starts[k] < skip_until:
            continue
        if picked:
            cur_shape = shape_of(picked[-1])
            if isinstance(cur_shape, CallShape) and starts[k] < cur_shape.end:
                continue  # inside one of cur's values
        if shape_of(k) == HEADER_FIRST:
            continue
        if not picked:
            if declared[k] or isinstance(shape_of(k), CallShape):
                picked.append(k)
            continue
        if sibling(picked[-1], k):
            picked.append(k)
        else:
            k_shape = shape_of(k)
            if isinstance(k_shape, CallShape):
                skip_until = max(skip_until, k_shape.end)

    def body_end_of(index: int, owner: int | None = None) -> tuple[int, bool]:
        """``(body_end, runs_on)`` of ``picked[index]`` in its region; the
        value runs on when the close of ``owner`` (default: the call itself)
        is left over after its structural end."""
        k = picked[index]
        unit = units[k]
        region_end = starts[picked[index + 1]] if index + 1 < len(picked) else len(text)
        shape = shape_of(k)
        if not isinstance(shape, CallShape):
            return text.rfind(closer, unit.end(), region_end), False
        # Only a leftover CLOSE says a value runs on. A header in the region
        # (a cut-off call, prose naming the format) does not: counting it
        # merged real parallel calls into one (review of 975a84f7b).
        pattern = orphan_for(k if owner is None else owner)
        runs_on = bool(pattern.search(text, shape.end, region_end))
        if runs_on:
            return text.rfind(closer, unit.end(), region_end), True
        return shape.end - len(closer), False

    # A later call whose value runs on (the earlier call's close is left
    # over after it) shows that SOME value runs over it; read on its own it
    # would hand the bytes after it to a call. The safer reading attributes
    # them to the earlier free call: merge, and look again. The sibling test
    # above sees only up to the next header, so two calls quoted back to back
    # inside one value passed it (review of 7a1d93272).
    merged = True
    while merged:
        merged = False
        for index in range(1, len(picked)):
            if (
                free(picked[index - 1])
                and body_end_of(index, owner=picked[index - 1])[1]
            ):
                del picked[index]
                merged = True
                break

    tops: list[TopCall] = []
    for index, k in enumerate(picked):
        unit = units[k]
        body_end, _runs_on = body_end_of(index)
        shape = shape_of(k)
        tops.append(
            TopCall(
                unit.start(),
                unit.end(),
                body_end,
                names[k],
                declared[k],
                shape if isinstance(shape, CallShape) else None,
            )
        )
    return tops


def _header_literal(pattern: str) -> str | None:
    """The literal prefix of a header regex (``<function=`` of
    ``<function=([^>]+)>``), used to spot a second header structurally."""
    literal = re.match(r"[^\\()\[\]{}.*+?^$|]+", pattern.split("(", 1)[0])
    if literal is None:
        return None
    prefix = literal.group(0)
    return prefix if prefix.startswith("<") and len(prefix) > 2 else None


def _split_declared_calls(
    text: str,
    opener: str,
    closer: str,
    outer: str | None,
    valid_names: frozenset[str] | set[str] | None,
) -> list[tuple[tuple[str, str, int, int] | None, tuple[int, int]]]:
    """The position-only pass: ``(call or None, region)`` per opener it owns.

    Used when the request declared no tools. ``call`` is ``None`` for an
    opener that was resolved but not emitted (no closer, or a missing
    required ``outer``).
    """
    openers = list(re.finditer(opener, text, re.DOTALL))
    calls: list[tuple[tuple[str, str, int, int] | None, tuple[int, int]]] = []
    i = 0
    while i < len(openers):
        # NO name deduplication here. That rule belongs to parameters, where
        # a repeat within one call is payload because each name carries one
        # value. Calls are the opposite: invoking the same tool twice in a
        # turn is ordinary agent behaviour — read_file /a then read_file /b —
        # and suppressing the second opener merges both into one body and
        # silently drops a call the model asked for.
        if valid_names is not None and openers[i].group(1).strip() not in valid_names:
            i += 1
            continue
        segmented = segment_by_next_opener(text, openers, i, closer, valid_names)
        sibling = _next_sibling(text, openers, i, closer, valid_names)
        limit = openers[sibling].start() if sibling < len(openers) else len(text)
        call: tuple[str, str, int, int] | None = None
        region_end = limit
        if segmented is not None:
            body, span_end = segmented
            region_end = span_end
            emit = True
            if outer:
                tail = re.match(r"\s*" + re.escape(outer), text[span_end:limit])
                if tail:
                    span_end += tail.end()
                    region_end = span_end
                else:
                    # ``outer`` is REQUIRED when requested, not opportunistic.
                    # The regex this replaced matched only with the enclosing
                    # marker present, so accepting a call whose wrapper never
                    # arrived would newly admit truncated emissions. Callers
                    # that tolerate a missing wrapper (nemotron_tool_parser
                    # documents that case) simply do not pass ``outer``.
                    emit = False
            if emit:
                call = (openers[i].group(1).strip(), body, openers[i].start(), span_end)
        calls.append((call, (openers[i].start(), region_end)))
        # See split_marked_parameters: skip openers swallowed by this value.
        i = sibling if sibling > i else i + 1
    return calls


def _exact_quoted_value(
    block: str, value_start: int, closer: str
) -> tuple[int, int] | None:
    """``(value_end, close_end)`` when the value is one JSON string, else None.

    A grammar-constrained value is a JSON string followed by its close
    (#1542). Its bytes are exact: a ``<parameter=…>`` or ``</parameter>`` in
    the quotes is data, never a sibling. Required: a terminated string, then
    only whitespace, then the close marker — anything else is a raw value
    that merely starts with a quote.
    """
    i = value_start
    n = len(block)
    while i < n and block[i].isspace():
        i += 1
    if i >= n or block[i] != '"':
        return None
    end = _json_string_end(block, i)
    if end == -1:
        return None
    k = end
    while k < n and block[k].isspace():
        k += 1
    if not block.startswith(closer, k):
        return None
    return k, k + len(closer)


def _nested_call_spans(
    block: str,
    value_start: int,
    closer: str,
    call_close: str,
    call_header: str,
    param_open: str | None = None,
) -> list[tuple[int, int]]:
    """The spans of OTHER calls written inside a body, from ``value_start``.

    Two shapes say a value holds another call's markup, so the body was read
    as one call (the safer reading) and that call's ``<parameter=…>`` openers
    must never become parameters of THIS call:

    * ``…</parameter></function>…<function=other>`` — that is how
      ``rm -rf ~`` from a following call overwrote ``shell``'s ``command``
      (review of 26452cbe3);
    * a header followed directly by its own ``<parameter=`` — a document
      that SHOWS a call (``<function=shell>\n<parameter=command>…``) gave its
      ``command`` to the ``write`` holding it (review of 7a1d93272). A
      header merely mentioned in prose (``see <function=x> below``) is not
      one.

    Each span runs from that header to the nested call's own ``call_close``
    when the body holds one. One that never closes runs to the end of the
    body when it followed a call's end (the rest IS that call), and otherwise
    ends at the first ``closer`` after its header (the value holding it), so
    a real parameter of THIS call after a quoted example still counts — ``s = '<function=x><parameter=y>'`` then ``path`` used to
    lose ``path`` into the value (review of 975a84f7b).
    """
    shaped = (
        re.compile(re.escape(call_header) + r"[^>\n]*>\s*" + re.escape(param_open))
        if param_open
        else None
    )
    end_of_params = _end_of_parameters(closer, call_close)
    spans: list[tuple[int, int]] = []
    pos = value_start
    while pos < len(block):
        found: list[tuple[int, bool]] = []  # (header start, after a call end)
        end = end_of_params.search(block, pos)
        if end is not None:
            start = block.find(call_header, end.end())
            if start != -1:
                found.append((start, True))
        if shaped is not None:
            header = shaped.search(block, pos)
            if header is not None:
                found.append((header.start(), False))
        if not found:
            break
        start, after_call_end = min(found)
        close = block.find(call_close, start)
        if close != -1:
            stop = close + len(call_close)
        elif after_call_end:
            # ``…</parameter></function>…<function=other>`` that never closes:
            # the rest of the body is that call's text.
            stop = len(block)
        else:
            value_close = block.find(closer, start)
            stop = value_close + len(closer) if value_close != -1 else len(block)
        spans.append((start, stop))
        pos = stop
    return spans


def split_marked_parameters(
    block: str,
    opener: str,
    closer: str,
    valid_names: frozenset[str] | set[str] | None = None,
    *,
    reject_undeclared_siblings: bool = False,
    call_close: str | None = None,
    call_header: str | None = None,
) -> list[tuple[str, str]] | None:
    """``(name, value)`` for each parameter in ``block``.

    ``opener`` must capture the parameter name in group 1. A value keeps
    every byte it was sent except the one wrapping newline per side that the
    wire format adds -- see ``trim_wrapping_newlines``. Parameter *names* are
    still stripped: those are identifiers, not payload.

    A JSON-quoted value is exact (``_exact_quoted_value``). With
    ``call_close``/``call_header`` given, parameter openers after a second
    call inside the body are that value's text (``_second_call_start``).
    """
    scanned = scan_marked_parameters(
        block,
        opener,
        closer,
        valid_names,
        reject_undeclared_siblings=reject_undeclared_siblings,
        call_close=call_close,
        call_header=call_header,
    )
    return None if scanned is None else scanned[0]


def scan_marked_parameters(
    block: str,
    opener: str,
    closer: str,
    valid_names: frozenset[str] | set[str] | None = None,
    *,
    reject_undeclared_siblings: bool = False,
    call_close: str | None = None,
    call_header: str | None = None,
) -> tuple[list[tuple[str, str]], list[tuple[int, int]]] | None:
    """``split_marked_parameters`` plus the spans no parameter opener may
    come from: exact (JSON-quoted) values and the text after a second call
    (see ``_second_call_start``). A recovery pass that re-reads the block
    with a looser rule must skip openers inside those spans."""
    openers = list(re.finditer(opener, block))
    out: list[tuple[str, str]] = []
    protected: list[tuple[int, int]] = []
    i = 0
    while i < len(openers):
        exact = _exact_quoted_value(block, openers[i].end(), closer)
        if exact is not None:
            value_end, close_end = exact
            out.append(
                (
                    openers[i].group(1).strip(),
                    trim_wrapping_newlines(block[openers[i].end() : value_end]),
                )
            )
            protected.append((openers[i].end(), close_end))
            i += 1
            while i < len(openers) and openers[i].start() < close_end:
                i += 1
            continue
        live = openers
        nested: list[tuple[int, int]] = []
        if call_close and call_header:
            nested = _nested_call_spans(
                block,
                openers[i].end(),
                closer,
                call_close,
                call_header,
                _header_literal(opener),
            )
            if nested:
                live = [
                    match
                    for match in openers
                    if not any(lo <= match.start() < hi for lo, hi in nested)
                ]
                protected.extend(nested)
        if reject_undeclared_siblings and valid_names is not None:
            current_end = live[i].end()
            for candidate in live[i + 1 :]:
                if candidate.group(1).strip() in valid_names:
                    continue
                if closer in block[current_end : candidate.start()]:
                    # Syntax cannot distinguish an undeclared sibling from
                    # marker-shaped payload after a close.  Continuing would
                    # splice the entire undeclared element into the previous
                    # value (#1541).  Refuse the call rather than execute a
                    # silently rewritten argument.
                    return None
        segmented = segment_by_next_opener(block, live, i, closer, valid_names)
        sibling = _next_sibling(block, live, i, closer, valid_names)
        if segmented is not None:
            out.append((live[i].group(1).strip(), trim_wrapping_newlines(segmented[0])))
        if sibling >= len(live):
            break
        # Jump to the sibling, not i+1: the openers in between are literal
        # text inside the value just consumed. Advancing one at a time is
        # what turns a value containing "<parameter=x>" into a phantom
        # parameter named x, passed to the tool as if the model sent it.
        # ``live`` only drops openers AFTER ``i`` (inside nested calls), so
        # ``live[i]`` is ``openers[i]``; map the sibling back to ``openers``.
        i = openers.index(live[sibling]) if sibling > i else i + 1
    return out, protected
