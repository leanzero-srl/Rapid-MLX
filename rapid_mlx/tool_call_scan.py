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
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any

__all__ = [
    "segment_by_next_opener",
    "declared_tool_names",
    "split_marked_calls",
    "split_marked_parameters",
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


def split_marked_calls(
    text: str,
    opener: str,
    closer: str,
    outer: str | None = None,
    valid_names: frozenset[str] | set[str] | None = None,
    *,
    surface_undeclared: bool = False,
) -> list[tuple[str, str, int, int]]:
    """``(name, body, span_start, span_end)`` for each call in ``text``.

    ``opener`` must capture the call name in group 1. ``outer``, when given,
    is an enclosing marker (e.g. ``</tool_call>`` around ``</function>``)
    that is absorbed into the span so callers can excise the whole
    invocation from surrounding prose in one step — a second, differently
    truncating regex pass over the same text is how tag fragments leak into
    user-visible content.

    ``surface_undeclared`` (only meaningful with ``valid_names``) also
    returns well-formed calls to names the request did NOT declare, provided
    they sit OUTSIDE every declared call's span. Declared names are resolved
    first and exactly as without the flag, so they still decide nesting: an
    undeclared opener inside a declared call's value stays part of that value
    and never becomes a sibling. What changes is only the top level — a call
    the model really made to a tool it was not offered reaches the client,
    which answers it with "tool not found", instead of being folded into the
    answer text where it silently ends the agent's turn. The result is in
    text order; callers can tell the undeclared entries by their name.
    """
    resolved = _split_declared_calls(text, opener, closer, outer, valid_names)
    found = [call for call, _region in resolved if call is not None]
    if not surface_undeclared or valid_names is None:
        return found
    regions = [region for _call, region in resolved]
    found.extend(
        _split_undeclared_calls(text, opener, closer, outer, valid_names, regions)
    )
    found.sort(key=lambda call: call[2])
    return found


def _split_undeclared_calls(
    text: str,
    opener: str,
    closer: str,
    outer: str | None,
    valid_names: frozenset[str] | set[str],
    regions: list[tuple[int, int]],
) -> list[tuple[str, str, int, int]]:
    """Well-formed calls to undeclared names that no declared call encloses.

    ``regions`` are the byte ranges the declared pass assigned to declared
    openers (their whole span, or up to their next sibling when they never
    closed). An undeclared opener inside one is payload of that call, full
    stop. Outside them, the position-only rules apply among the undeclared
    openers, bounded by the next declared region so an undeclared call can
    never absorb a declared one.
    """

    def _enclosed(pos: int) -> bool:
        return any(start <= pos < end for start, end in regions)

    free = [
        match
        for match in re.finditer(opener, text, re.DOTALL)
        if match.group(1).strip() not in valid_names and not _enclosed(match.start())
    ]
    calls: list[tuple[str, str, int, int]] = []
    k = 0
    while k < len(free):
        start = free[k].start()
        limit = min((s for s, _e in regions if s > start), default=len(text))
        bounded = text[:limit]
        group = [match for match in free[k:] if match.start() < limit]
        segmented = segment_by_next_opener(bounded, group, 0, closer)
        sibling = _next_sibling(bounded, group, 0, closer)
        if segmented is not None:
            body, span_end = segmented
            emit = True
            if outer:
                tail_limit = group[sibling].start() if sibling < len(group) else limit
                tail = re.match(r"\s*" + re.escape(outer), text[span_end:tail_limit])
                if tail:
                    span_end += tail.end()
                else:
                    emit = False
            if emit:
                calls.append((group[0].group(1).strip(), body, start, span_end))
        k += sibling if sibling > 0 else 1
    return calls


def _split_declared_calls(
    text: str,
    opener: str,
    closer: str,
    outer: str | None,
    valid_names: frozenset[str] | set[str] | None,
) -> list[tuple[tuple[str, str, int, int] | None, tuple[int, int]]]:
    """The declared-name pass: ``(call or None, region)`` per opener it owns.

    ``call`` is ``None`` for an opener that was resolved but not emitted (no
    closer, or a missing required ``outer``); its region still encloses
    whatever follows it, so nothing inside can surface as an undeclared call.
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
        # This pass resolves DECLARED openers only — the opener being emitted
        # as well as the search for the next sibling. Undeclared openers are
        # never siblings here, so marker text inside a declared value cannot
        # split it. Whether a top-level undeclared call is returned at all is
        # decided afterwards, outside every region this pass assigns
        # (``surface_undeclared``); without that flag it is not returned.
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


def split_marked_parameters(
    block: str,
    opener: str,
    closer: str,
    valid_names: frozenset[str] | set[str] | None = None,
    *,
    reject_undeclared_siblings: bool = False,
) -> list[tuple[str, str]] | None:
    """``(name, value)`` for each parameter in ``block``.

    ``opener`` must capture the parameter name in group 1. A value keeps
    every byte it was sent except the one wrapping newline per side that the
    wire format adds -- see ``trim_wrapping_newlines``. Parameter *names* are
    still stripped: those are identifiers, not payload.
    """
    openers = list(re.finditer(opener, block))
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(openers):
        if reject_undeclared_siblings and valid_names is not None:
            current_end = openers[i].end()
            for candidate in openers[i + 1 :]:
                if candidate.group(1).strip() in valid_names:
                    continue
                if closer in block[current_end : candidate.start()]:
                    # Syntax cannot distinguish an undeclared sibling from
                    # marker-shaped payload after a close.  Continuing would
                    # splice the entire undeclared element into the previous
                    # value (#1541).  Refuse the call rather than execute a
                    # silently rewritten argument.
                    return None
        segmented = segment_by_next_opener(block, openers, i, closer, valid_names)
        sibling = _next_sibling(block, openers, i, closer, valid_names)
        if segmented is not None:
            out.append(
                (openers[i].group(1).strip(), trim_wrapping_newlines(segmented[0]))
            )
        # Jump to the sibling, not i+1: the openers in between are literal
        # text inside the value just consumed. Advancing one at a time is
        # what turns a value containing "<parameter=x>" into a phantom
        # parameter named x, passed to the tool as if the model sent it.
        i = sibling if sibling > i else i + 1
    return out
