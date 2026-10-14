# SPDX-License-Identifier: Apache-2.0
"""LeanZero memory guard: report how a response ENDED back to the engine.

The guard keeps a conversation's cache entries while the client owes it the
next request -- the completion ended in a tool call (a tool, a subagent is
running for it) or a length cut (a continuation follows) -- and only then
(review 8). The chat route threads the holder through its engine calls; the
Anthropic Messages and OpenAI Responses routes are wrapped here instead:
the holder is put in ``SESSION_OUTCOME`` for the request (the engine picks
it up when it builds the ``SessionHint``) and filled from the response the
route returns -- its JSON body, or the SSE events once the stream ENDED (a
stream that is cut off reports nothing). Values are the chat API's
finish_reason vocabulary: "tool_calls", "length", "stop".
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable
from typing import Any

from ..request import SESSION_OUTCOME

_ANTHROPIC_STOP = re.compile(r'"stop_reason":\s*"([A-Za-z_]+)"')
_ANTHROPIC_MAP = {"tool_use": "tool_calls", "max_tokens": "length"}


def anthropic_outcome(text: str) -> str | None:
    """finish_reason of an Anthropic Messages body or SSE stream text."""
    found = _ANTHROPIC_STOP.findall(text)
    if not found:
        return None
    return _ANTHROPIC_MAP.get(found[-1], "stop")


def responses_outcome(text: str) -> str | None:
    """finish_reason of a Responses API body or SSE stream text: a function
    call item means the client runs a tool and comes back; an incomplete
    response a cut; a completed one a reply."""
    compact = text.replace(" ", "")
    if '"type":"function_call"' in compact or '"type":"custom_tool_call"' in compact:
        return "tool_calls"
    if '"status":"incomplete"' in compact or '"type":"response.incomplete"' in compact:
        return "length"
    if '"status":"completed"' in compact or '"type":"response.completed"' in compact:
        return "stop"
    return None


async def _scan_stream(stream, holder: list, parse: Callable[[str], str | None]):
    seen: list[str] = []
    async for chunk in stream:
        text = chunk.decode("utf-8", "ignore") if isinstance(chunk, bytes) else chunk
        if isinstance(text, str) and (
            "stop_reason" in text
            or "function_call" in text
            or "custom_tool_call" in text
            or "response." in text
            or '"status"' in text
        ):
            seen.append(text)
        yield chunk
    outcome = parse("".join(seen)) if seen else None
    if outcome is not None:
        holder[0] = outcome


def reports_outcome(parse: Callable[[str], str | None]):
    """Decorate a route handler: give its engine calls an outcome holder and
    fill it from the response the handler returns."""

    def wrap(handler):
        @functools.wraps(handler)
        async def wrapper(*args: Any, **kwargs: Any):
            holder: list = [None]
            # Not reset: a streaming body runs after the handler returned,
            # in a task that copied this request's context.
            SESSION_OUTCOME.set(holder)
            response = await handler(*args, **kwargs)
            iterator = getattr(response, "body_iterator", None)
            if iterator is not None:
                response.body_iterator = _scan_stream(iterator, holder, parse)
                return response
            body = getattr(response, "body", None)
            if (
                isinstance(body, (bytes, bytearray))
                and getattr(response, "status_code", 200) < 400
            ):
                try:
                    holder[0] = parse(bytes(body).decode("utf-8", "ignore"))
                except Exception:  # noqa: BLE001 -- a hint, never a failure
                    pass
            return response

        return wrapper

    return wrap


__all__ = ["anthropic_outcome", "reports_outcome", "responses_outcome"]
