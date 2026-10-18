# SPDX-License-Identifier: Apache-2.0
"""Dependency-free engine error types shared across runtime boundaries.

Keep these exceptions outside MLX-owning modules so API and service code can
translate engine failures without importing the Apple-only scheduler runtime.
"""


class BackpressureError(Exception):
    """Raised when admission control rejects a new request.

    Route handlers convert this to HTTP 503 with a Retry-After header.  It is
    distinct from ``ValueError`` so narrow batch-error handlers do not swallow
    admission failures.

    ``error_object`` is set when waiting cannot help -- the request does not fit
    the device even with nothing else running -- and is then the
    ``context_length_exceeded`` body a client compacts on (HTTP 400 before a
    stream commits, the SSE error after). ``None`` = transient load shedding: a
    503 with Retry-After, or mid-stream the OpenAI ``rate_limit_exceeded`` code
    goose retries with backoff (RUN-ISSUES I-17: it was masked as "Internal
    error during streaming" and the run was refused after 3 tries).
    """

    def __init__(self, *args, error_object: dict | None = None) -> None:
        super().__init__(*args)
        self.error_object = dict(error_object) if isinstance(error_object, dict) else None


class PagedCacheUnsupportedLayoutError(Exception):
    """Raised at startup when ``--use-paged-cache`` cannot serve the model.

    The paged prefix cache only supports cache layouts its block serializer
    can losslessly slice and reconstruct (plain full-attention KV layers).
    The check is structural — it inspects the loaded model's prompt-cache
    factory output, so it can only run after weights/model construction —
    and it fails closed: an explicit ``--use-paged-cache`` must abort before
    the server starts serving requests rather than degrade into a
    healthy-looking mode that provides zero reuse.

    ``incompatible_layers`` carries the offending cache class names (empty
    when the layout could not be determined at all).
    """

    def __init__(self, message: str, *, incompatible_layers: tuple[str, ...] = ()):
        super().__init__(message)
        self.incompatible_layers = incompatible_layers


# The client contract for a prompt this device cannot hold (memory guard,
# LeanZero fork). goose maps an HTTP 400 whose error code is
# ``context_length_exceeded`` (or one carrying ``n_prompt_tokens > n_ctx``) to
# ContextLengthExceeded, then compacts the conversation and resends -- so the
# refusal reuses that OpenAI code and says why in ``lz_reason``.
DEVICE_MEMORY_ERROR_KIND = "context_exceeds_device_memory"


class ContextExceedsDeviceMemoryError(Exception):
    """A prompt whose prefill cannot fit this device's memory.

    Raised (admission) or reported (mid-flight, as a terminal error output)
    by the step-peak memory guard BEFORE anything is allocated, never as a
    Metal out-of-memory. ``max_context`` is the honest largest context this
    device holds right now, so a client can compact to it.
    """

    def __init__(
        self,
        *,
        prompt_tokens: int,
        needed_bytes: int,
        available_bytes: int,
        max_context: int,
    ) -> None:
        self.prompt_tokens = int(prompt_tokens)
        self.needed_bytes = int(needed_bytes)
        self.available_bytes = int(available_bytes)
        self.max_context = max(0, int(max_context))
        super().__init__(self.message)

    @property
    def message(self) -> str:
        need, have = self.needed_bytes / 1e9, self.available_bytes / 1e9
        # A close call needs the digits that tell the two apart, and the
        # shortfall itself (RC12 Gabee: "needs 50.1 GB but this Mac can hold
        # 50.1 GB").
        digits = 1
        while digits < 3 and f"{need:.{digits}f}" == f"{have:.{digits}f}":
            digits += 1
        short = ""
        if digits > 1:
            short = f" ({max(0, self.needed_bytes - self.available_bytes) / 1e6:.0f} MB short)"
        return (
            f"This prompt ({self.prompt_tokens} tokens) needs "
            f"{need:.{digits}f} GB but this Mac can hold "
            f"{have:.{digits}f} GB{short}; the maximum context here "
            f"is {self.max_context} tokens."
        )

    def error_object(self) -> dict:
        return {
            "message": self.message,
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "n_prompt_tokens": self.prompt_tokens,
            "n_ctx": self.max_context,
            "lz_reason": DEVICE_MEMORY_ERROR_KIND,
        }
