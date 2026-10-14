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
    """


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
        return (
            f"This prompt ({self.prompt_tokens} tokens) needs "
            f"{self.needed_bytes / 1e9:.1f} GB but this Mac can hold "
            f"{self.available_bytes / 1e9:.1f} GB; the maximum context here "
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
