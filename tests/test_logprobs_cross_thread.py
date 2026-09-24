# SPDX-License-Identifier: Apache-2.0
"""A ``logprobs: true`` request must never abort the server (LeanZero lz.4).

The speculative decode paths (vendored MTP, suffix, DSpark) yield LAZY
logprobs rows (``lps[i]`` views) built on the mlx-step thread's stream. The
route thread converted them with ``np.array``; MLX then throws
``There is no Stream(gpu, N) in current thread`` inside the buffer protocol,
where it cannot become a Python exception, and libc++ aborts the process.

Each scenario runs in a subprocess because the failure mode is a process
abort that would take the test session with it. The negative control proves
the scenario reproduces the abort without the fix.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap

import pytest

mx = pytest.importorskip("mlx.core")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PRELUDE = textwrap.dedent(
    """
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    import mlx.core as mx
    import numpy as np

    step = ThreadPoolExecutor(1)

    def lazy_rows():
        # An evaluated verify stack, then per-token views taken AFTER the
        # eval — the exact shape ``mtp_generate_step`` yields.
        logits = mx.random.normal((4, 32))
        lps = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(lps)
        return [SimpleNamespace(logprobs=lps[i]) for i in range(4)]
    """
)


def _run(body: str) -> subprocess.CompletedProcess:
    env = dict(
        os.environ, PYTHONPATH=_REPO + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    return subprocess.run(
        [sys.executable, "-c", _PRELUDE + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


def test_negative_control_unmaterialized_row_aborts_the_process():
    proc = _run(
        """
        responses = step.submit(lazy_rows).result()
        np.array(responses[0].logprobs.astype(mx.float32))
        print("survived")
        """
    )
    assert proc.returncode == -signal.SIGABRT, (proc.returncode, proc.stderr[-400:])
    assert "There is no Stream" in proc.stderr


def test_step_thread_materialization_makes_route_conversion_safe():
    proc = _run(
        """
        from rapid_mlx.scheduler import _materialize_response_logprobs

        def step_body():
            responses = lazy_rows()
            _materialize_response_logprobs(responses)
            return responses

        responses = step.submit(step_body).result()
        for r in responses:
            row = np.array(r.logprobs.astype(mx.float32))
            assert row.shape == (32,)
            assert abs(float(np.exp(row).sum()) - 1.0) < 1e-3
        print("survived")
        """
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    assert "survived" in proc.stdout


def test_extractor_fails_the_request_not_the_process():
    proc = _run(
        """
        from rapid_mlx.service.helpers import _extract_token_logprob

        class Tok:
            def decode(self, ids):
                return f"<{ids[0]}>"

        responses = step.submit(lazy_rows).result()
        try:
            _extract_token_logprob(responses[0].logprobs, 1, Tok(), 2)
        except RuntimeError as exc:
            assert "logprobs row could not be materialized" in str(exc)
            print("raised")
        print("survived")
        """
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    assert "raised" in proc.stdout and "survived" in proc.stdout


def test_materialize_accepts_lists_and_absent_rows():
    from types import SimpleNamespace

    from rapid_mlx.scheduler import _materialize_response_logprobs

    row = mx.arange(4, dtype=mx.float32)
    _materialize_response_logprobs(
        [
            SimpleNamespace(logprobs=None),
            SimpleNamespace(logprobs=[row[1:], None]),
            SimpleNamespace(),
        ]
    )
