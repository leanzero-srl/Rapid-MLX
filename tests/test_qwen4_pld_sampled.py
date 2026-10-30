"""Prompt lookup (copy drafts) under sampling on the qwen4_exp family.

The Gauntlet runs Qwen3.8-Flash-Next at the model's own sampling defaults
(temperature 1.0), and the family policy used to qualify copy drafts for
greedy only, so every Gauntlet request made 0 prompt-lookup proposals. These
tests pin the policy and prove, on a real (tiny, random) qwen4_exp model with
its native MTP head, that turning copy drafts on under sampling changes how
fast tokens arrive and not which tokens are sampled.
"""

from __future__ import annotations

import importlib
from collections import Counter
from dataclasses import asdict

import mlx.core as mx
import mlx.nn as nn
import pytest

from rapid_mlx.models.qwen4_exp import Model, ModelArgs

from tests.test_qwen4_exp_vendored import _ple_args

scipy_stats = pytest.importorskip("scipy.stats")

ALPHABET = 4  # tokens the biased head may emit (and the prompt is written in)
GEN = 8  # generated tokens per sample
SAMPLES = 500  # seeded samples per arm
ALPHA = 1e-3  # per-test rejection level; seeds are fixed, so this is not flaky


def _de_bruijn(k: int, n: int) -> list[int]:
    """Every length-``n`` word over ``range(k)`` appears exactly once."""
    a = [0] * k * n
    seq: list[int] = []

    def db(t: int, p: int) -> None:
        if t > n:
            if n % p == 0:
                seq.extend(a[1 : p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    return seq


@pytest.fixture
def qwen4_mtp(monkeypatch):
    """A tiny qwen4_exp target with the family's own native MTP head.

    The head is given a bias that keeps sampling on a 4-token alphabet, and
    the prompt is a de Bruijn sequence over that alphabet, so every 2-gram
    suffix has a prompt match: copies are proposed on nearly every step, and
    the target's probability of the copied token is large enough that both
    verify branches (accept with ``p(d)``, residual draw on refusal) run
    hundreds of times per arm.
    """
    from rapid_mlx.spec_decode.mtp import dispatch_mtp_inject

    current_stream = mx.new_stream(mx.default_device())
    mx.set_default_stream(current_stream)
    importlib.import_module("mlx_lm.generate").generation_stream = current_stream
    # The family's production n-gram floor is 16 tokens; a 4-letter toy
    # language needs a 2-gram floor to find copies at all.
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MIN_NGRAM", "2")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MAX_NGRAM", "4")
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_MAX_TOKENS", "4")
    monkeypatch.delenv("RAPID_MLX_MTP_PROMPT_LOOKUP", raising=False)
    monkeypatch.delenv("RAPID_MLX_MTP_PROMPT_LOOKUP_SAMPLED", raising=False)

    mx.random.seed(0)
    args = _ple_args()
    args.mtp_num_hidden_layers = 1
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=asdict(args)))
    monkeypatch.setattr(nn, "quantize", lambda *_a, **_k: None)
    assert dispatch_mtp_inject(model, "qwen4_exp", allow_random_init=True) is True
    lm = model.language_model
    weight = lm.lm_head.weight
    head = nn.Linear(weight.shape[1], weight.shape[0], bias=True)
    head.weight = weight
    head.bias = mx.where(mx.arange(weight.shape[0]) < ALPHABET, 0.0, -30.0)
    lm.lm_head = head
    mx.eval(lm.parameters())
    seq = _de_bruijn(ALPHABET, 2)
    prompt = mx.array(seq + seq[:1], dtype=mx.uint32)
    return lm, prompt


def _sample(lm, prompt, *, prompt_lookup: bool, seed: int, stats: Counter):
    from rapid_mlx.spec_decode.mtp.generator import mtp_generate_step

    mx.random.seed(seed)
    timing: dict[str, float] = {}
    out = tuple(
        int(tok)
        for tok, _lp, _draft in mtp_generate_step(
            prompt,
            lm,
            max_tokens=GEN,
            temp=1.0,
            max_k=1,
            disable_auto_k=True,
            prompt_lookup_enabled=prompt_lookup,
            timing_stats=timing,
        )
    )
    for key, value in timing.items():
        if key.startswith("prompt_lookup_") and isinstance(value, (int, float)):
            stats[key] += value
    return out


def _arm(lm, prompt, *, prompt_lookup: bool, seed0: int):
    stats: Counter = Counter()
    runs = [
        _sample(lm, prompt, prompt_lookup=prompt_lookup, seed=seed0 + i, stats=stats)
        for i in range(SAMPLES)
    ]
    assert all(len(r) == GEN for r in runs)
    return runs, stats


def _copy_agreements(run: tuple, prompt: list[int]) -> int:
    """How many generated tokens equal the prompt's continuation of the
    two tokens before them -- i.e. what a copy draft would have proposed."""
    cont = {(prompt[i], prompt[i + 1]): prompt[i + 2] for i in range(len(prompt) - 2)}
    seq = list(prompt) + list(run)
    start = len(prompt)
    return sum(
        1 for t in range(start, len(seq)) if cont.get((seq[t - 2], seq[t - 1])) == seq[t]
    )


def _homogeneity_pvalues(a: list[tuple], b: list[tuple], prompt: list[int]) -> dict[str, float]:
    """Two-sample tests: every position, adjacent pairs, and copy agreement.

    Marginals and pairs alone are blind to a copy-biased verify, because the
    copied token depends on the two before it and averages out over a de
    Bruijn prompt. The copy-agreement count is the statistic a lossy verify
    moves directly: under exact speculative sampling it has the same law with
    copy drafts on or off.
    """
    pvals: dict[str, float] = {}
    pvals["copy_agreement"] = float(
        scipy_stats.mannwhitneyu(
            [_copy_agreements(r, prompt) for r in a],
            [_copy_agreements(r, prompt) for r in b],
        ).pvalue
    )

    def test(name, xs, ys):
        cx, cy = Counter(xs), Counter(ys)
        cells = sorted(set(cx) | set(cy))
        table = [[cx[c] for c in cells], [cy[c] for c in cells]]
        pvals[name] = float(scipy_stats.chi2_contingency(table)[1])

    for pos in range(GEN):
        test(f"pos{pos}", [r[pos] for r in a], [r[pos] for r in b])
    for pos in range(GEN - 1):
        test(f"pair{pos}", [r[pos : pos + 2] for r in a], [r[pos : pos + 2] for r in b])
    for pos in range(GEN - 2):
        test(f"triple{pos}", [r[pos : pos + 3] for r in a], [r[pos : pos + 3] for r in b])
    return pvals


def test_qwen4_family_qualifies_copy_drafts_under_sampling(qwen4_mtp, monkeypatch):
    from rapid_mlx.spec_decode.mtp.generator import _effective_prompt_lookup_policy

    lm, _prompt = qwen4_mtp
    assert lm.mtp_prompt_lookup_policy.enabled_under_sampling is True
    assert _effective_prompt_lookup_policy(lm).admits_temperature(1.0) is True
    # The operator's disable-only knob still puts sampled requests back on
    # the plain MTP route without touching greedy.
    monkeypatch.setenv("RAPID_MLX_MTP_PROMPT_LOOKUP_SAMPLED", "0")
    policy = _effective_prompt_lookup_policy(lm)
    assert policy.admits_temperature(1.0) is False
    assert policy.admits_temperature(0.0) is True


def test_qwen4_sampled_copy_drafts_leave_the_output_distribution_unchanged(qwen4_mtp):
    lm, prompt = qwen4_mtp
    on, on_stats = _arm(lm, prompt, prompt_lookup=True, seed0=0)
    off, off_stats = _arm(lm, prompt, prompt_lookup=False, seed0=100_000)

    # The copy path was actually exercised, on both of its branches.
    assert on_stats["prompt_lookup_proposals"] > SAMPLES
    assert on_stats["prompt_lookup_accepted_tokens"] > 100
    assert on_stats["prompt_lookup_rejections"] > 100
    assert off_stats["prompt_lookup_proposals"] == 0

    pvals = _homogeneity_pvalues(on, off, prompt.tolist())
    worst = min(pvals, key=pvals.get)
    assert pvals[worst] > ALPHA, (worst, pvals[worst], pvals)


def test_the_distribution_check_catches_a_residual_that_keeps_the_refused_copy(
    qwen4_mtp, monkeypatch
):
    """Power check: the test above must fail on a real lossy verify.

    A residual that leaves the refused copy in play over-emits it; the same
    seeds and statistic must flag that.
    """
    from rapid_mlx.spec_decode.mtp import generator

    lm, prompt = qwen4_mtp
    off, _ = _arm(lm, prompt, prompt_lookup=False, seed0=100_000)
    monkeypatch.setattr(
        generator,
        "_point_mass_residual_distribution",
        lambda target_logprobs, _proposed: mx.exp(target_logprobs),
    )
    broken, stats = _arm(lm, prompt, prompt_lookup=True, seed0=0)
    assert stats["prompt_lookup_rejections"] > 100
    pvals = _homogeneity_pvalues(broken, off, prompt.tolist())
    assert pvals["copy_agreement"] < ALPHA, pvals
