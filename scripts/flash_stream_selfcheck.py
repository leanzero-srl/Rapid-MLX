"""Self-check: harness `stream` (layer-streamed) == one direct full prefill.

Usage: flash_stream_selfcheck.py <dir with tok_tiny/, tp/ (pipe --pairs), ts/ (stream)> <prompts.json>
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pathlib import Path

import flash_pipeline_harness as h
import mlx.core as mx
from mlx_lm.generate import _make_cache

S = Path(sys.argv[1])
P = Path(sys.argv[2])
prompts = {p["name"]: p["ids"] for p in h._load_prompts(P)}
src = json.loads((S / "tp" / "rank1.json").read_text())
stream = json.loads((S / "ts" / "stream.json").read_text())["report"]
model = h._load_truncatable(S / "tok_tiny", None)
mx.eval(model.parameters())
groups = [[(r["name"], r["tokens"], r["trace"])] for r in src["results"]] + [
    list(zip(p["names"], p["tokens"], p["traces"])) for p in src["pairs"]
]
text = model.language_model
k = 0
for g in groups:
    seqs = [prompts[n] + t[:-1] for n, t, _ in g]
    toks, pad = h.pipe._left_pad(seqs, 0)
    cache = model.make_cache() if len(g) == 1 else _make_cache(model, pad, None)
    _, hidden = model(toks, cache=cache, return_hidden=True)
    mx.eval(hidden)
    for row, (n, t, tr) in enumerate(g):
        start = pad[row] + len(prompts[n]) - 1
        pos = mx.arange(start, start + len(t))
        logits = text.lm_head(
            text.model.hyper_connection_mixer(hidden[row : row + 1, pos])
        )[0]
        am = mx.argmax(logits, -1).tolist()
        matches = sum(a == b for a, b in zip(am, t))
        lp = logits.astype(mx.float32)
        lp = lp - mx.logsumexp(lp, -1, keepdims=True)
        worst = max(
            mx.max(
                mx.abs(
                    lp[i][mx.array(tr[i]["top_ids"])] - mx.array(tr[i]["top_logprobs"])
                )
            ).item()
            for i in range(len(t))
        )
        print(
            n,
            len(g),
            "direct",
            matches,
            round(worst, 6),
            "stream",
            stream[k]["argmax_matches"],
            stream[k]["max_abs_logprob_diff_top5"],
        )
        k += 1
