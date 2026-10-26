"""Greedy parity + throughput client for the qwen4 PLE / routed-expert offload.

Talks to an already running ``rapid-mlx serve`` (start it separately). For each
prompt: streams a temperature-0 chat completion, records TTFT, decode tok/s,
sha256 of the generated text, and the delta of the qwen4_offload counters from
/v1/status. Prints one line per prompt; full JSON to --out.

    python -m rapid_mlx.bench.qwen4_offload_bench --port 8197 --label ple-only \
        --out /path/result.json [--prompts code,prose,json,zh,long,copy]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _long_context(part: int = 0) -> str:
    # part 1/2 = different text -> mostly unseen PLE n-gram rows (cold page cache)
    src = (_REPO / "rapid_mlx" / "models" / "qwen4_exp.py").read_text()
    return src[part * 30000 : (part + 1) * 30000]


def _copy_source() -> str:
    src = (_REPO / "rapid_mlx" / "models" / "qwen4_offload.py").read_text()
    return src[:6000]


PROMPTS = {
    "code": "Write a Python function that parses an ISO-8601 duration string like "
    "'P3DT4H5M6S' into total seconds, with input validation and three doctest examples.",
    "prose": "Write a short story of about 300 words about a lighthouse keeper who finds "
    "a message in a bottle written in her own handwriting.",
    "json": "Return ONLY a JSON object describing three fictional employees with fields "
    "id (int), name, department, skills (array of strings), manager_id (int or null). "
    "No prose.",
    "zh": "请用中文详细解释一下什么是混合专家模型（Mixture of Experts），以及它在大语言模型推理时的内存和速度权衡。",
    "long": "Here is a source file:\n\n```python\n"
    + _long_context()
    + "\n```\n\nSummarize what this module implements, listing the main classes and "
    "what each one is responsible for.",
    **{
        f"long{part}": "Here is a source file:\n\n```python\n"
        + _long_context(part)
        + "\n```\n\nSummarize what this module implements."
        for part in (1, 2)
    },
    "copy": "Rewrite the following Python code exactly, but rename the class "
    "`ExpertOffloadManager` to `ExpertCache` everywhere. Output only the code.\n\n```python\n"
    + _copy_source()
    + "\n```",
}

MAX_TOKENS = {"long": 256, "long1": 32, "long2": 32, "copy": 768}


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def _offload(port: int) -> dict:
    try:
        st = _get(f"http://127.0.0.1:{port}/v1/status")
    except Exception:  # noqa: BLE001
        return {}
    out = dict(st.get("qwen4_offload") or {})
    mem = st.get("metal") or {}
    for k in ("active_memory_gb", "peak_memory_gb"):
        if k in mem:
            out[k] = mem[k]
    out["mtp_prompt_lookup"] = st.get("mtp_prompt_lookup")
    return out


def run_one(port: int, model: str, name: str, max_tokens: int) -> tuple[dict, str]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPTS[name]}],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS.get(name, max_tokens),
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    before = _offload(port)
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    text: list[str] = []
    usage: dict = {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("reasoning_content") or "") + (d.get("content") or "")
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text.append(piece)
    total = time.perf_counter() - t0
    after = _offload(port)
    out_text = "".join(text)
    ntok = int(usage.get("completion_tokens") or 0)
    decode_s = total - (ttft or 0.0)
    res = {
        "prompt": name,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": ntok,
        "ttft_s": round(ttft or 0.0, 3),
        "prefill_tok_s": round((usage.get("prompt_tokens") or 0) / ttft, 1) if ttft else None,
        "decode_tok_s": round((ntok - 1) / decode_s, 2) if ntok > 1 and decode_s > 0 else None,
        "total_s": round(total, 2),
        "sha": hashlib.sha256(out_text.encode()).hexdigest()[:16],
        "head": out_text[:80].replace("\n", " "),
    }
    if after:
        h = (after.get("hits") or 0) - (before.get("hits") or 0)
        m = (after.get("misses") or 0) - (before.get("misses") or 0)
        res["hits"], res["misses"] = h, m
        res["hit_rate"] = round(h / (h + m), 4) if h + m else None
        res["load_s"] = round((after.get("load_seconds") or 0) - (before.get("load_seconds") or 0), 2)
        if "ple_rows_gathered" in after:
            res["ple_rows"] = after["ple_rows_gathered"] - before.get("ple_rows_gathered", 0)
            res["ple_s"] = round(after["ple_gather_seconds"] - before.get("ple_gather_seconds", 0), 3)
        res["active_gb"] = after.get("active_memory_gb")
        res["peak_gb"] = after.get("peak_memory_gb")
    return res, out_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8197)
    ap.add_argument("--model", default="default")
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompts", default="code,prose,json,zh,long")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    results = []
    texts = {}
    for rep in range(a.repeat):
        for name in a.prompts.split(","):
            res, txt = run_one(a.port, a.model, name, a.max_tokens)
            res["rep"] = rep
            results.append(res)
            texts[f"{name}.{rep}"] = txt
            keys = ("prompt", "rep", "sha", "completion_tokens", "ttft_s", "prefill_tok_s",
                    "decode_tok_s", "hit_rate", "ple_rows", "ple_s", "load_s", "active_gb", "peak_gb")
            print(a.label, " ".join(f"{k}={res.get(k)}" for k in keys), flush=True)
    status = _offload(a.port)
    Path(a.out).write_text(json.dumps(
        {"label": a.label, "results": results, "status": status, "texts": texts}, indent=1,
        ensure_ascii=False))


if __name__ == "__main__":
    main()
