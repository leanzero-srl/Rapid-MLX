"""Pipeline-parallel qwen4_exp: split accounting, guardrails and 2/3-rank parity.

The parity tests launch real MLX distributed ranks with ``mlx.launch --backend
ring`` on 127.0.0.1 and compare them against the fork's unmodified single
process forward on the same quantized random-init checkpoint.
"""

import json
import random
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
pytestmark = pytest.mark.requires_mlx

from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.generate import _make_cache  # noqa: E402
from mlx_lm.utils import (  # noqa: E402
    load_model,
    quantize_model,
    save_config,
    save_model,
)

from rapid_mlx.distributed import pipeline_qwen4 as pipe  # noqa: E402
from rapid_mlx.models.qwen4_exp import Model, ModelArgs, TextModelArgs  # noqa: E402
from rapid_mlx.utils.tokenizer import _register_vendored_archs  # noqa: E402

PROMPTS = [
    [
        7,
        19,
        3,
        88,
        42,
        255,
        17,
        5,
        200,
        61,
        9,
        33,
        250,
        14,
        71,
        8,
        99,
        123,
        4,
        55,
        66,
        77,
        18,
        29,
        31,
        47,
        58,
        69,
        70,
        81,
        92,
        103,
        114,
        125,
        136,
        147,
        158,
    ],
    [
        12,
        34,
        56,
        78,
        90,
        11,
        22,
        255,
        44,
        13,
        24,
        35,
        46,
        57,
        68,
        79,
        80,
        91,
        102,
        113,
        124,
    ],
]
# Below the QSA budget (physical length // ratio <= block_topk) attention runs
# on the ordinary left-padded causal mask, and decode then crosses into the
# sparse selection: both regimes and the transition are covered.
SHORT_PROMPTS = [[5, 9, 200, 31, 77, 18, 42], [101, 12, 7, 64]]
DECODE_TOKENS = 64


def _tiny_text_config() -> dict:
    """Every Qwen4-Exp layer kind at dims the 4-bit quantizer accepts."""
    args = TextModelArgs(
        hidden_size=64,
        num_hidden_layers=8,
        vocab_size=256,
        max_position_embeddings=4096,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        hc_count=4,
        hc_lowrank=64,
        layer_types=["linear_attention"] * 3
        + ["full_attention"]
        + ["linear_attention"] * 3
        + ["full_attention"],
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=64,
        indexer_budget=8,
        indexer_compress_ratio=2,
        ple_layer_ids=[2],
        ple_embed_dim=64,
        heads_per_ngram=1,
        ngram_vocab_size_base=257,
        make_ngram_vocab_size_divisible_by=64,
        split_ngram_parts=4,
        eos_token_id=255,
    )
    config = asdict(args)
    config["model_type"] = "qwen4_exp_text"
    return config


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("qwen4_pipeline_ckpt")
    mx.random.seed(20260923)
    text_config = _tiny_text_config()
    config = {"model_type": "qwen4_exp", "text_config": text_config}
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=text_config))
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    model, config = quantize_model(model, config, group_size=64, bits=4)
    mx.eval(model.parameters())
    save_model(path, model, donate_model=False)
    save_config(config, path / "config.json")
    return path


def _reference(
    model_dir: Path,
    prompts: list[list[int]],
    max_tokens: int,
    prefill_step: int | None = None,
):
    """The fork's own single-process forward, mlx-lm's generate_step shape."""
    _register_vendored_archs()
    model, _ = load_model(model_dir)
    width = max(len(prompt) for prompt in prompts)
    padding = [width - len(prompt) for prompt in prompts]
    tokens = mx.array([[0] * pad + p for pad, p in zip(padding, prompts)])

    def fresh_cache():
        if len(prompts) == 1:
            return model.make_cache()
        return _make_cache(model, padding, None)

    logits = model(tokens, cache=fresh_cache())
    mx.eval(logits)

    cache = fresh_cache()
    prefix = tokens[:, :-1]
    step_size = prefill_step or prefix.shape[1]
    for offset in range(0, prefix.shape[1], step_size):
        model(prefix[:, offset : offset + step_size], cache=cache)
        mx.eval([layer.state for layer in cache])
    current = tokens[:, -1:]
    generated = []
    for _ in range(max_tokens):
        step = model(current, cache=cache)
        next_tokens = mx.argmax(step[:, -1, :], axis=-1).astype(mx.int32)
        mx.eval(next_tokens, [layer.state for layer in cache])
        generated.append(next_tokens.tolist())
        current = next_tokens[:, None]
    return np.array(logits.astype(mx.float32)), np.array(generated).T


def _free_port_block(count: int) -> int:
    """Consecutive listen ports BELOW the kernel's ephemeral range.

    Inside the ephemeral range a rank's outgoing connect can be handed the
    port a later rank is about to listen on ("[ring] Couldn't bind socket
    (error: 48)" — measured on this test's first flaky runs).
    """
    first = int(
        subprocess.run(
            ["sysctl", "-n", "net.inet.ip.portrange.first"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    rng = random.Random()
    for _ in range(256):
        base = rng.randrange(first // 2, first - count)
        held = []
        try:
            for offset in range(count):
                sock = socket.socket()
                held.append(sock)
                sock.bind(("127.0.0.1", base + offset))
        except OSError:
            continue
        finally:
            for sock in held:
                sock.close()
        return base
    raise RuntimeError("no free consecutive port block on 127.0.0.1")


def _launch(
    model_dir: Path,
    ranks: int,
    split: str | None,
    dump: Path,
    prompts,
    extra: tuple[str, ...] = (),
    expect_dump: bool = True,
):
    launcher = Path(sys.executable).parent / "mlx.launch"
    if not launcher.exists():
        pytest.skip("mlx.launch is not installed next to this interpreter")
    command = [
        str(launcher),
        "--backend",
        "ring",
        "-n",
        str(ranks),
        "--hosts",
        "127.0.0.1",
        "--starting-port",
        str(_free_port_block(ranks)),
        "--",
        sys.executable,
        "-m",
        "rapid_mlx.distributed.pipeline_qwen4",
        "run",
        "--model",
        str(model_dir),
        "--prompt-ids",
        ";".join(",".join(map(str, row)) for row in prompts),
        "--max-tokens",
        str(DECODE_TOKENS),
        "--context",
        "256",
        "--dump",
        str(dump),
    ]
    if split:
        command += ["--split", split]
    command += list(extra)
    # A hang guard for the test harness only; the engine itself has no clock.
    completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
    # mlx.launch can exit 0 after a rank died (measured: a rank's bind error
    # still returned 0), so the dump itself is the proof the run completed.
    if expect_dump:
        assert completed.returncode == 0 and dump.exists(), (
            completed.stdout + completed.stderr
        )
    return completed


def test_checkpoint_bytes_match_loaded_parameters(tiny_checkpoint):
    args = pipe.load_text_args(tiny_checkpoint)
    ckpt = pipe.read_checkpoint_bytes(tiny_checkpoint, args.num_hidden_layers)
    _register_vendored_archs()
    model, _ = load_model(tiny_checkpoint)
    per_layer = [
        sum(value.nbytes for _, value in tree_flatten(layer.parameters()))
        for layer in model.layers
    ]
    inner = model.language_model.model
    assert ckpt.layer_bytes == per_layer
    assert ckpt.head_bytes == sum(
        v.nbytes for _, v in tree_flatten(inner.embed_tokens.parameters())
    )
    assert ckpt.tail_bytes == sum(
        v.nbytes for _, v in tree_flatten(inner.hyper_connection_mixer.parameters())
    ) + sum(
        v.nbytes for _, v in tree_flatten(model.language_model.lm_head.parameters())
    )
    assert ckpt.activation_bytes == 2


def test_checkpoint_reader_refuses_an_unowned_tensor(tmp_path, tiny_checkpoint):
    for item in tiny_checkpoint.iterdir():
        shutil.copy(item, tmp_path / item.name)
    stray = {"language_model.model.stray.weight": mx.zeros((2,))}
    mx.save_safetensors(str(tmp_path / "model-stray.safetensors"), stray)
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    index["weight_map"]["language_model.model.stray.weight"] = "model-stray.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="no pipeline owner"):
        pipe.read_checkpoint_bytes(tmp_path, 8)


def _nodes(*budgets_mib: int) -> list[pipe.NodeBudget]:
    return [
        pipe.NodeBudget(f"n{i}", budget * 4 * 2**20, budget * 2**20, "test")
        for i, budget in enumerate(budgets_mib)
    ]


def test_split_equalizes_utilization_and_follows_budget(tiny_checkpoint):
    args = pipe.load_text_args(tiny_checkpoint)
    ckpt = pipe.read_checkpoint_bytes(tiny_checkpoint, args.num_hidden_layers)
    even = pipe.plan_pipeline(
        args, ckpt, _nodes(64, 64), context=256, batch=2, prefill_step=256
    )
    skewed = pipe.plan_pipeline(
        args, ckpt, _nodes(64, 16), context=256, batch=2, prefill_step=256
    )
    # The smaller second node receives fewer layers.
    assert skewed.stages[1].end - skewed.stages[1].start < (
        even.stages[1].end - even.stages[1].start
    )
    # The chosen cut is the best contiguous one: no other cut has a lower
    # worst-rank utilization.
    worst = max(stage.utilization for stage in skewed.stages)
    for cut in range(1, args.num_hidden_layers):
        other = pipe.plan_pipeline(
            args,
            ckpt,
            _nodes(64, 16),
            context=256,
            batch=2,
            prefill_step=256,
            starts=[0, cut],
        )
        assert worst <= max(stage.utilization for stage in other.stages) + 1e-12


def test_preflight_refuses_with_the_numbers(tiny_checkpoint):
    args = pipe.load_text_args(tiny_checkpoint)
    ckpt = pipe.read_checkpoint_bytes(tiny_checkpoint, args.num_hidden_layers)
    nodes = [
        pipe.NodeBudget("big", 2**30, 2**30, "test"),
        pipe.NodeBudget("starved", 2**20, 2**12, "test"),
    ]
    plan = pipe.plan_pipeline(args, ckpt, nodes, context=256, batch=1, prefill_step=256)
    with pytest.raises(pipe.PipelineDoesNotFitError) as refusal:
        pipe.require_fit(plan)
    message = str(refusal.value)
    assert "starved" in message and "DOES NOT FIT" in message and "GiB" in message


def test_guardrails_are_ratios_of_this_node(monkeypatch):
    calls = {}
    monkeypatch.setattr(mx, "set_memory_limit", lambda v: calls.setdefault("memory", v))
    monkeypatch.setattr(mx, "set_wired_limit", lambda v: calls.setdefault("wired", v))
    monkeypatch.setattr(mx, "set_cache_limit", lambda v: calls.setdefault("cache", v))
    monkeypatch.setattr(
        mx, "device_info", lambda: {"max_recommended_working_set_size": 2**40}
    )
    total = 96 * 2**30
    node = pipe.NodeMemory(total, total // 2, 50, 1)
    limits = pipe.apply_memory_guardrails(node, 10 * 2**30)
    assert calls["memory"] == int(total * pipe.MEMORY_LIMIT_RATIO)
    assert calls["wired"] == int(total * pipe.WIRED_LIMIT_RATIO)
    assert calls["cache"] == calls["memory"] - 10 * 2**30
    assert limits["budget"] == total // 2 - int(total * pipe.PRESSURE_FLOOR_RATIO)
    with pytest.raises(pipe.PipelineDoesNotFitError, match="budget"):
        pipe.apply_memory_guardrails(node, total)


def test_node_memory_reads_the_kernel_counters():
    node = pipe.measure_node_memory()
    assert node.total_bytes > 0
    assert 0 <= node.free_percent <= 100
    assert 0 < node.available_bytes <= node.total_bytes
    vm_stat = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(vm_stat.split("page size of ")[1].split()[0])
    pages = {
        line.split(":")[0].strip(): int(line.split(":")[1].strip(" ."))
        for line in vm_stat.splitlines()[1:]
        if ":" in line and line.split(":")[1].strip(" .").isdigit()
    }
    # vm_stat's "Pages free" is already free - speculative.
    expected = (
        pages["Pages free"] + pages["File-backed pages"] + pages["Pages purgeable"]
    ) * page
    assert abs(node.available_bytes - expected) < 0.02 * node.total_bytes
    assert node.pressure_level in (1, 2, 4)


def test_slice_drops_unowned_modules(tiny_checkpoint):
    _register_vendored_archs()
    first, _ = load_model(tiny_checkpoint, lazy=True)
    pipe.slice_model(first, 0, 2, 0, 3)
    assert len(first.layers) == 3
    assert "embed_tokens" in first.language_model.model
    assert "hyper_connection_mixer" not in first.language_model.model
    assert "lm_head" not in first.language_model
    last, _ = load_model(tiny_checkpoint, lazy=True)
    pipe.slice_model(last, 1, 2, 3, 8)
    assert [layer.layer_type for layer in last.layers][0] == "qwen_sparse_attention"
    assert "embed_tokens" not in last.language_model.model
    assert "lm_head" in last.language_model


@pytest.mark.parametrize(
    ("ranks", "split", "prompts", "prefill_step"),
    [
        (2, None, PROMPTS, None),  # the planner's own split for this budget
        (2, "1", PROMPTS, None),  # the PLE n-gram layer and caches on rank 1
        (2, "3", PROMPTS, None),  # rank 1 opens on a QSA layer
        (2, "3", SHORT_PROMPTS, None),  # dense masked attention, sparse in decode
        (2, "3", PROMPTS, 8),  # chunked prefill, as long real prompts run
        (3, "2,5", PROMPTS, None),  # a middle rank that receives and forwards
    ],
    ids=[
        "planned",
        "ple-on-rank1",
        "qsa-first",
        "short-dense",
        "chunked-prefill",
        "three-ranks",
    ],
)
def test_pipeline_matches_single_process(
    tiny_checkpoint, tmp_path, ranks, split, prompts, prefill_step
):
    ref_logits, ref_tokens = _reference(
        tiny_checkpoint, prompts, DECODE_TOKENS, prefill_step
    )
    dump = tmp_path / "pipeline.npz"
    extra = () if prefill_step is None else ("--prefill-step", str(prefill_step))
    stdout = _launch(tiny_checkpoint, ranks, split, dump, prompts, extra).stdout
    result = np.load(dump)
    padding = [max(map(len, prompts)) - len(p) for p in prompts]
    diffs = [
        float(np.max(np.abs(result["logits"][row, pad:] - ref_logits[row, pad:])))
        for row, pad in enumerate(padding)
    ]
    print(f"ranks={ranks} split={result['starts'].tolist()} max|dlogits|={diffs}")
    assert max(diffs) == 0.0, stdout
    np.testing.assert_array_equal(result["tokens"], ref_tokens)
    assert result["tokens"].shape == (len(prompts), DECODE_TOKENS)
    if split:
        assert result["starts"].tolist() == [0, *map(int, split.split(","))]


def test_memory_guard_trip_stops_every_rank_on_the_same_step(tiny_checkpoint, tmp_path):
    dump = tmp_path / "never.npz"
    completed = _launch(
        tiny_checkpoint,
        2,
        "4",
        dump,
        PROMPTS,
        extra=("--guard-limit-gib", "0"),
        expect_dump=False,
    )
    output = completed.stdout + completed.stderr
    assert not dump.exists()
    # The trip rides the step's all_sum, so every rank raises the guard's own
    # error instead of one rank dying and the other blocking in recv/all_sum
    # (a hang would hit the harness timeout). mlx.launch exits 0 here too.
    stops = [
        line for line in output.splitlines() if "PipelineMemoryStopError: rank" in line
    ]
    assert stops and all("MLX active memory" in line for line in stops), output


_SLOW_PEER = """
import sys, time
import mlx.core as mx
from rapid_mlx.distributed.pipeline_qwen4 import receive_stream
group = mx.distributed.init(strict=True)
if group.rank() == 0:
    x = mx.ones((1, 64, 1024), dtype=mx.bfloat16)
    mx.eval(x)
    time.sleep(float(sys.argv[1]))
    mx.eval(mx.distributed.send(x, 1, group=group))
else:
    hidden = receive_stream((1, 64, 1024), mx.bfloat16, 0, group)
    total = (hidden @ mx.ones((1024, 1024), dtype=mx.bfloat16)).sum()
    mx.eval(total)
    print("RECEIVED", total.item(), flush=True)
"""


def test_a_slow_upstream_rank_cannot_trip_the_gpu_watchdog(tmp_path):
    """Upstream takes longer than the Metal watchdog (6 s fails a fused recv)."""
    script = tmp_path / "slow_peer.py"
    script.write_text(_SLOW_PEER)
    launcher = Path(sys.executable).parent / "mlx.launch"
    completed = subprocess.run(
        [
            str(launcher),
            "--backend",
            "ring",
            "-n",
            "2",
            "--hosts",
            "127.0.0.1",
            "--starting-port",
            str(_free_port_block(2)),
            "--",
            sys.executable,
            str(script),
            "8",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = completed.stdout + completed.stderr
    assert "RECEIVED 67108864.0" in output, output
    assert "Timeout" not in output, output


def test_a_node_below_its_pressure_floor_refuses_instead_of_crashing(tiny_checkpoint):
    args = pipe.load_text_args(tiny_checkpoint)
    ckpt = pipe.read_checkpoint_bytes(tiny_checkpoint, args.num_hidden_layers)
    starved = pipe.NodeMemory(96 * 2**30, 10 * 2**30, 10, 2)
    assert starved.budget_bytes == 0
    nodes = [
        pipe.NodeBudget("big", 2**36, 2**35, "test"),
        pipe.NodeBudget("starved", starved.total_bytes, starved.budget_bytes, "test"),
    ]
    plan = pipe.plan_pipeline(args, ckpt, nodes, context=256, batch=1, prefill_step=256)
    assert not plan.stages[1].total_bytes <= plan.stages[1].node.budget_bytes
    assert "DOES NOT FIT" in pipe.format_plan(plan)
    assert pipe.plan_json(plan, pipe.wire_bytes_per_token(args, 2, 2))["fits"] is False
