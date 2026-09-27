"""A finished row's cache becomes its prefix-cache entry on the pipeline split (goose Q-179).

goose E2E #5b (3.0.58, Flash on the pipeline split, 2026-09-27): agent calls read
40,058 -> 46,926 of their prompt from the cache (97%) until an answer of 6,885
tokens took the conversation to 53,409; every call after it read 0 cached of
60,356 / 60,502 / 62,240 / 62,516 tokens, about 4 minutes each for answers of
73-327 tokens.  /v1/status: entries 0, evicted 37, skipped
{no_room_beside_the_batch: 6}, limit_bytes [5.74 GB, 9.92 GB].  The snapshot was
a COPY of the prefilling row's cache taken beside the running batch; with two
rows each reserving the whole 262,144-token context (goose sends no
max_tokens), rank 1 had 0.91 GB of room beside them and the 53,409-token copy
needed 0.98 GB.  Now the row keeps only a boundary record (its recurrent
states and QSA rings) and its own cache, cut back to the boundary, becomes the
entry when it leaves; a restore whose entry is evicted in the same plan moves
it.  One copy.

Nothing here launches a rank: rank 0's real tick loop runs the tiny float32
qwen4_exp on a singleton group, or the scheduler runs alone over budgets the
planner derives from Flash's own dimensions.  (The prefix-index unit tests that
lived in test_pipeline_qwen4_serve.py moved here: that file launches ranks.)
"""

import types

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
pytest.importorskip("fastapi")
pytestmark = pytest.mark.requires_mlx

from rapid_mlx.distributed import pipeline_qwen4 as pipe  # noqa: E402
from rapid_mlx.distributed import pipeline_qwen4_serve as serve  # noqa: E402

from .test_pipeline_qwen4_continuous import (  # noqa: E402, F401 - fixtures
    PREFILL_STEP,
    _job,
    _run_rank0,
    _solo,
    _wait,
    group,
    stage,
)

# ---------------------------------------------------------------------------
# rank 0's index alone (moved from test_pipeline_qwen4_serve.py)
# ---------------------------------------------------------------------------


class _FakeKv:
    """Reserves rows x longest; a snapshot costs its tokens on rank 0, double on rank 1."""

    def __init__(self, budgets: list[int]):
        self.budgets = budgets

    def reserve(self, lengths: list[int]) -> list[int]:
        return [len(lengths) * max(lengths)] * len(self.budgets)

    def entry_bytes(self, tokens: int) -> list[int]:
        return [tokens, 2 * tokens]

    def record_bytes(self) -> list[int]:
        return [1, 2]

    def fits(self, need: list[int]) -> bool:
        return all(n <= budget for n, budget in zip(need, self.budgets))


def _row(ids: list[int], boundary: int = 0):
    return serve._Row(ids, 4, 0.0, 1.0, boundary=boundary)


def _store(index, row) -> None:
    """The row left: what rank 0's on_stored callback does once every rank adopted its cache."""
    index.leaving([row])
    index.settle([0] * len(index.kv.budgets))
    index.stored(
        row.store_id, tuple(row.ids[: row.store_at]), index.kv.entry_bytes(row.store_at)
    )


def test_prefix_index_restores_only_an_exact_prefix_that_leaves_a_token_to_feed():
    """goose Q-164's shape on this split: a prompt the cache holds whole is never
    restored whole — the restore always leaves its last token to prefill (the
    hybrid layers keep only exact prefixes, so it prefills the whole prompt)."""
    index = serve._PrefixIndex(_FakeKv([1000, 1000]))
    cold = _row(list(range(20)), boundary=12)
    assert index.admit(cold, index.kv.reserve([24])) == []
    assert (cold.reuse_id, cold.cached, cold.store_at) == (0, 0, 12)
    _store(index, cold)

    warm = _row([*range(12), 99, 98, 97], boundary=13)
    index.admit(warm, index.kv.reserve([19]))
    assert (warm.reuse_id, warm.cached) == (cold.store_id, 12)
    assert warm.store_at == 13  # the longer boundary is a new entry

    other = _row([7, *range(1, 20)], boundary=12)
    index.admit(other, index.kv.reserve([24]))
    assert (other.reuse_id, other.cached) == (0, 0)  # the first token differs

    exact = _row(list(range(12)))
    index.admit(exact, index.kv.reserve([16]))
    # The prompt IS the entry: nothing would be left to feed, so no restore.
    assert exact.cached == 0
    assert serve.prefill_chunks(exact.cached, len(exact.ids), 4) != []
    assert index.status()["hits"] == 1


def test_prefix_index_yields_its_bytes_to_the_batch_oldest_first_hits_refresh():
    index = serve._PrefixIndex(_FakeKv([200, 200]))
    first = _row([1] * 30, boundary=20)
    index.admit(first, index.kv.reserve([30]))
    _store(index, first)  # held [20, 40]
    second = _row([2] * 30, boundary=20)
    index.admit(second, index.kv.reserve([30]))
    _store(index, second)  # held [40, 80]

    hit = _row([1] * 25)
    assert index.admit(hit, index.kv.reserve([25])) == []
    assert hit.reuse_id == first.store_id  # `first` is now the most recent

    # A row joining one running row reserves 150 between them, leaving room 50
    # on each rank; rank 1 holds 80, so the least recently used entry
    # (`second`) goes, and only it.
    joiner = _row([3] * 10)
    assert index.admit(joiner, index.kv.reserve([75, 75])) == [second.store_id]
    assert joiner.store_id == 0 and joiner.reuse_id == 0
    assert index.status()["bytes"] == [20, 40]
    assert index.status()["evicted"] == 1


def test_a_snapshot_is_directed_beside_a_full_batch_and_kept_once_its_row_leaves():
    """Q-179: the row's record fits beside a batch that leaves no room for a copy."""
    index = serve._PrefixIndex(_FakeKv([100, 100]))
    big = _row([5] * 60, boundary=40)  # rank 1's copy would need 80 beside 60
    assert index.admit(big, index.kv.reserve([60])) == []
    assert (big.store_at, index.pending) == (40, {big.store_id: [1, 2]})
    assert index.status()["skipped"] == {}
    # It leaves: the rows that stay need nothing, its entry [40, 80] fits.
    _store(index, big)
    assert index.status()["bytes"] == [40, 80] and index.pending == {}


# ---------------------------------------------------------------------------
# the entry a leaving row's cache becomes answers as a cold prefill (tiny model)
# ---------------------------------------------------------------------------

CONTEXT = 1024
P1 = [(i * 37 + 11) % 250 for i in range(83)]
BOUNDARY = 71  # odd: the QSA ring (ratio 2) holds half a group there
P2 = P1[:BOUNDARY] + [(i * 13 + 5) % 250 for i in range(29)]
LONG = [(i * 7 + 3) % 250 for i in range(40)]


def _tiny_plan(stage, context: int, slots: int):
    args = stage.args
    ckpt = pipe.CheckpointBytes(
        layer_bytes=[0] * args.num_hidden_layers,
        head_bytes=0,
        tail_bytes=0,
        excluded_bytes={},
        activation_bytes=4,
    )
    node = pipe.NodeBudget("cpu", 2**36, 2**35, "test")
    return pipe.plan_pipeline(
        args,
        ckpt,
        [node],
        context=context,
        batch=slots,
        prefill_step=PREFILL_STEP,
        starts=[0],
    )


def _serve_rank0(stage, group):
    kv = serve._KvBudget(_tiny_plan(stage, CONTEXT, slots=2), PREFILL_STEP)
    state = serve._State(
        served="tiny",
        context=CONTEXT,
        max_batch=kv.rows(),
        kv=kv,
        prefix=serve._PrefixIndex(kv),
    )
    store = serve._PrefixStore()
    engine = serve._Engine(stage, None, PREFILL_STEP, store=store)
    return state, engine, store, _run_rank0(state, engine, group)


def _stop(state, loop) -> None:
    state.jobs.put(None)
    loop.join(timeout=60)
    assert not loop.is_alive()


@pytest.mark.parametrize("beside", [False, True], ids=["alone", "in_a_batch"])
def test_a_leaving_rows_cache_becomes_an_entry_that_answers_as_a_cold_run(
    stage, group, beside
):
    """Turn 1 is directed to snapshot at 71 of its 83 tokens and writes 8
    more; the row leaves (alone, or extracted from a two-row batch) and its
    cache, cut back to 71, is the entry.  Turn 2 shares those 71 tokens: it
    restores them, prefills its last 41, and every token it writes is what a
    cold run of its whole prompt writes."""
    state, engine, store, loop = _serve_rank0(stage, group)
    other = None
    if beside:
        other = _job(LONG, 400)
        state.jobs.put(other)
        _wait(lambda: other.produced >= 2, "the other row never decoded")
    first = _job(P1, 8)
    first.row.boundary = BOUNDARY
    state.jobs.put(first)
    _wait(lambda: first.finished, "turn 1 never finished")
    _wait(
        lambda: state.prefix.status()["stored"] == 1, "turn 1's cache was never adopted"
    )
    assert first.row.store_at == BOUNDARY
    assert list(store.entries) == [first.row.store_id]
    entry = store.entries[first.row.store_id]
    held = serve._held_bytes(entry)
    assert held <= state.kv.entry_bytes(BOUNDARY)[0], (
        held,
        state.kv.entry_bytes(BOUNDARY),
    )
    # Measured on every rank and agreed in KiB (``_agree_bytes``).
    assert state.prefix.status()["bytes"] == [-(-held // 1024) * 1024]
    assert first.tokens() == _solo(stage, P1, 8)

    second = _job(P2, 12)
    state.jobs.put(second)
    _wait(lambda: second.finished, "turn 2 never finished")
    assert (second.row.reuse_id, second.row.cached) == (first.row.store_id, BOUNDARY)
    assert second.tokens() == _solo(stage, P2, 12)
    if other is not None:
        other.cancelled = True
        _wait(lambda: not state.active, "the other row never left")
        got = other.tokens()
        assert got == _solo(stage, LONG, len(got))
    _stop(state, loop)


def test_a_restore_evicted_in_its_own_plan_moves_the_entry(stage, group, monkeypatch):
    """The entry the restore reads is also evicted by the plan that admits it
    (the budget holds the rows and nothing beside): the row takes the entry
    itself, so the entry and its restore never sit side by side."""
    state, engine, store, loop = _serve_rank0(stage, group)
    first = _job(P1, 4)
    first.row.boundary = BOUNDARY
    state.jobs.put(first)
    _wait(
        lambda: state.prefix.status()["stored"] == 1, "turn 1's cache was never adopted"
    )
    entry = store.entries[first.row.store_id]

    admit = state.prefix.admit

    def everything_goes(row, need):
        evict = admit(row, need)
        with state.prefix.lock:
            gone = list(state.prefix.entries)
            state.prefix.entries.clear()
        return evict + gone

    monkeypatch.setattr(state.prefix, "admit", everything_goes)
    taken = []
    take = store.take

    def recording(entry_id, move=False):
        cache = take(entry_id, move)
        taken.append((move, cache is entry))
        return cache

    monkeypatch.setattr(store, "take", recording)
    second = _job(P2, 6)
    state.jobs.put(second)
    _wait(lambda: second.finished, "turn 2 never finished")
    assert taken == [(True, True)]
    assert first.row.store_id not in store.entries
    assert second.tokens() == _solo(stage, P2, 6)
    _stop(state, loop)


# ---------------------------------------------------------------------------
# E2E #5b's conversation over the budgets the planner derives for Flash
# ---------------------------------------------------------------------------

# Qwen3.8-Flash-Next-4bit's dimensions (its config.json text_config), all the
# planner's state and workspace formulas read.
FLASH = types.SimpleNamespace(
    layer_types=["linear_attention"] * 3 * 12,
    num_hidden_layers=48,
    num_key_value_heads=2,
    head_dim=256,
    indexer_compress_ratio=4,
    indexer_head_dim=128,
    linear_num_key_heads=16,
    linear_key_head_dim=128,
    linear_num_value_heads=48,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    ple_layer_ids=[2],
    hc_count=4,
    hidden_size=2560,
    ple_conv_kernel_size=4,
    ngram_size=3,
    vocab_size=248320,
    max_position_embeddings=262144,
)
FLASH.layer_types = [
    "qwen_sparse_attention" if index % 4 == 3 else "linear_attention"
    for index in range(48)
]
FLASH_CONTEXT = 262_144


def _flash_kv() -> serve._KvBudget:
    """E2E #5b's plan: ranks [0, 17) and [17, 48), 2 slots of 262,144 tokens,
    256-token chunks, the vision encode's measured workspace on rank 0."""
    ckpt = pipe.CheckpointBytes(
        layer_bytes=[0] * 48,
        head_bytes=0,
        tail_bytes=0,
        excluded_bytes={},
        activation_bytes=2,
    )
    nodes = [
        pipe.NodeBudget("mac", 128 << 30, 73 << 30, "test"),
        pipe.NodeBudget("studio", 96 << 30, 63 << 30, "test"),
    ]
    plan = pipe.plan_pipeline(
        FLASH,
        ckpt,
        nodes,
        context=FLASH_CONTEXT,
        batch=2,
        prefill_step=256,
        starts=[0, 17],
        vision=pipe.VisionCost(weight_bytes=448_092_512, workspace_bytes=1_229_312_000),
    )
    return serve._KvBudget(plan, 256)


def test_the_flash_budgets_are_the_ones_e2e_5b_reported():
    kv = _flash_kv()
    assert [round(b / 1e9, 2) for b in kv.budgets] == [5.74, 9.92]
    # The copy beside the batch, measured shape: beside a second row that
    # reserves the whole context, rank 1's room is below a 53,409-token entry
    # and above a 46,926-token one.
    room = [
        budget - need
        for budget, need in zip(
            kv.budgets,
            kv.admission([(48_000, FLASH_CONTEXT)], [(53_409, FLASH_CONTEXT)]),
        )
    ]
    assert kv.entry_bytes(46_926)[1] <= room[1] < kv.entry_bytes(53_409)[1]


def test_a_60k_conversation_beside_a_second_chat_keeps_its_prefix_cached():
    """E2E #5b's tool steps of one conversation, the live prompts from 53,409 on
    (60,356 / 60,502 / 62,240 / 62,516) and the 6,885-token answer, while a
    second chat decodes the whole time, both reserving the whole context.
    Each step's prompt is the previous one up to its boundary (goose's
    transient tail moves to the newest tool result), the answer and a tool
    result.  Every step restores the whole previous boundary: >= 90% of its
    prompt, except the step right after the 6,885-token answer, which must
    read that answer once (88%).  On lz-pipeline-qwen4.12 every step from the
    53,409-token one on restored nothing."""
    kv = _flash_kv()
    state = serve._State(
        served="flash",
        context=FLASH_CONTEXT,
        max_batch=kv.rows(),
        kv=kv,
        prefix=serve._PrefixIndex(kv),
    )
    scheduler = serve._Scheduler(state, serve._Engine(None, None, 256))
    other = _job([7] * 30_000, FLASH_CONTEXT - 30_000 - 1)
    other.produced = 1
    scheduler.running = [other]
    scheduler._publish()

    prompts = [47_500, 51_000, 53_409, 60_356, 60_502, 62_240, 62_516]
    answers = [200, 250, 6_885, 73, 327, 120, 200]
    tail = 300  # goose's transient tail: after each prompt's boundary
    fresh = iter(range(1, 10**6))
    ids: list[int] = []
    cached, boundaries = [], []
    for prompt, answer in zip(prompts, answers):
        keep = boundaries[-1] if boundaries else 0
        ids = ids[:keep] + [next(fresh) % 248_000 for _ in range(prompt - keep)]
        job = _job(ids, FLASH_CONTEXT - prompt - 1)
        job.row.boundary = prompt - tail
        boundaries.append(prompt - tail)
        state.jobs.put(job)
        plan = scheduler.plan(block=False)
        assert plan.joiner is job.row, "the step was not admitted beside the other chat"
        _apply_plan(scheduler, plan)
        cached.append(job.row.cached)
        while scheduler.engine.prefilling:
            _chunk(scheduler)
        job.produced = answer
        job.finished = True
        plan = scheduler.plan(block=False)
        assert plan.leave == [1] and plan.joiner is None
        _apply_plan(scheduler, plan)
    assert cached == [0, *boundaries[:-1]], (cached, boundaries)
    shares = [round(c / p, 3) for c, p in zip(cached, prompts)]
    assert shares[3] == round(53_109 / 60_356, 3)  # the step after 6,885 tokens
    assert all(share >= 0.9 for share in shares[1:3] + shares[4:]), shares
    status = state.prefix.status()
    assert status["skipped"] == {}, status
    assert status["entries"] == 1 and status["entry_tokens"] == [boundaries[-1]]
    # One copy: the entry sits where the rows leave room, on every rank.
    held = state.prefix._held()
    need = kv.held(scheduler._decoding(scheduler.running), [])
    assert kv.fits([h + n for h, n in zip(held, need)])


def _apply_plan(scheduler: serve._Scheduler, plan: serve._Plan) -> None:
    """``_ticks``' plan step on rank 0 with the engine's rows faked: a leaving
    row that took a snapshot hands its cache over (``on_stored``) unless the
    plan evicts it."""
    engine = scheduler.engine
    for index in plan.leave:
        row = scheduler.running[index].row
        if row.store_id and row.store_id not in plan.evict:
            scheduler._stored(row, scheduler.state.kv.entry_bytes(row.store_at))
    if plan.joiner is not None:
        row = plan.joiner
        engine.prefilling.append(
            serve._Joining(
                row,
                None,
                None,
                None,
                None,
                serve.prefill_chunks(
                    row.cached,
                    len(row.ids),
                    engine.prefill_step,
                    row.store_at if row.store_id else 0,
                ),
            )
        )
    scheduler.applied(plan)


def _chunk(scheduler: serve._Scheduler) -> None:
    engine = scheduler.engine
    target = engine.target
    joining = engine.prefilling[target]
    start, stop = joining.ranges.pop(0)
    first = None
    if not joining.ranges:
        engine.prefilling = [row for row in engine.prefilling if row is not joining]
        first = 5
    scheduler.chunked(target, stop - start, first, 0.0)
