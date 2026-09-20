# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only tests for the stable-row ``InputBatch`` layout.

``model_capabilities["stable_decode_slots"]`` (Qwen3.6: per-slot GDN state)
makes the front-packed persistent batch a fixed slot grid: a request keeps one
row -- its device state slot -- for its whole lifetime, a finished request's
row becomes a pad row (token 0, position -1, no blocks, neutral sampling), a
newcomer takes the lowest free row, and ``condense`` never moves anything. The
runner therefore never has to remap device state between slots
(``_decode_state_slot_remap`` is always ``None``), which is what these tests
pin down. No device execution.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.utils import torch_utils
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample import sampler as sampler_module
from vllm.v1.sample.logits_processor import build_logitsprocs
from vllm.v1.sample.ops import penalties as penalties_module
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu_input_batch import CachedRequestState

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.config import (
    get_tt_stable_decode_slots,
    store_tt_stable_decode_slots,
)
from vllm_tt_plugin.input_batch import (
    LOGPROBS_NONE_SENTINEL,
    InputBatch,
    select_live_decode_rows,
)
from vllm_tt_plugin.model_input import TTModelInput, slice_tt_sampling_params
from vllm_tt_plugin.model_runner import TTModelRunner

VOCAB = 64
BLOCK = 16
MAX_MODEL_LEN = 64
ROWS = 4


def _batch(stable_rows=True, max_num_reqs=ROWS):
    return InputBatch(
        max_num_reqs=max_num_reqs,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN * max_num_reqs,
        vocab_size=VOCAB,
        block_sizes=[BLOCK],
        kernel_block_sizes=[BLOCK],
        stable_rows=stable_rows,
    )


def _req(req_id, prompt, block, **sp):
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        mm_features=None,
        sampling_params=SamplingParams(**sp),
        generator=None,
        block_ids=([block],),
        num_computed_tokens=len(prompt),
        output_token_ids=[],
    )


# --------------------------------------------------------------------------
# Placement: hole -> pad row -> reuse, condense is a no-op
# --------------------------------------------------------------------------


def test_hole_becomes_pad_row_and_the_lowest_free_row_is_reused():
    b = _batch()
    for rid, block in (("a", 1), ("b", 2), ("c", 3)):
        b.add_request(_req(rid, [10, 11, 12], block, temperature=0.7, seed=5))
    assert b.req_id_to_index == {"a": 0, "b": 1, "c": 2}

    assert b.remove_request("b") == 1
    # Nobody moved; row 1 is a hole.
    assert b.req_id_to_index == {"a": 0, "c": 2}
    assert b.occupied_rows() == [0, 2]
    assert b.live_req_ids() == ["a", "c"]
    # ... and it reads as a pad row: no tokens, neutral sampling.
    assert b.num_tokens[1] == 0 and b.num_prompt_tokens[1] == 0
    assert float(b.sampling.temperature[1]) == 0.0
    assert int(b.sampling.seed[1]) == b.sampling.DEFAULTS["seed"]
    assert int(b.sampling.num_logprobs[1]) == LOGPROBS_NONE_SENTINEL
    assert b.all_greedy is False, "the live random request still counts"

    b.condense([1])
    assert b.req_id_to_index == {"a": 0, "c": 2}, "condense must not pull c down"

    # The decode view: gaps are token 0 at position -1.
    tokens, positions = b.decode_tokens_and_positions(list(range(ROWS)))
    assert tokens.tolist() == [[12], [0], [12], [0]]
    assert positions.tolist() == [2, -1, 2, -1]

    # Reuse: newcomers take the lowest free row, then the next one.
    b.add_request(_req("d", [1], 4))
    b.add_request(_req("e", [1], 5))
    assert b.req_id_to_index == {"a": 0, "c": 2, "d": 1, "e": 3}
    with pytest.raises(RuntimeError, match="no free row"):
        b.add_request(_req("f", [1], 6))


def test_stable_rows_reject_placing_onto_an_occupied_row():
    b = _batch()
    b.add_request(_req("a", [1], 1))
    with pytest.raises(ValueError, match="already occupied"):
        b.add_request(_req("b", [1], 2), req_index=0)


def test_pad_row_block_tables_are_zeroed():
    """vLLM's block table keeps the dead request's blocks on its row; the pad row
    must reach the device with none, or attention would touch freed blocks."""
    b = _batch()
    b.add_request(_req("a", [1] * BLOCK, 7))
    b.add_request(_req("b", [1] * BLOCK, 9))
    b.remove_request("a")
    stale = b.block_tables_for_rows([0], width=2)[0]
    assert stale[0, 0].item() == 7, "the raw table still holds a's block"
    (table,) = b.slot_block_tables(b.occupied_rows(), zero_gaps=True, total=ROWS, width=2)
    assert table.tolist() == [[0, 0], [9, 0], [0, 0], [0, 0]]


def test_front_packed_default_is_unchanged():
    """``stable_rows`` defaults off: the front-packed batch still appends at
    ``num_reqs`` and condenses holes, so other TT models see no change."""
    b = _batch(stable_rows=False)
    assert b.stable_rows is False
    b.add_request(_req("a", [1], 1))
    b.add_request(_req("b", [1], 2))
    b.add_request(_req("c", [1], 3))
    b.remove_request("a")
    b.condense([0])
    assert b.req_id_to_index == {"c": 0, "b": 1}
    assert b.sampling_rows == 2 and len(b.req_ids) == 2


# --------------------------------------------------------------------------
# Rows a step is built over
# --------------------------------------------------------------------------


def test_step_rows_prefill_covers_scheduled_rows_and_decode_reaches_the_top_live_row():
    b = _batch()
    for rid, block in (("a", 1), ("b", 2), ("c", 3)):
        b.add_request(_req(rid, [1], block))
    b.remove_request("b")
    # Prefill of a newcomer at the hole: only its row, the resident decodes stay put.
    b.add_request(_req("n", [1, 2], 4))
    assert b.step_rows(["n"], is_prompt=True) == [1]
    # Decode: rows ``[0, highest live row]`` with the gap as a pad row, once
    # every resident request is scheduled. The width stops at the highest live
    # row because that is what a bucketing model returns: with ``c`` at row 2
    # the device may hand back only 4 rows of a 32-row grid, and a wider
    # ``unpadded_batch_size`` would index past the end of that output.
    b.remove_request("n")
    assert b.step_rows(["c", "a"], is_prompt=False) == [0, 1, 2]
    b.remove_request("c")
    assert b.step_rows(["a"], is_prompt=False) == [0]
    # A lone request at the top row still needs the whole grid.
    b.add_request(_req("t", [1], 5), req_index=3)
    assert b.step_rows(["a", "t"], is_prompt=False) == [0, 1, 2, 3]
    # A resident request the scheduler left out of a decode would still be
    # advanced by the device: refuse rather than desync it.
    with pytest.raises(RuntimeError, match="unscheduled"):
        b.step_rows(["a"], is_prompt=False)
    with pytest.raises(RuntimeError, match="has no row"):
        b.step_rows(["ghost"], is_prompt=True)
    b.remove_request("a")
    b.remove_request("t")
    assert b.step_rows([], is_prompt=False) == []


def test_sampling_state_spans_the_whole_grid():
    """Per-row sampling state is indexed by persistent row, so it must cover the
    grid, not ``[:num_reqs]``: a lone request at row 3 would otherwise vanish."""
    seen: list[int] = []
    b = _batch()
    b.sampling.logitsprocs = SimpleNamespace(
        all=[SimpleNamespace(update_state=lambda upd: seen.append(upd.batch_size))]
    )
    for rid in ("a", "b", "c", "d"):
        b.add_request(_req(rid, [1], 1))
    for rid in ("a", "b", "c"):
        b.remove_request(rid)
    b.remove_request("d")
    b.add_request(_req("d", [1], 1, logprobs=5), req_index=3)
    assert b.max_num_logprobs == 5
    b.refresh_logitsprocs()
    assert seen == [ROWS]


# --------------------------------------------------------------------------
# Runner: rows are held for life, the slot map is the identity, no remap
# --------------------------------------------------------------------------


def _new_req(req_id, prompt, block):
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0),
        pooling_params=None,
        block_ids=([block],),
        num_computed_tokens=0,
        lora_request=None,
    )


def _step(*, new=(), scheduled=(), finished=(), preempted=None):
    out = SchedulerOutput.make_empty()
    out.scheduled_new_reqs = list(new)
    new_ids = {r.req_id for r in new}
    cached = [rid for rid in scheduled if rid not in new_ids]
    out.scheduled_cached_reqs = CachedRequestData(
        req_ids=cached,
        resumed_req_ids=set(),
        new_token_ids=[[] for _ in cached],
        all_token_ids={},
        new_block_ids=[None for _ in cached],
        num_computed_tokens=[1 for _ in cached],
        num_output_tokens=[0 for _ in cached],
    )
    out.num_scheduled_tokens = dict.fromkeys(scheduled, 1)
    out.total_num_scheduled_tokens = len(scheduled)
    out.finished_req_ids = set(finished)
    out.preempted_req_ids = preempted
    return out


def _runner(batch):
    """A fake runner wired to the real row/slot bookkeeping methods."""
    r = SimpleNamespace(
        input_batch=batch,
        requests={},
        encoder_cache={},
        tt_per_lane_max_num_seqs=batch.max_num_reqs,
        _req_state_slot={},
        _pending_state_slot_settle=None,
        _decode_layout_changed_since_last_decode=False,
        model=SimpleNamespace(),
    )
    for name in (
        "_update_states",
        "_release_model_request",
        "_release_dead_state_slots",
        "_pd_after_update_states",
        "_alloc_prefill_state_slots",
        "_decode_state_slot_remap",
        "note_decode_state_slots_settled",
    ):
        setattr(r, name, getattr(TTModelRunner, name).__get__(r))
    return r


def _grid(batch):
    return list(batch._req_ids)


def _prefill_step(r, scheduled):
    """What ``_prepare_model_inputs`` does on a prefill: rows of the scheduled
    requests, each request's slot is its row."""
    rows = r.input_batch.step_rows(scheduled, is_prompt=True)
    return rows, r._alloc_prefill_state_slots([r.input_batch.req_ids[i] for i in rows])


def _decode_step(r, scheduled):
    rows = r.input_batch.step_rows(scheduled, is_prompt=False)
    remap = r._decode_state_slot_remap([r.input_batch._req_ids[i] for i in rows])
    r.note_decode_state_slots_settled()
    return remap


def test_rows_are_held_for_life_and_no_remap_is_ever_issued():
    """Decode-instance lifecycle over a 4-row grid: prefill, finish in the middle,
    a prefill step that hides the resident decodes, decode again, churn. Rows
    never move, unscheduled requests are not evicted, and the state-slot remap is
    ``None`` on every decode."""
    b = _batch()
    r = _runner(b)
    condensed: list[list[int]] = []
    real_condense = b.condense
    b.condense = lambda idx: (condensed.append(list(idx)), real_condense(idx))

    # Step 1: four new requests prefill into rows 0..3 == slots 0..3.
    r._update_states(
        _step(new=[_new_req(f"r{i}", [1, 2], i + 1) for i in range(4)], scheduled=["r0", "r1", "r2", "r3"])
    )
    assert _prefill_step(r, ["r0", "r1", "r2", "r3"]) == ([0, 1, 2, 3], [0, 1, 2, 3])

    # Step 2: decode everyone -- identity.
    assert _decode_step(r, ["r0", "r1", "r2", "r3"]) is None

    # Step 3: r1 finishes; the others decode. Row 1 is a pad row, nobody moved.
    r._update_states(_step(scheduled=["r0", "r2", "r3"], finished=["r1"]))
    assert _grid(b) == ["r0", None, "r2", "r3"]
    assert "r1" not in r._req_state_slot
    assert _decode_step(r, ["r0", "r2", "r3"]) is None

    # Step 4: a prefill-only step (the scheduler hides the decodes). The resident
    # decodes keep their rows; the newcomer takes the hole and slot 1.
    r._update_states(_step(new=[_new_req("n", [5, 6], 9)], scheduled=["n"]))
    assert _grid(b) == ["r0", "n", "r2", "r3"], "unscheduled decodes were evicted"
    assert _prefill_step(r, ["n"]) == ([1], [1])

    # Step 5: everyone decodes again -- still the identity, nothing was re-added.
    r._update_states(_step(scheduled=["r0", "n", "r2", "r3"]))
    assert _decode_step(r, ["r0", "n", "r2", "r3"]) is None
    assert r._req_state_slot == {"r0": 0, "n": 1, "r2": 2, "r3": 3}

    # Step 6: churn -- three finishes and one import-style newcomer in one step.
    # The newcomer takes the lowest hole (row 0); the other holes stay pad rows.
    r._update_states(
        _step(
            new=[_new_req("m", [7], 8)],
            scheduled=["m", "r2"],
            finished=["r0", "n", "r3"],
        )
    )
    assert _grid(b) == ["m", None, "r2", None]
    assert _prefill_step(r, ["m"]) == ([0], [0])
    assert _decode_step(r, ["m", "r2"]) is None

    # Condense was reached (the runner still calls it on a removal) but is a no-op.
    assert condensed and b.req_id_to_index == {"m": 0, "r2": 2}
    assert r._pending_state_slot_settle is None


def test_preemption_frees_the_row_but_being_unscheduled_does_not():
    b = _batch()
    r = _runner(b)
    r._update_states(
        _step(new=[_new_req("a", [1], 1), _new_req("b", [1], 2)], scheduled=["a", "b"])
    )
    _prefill_step(r, ["a", "b"])
    # b unscheduled: stays resident on its row.
    r._update_states(_step(scheduled=["a"]))
    assert _grid(b) == ["a", "b", None, None]
    # b preempted: its KV is gone and it re-prefills from zero later, so the row
    # and the slot go back together.
    r._update_states(_step(scheduled=["a"], preempted={"b"}))
    assert _grid(b) == ["a", None, None, None]
    assert r._req_state_slot == {"a": 0}
    assert "b" in r.requests, "the request itself lives on for the resume"


# --------------------------------------------------------------------------
# Host sampling: the logits-processor state is keyed by persistent row, so the
# host sampler runs over the whole grid and the step's rows are picked out
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _disable_pinned_memory(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch_utils, "PIN_MEMORY", False)
    monkeypatch.setattr(sampler_module, "PIN_MEMORY", False)
    monkeypatch.setattr(penalties_module, "PIN_MEMORY", False)


def _sampling_batch():
    cfg = SimpleNamespace(
        speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=ROWS),
    )
    return InputBatch(
        max_num_reqs=ROWS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN * ROWS,
        vocab_size=VOCAB,
        block_sizes=[BLOCK],
        kernel_block_sizes=[BLOCK],
        logitsprocs=build_logitsprocs(
            cfg, torch.device("cpu"), is_pin_memory=False, is_pooling_model=False
        ),
        stable_rows=True,
    )


def _sampling_runner(batch):
    r = SimpleNamespace(
        input_batch=batch,
        host_sampler=Sampler(),
        _is_block_output_model=False,
        _output_tokens_per_step=1,
        vocab_size=VOCAB,
        tt_per_lane_max_num_seqs=ROWS,
    )
    for name in ("_get_output_tokens", "_host_sample_stable_rows", "apply_grammar_bitmask"):
        setattr(r, name, getattr(TTModelRunner, name).__get__(r))
    return r


def _model_input(batch, rows, *, is_decode, intermediate=None):
    """The slice of ``TTModelInput`` ``_get_output_tokens`` reads on the host path."""
    n = len(rows)
    return TTModelInput(
        input_tokens=torch.zeros((n, 1), dtype=torch.int32),
        input_positions=torch.zeros((n,), dtype=torch.int32),
        prompt_lens=None if is_decode else [1] * n,
        block_tables=torch.zeros((n, 1), dtype=torch.int32),
        block_tables_per_group=[torch.zeros((n, 1), dtype=torch.int32)],
        block_tables_per_layer=None,
        unpadded_batch_size=n,
        tt_sampling_params=slice_tt_sampling_params(batch.sampling, list(rows)),
        multi_modal_kwargs={},
        perform_device_sampling=False,
        grammar_bitmask=[None],
        logitsprocs_list=[batch.sampling.logitsprocs],
        bad_words_token_ids_list=[{}],
        allowed_token_ids_mask_list=[None],
        generators_list=[{}],
        max_num_logprobs=[batch.max_num_logprobs],
        row_req_ids=[batch._req_ids[i] for i in rows],
        intermediate_prefill_mask=intermediate,
    )


def _logits(*, rows, favourite, runner_up):
    """One ``[rows, 1, VOCAB]`` output whose argmax is ``favourite`` and whose
    next-best token is ``runner_up`` (the token a suppressed favourite yields)."""
    logits = torch.zeros(rows, 1, VOCAB)
    logits[:, :, favourite] = 10.0
    logits[:, :, runner_up] = 5.0
    return logits


def test_host_sampled_prefill_beside_a_resident_decode_lands_on_the_right_requests():
    """A prefill step covers only the scheduled rows, but ``min_tokens`` /
    ``logit_bias`` / ``min_p`` hold their state at the requests' persistent rows
    (``sampling_rows`` == the grid). Sampling the ``[n, vocab]`` prefill logits
    against that grid-sized state either fails on shape or lands the
    suppression on the wrong request; the stable-row path must sample the grid."""
    b = _sampling_batch()
    r = _sampling_runner(b)
    # Row 0: a resident greedy decode, not scheduled this step.
    b.add_request(_req("a", [1, 2], 1))
    b.refresh_logitsprocs()
    # Row 1: min_tokens keeps stop token 7 out until 5 tokens are out.
    b.add_request(_req("b", [3], 2, temperature=0.0, min_tokens=5, stop_token_ids=[7]))
    # Row 2: logit_bias hoists token 5; min_p makes the min-p processor size its
    # state to the grid (the shape failure in the bug).
    b.add_request(_req("c", [4], 3, temperature=1.0, min_p=0.1, logit_bias={5: 100.0}))
    b.refresh_logitsprocs()
    assert b.req_id_to_index == {"a": 0, "b": 1, "c": 2}

    rows = b.step_rows(["b", "c"], is_prompt=True)
    assert rows == [1, 2]
    tt_out = _logits(rows=2, favourite=7, runner_up=3)
    sampled, logprobs = r._get_output_tokens(
        tt_out=tt_out,
        tt_log_probs=None,
        sampling_params=slice_tt_sampling_params(b.sampling, rows),
        model_input=_model_input(b, rows, is_decode=False),
        batch_size_per_dp=[2],
        perform_device_sampling=False,
        is_decode=False,
    )
    assert sampled[0].tolist() == [[3], [5]], "b: 7 suppressed -> 3; c: biased -> 5"
    assert logprobs == [None]


def test_host_sampled_prefill_with_an_intermediate_chunk_keeps_its_generator_still():
    b = _sampling_batch()
    r = _sampling_runner(b)
    gen = torch.Generator()
    gen.manual_seed(11)
    req = _req("s", [1, 2, 3, 4], 1, temperature=1.0, seed=11)
    req.generator = gen
    req.num_computed_tokens = 2  # mid-prompt: this chunk emits no token
    b.add_request(req)
    b.refresh_logitsprocs()
    before = gen.get_state().clone()
    rows = [0]
    sampled, _ = r._get_output_tokens(
        tt_out=_logits(rows=1, favourite=2, runner_up=1),
        tt_log_probs=None,
        sampling_params=slice_tt_sampling_params(b.sampling, rows),
        model_input=_model_input(
            b, rows, is_decode=False, intermediate=torch.tensor([True])
        ),
        batch_size_per_dp=[1],
        perform_device_sampling=False,
        is_decode=False,
    )
    assert sampled[0].shape == (1, 1)
    assert torch.equal(gen.get_state(), before), "an intermediate chunk must not draw"


def test_host_sampled_stable_decode_reads_a_bucketed_prefix_against_the_grid():
    """A bucketing model returns logits only for rows ``[0, highest live row]``;
    the grid-keyed processors still apply, and the live rows come back in
    row order with pad rows in between."""
    b = _sampling_batch()
    r = _sampling_runner(b)
    b.add_request(_req("a", [1], 1, temperature=0.0, min_tokens=5, stop_token_ids=[7]))
    b.add_request(_req("gone", [1], 2))
    b.add_request(_req("c", [1], 3, temperature=0.0, logit_bias={5: 100.0}))
    b.refresh_logitsprocs()
    b.remove_request("gone")
    b.refresh_logitsprocs()
    rows = b.step_rows(["a", "c"], is_prompt=False)
    assert rows == [0, 1, 2]
    tt_out = _logits(rows=3, favourite=7, runner_up=3)  # the bucket, not the grid
    sampled, _ = r._get_output_tokens(
        tt_out=tt_out,
        tt_log_probs=None,
        sampling_params=slice_tt_sampling_params(b.sampling, rows),
        model_input=_model_input(b, rows, is_decode=True),
        batch_size_per_dp=[3],
        perform_device_sampling=False,
        is_decode=True,
    )
    tokens, _, req_ids = select_live_decode_rows(
        [b._req_ids[i] for i in rows], sampled[0], None
    )
    assert req_ids == ["a", "c"]
    assert tokens.tolist() == [[3], [5]]


def test_grid_host_logits_scatter_prefill_rows_and_pad_a_short_decode():
    b = _batch()
    prefill = torch.arange(2 * VOCAB, dtype=torch.float32).reshape(2, 1, VOCAB)
    full = b.grid_host_logits(prefill, [1, 3], is_decode=False, total=ROWS)
    assert full.shape == (ROWS, VOCAB)
    assert torch.equal(full[1], prefill[0, 0]) and torch.equal(full[3], prefill[1, 0])
    assert full[0].abs().sum() == 0 and full[2].abs().sum() == 0
    decode = torch.ones(3, 1, VOCAB)
    full = b.grid_host_logits(decode, [0, 1, 2], is_decode=True, total=ROWS)
    assert full.shape == (ROWS, VOCAB)
    assert full[:3].sum() == 3 * VOCAB and full[3].sum() == 0
    with pytest.raises(RuntimeError, match="does not reach"):
        b.grid_host_logits(decode, [0, 1, 2, 3], is_decode=True, total=ROWS)


# --------------------------------------------------------------------------
# Output: a full-width decode is read back at the live rows
# --------------------------------------------------------------------------


def test_select_live_decode_rows_drops_pad_rows_in_row_order():
    sampled = torch.tensor([[10], [99], [30], [99]], dtype=torch.int32)
    lp = LogprobsTensors(
        logprob_token_ids=torch.arange(8).reshape(4, 2),
        logprobs=torch.arange(8, dtype=torch.float32).reshape(4, 2),
        selected_token_ranks=torch.arange(4),
    )
    tokens, logprobs, req_ids = select_live_decode_rows(["a", None, "c", None], sampled, lp)
    assert tokens.tolist() == [[10], [30]]
    assert req_ids == ["a", "c"]
    assert logprobs is not None
    assert logprobs.logprob_token_ids.tolist() == [[0, 1], [4, 5]]
    assert logprobs.selected_token_ranks.tolist() == [0, 2]
    assert select_live_decode_rows(["a"], sampled[:1], None)[1] is None


def test_vectorized_state_write_refuses_a_stable_row_batch():
    """The ``req_ids=None`` write assumes rows ``[0, num_reqs)``; with gaps it
    would land tokens on the wrong requests, so it must fail instead."""
    b = _batch()
    b.add_request(_req("a", [1], 1))
    b.add_request(_req("b", [1], 2))
    b.remove_request("a")
    runner = SimpleNamespace(
        input_batch=b,
        requests={"b": SimpleNamespace(output_token_ids=[])},
        _output_tokens_per_step=1,
        model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
    )
    with pytest.raises(RuntimeError, match="explicit req_ids"):
        TTModelRunner._apply_sampled_tokens_to_state(
            runner, torch.tensor([[3]], dtype=torch.int32)
        )
    # The per-request path is what stable rows use.
    TTModelRunner._apply_sampled_tokens_to_state(
        runner, torch.tensor([[3]], dtype=torch.int32), req_ids=["b"]
    )
    assert b.num_tokens[1] == 2 and b.token_ids_cpu[1, 1] == 3
    assert runner.requests["b"].output_token_ids == [3]
    assert b.num_tokens[0] == 0, "the pad row was left alone"


# --------------------------------------------------------------------------
# Capability plumbing
# --------------------------------------------------------------------------


def test_stable_decode_slots_capability_roundtrips_through_the_config():
    cfg = SimpleNamespace(additional_config=None)
    assert get_tt_stable_decode_slots(cfg) is False, "absent means front-packed"
    store_tt_stable_decode_slots(cfg, True)
    assert get_tt_stable_decode_slots(cfg) is True
    assert cfg.additional_config == {"_tt_stable_decode_slots": True}
    with pytest.raises(ValueError, match="must be a bool"):
        store_tt_stable_decode_slots(cfg, 1)
