# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only tests of speculative decoding with a model-owned drafter
(spec_mtp.py, docs/SPECULATIVE.md): the policy helpers, the runner <-> scheduler
sidecars, the TT scheduler's admission-hold protocol and vLLM's token /
placeholder bookkeeping with a real ``TTScheduler`` (drafts scheduled,
rejections accounted), and the model runner's variable-length output packing."""

from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.worker.gpu_input_batch import CachedRequestState

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from tests.test_block_scheduler import BLOCK_SIZE, _request, _scheduler
from vllm_tt_plugin import spec_mtp
from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_input import TTModelInput
from vllm_tt_plugin.model_runner import TTModelRunner, _SyncForward
from vllm_tt_plugin.platform import TTPlatform
from vllm_tt_plugin.scheduler import TTScheduler

#
# ----------------------------------------------------------------------------------------------
# policy helpers


def test_request_forces_plain_rules():
    fp = spec_mtp.request_forces_plain
    assert not fp(SamplingParams(temperature=0.0))
    assert fp(SamplingParams(temperature=0.7))
    assert fp(SamplingParams(temperature=0.0, logprobs=1))
    assert fp(SamplingParams(temperature=0.0, prompt_logprobs=1))
    assert fp(SamplingParams(temperature=0.0, presence_penalty=0.5))
    assert fp(SamplingParams(temperature=0.0, repetition_penalty=1.1))
    assert not fp(
        SamplingParams(temperature=0.0, min_p=0.1)
    )  # vLLM neutralizes min_p at temperature 0
    assert fp(SamplingParams(temperature=0.0, logit_bias={3: 1.0}))
    assert fp(SamplingParams(temperature=0.0, bad_words=["x"]))
    assert fp(SamplingParams(temperature=0.0, min_tokens=2))
    assert fp(SamplingParams(temperature=0.0), use_structured_output=True)
    assert fp(None)


def test_hold_decision():
    hn = spec_mtp.hold_needed
    idle = spec_mtp.HoldInfo(pending_any=False)
    assert not hn(idle, 3, True) and not hn(None, 3, True)
    pending = spec_mtp.HoldInfo(pending_any=True, slots_before_crossing=2)
    assert not hn(pending, 0, False)
    assert not hn(pending, 2, False)  # fills free rows inside the band
    assert hn(pending, 3, False)  # the third admission leaves the band
    assert hn(
        pending, 1, True
    )  # a plain-forcing request needs every prefix committed first
    fits = spec_mtp.HoldInfo(pending_any=True, slots_before_crossing=None)
    assert not hn(fits, 40, False) and hn(fits, 1, True)


def test_drafts_for_rows_and_sidecars():
    rows = ["a", None, "b"]
    drafts = spec_mtp.drafts_for_rows({"a": [1, 2, -1], "c": [9]}, rows, pad_to=5)
    assert drafts == [[1, 2, -1], None, [], None, None]
    out = ModelRunnerOutput(
        req_ids=["a"],
        req_id_to_index={"a": 0},
        sampled_token_ids=[[1]],
        logprobs=None,
        prompt_logprobs_dict={"a": None},
        pooler_output=[],
    )
    assert spec_mtp.get_tt_spec_hold(out) is None
    spec_mtp.set_tt_spec_hold(
        out, SimpleNamespace(pending_any=True, slots_before_crossing=4)
    )
    hold = spec_mtp.get_tt_spec_hold(out)
    assert hold == spec_mtp.HoldInfo(pending_any=True, slots_before_crossing=4)
    so = SchedulerOutput.make_empty()
    assert not spec_mtp.get_tt_spec_flush(so)
    spec_mtp.set_tt_spec_flush(so)
    assert spec_mtp.get_tt_spec_flush(so)
    assert spec_mtp.is_spec_step_result(
        SimpleNamespace(committed=[], next_drafts=[], w=1)
    )
    assert not spec_mtp.is_spec_step_result(torch.zeros(1))


#
# ----------------------------------------------------------------------------------------------
# platform gate


def _cfg(method="mtp", async_scheduling=False, k=3):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method=method, num_speculative_tokens=k),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        model_config=SimpleNamespace(logits_processors=None),
        additional_config={},
    )


def test_platform_validates_speculative_config():
    cls = type("TTQwen36", (), {})
    TTPlatform._validate_speculative_config(
        _cfg(), {"supports_speculative_mtp": True}, cls
    )
    with pytest.raises(ValueError, match="supports_speculative_mtp"):
        TTPlatform._validate_speculative_config(_cfg(), {}, cls)
    with pytest.raises(ValueError, match="method 'mtp'"):
        TTPlatform._validate_speculative_config(
            _cfg(method="ngram"), {"supports_speculative_mtp": True}, cls
        )
    with pytest.raises(ValueError, match="synchronous"):
        TTPlatform._validate_speculative_config(
            _cfg(async_scheduling=True), {"supports_speculative_mtp": True}, cls
        )
    cfg = _cfg()
    cfg.speculative_config = None
    TTPlatform._validate_speculative_config(cfg, None, cls)  # off: nothing to check


#
# ----------------------------------------------------------------------------------------------
# real scheduler


class _FakeSpec:
    """The parts of a SpeculativeConfig(method='mtp') the scheduler reads."""

    method = "mtp"
    num_speculative_tokens_per_batch_size = None

    def __init__(self, k):
        self.num_speculative_tokens = k

    def use_eagle(self):
        return True

    def uses_draft_model(self):
        return False

    def use_dflash(self):
        return False

    def use_dspark(self):
        return False

    def uses_dynamic_speculative_decoding(self):
        return False


def _spec_scheduler(k=3, max_num_seqs=4):
    """A real TTScheduler over the local test model with a speculative config of k
    drafts (attached after the
    VllmConfig hook ran: the test model is not an MTP checkpoint)."""
    sched = _scheduler(1, max_model_len=256)
    cfg = sched.vllm_config
    cfg.speculative_config = _FakeSpec(k)
    cfg.scheduler_config.max_num_seqs = max_num_seqs
    return TTScheduler(
        vllm_config=cfg,
        kv_cache_config=sched.kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=sched.structured_output_manager,
    )


def _output(req_ids, tokens):
    return ModelRunnerOutput(
        req_ids=list(req_ids),
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=[list(t) for t in tokens],
        logprobs=None,
        prompt_logprobs_dict=dict.fromkeys(req_ids, None),
        pooler_output=[],
    )


def _new_request(rid, max_tokens=64):
    req = _request(max_tokens)
    req.request_id = rid
    req.sampling_params.temperature = 0.0  # greedy: eligible for the verify path
    return req


def test_real_scheduler_spec_bookkeeping_accepts_and_rejects():
    sched = _spec_scheduler(k=3)
    assert sched.num_spec_tokens == 3
    req = _new_request("r0")
    sched.add_request(req)
    so = sched.schedule()  # prefill: 32 prompt tokens
    assert so.num_scheduled_tokens["r0"] == 32
    sched.update_from_output(so, _output(["r0"], [[7]]))
    assert req.num_computed_tokens == 32 and req.num_tokens == 33
    # the runner proposed 3 drafts after that step
    sched.update_draft_token_ids(
        DraftTokenIds(req_ids=["r0"], draft_token_ids=[[11, 12, 13]])
    )
    so = sched.schedule()
    assert so.num_scheduled_tokens["r0"] == 4  # row-0 token + 3 drafts
    assert so.scheduled_spec_decode_tokens["r0"] == [11, 12, 13]
    assert req.num_computed_tokens == 36 and req.num_output_placeholders == 4
    # the verify accepted 2 drafts and produced the bonus token: 3 committed tokens
    sched.update_from_output(so, _output(["r0"], [[11, 12, 40]]))
    assert req.num_computed_tokens == 35  # 36 - 1 rejected
    assert req.num_output_placeholders == 0
    assert list(req.output_token_ids) == [7, 11, 12, 40]
    # no drafts (plain step next): exactly one token scheduled
    sched.update_draft_token_ids(DraftTokenIds(req_ids=["r0"], draft_token_ids=[[]]))
    so = sched.schedule()
    assert (
        so.num_scheduled_tokens["r0"] == 1
        and "r0" not in so.scheduled_spec_decode_tokens
    )
    sched.update_from_output(so, _output(["r0"], [[41]]))
    assert req.num_computed_tokens == 36 and req.num_output_placeholders == 0
    # a flush step: 3 drafts scheduled, all rejected -> one token, computed back by 3
    sched.update_draft_token_ids(
        DraftTokenIds(req_ids=["r0"], draft_token_ids=[[1, 2, 3]])
    )
    so = sched.schedule()
    assert so.num_scheduled_tokens["r0"] == 4
    sched.update_from_output(so, _output(["r0"], [[42]]))
    assert req.num_computed_tokens == 37 and req.num_output_placeholders == 0
    assert list(req.output_token_ids) == [7, 11, 12, 40, 41, 42]


def test_real_scheduler_holds_admission_for_a_flush():
    sched = _spec_scheduler(k=3)
    r0 = _new_request("r0")
    sched.add_request(r0)
    so = sched.schedule()
    sched.update_from_output(so, _output(["r0"], [[7]]))
    sched.update_draft_token_ids(
        DraftTokenIds(req_ids=["r0"], draft_token_ids=[[11, 12, 13]])
    )
    so = sched.schedule()
    # the runner reports a pending prefix that a further admission would break
    out = _output(["r0"], [[11, 12, 13, 40]])
    spec_mtp.set_tt_spec_hold(
        out, spec_mtp.HoldInfo(pending_any=True, slots_before_crossing=0)
    )
    sched.update_from_output(so, out)
    assert sched._tt_spec_hold is not None and sched._tt_spec_hold.pending_any
    sched.update_draft_token_ids(
        DraftTokenIds(req_ids=["r0"], draft_token_ids=[[21, 22, 23]])
    )
    r1 = _new_request("r1")
    sched.add_request(r1)
    so = sched.schedule()
    # held: a decode-only flush step, no admission, the drafts still scheduled (the
    # model rejects them)
    assert spec_mtp.get_tt_spec_flush(so)
    assert set(so.num_scheduled_tokens) == {"r0"} and so.num_scheduled_tokens["r0"] == 4
    assert not so.scheduled_new_reqs and sched._tt_spec_hold is None
    assert r1.status == RequestStatus.WAITING
    out = _output(["r0"], [[41]])
    spec_mtp.set_tt_spec_hold(out, spec_mtp.HoldInfo(pending_any=False))
    sched.update_from_output(so, out)
    sched.update_draft_token_ids(
        DraftTokenIds(req_ids=["r0"], draft_token_ids=[[31, 32, 33]])
    )
    so = sched.schedule()  # now the admission (prefill-only step for r1)
    assert not spec_mtp.get_tt_spec_flush(so)
    assert [r.req_id for r in so.scheduled_new_reqs] == ["r1"]


def test_real_scheduler_no_hold_when_admission_fits_the_band():
    sched = _spec_scheduler(k=3)
    r0 = _new_request("r0")
    sched.add_request(r0)
    so = sched.schedule()
    out = _output(["r0"], [[7]])
    spec_mtp.set_tt_spec_hold(
        out, spec_mtp.HoldInfo(pending_any=True, slots_before_crossing=3)
    )
    sched.update_from_output(so, out)
    r1 = _new_request("r1")
    sched.add_request(r1)
    so = sched.schedule()
    assert not spec_mtp.get_tt_spec_flush(so)
    assert [r.req_id for r in so.scheduled_new_reqs] == ["r1"]


def test_real_scheduler_holds_for_a_plain_forcing_admission():
    sched = _spec_scheduler(k=3)
    r0 = _new_request("r0")
    sched.add_request(r0)
    so = sched.schedule()
    out = _output(["r0"], [[7]])
    spec_mtp.set_tt_spec_hold(
        out, spec_mtp.HoldInfo(pending_any=True, slots_before_crossing=None)
    )
    sched.update_from_output(so, out)
    r1 = _new_request("r1")
    r1.sampling_params = SamplingParams(temperature=0.8, max_tokens=8)
    sched.add_request(r1)
    so = sched.schedule()
    assert spec_mtp.get_tt_spec_flush(so) and not so.scheduled_new_reqs


#
# ----------------------------------------------------------------------------------------------
# runner packing

VOCAB = 64
MAX_MODEL_LEN = 64
ROWS = 4


def _batch():
    return InputBatch(
        max_num_reqs=ROWS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN * ROWS,
        vocab_size=VOCAB,
        block_sizes=[16],
        kernel_block_sizes=[16],
        stable_rows=True,
    )


def _req(req_id, prompt, block, **sp):
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        mm_features=None,
        sampling_params=SamplingParams(temperature=0.0, **sp),
        generator=None,
        block_ids=([block],),
        num_computed_tokens=len(prompt),
        output_token_ids=[],
    )


def _runner(batch, requests, hold=None):
    return SimpleNamespace(
        input_batch=batch,
        requests=requests,
        model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
        model=SimpleNamespace(spec_hold_info=lambda: hold),
        _spec_enabled=True,
        _pending_draft_token_ids=None,
        _apply_spec_tokens_to_state=lambda *a,
        **k: TTModelRunner._apply_spec_tokens_to_state(runner, *a, **k),
        _build_spec_runner_output=lambda *a,
        **k: TTModelRunner._build_spec_runner_output(runner, *a, **k),
        _spec_hold_info=lambda: TTModelRunner._spec_hold_info(runner),
    )


def _model_input(rows, drafts):
    return TTModelInput(
        input_tokens=torch.zeros(ROWS, 1, dtype=torch.int32),
        input_positions=torch.zeros(ROWS, dtype=torch.int32),
        prompt_lens=None,
        block_tables=torch.zeros(ROWS, 4, dtype=torch.int32),
        block_tables_per_group=[torch.zeros(ROWS, 4, dtype=torch.int32)],
        block_tables_per_layer=None,
        unpadded_batch_size=len(rows),
        tt_sampling_params=None,
        multi_modal_kwargs={},
        perform_device_sampling=True,
        grammar_bitmask=[None],
        logitsprocs_list=[None],
        bad_words_token_ids_list=[{}],
        allowed_token_ids_mask_list=[None],
        generators_list=[{}],
        max_num_logprobs=[None],
        row_req_ids=rows,
        spec=spec_mtp.TTSpecStepInput(row_req_ids=rows, drafts=drafts, eligible=True),
    )


def test_finish_spec_step_applies_variable_tokens_and_hands_over_drafts():
    global runner
    batch = _batch()
    ra, rb = _req("a", [1, 2, 3], 1), _req("b", [4, 5], 2)
    batch.add_request(ra)
    batch.add_request(rb)
    for row, req in ((0, ra), (1, rb)):
        batch.token_ids_cpu[row, req.num_tokens] = (
            9  # the first sampled token (row-0 token of this step)
        )
        batch.num_tokens[row] = req.num_tokens + 1
        req.output_token_ids.append(9)
    runner = _runner(batch, {"a": ra, "b": rb})
    rows = ["a", "b", None, None]
    drafts = [[11, 12, 13], [21, 22, 23], None, None]
    res = SimpleNamespace(
        w=2,
        committed=[[11, 12, 40], [41]],
        next_drafts=[[51, 52, 53], [61, 62, 63]],
        hold=SimpleNamespace(pending_any=True, slots_before_crossing=1),
    )
    fwd = _SyncForward(
        tt_out=res,
        tt_log_probs=None,
        sampling_params=None,
        model_input=_model_input(rows, drafts),
        batch_size_per_dp=[2],
        perform_device_sampling=True,
        is_decode=True,
    )
    out = TTModelRunner._finish_spec_step(runner, fwd)
    assert out.req_ids == ["a", "b"] and out.sampled_token_ids == [[11, 12, 40], [41]]
    assert list(ra.output_token_ids) == [9, 11, 12, 40] and list(
        rb.output_token_ids
    ) == [9, 41]
    assert int(batch.num_tokens[0]) == 4 + 3 and int(batch.num_tokens[1]) == 3 + 1
    assert (
        batch.token_ids_cpu[0, 4:7].tolist() == [11, 12, 40]
        and int(batch.token_ids_cpu[1, 3]) == 41
    )
    hold = spec_mtp.get_tt_spec_hold(out)
    assert hold.pending_any and hold.slots_before_crossing == 1
    drafts_out = TTModelRunner.take_draft_token_ids(runner)
    assert drafts_out.req_ids == ["a", "b"] and drafts_out.draft_token_ids == [
        [51, 52, 53],
        [61, 62, 63],
    ]
    assert TTModelRunner.take_draft_token_ids(runner) is None  # consumed once
    # more tokens than the placeholders allow is a bug, not an output
    res_bad = SimpleNamespace(
        w=2, committed=[[1, 2, 3, 4, 5], [41]], next_drafts=[[], []], hold=None
    )
    fwd_bad = _SyncForward(
        tt_out=res_bad,
        tt_log_probs=None,
        sampling_params=None,
        model_input=_model_input(rows, drafts),
        batch_size_per_dp=[2],
        perform_device_sampling=True,
        is_decode=True,
    )
    with pytest.raises(RuntimeError, match="placeholder accounting"):
        TTModelRunner._finish_spec_step(runner, fwd_bad)


def test_spec_attach_plain_publishes_hold_and_clears_drafts():
    global runner
    batch = _batch()
    ra = _req("a", [1, 2, 3], 1)
    batch.add_request(ra)
    runner = _runner(
        batch,
        {"a": ra},
        hold=SimpleNamespace(pending_any=False, slots_before_crossing=None),
    )
    out = _output(["a"], [[5]])
    fwd = _SyncForward(
        tt_out=torch.zeros(1),
        tt_log_probs=None,
        sampling_params=None,
        model_input=_model_input(["a"], [[]]),
        batch_size_per_dp=[1],
        perform_device_sampling=True,
        is_decode=True,
    )
    out2 = TTModelRunner._spec_attach_plain(runner, out, fwd)
    assert out2 is out and spec_mtp.get_tt_spec_hold(out) == spec_mtp.HoldInfo(
        pending_any=False
    )
    drafts = TTModelRunner.take_draft_token_ids(runner)
    assert drafts.req_ids == ["a"] and drafts.draft_token_ids == [[]]


def test_build_spec_step_input_eligibility():
    batch = _batch()
    ra = _req("a", [1, 2, 3], 1)
    batch.add_request(ra)
    runner = SimpleNamespace(input_batch=batch)
    so = SchedulerOutput.make_empty()
    so.scheduled_spec_decode_tokens = {"a": [1, 2, -1]}
    spec_mtp.set_tt_spec_flush(so)
    inp = TTModelRunner._build_spec_step_input(runner, so, ["a"], 4, True, False)
    assert inp.eligible and inp.flush and inp.row_req_ids == ["a", None, None, None]
    assert inp.drafts == [[1, 2, -1], None, None, None]
    assert not TTModelRunner._build_spec_step_input(
        runner, so, ["a"], 4, False, False
    ).eligible
    assert not TTModelRunner._build_spec_step_input(
        runner, so, ["a"], 4, True, True
    ).eligible
    rb = _req("b", [4], 2, top_p=0.9)
    rb.sampling_params = SamplingParams(temperature=0.5)
    batch.add_request(rb)
    assert not TTModelRunner._build_spec_step_input(
        runner, so, ["a", "b"], 4, True, False
    ).eligible
