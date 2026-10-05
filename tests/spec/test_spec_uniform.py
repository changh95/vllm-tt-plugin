# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A verify step carries only requests the runner speculates for.

The ``argmax_ids`` accept walk compares token ids, so it certifies an argmax
continuation and nothing else, and a verify answers every row it carries with
the target argmax. ``TTModelRunner._publish_draft`` offers no drafts for a
sampled or penalized request, but that does not keep the request out of a
verify that another row's drafts, or another row's unresolved multi-token
commit, makes of the step. On such a step the request commits the argmax
rather than a token drawn with its own temperature, seed and penalties.

``SpecPlan.verify_requires_speculable_rows`` is how a model asks for uniform
steps instead: while the batch holds such a request, every decode step is the
model's ordinary decode, sampled by the tail a launch without speculation uses,
and nothing is proposed. These tests drive that through ``TTModelRunner``'s own
methods against a host target whose every position carries a real distribution
over the vocabulary, so that sampling, a seed and a penalty each change the
committed tokens, and they compare a speculating launch token for token with
one that does not speculate.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.sample.sampler import Sampler

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin import model_runner as model_runner_module
from vllm_tt_plugin.async_decode import TTAsyncDecodeController
from vllm_tt_plugin.config import get_tt_spec_plan, store_tt_spec_plan
from vllm_tt_plugin.input_batch import InputBatch, build_cached_request_state
from vllm_tt_plugin.model_runner import TTModelRunner, _SyncForward
from vllm_tt_plugin.spec_decode import (
    ACCEPT_MODE_ARGMAX_IDS,
    DRAFTER_STATE_INTERNAL,
    HIDDEN_HANDOFF_ROUNDTRIP,
    PLACEHOLDER_TOKEN_ID,
    SPEC_REQUIREMENT_DEVICE_PROPOSE,
    SPEC_REQUIREMENT_HIDDEN_FEED,
    DraftOutput,
    SpecPlan,
    VerifyOutput,
    check_spec_side_tensors,
)

VOCAB_SIZE = 512
BLOCK_SIZE = 16
MAX_MODEL_LEN = 256
MAX_NUM_REQS = 3
PROMPT_LEN = 4
DRAFT_LEN = 3
# Decode steps per comparison: enough for a sampled or penalized request to
# leave the argmax chain, and for a speculating greedy request to cross several
# accept walks.
STEPS = 12

# Every request the runner does not speculate for, as a client asks for one.
# Each reaches the batch on a speculating launch: the platform refuses only
# what no decode path applies (logprobs, structured output, token filters).
UNSPECULABLE = {
    "temperature": SamplingParams(temperature=1.0, seed=1234),
    "temperature with top-k and top-p": SamplingParams(
        temperature=0.8, top_k=40, top_p=0.9, seed=99
    ),
    "presence penalty": SamplingParams(temperature=0.0, presence_penalty=1.5),
    "frequency penalty": SamplingParams(temperature=0.0, frequency_penalty=1.5),
    "repetition penalty": SamplingParams(temperature=0.0, repetition_penalty=2.0),
}

# region The target


class ScoredTarget:
    """A target with a whole distribution behind every choice.

    Logits for the token after ``token`` at ``position`` are a fixed random row
    picked by ``(token, position)``, plus a bonus on ``token`` itself, so the
    argmax usually repeats the last token. The noise makes a sampled request's
    draws depend on its seed and its truncation. The bonus lets each penalty
    move the argmax, because the token it favours is the request's own last
    token, which every penalty counts once the request has emitted it. So
    every case in ``UNSPECULABLE`` commits tokens off the argmax chain, and a
    step that answered it by argmax shows in the output.

    The ordinary decode and the verify read the same rows: the verify's column
    ``j`` is the argmax after input column ``j``, and its drafter proposes the
    greedy chain, so a greedy request's drafts are always accepted and a
    speculated greedy run equals an unspeculated one.
    """

    model_capabilities = {
        "supports_spec_decode": True,
        "spec_requirements": [SPEC_REQUIREMENT_DEVICE_PROPOSE],
    }
    vocab_size = VOCAB_SIZE
    # Whether the plan asks for verify steps without unspeculable rows.
    declares = True

    _TABLE_ROWS = 997
    _REPEAT_BONUS = 3.0

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(20261002)
        self.table = torch.randn(self._TABLE_ROWS, VOCAB_SIZE, generator=generator)
        self.plain_calls = 0
        self.verify_calls: list[dict] = []
        self.propose_calls = 0

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs: int, requested_k: int):
        del vllm_config, max_num_seqs
        return SpecPlan(
            effective_k=requested_k,
            lanes_per_request=1,
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            accept_modes=(ACCEPT_MODE_ARGMAX_IDS,),
            drafter_state=DRAFTER_STATE_INTERNAL,
            supports_narrow_decode=True,
            verify_requires_speculable_rows=cls.declares,
        )

    def logits(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """``[..., V]`` logits for the token after each ``(token, position)``."""
        tokens = tokens.to(torch.int64)
        keys = (tokens * 31 + positions.to(torch.int64) * 7 + 11) % self._TABLE_ROWS
        logits = self.table[keys].clone()
        # A padding column carries token -1; its row is never read.
        logits.scatter_add_(
            -1,
            tokens.clamp(min=0).unsqueeze(-1),
            torch.full((*tokens.shape, 1), self._REPEAT_BONUS),
        )
        return logits

    def greedy(self, token: int, position: int, count: int) -> list[int]:
        """The ``count`` tokens a greedy, unpenalized request emits next."""
        emitted = []
        for _ in range(count):
            token = int(
                self.logits(torch.tensor(token), torch.tensor(position)).argmax()
            )
            position += 1
            emitted.append(token)
        return emitted

    def decode_forward(
        self,
        tokens,
        start_pos,
        num_valid_drafts=None,
        accepted_counts=None,
        spec_mode=None,
        **kwargs,
    ):
        del kwargs
        if spec_mode is None:
            if num_valid_drafts is not None or accepted_counts is not None:
                raise TypeError("a side tensor arrived without spec_mode")
            if tokens.shape[1] != 1 or start_pos.dim() != 1:
                raise ValueError(
                    f"an ordinary decode is [B, 1] against [B], got "
                    f"{tuple(tokens.shape)} and {tuple(start_pos.shape)}"
                )
            self.plain_calls += 1
            return self.logits(tokens[:, 0], start_pos).unsqueeze(1)
        if spec_mode != ACCEPT_MODE_ARGMAX_IDS:
            raise ValueError(f"ScoredTarget serves argmax_ids, asked for {spec_mode!r}")
        rows, width = tokens.shape
        check_spec_side_tensors(num_valid_drafts, accepted_counts, rows, width - 1)
        self.verify_calls.append(
            {
                "num_valid_drafts": num_valid_drafts.clone(),
                "accepted_counts": accepted_counts.clone(),
            }
        )
        argmax = self.logits(tokens, start_pos).argmax(dim=-1).to(torch.int32)
        return VerifyOutput(spec_mode=spec_mode, argmax_ids=argmax)

    def propose_draft_tokens(
        self,
        num_drafts,
        committed_tokens,
        committed_positions,
        accepted_counts,
        hidden=None,
    ) -> DraftOutput:
        """Draft each row's greedy chain from its last committed token."""
        del hidden
        self.propose_calls += 1
        last = (accepted_counts.to(torch.int64) - 1).unsqueeze(1)
        token = committed_tokens.to(torch.int64).gather(1, last).squeeze(1)
        position = committed_positions.to(torch.int64).gather(1, last).squeeze(1)
        drafts = []
        for _ in range(num_drafts):
            token = self.logits(token, position).argmax(dim=-1)
            position = position + 1
            drafts.append(token)
        return DraftOutput(draft_token_ids=torch.stack(drafts, dim=1).to(torch.int32))


class UndeclaredTarget(ScoredTarget):
    """The same target on a plan that leaves the declaration at its default."""

    declares = False


# endregion The target

# Requests whose tokens the host sampler filters (a grammar or a token
# filter). The persistent batch tracks them in none of its sampling sets, so the
# runner finds them through each request's own sampling params. Greedy, so a
# test can tell a held-back step from a verify by the calls alone.
HOST_FILTERED = {
    "structured outputs": SamplingParams(
        temperature=0.0,
        structured_outputs=StructuredOutputsParams(json={"type": "object"}),
    ),
    "bad_words": SamplingParams(temperature=0.0, bad_words=["x"]),
    "allowed_token_ids": SamplingParams(
        temperature=0.0, allowed_token_ids=list(range(0, VOCAB_SIZE, 7))
    ),
    "min_tokens": SamplingParams(temperature=0.0, min_tokens=STEPS),
}

# region Test helpers


def _runner(model: ScoredTarget, *, speculating: bool = True) -> SimpleNamespace:
    """A synchronous runner fake whose methods are the real ones.

    The speculative flags come from the model's own plan, the way
    ``TTModelRunner.__init__`` takes them from the admitted one.
    """
    plan = type(model).spec_plan(None, MAX_NUM_REQS, DRAFT_LEN) if speculating else None
    batch = InputBatch(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=VOCAB_SIZE,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )
    runner = SimpleNamespace(
        model=model,
        input_batch=batch,
        requests={},
        kv_caches=object(),
        trace_mode="decode_only",
        request_specific_rope=False,
        vocab_size=VOCAB_SIZE,
        _output_tokens_per_step=1,
        _is_block_output_model=False,
        _is_adaptive_block_output=False,
        _num_speculative_tokens=plan.effective_k if plan else 0,
        async_decode_scheduling=False,
        _spec_method="custom_class" if plan else None,
        _spec_supports_narrow_decode=bool(plan and plan.supports_narrow_decode),
        _spec_verify_requires_speculable_rows=bool(
            plan and plan.verify_requires_speculable_rows
        ),
        _spec_drafts_from_model=plan is not None,
        _req_accepted_counts={},
        _proposed_draft_token_ids={},
        _num_unspeculable_verify_rows=0,
        _ngram_proposer=None,
        tt_per_lane_max_num_seqs=MAX_NUM_REQS,
        tt_data_parallel_size=1,
        max_num_blocks_per_req=MAX_MODEL_LEN // BLOCK_SIZE,
        model_config=SimpleNamespace(
            is_multimodal_model=False, max_model_len=MAX_MODEL_LEN
        ),
        # Host sampling through vLLM's own sampler: the tail a launch without
        # speculation uses, which is the reference these tests compare with.
        check_perform_device_sampling=lambda **_: False,
        host_sampler=Sampler(),
        _block_tables_per_layer=lambda _: None,
        _alloc_prefill_state_slots=lambda row_req_ids: list(range(len(row_req_ids))),
        _decode_state_slot_remap=lambda row_req_ids: None,
        _decode_layout_changed_since_last_decode=False,
        note_decode_layout_consumed=lambda: None,
        note_decode_state_slots_settled=lambda: None,
    )
    # Every remaining method comes from the real class. The attributes set
    # above win, so the stubs stay stubs, and each descriptor keeps its own
    # binding: copying a static method onto an instance as a bound one shifts
    # every argument.
    for name, member in vars(TTModelRunner).items():
        if hasattr(runner, name):
            continue
        if isinstance(member, staticmethod):
            setattr(runner, name, member.__func__)
        elif isinstance(member, classmethod):
            setattr(runner, name, member.__func__.__get__(TTModelRunner))
        elif inspect.isfunction(member):
            setattr(runner, name, member.__get__(runner))
    return runner


def _add_request(
    runner: SimpleNamespace, req_id: str, first_token: int, params: SamplingParams
) -> None:
    """Add a fully prefilled request, built the way the runner builds one.

    ``build_cached_request_state`` is what seeds a seeded request's generator,
    so two runners given the same request draw from the same stream.
    """
    request = build_cached_request_state(
        NewRequestData(
            req_id=req_id,
            prompt_token_ids=list(range(first_token, first_token + PROMPT_LEN)),
            mm_features=[],
            sampling_params=params.clone(),
            pooling_params=None,
            block_ids=([0],),
            num_computed_tokens=PROMPT_LEN,
            lora_request=None,
        )
    )
    runner.requests[req_id] = request
    runner.input_batch.add_request(request)


def _finish(runner: SimpleNamespace, req_id: str) -> None:
    """Take a finished request out, the part of ``_update_states`` steps read."""
    row = runner.input_batch.remove_request(req_id)
    runner.input_batch.condense([row])
    del runner.requests[req_id]


def _tail(runner: SimpleNamespace, req_id: str) -> tuple[int, int]:
    """A request's last committed token and that token's position."""
    row = runner.input_batch.req_id_to_index[req_id]
    length = int(runner.input_batch.num_tokens[row])
    return int(runner.input_batch.token_ids_cpu[row, length - 1]), length - 1


def _build(
    runner: SimpleNamespace, *req_ids: str, drafts: dict[str, list[int]] | None = None
):
    """Build one decode step's model input for these decoding requests."""
    # Stands in for ``_update_states``: the scheduler's computed-token count
    # covers whatever the previous step committed, so the rows decode.
    for req_id in req_ids:
        row = runner.input_batch.req_id_to_index[req_id]
        runner.input_batch.num_computed_tokens_cpu[row] = runner.input_batch.num_tokens[
            row
        ]
    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.scheduled_spec_decode_tokens = dict(drafts or {})
    scheduler_output.num_scheduled_tokens = {
        req_id: 1 + len((drafts or {}).get(req_id, ())) for req_id in req_ids
    }
    scheduler_output.total_num_scheduled_tokens = sum(
        scheduler_output.num_scheduled_tokens.values()
    )
    scheduler_output.scheduled_cached_reqs = CachedRequestData(
        req_ids=list(req_ids),
        resumed_req_ids=set(),
        new_token_ids=[[] for _ in req_ids],
        all_token_ids={},
        new_block_ids=[None for _ in req_ids],
        num_computed_tokens=[
            int(runner.input_batch.num_tokens[runner.input_batch.req_id_to_index[r]])
            for r in req_ids
        ],
        num_output_tokens=[len(runner.requests[r].output_token_ids) for r in req_ids],
    )
    return TTModelRunner._prepare_model_inputs(runner, scheduler_output, None)


def _step(
    runner: SimpleNamespace, *req_ids: str, drafts: dict[str, list[int]] | None = None
) -> dict[str, list[int]]:
    """One decode step, build to commit to proposal; each request's tokens."""
    model_input = _build(runner, *req_ids, drafts=drafts)
    submission = TTAsyncDecodeController(runner).submit_decode(
        model_input, read_from_device=True
    )
    fwd = _SyncForward(
        tt_out=submission.tt_out,
        tt_log_probs=None,
        sampling_params=model_input.tt_sampling_params,
        model_input=model_input,
        batch_size_per_dp=[len(req_ids)],
        perform_device_sampling=False,
        is_decode=True,
        spec_hidden=submission.spec_hidden,
    )
    output = runner._finish_front_packed_sync(None, fwd=fwd)
    return dict(zip(output.req_ids, output.sampled_token_ids))


def _handoff(runner: SimpleNamespace) -> dict[str, list[int]]:
    """What the engine collects after a step and the scheduler sends back."""
    handed = runner.take_draft_token_ids()
    if handed is None:
        return {}
    return dict(zip(handed.req_ids, handed.draft_token_ids))


def _mixed_traffic(model: ScoredTarget, params: SamplingParams, *, speculating=True):
    """A greedy request mid-speculation, joined by one it cannot speculate for.

    The greedy request arrives holding what a speculating request holds the
    moment a peer joins: drafts the scheduler is about to verify, proposed
    before the peer existed, and the count of a verify that accepted them all.
    Either one alone makes the step a verify on a launch without the
    declaration. Returns the runner, every token each request emitted, and
    whether anything was handed to the scheduler.
    """
    runner = _runner(model, speculating=speculating)
    _add_request(runner, "greedy", 1, SamplingParams(temperature=0.0))
    _add_request(runner, "peer", 200, params)
    emitted: dict[str, list[int]] = {"greedy": [], "peer": []}
    drafts: dict[str, list[int]] = {}
    if speculating:
        token, position = _tail(runner, "greedy")
        drafts = {"greedy": model.greedy(token, position, DRAFT_LEN)}
        runner._req_accepted_counts["greedy"] = DRAFT_LEN + 1
    handed_any = False
    for _ in range(STEPS):
        committed = _step(runner, "greedy", "peer", drafts=drafts)
        for req_id, tokens in committed.items():
            emitted[req_id].extend(tokens)
        drafts = _handoff(runner) if speculating else {}
        handed_any = handed_any or bool(drafts)
    return runner, emitted, handed_any


# endregion Test helpers

# region The step a sampled request shares


@pytest.mark.parametrize("case", sorted(UNSPECULABLE))
def test_a_step_holding_an_unspeculable_request_never_verifies(case):
    """Neither scheduled drafts nor an unresolved count make it a verify.

    Both are present on the first step, and on a launch without the
    declaration each alone would send the whole step, the peer's row
    included, to the verify. Here every step is the model's ordinary decode,
    each row commits exactly one token, and nothing is proposed while the peer
    is live, although the drafter is still asked after every step.
    """
    model = ScoredTarget()
    runner, emitted, handed_any = _mixed_traffic(model, UNSPECULABLE[case])

    assert model.verify_calls == [], "a step holding an unspeculable request verified"
    assert model.plain_calls == STEPS
    assert len(emitted["greedy"]) == len(emitted["peer"]) == STEPS
    assert not handed_any, "a proposal was published while the peer was live"
    assert model.propose_calls == STEPS, "the drafter stopped being asked"
    # The count of the verify before the peer joined is gone: that step's one
    # token per row was a commit of 1.
    assert runner._req_accepted_counts == {}
    assert runner._num_unspeculable_verify_rows == 0


@pytest.mark.parametrize("case", sorted(UNSPECULABLE))
def test_the_request_is_sampled_as_a_launch_without_speculation_samples_it(case):
    """Token for token what an unspeculated launch emits, seed and all.

    The reference arm has no speculation configured at all, so its tokens come
    from the ordinary host sampler with the request's own temperature, top-k,
    top-p, seed and penalties. Equality needs the speculating arm to sample
    the same rows with the same parameters and to advance each seeded
    generator exactly once per step. A verify does neither: it answers the
    row with the unpenalized argmax, and it leaves a seeded request's
    generator a draw behind, which shifts every draw after it.
    """
    _, speculated, _ = _mixed_traffic(ScoredTarget(), UNSPECULABLE[case])
    _, reference, _ = _mixed_traffic(
        ScoredTarget(), UNSPECULABLE[case], speculating=False
    )

    assert speculated == reference


@pytest.mark.parametrize("case", sorted(UNSPECULABLE))
def test_without_the_declaration_the_same_traffic_commits_the_argmax(case):
    """The default leaves today's behaviour, hazard included, unchanged.

    A model whose plan does not set the field keeps the per-request gate only:
    the greedy request's drafts make the step a verify, the peer rides along,
    and the peer commits the target argmax instead of its own draw. The runner
    counts those rows. That divergence from the unspeculated reference is what
    the two tests above rule out, and it shows that they can tell the two
    behaviours apart for every case.
    """
    model = UndeclaredTarget()
    runner, speculated, _ = _mixed_traffic(model, UNSPECULABLE[case])
    _, reference, _ = _mixed_traffic(
        ScoredTarget(), UNSPECULABLE[case], speculating=False
    )

    assert model.verify_calls, "the greedy request's drafts did not cause a verify"
    assert runner._num_unspeculable_verify_rows > 0
    assert speculated["peer"] != reference["peer"]
    # The greedy request is lossless either way.
    assert speculated["greedy"][:STEPS] == reference["greedy"]


@pytest.mark.parametrize("case", sorted(HOST_FILTERED))
def test_a_step_holding_a_host_filtered_request_never_verifies(case):
    """A grammar or a token filter holds speculation back like sampling does.

    A verify answers each row with an unfiltered argmax, so the request must
    never ride along on one, and no draft is published for the greedy peer
    while it is live.
    """
    model = ScoredTarget()
    runner, emitted, handed_any = _mixed_traffic(model, HOST_FILTERED[case])

    assert model.verify_calls == [], "a step holding a host-filtered request verified"
    assert model.plain_calls == STEPS
    assert len(emitted["greedy"]) == len(emitted["peer"]) == STEPS
    assert not handed_any, "a proposal was published while the peer was live"
    assert runner._req_accepted_counts == {}
    assert runner._num_unspeculable_verify_rows == 0
    assert not runner._request_is_speculable("peer")
    assert runner._request_is_speculable("greedy")


@pytest.mark.parametrize(
    "case", ["bad_words", "allowed_token_ids", "min_tokens"]
)
def test_a_host_filtered_request_decodes_as_without_speculation(case):
    """The filtered request emits what an unspeculated launch emits.

    (Structured outputs need a grammar bitmask, which this harness does not
    build; the device E2E covers them.)
    """
    _, speculated, _ = _mixed_traffic(ScoredTarget(), HOST_FILTERED[case])
    _, reference, _ = _mixed_traffic(
        ScoredTarget(), HOST_FILTERED[case], speculating=False
    )
    assert speculated == reference


def test_speculation_resumes_once_the_structured_request_leaves():
    """Held back while a structured-output request is live, verifying after."""
    model = ScoredTarget()
    runner = _runner(model)
    _add_request(runner, "greedy", 1, SamplingParams(temperature=0.0))
    start = _tail(runner, "greedy")
    emitted: list[int] = []
    kinds: list[str] = []

    def step(*req_ids, drafts):
        verifies = len(model.verify_calls)
        emitted.extend(_step(runner, *req_ids, drafts=drafts)["greedy"])
        kinds.append("verify" if len(model.verify_calls) > verifies else "plain")
        return _handoff(runner)

    drafts = step("greedy", drafts={})
    drafts = step("greedy", drafts=drafts)
    _add_request(runner, "peer", 200, HOST_FILTERED["structured outputs"])
    drafts = step("greedy", "peer", drafts=drafts)
    assert drafts == {}
    drafts = step("greedy", "peer", drafts=drafts)
    assert drafts == {}
    _finish(runner, "peer")
    drafts = step("greedy", drafts=drafts)
    assert set(drafts) == {"greedy"}, "speculation did not resume"
    step("greedy", drafts=drafts)

    assert kinds == ["plain", "verify", "plain", "plain", "plain", "verify"]
    assert emitted == model.greedy(*start, len(emitted))


def test_an_all_greedy_batch_still_speculates():
    """The declaration costs a batch of greedy requests nothing.

    With no unspeculable request in the batch, drafts are published and
    verified as on any speculating launch, every verify accepts the drafter's
    greedy chain, and the committed tokens are the unspeculated ones.
    """
    model = ScoredTarget()
    runner = _runner(model)
    for req_id, first in (("a", 1), ("b", 200)):
        _add_request(runner, req_id, first, SamplingParams(temperature=0.0))
    starts = {req_id: _tail(runner, req_id) for req_id in ("a", "b")}

    emitted: dict[str, list[int]] = {"a": [], "b": []}
    drafts: dict[str, list[int]] = {}
    for _ in range(4):
        for req_id, tokens in _step(runner, "a", "b", drafts=drafts).items():
            emitted[req_id].extend(tokens)
        drafts = _handoff(runner)
        assert set(drafts) == {"a", "b"}, "a greedy request was not drafted for"

    # The first step had nothing in flight; each one after it verified, and
    # accepted every draft plus the bonus.
    assert model.plain_calls == 1
    assert len(model.verify_calls) == 3
    for req_id, (token, position) in starts.items():
        assert emitted[req_id] == model.greedy(token, position, 1 + 3 * (DRAFT_LEN + 1))
    assert runner._num_unspeculable_verify_rows == 0


def test_speculation_resumes_once_the_unspeculable_request_leaves():
    """Held back while the peer is live, and back to verifying without it.

    The step after the peer leaves is an ordinary decode, not a verify: its
    count was reset to 1 while it was held back, so nothing is left to
    resolve, and nothing was proposed for it to verify. It proposes, and the
    step after it verifies again. Throughout, the greedy request emits its own
    unspeculated continuation.
    """
    model = ScoredTarget()
    runner = _runner(model)
    _add_request(runner, "greedy", 1, SamplingParams(temperature=0.0))
    start = _tail(runner, "greedy")
    emitted: list[int] = []
    kinds: list[str] = []

    def step(*req_ids, drafts):
        verifies = len(model.verify_calls)
        emitted.extend(_step(runner, *req_ids, drafts=drafts)["greedy"])
        kinds.append("verify" if len(model.verify_calls) > verifies else "plain")
        return _handoff(runner)

    drafts = step("greedy", drafts={})
    drafts = step("greedy", drafts=drafts)
    assert runner._req_accepted_counts["greedy"] == DRAFT_LEN + 1

    _add_request(runner, "peer", 200, UNSPECULABLE["temperature"])
    drafts = step("greedy", "peer", drafts=drafts)
    assert "greedy" not in runner._req_accepted_counts
    assert drafts == {}
    drafts = step("greedy", "peer", drafts=drafts)
    assert drafts == {}

    _finish(runner, "peer")
    drafts = step("greedy", drafts=drafts)
    assert set(drafts) == {"greedy"}, "speculation did not resume"
    step("greedy", drafts=drafts)

    assert kinds == ["plain", "verify", "plain", "plain", "plain", "verify"]
    # The resumed verify reads the post-reset count, not the stale one.
    assert int(model.verify_calls[-1]["accepted_counts"][0]) == 1
    assert emitted == model.greedy(*start, len(emitted))


def _reserved_step_input(model: ScoredTarget):
    """An asynchronous build: a retained proposal and a lookahead reservation.

    Under asynchronous scheduling the scheduler sends ``-1`` placeholders, and
    ``_drafts_to_verify`` substitutes the proposal the runner retained for the
    request, consuming it.
    """
    runner = _runner(model)
    runner.async_decode_scheduling = True
    _add_request(runner, "greedy", 1, SamplingParams(temperature=0.0))
    _add_request(runner, "peer", 200, UNSPECULABLE["temperature"])
    token, position = _tail(runner, "greedy")
    runner._proposed_draft_token_ids["greedy"] = model.greedy(
        token, position, DRAFT_LEN
    )
    reservation = [PLACEHOLDER_TOKEN_ID] * DRAFT_LEN
    model_input = _build(
        runner, "greedy", "peer", drafts={"greedy": reservation, "peer": reservation}
    )
    return runner, model_input


def test_a_held_back_asynchronous_step_spends_the_retained_proposal():
    """The proposal is consumed by the step that drops it, not kept for later.

    A proposal is handed over once. Left in place, it would wait for the next
    step that verifies, and be checked there against a token the request has
    already moved past.
    """
    runner, model_input = _reserved_step_input(ScoredTarget())

    assert model_input.spec_mode is None
    assert tuple(model_input.input_tokens.shape) == (MAX_NUM_REQS, 1)
    assert runner._proposed_draft_token_ids == {}


def test_without_the_declaration_the_retained_proposal_is_verified():
    runner, model_input = _reserved_step_input(UndeclaredTarget())

    assert model_input.spec_mode == ACCEPT_MODE_ARGMAX_IDS
    assert model_input.num_valid_drafts.tolist() == [DRAFT_LEN, 0, 0]
    assert runner._proposed_draft_token_ids == {}


# endregion The step a sampled request shares

# region The publish gate


class _FakeBatch:
    def __init__(self, random=(), presence=(), frequency=(), repetition=()):
        self.random_reqs = set(random)
        self.presence_penalties_reqs = set(presence)
        self.frequency_penalties_reqs = set(frequency)
        self.repetition_penalties_reqs = set(repetition)


def _publish(
    declares: bool, requests=None, **batch_kwargs
) -> dict[str, list[int]]:
    """Publish a greedy request's offer next to whatever else the batch holds."""
    runner = object.__new__(TTModelRunner)
    runner.input_batch = _FakeBatch(**batch_kwargs)
    if requests is not None:
        runner.requests = requests
        runner.input_batch.req_id_to_index = {
            req_id: row for row, req_id in enumerate(requests)
        }
    runner._spec_verify_requires_speculable_rows = declares
    runner._proposed_draft_token_ids = {"greedy": [9, 9]}  # a stale earlier offer
    TTModelRunner._publish_draft(runner, "greedy", [1, 2, 3])
    return runner._proposed_draft_token_ids


@pytest.mark.parametrize(
    "sampling_set", ["random", "presence", "frequency", "repetition"]
)
def test_nothing_is_published_while_an_unspeculable_request_is_live(sampling_set):
    """The drafts would be scheduled, reserved, and then dropped unverified.

    The stale offer is erased too: this map is what the scheduler verifies
    next, so a leftover entry would be verified against a token the request
    has already moved past.
    """
    assert _publish(True, **{sampling_set: ["peer"]}) == {}


@pytest.mark.parametrize("case", sorted(HOST_FILTERED))
def test_nothing_is_published_while_a_host_filtered_request_is_live(case):
    requests = {
        "greedy": SimpleNamespace(sampling_params=SamplingParams(temperature=0.0)),
        "peer": SimpleNamespace(sampling_params=HOST_FILTERED[case]),
    }
    assert _publish(True, requests=requests) == {}
    # Without the declaration the gate stays per request.
    assert _publish(False, requests=requests) == {"greedy": [1, 2, 3]}


def test_the_hold_back_is_total_for_a_paramless_request():
    requests = {
        "greedy": SimpleNamespace(sampling_params=SamplingParams(temperature=0.0)),
        "pooled": SimpleNamespace(sampling_params=None),
    }
    assert _publish(True, requests=requests) == {"greedy": [1, 2, 3]}


def test_a_batch_of_speculable_requests_still_publishes():
    assert _publish(True) == {"greedy": [1, 2, 3]}


def test_without_the_declaration_the_gate_stays_per_request():
    """A greedy request beside a sampled one keeps its own proposal."""
    assert _publish(False, random=["peer"]) == {"greedy": [1, 2, 3]}


# endregion The publish gate

# region The declaration


def _plan(**overrides) -> SpecPlan:
    fields = {
        "effective_k": DRAFT_LEN,
        "lanes_per_request": 1,
        "extra_bytes_per_seq": 0,
        "extra_bytes_per_token": 0,
        "accept_modes": (ACCEPT_MODE_ARGMAX_IDS,),
        "drafter_state": DRAFTER_STATE_INTERNAL,
        "supports_narrow_decode": True,
    }
    fields.update(overrides)
    return SpecPlan(**fields)


def test_the_declaration_defaults_off():
    """Absent means today's behaviour, so no existing model changes."""
    assert _plan().verify_requires_speculable_rows is False


def test_the_declaration_needs_the_ordinary_decode_call():
    """A held-back step is sent as the narrow call, which such a model lacks."""
    with pytest.raises(ValueError, match="needs supports_narrow_decode=True"):
        _plan(verify_requires_speculable_rows=True, supports_narrow_decode=False)


def _runner_config() -> SimpleNamespace:
    """Just what ``TTModelRunner.__init__`` reads from a launch's config."""
    return SimpleNamespace(
        additional_config={},
        model_config=SimpleNamespace(
            is_encoder_decoder=False,
            uses_mrope=False,
            hf_config=SimpleNamespace(model_type="scored_target"),
            get_vocab_size=lambda: VOCAB_SIZE,
            logits_processors=None,
            max_model_len=MAX_MODEL_LEN,
        ),
        cache_config=None,
        lora_config=None,
        load_config=None,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        scheduler_config=SimpleNamespace(async_scheduling=False, max_num_seqs=4),
        speculative_config=SimpleNamespace(method="custom_class"),
        observability_config=None,
        device_config=None,
    )


@pytest.mark.parametrize("declares", [True, False])
def test_the_runner_takes_the_declaration_from_the_admitted_plan(declares):
    """Through the stored plan, the one path from admission to the runner.

    ``store_tt_spec_plan`` keeps the plan as a dictionary on
    ``additional_config`` and ``get_tt_spec_plan`` rebuilds it, so a field the
    round trip dropped would silently fall back to the default.
    """
    vllm_config = _runner_config()
    store_tt_spec_plan(vllm_config, _plan(verify_requires_speculable_rows=declares))
    assert get_tt_spec_plan(vllm_config).verify_requires_speculable_rows is declares

    runner = TTModelRunner(
        vllm_config,
        mesh_device=None,
        trace_mode="decode_only",
        enable_model_warmup=False,
        num_devices=1,
    )

    assert runner._spec_verify_requires_speculable_rows is declares


class _RoundtripHiddenTarget(ScoredTarget):
    model_capabilities = {
        "supports_spec_decode": True,
        "spec_requirements": [
            SPEC_REQUIREMENT_DEVICE_PROPOSE,
            SPEC_REQUIREMENT_HIDDEN_FEED,
        ],
        "spec_hidden_handoff": [HIDDEN_HANDOFF_ROUNDTRIP],
    }


def test_a_drafter_fed_through_the_runner_cannot_declare_it(monkeypatch):
    """A held-back step returns no hidden handle for such a drafter to read.

    ``load_model`` already keeps every step of that pairing a verify. With the
    declaration it cannot, so it refuses the launch by name rather than hand
    the drafter ``None`` after each held-back step.
    """
    monkeypatch.setattr(
        model_runner_module,
        "TTModelLoader",
        lambda load_config: SimpleNamespace(
            load_model=lambda **_: _RoundtripHiddenTarget()
        ),
    )
    vllm_config = _runner_config()
    store_tt_spec_plan(vllm_config, _plan(verify_requires_speculable_rows=True))
    runner = TTModelRunner(
        vllm_config,
        mesh_device=None,
        trace_mode="decode_only",
        enable_model_warmup=False,
        num_devices=1,
    )
    assert runner._spec_drafts_from_model and runner._spec_supports_narrow_decode

    with pytest.raises(ValueError, match="verify_requires_speculable_rows"):
        runner.load_model()


# endregion The declaration
