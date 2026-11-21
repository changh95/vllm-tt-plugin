# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only tests: logits-only controls on a narrow-decode ``argmax_ids`` launch.

Admission lets a request carrying logprobs, structured output,
``allowed_token_ids``, ``bad_words`` or ``min_tokens`` onto a speculating launch
whose model serves only ``argmax_ids`` when its ``SpecPlan`` keeps
``supports_narrow_decode``
(``TTPlatform._reject_unsupported_speculative_request``). The greedy walk
cannot apply those controls, so such a request must never be answered by a
verify: ``TTModelRunner._request_is_argmax_certifiable`` is false for it, it is
offered no drafts, and every step it shares is the ordinary decode
(``_step_verifies``), which applies the controls as a launch without
speculation does. These tests pin that on the runner side, row by row:

- a step that carries such a row is an ordinary decode even when another row
  offers drafts, and the request's own proposal is withheld;
- a multi-token commit still unresolved when such a request arrives reaches
  that ordinary decode as ``accepted_counts`` and is then forgotten;
- a pending ``min_tokens`` keeps the request off the drafts only while it is
  pending, and a speculating launch builds the logits processor that applies it;
- the grammar bitmask such a step samples with is each request's committed row.
"""

import pytest
import torch
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_tt_plugin.structured_output import spec_grammar_bitmask_for_tt_batch

from .test_spec_inputs import (
    MAX_NUM_REQS,
    OUTPUT_LEN,
    PROMPT_LEN,
    _add_decoding_request,
    _batch,
    _decode,
    _drafts,
    _fake_runner,
)

STOP_TOKEN = 2


def _feature_params(control: str) -> SamplingParams:
    """A greedy request carrying exactly one control only a logits walk applies."""
    if control == "logprobs":
        return SamplingParams(temperature=0.0, logprobs=1)
    if control == "logprobs-0":
        return SamplingParams(temperature=0.0, logprobs=0)
    if control == "structured_outputs":
        return SamplingParams(
            temperature=0.0, structured_outputs=StructuredOutputsParams(regex="[a-z]+")
        )
    if control == "allowed_token_ids":
        return SamplingParams(temperature=0.0, allowed_token_ids=[1, 2, 3])
    if control == "bad_words":
        params = SamplingParams(temperature=0.0, bad_words=["no"])
        # What SamplingParams.update_from_tokenizer would leave on the request.
        params._bad_words_token_ids = [[7]]
        return params
    if control == "min_tokens":
        # A pending min_tokens: the request has OUTPUT_LEN tokens and a stop
        # token to withhold.
        return SamplingParams(
            temperature=0.0, min_tokens=OUTPUT_LEN + 4, stop_token_ids=[STOP_TOKEN]
        )
    raise ValueError(control)


CONTROLS = [
    "logprobs",
    "logprobs-0",
    "structured_outputs",
    "allowed_token_ids",
    "bad_words",
    "min_tokens",
]


def _add_feature_request(batch, req_id: str, params: SamplingParams):
    request = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(range(PROMPT_LEN)),
        mm_features=None,
        sampling_params=params,
        generator=None,
        block_ids=([0],),
        num_computed_tokens=PROMPT_LEN + OUTPUT_LEN - 1,
        output_token_ids=list(range(PROMPT_LEN, PROMPT_LEN + OUTPUT_LEN)),
    )
    batch.add_request(request)
    return request


def _runner_with_feature_row(control: str, accepted_counts=None):
    batch = _batch()
    requests = {
        "greedy": _add_decoding_request(batch, "greedy"),
        "feature": _add_feature_request(batch, "feature", _feature_params(control)),
    }
    return _fake_runner(
        batch,
        requests,
        supports_narrow_decode=True,
        accepted_counts=accepted_counts,
    )


@pytest.mark.parametrize("control", CONTROLS)
def test_a_feature_row_is_never_answered_by_a_verify(control):
    """Another row's drafts do not pull the feature request into a verify.

    An ``argmax_ids`` verify answers every row with the target argmax: no
    logprobs, no grammar, no token filter. With the feature row present the step
    is the ordinary decode instead: plain shapes, no ``spec_mode``, no draft
    columns, and the greedy row's drafts are dropped.
    """
    runner = _runner_with_feature_row(control)
    assert runner._request_is_speculable("feature") is False
    assert runner._request_is_speculable("greedy") is True

    model_input = _decode(runner, "greedy", "feature", drafts=_drafts(greedy=[21, 22]))

    assert model_input.spec_mode is None
    assert model_input.input_tokens.shape == (MAX_NUM_REQS, 1)
    assert model_input.input_positions.shape == (MAX_NUM_REQS,)
    assert model_input.num_valid_drafts is None
    assert model_input.draft_token_ids is None
    assert model_input.accepted_counts is None


@pytest.mark.parametrize("control", CONTROLS)
def test_a_feature_request_is_offered_no_drafts(control):
    """Its own proposal is withheld, so it never carries a draft to verify."""
    runner = _runner_with_feature_row(control)

    runner._publish_draft("feature", [31, 32], False)
    runner._publish_draft("greedy", [41, 42], False)

    assert "feature" not in runner._proposed_draft_token_ids
    assert runner._proposed_draft_token_ids["greedy"] == [41, 42]
    # Erasing on a withheld offer: a stale entry would be verified next step.
    runner._proposed_draft_token_ids["feature"] = [99]
    runner._publish_draft("feature", [31, 32], False)
    assert "feature" not in runner._proposed_draft_token_ids


@pytest.mark.parametrize("control", CONTROLS)
def test_an_unresolved_commit_reaches_the_ordinary_decode_a_feature_row_forces(control):
    """A feature request arriving right after a multi-token commit.

    The greedy row's previous verify committed 3 tokens, which a verify would
    resolve. The feature row makes the step an ordinary decode, so the count
    travels on that call, and the step then commits one token per row: the
    runner forgets the 3.
    """
    runner = _runner_with_feature_row(control, accepted_counts={"greedy": 3})

    model_input = _decode(runner, "greedy", "feature")

    assert model_input.spec_mode is None
    assert model_input.input_tokens.shape == (MAX_NUM_REQS, 1)
    assert model_input.num_valid_drafts is None
    assert model_input.accepted_counts.dtype == torch.int32
    assert model_input.accepted_counts.tolist() == [3] + [1] * (MAX_NUM_REQS - 1)
    assert runner._req_accepted_counts == {}


def test_speculation_resumes_once_the_feature_request_leaves():
    """Only the steps the feature request shares are ordinary decodes."""
    runner = _runner_with_feature_row("logprobs")

    model_input = _decode(runner, "greedy", "feature", drafts=_drafts(greedy=[21, 22]))
    assert model_input.spec_mode is None

    # The feature request finishes: its row leaves the persistent batch.
    removed = runner.input_batch.remove_request("feature")
    runner.input_batch.condense([removed])
    del runner.requests["feature"]

    model_input = _decode(runner, "greedy", drafts=_drafts(greedy=[21, 22]))
    assert model_input.spec_mode is not None
    assert model_input.input_tokens.shape == (MAX_NUM_REQS, 4)


def test_a_min_tokens_request_is_held_off_the_drafts_only_while_pending():
    """Once its output reaches ``min_tokens`` the stop token may be the argmax."""
    batch = _batch()
    pending = _add_feature_request(batch, "pending", _feature_params("min_tokens"))
    reached = _add_feature_request(
        batch,
        "reached",
        SamplingParams(temperature=0.0, min_tokens=OUTPUT_LEN, stop_token_ids=[2]),
    )
    runner = _fake_runner(
        batch, {"pending": pending, "reached": reached}, supports_narrow_decode=True
    )

    assert runner._request_is_speculable("pending") is False
    assert runner._request_is_speculable("reached") is True


def test_a_speculating_launch_builds_the_min_tokens_processor(monkeypatch):
    """The ordinary decode applies ``min_tokens`` through vLLM's own processor.

    ``build_logitsprocs`` builds only ``MinTokensLogitsProcessor`` when
    speculation is configured (no min_p, no logit_bias), which is the one this
    request needs; ``TTModelRunner`` builds its host processors that way.
    """
    from types import SimpleNamespace

    from vllm.v1.sample.logits_processor import (
        MinTokensLogitsProcessor,
        build_logitsprocs,
    )

    vllm_config = SimpleNamespace(
        speculative_config=object(),
        scheduler_config=SimpleNamespace(max_num_seqs=MAX_NUM_REQS),
    )
    procs = build_logitsprocs(
        vllm_config=vllm_config,
        device=torch.device("cpu"),
        is_pin_memory=False,
        is_pooling_model=False,
    )

    assert [type(p) for p in procs.all] == [MinTokensLogitsProcessor]


def test_an_ordinary_decode_samples_with_each_requests_committed_grammar_row():
    """The bitmask an ordinary decode inside a speculating launch applies.

    On such a launch vLLM writes each structured request one row per scheduled
    draft plus the bonus, and an asynchronous launch reserves placeholder rows
    on every step. A step that commits one token per row (``width=None``) must
    apply each request's row 0, its grammar at the committed output, and leave
    a row without a structured request unconstrained.
    """
    words = 4
    bitmask = torch.arange(3 * words, dtype=torch.int32).reshape(3, words)
    out = spec_grammar_bitmask_for_tt_batch(
        bitmask=bitmask,
        structured_output_request_ids=["feature"],
        rows_per_request={"feature": 3},
        row_req_ids=["greedy", "feature", None, None],
        batch_length=MAX_NUM_REQS,
        width=None,
    )

    assert out.shape == (MAX_NUM_REQS, words)
    assert out[1].tolist() == bitmask[0].tolist()
    assert out[0].tolist() == [-1] * words
    assert out[2].tolist() == [-1] * words
