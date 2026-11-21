# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only tests: a logits-only control on an asynchronous narrow-decode launch.

``test_narrow_decode_feature_requests.py`` pins, on the synchronous runner,
that a request carrying logprobs, structured output, ``allowed_token_ids``,
``bad_words`` or a pending ``min_tokens`` is never answered by a verify on a
narrow-decode ``argmax_ids`` launch: every step it shares is the ordinary
decode. These tests drive the same sequence through the asynchronous path,
``execute_model`` then ``sample_tokens`` with the readback held, and pin the
two things only that path can get wrong:

- the forced ordinary decode never overlaps an outstanding step. With such a
  row live, ``TTModelRunner.check_perform_device_sampling`` sends the step to
  the host sampler on a 4-chip mesh with no device grammar, and
  ``steady_decode_scheduler_invariants_met`` refuses it on the control itself
  (a grammar from the request state; the allowlist, bad words, logprobs and
  the ``min_tokens`` processor from the persistent batch). So the build waits
  for every outstanding step, and the step registers as not overlap-safe, so
  the next build waits for it;
- a multi-token commit still unresolved when the request arrives reaches that
  ordinary decode as ``accepted_counts``, as it does synchronously.

The runner is ``test_spec_async._baseline_runner``, on which an ordinary
greedy decode does overlap, with the real device-sampling predicate and the
logits processors ``build_logitsprocs`` gives a speculating launch, so the
refusal comes from the controls and not from the harness.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.sample.logits_processor import build_logitsprocs

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.host_sampler import create_host_sampler
from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.scheduler import get_tt_forced_reset_discard_counts

from . import test_spec_async as async_harness
from .deterministic_target import TARGET_VOCAB_SIZE
from .test_narrow_decode_feature_requests import CONTROLS, _feature_params

# The held-readback fixture, applied to every test here as it is there.
wait_for_released_reads = async_harness.wait_for_released_reads

MAX_NUM_REQS = async_harness.MAX_NUM_REQS
PROMPT_LEN = async_harness.PROMPT_LEN


def _feature_runner():
    """An overlapping narrow launch whose sampling route the runner decides.

    ``_baseline_runner`` answers every device-sampling question with yes. Here
    the real ``check_perform_device_sampling`` answers, with the values a
    4-chip launch without device grammar resolves, and the persistent batch
    carries the processors a speculating launch builds (``min_tokens`` only),
    which is what ``TTModelRunner.__init__`` hands its ``InputBatch``.
    """
    model = async_harness.DeferredVerifyTarget()
    # A drafter fed no hidden state, so a draftless step may precede a verify.
    model.offer_drafts_only_when_solo()
    runner = async_harness._baseline_runner(model)
    runner.check_perform_device_sampling = (
        TTModelRunner.check_perform_device_sampling.__get__(runner)
    )
    runner.sample_on_device_mode = "decode_only"
    runner.num_devices = 4
    runner.seeded_sampling_policy = "auto"
    runner.supports_device_grammar = False
    runner.supports_topk_logprobs = False
    runner.vocab_size = TARGET_VOCAB_SIZE
    runner.host_sampler = create_host_sampler()
    runner.input_batch = InputBatch(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=async_harness.MAX_MODEL_LEN,
        max_num_batched_tokens=async_harness.MAX_MODEL_LEN,
        vocab_size=TARGET_VOCAB_SIZE,
        block_sizes=[async_harness.BLOCK_SIZE],
        kernel_block_sizes=[async_harness.BLOCK_SIZE],
        logitsprocs=build_logitsprocs(
            vllm_config=SimpleNamespace(
                speculative_config=object(),
                scheduler_config=SimpleNamespace(max_num_seqs=MAX_NUM_REQS),
            ),
            device=torch.device("cpu"),
            is_pin_memory=False,
            is_pooling_model=False,
        ),
    )
    return model, runner


def _drain_admitting(runner, decoding, req_id, params):
    """``test_spec_async._drain``, admitting a request with its own params."""
    output = async_harness._scheduler_output(runner, decoding=decoding)
    output.scheduled_new_reqs = [
        NewRequestData(
            req_id=req_id,
            prompt_token_ids=list(range(71, 71 + PROMPT_LEN)),
            mm_features=[],
            sampling_params=params,
            pooling_params=None,
            block_ids=([0],),
            num_computed_tokens=PROMPT_LEN,
            lora_request=None,
        )
    ]
    output.num_scheduled_tokens[req_id] = 1
    output.total_num_scheduled_tokens += 1
    runner._update_states(output)
    runner.async_decode.apply_ready_completed_decode_steps(
        suppress_output_req_ids=runner.async_decode.suppressed_output_req_ids(output),
        forced_reset_discard_counts=get_tt_forced_reset_discard_counts(output),
    )


def _steady_candidate(runner, *req_ids) -> bool:
    """What ``execute_model`` asks before it decides whether to drain."""
    return runner.async_decode.can_attempt_steady_decode_from_scheduler(
        async_harness._scheduler_output(runner, decoding=req_ids)
    )


@pytest.mark.parametrize("control", CONTROLS)
def test_an_async_launch_never_overlaps_the_ordinary_decode_a_feature_row_forces(
    control,
):
    """A verify's commit, the feature request's arrival, and the steps after.

    The greedy request speculates alone, so its last verify commits a whole
    block. The feature request arrives in the step that applies that commit.
    The next step is the ordinary decode the feature row forces, which carries
    the greedy row's count and samples on the host; it is built with nothing
    outstanding and registers as not overlap-safe. Once the feature request
    leaves, an ordinary greedy step overlaps again.
    """
    model, runner = _feature_runner()
    controller = runner.async_decode
    async_harness._admit(runner, ("g", 11))
    async_harness._settle_layout(runner)

    # Solo and greedy: a draftless step overlaps, then a verify.
    plain = async_harness._submit_plain_step(runner, "g")
    assert list(runner._pending_async_overlap_ok) == [True], (
        "an ordinary greedy step could not overlap, so this launch proves nothing"
    )
    model.release()
    plain.get_output()
    async_harness._drain(runner, "g")
    drafts = async_harness._take_proposals(runner)
    assert drafts is not None and drafts.draft_token_ids[0], (
        "a solo request was offered no drafts"
    )
    async_harness._settle_layout(runner)
    verified = async_harness._submit_step(
        runner, "g", drafts={"g": list(drafts.draft_token_ids[0])}
    )
    model.release()
    verified.get_output()

    # The feature request arrives in the step that applies that commit.
    _drain_admitting(runner, ("g",), "feature", _feature_params(control))
    committed = runner._req_accepted_counts.get("g", 0)
    assert committed > 1, "the verify committed no block to resolve"
    async_harness._settle_layout(runner)
    assert not _steady_candidate(runner, "g", "feature"), (
        f"a step carrying {control} was a steady-decode candidate"
    )
    verifies = len(model.verify_calls)
    overlapped = controller._overlapped_submissions

    forced = async_harness._submit_plain_step(runner, "g", "feature")

    model_input = forced._model_input
    assert len(model.verify_calls) == verifies, "a feature row's step verified"
    assert model_input.spec_mode is None
    assert model_input.perform_device_sampling is False
    rows = runner.input_batch.req_id_to_index
    expected = [1] * MAX_NUM_REQS
    expected[rows["g"]] = committed
    assert model_input.accepted_counts.dtype == torch.int32
    assert model_input.accepted_counts.tolist() == expected
    assert runner._req_accepted_counts == {}
    # Built with nothing outstanding, and the next build waits for it.
    assert controller._overlapped_submissions == overlapped
    assert controller._overlapped_unsafe_submissions == 0
    assert list(runner._pending_async_overlap_ok) == [False]
    assert controller.must_drain_pending_async_steps(steady_decode_candidate=True)
    model.release()
    forced.get_output()
    async_harness._drain(runner, "g", "feature")
    async_harness._settle_layout(runner)

    # While the feature request is live, no step it shares overlaps.
    assert not _steady_candidate(runner, "g", "feature")
    again = async_harness._submit_plain_step(runner, "g", "feature")
    assert again._model_input.perform_device_sampling is False
    assert list(runner._pending_async_overlap_ok) == [False]
    model.release()
    again.get_output()
    async_harness._drain(runner, "g", "feature")

    # It leaves, and an ordinary greedy step overlaps again.
    async_harness._drain(runner, "g", finished=["feature"])
    async_harness._settle_layout(runner)
    assert _steady_candidate(runner, "g")
    resumed = async_harness._submit_plain_step(runner, "g")
    assert list(runner._pending_async_overlap_ok) == [True]
    model.release()
    resumed.get_output()
    assert controller._overlapped_unsafe_submissions == 0
