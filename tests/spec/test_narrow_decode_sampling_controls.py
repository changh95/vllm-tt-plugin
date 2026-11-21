# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only tests: ``logit_bias`` and ``min_p`` on a narrow-decode launch.

No accept walk applies either control, the ``logits`` walk included, and vLLM
refuses both on every speculating launch. A TT launch whose ``SpecPlan`` keeps
``supports_narrow_decode`` serves them without speculation, through three
pieces this file pins:

- ``_install_tt_spec_sampling_validation_patch`` lifts vLLM's refusal
  (``SamplingParams._validate_spec_decode``) for the requesting engine only,
  when its plan keeps narrow decode;
- ``build_host_logitsprocs`` gives such a launch vLLM's whole builtin
  processor set, so the ordinary decode applies both on the host;
- ``TTModelRunner._request_is_speculable`` is false for such a request in
  every accept mode, so every step it shares is the ordinary decode.
"""

from types import SimpleNamespace

import pytest
import torch

# vLLM's own bootstrap resolves the platform plugin, which imports plugin
# modules. Let vLLM finish importing itself first.
import vllm  # noqa: F401
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
    MinTokensLogitsProcessor,
)
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_tt_plugin.host_sampler import build_host_logitsprocs
from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.spec_admission import ordinary_decode_controls

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
from .test_spec_platform_integration import (
    _input_processor_for_launch,
    _process_until_admission,
)


def _with_logit_bias(bias=None, **kwargs) -> SamplingParams:
    kwargs.setdefault("temperature", 0.0)
    return SamplingParams(logit_bias={3: 1.0} if bias is None else bias, **kwargs)


CONTROLLED = {
    "logit_bias": lambda: _with_logit_bias(),
    "min_p": lambda: SamplingParams(temperature=0.7, min_p=0.1, seed=5),
    "both": lambda: _with_logit_bias(temperature=0.7, min_p=0.1, seed=5),
}

# region Which requests carry a control no walk applies


def test_vllm_resets_min_p_on_a_greedy_request():
    """The ``min_p`` rule relies on this vLLM 0.31 behaviour; pin it.

    ``SamplingParams.__post_init__`` zeroes ``min_p`` below ``_SAMPLING_EPS``
    temperature, before ``verify`` reaches ``_validate_spec_decode``. A vLLM
    that stopped doing so would turn greedy + min_p into a refusal on every
    launch without narrow decode.
    """
    params = SamplingParams(temperature=0.0, min_p=0.5)

    assert params.min_p == 0.0
    assert ordinary_decode_controls(params) == []
    # vLLM's own check (no test starts with the wrapper installed) accepts it
    # on a speculating launch.
    params._validate_spec_decode(SimpleNamespace(method="ngram"))


def test_a_tiny_temperature_is_sampled_and_keeps_its_min_p():
    """vLLM raises a temperature in (0, 0.01) to 0.01 before its greedy test.

    So only temperature 0 is greedy: a request asking for 1e-6 samples, its
    ``min_p`` stays, and the ordinary decode applies it.
    """
    params = SamplingParams(temperature=1e-6, min_p=0.5)

    assert params.temperature == pytest.approx(0.01)
    assert params.min_p == 0.5
    assert ordinary_decode_controls(params) == ["min_p=0.5"]


@pytest.mark.parametrize(
    "params, controls",
    [
        (SamplingParams(temperature=0.0), []),
        (SamplingParams(temperature=0.7, min_p=0.1), ["min_p=0.1"]),
        # Below vLLM's refusal threshold, but MinPLogitsProcessor applies it.
        (SamplingParams(temperature=0.7, min_p=1e-6), ["min_p=1e-06"]),
        (_with_logit_bias(), ["logit_bias"]),
        # An all-zero bias still counts: it costs speculation, nothing else.
        (_with_logit_bias({3: 0.0}), ["logit_bias"]),
        (_with_logit_bias(temperature=0.7, min_p=0.1), ["logit_bias", "min_p=0.1"]),
    ],
)
def test_the_controls_no_walk_applies_are_named(params, controls):
    assert ordinary_decode_controls(params) == controls


# endregion

# region vLLM's refusal, lifted for a narrow-decode engine only


def _processor_running_vllms_check(monkeypatch, vllm_config, launch):
    """A real ``InputProcessor`` whose parameter validation is vLLM's refusal.

    ``SamplingParams.verify`` needs a real model config for its other checks;
    ``_validate_spec_decode`` is the one the patch wraps, and it runs here
    inside ``InputProcessor.process_inputs``, where ``process_inputs_tt``
    binds the requesting engine.
    """
    processor = _input_processor_for_launch(monkeypatch, vllm_config, launch)
    processor._validate_params = lambda params, _tasks: params._validate_spec_decode(
        processor.vllm_config.speculative_config
    )
    return processor


@pytest.mark.parametrize("control", list(CONTROLLED))
def test_a_narrow_decode_engine_admits_them_through_the_input_processor(
    monkeypatch, vllm_config, control
):
    processor = _processor_running_vllms_check(monkeypatch, vllm_config, "narrow")
    _process_until_admission(processor, CONTROLLED[control]())


@pytest.mark.parametrize("control", list(CONTROLLED))
def test_vllm_itself_still_refuses_them_where_every_step_verifies(
    monkeypatch, vllm_config, control
):
    processor = _processor_running_vllms_check(monkeypatch, vllm_config, "verify-only")
    with pytest.raises(VLLMValidationError, match="not yet supported"):
        _process_until_admission(processor, CONTROLLED[control]())


@pytest.mark.parametrize(
    "launches", [("narrow", "verify-only"), ("verify-only", "narrow")]
)
def test_each_live_engine_keeps_its_own_answer(monkeypatch, vllm_config, launches):
    """The lift reads the requesting engine, not the last one configured."""
    from copy import deepcopy

    processors = [
        _processor_running_vllms_check(monkeypatch, deepcopy(vllm_config), launch)
        for launch in launches
    ]
    for launch, processor in zip(launches, processors):
        params = CONTROLLED["both"]()
        if launch == "verify-only":
            with pytest.raises(VLLMValidationError, match="not yet supported"):
                _process_until_admission(processor, params)
        else:
            _process_until_admission(processor, params)


def test_outside_a_request_the_original_check_runs(monkeypatch, vllm_config):
    """No TT request context: vLLM's behaviour, even with a narrow engine live."""
    _processor_running_vllms_check(monkeypatch, vllm_config, "narrow")
    with pytest.raises(VLLMValidationError, match="not yet supported"):
        CONTROLLED["min_p"]()._validate_spec_decode(SimpleNamespace(method="ngram"))


def test_the_lift_leaves_the_callers_params_untouched(monkeypatch, vllm_config):
    """vLLM validates the caller-owned params before cloning them."""
    processor = _processor_running_vllms_check(monkeypatch, vllm_config, "narrow")
    params = CONTROLLED["both"]()
    _process_until_admission(processor, params)
    assert params.logit_bias == {3: 1.0}
    assert params.min_p == 0.1


def test_the_patch_installs_once():
    import vllm.sampling_params as sampling_params

    from vllm_tt_plugin.platform import _install_tt_spec_sampling_validation_patch

    original = SamplingParams._validate_spec_decode
    _install_tt_spec_sampling_validation_patch()
    wrapper = SamplingParams._validate_spec_decode
    _install_tt_spec_sampling_validation_patch()

    assert wrapper is not original
    assert SamplingParams._validate_spec_decode is wrapper
    assert sampling_params._tt_original_validate_spec_decode is original


def test_the_patch_refuses_a_reshaped_method(monkeypatch):
    """A vLLM that changed the method fails at install, naming it."""
    from vllm_tt_plugin.platform import _install_tt_spec_sampling_validation_patch

    monkeypatch.setattr(
        SamplingParams, "_validate_spec_decode", lambda self, config, strict: None
    )
    with pytest.raises(RuntimeError, match="_validate_spec_decode"):
        _install_tt_spec_sampling_validation_patch()


def test_no_test_inherits_the_wrapper():
    """``reset_tt_platform_class_state`` restores vLLM's method after each test.

    Earlier tests here installed the wrapper; a leaked one would decide
    admission for tests that never configured a TT engine.
    """
    import vllm.sampling_params as sampling_params

    method = SamplingParams._validate_spec_decode
    assert method.__module__ == "vllm.sampling_params"
    assert method.__qualname__ == "SamplingParams._validate_spec_decode"
    assert not hasattr(sampling_params, "_tt_original_validate_spec_decode")


# endregion

# region The processors the ordinary decode samples with


def _plan(supports_narrow_decode):
    return SimpleNamespace(supports_narrow_decode=supports_narrow_decode)


def _config(speculating=True):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method="ngram") if speculating else None,
        scheduler_config=SimpleNamespace(max_num_seqs=MAX_NUM_REQS),
    )


def _types(procs):
    return sorted(type(proc).__name__ for proc in procs.all)


def test_a_narrow_speculating_launch_samples_with_every_builtin_processor():
    procs = build_host_logitsprocs(_config(), _plan(True))
    assert _types(procs) == sorted(
        cls.__name__
        for cls in (
            MinPLogitsProcessor,
            LogitBiasLogitsProcessor,
            MinTokensLogitsProcessor,
        )
    )


def test_a_launch_that_verifies_every_step_keeps_vllms_speculative_set():
    procs = build_host_logitsprocs(_config(), _plan(False))
    assert _types(procs) == [MinTokensLogitsProcessor.__name__]


@pytest.mark.parametrize("narrow", [True, False])
def test_custom_processors_stay_refused_on_any_speculating_launch(narrow):
    from vllm.v1.sample.logits_processor import STR_SPEC_DEC_REJECTS_LOGITSPROCS

    with pytest.raises(ValueError) as refused:
        build_host_logitsprocs(
            _config(), _plan(narrow), custom_logitsprocs=("some.module:Proc",)
        )
    assert str(refused.value) == STR_SPEC_DEC_REJECTS_LOGITSPROCS


def _host_batch(params: SamplingParams) -> InputBatch:
    """One request in a batch holding the narrow speculating launch's set."""
    batch = InputBatch(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=64,
        max_num_batched_tokens=64,
        vocab_size=16,
        block_sizes=[16],
        kernel_block_sizes=[16],
        logitsprocs=build_host_logitsprocs(_config(), _plan(True)),
    )
    batch.add_request(
        CachedRequestState(
            req_id="r",
            prompt_token_ids=[1, 2, 3],
            mm_features=None,
            sampling_params=params,
            generator=None,
            block_ids=([0],),
            num_computed_tokens=3,
            output_token_ids=[],
        )
    )
    batch.refresh_logitsprocs()
    return batch


def _apply(batch: InputBatch, logits: torch.Tensor) -> torch.Tensor:
    for proc in batch.sampling.logitsprocs.all:
        logits = proc.apply(logits)
    return logits


def test_the_ordinary_decode_applies_logit_bias():
    """A large bias makes a greedy request take the biased token."""
    batch = _host_batch(_with_logit_bias({11: 100.0}))
    assert batch.sampling.has_active_logitsprocs()

    logits = torch.zeros(1, 16)
    logits[0, 4] = 5.0
    assert int(_apply(batch, logits).argmax()) == 11


def test_the_ordinary_decode_applies_min_p():
    """Tokens below ``min_p`` times the top probability are masked."""
    batch = _host_batch(SamplingParams(temperature=1.0, min_p=0.5, seed=1))
    assert batch.sampling.has_active_logitsprocs()

    logits = torch.full((1, 16), -30.0)
    logits[0, 2] = 2.0  # the top token
    logits[0, 5] = 2.0 - 0.5  # p ratio exp(-0.5) ~ 0.61: kept
    logits[0, 7] = 2.0 - 1.0  # p ratio exp(-1) ~ 0.37: masked
    out = _apply(batch, logits)
    kept = torch.isfinite(out[0]).nonzero().flatten().tolist()
    assert kept == [2, 5]


def test_the_ordinary_decode_applies_a_min_p_below_vllms_refusal_threshold():
    """Why ``ordinary_decode_controls`` counts any ``min_p`` above 0.

    vLLM refuses ``min_p`` on a speculating launch only above 1e-5, but its
    processor applies a smaller one too; the ``logits`` walk applies none.
    """
    batch = _host_batch(SamplingParams(temperature=1.0, min_p=1e-6, seed=1))
    assert batch.sampling.has_active_logitsprocs()


# endregion

# region The runner never verifies such a request


def _add_controlled_request(batch, req_id, params):
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


def _runner_with_controlled_row(control, modes, accepted_counts=None):
    batch = _batch()
    requests = {
        "greedy": _add_decoding_request(batch, "greedy"),
        "controlled": _add_controlled_request(
            batch, "controlled", CONTROLLED[control]()
        ),
    }
    runner = _fake_runner(
        batch,
        requests,
        supports_narrow_decode=True,
        accepted_counts=accepted_counts,
    )
    runner._spec_accept_modes = modes
    return runner


MODES = [("argmax_ids",), ("argmax_ids", "logits"), ("logits",)]


@pytest.mark.parametrize("modes", MODES)
@pytest.mark.parametrize("control", list(CONTROLLED))
def test_such_a_row_is_never_answered_by_a_verify(control, modes):
    """Even where the ``logits`` walk would speculate every other control."""
    runner = _runner_with_controlled_row(control, modes)
    assert runner._request_is_speculable("controlled") is False
    assert runner._request_is_speculable("greedy") is True

    model_input = _decode(
        runner, "greedy", "controlled", drafts=_drafts(greedy=[21, 22])
    )

    assert model_input.spec_mode is None
    assert model_input.input_tokens.shape == (MAX_NUM_REQS, 1)
    assert model_input.num_valid_drafts is None
    assert model_input.draft_token_ids is None


@pytest.mark.parametrize("modes", MODES)
@pytest.mark.parametrize("control", list(CONTROLLED))
def test_such_a_request_is_offered_no_drafts(control, modes):
    runner = _runner_with_controlled_row(control, modes)
    runner._proposed_draft_token_ids["controlled"] = [99]

    runner._publish_draft("controlled", [31, 32], False)
    runner._publish_draft("greedy", [41, 42], False)

    assert "controlled" not in runner._proposed_draft_token_ids
    assert runner._proposed_draft_token_ids["greedy"] == [41, 42]


@pytest.mark.parametrize("control", list(CONTROLLED))
def test_an_unresolved_commit_reaches_the_ordinary_decode_such_a_row_forces(control):
    runner = _runner_with_controlled_row(
        control, ("argmax_ids", "logits"), accepted_counts={"greedy": 3}
    )

    model_input = _decode(runner, "greedy", "controlled")

    assert model_input.spec_mode is None
    assert model_input.accepted_counts.tolist() == [3] + [1] * (MAX_NUM_REQS - 1)
    assert runner._req_accepted_counts == {}


# endregion
