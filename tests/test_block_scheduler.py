# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_tt_plugin.config import (
    store_tt_adaptive_block_batched,
    store_tt_adaptive_block_output,
    store_tt_adaptive_block_ragged,
    store_tt_block_output_kv_lookahead_tokens,
    store_tt_output_tokens_per_step,
)
from vllm_tt_plugin.scheduler import (
    TTScheduler,
    get_tt_forced_reset_discard_counts,
)

BLOCK_SIZE = 128
CANVAS = 16
MAX_MODEL_LEN = 256
LOCAL_MODEL_CONFIG = Path(__file__).parent / "model_configs" / "qwen2"


class _StubModel:
    """No model_capabilities: the platform hook resolves
    output_tokens_per_step=1; tests inject the block width afterwards."""


@contextmanager
def _stub_model_resolution():
    """Resolve the stub instead of tt-metal's TTQwen2ForCausalLM while the
    platform hook runs inside VllmConfig.__post_init__ (mirrors
    test_block_request_validation._patch_model_resolution)."""
    with (
        # Fresh-process semantics: don't leave this file's configs as the
        # platform's process-level admission handle across tests.
        patch("vllm_tt_plugin.platform.TTPlatform._tt_vllm_config", None),
        patch("vllm_tt_plugin.platform.register_tt_models"),
        patch(
            "vllm_tt_plugin.platform._resolve_standard_dp_visible_device_groups",
            return_value=None,
        ),
        patch(
            "vllm.model_executor.models.registry.ModelRegistry.get_supported_archs",
            return_value=["TTQwen2ForCausalLM"],
        ),
        patch(
            "vllm.model_executor.model_loader.utils.get_model_architecture",
            return_value=(_StubModel, None),
        ),
    ):
        yield


def _scheduler(
    output_width: int = CANVAS,
    *,
    diffusion_checkpoint: bool = False,
    max_model_len: int = MAX_MODEL_LEN,
    async_scheduling: bool = False,
    adaptive: bool = False,
    max_num_seqs: int = 1,
    kv_lookahead: int = 0,
    batched: bool = False,
    ragged: bool = False,
) -> TTScheduler:
    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    model_config.max_model_len = max_model_len
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_model_len,
        max_model_len=max_model_len,
        enable_chunked_prefill=False,
        async_scheduling=False,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    with _stub_model_resolution():
        config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
        )
    config.scheduler_config.async_scheduling = async_scheduling
    if diffusion_checkpoint:
        # Reproduce the platform hook's post-update state, including
        # invalidation of ModelConfig.is_diffusion's cached True value.
        config.model_config.hf_config.canvas_length = output_width
        config.model_config.__dict__.pop("is_diffusion", None)
        assert config.model_config.is_diffusion is True
        delattr(config.model_config.hf_config, "canvas_length")
        config.model_config.__dict__.pop("is_diffusion", None)
        assert config.model_config.is_diffusion is False
    store_tt_output_tokens_per_step(config, output_width)
    if adaptive:
        store_tt_adaptive_block_output(config, True)
    if batched:
        store_tt_adaptive_block_batched(config, True)
    if ragged:
        store_tt_adaptive_block_ragged(config, True)
    if kv_lookahead:
        store_tt_block_output_kv_lookahead_tokens(config, kv_lookahead)
    num_blocks = max_model_len // BLOCK_SIZE + 2
    cache_config.num_gpu_blocks = num_blocks
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    return TTScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )


def _request(
    max_tokens: int,
    *,
    ignore_eos: bool = True,
    request_id: str = "req-0",
    prompt_len: int = 32,
) -> Request:
    init_none_hash(sha256)
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        ignore_eos=ignore_eos,
    )
    sampling_params.update_from_generation_config({}, eos_token_id=2)
    return Request(
        request_id=request_id,
        prompt_token_ids=[1] * prompt_len,
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _scheduled(
    max_tokens: int = CANVAS * 2,
    *,
    output_width: int = CANVAS,
    ignore_eos: bool = True,
    async_scheduling: bool = False,
) -> tuple[TTScheduler, Request, SchedulerOutput]:
    scheduler = _scheduler(output_width, async_scheduling=async_scheduling)
    request = _request(max_tokens, ignore_eos=ignore_eos)
    scheduler.add_request(request)
    return scheduler, request, scheduler.schedule()


def _runner_output(
    scheduler_output: SchedulerOutput, tokens: list[int]
) -> ModelRunnerOutput:
    req_id = next(iter(scheduler_output.num_scheduled_tokens))
    return ModelRunnerOutput(
        req_ids=[req_id],
        req_id_to_index={req_id: 0},
        sampled_token_ids=[tokens],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


@pytest.mark.parametrize(
    ("max_tokens", "ignore_eos", "canvas", "kept", "status"),
    [
        (3, True, list(range(CANVAS)), [0, 1, 2], RequestStatus.FINISHED_LENGTH_CAPPED),
        (
            CANVAS,
            False,
            [0, 2, *range(2, CANVAS)],
            [0, 2],
            RequestStatus.FINISHED_STOPPED,
        ),
    ],
    ids=["max_tokens", "eos"],
)
def test_trimmed_canvas_consumes_physical_reservation(
    max_tokens, ignore_eos, canvas, kept, status
):
    scheduler, request, prefill = _scheduled(max_tokens, ignore_eos=ignore_eos)

    outputs = scheduler.update_from_output(prefill, _runner_output(prefill, canvas))

    assert outputs[0].outputs[0].new_token_ids == kept
    assert list(request.output_token_ids) == kept
    assert request.num_output_placeholders == 0
    assert request.status == status


def test_add_request_clamps_max_tokens_that_would_overshoot_max_model_len():
    """A prebuilt request's leftover max_tokens must not schedule a canvas
    past max_model_len; that path raises in the runner and kills the engine."""
    scheduler = _scheduler()
    request = _request(max_tokens=MAX_MODEL_LEN)

    scheduler.add_request(request)

    # prompt=32, max_model_len=256, canvas=16 → 224 tokens of whole canvases.
    assert request.max_tokens == 224
    assert request.sampling_params.max_tokens == 224


def test_add_request_strips_host_sampling_controls_from_bypassed_request():
    """A prebuilt EngineCoreRequest skips frontend validation; any of these
    controls flips the step onto host sampling, which cannot construct a
    multi-token canvas and would kill the engine."""
    scheduler = _scheduler()
    init_none_hash(sha256)
    params = SamplingParams(
        max_tokens=CANVAS,
        ignore_eos=True,
        min_p=0.2,
        min_tokens=1,
        logit_bias={2: 1.0},
        allowed_token_ids=[1, 2],
        bad_words=["bad"],
        presence_penalty=0.5,
        frequency_penalty=0.5,
        repetition_penalty=1.1,
        structured_outputs=StructuredOutputsParams(json_object=True),
    )
    params.update_from_generation_config({}, eos_token_id=2)
    # The tokenized form is what actually flips the worker onto host sampling
    # (InputBatch reads bad_words_token_ids, not the strings); with
    # skip_tokenizer_init it stays unset unless seeded here.
    params._bad_words_token_ids = [[7]]
    request = Request(
        request_id="bypass-0",
        prompt_token_ids=[1] * 32,
        sampling_params=params,
        pooling_params=None,
        resumable=True,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    assert request.use_structured_output
    assert request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR

    scheduler.add_request(request)

    assert not request.use_structured_output
    assert params.structured_outputs is None
    assert params.min_p == 0.0
    assert params.min_tokens == 0
    assert params.logit_bias is None
    assert params.allowed_token_ids is None
    assert params.bad_words is None
    assert params._bad_words_token_ids is None
    # Non-neutral penalties make every block step build session-length
    # penalty tensors it then discards.
    assert params.presence_penalty == 0.0
    assert params.frequency_penalty == 0.0
    assert params.repetition_penalty == 1.0
    # A resumable session would park the stopped request forever and leak the
    # model-owned state slot.
    assert request.resumable is False
    # With the structured-output request gone, nothing could ever promote the
    # request out of skipped_waiting; it must be schedulable immediately.
    assert request.status == RequestStatus.WAITING
    scheduled = scheduler.schedule()
    assert scheduled.num_scheduled_tokens == {"bypass-0": 32}


# Largest tile-aligned prompt that still fits one whole canvas: the
# truncation target for unservable bypassed prompts (mml=256, K=16 -> 224).
SERVABLE_PROMPT = (MAX_MODEL_LEN // 32 * 32 - CANVAS) // 32 * 32


@pytest.mark.parametrize(
    "prompt_len",
    [MAX_MODEL_LEN - 15, MAX_MODEL_LEN + 40],
    ids=["dead-zone-band", "beyond-max-model-len"],
)
def test_unservable_bypassed_prompt_is_truncated_and_served(prompt_len):
    """A bypassed prompt with no room for a whole canvas is otherwise fatal:
    parked forever when it exceeds the token budget, overflowing the worker's
    max_model_len-wide buffer when it doesn't, or — even when it fits
    max_model_len — killing the engine via the adapter's own capacity check
    in eager mode, which raises instead of returning a clippable canvas.
    Truncation makes the request genuinely servable end to end."""
    scheduler = _scheduler()
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=64, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=2)
    request = Request(
        request_id="unservable-0",
        prompt_token_ids=[1] * prompt_len,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )

    scheduler.add_request(request)

    assert request.num_prompt_tokens == SERVABLE_PROMPT
    assert len(request.prompt_token_ids) == SERVABLE_PROMPT
    assert request.num_tokens == SERVABLE_PROMPT
    # Whole canvases still fitting after the tile-aligned truncation.
    assert request.max_tokens == 32

    prefill = scheduler.schedule()
    assert prefill.num_scheduled_tokens == {"unservable-0": SERVABLE_PROMPT}
    scheduler.update_from_output(prefill, _runner_output(prefill, list(range(CANVAS))))
    assert request.status == RequestStatus.RUNNING

    decode = scheduler.schedule()
    scheduler.update_from_output(decode, _runner_output(decode, list(range(CANVAS))))

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert scheduler.running == []


def test_continuation_of_scrubbed_resumable_session_is_dropped():
    """Scrubbing resumable admits the first chunk with streaming_queue=None,
    and the streaming protocol always sends a same-id follow-up (the next
    chunk or the closing sentinel); the base scheduler's duplicate-id assert
    on the missing queue would tear down EngineCore."""
    scheduler = _scheduler()
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=CANVAS, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=2)
    first = Request(
        request_id="stream-0",
        prompt_token_ids=[1] * 32,
        sampling_params=params,
        pooling_params=None,
        resumable=True,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    scheduler.add_request(first)
    assert first.resumable is False
    assert first.streaming_queue is None

    sentinel_params = SamplingParams(max_tokens=1)
    sentinel_params.update_from_generation_config({}, eos_token_id=2)
    sentinel = Request(
        request_id="stream-0",
        prompt_token_ids=[0],
        sampling_params=sentinel_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    scheduler.add_request(sentinel)  # must not raise

    assert scheduler.requests["stream-0"] is first
    prefill = scheduler.schedule()
    assert prefill.num_scheduled_tokens == {"stream-0": 32}


def test_multimodal_features_are_dropped_from_bypassed_request():
    """A text-only block model has a zero encoder budget: an mm feature at
    offset 0 parks the request in WAITING forever (head-of-line stall), and
    an interior offset carves a partial prefill chunk that flips the step
    onto host sampling and kills the engine."""
    from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange

    scheduler = _scheduler()
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=3, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=2)
    request = Request(
        request_id="mm-0",
        prompt_token_ids=[1] * 32,
        sampling_params=params,
        pooling_params=None,
        mm_features=[
            MultiModalFeatureSpec(
                data=None,
                modality="image",
                identifier="img-0",
                mm_position=PlaceholderRange(offset=0, length=16),
            )
        ],
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    assert request.has_encoder_inputs

    scheduler.add_request(request)

    assert request.mm_features == []
    assert not request.has_encoder_inputs

    prefill = scheduler.schedule()
    assert prefill.num_scheduled_tokens == {"mm-0": 32}
    scheduler.update_from_output(prefill, _runner_output(prefill, list(range(CANVAS))))

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert scheduler.running == []


@pytest.mark.parametrize(
    ("output_width", "max_model_len", "expected_prompt", "expected_max_tokens"),
    [
        # keep = (256 - 64) // 32 * 32 = 192; remaining 64 = one 64-canvas.
        pytest.param(64, 256, 192, 64, id="width-above-tile"),
        # aligned mml = 250 // 32 * 32 = 224; keep = (224 - 16) // 32 * 32
        # = 192; remaining 32 = two 16-canvases.
        pytest.param(CANVAS, 250, 192, 32, id="unaligned-max-model-len"),
    ],
)
def test_truncation_respects_width_and_alignment(
    output_width, max_model_len, expected_prompt, expected_max_tokens
):
    """Regimes where the width term and tile alignment actually matter; the
    default fixture (canvas 16, aligned limit) is insensitive to both."""
    scheduler = _scheduler(output_width, max_model_len=max_model_len)
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=64, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=2)
    request = Request(
        request_id="regime-0",
        prompt_token_ids=[1] * 300,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )

    scheduler.add_request(request)

    assert request.num_prompt_tokens == expected_prompt
    assert request.max_tokens == expected_max_tokens


def test_plain_prefix_cache_reset_leaves_running_block_requests_alone():
    """A plain /reset_prefix_cache (reset_running_requests=False) must
    delegate upstream instead of raising or preempting live block work."""
    scheduler = _scheduler()
    init_none_hash(sha256)
    request = _request(CANVAS * 2)
    scheduler.add_request(request)
    prefill = scheduler.schedule()
    scheduler.update_from_output(prefill, _runner_output(prefill, list(range(CANVAS))))
    assert request.status == RequestStatus.RUNNING

    result = scheduler.reset_prefix_cache(
        reset_running_requests=False, reset_connector=False
    )

    assert result is False
    assert request.status == RequestStatus.RUNNING
    assert scheduler.running == [request]


def test_mixed_token_embeds_bypassed_prompt_drops_the_embeds():
    """Mixed token+embeds prompts skip the embeds-only branch; the embeds
    must still be scrubbed rather than pinned for the request lifetime."""
    scheduler = _scheduler()
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=3, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=2)
    request = Request(
        request_id="mixed-0",
        prompt_token_ids=[1, 0, 3, 0, 5, 6, 7, 8],
        prompt_embeds=torch.zeros(8, 4),
        prompt_is_token_ids=[True, False, True, False, True, True, True, True],
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )

    scheduler.add_request(request)

    assert request.prompt_embeds is None
    assert request.prompt_is_token_ids is None
    assert request.prompt_token_ids == [1, 0, 3, 0, 5, 6, 7, 8]


def test_embeds_only_bypassed_prompt_is_replaced_with_placeholders():
    """The frontend rejects prompt_embeds for every TT model; admitted bare,
    the worker's request-state builder raises NotImplementedError out of
    execute_model and kills the engine."""
    scheduler = _scheduler()
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=3, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=2)
    request = Request(
        request_id="embeds-0",
        prompt_token_ids=None,
        prompt_embeds=torch.zeros(8, 4),
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )

    scheduler.add_request(request)

    assert request.prompt_token_ids == [0] * 8
    assert request.prompt_embeds is None
    assert request.num_prompt_tokens == 8
    # The max_tokens clamp no longer early-returns on the missing token ids.
    assert request.max_tokens == 3

    prefill = scheduler.schedule()
    assert prefill.num_scheduled_tokens == {"embeds-0": 8}
    scheduler.update_from_output(prefill, _runner_output(prefill, list(range(CANVAS))))

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert scheduler.running == []


def test_oversized_embeds_only_prompt_is_replaced_and_truncated():
    """Embeds replacement and prompt truncation mutate the same four fields
    in sequence; a refactor breaking only the composition would admit an
    oversized or internally inconsistent request while the single-step tests
    stay green."""
    scheduler = _scheduler()
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=64, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=2)
    request = Request(
        request_id="embeds-big-0",
        prompt_token_ids=None,
        prompt_embeds=torch.zeros(MAX_MODEL_LEN + 40, 4),
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )

    scheduler.add_request(request)

    assert request.prompt_token_ids == [0] * SERVABLE_PROMPT
    assert request.prompt_embeds is None
    assert request.num_prompt_tokens == SERVABLE_PROMPT
    assert request.num_tokens == SERVABLE_PROMPT
    assert request.max_tokens == 32

    prefill = scheduler.schedule()
    assert prefill.num_scheduled_tokens == {"embeds-big-0": SERVABLE_PROMPT}
    scheduler.update_from_output(prefill, _runner_output(prefill, list(range(CANVAS))))
    decode = scheduler.schedule()
    scheduler.update_from_output(decode, _runner_output(decode, list(range(CANVAS))))

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert scheduler.running == []


def test_empty_bypassed_prompt_is_padded_and_served():
    """The frontend rejects empty prompts; admitted bare, the waiting loop
    schedules zero new tokens and upstream's num_new_tokens assert tears
    down the engine."""
    scheduler = _scheduler()
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=3, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=2)
    request = Request(
        request_id="empty-0",
        prompt_token_ids=[],
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )

    scheduler.add_request(request)

    assert request.prompt_token_ids == [0]
    assert request.num_prompt_tokens == 1
    assert request.num_tokens == 1

    prefill = scheduler.schedule()
    assert prefill.num_scheduled_tokens == {"empty-0": 1}
    scheduler.update_from_output(prefill, _runner_output(prefill, list(range(CANVAS))))

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert scheduler.running == []


def test_zero_max_tokens_bypassed_request_finishes_after_first_canvas():
    """A hand-crafted prebuilt request can carry max_tokens=0 (SamplingParams
    itself forbids it); the stop check must finish it length-capped on its
    first canvas instead of generating forever."""
    scheduler = _scheduler()
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=1, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=2)
    params.max_tokens = 0
    request = Request(
        request_id="zero-0",
        prompt_token_ids=[1] * 32,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    assert request.max_tokens == 0

    scheduler.add_request(request)
    assert request.max_tokens == 0

    prefill = scheduler.schedule()
    outputs = scheduler.update_from_output(
        prefill, _runner_output(prefill, list(range(CANVAS)))
    )

    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    # The stopping token is the only client-visible overshoot of the zero
    # budget; the request finishes through the normal output path.
    assert outputs[0].outputs[0].new_token_ids == [0]
    assert request.num_output_placeholders == 0
    assert scheduler.running == []


def test_running_block_request_rejects_prefix_cache_reset():
    # Direct scheduler-level backstop: the engine-layer patch aborts running
    # block requests first, so reaching this guard means it was bypassed.
    scheduler, request, _ = _scheduled()

    assert scheduler.running == [request]
    with pytest.raises(RuntimeError, match="Cannot reset prefix cache"):
        scheduler.reset_prefix_cache(
            reset_running_requests=True,
            reset_connector=False,
        )
    assert scheduler.running == [request]
    assert request.status == RequestStatus.RUNNING


def _patched_engine(scheduler: TTScheduler, sent: list) -> SimpleNamespace:
    from vllm_tt_plugin.platform import _install_block_output_reset_abort_patch

    _install_block_output_reset_abort_patch()
    return SimpleNamespace(scheduler=scheduler, _send_abort_outputs=sent.append)


def test_engine_level_reset_aborts_running_block_requests():
    scheduler, request, _ = _scheduled()
    sent: list = []
    engine = _patched_engine(scheduler, sent)

    assert (
        EngineCore.reset_prefix_cache(
            engine, reset_running_requests=True, reset_connector=False
        )
        is True
    )
    assert scheduler.running == []
    assert request.status == RequestStatus.FINISHED_ABORTED
    assert sent == [[request]]


def test_engine_reset_patch_requires_resolved_output_width():
    scheduler, _, _ = _scheduled()
    scheduler.vllm_config.additional_config.clear()
    engine = _patched_engine(scheduler, [])

    with pytest.raises(RuntimeError, match="was not initialized"):
        EngineCore.reset_prefix_cache(
            engine, reset_running_requests=True, reset_connector=False
        )


def test_engine_without_abort_notifier_refuses_reset():
    # A bare in-process EngineCore lacks _send_abort_outputs: aborting there
    # would silently remove a request its caller is still waiting on, so the
    # reset must fall through to the scheduler guard's raise instead.
    scheduler, request, _ = _scheduled()
    from vllm_tt_plugin.platform import _install_block_output_reset_abort_patch

    _install_block_output_reset_abort_patch()
    engine = SimpleNamespace(scheduler=scheduler)

    with pytest.raises(RuntimeError, match="Cannot reset prefix cache"):
        EngineCore.reset_prefix_cache(
            engine, reset_running_requests=True, reset_connector=False
        )
    assert scheduler.running == [request]
    assert request.status == RequestStatus.RUNNING


def test_engine_level_keep_pause_reset_preserves_block_requests():
    scheduler, request, _ = _scheduled()
    scheduler.set_pause_state(PauseState.PAUSED_ALL)
    sent: list = []
    engine = _patched_engine(scheduler, sent)

    assert (
        EngineCore.reset_prefix_cache(
            engine, reset_running_requests=True, reset_connector=False
        )
        is False
    )
    assert scheduler.running == [request]
    assert request.status == RequestStatus.RUNNING
    assert sent == []


def _pause_guarded_engine(scheduler: TTScheduler) -> SimpleNamespace:
    from vllm_tt_plugin.platform import _install_block_output_pause_guard_patch

    _install_block_output_pause_guard_patch()
    return SimpleNamespace(scheduler=scheduler)


def test_keep_pause_with_clear_cache_is_refused_up_front():
    # The keep-mode reset runs from an idle callback whose result upstream
    # discards, so the only honest failure is a synchronous one before any
    # pause state changes.
    scheduler, request, _ = _scheduled()
    engine = _pause_guarded_engine(scheduler)

    with pytest.raises(ValueError, match="clear_cache=False or mode='abort'"):
        EngineCore.pause_scheduler(engine, mode="keep", clear_cache=True)

    assert scheduler.pause_state == PauseState.UNPAUSED
    assert scheduler.running == [request]
    assert request.status == RequestStatus.RUNNING


def test_keep_pause_patch_requires_resolved_output_width():
    scheduler, _, _ = _scheduled()
    scheduler.vllm_config.additional_config.clear()
    engine = _pause_guarded_engine(scheduler)

    with pytest.raises(RuntimeError, match="was not initialized"):
        EngineCore.pause_scheduler(engine, mode="keep", clear_cache=True)


def test_keep_pause_guard_covers_engine_core_proc():
    # EngineCoreProc overrides pause_scheduler, so the guard must wrap it too.
    scheduler, request, _ = _scheduled()
    engine = _pause_guarded_engine(scheduler)

    with pytest.raises(ValueError, match="live block-output request"):
        EngineCoreProc.pause_scheduler(engine, mode="keep", clear_cache=True)

    assert scheduler.running == [request]


def test_keep_pause_without_clear_cache_pauses_block_requests():
    scheduler, request, _ = _scheduled()
    engine = _pause_guarded_engine(scheduler)

    assert EngineCore.pause_scheduler(engine, mode="keep", clear_cache=False) is None

    assert scheduler.pause_state == PauseState.PAUSED_ALL
    assert scheduler.running == [request]
    assert request.status == RequestStatus.RUNNING


def test_deferred_keep_reset_returns_false_without_preempting_block_request():
    scheduler, request, _ = _scheduled()
    scheduler.set_pause_state(PauseState.PAUSED_ALL)

    assert (
        scheduler.reset_prefix_cache(
            reset_running_requests=True,
            reset_connector=False,
        )
        is False
    )
    assert scheduler.running == [request]
    assert request.status == RequestStatus.RUNNING
    assert request.async_tokens_to_discard == 0


def test_ar_prefix_cache_reset_delegates_to_upstream_preemption():
    scheduler, request, _ = _scheduled(output_width=1)

    assert scheduler.reset_prefix_cache(
        reset_running_requests=True,
        reset_connector=False,
    )
    assert scheduler.running == []
    assert request.status == RequestStatus.PREEMPTED
    assert request.async_tokens_to_discard == 1
    assert request.num_output_placeholders == 0


def test_ordinary_async_preemption_keeps_inflight_token_for_resume():
    scheduler, request, submitted = _scheduled(output_width=1, async_scheduling=True)
    scheduler.running.remove(request)
    scheduler._preempt_request(request, time.monotonic())

    assert request.num_output_placeholders == 1
    outputs = scheduler.update_from_output(submitted, _runner_output(submitted, [7]))
    assert outputs[0].outputs[0].new_token_ids == [7]
    assert list(request.output_token_ids) == [7]
    assert request.num_output_placeholders == 0

    resumed = scheduler.schedule()
    assert request.request_id in resumed.scheduled_cached_reqs.resumed_req_ids


def test_preempted_request_waits_for_inflight_output_before_resume():
    scheduler, request, submitted = _scheduled(output_width=1, async_scheduling=True)
    scheduler.running.remove(request)
    scheduler._preempt_request(request, time.monotonic())

    blocked = scheduler.schedule()
    assert blocked.total_num_scheduled_tokens == 0
    assert request.status == RequestStatus.PREEMPTED
    assert request.num_output_placeholders == 1

    older = scheduler.update_from_output(submitted, _runner_output(submitted, [7]))
    assert older[0].outputs[0].new_token_ids == [7]
    assert request.num_output_placeholders == 0

    resumed = scheduler.schedule()
    assert request.request_id in resumed.scheduled_cached_reqs.resumed_req_ids
    assert request.num_output_placeholders == 1

    following = scheduler.update_from_output(resumed, _runner_output(resumed, [8]))
    assert following[0].outputs[0].new_token_ids == [8]
    assert list(request.output_token_ids) == [7, 8]
    assert request.num_output_placeholders == 0


def test_forced_reset_discards_stale_frame_before_following_valid_frame():
    scheduler, request, submitted = _scheduled(output_width=1, async_scheduling=True)

    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    resumed = scheduler.schedule()

    assert get_tt_forced_reset_discard_counts(resumed) == {request.request_id: 1}
    stale = scheduler.update_from_output(submitted, _runner_output(submitted, [7]))
    assert stale[0].outputs == []
    assert request.async_tokens_to_discard == 0
    assert list(request.output_token_ids) == []

    valid = scheduler.update_from_output(resumed, _runner_output(resumed, [8]))
    assert valid[0].outputs[0].new_token_ids == [8]
    assert list(request.output_token_ids) == [8]
    assert request.num_output_placeholders == 0


@pytest.mark.parametrize(
    ("mutate", "width", "exc", "match"),
    [
        pytest.param(
            lambda r: setattr(r, "async_tokens_to_discard", 1),
            CANVAS,
            RuntimeError,
            "stale async output",
            id="stale-async-frame",
        ),
        pytest.param(
            None,
            CANVAS - 1,
            ValueError,
            r"15 != 16",
            id="narrow-output",
        ),
        pytest.param(
            None,
            CANVAS + 1,
            ValueError,
            r"17 != 16",
            id="wide-output",
        ),
        pytest.param(
            lambda r: setattr(r, "num_output_placeholders", CANVAS - 1),
            CANVAS,
            RuntimeError,
            "placeholders underflowed",
            id="placeholder-underflow",
        ),
    ],
)
def test_block_output_update_guards(mutate, width, exc, match):
    scheduler, request, prefill = _scheduled()
    if mutate is not None:
        mutate(request)

    with pytest.raises(exc, match=match):
        scheduler.update_from_output(
            prefill, _runner_output(prefill, list(range(width)))
        )


def test_k1_delegates_to_upstream_async_scheduler():
    scheduler, request, prefill = _scheduled(2, output_width=1)
    cache_calls = []
    scheduler.kv_cache_manager.cache_blocks = lambda *args: cache_calls.append(args)

    outputs = scheduler.update_from_output(prefill, _runner_output(prefill, [7]))

    assert outputs[0].outputs[0].new_token_ids == [7]
    assert request.num_output_placeholders == 0
    assert cache_calls


def test_diffusion_checkpoint_books_exactly_one_canvas():
    """After the platform removes the diffusion marker, upstream contributes
    one normal sampled-token placeholder and the plugin reserves only K-1 more."""
    scheduler = _scheduler(diffusion_checkpoint=True)

    assert scheduler.vllm_config.model_config.is_diffusion is False
    assert scheduler.num_sampled_tokens_per_step == 1
    assert scheduler.num_spec_tokens == 0
    assert scheduler.vllm_config.num_speculative_tokens == 0

    request = _request(CANVAS * 2)
    scheduler.add_request(request)
    prefill = scheduler.schedule()

    assert prefill.num_scheduled_tokens == {"req-0": 32}
    assert prefill.num_spec_tokens_to_schedule == 0
    assert list(request.spec_token_ids) == []
    assert request.num_output_placeholders == CANVAS
    assert request.num_computed_tokens == 32

    cache_calls = []
    scheduler.kv_cache_manager.cache_blocks = lambda *args: cache_calls.append(args)
    outputs = scheduler.update_from_output(
        prefill, _runner_output(prefill, list(range(CANVAS)))
    )
    assert outputs[0].outputs[0].new_token_ids == list(range(CANVAS))
    assert request.num_output_placeholders == 0
    assert cache_calls == []

    decode = scheduler.schedule()
    assert decode.num_scheduled_tokens == {"req-0": CANVAS}
    assert decode.num_spec_tokens_to_schedule == 0
    assert request.num_output_placeholders == CANVAS
    assert request.num_computed_tokens == 32 + CANVAS

    scheduler.update_from_output(decode, _runner_output(decode, list(range(CANVAS))))
    assert request.num_output_placeholders == 0
    assert cache_calls == []
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED


# ── Adaptive block-output (tt_adaptive_block_output) ─────────────────────────


def _adaptive_anchor(scheduler, request, token=5):
    """Drive the prefill step: adaptive prefills commit ONE anchor token."""
    submitted = scheduler.schedule()
    assert request._tt_block_step is False
    assert request.num_output_placeholders == 1
    outputs = scheduler.update_from_output(
        submitted, _runner_output(submitted, [token])
    )
    assert outputs[0].outputs[0].new_token_ids == [token]
    assert request.num_output_placeholders == 0
    return outputs


def test_adaptive_prefill_commits_single_anchor_then_solo_decode_blocks():
    """Prefill is a plain one-token step; the following solo decode reserves
    and commits the full block. On the old code the prefill itself reserved
    the block and a 1-token anchor commit was rejected."""
    scheduler = _scheduler(adaptive=True, max_num_seqs=2)
    request = _request(CANVAS * 2)
    scheduler.add_request(request)
    _adaptive_anchor(scheduler, request)

    submitted = scheduler.schedule()
    assert request._tt_block_step is True
    assert request.num_output_placeholders == CANVAS

    block = list(range(10, 10 + CANVAS))
    outputs = scheduler.update_from_output(submitted, _runner_output(submitted, block))
    assert outputs[0].outputs[0].new_token_ids == block
    assert request.num_output_placeholders == 0


def test_adaptive_batched_decode_commits_single_tokens():
    """Two decodes in one step each get ONE placeholder and commit one
    baseline token."""
    scheduler = _scheduler(adaptive=True, max_num_seqs=2)
    req_a = _request(CANVAS * 2, request_id="req-a")
    req_b = _request(CANVAS * 2, request_id="req-b")
    scheduler.add_request(req_a)
    scheduler.add_request(req_b)
    # Batched prefill step: both commit their anchors.
    submitted = scheduler.schedule()
    assert len(submitted.num_scheduled_tokens) == 2
    anchor_output = ModelRunnerOutput(
        req_ids=["req-a", "req-b"],
        req_id_to_index={"req-a": 0, "req-b": 1},
        sampled_token_ids=[[5], [6]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(submitted, anchor_output)

    # Batched DECODE step: still plain one-token baseline for both.
    submitted = scheduler.schedule()
    for req in (req_a, req_b):
        assert req._tt_block_step is False
        assert req.num_output_placeholders == 1
    decode_output = ModelRunnerOutput(
        req_ids=["req-a", "req-b"],
        req_id_to_index={"req-a": 0, "req-b": 1},
        sampled_token_ids=[[7], [9]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    outputs = scheduler.update_from_output(submitted, decode_output)
    committed = {o.request_id: o.new_token_ids for o in outputs[0].outputs}
    assert committed == {"req-a": [7], "req-b": [9]}
    assert req_a.num_output_placeholders == 0
    assert req_b.num_output_placeholders == 0


def test_adaptive_batched_blocks_every_decode_in_the_step():
    """With tt_adaptive_block_batched, a decode step with two requests reserves
    the block for BOTH and each commits a full block; the batched prefill step
    before it still commits one anchor per request."""
    scheduler = _scheduler(adaptive=True, max_num_seqs=2, batched=True)
    req_a = _request(CANVAS * 2, request_id="req-a")
    req_b = _request(CANVAS * 2, request_id="req-b")
    scheduler.add_request(req_a)
    scheduler.add_request(req_b)
    submitted = scheduler.schedule()
    assert len(submitted.num_scheduled_tokens) == 2
    for req in (req_a, req_b):
        assert req._tt_block_step is False  # prefill anchors stay width-1
        assert req.num_output_placeholders == 1
    anchor_output = ModelRunnerOutput(
        req_ids=["req-a", "req-b"],
        req_id_to_index={"req-a": 0, "req-b": 1},
        sampled_token_ids=[[5], [6]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(submitted, anchor_output)

    submitted = scheduler.schedule()
    assert len(submitted.num_scheduled_tokens) == 2
    for req in (req_a, req_b):
        assert req._tt_block_step is True
        assert req.num_output_placeholders == CANVAS
    block_a = list(range(100, 100 + CANVAS))
    block_b = list(range(200, 200 + CANVAS))
    decode_output = ModelRunnerOutput(
        req_ids=["req-a", "req-b"],
        req_id_to_index={"req-a": 0, "req-b": 1},
        sampled_token_ids=[block_a, block_b],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    outputs = scheduler.update_from_output(submitted, decode_output)
    committed = {o.request_id: o.new_token_ids for o in outputs[0].outputs}
    assert committed == {"req-a": block_a, "req-b": block_b}
    assert req_a.num_output_placeholders == 0
    assert req_b.num_output_placeholders == 0


def test_adaptive_batched_without_flag_keeps_single_token_batched_decodes():
    """The flag is opt-in: an adaptive model without it keeps the solo-only gate
    (the mechanism test_adaptive_batched_decode_commits_single_tokens pins)."""
    scheduler = _scheduler(adaptive=True, max_num_seqs=2, batched=False)
    assert scheduler._adaptive_block_batched is False


def test_adaptive_returns_to_block_width_when_solo_again():
    """After a peer finishes, the survivor's next solo decode reserves the
    block again."""
    scheduler = _scheduler(adaptive=True, max_num_seqs=2)
    req_a = _request(CANVAS * 4, request_id="req-a")
    req_b = _request(1, request_id="req-b", ignore_eos=False)
    scheduler.add_request(req_a)
    scheduler.add_request(req_b)
    submitted = scheduler.schedule()  # batched prefill anchors
    anchor_output = ModelRunnerOutput(
        req_ids=["req-a", "req-b"],
        req_id_to_index={"req-a": 0, "req-b": 1},
        sampled_token_ids=[[5], [2]],  # req-b hits max_tokens=1 and finishes
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(submitted, anchor_output)
    assert req_b.is_finished()

    resumed = scheduler.schedule()
    assert len(resumed.num_scheduled_tokens) == 1
    assert req_a._tt_block_step is True
    assert req_a.num_output_placeholders == CANVAS


def test_adaptive_commit_without_scheduling_decision_raises():
    """A committing request the placeholder pass never stamped is a broken
    invariant, not a silent block-path default."""
    scheduler = _scheduler(adaptive=True, max_num_seqs=2)
    request = _request(CANVAS * 2)
    with pytest.raises(RuntimeError, match="without a scheduling decision"):
        scheduler._update_request_with_output(request, list(range(CANVAS)))


def test_adaptive_over_frontier_prompt_never_blocks():
    """A prompt over tt_adaptive_block_max_prompt_tokens is served as plain
    baseline for its whole lifetime: width-1 reservation even on solo decode.
    On the old code the solo decode reserved the full block and the model's
    width-1 baseline output killed the engine."""
    from vllm_tt_plugin.config import store_tt_adaptive_block_max_prompt_tokens

    scheduler = _scheduler(adaptive=True, max_num_seqs=2)
    # frontier below this request's 32-token prompt
    store_tt_adaptive_block_max_prompt_tokens(scheduler.vllm_config, 16)
    scheduler._adaptive_block_max_prompt = 16
    request = _request(CANVAS * 2)
    scheduler.add_request(request)
    _adaptive_anchor(scheduler, request)

    submitted = scheduler.schedule()  # solo decode -- but over the frontier
    assert request._tt_block_step is False
    assert request.num_output_placeholders == 1
    outputs = scheduler.update_from_output(submitted, _runner_output(submitted, [9]))
    assert outputs[0].outputs[0].new_token_ids == [9]
    assert request.num_output_placeholders == 0


def test_adaptive_under_frontier_prompt_still_blocks():
    from vllm_tt_plugin.config import store_tt_adaptive_block_max_prompt_tokens

    scheduler = _scheduler(adaptive=True, max_num_seqs=2)
    store_tt_adaptive_block_max_prompt_tokens(scheduler.vllm_config, 64)
    scheduler._adaptive_block_max_prompt = 64
    request = _request(CANVAS * 2)  # 32-token prompt <= 64
    scheduler.add_request(request)
    _adaptive_anchor(scheduler, request)

    scheduler.schedule()
    assert request._tt_block_step is True
    assert request.num_output_placeholders == CANVAS


def test_block_kv_lookahead_allocates_the_block_before_the_step():
    """A speculative block model writes the whole block (plus its rejected-draft
    tail) into the paged KV inside one step; the declared lookahead must make
    allocate_slots cover that reach on the very step that writes it."""
    lookahead = CANVAS + 3
    scheduler, request, prefill = (
        _scheduler(adaptive=True, kv_lookahead=lookahead),
        None,
        None,
    )
    assert scheduler.num_lookahead_tokens == lookahead
    request = _request(CANVAS * 2)
    scheduler.add_request(request)
    prefill = scheduler.schedule()
    scheduler.update_from_output(prefill, _runner_output(prefill, [7]))
    decode = scheduler.schedule()
    assert decode.num_scheduled_tokens[request.request_id] == 1
    # Prompt (32) + anchor + one scheduled token + lookahead slots, rounded up
    # to whole blocks, are all allocated before the model runs the block step.
    blocks = scheduler.kv_cache_manager.get_block_ids(request.request_id)[0]
    need = request.num_computed_tokens + 1 + lookahead
    assert len(blocks) * BLOCK_SIZE >= need


def test_block_kv_lookahead_defaults_to_upstream():
    scheduler = _scheduler(adaptive=True)
    assert scheduler.num_lookahead_tokens == 0


# ── Ragged batched blocks (tt_adaptive_block_ragged) ─────────────────────────
#
# Contract: on a decode-only block step the model returns a rectangular int32
# [num_reqs, W] tensor whose row i holds 1 <= n_i <= W real ids followed by -1
# padding. The runner strips the padding, so the scheduler sees n_i tokens per
# request; it accepts 1..W but still consumes the whole W reservation.

PROMPT = 32  # _request's default prompt length


def _multi_runner_output(tokens_by_req: dict[str, list[int]]) -> ModelRunnerOutput:
    req_ids = list(tokens_by_req)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: idx for idx, req_id in enumerate(req_ids)},
        sampled_token_ids=[list(tokens_by_req[req_id]) for req_id in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def _batched_pair(
    *,
    ragged: bool,
    max_tokens_a: int = CANVAS * 4,
    max_tokens_b: int = CANVAS * 4,
    ignore_eos: bool = True,
    prompt_len: int = PROMPT,
    kv_lookahead: int = 0,
) -> tuple[TTScheduler, Request, Request]:
    """Two requests driven through their batched prefill anchors (5 and 6)."""
    scheduler = _scheduler(
        adaptive=True,
        max_num_seqs=2,
        batched=True,
        ragged=ragged,
        kv_lookahead=kv_lookahead,
    )
    req_a = _request(
        max_tokens_a, request_id="req-a", ignore_eos=ignore_eos, prompt_len=prompt_len
    )
    req_b = _request(
        max_tokens_b, request_id="req-b", ignore_eos=ignore_eos, prompt_len=prompt_len
    )
    scheduler.add_request(req_a)
    scheduler.add_request(req_b)
    submitted = scheduler.schedule()
    assert len(submitted.num_scheduled_tokens) == 2
    scheduler.update_from_output(
        submitted, _multi_runner_output({"req-a": [5], "req-b": [6]})
    )
    for req in (req_a, req_b):
        assert req.num_output_placeholders == 0
        assert req.num_computed_tokens == prompt_len
        assert req.num_tokens == prompt_len + 1
    return scheduler, req_a, req_b


def test_ragged_rows_commit_their_own_widths_and_keep_the_w_reservation():
    """A ragged decode step commits n_a=5 and n_b=1 (padding already stripped
    by the runner). Each request consumes its WHOLE W reservation, so the next
    schedule advances its computed tokens by exactly n (never past
    num_tokens) and stamps a fresh W-wide reservation on top."""
    scheduler, req_a, req_b = _batched_pair(ragged=True)
    assert scheduler._adaptive_block_ragged is True

    decode = scheduler.schedule()
    for req in (req_a, req_b):
        assert req._tt_block_step is True
        assert req.num_output_placeholders == CANVAS
        assert decode.num_scheduled_tokens[req.request_id] == 1  # the anchor
    block_a = list(range(100, 105))
    block_b = [200]
    outputs = scheduler.update_from_output(
        decode, _multi_runner_output({"req-a": block_a, "req-b": block_b})
    )
    committed = {o.request_id: o.new_token_ids for o in outputs[0].outputs}
    assert committed == {"req-a": block_a, "req-b": block_b}
    for req, anchor, block in ((req_a, 5, block_a), (req_b, 6, block_b)):
        assert list(req.output_token_ids) == [anchor, *block]
        assert req.num_output_placeholders == 0  # W consumed, not n
        assert req.num_computed_tokens == PROMPT + 1  # pre-step: the anchor
        assert req.num_tokens == PROMPT + 1 + len(block)
        assert req.status == RequestStatus.RUNNING

    decode = scheduler.schedule()
    assert decode.num_scheduled_tokens == {"req-a": 5, "req-b": 1}
    for req in (req_a, req_b):
        assert req.num_computed_tokens == req.num_tokens
        assert req.is_prefill_chunk is False
        assert req._tt_block_step is True
        assert req.num_output_placeholders == CANVAS

    # A full-width row next to a short one in the same step.
    block_a = list(range(300, 300 + CANVAS))
    block_b = [400, 401, 402]
    outputs = scheduler.update_from_output(
        decode, _multi_runner_output({"req-a": block_a, "req-b": block_b})
    )
    committed = {o.request_id: o.new_token_ids for o in outputs[0].outputs}
    assert committed == {"req-a": block_a, "req-b": block_b}
    assert req_a.num_output_placeholders == 0
    assert req_b.num_output_placeholders == 0
    decode = scheduler.schedule()
    assert decode.num_scheduled_tokens == {"req-a": CANVAS, "req-b": 3}
    for req in (req_a, req_b):
        assert req.num_computed_tokens == req.num_tokens
        assert req.num_output_placeholders == CANVAS


def test_ragged_solo_decode_commits_a_single_token_row():
    scheduler = _scheduler(adaptive=True, max_num_seqs=2, batched=True, ragged=True)
    request = _request(CANVAS * 2)
    scheduler.add_request(request)
    _adaptive_anchor(scheduler, request)

    decode = scheduler.schedule()
    assert request._tt_block_step is True
    assert request.num_output_placeholders == CANVAS
    outputs = scheduler.update_from_output(decode, _runner_output(decode, [9]))
    assert outputs[0].outputs[0].new_token_ids == [9]
    assert request.num_output_placeholders == 0

    decode = scheduler.schedule()
    assert decode.num_scheduled_tokens == {"req-0": 1}
    assert request.num_computed_tokens == request.num_tokens
    assert request.num_output_placeholders == CANVAS


@pytest.mark.parametrize(
    ("row", "match"),
    [
        pytest.param([], r"0 not in 1\.\.16", id="empty-row"),
        pytest.param(list(range(CANVAS + 1)), r"17 not in 1\.\.16", id="over-wide"),
        pytest.param([7, -1, -1], "with padding", id="pad-reached-scheduler"),
    ],
)
def test_ragged_width_guards(row, match):
    scheduler, req_a, _ = _batched_pair(ragged=True)
    scheduler.schedule()
    assert req_a._tt_block_step is True
    with pytest.raises(ValueError, match=match):
        # Direct call: upstream update_from_output never forwards an empty row.
        scheduler._update_request_with_output(req_a, list(row))


def test_batched_without_ragged_keeps_the_fixed_width_check():
    """The flag is opt-in: a batched model without it still rejects a short
    row exactly as before (the old contract fills every row to W)."""
    scheduler, _, _ = _batched_pair(ragged=False)
    assert scheduler._adaptive_block_ragged is False
    decode = scheduler.schedule()
    with pytest.raises(ValueError, match=r"5 != 16"):
        scheduler.update_from_output(
            decode,
            _multi_runner_output(
                {"req-a": list(range(5)), "req-b": list(range(CANVAS))}
            ),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"adaptive": False}, id="no-adaptive"),
        pytest.param({"adaptive": True, "batched": False}, id="no-batched"),
    ],
)
def test_ragged_requires_the_batched_contract(kwargs):
    with pytest.raises(ValueError, match="requires tt_adaptive_block_batched"):
        _scheduler(max_num_seqs=2, ragged=True, **kwargs)


@pytest.mark.parametrize(
    ("max_tokens", "ignore_eos", "row", "kept", "status"),
    [
        # anchor + 3 of the 5-token row reach max_tokens=4
        (
            4,
            True,
            [10, 11, 12, 13, 14],
            [10, 11, 12],
            RequestStatus.FINISHED_LENGTH_CAPPED,
        ),
        (CANVAS, False, [10, 2, 12], [10, 2], RequestStatus.FINISHED_STOPPED),
    ],
    ids=["max_tokens", "eos"],
)
def test_ragged_row_stops_trim_and_consume_the_reservation(
    max_tokens, ignore_eos, row, kept, status
):
    """Stops inside a ragged row are vLLM's as today: the row is trimmed at the
    stop, the finished request still consumes its whole W, and the survivor
    carries on alone as a solo block step scheduling exactly its n tokens."""
    scheduler, req_a, req_b = _batched_pair(
        ragged=True, max_tokens_a=max_tokens, ignore_eos=ignore_eos
    )
    decode = scheduler.schedule()
    outputs = scheduler.update_from_output(
        decode, _multi_runner_output({"req-a": row, "req-b": [200, 201]})
    )
    committed = {o.request_id: o.new_token_ids for o in outputs[0].outputs}
    assert committed == {"req-a": kept, "req-b": [200, 201]}
    assert req_a.status == status
    assert req_a.num_output_placeholders == 0
    assert req_b.status == RequestStatus.RUNNING
    assert req_b.num_output_placeholders == 0

    decode = scheduler.schedule()
    assert decode.num_scheduled_tokens == {"req-b": 2}
    assert req_b._tt_block_step is True
    assert req_b.num_output_placeholders == CANVAS
    assert req_b.num_computed_tokens == req_b.num_tokens


def test_ragged_block_kv_lookahead_covers_the_next_block_after_a_short_row():
    """After a short row lands, the next schedule allocates n new slots plus
    the declared lookahead: the model's next block (W) and rejected-draft tail
    are covered from the tokens it actually has, exactly as with full rows."""
    lookahead = CANVAS + 16
    prompt_len = 92
    scheduler, req_a, req_b = _batched_pair(
        ragged=True, prompt_len=prompt_len, kv_lookahead=lookahead
    )
    assert scheduler.num_lookahead_tokens == lookahead
    decode = scheduler.schedule()
    scheduler.update_from_output(
        decode, _multi_runner_output({"req-a": [100, 101], "req-b": [200]})
    )

    decode = scheduler.schedule()
    assert decode.num_scheduled_tokens == {"req-a": 2, "req-b": 1}
    for req in (req_a, req_b):
        assert req.num_computed_tokens == req.num_tokens
        blocks = scheduler.kv_cache_manager.get_block_ids(req.request_id)[0]
        # Upstream allocates num_computed(pre) + n + lookahead == num_tokens
        # + lookahead slots: everything the model holds, plus its whole next
        # block and tail.
        need = req.num_tokens + lookahead
        assert len(blocks) * BLOCK_SIZE >= need
        assert len(blocks) == -(-need // BLOCK_SIZE)
