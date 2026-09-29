# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Block-output chunked prefill (``tt_block_output_chunked_prefill``).

A batched ragged block-output model (the Qwen3.6 DFlash2 class under
QWEN36_DFLASH_CHUNKED_PREFILL=1) that resumes split prompts: the platform keeps
the TT chunk policy for it, the scheduler interleaves 0-token intermediate
chunks with the other requests' block steps, the runner emits nothing for an
intermediate row and keeps a continuation on its state slot. These tests drive
the real platform hook, a real ``TTScheduler`` and the runner's slot/output
functions; the only fakes are the model class and the runner's token rows.
"""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

# vLLM's own bootstrap resolves the platform plugin, which imports this module.
import vllm  # noqa: F401
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from vllm_tt_plugin.config import (
    get_tt_prefill_chunk_extras,
    get_tt_prefill_chunk_policy,
    is_tt_block_output_chunked_prefill,
)
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.platform import (
    _apply_chunked_prefill_policy,
    _finalize_tt_prefill_chunk_policy,
)
from vllm_tt_plugin.scheduler import TTScheduler

BLOCK = 16
CHUNK = 64
W = 4
LOOKAHEAD = 8
MAX_MODEL_LEN = 1024
LOCAL_MODEL_CONFIG = Path(__file__).parent / "model_configs" / "qwen2"

# The DFlash2 class's declaration under its knob, scaled down.
BLOCK_CHUNK_CAPS = {
    "supports_chunked_prefill": True,
    "tt_prefill_chunk_tokens": CHUNK,
    "tt_block_output_chunked_prefill": True,
    "output_tokens_per_step": W,
    "tt_adaptive_block_output": True,
    "tt_adaptive_block_batched": True,
    "tt_adaptive_block_ragged": True,
    "tt_adaptive_block_max_prompt_tokens": 0,
    "tt_block_output_kv_lookahead_tokens": LOOKAHEAD,
    "supports_sample_on_device": True,
}


# ------------------------------------------------------------------ platform policy


class _FakeModel:
    """Stand-in for the resolved TT model class; the policy only reads its name."""


def _policy_config(*, tt=None, async_scheduling=False, max_num_seqs=8):
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=True,
            max_num_batched_tokens=262144,
            long_prefill_token_threshold=0,
            disable_chunked_mm_input=False,
            max_num_seqs=max_num_seqs,
            async_scheduling=async_scheduling,
        ),
        model_config=SimpleNamespace(max_model_len=262144),
        cache_config=SimpleNamespace(block_size=64),
        additional_config={"tt": dict(tt or {})},
    )


def _dflash_caps(**overrides):
    caps = dict(
        BLOCK_CHUNK_CAPS, tt_prefill_chunk_tokens=2048, output_tokens_per_step=32
    )
    caps.update(overrides)
    return {k: v for k, v in caps.items() if v is not None}


def test_block_output_model_without_the_key_keeps_todays_refusal():
    config = _policy_config()
    caps = _dflash_caps(tt_block_output_chunked_prefill=None)
    _apply_chunked_prefill_policy(config, caps, _FakeModel)
    assert config.scheduler_config.enable_chunked_prefill is False
    assert get_tt_prefill_chunk_policy(config) is None
    assert is_tt_block_output_chunked_prefill(config) is False


def test_the_key_with_its_prerequisites_resolves_the_policy():
    config = _policy_config()
    _apply_chunked_prefill_policy(config, _dflash_caps(), _FakeModel)
    sc = config.scheduler_config
    assert sc.enable_chunked_prefill is True
    assert get_tt_prefill_chunk_policy(config) == (2048, 4)
    assert sc.long_prefill_token_threshold == 2048
    assert sc.max_num_batched_tokens == 262144 + 8 * 2048 == 278528
    assert is_tt_block_output_chunked_prefill(config) is True
    extras = get_tt_prefill_chunk_extras(config)
    assert extras["rider_tokens"] == 512
    assert extras["max_riders"] == 1
    assert extras["cadence_after_final"] is True
    assert extras["burst_longs"] == 2
    assert extras["min_tokens"] == 8192
    assert extras["chunk_without_decoders"] is True
    assert extras["oversized_rider_step"] is True


@pytest.mark.parametrize(
    "missing",
    [
        "supports_chunked_prefill",
        "tt_prefill_chunk_tokens",
        "tt_adaptive_block_output",
        "tt_adaptive_block_batched",
    ],
)
def test_the_key_without_a_prerequisite_raises(missing):
    caps = _dflash_caps(**{missing: None})
    with pytest.raises(ValueError, match="tt_block_output_chunked_prefill without"):
        _apply_chunked_prefill_policy(_policy_config(), caps, _FakeModel)


def test_the_key_on_a_width_one_model_raises():
    caps = _dflash_caps(output_tokens_per_step=1)
    with pytest.raises(ValueError, match="output_tokens_per_step > 1"):
        _apply_chunked_prefill_policy(_policy_config(), caps, _FakeModel)


@pytest.mark.parametrize(
    "tt, match",
    [
        ({"chunked_prefill_rider_tokens": -1}, "rider_tokens"),
        ({"chunked_prefill_rider_tokens": True}, "rider_tokens"),
        ({"chunked_prefill_cadence_after_final": 1}, "cadence_after_final"),
        ({"chunked_prefill_burst_longs": 1}, "burst_longs"),
        ({"chunked_prefill_max_riders": 0}, "max_riders"),
        ({"chunked_prefill_chunk_without_decoders": 1}, "chunk_without_decoders"),
        ({"chunked_prefill_oversized_rider_step": "yes"}, "oversized_rider_step"),
        ({"chunked_prefill_min_tokens": -1}, "min_tokens"),
        ({"chunked_prefill_min_tokens": 8192.0}, "min_tokens"),
    ],
)
def test_policy_extra_knobs_are_validated(tt, match):
    with pytest.raises(ValueError, match=match):
        _apply_chunked_prefill_policy(_policy_config(tt=tt), _dflash_caps(), _FakeModel)


def test_plain_models_keep_the_plain_policy_extras():
    config = _policy_config()
    _apply_chunked_prefill_policy(
        config,
        {"supports_chunked_prefill": True, "tt_prefill_chunk_tokens": 2048},
        _FakeModel,
    )
    assert get_tt_prefill_chunk_extras(config) == {
        "block_output": False,
        "rider_tokens": None,
        "max_riders": None,
        "cadence_after_final": False,
        "burst_longs": 0,
        "min_tokens": 0,
        "chunk_without_decoders": False,
        "oversized_rider_step": False,
    }


@pytest.mark.parametrize("why", ["async", "kv_transfer", "lane"])
def test_turning_the_policy_off_clears_the_block_output_flag(why):
    config = _policy_config()
    _apply_chunked_prefill_policy(config, _dflash_caps(), _FakeModel)
    assert is_tt_block_output_chunked_prefill(config) is True
    if why == "async":
        config.scheduler_config.async_scheduling = True
    elif why == "kv_transfer":
        config.kv_transfer_config = object()
    _finalize_tt_prefill_chunk_policy(config, is_lane_mode=why == "lane")
    assert get_tt_prefill_chunk_policy(config) is None
    assert is_tt_block_output_chunked_prefill(config) is False
    assert config.scheduler_config.enable_chunked_prefill is False


# ------------------------------------------------------------------ scheduler


class _BlockChunkModel:
    model_capabilities = BLOCK_CHUNK_CAPS

    def release_request(self, row):  # block-output lifecycle hooks (platform check)
        pass

    def release_persistent_capture(self):
        pass


@contextmanager
def _model_resolution(model_cls):
    with (
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
            return_value=(model_cls, None),
        ),
    ):
        yield


def _scheduler(*, max_num_seqs=4, decode_steps=2, num_blocks=None, **tt_extra):
    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    model_config.max_model_len = MAX_MODEL_LEN
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=MAX_MODEL_LEN,
        max_model_len=MAX_MODEL_LEN,
        enable_chunked_prefill=True,
        async_scheduling=False,
        is_encoder_decoder=False,
    )
    cache_config = CacheConfig(
        block_size=BLOCK,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    tt = {
        "chunked_prefill_decode_steps": decode_steps,
        "sample_on_device_mode": "decode_only",
        # The tests below that predate these two extras run with them off
        # (every prompt above the chunk is long, no burst fallback); the tests
        # of the extras set them explicitly.
        "chunked_prefill_min_tokens": 0,
        "chunked_prefill_burst_longs": 0,
        **tt_extra,
    }
    with _model_resolution(_BlockChunkModel):
        config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
            additional_config={"tt": tt},
        )
    if num_blocks is None:
        num_blocks = max_num_seqs * MAX_MODEL_LEN // BLOCK + 1
    cache_config.num_gpu_blocks = num_blocks
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
            )
        ],
    )
    return TTScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )


def _request(req_id, prompt_len, max_tokens=60):
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=2)
    return Request(
        request_id=req_id,
        prompt_token_ids=[7] * prompt_len,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


class _Trace:
    def __init__(self):
        self.steps = []  # (kind, {req_id: (start, end)}, {req_id: placeholders})

    def kinds(self):
        return [k for k, _, _ in self.steps]


def _step(s, trace=None, ragged_n=3):
    """One engine step with a fake ragged block-output runner: a prefill row
    emits [] (intermediate chunk) or one anchor token; a decode (block) row
    emits ``ragged_n`` real ids (1..W)."""
    starts = {r: q.num_computed_tokens for r, q in s.requests.items()}
    out = s.schedule()
    spans = {
        r: (starts.get(r, 0), starts.get(r, 0) + n)
        for r, n in out.num_scheduled_tokens.items()
    }
    is_prefill = any(spans[r][0] < s.requests[r].num_prompt_tokens for r in spans)
    decodes = [r for r in spans if spans[r][0] >= s.requests[r].num_prompt_tokens]
    if is_prefill and decodes:
        raise AssertionError(f"mixed prefill/decode step: {spans}")
    kind = "idle" if not spans else ("prefill" if is_prefill else "decode")
    placeholders = {r: s.requests[r].num_output_placeholders for r in spans}
    if trace is not None:
        trace.steps.append((kind, spans, placeholders))
    req_ids = list(out.num_scheduled_tokens)
    rows = []
    for r in req_ids:
        req = s.requests[r]
        if kind == "prefill":
            rows.append([] if req.is_prefill_chunk else [5])
        else:
            assert req._tt_block_step is True, f"{r} decoded without a block step"
            rows.append([5] * ragged_n)
    s.update_from_output(
        out,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=rows,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    return kind, spans


def _start_decoders(s, n, prompt_len=16, max_tokens=400):
    for i in range(n):
        s.add_request(_request(f"d{i}", prompt_len, max_tokens))
    _step(s)  # one prefill step admits them all
    _step(s)  # their first block step


def _run_until(s, cond, trace, limit=600):
    for _ in range(limit):
        if cond():
            return
        _step(s, trace)
    raise AssertionError("condition not reached")


def test_block_output_scheduler_resolves_the_policy_and_the_lookahead():
    s = _scheduler(decode_steps=3)
    assert s._chunk_policy == (CHUNK, 3)
    assert s._cp_extras["block_output"] is True
    assert s._adaptive_block_ragged and s._adaptive_block_batched
    assert s.num_lookahead_tokens >= LOOKAHEAD


def test_intermediate_chunks_commit_nothing_and_decoders_keep_their_blocks():
    """0-token intermediate chunks between the decoders' ragged block steps:
    no placeholders or block stamp on the partial, W reserved for the decoders
    only, the final chunk is a width-1 anchor, and the joined request's next
    decode step is a block step with the others. The mixing check never fires
    (``_step`` would raise)."""
    s = _scheduler(decode_steps=2)
    _start_decoders(s, 2)
    s.add_request(_request("L", 3 * CHUNK + 17))
    trace = _Trace()
    _run_until(s, lambda: s.requests["L"].num_output_tokens >= 2, trace)
    l_steps = [(k, sp, ph) for k, sp, ph in trace.steps if "L" in sp]
    assert [sp["L"] for k, sp, ph in l_steps if k == "prefill"] == [
        (0, CHUNK),
        (CHUNK, 2 * CHUNK),
        (2 * CHUNK, 3 * CHUNK),
        (3 * CHUNK, 3 * CHUNK + 17),
    ]
    for k, sp, ph in l_steps:
        if k == "prefill":
            # an intermediate chunk reserves nothing; the final chunk only the
            # upstream one-token anchor placeholder (never a W-wide block)
            final = sp["L"][1] == 3 * CHUNK + 17
            assert ph["L"] == (1 if final else 0), (sp, ph)
    for k, sp, ph in trace.steps:
        if k == "decode":
            assert all(v == W for v in ph.values()), ph  # the W-wide block reservation
    # the anchor (one token) then block tokens
    assert s.requests["L"].num_output_tokens >= 1 + 3
    idx = [
        i for i, (k, sp, _) in enumerate(trace.steps) if "L" in sp and k == "prefill"
    ]
    assert all(b - a == 3 for a, b in zip(idx, idx[1:])), trace.kinds()
    first_decode = next(i for i, (k, sp, _) in enumerate(trace.steps) if i > idx[-1])
    assert trace.steps[first_decode][0] == "decode"
    assert "L" in trace.steps[first_decode][1]
    assert {"d0", "d1"} <= set(trace.steps[first_decode][1])


def test_the_partial_is_never_a_decode_row():
    s = _scheduler(decode_steps=1)
    _start_decoders(s, 3)
    s.add_request(_request("L", 5 * CHUNK))
    trace = _Trace()
    _run_until(s, lambda: not s.requests["L"].is_prefill_chunk, trace)
    for k, sp, _ in trace.steps:
        if k == "decode":
            assert "L" not in sp


def test_riders_beyond_the_budget_wait_for_the_next_chunk_step():
    """Perf critique M2: at most ``chunked_prefill_rider_tokens`` of short
    prompts share a chunk step (the oldest always may)."""
    s = _scheduler(max_num_seqs=8, decode_steps=1, chunked_prefill_rider_tokens=40)
    _start_decoders(s, 1)
    s.add_request(_request("L", 4 * CHUNK))
    trace = _Trace()
    _step(s, trace)  # chunk 0
    for i in range(3):
        s.add_request(_request(f"s{i}", 30))
    _run_until(
        s, lambda: all(s.requests[f"s{i}"].num_computed_tokens for i in range(3)), trace
    )
    chunk_steps = [sp for k, sp, _ in trace.steps if k == "prefill" and "L" in sp]
    riders = [[r for r in sp if r.startswith("s")] for sp in chunk_steps]
    assert all(len(x) <= 1 for x in riders), riders  # 30 + 30 > 40
    assert sum(riders, []) == ["s0", "s1", "s2"]


def test_riders_beyond_the_count_cap_wait_for_the_next_chunk_step():
    """Short riders cost a nearly fixed masked-bucket prefill each: the count cap
    (block-output default 1) bounds the chunk step where the token budget
    would let many tiny riders through."""
    s = _scheduler(max_num_seqs=8, decode_steps=1, chunked_prefill_rider_tokens=512)
    assert s._cp_extras["max_riders"] == 1
    _start_decoders(s, 1)
    s.add_request(_request("L", 4 * CHUNK))
    trace = _Trace()
    _step(s, trace)  # chunk 0
    for i in range(3):
        s.add_request(_request(f"s{i}", 10))
    _run_until(
        s, lambda: all(s.requests[f"s{i}"].num_computed_tokens for i in range(3)), trace
    )
    chunk_steps = [sp for k, sp, _ in trace.steps if k == "prefill" and "L" in sp]
    riders = [[r for r in sp if r.startswith("s")] for sp in chunk_steps]
    assert all(len(x) <= 1 for x in riders), riders
    assert sum(riders, []) == ["s0", "s1", "s2"]


def test_no_count_cap_lets_small_riders_share_one_chunk_step():
    s = _scheduler(
        max_num_seqs=8,
        decode_steps=1,
        chunked_prefill_rider_tokens=512,
        chunked_prefill_max_riders=None,
    )
    _start_decoders(s, 1)
    s.add_request(_request("L", 4 * CHUNK))
    _step(s)
    for i in range(3):
        s.add_request(_request(f"s{i}", 10))
    trace = _Trace()
    _run_until(s, lambda: s.requests["s0"].num_computed_tokens > 0, trace)
    assert set(trace.steps[-1][1]) == {"L", "s0", "s1", "s2"}


def test_without_the_hard_budget_the_oldest_rider_rides_above_it():
    s = _scheduler(
        max_num_seqs=8,
        decode_steps=1,
        chunked_prefill_rider_tokens=0,
        chunked_prefill_oversized_rider_step=False,
    )
    _start_decoders(s, 1)
    s.add_request(_request("L", 4 * CHUNK))
    _step(s)
    s.add_request(_request("s0", CHUNK))
    trace = _Trace()
    _run_until(s, lambda: s.requests["s0"].num_computed_tokens > 0, trace)
    assert set(trace.steps[-1][1]) == {"L", "s0"}


def test_an_oversized_rider_gets_its_own_step_counted_by_the_cadence():
    """Review F5: the rider budget is hard. A short above it never shares a
    chunk step (chunk + rider would exceed one chunk's stall); it takes a
    prefill step alone, the partial held, and the cadence then separates it
    from the next chunk step."""
    s = _scheduler(max_num_seqs=8, decode_steps=2, chunked_prefill_rider_tokens=40)
    _start_decoders(s, 1)
    s.add_request(_request("L", 4 * CHUNK))
    trace = _Trace()
    _step(s, trace)  # chunk 0
    s.add_request(_request("big", 60))  # above the budget, still short (<= CHUNK)
    s.add_request(_request("small", 20))
    _run_until(s, lambda: not s.requests["L"].is_prefill_chunk, trace)
    prefills = [(i, sp) for i, (k, sp, _) in enumerate(trace.steps) if k == "prefill"]
    assert any(set(sp) == {"big"} for _, sp in prefills), prefills
    assert not any("big" in sp and "L" in sp for _, sp in prefills), prefills
    small = [sp for _, sp in prefills if "small" in sp]
    assert small and "L" in small[0], prefills  # a small rider still rides
    idx = [i for i, _ in prefills]
    assert all(b - a >= 3 for a, b in zip(idx, idx[1:])), trace.kinds()


def test_oversized_riders_alternate_with_chunks_and_never_starve_the_partial():
    s = _scheduler(max_num_seqs=8, decode_steps=1, chunked_prefill_rider_tokens=10)
    _start_decoders(s, 1)
    s.add_request(_request("L", 4 * CHUNK))
    trace = _Trace()
    _step(s, trace)  # chunk 0
    for i in range(4):
        s.add_request(_request(f"b{i}", 50))
    _run_until(s, lambda: not s.requests["L"].is_prefill_chunk, trace)
    seq = ["L" if "L" in sp else "b" for k, sp, _ in trace.steps if k == "prefill"]
    assert "bb" not in "".join(seq), seq
    assert seq.count("L") == 4, seq


def test_an_oversized_rider_goes_before_a_new_long_prompts_first_chunk():
    s = _scheduler(max_num_seqs=8, decode_steps=2, chunked_prefill_rider_tokens=10)
    _start_decoders(s, 1)
    s.add_request(_request("big", 50))
    s.add_request(_request("L", 3 * CHUNK))
    trace = _Trace()
    _run_until(s, lambda: s.requests["L"].num_computed_tokens > 0, trace)
    prefills = [(i, sp) for i, (k, sp, _) in enumerate(trace.steps) if k == "prefill"]
    assert [set(sp) for _, sp in prefills] == [{"big"}, {"L"}], prefills
    assert prefills[1][0] - prefills[0][0] >= 3, trace.kinds()


def test_short_prompts_alone_keep_todays_admission():
    s = _scheduler(max_num_seqs=8, decode_steps=2, chunked_prefill_rider_tokens=0)
    _start_decoders(s, 1)
    for i in range(3):
        s.add_request(_request(f"s{i}", 30))
    kind, spans = _step(s)
    assert kind == "prefill" and set(spans) == {"s0", "s1", "s2"}


def test_no_back_to_back_chunk_steps_after_a_final_chunk():
    """Lossless critique m3: the next long prompt's first chunk waits the
    cadence after the previous partial's final chunk."""
    s = _scheduler(max_num_seqs=8, decode_steps=2)
    _start_decoders(s, 1)
    s.add_request(_request("L1", 2 * CHUNK + 5))
    s.add_request(_request("L2", 2 * CHUNK + 5))
    trace = _Trace()
    _run_until(s, lambda: s.requests["L2"].num_computed_tokens >= 2 * CHUNK + 5, trace)
    kinds = trace.kinds()
    prefill_idx = [i for i, k in enumerate(kinds) if k == "prefill"]
    assert all(b - a >= 3 for a, b in zip(prefill_idx, prefill_idx[1:])), kinds


def test_without_the_extra_a_new_long_prompt_may_follow_a_final_chunk():
    s = _scheduler(
        max_num_seqs=8, decode_steps=2, chunked_prefill_cadence_after_final=False
    )
    _start_decoders(s, 1)
    s.add_request(_request("L1", 2 * CHUNK + 5))
    s.add_request(_request("L2", 2 * CHUNK + 5))
    trace = _Trace()
    _run_until(s, lambda: s.requests["L2"].num_computed_tokens >= 2 * CHUNK + 5, trace)
    kinds = trace.kinds()
    prefill_idx = [i for i, k in enumerate(kinds) if k == "prefill"]
    assert any(b - a == 1 for a, b in zip(prefill_idx, prefill_idx[1:])), kinds


def test_a_partial_keeps_one_chunk_per_step_when_the_last_decoder_leaves():
    """Review F1: when the decoder count reaches 0 mid-partial, the remainder
    must not run as one call (a request arriving right after would wait for
    all of it): the partial advances one chunk per step, back to back, and a
    newcomer rides the next chunk step."""
    s = _scheduler(max_num_seqs=8, decode_steps=2)
    _start_decoders(s, 1, max_tokens=12)
    s.add_request(_request("L", 8 * CHUNK + 5))
    trace = _Trace()
    _run_until(s, lambda: "d0" not in s.requests, trace)
    assert s.requests["L"].is_prefill_chunk
    s.add_request(_request("late", 20))
    _run_until(s, lambda: s.requests["L"].num_computed_tokens >= 8 * CHUNK + 5, trace)
    spans = [sp["L"] for k, sp, _ in trace.steps if k == "prefill" and "L" in sp]
    assert all(b - a <= CHUNK for a, b in spans), spans
    late = next(sp for k, sp, _ in trace.steps if "late" in sp)
    assert "L" in late and late["L"][1] < 8 * CHUNK + 5, late


def test_without_the_extra_the_remainder_runs_whole_when_nobody_decodes():
    """Negative control for the test above: lane C's plain behaviour."""
    s = _scheduler(
        max_num_seqs=8, decode_steps=2, chunked_prefill_chunk_without_decoders=False
    )
    _start_decoders(s, 1, max_tokens=12)
    s.add_request(_request("L", 8 * CHUNK + 5))
    trace = _Trace()
    _run_until(s, lambda: s.requests["L"].num_computed_tokens >= 8 * CHUNK + 5, trace)
    spans = [sp["L"] for k, sp, _ in trace.steps if k == "prefill" and "L" in sp]
    assert len(spans) >= 2 and spans[-1][1] - spans[-1][0] > CHUNK, spans


def test_a_lone_long_prompt_with_nobody_decoding_is_still_one_call():
    s = _scheduler(max_num_seqs=8, decode_steps=2)
    s.add_request(_request("L", 8 * CHUNK + 5))
    kind, spans = _step(s)
    assert kind == "prefill" and spans == {"L": (0, 8 * CHUNK + 5)}


def test_a_small_kv_pool_warns_at_boot():
    import vllm_tt_plugin.scheduler as sched_mod

    with patch.object(sched_mod.logger, "warning") as warn:
        _scheduler(max_num_seqs=4, num_blocks=4 * MAX_MODEL_LEN // BLOCK - 1)
    assert warn.call_count == 1 and "preempt" in warn.call_args[0][0]
    with patch.object(sched_mod.logger, "warning") as warn:
        _scheduler(max_num_seqs=4)
    assert warn.call_count == 0


def test_a_burst_of_long_prompts_falls_back_to_prefill_first():
    """A burst is prefilled as without the policy: every waiting prompt whole in
    one step (not one long prompt per step, which would hand each its first
    token early and then stall it behind the others: a different TTFT/TPOT
    split than the unchunked server's)."""
    s = _scheduler(max_num_seqs=8, decode_steps=2, chunked_prefill_burst_longs=2)
    _start_decoders(s, 1)
    for i in range(3):
        s.add_request(_request(f"L{i}", 3 * CHUNK + 5))
    kind, spans = _step(s)
    assert kind == "prefill"
    assert spans == {f"L{i}": (0, 3 * CHUNK + 5) for i in range(3)}
    s.add_request(_request("L3", 3 * CHUNK + 5))
    kind, spans = _step(s)  # 1 pending: no burst, chunk again
    assert kind == "prefill" and spans == {"L3": (0, CHUNK)}


def test_a_burst_admits_only_what_fits_the_token_budget_and_never_splits():
    s = _scheduler(max_num_seqs=8, decode_steps=2, chunked_prefill_burst_longs=2)
    # the unchunked server's budget (max_model_len): two fit, the third does not
    n = MAX_MODEL_LEN // 3 + 1
    _start_decoders(s, 1)
    for i in range(3):
        s.add_request(_request(f"L{i}", n))
    kind, spans = _step(s)
    assert kind == "prefill" and spans == {"L0": (0, n), "L1": (0, n)}
    kind, spans = _step(s)  # L2 is the only long prompt left: no burst, chunked
    assert kind == "prefill" and spans == {"L2": (0, CHUNK)}, spans


def test_one_long_prompt_with_decoders_is_still_chunked_under_the_burst_extra():
    s = _scheduler(max_num_seqs=8, decode_steps=2, chunked_prefill_burst_longs=2)
    _start_decoders(s, 1)
    s.add_request(_request("L1", 3 * CHUNK + 5))
    kind, spans = _step(s)
    assert spans == {"L1": (0, CHUNK)}


# ------------------------------------------- E2 policy: min_tokens, burst, no decoders

MIN = 4 * CHUNK  # scaled chunked_prefill_min_tokens (8192 / 2048 on the server)


def _e2_scheduler(**kw):
    """The block-output defaults of the E2 policy, scaled: min_tokens = 4
    chunks, burst fallback at 2 long prompts."""
    kw.setdefault("max_num_seqs", 8)
    kw.setdefault("decode_steps", 2)
    return _scheduler(
        chunked_prefill_min_tokens=MIN, chunked_prefill_burst_longs=2, **kw
    )


def test_with_nobody_decoding_every_waiting_prompt_runs_whole_in_one_step():
    """Nothing to protect: the step is the unchunked server's step (all waiting
    prompts, whole), not one long prompt per step."""
    s = _e2_scheduler()
    for i in range(2):
        s.add_request(_request(f"L{i}", 6 * CHUNK))
    s.add_request(_request("M", 2 * CHUNK + 3))
    s.add_request(_request("s", 10))
    kind, spans = _step(s)
    assert kind == "prefill"
    assert spans == {
        "L0": (0, 6 * CHUNK),
        "L1": (0, 6 * CHUNK),
        "M": (0, 2 * CHUNK + 3),
        "s": (0, 10),
    }


def test_with_nobody_decoding_and_no_burst_extra_one_long_prompt_per_step():
    """Negative control of the test above: burst_longs 0 and min_tokens 0 keep
    lane C's rule."""
    s = _scheduler(max_num_seqs=8, decode_steps=2)
    for i in range(2):
        s.add_request(_request(f"L{i}", 6 * CHUNK))
    kind, spans = _step(s)
    assert kind == "prefill" and spans == {"L0": (0, 6 * CHUNK)}


def test_with_nobody_decoding_min_tokens_alone_also_admits_everything():
    s = _scheduler(max_num_seqs=8, decode_steps=2, chunked_prefill_min_tokens=MIN)
    s.add_request(_request("M1", 3 * CHUNK))
    s.add_request(_request("M2", 3 * CHUNK))
    kind, spans = _step(s)
    assert kind == "prefill"
    assert spans == {"M1": (0, 3 * CHUNK), "M2": (0, 3 * CHUNK)}


def test_an_older_long_prompt_is_not_starved_by_a_stream_of_medium_prompts():
    """Review E2-1: a medium pass hides the long prompts, so it may run only
    when the medium is older than every waiting long prompt; otherwise the
    long prompt's chunk step comes first (the medium gets its own step)."""
    s = _e2_scheduler(max_num_seqs=4, decode_steps=1)
    _start_decoders(s, 3, max_tokens=30)
    s.add_request(_request("L", 8 * CHUNK))
    trace = _Trace()
    for i in range(60):
        if i % 3 == 0:
            s.add_request(_request(f"M{i}", 3 * CHUNK, max_tokens=30))
        _step(s, trace)
    assert any("L" in sp for k, sp, _ in trace.steps if k == "prefill"), trace.kinds()


def test_a_medium_rider_that_does_not_fit_does_not_stall_the_partial():
    """Review E2-2: a medium's own step that schedules nothing (its KV blocks do
    not fit, or no seat) must not take the step from the partial's chunk."""
    s = _e2_scheduler(max_num_seqs=3, decode_steps=1)
    _start_decoders(s, 2, max_tokens=500)
    s.add_request(_request("L", 8 * CHUNK))
    _step(s)
    assert s.requests["L"].is_prefill_chunk
    s.add_request(_request("M", 3 * CHUNK))  # no seat: 2 decoders + the partial
    trace = _Trace()
    for _ in range(20):
        _step(s, trace)
    l_chunks = [sp["L"] for k, sp, _ in trace.steps if k == "prefill" and "L" in sp]
    assert len(l_chunks) >= 4, trace.kinds()


def _admission_trace(s, arrivals, steps):
    out = []
    for t in range(steps):
        for rid, n, mt in arrivals.get(t, ()):
            s.add_request(_request(rid, n, mt))
        kind, spans = _step(s)
        out.append((kind, spans))
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_prompts_up_to_min_tokens_schedule_exactly_like_the_unchunked_policy(seed):
    """Differential: an arrival trace whose prompts never exceed min_tokens,
    with and without decoders, gives the same steps under the E2 policy as
    under the unchunked TT default policy (``_chunk_policy`` None)."""
    rng = np.random.default_rng(seed)
    arrivals = {}
    for i in range(24):
        t = int(rng.integers(0, 120))
        n = int(rng.choice([10, CHUNK, CHUNK + 1, 3 * CHUNK, MIN]))
        arrivals.setdefault(t, []).append((f"r{i}", n, int(rng.integers(4, 60))))
    e2 = _e2_scheduler(max_num_seqs=4, decode_steps=3)
    # the unchunked server: no chunk policy, no chunked prefill, budget
    # max_model_len
    plain = _e2_scheduler(max_num_seqs=4, decode_steps=3)
    plain._chunk_policy = None
    plain.scheduler_config.long_prefill_token_threshold = 0
    plain.scheduler_config.enable_chunked_prefill = False
    plain.max_num_scheduled_tokens = MAX_MODEL_LEN
    assert _admission_trace(e2, arrivals, 200) == _admission_trace(plain, arrivals, 200)


def test_a_prompt_above_min_tokens_is_chunked_while_others_decode():
    s = _e2_scheduler()
    _start_decoders(s, 1)
    s.add_request(_request("L", MIN + 1))
    kind, spans = _step(s)
    assert kind == "prefill" and spans == {"L": (0, CHUNK)}


def test_a_medium_prompt_runs_whole_and_the_cadence_follows_it():
    """min_tokens: a prompt of more than one chunk but at most min_tokens runs
    whole next to decoders (the old policy split it), and the waiting long
    prompt's first chunk waits the cadence after it, like after an oversized
    rider's step."""
    s = _e2_scheduler(decode_steps=2)
    _start_decoders(s, 1)
    s.add_request(_request("M", MIN))
    s.add_request(_request("L", MIN + 2 * CHUNK))
    trace = _Trace()
    _step(s, trace)
    assert trace.steps[-1][:2] == ("prefill", {"M": (0, MIN)})
    _run_until(s, lambda: s.requests["L"].num_computed_tokens > 0, trace)
    kinds = trace.kinds()
    idx = [i for i, k in enumerate(kinds) if k == "prefill"]
    assert idx[1] - idx[0] >= 3, kinds  # two decode steps between


def test_a_medium_prompt_with_no_long_prompt_waiting_holds_nothing():
    """With only medium and short prompts the policy admits like the unchunked
    server: a prompt arriving right after a medium one's step is prefilled at
    once, not held for the cadence."""
    s = _e2_scheduler(decode_steps=4)
    _start_decoders(s, 1)
    s.add_request(_request("M", MIN))
    assert _step(s) == ("prefill", {"M": (0, MIN)})
    s.add_request(_request("M2", 2 * CHUNK + 1))
    assert _step(s) == ("prefill", {"M2": (0, 2 * CHUNK + 1)})


def test_without_min_tokens_the_same_prompt_is_chunked():
    """Negative control: min_tokens 0 (the pre-E2 policy) splits it."""
    s = _scheduler(max_num_seqs=8, decode_steps=2)
    _start_decoders(s, 1)
    s.add_request(_request("M", MIN))
    kind, spans = _step(s)
    assert spans == {"M": (0, CHUNK)}


def test_older_medium_prompts_run_whole_and_hide_a_newer_long_one():
    s = _e2_scheduler(decode_steps=1)
    _start_decoders(s, 1)
    s.add_request(_request("M1", 3 * CHUNK))
    s.add_request(_request("M2", 2 * CHUNK + 1))
    s.add_request(_request("s", 10))
    s.add_request(_request("L", 8 * CHUNK))
    kind, spans = _step(s)  # one long waiting: no burst; the mediums whole, L hidden
    assert kind == "prefill"
    assert spans == {"M1": (0, 3 * CHUNK), "M2": (0, 2 * CHUNK + 1), "s": (0, 10)}
    trace = _Trace()
    _run_until(s, lambda: s.requests["L"].num_computed_tokens > 0, trace)
    assert trace.steps[-1][1] == {"L": (0, CHUNK)}


def test_a_newer_medium_prompt_takes_its_own_step_next_to_an_older_long_one():
    s = _e2_scheduler(decode_steps=1)
    _start_decoders(s, 1)
    s.add_request(_request("L", 8 * CHUNK))
    s.add_request(_request("M", 3 * CHUNK))
    trace = _Trace()
    _run_until(s, lambda: s.requests["L"].num_computed_tokens > 0, trace)
    prefills = [sp for k, sp, _ in trace.steps if k == "prefill"]
    assert prefills == [{"M": (0, 3 * CHUNK)}, {"L": (0, CHUNK)}], prefills


def test_two_long_prompts_waiting_next_to_decoders_are_a_burst():
    s = _e2_scheduler()
    _start_decoders(s, 1)
    s.add_request(_request("L0", 6 * CHUNK))
    s.add_request(_request("L1", 6 * CHUNK))
    kind, spans = _step(s)
    assert spans == {"L0": (0, 6 * CHUNK), "L1": (0, 6 * CHUNK)}


@pytest.mark.parametrize("hard_budget", [True, False])
def test_a_medium_prompt_gets_its_own_step_while_a_partial_is_in_flight(hard_budget):
    """A medium prompt cannot ride a chunk step (the step's threshold would split
    it into a second partial): with a partial in flight it takes a prefill step
    alone, whole, whatever the hard rider budget says, and the partial goes on."""
    s = _e2_scheduler(decode_steps=1, chunked_prefill_oversized_rider_step=hard_budget)
    _start_decoders(s, 1)
    s.add_request(_request("L", 8 * CHUNK))
    trace = _Trace()
    _step(s, trace)
    assert trace.steps[-1][1] == {"L": (0, CHUNK)}
    s.add_request(_request("M", 3 * CHUNK + 7))
    _run_until(s, lambda: not s.requests["L"].is_prefill_chunk, trace)
    prefills = [sp for k, sp, _ in trace.steps if k == "prefill"]
    assert {"M": (0, 3 * CHUNK + 7)} in prefills, prefills
    assert all(b - a <= CHUNK for sp in prefills if "L" in sp for a, b in [sp["L"]])
    assert not any("M" in sp and "L" in sp for sp in prefills), prefills


def test_a_burst_arriving_on_a_partial_finishes_it_whole_then_admits_the_rest():
    s = _e2_scheduler(decode_steps=1)
    _start_decoders(s, 1)
    s.add_request(_request("L0", 8 * CHUNK))
    _step(s)  # L0's first chunk
    assert s.requests["L0"].is_prefill_chunk
    for i in (1, 2):
        s.add_request(_request(f"L{i}", 6 * CHUNK))
    trace = _Trace()
    _run_until(s, lambda: s.requests["L2"].num_computed_tokens > 0, trace)
    prefills = [sp for k, sp, _ in trace.steps if k == "prefill"]
    assert prefills == [
        {"L0": (CHUNK, 8 * CHUNK)},
        {"L1": (0, 6 * CHUNK), "L2": (0, 6 * CHUNK)},
    ], prefills


def test_a_preempted_decoders_replay_is_never_split():
    """KV pressure (found on device): a preempted decoder comes back with prompt +
    outputs to replay. Split at the chunk size, its continuation chunk starts past
    the prompt, so the block accounting's decode test counts it as a decode row (a
    step with a rider trips the mixed-step refusal; a solo tail underflows the block
    reservation). The policy admits a long replay whole instead of chunking it."""
    s = _scheduler(max_num_seqs=8, decode_steps=1)
    _start_decoders(s, 2, max_tokens=600)
    for _ in range(40):  # d1 accumulates > CHUNK outputs (3 per ragged block step)
        _step(s)
    victim = s.requests["d1"]
    assert victim.num_output_tokens > CHUNK
    s.running.remove(victim)
    s._preempt_request(victim, 0.0)
    replay = victim.num_tokens
    assert replay > CHUNK and victim.num_computed_tokens == 0
    s.add_request(_request("s0", 20))  # a rider waiting next to the replay
    trace = _Trace()
    _run_until(s, lambda: victim.num_computed_tokens >= replay, trace)
    spans = [sp["d1"] for k, sp, _ in trace.steps if "d1" in sp and k == "prefill"]
    assert spans == [(0, replay)], spans
    for _ in range(
        10
    ):  # and it decodes as a block row afterwards (_step raises on a mixed step)
        _step(s)


# ------------------------------------------------------------- runner: sticky slots


def _runner(slots=8, block_chunked=True):
    return SimpleNamespace(
        tt_per_lane_max_num_seqs=slots,
        _req_state_slot={},
        _pending_state_slot_settle=None,
        requests={},
        _tt_block_output_chunked=block_chunked,
    )


def _prefill(runner, row_req_ids, continuing=None):
    if continuing:
        out = TTModelRunner._alloc_prefill_state_slots(
            runner, list(row_req_ids), continuing=set(continuing)
        )
    else:
        out = TTModelRunner._alloc_prefill_state_slots(runner, list(row_req_ids))
    runner.requests.update(dict.fromkeys(row_req_ids))
    return out


def test_a_continuation_keeps_its_slot_across_chunk_steps_with_riders():
    r = _runner()
    r.requests.update(dict.fromkeys(["d0", "d1"]))
    r._req_state_slot.update({"d0": 0, "d1": 1})
    (slot_l,) = _prefill(r, ["L"])  # first chunk: row 0 is held -> first free slot
    assert slot_l == 2
    for k in range(3):  # three chunk steps; the rider sits at row 0/1, before L
        rider = f"s{k}"
        slots = _prefill(r, [rider, "L"], continuing={"L"})
        assert slots[1] == slot_l
        assert slots[0] != slot_l
        r.requests.pop(rider)
        r._req_state_slot.pop(rider)


def test_without_sticky_slots_a_rider_takes_the_continuations_slot():
    """Negative control for the model's owner check: today's allocation (no
    ``continuing``) lets a rider at row 2 take the partial's slot 2."""
    r = _runner()
    r.requests.update(dict.fromkeys(["d0", "d1"]))
    r._req_state_slot.update({"d0": 0, "d1": 1})
    (slot_l,) = _prefill(r, ["L"])
    assert slot_l == 2
    slots = _prefill(r, ["s0", "x", "L"])  # rows 0, 1, 2
    assert slots[0] == slot_l  # the rider lands on the partial's slot ...
    assert slots[2] != slot_l  # ... and the continuation moves
    # with the continuation declared, it keeps its slot and the rider goes elsewhere
    r2 = _runner()
    r2.requests.update(dict.fromkeys(["d0", "d1"]))
    r2._req_state_slot.update({"d0": 0, "d1": 1})
    assert _prefill(r2, ["L"]) == [2]
    slots = _prefill(r2, ["s0", "x", "L"], continuing={"L"})
    assert slots[2] == 2 and 2 not in slots[:2]


def test_a_decode_remap_moves_the_continuations_logical_slot():
    r = _runner()
    r.requests.update(dict.fromkeys(["d0", "d1"]))
    r._req_state_slot.update({"d0": 3, "d1": 5})
    (slot_l,) = _prefill(r, ["L"])
    remap = TTModelRunner._decode_state_slot_remap(r, ["d0", "d1"])
    TTModelRunner.note_decode_state_slots_settled(r)
    assert remap is not None
    moved = r._req_state_slot["L"]
    assert remap.tolist()[moved] == slot_l  # the device row `moved` now holds L's state
    slots = _prefill(r, ["s0", "L"], continuing={"L"})
    assert slots[1] == moved


def test_a_continuation_without_a_slot_raises():
    r = _runner()
    with pytest.raises(RuntimeError, match="has no device state slot"):
        _prefill(r, ["L"], continuing={"L"})


def test_two_claims_on_a_continuation_slot_raise():
    r = _runner()
    r.requests.update(dict.fromkeys(["d0"]))
    r._req_state_slot.update({"d0": 2, "L": 2})
    with pytest.raises(RuntimeError, match="another request also claims"):
        _prefill(r, ["L"], continuing={"L"})


def test_allocation_without_continuations_is_unchanged():
    a, b = _runner(), _runner(block_chunked=False)
    for r in (a, b):
        r.requests.update(dict.fromkeys(["d0", "d1"]))
        r._req_state_slot.update({"d0": 0, "d1": 3})
    assert _prefill(a, ["x", "y", "z"]) == _prefill(b, ["x", "y", "z"]) == [1, 2, 4]


# ------------------------------------------------------------------ runner: outputs


def _output_runner(width, flag):
    return SimpleNamespace(
        _output_tokens_per_step=width,
        _tt_block_output_chunked=flag,
        _apply_sampled_tokens_to_state=lambda *a, **k: None,
        _enqueue_deferred_state_apply=lambda *a, **k: None,
    )


def test_block_output_chunk_rows_emit_nothing_or_one_anchor():
    runner = _output_runner(W, True)
    out = TTModelRunner._build_chunked_prefill_output(
        runner,
        req_ids=["L", "s0"],
        sampled_token_ids=torch.tensor([[0], [42]], dtype=torch.int32),
        logprobs=None,
        intermediate_mask=np.array([True, False]),
    )
    assert out.sampled_token_ids == [[], [42]]


def test_block_output_all_intermediate_rows_emit_nothing():
    runner = _output_runner(W, True)
    out = TTModelRunner._build_chunked_prefill_output(
        runner,
        req_ids=["L"],
        sampled_token_ids=torch.zeros(1, dtype=torch.int32),
        logprobs=None,
        intermediate_mask=np.array([True]),
    )
    assert out.sampled_token_ids == [[]]


def test_block_output_chunk_rows_without_the_flag_raise():
    runner = _output_runner(W, False)
    with pytest.raises(RuntimeError, match="tt_block_output_chunked_prefill"):
        TTModelRunner._build_chunked_prefill_output(
            runner,
            req_ids=["L"],
            sampled_token_ids=torch.zeros(1, dtype=torch.int32),
            logprobs=None,
            intermediate_mask=np.array([True]),
        )
