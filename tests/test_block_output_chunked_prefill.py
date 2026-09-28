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
    assert extras["burst_longs"] == 0


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


def test_the_oldest_rider_is_admitted_even_above_the_budget():
    s = _scheduler(max_num_seqs=8, decode_steps=1, chunked_prefill_rider_tokens=0)
    _start_decoders(s, 1)
    s.add_request(_request("L", 4 * CHUNK))
    _step(s)
    s.add_request(_request("s0", CHUNK))
    trace = _Trace()
    _run_until(s, lambda: s.requests["s0"].num_computed_tokens > 0, trace)
    assert set(trace.steps[-1][1]) == {"L", "s0"}


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


def test_a_burst_of_long_prompts_falls_back_to_prefill_first():
    s = _scheduler(max_num_seqs=8, decode_steps=2, chunked_prefill_burst_longs=2)
    _start_decoders(s, 1)
    for i in range(3):
        s.add_request(_request(f"L{i}", 3 * CHUNK + 5))
    kind, spans = _step(s)  # 3 long pending: whole prompt, one at a time
    assert kind == "prefill" and spans == {"L0": (0, 3 * CHUNK + 5)}
    kind, spans = _step(s)  # 2 pending: still the burst
    assert kind == "prefill" and spans == {"L1": (0, 3 * CHUNK + 5)}
    kind, spans = _step(s)  # 1 pending: the burst drained, chunk again
    assert kind == "prefill" and spans == {"L2": (0, CHUNK)}


def test_one_long_prompt_with_decoders_is_still_chunked_under_the_burst_extra():
    s = _scheduler(max_num_seqs=8, decode_steps=2, chunked_prefill_burst_longs=2)
    _start_decoders(s, 1)
    s.add_request(_request("L1", 3 * CHUNK + 5))
    kind, spans = _step(s)
    assert spans == {"L1": (0, CHUNK)}


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
