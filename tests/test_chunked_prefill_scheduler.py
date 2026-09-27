# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""TT chunk policy on a real ``TTScheduler`` (vLLM scheduler + KV cache manager).

The model declares ``tt_prefill_chunk_tokens``; the platform hook resolves the
policy (chunk C, cadence N) while ``VllmConfig`` is built, exactly as in
serving. Each test drives schedule() / update_from_output() with a fake runner
that returns ``[]`` for an intermediate chunk and one token otherwise.
"""

from contextlib import contextmanager
from pathlib import Path
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

from vllm_tt_plugin.config import get_tt_prefill_chunk_policy
from vllm_tt_plugin.scheduler import TTScheduler

BLOCK = 16
CHUNK = 64
MAX_MODEL_LEN = 1024
LOCAL_MODEL_CONFIG = Path(__file__).parent / "model_configs" / "qwen2"


class _ChunkModel:
    model_capabilities = {
        "supports_chunked_prefill": True,
        "tt_prefill_chunk_tokens": CHUNK,
    }


@contextmanager
def _model_resolution():
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
            return_value=(_ChunkModel, None),
        ),
    ):
        yield


def _scheduler(
    *, max_num_seqs=4, num_blocks=None, decode_steps=2, chunk=None
) -> TTScheduler:
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
    tt = {"chunked_prefill_decode_steps": decode_steps}
    if chunk is not None:
        tt["prefill_chunk_tokens"] = chunk
    with _model_resolution():
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


def _request(req_id: str, prompt_len: int, max_tokens: int = 50) -> Request:
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
    """Per step: kind ('prefill'/'decode'/'idle') and {req_id: (start, end)}."""

    def __init__(self):
        self.steps = []

    def kinds(self):
        return [k for k, _ in self.steps]


def _step(scheduler: TTScheduler, trace: _Trace | None = None):
    starts = {r: q.num_computed_tokens for r, q in scheduler.requests.items()}
    out = scheduler.schedule()
    spans = {
        r: (starts.get(r, 0), starts.get(r, 0) + n)
        for r, n in out.num_scheduled_tokens.items()
    }
    is_prefill = any(
        spans[r][0] < scheduler.requests[r].num_prompt_tokens for r in spans
    )
    decodes = [
        r for r in spans if spans[r][0] >= scheduler.requests[r].num_prompt_tokens
    ]
    if is_prefill and decodes:
        raise AssertionError(f"mixed prefill/decode step: {spans}")
    kind = "idle" if not spans else ("prefill" if is_prefill else "decode")
    if trace is not None:
        trace.steps.append((kind, spans))
    req_ids = list(out.num_scheduled_tokens)
    runner_out = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=[
            [] if scheduler.requests[r].is_prefill_chunk else [5] for r in req_ids
        ],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(out, runner_out)
    return kind, spans


def _run_until(scheduler, cond, trace, limit=500):
    for _ in range(limit):
        if cond():
            return
        _step(scheduler, trace)
    raise AssertionError("condition not reached")


def _start_decoders(scheduler, n, prompt_len=16, max_tokens=400):
    for i in range(n):
        scheduler.add_request(_request(f"d{i}", prompt_len, max_tokens))
    _step(scheduler)  # one prefill step admits them all
    for i in range(n):
        assert scheduler.requests[f"d{i}"].num_computed_tokens == prompt_len


# ------------------------------------------------------------------ resolution


def test_platform_resolves_the_chunk_policy():
    s = _scheduler(max_num_seqs=4, decode_steps=3)
    cfg = s.vllm_config.scheduler_config
    assert get_tt_prefill_chunk_policy(s.vllm_config) == (CHUNK, 3)
    assert cfg.enable_chunked_prefill is True
    assert cfg.long_prefill_token_threshold == CHUNK
    assert cfg.max_num_batched_tokens == MAX_MODEL_LEN + 4 * CHUNK


def test_chunk_knob_must_be_a_multiple_of_the_model_unit():
    with pytest.raises(ValueError, match="multiple"):
        _scheduler(chunk=CHUNK + BLOCK)
    assert _scheduler(chunk=2 * CHUNK)._chunk_policy == (2 * CHUNK, 2)


# ------------------------------------------------------------------ cadence


def test_no_decoders_runs_the_whole_prompt_in_one_step():
    s = _scheduler()
    s.add_request(_request("L", 300))
    kind, spans = _step(s)
    assert kind == "prefill" and spans == {"L": (0, 300)}
    assert s.vllm_config.scheduler_config.long_prefill_token_threshold == CHUNK


def test_chunks_are_aligned_and_separated_by_n_decode_steps():
    s = _scheduler(decode_steps=2)
    _start_decoders(s, 2)
    s.add_request(_request("L", 5 * CHUNK + 17))
    trace = _Trace()
    _run_until(
        s,
        lambda: not s.requests["L"].is_prefill_chunk
        and s.requests["L"].num_output_tokens > 0,
        trace,
    )
    l_spans = [sp["L"] for k, sp in trace.steps if "L" in sp and k == "prefill"]
    assert l_spans == [(i * CHUNK, (i + 1) * CHUNK) for i in range(5)] + [
        (5 * CHUNK, 5 * CHUNK + 17)
    ]
    # between consecutive chunk steps exactly N=2 decode steps
    idx = [i for i, (k, sp) in enumerate(trace.steps) if "L" in sp and k == "prefill"]
    assert all(b - a == 3 for a, b in zip(idx, idx[1:])), trace.kinds()


def test_decoders_finishing_mid_prompt_lift_the_cap():
    s = _scheduler(decode_steps=1)
    _start_decoders(s, 1, max_tokens=3)
    s.add_request(_request("L", 10 * CHUNK))
    trace = _Trace()
    _run_until(s, lambda: s.requests["L"].num_computed_tokens >= 10 * CHUNK, trace)
    l_spans = [sp["L"] for k, sp in trace.steps if "L" in sp]
    # a few aligned chunks while d0 decodes, then the whole remainder at once
    assert l_spans[-1][1] == 10 * CHUNK and l_spans[-1][1] - l_spans[-1][0] > CHUNK
    assert all(a % CHUNK == 0 and b % CHUNK == 0 for a, b in l_spans)


def test_one_long_prompt_in_flight_second_waits_in_fcfs_order():
    s = _scheduler(decode_steps=0)
    _start_decoders(s, 1)
    s.add_request(_request("L1", 3 * CHUNK))
    s.add_request(_request("L2", 3 * CHUNK))
    trace = _Trace()
    _run_until(s, lambda: s.requests["L2"].num_computed_tokens > 0, trace)
    assert not s.requests[
        "L1"
    ].is_prefill_chunk  # L1 finished its prompt before L2 started
    first = [i for i, (_, sp) in enumerate(trace.steps) if "L2" in sp][0]
    assert all("L2" not in sp for _, sp in trace.steps[:first])
    assert [r.request_id for r in s.waiting] == []


def test_short_prompt_rides_with_the_chunk_and_is_held_by_the_cadence():
    s = _scheduler(decode_steps=3)
    _start_decoders(s, 1)
    s.add_request(_request("L", 4 * CHUNK))
    trace = _Trace()
    _step(s, trace)  # chunk 0
    assert trace.steps[-1][1] == {"L": (0, CHUNK)}
    s.add_request(_request("S", 40))
    for _ in range(3):  # the cadence holds S (and L) for N=3 decode steps
        kind, spans = _step(s, trace)
        assert kind == "decode" and "S" not in spans
    kind, spans = _step(s, trace)
    assert kind == "prefill" and spans == {"L": (CHUNK, 2 * CHUNK), "S": (0, 40)}
    assert not s.requests["S"].is_prefill_chunk


def test_riders_ahead_of_a_long_prompt_never_split_it_unaligned():
    """Review M3 / perf M1: riders queued before the long prompt take budget
    first; the budget sizing keeps the long prompt's split on the chunk grid,
    with and without running decoders."""
    s = _scheduler(max_num_seqs=8, decode_steps=0)
    for i in range(6):
        s.add_request(_request(f"r{i}", CHUNK))
    s.add_request(_request("L", MAX_MODEL_LEN - 8))
    kind, spans = _step(s)
    assert spans["L"] == (0, MAX_MODEL_LEN - 8)  # dynamic cap: whole prompt, one step
    s2 = _scheduler(max_num_seqs=8, decode_steps=0)
    _start_decoders(s2, 1)
    for i in range(6):
        s2.add_request(_request(f"r{i}", CHUNK))
    s2.add_request(_request("L", MAX_MODEL_LEN - 8))
    kind, spans = _step(s2)
    assert spans["L"] == (0, CHUNK) and all(
        spans[f"r{i}"] == (0, CHUNK) for i in range(6)
    )


def test_hidden_partial_seat_counts_against_max_num_seqs():
    """Review M4: max_num_seqs=4, 2 decoders, 1 partial, 3 short prompts waiting:
    the next prefill step admits exactly one."""
    s = _scheduler(max_num_seqs=4, decode_steps=1)
    _start_decoders(s, 2)
    s.add_request(_request("L", 4 * CHUNK))
    _step(s)  # chunk 0 -> L partial
    for i in range(3):
        s.add_request(_request(f"s{i}", 20))
    trace = _Trace()
    _run_until(
        s, lambda: any(f"s{i}" in sp for _, sp in trace.steps for i in range(3)), trace
    )
    admitted = [r for r in trace.steps[-1][1] if r.startswith("s")]
    assert len(admitted) == 1
    for _ in range(20):  # no step may exceed the seats either
        _step(s)
        live = [r for r in s.running]
        assert len(live) <= 4


def test_partial_under_kv_pressure_never_restarts_from_zero():
    """Review M2: riders admitted between chunks take the blocks the partial's
    later chunks need. The partial is not scheduled (so not self-preempted)
    until its next chunk fits; decode steps preempt/finish decoders instead,
    and its computed tokens never go back."""
    s = _scheduler(max_num_seqs=8, num_blocks=41, decode_steps=1)
    _start_decoders(s, 2, max_tokens=300)
    s.add_request(_request("L", 6 * CHUNK))
    _step(s)  # chunk 0
    assert s.requests["L"].num_computed_tokens == CHUNK
    for i in range(5):
        s.add_request(_request(f"r{i}", CHUNK, max_tokens=300))
    seen, starved_steps = [], 0
    for _ in range(2000):
        kind, spans = _step(s)
        if "L" not in s.requests:
            break
        L = s.requests["L"]
        assert L.num_preemptions == 0, "the partial was preempted and lost its chunks"
        seen.append(L.num_computed_tokens)
        if L.is_prefill_chunk and kind == "decode" and not s._cp_partial_fits(L, CHUNK):
            starved_steps += 1
        if L.num_output_tokens > 0:
            break
    assert starved_steps > 0, "the test never put the partial under KV pressure"
    assert seen == sorted(seen), "the partial lost computed tokens"
    assert s.requests["L"].num_output_tokens > 0


def test_threshold_restored_after_a_discarded_prefill_pass():
    s = _scheduler(decode_steps=0)
    _start_decoders(s, 1)
    s.add_request(_request("L", 3 * CHUNK))
    for _ in range(6):
        _step(s)
        assert s.vllm_config.scheduler_config.long_prefill_token_threshold == CHUNK
