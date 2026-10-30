# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""TT chunk core on a real ``TTScheduler`` (vLLM scheduler + KV cache manager).

The model declares ``tt_prefill_chunk_tokens``; the platform hook resolves the
core while ``VllmConfig`` is built, exactly as in serving. Each test drives
schedule() / update_from_output() with a fake runner that returns ``[]`` for an
intermediate chunk and one token otherwise.
"""

from collections import deque
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

from vllm_tt_plugin.config import (
    TTPrefillChunkPolicy,
    get_tt_prefill_chunk_policy,
    get_tt_prefill_chunk_unit,
)
from vllm_tt_plugin.scheduler import TTScheduler

BLOCK = 16
CHUNK = 64
MAX_MODEL_LEN = 1024
LOCAL_MODEL_CONFIG = Path(__file__).parent / "model_configs" / "qwen2"


class _ChunkModel:
    model_capabilities = {
        "supports_chunked_prefill": True,
        "supports_async_decode": True,
        "tt_prefill_chunk_tokens": CHUNK,
    }


class _ExclusiveChunkModel:
    model_capabilities = {
        **_ChunkModel.model_capabilities,
        "tt_prefill_exclusive_chunks": True,
    }


class _PlainChunkedModel:
    """Supports chunked prefill but declares no resume unit: base chunking."""

    model_capabilities = {
        "supports_chunked_prefill": True,
        "supports_async_decode": True,
    }


class _UnchunkedModel:
    model_capabilities = {"supports_async_decode": True}


@contextmanager
def _model_resolution(model_class):
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
            return_value=(model_class, None),
        ),
    ):
        yield


def _scheduler(
    model_class=_ChunkModel,
    *,
    max_num_seqs=4,
    num_blocks=None,
    interleave=False,
    async_scheduling=False,
    long_prefill_token_threshold=0,
    tt=None,
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
        long_prefill_token_threshold=long_prefill_token_threshold,
        async_scheduling=async_scheduling,
        is_encoder_decoder=False,
    )
    cache_config = CacheConfig(
        block_size=BLOCK,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    with _model_resolution(model_class):
        config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
            additional_config={
                "tt": {"decode_interleave_enabled": interleave, **(tt or {})}
            },
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


def _runner_output(tokens: dict[str, list[int]]) -> ModelRunnerOutput:
    req_ids = list(tokens)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=[tokens[r] for r in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


class _Trace:
    """Per step: kind ('prefill'/'decode'/'idle') and {req_id: (start, end)}."""

    def __init__(self):
        self.steps = []

    def kinds(self):
        return [k for k, _ in self.steps]

    def spans(self, req_id):
        """The request's prefill spans."""
        return [sp[req_id] for k, sp in self.steps if k == "prefill" and req_id in sp]


def _schedule(scheduler: TTScheduler, trace: _Trace | None):
    """One schedule(); returns it with the tokens the fake runner will commit,
    decided at schedule time (an intermediate chunk samples nothing)."""
    starts = {r: q.num_computed_tokens for r, q in scheduler.requests.items()}
    out = scheduler.schedule()
    spans = {
        r: (starts.get(r, 0), starts.get(r, 0) + n)
        for r, n in out.num_scheduled_tokens.items()
    }
    # A decode schedules one token; more is prefill work, including a resumed
    # replay's later chunks, which start past the prompt.
    prefills = [
        r
        for r, (start, end) in spans.items()
        if start < scheduler.requests[r].num_prompt_tokens or end - start > 1
    ]
    if prefills and len(prefills) != len(spans):
        raise AssertionError(f"mixed prefill/decode step: {spans}")
    kind = "idle" if not spans else ("prefill" if prefills else "decode")
    if trace is not None:
        trace.steps.append((kind, spans))
    tokens = {
        r: [] if scheduler.requests[r].is_prefill_chunk else [5]
        for r in out.num_scheduled_tokens
    }
    return out, tokens, kind, spans


def _step(scheduler: TTScheduler, trace: _Trace | None = None):
    out, tokens, kind, spans = _schedule(scheduler, trace)
    scheduler.update_from_output(out, _runner_output(tokens))
    return kind, spans


def _run_until(scheduler, cond, trace=None, limit=500):
    for _ in range(limit):
        if cond():
            return
        _step(scheduler, trace)
    raise AssertionError("condition not reached")


def _start_decoders(scheduler, n, prompt_len=16, max_tokens=400):
    for i in range(n):
        scheduler.add_request(_request(f"d{i}", prompt_len, max_tokens))
    _step(scheduler)  # one prefill step admits them all, whole
    for i in range(n):
        assert scheduler.requests[f"d{i}"].num_computed_tokens == prompt_len


def _prompt_done(scheduler, req_id):
    request = scheduler.requests[req_id]
    return not request.is_prefill_chunk and request.num_output_tokens > 0


# ------------------------------------------------------------------ resolution


def test_platform_resolves_the_chunk_core_on_a_real_config():
    s = _scheduler(max_num_seqs=4)
    cfg = s.vllm_config.scheduler_config
    assert s._prefill_chunk_policy == TTPrefillChunkPolicy(CHUNK, False)
    assert get_tt_prefill_chunk_unit(s.vllm_config) == CHUNK
    assert cfg.enable_chunked_prefill is True
    assert cfg.long_prefill_token_threshold == CHUNK
    assert cfg.max_num_batched_tokens == MAX_MODEL_LEN + 4 * CHUNK


def test_async_scheduling_keeps_the_chunk_core():
    s = _scheduler(async_scheduling=True)
    assert s.vllm_config.scheduler_config.async_scheduling is True
    assert s._prefill_chunk_policy == TTPrefillChunkPolicy(CHUNK, False)


def test_max_num_seqs_one_schedules_like_a_model_without_chunked_prefill():
    """The chunk core is off at one seat, and the scheduler config is the one
    an unchunked model gets: prompts never split, single-user unchanged."""
    chunked = _scheduler(_ChunkModel, max_num_seqs=1)
    unchunked = _scheduler(_UnchunkedModel, max_num_seqs=1)
    assert chunked._prefill_chunk_policy is None
    assert get_tt_prefill_chunk_policy(chunked.vllm_config) is None
    fields = (
        "enable_chunked_prefill",
        "max_num_batched_tokens",
        "long_prefill_token_threshold",
        "disable_chunked_mm_input",
    )
    for field in fields:
        assert getattr(chunked.vllm_config.scheduler_config, field) == getattr(
            unchunked.vllm_config.scheduler_config, field
        ), field
    assert chunked.vllm_config.scheduler_config.enable_chunked_prefill is False
    chunked.add_request(_request("L", 5 * CHUNK + 3))
    _, spans = _step(chunked)
    assert spans == {"L": (0, 5 * CHUNK + 3)}


# ------------------------------------------------------------------ chunk shape


def test_no_decoders_runs_every_scheduled_prompt_whole():
    s = _scheduler()
    s.add_request(_request("L", 5 * CHUNK + 44))
    s.add_request(_request("S", 40))
    kind, spans = _step(s)
    assert kind == "prefill"
    assert spans == {"L": (0, 5 * CHUNK + 44), "S": (0, 40)}
    # The per-pass threshold is restored to the configured value.
    assert s.vllm_config.scheduler_config.long_prefill_token_threshold == CHUNK


def test_chunks_are_aligned_while_requests_decode():
    s = _scheduler()
    _start_decoders(s, 2)
    s.add_request(_request("L", 5 * CHUNK + 17))
    trace = _Trace()
    _run_until(s, lambda: _prompt_done(s, "L"), trace)
    assert trace.spans("L") == [(i * CHUNK, (i + 1) * CHUNK) for i in range(5)] + [
        (5 * CHUNK, 5 * CHUNK + 17)
    ]


def test_an_unaligned_intermediate_chunk_is_refused():
    """Negative control for the alignment check: a token budget below the
    chunk splits the prompt off the grid, and the pass refuses it."""
    s = _scheduler()
    _start_decoders(s, 1)
    s.max_num_scheduled_tokens = CHUNK - 8
    s.add_request(_request("L", 3 * CHUNK))
    with pytest.raises(RuntimeError, match=r"unaligned intermediate chunk \[0, 56\)"):
        s.schedule()


def test_a_partial_whose_decoders_left_finishes_in_one_step():
    # The interleave gives the decoder the steps it needs to finish.
    s = _scheduler(interleave=True)
    _start_decoders(s, 1, max_tokens=3)
    s.add_request(_request("L", 10 * CHUNK))
    trace = _Trace()
    _run_until(s, lambda: _prompt_done(s, "L"), trace)
    l_spans = trace.spans("L")
    assert len(l_spans) >= 2, "L was never chunked"
    assert all(b - a == CHUNK for a, b in l_spans[:-1])
    last_start, last_end = l_spans[-1]
    assert last_end == 10 * CHUNK and last_end - last_start > CHUNK
    assert "d0" not in s.requests  # the decoder finished before the remainder


# ------------------------------------------------------------------ one long in flight


def test_one_long_prompt_in_flight_and_the_oldest_goes_first():
    s = _scheduler()
    _start_decoders(s, 1)
    s.add_request(_request("L1", 3 * CHUNK))
    s.add_request(_request("L2", 3 * CHUNK))
    trace = _Trace()
    _run_until(s, lambda: s.requests["L2"].num_computed_tokens > 0, trace)
    first_l2 = next(i for i, (_, sp) in enumerate(trace.steps) if "L2" in sp)
    assert trace.spans("L1")[-1][1] == 3 * CHUNK
    assert all("L1" not in sp for _, sp in trace.steps[first_l2:])
    for _ in range(10):
        _step(s)
        assert sum(r.is_prefill_chunk for r in s.running) <= 1


def test_plain_chunking_puts_two_long_prompts_in_flight():
    """Negative control: a model without the unit keeps base chunking at the
    same threshold, which splits both long prompts in one step."""
    s = _scheduler(_PlainChunkedModel, long_prefill_token_threshold=CHUNK)
    assert s._prefill_chunk_policy is None
    _start_decoders(s, 1)
    s.add_request(_request("L1", 3 * CHUNK))
    s.add_request(_request("L2", 3 * CHUNK))
    _, spans = _step(s)
    assert spans == {"L1": (0, CHUNK), "L2": (0, CHUNK)}
    assert sum(r.is_prefill_chunk for r in s.running) == 2


# ------------------------------------------------------------------ exclusivity


def test_short_prompt_rides_with_a_chunk_without_exclusivity():
    s = _scheduler()
    _start_decoders(s, 1)
    s.add_request(_request("L", 3 * CHUNK))
    assert _step(s)[1] == {"L": (0, CHUNK)}
    s.add_request(_request("S", 40))
    kind, spans = _step(s)
    assert kind == "prefill" and spans == {"L": (CHUNK, 2 * CHUNK), "S": (0, 40)}


def test_exclusive_short_prompt_waits_until_the_partial_finishes():
    s = _scheduler(_ExclusiveChunkModel)
    assert s._prefill_chunk_policy == TTPrefillChunkPolicy(CHUNK, True)
    _start_decoders(s, 1)
    s.add_request(_request("L", 3 * CHUNK + 10))
    assert _step(s)[1] == {"L": (0, CHUNK)}
    s.add_request(_request("S", 40))
    trace = _Trace()
    _run_until(
        s, lambda: "S" in s.requests and s.requests["S"].num_computed_tokens > 0, trace
    )
    prefill_steps = [sp for kind, sp in trace.steps if kind == "prefill"]
    assert prefill_steps == [
        {"L": (CHUNK, 2 * CHUNK)},
        {"L": (2 * CHUNK, 3 * CHUNK)},
        {"L": (3 * CHUNK, 3 * CHUNK + 10)},  # the final chunk runs alone too
        {"S": (0, 40)},
    ]


def test_exclusive_step_without_a_partial_carries_whole_prompts_and_one_first_chunk():
    s = _scheduler(_ExclusiveChunkModel, max_num_seqs=6)
    _start_decoders(s, 1)
    s.add_request(_request("S1", 40))
    s.add_request(_request("L1", 3 * CHUNK))
    s.add_request(_request("S2", 30))
    s.add_request(_request("L2", 2 * CHUNK + 1))
    _, spans = _step(s)
    assert spans == {"S1": (0, 40), "L1": (0, CHUNK), "S2": (0, 30)}
    assert [r.request_id for r in s.waiting] == ["L2"]


def test_hidden_long_prompts_keep_their_queue_order():
    s = _scheduler(max_num_seqs=8)
    _start_decoders(s, 1)
    for req_id, length in (("L1", 3 * CHUNK), ("L2", 4 * CHUNK), ("L3", 5 * CHUNK)):
        s.add_request(_request(req_id, length))
    s.add_request(_request("S", 20))
    _, spans = _step(s)
    assert spans == {"L1": (0, CHUNK), "S": (0, 20)}
    assert [r.request_id for r in s.waiting] == ["L2", "L3"]


# ------------------------------------------------------------------ KV pressure


def _kv_pressure_run(s):
    """Riders admitted between the partial's chunks take the blocks its later
    chunks need. Returns (starved decode steps, L's preemptions, L's computed
    tokens per step)."""
    _start_decoders(s, 2, max_tokens=300)
    s.add_request(_request("L", 6 * CHUNK))
    _step(s)  # chunk 0
    assert s.requests["L"].num_computed_tokens == CHUNK
    for i in range(5):
        s.add_request(_request(f"r{i}", CHUNK, max_tokens=300))
    seen, starved = [], 0
    for _ in range(2000):
        L = s.requests["L"]
        was_starved = L.is_prefill_chunk and s._prefill_chunk_partial_starved()
        kind, _ = _step(s)
        starved += was_starved and kind == "decode"
        seen.append(L.num_computed_tokens)
        if L.num_preemptions or L.num_output_tokens > 0:
            break
    return starved, L.num_preemptions, seen


def test_kv_pressure_never_preempts_the_partial():
    s = _scheduler(max_num_seqs=8, num_blocks=41)
    starved, preemptions, seen = _kv_pressure_run(s)
    assert starved > 0, "the test never put the partial under KV pressure"
    assert preemptions == 0, "the partial was preempted and lost its chunks"
    assert seen == sorted(seen)
    assert s.requests["L"].num_output_tokens > 0


def test_without_the_kv_guard_the_partial_is_preempted(monkeypatch):
    """Negative control: the same run with the guard disabled restarts the
    partial from zero."""
    s = _scheduler(max_num_seqs=8, num_blocks=41)
    monkeypatch.setattr(s, "_prefill_chunk_partial_starved", lambda: False)
    _, preemptions, _ = _kv_pressure_run(s)
    assert preemptions > 0


# ------------------------------------------------------------------ composition


def test_chunk_core_composes_with_the_decode_interleave():
    """The interleave policy (2 prefill steps, then 1 decode step) decides the
    cadence; the core only shapes the prefill steps, which stay aligned."""
    s = _scheduler(interleave=True)
    _start_decoders(s, 1)
    assert _step(s)[0] == "decode"  # steady decode clears the interleave run
    s.add_request(_request("L", 6 * CHUNK))
    trace = _Trace()
    _run_until(s, lambda: _prompt_done(s, "L"), trace)
    assert trace.kinds() == ["prefill", "prefill", "decode"] * 2 + ["prefill"] * 2
    assert trace.spans("L") == [(i * CHUNK, (i + 1) * CHUNK) for i in range(6)]


def test_chunk_core_under_async_scheduling():
    """Async engine loop: the next step is scheduled before the previous one
    commits. The core reads only schedule-time state, so the chunks stay
    aligned and one long prompt stays in flight."""
    s = _scheduler(async_scheduling=True, interleave=True)
    s.add_request(_request("d0", 16, max_tokens=400))
    s.add_request(_request("L1", 5 * CHUNK + 9))
    s.add_request(_request("L2", 3 * CHUNK))
    trace = _Trace()
    inflight: deque = deque()
    for _ in range(200):
        if all(r in s.requests and _prompt_done(s, r) for r in ("L1", "L2")):
            break
        out, tokens, kind, _ = _schedule(s, trace)
        inflight.append((out, tokens))
        if len(inflight) > 1 or kind == "idle":
            done_out, done_tokens = inflight.popleft()
            s.update_from_output(done_out, _runner_output(done_tokens))
        assert sum(r.is_prefill_chunk for r in s.running) <= 1
    else:
        raise AssertionError("prompts did not finish")
    # d0 and L1 arrive together with nothing decoding: both run whole; L2 is
    # the second long prompt and waits, then chunks next to the decoders.
    assert trace.spans("L1") == [(0, 5 * CHUNK + 9)]
    assert trace.spans("L2") == [(i * CHUNK, (i + 1) * CHUNK) for i in range(3)]


# ------------------------------------------------------------------ gangs


def _start_decoding(scheduler, n):
    """n requests past their first decode step (gang_tokens caps a gang once a
    running request has decoded)"""
    _start_decoders(scheduler, n)
    assert _step(scheduler)[0] == "decode"


def _gang_scheduler(gang_tokens, **kwargs):
    return _scheduler(
        _ExclusiveChunkModel, tt={"prefill_gang_tokens": gang_tokens}, **kwargs
    )


def test_gang_knob_resolves_into_the_policy():
    s = _gang_scheduler(8 * CHUNK)
    assert s._prefill_chunk_policy == TTPrefillChunkPolicy(CHUNK, True, 8 * CHUNK)
    assert _scheduler(_ExclusiveChunkModel)._prefill_chunk_policy.gang_tokens == 0


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_gang_knob_must_be_a_non_negative_integer(value):
    with pytest.raises(ValueError, match="prefill_gang_tokens"):
        _gang_scheduler(value)


def test_gang_knob_needs_the_chunk_unit():
    with pytest.raises(ValueError, match="prefill_gang_tokens"):
        _scheduler(_PlainChunkedModel, tt={"prefill_gang_tokens": 1024})


def test_gang_idle_cap_needs_gangs():
    with pytest.raises(ValueError, match="prefill_gang_idle_tokens"):
        _scheduler(_ExclusiveChunkModel, tt={"prefill_gang_idle_tokens": 1024})


def test_gang_idle_cap_splits_a_burst_into_whole_gangs():
    """With nothing decoding the idle cap bounds the gang; the rest follow as the
    next gang while the first decode (gang_tokens), never split."""
    s = _scheduler(
        _ExclusiveChunkModel,
        max_num_seqs=8,
        tt={"prefill_gang_tokens": 4 * CHUNK, "prefill_gang_idle_tokens": 4 * CHUNK},
    )
    assert s._prefill_chunk_policy == TTPrefillChunkPolicy(
        CHUNK, True, 4 * CHUNK, 4 * CHUNK
    )
    for i in range(4):
        s.add_request(_request(f"g{i}", 2 * CHUNK))
    assert _step(s)[1] == {"g0": (0, 2 * CHUNK), "g1": (0, 2 * CHUNK)}
    assert _step(s)[1] == {"g2": (0, 2 * CHUNK), "g3": (0, 2 * CHUNK)}


def test_gang_with_nothing_decoding_runs_every_waiting_prompt_whole_in_one_step():
    """Without gangs one long prompt runs whole and the rest wait; a gang step
    carries them all, so their first tokens come out of the same step."""
    lengths = {"L1": 3 * CHUNK + 5, "L2": 4 * CHUNK, "S": 20, "L3": 2 * CHUNK + 1}
    s = _gang_scheduler(CHUNK, max_num_seqs=8)
    for req_id, n in lengths.items():
        s.add_request(_request(req_id, n))
    kind, spans = _step(s)
    assert kind == "prefill"
    assert spans == {r: (0, n) for r, n in lengths.items()}
    assert not s.waiting


def test_gang_while_decoding_is_bounded_by_gang_tokens():
    s = _gang_scheduler(7 * CHUNK, max_num_seqs=8)
    _start_decoding(s, 1)
    for req_id in ("A", "B", "C"):
        s.add_request(_request(req_id, 3 * CHUNK))
    _, spans = _step(s)
    assert spans == {"A": (0, 3 * CHUNK), "B": (0, 3 * CHUNK)}  # C would make 9
    assert [r.request_id for r in s.waiting] == ["C"]
    assert not any(r.is_prefill_chunk for r in s.running)


def test_gang_keeps_admission_order_and_stops_at_the_first_misfit():
    """A prefix: a small prompt behind one that does not fit waits its turn."""
    s = _gang_scheduler(5 * CHUNK, max_num_seqs=8)
    _start_decoding(s, 1)
    s.add_request(_request("A", 2 * CHUNK))
    s.add_request(_request("B", 4 * CHUNK))
    s.add_request(_request("S", 10))
    _, spans = _step(s)
    assert spans == {"A": (0, 2 * CHUNK)}
    assert [r.request_id for r in s.waiting] == ["B", "S"]


def test_gang_first_prompt_over_the_budget_is_chunked_as_before():
    s = _gang_scheduler(2 * CHUNK, max_num_seqs=8)
    _start_decoding(s, 1)
    s.add_request(_request("L", 5 * CHUNK))
    s.add_request(_request("S", 30))
    trace = _Trace()
    _run_until(s, lambda: _prompt_done(s, "L"), trace)
    assert trace.spans("L") == [(i * CHUNK, (i + 1) * CHUNK) for i in range(5)]
    # exclusive: S rode with L's first chunk (no partial yet), as without gangs
    assert trace.spans("S") == [(0, 30)]
    assert trace.steps[0][1] == {"L": (0, CHUNK), "S": (0, 30)}


def test_gang_respects_free_seats():
    s = _gang_scheduler(100 * CHUNK, max_num_seqs=3)
    _start_decoding(s, 1)
    for req_id in ("A", "B", "C"):
        s.add_request(_request(req_id, 2 * CHUNK))
    _, spans = _step(s)
    assert spans == {"A": (0, 2 * CHUNK), "B": (0, 2 * CHUNK)}
    assert [r.request_id for r in s.waiting] == ["C"]


def test_gang_waits_for_a_partial_to_finish():
    """A prompt split before the gang knob took effect (or one over the budget)
    keeps the pipeline exclusively; the gang starts after its final chunk."""
    s = _gang_scheduler(2 * CHUNK, max_num_seqs=8)
    _start_decoding(s, 1)
    s.add_request(_request("L", 4 * CHUNK))
    assert _step(s)[1] == {"L": (0, CHUNK)}
    s.add_request(_request("A", CHUNK + 1))
    s.add_request(_request("B", 20))
    trace = _Trace()
    _run_until(s, lambda: _prompt_done(s, "A"), trace)
    prefills = [sp for kind, sp in trace.steps if kind == "prefill"]
    assert prefills[:4] == [
        {"L": (CHUNK, 2 * CHUNK)},
        {"L": (2 * CHUNK, 3 * CHUNK)},
        {"L": (3 * CHUNK, 4 * CHUNK)},
        {"A": (0, CHUNK + 1), "B": (0, 20)},
    ]


def test_gang_steps_compose_with_the_decode_interleave():
    s = _gang_scheduler(4 * CHUNK, interleave=True, max_num_seqs=8)
    _start_decoding(s, 1)
    assert _step(s)[0] == "decode"
    kinds = []
    for i in range(3):
        s.add_request(_request(f"g{i}", 3 * CHUNK))
        kinds.append(_step(s)[0])
    # two gang steps, then the interleave's decode step; g2 runs after it
    assert kinds == ["prefill", "prefill", "decode"]
    assert _step(s)[1] == {"g2": (0, 3 * CHUNK)}


def test_gang_under_async_scheduling():
    s = _gang_scheduler(
        6 * CHUNK, async_scheduling=True, interleave=True, max_num_seqs=8
    )
    s.add_request(_request("d0", 16, max_tokens=400))
    for i in range(3):
        s.add_request(_request(f"L{i}", 3 * CHUNK + i))
    trace = _Trace()
    inflight: deque = deque()
    names = ("L0", "L1", "L2", "M0", "M1", "M2")
    for step in range(200):
        if step == 3:
            for i in range(3):
                s.add_request(_request(f"M{i}", 2 * CHUNK + i))
        if step > 3 and all(r in s.requests and _prompt_done(s, r) for r in names):
            break
        out, tokens, kind, _ = _schedule(s, trace)
        inflight.append((out, tokens))
        if len(inflight) > 1 or kind == "idle":
            done_out, done_tokens = inflight.popleft()
            s.update_from_output(done_out, _runner_output(done_tokens))
        assert not any(r.is_prefill_chunk for r in s.running)
    else:
        raise AssertionError("prompts did not finish")
    # every prompt ran whole: the first four together (nothing decoded yet),
    # the M prompts while d0 and the L prompts decode, at most 6 chunks a step
    for r in names:
        assert len(trace.spans(r)) == 1, (r, trace.spans(r))
    assert {"d0", "L0", "L1", "L2"} == set(trace.steps[0][1])
    for kind, sp in trace.steps[1:]:
        if kind == "prefill":
            assert sum(b - a for a, b in sp.values()) <= 6 * CHUNK


# ------------------------------------------------------------------ policy file


def test_policy_file_switches_gangs_and_the_interleave_at_runtime(
    tmp_path, monkeypatch
):
    policy = tmp_path / "policy.json"
    monkeypatch.setenv("TT_SCHED_POLICY_FILE", str(policy))
    s = _scheduler(_ExclusiveChunkModel, interleave=True, max_num_seqs=8)
    _start_decoders(s, 1)
    s.add_request(_request("A", 3 * CHUNK))
    assert _step(s)[1] == {"A": (0, CHUNK)}  # no file yet: the chunk path
    _run_until(s, lambda: _prompt_done(s, "A"))
    policy.write_text(
        '{"prefill_gang_tokens": 4096, "decode_interleave_prefill_steps": 1}'
    )
    s.add_request(_request("B", 3 * CHUNK))
    s.add_request(_request("C", 3 * CHUNK))
    assert _step(s)[1] == {"B": (0, 3 * CHUNK), "C": (0, 3 * CHUNK)}
    assert s._decode_interleave._prefill_steps == 1
    s.add_request(_request("D", 3 * CHUNK))
    assert _step(s)[0] == "decode"  # one prefill step, then the decode step
    # a key left out of a new version returns to the configured value
    policy.write_text("{}")
    import os

    os.utime(policy, ns=(1, 1))
    assert _step(s)[1] == {"D": (0, CHUNK)}
    assert s._prefill_chunk_policy == TTPrefillChunkPolicy(CHUNK, True)
    assert s._decode_interleave._prefill_steps == 2


def test_sched_trace_records_each_step(tmp_path, monkeypatch):
    import json

    trace_file = tmp_path / "trace.jsonl"
    monkeypatch.setenv("TT_SCHED_TRACE", str(trace_file))
    s = _scheduler(_ExclusiveChunkModel, max_num_seqs=8)
    _start_decoders(s, 1)
    s.add_request(_request("L", 3 * CHUNK))
    _step(s)
    _step(s)
    lines = [json.loads(x) for x in trace_file.read_text().splitlines()]
    assert [x["kind"] for x in lines] == ["prefill", "prefill", "prefill"]
    assert lines[1]["reqs"] == {"L": [0, CHUNK, 3 * CHUNK]}
    assert lines[1]["decoding"] == 1 and lines[1]["waiting"] == 0


def test_gang_requests_without_a_decode_step_do_not_cap_the_next_gang():
    """A burst that reaches the engine over two steps: the first gang's requests
    have their first token but have not decoded, so the stragglers still form a
    gang at the idle cap; once one of them has decoded, gang_tokens applies."""
    s = _gang_scheduler(2 * CHUNK, max_num_seqs=8)
    s.add_request(_request("a", 3 * CHUNK))
    assert _step(s)[1] == {"a": (0, 3 * CHUNK)}
    for req_id in ("b", "c"):
        s.add_request(_request(req_id, 3 * CHUNK))
    # a is running with its first token only: b and c run whole together
    assert _step(s)[1] == {"b": (0, 3 * CHUNK), "c": (0, 3 * CHUNK)}
    assert _step(s)[0] == "decode"
    s.add_request(_request("d", 3 * CHUNK))
    # now a, b, c have decoded: d is over gang_tokens and splits
    assert _step(s)[1] == {"d": (0, CHUNK)}


def test_gang_decode_started_counts_async_placeholders():
    from vllm_tt_plugin.scheduler import _decode_started

    r = _request("x", 10)
    r.num_output_placeholders = 1  # the prefill's token in flight
    assert not _decode_started(r)
    r.num_output_placeholders = 2  # and the first decode's
    assert _decode_started(r)
