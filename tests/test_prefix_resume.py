# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Prefix resume (prefix_resume.py): the hit cap, the tail directory's orders,
prompt-only caching and the runner kwargs, on vLLM's own scheduler and KV
cache manager. Small geometry: 16-token blocks, a 64-token grid and budget,
a 32-token minimum suffix."""

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
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from vllm_tt_plugin.config import get_tt_prefix_resume, store_tt_prefix_resume
from vllm_tt_plugin.prefix_resume import (
    get_tail_orders,
    prefix_resume_kwargs,
    tail_orders_for_rows,
)
from vllm_tt_plugin.scheduler import TTScheduler

BLOCK, G, T, BUDGET, MAX_LEN = 16, 64, 32, 64, 1024
LOCAL_MODEL_CONFIG = Path(__file__).parent / "model_configs" / "qwen2"


class _StubModel:
    pass


@contextmanager
def _stub_model_resolution():
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
            return_value=(_StubModel, None),
        ),
    ):
        yield


def _spec(sliding=None):
    common = dict(block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32)
    if sliding:
        return SlidingWindowSpec(**common, sliding_window=sliding)
    return FullAttentionSpec(**common)


def _scheduler(*, slots=2, prompt_only=True, hybrid=False, retention=0, max_num_seqs=4):
    model_config = ModelConfig(
        model=str(LOCAL_MODEL_CONFIG),
        dtype="float16",
        seed=42,
        skip_tokenizer_init=True,
    )
    model_config.max_model_len = MAX_LEN
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=BUDGET,
        max_model_len=MAX_LEN,
        enable_chunked_prefill=True,
        async_scheduling=False,
        is_encoder_decoder=False,
    )
    cache_config = CacheConfig(
        block_size=BLOCK,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=True,
        prefix_cache_retention_interval=retention,
    )
    with _stub_model_resolution():
        config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=ParallelConfig(),
            device_config=DeviceConfig(device="cpu"),
            additional_config={"tt": {"prefill_chunk_alignment": BLOCK}},
        )
    # The stub declares nothing, so the platform switched both off; a
    # prefix-resume model declares both (and the platform would store this).
    config.cache_config.enable_prefix_caching = True
    config.scheduler_config.enable_chunked_prefill = True
    config.scheduler_config.max_num_batched_tokens = BUDGET
    if getattr(config.scheduler_config, "max_num_scheduled_tokens", None) is not None:
        config.scheduler_config.max_num_scheduled_tokens = BUDGET
    store_tt_prefix_resume(
        config,
        {
            "granularity": G,
            "min_suffix": T,
            "tail_slots": slots,
            "prompt_only": prompt_only,
            "req_ids": True,
        },
    )
    num_blocks = 4 * MAX_LEN // BLOCK
    cache_config.num_gpu_blocks = num_blocks
    groups = [KVCacheGroupSpec(["full"], _spec())]
    if hybrid:
        groups.append(KVCacheGroupSpec(["sliding"], _spec(sliding=2 * BLOCK)))
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
    )
    return TTScheduler(
        vllm_config=config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(config),
    )


def _tokens(seed, n):
    return [(1000 + 37 * seed + 11 * j) % 30000 + 3 for j in range(n)]


def _request(req_id, prompt, max_tokens=4):
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=2)
    return Request(
        request_id=req_id,
        prompt_token_ids=list(prompt),
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
    )


def _step(scheduler):
    """One schedule() plus a model output: a token for every request whose
    prefill is complete (or decoding), none for an intermediate chunk."""
    out = scheduler.schedule()
    req_ids = list(out.num_scheduled_tokens)
    tokens = []
    for req_id in req_ids:
        request = scheduler.requests[req_id]
        done = request.num_computed_tokens >= request.num_tokens
        tokens.append([7] if done else [])
    scheduler.update_from_output(
        out,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=tokens,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    return out


def _run(scheduler, steps=100):
    log = []
    for _ in range(steps):
        if not scheduler.requests:
            return log
        out = _step(scheduler)
        log.append(
            (
                {r.req_id: r.num_computed_tokens for r in out.scheduled_new_reqs},
                dict(out.num_scheduled_tokens),
                get_tail_orders(out),
            )
        )
    raise RuntimeError("requests did not finish")


def _orders(log, req_id):
    return [o[req_id] for _, _, o in log if req_id in o]


def test_a_canonical_run_stores_its_two_latest_grid_tails():
    s = _scheduler(slots=2)
    s.add_request(_request("A", _tokens(1, 200)))
    log = _run(s)
    assert _orders(log, "A") == [
        {"store": (0, 64, 1)},
        {"store": (1, 128, 2)},
        {"store": (0, 192, 3)},  # its oldest entry's slot: at most two per request
    ]


def test_a_hit_resumes_at_the_largest_saved_point_with_a_restore():
    s = _scheduler(slots=2)
    a = _tokens(2, 200)
    s.add_request(_request("A", a))
    _run(s)
    s.add_request(_request("B", a[:192] + _tokens(3, 60)))  # h = 192 <= 252 - 32
    log = _run(s)
    assert log[0][0] == {"B": 192} and log[0][1] == {"B": 60}
    assert _orders(log, "B")[0] == {"restore": (0, 192, 3)}
    assert s._prefix_resume.counters["restores"] == 1


def test_the_cap_keeps_the_minimum_suffix():
    s = _scheduler(slots=2)
    a = _tokens(4, 200)
    s.add_request(_request("A", a))
    _run(s)
    s.add_request(
        _request("B", a[:192] + _tokens(5, 10))
    )  # 202 - 32 = 170: the 128 entry
    log = _run(s)
    assert log[0][0] == {"B": 128}
    # B's own canonical chunk [128, 192) ends on A's saved 192 point: refreshed,
    # not stored again
    assert _orders(log, "B")[0] == {"restore": (1, 128, 2)}
    assert s._prefix_resume.counters["capped"] == 1


def test_an_identical_tail_is_refreshed_not_stored_again():
    s = _scheduler(slots=2)
    a = _tokens(30, 200)
    s.add_request(_request("A", a))
    _run(s)  # slot 1: (128, gen 2); slot 0: (192, gen 3)
    directory = s._prefix_resume.directory
    before = directory.entries[0].last_use
    s.add_request(_request("B", a[:192] + _tokens(31, 10)))  # restores 128
    log = _run(s)
    assert [o for o in _orders(log, "B") if "store" in o] == []
    assert s._prefix_resume.counters["dedups"] == 1
    assert (directory.entries[0].end, directory.entries[0].generation) == (192, 3)
    assert directory.entries[0].last_use > before
    s.add_request(_request("C", a[:192] + _tokens(32, 60)))  # A's own 192 tail
    log = _run(s)
    assert _orders(log, "C")[0] == {"restore": (0, 192, 3)}


def test_the_same_tokens_under_another_salt_store_their_own_tail():
    s = _scheduler(slots=2)
    a = _tokens(33, 70)
    s.add_request(_request("A", a))
    _run(s)
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=4, ignore_eos=True)
    params.update_from_generation_config({}, eos_token_id=2)
    salted = Request(
        request_id="B",
        prompt_token_ids=list(a),
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK, sha256),
        cache_salt="tenant-b",
    )
    s.add_request(salted)
    log = _run(s)
    assert log[0][0] == {"B": 0}
    assert _orders(log, "B") == [{"store": (1, 64, 2)}]
    assert s._prefix_resume.counters["dedups"] == 0


def test_without_a_saved_point_the_hit_is_zero():
    s = _scheduler(slots=0)
    a = _tokens(6, 200)
    s.add_request(_request("A", a))
    _run(s)
    s.add_request(_request("B", a[:192] + _tokens(7, 60)))
    log = _run(s)
    assert log[0][0] == {"B": 0} and not _orders(log, "B")
    assert s._prefix_resume.counters["cold"] == 1


def test_a_non_canonical_source_stores_nothing():
    """D1: a request whose chunks did not all start on the grid (here B, which
    took a step's 24-token leftover, rounded to 16) never stores a tail."""
    s = _scheduler(slots=2)
    s.add_request(_request("A", _tokens(8, 40)))
    b = _tokens(9, 200)
    s.add_request(_request("B", b))
    log = _run(s)
    assert log[0][1] == {"A": 40, "B": 16}
    assert not _orders(log, "B") and not _orders(log, "A")
    s.add_request(_request("C", b[:192] + _tokens(10, 60)))
    log = _run(s)
    assert log[0][0] == {"C": 0}


def test_prompt_only_caching_never_offers_decode_written_blocks():
    s = _scheduler(slots=0, prompt_only=True)
    a = _request("A", _tokens(11, 40), max_tokens=40)
    s.add_request(a)
    _run(s)
    echo = _request("B", list(a.prompt_token_ids) + [7] * 40 + _tokens(12, 30))
    _, hit, _ = s.kv_cache_manager.get_computed_blocks(echo)
    assert hit == 32  # A's full prompt blocks only; its 40 outputs were never cached
    plain = _scheduler(slots=0, prompt_only=False)
    a2 = _request("A", _tokens(11, 40), max_tokens=40)
    plain.add_request(a2)
    _run(plain)
    echo2 = _request("B", list(a2.prompt_token_ids) + [7] * 40 + _tokens(12, 30))
    _, hit2, _ = plain.kv_cache_manager.get_computed_blocks(echo2)
    assert hit2 > 40  # without it, the outputs' blocks are hits


def test_a_slot_restored_in_a_step_is_not_restored_and_stored_together():
    s = _scheduler(slots=1)
    a = _tokens(13, 64)
    s.add_request(_request("A", a + _tokens(14, 40)))  # stores (0, 64)
    _run(s)
    s.add_request(_request("B", a + _tokens(15, 40)))  # restores slot 0 at 64
    s.add_request(_request("C", _tokens(16, 30)))  # no grid end: no store
    first = _step(s)
    orders = get_tail_orders(first)
    assert orders == {"B": {"restore": (0, 64, 1)}}
    s2 = _scheduler(slots=1)
    s2.add_request(_request("A", a + _tokens(14, 40)))
    _run(s2)
    s2.add_request(_request("B", a + _tokens(15, 40)))
    s2.add_request(
        _request("D", _tokens(17, 100))
    )  # its first chunk would end at... 64 - 40 = 24: no grid end
    orders = get_tail_orders(_step(s2))
    assert all(
        "store" not in o or o["store"][0] != 0 for r, o in orders.items() if r != "B"
    )


def test_a_hybrid_hit_re_looks_up_at_the_saved_point():
    """With sliding-window groups, the cap's re-lookup must find the sliding
    blocks at the saved point: the retention interval keeps them there."""
    s = _scheduler(slots=2, hybrid=True, retention=G)
    a = _tokens(18, 200)
    s.add_request(_request("A", a))
    _run(s)
    s.add_request(
        _request("B", a[:150] + _tokens(19, 80))
    )  # full hit 144: the 128 entry
    log = _run(s)
    assert log[0][0] == {"B": 128}
    assert _orders(log, "B")[0] == {"restore": (1, 128, 2), "store": (0, 192, 4)}


def test_a_preempted_request_resumes_at_its_own_saved_point():
    s = _scheduler(slots=2)
    a = _request("A", _tokens(20, 200), max_tokens=30)
    s.add_request(a)
    for _ in range(6):
        _step(s)  # prefill done, a couple of decodes
    s._preempt_request(a, 0.0)
    out = _step(s)
    orders = get_tail_orders(out)
    # min(h, n - 32, P) falls below 192 once the outputs count: its own 128 entry
    assert orders["A"]["restore"] == (1, 128, 2)
    assert "A" in out.scheduled_cached_reqs.resumed_req_ids


def test_runner_kwargs():
    cfg = {
        "granularity": G,
        "min_suffix": T,
        "tail_slots": 2,
        "prompt_only": True,
        "req_ids": True,
    }
    mi = SimpleNamespace(
        row_req_ids=["x", "y"], prefix_tail_orders=[None, {"store": (0, 64, 1)}]
    )
    assert prefix_resume_kwargs(cfg, mi, prefill=True) == {
        "req_ids": ["x", "y"],
        "tail_orders": [None, {"store": (0, 64, 1)}],
    }
    assert prefix_resume_kwargs(cfg, mi, prefill=False) == {"req_ids": ["x", "y"]}
    assert prefix_resume_kwargs({**cfg, "req_ids": False}, mi, prefill=False) == {}
    with pytest.raises(RuntimeError, match="declares no prefix_resume"):
        prefix_resume_kwargs(None, mi, prefill=True)
    assert (
        prefix_resume_kwargs(
            None, SimpleNamespace(prefix_tail_orders=None), prefill=True
        )
        == {}
    )
    assert tail_orders_for_rows({"y": {"store": (0, 64, 1)}}, ["x", "y"]) == [
        None,
        {"store": (0, 64, 1)},
    ]
    with pytest.raises(RuntimeError, match="outside the step's rows"):
        tail_orders_for_rows({"z": {}}, ["x", "y"])
    assert tail_orders_for_rows({}, ["x"]) is None


def test_platform_resolution():
    from vllm_tt_plugin.prefix_resume import (
        resolve_and_store_prefix_resume as _resolve_prefix_resume,
    )

    s = _scheduler()
    config = s.vllm_config
    declared = {
        "granularity": 8192,
        "min_suffix": 2080,
        "tail_slots": 4,
        "prompt_only": True,
        "req_ids": True,
    }
    model = type("M", (), {})
    _resolve_prefix_resume(config, {"prefix_resume": declared}, model, True)
    assert get_tt_prefix_resume(config) == declared
    _resolve_prefix_resume(config, {}, model, True)
    assert get_tt_prefix_resume(config) is None
    with pytest.raises(
        ValueError, match="without model_capabilities\\['supports_prefix_caching'\\]"
    ):
        _resolve_prefix_resume(config, {"prefix_resume": declared}, model, False)
    with pytest.raises(ValueError, match="exactly the keys"):
        _resolve_prefix_resume(
            config, {"prefix_resume": {"granularity": 8192}}, model, True
        )
    with pytest.raises(ValueError, match="multiple of the"):
        _resolve_prefix_resume(
            config, {"prefix_resume": {**declared, "granularity": 100}}, model, True
        )
    config.additional_config["_tt_resolved_lane_count"] = 2
    with pytest.raises(ValueError, match="one TT lane per engine"):
        _resolve_prefix_resume(config, {"prefix_resume": declared}, model, True)
    config.additional_config.pop("_tt_resolved_lane_count")
    config.cache_config.enable_prefix_caching = False
    _resolve_prefix_resume(config, {"prefix_resume": declared}, model, True)
    assert get_tt_prefix_resume(config) is None
