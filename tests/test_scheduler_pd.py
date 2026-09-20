# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only tests for TTScheduler under prefill/decode disaggregation.

A real ``TTScheduler`` (real ``KVCacheManager``) drives the vLLM base
``schedule()`` with a STUB scheduler-role connector that follows the
``TTKVConnector`` contract of ``profiles/pd/PHASE2_DESIGN.md`` 3.2 (consumer
role): ``(T-1, True)`` for a valid remote request, ``(None, False)`` for a
deferral or a demotion, ``(0, False)`` for a plain request, and the B2
failed-load re-admission rule. The runner is simulated by
``ModelRunnerOutput`` objects fed to ``update_from_output``.

Covers PHASE2_DESIGN 8.3 and the round-3 AM1 slot-capacity amendment.
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
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from vllm_tt_plugin.scheduler import TTScheduler

BLOCK_SIZE = 16
MAX_MODEL_LEN = 512
LOCAL_MODEL_CONFIG = Path(__file__).parent / "model_configs" / "qwen2"
PROMPT = 32  # T of every request unless stated; T-1 = 31 -> 2 blocks
TOKEN = 7  # a sampled token that is not EOS (2)
# KV pressure: a 24-block pool has 23 usable blocks (block 0 is the null
# block); a running 32-token decode holds 3 and a pinned remote load 2, so a
# 22-block prompt can never be admitted while one decode runs.
PRESSURE_BLOCKS = 24
STARVED = 22 * BLOCK_SIZE - 4


class _StubModel:
    """No model_capabilities: the platform hook resolves plain AR serving."""


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


class _StubMeta(KVConnectorMetadata):
    def __init__(self, reqs_to_recv: dict[str, dict]):
        self.reqs_to_recv = reqs_to_recv


class _StubWorkerMeta(KVConnectorWorkerMetadata):
    def __init__(self, free_state_slots: int):
        self.free_state_slots = free_state_slots

    def aggregate(self, other):
        return _StubWorkerMeta(min(self.free_state_slots, other.free_state_slots))


class StubConsumerConnector:
    """Scheduler-role stub of ``TTKVConnector`` (consumer, PHASE2_DESIGN 3.2).

    Observable state: ``meta_calls`` (one ``build_connector_meta`` per step),
    ``metas`` (the drained descriptors per call), ``demoted``, ``free_slots``.
    ``reject`` names request ids whose params fail validation.
    """

    def __init__(self, free_slots: int, reject: set[str] | None = None):
        self.free_slots = free_slots
        self.reject = set(reject or ())
        self.meta_calls = 0
        self.metas: list[_StubMeta] = []
        self.demoted: list[str] = []
        self._reqs_need_recv: dict[str, dict] = {}

    # --- 3.2 ---
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        params = request.kv_transfer_params
        if not params:
            return 0, False
        if (
            params.get("_tt_recv_recorded")
            and not params.get("_tt_demoted")
            and num_computed_tokens == 0
            and request.status == RequestStatus.WAITING
        ):
            # B2: failed load promoted with num_computed_tokens == 0.
            return self._demote(request)
        if params.get("do_remote_prefill"):
            if request.request_id in self.reject:
                return self._demote(request)
            if self.free_slots <= 0:
                return None, False
            assert num_computed_tokens == 0
            return request.num_prompt_tokens - 1, True
        return 0, False

    def _demote(self, request):
        params = request.kv_transfer_params
        params["do_remote_prefill"] = False
        params["_tt_demoted"] = True
        self.demoted.append(request.request_id)
        return None, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens == 0:
            return
        assert num_external_tokens == request.num_prompt_tokens - 1
        local = blocks.get_block_ids()[0]
        self._reqs_need_recv[request.request_id] = {
            "local_block_ids": list(local),
            "num_tokens": num_external_tokens,
        }
        request.kv_transfer_params["do_remote_prefill"] = False
        request.kv_transfer_params["_tt_recv_recorded"] = True
        self.free_slots -= 1

    def build_connector_meta(self, scheduler_output):
        self.meta_calls += 1
        meta = _StubMeta(self._reqs_need_recv)
        self._reqs_need_recv = {}
        self.metas.append(meta)
        return meta

    def request_finished(self, request, block_ids):
        return False, None

    def update_connector_output(self, kv_connector_output):
        wm = kv_connector_output.kv_connector_worker_meta
        if wm:
            self.free_slots = wm.free_state_slots

    # --- inert base-class surface the scheduler touches ---
    def on_new_request(self, request):
        pass

    def bind_gpu_block_pool(self, pool):
        pass

    def has_pending_push_work(self):
        return False

    def take_events(self):
        return ()

    def get_kv_connector_stats(self):
        return None

    def shutdown(self):
        pass


def _scheduler(*, max_num_seqs: int, num_blocks: int) -> TTScheduler:
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


def _pd_scheduler(
    *,
    max_num_seqs: int = 8,
    num_blocks: int = 64,
    free_slots: int | None = None,
    reject: set[str] | None = None,
    recompute: bool = True,
) -> tuple[TTScheduler, StubConsumerConnector]:
    """A TTScheduler with the stub connector injected the way
    ``Scheduler.__init__`` would have created one from a kv_transfer_config."""
    scheduler = _scheduler(max_num_seqs=max_num_seqs, num_blocks=num_blocks)
    connector = StubConsumerConnector(
        free_slots=max_num_seqs if free_slots is None else free_slots, reject=reject
    )
    scheduler.connector = connector
    scheduler.recompute_kv_load_failures = recompute
    return scheduler, connector


def _request(
    request_id: str,
    *,
    prompt_len: int = PROMPT,
    max_tokens: int = 64,
    remote: bool = False,
) -> Request:
    init_none_hash(sha256)
    extra_args = None
    if remote:
        extra_args = {
            "kv_transfer_params": {
                "do_remote_prefill": True,
                "do_remote_decode": False,
                "remote_num_tokens": prompt_len - 1,
                "xfer_id": f"p0:{request_id}",
            }
        }
    sampling_params = SamplingParams(
        max_tokens=max_tokens, ignore_eos=True, extra_args=extra_args
    )
    sampling_params.update_from_generation_config({}, eos_token_id=2)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(1, prompt_len + 1)),
        sampling_params=sampling_params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _runner_output(
    scheduler_output: SchedulerOutput,
    kv_connector_output: KVConnectorOutput | None = None,
) -> ModelRunnerOutput:
    """What the runner returns for ``scheduler_output``: one sampled token per
    scheduled request (a prefill anchor or a decode token), plus the connector
    output. A zero-token step carries only the connector output."""
    req_ids = list(scheduler_output.num_scheduled_tokens)
    out = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=[[TOKEN] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict=dict.fromkeys(req_ids, None),
        pooler_output=[],
    )
    out.kv_connector_output = kv_connector_output
    return out


def _step(
    scheduler: TTScheduler,
    kv_connector_output: KVConnectorOutput | None = None,
) -> SchedulerOutput:
    """One engine step: schedule, execute (simulated), update."""
    scheduler_output = scheduler.schedule()
    scheduler.update_from_output(
        scheduler_output, _runner_output(scheduler_output, kv_connector_output)
    )
    return scheduler_output


def _decode_ids(scheduler_output: SchedulerOutput) -> list[str]:
    return list(scheduler_output.scheduled_cached_reqs.req_ids)


def _new_ids(scheduler_output: SchedulerOutput) -> list[str]:
    return [r.req_id for r in scheduler_output.scheduled_new_reqs]


def _admit_decodes(scheduler: TTScheduler, ids: list[str]) -> None:
    """Prefill ``ids`` (plain requests) so they are running decodes."""
    for req_id in ids:
        scheduler.add_request(_request(req_id))
    out = _step(scheduler)
    assert sorted(_new_ids(out)) == sorted(ids)
    assert all(scheduler.requests[r].status == RequestStatus.RUNNING for r in ids)


def _assert_never_mixed(scheduler_output: SchedulerOutput, scheduler) -> None:
    """No output puts a prefill row next to a decode row (P/scheduler.py:66-68).

    A remote-ready continuation (``num_computed_tokens == T-1``, 1 token) is a
    decode row and may share the step with running decodes (6.2)."""
    prefill_new = [
        r.req_id
        for r in scheduler_output.scheduled_new_reqs
        if r.num_computed_tokens != scheduler.requests[r.req_id].num_prompt_tokens - 1
        or scheduler_output.num_scheduled_tokens[r.req_id] != 1
    ]
    if prefill_new:
        assert not _decode_ids(scheduler_output), (prefill_new, scheduler_output)


def _kv_out(
    *,
    finished_recving: set[str] | None = None,
    invalid_block_ids: set[int] | None = None,
    free_state_slots: int | None = None,
) -> KVConnectorOutput:
    out = KVConnectorOutput(finished_recving=finished_recving)
    if invalid_block_ids:
        out.invalid_block_ids = set(invalid_block_ids)
    if free_state_slots is not None:
        out.kv_connector_worker_meta = _StubWorkerMeta(free_state_slots)
    return out


def _count_base_schedules(monkeypatch) -> list[int]:
    calls: list[int] = []
    original = AsyncScheduler.schedule

    def counting(self, throttle_prefills=False):
        calls.append(1)
        return original(self, throttle_prefills)

    monkeypatch.setattr(AsyncScheduler, "schedule", counting)
    return calls


# ---------------------------------------------------------------------------
# 8.3: metadata exactly once, admission descriptor with a decode running


def test_build_connector_meta_exactly_once_per_step_on_every_path(monkeypatch):
    scheduler, connector = _pd_scheduler(num_blocks=PRESSURE_BLOCKS)
    base_calls = _count_base_schedules(monkeypatch)

    # Path 1: prefill-only pass returning tokens (plain request).
    scheduler.add_request(_request("plain"))
    out = _step(scheduler)
    assert _new_ids(out) == ["plain"]
    assert (
        connector.meta_calls == 1 and out.kv_connector_metadata is connector.metas[-1]
    )

    # Path 2: natural pass (running decode, remote admission only).
    scheduler.add_request(_request("remote", remote=True))
    out = _step(scheduler)
    assert (
        connector.meta_calls == 2 and out.kv_connector_metadata is connector.metas[-1]
    )
    assert len(base_calls) == 2, "the natural pass is a single base schedule()"

    # Path 3: default-mode fallback (prefill-only yields 0 tokens under KV
    # pressure, decode-only pass returned): the discarded pass must not have
    # built (and drained) metadata.
    scheduler.add_request(_request("starved", prompt_len=STARVED))
    out = _step(scheduler)
    assert scheduler.requests["starved"].status == RequestStatus.WAITING
    assert (
        connector.meta_calls == 3 and out.kv_connector_metadata is connector.metas[-1]
    )
    assert len(base_calls) == 4, "fallback = two base passes, one metadata"


def test_admission_descriptor_reaches_the_output_while_a_decode_runs():
    scheduler, connector = _pd_scheduler()
    _admit_decodes(scheduler, ["d0"])

    scheduler.add_request(_request("remote", remote=True))
    out = scheduler.schedule()

    # One output: the running decode AND the 0-token remote admission.
    assert _decode_ids(out) == ["d0"] and out.total_num_scheduled_tokens == 1
    request = scheduler.requests["remote"]
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.num_computed_tokens == PROMPT - 1
    assert "remote" in out.kv_connector_metadata.reqs_to_recv
    assert out.kv_connector_metadata.reqs_to_recv["remote"]["num_tokens"] == PROMPT - 1
    assert connector.free_slots == 7


def test_finished_and_preempted_ids_survive_the_discarded_prefill_pass(monkeypatch):
    monkeypatch.setenv("TT_SCHED_ASSERT_NO_DROPPED_SIDE_EFFECTS", "1")
    scheduler, connector = _pd_scheduler(num_blocks=PRESSURE_BLOCKS)
    _admit_decodes(scheduler, ["d0", "d1"])
    # d1 finishes between steps (client abort): its id must reach the worker.
    scheduler.finish_requests("d1", RequestStatus.FINISHED_ABORTED)
    # A local prefill the pool cannot hold forces the empty prefill-only pass.
    scheduler.add_request(_request("starved", prompt_len=STARVED))

    out = scheduler.schedule()

    assert _decode_ids(out) == ["d0"], "fallback decode-only pass"
    assert scheduler.requests["starved"].status == RequestStatus.WAITING
    assert "d1" in out.finished_req_ids, "lost finished id = leaked runner slot"
    assert connector.meta_calls == 2


# ---------------------------------------------------------------------------
# 8.3: the natural pass and the partition


def test_natural_pass_schedules_running_decodes_with_the_promoted_request():
    scheduler, _ = _pd_scheduler()
    _admit_decodes(scheduler, ["d0", "d1"])
    scheduler.add_request(_request("remote", remote=True))
    # Step k: admission (0 tokens) with the decodes; the worker imports the
    # K/V blocks at step-end and reports finished_recving.
    out_k = _step(scheduler, _kv_out(finished_recving={"remote"}))
    assert _decode_ids(out_k) == ["d0", "d1"] and _new_ids(out_k) == []

    # Step k+1: promotion + 1-token continuation TOGETHER with the decodes.
    out = scheduler.schedule()
    assert _new_ids(out) == ["remote"]
    (new_req,) = out.scheduled_new_reqs
    assert new_req.num_computed_tokens == PROMPT - 1
    assert out.num_scheduled_tokens["remote"] == 1
    assert _decode_ids(out) == ["d0", "d1"]
    assert out.total_num_scheduled_tokens == 3
    scheduler.update_from_output(out, _runner_output(out))
    assert scheduler.requests["remote"].status == RequestStatus.RUNNING
    assert scheduler.requests["remote"].num_computed_tokens == PROMPT


def test_prefill_only_pass_hides_remote_class_and_decode_only_pass_keeps_them():
    scheduler, connector = _pd_scheduler(num_blocks=PRESSURE_BLOCKS)
    _admit_decodes(scheduler, ["d0"])
    scheduler.add_request(_request("remote", remote=True))
    scheduler.add_request(_request("plain"))

    # Pending local prefill -> prefill-only pass: admits the plain request,
    # never the remote one (it would be a 0-token admission in a prefill step
    # whose descriptor the worker executes as a prefill batch).
    out = scheduler.schedule()
    assert _new_ids(out) == ["plain"] and _decode_ids(out) == []
    assert scheduler.requests["remote"].status == RequestStatus.WAITING
    assert out.kv_connector_metadata.reqs_to_recv == {}
    assert len(scheduler.waiting) == 1, "the hidden remote request is restored"
    scheduler.update_from_output(out, _runner_output(out))

    # Now a starved local prefill + running decodes -> decode-only fallback:
    # the remote request is admitted in that pass.
    scheduler.add_request(_request("starved", prompt_len=STARVED))
    out = scheduler.schedule()
    assert sorted(_decode_ids(out)) == ["d0", "plain"]
    assert scheduler.requests["remote"].status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert "remote" in out.kv_connector_metadata.reqs_to_recv
    assert scheduler.requests["starved"].status == RequestStatus.WAITING
    assert len(scheduler.waiting) == 1 and len(scheduler.skipped_waiting) == 1


def test_deferred_remote_request_causes_no_double_pass(monkeypatch):
    scheduler, connector = _pd_scheduler(free_slots=0)
    _admit_decodes(scheduler, ["d0"])
    base_calls = _count_base_schedules(monkeypatch)
    scheduler.add_request(_request("remote", remote=True))

    out = scheduler.schedule()

    # (None, False): deferred, stays remote class, joins the natural pass.
    assert len(base_calls) == 1
    assert _decode_ids(out) == ["d0"] and _new_ids(out) == []
    assert scheduler.requests["remote"].status == RequestStatus.WAITING
    assert scheduler.requests["remote"].num_computed_tokens == 0
    assert connector.demoted == []
    assert len(scheduler.skipped_waiting) == 1
    scheduler.update_from_output(out, _runner_output(out, _kv_out(free_state_slots=7)))
    assert connector.free_slots == 7

    # Once a slot is free the same request is admitted, still one pass.
    out = scheduler.schedule()
    assert len(base_calls) == 2
    assert scheduler.requests["remote"].status == RequestStatus.WAITING_FOR_REMOTE_KVS


def test_rejected_inside_decode_pass_is_demoted_then_prefilled_locally():
    scheduler, connector = _pd_scheduler(reject={"remote"})
    _admit_decodes(scheduler, ["d0", "d1"])
    scheduler.add_request(_request("remote", remote=True))

    # Natural pass: the stub demotes -> (None, False) -> not scheduled here.
    out = _step(scheduler)
    _assert_never_mixed(out, scheduler)
    assert _decode_ids(out) == ["d0", "d1"] and _new_ids(out) == []
    assert connector.demoted == ["remote"]
    request = scheduler.requests["remote"]
    assert request.status == RequestStatus.WAITING
    assert request.kv_transfer_params["do_remote_prefill"] is False

    # Plain class now: the next prefill-only pass admits it as a T-token
    # local prefill, alone (no decode rows next to it).
    out = _step(scheduler)
    _assert_never_mixed(out, scheduler)
    assert _new_ids(out) == ["remote"] and _decode_ids(out) == []
    assert out.num_scheduled_tokens["remote"] == PROMPT
    assert request.status == RequestStatus.RUNNING


@pytest.mark.parametrize("recompute", [True, False], ids=["recompute", "fail"])
def test_failed_load_with_decodes_running(recompute):
    scheduler, connector = _pd_scheduler(recompute=recompute)
    _admit_decodes(scheduler, ["d0", "d1", "d2"])
    scheduler.add_request(_request("remote", remote=True))
    out = scheduler.schedule()
    request = scheduler.requests["remote"]
    assert request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    blocks = out.kv_connector_metadata.reqs_to_recv["remote"]["local_block_ids"]
    free_before = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()

    # Worker: the load FAILED -> the id and ALL its blocks in the same output.
    scheduler.update_from_output(
        out,
        _runner_output(
            out,
            _kv_out(finished_recving={"remote"}, invalid_block_ids=set(blocks)),
        ),
    )

    if not recompute:
        # ``fail``: finished with FINISHED_ERROR and freed in that update.
        assert request.status == RequestStatus.FINISHED_ERROR
        assert "remote" not in scheduler.requests
        assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == (
            free_before + len(blocks)
        )
        return

    # ``recompute``: fully invalidated.
    assert request.num_computed_tokens == 0
    assert "remote" in scheduler.failed_recving_kv_req_ids
    assert "remote" in scheduler.finished_recving_kv_req_ids

    # Next natural pass: the base loop promotes it (frees the blocks, leaves
    # num_computed_tokens == 0) and falls through into the admission body; the
    # connector demotes (B2) so NO T-token prefill joins the decodes.
    out = _step(scheduler)
    _assert_never_mixed(out, scheduler)
    assert _new_ids(out) == [] and sorted(_decode_ids(out)) == ["d0", "d1", "d2"]
    assert connector.demoted == ["remote"]
    assert request.status == RequestStatus.WAITING
    assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == (
        free_before + len(blocks)
    )

    # Following prefill-only pass: a full local prefill of T tokens, alone.
    out = _step(scheduler)
    _assert_never_mixed(out, scheduler)
    assert _new_ids(out) == ["remote"] and _decode_ids(out) == []
    assert out.num_scheduled_tokens["remote"] == PROMPT
    assert request.status == RequestStatus.RUNNING


def test_queue_swaps_move_entries_never_copy_them():
    """50 alternating prefill-only / decode-only passes with deferred remote
    requests and a starved local prefill interleaved: every id appears exactly
    once across waiting + skipped_waiting (G3)."""
    scheduler, connector = _pd_scheduler(free_slots=0, num_blocks=PRESSURE_BLOCKS)
    _admit_decodes(scheduler, ["d0"])
    ids = []
    for i in range(3):
        ids.append(f"remote{i}")
        scheduler.add_request(_request(ids[-1], remote=True))
        ids.append(f"starved{i}")
        scheduler.add_request(_request(ids[-1], prompt_len=STARVED))

    for _ in range(50):
        out = _step(scheduler)
        _assert_never_mixed(out, scheduler)
        assert _decode_ids(out) == ["d0"] and _new_ids(out) == []
        queued = [r.request_id for r in scheduler.waiting] + [
            r.request_id for r in scheduler.skipped_waiting
        ]
        assert sorted(queued) == sorted(ids), queued
    assert connector.demoted == []


def test_liveness_remote_only_pass_unblocks_a_starved_local_prefill():
    """C3: no running decode, a local prefill starved by KV pressure, two
    remote-class requests holding blocks -> the remote-only fallback pass
    promotes them, they run to completion, the local prefill is admitted."""
    # Pool: 24 blocks (one is the null block). Two 32-token remote loads pin
    # 2 blocks each (+1 for the continuation); the local prefill needs 20.
    scheduler, connector = _pd_scheduler(num_blocks=PRESSURE_BLOCKS)
    scheduler.add_request(_request("remote0", remote=True, max_tokens=1))
    scheduler.add_request(_request("remote1", remote=True, max_tokens=1))
    out = _step(scheduler)  # natural pass: two 0-token admissions
    assert all(
        scheduler.requests[r].status == RequestStatus.WAITING_FOR_REMOTE_KVS
        for r in ("remote0", "remote1")
    )
    scheduler.add_request(_request("local", prompt_len=20 * BLOCK_SIZE - 4))

    # No running decode: the prefill-only pass is empty (KV pressure) and the
    # remote-only fallback pass must still include the remote requests.
    out = _step(scheduler, _kv_out(finished_recving={"remote0", "remote1"}))
    assert out.total_num_scheduled_tokens == 0
    assert scheduler.requests["local"].status == RequestStatus.WAITING
    # Both loads done -> promoted and decoded (max_tokens=1 -> they finish).
    out = _step(scheduler)
    assert sorted(_new_ids(out)) == ["remote0", "remote1"]
    assert all(out.num_scheduled_tokens[r] == 1 for r in _new_ids(out))
    assert "remote0" not in scheduler.requests and "remote1" not in scheduler.requests
    # Their blocks are free: the local prefill is admitted.
    out = _step(scheduler)
    assert _new_ids(out) == ["local"]
    assert out.num_scheduled_tokens["local"] == 20 * BLOCK_SIZE - 4


def test_promotion_waits_for_a_block_and_the_row_joins_only_then():
    """A KV_DONE request whose 1-token continuation needs a KV block the pool
    cannot supply (``(T-1) % block_size == 0``) is promoted to WAITING with
    ``num_computed_tokens == T-1`` and stays there, remote class and a slot
    holder, until a decode finishes; it appears in ``scheduled_new_reqs``
    exactly once, in the step whose batch it joins (the runner installs its
    GDN row at that step's begin, never earlier: I11). The other way into this
    state, ``running == max_num_seqs``, is closed by AM1: a remote holder's
    seat cannot be taken by a local prefill.

    Pool arithmetic (23 usable blocks): 5 decodes of 47 prompt tokens hold 3
    blocks each and cross into a 4th at the promotion step; load A pins 2
    (T-1 = 32) and needs 1 more to continue; a filler load B pins 1 and never
    finishes loading. 15 + 2 + 1 + 5 = 23, so A's block is gone exactly when
    A is promoted.
    """
    scheduler, connector = _pd_scheduler(num_blocks=PRESSURE_BLOCKS)
    pool = scheduler.kv_cache_manager.block_pool
    decodes = [f"d{i}" for i in range(5)]
    for req_id in decodes:
        scheduler.add_request(_request(req_id, prompt_len=3 * BLOCK_SIZE - 1))
    _step(scheduler)
    assert pool.get_num_free_blocks() == 8
    scheduler.add_request(_request("A", remote=True, prompt_len=2 * BLOCK_SIZE + 1))
    scheduler.add_request(_request("B", remote=True, prompt_len=BLOCK_SIZE + 1))
    out = _step(scheduler, _kv_out(finished_recving={"A"}))  # both admitted
    assert sorted(out.kv_connector_metadata.reqs_to_recv) == ["A", "B"]
    assert pool.get_num_free_blocks() == 5
    request = scheduler.requests["A"]

    for _ in range(3):
        out = _step(scheduler)
        assert pool.get_num_free_blocks() == 0
        assert _new_ids(out) == [], "no block: the promoted row must not join"
        assert sorted(_decode_ids(out)) == decodes
        # Promoted (status WAITING, T-1 computed) -> still remote class and a
        # slot holder for the prefill-only capacity count.
        assert request.status == RequestStatus.WAITING
        assert request.num_computed_tokens == 2 * BLOCK_SIZE
        assert TTScheduler._is_remote_class(request)
        assert TTScheduler._holds_remote_slot(request)
        assert scheduler.requests["B"].status == RequestStatus.WAITING_FOR_REMOTE_KVS

    scheduler.finish_requests("d1", RequestStatus.FINISHED_ABORTED)
    out = _step(scheduler)
    assert _new_ids(out) == ["A"] and out.num_scheduled_tokens["A"] == 1
    assert sorted(_decode_ids(out)) == ["d0", "d2", "d3", "d4"]
    assert request.status == RequestStatus.RUNNING


# ---------------------------------------------------------------------------
# AM1: decode-node slot capacity counts remote slot holders


def test_am1_local_prefill_waits_while_remote_holders_fill_the_seats():
    """7 running decodes + 1 remote load in flight + a param-less local prefill
    on an 8-seat node: the local prefill must wait. Admitting it would make the
    runner claim a 9th device state slot (``RuntimeError("no free device state
    slot")``, model_runner.py ``_alloc_prefill_state_slots``)."""
    scheduler, connector = _pd_scheduler(max_num_seqs=8)
    _admit_decodes(scheduler, [f"d{i}" for i in range(7)])
    scheduler.add_request(_request("remote", remote=True))
    out = _step(scheduler)  # 0-token admission with the 7 decodes
    assert scheduler.requests["remote"].status == RequestStatus.WAITING_FOR_REMOTE_KVS
    scheduler.add_request(_request("local"))

    for _ in range(3):
        out = _step(scheduler)
        _assert_never_mixed(out, scheduler)
        assert _new_ids(out) == [], "8th seat is the remote load's"
        assert len(_decode_ids(out)) == 7
        assert scheduler.requests["local"].status == RequestStatus.WAITING

    # The promoted-but-unadmitted state holds the seat too.
    out = _step(scheduler, _kv_out(finished_recving={"remote"}))
    remote = scheduler.requests["remote"]
    # Promotion needs running < max_num_seqs: 7 < 8 -> it joins the batch.
    out = scheduler.schedule()
    assert _new_ids(out) == ["remote"] and out.num_scheduled_tokens["remote"] == 1
    assert scheduler.requests["local"].status == RequestStatus.WAITING
    scheduler.update_from_output(out, _runner_output(out))
    assert remote.status == RequestStatus.RUNNING

    # A seat opens only when a request finishes.
    scheduler.finish_requests("d0", RequestStatus.FINISHED_ABORTED)
    out = _step(scheduler)
    assert _new_ids(out) == ["local"] and _decode_ids(out) == []


def test_am1_negative_control_plain_scheduler_would_have_admitted():
    """The same seats without the holder subtraction admit the local prefill
    (what a runner without AM1 would then trip over)."""
    scheduler, connector = _pd_scheduler(max_num_seqs=8)
    _admit_decodes(scheduler, [f"d{i}" for i in range(7)])
    scheduler.add_request(_request("remote", remote=True))
    _step(scheduler)
    scheduler.add_request(_request("local"))

    # Bypass the subtraction the way the pre-AM1 scheduler behaved.
    saved = TTScheduler._holds_remote_slot
    try:
        TTScheduler._holds_remote_slot = staticmethod(lambda request: False)
        out = scheduler.schedule()
    finally:
        TTScheduler._holds_remote_slot = saved
    assert _new_ids(out) == ["local"], "control: without AM1 the 9th claim happens"


def test_promoted_and_deferred_requests_are_classified_by_slot_ownership():
    waiting_remote = _request("a", remote=True)
    assert TTScheduler._is_remote_class(waiting_remote)
    assert not TTScheduler._holds_remote_slot(waiting_remote), "deferred: no slot"

    waiting_remote.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    assert TTScheduler._is_remote_class(waiting_remote)
    assert TTScheduler._holds_remote_slot(waiting_remote)

    promoted = _request("b", remote=True)
    promoted.kv_transfer_params["do_remote_prefill"] = False
    promoted.num_computed_tokens = PROMPT - 1
    assert TTScheduler._is_remote_class(promoted)
    assert TTScheduler._holds_remote_slot(promoted)

    demoted = _request("c", remote=True)
    demoted.kv_transfer_params.update(do_remote_prefill=False, _tt_demoted=True)
    assert not TTScheduler._is_remote_class(demoted)

    plain = _request("d")
    assert not TTScheduler._is_remote_class(plain)
    plain.status = RequestStatus.PREEMPTED
    assert not TTScheduler._is_remote_class(plain)


def test_existing_scheduler_behaviour_without_a_connector_is_unchanged():
    scheduler = _scheduler(max_num_seqs=8, num_blocks=64)
    assert scheduler.connector is None
    _admit_decodes(scheduler, ["d0"])
    scheduler.add_request(_request("plain"))
    out = scheduler.schedule()
    assert _new_ids(out) == ["plain"] and _decode_ids(out) == []
    assert out.kv_connector_metadata is None
