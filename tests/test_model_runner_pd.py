# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only tests for the runner/worker/platform side of prefill/decode
disaggregation (PHASE2_DESIGN 6.1, 6.2, 6.4; test list 8.4).

Fake runners as in ``tests/test_state_slots.py`` and
``tests/test_model_runner.py``; a fake worker-role connector records the
order of its hook calls. No device.
"""

from collections import deque
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
)
from vllm.v1.worker.gpu_input_batch import CachedRequestState

import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.input_batch import InputBatch
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.platform import TT_SCHEDULER_CLS, _validate_tt_kv_transfer_config

SLOTS = 8
VOCAB_SIZE = 64
BLOCK_SIZE = 16
MAX_MODEL_LEN = 64


class FakeConnector:
    """Worker-role connector recording the hook order (3.3)."""

    def __init__(self, *, finished=(None, None), invalid=(), worker_meta=None):
        self.calls: list[str] = []
        self.metadata = None
        self._finished = finished
        self._invalid = set(invalid)
        self._worker_meta = worker_meta
        self.runner = None

    def attach_runner(self, runner):
        self.runner = runner

    def bind_connector_metadata(self, metadata):
        self.metadata = metadata
        self.calls.append("bind")

    def has_connector_metadata(self):
        return self.metadata is not None

    def clear_connector_metadata(self):
        self.metadata = None
        self.calls.append("clear")

    def start_load_kv(self, forward_context, **kwargs):
        self.calls.append("start_load_kv")
        self.step_kwargs = kwargs

    def wait_for_save(self):
        self.calls.append("wait_for_save")

    def get_finished(self, finished_req_ids):
        self.calls.append("get_finished")
        return self._finished

    def get_block_ids_with_load_errors(self):
        return set(self._invalid)

    def build_connector_worker_meta(self):
        return self._worker_meta

    def get_kv_connector_stats(self):
        return None


def _bare_runner(**attrs) -> TTModelRunner:
    """A TTModelRunner with the real methods and only the attributes a test
    sets (no mesh device, no model); instance attributes shadow methods, so a
    test can stub ``build_model_input`` etc. per instance."""
    runner = TTModelRunner.__new__(TTModelRunner)
    for name, value in attrs.items():
        setattr(runner, name, value)
    return runner


def _slot_runner(slots=SLOTS):
    """The state-slot map plus the PD sets (see ``TTModelRunner.__init__``)."""
    return _bare_runner(
        tt_per_lane_max_num_seqs=slots,
        _req_state_slot={},
        _pending_state_slot_settle=None,
        requests={},
        _remote_loading={},
        _remote_ready=set(),
        _kv_connector=None,
    )


def _prefill(runner, row_req_ids):
    out = TTModelRunner._alloc_prefill_state_slots(runner, list(row_req_ids))
    runner.requests.update(dict.fromkeys(row_req_ids))
    return out


def _decode(runner, row_req_ids):
    remap = TTModelRunner._decode_state_slot_remap(runner, list(row_req_ids))
    TTModelRunner.note_decode_state_slots_settled(runner)
    return None if remap is None else remap.tolist()


def _so(*, finished=(), preempted=None, new=(), cached=()):
    out = SchedulerOutput.make_empty()
    out.finished_req_ids = set(finished)
    out.preempted_req_ids = preempted
    out.scheduled_new_reqs = [SimpleNamespace(req_id=r) for r in new]
    out.num_scheduled_tokens = {r: 1 for r in (*new, *cached)}
    out.total_num_scheduled_tokens = len(out.num_scheduled_tokens)
    return out


# ---------------------------------------------------------------------------
# Slot bookkeeping (I7)


def test_claim_takes_the_lowest_free_slot_and_never_raises():
    r = _slot_runner()
    assert _prefill(r, ["A", "B"]) == [0, 1]
    assert TTModelRunner.claim_remote_state_slot(r, "L") == 2
    assert r._remote_loading == {"L": 2} and r._req_state_slot["L"] == 2
    assert TTModelRunner.held_state_slots(r, include_loading=True) == {0, 1, 2}
    assert TTModelRunner.held_state_slots(r, include_loading=False) == {0, 1}
    # Exhaustion: None, no exception, nothing recorded.
    for i in range(5):
        assert TTModelRunner.claim_remote_state_slot(r, f"M{i}") == 3 + i
    assert TTModelRunner.claim_remote_state_slot(r, "overflow") is None
    assert "overflow" not in r._req_state_slot
    # A second claim for a live id is a bookkeeping error.
    with pytest.raises(RuntimeError, match="already holds slot"):
        TTModelRunner.claim_remote_state_slot(r, "L")


def test_held_loading_slot_blocks_a_concurrent_local_prefill():
    """Today's predicate (``req_id in self.requests``) would hand the loading
    slot to a local prefill; a load's claim must count as held (6.2)."""
    r = _slot_runner()
    assert TTModelRunner.claim_remote_state_slot(r, "L") == 0
    assert _prefill(r, ["P0", "P1"]) == [1, 2], "slot 0 is the load's"
    TTModelRunner.mark_remote_ready(r, "L")
    assert r._remote_loading == {} and r._remote_ready == {"L"}
    assert _prefill(r, ["P2"]) == [3], "a KV_DONE row still holds its slot"


def test_am1_without_the_scheduler_guard_the_runner_would_raise():
    """7 live decodes + 1 loading claim + 1 local prefill on 8 slots: the
    exhaustion the AM1 scheduler rule keeps unreachable."""
    r = _slot_runner()
    _prefill(r, [f"d{i}" for i in range(7)])
    assert TTModelRunner.claim_remote_state_slot(r, "L") == 7
    with pytest.raises(RuntimeError, match="no free device state slot"):
        TTModelRunner._alloc_prefill_state_slots(r, ["local"])


def test_claim_made_before_the_remap_survives_a_non_identity_gather():
    """A claim recorded before ``_prepare_model_inputs`` rides ``moved``; after
    settle ``remote_slot_of`` reports the moved slot (why step-begin must
    precede the decode remap, 6.2 (a))."""
    r = _slot_runner()
    r._req_state_slot.update({"A": 3, "B": 1})
    r.requests.update(dict.fromkeys(["A", "B"]))
    assert TTModelRunner.claim_remote_state_slot(r, "L") == 0
    remap = TTModelRunner._decode_state_slot_remap(r, ["A", "B"])
    assert remap.tolist() == [3, 1, 0, 2, 4, 5, 6, 7]
    assert r._pending_state_slot_settle["L"] == 2, "the claim is in the pending map"
    TTModelRunner.note_decode_state_slots_settled(r)
    assert TTModelRunner.remote_slot_of(r, "L") == 2
    # The claim vanished (abort race): loud, not a stale slot.
    TTModelRunner.release_remote_slot(r, "L")
    with pytest.raises(RuntimeError, match="no device state slot claim"):
        TTModelRunner.remote_slot_of(r, "L")


def test_finished_and_preempted_ids_release_claims_and_remote_sets():
    r = _slot_runner()
    TTModelRunner.claim_remote_state_slot(r, "L")
    TTModelRunner.claim_remote_state_slot(r, "R")
    TTModelRunner.mark_remote_ready(r, "R")
    _prefill(r, ["P"])
    TTModelRunner._release_dead_state_slots(
        r, _so(finished=["L"], preempted={"R", "P"})
    )
    assert (
        r._req_state_slot == {} and r._remote_loading == {} and r._remote_ready == set()
    )
    # Idempotent: a second release (the worker's) is harmless.
    TTModelRunner.release_remote_slot(r, "L")


# ---------------------------------------------------------------------------
# Step hooks (I5, I11)


def _hook_runner(connector, *, ready=(), pending=0):
    r = _slot_runner()
    r._kv_connector = connector
    r._remote_ready = set(ready)
    r._pending_kv_outputs = deque([KVConnectorOutput()] * pending)
    r._step_finished_ids = set()
    r._step_join_ids = set()
    return r


def test_step_begin_computes_join_ids_binds_metadata_and_starts_loads():
    connector = FakeConnector()
    r = _hook_runner(connector, ready={"R", "S"})
    so = _so(finished=["X"], new=["R", "P"], cached=["d0"])
    so.kv_connector_metadata = object()

    TTModelRunner._kv_connector_step_begin(r, so)

    assert r._step_join_ids == {"R"}, "only remote-ready rows of THIS batch"
    assert r._step_finished_ids == {"X"}
    assert connector.metadata is so.kv_connector_metadata
    assert connector.calls == ["bind", "start_load_kv"]
    assert connector.step_kwargs == {
        "finished_req_ids": {"X"},
        "join_req_ids": {"R"},
        "num_scheduled_tokens": 3,
    }


def test_step_begin_refuses_missing_metadata_and_a_stale_pending_output():
    connector = FakeConnector()
    r = _hook_runner(connector)
    with pytest.raises(RuntimeError, match="kv_connector_metadata"):
        TTModelRunner._kv_connector_step_begin(r, _so())
    so = _so()
    so.kv_connector_metadata = object()
    r = _hook_runner(connector, pending=2)
    with pytest.raises(RuntimeError, match="pending at step-begin"):
        TTModelRunner._kv_connector_step_begin(r, so)


def test_step_end_reports_finished_failed_blocks_and_meta_in_one_output():
    meta = SimpleNamespace(free_state_slots=5)
    connector = FakeConnector(
        finished=(None, {"F", "K"}), invalid={4, 5}, worker_meta=meta
    )
    r = _hook_runner(connector)
    connector.metadata = object()

    out = TTModelRunner._kv_connector_step_end(r, _so(finished=["X"]))

    assert connector.calls == ["wait_for_save", "get_finished", "clear"]
    assert out.finished_recving == {"F", "K"} and out.finished_sending is None
    assert out.invalid_block_ids == {4, 5}, "same output as the failed id (I8)"
    assert out.kv_connector_worker_meta is meta
    assert connector.metadata is None


def test_zero_token_step_returns_the_connector_output_only():
    """No forward: the step-end runs and its output is returned directly
    (EMPTY_MODEL_RUNNER_OUTPUT when nothing happened)."""
    connector = FakeConnector()
    r = _hook_runner(connector)
    r._pending_samples = deque()
    r.build_model_input = lambda so, grammar: None

    out = TTModelRunner._execute_model_with_kv_connector(r, _so())
    assert out is EMPTY_MODEL_RUNNER_OUTPUT
    assert r._pending_kv_outputs == deque() and r._pending_samples == deque()

    connector = FakeConnector(finished=(None, {"F"}))
    r = _hook_runner(connector)
    r._pending_samples = deque()
    r.build_model_input = lambda so, grammar: None
    out = TTModelRunner._execute_model_with_kv_connector(r, _so())
    assert out is not EMPTY_MODEL_RUNNER_OUTPUT
    assert out.kv_connector_output.finished_recving == {"F"}
    assert EMPTY_MODEL_RUNNER_OUTPUT.kv_connector_output is None, "singleton intact"


def test_forward_step_queues_the_connector_output_and_sample_tokens_attaches_it():
    connector = FakeConnector(finished=({"S"}, None))
    r = _hook_runner(connector)
    r._pending_samples = deque()
    order: list[str] = []
    r.build_model_input = lambda so, grammar: order.append("build") or "input"
    r._forward_with_model_input = lambda mi: order.append("forward") or "fwd"

    def finish(grammar_output, *, fwd):
        order.append("sample")
        return ModelRunnerOutput(
            req_ids=["d0"], req_id_to_index={"d0": 0}, sampled_token_ids=[[1]]
        )

    r._finish_front_packed_sync = finish

    assert TTModelRunner._execute_model_with_kv_connector(r, _so(cached=["d0"])) is None
    assert order == ["build", "forward"]
    assert connector.calls == ["wait_for_save", "get_finished", "clear"], (
        "step-end after the forward returned"
    )
    assert len(r._pending_samples) == 1 and len(r._pending_kv_outputs) == 1

    out = TTModelRunner.sample_tokens(r, None)
    assert order == ["build", "forward", "sample"]
    assert out.kv_connector_output.finished_sending == {"S"}
    assert r._pending_kv_outputs == deque()


def test_sample_tokens_copies_the_empty_singleton_before_attaching():
    r = _hook_runner(FakeConnector())
    kv_output = KVConnectorOutput(finished_recving={"F"})
    r._pending_kv_outputs = deque([kv_output])
    r._pending_samples = deque([lambda grammar_output: EMPTY_MODEL_RUNNER_OUTPUT])

    out = TTModelRunner.sample_tokens(r, None)

    assert out is not EMPTY_MODEL_RUNNER_OUTPUT
    assert out.kv_connector_output is kv_output
    assert EMPTY_MODEL_RUNNER_OUTPUT.kv_connector_output is None


def test_sample_tokens_leaves_the_output_alone_when_nothing_happened():
    r = _hook_runner(FakeConnector())
    r._pending_kv_outputs = deque([KVConnectorOutput()])
    plain = ModelRunnerOutput(req_ids=["d0"], req_id_to_index={"d0": 0})
    r._pending_samples = deque([lambda grammar_output: plain])
    out = TTModelRunner.sample_tokens(r, None)
    assert out is plain and out.kv_connector_output is None
    assert r._pending_kv_outputs == deque()


def test_execute_model_exception_clears_both_fifos():
    connector = FakeConnector()
    r = _hook_runner(connector)
    r._pending_samples = deque([object()])
    r._pending_kv_outputs = deque([KVConnectorOutput()])
    connector.metadata = object()

    def boom(so, grammar):
        raise ValueError("device")

    r.build_model_input = boom
    with pytest.raises(ValueError, match="device"):
        TTModelRunner._execute_model_with_kv_connector(r, _so(cached=["d0"]))
    assert r._pending_samples == deque() and r._pending_kv_outputs == deque()
    assert connector.metadata is None


# ---------------------------------------------------------------------------
# The remote-KV continuation dispatches through the decode branch (6.2)


def _batch(reqs):
    """``reqs``: (req_id, prompt_len, num_computed_tokens, output_len)."""
    batch = InputBatch(
        max_num_reqs=SLOTS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=VOCAB_SIZE,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )
    states = {}
    for req_id, prompt_len, num_computed, output_len in reqs:
        state = CachedRequestState(
            req_id=req_id,
            prompt_token_ids=list(range(1, prompt_len + 1)),
            mm_features=None,
            sampling_params=SamplingParams(temperature=0.0),
            generator=None,
            block_ids=([0, 1],),
            num_computed_tokens=num_computed,
            output_token_ids=list(range(50, 50 + output_len)),
        )
        batch.add_request(state)
        states[req_id] = state
    return batch, states


def _prepare_runner(batch, states, *, connector=None, ready=(), join=None):
    runner = _bare_runner(
        input_batch=batch,
        requests=dict(states),
        _output_tokens_per_step=1,
        _is_adaptive_block_output=False,
        tt_per_lane_max_num_seqs=SLOTS,
        tt_data_parallel_size=1,
        max_num_blocks_per_req=MAX_MODEL_LEN // BLOCK_SIZE,
        model_config=SimpleNamespace(is_multimodal_model=False),
        check_perform_device_sampling=lambda **_: False,
        _block_tables_per_layer=lambda _: None,
        _sampling_params_for_padded_decode=lambda params, req_indices, n: params,
        _decode_layout_changed_since_last_decode=False,
        _build_host_generators=TTModelRunner._build_host_generators,
        _req_state_slot={},
        _pending_state_slot_settle=None,
        _remote_loading={},
        _remote_ready=set(ready),
        _kv_connector=connector,
    )
    runner._step_join_ids = set(ready) if join is None else set(join)
    return runner


def _cached(rows):
    """``rows``: (req_id, num_computed, num_output) running decodes."""
    return CachedRequestData(
        req_ids=[r for r, *_ in rows],
        resumed_req_ids=set(),
        new_token_ids=[[] for _ in rows],
        all_token_ids={},
        new_block_ids=[None for _ in rows],
        num_computed_tokens=[c for _, c, _ in rows],
        num_output_tokens=[o for *_, o in rows],
    )


def test_remote_ready_new_request_runs_as_a_decode_row_on_its_claimed_slot():
    T = 20
    batch, states = _batch([("d0", 8, 8, 1), ("R", T, T - 1, 0)])
    runner = _prepare_runner(batch, states, connector=FakeConnector(), ready={"R"})
    runner._req_state_slot["d0"] = 0
    assert TTModelRunner.claim_remote_state_slot(runner, "R") == 1
    so = _so(new=["R"], cached=["d0"])
    so.scheduled_cached_reqs = _cached([("d0", 8, 1)])

    model_input = TTModelRunner._prepare_model_inputs(runner, so, None)

    assert model_input.prompt_lens is None, "decode branch, not prefill"
    assert model_input.input_positions.tolist()[:2] == [8, T - 1]
    assert model_input.input_tokens[1, 0] == T, "prompt[T-1]"
    assert model_input.prefill_empty_slots is None
    assert model_input.slot_remap is None, "claimed slot 1 == row 1: identity"
    assert runner._remote_ready == set(), "consumed: installed this step"
    assert runner._req_state_slot["R"] == 1


def test_remote_ready_row_at_a_different_slot_resolves_through_the_remap():
    T = 20
    batch, states = _batch([("d0", 8, 8, 1), ("R", T, T - 1, 0)])
    runner = _prepare_runner(batch, states, connector=FakeConnector(), ready={"R"})
    runner._req_state_slot["d0"] = 3  # the decode's state sits off-row
    assert TTModelRunner.claim_remote_state_slot(runner, "R") == 0
    so = _so(new=["R"], cached=["d0"])
    so.scheduled_cached_reqs = _cached([("d0", 8, 1)])

    model_input = TTModelRunner._prepare_model_inputs(runner, so, None)

    assert model_input.slot_remap.tolist()[:2] == [3, 0], "row 1 (R) reads slot 0"
    TTModelRunner.note_decode_state_slots_settled(runner)
    assert TTModelRunner.remote_slot_of(runner, "R") == 1


def test_remote_ready_row_not_installed_this_step_is_refused():
    T = 20
    batch, states = _batch([("R", T, T - 1, 0)])
    runner = _prepare_runner(
        batch, states, connector=FakeConnector(), ready={"R"}, join=set()
    )
    TTModelRunner.claim_remote_state_slot(runner, "R")
    with pytest.raises(RuntimeError, match="installed at step-begin"):
        TTModelRunner._prepare_model_inputs(runner, _so(new=["R"]), None)


def test_remote_ready_row_must_be_a_t_minus_one_continuation():
    T = 20
    batch, states = _batch([("R", T, T - 2, 0)])
    runner = _prepare_runner(batch, states, connector=FakeConnector(), ready={"R"})
    TTModelRunner.claim_remote_state_slot(runner, "R")
    with pytest.raises(RuntimeError, match="not a T-1 continuation"):
        TTModelRunner._prepare_model_inputs(runner, _so(new=["R"]), None)


def test_remote_new_next_to_a_plain_new_row_raises():
    batch, states = _batch([("R", 20, 19, 0), ("P", 8, 0, 0)])
    runner = _prepare_runner(batch, states, connector=FakeConnector(), ready={"R"})
    TTModelRunner.claim_remote_state_slot(runner, "R")
    with pytest.raises(RuntimeError, match="must not share a step with prefill rows"):
        TTModelRunner._prepare_model_inputs(runner, _so(new=["R", "P"]), None)


def test_plain_prefill_next_to_decode_rows_raises_only_with_a_connector():
    """B2 belt-and-braces: the prefill branch over live decode rows would
    ``write_slot`` over their GDN state. Plain serving keeps today's behaviour
    (the batch is treated as all-prefill)."""
    batch, states = _batch([("d0", 8, 8, 1), ("P", 8, 0, 0)])
    so = _so(new=["P"], cached=["d0"])
    so.num_scheduled_tokens = {"d0": 1, "P": 8}
    so.total_num_scheduled_tokens = 9
    so.scheduled_cached_reqs = _cached([("d0", 8, 1)])

    runner = _prepare_runner(batch, states, connector=FakeConnector())
    runner._req_state_slot["d0"] = 0
    with pytest.raises(RuntimeError, match="prefill rows .* next to decode rows"):
        TTModelRunner._prepare_model_inputs(runner, so, None)

    runner = _prepare_runner(batch, states, connector=None)
    runner._req_state_slot["d0"] = 0
    model_input = TTModelRunner._prepare_model_inputs(runner, so, None)
    assert model_input.prompt_lens is not None, "unchanged plain-serving rule"


def test_a_step_of_running_decodes_with_a_load_pending_installs_nothing():
    """I11: a pending (loading or KV_DONE) row is not in ``scheduled_new_reqs``,
    so ``_step_join_ids`` is empty and the decode batch does not touch it."""
    connector = FakeConnector()
    r = _hook_runner(connector, ready={"K"})
    r._remote_loading["L"] = 5
    so = _so(cached=["d0", "d1"])
    so.kv_connector_metadata = object()
    TTModelRunner._kv_connector_step_begin(r, so)
    assert r._step_join_ids == set()
    assert r._remote_ready == {"K"}, "still waiting for its join step"


# ---------------------------------------------------------------------------
# Platform guard (6.4)


def _pd_config(**overrides):
    cfg = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(
            scheduler_cls=TT_SCHEDULER_CLS, async_scheduling=False
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        additional_config={},
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_platform_guard_accepts_the_phase2_configuration(monkeypatch):
    monkeypatch.delenv("QWEN36_PREFILL_BUCKET_TRACE", raising=False)
    monkeypatch.delenv("QWEN36_PREFILL_BUCKET_EXTRA_BLOCK", raising=False)
    _validate_tt_kv_transfer_config(_pd_config(), is_lane_mode=False)


@pytest.mark.parametrize(
    ("kwargs", "env", "match"),
    [
        (dict(is_lane_mode=True), {}, "lane mode"),
        (
            dict(
                config=_pd_config(
                    scheduler_config=SimpleNamespace(
                        scheduler_cls="vllm_tt_plugin.lane_scheduler.TTLaneCoordinator",
                        async_scheduling=False,
                    )
                )
            ),
            {},
            "requires the TT scheduler",
        ),
        (
            dict(
                config=_pd_config(additional_config={"_tt_output_tokens_per_step": 8})
            ),
            {},
            "block-output",
        ),
        (
            dict(
                config=_pd_config(
                    cache_config=SimpleNamespace(enable_prefix_caching=True)
                )
            ),
            {},
            "prefix caching off",
        ),
        (
            dict(
                config=_pd_config(
                    scheduler_config=SimpleNamespace(
                        scheduler_cls=TT_SCHEDULER_CLS, async_scheduling=True
                    )
                )
            ),
            {},
            "synchronous scheduling",
        ),
        (dict(), {"QWEN36_PREFILL_BUCKET_TRACE": "0"}, "QWEN36_PREFILL_BUCKET_TRACE=1"),
        (
            dict(),
            {"QWEN36_PREFILL_BUCKET_EXTRA_BLOCK": "0"},
            "QWEN36_PREFILL_BUCKET_EXTRA_BLOCK",
        ),
    ],
    ids=[
        "lane",
        "scheduler_cls",
        "block_output",
        "prefix_caching",
        "async",
        "bucket_trace",
        "extra_block",
    ],
)
def test_platform_guard_rejects(kwargs, env, match, monkeypatch):
    monkeypatch.delenv("QWEN36_PREFILL_BUCKET_TRACE", raising=False)
    monkeypatch.delenv("QWEN36_PREFILL_BUCKET_EXTRA_BLOCK", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    config = kwargs.pop("config", None) or _pd_config()
    with pytest.raises(ValueError, match=match):
        _validate_tt_kv_transfer_config(
            config, is_lane_mode=kwargs.get("is_lane_mode", False)
        )


def test_block_output_config_key_is_what_the_guard_reads():
    """Sanity: the block-output check reads the stored output width."""
    from vllm_tt_plugin.config import get_tt_output_tokens_per_step

    assert get_tt_output_tokens_per_step(_pd_config()) == 1
    assert (
        get_tt_output_tokens_per_step(
            _pd_config(additional_config={"_tt_output_tokens_per_step": 8})
        )
        == 8
    )


def test_prepare_inputs_padding_uses_torch_int32_positions_for_remote_rows():
    """The remote row's position tensor stays int-typed like every decode row
    (the generator writes it into the traced position buffer)."""
    T = 20
    batch, states = _batch([("R", T, T - 1, 0)])
    runner = _prepare_runner(batch, states, connector=FakeConnector(), ready={"R"})
    TTModelRunner.claim_remote_state_slot(runner, "R")
    model_input = TTModelRunner._prepare_model_inputs(runner, _so(new=["R"]), None)
    assert model_input.input_positions.dtype in (torch.int32, torch.int64)
    assert model_input.input_positions.tolist() == [T - 1] + [-1] * (SLOTS - 1)
