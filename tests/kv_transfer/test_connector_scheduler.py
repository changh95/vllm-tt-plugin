# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only tests for the SCHEDULER role of ``TTKVConnector`` (PHASE2_DESIGN.md
3.1/3.2, step 8.1). Real vLLM ``Request``/``KVTransferConfig`` objects, fake
``VllmConfig``/``KVCacheConfig``/``KVCacheBlocks``/``SchedulerOutput``. No device.
"""

from __future__ import annotations

import ast
import copy
import pathlib
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorLogging
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus

from vllm_tt_plugin.kv_transfer import tt_connector as tc
from vllm_tt_plugin.kv_transfer.metadata import (
    XFER_ID_RE,
    RecvMeta,
    SaveMeta,
    TransferDescriptor,
    TTKVConnectorMetadata,
    TTKVWorkerMeta,
)

MODULE_PATH = "vllm_tt_plugin.kv_transfer.tt_connector"
LAYER_TYPES = ["full_attention"] * 16 + ["linear_attention"] * 48
BLOCK = 64
KV_CACHE_CONFIG = SimpleNamespace(
    kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=BLOCK))]
)


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
def make_vllm_config(
    engine_id,
    role,
    extra=None,
    *,
    max_num_seqs=8,
    layer_types=None,
    model="/w/weights_qwen38",
):
    layer_types = LAYER_TYPES if layer_types is None else layer_types
    ktc = KVTransferConfig(
        kv_connector="TTKVConnector",
        kv_connector_module_path=MODULE_PATH,
        kv_role=role,
        engine_id=engine_id,
        kv_connector_extra_config=dict(extra or {}),
    )
    hf = SimpleNamespace(
        layer_types=list(layer_types), num_hidden_layers=len(layer_types)
    )
    return SimpleNamespace(
        kv_transfer_config=ktc,
        model_config=SimpleNamespace(model=model, hf_config=hf, hf_text_config=hf),
        cache_config=SimpleNamespace(block_size=BLOCK, enable_prefix_caching=False),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )


def make_connector(role="kv_consumer", engine_id="d0", **kw):
    return tc.TTKVConnector(
        make_vllm_config(engine_id, role, **kw),
        KVConnectorRole.SCHEDULER,
        KV_CACHE_CONFIG,
    )


def make_request(rid, T, params=None, mm=False, max_tokens=16):
    extra = {"kv_transfer_params": params} if params is not None else None
    req = Request(
        rid,
        list(range(1, T + 1)),
        SamplingParams(max_tokens=max_tokens, extra_args=extra),
        None,
    )
    if mm:
        req.mm_features = [object()]
    return req


def blocks_of(ids):
    return SimpleNamespace(get_block_ids=lambda allow_none=False: (list(ids),))


def sched_output(new=()):
    return SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id=rid, block_ids=(list(ids),)) for rid, ids in new
        ],
        finished_req_ids=set(),
        total_num_scheduled_tokens=0,
    )


def p_params():
    return {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }


def produce(pc, rid, T, block_ids=None):
    """Run the whole P leg through ``pc``: admission (truncation), metadata, finish."""
    if block_ids is None:
        block_ids = list(range(100, 100 + -(-(T - 1) // BLOCK)))
    req = make_request(rid, T, p_params())
    assert pc.get_num_new_matched_tokens(req, 0) == (0, False)
    meta = pc.build_connector_meta(sched_output([(rid, block_ids)]))
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    req.num_computed_tokens = req.num_prompt_tokens
    delay, params = pc.request_finished(req, block_ids)
    return req, meta, delay, params


@pytest.fixture
def pair():
    """Producer p0 and consumer d0 with identical model config (same fingerprint)."""
    return make_connector("kv_producer", "p0"), make_connector("kv_consumer", "d0")


def consumer_offer(dc, rid, T, params):
    req = make_request(rid, T, copy.deepcopy(params))
    return req, dc.get_num_new_matched_tokens(req, 0)


# --------------------------------------------------------------------------- #
# construction / module hygiene
# --------------------------------------------------------------------------- #
def test_constructor_roles_and_config():
    c = make_connector(
        "kv_both", "x1", extra={"shm_mode": "raw", "max_inflight_loads": 3}
    )
    assert c.is_producer and c.is_consumer
    assert c.block_size == BLOCK and c.hybrid_state is True
    assert c.transport_descriptor == {"kind": "shm", "mode": "raw", "layout_version": 1}
    assert c.max_inflight_loads == 3 and c._free_slots_estimate == 8
    assert "/" not in str(c.transport_descriptor)  # no paths in the descriptor (I10)


def test_engine_id_validated():
    with pytest.raises(ValueError):
        make_connector("kv_consumer", "bad/id")
    with pytest.raises(ValueError):
        make_connector("kv_consumer", "x" * 33)


def test_hybrid_state_resolution():
    assert (
        tc._resolve_hybrid_state(make_vllm_config("a", "kv_consumer"), "auto") is True
    )
    assert (
        tc._resolve_hybrid_state(
            make_vllm_config("a", "kv_consumer", layer_types=["full_attention"] * 4),
            "auto",
        )
        is False
    )
    assert (
        tc._resolve_hybrid_state(make_vllm_config("a", "kv_consumer"), "false") is False
    )
    assert make_connector(extra={"hybrid_state": True}).hybrid_state is True
    with pytest.raises(ValueError):
        tc._resolve_hybrid_state(make_vllm_config("a", "kv_consumer"), "maybe")


def test_fingerprint_sensitivity(monkeypatch):
    base = tc._fingerprint(make_vllm_config("a", "kv_consumer"))
    assert base == tc._fingerprint(
        make_vllm_config("b", "kv_producer")
    )  # engine id irrelevant
    assert base != tc._fingerprint(
        make_vllm_config("a", "kv_consumer", layer_types=["full_attention"] * 64)
    )
    monkeypatch.setenv("QWEN_SDPA_BF8", "0")
    assert base != tc._fingerprint(make_vllm_config("a", "kv_consumer"))
    monkeypatch.setenv("QWEN_SDPA_BF8", "1")
    # QWEN36_GDN_DECODE_FUSED is not part of the default fingerprint (AM3 / M2)
    monkeypatch.setenv("QWEN36_GDN_DECODE_FUSED", "0")
    assert base == tc._fingerprint(make_vllm_config("a", "kv_consumer"))
    assert base != tc._fingerprint(
        make_vllm_config("a", "kv_consumer"), ["QWEN36_GDN_DECODE_FUSED"]
    )


def test_kv_connector_logging_accepts_class():
    ktc = make_vllm_config("d0", "kv_consumer").kv_transfer_config
    assert KVConnectorLogging(ktc).connector_cls is tc.TTKVConnector
    stats = tc.TTKVConnector.build_kv_connector_stats(None)
    assert stats is not None and stats.is_empty()
    stats.record_export("r", 12.0, 1000, True)
    other = tc.TTKVConnector.build_kv_connector_stats(
        {"export_ms": [3.0], "records": {"q": {"export_ms": 3.0}}}
    )
    stats.aggregate(other)
    red = stats.reduce()
    assert red["Exports"] == 2 and set(stats.data["records"]) == {"r", "q"}
    assert not stats.is_empty()
    old = stats.clone_and_reset()
    assert stats.is_empty() and not old.is_empty()


def test_module_import_has_no_ttnn_side_effect():
    pkg = pathlib.Path(tc.__file__).parent
    for name in (
        "__init__.py",
        "metadata.py",
        "hooks.py",
        "tt_connector.py",
        "worker.py",
    ):
        tree = ast.parse((pkg / name).read_text())
        for node in tree.body:  # module level only
            if isinstance(node, ast.Import):
                assert not any(a.name.split(".")[0] == "ttnn" for a in node.names), name
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "ttnn", name
    # And dynamically: purge + block ttnn AFTER vllm is loaded, then import the package.
    code = (
        "import sys, importlib.abc\n"
        "import vllm\n"
        "import vllm.distributed.kv_transfer.kv_connector.v1.base\n"
        "import vllm.distributed.kv_transfer.kv_connector.v1.metrics\n"
        "import vllm.v1.request, vllm.utils.math_utils, vllm.logger\n"
        "for k in [k for k in sys.modules if k == 'ttnn' or k.startswith('ttnn.')]:\n"
        "    del sys.modules[k]\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        if name == 'ttnn' or name.startswith('ttnn.'):\n"
        "            raise ImportError('ttnn blocked')\n"
        "sys.meta_path.insert(0, Block())\n"
        "import vllm_tt_plugin.kv_transfer.metadata, vllm_tt_plugin.kv_transfer.hooks\n"
        "import vllm_tt_plugin.kv_transfer.tt_connector\n"
        "import vllm_tt_plugin.kv_transfer.worker\n"
        "assert 'ttnn' not in sys.modules\n"
        "print('NO_TTNN_OK')\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert "NO_TTNN_OK" in out.stdout, out.stderr[-3000:]


# --------------------------------------------------------------------------- #
# producer
# --------------------------------------------------------------------------- #
def test_producer_truncates_once_and_records_save():
    pc = make_connector("kv_producer", "p0")
    req = make_request("a-p", 5, p_params())
    assert pc.get_num_new_matched_tokens(req, 0) == (0, False)
    assert req.prompt_token_ids == [1, 2, 3, 4] and len(req._all_token_ids) == 4
    assert req.num_prompt_tokens == 4 and req.num_tokens == 4 and req.max_tokens == 1
    assert req.kv_transfer_params["_p_side_truncated"] is True
    # re-offer after an allocate_slots failure: no second truncation, still recorded
    assert pc.get_num_new_matched_tokens(req, 0) == (0, False)
    assert req.num_prompt_tokens == 4 and pc._reqs_need_save == {"a-p": req}


def test_producer_without_hybrid_state_does_not_truncate():
    pc = make_connector("kv_producer", "p0", layer_types=["full_attention"] * 8)
    req = make_request("a-p", 5, p_params())
    assert pc.get_num_new_matched_tokens(req, 0) == (0, False)
    assert req.num_prompt_tokens == 5 and req.max_tokens == 16


def test_producer_multimodal_is_not_exported():
    pc = make_connector("kv_producer", "p0")
    req = make_request("mm-p", 5, p_params(), mm=True)
    assert pc.get_num_new_matched_tokens(req, 0) == (0, False)
    assert (
        req.kv_transfer_params["do_remote_decode"] is False
        and req.num_prompt_tokens == 5
    )
    assert "mm-p" not in pc._reqs_need_save
    req.status = RequestStatus.FINISHED_LENGTH_CAPPED
    assert pc.request_finished(req, [1]) == (
        False,
        None,
    )  # xfer_id None -> plain finish


def test_producer_T1_prompt_is_prefilled_whole():
    pc = make_connector("kv_producer", "p0")
    req = make_request("one-p", 1, p_params())
    assert pc.get_num_new_matched_tokens(req, 0) == (0, False)
    assert (
        req.num_prompt_tokens == 1 and "_p_side_truncated" not in req.kv_transfer_params
    )


def test_producer_request_finished_emits_params_and_arms_once():
    pc = make_connector("kv_producer", "p0")
    client_bytes = "../../etc/passwd -p"
    req, meta, delay, params = produce(pc, client_bytes, 130, block_ids=[7, 3, 9])
    assert meta.reqs_to_save == {
        client_bytes: SaveMeta(
            block_ids=[7, 3, 9], num_tokens=129, xfer_id=pc._xfer_id(client_bytes)
        )
    }
    assert meta.reqs_to_send == {} and meta.is_empty() is False
    assert delay is True
    assert params["do_remote_prefill"] is True and params["do_remote_decode"] is False
    assert (
        params["remote_engine_id"] == "p0"
        and params["remote_request_id"] == client_bytes
    )
    assert (
        params["remote_block_ids"] == [7, 3, 9] and params["remote_num_tokens"] == 129
    )
    assert params["remote_prompt_hash"] == tc._prompt_hash(list(range(1, 130)))
    assert params["remote_fingerprint"] == pc.fingerprint
    assert params["remote_transport"] == {
        "kind": "shm",
        "mode": "dumpfile",
        "layout_version": 1,
    }
    assert params["tt_layout_version"] == 1 and params["tt_chunk_tokens"] == 2048
    assert params["remote_blocks_expiry_time"] > time.time()
    xid = params["xfer_id"]
    assert (
        XFER_ID_RE.match(xid)
        and xid.startswith("p0:")
        and "etc" not in xid
        and "/" not in xid
    )
    assert TransferDescriptor.from_params(params).xfer_hex == xid.split(":")[1]
    # armed exactly once
    assert pc._reqs_to_arm == {client_bytes: params["remote_blocks_expiry_time"]}
    meta2 = pc.build_connector_meta(sched_output())
    assert set(meta2.reqs_to_send) == {client_bytes} and meta2.reqs_to_save == {}
    meta3 = pc.build_connector_meta(sched_output())
    assert meta3.reqs_to_send == {} and meta3.is_empty()
    assert client_bytes in pc._reqs_need_send
    pc.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=None,
            finished_recving=None,
            finished_sending={client_bytes},
        )
    )
    assert pc._reqs_need_send == {}


def test_producer_aborted_request_is_not_processed():
    pc = make_connector("kv_producer", "p0")
    req = make_request("ab-p", 10, p_params())
    pc.get_num_new_matched_tokens(req, 0)
    req.status = RequestStatus.FINISHED_ABORTED
    assert pc.request_finished(req, [1]) == (False, None)
    assert "ab-p" not in pc._reqs_need_save and pc._reqs_not_processed == {"ab-p"}
    meta = pc.build_connector_meta(sched_output())
    assert meta.reqs_not_processed == {"ab-p"} and pc._reqs_not_processed == set()
    assert pc._reqs_to_arm == {}


def test_producer_finished_stopped_is_accepted_and_asserts_token_count():
    pc = make_connector("kv_producer", "p0")
    req = make_request("s-p", 10, p_params())
    pc.get_num_new_matched_tokens(req, 0)
    req.status = RequestStatus.FINISHED_STOPPED
    req.num_computed_tokens = 5  # chunked prefill would produce this: forbidden
    with pytest.raises(RuntimeError):
        pc.request_finished(req, [1])
    req.num_computed_tokens = 9
    delay, params = pc.request_finished(req, [1])
    assert delay is True and params["remote_num_tokens"] == 9


def test_producer_ignores_update_state_after_alloc():
    pc = make_connector("kv_producer", "p0")
    req = make_request("u-p", 10, p_params())
    pc.get_num_new_matched_tokens(req, 0)
    pc.update_state_after_alloc(req, blocks_of([1]), 0)  # must be a no-op
    assert pc._reqs_need_recv == {} and pc._inflight == set()


# --------------------------------------------------------------------------- #
# consumer: admission
# --------------------------------------------------------------------------- #
def test_no_params_returns_0_false(pair):
    _, dc = pair
    req = make_request("plain", 100)
    assert dc.get_num_new_matched_tokens(req, 0) == (0, False)
    assert dc.request_finished(req, [1]) == (False, None)


def test_consumer_valid_params_return_T_minus_1_true(pair, monkeypatch):
    pc, dc = pair
    T = 130
    _, _, _, params = produce(pc, "v-p", T)
    calls = []
    real = tc._prompt_hash
    monkeypatch.setattr(tc, "_prompt_hash", lambda ids: (calls.append(1), real(ids))[1])
    req, res = consumer_offer(dc, "v", T, params)
    assert res == (T - 1, True)
    assert (
        req.kv_transfer_params["_tt_validated"] is True
        and req.kv_transfer_params["do_remote_prefill"] is True
    )
    assert dc._first_token_steps == {"v": 0}
    # pool-starved re-offer: validation cached (hash computed once), lease re-checked
    assert dc.get_num_new_matched_tokens(req, 0) == (T - 1, True)
    assert dc.get_num_new_matched_tokens(req, 0) == (T - 1, True)
    assert len(calls) == 1


def test_consumer_never_returns_0_true(pair):
    pc, dc = pair
    _, _, _, params = produce(pc, "z-p", 66)
    for T in (1, 2, 65, 66, 67):
        req, res = consumer_offer(dc, f"z{T}", T, params)
        assert res != (0, True)
        assert res in ((T - 1, True), (None, False))
        dc._to_release.clear()


def _mutate(name, params, T):
    p = copy.deepcopy(params)
    if name == "hash":
        p["remote_prompt_hash"] = "00" * 8
    elif name == "fingerprint":
        p["remote_fingerprint"] = "ff" * 16
    elif name == "num_tokens":
        p["remote_num_tokens"] = T
    elif name == "transport":
        p["remote_transport"] = {"kind": "shm", "mode": "raw", "layout_version": 1}
    elif name == "missing_key":
        del p["remote_fingerprint"]
    elif name == "layout":
        p["tt_layout_version"] = 2
    elif name == "lease":
        p["remote_blocks_expiry_time"] = time.time() - 1
    elif name == "xfer_id_path":
        p["xfer_id"] = "p0:../../etc"
    elif name == "xfer_id_engine":
        p["xfer_id"] = "q9:" + p["xfer_id"].split(":")[1]
    elif name == "mistyped":
        p["remote_num_tokens"] = "many"
    elif name == "chunk_tokens":
        p["tt_chunk_tokens"] = 4096  # NIT-3: demoted at admission, not at import
    return p


@pytest.mark.parametrize(
    "case",
    [
        "hash",
        "fingerprint",
        "num_tokens",
        "transport",
        "missing_key",
        "layout",
        "lease",
        "xfer_id_path",
        "xfer_id_engine",
        "mistyped",
        "chunk_tokens",
        "mm",
        "T1",
    ],
)
def test_consumer_demotes_on_bad_params(pair, case):
    pc, dc = pair
    T = 130
    _, _, _, params = produce(pc, "d-p", T)
    if case == "T1":
        _, _, _, params = produce(pc, "d1-p", 1)
        T = 1
    p = _mutate(case, params, T)
    req = make_request("d", T, p, mm=(case == "mm"))
    res = dc.get_num_new_matched_tokens(req, 0)
    assert res == (None, False)
    kp = req.kv_transfer_params
    assert kp["do_remote_prefill"] is False and kp["_tt_demoted"] is True
    assert dc._to_release == {p["xfer_id"]}
    assert "d" not in dc._first_token_steps and "d" not in dc._inflight
    # the following offer is a PLAIN local prefill
    assert dc.get_num_new_matched_tokens(req, 0) == (0, False)
    assert dc.get_num_new_matched_tokens(req, 0) == (0, False)
    # request_finished after demotion: no second release
    meta = dc.build_connector_meta(sched_output())
    assert meta.to_release == {p["xfer_id"]} and dc._to_release == set()
    req.status = RequestStatus.FINISHED_STOPPED
    assert dc.request_finished(req, [1]) == (False, None)
    assert dc._to_release == set()
    assert dc._sched_stats.data["num_demotions"] == [1]


def test_lease_rechecked_on_every_offer(pair):
    pc, dc = pair
    _, _, _, params = produce(pc, "l-p", 130)
    req, res = consumer_offer(dc, "l", 130, params)
    assert res == (129, True)
    req.kv_transfer_params["remote_blocks_expiry_time"] = time.time() - 0.001
    assert dc.get_num_new_matched_tokens(req, 0) == (None, False)
    assert req.kv_transfer_params["_tt_demoted"] is True
    assert dc.get_num_new_matched_tokens(req, 0) == (0, False)


def test_consumer_defers_without_demotion(pair):
    pc, dc = pair
    _, _, _, params = produce(pc, "f-p", 130)
    dc._free_slots_estimate = 0
    req, res = consumer_offer(dc, "f", 130, params)
    assert res == (None, False)
    assert (
        req.kv_transfer_params["do_remote_prefill"] is True
        and "_tt_demoted" not in req.kv_transfer_params
    )
    assert dc._to_release == set()
    dc._free_slots_estimate = 8
    dc._inflight = {"x", "y"}  # max_inflight_loads = 2
    assert dc.get_num_new_matched_tokens(req, 0) == (None, False)
    dc._inflight = {"x"}
    assert dc.get_num_new_matched_tokens(req, 0) == (129, True)


def test_update_state_after_alloc_records_recv(pair):
    pc, dc = pair
    T = 130
    _, _, _, params = produce(pc, "u-p", T)
    req, res = consumer_offer(dc, "u", T, params)
    assert res == (T - 1, True)
    with pytest.raises(RuntimeError):
        dc.update_state_after_alloc(req, blocks_of([5, 6, 7]), T)  # never T (I2)
    with pytest.raises(RuntimeError):
        dc.update_state_after_alloc(req, blocks_of([5, 6]), T - 1)  # wrong block count
    dc.update_state_after_alloc(req, blocks_of([5, 6, 7]), T - 1)
    rm = dc._reqs_need_recv["u"]
    assert (
        isinstance(rm, RecvMeta)
        and rm.local_block_ids == [5, 6, 7]
        and rm.num_tokens == T - 1
    )
    assert (
        rm.xfer.xfer_id == params["xfer_id"]
        and rm.xfer.engine_id == "p0"
        and rm.xfer.num_tokens == T - 1
    )
    assert req.kv_transfer_params["do_remote_prefill"] is False
    assert req.kv_transfer_params["_tt_recv_recorded"] is True
    assert dc._free_slots_estimate == 7 and dc._inflight == {"u"}
    meta = dc.build_connector_meta(sched_output())
    assert meta.reqs_to_recv == {"u": rm} and dc._reqs_need_recv == {}
    assert dc._first_token_steps == {"u": 1}
    # a re-admission after preemption / the post-promotion call: no-op except stats
    dc.update_state_after_alloc(req, blocks_of([8]), 0)
    assert dc._reqs_need_recv == {} and dc._first_token_steps == {}
    assert dc._sched_stats.data["steps_to_first_token"] == [1]
    assert dc._sched_stats.data["records"]["u"]["steps_to_first_token"] == 1
    st = dc.get_kv_connector_stats()
    assert st is not None and st.data["steps_to_first_token"] == [1]
    assert dc.get_kv_connector_stats() is None


def test_failed_load_readmission_is_demoted(pair):
    """B2: after a failed load (recompute) the request is re-offered with
    num_computed_tokens == 0, status WAITING and do_remote_prefill already False."""
    pc, dc = pair
    T = 130
    _, _, _, params = produce(pc, "b2-p", T)
    req, res = consumer_offer(dc, "b2", T, params)
    dc.update_state_after_alloc(req, blocks_of([1, 2, 3]), T - 1)
    dc.build_connector_meta(sched_output())
    assert dc._inflight == {"b2"} and "b2" in dc._first_token_steps
    req.status = RequestStatus.WAITING
    req.num_computed_tokens = 0
    res = dc.get_num_new_matched_tokens(req, 0)
    assert res == (None, False)
    assert req.kv_transfer_params["_tt_demoted"] is True
    assert "b2" not in dc._first_token_steps and "b2" not in dc._inflight
    assert dc._to_release == {params["xfer_id"]}  # released idempotently
    assert dc.get_num_new_matched_tokens(req, 0) == (0, False)
    # the local prefill's update_state_after_alloc(..., 0) is not a promotion (G4)
    dc.update_state_after_alloc(req, blocks_of([1, 2, 3]), 0)
    assert dc._sched_stats.data["steps_to_first_token"] == []


def test_preempted_resume_is_plain_class(pair):
    pc, dc = pair
    T = 130
    _, _, _, params = produce(pc, "pr-p", T)
    req, res = consumer_offer(dc, "pr", T, params)
    dc.update_state_after_alloc(req, blocks_of([1, 2, 3]), T - 1)
    req.status = RequestStatus.PREEMPTED
    req.num_computed_tokens = 0
    assert dc.get_num_new_matched_tokens(req, 0) == (0, False)
    assert "_tt_demoted" not in req.kv_transfer_params


def test_promoted_request_is_not_consulted_as_remote(pair):
    pc, dc = pair
    T = 130
    _, _, _, params = produce(pc, "pm-p", T)
    req, _ = consumer_offer(dc, "pm", T, params)
    dc.update_state_after_alloc(req, blocks_of([1, 2, 3]), T - 1)
    # the base scheduler never calls the connector for num_computed_tokens > 0; if it
    # did, the answer is the plain (0, False) (do_remote_prefill is already False)
    req.num_computed_tokens = T - 1
    assert dc.get_num_new_matched_tokens(req, T - 1) == (0, False)


def test_build_connector_meta_drains_and_does_not_mutate_output(pair):
    pc, dc = pair
    T = 130
    _, _, _, params = produce(pc, "m-p", T)
    req, _ = consumer_offer(dc, "m", T, params)
    dc.update_state_after_alloc(req, blocks_of([1, 2, 3]), T - 1)
    dc._to_release.add("p0:" + "a" * 32)
    so = sched_output([("other", [9])])
    snapshot = repr(so)
    meta = dc.build_connector_meta(so)
    assert repr(so) == snapshot
    assert isinstance(meta, TTKVConnectorMetadata)
    assert set(meta.reqs_to_recv) == {"m"} and meta.to_release == {"p0:" + "a" * 32}
    assert (
        meta.reqs_to_save == {}
        and meta.reqs_to_send == {}
        and meta.reqs_not_processed == set()
    )
    again = dc.build_connector_meta(sched_output())
    assert again.is_empty()


def test_consumer_request_finished_release_paths(pair):
    pc, dc = pair
    T = 130
    _, _, _, params = produce(pc, "rf-p", T)
    # (a) aborted before the load was recorded (still do_remote_prefill) -> release once
    req, res = consumer_offer(dc, "rf", T, params)
    assert res == (T - 1, True)
    req.status = RequestStatus.FINISHED_ABORTED
    assert dc.request_finished(req, [1, 2]) == (False, None)
    assert dc._to_release == {params["xfer_id"]} and "rf" not in dc._first_token_steps
    assert dc.request_finished(req, [1, 2]) == (False, None)
    assert dc._to_release == {params["xfer_id"]}  # a set: no double add
    dc._to_release.clear()
    # (b) recorded load (do_remote_prefill False, not demoted): the worker owns it
    req2, _ = consumer_offer(dc, "rf2", T, params)
    dc.update_state_after_alloc(req2, blocks_of([1, 2, 3]), T - 1)
    req2.status = RequestStatus.FINISHED_ABORTED
    assert dc.request_finished(req2, [1, 2, 3]) == (False, None)
    assert dc._to_release == set() and "rf2" not in dc._inflight
    # (c) synthetic serving-layer rejection request: no xfer_id -> no KeyError
    req3 = make_request("rf3", 5, {"do_remote_prefill": True})
    req3.status = RequestStatus.FINISHED_ABORTED
    assert dc.request_finished(req3, []) == (False, None)
    assert dc.request_finished_all_groups(req3, ([],)) == (False, None)


def test_update_connector_output_refreshes_estimate_and_inflight(pair):
    _, dc = pair
    dc._inflight = {"a", "b"}
    dc._reqs_need_send = {"z": time.time()}
    dc.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=TTKVWorkerMeta(3),
            finished_recving={"a"},
            finished_sending=None,
        )
    )
    assert (
        dc._free_slots_estimate == 3
        and dc._inflight == {"b"}
        and "z" in dc._reqs_need_send
    )
    dc.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=None, finished_recving=None, finished_sending={"z"}
        )
    )
    assert dc._free_slots_estimate == 3 and dc._reqs_need_send == {}


def test_worker_meta_aggregate():
    assert TTKVWorkerMeta(3).aggregate(TTKVWorkerMeta(5)) == TTKVWorkerMeta(3)


def test_kv_both_loopback_round_trip():
    """R1: one connector plays both legs; per-leg ids keep the two Requests distinct."""
    c = make_connector("kv_both", "l0")
    T = 200
    req_p, meta, delay, params = produce(c, "rid-p", T)
    assert delay is True and set(meta.reqs_to_save) == {"rid-p"}
    req_d, res = consumer_offer(c, "rid", T, params)
    assert res == (T - 1, True)
    c.update_state_after_alloc(req_d, blocks_of(list(range(4))), T - 1)
    meta2 = c.build_connector_meta(sched_output())
    assert set(meta2.reqs_to_recv) == {"rid"} and set(meta2.reqs_to_send) == {"rid-p"}
    # the P-leg multimodal case lands in the final return (xfer_id None)
    mm = make_request("mm-p", 5, p_params(), mm=True)
    c.get_num_new_matched_tokens(mm, 0)
    mm.status = RequestStatus.FINISHED_LENGTH_CAPPED
    assert c.request_finished(mm, [1]) == (False, None)


def test_transfer_descriptor_validation():
    good = {
        "remote_engine_id": "p0",
        "remote_request_id": "x",
        "xfer_id": "p0:" + "0" * 32,
        "remote_num_tokens": 10,
        "remote_transport": {"kind": "shm", "mode": "dumpfile", "layout_version": 1},
        "tt_layout_version": 1,
        "remote_prompt_hash": "ab",
        "remote_fingerprint": "cd",
    }
    d = TransferDescriptor.from_params(good)
    assert d.chunk_tokens == 2048 and d.expiry is None and d.xfer_hex == "0" * 32
    for bad in (
        {"xfer_id": "p0:zz"},
        {"remote_engine_id": "p1"},
        {"remote_num_tokens": 0},
        {"remote_blocks_expiry_time": "soon"},
    ):
        with pytest.raises((ValueError, TypeError)):
            TransferDescriptor.from_params({**good, **bad})
    with pytest.raises(KeyError):
        TransferDescriptor.from_params(
            {k: v for k, v in good.items() if k != "xfer_id"}
        )


# --------------------------------------------------------------------------- #
# [fix] PD polish: NIT-3 reason text, NIT-5 finished ids reach end_step
# --------------------------------------------------------------------------- #
def test_chunk_tokens_mismatch_is_demoted_at_admission_with_a_clear_reason(pair):
    pc, dc = pair
    _, _, _, params = produce(pc, "c-p", 130)
    ok, why = dc._params_ok(params, make_request("c", 130, params), 129)
    assert ok and why == ""
    bad = {**params, "tt_chunk_tokens": 4096}
    ok, why = dc._params_ok(bad, make_request("c", 130, bad), 129)
    assert not ok and "tt_chunk_tokens 4096 != local xfer_chunk_tokens 2048" in why
    # The transport descriptor itself is unchanged (the fabric handshake may
    # replace it wholesale), so chunk_tokens rides its own params key.
    assert "chunk_tokens" not in dc.transport_descriptor
    assert params["tt_chunk_tokens"] == pc.xfer_chunk_tokens


def test_wait_for_save_hands_this_steps_finished_ids_to_end_step():
    dc = make_connector()

    class W:
        def __init__(self):
            self.calls = []

        def begin_step(self, meta, finished, join, *, num_scheduled_tokens=None):
            self.calls.append(("begin", set(finished), set(join)))

        def end_step(self, finished_req_ids=None):
            self.calls.append(
                ("end", None if finished_req_ids is None else set(finished_req_ids))
            )

    dc._w = W()
    dc.bind_connector_metadata(TTKVConnectorMetadata())
    dc.start_load_kv(
        None, finished_req_ids={"a"}, join_req_ids=set(), num_scheduled_tokens=0
    )
    dc.wait_for_save()
    assert dc._w.calls == [("begin", {"a"}, set()), ("end", {"a"})]
    # The next step's begin replaces the set.
    dc.start_load_kv(None, finished_req_ids=set(), join_req_ids=set())
    dc.wait_for_save()
    assert dc._w.calls[-1] == ("end", set())
