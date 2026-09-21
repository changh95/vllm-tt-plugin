# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only tests for the consumer's pull-thread import preparation: ``_pull`` unpacks
the payload and runs the GDN import's host prep (the model's ``pd_gdn_host_packer``)
off the main thread, ``_drain_fetched`` parks the prepared form for the runner without
touching the bytes again, a prep failure releases the payload (DONE / pool) and is
reported like a failed pull, and a ``_Fetched`` without the unpacked fields (older
record, tests) is still unpacked by the drain."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from vllm_tt_plugin.kv_connector import tt_mooncake_connector as mc
from vllm_tt_plugin.kv_connector.tt_mooncake_connector import (
    RecvReq,
    _HostBufferPool,
    _WorkerSide,
    pack_payload,
)

from . import test_pd_handoff as handoff
from .test_pd_handoff import FakeEngine, _connector_stub, _payload, _staged_reply

# fixtures shared with the hand-off tests (bound by assignment so pytest sees them here)
small_pool = handoff.small_pool
producer = handoff.producer
fake_pd_transfer = handoff.fake_pd_transfer


class FakePacker:
    """Stands in for pd_transfer.GdnHostPacker: records the thread it ran on."""

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def prepare(self, rec, taps):
        self.calls.append((threading.current_thread().name, rec, taps))
        if self.fail:
            raise RuntimeError("vectorized packed-history layout differs")
        return SimpleNamespace(kind="prepared", rec=rec, taps=taps, hist=torch.zeros(2))


@pytest.fixture
def consumer(small_pool):
    w = _WorkerSide(_connector_stub(False))
    w.engine = FakeEngine()
    w.pool = _HostBufferPool(w.engine, w._engine_lock, "receive")
    w._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pull")
    w._ctrl_pool = ThreadPoolExecutor(max_workers=1)
    w.runner = SimpleNamespace(pd_pending_gdn={})
    w.model = SimpleNamespace(pd_gdn_host_packer=FakePacker())
    yield w
    w._pool.shutdown(wait=True)
    w._ctrl_pool.shutdown(wait=True)


def _rr(num_tokens=7):
    return RecvReq("r1", [3, 4], "127.0.0.1", 1, "t1", num_tokens)


def _pull_on_worker(w, rr):
    fut = w._pool.submit(w._pull, rr)
    fut.result(timeout=10)


def test_pull_unpacks_and_prepares_on_the_worker_thread(
    consumer, producer, monkeypatch, fake_pd_transfer
):
    w = consumer
    kv, rec, taps = _payload()
    buf, header = pack_payload(kv, rec, taps, 7, 2)
    prod = producer.pool
    staged = prod.acquire(buf.numel())
    staged[: buf.numel()].copy_(buf)
    try:
        w._shm_ok = False  # force the copy path (deterministic release semantics)
        rep = _staged_reply(prod, staged, header, "elsewhere:0", None)
        sent = []
        monkeypatch.setattr(
            w, "_side_channel_call", lambda h, p, msg: sent.append(msg) or rep
        )
        rr = _rr()
        w._inflight[rr.req_id] = rr
        _pull_on_worker(w, rr)
        f = w._fetched.get_nowait()
        packer = w.model.pd_gdn_host_packer
        # prepared once, on the pull worker, from views into the fetched bytes
        assert len(packer.calls) == 1 and packer.calls[0][0].startswith("pull")
        assert f.rec is not None and f.gdn.kind == "prepared"
        assert torch.equal(f.rec, rec) and torch.equal(f.gdn.taps, taps)
        assert f.rec.data_ptr() >= f.buf.data_ptr()  # a view, no copy
        assert len(f.kv) == 2 and torch.equal(f.kv[1][0], kv[1][0])
        assert f.t_prep >= 0.0
        # the drain does not unpack again and parks the prepared form for the runner
        calls = []
        monkeypatch.setattr(
            mc, "unpack_payload", lambda *a, **k: calls.append(1) or (None,) * 3
        )
        w._fetched.put(f)
        w._drain_fetched()
        assert calls == [] and len(packer.calls) == 1
        assert fake_pd_transfer and fake_pd_transfer[0][0] == [3, 4]
        entry = w.runner.pd_pending_gdn[rr.req_id]
        assert entry[1] is f.gdn and torch.equal(entry[0], rec)
        assert len(w.pool._free) == 0  # released by the runner, not the drain
        entry[3]()
        assert len(w.pool._free) == 1
        assert w.take_finished(set()) == (None, {rr.req_id})
    finally:
        prod.release(staged)


def test_kv_payload_is_prepared_on_the_worker_when_pd_transfer_offers_it(
    consumer, producer, monkeypatch, fake_pd_transfer
):
    """With pd_transfer.prepare_kv_import available the worker also stages the KV
    payload; the drain hands that prepared object to import_kv_blocks."""
    import sys

    w = consumer
    mod = sys.modules["models.demos.blackhole.qwen36.tt.pd_transfer"]
    seen = []

    def prepare_kv_import(model, kv):
        seen.append(threading.current_thread().name)
        return SimpleNamespace(kind="kv-prepared", kv=kv)

    mod.prepare_kv_import = prepare_kv_import
    kv, rec, taps = _payload()
    buf, header = pack_payload(kv, rec, taps, 7, 2)
    prod = producer.pool
    staged = prod.acquire(buf.numel())
    staged[: buf.numel()].copy_(buf)
    try:
        w._shm_ok = False
        rep = _staged_reply(prod, staged, header, "elsewhere:0", None)
        monkeypatch.setattr(w, "_side_channel_call", lambda h, p, msg: rep)
        rr = _rr()
        w._inflight[rr.req_id] = rr
        _pull_on_worker(w, rr)
        f = w._fetched.get_nowait()
        assert seen and seen[0].startswith("pull") and f.kv.kind == "kv-prepared"
        assert torch.equal(f.kv.kv[0][1], kv[0][1])
        w._fetched.put(f)
        w._drain_fetched()
        assert fake_pd_transfer[0][0] == [3, 4] and fake_pd_transfer[0][1] is f.kv
        w.runner.pd_pending_gdn[rr.req_id][3]()
    finally:
        prod.release(staged)


def test_prep_failure_releases_the_payload_and_reports_the_pull_failed(
    consumer, producer, monkeypatch, fake_pd_transfer
):
    w = consumer
    w.model = SimpleNamespace(pd_gdn_host_packer=FakePacker(fail=True))
    kv, rec, taps = _payload()
    buf, header = pack_payload(kv, rec, taps, 7, 2)
    prod = producer.pool
    staged = prod.acquire(buf.numel())
    staged[: buf.numel()].copy_(buf)
    try:
        w._shm_ok = False
        rep = _staged_reply(prod, staged, header, "elsewhere:0", None)
        sent = []
        monkeypatch.setattr(
            w, "_side_channel_call", lambda h, p, msg: sent.append(msg) or rep
        )
        rr = _rr()
        w._inflight[rr.req_id] = rr
        _pull_on_worker(w, rr)
        assert w._fetched.empty()
        assert len(w.pool._free) == 1  # the pooled receive buffer went back
        w._drain_fetched()
        assert fake_pd_transfer == [] and rr.req_id not in w.runner.pd_pending_gdn
        # reported as received so the scheduler proceeds (the runner prefills locally)
        assert w.take_finished(set()) == (None, {rr.req_id})
    finally:
        prod.release(staged)


def test_without_a_packer_the_raw_taps_are_parked(
    consumer, producer, monkeypatch, fake_pd_transfer
):
    w = consumer
    w.model = object()  # no pd_gdn_host_packer (post_warmup not run)
    kv, rec, taps = _payload()
    buf, header = pack_payload(kv, rec, taps, 7, 2)
    prod = producer.pool
    staged = prod.acquire(buf.numel())
    staged[: buf.numel()].copy_(buf)
    try:
        w._shm_ok = False
        rep = _staged_reply(prod, staged, header, "elsewhere:0", None)
        monkeypatch.setattr(w, "_side_channel_call", lambda h, p, msg: rep)
        rr = _rr()
        w._inflight[rr.req_id] = rr
        _pull_on_worker(w, rr)
        w._drain_fetched()
        entry = w.runner.pd_pending_gdn[rr.req_id]
        assert torch.equal(entry[0], rec) and torch.equal(entry[1], taps)
        entry[3]()
    finally:
        prod.release(staged)


def test_drain_unpacks_a_record_the_worker_did_not(consumer, fake_pd_transfer):
    """A _Fetched with rec=None (older record / tests) is unpacked and prepared by the
    drain itself."""
    w = consumer
    kv, rec, taps = _payload()
    buf, header = pack_payload(kv, rec, taps, 7, 2)
    rr = _rr()
    w._inflight[rr.req_id] = rr
    released = []
    w._fetched.put(
        mc._Fetched(rr, buf, header, lambda: released.append(1), "pull", 0.0, 0.0)
    )
    w._drain_fetched()
    packer = w.model.pd_gdn_host_packer
    assert len(packer.calls) == 1  # prepared on this (main) thread instead
    entry = w.runner.pd_pending_gdn[rr.req_id]
    assert entry[1].kind == "prepared" and torch.equal(entry[0], rec)
    assert released == []
    entry[3]()
    assert released == [1]


def test_aborted_pull_skips_the_prep(consumer, producer, monkeypatch, fake_pd_transfer):
    """A request that finished while its bytes were in flight is not prepared (the
    drain drops and releases it)."""
    w = consumer
    kv, rec, taps = _payload()
    buf, header = pack_payload(kv, rec, taps, 7, 2)
    prod = producer.pool
    staged = prod.acquire(buf.numel())
    staged[: buf.numel()].copy_(buf)
    try:
        w._shm_ok = False
        rep = _staged_reply(prod, staged, header, "elsewhere:0", None)

        def get(h, p, msg):
            if msg["op"] == "GET":
                with w._lock:  # the client hangs up right after the GET was answered
                    w._aborted.add("r1")
            return rep

        monkeypatch.setattr(w, "_side_channel_call", get)
        rr = _rr()
        w._inflight[rr.req_id] = rr
        _pull_on_worker(w, rr)
        f = w._fetched.get_nowait()
        assert f.rec is None and w.model.pd_gdn_host_packer.calls == []
        w._fetched.put(f)
        w._drain_fetched()
        assert fake_pd_transfer == [] and len(w.pool._free) == 1
        assert w.take_finished(set()) == (None, {rr.req_id})
    finally:
        prod.release(staged)
