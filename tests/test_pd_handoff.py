# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only tests for the P/D hand-off mechanics of ``tt_mooncake_connector``:

* the producer's staging pool backed by ``/dev/shm`` files, mapped read-only by a
  consumer on the same host (no copy) with DONE deferred to the consumer's release;
* the ROUTER side channel that parks a GET until the transfer is staged;
* the proxy-chosen ``transfer_id`` (echoed by the producer) and the consumer-derived
  ``num_tokens`` that let a proxy post to both instances at once;
* aborts of in-flight pulls, including the CANCEL the producer remembers for a transfer
  it has not staged yet, and DONE/CANCEL never queueing behind parked pulls.

No Mooncake engine and no device: the engine is a stub whose ``transfer_sync_read``
is a same-process memcpy, and the model import is a recorded stub module.
"""

from __future__ import annotations

import ctypes
import os
import socket
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.request import RequestStatus

from vllm_tt_plugin.kv_connector import tt_mooncake_connector as mc
from vllm_tt_plugin.kv_connector.tt_mooncake_connector import (
    RecvReq,
    StageReq,
    TTMooncakeConnectorMetadata,
    _HostBufferPool,
    _SchedulerSide,
    _Staged,
    _WorkerSide,
    host_identity,
    map_shm_segment,
    pack_payload,
    payload_nbytes,
    unlink_stale_shm_segments,
    unpack_payload,
)

needs_shm = pytest.mark.skipif(
    not os.path.isdir("/dev/shm"), reason="/dev/shm is not available"
)


class FakeEngine:
    def __init__(self):
        self.registered: list[tuple[int, int]] = []
        self.reads: list[tuple[str, int, int, int]] = []

    def register_memory(self, ptr, nbytes):
        self.registered.append((ptr, nbytes))
        return 0

    def transfer_sync_read(self, segment, dst, src, nbytes):
        self.reads.append((segment, dst, src, nbytes))
        ctypes.memmove(dst, src, nbytes)
        return 0


def _payload(seed=0):
    g = torch.Generator().manual_seed(seed)
    kv = [
        (
            torch.randn(2, 4, 2, 8, generator=g).to(torch.bfloat16),
            torch.randn(2, 4, 2, 8, generator=g).to(torch.bfloat16),
        )
        for _ in range(2)
    ]
    rec = torch.randn(2, 3, 2, 8, 8, generator=g, dtype=torch.float32)
    taps = torch.randn(2, 3, 4, 16, generator=g).to(torch.bfloat16)
    return kv, rec, taps


def _inside(view: torch.Tensor, buf: torch.Tensor) -> bool:
    lo = buf.data_ptr()
    hi = lo + buf.numel() * buf.element_size()
    return lo <= view.data_ptr() < view.data_ptr() + view.numel() <= hi


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _connector_stub(producer: bool, port: int = 0):
    return SimpleNamespace(
        _is_producer=producer,
        _is_consumer=not producer,
        _side_host="127.0.0.1",
        _side_port=port,
        _block_size=4,
    )


@pytest.fixture
def small_pool(monkeypatch):
    # 256 MiB per buffer is the production minimum; tests want small files
    monkeypatch.setattr(_HostBufferPool, "_MIN", 1 << 16)
    monkeypatch.setattr(_HostBufferPool, "_STEP", 1 << 12)


@pytest.fixture
def producer(small_pool):
    """A producer worker with a shm staging pool and its side channel on a free port."""
    port = _free_port()
    w = _WorkerSide(_connector_stub(True, port))
    w.engine = FakeEngine()
    w.pool = _HostBufferPool(w.engine, w._engine_lock, "staging", shm=True)
    w.local_host, w.rpc_port = "127.0.0.1", 12345
    thread = threading.Thread(target=w._serve_side_channel, daemon=True)
    thread.start()
    yield w
    w._stop.set()
    thread.join(timeout=5)
    w.pool.close()


@pytest.fixture
def consumer(small_pool):
    w = _WorkerSide(_connector_stub(False))
    w.engine = FakeEngine()
    w.pool = _HostBufferPool(w.engine, w._engine_lock, "receive")
    w._pool = ThreadPoolExecutor(max_workers=1)
    w._ctrl_pool = ThreadPoolExecutor(max_workers=1)
    w.runner = SimpleNamespace(pd_pending_gdn={})
    w.model = object()
    yield w
    w._pool.shutdown(wait=True)
    w._ctrl_pool.shutdown(wait=True)


@pytest.fixture
def fake_pd_transfer(monkeypatch):
    """``models.demos.blackhole.qwen36.tt.pd_transfer`` as the drain imports it."""
    calls = []
    mod = types.ModuleType("models.demos.blackhole.qwen36.tt.pd_transfer")
    mod.import_kv_blocks = lambda model, block_ids, kv: calls.append(
        (list(block_ids), kv)
    )
    mod.export_kv_blocks = lambda model, block_ids: _payload()[0]
    parent = None
    for i, name in enumerate(["models", "demos", "blackhole", "qwen36", "tt"]):
        full = ".".join(["models", "demos", "blackhole", "qwen36", "tt"][: i + 1])
        pkg = sys.modules.get(full) or types.ModuleType(full)
        pkg.__path__ = []
        monkeypatch.setitem(sys.modules, full, pkg)
        if parent is not None:
            monkeypatch.setattr(parent, name, pkg, raising=False)
        parent = pkg
    monkeypatch.setattr(parent, "pd_transfer", mod, raising=False)
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    return calls


# ---- F: shm-backed staging ----------------------------------------------------------


@needs_shm
def test_shm_pool_buffers_are_dev_shm_files_registered_once(small_pool):
    engine = FakeEngine()
    pool = _HostBufferPool(engine, threading.Lock(), "staging", shm=True)
    try:
        b = pool.acquire(1000)
        name = pool.shm_name(b)
        assert name is not None and name.startswith(f"qwen36-pd-{os.getpid()}-")
        path = os.path.join("/dev/shm", name)
        assert os.path.getsize(path) == b.numel() >= 1000
        # the pointer Mooncake knows is the mapping itself (cross-host pulls unchanged)
        assert engine.registered == [(b.data_ptr(), b.numel())]
        pool.release(b)
        assert (
            pool.acquire(1000).data_ptr() == b.data_ptr()
        )  # reused, not re-registered
        assert len(engine.registered) == 1
    finally:
        pool.close()
    assert not os.path.exists(path)


@needs_shm
def test_consumer_mapping_is_a_zero_copy_view_of_the_producer_buffer(small_pool):
    kv, rec, taps = _payload()
    pool = _HostBufferPool(FakeEngine(), threading.Lock(), "staging", shm=True)
    try:
        buf = pool.acquire(payload_nbytes(kv, rec, taps))
        buf, header = pack_payload(kv, rec, taps, 7, 2, out=buf)
        seg = map_shm_segment(pool.shm_name(buf))
        assert seg is not None and seg.numel() == buf.numel()
        assert seg.data_ptr() != buf.data_ptr()  # a second mapping of the same pages
        kv2, rec2, taps2 = unpack_payload(seg[: header["nbytes"]], header)
        assert torch.equal(rec2, rec) and torch.equal(taps2, taps)
        for (k, v), (k2, v2) in zip(kv, kv2):
            assert torch.equal(k, k2) and torch.equal(v, v2)
        assert _inside(rec2, seg) and _inside(taps2, seg)  # views, no copy
        # bytes the producer writes later show up in the consumer's mapping
        buf[0] = 0xAB
        assert int(seg[0]) == 0xAB
    finally:
        pool.close()


def test_map_shm_segment_rejects_foreign_names_and_missing_files():
    assert map_shm_segment("../etc/passwd") is None
    assert map_shm_segment("not-ours") is None
    assert map_shm_segment("qwen36-pd-0-doesnotexist") is None


def test_shm_knob_off_gives_plain_buffers(monkeypatch, small_pool):
    monkeypatch.setenv("QWEN36_PD_SHM", "0")
    assert mc.shm_enabled() is False
    w = _WorkerSide(_connector_stub(True))
    assert w._shm_ok is False
    pool = _HostBufferPool(FakeEngine(), threading.Lock(), "staging", shm=w._shm_ok)
    b = pool.acquire(10)
    assert pool.shm_name(b) is None
    w.pool = pool
    w._staged["t"] = _Staged(b, b.data_ptr(), 10, {}, time.time(), pool.shm_name(b))
    assert w._get_reply("t", "h")["shm"] is None


@needs_shm
def test_unlink_stale_shm_segments_only_removes_dead_producers(tmp_path, monkeypatch):
    monkeypatch.setattr(mc, "_SHM_DIR", str(tmp_path))
    for name in ("qwen36-pd-111-aa", "qwen36-pd-222-bb", "other-file", "qwen36-pd-x"):
        (tmp_path / name).write_bytes(b"x")
    removed = unlink_stale_shm_segments(pids_alive={222})
    assert removed == ["qwen36-pd-111-aa"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "other-file",
        "qwen36-pd-222-bb",
        "qwen36-pd-x",
    ]


# ---- side channel: parked GET, DONE releases -----------------------------------------


@needs_shm
def test_get_parks_until_staged_then_carries_shm_and_host(producer):
    w = producer
    kv, rec, taps = _payload()
    replies = []

    def get():
        replies.append(
            _WorkerSide._side_channel_call(
                "127.0.0.1", w.c._side_port, {"op": "GET", "transfer_id": "tid-1"}
            )
        )

    t = threading.Thread(target=get)
    t0 = time.perf_counter()
    t.start()
    time.sleep(0.15)
    assert not replies  # parked, not answered "pending"
    buf = w.pool.acquire(payload_nbytes(kv, rec, taps))
    buf, header = pack_payload(kv, rec, taps, 7, 2, out=buf)
    with w._lock:
        w._staged["tid-1"] = _Staged(
            buf,
            buf.data_ptr(),
            header["nbytes"],
            header,
            time.time(),
            w.pool.shm_name(buf),
        )
    t.join(timeout=5)
    assert replies and replies[0]["status"] == "ok"
    rep = replies[0]
    assert (
        time.perf_counter() - t0 < mc._GET_WAIT_S
    )  # answered by the stage, not the timeout
    assert rep["host"] == host_identity()
    assert rep["shm"] == {
        "name": w.pool.shm_name(buf),
        "offset": 0,
        "nbytes": header["nbytes"],
    }
    assert rep["addr"] == buf.data_ptr() and rep["header"]["nbytes"] == header["nbytes"]
    # the buffer stays staged until DONE
    assert w.pool._free == []
    done = _WorkerSide._side_channel_call(
        "127.0.0.1", w.c._side_port, {"op": "DONE", "transfer_id": "tid-1"}
    )
    assert done == {"status": "ok"}
    assert len(w.pool._free) == 1 and "tid-1" not in w._staged


def test_get_answers_pending_after_the_bounded_wait(producer, monkeypatch):
    monkeypatch.setattr(mc, "_GET_WAIT_S", 0.05)
    t0 = time.perf_counter()
    rep = _WorkerSide._side_channel_call(
        "127.0.0.1", producer.c._side_port, {"op": "GET", "transfer_id": "never"}
    )
    assert rep == {"status": "pending"}
    assert 0.05 <= time.perf_counter() - t0 < 2.0


def test_other_clients_are_served_while_a_get_is_parked(producer):
    t = threading.Thread(
        target=_WorkerSide._side_channel_call,
        args=("127.0.0.1", producer.c._side_port, {"op": "GET", "transfer_id": "p"}),
    )
    t.start()
    time.sleep(0.05)
    t0 = time.perf_counter()
    rep = _WorkerSide._side_channel_call(
        "127.0.0.1", producer.c._side_port, {"op": "CANCEL", "transfer_id": "other"}
    )
    assert rep == {"status": "ok"} and time.perf_counter() - t0 < 1.0
    # cancelling the parked transfer wakes its waiter
    _WorkerSide._side_channel_call(
        "127.0.0.1", producer.c._side_port, {"op": "CANCEL", "transfer_id": "p"}
    )
    t.join(timeout=5)
    assert not t.is_alive()


# ---- consumer pull: shm mapping vs Mooncake copy -------------------------------------


def _staged_reply(pool, buf, header, host, shm_name):
    rep = {
        "status": "ok",
        "segment": "127.0.0.1:1",
        "addr": buf.data_ptr(),
        "nbytes": header["nbytes"],
        "header": header,
        "host": host,
        "shm": None,
    }
    if shm_name:
        rep["shm"] = {"name": shm_name, "offset": 0, "nbytes": header["nbytes"]}
    return rep


def _rr(req_id="r1", num_tokens=7):
    return RecvReq(req_id, [3, 4], "127.0.0.1", 1, "tid", num_tokens)


@needs_shm
def test_pull_maps_same_host_segment_and_defers_done(consumer, monkeypatch):
    w = consumer
    kv, rec, taps = _payload()
    prod = _HostBufferPool(FakeEngine(), threading.Lock(), "staging", shm=True)
    try:
        buf = prod.acquire(payload_nbytes(kv, rec, taps))
        buf, header = pack_payload(kv, rec, taps, 7, 2, out=buf)
        sent = []
        rep = _staged_reply(prod, buf, header, host_identity(), prod.shm_name(buf))
        monkeypatch.setattr(
            w, "_side_channel_call", lambda h, p, msg: sent.append(msg) or rep
        )
        w._pull(_rr())
        f = w._fetched.get_nowait()
        assert f.via == "shm"
        assert w.engine.reads == []  # no copy
        seg = w._shm_segments[prod.shm_name(buf)]
        assert _inside(f.buf, seg) and f.buf.numel() == header["nbytes"]
        _, rec2, _ = unpack_payload(f.buf, header)
        assert torch.equal(rec2, rec)
        # DONE only when the consumer releases the mapping (through the control
        # executor, never the pull workers)
        assert [m["op"] for m in sent] == ["GET"]
        f.release()
        w._ctrl_pool.shutdown(wait=True)
        assert [m["op"] for m in sent] == ["GET", "DONE"]
        # a second request on the same segment reuses the mapping
        w._pull(_rr("r2"))
        assert len(w._shm_segments) == 1
    finally:
        prod.close()


@needs_shm
@pytest.mark.parametrize("reason", ["other_host", "segment_missing", "knob_off"])
def test_pull_falls_back_to_mooncake_copy(consumer, monkeypatch, reason):
    w = consumer
    kv, rec, taps = _payload(1)
    prod = _HostBufferPool(FakeEngine(), threading.Lock(), "staging", shm=True)
    try:
        buf = prod.acquire(payload_nbytes(kv, rec, taps))
        buf, header = pack_payload(kv, rec, taps, 7, 2, out=buf)
        host, name = host_identity(), prod.shm_name(buf)
        if reason == "other_host":
            host = "elsewhere:0"
        elif reason == "segment_missing":
            name = "qwen36-pd-0-gone"
        else:
            w._shm_ok = False
        sent = []
        rep = _staged_reply(prod, buf, header, host, name)
        monkeypatch.setattr(
            w, "_side_channel_call", lambda h, p, msg: sent.append(msg) or rep
        )
        w._pull(_rr())
        f = w._fetched.get_nowait()
        assert f.via == "pull" and len(w.engine.reads) == 1
        assert [m["op"] for m in sent] == ["GET", "DONE"]  # copied: producer freed now
        assert not _inside(f.buf, buf) and torch.equal(f.buf, buf[: header["nbytes"]])
        assert w.pool._free == []
        f.release()
        assert len(w.pool._free) == 1
    finally:
        prod.close()


# ---- drain: header check, abort ------------------------------------------------


def _fetched_for(w, num_tokens_in_header, rr):
    kv, rec, taps = _payload()
    buf, header = pack_payload(kv, rec, taps, num_tokens_in_header, 2)
    released = []
    w._fetched.put(
        mc._Fetched(rr, buf, header, lambda: released.append(1), "pull", 0.0, 0.0)
    )
    return released


def test_drain_imports_and_parks_release_for_the_runner(consumer, fake_pd_transfer):
    w = consumer
    rr = _rr(num_tokens=7)
    w._inflight[rr.req_id] = rr
    released = _fetched_for(w, 7, rr)
    w._drain_fetched()
    assert len(fake_pd_transfer) == 1 and fake_pd_transfer[0][0] == [3, 4]
    assert released == []  # the runner releases after import_gdn_slot
    entry = w.runner.pd_pending_gdn[rr.req_id]
    entry[3]()
    assert released == [1]
    assert w.take_finished(set()) == (None, {rr.req_id})


def test_drain_rejects_a_token_count_mismatch(consumer, fake_pd_transfer):
    w = consumer
    rr = _rr(num_tokens=9)  # this instance tokenized to 10 tokens; producer staged 7
    w._inflight[rr.req_id] = rr
    released = _fetched_for(w, 7, rr)
    w._drain_fetched()
    assert fake_pd_transfer == [] and released == [1]
    assert rr.req_id not in w.runner.pd_pending_gdn
    assert w.take_finished(set()) == (None, {rr.req_id})  # scheduler proceeds


def test_finished_request_aborts_its_inflight_pull(
    consumer, fake_pd_transfer, monkeypatch
):
    w = consumer
    rr = _rr()
    w._inflight[rr.req_id] = rr
    calls = []
    monkeypatch.setattr(
        w,
        "_side_channel_call",
        lambda h, p, msg: calls.append(msg) or {"status": "pending"},
    )
    monkeypatch.setattr(mc, "_GET_POLL_S", 0.001)
    # the request finishes (client hang-up) while the pull polls the producer
    threading.Timer(0.05, lambda: w.take_finished({rr.req_id})).start()
    w._pull(rr)
    w._ctrl_pool.shutdown(wait=True)
    assert w._fetched.empty() and calls and calls[0]["op"] == "GET"
    # the producer is told (it may still stage the transfer later)
    assert [m["op"] for m in calls if m["op"] != "GET"] == ["CANCEL"]
    assert calls[-1] == {"op": "CANCEL", "transfer_id": rr.transfer_id}
    w._drain_fetched()
    assert fake_pd_transfer == []
    assert rr.req_id not in w._inflight and rr.req_id not in w._aborted
    # reported so the scheduler frees the blocks it held back for the transfer
    assert w.take_finished(set()) == (None, {rr.req_id})


def test_fetched_after_abort_is_released_not_imported(consumer, fake_pd_transfer):
    w = consumer
    rr = _rr()
    w._inflight[rr.req_id] = rr
    released = _fetched_for(w, 7, rr)
    # the step that finishes the request also drains: dropped, released, reported
    assert w.take_finished({rr.req_id}) == (None, {rr.req_id})
    assert fake_pd_transfer == [] and released == [1]
    assert rr.req_id not in w.runner.pd_pending_gdn


def test_parked_import_of_a_finished_request_is_released(consumer, fake_pd_transfer):
    w = consumer
    rr = _rr()
    w._inflight[rr.req_id] = rr
    released = _fetched_for(w, 7, rr)
    w._drain_fetched()
    assert rr.req_id in w.runner.pd_pending_gdn and released == []
    # aborted after the import was parked but before it got a decode slot
    w.take_finished({rr.req_id})
    assert rr.req_id not in w.runner.pd_pending_gdn and released == [1]


# ---- DONE/CANCEL never queue behind parked pulls; CANCEL before stage ----------------


def _stage_on(producer, tid, num_tokens=7):
    kv, rec, taps = _payload()
    buf = producer.pool.acquire(payload_nbytes(kv, rec, taps))
    buf, header = pack_payload(kv, rec, taps, num_tokens, 2, out=buf)
    with producer._lock:
        producer._staged[tid] = _Staged(
            buf,
            buf.data_ptr(),
            header["nbytes"],
            header,
            time.time(),
            producer.pool.shm_name(buf),
        )
    return buf


def _wait_until(pred, timeout=5.0):
    t0 = time.perf_counter()
    while not pred():
        if time.perf_counter() - t0 > timeout:
            return False
        time.sleep(0.002)
    return True


def _rr_on(producer, req_id, tid, num_tokens=7):
    return RecvReq(req_id, [3, 4], "127.0.0.1", producer.c._side_port, tid, num_tokens)


@needs_shm
def test_release_done_reaches_producer_while_every_pull_worker_is_parked(
    producer, consumer, monkeypatch
):
    """A release issued while the (single) pull worker is parked on the producer for a
    not-yet-staged transfer frees the producer's staging promptly: DONE goes through
    the control executor, not the pull FIFO."""
    monkeypatch.setattr(mc, "_GET_WAIT_S", 3.0)
    p, w = producer, consumer
    _stage_on(p, "tid-a")
    w._pull(_rr_on(p, "ra", "tid-a"))  # inline: fetched via shm, DONE deferred
    f = w._fetched.get_nowait()
    assert f.via == "shm" and p.pool._free == [] and "tid-a" in p._staged
    # the only pull worker parks on the producer (tid-b is not staged)
    rr_b = _rr_on(p, "rb", "tid-b")
    w._inflight[rr_b.req_id] = rr_b
    fut = w._pool.submit(w._pull, rr_b)
    time.sleep(0.1)
    assert not fut.done()
    t0 = time.perf_counter()
    f.release()
    assert _wait_until(lambda: "tid-a" not in p._staged, timeout=1.0)
    assert time.perf_counter() - t0 < 1.0 and len(p.pool._free) == 1
    assert not fut.done()  # the pull is still parked; DONE did not wait for it
    # let the parked pull finish: staging tid-b answers its GET
    _stage_on(p, "tid-b")
    fut.result(timeout=5)
    assert w._fetched.get(timeout=1).via == "shm"


def test_cancel_before_stage_is_remembered_and_the_stage_is_dropped(
    producer, fake_pd_transfer
):
    p = producer
    kv, rec, taps = _payload()
    released = []
    p.runner = SimpleNamespace(_req_state_slot={"r": 0})
    p.model = SimpleNamespace(
        pd_gdn_capture={0: (rec, taps)},
        pd_gdn_snapshot_release=lambda r, c: released.append(1),
    )
    rep = _WorkerSide._side_channel_call(
        "127.0.0.1", p.c._side_port, {"op": "CANCEL", "transfer_id": "tid-c"}
    )
    assert rep == {"status": "ok"} and "tid-c" in p._cancelled
    meta = TTMooncakeConnectorMetadata(stage=[StageReq("r", [0, 1], 7, "tid-c")])
    p.stage_after_step(meta)
    # nothing staged, no buffer acquired, snapshot given back, scheduler still told
    assert "tid-c" not in p._staged and p._cancelled == {}
    assert p.pool.total == 0 and p.pool._free == []
    assert released == [1] and p.model.pd_gdn_capture == {}
    assert p.take_finished(set()) == ({"r"}, None)


def test_cancel_racing_the_stage_puts_the_buffer_back_in_the_pool(
    producer, fake_pd_transfer, monkeypatch
):
    p = producer
    kv, rec, taps = _payload()
    p.runner = SimpleNamespace(_req_state_slot={"r": 0})
    p.model = SimpleNamespace(pd_gdn_capture={0: (rec, taps)})
    real_pack = mc.pack_payload

    def pack_then_cancel(*a, **k):
        out = real_pack(*a, **k)
        p._release("tid-d", remember_cancel=True)  # CANCEL lands mid-stage
        return out

    monkeypatch.setattr(mc, "pack_payload", pack_then_cancel)
    p.stage_after_step(
        TTMooncakeConnectorMetadata(stage=[StageReq("r", [0, 1], 7, "tid-d")])
    )
    assert "tid-d" not in p._staged and p._cancelled == {}
    assert len(p.pool._free) == 1 and p.pool.total > 0  # acquired, then freed
    assert p.take_finished(set()) == ({"r"}, None)


def test_abort_during_parked_get_cancels_on_the_producer(
    producer, consumer, fake_pd_transfer, monkeypatch
):
    """End to end over the real side channel: the request finishes while its GET is
    parked -> CANCEL -> the producer's later stage is dropped, not held for the GC."""
    monkeypatch.setattr(mc, "_GET_WAIT_S", 0.2)
    monkeypatch.setattr(mc, "_GET_POLL_S", 0.001)
    p, w = producer, consumer
    rr = _rr_on(p, "re", "tid-e")
    w._inflight[rr.req_id] = rr
    threading.Timer(0.05, lambda: w.take_finished({rr.req_id})).start()
    w._pull(rr)  # GET parks 0.2 s, comes back "pending", the abort is seen
    w._ctrl_pool.shutdown(wait=True)
    assert "tid-e" in p._cancelled and w.take_finished(set()) == (None, {rr.req_id})
    kv, rec, taps = _payload()
    released = []
    p.runner = SimpleNamespace(_req_state_slot={"r": 0})
    p.model = SimpleNamespace(
        pd_gdn_capture={0: (rec, taps)},
        pd_gdn_snapshot_release=lambda r, c: released.append(1),
    )
    p.stage_after_step(
        TTMooncakeConnectorMetadata(stage=[StageReq("r", [0, 1], 7, "tid-e")])
    )
    assert "tid-e" not in p._staged and released == [1] and p.pool.total == 0


def test_stale_cancel_memory_expires_with_the_gc(producer, fake_pd_transfer):
    p = producer
    kv, rec, taps = _payload()
    p.runner = SimpleNamespace(_req_state_slot={"r": 0})
    p.model = SimpleNamespace(pd_gdn_capture={0: (rec, taps)})
    p._cancelled["old"] = time.time() - mc._GET_TIMEOUT_S - 1
    p._cancelled["fresh"] = time.time()
    p.stage_after_step(
        TTMooncakeConnectorMetadata(stage=[StageReq("r", [0, 1], 7, "tid-f")])
    )
    assert "tid-f" in p._staged and set(p._cancelled) == {"fresh"}


# ---- H: proxy-chosen transfer_id, consumer-derived num_tokens ------------------------


def _request(req_id, params, n_prompt):
    return SimpleNamespace(
        request_id=req_id,
        kv_transfer_params=params,
        prompt_token_ids=list(range(n_prompt)),
        num_prompt_tokens=n_prompt,
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
    )


def test_producer_echoes_the_proxy_transfer_id():
    sched = _SchedulerSide(_connector_stub(True, 18100))
    blocks = SimpleNamespace(get_block_ids=lambda: [[5, 6]])
    params = {"do_remote_decode": True, "transfer_id": "proxy-chose-this"}
    req = _request("chatcmpl-x-1a2b3c4d", params, 9)
    sched.update_state_after_alloc(req, blocks, 0)
    assert sched._to_stage[req.request_id] == StageReq(
        req.request_id, [5, 6], 9, "proxy-chose-this"
    )
    ok, out = sched.request_finished(req, [5, 6])
    assert ok and out["transfer_id"] == "proxy-chose-this"
    assert out["remote_host"] == "127.0.0.1" and out["remote_port"] == 18100
    assert out["num_tokens"] == 9


def test_producer_falls_back_to_the_engine_request_id():
    sched = _SchedulerSide(_connector_stub(True, 18100))
    req = _request("chatcmpl-x-1a2b3c4d", {"do_remote_decode": True}, 9)
    _, out = sched.request_finished(req, [])
    assert out["transfer_id"] == "chatcmpl-x-1a2b3c4d"


def test_consumer_derives_num_tokens_when_the_proxy_omits_it():
    sched = _SchedulerSide(_connector_stub(False))
    blocks = SimpleNamespace(get_block_ids=lambda: [[1, 2, 3]])
    params = {
        "do_remote_prefill": True,
        "remote_host": "127.0.0.1",
        "remote_port": 18100,
        "transfer_id": "t",
    }
    req = _request("d-1", params, 10)
    n, async_load = sched.get_num_new_matched_tokens(req, 0)
    assert (n, async_load) == (9, True)  # last token is recomputed on the consumer
    sched.update_state_after_alloc(req, blocks, n)
    rr = sched._to_recv["d-1"]
    assert rr.num_tokens == 9 and rr.transfer_id == "t" and rr.block_ids == [1, 2, 3]
    assert params["do_remote_prefill"] is False
    # an explicit count (serial proxy) still wins
    params2 = dict(params, do_remote_prefill=True, num_tokens=4)
    sched.update_state_after_alloc(_request("d-2", params2, 10), blocks, 9)
    assert sched._to_recv["d-2"].num_tokens == 4


# ---- producer: every request that finished in its prefill step is staged; strays never reach vLLM's assert ----


def test_producer_delays_the_free_for_a_stopped_first_token():
    """P's one-token completion may finish STOPPED (EOS / stop string as the first token): the worker still stages
    it in that step, so the free must be delayed exactly as for the length-capped case -- vLLM asserts that every
    finished_sending id is still tracked (scheduler._update_from_kv_xfer_finished), and 2026-09-22's grid lost P's
    engine to that assert."""
    sched = _SchedulerSide(_connector_stub(True, 18100))
    req = _request("chatcmpl-eos-1a2b3c4d", {"do_remote_decode": True}, 9)
    req.status = RequestStatus.FINISHED_STOPPED
    ok, out = sched.request_finished(req, [1])
    assert (
        ok
        and out["do_remote_prefill"]
        and out["transfer_id"] == "chatcmpl-eos-1a2b3c4d"
    )
    aborted = _request("chatcmpl-abort-1a2b3c4d", {"do_remote_decode": True}, 9)
    aborted.status = RequestStatus.FINISHED_ABORTED
    assert sched.request_finished(aborted, [2]) == (False, None)


def test_stray_finished_sending_ids_are_dropped_before_vllm_sees_them():
    sched = _SchedulerSide(_connector_stub(True, 18100))
    kept = _request("chatcmpl-kept-1a2b3c4d", {"do_remote_decode": True}, 9)
    assert sched.request_finished(kept, [1])[0]
    out = SimpleNamespace(
        finished_sending={"chatcmpl-kept-1a2b3c4d", "chatcmpl-gone-1a2b3c4d"},
        finished_recving=None,
    )
    sched.update_connector_output(out)
    assert out.finished_sending == {"chatcmpl-kept-1a2b3c4d"}
    assert not sched._delayed_free  # reported once, then forgotten
    empty = SimpleNamespace(finished_sending=None, finished_recving=None)
    sched.update_connector_output(empty)  # None stays None
    assert empty.finished_sending is None
