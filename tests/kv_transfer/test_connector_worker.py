# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only tests for ``TTKVWorker`` (PHASE2_DESIGN.md 3.3, step 8.2) with a fake
transport (4.1 names), a fake model hook (5.1 names) and a fake runner (6.2
remote-slot API). No device, no ttnn."""

from __future__ import annotations

import time

import pytest
from vllm.utils.math_utils import cdiv

from vllm_tt_plugin.kv_transfer import worker as wk
from vllm_tt_plugin.kv_transfer.hooks import build_manifest
from vllm_tt_plugin.kv_transfer.metadata import (
    RecvMeta,
    SaveMeta,
    TransferDescriptor,
    TTKVConnectorMetadata,
    TTKVWorkerMeta,
)
from vllm_tt_plugin.kv_transfer.tt_connector import TTKVConnectorStats, xfer_id_for
from vllm_tt_plugin.kv_transfer.worker import LoadState, TTKVWorker

BLOCK = 64
DESC = {"kind": "shm", "mode": "dumpfile", "layout_version": 1}


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FakePut:
    def __init__(self, xfer_id, manifest):
        self.xfer_id, self.manifest = xfer_id, manifest
        self.sinks = {p.name: object() for p in manifest.parts}


class FakeGet:
    def __init__(self, xfer_id, manifest, status):
        self.xfer_id, self.manifest, self.status = xfer_id, manifest, status
        self.sources = {p.name: object() for p in manifest.parts} if manifest else {}

    def ready(self):
        return self.status == "READY"


class FakeTransport:
    """Header state machine only:
    WRITING -> READY|FAILED -> CONSUMED|LOAD_FAILED|RELEASED."""

    def __init__(self):
        self.segments: dict[str, str] = {}
        self.manifests: dict[str, object] = {}
        self.calls: list[tuple] = []
        self.refuse_put = False
        self.started = self.stopped = 0

    def descriptor(self):
        return dict(DESC)

    def start(self):
        self.started += 1

    def shutdown(self):
        self.stopped += 1

    # producer
    def open_put(self, xfer_id, manifest):
        self.calls.append(("open_put", xfer_id))
        if self.refuse_put:
            return None
        self.segments[xfer_id] = "WRITING"
        self.manifests[xfer_id] = manifest
        return FakePut(xfer_id, manifest)

    def finish_export(self, h, status):
        self.calls.append(("finish_export", h.xfer_id, status))
        self.segments[h.xfer_id] = status

    def abandon(self, xfer_id):
        self.calls.append(("abandon", xfer_id))
        self.segments.pop(xfer_id, None)

    # consumer
    def publish(self, xfer_id, status="READY", num_tokens=129):
        self.segments[xfer_id] = status
        self.manifests[xfer_id] = build_manifest(
            num_tokens, list(range(-(-num_tokens // BLOCK)))
        )

    def open_get(self, desc: TransferDescriptor):
        self.calls.append(("open_get", desc.xfer_id))
        st = self.segments.get(desc.xfer_id, "MISSING")
        if st == "WRITING":
            return None
        if st in ("READY", "FAILED"):
            return FakeGet(desc.xfer_id, self.manifests.get(desc.xfer_id), st)
        return FakeGet(desc.xfer_id, None, "MISSING")

    def finish_import(self, h, ok):
        self.calls.append(("finish_import", h.xfer_id, ok))
        self.segments[h.xfer_id] = "CONSUMED" if ok else "LOAD_FAILED"

    def release_remote(self, xfer_id):
        self.calls.append(("release_remote", xfer_id))
        if xfer_id in self.segments and self.segments[xfer_id] in (
            "WRITING",
            "READY",
            "FAILED",
        ):
            self.segments[xfer_id] = "RELEASED"

    def count(self, op, xfer_id=None):
        return sum(
            1 for c in self.calls if c[0] == op and (xfer_id is None or c[1] == xfer_id)
        )


class FakeModel:
    kv_transfer_hybrid_state = True

    def __init__(self):
        self.calls: list[tuple] = []
        self.validate_error = None
        self.install_error = None
        self.export_error = None
        self.import_error = None

    def describe_request_state(self, num_tokens, block_ids):
        self.calls.append(("describe", num_tokens, list(block_ids)))
        return build_manifest(num_tokens, block_ids)

    def export_request_state(self, block_ids, num_tokens, slot, sinks):
        self.calls.append(
            ("export", list(block_ids), num_tokens, slot, sorted(sinks)[:1])
        )
        if self.export_error:
            raise self.export_error

    def import_kv_blocks(self, sources, block_ids, num_tokens, *, chunk_range=None):
        self.calls.append(
            ("import_kv_blocks", list(block_ids), num_tokens, chunk_range)
        )
        if self.import_error:
            raise self.import_error
        nchunks = cdiv(cdiv(num_tokens, BLOCK), 32)
        return len(
            range(nchunks) if chunk_range is None else range(nchunks)[chunk_range]
        )

    def validate_gdn_parts(self, sources):
        self.calls.append(("validate_gdn_parts",))
        if self.validate_error:
            raise self.validate_error

    def install_gdn_state(self, sources, slot):
        self.calls.append(("install_gdn_state", slot))
        if self.install_error:
            raise self.install_error

    def names(self):
        return [c[0] for c in self.calls]


class FakeRunner:
    """The 6.2 remote-slot API over ``_req_state_slot``."""

    def __init__(self, slots=8):
        self.tt_per_lane_max_num_seqs = slots
        self._req_state_slot: dict[str, int] = {}
        self.requests: dict[str, object] = {}
        self._remote_loading: dict[str, int] = {}
        self._remote_ready: set[str] = set()
        self.calls: list[tuple] = []

    def held_state_slots(self, include_loading):
        return {
            s
            for r, s in self._req_state_slot.items()
            if r in self.requests
            or (
                include_loading
                and (r in self._remote_loading or r in self._remote_ready)
            )
        }

    def claim_remote_state_slot(self, req_id):
        self.calls.append(("claim", req_id))
        held = self.held_state_slots(include_loading=True)
        free = [s for s in range(self.tt_per_lane_max_num_seqs) if s not in held]
        if not free:
            return None
        self._req_state_slot[req_id] = free[0]
        self._remote_loading[req_id] = free[0]
        return free[0]

    def remote_slot_of(self, req_id):
        return self._req_state_slot[req_id]

    def mark_remote_ready(self, req_id):
        self.calls.append(("ready", req_id))
        self._remote_loading.pop(req_id, None)
        self._remote_ready.add(req_id)

    def release_remote_slot(self, req_id):
        self.calls.append(("release", req_id))
        self._remote_loading.pop(req_id, None)
        self._remote_ready.discard(req_id)
        self._req_state_slot.pop(req_id, None)

    # test helpers
    def fill_local(self, n, prefix="local"):
        for i in range(n):
            rid = f"{prefix}{i}"
            self.requests[rid] = object()
            self._req_state_slot[rid] = i

    def move_slot(self, req_id, new_slot):
        """What a decode-step gather remap does to a loading claim (6.2 `moved`)."""
        self._req_state_slot[req_id] = new_slot
        if req_id in self._remote_loading:
            self._remote_loading[req_id] = new_slot

    def join(self, req_id):
        """The request's row entered the batch (`_update_states`)."""
        self.requests[req_id] = object()
        self._remote_ready.discard(req_id)


def recv_meta(rid, num_tokens=129, blocks=None, chunk_tokens=2048):
    xid = xfer_id_for("p0", rid + "-p")
    if blocks is None:
        blocks = list(range(1000, 1000 + -(-num_tokens // BLOCK)))
    return RecvMeta(
        local_block_ids=blocks,
        num_tokens=num_tokens,
        xfer=TransferDescriptor(
            engine_id="p0",
            request_id=rid + "-p",
            xfer_id=xid,
            num_tokens=num_tokens,
            transport=dict(DESC),
            layout_version=1,
            chunk_tokens=chunk_tokens,
            expiry=None,
            prompt_hash="",
            fingerprint="",
        ),
    )


def make_worker(role="consumer", slots=8, **kw):
    tr, model, runner = FakeTransport(), FakeModel(), FakeRunner(slots)
    syncs = []
    w = TTKVWorker(
        None,
        runner,
        None,
        tr,
        model=model,
        is_producer=role in ("producer", "both"),
        is_consumer=role in ("consumer", "both"),
        block_size=BLOCK,
        idle_sleep_s=0.0,
        sync_device=lambda: syncs.append(1),
        xfer_id_fn=lambda r: xfer_id_for("p0", r),
        **kw,
    )
    w._syncs = syncs
    return w, tr, model, runner


def meta(**kw):
    return TTKVConnectorMetadata(**kw)


def step(w, m=None, finished=(), join=()):
    """One engine step as the runner drives it (6.2): begin, forward, end, report."""
    w.begin_step(m or meta(), set(finished), set(join))
    w.end_step()
    fin = w.get_finished(set(finished))
    inv = w.take_invalid_block_ids()
    wm = w.build_worker_meta()
    return fin, inv, wm


# --------------------------------------------------------------------------- #
# consumer: happy path and ORDER (I11)
# --------------------------------------------------------------------------- #
def test_consumer_two_phase_import_order_and_exactly_once():
    w, tr, model, runner = make_worker()
    rm = recv_meta("r1")
    tr.publish(rm.xfer.xfer_id)
    # step k: admission. begin -> claim + poll only;
    # end -> K/V import + validate -> KV_DONE
    w.begin_step(meta(reqs_to_recv={"r1": rm}), set(), set())
    assert model.calls == []  # NO device write at step-begin for a fresh admission
    assert runner._req_state_slot["r1"] == 0 and "r1" in runner._remote_loading
    assert w.loads()["r1"].state == LoadState.IMPORTING_KV
    w.end_step()
    assert model.names() == ["import_kv_blocks", "validate_gdn_parts"]
    assert model.calls[0][1:] == (rm.local_block_ids, 129, slice(0, 1))
    assert w._syncs == [1]
    assert "r1" in runner._remote_ready and w.loads()["r1"].state == LoadState.KV_DONE
    assert w.get_finished(set()) == (None, {"r1"})
    assert w.take_invalid_block_ids() == set()
    wm = w.build_worker_meta()
    assert wm == TTKVWorkerMeta(7)  # first emission + event
    assert tr.segments[rm.xfer.xfer_id] == "READY"  # handle held until the join step
    # a step where the request is not yet promoted: nothing happens, nothing re-reported
    fin, inv, wm = step(w)
    assert fin == (None, None) and inv == set() and wm is None
    assert model.names() == ["import_kv_blocks", "validate_gdn_parts"]
    # step k+1: the request joins the batch -> GDN install at step-BEGIN, exactly once
    runner.move_slot("r1", 3)  # a decode remap moved the claim in between
    w.begin_step(meta(), set(), {"r1"})
    assert model.names()[-1:] == ["install_gdn_state"] and model.calls[-1] == (
        "install_gdn_state",
        3,
    )
    assert w.loads()["r1"].state == LoadState.INSTALLED
    assert tr.segments[rm.xfer.xfer_id] == "CONSUMED"
    runner.join("r1")
    w.end_step()
    assert w.get_finished(set()) == (None, None)
    assert "r1" not in w.loads()
    assert w.build_worker_meta() is not None  # INSTALLED is a terminal event
    assert model.names().count("install_gdn_state") == 1
    st = w.take_stats()
    assert st is not None and st.data["records"]["r1"]["import_install_ms"] >= 0
    assert st.data["import_ms"] and st.data["bytes_import"] == [
        tr.manifests[rm.xfer.xfer_id].total_nbytes
    ]


def test_join_requires_kv_done():
    w, tr, model, runner = make_worker()
    rm = recv_meta("r1")
    tr.publish(rm.xfer.xfer_id, "WRITING")
    w.begin_step(meta(reqs_to_recv={"r1": rm}), set(), set())
    assert w.loads()["r1"].state == LoadState.PENDING_READY
    with pytest.raises(RuntimeError):
        w.begin_step(meta(), set(), {"r1"})
    with pytest.raises(RuntimeError):
        w.begin_step(meta(), set(), {"unknown"})
    with pytest.raises(RuntimeError):
        w.begin_step(meta(reqs_to_recv={"r2": recv_meta("r2")}), set(), {"r2"})


def test_install_failure_is_fatal():
    w, tr, model, runner = make_worker()
    rm = recv_meta("r1")
    tr.publish(rm.xfer.xfer_id)
    step(w, meta(reqs_to_recv={"r1": rm}))
    model.install_error = ValueError("device error")
    with pytest.raises(RuntimeError, match="after finished_recving"):
        w.begin_step(meta(), set(), {"r1"})


def test_writing_header_is_polled_until_ready():
    w, tr, model, runner = make_worker()
    rm = recv_meta("r1")
    tr.publish(rm.xfer.xfer_id, "WRITING")
    fin, inv, wm = step(w, meta(reqs_to_recv={"r1": rm}))
    assert (
        fin == (None, None)
        and model.calls == []
        and w.loads()["r1"].state == LoadState.PENDING_READY
    )
    assert wm == TTKVWorkerMeta(7)  # slot claimed -> count changed
    fin, inv, wm = step(w)
    # polled at step begin AND step end (the fabric claim's sends land during the
    # forward): two open_get per step
    assert fin == (None, None) and wm is None and tr.count("open_get") == 4
    tr.segments[rm.xfer.xfer_id] = "READY"
    fin, inv, wm = step(w)
    assert fin == (None, {"r1"}) and w.loads()["r1"].state == LoadState.KV_DONE
    assert wm == TTKVWorkerMeta(7)  # event


# --------------------------------------------------------------------------- #
# consumer: failures (I8: id + FULL block list in the same call)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("header", ["FAILED", "MISSING"])
def test_bad_header_fails_load_same_call(header):
    w, tr, model, runner = make_worker()
    rm = recv_meta("r1", blocks=[3, 9, 27])
    if header == "FAILED":
        tr.publish(rm.xfer.xfer_id, "FAILED")
    fin, inv, wm = step(w, meta(reqs_to_recv={"r1": rm}))
    assert fin == (None, {"r1"}) and inv == {3, 9, 27}
    assert model.calls == [] and "r1" not in runner._req_state_slot
    assert (
        tr.count("release_remote", rm.xfer.xfer_id) == 1
        and tr.count("finish_import") == 0
    )
    assert wm == TTKVWorkerMeta(8) and "r1" not in w.loads()
    assert step(w)[0] == (None, None)  # never re-reported


def test_validate_failure_fails_load_same_call():
    w, tr, model, runner = make_worker()
    rm = recv_meta("r1", blocks=[3, 9, 27])
    tr.publish(rm.xfer.xfer_id)
    model.validate_error = RuntimeError("crc mismatch")
    fin, inv, wm = step(w, meta(reqs_to_recv={"r1": rm}))
    assert fin == (None, {"r1"}) and inv == {3, 9, 27}
    assert model.names() == ["import_kv_blocks", "validate_gdn_parts"]
    assert "r1" not in runner._req_state_slot and "r1" not in runner._remote_ready
    assert tr.segments[rm.xfer.xfer_id] == "LOAD_FAILED"
    assert "r1" not in w.loads() and wm == TTKVWorkerMeta(8)
    assert w.take_stats().data["num_failed_loads"] == [1]


def test_import_exception_fails_load_and_oom_reraises():
    w, tr, model, runner = make_worker()
    rm = recv_meta("r1")
    tr.publish(rm.xfer.xfer_id)
    model.import_error = RuntimeError("paged_fill_cache failed")
    fin, inv, wm = step(w, meta(reqs_to_recv={"r1": rm}))
    assert fin == (None, {"r1"}) and inv == set(rm.local_block_ids)
    # OOM: bookkeeping first, then the exception propagates (R17)
    w2, tr2, model2, runner2 = make_worker()
    rm2 = recv_meta("r2")
    tr2.publish(rm2.xfer.xfer_id)
    model2.import_error = RuntimeError(
        "TT_THROW: Out of Memory: Not enough space to allocate"
    )
    w2.begin_step(meta(reqs_to_recv={"r2": rm2}), set(), set())
    with pytest.raises(RuntimeError, match="Out of Memory"):
        w2.end_step()
    assert w2.loads()[
        "r2"
    ].state == LoadState.FAILED and w2.take_invalid_block_ids() == set(
        rm2.local_block_ids
    )
    assert "r2" not in runner2._req_state_slot


def test_no_progress_import_is_a_failure_not_a_spin():
    w, tr, model, runner = make_worker()
    rm = recv_meta("r1")
    tr.publish(rm.xfer.xfer_id)
    model.import_kv_blocks = lambda *a, **k: 0
    fin, inv, wm = step(w, meta(reqs_to_recv={"r1": rm}))
    assert fin == (None, {"r1"}) and inv == set(rm.local_block_ids)


# --------------------------------------------------------------------------- #
# consumer: aborts (N5) and exactly-once
# --------------------------------------------------------------------------- #
def test_abort_at_begin_step_releases_without_claim():
    w, tr, model, runner = make_worker()
    rm = recv_meta("r1")
    tr.publish(rm.xfer.xfer_id)
    fin, inv, wm = step(w, meta(reqs_to_recv={"r1": rm}), finished={"r1"})
    assert fin == (None, {"r1"}) and inv == set()
    assert runner.calls == [] and model.calls == []
    assert tr.segments[rm.xfer.xfer_id] == "RELEASED" and "r1" not in w.loads()
    assert wm == TTKVWorkerMeta(8)


def test_abort_while_pending_slot_releases_and_emits_meta():
    w, tr, model, runner = make_worker(slots=2)
    runner.fill_local(2)  # no free slot -> PENDING_SLOT
    rm = recv_meta("r1")
    tr.publish(rm.xfer.xfer_id)
    fin, inv, wm = step(w, meta(reqs_to_recv={"r1": rm}))
    assert fin == (None, None) and w.loads()["r1"].state == LoadState.PENDING_SLOT
    assert wm == TTKVWorkerMeta(0)  # first emission
    assert step(w)[2] is None
    fin, inv, wm = step(w, finished={"r1"})
    assert fin == (None, {"r1"}) and inv == set()
    assert tr.count("release_remote", rm.xfer.xfer_id) == 1 and "r1" not in w.loads()
    assert wm == TTKVWorkerMeta(0)  # abort counts as a terminal event (N5)
    assert ("release", "r1") in runner.calls


def test_abort_while_importing_after_ready_finish_imports_failed():
    w, tr, model, runner = make_worker()
    rm = recv_meta("r1")
    tr.publish(rm.xfer.xfer_id, "WRITING")
    step(w, meta(reqs_to_recv={"r1": rm}))
    tr.segments[rm.xfer.xfer_id] = "READY"
    w.begin_step(meta(), set(), set())  # -> IMPORTING_KV (handle claimed)
    assert w.loads()["r1"].state == LoadState.IMPORTING_KV
    w.begin_step(meta(), {"r1"}, set())  # aborted this step: skipped by the FIFO loop
    w.end_step()
    assert model.calls == []  # no import for a finished id
    assert w.get_finished({"r1"}) == (None, {"r1"})
    assert (
        tr.segments[rm.xfer.xfer_id] == "LOAD_FAILED"
        and "r1" not in runner._req_state_slot
    )


def test_kv_done_aborted_before_join_not_reported_twice():
    w, tr, model, runner = make_worker()
    rm = recv_meta("r1")
    tr.publish(rm.xfer.xfer_id)
    fin, _, _ = step(w, meta(reqs_to_recv={"r1": rm}))
    assert fin == (None, {"r1"})
    fin, inv, wm = step(w, finished={"r1"})
    assert fin == (None, None) and inv == set()
    assert (
        "r1" not in w.loads()
        and "r1" not in runner._req_state_slot
        and "r1" not in runner._remote_ready
    )
    assert tr.segments[rm.xfer.xfer_id] == "LOAD_FAILED"
    assert model.names().count("install_gdn_state") == 0
    assert wm == TTKVWorkerMeta(8)


def test_release_remote_twice_and_duplicate_recv_are_harmless():
    w, tr, model, runner = make_worker()
    xid = xfer_id_for("p0", "gone")
    step(w, meta(to_release={xid}))
    step(w, meta(to_release={xid}))
    assert tr.count("release_remote", xid) == 2
    rm = recv_meta("r1")
    tr.publish(rm.xfer.xfer_id)
    w.begin_step(meta(reqs_to_recv={"r1": rm}), set(), set())
    w.begin_step(meta(reqs_to_recv={"r1": rm}), set(), set())  # duplicate ignored
    assert len(w.loads()) == 1 and runner.calls.count(("claim", "r1")) == 1


# --------------------------------------------------------------------------- #
# consumer: chunked import, slot capacity, worker meta
# --------------------------------------------------------------------------- #
def test_chunk_cursor_resumes_and_install_uses_slot_read_at_join():
    w, tr, model, runner = make_worker(max_import_chunks_per_step=1)
    rm = recv_meta("r1", num_tokens=4095)  # 64 blocks -> 2 chunks of 32
    tr.publish(rm.xfer.xfer_id, num_tokens=4095)
    fin, inv, wm = step(w, meta(reqs_to_recv={"r1": rm}))
    assert fin == (None, None) and w.loads()["r1"].state == LoadState.IMPORTING_KV
    assert model.calls[-1] == (
        "import_kv_blocks",
        rm.local_block_ids,
        4095,
        slice(0, 1),
    )
    assert (
        w.loads()["r1"].chunk_cursor == 1 and "validate_gdn_parts" not in model.names()
    )
    runner.move_slot("r1", 6)  # remap between the two chunk steps (A-G1)
    fin, inv, wm = step(w)
    assert fin == (None, {"r1"}) and model.calls[-2] == (
        "import_kv_blocks",
        rm.local_block_ids,
        4095,
        slice(1, 2),
    )
    assert model.names()[-1] == "validate_gdn_parts"
    runner.move_slot("r1", 2)  # and again before the join step
    w.begin_step(meta(), set(), {"r1"})
    assert model.calls[-1] == ("install_gdn_state", 2)


def test_budget_is_shared_fifo_across_jobs():
    w, tr, model, runner = make_worker(max_import_chunks_per_step=2)
    a, b = recv_meta("a", num_tokens=4095), recv_meta("b", num_tokens=4095)
    tr.publish(a.xfer.xfer_id, num_tokens=4095)
    tr.publish(b.xfer.xfer_id, num_tokens=4095)
    fin, _, _ = step(w, meta(reqs_to_recv={"a": a, "b": b}))
    assert fin == (None, {"a"}) and w.loads()["b"].chunk_cursor == 0
    fin, _, _ = step(w)
    assert fin == (None, {"b"})


def test_pending_slot_waits_for_capacity_and_free_slots_counts_loading():
    w, tr, model, runner = make_worker(slots=2)
    runner.fill_local(2)
    rm = recv_meta("r1")
    tr.publish(rm.xfer.xfer_id)
    fin, inv, wm = step(w, meta(reqs_to_recv={"r1": rm}))
    assert fin == (None, None) and model.calls == [] and wm == TTKVWorkerMeta(0)
    del runner.requests[
        "local1"
    ]  # a decode finished; _release_dead_state_slots popped its claim
    del runner._req_state_slot["local1"]
    fin, inv, wm = step(w)
    assert fin == (None, {"r1"}) and runner._req_state_slot["r1"] == 1
    assert w.free_state_slots() == 0 and wm == TTKVWorkerMeta(0)


def test_worker_meta_only_on_change_or_event():
    w, tr, model, runner = make_worker()
    assert w.build_worker_meta() == TTKVWorkerMeta(
        8
    )  # initial sync always emits (NIT-1)
    assert w.build_worker_meta() is None
    runner.fill_local(3)
    assert w.build_worker_meta() == TTKVWorkerMeta(5)
    assert w.build_worker_meta() is None
    w._events_this_step = True
    assert w.build_worker_meta() == TTKVWorkerMeta(5)
    assert w.build_worker_meta() is None


def test_producer_only_worker_emits_no_meta_and_rejects_join():
    w, tr, model, runner = make_worker("producer")
    assert w.build_worker_meta() is None
    with pytest.raises(RuntimeError):
        w.begin_step(meta(), set(), {"r1"})


# --------------------------------------------------------------------------- #
# producer
# --------------------------------------------------------------------------- #
def test_export_then_exactly_once_finished_sending(monkeypatch):
    w, tr, model, runner = make_worker("producer")
    runner._req_state_slot["p1"] = 0
    xid = xfer_id_for("p0", "p1")
    sm = SaveMeta(block_ids=[4, 5, 6], num_tokens=129, xfer_id=xid)
    fin, inv, wm = step(w, meta(reqs_to_save={"p1": sm}))
    assert model.names() == ["describe", "export"] and model.calls[1][1:4] == (
        [4, 5, 6],
        129,
        0,
    )
    assert tr.segments[xid] == "READY" and tr.calls[:2] == [
        ("open_put", xid),
        ("finish_export", xid, "READY"),
    ]
    assert fin == (None, None)  # not armed yet: the id is not finished in this step
    assert wm is None
    fin, inv, wm = step(
        w, meta(reqs_to_send={"p1": 1e12})
    )  # request_finished happened -> armed
    assert fin == ({"p1"}, None)
    assert step(w)[0] == (None, None)  # exactly once
    st = w.take_stats()
    assert (
        st.data["records"]["p1"]["export_ok"] is True and st.data["bytes_export"][0] > 0
    )


def test_failed_export_is_still_reported_and_marks_header_failed():
    w, tr, model, runner = make_worker("producer")
    xid = xfer_id_for("p0", "p1")
    model.export_error = RuntimeError("ttnn slice failed")
    step(w, meta(reqs_to_save={"p1": SaveMeta([1], 60, xid)}))
    assert tr.segments[xid] == "FAILED"
    fin, _, _ = step(w, meta(reqs_to_send={"p1": 1e12}))
    assert fin == ({"p1"}, None)
    assert w.take_stats().data["num_failed_exports"] == [1]


def test_refused_open_put_is_a_failed_export():
    w, tr, model, runner = make_worker("producer")
    tr.refuse_put = True
    xid = xfer_id_for("p0", "p1")
    step(w, meta(reqs_to_save={"p1": SaveMeta([1], 60, xid)}))
    assert "export" not in model.names() and xid not in tr.segments
    fin, _, _ = step(w, meta(reqs_to_send={"p1": 1e12}))
    assert fin == ({"p1"}, None)


def test_armed_but_never_exported_is_reported_with_warning(monkeypatch):
    w, tr, model, runner = make_worker("producer")
    warnings = []
    monkeypatch.setattr(
        wk.logger, "warning", lambda msg, *a, **k: warnings.append(msg % a)
    )
    fin, _, _ = step(w, meta(reqs_to_send={"ghost": 1e12}))
    assert fin == ({"ghost"}, None)
    assert any("never exported" in m for m in warnings)
    assert step(w)[0] == (None, None)


def test_not_processed_and_abort_mid_step_abandon_the_segment():
    w, tr, model, runner = make_worker("producer")
    xid1, xid2 = xfer_id_for("p0", "p1"), xfer_id_for("p0", "p2")
    # p1: save requested but the request was aborted before the step ran
    w.begin_step(
        meta(reqs_to_save={"p1": SaveMeta([1], 60, xid1)}, reqs_not_processed={"p1"}),
        set(),
        set(),
    )
    w.end_step()
    assert model.calls == [] and tr.calls == [("abandon", xid1)]
    # p2: pending save whose id is finished at get_finished (aborted mid-step)
    w.begin_step(meta(reqs_to_save={"p2": SaveMeta([2], 60, xid2)}), set(), set())
    # (no end_step: execute_model raised)
    assert w.get_finished({"p2"}) == (None, None)
    assert ("abandon", xid2) in tr.calls and w._pending_saves == {}


def test_kv_both_worker_runs_both_roles_in_one_step():
    w, tr, model, runner = make_worker("both")
    runner._req_state_slot["rid-p"] = 0
    runner.requests["rid-p"] = object()
    runner.requests["other"] = object()  # an unrelated running decode
    runner._req_state_slot["other"] = 1
    xid_p = xfer_id_for("p0", "rid-p")
    rm = recv_meta("rid")
    tr.publish(rm.xfer.xfer_id)
    fin, inv, wm = step(
        w,
        meta(
            reqs_to_save={"rid-p": SaveMeta([1, 2, 3], 129, xid_p)},
            reqs_to_recv={"rid": rm},
        ),
    )
    assert fin == (None, {"rid"})
    assert tr.segments[xid_p] == "READY" and w.loads()["rid"].state == LoadState.KV_DONE
    assert wm == TTKVWorkerMeta(5)  # rid-p slot 0, other slot 1, the load slot 2
    fin, _, _ = step(w, meta(reqs_to_send={"rid-p": 1e12}))
    assert fin == ({"rid-p"}, None)
    # stall accounting: device work happened while another request (other) was live
    st = w.take_stats()
    assert st.data["stall_ms_other_users"]


def test_shutdown_stops_transport():
    w, tr, model, runner = make_worker()
    w.shutdown()
    assert tr.stopped == 1


def test_stats_default_instance():
    w, tr, model, runner = make_worker()
    assert isinstance(w.stats, TTKVConnectorStats) and w.take_stats() is None


# --------------------------------------------------------------------------- #
# [fix] PD polish: NIT-5 (finished ids at step-end), lease check before the claim
# --------------------------------------------------------------------------- #
def test_end_step_skips_ids_finished_after_step_begin():
    """Critic NIT-5: ``end_step`` takes this step's finished ids too, so an
    aborted IMPORTING_KV job never imports a chunk into blocks that may already
    belong to another request -- even when ``begin_step`` did not see the id."""
    w, tr, model, runner = make_worker()
    rm = recv_meta("r1")
    tr.publish(rm.xfer.xfer_id)
    w.begin_step(meta(reqs_to_recv={"r1": rm}), set(), set())
    assert w.loads()["r1"].state == LoadState.IMPORTING_KV

    w.end_step(finished_req_ids={"r1"})

    assert model.calls == [], "no K/V device write for a finished id"
    assert w.loads()["r1"].state == LoadState.IMPORTING_KV
    assert w.get_finished({"r1"}) == (None, {"r1"})
    assert tr.segments[rm.xfer.xfer_id] == "LOAD_FAILED" and "r1" not in w.loads()
    # Control: without the id the same step imports.
    rm2 = recv_meta("r2")
    tr.publish(rm2.xfer.xfer_id)
    w.begin_step(meta(reqs_to_recv={"r2": rm2}), set(), set())
    w.end_step(finished_req_ids=set())
    assert model.names() == ["import_kv_blocks", "validate_gdn_parts"]
    assert w.loads()["r2"].state == LoadState.KV_DONE


def test_pending_ready_job_with_an_expired_lease_fails_without_claiming():
    """Audit minor (janitor-vs-claim race): a job that waited for a slot past
    the producer's lease must not race the janitor's sweep with a claim; it
    FAILS (blocks invalidated, recompute) and releases the segment."""
    w, tr, model, runner = make_worker(slots=1)
    runner.fill_local(1)  # no free slot -> PENDING_SLOT
    rm = recv_meta("r1")
    rm.xfer.expiry = time.time() + 3600.0  # valid at the offer
    tr.publish(rm.xfer.xfer_id)
    fin, inv, wm = step(w, meta(reqs_to_recv={"r1": rm}))
    assert fin == (None, None) and w.loads()["r1"].state == LoadState.PENDING_SLOT

    rm.xfer.expiry = time.time() - 1.0  # ran out while waiting for the slot
    runner.requests.clear()
    runner._req_state_slot.clear()  # the slot frees
    fin, inv, wm = step(w)

    assert fin == (None, {"r1"}) and inv == set(rm.local_block_ids)
    assert tr.count("open_get", rm.xfer.xfer_id) == 0, "never claimed"
    assert tr.count("release_remote", rm.xfer.xfer_id) == 1
    assert tr.segments[rm.xfer.xfer_id] == "RELEASED"
    assert "r1" not in runner._req_state_slot and "r1" not in w.loads()
    assert model.calls == []

    # Control: a valid lease claims and imports as before.
    rm2 = recv_meta("r2")
    rm2.xfer.expiry = time.time() + 3600.0
    tr.publish(rm2.xfer.xfer_id)
    fin, inv, wm = step(w, meta(reqs_to_recv={"r2": rm2}))
    assert fin == (None, {"r2"}) and w.loads()["r2"].state == LoadState.KV_DONE
    assert tr.count("open_get", rm2.xfer.xfer_id) == 1


def test_pending_ready_claim_that_becomes_ready_during_the_forward_imports_same_step():
    """Fabric claim-gated sends: begin_step claims (open_get -> None), the
    producer sends during our forward, end_step re-polls and imports in the SAME
    step instead of the next one."""
    w, tr, model, runner = make_worker()
    rm = recv_meta("r1")
    tr.publish(rm.xfer.xfer_id, "READY")
    polls = {"n": 0}
    orig_open_get = tr.open_get

    def open_get(desc):
        polls["n"] += 1
        if polls["n"] == 1:
            tr.calls.append(("open_get", desc.xfer_id))
            return None  # claimed, the producer has not pumped yet
        return orig_open_get(desc)

    tr.open_get = open_get
    fin, inv, wm = step(w, meta(reqs_to_recv={"r1": rm}))
    assert polls["n"] == 2 and tr.count("open_get") == 2
    assert w.loads()["r1"].state == LoadState.KV_DONE and fin == (None, {"r1"})
    assert "import_kv_blocks" in model.names()


def _claimed_then_gated(tr, deadlines):
    """A fabric-shaped FakeTransport: open_get claims (None) until ``gate['open']``
    and reports a claim deadline from ``deadlines`` (claim_deadline seam)."""
    gate = {"open": False}
    orig = tr.open_get
    tr.claim_deadline = lambda xid: deadlines.get(xid)

    def open_get(desc):
        if not gate["open"]:
            tr.calls.append(("open_get", desc.xfer_id))
            return None  # claimed; the producer has not pumped yet
        return orig(desc)

    tr.open_get = open_get
    return gate


def test_claimed_job_outlives_the_ready_lease_while_the_producer_is_busy():
    """Claim-gated fabric: the READY lease started at P's publish, but P is
    mid-prefill of the NEXT queued request (a whole prompt per P step: 26 s @32k,
    longer than the 30 s lease above ~36k tokens).  After the claim the worker
    asks the transport for the CLAIM deadline and keeps polling; the import
    completes when the marker lands."""
    w, tr, model, runner = make_worker()
    rm = recv_meta("r1")
    rm.xfer.expiry = time.time() + 3600.0  # valid at the offer
    tr.publish(rm.xfer.xfer_id, "READY")
    deadlines = {rm.xfer.xfer_id: time.time() + 3600.0}  # claim + claim_lease_s
    gate = _claimed_then_gated(tr, deadlines)
    fin, inv, wm = step(w, meta(reqs_to_recv={"r1": rm}))
    assert w.loads()["r1"].state == LoadState.PENDING_READY and fin == (None, None)
    rm.xfer.expiry = time.time() - 1.0  # the READY lease ran out while P prefills
    for _ in range(3):
        fin, inv, wm = step(w)
        assert w.loads()["r1"].state == LoadState.PENDING_READY, (
            "must not expire on the READY lease once the claim is made"
        )
        assert fin == (None, None) and inv == set()
    assert tr.count("release_remote", rm.xfer.xfer_id) == 0
    gate["open"] = True  # the marker landed: P's step ended
    fin, inv, wm = step(w)
    assert w.loads()["r1"].state == LoadState.KV_DONE and fin == (None, {"r1"})
    assert "import_kv_blocks" in model.names()


def test_claimed_job_fails_at_the_claim_deadline_and_releases():
    """The hard bound: a producer that never sends (dead / stuck) is caught at the
    transport's claim deadline (claim + claim_lease_s), even though the READY
    lease is still valid; the claim is released (fence + drop) and the request
    recomputes.  A raising claim_deadline falls back to the READY lease."""
    w, tr, model, runner = make_worker()
    rm = recv_meta("r1")
    rm.xfer.expiry = time.time() + 3600.0
    tr.publish(rm.xfer.xfer_id, "READY")
    deadlines: dict = {}
    _claimed_then_gated(tr, deadlines)
    fin, inv, wm = step(w, meta(reqs_to_recv={"r1": rm}))
    assert w.loads()["r1"].state == LoadState.PENDING_READY
    deadlines[rm.xfer.xfer_id] = time.time() - 1.0  # claim lease ran out
    fin, inv, wm = step(w)
    assert fin == (None, {"r1"}) and inv == set(rm.local_block_ids)
    assert tr.count("release_remote", rm.xfer.xfer_id) == 1
    assert "r1" not in w.loads() and model.calls == []

    def boom(xid):
        raise RuntimeError("no such claim")

    tr.claim_deadline = boom
    rm2 = recv_meta("r2")
    rm2.xfer.expiry = time.time() - 1.0  # READY lease gone, deadline unknown
    tr.publish(rm2.xfer.xfer_id, "READY")
    fin, inv, wm = step(w, meta(reqs_to_recv={"r2": rm2}))
    assert fin == (None, {"r2"}) and inv == set(rm2.local_block_ids)
    assert tr.count("open_get", rm2.xfer.xfer_id) == 0  # READY lease rule applied


def test_end_step_pumps_a_transport_that_has_pump_and_survives_its_errors():
    """The claim-gated protocol's per-step clock lives in end_step: a transport
    with pump() is pumped after the exports / imports of every step, a raising
    pump is logged and never takes the step down, shm (no pump) is a no-op."""
    w, tr, model, runner = make_worker(role="both")
    log: list = []
    tr.pump = lambda: log.append(("pump", len(tr.calls)))
    step(w)
    step(w)
    assert [k for k, _ in log] == ["pump", "pump"]
    rm = recv_meta("r1")
    tr.publish(rm.xfer.xfer_id, "READY")
    step(w, meta(reqs_to_recv={"r1": rm}))
    # the pump ran AFTER this step's import (open_get / finish_import calls precede)
    assert log[-1][1] >= tr.count("open_get") and w.loads()["r1"].state == (
        LoadState.KV_DONE
    )

    def boom():
        raise ValueError("pump broke")

    tr.pump = boom
    step(w)  # no raise
    del tr.pump
    step(w)  # shm shape: no pump attribute, no-op
