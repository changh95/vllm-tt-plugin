# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Host-only tests for ``FabricSocketTransport`` (PHASE3_NOTES 3, hybrid control plane).

No ttnn: ``FakeSocketLayer`` implements the ``fabric_socket.SocketLayer`` seam over a
``FakeWorld`` that simulates the two MPI ranks in one process -- a ``threading.Barrier``
for ``distributed_context_barrier``, a socket rendezvous, and a ``FakeChannel`` that
pairs sends and recvs strictly in FIFO order (a spec mismatch between a paired send
and recv is an error, as direct mode would deliver garbage).  Tensors are numpy byte
arrays; the block-major -> head-major relayout is checked against an independent
permutation.  Covered: start (one socket, lockstep warm-up, rendezvous spec table,
every construction-time failure), producer pool accounting, canonical send order,
sidecar + shm header publish, consumer sequence gate, drains (expired / released /
aborted mid-import / out-of-order), rec-set exhaustion, byte-exact end-to-end identity
(single thread and two rank threads with blocking recvs), taps roundtrip, chunk_range
across steps, reclaim, pump, shutdown, and the factory wiring.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_tt_plugin.kv_transfer.transport import fabric_socket, make_transport
from vllm_tt_plugin.kv_transfer.transport.base import (
    KV_CHUNK_SHAPE,
    TILE_RECORD_BYTES,
    Manifest,
    build_manifest,
)
from vllm_tt_plugin.kv_transfer.transport.fabric import (
    CLOSING_SUFFIX,
    SIDECAR_NAME,
    FabricConfig,
    FabricSink,
    FabricSocketTransport,
    FabricSource,
    _ControlSegments,
    export_pool_bytes,
    max_export_kv_buffers,
    spec_nbytes,
)
from vllm_tt_plugin.kv_transfer.transport.fabric_socket import RendezvousTimeout
from vllm_tt_plugin.kv_transfer.transport.shm import (
    FAILED,
    READY,
    RELEASED,
    DumpfileSink,
    DumpfileSource,
    read_header,
    read_status,
)

P_ENGINE, D_ENGINE = "prefill-0", "decode-0"
HX = ["%032x" % (0x1000 + i) for i in range(8)]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def xfer(i: int = 0, engine: str = P_ENGINE) -> str:
    return f"{engine}:{HX[i]}"


@dataclass
class Desc:  # TransferDescriptor look-alike (design 3.4); config-derived descriptor
    xfer_id: str
    transport: dict | None = None

    def __post_init__(self):
        if self.transport is None:
            self.transport = {"kind": "fabric", "mode": "dumpfile", "layout_version": 1}


class Clock:
    def __init__(self, t0: float = 1_000_000.0) -> None:
        self.now = t0

    def __call__(self) -> float:
        return self.now


# --- fakes ----------------------------------------------------------------------------


@dataclass
class FakeTensor:
    shape: tuple[int, ...]
    dtype: str
    layout: str
    data: np.ndarray
    on_device: bool = True
    freed: bool = False

    @property
    def spec(self):
        return (tuple(self.shape), self.dtype, self.layout)

    @classmethod
    def zeros(cls, spec, on_device=True):
        return cls(
            tuple(spec[0]),
            spec[1],
            spec[2],
            np.zeros(spec_nbytes(spec), np.uint8),
            on_device,
        )

    @classmethod
    def random(cls, spec, rng, on_device=True):
        return cls(
            tuple(spec[0]),
            spec[1],
            spec[2],
            rng.integers(0, 256, spec_nbytes(spec), dtype=np.uint8),
            on_device,
        )


class FakeSpecMismatch(AssertionError):
    pass


class FakeChannel:
    """One FIFO channel: sends and recvs pair up strictly in posting order."""

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.sends: deque[tuple[np.ndarray, tuple]] = deque()
        self.recvs: deque[tuple[FakeTensor, tuple, threading.Event]] = deque()
        self.delivered = 0
        self.log: list[tuple] = []  # (spec) per delivered item

    def _match_locked(self) -> None:
        while self.sends and self.recvs:
            data, s_spec = self.sends.popleft()
            t, r_spec, ev = self.recvs.popleft()
            if s_spec != r_spec:
                raise FakeSpecMismatch(f"send {s_spec} paired with recv {r_spec}")
            t.data[:] = data
            self.delivered += 1
            self.log.append(s_spec)
            ev.set()

    def send(self, data: np.ndarray, spec: tuple) -> None:
        with self.cond:
            self.sends.append((data, spec))
            self._match_locked()
            self.cond.notify_all()

    def recv(
        self, t: FakeTensor, spec: tuple, *, block: bool, timeout: float
    ) -> threading.Event:
        ev = threading.Event()
        with self.cond:
            self.recvs.append((t, spec, ev))
            self._match_locked()
            self.cond.notify_all()
        if block and not ev.wait(timeout):
            raise TimeoutError("fake recv: no matching send arrived")
        return ev

    @property
    def pending_sends(self) -> int:
        with self.cond:
            return len(self.sends)

    @property
    def pending_recvs(self) -> int:
        with self.cond:
            return len(self.recvs)


class FakeSocket:
    def __init__(self, channel: FakeChannel, cfg: tuple) -> None:
        self.channel, self.cfg, self.closed = channel, cfg, False

    def close(self) -> None:
        self.closed = True


class FakeWorld:
    """Two ranks in one process."""

    def __init__(self, size: int = 2, barrier_timeout: float = 5.0) -> None:
        self.size = size
        self.barrier = threading.Barrier(size, timeout=barrier_timeout)
        self.sock_barrier = threading.Barrier(size, timeout=barrier_timeout)
        self.channel = FakeChannel()
        self.sock_cfgs: dict[int, tuple] = {}
        self.sockets_created = 0
        self.lock = threading.Lock()

    def layer(self, rank: int, **kw) -> FakeSocketLayer:
        return FakeSocketLayer(self, rank, **kw)


class FakeSocketLayer:
    """``SocketLayer`` over ``FakeWorld``; records program keys like a program cache."""

    def __init__(
        self,
        world: FakeWorld,
        rank: int,
        *,
        distributed: bool = True,
        blocking_recv: bool = False,
        recv_timeout: float = 5.0,
    ) -> None:
        self.world, self._rank = world, rank
        self.distributed, self.blocking_recv, self.recv_timeout = (
            distributed,
            blocking_recv,
            recv_timeout,
        )
        self.compiled: set[tuple] = set()
        self.allocated: list[FakeTensor] = []
        self.deallocated: list[FakeTensor] = []
        self.barriers = 0
        self.sends = self.recvs = 0
        self.sockets: list[FakeSocket] = []
        self._recv_events: list[threading.Event] = []

    # distributed context
    def is_distributed(self):
        return self.distributed

    def rank(self):
        return self._rank

    def size(self):
        return self.world.size

    def barrier(self, timeout_s=None):
        self.barriers += 1
        try:
            self.world.barrier.wait(timeout=timeout_s)
        except threading.BrokenBarrierError as e:
            raise RuntimeError("fabric barrier timed out (peer never arrived)") from e

    # socket
    def create_socket(
        self,
        mesh,
        *,
        connections,
        fifo_bytes,
        sender_rank,
        receiver_rank,
        timeout_s=None,
    ):
        cfg = (connections, fifo_bytes, sender_rank, receiver_rank)
        with self.world.lock:
            self.world.sock_cfgs[self._rank] = cfg
            self.world.sockets_created += 1
        try:
            self.world.sock_barrier.wait(timeout=timeout_s)
        except threading.BrokenBarrierError as e:
            raise RuntimeError("socket handshake timed out") from e
        assert len(set(self.world.sock_cfgs.values())) == 1, self.world.sock_cfgs
        s = FakeSocket(self.world.channel, cfg)
        self.sockets.append(s)
        return s

    def close_socket(self, sock):
        sock.close()

    # tensors
    def allocate(self, mesh, spec):
        t = FakeTensor.zeros(spec)
        self.allocated.append(t)
        return t

    def deallocate(self, t):
        assert not t.freed, "double deallocate"
        t.freed = True
        self.deallocated.append(t)

    def spec_of(self, t):
        return t.spec

    def relayout_copy(self, blk, dst):
        assert blk.on_device and dst.on_device and not blk.freed and not dst.freed
        assert tuple(blk.shape) == KV_CHUNK_SHAPE and blk.dtype == dst.dtype
        self.compiled.add(("relayout", blk.spec, dst.spec))
        dst.data[:] = ref_relayout(blk.data, TILE_RECORD_BYTES[blk.dtype])

    def copy(self, src, dst):
        assert src.spec == dst.spec, (src.spec, dst.spec)
        assert src.on_device and dst.on_device and not src.freed and not dst.freed
        self.compiled.add(("copy", src.spec))
        dst.data[:] = src.data

    def copy_host_to_device(self, host, dst):
        assert not host.on_device and dst.on_device and host.spec == dst.spec
        dst.data[:] = host.data

    def send(self, t, sock):
        assert t.on_device and not t.freed and not sock.closed
        self.compiled.add(("send", t.spec))
        self.sends += 1
        sock.channel.send(t.data.copy(), t.spec)

    def recv(self, t, sock):
        assert t.on_device and not t.freed and not sock.closed
        self.compiled.add(("recv", t.spec))
        self.recvs += 1
        ev = sock.channel.recv(
            t, t.spec, block=self.blocking_recv, timeout=self.recv_timeout
        )
        self._recv_events.append(ev)

    def sync(self, mesh):
        # synchronize_device: every posted recv of this rank has landed
        for ev in self._recv_events:
            if not ev.wait(self.recv_timeout):
                raise TimeoutError("fake sync: a posted recv never completed")
        self._recv_events.clear()

    def num_program_cache_entries(self, mesh):
        return len(self.compiled)


def ref_relayout(data: np.ndarray, rec_bytes: int) -> np.ndarray:
    """Independent block-major [32,4,64,256] -> head-major [1,4,2048,256] gather."""
    v = np.asarray(data).reshape(
        32, 4, 2, 8, rec_bytes
    )  # [b, h, tile-row, tile-col, rec]
    return np.ascontiguousarray(v.transpose(1, 0, 2, 3, 4)).reshape(-1)


# --- helpers --------------------------------------------------------------------------

KV_BM = (KV_CHUNK_SHAPE, "bfloat8_b", "TILE")
KV_HM = ((1, 4, 2048, 256), "bfloat8_b", "TILE")
REC = ((1, 48, 128, 128), "float32", "TILE")
KV_NBYTES = spec_nbytes(KV_HM)


def manifest(num_tokens=4160, kv_layers=2, gdn_layers=2) -> Manifest:
    # 4160 tokens -> 65 blocks -> 3 chunks (the last one holds a single valid block)
    return build_manifest(
        num_tokens,
        model_sig="sig",
        prompt_hash="ph",
        num_attn_layers=kv_layers,
        num_gdn_layers=gdn_layers,
    )


def make_pair(
    tmp_path,
    world=None,
    *,
    kv_bufs=16,
    rec_sets=2,
    rec_parts=2,
    lease=30.0,
    clock=None,
    d_kw=None,
    **kw,
):
    world = world or FakeWorld()
    clock = clock or Clock()
    common = dict(
        control_dir=str(tmp_path / "ctrl"),
        mesh_device=object(),
        janitor_period=0,
        clock=clock,
        rec_sets=rec_sets,
        rec_parts=rec_parts,
        export_budget_bytes=kv_bufs * KV_NBYTES,
        # the test manifests carry <= 4 K/V parts: a max-length export of
        # kv_bufs // 4 chunks fits the pool exactly (start() checks this)
        kv_parts=4,
        max_model_len=(kv_bufs // 4) * 2048,
        lease_duration=lease,
        socket_timeout_s=5.0,
        claim_wait_s=0.0,  # tests drive the producer's pump by hand
    )
    common.update(kw)
    P = FabricSocketTransport(
        engine_id=P_ENGINE, role="producer", socket_layer=world.layer(0), **common
    )
    D = FabricSocketTransport(
        engine_id=D_ENGINE,
        role="consumer",
        socket_layer=world.layer(1),
        **{**common, **(d_kw or {})},
    )
    return world, clock, P, D


def start_both(*transports):
    errors: dict[int, BaseException] = {}

    def run(i, t):
        try:
            t.start()
        except BaseException as e:  # noqa: BLE001
            errors[i] = e

    ths = [threading.Thread(target=run, args=(i, t)) for i, t in enumerate(transports)]
    for th in ths:
        th.start()
    for th in ths:
        th.join(20)
    return errors


def started_pair(tmp_path, **kw):
    world, clock, P, D = make_pair(tmp_path, **kw)
    errors = start_both(P, D)
    assert not errors, errors
    return world, clock, P, D


def rows_tensor(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(4, 10240, generator=g).to(torch.bfloat16)


def kv_parts(m):
    return [p for p in m.parts if p.kind == "kv_blocks"]


def rec_parts(m):
    return [p for p in m.parts if p.kind == "gdn_rec"]


def taps_parts(m):
    return [p for p in m.parts if p.kind == "gdn_taps"]


def export_like_hook(h, m, rng, *, part_outer=False):
    """The qwen hook's order (chunk-outer, part-inner) or DefaultKVTransferable's
    (part-outer, chunk-inner); returns {(part, chunk): head-major bytes, rec: bytes,
    taps: rows}."""
    want = {}
    nch = kv_parts(m)[0].nchunks
    order = (
        [(p, c) for p in kv_parts(m) for c in range(nch)]
        if part_outer
        else [(p, c) for c in range(nch) for p in kv_parts(m)]
    )
    for p, c in order:
        blk = FakeTensor.random(KV_BM, rng)
        h.sinks[p.name].write_from_device(blk, chunk=c, blocking=True)
        blk.freed = True  # the hook deallocates right after the call
        want[(p.name, c)] = ref_relayout(blk.data, TILE_RECORD_BYTES["bfloat8_b"])
    for j, p in enumerate(rec_parts(m)):
        rec = FakeTensor.random(REC, rng)
        h.sinks[p.name].write_from_device(rec, chunk=0, blocking=False)
        want[p.name] = rec.data.copy()
    for j, p in enumerate(taps_parts(m)):
        r = rows_tensor(j)
        h.sinks[p.name].write_rows(r)
        want[p.name] = r
    return want


def import_like_hook(g, m, staging, chunk_range=None):
    """import_kv_blocks: chunk-outer, part-inner; the staging read right after
    ``read_into_device`` stands for ``paged_fill_cache`` on the same CQ."""
    nch = kv_parts(m)[0].nchunks
    chunks = range(nch) if chunk_range is None else range(nch)[chunk_range]
    got = {}
    for c in chunks:
        for p in kv_parts(m):
            src = g.sources[p.name].chunk(c)
            assert src.is_head_major and src.is_device_readable
            st = staging[c % 2]
            src.read_into_device(st)
            got[(p.name, c)] = st.data.copy()
    return got


def validate_like_hook(g, m):
    for p in rec_parts(m) + taps_parts(m):
        s = g.sources[p.name]
        assert s.spec.spec_key() == p.spec_key()
        assert int(s.nbytes_present) == p.nbytes, (p.name, s.nbytes_present, p.nbytes)
    for p in kv_parts(m):
        assert p.name in g.sources


def install_like_hook(g, m, rec_staging):
    got = {}
    for p in rec_parts(m):
        src = g.sources[p.name].chunk(0)
        assert src.is_device_readable
        src.read_into_device(rec_staging)
        got[p.name] = rec_staging.data.copy()
    for p in taps_parts(m):
        got[p.name] = g.sources[p.name].read_rows()
    return got


def publish(P, i, m, rng, **kw):
    h = P.open_put(xfer(i), m)
    assert h is not None
    want = export_like_hook(h, m, rng, **kw)
    P.finish_export(h, "READY")
    return h, want


def seg_dirs(D, i):
    """(tmp, pub, D-claim) of the producer segment i; claims carry the CONSUMER id."""
    return D.control.segment_dirs(P_ENGINE, HX[i])


def marker_path(D, i):
    """The producer's send marker of segment i (claim-gated sends)."""
    return D._marker_path(P_ENGINE, HX[i])


def claim(D, i):
    """The consumer's CLAIM: open_get renames the segment and returns None until the
    producer's marker says the sends are on the channel."""
    g = D.open_get(Desc(xfer(i)))
    assert g is None, g
    tmp, pub, mine = seg_dirs(D, i)
    assert os.path.isdir(mine) and not os.path.isdir(pub)
    return mine


def claim_and_send(P, D, i):
    """Claim, the producer's next pump (its step / idle tick) sends, READY handle."""
    claim(D, i)
    P.pump()
    g = D.open_get(Desc(xfer(i)))
    assert g is not None and g.ready(), g
    return g


# --- construction / seam --------------------------------------------------------------


def test_seam_not_started_and_descriptor(tmp_path):
    f = FabricSocketTransport(engine_id="p")
    assert f.descriptor() == {"kind": "fabric", "mode": "direct", "layout_version": 1}
    with pytest.raises(NotImplementedError):
        f.open_put(xfer(), manifest())
    with pytest.raises(NotImplementedError):
        f.open_get(Desc(xfer()))
    f.abandon(xfer())  # tolerant before start
    f.release_remote(xfer())
    f.shutdown()
    f.shutdown()  # idempotent
    with pytest.raises(ValueError):
        FabricSocketTransport(engine_id="p", role="both")
    with pytest.raises(ValueError):
        FabricSocketTransport(engine_id="p", role="mixed")


def test_factory_forwards_extra_config(tmp_path):
    cfg = SimpleNamespace(
        engine_id="decode-0",
        kv_role="kv_consumer",
        kv_connector_extra_config={
            "transport": "fabric",
            "shm_mode": "dumpfile",
            "kv_lease_duration": 7.5,
            "fabric_control_dir": str(tmp_path / "c"),
            "socket_connections": 1,
            "fabric_rec_sets": 3,
            "fabric_export_budget_bytes": 4 * KV_NBYTES,
            "fabric_max_packet_payload_bytes": 15232,
        },
    )
    t = make_transport("fabric", None, cfg, cfg.kv_role)
    assert isinstance(t, FabricSocketTransport)
    assert (t.engine_id, t.role, t.lease_duration) == ("decode-0", "consumer", 7.5)
    assert t.cfg.control_dir == str(tmp_path / "c")
    assert (t.cfg.socket_connections, t.cfg.rec_sets, t.cfg.packet_bytes) == (
        1,
        3,
        15232,
    )
    assert t.cfg.export_budget_bytes == 4 * KV_NBYTES
    # explicit kwargs win (tests inject the fake layer this way)
    t2 = make_transport(
        "fabric", None, cfg, "kv_producer", control_dir=str(tmp_path / "x"), rec_sets=1
    )
    assert (t2.role, t2.cfg.control_dir, t2.cfg.rec_sets) == (
        "producer",
        str(tmp_path / "x"),
        1,
    )
    # bare factory call as in test_shm_transport
    assert make_transport("fabric", engine_id="p").engine_id == "p"
    # the launch lane's Phase 2 spellings are accepted too (nothing silently ignored)
    cfg2 = SimpleNamespace(
        engine_id="p0",
        kv_role="kv_producer",
        kv_connector_extra_config={
            "transport": "fabric",
            "control_dir": str(tmp_path / "alias"),
            "socket_fifo_bytes": 256,
        },
    )
    t3 = make_transport("fabric", None, cfg2, None)
    assert (t3.cfg.control_dir, t3.cfg.fifo_bytes) == (str(tmp_path / "alias"), 256)


def test_env_knobs(tmp_path, monkeypatch):
    monkeypatch.setenv("TT_PD_FABRIC_DIR", str(tmp_path / "e"))
    monkeypatch.setenv("TT_PD_FABRIC_REC_SETS", "6")
    monkeypatch.setenv("TT_PD_FABRIC_SOCKET_TIMEOUT_S", "12")
    t = FabricSocketTransport(engine_id="p")
    assert (t.cfg.control_dir, t.cfg.rec_sets, t.cfg.socket_timeout_s) == (
        str(tmp_path / "e"),
        6,
        12.0,
    )


# --- start ----------------------------------------------------------------------------


def test_start_pair_socket_once_warmup_rendezvous(tmp_path):
    world, clock, P, D = started_pair(tmp_path)
    assert world.sockets_created == 2 and len(set(world.sock_cfgs.values())) == 1
    assert world.sock_cfgs[0] == (2, 128, 0, 1)
    assert P.layer.barriers == 2 and D.layer.barriers == 2
    assert (P.role, D.role, P.rank, D.rank) == ("producer", "consumer", 0, 1)
    assert P.peer_engine_id == D_ENGINE and D.peer_engine_id == P_ENGINE
    # lockstep warm-up: one kv + one rec crossed the channel, nothing left parked
    assert world.channel.delivered == 2 and world.channel.pending_sends == 0
    assert {k[0] for k in P.layer.compiled} == {"relayout", "copy", "send"}
    assert ("send", KV_HM) in P.layer.compiled and ("send", REC) in P.layer.compiled
    assert ("copy", KV_HM) in P.layer.compiled and ("copy", REC) in P.layer.compiled
    assert D.layer.compiled == {("recv", KV_HM), ("recv", REC), ("copy", REC)}
    assert P.stats["warm_programs"] == len(P.layer.compiled)
    # pools
    assert len(P._kv_pool) == 16 and P._kv_pool.free == 16
    assert len(P._rec_pool) == 4 and P._rec_pool.free == 4
    assert len(D._rec_sets) == 2 and all(len(s) == 2 for s in D._rec_sets)
    # rendezvous spec table on disk
    rv = tmp_path / "ctrl" / ".fabric_rendezvous"
    r0 = json.loads((rv / "rank0.json").read_text())
    assert r0["engine_id"] == P_ENGINE and r0["role"] == "producer"
    assert r0["kv_spec"] == [[1, 4, 2048, 256], "bfloat8_b", "TILE"]
    assert D._recv_epoch == r0["epoch"] and D.pending_receive_seq() == 0
    # both started transports are registered for the rank's close_mesh_device wrapper
    # (fabric_socket.shutdown_registered_transports runs before the mesh closes)
    assert P in fabric_socket.registered_transports()
    assert D in fabric_socket.registered_transports()
    # start twice is a no-op
    P.start()
    assert world.sockets_created == 2
    # shutdown: socket closed, every pool tensor deallocated, rendezvous file gone,
    # registry entry dropped
    P.shutdown()
    D.shutdown()
    assert P not in fabric_socket.registered_transports()
    assert D not in fabric_socket.registered_transports()
    assert all(s.closed for s in P.layer.sockets + D.layer.sockets)
    for L in (P.layer, D.layer):
        assert {id(t) for t in L.allocated} == {id(t) for t in L.deallocated}
    assert not (rv / "rank0.json").exists() and not (rv / "rank1.json").exists()
    P.shutdown()  # idempotent


def test_start_skip_warmup_knob(tmp_path, monkeypatch):
    """TT_PD_FABRIC_SKIP_WARMUP=1 (or fabric_skip_warmup) makes start() create the
    socket and pools but enqueue NO warm-up send/recv (p3 device validation: a pair
    must reach READY on a link that does not pass payloads)."""
    monkeypatch.setenv("TT_PD_FABRIC_SKIP_WARMUP", "1")
    world, clock, P, D = make_pair(tmp_path)
    assert P.cfg.skip_warmup and D.cfg.skip_warmup
    errors = start_both(P, D)
    assert not errors, errors
    assert world.sockets_created == 2
    assert world.channel.delivered == 0 and world.channel.pending_sends == 0
    compiled = P.layer.compiled | D.layer.compiled
    assert not any(k[0] in ("send", "recv") for k in compiled)
    assert len(P._kv_pool) == 16 and P._kv_pool.free == 16
    P.shutdown()
    D.shutdown()
    monkeypatch.delenv("TT_PD_FABRIC_SKIP_WARMUP")
    # the extra_config spelling; default stays off
    world, clock, P, D = make_pair(
        tmp_path / "b", extra_config={"fabric_skip_warmup": "1"}
    )
    assert P.cfg.skip_warmup
    plain = FabricSocketTransport(
        engine_id="x", role="producer", socket_layer=world.layer(0)
    )
    assert not plain.cfg.skip_warmup


def test_start_failures_raise_clearly(tmp_path, monkeypatch):
    world = FakeWorld()
    base = dict(
        control_dir=str(tmp_path / "ctrl"), janitor_period=0, socket_timeout_s=0.5
    )
    # role / rank mismatch (the prefill engine must be rank 0)
    t = FabricSocketTransport(
        engine_id="p",
        role="producer",
        socket_layer=world.layer(1),
        mesh_device=object(),
        **base,
    )
    with pytest.raises(RuntimeError, match="must be rank 0"):
        t.start()
    # configured rank != MPI rank
    t = FabricSocketTransport(
        engine_id="p",
        role="producer",
        rank=1,
        socket_layer=world.layer(0),
        mesh_device=object(),
        **base,
    )
    with pytest.raises(RuntimeError, match="configured rank"):
        t.start()
    # world size
    w3 = FakeWorld(size=3)
    t = FabricSocketTransport(
        engine_id="p",
        role="producer",
        socket_layer=w3.layer(0),
        mesh_device=object(),
        **base,
    )
    with pytest.raises(RuntimeError, match="world of 2"):
        t.start()
    # not under tt-run
    t = FabricSocketTransport(
        engine_id="p",
        role="producer",
        socket_layer=world.layer(0, distributed=False),
        mesh_device=object(),
        **base,
    )
    with pytest.raises(RuntimeError, match="MPI ranks"):
        t.start()
    # no mesh registered
    t = FabricSocketTransport(
        engine_id="p", role="producer", socket_layer=world.layer(0), **base
    )
    with pytest.raises(RuntimeError, match="pd_fabric_rank"):
        t.start()
    # checksums are unsupported over fabric
    monkeypatch.setenv("TT_PD_CHECKSUM", "1")
    t = FabricSocketTransport(
        engine_id="p",
        role="producer",
        socket_layer=world.layer(0),
        mesh_device=object(),
        **base,
    )
    with pytest.raises(RuntimeError, match="TT_PD_CHECKSUM"):
        t.start()
    monkeypatch.delenv("TT_PD_CHECKSUM")
    # rank 1 must be the consumer
    t = FabricSocketTransport(
        engine_id="d",
        role="producer",
        socket_layer=world.layer(1),
        mesh_device=object(),
        **base,
    )
    with pytest.raises(RuntimeError):
        t.start()
    # every failure left nothing started and no socket
    assert world.sockets_created == 0


def test_start_peer_absent_times_out(tmp_path):
    world = FakeWorld(barrier_timeout=0.3)
    P = FabricSocketTransport(
        engine_id=P_ENGINE,
        role="producer",
        socket_layer=world.layer(0),
        mesh_device=object(),
        control_dir=str(tmp_path / "ctrl"),
        janitor_period=0,
        socket_timeout_s=0.3,
        rec_parts=2,
        rec_sets=1,
        export_budget_bytes=2 * KV_NBYTES,
        kv_parts=2,
        max_model_len=2048,
    )
    t0 = time.perf_counter()
    with pytest.raises(RuntimeError, match="timed out"):
        P.start()
    assert time.perf_counter() - t0 < 5
    assert not P._started and P._sock is None and P._ctrl is None
    # pools were released on the failure path
    assert {id(t) for t in P.layer.allocated} == {id(t) for t in P.layer.deallocated}
    with pytest.raises(NotImplementedError):
        P.open_put(xfer(), manifest())
    P.shutdown()


def test_start_spec_table_mismatch(tmp_path):
    world = FakeWorld()
    clock = Clock()
    common = dict(
        control_dir=str(tmp_path / "ctrl"),
        mesh_device=object(),
        janitor_period=0,
        clock=clock,
        rec_sets=1,
        rec_parts=2,
        export_budget_bytes=2 * KV_NBYTES,
        kv_parts=2,
        max_model_len=2048,
        socket_timeout_s=5.0,
    )
    P = FabricSocketTransport(
        engine_id=P_ENGINE,
        role="producer",
        socket_layer=world.layer(0),
        kv_dtype="bfloat8_b",
        **common,
    )
    D = FabricSocketTransport(
        engine_id=D_ENGINE,
        role="consumer",
        socket_layer=world.layer(1),
        kv_dtype="bfloat16",
        **common,
    )
    errors = start_both(P, D)
    assert set(errors) == {0, 1}
    assert all("spec table mismatch" in str(e) for e in errors.values())
    assert world.sockets_created == 0


def test_start_both_producers_rejected(tmp_path):
    world = FakeWorld()
    common = dict(
        control_dir=str(tmp_path / "ctrl"),
        mesh_device=object(),
        janitor_period=0,
        rec_sets=1,
        rec_parts=2,
        export_budget_bytes=2 * KV_NBYTES,
        kv_parts=2,
        max_model_len=2048,
        socket_timeout_s=5.0,
    )
    P = FabricSocketTransport(
        engine_id=P_ENGINE, role=None, socket_layer=world.layer(0), **common
    )
    D = FabricSocketTransport(
        engine_id=D_ENGINE, role=None, socket_layer=world.layer(1), **common
    )
    # role None: derived from the rank
    assert not start_both(P, D)
    assert (P.role, D.role) == ("producer", "consumer")
    P.shutdown()
    D.shutdown()


# --- producer -------------------------------------------------------------------------


def test_open_put_pool_accounting_and_sinks(tmp_path):
    world, clock, P, D = started_pair(tmp_path, kv_bufs=16)
    m = manifest()  # 2 layers x k,v x 3 chunks = 12 kv items, 2 recs, 2 taps
    h = P.open_put(xfer(0), m)
    assert h is not None and h.lease_expiry_ts == clock.now + 30.0
    assert P._kv_pool.free == 4 and P._rec_pool.free == 2
    for p in kv_parts(m) + rec_parts(m):
        assert (
            isinstance(h.sinks[p.name], FabricSink)
            and h.sinks[p.name].supports_regions is False
        )
    for p in taps_parts(m):
        assert isinstance(h.sinks[p.name], DumpfileSink)
    tmp, pub, mine = seg_dirs(D, 0)
    assert (
        os.path.isdir(tmp) and read_status(os.path.join(tmp, "header")) == 1
    )  # WRITING
    # a second export of the same id while the first is open is refused
    assert P.open_put(xfer(0), m) is None
    # pool short: 4 free kv buffers, a 1-layer 3-chunk manifest needs 6
    assert P.open_put(xfer(1), manifest(kv_layers=1)) is None
    assert P.stats["budget_refusals"] == 2
    # 1 layer, 1 chunk = 2 kv items fits
    h1 = P.open_put(xfer(2), manifest(num_tokens=100, kv_layers=1))
    assert h1 is not None and P._kv_pool.free == 2 and P._rec_pool.free == 0
    # rec pool exhausted now
    assert P.open_put(xfer(3), manifest(num_tokens=100, kv_layers=1)) is None
    # wrong-shape writes are rejected, nothing recorded
    with pytest.raises(ValueError):
        h.sinks["kv.L0.k"].write_from_device(FakeTensor.zeros(REC), chunk=0)
    with pytest.raises(ValueError):
        h.sinks["gdn.L0.rec"].write_from_device(FakeTensor.zeros(KV_HM), chunk=0)
    with pytest.raises(IndexError):
        h.sinks["kv.L0.k"].write_from_device(FakeTensor.zeros(KV_BM), chunk=3)
    with pytest.raises(NotImplementedError):
        h.sinks["kv.L0.k"].write_region_from_device(
            None, 0, 0, chunk=0, dst_offset_bytes=0
        )
    # abandon (never published) returns the buffers and removes the .tmp
    P.abandon(xfer(2))
    assert P._kv_pool.free == 4 and P._rec_pool.free == 2
    assert not os.path.isdir(seg_dirs(D, 2)[0])
    P.abandon(xfer(2))  # idempotent
    # FAILED export: buffers back at once, nothing on the channel, header FAILED
    h.sinks["kv.L0.k"].write_from_device(FakeTensor.zeros(KV_BM), chunk=0)
    P.finish_export(h, "FAILED")
    assert P._kv_pool.free == 16 and P._rec_pool.free == 4
    assert world.channel.pending_sends == 0
    assert read_status(os.path.join(pub, "header")) == FAILED
    assert load_json(os.path.join(pub, SIDECAR_NAME))["nitems"] == 0
    g = D.open_get(Desc(xfer(0)))
    assert g is not None and g.status == "FAILED" and not os.path.isdir(pub)
    P.finish_export(h, "READY")  # idempotent after publish
    assert world.channel.pending_sends == 0
    P.shutdown()
    D.shutdown()


def test_finish_export_publishes_without_sends_then_the_claim_triggers_them(tmp_path):
    """Claim-gated sends (D5): READY puts NOTHING on the channel; the consumer's
    claim (open_get -> rename -> None) is the trigger; the producer's next pump
    enqueues every item once, in canonical order, assigns the seq in send order
    and writes the marker; the consumer's next open_get is READY."""
    world, clock, P, D = started_pair(tmp_path)
    m = manifest()
    rng = np.random.default_rng(1)
    h = P.open_put(xfer(0), m)
    want = export_like_hook(h, m, rng)
    # relayed head-major into the assigned buffers before publish
    for p in kv_parts(m):
        for c in range(3):
            buf = P._exports[xfer(0)].kv_bufs[(p.name, c)]
            assert buf.spec == KV_HM and np.array_equal(
                buf.tensor.data, want[(p.name, c)]
            )
    P.finish_export(h, "READY")
    assert world.channel.pending_sends == 0 and P.stats["sends"] == 0
    P.pump()
    P.pump()  # no claim: nothing is ever enqueued
    assert world.channel.pending_sends == 0 and P.unsent_exports() == 1
    assert P.stats["exports_ready"] == 1 and P.stats["exports_sent"] == 0
    tmp, pub, mine = seg_dirs(D, 0)
    assert os.path.isdir(pub) and not os.path.isdir(tmp)
    hdr = read_header(os.path.join(pub, "header"), full=True)
    assert hdr.status == READY and hdr.manifest.num_tokens == m.num_tokens
    assert {
        r.name: r.chunks_written for r in hdr.parts if r.spec.kind == "kv_blocks"
    } == {p.name: 3 for p in kv_parts(m)}
    side = load_json(os.path.join(pub, SIDECAR_NAME))
    names = [p.name for p in kv_parts(m)]
    items = [[n, c] for c in range(3) for n in names] + [
        [p.name, 0] for p in rec_parts(m)
    ]
    assert (
        side["seq"],
        side["claim_gated"],
        side["nitems"],
        side["n_kv"],
        side["epoch"],
    ) == (None, True, 14, 12, P._epoch)
    assert side["items"] == items
    assert side["kv_spec"] == [[1, 4, 2048, 256], "bfloat8_b", "TILE"]
    assert os.path.isfile(os.path.join(pub, "gdn.L0.taps.rows.pt"))
    # the consumer CLAIMS (rename) and gets None: nothing is on the channel yet
    assert D.open_get(Desc(xfer(0))) is None
    assert os.path.isdir(mine) and not os.path.isdir(pub) and D.stats["imports"] == 1
    assert world.channel.pending_sends == 0 and not os.path.exists(marker_path(D, 0))
    assert D.open_get(Desc(xfer(0))) is None  # re-poll: still no marker
    # the producer's pump sees the claim: 14 sends in canonical order, seq 0, marker
    P.pump()
    assert world.channel.pending_sends == 14 and P.stats["sends"] == 14
    assert P.layer.sends == 14 + 2  # + the two warm-up items
    with world.channel.cond:
        specs = [s for _, s in world.channel.sends]
    assert specs == [KV_HM] * 12 + [REC] * 2
    mk = load_json(marker_path(D, 0))
    assert (mk["seq"], mk["status"], mk["nitems"], mk["n_kv"], mk["epoch"]) == (
        0,
        "READY",
        14,
        12,
        P._epoch,
    )
    assert mk["items"] == items and mk["xfer_id"] == xfer(0)
    # exactly once: further pumps enqueue nothing more
    P.pump()
    P.pump()
    assert world.channel.pending_sends == 14 and P.stats["sends"] == 14
    assert P.stats["exports_sent"] == 1 and P.unsent_exports() == 0
    g = D.open_get(Desc(xfer(0)))  # channel head seq 0: READY
    assert g is not None and g.ready()
    # a second export claimed later gets seq 1 in SEND order and waits behind A
    publish(P, 1, manifest(num_tokens=100, kv_layers=1), rng)
    assert load_json(os.path.join(seg_dirs(D, 1)[1], SIDECAR_NAME))["seq"] is None
    claim(D, 1)
    P.pump()
    assert load_json(marker_path(D, 1))["seq"] == 1
    assert D.open_get(Desc(xfer(1))) is None  # A (seq 0) is mid-receive: B waits
    assert P.outstanding_exports() == 2 and not P.export_complete(xfer(0))
    P.shutdown()
    D.shutdown()


def test_default_hook_part_outer_write_order_is_canonicalised(tmp_path):
    """DefaultKVTransferable writes part-outer/chunk-inner; the channel must still
    carry chunk-outer/part-inner because both import loops read it that way."""
    world, clock, P, D = started_pair(tmp_path)
    m = manifest(kv_layers=2, gdn_layers=0)
    rng = np.random.default_rng(3)
    h, want = publish(P, 0, m, rng, part_outer=True)
    side = load_json(os.path.join(seg_dirs(D, 0)[1], SIDECAR_NAME))
    names = [p.name for p in kv_parts(m)]
    assert side["items"] == [[n, c] for c in range(3) for n in names]
    g = claim_and_send(P, D, 0)
    assert load_json(marker_path(D, 0))["items"] == side["items"]
    staging = [FakeTensor.zeros(KV_HM), FakeTensor.zeros(KV_HM)]
    got = import_like_hook(g, m, staging)
    assert all(np.array_equal(got[k], want[k]) for k in got) and len(got) == 12
    assert world.channel.pending_sends == 0 and D.pending_receive_seq() == 1
    D.finish_import(g, ok=True)
    P.shutdown()
    D.shutdown()


# --- consumer -------------------------------------------------------------------------


def test_end_to_end_identity_two_steps(tmp_path):
    world, clock, P, D = started_pair(tmp_path)
    m = manifest()
    rng = np.random.default_rng(7)
    keys_p, keys_d = set(P.layer.compiled), set(D.layer.compiled)
    h = P.open_put(xfer(0), m)
    assert D.open_get(Desc(xfer(0))) is None  # WRITING
    want = export_like_hook(h, m, rng)
    assert D.open_get(Desc(xfer(0))) is None  # still WRITING
    P.finish_export(h, "READY")
    tmp, pub, mine = seg_dirs(D, 0)
    assert D.open_get(Desc(xfer(0))) is None  # CLAIMED, the producer has not pumped
    assert os.path.isdir(mine) and not os.path.isdir(pub)
    assert len(D._rec_sets) == 1  # the rec set is reserved at the claim
    assert world.channel.pending_sends == 0
    P.pump()  # the producer's next step / idle tick: sends + marker
    g = D.open_get(Desc(xfer(0)))
    assert g is not None and g.ready() and g.manifest.nblk == 65
    for p in kv_parts(m) + rec_parts(m):
        assert isinstance(g.sources[p.name], FabricSource)
        assert g.sources[p.name].nbytes_present == p.nbytes
    for p in taps_parts(m):
        assert isinstance(g.sources[p.name], DumpfileSource)
    assert D.open_get(Desc(xfer(0))) is not None  # re-open of our own claim
    staging = [FakeTensor.zeros(KV_HM), FakeTensor.zeros(KV_HM)]
    # step 1: chunks 0..1 (max_import_chunks_per_step = 2)
    got = import_like_hook(g, m, staging, slice(0, 2))
    assert len(got) == 8 and world.channel.pending_sends == 14 - 8
    assert D.pending_receive_seq() == 0  # xfer still in flight
    # recs are not served before the K/V items are all in
    with pytest.raises(RuntimeError, match="before the K/V"):
        g.sources["gdn.L0.rec"].chunk(0).read_into_device(FakeTensor.zeros(REC))
    # step 2: chunk 2 -> the 2 rec recvs are posted right after the last kv item
    got.update(import_like_hook(g, m, staging, slice(2, 3)))
    assert world.channel.pending_sends == 0 and D.pending_receive_seq() == 1
    for k, v in want.items():
        if isinstance(k, tuple):
            assert np.array_equal(got[k], v), k
    validate_like_hook(g, m)
    D.layer.sync(None)  # the worker's synchronize_device -> KV_DONE
    # join step: install
    rec_staging = FakeTensor.zeros(REC)
    inst = install_like_hook(g, m, rec_staging)
    for p in rec_parts(m):
        assert np.array_equal(inst[p.name], want[p.name])
    for p in taps_parts(m):
        assert torch.equal(inst[p.name], want[p.name])
    with pytest.raises(NotImplementedError):
        g.sources["kv.L0.k"].chunk(0).read_device(None)
    with pytest.raises(NotImplementedError):
        g.sources["kv.L0.k"].crc32c()
    assert os.path.exists(marker_path(D, 0))  # marker lives until the claim is done
    D.finish_import(g, ok=True)
    assert not os.path.isdir(mine) and len(D._rec_sets) == 2
    assert not os.path.exists(marker_path(D, 0)) and D._markers == {}
    assert not D.wants_pump()
    # producer reclaims the buffers (engine thread, via open_put/pump)
    assert P._kv_pool.free == 4 and P.wants_pump()
    P.pump()
    assert (
        P._kv_pool.free == 16 and P._rec_pool.free == 4 and P.export_complete(xfer(0))
    )
    assert P.stats["reclaimed"] == 1 and not P.wants_pump()
    # no program compiled after start() on either side (TT_PD_STRICT_SHAPES)
    assert P.layer.compiled == keys_p and D.layer.compiled == keys_d
    # janitor: nothing left to sweep, no _puts entry
    P.control.janitor_once()
    assert P.control.put_state(xfer(0)) is None
    P.shutdown()
    D.shutdown()


def test_orphan_export_never_parks_and_is_swept_at_lease_expiry(tmp_path):
    """V7 (D5): an export whose D leg never comes has NOTHING on the channel; a
    later claimed export flows past it (seq 0 goes to the first SENT xfer); the
    producer janitor sweeps the orphan at lease expiry and its buffers come back
    -- no drain, no parked CQ, no lease-long stall of the producer."""
    world, clock, P, D = started_pair(tmp_path, kv_bufs=32, lease=10.0)
    rng = np.random.default_rng(11)
    mA, mB = manifest(kv_layers=1), manifest(num_tokens=100, kv_layers=1)
    publish(P, 0, mA, rng)  # the D leg never names A
    _, wantB = publish(P, 1, mB, rng)
    P.pump()
    D.pump()
    assert world.channel.pending_sends == 0 and P.stats["sends"] == 0
    g = claim_and_send(P, D, 1)
    assert load_json(marker_path(D, 1))["seq"] == 0
    assert world.channel.pending_sends == 4 and P.stats["sends"] == 4
    assert os.path.isdir(seg_dirs(D, 0)[1])  # A: published, unclaimed, unsent
    staging = [FakeTensor.zeros(KV_HM), FakeTensor.zeros(KV_HM)]
    got = import_like_hook(g, mB, staging)
    assert all(np.array_equal(got[k], wantB[k]) for k in got)
    D.layer.sync(None)
    inst = install_like_hook(g, mB, FakeTensor.zeros(REC))
    assert all(np.array_equal(inst[p.name], wantB[p.name]) for p in rec_parts(mB))
    D.finish_import(g, ok=True)
    assert D.pending_receive_seq() == 1 and D.stats["drains"] == 0
    # A's lease expires: nothing for the consumer to drain, the janitor sweeps it
    clock.now += 10.5
    D.pump()
    assert D.stats["drained_xfers"] == 0 and world.channel.pending_sends == 0
    assert P.outstanding_exports() == 2  # A's buffers held while its segment exists
    P.control.janitor_once()
    assert not os.path.isdir(seg_dirs(D, 0)[1]) and P.control.stats["expired"] == 1
    P.pump()
    assert P._kv_pool.free == 32 and P.outstanding_exports() == 0
    assert P.stats["sends"] == 4 and P.stats["exports_sent"] == 1
    # a late request for the swept A -> definite MISSING
    gA = D.open_get(Desc(xfer(0)))
    assert gA is not None and gA.status == "MISSING"
    P.shutdown()
    D.shutdown()


def test_mixed_orphans_and_claims_keep_send_order(tmp_path):
    """Four exports published A B C Dd; the consumer claims C then A: the producer
    sends the claimed ones in PUBLISH order (A seq 0, C seq 1), the consumer
    receives in seq order (C waits for A), B and Dd never touch the channel; B
    claimed afterwards is seq 2."""
    world, clock, P, D = started_pair(tmp_path, kv_bufs=32, rec_sets=4, d_kw={})
    rng = np.random.default_rng(17)
    m = manifest(num_tokens=100, kv_layers=1)  # 2 kv + 2 rec items each
    wants = [publish(P, i, m, rng)[1] for i in range(4)]
    claim(D, 2)
    claim(D, 0)
    P.pump()
    assert world.channel.pending_sends == 8 and P.unsent_exports() == 2
    assert load_json(marker_path(D, 0))["seq"] == 0
    assert load_json(marker_path(D, 2))["seq"] == 1
    assert D.open_get(Desc(xfer(2))) is None  # A (seq 0) is the channel head
    gA = D.open_get(Desc(xfer(0)))
    assert gA is not None and gA.ready()
    staging = [FakeTensor.zeros(KV_HM), FakeTensor.zeros(KV_HM)]
    got = import_like_hook(gA, m, staging)
    assert all(np.array_equal(got[k], wants[0][k]) for k in got)
    gC = D.open_get(Desc(xfer(2)))
    assert gC is not None and gC.ready() and D.pending_receive_seq() == 1
    got = import_like_hook(gC, m, staging)
    assert all(np.array_equal(got[k], wants[2][k]) for k in got)
    assert world.channel.pending_sends == 0 and D.pending_receive_seq() == 2
    D.finish_import(gA, ok=True)
    D.finish_import(gC, ok=True)
    gB = claim_and_send(P, D, 1)
    assert load_json(marker_path(D, 1))["seq"] == 2
    got = import_like_hook(gB, m, staging)
    assert all(np.array_equal(got[k], wants[1][k]) for k in got)
    D.finish_import(gB, ok=True)
    assert D.pending_receive_seq() == 3 and D.stats["drains"] == 0
    assert P.stats["sends"] == 12 and P.unsent_exports() == 1  # Dd: orphan
    P.shutdown()
    D.shutdown()


def test_release_remote_unclaimed_marks_released_nothing_to_drain(tmp_path):
    world, clock, P, D = started_pair(tmp_path)
    rng = np.random.default_rng(5)
    publish(P, 0, manifest(kv_layers=1), rng)
    D.release_remote(xfer(0))  # scheduler demotion before the worker ever claimed it
    pub = seg_dirs(D, 0)[1]
    assert os.path.isdir(pub) and read_status(os.path.join(pub, "header")) == RELEASED
    P.pump()  # no claim: nothing sent; the segment is the janitor's
    assert world.channel.pending_sends == 0 and P.stats["sends"] == 0
    P.control.janitor_once()
    assert not any(os.path.isdir(d) for d in seg_dirs(D, 0))
    P.pump()
    assert P._kv_pool.free == 16 and P.outstanding_exports() == 0
    D.release_remote(xfer(0))  # idempotent
    D.pump()
    assert D.pending_receive_seq() == 0 and D.stats["drains"] == 0
    P.shutdown()
    D.shutdown()


def test_released_unclaimed_export_does_not_block_a_later_claim(tmp_path):
    world, clock, P, D = started_pair(tmp_path, kv_bufs=32)
    rng = np.random.default_rng(6)
    mA = manifest(kv_layers=1)
    _, wantA = publish(P, 0, mA, rng)
    publish(P, 1, manifest(num_tokens=100, kv_layers=1), rng)
    D.release_remote(xfer(1))  # B released while A, published earlier, is unclaimed
    pubB = seg_dirs(D, 1)[1]
    assert os.path.isdir(pubB) and read_status(os.path.join(pubB, "header")) == RELEASED
    g = claim_and_send(P, D, 0)  # A flows: nothing of B is on the channel
    assert load_json(marker_path(D, 0))["seq"] == 0
    got = import_like_hook(g, mA, [FakeTensor.zeros(KV_HM), FakeTensor.zeros(KV_HM)])
    assert all(np.array_equal(got[k], wantA[k]) for k in got)
    # a worker asking for the released B gets a definite answer, not None forever
    gB = D.open_get(Desc(xfer(1)))
    assert gB is not None and gB.status == "FAILED" and "RELEASED" in gB.reason
    assert not os.path.isdir(pubB) and world.channel.pending_sends == 0
    D.finish_import(g, ok=True)
    P.pump()
    assert P.outstanding_exports() == 0 and P.stats["sends"] == 8
    P.shutdown()
    D.shutdown()


def test_abandon_after_publish_unsent_and_claimed(tmp_path):
    world, clock, P, D = started_pair(tmp_path, kv_bufs=32, lease=5.0)
    rng = np.random.default_rng(8)
    publish(P, 0, manifest(kv_layers=1), rng)
    publish(P, 1, manifest(num_tokens=100, kv_layers=1), rng)
    D.pump()
    P.pump()  # nothing claimed: nothing on the channel
    assert world.channel.pending_sends == 0
    # the producer gives up a PUBLISHED, unsent xfer: segment removed, buffers back
    P.abandon(xfer(0))
    assert not any(os.path.isdir(d) for d in seg_dirs(D, 0))
    assert P.outstanding_exports() == 1 and P._kv_pool.free == 30
    # B claimed, then abandoned before the producer pumped: the claim's header
    # flips to RELEASED, nothing is ever sent, the consumer drops the claim
    claim(D, 1)
    P.abandon(xfer(1))
    mineB = seg_dirs(D, 1)[2]
    assert (
        os.path.isdir(mineB) and read_status(os.path.join(mineB, "header")) == RELEASED
    )
    P.pump()
    assert world.channel.pending_sends == 0 and P.stats["sends"] == 0
    g = D.open_get(Desc(xfer(1)))
    assert (
        g is not None
        and g.status == "MISSING"
        and "released by the producer" in (g.reason)
    )
    assert not any(os.path.isdir(d) for d in seg_dirs(D, 1)) and len(D._rec_sets) == 2
    P.pump()
    assert P.outstanding_exports() == 0 and P._kv_pool.free == 32
    D.pump()
    assert D.pending_receive_seq() == 0 and D.stats["drains"] == 0
    P.shutdown()
    D.shutdown()


def test_out_of_order_read_raises_then_failed_import_drains(tmp_path):
    world, clock, P, D = started_pair(tmp_path)
    rng = np.random.default_rng(9)
    m = manifest(kv_layers=1)
    publish(P, 0, m, rng)
    g = claim_and_send(P, D, 0)
    st = FakeTensor.zeros(KV_HM)
    g.sources["kv.L0.k"].chunk(0).read_into_device(st)
    with pytest.raises(RuntimeError, match="order violation"):
        g.sources["kv.L0.k"].chunk(1).read_into_device(st)  # channel head is kv.L0.v c0
    assert world.channel.pending_sends == 7 and D.stats["order_violations"] == 1
    with pytest.raises(RuntimeError, match="already failed"):
        g.sources["kv.L0.v"].chunk(0).read_into_device(st)
    # worker _fail -> finish_import(ok=False): the rest is drained into scratch
    D.finish_import(g, ok=False)
    assert world.channel.pending_sends == 0 and D.stats["drains"] == 7
    assert len(D._rec_sets) == 2 and D.pending_receive_seq() == 1
    assert not any(os.path.isdir(d) for d in seg_dirs(D, 0))
    assert not os.path.exists(marker_path(D, 0))
    P.pump()
    assert P._kv_pool.free == 16
    # a fresh xfer after the failure flows normally
    _, want = publish(P, 1, m, rng)
    g2 = claim_and_send(P, D, 1)
    got = import_like_hook(g2, m, [FakeTensor.zeros(KV_HM), FakeTensor.zeros(KV_HM)])
    assert all(np.array_equal(got[k], want[k]) for k in got)
    D.finish_import(g2, ok=True)
    P.shutdown()
    D.shutdown()


def test_bad_staging_spec_rejected_without_posting(tmp_path):
    world, clock, P, D = started_pair(tmp_path)
    m = manifest(kv_layers=1)
    publish(P, 0, m, np.random.default_rng(2))
    g = claim_and_send(P, D, 0)
    with pytest.raises(ValueError, match="staging spec"):
        g.sources["kv.L0.k"].chunk(0).read_into_device(FakeTensor.zeros(KV_BM))
    assert world.channel.pending_sends == 8 and world.channel.pending_recvs == 0
    D.finish_import(g, ok=False)
    assert world.channel.pending_sends == 0
    P.shutdown()
    D.shutdown()


def test_rec_set_exhaustion_returns_none_before_claiming(tmp_path):
    # the consumer has ONE rec set; the producer keeps two exports in flight
    world, clock, P, D = started_pair(
        tmp_path, kv_bufs=32, rec_sets=2, d_kw={"rec_sets": 1}
    )
    rng = np.random.default_rng(12)
    mA, mB = manifest(kv_layers=1), manifest(num_tokens=100, kv_layers=1)
    _, wantA = publish(P, 0, mA, rng)
    _, wantB = publish(P, 1, mB, rng)
    gA = claim_and_send(P, D, 0)
    import_like_hook(gA, mA, [FakeTensor.zeros(KV_HM), FakeTensor.zeros(KV_HM)])
    assert D.pending_receive_seq() == 1 and len(D._rec_sets) == 0
    # A holds the only rec set until its join step: B waits UNCLAIMED, so the
    # producer sends nothing for it
    assert D.open_get(Desc(xfer(1))) is None
    assert os.path.isdir(seg_dirs(D, 1)[1])
    P.pump()
    assert world.channel.pending_sends == 0 and P.unsent_exports() == 1
    D.layer.sync(None)
    install_like_hook(gA, mA, FakeTensor.zeros(REC))
    D.finish_import(gA, ok=True)  # join step: set returned
    gB = claim_and_send(P, D, 1)
    got = import_like_hook(gB, mB, [FakeTensor.zeros(KV_HM), FakeTensor.zeros(KV_HM)])
    assert all(np.array_equal(got[k], wantB[k]) for k in got)
    inst = install_like_hook(gB, mB, FakeTensor.zeros(REC))
    assert all(np.array_equal(inst[p.name], wantB[p.name]) for p in rec_parts(mB))
    D.finish_import(gB, ok=True)
    P.shutdown()
    D.shutdown()


def test_aborted_claimed_xfer_release_drains(tmp_path):
    world, clock, P, D = started_pair(tmp_path)
    m = manifest(kv_layers=1)
    publish(P, 0, m, np.random.default_rng(4))
    g = claim_and_send(P, D, 0)
    import_like_hook(
        g, m, [FakeTensor.zeros(KV_HM), FakeTensor.zeros(KV_HM)], slice(0, 1)
    )
    assert world.channel.pending_sends == 6
    D.release_remote(xfer(0))  # abort while IMPORTING_KV: at the head -> drained now
    assert world.channel.pending_sends == 0 and len(D._rec_sets) == 2
    assert not any(os.path.isdir(d) for d in seg_dirs(D, 0))
    assert not os.path.exists(marker_path(D, 0)) and D.pending_receive_seq() == 1
    P.pump()
    assert P._kv_pool.free == 16 and P.outstanding_exports() == 0
    P.shutdown()
    D.shutdown()


def test_release_before_the_send_fences_the_claim(tmp_path):
    """A claim dropped before the producer pumped (abort / lease expiry on D) is
    FENCED (rename to .closing) then dropped: the producer never sends for it,
    nothing is parked, its buffers come back."""
    world, clock, P, D = started_pair(tmp_path)
    publish(P, 0, manifest(kv_layers=1), np.random.default_rng(4))
    claim(D, 0)
    mine = seg_dirs(D, 0)[2]
    assert os.path.isdir(mine) and len(D._rec_sets) == 1
    D.release_remote(xfer(0))
    assert not any(os.path.isdir(d) for d in seg_dirs(D, 0))
    assert not os.path.isdir(mine + CLOSING_SUFFIX) and len(D._rec_sets) == 2
    assert xfer(0) not in D._xfers
    P.pump()  # the producer never sends for a dropped claim
    assert world.channel.pending_sends == 0 and P.stats["sends"] == 0
    assert P.outstanding_exports() == 0 and P._kv_pool.free == 16
    D.pump()
    assert D.pending_receive_seq() == 0 and not D.wants_pump()
    P.shutdown()
    D.shutdown()


def test_released_claim_behind_a_live_import_is_drained_by_pump_at_its_turn(tmp_path):
    """The request of a SENT claim dies (abort) while an older xfer is still being
    received: the drain is deferred to its channel turn and done by the next pump
    -- one consumer step after the head clears, no lease involved."""
    world, clock, P, D = started_pair(tmp_path, kv_bufs=32)
    rng = np.random.default_rng(14)
    mA, mB = manifest(kv_layers=1), manifest(num_tokens=100, kv_layers=1)
    _, wantA = publish(P, 0, mA, rng)
    publish(P, 1, mB, rng)
    gA = claim_and_send(P, D, 0)  # seq 0
    claim(D, 1)
    P.pump()  # B seq 1
    assert world.channel.pending_sends == 12 and len(D._rec_sets) == 0
    assert D.open_get(Desc(xfer(1))) is None  # A is the head
    D.release_remote(xfer(1))  # B's request aborted: deferred behind A
    assert world.channel.pending_sends == 12
    assert xfer(1) in D._xfers and D._xfers[xfer(1)].released
    assert len(D._rec_sets) == 1  # B's set back at once (no recv targets it)
    D.pump()  # not at the head yet: nothing to do
    assert world.channel.pending_sends == 12 and D.wants_pump()
    staging = [FakeTensor.zeros(KV_HM), FakeTensor.zeros(KV_HM)]
    got = import_like_hook(gA, mA, staging)
    assert all(np.array_equal(got[k], wantA[k]) for k in got)
    assert world.channel.pending_sends == 4 and D.pending_receive_seq() == 1
    D.pump()  # B is the head now: drained
    assert world.channel.pending_sends == 0 and D.pending_receive_seq() == 2
    assert xfer(1) not in D._xfers and D.stats["drained_xfers"] == 1
    assert not any(os.path.isdir(d) for d in seg_dirs(D, 1))
    D.finish_import(gA, ok=True)
    P.pump()
    assert P.outstanding_exports() == 0 and P._kv_pool.free == 32
    P.shutdown()
    D.shutdown()


class CallbackSendLayer(FakeSocketLayer):
    """``send`` runs ``on_send`` once, on the first data send (warm-up excluded)."""

    def __init__(self, world, rank, **kw):
        super().__init__(world, rank, **kw)
        self.on_send = None
        self.armed = False

    def send(self, t, sock):
        if self.armed and self.on_send is not None:
            cb, self.on_send = self.on_send, None
            cb()
        super().send(t, sock)


def _pair_with_layer0(tmp_path, layer0, world):
    common = dict(
        control_dir=str(tmp_path / "ctrl"),
        mesh_device=object(),
        janitor_period=0,
        rec_sets=2,
        rec_parts=2,
        export_budget_bytes=16 * KV_NBYTES,
        kv_parts=4,
        max_model_len=8192,
        socket_timeout_s=5.0,
        claim_wait_s=0.0,
    )
    P = FabricSocketTransport(
        engine_id=P_ENGINE, role="producer", socket_layer=layer0, **common
    )
    D = FabricSocketTransport(
        engine_id=D_ENGINE, role="consumer", socket_layer=world.layer(1), **common
    )
    assert not start_both(P, D)
    return P, D


def test_release_racing_the_send_leaves_an_orphan_marker_drained_by_pump(tmp_path):
    """The residual window: the producer listed the claim and is enqueueing when
    the consumer fences and drops it (no marker yet).  The marker then lands as
    an ORPHAN; the consumer's next pump drains the parked items at their turn and
    a later export flows with the next seq."""
    world = FakeWorld()
    layer0 = CallbackSendLayer(world, 0)
    P, D = _pair_with_layer0(tmp_path, layer0, world)
    rng = np.random.default_rng(3)
    m = manifest(kv_layers=1)  # 6 kv + 2 rec items
    publish(P, 0, m, rng)
    claim(D, 0)
    layer0.armed = True
    layer0.on_send = lambda: D.release_remote(xfer(0))  # fence + drop mid-enqueue
    P.pump()
    assert world.channel.pending_sends == 8 and os.path.exists(marker_path(D, 0))
    assert not any(os.path.isdir(d) for d in seg_dirs(D, 0))
    assert xfer(0) not in D._xfers and len(D._rec_sets) == 2
    assert D.wants_pump()  # the orphan marker
    D.pump()  # the consumer's next step / idle tick
    assert world.channel.pending_sends == 0 and D.pending_receive_seq() == 1
    assert D.stats["orphan_markers"] == 1 and D.stats["drains"] == 8
    assert not os.path.exists(marker_path(D, 0)) and not D.wants_pump()
    P.pump()
    assert P.outstanding_exports() == 0 and P._kv_pool.free == 16
    mB = manifest(num_tokens=100, kv_layers=1)
    _, wantB = publish(P, 1, mB, rng)
    gB = claim_and_send(P, D, 1)
    assert load_json(marker_path(D, 1))["seq"] == 1
    got = import_like_hook(gB, mB, [FakeTensor.zeros(KV_HM), FakeTensor.zeros(KV_HM)])
    assert all(np.array_equal(got[k], wantB[k]) for k in got)
    D.finish_import(gB, ok=True)
    P.shutdown()
    D.shutdown()


def test_marker_landing_between_the_release_check_and_the_drop_survives(tmp_path):
    """The other half of the residual window (review MAJOR 1): the producer listed
    the OPEN claim before the consumer's fence and its marker lands AFTER the
    consumer's absent-marker read but BEFORE the drop.  The drop must not unlink a
    marker it never read: it survives as an ORPHAN, ``wants_pump`` is True, the
    pump drains every parked item and the next export flows at seq 1.  (The
    sibling test above fires the release before the marker write; this one hooks
    the consumer's marker read so the producer writes right after the None.)"""
    world, clock, P, D = started_pair(tmp_path)
    rng = np.random.default_rng(7)
    m = manifest(kv_layers=1)  # 6 kv + 2 rec items
    publish(P, 0, m, rng)
    claim(D, 0)
    exp = P._exports[xfer(0)]
    orig_marker = D._marker
    fired: list[int] = []

    def marker_then_the_producer_lands(engine, hx):
        got = orig_marker(engine, hx)
        if got is None and not fired:
            fired.append(1)
            with P._lock:  # P snapshotted the OPEN claim before our fence: finishes now
                P._send_export(exp)
            assert os.path.exists(marker_path(D, 0))
        return got

    D._marker = marker_then_the_producer_lands
    try:
        D.release_remote(xfer(0))  # lease expiry / abort / demotion after the claim
    finally:
        D._marker = orig_marker
    assert fired and exp.sent and exp.seq == 0
    assert world.channel.pending_sends == 8
    assert os.path.exists(marker_path(D, 0)), "the drop unlinked a marker it never read"
    assert D.stats["late_markers"] == 1
    assert xfer(0) not in D._xfers and len(D._rec_sets) == 2
    assert not any(os.path.isdir(d) for d in seg_dirs(D, 0))
    assert D.wants_pump()  # the orphan marker: the idle ticker wakes the engine
    D.pump()  # the consumer's next step / idle tick
    assert world.channel.pending_sends == 0 and D.pending_receive_seq() == 1
    assert D.stats["orphan_markers"] == 1 and D.stats["drains"] == 8
    assert not os.path.exists(marker_path(D, 0)) and not D.wants_pump()
    P.pump()
    assert P.outstanding_exports() == 0 and P._kv_pool.free == 16
    mB = manifest(num_tokens=100, kv_layers=1)
    _, wantB = publish(P, 1, mB, rng)
    gB = claim_and_send(P, D, 1)
    assert load_json(marker_path(D, 1))["seq"] == 1
    got = import_like_hook(gB, mB, [FakeTensor.zeros(KV_HM), FakeTensor.zeros(KV_HM)])
    assert all(np.array_equal(got[k], wantB[k]) for k in got)
    D.finish_import(gB, ok=True)
    assert not os.path.exists(marker_path(D, 1))  # an ADOPTED marker is unlinked
    P.shutdown()
    D.shutdown()


def test_claim_deadline_is_clocked_from_the_claim_then_from_the_marker(tmp_path):
    """The claim lease (review MAJOR 2): after D's claim, P is mid-prefill of the
    next queued request for longer than one READY lease (a whole prompt per P
    step, 26 s @32k).  The transport's claim deadline is claim + claim_lease_s
    (default 3 x lease: the hard bound a dead producer is caught by) until the
    marker, then marker + lease for the channel head; the transfer completes
    byte-exact although the READY lease is long gone."""
    world, clock, P, D = started_pair(tmp_path)  # lease 30 -> claim lease 90
    assert D.cfg.claim_lease_s == 90.0 and P.cfg.claim_lease_s == 90.0
    rng = np.random.default_rng(31)
    m = manifest(kv_layers=1)
    _, want = publish(P, 0, m, rng)
    assert D.claim_deadline(xfer(0)) is None  # not claimed: the READY lease applies
    t_claim = clock.now
    claim(D, 0)
    assert D.claim_deadline(xfer(0)) == t_claim + 90.0
    clock.now += 45.0  # P busy: one full prefill of the next prompt, > one lease
    assert D.open_get(Desc(xfer(0))) is None  # still waiting; no expiry here
    assert D.claim_deadline(xfer(0)) == t_claim + 90.0  # the hard bound, unchanged
    P.pump()  # P's step ended: the marker lands 45 s after READY
    assert D.claim_deadline(xfer(0)) == t_claim + 90.0  # not read by D yet
    g = D.open_get(Desc(xfer(0)))
    assert g is not None and g.ready()
    assert D.claim_deadline(xfer(0)) is None  # posted at the head: nothing to time out
    # restart from the marker: B is claimed and sent while A holds the head
    mB = manifest(num_tokens=100, kv_layers=1)
    _, wantB = publish(P, 1, mB, rng)
    claim(D, 1)
    clock.now += 40.0
    P.pump()  # B seq 1, behind A on the channel
    clock.now += 1.0
    t_seen_b = clock.now  # D reads the marker at its next poll: the lease restarts here
    assert D.open_get(Desc(xfer(1))) is None  # A is the head
    assert D.claim_deadline(xfer(1)) == t_seen_b + 30.0  # restarted from the marker
    clock.now += 5.0
    assert D.claim_deadline(xfer(1)) == t_seen_b + 30.0
    staging = [FakeTensor.zeros(KV_HM), FakeTensor.zeros(KV_HM)]
    got = import_like_hook(g, m, staging)
    assert all(np.array_equal(got[k], want[k]) for k in got)
    D.finish_import(g, ok=True)
    gB = D.open_get(Desc(xfer(1)))
    assert gB is not None and gB.ready() and D.claim_deadline(xfer(1)) is None
    got = import_like_hook(gB, mB, staging)
    assert all(np.array_equal(got[k], wantB[k]) for k in got)
    D.finish_import(gB, ok=True)
    assert D.claim_deadline(xfer(1)) is None and not D.wants_pump()
    P.pump()
    assert P.outstanding_exports() == 0
    P.shutdown()
    D.shutdown()


def test_claim_lease_knob(monkeypatch):
    """claim_lease_s: default 3 x lease; fabric_claim_lease_s and
    TT_PD_FABRIC_CLAIM_LEASE_S override; a claim lease below the READY lease is
    refused."""
    t = FabricSocketTransport(engine_id="p", lease_duration=20.0)
    assert t.cfg.claim_lease_s == 60.0
    t = FabricSocketTransport(
        engine_id="p", lease_duration=20.0, extra_config={"fabric_claim_lease_s": 45}
    )
    assert t.cfg.claim_lease_s == 45.0
    monkeypatch.setenv("TT_PD_FABRIC_CLAIM_LEASE_S", "120")
    assert FabricSocketTransport(engine_id="p").cfg.claim_lease_s == 120.0
    monkeypatch.setenv("TT_PD_FABRIC_CLAIM_LEASE_S", "0")  # 0 = derived
    assert FabricSocketTransport(engine_id="p").cfg.claim_lease_s == 90.0
    with pytest.raises(ValueError, match="fabric_claim_lease_s"):
        FabricSocketTransport(engine_id="p", lease_duration=30.0, claim_lease_s=10.0)


def test_producer_skips_a_claim_whose_consumer_pid_is_dead(tmp_path, monkeypatch):
    """A consumer that dies right after claiming must not trigger the sends: nobody
    would post the recvs and the channel would park until the janitor sweeps the
    dead claim.  The producer's pump reads consumer.pid and skips a dead one (the
    janitor then frees the buffers); a live claim is sent for as before."""
    from vllm_tt_plugin.kv_transfer.transport import shm as shm_mod

    world, clock, P, D = started_pair(tmp_path)
    rng = np.random.default_rng(21)
    publish(P, 0, manifest(kv_layers=1), rng)
    mine = claim(D, 0)
    pid_path = os.path.join(mine, shm_mod.CONSUMER_PID_FILE)
    with open(pid_path) as f:
        assert int(f.read()) == os.getpid()
    dead_pid = 4194303
    monkeypatch.setattr(shm_mod, "pid_alive", lambda pid: pid != dead_pid)
    with open(pid_path, "w") as f:
        f.write(str(dead_pid))  # the consumer died right after the claim
    P.pump()
    P.pump()
    assert world.channel.pending_sends == 0 and P.stats["sends"] == 0
    assert P.stats["dead_claims_skipped"] == 2 and not P._exports[xfer(0)].sent
    assert not os.path.exists(marker_path(D, 0))
    P.control.janitor_once()  # the C2 rule sweeps the dead claim
    assert not os.path.isdir(mine) and P.control.stats["stale_claims"] == 1
    P.pump()
    assert P.outstanding_exports() == 0 and P._kv_pool.free == 16
    # control: a live consumer's claim is sent for
    D.release_remote(xfer(0))  # D's view of the dead claim: gone, nothing to drain
    m = manifest(num_tokens=100, kv_layers=1)
    _, want = publish(P, 1, m, rng)
    g = claim_and_send(P, D, 1)
    assert load_json(marker_path(D, 1))["seq"] == 0 and P.stats["exports_sent"] == 1
    got = import_like_hook(g, m, [FakeTensor.zeros(KV_HM), FakeTensor.zeros(KV_HM)])
    assert all(np.array_equal(got[k], want[k]) for k in got)
    D.finish_import(g, ok=True)
    P.shutdown()
    D.shutdown()


def test_claim_wait_spins_for_an_idle_producers_marker(tmp_path):
    """The common 1-user path: the producer's idle tick answers the claim within a
    few ms; the FIRST open_get after the claim spins up to claim_wait_s so the
    recvs go out in the same consumer step."""
    world, clock, P, D = started_pair(tmp_path, d_kw={"claim_wait_s": 1.0})
    m = manifest(kv_layers=1)
    _, want = publish(P, 0, m, np.random.default_rng(15))

    def producer_tick():
        deadline = time.perf_counter() + 2.0
        while P.unsent_exports() and time.perf_counter() < deadline:
            time.sleep(0.005)
            P.pump()

    th = threading.Thread(target=producer_tick)
    th.start()
    t0 = time.perf_counter()
    g = D.open_get(Desc(xfer(0)))  # claim + spin: READY in this call
    th.join(5)
    assert g is not None and g.ready(), g
    assert time.perf_counter() - t0 < 1.0  # returned as soon as the marker landed
    got = import_like_hook(g, m, [FakeTensor.zeros(KV_HM), FakeTensor.zeros(KV_HM)])
    assert all(np.array_equal(got[k], want[k]) for k in got)
    D.finish_import(g, ok=True)
    P.shutdown()
    D.shutdown()


def test_wants_pump_is_a_host_only_hint_for_the_idle_ticker(tmp_path):
    world, clock, P, D = started_pair(tmp_path)
    assert not P.wants_pump() and not D.wants_pump()
    publish(P, 0, manifest(kv_layers=1), np.random.default_rng(0))
    assert P.wants_pump() and not D.wants_pump()  # a claim may appear any time
    claim(D, 0)
    assert D.wants_pump()  # a claim of ours is open
    P.pump()
    g = D.open_get(Desc(xfer(0)))
    import_like_hook(g, manifest(kv_layers=1), [FakeTensor.zeros(KV_HM)] * 2)
    D.finish_import(g, ok=True)
    assert not D.wants_pump() and P.wants_pump()  # buffers not reclaimed yet
    P.pump()
    assert not P.wants_pump()
    P.shutdown()
    D.shutdown()
    assert not P.wants_pump() and not D.wants_pump()


def test_open_get_misc_statuses(tmp_path):
    world, clock, P, D = started_pair(tmp_path)
    # unknown segment
    g = D.open_get(Desc(xfer(5)))
    assert g is not None and g.status == "MISSING"
    # transport kind mismatch (config drift between nodes)
    g = D.open_get(
        Desc(
            xfer(5), transport={"kind": "shm", "mode": "dumpfile", "layout_version": 1}
        )
    )
    assert g is not None and g.status == "FAILED" and "mismatch" in g.reason
    # the connector's config-derived descriptor says mode dumpfile: accepted
    publish(P, 0, manifest(kv_layers=1), np.random.default_rng(0))
    desc = Desc(
        xfer(0), transport={"kind": "fabric", "mode": "dumpfile", "layout_version": 1}
    )
    assert D.open_get(desc) is None  # claimed
    P.pump()
    g = D.open_get(desc)
    assert g is not None and g.ready()
    with pytest.raises(RuntimeError):
        P.open_get(Desc(xfer(0)))
    with pytest.raises(RuntimeError):
        D.open_put(xfer(0, D_ENGINE), manifest())
    D.finish_import(g, ok=False)
    P.shutdown()
    D.shutdown()


def test_manifest_spec_mismatch_rejected_at_open_put(tmp_path):
    world, clock, P, D = started_pair(tmp_path)
    m = build_manifest(
        100,
        model_sig="s",
        prompt_hash="p",
        kv_dtype="bfloat16",
        num_attn_layers=1,
        num_gdn_layers=1,
    )
    with pytest.raises(ValueError, match="fabric kv chunk spec"):
        P.open_put(xfer(0), m)
    with pytest.raises(ValueError, match="rec parts"):
        P.open_put(xfer(0), manifest(kv_layers=1, gdn_layers=3))  # rec_parts=2
    assert P._kv_pool.free == 16 and P._rec_pool.free == 4
    P.shutdown()
    D.shutdown()


def test_write_host_rec_row(tmp_path):
    """B > 1 producer (kv_both-style hook): the rec row arrives as a host tensor."""
    world, clock, P, D = started_pair(tmp_path)
    m = manifest(num_tokens=100, kv_layers=1)
    rng = np.random.default_rng(13)
    h = P.open_put(xfer(0), m)
    want = {}
    for p in kv_parts(m):
        blk = FakeTensor.random(KV_BM, rng)
        h.sinks[p.name].write_from_device(blk, chunk=0)
        want[(p.name, 0)] = ref_relayout(blk.data, 1088)
    for p in rec_parts(m):
        row = FakeTensor.random(REC, rng, on_device=False)
        h.sinks[p.name].write_host(row, chunk=0)
        want[p.name] = row.data.copy()
    for j, p in enumerate(taps_parts(m)):
        h.sinks[p.name].write_rows(rows_tensor(j))
    P.finish_export(h, "READY")
    g = claim_and_send(P, D, 0)
    got = import_like_hook(g, m, [FakeTensor.zeros(KV_HM), FakeTensor.zeros(KV_HM)])
    inst = install_like_hook(g, m, FakeTensor.zeros(REC))
    assert all(np.array_equal(got[k], want[k]) for k in got)
    assert all(np.array_equal(inst[p.name], want[p.name]) for p in rec_parts(m))
    D.finish_import(g, ok=True)
    P.shutdown()
    D.shutdown()


def test_partial_export_nbytes_present(tmp_path):
    """A chunk the hook never wrote is absent from the channel and from nbytes_present
    (validate_gdn_parts / the hook's chunk() then fail the load, never a hang)."""
    world, clock, P, D = started_pair(tmp_path)
    m = manifest(kv_layers=1)
    h = P.open_put(xfer(0), m)
    rng = np.random.default_rng(0)
    for p in kv_parts(m):
        h.sinks[p.name].write_from_device(FakeTensor.random(KV_BM, rng), chunk=0)
    # gdn.L1.rec never written, chunks 1..2 never written
    h.sinks["gdn.L0.rec"].write_from_device(FakeTensor.random(REC, rng), chunk=0)
    for j, p in enumerate(taps_parts(m)):
        h.sinks[p.name].write_rows(rows_tensor(j))
    P.finish_export(h, "READY")
    assert load_json(os.path.join(seg_dirs(D, 0)[1], SIDECAR_NAME))["nitems"] == 3
    g = claim_and_send(P, D, 0)
    assert world.channel.pending_sends == 3
    assert load_json(marker_path(D, 0))["nitems"] == 3
    assert g.sources["kv.L0.k"].nbytes_present == g.sources["kv.L0.k"].spec.chunk_nbytes
    assert g.sources["gdn.L1.rec"].nbytes_present == 0
    assert g.sources["gdn.L0.rec"].nbytes_present == g.sources["gdn.L0.rec"].spec.nbytes
    st = FakeTensor.zeros(KV_HM)
    g.sources["kv.L0.k"].chunk(0).read_into_device(st)
    g.sources["kv.L0.v"].chunk(0).read_into_device(st)
    assert (
        world.channel.pending_sends == 0
    )  # the single rec was posted after the last kv
    with pytest.raises(RuntimeError, match="after all"):
        g.sources["kv.L0.k"].chunk(1).read_into_device(st)
    with pytest.raises(RuntimeError, match="no rec item"):
        g.sources["gdn.L1.rec"].chunk(0).read_into_device(FakeTensor.zeros(REC))
    D.finish_import(g, ok=False)
    P.shutdown()
    D.shutdown()


def test_two_rank_threads_blocking_recv_identity(tmp_path):
    """Both engines on their own thread: the producer pumps like its step / idle
    tick, the consumer's first open_get after each claim spins for the marker
    (claim_wait_s) and its recvs block until the sends arrive (the CQ parks), two
    requests back to back."""
    world = FakeWorld()
    world_layers = {
        0: world.layer(0),
        1: world.layer(1, blocking_recv=True, recv_timeout=10),
    }
    clock = Clock()
    common = dict(
        control_dir=str(tmp_path / "ctrl"),
        mesh_device=object(),
        janitor_period=0,
        clock=clock,
        rec_sets=2,
        rec_parts=2,
        export_budget_bytes=32 * KV_NBYTES,
        kv_parts=4,
        max_model_len=16384,
        socket_timeout_s=5.0,
        claim_wait_s=0.2,
    )
    P = FabricSocketTransport(
        engine_id=P_ENGINE, role="producer", socket_layer=world_layers[0], **common
    )
    D = FabricSocketTransport(
        engine_id=D_ENGINE, role="consumer", socket_layer=world_layers[1], **common
    )
    assert not start_both(P, D)
    ms = [manifest(kv_layers=2), manifest(num_tokens=100, kv_layers=1)]
    want: dict = {}
    got: dict = {}
    errors: list = []
    go = threading.Event()
    done = threading.Event()

    def producer():
        try:
            go.wait(5)
            rng = np.random.default_rng(21)
            for i, m in enumerate(ms):
                time.sleep(0.02)  # the consumer polls open_get meanwhile
                _, w = publish(P, i, m, rng)
                want[i] = w
            while not done.is_set():  # the step / idle-tick pump
                P.pump()
                time.sleep(0.002)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    def consumer():
        try:
            for i, m in enumerate(ms):
                g = None
                t0 = time.perf_counter()
                while g is None or g.status == "MISSING":
                    # MISSING only before the producer created the segment (in the
                    # real system D asks after P's response, so never), None =
                    # WRITING, or claimed and waiting for the producer's sends
                    g = D.open_get(Desc(xfer(i)))
                    if g is None or g.status == "MISSING":
                        assert time.perf_counter() - t0 < 10, (
                            "open_get never became READY"
                        )
                        time.sleep(0.001)
                assert g.ready(), g.reason
                staging = [FakeTensor.zeros(KV_HM), FakeTensor.zeros(KV_HM)]
                r = import_like_hook(g, m, staging)  # blocking recvs: st holds the data
                validate_like_hook(g, m)
                D.layer.sync(None)
                r.update(install_like_hook(g, m, FakeTensor.zeros(REC)))
                D.finish_import(g, ok=True)
                got[i] = r
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
        finally:
            done.set()

    tp, tc = threading.Thread(target=producer), threading.Thread(target=consumer)
    tc.start()
    tp.start()
    go.set()
    tc.join(20)
    done.set()
    tp.join(20)
    assert not errors, errors
    for i, m in enumerate(ms):
        for k, v in want[i].items():
            if isinstance(v, torch.Tensor):
                assert torch.equal(got[i][k], v)
            else:
                assert np.array_equal(got[i][k], v), (i, k)
    assert world.channel.pending_sends == 0 and world.channel.pending_recvs == 0
    P.pump()
    assert P.outstanding_exports() == 0 and P._kv_pool.free == 32
    assert P.stats["exports_sent"] == 2 and D.stats["drains"] == 0
    P.shutdown()
    D.shutdown()


def test_producer_janitor_sweeps_expired_unsent_ready_segments(tmp_path):
    """An unclaimed READY segment has nothing on the channel (claim-gated sends):
    the shm janitor's lease-expiry rule applies and the buffers come back."""
    world, clock, P, D = started_pair(tmp_path, lease=1.0)
    publish(P, 0, manifest(kv_layers=1), np.random.default_rng(0))
    clock.now += 100
    P.control.janitor_once()
    pub = seg_dirs(D, 0)[1]
    assert not os.path.isdir(pub) and P.control.stats["expired"] == 1
    P.pump()
    assert P.outstanding_exports() == 0 and P._kv_pool.free == 16
    D.pump()  # nothing to drain
    assert world.channel.pending_sends == 0 and D.stats["drained_xfers"] == 0
    P.shutdown()
    D.shutdown()


# --- pool sizing / control-plane budget (review findings 2 + 3) -----------------------


def test_default_export_budget_covers_one_max_length_export(monkeypatch):
    monkeypatch.delenv("TT_PD_FABRIC_EXPORT_BUDGET", raising=False)
    monkeypatch.delenv("TT_PD_FABRIC_EXPORT_SLOTS", raising=False)
    monkeypatch.delenv("TT_PD_FABRIC_MAX_MODEL_LEN", raising=False)
    # 65536 tokens -> 1024 blocks -> 32 chunks x 32 K/V parts = 1024 head-major buffers
    assert max_export_kv_buffers(65536) == max_export_kv_buffers(65535) == 1024
    assert export_pool_bytes(65536) == 1024 * KV_NBYTES == 2_281_701_376
    cfg = FabricConfig.build(None)
    assert (cfg.max_model_len, cfg.kv_parts, cfg.export_slots) == (65536, 32, 1)
    assert cfg.export_budget_bytes == export_pool_bytes(65536)
    assert cfg.export_pool_buffers == cfg.max_export_kv_buffers == 1024
    # the old fixed 2 GiB default held 963 buffers: below one max-length export
    old = FabricConfig.build(None, export_budget_bytes=2 << 30)
    assert old.export_pool_buffers == 963 < old.max_export_kv_buffers
    # two slots pipeline two full-length exports
    two = FabricConfig.build({"fabric_export_slots": 2, "fabric_max_model_len": 65536})
    assert two.export_budget_bytes == 2 * export_pool_bytes(65536)
    # a shorter served context shrinks the derived pool; kv_parts follows the model
    small = FabricConfig.build({"fabric_max_model_len": 4096, "fabric_kv_parts": 8})
    assert small.export_pool_buffers == 2 * 8 == small.max_export_kv_buffers
    # env fallback (0 / unset = derived)
    monkeypatch.setenv("TT_PD_FABRIC_EXPORT_BUDGET", "0")
    assert FabricConfig.build(None).export_budget_bytes == export_pool_bytes(65536)
    monkeypatch.setenv("TT_PD_FABRIC_EXPORT_BUDGET", str(3 * KV_NBYTES))
    assert FabricConfig.build(None).export_budget_bytes == 3 * KV_NBYTES
    with pytest.raises(ValueError, match="export_slots"):
        FabricConfig.build({"fabric_export_slots": 0})


def test_start_refuses_pool_below_one_max_length_export(tmp_path):
    """Boundary: 32 parts x 1 chunk needs 32 buffers; 31 is refused BEFORE any
    allocation, 32 is accepted (and the refusal names the numbers)."""
    world = FakeWorld()
    base = dict(
        control_dir=str(tmp_path / "ctrl"),
        mesh_device=object(),
        janitor_period=0,
        socket_timeout_s=5.0,
        rec_sets=1,
        rec_parts=2,
        kv_parts=32,
        max_model_len=2048,
    )
    P = FabricSocketTransport(
        engine_id=P_ENGINE,
        role="producer",
        socket_layer=world.layer(0),
        export_budget_bytes=31 * KV_NBYTES,
        **base,
    )
    with pytest.raises(RuntimeError, match="holds 31 K/V buffers .* needs 32"):
        P.start()
    assert P.layer.allocated == [] and not P._started and P._ctrl is None
    # the 963-buffer old default vs the served 65536 context, same refusal
    P2 = FabricSocketTransport(
        engine_id=P_ENGINE,
        role="producer",
        socket_layer=world.layer(0),
        export_budget_bytes=2 << 30,
        **{**base, "max_model_len": 65536},
    )
    with pytest.raises(RuntimeError, match="holds 963 K/V buffers .* needs 1024"):
        P2.start()
    assert P2.layer.allocated == []
    # exactly one max-length export fits: the pair starts and the pool is 32 deep
    P3 = FabricSocketTransport(
        engine_id=P_ENGINE,
        role="producer",
        socket_layer=world.layer(0),
        export_budget_bytes=32 * KV_NBYTES,
        **base,
    )
    D3 = FabricSocketTransport(
        engine_id=D_ENGINE,
        role="consumer",
        socket_layer=world.layer(1),
        export_budget_bytes=32 * KV_NBYTES,
        **base,
    )
    assert not start_both(P3, D3)
    assert len(P3._kv_pool) == 32 == P3.cfg.max_export_kv_buffers
    P3.shutdown()
    D3.shutdown()


def full_manifest(num_tokens: int) -> Manifest:
    """The served model's manifest: 32 K/V parts, 48 recs, 48 taps."""
    return build_manifest(num_tokens, model_sig="sig", prompt_hash="ph")


def test_control_plane_charges_only_stored_bytes(tmp_path):
    """The reused shm control segments must not charge the manifest's K/V + rec
    bytes (they travel over the socket): a 32768- and a 65535-token export open
    back-to-back under the default 1 GiB control budget."""
    ctrl = _ControlSegments(
        engine_id=P_ENGINE,
        shm_dir=str(tmp_path / "ctrl"),
        role="producer",
        janitor_period=0,
    )
    ctrl.start()
    try:
        m32, m64 = full_manifest(32768), full_manifest(65535)
        assert m64.total_nbytes > 2 << 30  # 2.27 GiB on the wire
        assert m32.total_nbytes + m64.total_nbytes > ctrl.budget_bytes
        h32 = ctrl.open_put(xfer(0), m32)
        h64 = ctrl.open_put(xfer(1), m64)
        assert h32 is not None and h64 is not None
        assert ctrl.stats["budget_refusals"] == 0
        taps = sum(p.nbytes for p in m64.parts if p.kind == "gdn_taps")
        assert taps == 48 * 81_920
        charged = ctrl.outstanding_bytes()
        # header + manifest json + taps rows per segment, nothing else
        assert 2 * taps < charged < 2 * (taps + (1 << 20))
        ctrl.finish_export(h32, "FAILED")
        ctrl.finish_export(h64, "FAILED")
    finally:
        ctrl.shutdown()


def test_transport_accepts_65535_token_manifest(tmp_path):
    """End-to-end at the served size: the derived default pool holds the 1024
    head-major buffers of a 65535-token manifest and the control plane does not
    refuse it (open_put returns a handle); FAILED returns everything."""
    world = FakeWorld()
    common = dict(
        control_dir=str(tmp_path / "ctrl"),
        mesh_device=object(),
        janitor_period=0,
        socket_timeout_s=5.0,
        rec_sets=1,
        rec_parts=48,
        max_model_len=65536,
        kv_parts=32,
    )
    P = FabricSocketTransport(
        engine_id=P_ENGINE, role="producer", socket_layer=world.layer(0), **common
    )
    D = FabricSocketTransport(
        engine_id=D_ENGINE, role="consumer", socket_layer=world.layer(1), **common
    )
    assert P.cfg.export_budget_bytes == export_pool_bytes(65536)
    assert not start_both(P, D)
    assert len(P._kv_pool) == 1024 and len(P._rec_pool) == 48
    m = full_manifest(65535)
    assert sum(p.nchunks for p in kv_parts(m)) == 1024
    h = P.open_put(xfer(0), m)
    assert h is not None and P.stats["budget_refusals"] == 0
    assert P._kv_pool.free == 0 and P._rec_pool.free == 0
    assert P.control.stats["budget_refusals"] == 0
    # a second full-length export while the first is open: pool short (one slot),
    # refused by the pool -- never by the control plane
    assert P.open_put(xfer(1), full_manifest(65535)) is None
    assert P.control.stats["budget_refusals"] == 0
    P.finish_export(h, "FAILED")
    assert P._kv_pool.free == 1024 and P._rec_pool.free == 48
    P.shutdown()
    D.shutdown()


# --- finish_export partial-send failure (review minor) ----------------------------


class FailingSendLayer(FakeSocketLayer):
    """``send`` raises on the ``fail_at``-th data send (warm-up sends excluded)."""

    def __init__(self, world, rank, *, fail_at: int, **kw):
        super().__init__(world, rank, **kw)
        self.fail_at, self.data_sends, self.armed = fail_at, 0, False

    def send(self, t, sock):
        if self.armed:
            self.data_sends += 1
            if self.data_sends == self.fail_at:
                raise RuntimeError("simulated send_direct_async failure")
        super().send(t, sock)


def test_partial_send_failure_marks_failed_and_consumer_drains(tmp_path):
    world = FakeWorld()
    layer0 = FailingSendLayer(world, 0, fail_at=5)
    P, D = _pair_with_layer0(tmp_path, layer0, world)
    layer0.armed = True
    m = manifest(kv_layers=1)  # 6 kv + 2 rec items
    rng = np.random.default_rng(3)
    publish(P, 0, m, rng)
    claim(D, 0)
    P.pump()  # send #5 raises: 4 items parked, marker FAILED (seq consumed)
    mk = load_json(marker_path(D, 0))
    assert (mk["status"], mk["seq"], mk["nitems"], mk["n_kv"]) == ("FAILED", 0, 4, 4)
    assert len(mk["items"]) == 4 and world.channel.pending_sends == 4
    assert P.stats["sends"] == 4 and P.stats["exports_failed"] == 1
    # the seq is consumed and the buffers stay with the parked sends
    assert P._next_seq == 1 and P.outstanding_exports() == 1
    assert P._kv_pool.free == 10 and P._rec_pool.free == 2
    # the consumer asks for the failed xfer: definite FAILED, its items drained
    g = D.open_get(Desc(xfer(0)))
    assert g is not None and g.status == "FAILED" and "after 4 items" in g.reason
    assert world.channel.pending_sends == 0 and D.stats["drains"] == 4
    assert D.pending_receive_seq() == 1 and len(D._rec_sets) == 2
    assert not any(os.path.isdir(d) for d in seg_dirs(D, 0))
    assert not os.path.exists(marker_path(D, 0))
    P.pump()
    assert (
        P._kv_pool.free == 16 and P._rec_pool.free == 4 and P.outstanding_exports() == 0
    )
    # a later export flows normally with the next seq
    mB = manifest(num_tokens=100, kv_layers=1)
    _, wantB = publish(P, 1, mB, rng)
    gB = claim_and_send(P, D, 1)
    assert load_json(marker_path(D, 1))["seq"] == 1
    staging = [FakeTensor.zeros(KV_HM), FakeTensor.zeros(KV_HM)]
    got = import_like_hook(gB, mB, staging)
    assert all(np.array_equal(got[k], wantB[k]) for k in got)
    D.finish_import(gB, ok=True)
    # second failure (seq 2), then a good export (seq 3), both claimed: asking for
    # the good one first drains the failed head, like a released claim
    layer0.data_sends = 0
    publish(P, 2, m, rng)
    _, wantD = publish(P, 3, mB, rng)
    claim(D, 2)
    claim(D, 3)
    P.pump()
    assert world.channel.pending_sends == 8
    assert load_json(marker_path(D, 2))["status"] == "FAILED"
    assert load_json(marker_path(D, 3))["seq"] == 3
    gD = D.open_get(Desc(xfer(3)))
    assert gD is not None and gD.ready()
    assert world.channel.pending_sends == 4 and D.stats["drains"] == 8
    assert not any(os.path.isdir(d) for d in seg_dirs(D, 2))
    gC = D.open_get(Desc(xfer(2)))
    assert gC is not None and gC.status == "MISSING"
    got = import_like_hook(gD, mB, staging)
    assert all(np.array_equal(got[k], wantD[k]) for k in got)
    D.finish_import(gD, ok=True)
    P.pump()
    assert P.outstanding_exports() == 0 and P._kv_pool.free == 16
    assert P.stats["exports_failed"] == 2 and P.stats["exports_sent"] == 2
    assert P.stats["exports_ready"] == 4
    P.shutdown()
    D.shutdown()


def test_first_send_failure_consumes_no_seq(tmp_path):
    world = FakeWorld()
    layer0 = FailingSendLayer(world, 0, fail_at=1)
    P, D = _pair_with_layer0(tmp_path, layer0, world)
    layer0.armed = True
    m = manifest(kv_layers=1)
    rng = np.random.default_rng(4)
    publish(P, 0, m, rng)
    claim(D, 0)
    P.pump()
    mk = load_json(marker_path(D, 0))
    assert (mk["status"], mk["seq"], mk["nitems"]) == ("FAILED", None, 0)
    # nothing parked: buffers back at once, no seq consumed
    assert world.channel.pending_sends == 0 and P._next_seq == 0
    assert P._kv_pool.free == 16 and P.outstanding_exports() == 0
    g = D.open_get(Desc(xfer(0)))  # definite FAILED, claim dropped, marker gone
    assert g is not None and g.status == "FAILED" and "after 0 items" in g.reason
    assert not any(os.path.isdir(d) for d in seg_dirs(D, 0))
    assert not os.path.exists(marker_path(D, 0)) and len(D._rec_sets) == 2
    layer0.armed = False
    h2, want = publish(P, 1, m, rng)
    g2 = claim_and_send(P, D, 1)
    assert load_json(marker_path(D, 1))["seq"] == 0
    got = import_like_hook(g2, m, [FakeTensor.zeros(KV_HM), FakeTensor.zeros(KV_HM)])
    assert all(np.array_equal(got[k], want[k]) for k in got)
    D.finish_import(g2, ok=True)
    P.shutdown()
    D.shutdown()


# --- rendezvous timeout teardown (review minor) --------------------------------------


class TimingOutBarrierLayer(FakeSocketLayer):
    def barrier(self, timeout_s=None):
        self.barriers += 1
        raise RendezvousTimeout("fabric barrier did not complete within 1 s")


def test_rendezvous_timeout_leaves_device_tensors_alone(tmp_path):
    """The abandoned helper thread may still be inside the native call: start()
    must raise without deallocating the pools (the process exits)."""
    world = FakeWorld()
    P = FabricSocketTransport(
        engine_id=P_ENGINE,
        role="producer",
        socket_layer=TimingOutBarrierLayer(world, 0),
        mesh_device=object(),
        control_dir=str(tmp_path / "ctrl"),
        janitor_period=0,
        socket_timeout_s=1.0,
        rec_sets=1,
        rec_parts=2,
        export_budget_bytes=2 * KV_NBYTES,
        kv_parts=2,
        max_model_len=2048,
    )
    with pytest.raises(RendezvousTimeout):
        P.start()
    assert len(P.layer.allocated) == 4 and P.layer.deallocated == []
    assert not P._started and P._ctrl is None and P._kv_pool is None
    P.shutdown()  # nothing left to free; must not raise
    assert P.layer.deallocated == []


def test_spec_nbytes_matches_wire_layout():
    assert spec_nbytes(KV_HM) == 2_228_224 == spec_nbytes(KV_BM)
    assert spec_nbytes(REC) == 3_145_728
    assert spec_nbytes(((4, 10240), "bfloat16", "ROW_MAJOR")) == 81_920
    assert spec_nbytes(((1, 4, 2048, 256), "bfloat16", "TILE")) == 4_194_304


# --------------------------------------------------------------------------- #
# p1d1_opt lane B: the step-begin hold seam and the zero-copy rec buffer
# --------------------------------------------------------------------------- #
def test_wait_for_claims_sends_at_the_claim_and_returns_at_the_deadline(tmp_path):
    """The producer's step-begin hold: wait_for_claims pumps until a claim of a
    published-unsent export arrives (its sends go out inside the call, marker
    written) or the deadline passes (nothing enqueued, 0 returned);
    newest_unsent_ready_ts is the hold's clock (None = nothing to wait for)."""
    world, clock, P, D = started_pair(tmp_path, kv_bufs=32, rec_sets=4)
    rng = np.random.default_rng(3)
    assert P.newest_unsent_ready_ts() is None
    assert P.wait_for_claims(time.perf_counter() + 0.02) == 0  # nothing unsent
    publish(P, 0, manifest(), rng)
    ts = P.newest_unsent_ready_ts()
    assert ts is not None and ts <= time.perf_counter()
    t0 = time.perf_counter()
    assert P.wait_for_claims(t0 + 0.03) == 0  # no claim: full hold, nothing sent
    assert 0.025 <= time.perf_counter() - t0 < 1.0
    assert world.channel.pending_sends == 0 and P.unsent_exports() == 1
    # a claim landing during the hold is answered inside it
    th = threading.Timer(0.02, lambda: claim(D, 0))
    th.start()
    t0 = time.perf_counter()
    n = P.wait_for_claims(t0 + 2.0)
    dt = time.perf_counter() - t0
    th.join()
    assert n == 1 and 0.015 <= dt < 1.0, (n, dt)
    assert os.path.isfile(marker_path(D, 0)) and world.channel.pending_sends == 14
    assert P.newest_unsent_ready_ts() is None and P.unsent_exports() == 0
    # the NEWEST of two unsent exports clocks the hold (review finding 2: an orphan
    # published long ago must not disable the hold for a fresh export)
    publish(P, 1, manifest(num_tokens=100, kv_layers=1), rng)
    t1 = P.newest_unsent_ready_ts()
    P._exports[
        xfer(1)
    ].ready_ts -= 100.0  # the orphan: READY 100 s "ago", never claimed
    assert P.newest_unsent_ready_ts() == t1 - 100.0
    publish(P, 2, manifest(num_tokens=100, kv_layers=1), rng)
    t2 = P.newest_unsent_ready_ts()
    assert t2 > t1 - 50.0 and t2 <= time.perf_counter()
    # ... and the hold waits for the fresh export's claim although the orphan is stale
    th = threading.Timer(0.02, lambda: claim(D, 2))
    th.start()
    t0 = time.perf_counter()
    n = P.wait_for_claims(t2 + 2.0)
    th.join()
    assert n == 1 and 0.015 <= time.perf_counter() - t0 < 1.0
    assert (
        os.path.isfile(marker_path(D, 2)) and P.unsent_exports() == 1
    )  # the orphan stays
    # a consumer has nothing to hold for
    assert D.newest_unsent_ready_ts() is None
    assert D.wait_for_claims(time.perf_counter() + 0.01) == 0
    g = D.open_get(Desc(xfer(0)))
    assert g is not None and g.ready()
    P.shutdown()
    D.shutdown()


def test_rec_chunk_device_tensor_is_the_received_buffer_without_a_copy(tmp_path):
    """install_gdn_state fill_cache's the rec row straight from the received pool
    buffer: device_tensor() hands out that buffer (same bytes as the staging copy
    path), only for rec items and only once every K/V item was received."""
    world, clock, P, D = started_pair(tmp_path)
    m = manifest()
    rng = np.random.default_rng(5)
    h, want = publish(P, 0, m, rng)
    claim(D, 0)
    P.pump()
    g = D.open_get(Desc(xfer(0)))
    assert g is not None and g.ready()
    rec_src = g.sources[rec_parts(m)[0].name].chunk(0)
    with pytest.raises(RuntimeError, match="before the K/V items"):
        rec_src.device_tensor()
    with pytest.raises(NotImplementedError):
        g.sources[kv_parts(m)[0].name].chunk(0).device_tensor()
    staging = [FakeTensor.zeros(KV_HM), FakeTensor.zeros(KV_HM)]
    import_like_hook(g, m, staging)  # every K/V item received -> the recs are posted
    for p in rec_parts(m):
        src = g.sources[p.name].chunk(0)
        t = src.device_tensor()
        assert t.spec == REC and np.array_equal(t.data, want[p.name])
        assert t is g.sources[p.name].chunk(0).device_tensor()  # the pool buffer itself
        st = FakeTensor.zeros(REC)
        src.read_into_device(st)
        assert np.array_equal(st.data, t.data)
    D.finish_import(g, ok=True)
    P.pump()
    P.shutdown()
    D.shutdown()


# --- rendezvous: wait for a late peer, ignore a dead peer's leftover, time out ------
# A rank with a warm tensor cache reaches start() in seconds while its peer converts
# a cold cache for minutes; the peer's rendezvous file must be awaited, not read once
# (2026-09-22: the p1d1 container's prefill rank failed "peer rank 1 left no
# rank1.json" and the pair wedged).


def _peer_file(t, role, pid):
    d = dict(t.cfg.spec_table(), engine_id="peer", role=role, epoch="e1", pid=pid)
    peer_rank = t.cfg.receiver_rank if t.is_producer else t.cfg.sender_rank
    p = t._rendezvous_path(peer_rank)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, p)
    return p


def test_rendezvous_waits_for_a_late_peer(tmp_path):
    world, clock, P, D = make_pair(tmp_path)
    P.cfg.socket_timeout_s = 5.0
    t = threading.Timer(0.3, _peer_file, args=(P, "consumer", os.getpid()))
    t0 = time.monotonic()
    t.start()
    try:
        P._read_peer_rendezvous()
    finally:
        t.join()
    assert P._peer["role"] == "consumer"
    assert 0.25 <= time.monotonic() - t0 < 4.0


def test_rendezvous_ignores_a_dead_peers_leftover(tmp_path):
    world, clock, P, D = make_pair(tmp_path)
    P.cfg.socket_timeout_s = 5.0
    dead = 2**22 + 12345  # beyond pid_max on this box: certainly not alive
    stale = _peer_file(P, "consumer", dead)
    t = threading.Timer(0.3, _peer_file, args=(P, "consumer", os.getpid()))
    t.start()
    try:
        P._read_peer_rendezvous()
    finally:
        t.join()
    assert P._peer["pid"] == os.getpid()
    assert os.path.exists(stale)  # replaced by the live peer's file


def test_rendezvous_times_out_with_a_clear_message(tmp_path):
    world, clock, P, D = make_pair(tmp_path)
    P.cfg.socket_timeout_s = 0.3
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="left no .*rank1.json within 0 s"):
        P._read_peer_rendezvous()
    assert time.monotonic() - t0 < 3.0


def test_rendezvous_consumer_waits_for_producer(tmp_path):
    world, clock, P, D = make_pair(tmp_path)
    D.cfg.socket_timeout_s = 5.0
    t = threading.Timer(0.2, _peer_file, args=(D, "producer", os.getpid()))
    t.start()
    try:
        D._read_peer_rendezvous()
    finally:
        t.join()
    assert D._recv_epoch == "e1" and D.pending_receive_seq() == 0
