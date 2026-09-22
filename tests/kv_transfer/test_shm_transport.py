# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""Host-only tests for the v1 shm KV transport (PHASE2_DESIGN 4.1/4.2/3.5, 8 step 5).

No ttnn: a numpy-backed ``FakeTensor`` / ``FakeDeviceLayer`` stands in for the device
layer, so every path of ``ShmTransport`` (dumpfile AND raw mode, with and without
the step-5 byte helpers) runs against real files under a temporary shm_dir.
Covered: header state machine, atomic publish + claim rename + consumer-side unlink,
``open_get`` on a foreign claim -> MISSING, dead-producer WRITING -> FAILED, janitor
(terminal states, lease expiry, stale ``.claimed-*``, budget -> ``open_put`` None,
startup sweep), ``release_remote`` idempotent on missing/claimed/terminal, chunk
offsets == ``c * chunk_nbytes``, region writes land at ``chunk_offsets[c] +
dst_offset``, block-major -> head-major tile-record gather vs a reference
permutation, ``read/write_rows``, descriptor carries no path, the 5.6 byte-size
table, and concurrent producer/consumer PROCESSES (multiprocessing fork).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import time
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from vllm_tt_plugin.kv_transfer.transport import base, shm
from vllm_tt_plugin.kv_transfer.transport.base import (
    Manifest,
    PartSpec,
    build_manifest,
    chunk_offsets,
    kv_part_spec,
    rec_part_spec,
    request_nbytes,
    taps_part_spec,
)
from vllm_tt_plugin.kv_transfer.transport.fabric import FabricSocketTransport
from vllm_tt_plugin.kv_transfer.transport.shm import (
    FAILED,
    LOAD_FAILED,
    READY,
    RELEASED,
    WRITING,
    ShmTransport,
    crc32c,
    read_header,
    read_status,
    relayout_block_major_to_head_major,
)

XFER_HEX = "0123456789abcdef0123456789abcdef"
XFER_HEX2 = "fedcba9876543210fedcba9876543210"
P_ENGINE, D_ENGINE, OTHER = "prefill-0", "decode-0", "decode-9"


# --- fake device layer --------------------------------------------------------------


@dataclass
class FakeTensor:
    shape: tuple[int, ...]
    dtype: str
    layout: str
    data: np.ndarray  # uint8, the tensor's bytes
    on_device: bool

    @classmethod
    def zeros(cls, spec: PartSpec, on_device: bool, nbytes: int | None = None):
        n = spec.chunk_nbytes if nbytes is None else nbytes
        return cls(
            spec.shape, spec.dtype, spec.layout, np.zeros(n, np.uint8), on_device
        )

    @classmethod
    def random(cls, spec: PartSpec, rng, on_device=True, shape=None, nbytes=None):
        n = spec.chunk_nbytes if nbytes is None else nbytes
        return cls(
            shape or spec.shape,
            spec.dtype,
            spec.layout,
            rng.integers(0, 256, n, dtype=np.uint8),
            on_device,
        )


class FakeDeviceLayer:
    """numpy stand-in for ``DeviceLayer``; ``byte_helpers`` toggles the step-5 path."""

    def __init__(self, byte_helpers: bool = False) -> None:
        self.byte_helpers = byte_helpers
        self.calls: list[str] = []

    def spec_of(self, t):
        return (tuple(t.shape), t.dtype, t.layout)

    def from_device(self, t):
        assert t.on_device
        self.calls.append("from_device")
        return FakeTensor(t.shape, t.dtype, t.layout, t.data.copy(), False)

    def dump_tensor(self, path, h):
        assert path.endswith(".tensorbin") and not h.on_device
        with open(path, "wb") as f:
            pickle.dump((h.shape, h.dtype, h.layout, h.data.tobytes()), f)

    def load_tensor(self, path, device):
        with open(path, "rb") as f:
            shape, dtype, layout, raw = pickle.load(f)
        self.calls.append("load_tensor")
        return FakeTensor(
            tuple(shape), dtype, layout, np.frombuffer(raw, np.uint8).copy(), True
        )

    def allocate_host_like(self, t):
        self.calls.append("allocate_host_like")
        return FakeTensor(t.shape, t.dtype, t.layout, np.zeros_like(t.data), False)

    def allocate_host(self, spec, device):
        return FakeTensor.zeros(spec, False)

    def copy_device_to_host(self, d, h, *, blocking, cq_id):
        assert d.on_device and not h.on_device and blocking
        self.calls.append("copy_device_to_host")
        h.data[:] = d.data

    def copy_host_to_device(self, h, d, *, cq_id):
        assert d.on_device and not h.on_device
        self.calls.append("copy_host_to_device")
        d.data[:] = h.data

    def host_bytes(self, h):
        assert not h.on_device
        return h.data

    def read_tensor_bytes(self, t, dst, src_offset, nbytes, *, blocking):
        if not self.byte_helpers:
            raise NotImplementedError("not built")
        self.calls.append("read_tensor_bytes")
        np.frombuffer(dst, np.uint8)[:] = t.data[src_offset : src_offset + nbytes]


def small_manifest(num_tokens=100, *, kv_layers=1, gdn_layers=1, kv_dtype="bfloat8_b"):
    return build_manifest(
        num_tokens,
        model_sig="sig",
        prompt_hash="ph",
        kv_dtype=kv_dtype,
        num_attn_layers=kv_layers,
        num_gdn_layers=gdn_layers,
    )


def make(tmp_path, engine, mode="dumpfile", role="both", **kw):
    kw.setdefault("device_layer", FakeDeviceLayer())
    kw.setdefault("janitor_period", 0)  # tests drive janitor_once() by hand
    return ShmTransport(
        engine_id=engine, shm_dir=str(tmp_path / "shm"), mode=mode, role=role, **kw
    )


def xfer(engine=P_ENGINE, hx=XFER_HEX):
    return f"{engine}:{hx}"


@dataclass
class Desc:  # TransferDescriptor look-alike (design 3.4)
    xfer_id: str
    engine_id: str = P_ENGINE
    transport: dict | None = None


def rows_tensor(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(4, 10240, generator=g).to(torch.bfloat16)


def fill_all(h, rng, layer, mode):
    """Write every part of a PutHandle; return {part: {chunk: bytes}} written."""
    want = {}
    for name, sink in h.sinks.items():
        spec = sink.spec
        if spec.kind == "gdn_taps":
            r = rows_tensor(len(want))
            sink.write_rows(r)
            want[name] = {0: r}
            continue
        for c in range(spec.nchunks):
            t = FakeTensor.random(spec, rng)
            if c % 2 == 1 and mode == "dumpfile":
                sink.write_host(layer.from_device(t), chunk=c)  # write_host path
            else:
                sink.write_from_device(t, chunk=c, blocking=False)
            want.setdefault(name, {})[c] = t.data.copy()
    return want


# --- geometry / byte sizes (4.1, 5.6) ------------------------------------------


def test_chunk_offsets_and_part_specs():
    assert chunk_offsets(3, 10) == [0, 10, 20]
    kv = kv_part_spec("kv.L0.k", nblk=64)
    assert (kv.nchunks, kv.chunk_nbytes, kv.nbytes) == (2, 2_228_224, 4_456_448)
    assert kv.chunk_offsets == [0, 2_228_224] and kv.shape == (32, 4, 64, 256)
    assert kv_part_spec("x", 33).nchunks == 2 and kv_part_spec("x", 32).nchunks == 1
    assert kv_part_spec("x", 64, "bfloat16").chunk_nbytes == 32 * 131_072
    assert base.kv_block_nbytes("bfloat8_b") == 69_632
    rec, taps = rec_part_spec("gdn.L0.rec"), taps_part_spec("gdn.L0.taps")
    assert (rec.chunk_nbytes, rec.shape) == (3_145_728, (1, 48, 128, 128))
    assert (taps.chunk_nbytes, taps.shape, taps.layout) == (
        81_920,
        (4, 10240),
        "ROW_MAJOR",
    )
    for spec in (kv, rec, taps):
        assert spec.chunk_nbytes % 4096 == 0  # 4 KiB-aligned raw regions


def test_byte_sizes_match_design_5_6_table():
    assert request_nbytes(4095) == 297_533_440
    assert request_nbytes(32767) == 1_295_777_792
    assert (
        request_nbytes(4095, kv_dtype="bfloat16")
        == 256 * 2**20 + 150_994_944 + 3_932_160
    )
    m = build_manifest(4095, model_sig="s", prompt_hash="p")
    assert len(m.parts) == 32 + 48 + 48 and m.nblk == 64
    assert m.total_nbytes == 297_533_440  # 64 blocks = whole chunks: no padding
    m = build_manifest(32767, model_sig="s", prompt_hash="p")
    assert m.total_nbytes == 1_295_777_792 and m.part("kv.L15.v").nchunks == 16
    # a partial chunk pads to 32 blocks on the wire
    m = build_manifest(65, model_sig="s", prompt_hash="p")
    assert m.nblk == 2 and m.part("kv.L0.k").nbytes == 2_228_224
    rt = Manifest.from_json(m.to_json())
    assert rt == m


def test_coerce_manifest_accepts_metadata_dataclasses():
    from vllm_tt_plugin.kv_transfer import metadata as md

    mine = small_manifest(100)
    theirs = md.Manifest(
        **{
            **mine.__dict__,
            "parts": [
                md.PartSpec(**p.to_dict() | {"shape": tuple(p.shape)})
                for p in mine.parts
            ],
        }
    )
    assert base.coerce_manifest(theirs) == mine


def test_crc32c_vector_and_relayout_reference():
    assert crc32c(b"123456789") == 0xE3069283
    assert crc32c(np.frombuffer(b"123456789", np.uint8)) == 0xE3069283
    for rec in (1088, 2048):
        n, h, r, col = 32, 4, 2, 8
        src = np.random.default_rng(rec).integers(
            0, 256, n * h * r * col * rec, np.uint8
        )
        nvalid = 20
        out = relayout_block_major_to_head_major(src, rec, nvalid)
        ref = np.zeros_like(out)
        for b in range(n):
            for hh in range(h):
                for rr in range(r):
                    for cc in range(col):
                        si = ((b * 4 + hh) * 2 + rr) * 8 + cc
                        di = (hh * (2 * n) + (b * 2 + rr)) * 8 + cc
                        if b < nvalid:
                            ref[di * rec : (di + 1) * rec] = src[
                                si * rec : (si + 1) * rec
                            ]
        assert np.array_equal(out, ref)
        assert out.nbytes == src.nbytes


# --- header state machine, publish, claim, unlink (4.2) -------------------------


@pytest.mark.parametrize("mode", ["dumpfile", "raw"])
def test_publish_claim_import_unlink(tmp_path, mode):
    rng = np.random.default_rng(1)
    P = make(tmp_path, P_ENGINE, mode, role="producer")
    D = make(tmp_path, D_ENGINE, mode, role="consumer")
    P.start()
    assert P.descriptor() == {"kind": "shm", "mode": mode, "layout_version": 1}
    assert not any("/" in str(v) for v in P.descriptor().values())  # no paths
    m = small_manifest(100)  # nblk 2 -> 1 kv chunk with 30 pad blocks
    xid = xfer()
    h = P.open_put(xid, m)
    assert h is not None and set(h.sinks) == {p.name for p in m.parts}
    assert h.lease_expiry_ts > time.time()
    edir = tmp_path / "shm" / P_ENGINE
    assert (edir / f"{XFER_HEX}.tmp").is_dir() and not (edir / XFER_HEX).exists()
    hp = edir / f"{XFER_HEX}.tmp" / ("data" if mode == "raw" else "header")
    assert read_status(hp) == WRITING
    assert D.open_get(Desc(xid)) is None  # WRITING, producer alive -> not yet
    want = fill_all(h, rng, P.layer, mode)
    P.finish_export(h, "READY")
    P.finish_export(h, "READY")  # idempotent
    assert (edir / XFER_HEX).is_dir() and not (edir / f"{XFER_HEX}.tmp").exists()
    assert read_status(edir / XFER_HEX / hp.name) == READY
    P.janitor_once()  # READY + unexpired: untouched
    assert (edir / XFER_HEX).is_dir() and P.outstanding_bytes() > 0

    g = D.open_get(Desc(xid, transport=P.descriptor()))
    assert g is not None and g.ready() and g.manifest == m
    claim = edir / f"{XFER_HEX}.claimed-{D_ENGINE}"
    assert claim.is_dir() and not (edir / XFER_HEX).exists()
    assert D.open_get(Desc(xid)).ready()  # re-open of our own claim
    P.janitor_once()  # a young claim is never touched by the producer's janitor
    assert claim.is_dir()

    for name, chunks in want.items():
        src = g.sources[name]
        assert src.spec == m.part(name)
        assert src.nbytes_present == src.spec.nbytes
        if src.spec.kind == "gdn_taps":
            assert torch.equal(src.read_rows(), chunks[0])
            continue
        for c, data in chunks.items():
            ch = src.chunk(c)
            if mode == "dumpfile":
                assert not ch.is_head_major and not ch.is_device_readable
                t = ch.read_device(mesh_device=None)
                assert (t.shape, t.dtype, t.layout) == src.spec.spec_key()
                assert np.array_equal(t.data, data)
                with pytest.raises(NotImplementedError):
                    ch.read_into_device(None)
            else:
                assert ch.is_device_readable
                if src.spec.kind == "kv_blocks":
                    assert ch.is_head_major
                    st = FakeTensor(
                        (1, 4, 2048, 256),
                        src.spec.dtype,
                        "TILE",
                        np.zeros(src.spec.chunk_nbytes, np.uint8),
                        True,
                    )
                    ch.read_into_device(st)
                    exp = relayout_block_major_to_head_major(
                        data, base.TILE_RECORD_BYTES[src.spec.dtype], m.nblk - 32 * c
                    )
                    assert np.array_equal(st.data, exp)
                else:
                    assert not ch.is_head_major
                    st = FakeTensor.zeros(src.spec, True)
                    ch.read_into_device(st)
                    assert np.array_equal(st.data, data)
                with pytest.raises(NotImplementedError):
                    ch.read_device(None)
    D.finish_import(g, ok=True)
    assert not claim.exists() and not (edir / XFER_HEX).exists()
    D.finish_import(g, ok=True)  # idempotent
    D.release_remote(xid)  # idempotent on missing
    assert D.open_get(Desc(xid)).status == "MISSING"
    P.janitor_once()
    assert P.outstanding_bytes() == 0 and xid not in P._puts
    P.shutdown()
    D.shutdown()


def test_failed_export_and_load_failed(tmp_path):
    P, D = make(tmp_path, P_ENGINE), make(tmp_path, D_ENGINE)
    xid = xfer()
    h = P.open_put(xid, small_manifest(64))
    P.finish_export(h, "FAILED")
    edir = tmp_path / "shm" / P_ENGINE
    assert read_status(edir / XFER_HEX / "header") == FAILED
    g = D.open_get(Desc(xid))
    assert g is not None and not g.ready() and g.status == "FAILED"
    assert not (edir / XFER_HEX).exists()  # claimed and dropped by the consumer
    assert D.open_get(Desc(xid)).status == "MISSING"
    # LOAD_FAILED path: a READY segment the consumer could not install
    h2 = P.open_put(xfer(hx=XFER_HEX2), small_manifest(64))
    fill_all(h2, np.random.default_rng(0), P.layer, "dumpfile")
    P.finish_export(h2, "READY")
    g2 = D.open_get(Desc(h2.xfer_id))
    assert g2.ready()
    hp = edir / f"{XFER_HEX2}.claimed-{D_ENGINE}" / "header"
    orig_write = shm.write_status
    seen = []
    shm.write_status = lambda p, s: (seen.append(s), orig_write(p, s))
    try:
        D.finish_import(g2, ok=False)
    finally:
        shm.write_status = orig_write
    assert seen == [LOAD_FAILED] and not hp.exists()


def test_open_get_edge_cases(tmp_path):
    P, D = make(tmp_path, P_ENGINE), make(tmp_path, D_ENGINE)
    edir = tmp_path / "shm" / P_ENGINE
    # missing
    assert D.open_get(Desc(xfer())).status == "MISSING"
    # malformed id
    assert D.open_get(Desc("not an id")).status == "FAILED"
    # transport mismatch
    h = P.open_put(xfer(), small_manifest(64))
    P.finish_export(h, "READY")
    g = D.open_get(
        Desc(xfer(), transport={"kind": "shm", "mode": "raw", "layout_version": 1})
    )
    assert g.status == "FAILED" and "mismatch" in g.reason
    assert (edir / XFER_HEX).is_dir()  # not claimed by a mismatching consumer
    # foreign claim -> MISSING
    other = make(tmp_path, OTHER)
    assert other.open_get(Desc(xfer())).ready()
    assert D.open_get(Desc(xfer())).status == "MISSING"
    # a dead producer left a WRITING .tmp -> FAILED
    h2 = P.open_put(xfer(hx=XFER_HEX2), small_manifest(64))
    hp = edir / f"{XFER_HEX2}.tmp" / "header"
    raw = bytearray(hp.read_bytes())
    import struct

    off = shm.PRODUCER_PID_OFFSET  # patch producer_pid to a pid that cannot exist
    raw[off : off + 4] = struct.pack("<I", 2**22 - 1)
    hp.write_bytes(bytes(raw))
    hdr = read_header(hp, full=False)
    assert hdr.producer_pid == 2**22 - 1 and hdr.status == WRITING
    g = D.open_get(Desc(h2.xfer_id))
    assert g.status == "FAILED" and "died" in g.reason
    # same .tmp with a live pid -> None
    raw[off : off + 4] = struct.pack("<I", os.getpid())
    hp.write_bytes(bytes(raw))
    assert D.open_get(Desc(h2.xfer_id)) is None
    P.abandon(h2.xfer_id)
    P.abandon(h2.xfer_id)  # idempotent
    assert not (edir / f"{XFER_HEX2}.tmp").exists()


def test_release_remote_idempotent_and_janitor_terminal(tmp_path):
    P, D = make(tmp_path, P_ENGINE), make(tmp_path, D_ENGINE)
    edir = tmp_path / "shm" / P_ENGINE
    D.release_remote(xfer())  # missing: no error
    D.release_remote("garbage")
    h = P.open_put(xfer(), small_manifest(64))
    P.finish_export(h, "READY")
    D.release_remote(xfer())  # unclaimed -> RELEASED, producer's janitor unlinks
    D.release_remote(xfer())
    assert read_status(edir / XFER_HEX / "header") == RELEASED
    P.janitor_once()
    assert not (edir / XFER_HEX).exists() and P.stats["swept"] == 1
    assert P.outstanding_bytes() == 0
    # claimed -> unlink directly
    h = P.open_put(xfer(hx=XFER_HEX2), small_manifest(64))
    P.finish_export(h, "READY")
    g = D.open_get(Desc(h.xfer_id))
    assert g.ready()
    D.release_remote(h.xfer_id)
    assert not any(edir.iterdir())
    D.release_remote(h.xfer_id)
    D.finish_import(g, ok=True)  # after release: no error


def test_janitor_lease_expiry_stale_claims_startup_sweep_and_budget(tmp_path):
    now = [1000.0]
    P = make(tmp_path, P_ENGINE, lease_duration=10.0, clock=lambda: now[0])
    D = make(tmp_path, D_ENGINE, clock=lambda: now[0])
    edir = tmp_path / "shm" / P_ENGINE
    h = P.open_put(xfer(), small_manifest(64))
    assert h.lease_expiry_ts == 1010.0
    P.finish_export(h, "READY")
    now[0] = 1009.0
    P.janitor_once()
    assert (edir / XFER_HEX).is_dir()
    now[0] = 1010.5
    P.janitor_once()  # consumer never came
    assert not (edir / XFER_HEX).exists() and P.stats["expired"] == 1
    assert D.open_get(Desc(xfer())).status == "MISSING"
    # stale .claimed-* (consumer died mid-import): swept only when the consumer pid
    # stamped at the claim is dead, NEVER on age alone (critic C2: a KV_DONE handle
    # is held across an unbounded promotion wait)
    h = P.open_put(xfer(hx=XFER_HEX2), small_manifest(64))
    P.finish_export(h, "READY")
    g = D.open_get(Desc(h.xfer_id))
    assert g.ready()
    claim = edir / f"{XFER_HEX2}.claimed-{D_ENGINE}"
    pid_file = claim / shm.CONSUMER_PID_FILE
    assert pid_file.read_text() == str(os.getpid())
    P.janitor_once()
    assert claim.is_dir()
    old = now[0] - 25.0
    os.utime(claim, (old, old))
    P.janitor_once()
    assert claim.is_dir() and P.stats["stale_claims"] == 0  # live pid, old: kept
    pid_file.write_text("2147483000")  # a pid that does not exist
    P.janitor_once()
    assert not claim.exists() and P.stats["stale_claims"] == 1
    # legacy claim without a pid file: age rule
    legacy = edir / f"{XFER_HEX2}.claimed-legacy"
    legacy.mkdir()
    P.janitor_once()
    assert legacy.is_dir()
    os.utime(legacy, (old, old))
    P.janitor_once()
    assert not legacy.exists() and P.stats["stale_claims"] == 2
    P.janitor_once()
    assert P.outstanding_bytes() == 0
    # a foreign .tmp (previous process) is swept by the janitor, an owned one is not
    (edir / "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.tmp").mkdir()
    h = P.open_put(xfer(), small_manifest(64))
    P.janitor_once()
    assert not (edir / "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.tmp").exists()
    assert (edir / f"{XFER_HEX}.tmp").is_dir()
    P.finish_export(h, "READY")
    # startup sweep: a restarted producer inherits nothing but young claims
    g = D.open_get(Desc(xfer()))
    assert g.ready()
    h2 = P.open_put(xfer(hx=XFER_HEX2), small_manifest(64))
    P.finish_export(h2, "READY")
    (edir / "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.tmp").mkdir()
    P2 = make(tmp_path, P_ENGINE, lease_duration=10.0, clock=lambda: now[0])
    P2.start()
    P2.shutdown()
    names = sorted(p.name for p in edir.iterdir())
    assert names == [f"{XFER_HEX}.claimed-{D_ENGINE}"]
    assert P2.outstanding_bytes() > 0  # the live claim counts until it vanishes
    D.finish_import(g, ok=True)
    P2.janitor_once()
    assert P2.outstanding_bytes() == 0
    # budget: open_put returns None instead of ENOSPC mid-write
    m = small_manifest(64)
    one = shm.segment_layout(m, "dumpfile")[1]
    total = one.data_off + one.total_nbytes
    Pb = make(tmp_path, P_ENGINE, budget_bytes=int(total * 1.5), clock=lambda: now[0])
    a = Pb.open_put(xfer(), m)
    assert a is not None
    assert (
        Pb.open_put(xfer(hx=XFER_HEX2), m) is None and Pb.stats["budget_refusals"] == 1
    )
    Pb.abandon(a.xfer_id)
    assert Pb.open_put(xfer(hx=XFER_HEX2), m) is not None
    with pytest.raises(ValueError):
        Pb.open_put(xfer(engine=D_ENGINE), m)  # not our engine


# --- raw mode: regions, offsets, fallbacks, checksum ----------------------------


@pytest.mark.parametrize("byte_helpers", [False, True])
def test_raw_regions_land_at_chunk_offsets(tmp_path, byte_helpers):
    layer = FakeDeviceLayer(byte_helpers=byte_helpers)
    P = make(tmp_path, P_ENGINE, "raw", device_layer=layer)
    D = make(tmp_path, D_ENGINE, "raw", device_layer=layer, relayout_workers=0)
    m = build_manifest(
        64 * 40, model_sig="s", prompt_hash="p", num_attn_layers=1, num_gdn_layers=1
    )  # nblk 40 -> 2 chunks (32 + 8)
    BLK = base.kv_block_nbytes("bfloat8_b")
    h = P.open_put(xfer(), m)
    kv = h.sinks["kv.L0.k"]
    assert kv.supports_regions and kv.spec.chunk_offsets == [0, 2_228_224]
    # the producer's paged cache: 50 blocks, a fragmented block list of 40
    rng = np.random.default_rng(7)
    cache = FakeTensor(
        (50, 4, 64, 256),
        "bfloat8_b",
        "TILE",
        rng.integers(0, 256, 50 * BLK, np.uint8),
        True,
    )
    ids = [3, 4, 5, 10, 11] + list(range(20, 50)) + [7, 8, 9, 12, 13]
    ids = ids[:40]
    expected = np.zeros(kv.spec.nbytes, np.uint8)
    for c in range(2):
        chunk_ids = ids[32 * c : 32 * c + 32]
        k0 = 0
        while k0 < len(chunk_ids):  # contiguous runs -> ONE region write each
            b0, n = chunk_ids[k0], 1
            while k0 + n < len(chunk_ids) and chunk_ids[k0 + n] == b0 + n:
                n += 1
            kv.write_region_from_device(
                cache,
                b0 * BLK,
                n * BLK,
                chunk=c,
                dst_offset_bytes=k0 * BLK,
                blocking=False,
            )
            dst = kv.spec.chunk_offsets[c] + k0 * BLK
            expected[dst : dst + n * BLK] = cache.data[b0 * BLK : (b0 + n) * BLK]
            k0 += n
    if byte_helpers:
        assert (
            "read_tensor_bytes" in layer.calls
            and "copy_device_to_host" not in layer.calls
        )
    else:
        assert "copy_device_to_host" in layer.calls
        # the fallback host tensor is allocated once per spec and cached
        assert layer.calls.count("allocate_host_like") == 1
    with pytest.raises(ValueError):
        kv.write_region_from_device(cache, 0, 100, chunk=0, dst_offset_bytes=0)
    with pytest.raises(ValueError):
        kv.write_region_from_device(cache, 0, BLK, chunk=1, dst_offset_bytes=32 * BLK)
    with pytest.raises(IndexError):
        kv.write_region_from_device(cache, 0, BLK, chunk=2, dst_offset_bytes=0)
    # rec: B>1 producer writes byte region of row `slot`; whole-chunk write for v
    ROW = 3_145_728
    rec_state = FakeTensor(
        (8, 48, 128, 128),
        "float32",
        "TILE",
        rng.integers(0, 256, 8 * ROW, np.uint8),
        True,
    )
    h.sinks["gdn.L0.rec"].write_region_from_device(
        rec_state, 5 * ROW, ROW, chunk=0, dst_offset_bytes=0
    )
    v = FakeTensor.random(m.part("kv.L0.v"), rng)
    with pytest.raises(ValueError):  # wrong spec is refused
        h.sinks["kv.L0.v"].write_from_device(cache, chunk=0)
    h.sinks["kv.L0.v"].write_from_device(v, chunk=0)
    h.sinks["kv.L0.v"].write_from_device(v, chunk=1)
    r = rows_tensor(3)
    h.sinks["gdn.L0.taps"].write_rows(r)
    with pytest.raises(ValueError):
        h.sinks["gdn.L0.taps"].write_rows(r.float())
    # write_host on a raw sink: memcpy of a host tensor
    hv = layer.from_device(v)
    h.sinks["kv.L0.v"].write_host(hv, chunk=1)
    P.finish_export(h, "READY")
    # raw file layout: parts at 4 KiB-aligned absolute offsets
    hdr = read_header(tmp_path / "shm" / P_ENGINE / XFER_HEX / "data")
    offs = {p.name: p.offset for p in hdr.parts}
    assert all(o % 4096 == 0 for o in offs.values()) and offs["kv.L0.k"] == hdr.data_off
    assert offs["kv.L0.v"] == offs["kv.L0.k"] + kv.spec.nbytes
    data = (tmp_path / "shm" / P_ENGINE / XFER_HEX / "data").read_bytes()
    got = np.frombuffer(
        data[offs["kv.L0.k"] : offs["kv.L0.k"] + kv.spec.nbytes], np.uint8
    )
    assert np.array_equal(got, expected)
    g = D.open_get(Desc(xfer(), transport=P.descriptor()))
    assert g.ready()
    s = g.sources["kv.L0.k"]
    assert s.nbytes_present == s.spec.nbytes and s.spec_crc == 0
    for c in range(2):
        st = FakeTensor(
            (1, 4, 2048, 256),
            "bfloat8_b",
            "TILE",
            np.zeros(kv.spec.chunk_nbytes, np.uint8),
            True,
        )
        s.chunk(c).read_into_device(st)
        nvalid = min(32, 40 - 32 * c)
        ref = relayout_block_major_to_head_major(
            expected[c * kv.spec.chunk_nbytes : (c + 1) * kv.spec.chunk_nbytes],
            1088,
            nvalid,
        )
        assert np.array_equal(st.data, ref)
        if c == 1:  # blocks 8..31 of chunk 1 zero-filled head-major
            hm = st.data.reshape(4, 32, 2 * 8 * 1088)
            assert not hm[:, 8:].any() and hm[:, :8].any()
    with pytest.raises(ValueError):  # staging of the wrong spec
        s.chunk(0).read_into_device(
            FakeTensor(
                (32, 4, 64, 256),
                "bfloat8_b",
                "TILE",
                np.zeros(kv.spec.chunk_nbytes, np.uint8),
                True,
            )
        )
    rs = g.sources["gdn.L0.rec"]
    st = FakeTensor.zeros(rs.spec, True)
    rs.chunk(0).read_into_device(st)
    assert np.array_equal(st.data, rec_state.data[5 * ROW : 6 * ROW])
    assert torch.equal(g.sources["gdn.L0.taps"].read_rows(), r)
    st = FakeTensor(
        (1, 4, 2048, 256),
        "bfloat8_b",
        "TILE",
        np.zeros(kv.spec.chunk_nbytes, np.uint8),
        True,
    )
    g.sources["kv.L0.v"].chunk(1).read_into_device(st)
    assert np.array_equal(st.data, relayout_block_major_to_head_major(v.data, 1088, 8))
    D.finish_import(g, ok=True)
    D.shutdown()
    P.shutdown()


def test_raw_relayout_threadpool_prefetch(tmp_path):
    layer = FakeDeviceLayer()
    P = make(tmp_path, P_ENGINE, "raw", device_layer=layer)
    D = make(tmp_path, D_ENGINE, "raw", device_layer=layer, relayout_workers=2)
    m = build_manifest(
        64 * 96, model_sig="s", prompt_hash="p", num_attn_layers=1, num_gdn_layers=0
    )  # 3 chunks
    h = P.open_put(xfer(), m)
    rng = np.random.default_rng(2)
    want = fill_all(h, rng, layer, "raw")
    P.finish_export(h, "READY")
    g = D.open_get(Desc(xfer()))
    s = g.sources["kv.L0.k"]
    ch0 = s.chunk(0)
    assert set(s._futures) == {0, 1}  # c and the prefetched c+1
    for c, ch in ((0, ch0), (1, s.chunk(1)), (2, s.chunk(2))):
        st = FakeTensor(
            (1, 4, 2048, 256),
            "bfloat8_b",
            "TILE",
            np.zeros(s.spec.chunk_nbytes, np.uint8),
            True,
        )
        ch.read_into_device(st)
        assert np.array_equal(
            st.data, relayout_block_major_to_head_major(want["kv.L0.k"][c], 1088, 32)
        )
    D.finish_import(g, ok=True)
    D.shutdown()


@pytest.mark.parametrize("mode", ["dumpfile", "raw"])
def test_checksum_flag(tmp_path, mode):
    layer = FakeDeviceLayer()
    P = make(tmp_path, P_ENGINE, mode, device_layer=layer, checksum=True)
    D = make(tmp_path, D_ENGINE, mode, device_layer=layer)
    m = Manifest(
        layout_version=1,
        model_sig="s",
        num_tokens=1,
        nblk=1,
        block_size=64,
        chunk_tokens=2048,
        kv_dtype="bfloat8_b",
        rec_dtype="float32",
        parts=[
            taps_part_spec("gdn.L0.taps"),
            PartSpec(
                "gdn.L0.rec", "gdn_rec", (1, 2, 32, 32), "float32", "TILE", 1, 8192
            ),
        ],
        prompt_hash="p",
    )
    h = P.open_put(xfer(), m)
    r = rows_tensor(5)
    h.sinks["gdn.L0.taps"].write_rows(r)
    t = FakeTensor.random(m.parts[1], np.random.default_rng(0))
    h.sinks["gdn.L0.rec"].write_from_device(t, chunk=0)
    P.finish_export(h, "READY")
    g = D.open_get(Desc(xfer()))
    for name in ("gdn.L0.taps", "gdn.L0.rec"):
        s = g.sources[name]
        assert s.spec_crc != 0 and s.crc32c() == s.spec_crc
    # tamper -> mismatch
    seg = tmp_path / "shm" / P_ENGINE / f"{XFER_HEX}.claimed-{D_ENGINE}"
    if mode == "dumpfile":
        p = seg / "gdn.L0.taps.rows.pt"
        raw = bytearray(p.read_bytes())
        raw[-1] ^= 0xFF
        p.write_bytes(bytes(raw))
    else:
        p = seg / "data"
        hdr = read_header(p)
        off = hdr.parts[0].offset
        with open(p, "r+b") as f:
            f.seek(off + 100)
            b = f.read(1)
            f.seek(off + 100)
            f.write(bytes([b[0] ^ 0xFF]))
    assert g.sources["gdn.L0.taps"].crc32c() != g.sources["gdn.L0.taps"].spec_crc
    D.finish_import(g, ok=False)
    D.shutdown()


def test_dumpfile_partial_export_nbytes_present(tmp_path):
    P, D = make(tmp_path, P_ENGINE), make(tmp_path, D_ENGINE)
    m = small_manifest(64 * 40)  # 2 kv chunks
    h = P.open_put(xfer(), m)
    rng = np.random.default_rng(0)
    h.sinks["kv.L0.k"].write_from_device(
        FakeTensor.random(m.part("kv.L0.k"), rng), chunk=0
    )
    P.finish_export(h, "READY")  # chunk 1 of kv.L0.k never written
    g = D.open_get(Desc(xfer()))
    s = g.sources["kv.L0.k"]
    assert s.nbytes_present == s.spec.chunk_nbytes < s.spec.nbytes
    # validate_gdn_parts (design 5.4) catches a never-written rec part here
    assert g.sources["gdn.L0.rec"].nbytes_present == 0
    with pytest.raises(NotImplementedError):
        h.sinks["kv.L0.k"].write_region_from_device(
            None, 0, 0, chunk=0, dst_offset_bytes=0
        )
    D.finish_import(g, ok=False)


def test_fabric_stub_seam():
    f = FabricSocketTransport(engine_id="p")
    assert f.descriptor() == {"kind": "fabric", "mode": "direct", "layout_version": 1}
    with pytest.raises(NotImplementedError):
        f.open_put(xfer(), small_manifest(64))
    with pytest.raises(NotImplementedError):
        f.open_get(Desc(xfer()))
    f.shutdown()  # tolerant
    from vllm_tt_plugin.kv_transfer.transport import make_transport

    assert isinstance(make_transport("fabric", engine_id="p"), FabricSocketTransport)
    assert isinstance(
        make_transport("shm", engine_id="p", janitor_period=0), ShmTransport
    )
    with pytest.raises(ValueError):
        make_transport("nixl", engine_id="p")


def test_make_transport_from_kv_transfer_config(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from vllm_tt_plugin.kv_transfer.transport import make_transport

    cfg = SimpleNamespace(
        engine_id="decode-0",
        kv_role="kv_consumer",
        kv_connector_extra_config={
            "shm_dir": str(tmp_path / "s"),
            "shm_budget_bytes": 1 << 20,
            "kv_lease_duration": 7.5,
        },
    )
    monkeypatch.setenv("TT_PD_CHECKSUM", "1")
    # the connector's positional call shape (tt_connector.attach_runner)
    t = make_transport("shm", "raw", cfg, cfg.kv_role, janitor_period=0)
    assert isinstance(t, ShmTransport)
    assert (t.engine_id, t.mode, t.role, t.shm_dir) == (
        "decode-0",
        "raw",
        "consumer",
        str(tmp_path / "s"),
    )
    assert (t.budget_bytes, t.lease_duration, t.checksum) == (1 << 20, 7.5, True)
    t.start()  # consumer: no janitor thread, no sweep
    assert t._janitor is None
    t.shutdown()
    f = make_transport("fabric", None, cfg, "kv_producer")
    assert isinstance(f, FabricSocketTransport) and f.engine_id == "decode-0"


# --- concurrent producer / consumer processes -----------------------------------------


def _producer_proc(shm_dir, mode, n, lease, ready_ev, done_ev):
    layer = FakeDeviceLayer()
    P = ShmTransport(
        engine_id=P_ENGINE,
        shm_dir=shm_dir,
        mode=mode,
        role="producer",
        lease_duration=lease,
        device_layer=layer,
        janitor_period=0.01,
    )
    P.start()
    rng = np.random.default_rng(123)
    for i in range(n):
        m = small_manifest(64 * 3, kv_layers=1, gdn_layers=1)
        h = P.open_put(f"{P_ENGINE}:{i:032x}", m)
        assert h is not None
        # deterministic contents: chunk bytes = i, rows = rows_tensor(i)
        for name, sink in h.sinks.items():
            if sink.spec.kind == "gdn_taps":
                sink.write_rows(rows_tensor(i))
            else:
                for c in range(sink.spec.nchunks):
                    t = FakeTensor(
                        sink.spec.shape,
                        sink.spec.dtype,
                        sink.spec.layout,
                        np.full(sink.spec.chunk_nbytes, (i + c) % 251, np.uint8),
                        True,
                    )
                    sink.write_from_device(t, chunk=c)
        time.sleep(0.005 * rng.integers(0, 4))
        P.finish_export(h, "READY")
    ready_ev.set()
    done_ev.wait(30)
    # keep sweeping so RELEASED / expired segments the consumer left behind vanish
    deadline = time.time() + 10
    edir = os.path.join(shm_dir, P_ENGINE)
    while time.time() < deadline and any(
        not n_.endswith(".keep") for n_ in os.listdir(edir)
    ):
        time.sleep(0.02)
    P.shutdown()


@pytest.mark.parametrize("mode", ["dumpfile", "raw"])
def test_concurrent_producer_consumer_processes(tmp_path, mode):
    # spawn, not fork: torch (OpenMP) deadlocks in a fork child of a torch-using parent
    ctx = mp.get_context("spawn")
    shm_dir = str(tmp_path / "shm")
    N = 6
    ready_ev, done_ev = ctx.Event(), ctx.Event()
    proc = ctx.Process(
        target=_producer_proc, args=(shm_dir, mode, N, 2.0, ready_ev, done_ev)
    )
    proc.start()
    layer = FakeDeviceLayer()
    D = ShmTransport(
        engine_id=D_ENGINE,
        shm_dir=shm_dir,
        mode=mode,
        role="consumer",
        device_layer=layer,
        janitor_period=0,
    )
    got = {}
    deadline = time.time() + 30
    pending = {i: f"{P_ENGINE}:{i:032x}" for i in range(N)}
    while pending and time.time() < deadline:
        for i, xid in list(pending.items()):
            g = D.open_get(Desc(xid))
            if g is None:
                continue  # WRITING (or not yet created)
            if g.status == "MISSING":
                continue  # not created yet
            assert g.ready(), g.reason
            if i == N - 1:  # exercise release without reading on the last one
                D.release_remote(xid)
                got[i] = "released"
                del pending[i]
                continue
            for name, s in g.sources.items():
                if s.spec.kind == "gdn_taps":
                    assert torch.equal(s.read_rows(), rows_tensor(i))
                    continue
                for c in range(s.spec.nchunks):
                    ch = s.chunk(c)
                    if mode == "dumpfile":
                        data = ch.read_device(None).data
                    else:
                        shape = (1, 4, 2048, 256) if ch.is_head_major else s.spec.shape
                        st = FakeTensor(
                            shape,
                            s.spec.dtype,
                            s.spec.layout,
                            np.zeros(s.spec.chunk_nbytes, np.uint8),
                            True,
                        )
                        ch.read_into_device(st)
                        data = st.data
                    assert set(np.unique(data)) <= {(i + c) % 251, 0}
                    assert data[0] == (i + c) % 251
            D.finish_import(g, ok=True)
            got[i] = "ok"
            del pending[i]
        time.sleep(0.005)
    assert not pending, f"never saw {pending}"
    assert ready_ev.wait(10)
    done_ev.set()
    proc.join(30)
    assert proc.exitcode == 0
    edir = tmp_path / "shm" / P_ENGINE
    # consumed segments unlinked by D, the released one swept by P's janitor
    assert sorted(p.name for p in edir.iterdir()) == []
    D.shutdown()


# --- [fix] PD polish: the two audited minors ------------------------------------


def test_dumpfile_truncated_file_is_not_present(tmp_path):
    """Audit minor: design 5.4 says "exists + size". The producer records every
    data file's byte length at finish_export; a file that later lost bytes is
    ABSENT for ``nbytes_present``, so ``validate_gdn_parts`` fails the load
    before ``finished_recving`` (recompute) instead of ``ttnn.load_tensor``
    raising inside the fatal join-step install."""
    rng = np.random.default_rng(3)
    P = make(tmp_path, P_ENGINE, role="producer")
    D = make(tmp_path, D_ENGINE, role="consumer")
    m = small_manifest(64)
    h = P.open_put(xfer(), m)
    fill_all(h, rng, P.layer, "dumpfile")
    P.finish_export(h, "READY")
    pub = tmp_path / "shm" / P_ENGINE / XFER_HEX
    sizes = json.loads((pub / shm.SIZES_FILE).read_text())
    rec_file, kv_file = "gdn.L0.rec.c0.tensorbin", "kv.L0.k.c0.tensorbin"
    assert {rec_file, kv_file, "gdn.L0.taps.rows.pt"} <= set(sizes)
    assert sizes[rec_file] == (pub / rec_file).stat().st_size > 0
    assert "header" not in sizes and shm.SIZES_FILE not in sizes

    full = (pub / rec_file).read_bytes()
    (pub / rec_file).write_bytes(full[: len(full) // 2])  # truncated after READY
    g = D.open_get(Desc(xfer()))
    assert g.ready()
    rec, kv = g.sources["gdn.L0.rec"], g.sources["kv.L0.k"]
    assert rec.nbytes_present == 0, "truncated -> absent"
    assert kv.nbytes_present == kv.spec.chunk_nbytes * kv.spec.nchunks
    taps = g.sources["gdn.L0.taps"]
    assert taps.nbytes_present == taps.spec.nbytes
    D.finish_import(g, ok=False)

    # Legacy segment without a size record: the old exists-and-non-empty rule.
    h = P.open_put(xfer(hx=XFER_HEX2), m)
    fill_all(h, rng, P.layer, "dumpfile")
    P.finish_export(h, "READY")
    pub2 = tmp_path / "shm" / P_ENGINE / XFER_HEX2
    (pub2 / shm.SIZES_FILE).unlink()
    (pub2 / rec_file).write_bytes((pub2 / rec_file).read_bytes()[:10])
    g2 = D.open_get(Desc(h.xfer_id))
    assert g2.sources["gdn.L0.rec"].nbytes_present == rec.spec.chunk_nbytes
    D.finish_import(g2, ok=False)


def test_janitor_renames_before_rmtree_so_a_racing_claim_misses(tmp_path, monkeypatch):
    """Audit minor: ``shutil.rmtree`` is dir-fd based, so a consumer claim
    rename landing mid-sweep used to leave the consumer owning a directory
    whose files were being deleted. The janitor now renames the segment to
    ``{hx}.expired-{pid}`` first (atomic): a claim that comes later finds no
    segment (MISSING -> recompute); a claim that came first wins."""
    now = [1000.0]
    P = make(tmp_path, P_ENGINE, lease_duration=10.0, clock=lambda: now[0])
    D = make(tmp_path, D_ENGINE, clock=lambda: now[0])
    edir = tmp_path / "shm" / P_ENGINE
    h = P.open_put(xfer(), small_manifest(64))
    P.finish_export(h, "READY")
    pub = edir / XFER_HEX

    seen = []
    real_rmtree = shm.ShmTransport._rmtree

    def racing_rmtree(path):
        # The consumer's claim lands while the janitor is deleting.
        g = D.open_get(Desc(xfer()))
        seen.append((os.path.basename(path), g.status, pub.exists()))
        real_rmtree(path)

    P._rmtree = racing_rmtree
    now[0] = 1010.5
    P.janitor_once()
    assert seen == [(f"{XFER_HEX}{shm.EXPIRED_MARK}{os.getpid()}", "MISSING", False)]
    assert P.stats["expired"] == 1 and list(edir.iterdir()) == []
    P._rmtree = real_rmtree

    # The claim came first: the rename fails, nothing is expired, the claim stands.
    h = P.open_put(xfer(hx=XFER_HEX2), small_manifest(64))
    P.finish_export(h, "READY")
    pub2 = edir / XFER_HEX2
    real_rename = os.rename
    claimed = []

    def claim_then_rename(src, dst, *a, **kw):
        if src == str(pub2) and shm.EXPIRED_MARK in dst:
            claimed.append(D.open_get(Desc(h.xfer_id)))  # consumer wins the race
        return real_rename(src, dst, *a, **kw)

    monkeypatch.setattr(shm.os, "rename", claim_then_rename)
    now[0] = 1030.0
    P.janitor_once()
    monkeypatch.undo()
    assert len(claimed) == 1 and claimed[0].ready()
    assert P.stats["expired"] == 1 and P.stats["swept"] == 0
    assert (edir / f"{XFER_HEX2}.claimed-{D_ENGINE}").is_dir()
    D.finish_import(claimed[0], ok=True)

    # A sweep that died between rename and rmtree left an .expired-* dir: swept.
    leftover = edir / f"{XFER_HEX2}{shm.EXPIRED_MARK}4242"
    leftover.mkdir()
    (leftover / "header").write_bytes(b"junk")
    P.janitor_once()
    assert not leftover.exists()
    # Terminal unclaimed segments (RELEASED) take the same rename-first path.
    h = P.open_put(xfer(), small_manifest(64))
    P.finish_export(h, "READY")
    D.release_remote(xfer())
    seen.clear()
    P._rmtree = racing_rmtree
    P.janitor_once()
    assert seen == [(f"{XFER_HEX}{shm.EXPIRED_MARK}{os.getpid()}", "MISSING", False)]
    assert P.stats["swept"] == 1 and list(edir.iterdir()) == []


def test_dumpfile_taps_rows_file_is_compact_for_a_view_of_a_larger_storage(tmp_path):
    """The hook hands the taps sink a slice of its batched [n_layers, K, D] read;
    torch.save would serialize the view's WHOLE 3.75 MiB storage per part (48 x =
    50 ms on the prefill node, py-spy 2026-09-22). The sink writes a compact 80 KiB
    file either way."""
    import os

    import torch

    from vllm_tt_plugin.kv_transfer.metadata import PartSpec
    from vllm_tt_plugin.kv_transfer.transport.shm import DumpfileSink, _PutState

    nb = 4 * 10240 * 2
    spec = PartSpec(
        name="gdn.L3.taps",
        kind="gdn_taps",
        shape=(4, 10240),
        dtype="bfloat16",
        layout="ROW_MAJOR",
        nchunks=1,
        chunk_nbytes=nb,
        chunk_offsets=[0],
        nbytes=nb,
    )
    st = _PutState(
        "p:0", "0" * 32, str(tmp_path), str(tmp_path / "header"), None, 0, 0.0
    )
    sink = DumpfileSink(spec, str(tmp_path), None, st)
    big = torch.randn(48, 4, 10240).to(torch.bfloat16)
    sink.write_rows(big[3])  # a view sharing the 3.75 MiB storage
    path = tmp_path / "gdn.L3.taps.rows.pt"
    size = os.path.getsize(path)
    assert size < 2 * nb, size  # 80 KiB + pickle overhead, not 3.75 MiB
    back = torch.load(path, weights_only=True)
    assert torch.equal(back, big[3]) and back.untyped_storage().nbytes() == nb
    assert st.written == {"gdn.L3.taps": {0}}
