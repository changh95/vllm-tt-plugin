# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""ShmTransport: the v1 shared-memory data plane (PHASE2_DESIGN 4.2, 3.5).

Segment directory ``{shm_dir}/{producer_engine_id}/{xfer_hex}`` where ``xfer_hex``
is the 32-hex-digit half of ``xfer_id`` (I10).  Lifecycle::

    producer  open_put       -> {xfer_hex}.tmp/          status WRITING
              finish_export  -> status READY|FAILED written LAST, then
                                os.rename({xfer_hex}.tmp, {xfer_hex})   (atomic publish)
    consumer  open_get       -> os.rename({xfer_hex}, {xfer_hex}.claimed-{consumer})
              finish_import  -> status CONSUMED|LOAD_FAILED, unmap, unlink (consumer
                                owns a claimed segment)
              release_remote -> claimed: unlink; unclaimed: status RELEASED (the
                                janitor unlinks); missing: return
    janitor   (producer, 100 ms) unlinks unclaimed RELEASED/FAILED segments, unclaimed
              segments past ``lease_expiry_ts``, ``.tmp`` dirs it does not own, and
              ``.claimed-*`` dirs whose consumer pid (``consumer.pid``, stamped at the
              claim) is dead -- never on age alone: a KV_DONE handle is legitimately
              held across an unbounded promotion wait (round-2 critic C2); a claim
              without a pid file (legacy) falls back to the 2 x lease age rule

Two data-plane modes behind the same ``Sink`` / ``SourceChunk``:

``dumpfile`` (bring-up default): one header file plus one ``ttnn.dump_tensor``
file per chunk (``{part}.c{c}.tensorbin``) and ``{part}.rows.pt`` for the taps.
``raw`` (performance path): one POSIX shm file ``data`` (``O_CREAT|O_EXCL``,
``posix_fallocate``, ``mmap``) holding the header, the part table, the manifest
and the data region with every part at a 4 KiB-aligned offset; chunk ``c`` of a
part sits at ``part_offset + chunk_offsets[c]``.  The raw D2H/H2D primitives are
behind ``DeviceLayer``; ``TtnnDeviceLayer`` is the real one and falls back from the
step-5 nanobind byte helpers (not built yet) to
``copy_device_to_host_tensor`` + ``host_bytes`` memcpy.

Raw-mode consumer relayout (design 4.2, qualified per AM4 / critic C2): K/V chunks
are stored BLOCK-major exactly as read from the producer's cache.  ``Source.chunk(c)``
of a ``kv_blocks`` part submits a pure-numpy gather of the tile records of blocks
``[32c, 32c+32)`` into a HEAD-major ``[1, 4, 2048, 256]`` host buffer to a helper
thread pool (and prefetches chunk ``c+1``); ``SourceChunk.read_into_device`` blocks
on that future, so the handoff is a per-(part, chunk) ``concurrent.futures.Future``.
This MATERIALISES one head-major host chunk buffer per (part, chunk) that is
read (2.2 MB bfp8 / 4.2 MB bf16 each, released when the chunk has been read into
the device), i.e. raw mode is "no extra copy" for the producer and for the rec /
taps parts only; the K/V import costs one host pass per chunk in a helper thread.

Deviation from the doc text: the manifest is stored as JSON, not msgpack
(``msgpack`` is not installed in the serving venv; the manifest is ~30 KB and
read once per transfer).  ``crc32c`` uses the ``crc32c`` module when present and
a pure-Python table otherwise (debug flag only).
"""

from __future__ import annotations

import contextlib
import json
import logging
import mmap
import os
import re
import shutil
import struct
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .base import (
    ALIGN,
    BLOCKS_PER_CHUNK,
    HEAD_MAJOR_KV_SHAPE_HINT,
    KV_HEAD_DIM,
    KV_HEADS,
    LAYOUT_VERSION,
    TAPS_SHAPE,
    TILE,
    TILE_RECORD_BYTES,
    GetHandle,
    Manifest,
    PartSpec,
    PutHandle,
    Sink,
    Source,
    SourceChunk,
    TTKVTransport,
    coerce_manifest,
)

try:  # vLLM's logger tree when the plugin is installed; plain logging otherwise
    from vllm_tt_plugin.logger import init_tt_logger

    logger = init_tt_logger(__name__)
except Exception:  # pragma: no cover
    logger = logging.getLogger(__name__)

# --- header -----------------------------------------------------------------------

WRITING, READY, FAILED, CONSUMED, LOAD_FAILED, RELEASED = 1, 2, 3, 4, 5, 6
STATUS_NAMES = {
    WRITING: "WRITING",
    READY: "READY",
    FAILED: "FAILED",
    CONSUMED: "CONSUMED",
    LOAD_FAILED: "LOAD_FAILED",
    RELEASED: "RELEASED",
}
TERMINAL_UNCLAIMED = {FAILED, RELEASED, CONSUMED, LOAD_FAILED}

HEADER_MAGIC = b"TTPD"
HEADER_VERSION = 1
HEADER_FIXED = 4096  # the fixed header occupies the first 4 KiB
FLAG_RAW = 1  # header.flags bit: raw mode segment (else dumpfile)
CHECKSUM_CRC32C = 1  # header.checksum_flags bit

# magic, version, status, flags, producer_engine_id, xfer_hex, num_tokens, nblk,
# block_size, chunk_tokens, nparts, kv_dtype, rec_dtype, model_sig, prompt_hash,
# created_ts, lease_expiry_ts, producer_pid, checksum_flags, table_off, table_len,
# manifest_off, manifest_len, data_off, total_nbytes
_HDR = struct.Struct("<4sIII32s32sQIIII16s16s32s16sddIIIIIIQQ")
STATUS_OFFSET = 8  # magic(4) + version(4)
CONSUMER_PID_FILE = "consumer.pid"  # inside a .claimed-* dir; C2 sweep rule
# dumpfile mode: {file name: byte length} of every data file the producer wrote,
# written at finish_export before the status flip; ``DumpfileSource`` treats a
# file whose length differs as ABSENT (truncated/corrupt -> validate_gdn_parts
# fails BEFORE finished_recving -> recompute, never a fatal join-step install).
SIZES_FILE = "sizes.json"
EXPIRED_MARK = ".expired-"  # janitor: renamed here first, then removed
PRODUCER_PID_OFFSET = struct.calcsize("<4sIII32s32sQIIII16s16s32s16sdd")
assert _HDR.size <= HEADER_FIXED
# name, offset, nbytes, nchunks, chunk_nbytes, spec_json, crc32c, chunks_written
_PART = struct.Struct("<64sQQIQ512sII")

HEAD_MAJOR_KV_SHAPE = HEAD_MAJOR_KV_SHAPE_HINT  # [1,4,2048,256]
_XFER_RE = re.compile(r"^([A-Za-z0-9_-]{1,32}):([0-9a-f]{32})$")


def parse_xfer_id(xfer_id: str) -> tuple[str, str]:
    m = _XFER_RE.match(xfer_id)
    if not m:
        raise ValueError(f"malformed xfer_id {xfer_id!r}")
    return m.group(1), m.group(2)


def _fx(s: str, n: int) -> bytes:
    b = s.encode()
    if len(b) > n:
        b = b[:n]
    return b


def _unfx(b: bytes) -> str:
    return b.rstrip(b"\0").decode(errors="replace")


def _align(n: int, a: int = ALIGN) -> int:
    return -(-n // a) * a


@dataclass
class PartRecord:
    name: str
    offset: int  # absolute offset inside the raw ``data`` file (0 in dumpfile mode)
    nbytes: int
    nchunks: int
    chunk_nbytes: int
    spec: PartSpec
    crc32c: int = 0
    chunks_written: int = 0

    def pack(self) -> bytes:
        d = self.spec.to_dict()
        d.pop("chunk_offsets", None)  # deterministic: c * chunk_nbytes
        sj = json.dumps(d, separators=(",", ":")).encode()
        if len(sj) > 512:
            raise ValueError("PartSpec json exceeds 512 bytes")
        return _PART.pack(
            _fx(self.name, 64),
            self.offset,
            self.nbytes,
            self.nchunks,
            self.chunk_nbytes,
            sj,
            self.crc32c,
            self.chunks_written,
        )

    @classmethod
    def unpack(cls, raw: bytes) -> PartRecord:
        name, off, nb, nch, cnb, sj, crc, cw = _PART.unpack(raw)
        spec = PartSpec.from_dict(json.loads(_unfx(sj)))
        return cls(_unfx(name), off, nb, nch, cnb, spec, crc, cw)


@dataclass
class Header:
    status: int
    flags: int
    producer_engine_id: str
    xfer_hex: str
    num_tokens: int
    nblk: int
    block_size: int
    chunk_tokens: int
    nparts: int
    kv_dtype: str
    rec_dtype: str
    model_sig: str
    prompt_hash: str
    created_ts: float
    lease_expiry_ts: float
    producer_pid: int
    checksum_flags: int
    table_off: int
    table_len: int
    manifest_off: int
    manifest_len: int
    data_off: int
    total_nbytes: int
    version: int = HEADER_VERSION
    parts: list[PartRecord] = field(default_factory=list)
    manifest: Manifest | None = None

    @property
    def mode(self) -> str:
        return "raw" if self.flags & FLAG_RAW else "dumpfile"

    def pack_fixed(self) -> bytes:
        b = _HDR.pack(
            HEADER_MAGIC,
            self.version,
            self.status,
            self.flags,
            _fx(self.producer_engine_id, 32),
            _fx(self.xfer_hex, 32),
            self.num_tokens,
            self.nblk,
            self.block_size,
            self.chunk_tokens,
            self.nparts,
            _fx(self.kv_dtype, 16),
            _fx(self.rec_dtype, 16),
            _fx(self.model_sig, 32),
            _fx(self.prompt_hash, 16),
            self.created_ts,
            self.lease_expiry_ts,
            self.producer_pid,
            self.checksum_flags,
            self.table_off,
            self.table_len,
            self.manifest_off,
            self.manifest_len,
            self.data_off,
            self.total_nbytes,
        )
        return b.ljust(HEADER_FIXED, b"\0")

    def pack_table(self) -> bytes:
        return b"".join(p.pack() for p in self.parts)

    @classmethod
    def unpack_fixed(cls, raw: bytes) -> Header:
        if len(raw) < _HDR.size:
            raise ValueError("short header")
        f = _HDR.unpack_from(raw, 0)
        if f[0] != HEADER_MAGIC:
            raise ValueError("bad magic")
        return cls(
            version=f[1],
            status=f[2],
            flags=f[3],
            producer_engine_id=_unfx(f[4]),
            xfer_hex=_unfx(f[5]),
            num_tokens=f[6],
            nblk=f[7],
            block_size=f[8],
            chunk_tokens=f[9],
            nparts=f[10],
            kv_dtype=_unfx(f[11]),
            rec_dtype=_unfx(f[12]),
            model_sig=_unfx(f[13]),
            prompt_hash=_unfx(f[14]),
            created_ts=f[15],
            lease_expiry_ts=f[16],
            producer_pid=f[17],
            checksum_flags=f[18],
            table_off=f[19],
            table_len=f[20],
            manifest_off=f[21],
            manifest_len=f[22],
            data_off=f[23],
            total_nbytes=f[24],
        )


def segment_layout(manifest: Manifest, mode: str) -> tuple[list[PartRecord], Header]:
    """Deterministic file layout: fixed header, part table, JSON manifest, data."""
    nparts = len(manifest.parts)
    table_off = HEADER_FIXED
    table_len = nparts * _PART.size
    mjson = manifest.to_json()
    manifest_off = table_off + table_len
    data_off = _align(manifest_off + len(mjson))
    records: list[PartRecord] = []
    off = data_off
    for p in manifest.parts:
        records.append(
            PartRecord(
                p.name,
                off if mode == "raw" else 0,
                p.nbytes,
                p.nchunks,
                p.chunk_nbytes,
                p,
            )
        )
        off += _align(p.nbytes)
    total = off - data_off
    hdr = Header(
        status=WRITING,
        flags=FLAG_RAW if mode == "raw" else 0,
        producer_engine_id="",
        xfer_hex="",
        num_tokens=manifest.num_tokens,
        nblk=manifest.nblk,
        block_size=manifest.block_size,
        chunk_tokens=manifest.chunk_tokens,
        nparts=nparts,
        kv_dtype=manifest.kv_dtype,
        rec_dtype=manifest.rec_dtype,
        model_sig=manifest.model_sig,
        prompt_hash=manifest.prompt_hash,
        created_ts=0.0,
        lease_expiry_ts=0.0,
        producer_pid=os.getpid(),
        checksum_flags=0,
        table_off=table_off,
        table_len=table_len,
        manifest_off=manifest_off,
        manifest_len=len(mjson),
        data_off=data_off,
        total_nbytes=total,
        parts=records,
        manifest=manifest,
    )
    return records, hdr


def read_header(path: str, *, full: bool = True) -> Header:
    """Read the header (and, with ``full``, the part table + manifest) of a segment."""
    with open(path, "rb") as f:
        hdr = Header.unpack_fixed(f.read(HEADER_FIXED))
        if full:
            f.seek(hdr.table_off)
            table = f.read(hdr.table_len)
            if len(table) != hdr.table_len:
                raise ValueError("short part table")
            hdr.parts = [
                PartRecord.unpack(table[i : i + _PART.size])
                for i in range(0, hdr.table_len, _PART.size)
            ]
            f.seek(hdr.manifest_off)
            mj = f.read(hdr.manifest_len)
            if len(mj) != hdr.manifest_len:
                raise ValueError("short manifest")
            hdr.manifest = Manifest.from_json(mj)
    return hdr


def read_status(path: str) -> int:
    with open(path, "rb") as f:
        f.seek(STATUS_OFFSET)
        raw = f.read(4)
    if len(raw) != 4:
        raise ValueError("short header")
    return struct.unpack("<I", raw)[0]


def write_status(path: str, status: int) -> None:
    """The ONE 4-byte write the state machine turns on (atomic on tmpfs)."""
    fd = os.open(path, os.O_WRONLY)
    try:
        os.pwrite(fd, struct.pack("<I", status), STATUS_OFFSET)
    finally:
        os.close(fd)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# --- crc32c -------------------------------------------------------------------------

try:  # pragma: no cover - optional accelerator
    import crc32c as _crc32c_mod
except Exception:  # pragma: no cover
    _crc32c_mod = None

_CRC_TABLE: list[int] | None = None


def _crc_table() -> list[int]:
    global _CRC_TABLE
    if _CRC_TABLE is None:
        tbl = []
        for i in range(256):
            c = i
            for _ in range(8):
                c = (c >> 1) ^ 0x82F63B78 if c & 1 else c >> 1
            tbl.append(c)
        _CRC_TABLE = tbl
    return _CRC_TABLE


def crc32c(data: Any, crc: int = 0) -> int:
    """CRC-32C (Castagnoli) of ``data`` (bytes / memoryview / uint8 ndarray)."""
    mv = (
        memoryview(np.ascontiguousarray(data).view(np.uint8))
        if isinstance(data, np.ndarray)
        else memoryview(data).cast("B")
    )
    if _crc32c_mod is not None:  # pragma: no cover
        return _crc32c_mod.crc32c(mv, crc)
    tbl = _crc_table()
    c = crc ^ 0xFFFFFFFF
    for b in mv.tobytes():
        c = tbl[(c ^ b) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


# --- head-major relayout (raw mode, host-only numpy) ------------------------------


def relayout_block_major_to_head_major(
    chunk: np.ndarray, rec_bytes: int, nvalid: int = BLOCKS_PER_CHUNK
) -> np.ndarray:
    """Gather the tile records of one block-major K/V chunk head-major.

    Block-major ``[32, 4, 64, 256]`` TILE: tile ``((b*4 + h)*2 + r)*8 + col``.
    Head-major ``[1, 4, 2048, 256]`` TILE: tile ``(h*64 + b*2 + r)*8 + col``.
    Tile records are ``rec_bytes`` each (1088 bfp8_b / 2048 bf16).  Blocks with index
    ``>= nvalid`` are zero-filled.  Pure numpy; returns a fresh contiguous uint8 array.
    """
    n, h, r, col = BLOCKS_PER_CHUNK, KV_HEADS, 64 // TILE, KV_HEAD_DIM // TILE
    src = np.asarray(chunk).view(np.uint8)
    if src.size != n * h * r * col * rec_bytes:
        raise ValueError(
            f"chunk of {src.size} B is not a [32,4,64,256] chunk of "
            f"{rec_bytes}-B tile records"
        )
    v = src.reshape(n, h, r, col, rec_bytes)
    out = np.ascontiguousarray(v.transpose(1, 0, 2, 3, 4))  # [h, n, r, col, rec]
    if nvalid < n:
        out[:, nvalid:] = 0
    return out.reshape(-1)


# --- real ttnn adapter ------------------------------------------------------------


class TtnnDeviceLayer:
    """``DeviceLayer`` over ttnn (imported lazily; never at module import)."""

    def __init__(self) -> None:
        import ttnn  # noqa: PLC0415 - lazy by design (design 3: no ttnn in API server)

        self.ttnn = ttnn

    @staticmethod
    def _shape(t: Any) -> tuple[int, ...]:
        return tuple(int(x) for x in t.shape)

    def spec_of(self, t: Any) -> tuple[tuple[int, ...], str, str]:
        return (self._shape(t), t.dtype.name.lower(), t.layout.name.upper())

    def from_device(self, t: Any) -> Any:
        return self.ttnn.from_device(t)

    def dump_tensor(self, path: str, h: Any) -> None:
        self.ttnn.dump_tensor(path, h, mode=self.ttnn.DumpTensorMode.LOCAL)

    def load_tensor(self, path: str, device: Any) -> Any:
        return self.ttnn.load_tensor(path, device=device)

    def allocate_host_like(self, t: Any) -> Any:
        return self.ttnn.allocate_tensor_on_host(t.spec, t.device())

    def allocate_host(self, spec: PartSpec, device: Any) -> Any:
        ttnn = self.ttnn
        return ttnn.allocate_tensor_on_host(
            ttnn.Shape(list(spec.shape)),
            getattr(ttnn.DataType, spec.dtype.upper()),
            getattr(ttnn.Layout, spec.layout.upper()),
            device,
        )

    def _cq(self, cq_id: int | None) -> Any:
        return None if cq_id is None else self.ttnn.QueueId(cq_id)

    def copy_device_to_host(
        self, d: Any, h: Any, *, blocking: bool, cq_id: int | None
    ) -> None:
        self.ttnn.copy_device_to_host_tensor(
            d, h, blocking=blocking, cq_id=self._cq(cq_id)
        )

    def copy_host_to_device(self, h: Any, d: Any, *, cq_id: int | None) -> None:
        self.ttnn.copy_host_to_device_tensor(h, d, cq_id=self._cq(cq_id))

    def host_bytes(self, h: Any) -> np.ndarray:
        fn = getattr(h, "host_bytes", None)  # design 8 step 5 nanobind helper
        if fn is None:
            raise NotImplementedError(
                "ttnn.Tensor.host_bytes is not bound in this tt-metal build; raw shm "
                "mode needs the step-5 nanobind helpers (design 8); use "
                "shm_mode=dumpfile until they are built"
            )
        return np.asarray(fn())

    def read_tensor_bytes(
        self, t: Any, dst: memoryview, src_offset: int, nbytes: int, *, blocking: bool
    ) -> None:
        fn = getattr(self.ttnn.experimental, "read_tensor_bytes", None)
        if fn is None:
            raise NotImplementedError("ttnn.experimental.read_tensor_bytes not built")
        arr = np.frombuffer(dst, dtype=np.uint8)
        fn(t, arr.ctypes.data, src_offset, nbytes, blocking)


# --- sinks ----------------------------------------------------------------------------


def _check_spec(
    layer: Any,
    tensor: Any,
    spec: PartSpec,
    shape_ok: tuple[tuple[int, ...], ...] | None = None,
) -> None:
    shape, dtype, layout = layer.spec_of(tensor)
    shapes = shape_ok or (spec.shape,)
    if tuple(shape) not in shapes or dtype != spec.dtype or layout != spec.layout:
        raise ValueError(
            f"tensor ({shape}, {dtype}, {layout}) does not match part "
            f"{spec.name} spec ({shapes}, {spec.dtype}, {spec.layout})"
        )


def _check_rows(rows: Any, spec: PartSpec) -> None:
    import torch  # noqa: PLC0415 - rows are torch by contract

    if tuple(rows.shape) != TAPS_SHAPE or rows.dtype != torch.bfloat16:
        raise ValueError(
            f"taps rows must be {TAPS_SHAPE} bf16, got {tuple(rows.shape)} {rows.dtype}"
        )


def _rows_to_u8(rows: Any) -> np.ndarray:
    import torch  # noqa: PLC0415

    return rows.contiguous().cpu().view(torch.int16).numpy().view(np.uint8).reshape(-1)


def _u8_to_rows(buf: np.ndarray) -> Any:
    import torch  # noqa: PLC0415

    return (
        torch.from_numpy(np.array(buf, dtype=np.uint8).view(np.int16))
        .view(torch.bfloat16)
        .reshape(TAPS_SHAPE)
        .clone()
    )


class _PutState:
    def __init__(
        self,
        xfer_id: str,
        xfer_hex: str,
        tmp_dir: str,
        header_path: str,
        hdr: Header,
        nbytes: int,
        created_ts: float,
    ) -> None:
        self.xfer_id, self.xfer_hex = xfer_id, xfer_hex
        self.tmp_dir, self.header_path, self.hdr = tmp_dir, header_path, hdr
        self.nbytes, self.created_ts = nbytes, created_ts
        self.written: dict[str, set[int]] = {}
        self.fd: int | None = None
        self.mm: mmap.mmap | None = None
        self.published = False
        self.lock = threading.Lock()

    def note(self, part: str, chunk: int) -> None:
        with self.lock:
            self.written.setdefault(part, set()).add(chunk)

    def close(self) -> None:
        if self.mm is not None:
            with contextlib.suppress(Exception):  # pragma: no cover
                self.mm.flush()
            self.mm.close()
            self.mm = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class DumpfileSink(Sink):
    supports_regions = False

    def __init__(self, spec: PartSpec, seg_dir: str, layer: Any, st: _PutState) -> None:
        self.spec, self._dir, self._layer, self._st = spec, seg_dir, layer, st

    def _path(self, chunk: int) -> str:
        if not 0 <= chunk < self.spec.nchunks:
            raise IndexError(f"chunk {chunk} of {self.spec.nchunks}")
        return os.path.join(self._dir, f"{self.spec.name}.c{chunk}.tensorbin")

    def write_from_device(
        self,
        device_tensor: Any,
        *,
        chunk: int,
        blocking: bool = True,
        cq_id: int | None = None,
    ) -> None:
        _check_spec(self._layer, device_tensor, self.spec)
        h = self._layer.from_device(device_tensor)  # inherently blocking
        self._layer.dump_tensor(self._path(chunk), h)
        self._st.note(self.spec.name, chunk)

    def write_region_from_device(
        self,
        device_tensor: Any,
        src_offset_bytes: int,
        nbytes: int,
        *,
        chunk: int,
        dst_offset_bytes: int,
        blocking: bool = True,
    ) -> None:
        raise NotImplementedError(
            "dumpfile mode has no byte regions (supports_regions)"
        )

    def write_rows(self, rows: Any) -> None:
        import torch  # noqa: PLC0415

        _check_rows(rows, self.spec)
        rows = rows.detach().contiguous().cpu()
        if rows.untyped_storage().nbytes() != rows.numel() * rows.element_size():
            # a view of a larger storage (the hook's batched taps read): torch.save
            # serializes the WHOLE storage, 3.75 MiB instead of 80 KiB per part
            rows = rows.clone()
        torch.save(rows, os.path.join(self._dir, f"{self.spec.name}.rows.pt"))
        self._st.note(self.spec.name, 0)

    def write_host(self, host_tensor: Any, *, chunk: int) -> None:
        _check_spec(self._layer, host_tensor, self.spec)
        self._layer.dump_tensor(self._path(chunk), host_tensor)
        self._st.note(self.spec.name, chunk)


class _HostCache:
    """Preallocated host tensors keyed by spec (raw-mode fallback path)."""

    def __init__(self, layer: Any) -> None:
        self._layer = layer
        self._t: dict[Any, Any] = {}
        self._lock = threading.Lock()

    def like(self, tensor: Any) -> Any:
        key = self._layer.spec_of(tensor)
        with self._lock:
            h = self._t.get(key)
            if h is None:
                h = self._t[key] = self._layer.allocate_host_like(tensor)
        return h


class RawSink(Sink):
    supports_regions = True

    def __init__(
        self,
        spec: PartSpec,
        mm: mmap.mmap,
        part_off: int,
        layer: Any,
        hosts: _HostCache,
        st: _PutState,
    ) -> None:
        self.spec, self._mm, self._off = spec, mm, part_off
        self._layer, self._hosts, self._st = layer, hosts, st

    def _region(self, start: int, nbytes: int) -> memoryview:
        if start < 0 or start + nbytes > self.spec.nbytes:
            raise ValueError(
                f"region [{start}, {start + nbytes}) outside part "
                f"{self.spec.name} ({self.spec.nbytes} B)"
            )
        a = self._off + start
        return memoryview(self._mm)[
            a : a + nbytes
        ]  # short-lived: consumed in the caller

    def _d2h_bytes(
        self,
        t: Any,
        src_off: int,
        nbytes: int,
        dst: memoryview,
        blocking: bool,
        cq_id: int | None,
    ) -> None:
        try:
            self._layer.read_tensor_bytes(t, dst, src_off, nbytes, blocking=blocking)
            return
        except NotImplementedError:
            pass
        # Fallback (design 4.2): whole-tensor D2H into a preallocated host tensor,
        # then one host memcpy of the range.  Must be blocking: the memcpy reads it.
        h = self._hosts.like(t)
        self._layer.copy_device_to_host(t, h, blocking=True, cq_id=cq_id)
        hb = np.asarray(self._layer.host_bytes(h)).view(np.uint8).reshape(-1)
        if src_off + nbytes > hb.size:
            raise ValueError(
                f"source range [{src_off}, {src_off + nbytes}) outside "
                f"tensor of {hb.size} B"
            )
        np.frombuffer(dst, dtype=np.uint8)[:] = hb[src_off : src_off + nbytes]

    def write_from_device(
        self,
        device_tensor: Any,
        *,
        chunk: int,
        blocking: bool = True,
        cq_id: int | None = None,
    ) -> None:
        _check_spec(self._layer, device_tensor, self.spec)
        if not 0 <= chunk < self.spec.nchunks:
            raise IndexError(f"chunk {chunk} of {self.spec.nchunks}")
        dst = self._region(self.spec.chunk_offsets[chunk], self.spec.chunk_nbytes)
        self._d2h_bytes(device_tensor, 0, self.spec.chunk_nbytes, dst, blocking, cq_id)
        self._st.note(self.spec.name, chunk)

    def write_region_from_device(
        self,
        device_tensor: Any,
        src_offset_bytes: int,
        nbytes: int,
        *,
        chunk: int,
        dst_offset_bytes: int,
        blocking: bool = True,
    ) -> None:
        if not 0 <= chunk < self.spec.nchunks:
            raise IndexError(f"chunk {chunk} of {self.spec.nchunks}")
        if nbytes % ALIGN or src_offset_bytes % ALIGN or dst_offset_bytes % ALIGN:
            raise ValueError("region offsets/length must be 4 KiB multiples (BLK, ROW)")
        if dst_offset_bytes + nbytes > self.spec.chunk_nbytes:
            raise ValueError("region does not fit inside the chunk")
        dst = self._region(self.spec.chunk_offsets[chunk] + dst_offset_bytes, nbytes)
        self._d2h_bytes(device_tensor, src_offset_bytes, nbytes, dst, blocking, None)
        self._st.note(self.spec.name, chunk)

    def write_rows(self, rows: Any) -> None:
        _check_rows(rows, self.spec)
        dst = self._region(self.spec.chunk_offsets[0], self.spec.chunk_nbytes)
        np.frombuffer(dst, dtype=np.uint8)[:] = _rows_to_u8(rows)
        self._st.note(self.spec.name, 0)

    def write_host(self, host_tensor: Any, *, chunk: int) -> None:
        _check_spec(self._layer, host_tensor, self.spec)
        dst = self._region(self.spec.chunk_offsets[chunk], self.spec.chunk_nbytes)
        hb = np.asarray(self._layer.host_bytes(host_tensor)).view(np.uint8).reshape(-1)
        np.frombuffer(dst, dtype=np.uint8)[:] = hb[: self.spec.chunk_nbytes]
        self._st.note(self.spec.name, chunk)


# --- sources ------------------------------------------------------------------


class DumpfileChunk(SourceChunk):
    is_head_major = False
    is_device_readable = False

    def __init__(self, path: str, spec: PartSpec, layer: Any) -> None:
        self._path, self.spec, self._layer = path, spec, layer

    def read_into_device(
        self, staging_tensor: Any, *, cq_id: int | None = None
    ) -> None:
        raise NotImplementedError("dumpfile chunks are read with read_device")

    def read_device(self, mesh_device: Any) -> Any:
        t = self._layer.load_tensor(self._path, mesh_device)  # spec self-describing
        _check_spec(self._layer, t, self.spec)
        return t


class DumpfileSource(Source):
    def __init__(self, rec: PartRecord, seg_dir: str, layer: Any) -> None:
        self.spec, self._rec, self._dir, self._layer = rec.spec, rec, seg_dir, layer
        self.spec_crc = rec.crc32c
        self._sizes: dict[str, int] | None = None
        self._sizes_loaded = False

    def _recorded_sizes(self) -> dict[str, int] | None:
        """The producer's ``SIZES_FILE`` (None for a legacy segment without one)."""
        if not self._sizes_loaded:
            self._sizes_loaded = True
            try:
                with open(os.path.join(self._dir, SIZES_FILE)) as f:
                    raw = json.load(f)
                self._sizes = {str(k): int(v) for k, v in dict(raw).items()}
            except (OSError, ValueError, TypeError, AttributeError):
                self._sizes = None
        return self._sizes

    def _chunk_path(self, c: int) -> str:
        return os.path.join(self._dir, f"{self.spec.name}.c{c}.tensorbin")

    def _rows_path(self) -> str:
        return os.path.join(self._dir, f"{self.spec.name}.rows.pt")

    def _present_files(self) -> list[str]:
        """Files that exist AND have the byte length the producer recorded
        (design 5.4 "exists + size"); without a size record: exists and non-empty."""
        if self.spec.kind == "gdn_taps":
            paths = [self._rows_path()]
        else:
            paths = [self._chunk_path(c) for c in range(self.spec.nchunks)]
        sizes = self._recorded_sizes()
        present: list[str] = []
        for p in paths:
            if not os.path.isfile(p):
                continue
            n = os.path.getsize(p)
            if n <= 0:
                continue
            want = None if sizes is None else sizes.get(os.path.basename(p))
            if want is not None and n != want:
                continue  # truncated or corrupt: not present
            present.append(p)
        return present

    @property
    def nbytes_present(self) -> int:
        # dumpfile: "file exists + size" (design 5.4) -> payload bytes of the chunks the
        # producer completed (the flatbuffer file itself carries metadata overhead)
        n = min(len(self._present_files()), self._rec.chunks_written)
        return n * self.spec.chunk_nbytes

    def crc32c(self) -> int:
        c = 0
        for p in self._present_files():
            with open(p, "rb") as f:
                c = crc32c(f.read(), c)
        return c

    def chunk(self, c: int) -> SourceChunk:
        if not 0 <= c < self.spec.nchunks:
            raise IndexError(f"chunk {c} of {self.spec.nchunks}")
        return DumpfileChunk(self._chunk_path(c), self.spec, self._layer)

    def read_rows(self) -> Any:
        import torch  # noqa: PLC0415

        rows = torch.load(self._rows_path(), weights_only=True)
        _check_rows(rows, self.spec)
        return rows


class RawChunk(SourceChunk):
    is_device_readable = True

    def __init__(
        self,
        spec: PartSpec,
        data: Callable[[], np.ndarray] | Future,
        head_major: bool,
        layer: Any,
        hosts: _HostCache,
        release: Callable[[], None] | None = None,
    ) -> None:
        self.spec, self._data, self.is_head_major = spec, data, head_major
        self._layer, self._hosts, self._release = layer, hosts, release

    def _bytes(self) -> np.ndarray:
        if isinstance(self._data, Future):
            return self._data.result()
        return self._data()  # a fresh view of the mapping; nothing exported at rest

    def read_into_device(
        self, staging_tensor: Any, *, cq_id: int | None = None
    ) -> None:
        shapes = (HEAD_MAJOR_KV_SHAPE,) if self.is_head_major else (self.spec.shape,)
        _check_spec(self._layer, staging_tensor, self.spec, shapes)
        h = self._hosts.like(staging_tensor)
        hb = np.asarray(self._layer.host_bytes(h)).view(np.uint8).reshape(-1)
        src = self._bytes()
        if hb.size != src.size:
            raise ValueError(f"staging tensor is {hb.size} B, chunk is {src.size} B")
        hb[:] = src
        del src
        self._layer.copy_host_to_device(h, staging_tensor, cq_id=cq_id)
        if self._release is not None:
            self._release()  # drop the materialised head-major buffer (raw mode)
            self._release = None

    def read_device(self, mesh_device: Any) -> Any:
        raise NotImplementedError("raw chunks are read with read_into_device")


class RawSource(Source):
    def __init__(
        self,
        rec: PartRecord,
        mm: mmap.mmap,
        nblk: int,
        layer: Any,
        hosts: _HostCache,
        pool: ThreadPoolExecutor | None,
    ) -> None:
        self.spec, self._rec, self._mm, self._nblk = rec.spec, rec, mm, nblk
        self._layer, self._hosts, self._pool = layer, hosts, pool
        self.spec_crc = rec.crc32c
        self._futures: dict[int, Future] = {}
        self._lock = threading.Lock()

    @property
    def nbytes_present(self) -> int:
        end = min(self._rec.offset + self.spec.nbytes, len(self._mm))
        return max(0, end - self._rec.offset)

    def _region(self, c: int) -> np.ndarray:
        a = self._rec.offset + self.spec.chunk_offsets[c]
        return np.frombuffer(
            self._mm, dtype=np.uint8, count=self.spec.chunk_nbytes, offset=a
        )

    def crc32c(self) -> int:
        return crc32c(
            np.frombuffer(
                self._mm,
                dtype=np.uint8,
                count=self.nbytes_present,
                offset=self._rec.offset,
            )
        )

    def _relayout(self, c: int) -> np.ndarray:
        nvalid = max(0, min(BLOCKS_PER_CHUNK, self._nblk - c * BLOCKS_PER_CHUNK))
        return relayout_block_major_to_head_major(
            self._region(c), TILE_RECORD_BYTES[self.spec.dtype], nvalid
        )

    def _drop(self, c: int) -> None:
        with self._lock:
            self._futures.pop(c, None)

    def _submit(self, c: int) -> Future:
        with self._lock:
            f = self._futures.get(c)
            if f is None:
                if self._pool is None:
                    f = Future()
                    try:
                        f.set_result(self._relayout(c))
                    except Exception as e:  # pragma: no cover
                        f.set_exception(e)
                else:
                    f = self._pool.submit(self._relayout, c)
                self._futures[c] = f
        return f

    def chunk(self, c: int) -> SourceChunk:
        if not 0 <= c < self.spec.nchunks:
            raise IndexError(f"chunk {c} of {self.spec.nchunks}")
        if self.spec.kind != "kv_blocks":
            return RawChunk(
                self.spec, lambda: self._region(c), False, self._layer, self._hosts
            )
        f = self._submit(c)
        if c + 1 < self.spec.nchunks:
            self._submit(c + 1)  # prefetch: the helper relays c+1 while c is imported
        return RawChunk(
            self.spec, f, True, self._layer, self._hosts, release=lambda: self._drop(c)
        )

    def read_rows(self) -> Any:
        return _u8_to_rows(self._region(0))

    def close(self) -> None:
        with self._lock:
            futs, self._futures = list(self._futures.values()), {}
        for f in futs:
            if not f.cancel():  # running: let it finish so the mapping can close
                with contextlib.suppress(Exception):  # pragma: no cover
                    f.result(timeout=10.0)


class _GetState:
    def __init__(
        self, xfer_id: str, claim_dir: str, header_path: str, hdr: Header
    ) -> None:
        self.xfer_id, self.claim_dir, self.header_path, self.hdr = (
            xfer_id,
            claim_dir,
            header_path,
            hdr,
        )
        self.fd: int | None = None
        self.mm: mmap.mmap | None = None
        self.sources: list[RawSource] = []

    def close(self) -> None:
        for s in self.sources:
            s.close()
        if self.mm is not None:
            try:
                self.mm.close()
            except BufferError:  # a live SourceChunk still views it; GC unmaps later
                logger.warning(
                    "shm mapping of %s still referenced at close", self.xfer_id
                )
            self.mm = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


# --- transport ----------------------------------------------------------------


class ShmTransport(TTKVTransport):
    """Design 4.2 / 3.5.  One instance per connector role; ``role`` picks the janitor.

    ``device_layer`` is any ``DeviceLayer``; ``None`` = ``TtnnDeviceLayer`` created on
    first use (no ttnn import at construction).
    """

    KIND = "shm"

    def __init__(
        self,
        *,
        engine_id: str,
        shm_dir: str = "/dev/shm/tt_pd",
        mode: Literal["dumpfile", "raw"] = "dumpfile",
        budget_bytes: int = 8 << 30,
        lease_duration: float = 30.0,
        checksum: bool = False,
        role: str = "both",
        device_layer: Any | None = None,
        janitor_period: float = 0.1,
        relayout_workers: int = 2,
        free_space_headroom: int = 1 << 30,
        clock: Any = time.time,
    ) -> None:
        if mode not in ("dumpfile", "raw"):
            raise ValueError(f"shm_mode must be dumpfile|raw, got {mode!r}")
        if not re.match(r"^[A-Za-z0-9_-]{1,32}$", engine_id):
            raise ValueError(
                f"engine_id {engine_id!r} must match [A-Za-z0-9_-]{{1,32}}"
            )
        if role not in ("producer", "consumer", "both"):
            raise ValueError(role)
        self.engine_id, self.shm_dir, self.mode = engine_id, shm_dir, mode
        self.budget_bytes, self.lease_duration = (
            int(budget_bytes),
            float(lease_duration),
        )
        self.checksum, self.role = checksum, role
        self.janitor_period, self.free_space_headroom = (
            janitor_period,
            free_space_headroom,
        )
        self._clock = clock
        self._layer = device_layer
        self._hosts: _HostCache | None = None
        self._pool: ThreadPoolExecutor | None = None
        self._relayout_workers = relayout_workers
        self._lock = threading.RLock()
        self._puts: dict[str, _PutState] = {}  # xfer_id -> state (owned segments)
        # xfer_id -> nbytes of the .claimed-* dirs found by the startup sweep
        self._inherited: dict[str, int] = {}
        self._gets: dict[str, _GetState] = {}
        self._stop = threading.Event()
        self._janitor: threading.Thread | None = None
        self.stats = {"expired": 0, "swept": 0, "budget_refusals": 0, "stale_claims": 0}

    # -- helpers
    @property
    def layer(self) -> Any:
        if self._layer is None:
            self._layer = TtnnDeviceLayer()
        return self._layer

    @property
    def hosts(self) -> _HostCache:
        if self._hosts is None:
            self._hosts = _HostCache(self.layer)
        return self._hosts

    @property
    def is_producer(self) -> bool:
        return self.role in ("producer", "both")

    def _engine_dir(self, engine: str | None = None) -> str:
        return os.path.join(self.shm_dir, engine or self.engine_id)

    def _tmp_dir(self, engine: str, hx: str) -> str:
        return os.path.join(self._engine_dir(engine), f"{hx}.tmp")

    def _pub_dir(self, engine: str, hx: str) -> str:
        return os.path.join(self._engine_dir(engine), hx)

    def _claim_dir(self, engine: str, hx: str) -> str:
        return os.path.join(self._engine_dir(engine), f"{hx}.claimed-{self.engine_id}")

    @staticmethod
    def _stamp_consumer_pid(claim_dir: str) -> None:
        """Record the claiming consumer's pid (C2: the producer janitor sweeps a
        claim only when this pid is dead, never on age)."""
        try:
            tmp = os.path.join(claim_dir, CONSUMER_PID_FILE + ".tmp")
            with open(tmp, "w") as f:
                f.write(str(os.getpid()))
            os.replace(tmp, os.path.join(claim_dir, CONSUMER_PID_FILE))
        except OSError:
            logger.warning("could not stamp %s in %s", CONSUMER_PID_FILE, claim_dir)

    def _claim_is_stale(self, claim_dir: str, now: float) -> tuple[bool, str]:
        """A ``.claimed-*`` dir is stale when its consumer pid is dead; without a
        pid file (legacy claim) fall back to the 2 x lease age rule."""
        pid_path = os.path.join(claim_dir, CONSUMER_PID_FILE)
        try:
            with open(pid_path) as f:
                pid = int(f.read().strip() or "0")
        except FileNotFoundError:
            try:
                age = now - os.stat(claim_dir).st_mtime
            except FileNotFoundError:
                return False, ""
            if age > 2 * self.lease_duration:
                return True, f"no consumer pid, {age:.1f}s old"
            return False, ""
        except (OSError, ValueError):
            return False, ""
        if not pid_alive(pid):
            return True, f"consumer pid {pid} is dead"
        return False, ""

    def _header_name(self, mode: str | None = None) -> str:
        return "data" if (mode or self.mode) == "raw" else "header"

    def _find_header(self, seg_dir: str) -> str | None:
        for name in ("data", "header"):
            p = os.path.join(seg_dir, name)
            if os.path.isfile(p):
                return p
        return None

    @staticmethod
    def _rmtree(path: str) -> None:
        shutil.rmtree(path, ignore_errors=True)

    def descriptor(self) -> dict[str, Any]:
        return {"kind": self.KIND, "mode": self.mode, "layout_version": LAYOUT_VERSION}

    def outstanding_bytes(self) -> int:
        with self._lock:
            return sum(s.nbytes for s in self._puts.values()) + sum(
                self._inherited.values()
            )

    def _segment_charge(self, hdr: Header, manifest: Manifest) -> int:
        """Bytes ``open_put`` charges against ``budget_bytes`` / the tmpfs free space
        for one segment: the whole file here (header + every part).  The fabric
        control plane (``fabric._ControlSegments``) overrides this to charge only
        what it really stores (header + taps rows), its K/V and recs travel over the
        socket."""
        return hdr.data_off + hdr.total_nbytes

    # -- producer
    def open_put(self, xfer_id: str, manifest: Manifest) -> PutHandle | None:
        engine, hx = parse_xfer_id(xfer_id)
        if engine != self.engine_id:
            raise ValueError(
                f"xfer_id {xfer_id} does not belong to producer {self.engine_id}"
            )
        manifest = coerce_manifest(manifest)
        if manifest.layout_version != LAYOUT_VERSION:
            raise ValueError("manifest layout_version mismatch")
        records, hdr = segment_layout(manifest, self.mode)
        total_file = hdr.data_off + hdr.total_nbytes  # raw mode: the mmap size
        charge = self._segment_charge(hdr, manifest)
        with self._lock:
            if xfer_id in self._puts:
                raise ValueError(f"{xfer_id} already open")
            if self.outstanding_bytes() + charge > self.budget_bytes:
                self.stats["budget_refusals"] += 1
                logger.warning(
                    "shm budget exhausted (%d + %d > %d): refusing %s",
                    self.outstanding_bytes(),
                    charge,
                    self.budget_bytes,
                    xfer_id,
                )
                return None
            os.makedirs(self._engine_dir(), exist_ok=True)
            try:
                st = os.statvfs(self._engine_dir())
                free = st.f_bavail * st.f_frsize
            except OSError:  # pragma: no cover
                free = charge + self.free_space_headroom
            if free < charge + self.free_space_headroom:
                self.stats["budget_refusals"] += 1
                logger.warning(
                    "tmpfs has %d B free, need %d + headroom: refusing %s",
                    free,
                    charge,
                    xfer_id,
                )
                return None
            tmp = self._tmp_dir(engine, hx)
            pub = self._pub_dir(engine, hx)
            for stale in (tmp, pub):  # an older attempt of the same id (unclaimed)
                if os.path.isdir(stale):
                    self._rmtree(stale)
            os.mkdir(tmp)
            now = self._clock()
            hdr.producer_engine_id, hdr.xfer_hex = engine, hx
            hdr.created_ts, hdr.lease_expiry_ts = now, now + self.lease_duration
            hdr.checksum_flags = CHECKSUM_CRC32C if self.checksum else 0
            hpath = os.path.join(tmp, self._header_name())
            state = _PutState(xfer_id, hx, tmp, hpath, hdr, charge, now)
            head = hdr.pack_fixed() + hdr.pack_table() + manifest.to_json()
            fd = os.open(hpath, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                if self.mode == "raw":
                    os.posix_fallocate(fd, 0, total_file)
                    os.pwrite(fd, head, 0)
                    state.fd = fd
                    state.mm = mmap.mmap(
                        fd,
                        total_file,
                        mmap.MAP_SHARED,
                        mmap.PROT_READ | mmap.PROT_WRITE,
                    )
                else:
                    os.pwrite(fd, head, 0)
                    os.close(fd)
            except OSError:
                os.close(fd)
                self._rmtree(tmp)
                raise
            sinks: dict[str, Sink] = {}
            for rec in records:
                if self.mode == "raw":
                    assert state.mm is not None
                    sinks[rec.name] = RawSink(
                        rec.spec, state.mm, rec.offset, self.layer, self.hosts, state
                    )
                else:
                    sinks[rec.name] = DumpfileSink(rec.spec, tmp, self.layer, state)
            self._puts[xfer_id] = state
        return PutHandle(
            xfer_id=xfer_id,
            manifest=manifest,
            sinks=sinks,
            lease_expiry_ts=hdr.lease_expiry_ts,
        )

    def _part_crc(self, st: _PutState, rec: PartRecord) -> int:
        if self.mode == "raw":
            assert st.mm is not None
            return crc32c(memoryview(st.mm)[rec.offset : rec.offset + rec.nbytes])
        c = 0
        names = (
            [f"{rec.name}.rows.pt"]
            if rec.spec.kind == "gdn_taps"
            else [f"{rec.name}.c{i}.tensorbin" for i in range(rec.nchunks)]
        )
        for n in names:
            p = os.path.join(st.tmp_dir, n)
            if os.path.isfile(p):
                with open(p, "rb") as f:
                    c = crc32c(f.read(), c)
        return c

    def finish_export(self, h: PutHandle, status: Literal["READY", "FAILED"]) -> None:
        code = {"READY": READY, "FAILED": FAILED}[status]
        with self._lock:
            st = self._puts.get(h.xfer_id)
            if st is None or st.published:
                return  # abandoned or already published: idempotent
            engine, hx = parse_xfer_id(h.xfer_id)
            # part table: chunks_written (+ crc under the debug flag), before the status
            for rec in st.hdr.parts:
                rec.chunks_written = len(st.written.get(rec.name, ()))
                if self.checksum and code == READY:
                    rec.crc32c = self._part_crc(st, rec)
            fd = os.open(st.header_path, os.O_WRONLY)
            try:
                os.pwrite(fd, st.hdr.pack_table(), st.hdr.table_off)
            finally:
                os.close(fd)
            st.close()  # producer keeps nothing mapped after publish
            if self.mode == "dumpfile":
                self._write_sizes(st.tmp_dir)  # before the status flip
            write_status(st.header_path, code)  # LAST
            st.hdr.status = code
            os.rename(st.tmp_dir, self._pub_dir(engine, hx))  # atomic publish
            st.published = True

    def _write_sizes(self, seg_dir: str) -> None:
        """Dumpfile mode: record every data file's byte length (``SIZES_FILE``) so
        the consumer's ``nbytes_present`` sees a truncated file as absent. Best
        effort: a failure here leaves a legacy segment (size > 0 rule)."""
        try:
            sizes: dict[str, int] = {}
            for n in os.listdir(seg_dir):
                if n in ("header", "data", SIZES_FILE) or n.endswith(".tmp"):
                    continue
                p = os.path.join(seg_dir, n)
                if os.path.isfile(p):
                    sizes[n] = os.path.getsize(p)
            tmp = os.path.join(seg_dir, SIZES_FILE + ".tmp")
            with open(tmp, "w") as f:
                json.dump(sizes, f, separators=(",", ":"))
            os.replace(tmp, os.path.join(seg_dir, SIZES_FILE))
        except OSError:  # pragma: no cover - tmpfs trouble; the export still publishes
            logger.warning("could not write %s in %s", SIZES_FILE, seg_dir)

    def _retire(self, path: str) -> bool:
        """Janitor removal of a PUBLISHED (unclaimed) segment: rename it to a
        name no consumer's ``open_get`` can claim, THEN rmtree. ``shutil.rmtree``
        is dir-fd based, so a consumer claim rename landing mid-sweep would
        otherwise leave it owning a directory whose files are being deleted
        (audit: janitor-vs-claim race); after the rename its claim fails
        (FileNotFoundError -> MISSING -> recompute). False when the directory
        vanished or was claimed first."""
        target = f"{path}{EXPIRED_MARK}{os.getpid()}"
        try:
            os.rename(path, target)
        except FileNotFoundError:
            return False  # claimed (renamed away) between listdir and here
        except OSError:  # pragma: no cover - same-directory rename; fall back
            self._rmtree(path)
            return True
        self._rmtree(target)
        return True

    def abandon(self, xfer_id: str) -> None:
        try:
            engine, hx = parse_xfer_id(xfer_id)
        except ValueError:
            return
        with self._lock:
            st = self._puts.pop(xfer_id, None)
            if st is not None:
                st.close()
            self._inherited.pop(xfer_id, None)
            removed = []
            for d in (self._tmp_dir(engine, hx), self._pub_dir(engine, hx)):
                if os.path.isdir(d):
                    self._rmtree(d)  # a claimed segment belongs to the consumer: kept
                    removed.append(os.path.basename(d))
            logger.info("shm: abandoned %s (removed %s)", xfer_id, removed or "nothing")

    # -- consumer
    def open_get(self, desc: Any) -> GetHandle | None:
        xfer_id = str(getattr(desc, "xfer_id", desc))
        try:
            engine, hx = parse_xfer_id(xfer_id)
        except ValueError as e:
            return GetHandle(xfer_id, None, "FAILED", {}, str(e))
        tdesc = getattr(desc, "transport", None)
        if isinstance(tdesc, dict) and (
            tdesc.get("kind", self.KIND) != self.KIND
            or tdesc.get("mode", self.mode) != self.mode
        ):
            return GetHandle(
                xfer_id,
                None,
                "FAILED",
                {},
                f"transport mismatch: {tdesc} vs {self.descriptor()}",
            )
        with self._lock:
            gs = self._gets.get(xfer_id)
            if gs is not None:  # re-open of our own claim
                return self._handle_from_state(gs)
            mine = self._claim_dir(engine, hx)
            pub = self._pub_dir(engine, hx)
            if not os.path.isdir(mine) and os.path.isdir(pub):
                hp = self._find_header(pub)
                if hp is not None and os.path.basename(hp) != self._header_name():
                    return GetHandle(
                        xfer_id,
                        None,
                        "FAILED",
                        {},
                        f"segment is {os.path.basename(hp)}-mode, consumer "
                        f"is {self.mode}",
                    )  # not ours to destroy
                try:
                    os.rename(pub, mine)  # CLAIM
                    os.utime(mine, None)  # stale-claim age counts from the claim
                    self._stamp_consumer_pid(mine)
                except (FileNotFoundError, OSError):
                    pass
            if os.path.isdir(mine):
                return self._open_claimed(xfer_id, mine)
            edir = self._engine_dir(engine)
            try:
                names = os.listdir(edir)
            except FileNotFoundError:
                names = []
            if any(n.startswith(f"{hx}.claimed-") for n in names):
                return GetHandle(
                    xfer_id, None, "MISSING", {}, "claimed by another consumer"
                )
            tmp = self._tmp_dir(engine, hx)
            if os.path.isdir(tmp):
                hp = self._find_header(tmp)
                if hp is None:
                    return None  # header not yet written
                try:
                    hdr = read_header(hp, full=False)
                except (ValueError, OSError):
                    return None  # torn / partial header: not yet
                if hdr.status == WRITING and not pid_alive(hdr.producer_pid):
                    return GetHandle(
                        xfer_id,
                        None,
                        "FAILED",
                        {},
                        f"producer pid {hdr.producer_pid} died while WRITING",
                    )
                return None  # WRITING (or READY an instant before the rename)
            return GetHandle(xfer_id, None, "MISSING", {}, "no such segment")

    def _open_claimed(self, xfer_id: str, mine: str) -> GetHandle:
        hp = self._find_header(mine)
        try:
            if hp is None:
                raise ValueError("segment has no header file")
            hdr = read_header(hp, full=True)
        except (ValueError, OSError) as e:
            self._rmtree(mine)
            return GetHandle(xfer_id, None, "FAILED", {}, f"unreadable header: {e}")
        if hdr.status != READY:
            self._rmtree(mine)  # ours now; nothing to import
            return GetHandle(
                xfer_id,
                hdr.manifest,
                "FAILED",
                {},
                f"segment status {STATUS_NAMES.get(hdr.status, hdr.status)}",
            )
        if hdr.mode != self.mode:
            self._rmtree(mine)
            return GetHandle(
                xfer_id,
                hdr.manifest,
                "FAILED",
                {},
                f"segment mode {hdr.mode} != {self.mode}",
            )
        gs = _GetState(xfer_id, mine, hp, hdr)
        if self.mode == "raw":
            gs.fd = os.open(hp, os.O_RDWR)
            gs.mm = mmap.mmap(gs.fd, 0, mmap.MAP_SHARED, mmap.PROT_READ)
            if self._pool is None and self._relayout_workers > 0:
                self._pool = ThreadPoolExecutor(
                    self._relayout_workers, thread_name_prefix="tt_pd_relayout"
                )
        self._gets[xfer_id] = gs
        return self._handle_from_state(gs)

    def _handle_from_state(self, gs: _GetState) -> GetHandle:
        hdr = gs.hdr
        sources: dict[str, Source] = {}
        if self.mode == "raw":
            assert gs.mm is not None
            gs.sources = []
            for rec in hdr.parts:
                s = RawSource(rec, gs.mm, hdr.nblk, self.layer, self.hosts, self._pool)
                gs.sources.append(s)
                sources[rec.name] = s
        else:
            for rec in hdr.parts:
                sources[rec.name] = DumpfileSource(rec, gs.claim_dir, self.layer)
        return GetHandle(gs.xfer_id, hdr.manifest, "READY", sources)

    def _drop_claim(self, gs: _GetState, status: int | None) -> None:
        gs.close()
        if status is not None:
            with contextlib.suppress(OSError):
                write_status(gs.header_path, status)
        self._rmtree(gs.claim_dir)

    def finish_import(self, h: GetHandle, ok: bool) -> None:
        with self._lock:
            gs = self._gets.pop(h.xfer_id, None)
            if gs is None:
                return  # never claimed / already finished: idempotent
            status = CONSUMED if ok else LOAD_FAILED
            self._drop_claim(gs, status)
            logger.info(
                "shm: segment %s %s; claim dir %s removed",
                h.xfer_id,
                STATUS_NAMES[status],
                os.path.basename(gs.claim_dir),
            )

    def release_remote(self, xfer_id: str) -> None:
        try:
            engine, hx = parse_xfer_id(xfer_id)
        except ValueError:
            return
        with self._lock:
            gs = self._gets.pop(xfer_id, None)
            if gs is not None:
                self._drop_claim(gs, RELEASED)
                logger.info(
                    "shm: released %s (our open claim dropped, dir removed)", xfer_id
                )
                return
            mine = self._claim_dir(engine, hx)
            if os.path.isdir(mine):  # claimed by us (e.g. an earlier instance): ours
                self._rmtree(mine)
                logger.info(
                    "shm: released %s (stale claim dir of ours removed)", xfer_id
                )
                return
            pub = self._pub_dir(engine, hx)
            hp = self._find_header(pub) if os.path.isdir(pub) else None
            if hp is not None:
                with contextlib.suppress(OSError):
                    write_status(hp, RELEASED)  # the producer's janitor unlinks it
                logger.info(
                    "shm: released %s (unclaimed segment marked RELEASED for the "
                    "producer's janitor)",
                    xfer_id,
                )
                return
            # missing (or still .tmp / foreign claim): nothing to do
            logger.info("shm: released %s (no segment on disk: nothing to do)", xfer_id)

    # -- janitor (producer)
    def janitor_once(self, now: float | None = None) -> None:
        """One sweep of ``{shm_dir}/{engine_id}`` (design 3.5)."""
        now = self._clock() if now is None else now
        edir = self._engine_dir()
        try:
            names = os.listdir(edir)
        except FileNotFoundError:
            return
        with self._lock:
            owned_tmp = {
                os.path.basename(s.tmp_dir)
                for s in self._puts.values()
                if not s.published
            }
            for n in names:
                p = os.path.join(edir, n)
                if not os.path.isdir(p):
                    continue
                if n.endswith(".tmp"):
                    if n not in owned_tmp:  # a previous process's unfinished export
                        self._rmtree(p)
                        self.stats["swept"] += 1
                elif ".claimed-" in n:
                    stale, why = self._claim_is_stale(p, now)
                    if stale:
                        logger.warning("removing stale claim %s (%s)", n, why)
                        self._rmtree(p)
                        self.stats["stale_claims"] += 1
                elif EXPIRED_MARK in n:
                    self._rmtree(p)  # a previous sweep died between rename and rmtree
                else:
                    hp = self._find_header(p)
                    status, expiry = None, None
                    if hp is not None:
                        try:
                            hdr = read_header(hp, full=False)
                            status, expiry = hdr.status, hdr.lease_expiry_ts
                        except (ValueError, OSError):
                            pass
                    if status is None or status in TERMINAL_UNCLAIMED:
                        if self._retire(p):
                            self.stats["swept"] += 1
                            logger.info(
                                "shm janitor: swept %s (status %s)",
                                n,
                                STATUS_NAMES.get(status, status),
                            )
                    elif expiry is not None and now > expiry and self._retire(p):
                        logger.warning("lease expired on %s: consumer never came", n)
                        self.stats["expired"] += 1
            # budget: an entry is outstanding while any of its directories exists
            try:
                names = set(os.listdir(edir))
            except FileNotFoundError:
                names = set()

            def _exists(hx: str) -> bool:
                return (
                    hx in names
                    or f"{hx}.tmp" in names
                    or any(n.startswith(f"{hx}.claimed-") for n in names)
                )

            for xid, st in list(self._puts.items()):
                if st.published and not _exists(st.xfer_hex):
                    st.close()
                    del self._puts[xid]
                    logger.info(
                        "shm janitor: segment %s gone (claimed and finished by the "
                        "consumer, or swept); budget charge released",
                        xid,
                    )
            for xid in list(self._inherited):
                if not _exists(parse_xfer_id(xid)[1]):
                    del self._inherited[xid]

    def _janitor_loop(self) -> None:
        while not self._stop.wait(self.janitor_period):
            try:
                self.janitor_once()
            except Exception:  # pragma: no cover
                logger.exception("shm janitor sweep failed")

    def startup_sweep(self) -> None:
        """A restarted producer inherits nothing usable: unlink everything but the
        ``.claimed-*`` dirs of a live consumer (it may be importing them, or holding a
        KV_DONE handle until its join step) and count those toward the budget until
        they vanish."""
        edir = self._engine_dir()
        os.makedirs(edir, exist_ok=True)
        now = self._clock()
        with self._lock:
            for n in os.listdir(edir):
                p = os.path.join(edir, n)
                if not os.path.isdir(p):
                    continue
                if ".claimed-" in n and not self._claim_is_stale(p, now)[0]:
                    hx = n.split(".claimed-", 1)[0]
                    hp = self._find_header(p)
                    nbytes = 0
                    if hp is not None:
                        try:
                            hdr = read_header(hp, full=False)
                            nbytes = hdr.data_off + hdr.total_nbytes
                        except (ValueError, OSError):
                            pass
                    self._inherited[f"{self.engine_id}:{hx}"] = nbytes
                else:
                    self._rmtree(p)
                    self.stats["swept"] += 1

    def start(self) -> None:
        if self.is_producer:
            self.startup_sweep()
            if self._janitor is None and self.janitor_period > 0:
                self._stop.clear()
                self._janitor = threading.Thread(
                    target=self._janitor_loop, name="tt_pd_shm_janitor", daemon=True
                )
                self._janitor.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._janitor is not None:
            self._janitor.join(timeout=5.0)
            self._janitor = None
        with self._lock:
            for gs in list(self._gets.values()):  # owned (claimed) segments
                self._drop_claim(gs, RELEASED)
            self._gets.clear()
            for st in self._puts.values():
                st.close()
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None


__all__ = [
    "CONSUMED",
    "FAILED",
    "LOAD_FAILED",
    "READY",
    "RELEASED",
    "WRITING",
    "STATUS_NAMES",
    "HEAD_MAJOR_KV_SHAPE",
    "Header",
    "PartRecord",
    "ShmTransport",
    "TtnnDeviceLayer",
    "crc32c",
    "parse_xfer_id",
    "read_header",
    "read_status",
    "relayout_block_major_to_head_major",
    "segment_layout",
    "write_status",
]
