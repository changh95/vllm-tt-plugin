# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""TTKVTransport interface and the ONE on-the-wire layout (PHASE2_DESIGN 4.1).

The model hook (``M/kv_transfer.py``, design section 5) and every transport
implementation (``shm.py`` today, ``fabric.py`` in Phase 3) share exactly the names
in this module and nothing else.  Nothing here imports ``ttnn`` or ``torch`` at
module level: the connector module is imported in the API-server process
(design 3), so this file must stay importable without a device stack.

Wire layout, ``LAYOUT_VERSION = 1`` (identical in ``raw`` and ``dumpfile`` mode):

===================  ======================================  ======  ============
Part                 chunk spec                              chunks  chunk bytes
===================  ======================================  ======  ============
``kv.L{i}.{k,v}``    ``[32, 4, 64, 256]`` TILE, cache dtype  cdiv(nblk, 32)  32 x BLK
``gdn.L{j}.rec``     ``[1, 48, 128, 128]`` TILE fp32         1       3,145,728
``gdn.L{j}.taps``    ``[4, 10240]`` ROW_MAJOR bf16 (host)    1       81,920
===================  ======================================  ======  ============

K/V chunks are BLOCK-major: block ``32c + j`` of the request sits at index ``j`` of
chunk ``c``; indices beyond ``nblk`` hold don't-care bytes.  ``BLK`` is one block of
one paged K or V tensor: 64 tiles of 1088 B (bfp8_b: 1024 B mantissas + 64 B
shared exponents) = 69,632 B, or 64 x 2048 B = 131,072 B for a bf16 cache.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

LAYOUT_VERSION = 1

# --- geometry constants (design 4.1 / 5.2 / 5.6) -------------------------------
BLOCK_TOKENS = 64  # vLLM --block-size on both nodes
KV_HEADS = 4
KV_HEAD_DIM = 256
BLOCKS_PER_CHUNK = 32  # xfer_chunk_tokens 2048 / 64
CHUNK_TOKENS = BLOCKS_PER_CHUNK * BLOCK_TOKENS
TILE = 32
TILE_RECORD_BYTES = {
    "bfloat8_b": 1088,
    "bfloat4_b": 576,
    "bfloat16": 2048,
    "float32": 4096,
}
TILES_PER_BLOCK = (BLOCK_TOKENS // TILE) * (KV_HEAD_DIM // TILE) * KV_HEADS  # 64
REC_SHAPE = (1, 48, 128, 128)
TAPS_SHAPE = (4, 10240)
KV_CHUNK_SHAPE = (BLOCKS_PER_CHUNK, KV_HEADS, BLOCK_TOKENS, KV_HEAD_DIM)
# consumer staging spec of a K/V chunk after the head-major relayout (raw/fabric)
HEAD_MAJOR_KV_SHAPE_HINT = (1, KV_HEADS, BLOCKS_PER_CHUNK * BLOCK_TOKENS, KV_HEAD_DIM)
TAPS_NBYTES = TAPS_SHAPE[0] * TAPS_SHAPE[1] * 2  # 81,920 (bf16 host rows)
ALIGN = 4096

PartKind = Literal["kv_blocks", "gdn_rec", "gdn_taps"]


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def kv_block_nbytes(kv_dtype: str) -> int:
    """Bytes of ONE block of ONE paged K or V tensor (``BLK`` in the design)."""
    return TILES_PER_BLOCK * TILE_RECORD_BYTES[kv_dtype]


def rec_nbytes(rec_dtype: str) -> int:
    """Bytes of one ``[1, 48, 128, 128]`` TILE rec row (fp32 default)."""
    n = 48 * 128 * 128
    return n * {"float32": 4, "bfloat16": 2}[rec_dtype]


def chunk_offsets(nchunks: int, chunk_nbytes: int) -> list[int]:
    """Byte offset of chunk ``c`` inside a part's data region: ``c * chunk_nbytes``."""
    return [c * chunk_nbytes for c in range(nchunks)]


@dataclass
class PartSpec:
    name: str  # "kv.L{i}.k" | "kv.L{i}.v" | "gdn.L{j}.rec" | "gdn.L{j}.taps"
    kind: PartKind
    shape: tuple[int, ...]  # on-wire CHUNK shape
    dtype: str  # ttnn dtype name: "bfloat8_b" | "bfloat16" | "float32"
    layout: str  # "TILE" (device parts) | "ROW_MAJOR" (host rows)
    nchunks: int
    chunk_nbytes: int
    chunk_offsets: list[int] = field(default_factory=list)
    nbytes: int = 0

    def __post_init__(self) -> None:
        self.shape = tuple(int(s) for s in self.shape)
        if not self.chunk_offsets:
            self.chunk_offsets = chunk_offsets(self.nchunks, self.chunk_nbytes)
        if not self.nbytes:
            self.nbytes = self.nchunks * self.chunk_nbytes

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["shape"] = list(self.shape)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PartSpec:
        return cls(
            name=str(d["name"]),
            kind=d["kind"],
            shape=tuple(d["shape"]),
            dtype=str(d["dtype"]),
            layout=str(d["layout"]),
            nchunks=int(d["nchunks"]),
            chunk_nbytes=int(d["chunk_nbytes"]),
            chunk_offsets=[int(x) for x in d.get("chunk_offsets", [])],
            nbytes=int(d.get("nbytes", 0)),
        )

    def spec_key(self) -> tuple[tuple[int, ...], str, str]:
        """(shape, dtype, layout) -- what a device/host tensor must match."""
        return (self.shape, self.dtype, self.layout)


def kv_part_spec(name: str, nblk: int, kv_dtype: str = "bfloat8_b") -> PartSpec:
    blk = kv_block_nbytes(kv_dtype)
    return PartSpec(
        name=name,
        kind="kv_blocks",
        shape=KV_CHUNK_SHAPE,
        dtype=kv_dtype,
        layout="TILE",
        nchunks=cdiv(nblk, BLOCKS_PER_CHUNK),
        chunk_nbytes=BLOCKS_PER_CHUNK * blk,
    )


def rec_part_spec(name: str, rec_dtype: str = "float32") -> PartSpec:
    return PartSpec(
        name=name,
        kind="gdn_rec",
        shape=REC_SHAPE,
        dtype=rec_dtype,
        layout="TILE",
        nchunks=1,
        chunk_nbytes=rec_nbytes(rec_dtype),
    )


def taps_part_spec(name: str) -> PartSpec:
    return PartSpec(
        name=name,
        kind="gdn_taps",
        shape=TAPS_SHAPE,
        dtype="bfloat16",
        layout="ROW_MAJOR",
        nchunks=1,
        chunk_nbytes=TAPS_NBYTES,
    )


@dataclass
class Manifest:
    layout_version: int
    model_sig: str  # == connector fingerprint
    num_tokens: int  # T-1 (the T-1 rule, design 2.1)
    nblk: int
    block_size: int
    chunk_tokens: int
    kv_dtype: str
    rec_dtype: str
    parts: list[PartSpec]
    prompt_hash: str

    @property
    def total_nbytes(self) -> int:
        return sum(p.nbytes for p in self.parts)

    def part(self, name: str) -> PartSpec:
        for p in self.parts:
            if p.name == name:
                return p
        raise KeyError(name)

    def to_json(self) -> bytes:
        d = asdict(self)
        d["parts"] = [p.to_dict() for p in self.parts]
        return json.dumps(d, separators=(",", ":")).encode()

    @classmethod
    def from_json(cls, raw: bytes | str) -> Manifest:
        d = json.loads(raw)
        d["parts"] = [PartSpec.from_dict(p) for p in d["parts"]]
        return cls(**d)


def build_manifest(
    num_tokens: int,
    *,
    model_sig: str,
    prompt_hash: str,
    kv_dtype: str = "bfloat8_b",
    rec_dtype: str = "float32",
    num_attn_layers: int = 16,
    num_gdn_layers: int = 48,
    block_size: int = BLOCK_TOKENS,
    chunk_tokens: int = CHUNK_TOKENS,
) -> Manifest:
    """The Manifest ``describe_request_state`` returns for ``num_tokens`` (= T-1).

    32 ``kv_blocks`` parts of ``cdiv(nblk, 32)`` chunks, 48 ``gdn_rec`` and 48
    ``gdn_taps`` parts, ``nblk = cdiv(num_tokens, block_size)``.
    """
    if block_size != BLOCK_TOKENS or chunk_tokens != CHUNK_TOKENS:
        raise ValueError("layout_version 1 is fixed at block 64 / chunk 2048 tokens")
    nblk = cdiv(num_tokens, block_size)
    parts: list[PartSpec] = []
    for i in range(num_attn_layers):
        parts.append(kv_part_spec(f"kv.L{i}.k", nblk, kv_dtype))
        parts.append(kv_part_spec(f"kv.L{i}.v", nblk, kv_dtype))
    for j in range(num_gdn_layers):
        parts.append(rec_part_spec(f"gdn.L{j}.rec", rec_dtype))
    for j in range(num_gdn_layers):
        parts.append(taps_part_spec(f"gdn.L{j}.taps"))
    return Manifest(
        layout_version=LAYOUT_VERSION,
        model_sig=model_sig,
        num_tokens=num_tokens,
        nblk=nblk,
        block_size=block_size,
        chunk_tokens=chunk_tokens,
        kv_dtype=kv_dtype,
        rec_dtype=rec_dtype,
        parts=parts,
        prompt_hash=prompt_hash,
    )


def coerce_manifest(m: Any) -> Manifest:
    """Accept this module's Manifest or the field-compatible ``metadata.Manifest``."""
    if isinstance(m, Manifest):
        return m
    parts = [
        p
        if isinstance(p, PartSpec)
        else PartSpec(
            name=p.name,
            kind=p.kind,
            shape=tuple(p.shape),
            dtype=p.dtype,
            layout=p.layout,
            nchunks=int(p.nchunks),
            chunk_nbytes=int(p.chunk_nbytes),
            chunk_offsets=list(getattr(p, "chunk_offsets", []) or []),
            nbytes=int(getattr(p, "nbytes", 0) or 0),
        )
        for p in m.parts
    ]
    return Manifest(
        layout_version=int(m.layout_version),
        model_sig=str(m.model_sig),
        num_tokens=int(m.num_tokens),
        nblk=int(m.nblk),
        block_size=int(m.block_size),
        chunk_tokens=int(m.chunk_tokens),
        kv_dtype=str(m.kv_dtype),
        rec_dtype=str(m.rec_dtype),
        parts=parts,
        prompt_hash=str(m.prompt_hash),
    )


def request_nbytes(
    num_tokens: int,
    kv_dtype: str = "bfloat8_b",
    rec_dtype: str = "float32",
    num_attn_layers: int = 16,
    num_gdn_layers: int = 48,
) -> int:
    """Payload bytes of one handoff (design 5.6 table, ``nblk = cdiv(T-1, 64)``).

    Counts real blocks only (the table's ``nblk x 2,228,224``); the on-wire size
    (``Manifest.total_nbytes``) rounds each K/V part up to whole 32-block chunks.
    """
    nblk = cdiv(num_tokens, BLOCK_TOKENS)
    kv = 2 * num_attn_layers * nblk * kv_block_nbytes(kv_dtype)
    return kv + num_gdn_layers * (rec_nbytes(rec_dtype) + TAPS_NBYTES)


# --- device layer seam --------------------------------------------------------


class DeviceLayer(Protocol):
    """Every ttnn call a transport makes, behind one thin adapter (design 4.2).

    The real implementation (``shm.TtnnDeviceLayer``) imports ``ttnn`` lazily; the
    unit tests inject a numpy-backed fake.  Tensors are opaque to the transport:
    only ``spec_of`` looks inside one.
    """

    def spec_of(self, tensor: Any) -> tuple[tuple[int, ...], str, str]:
        """(shape, dtype name, layout name) of a host or device tensor."""

    # dumpfile mode
    def from_device(self, tensor: Any) -> Any:
        """Raw D2H (no untilize) -> host tensor of the same spec; blocking."""

    def dump_tensor(self, path: str, host_tensor: Any) -> None:
        """``ttnn.dump_tensor(path, t, mode=LOCAL)``; ``path`` ends in .tensorbin."""

    def load_tensor(self, path: str, device: Any) -> Any:
        """``ttnn.load_tensor(path, device=device)`` -> fresh device tensor."""

    # raw mode
    def allocate_host_like(self, tensor: Any) -> Any:
        """``ttnn.allocate_tensor_on_host(t.spec, t.device())``."""

    def allocate_host(self, spec: PartSpec, device: Any) -> Any:
        """``ttnn.allocate_tensor_on_host(shape, dtype, layout, device)``."""

    def copy_device_to_host(
        self, device_tensor: Any, host_tensor: Any, *, blocking: bool, cq_id: int | None
    ) -> None:
        """``ttnn.copy_device_to_host_tensor``."""

    def copy_host_to_device(
        self, host_tensor: Any, device_tensor: Any, *, cq_id: int | None
    ) -> None:
        """``ttnn.copy_host_to_device_tensor``."""

    def host_bytes(self, host_tensor: Any) -> Any:
        """Writable uint8 numpy view of a host tensor's buffer (zero copy)."""

    def read_tensor_bytes(
        self,
        device_tensor: Any,
        dst: memoryview,
        src_offset: int,
        nbytes: int,
        *,
        blocking: bool,
    ) -> None:
        """Byte-range D2H straight into ``dst`` (nanobind helper of design 8 step 5).

        Raise ``NotImplementedError`` when the helper is not built; the raw Sink
        then takes the ``copy_device_to_host`` + ``host_bytes`` memcpy fallback.
        """


# --- interface (design 4.1, verbatim names) -------------------------------------


class Sink(ABC):
    """Producer side, one per PartSpec; the model hook never sees bytes."""

    spec: PartSpec
    supports_regions: bool  # True in raw mode; False in dumpfile mode

    @abstractmethod
    def write_from_device(
        self,
        device_tensor: Any,
        *,
        chunk: int,
        blocking: bool = True,
        cq_id: int | None = None,
    ) -> None:
        """Whole chunk ``chunk`` from a device tensor of exactly the chunk spec."""

    @abstractmethod
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
        """raw only: bytes [src, src+n) -> ``chunk_offsets[chunk] + dst_offset``."""

    @abstractmethod
    def write_rows(self, rows: torch.Tensor) -> None:
        """gdn_taps: ``[4, 10240]`` bf16 host rows, chunk 0."""

    @abstractmethod
    def write_host(self, host_tensor: Any, *, chunk: int) -> None:
        """A HOST tensor of exactly the chunk spec (dumpfile: dump; raw: memcpy)."""


class SourceChunk(ABC):
    """Consumer side, one chunk of one part."""

    is_head_major: bool  # K/V chunk already relaid head-major (raw mode)
    is_device_readable: bool  # read_into_device available (raw mode)

    @abstractmethod
    def read_into_device(
        self, staging_tensor: Any, *, cq_id: int | None = None
    ) -> None:
        """raw H2D into a preallocated staging tensor of the CHUNK's spec."""

    @abstractmethod
    def read_device(self, mesh_device: Any) -> Any:
        """dumpfile: materialize a fresh device tensor of ``spec.shape``."""


class Source(ABC):
    spec: PartSpec
    nbytes_present: int  # bytes actually present (validate_gdn_parts)
    spec_crc: int  # header crc32c (0 unless the checksum flag is on)

    @abstractmethod
    def crc32c(self) -> int:
        """Host-only, over the present bytes."""

    @abstractmethod
    def chunk(self, c: int) -> SourceChunk:
        pass

    @abstractmethod
    def read_rows(self) -> torch.Tensor:
        """gdn_taps: ``[4, 10240]`` bf16."""


@dataclass
class PutHandle:
    xfer_id: str
    manifest: Manifest
    sinks: dict[str, Sink]
    lease_expiry_ts: float = 0.0  # == remote_blocks_expiry_time of the params


@dataclass
class GetHandle:
    xfer_id: str
    manifest: Manifest | None
    status: str  # "READY" | "FAILED" | "MISSING"
    sources: dict[str, Source]
    reason: str = ""

    def ready(self) -> bool:
        return self.status == "READY"


class TTKVTransport(ABC):
    @abstractmethod
    def descriptor(self) -> dict[str, Any]:
        """{"kind", "mode", "layout_version"} -- NO paths (design 3.1)."""

    @abstractmethod
    def open_put(self, xfer_id: str, manifest: Manifest) -> PutHandle | None:
        """None when the budget / tmpfs cannot take it (-> FAILED export)."""

    @abstractmethod
    def finish_export(self, h: PutHandle, status: Literal["READY", "FAILED"]) -> None:
        """Header status written LAST, then the atomic publish."""

    @abstractmethod
    def abandon(self, xfer_id: str) -> None:
        """Producer gives up (reqs_not_processed); idempotent."""

    @abstractmethod
    def open_get(self, desc: Any) -> GetHandle | None:
        """None while WRITING; ``handle.status`` FAILED / MISSING otherwise."""

    @abstractmethod
    def finish_import(self, h: GetHandle, ok: bool) -> None:
        """header -> CONSUMED | LOAD_FAILED; unmap; unlink (consumer owns it)."""

    @abstractmethod
    def release_remote(self, xfer_id: str) -> None:
        """header -> RELEASED without reading; idempotent, tolerant."""

    @abstractmethod
    def start(self) -> None:
        """Startup sweep + janitor thread (producer)."""

    @abstractmethod
    def shutdown(self) -> None:
        """Stop the janitor; owned-segment cleanup."""


__all__ = [
    "ALIGN",
    "BLOCKS_PER_CHUNK",
    "BLOCK_TOKENS",
    "CHUNK_TOKENS",
    "HEAD_MAJOR_KV_SHAPE_HINT",
    "KV_CHUNK_SHAPE",
    "LAYOUT_VERSION",
    "REC_SHAPE",
    "TAPS_NBYTES",
    "TAPS_SHAPE",
    "TILE_RECORD_BYTES",
    "DeviceLayer",
    "GetHandle",
    "Manifest",
    "PartSpec",
    "PutHandle",
    "Sink",
    "Source",
    "SourceChunk",
    "TTKVTransport",
    "build_manifest",
    "cdiv",
    "chunk_offsets",
    "coerce_manifest",
    "kv_block_nbytes",
    "kv_part_spec",
    "rec_nbytes",
    "rec_part_spec",
    "request_nbytes",
    "taps_part_spec",
]
