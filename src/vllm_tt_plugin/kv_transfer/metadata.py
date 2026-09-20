# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Scheduler <-> worker metadata and the on-the-wire manifest (PHASE2_DESIGN.md 3.4).

Plain dataclasses: the uniproc executor hands the same ``SchedulerOutput`` object
to the in-process worker, nothing here is serialized. No ``ttnn`` import (this
module is loaded in the API-server process through ``tt_connector``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)

LAYOUT_VERSION = 1

# I10: an ``xfer_id`` is ``{engine_id}:{blake2b(request_id, 16).hexdigest()}``;
# the client-controlled request id never reaches a path or a header field.
ENGINE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
XFER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}:[0-9a-f]{32}$")

REQUIRED_PARAM_KEYS = (
    "remote_engine_id",
    "remote_request_id",
    "remote_num_tokens",
    "remote_prompt_hash",
    "remote_transport",
    "remote_fingerprint",
    "xfer_id",
    "tt_layout_version",
)


@dataclass
class TransferDescriptor:
    """Everything the consumer needs to find and check one exported segment.

    Built from the producer's ``kv_transfer_params`` by :meth:`from_params`;
    carries NO path: the consumer resolves ``xfer_id`` under its own transport
    root (I10).
    """

    engine_id: str
    request_id: str
    xfer_id: str
    num_tokens: int
    transport: dict
    layout_version: int
    chunk_tokens: int
    expiry: float | None
    prompt_hash: str
    fingerprint: str

    @property
    def xfer_hex(self) -> str:
        return self.xfer_id.split(":", 1)[1]

    @classmethod
    def from_params(cls, p: dict[str, Any]) -> TransferDescriptor:
        """KeyError/TypeError/ValueError on a missing, mistyped or malformed key.

        The caller (``TTKVConnector._params_ok``) turns any of them into a
        demotion.
        """
        d = cls(
            engine_id=str(p["remote_engine_id"]),
            request_id=str(p["remote_request_id"]),
            xfer_id=str(p["xfer_id"]),
            num_tokens=int(p["remote_num_tokens"]),
            transport=dict(p["remote_transport"]),
            layout_version=int(p["tt_layout_version"]),
            chunk_tokens=int(p.get("tt_chunk_tokens", 2048)),
            expiry=p.get("remote_blocks_expiry_time"),
            prompt_hash=str(p["remote_prompt_hash"]),
            fingerprint=str(p["remote_fingerprint"]),
        )
        if not XFER_ID_RE.match(d.xfer_id):
            raise ValueError(f"malformed xfer_id {d.xfer_id!r}")
        if not ENGINE_ID_RE.match(d.engine_id):
            raise ValueError(f"malformed remote_engine_id {d.engine_id!r}")
        if d.xfer_id.split(":", 1)[0] != d.engine_id:
            raise ValueError("xfer_id prefix does not match remote_engine_id")
        if d.expiry is not None and not isinstance(d.expiry, (int, float)):
            raise TypeError("remote_blocks_expiry_time must be a number")
        if d.num_tokens <= 0 or d.chunk_tokens <= 0:
            raise ValueError("remote_num_tokens / tt_chunk_tokens must be positive")
        return d


@dataclass
class RecvMeta:
    """Consumer: one load to issue (scheduler -> worker)."""

    local_block_ids: list[int]
    num_tokens: int
    xfer: TransferDescriptor


@dataclass
class SaveMeta:
    """Producer: one export to run after this step's prefill (scheduler -> worker)."""

    block_ids: list[int]
    num_tokens: int
    xfer_id: str


@dataclass
class TTKVConnectorMetadata(KVConnectorMetadata):
    """Built EXACTLY ONCE per engine step
    (``TTScheduler._finalize_scheduler_output``)."""

    reqs_to_recv: dict[str, RecvMeta] = field(default_factory=dict)
    reqs_to_save: dict[str, SaveMeta] = field(default_factory=dict)
    reqs_to_send: dict[str, float] = field(default_factory=dict)
    reqs_not_processed: set[str] = field(default_factory=set)
    to_release: set[str] = field(default_factory=set)

    def is_empty(self) -> bool:
        return not (
            self.reqs_to_recv
            or self.reqs_to_save
            or self.reqs_to_send
            or self.reqs_not_processed
            or self.to_release
        )


@dataclass
class TTKVWorkerMeta(KVConnectorWorkerMetadata):
    """Worker -> scheduler: free device state slots
    (refreshes ``_free_slots_estimate``)."""

    free_state_slots: int

    def aggregate(self, other: KVConnectorWorkerMetadata) -> TTKVWorkerMeta:
        if not isinstance(other, TTKVWorkerMeta):
            raise TypeError(
                f"cannot aggregate TTKVWorkerMeta with {type(other).__name__}"
            )
        return TTKVWorkerMeta(min(self.free_state_slots, other.free_state_slots))


@dataclass
class PartSpec:
    """One part of the on-the-wire layout (section 4.1 table)."""

    name: str  # "kv.L{i}.k" | "kv.L{i}.v" | "gdn.L{j}.rec" | "gdn.L{j}.taps"
    kind: Literal["kv_blocks", "gdn_rec", "gdn_taps"]
    shape: tuple[int, ...]  # on-wire chunk shape
    dtype: str  # ttnn dtype name: "bfloat8_b" | "bfloat16" | "float32"
    layout: str  # "TILE" (device parts) | "ROW_MAJOR" (host rows)
    nchunks: int  # cdiv(nblk, blocks_per_chunk) for kv_blocks, 1 otherwise
    chunk_nbytes: int
    chunk_offsets: list[int]  # byte offset of chunk c inside the part's data region
    nbytes: int  # nchunks * chunk_nbytes


@dataclass
class Manifest:
    layout_version: int
    model_sig: str  # == connector fingerprint
    num_tokens: int
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
