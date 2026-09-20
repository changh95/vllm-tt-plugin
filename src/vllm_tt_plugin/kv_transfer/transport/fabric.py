# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""FabricSocketTransport: the Phase 3 seam (PHASE2_DESIGN 4.3). NOT built in Phase 2.

Interface identical to ``ShmTransport``; only the ``Sink`` / ``SourceChunk`` data
plane and the control plane change:

* ``Sink.write_from_device(t, chunk=c)`` = ``ttnn.experimental.send_direct_async(t,
  socket)`` (direct mode: 128 B L1 footprint, data lands in the receiver's staging
  tensor of the IDENTICAL TensorSpec + memory config).
* ``SourceChunk.read_into_device(staging)`` = ``ttnn.experimental.recv_direct_async(
  staging, socket)``; ``is_head_major`` is True only if the sender relays (v2 decision).
* Control plane = ZMQ REQ/REP (``xfer_ready``, ``recv_posted``, ``consumed``,
  heartbeat) as in PHASE3_NOTES 3.3-3.5; the lease travels as a DURATION relative
  to the handshake (one host clock per node), never as an absolute wall-clock.
* Both sides use fixed-shape staging tensors from ``STAGING_SPECS`` below; recv posts
  and fills interleave on the same CQ; ``finished_sending`` waits for ``consumed``
  (the producer's export tensors are read by the send kernels: the ``armed & done``
  rule becomes ``armed & consumed``).
* Process shape: ONE tt-run/mpirun job with two headless ``EngineCoreProc`` ranks
  (rank 0 = prefill on mesh 0, rank 1 = decode on mesh 1), chip pair (0, 3) or (1, 2)
  on the P300x2 box, ``FABRIC_2D``; ``P/worker.py get_fabric_config`` must stop
  returning ``None`` for 1-device meshes.  ``D2DStreamService`` is ruled out
  (UINT32 ROW_MAJOR only).

Everything below raises ``NotImplementedError`` so a misconfigured
``transport=fabric`` fails loudly at construction time of the first handle.
"""

from __future__ import annotations

from typing import Any, Literal

from .base import (
    HEAD_MAJOR_KV_SHAPE_HINT,
    LAYOUT_VERSION,
    REC_SHAPE,
    TAPS_SHAPE,
    GetHandle,
    Manifest,
    PutHandle,
    TTKVTransport,
)

# One spec table for both ranks (design 4.3): the v1 wire layout already honours it.
STAGING_SPECS: dict[str, tuple[tuple[int, ...], str, str]] = {
    "kv_blocks": (HEAD_MAJOR_KV_SHAPE_HINT, "<cache dtype>", "TILE"),  # [1,4,2048,256]
    "gdn_rec": (REC_SHAPE, "float32", "TILE"),  # [1,48,128,128]
    "gdn_taps": (TAPS_SHAPE, "bfloat16", "ROW_MAJOR"),  # host rows over ZMQ
}

_MSG = (
    "FabricSocketTransport is the Phase 3 seam (PHASE2_DESIGN 4.3 / PHASE3_NOTES); "
    "Phase 2 ships ShmTransport only -- set kv_connector_extra_config.transport=shm"
)


class FabricSocketTransport(TTKVTransport):
    KIND = "fabric"

    def __init__(
        self,
        *,
        engine_id: str,
        rank: int | None = None,
        control_endpoint: str | None = None,
        lease_duration: float = 30.0,
        **_unused: Any,
    ) -> None:
        self.engine_id, self.rank = engine_id, rank
        self.control_endpoint, self.lease_duration = control_endpoint, lease_duration

    def descriptor(self) -> dict[str, Any]:
        # Path-free like shm's; the ZMQ endpoint is exchanged at the handshake.
        return {"kind": self.KIND, "mode": "direct", "layout_version": LAYOUT_VERSION}

    def open_put(self, xfer_id: str, manifest: Manifest) -> PutHandle | None:
        raise NotImplementedError(_MSG)

    def finish_export(self, h: PutHandle, status: Literal["READY", "FAILED"]) -> None:
        raise NotImplementedError(_MSG)

    def abandon(self, xfer_id: str) -> None:
        raise NotImplementedError(_MSG)

    def open_get(self, desc: Any) -> GetHandle | None:
        raise NotImplementedError(_MSG)

    def finish_import(self, h: GetHandle, ok: bool) -> None:
        raise NotImplementedError(_MSG)

    def release_remote(self, xfer_id: str) -> None:
        raise NotImplementedError(_MSG)

    def start(self) -> None:
        raise NotImplementedError(_MSG)

    def shutdown(self) -> None:
        # Tolerant: a connector tearing down a never-started transport must not raise.
        return None


__all__ = ["STAGING_SPECS", "FabricSocketTransport"]
