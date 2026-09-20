# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""KV handoff transports (PHASE2_DESIGN section 4).

``base`` holds the interface and the wire layout, ``shm`` the v1 shared-memory
data plane (``dumpfile`` and ``raw`` modes), ``fabric`` the Phase 3 seam.  Nothing
here imports ``ttnn`` at module level; ``make_transport`` is the one factory the
connector needs.
"""

from __future__ import annotations

import os
from typing import Any

from .base import (
    LAYOUT_VERSION,
    DeviceLayer,
    GetHandle,
    Manifest,
    PartSpec,
    PutHandle,
    Sink,
    Source,
    SourceChunk,
    TTKVTransport,
    build_manifest,
    chunk_offsets,
    request_nbytes,
)
from .fabric import FabricSocketTransport
from .shm import ShmTransport


def make_transport(
    kind: str,
    mode: str | None = None,
    kv_transfer_config: Any = None,
    kv_role: str | None = None,
    *,
    engine_id: str | None = None,
    **kwargs: Any,
) -> TTKVTransport:
    """Factory used by ``TTKVConnector.attach_runner``.

    ``make_transport(kind, shm_mode, kv_transfer_config, kv_role)`` derives
    ``engine_id`` / ``shm_dir`` / ``budget_bytes`` / ``lease_duration`` / ``checksum``
    from ``kv_transfer_config`` (``engine_id`` + ``kv_connector_extra_config`` keys of
    design 3.1, plus ``TT_PD_CHECKSUM=1``); explicit keyword arguments win.  Tests call
    ``make_transport("shm", engine_id=..., **ShmTransport kwargs)`` directly.
    """
    opts: dict[str, Any] = {}
    if kv_transfer_config is not None:
        extra = dict(
            getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
        )
        opts["engine_id"] = getattr(kv_transfer_config, "engine_id", None)
        opts["shm_dir"] = extra.get("shm_dir", "/dev/shm/tt_pd")
        opts["budget_bytes"] = int(extra.get("shm_budget_bytes", 8 << 30))
        opts["lease_duration"] = float(extra.get("kv_lease_duration", 30.0))
        opts["checksum"] = bool(
            extra.get("checksum", os.environ.get("TT_PD_CHECKSUM", "0") == "1")
        )
        if mode is None:
            mode = extra.get("shm_mode", "dumpfile")
        if kv_role is None:
            kv_role = getattr(kv_transfer_config, "kv_role", None)
    if kv_role is not None:
        role = str(getattr(kv_role, "value", kv_role))
        opts["role"] = {
            "kv_producer": "producer",
            "kv_consumer": "consumer",
            "kv_both": "both",
        }.get(role, role)
    if engine_id is not None:
        opts["engine_id"] = engine_id
    opts.update(kwargs)
    if opts.get("engine_id") is None:
        raise ValueError("make_transport needs engine_id or kv_transfer_config")
    if kind == "shm":
        return ShmTransport(mode=mode or "dumpfile", **opts)
    if kind == "fabric":
        opts.pop("shm_dir", None)
        opts.pop("budget_bytes", None)
        opts.pop("checksum", None)
        return FabricSocketTransport(**opts)
    raise ValueError(f"unknown transport kind {kind!r} (shm | fabric)")


__all__ = [
    "LAYOUT_VERSION",
    "DeviceLayer",
    "FabricSocketTransport",
    "GetHandle",
    "Manifest",
    "PartSpec",
    "PutHandle",
    "ShmTransport",
    "Sink",
    "Source",
    "SourceChunk",
    "TTKVTransport",
    "build_manifest",
    "chunk_offsets",
    "make_transport",
    "request_nbytes",
]
