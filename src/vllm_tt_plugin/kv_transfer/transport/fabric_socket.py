# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""The ONE module of the fabric transport that touches ``ttnn`` (PHASE3_NOTES 3).

``SocketLayer`` is every device / fabric / MPI call ``fabric.py`` makes, behind a
thin adapter, so the host-only unit tests can inject a fake that simulates two
ranks in one process.  ``TtnnSocketLayer`` is the real one; it imports ``ttnn``
lazily in ``__init__`` and never at module import (the connector module is imported
by the API-server process, design 3).

Device facts this adapter encodes (all measured in PHASE3_MICROBENCH on the
p150 pair, chips 1/2, under ``tt-run --bare``):

* one ``ttnn.MeshSocket`` per pair, ``SocketConfig(connections,
  SocketMemoryConfig(L1, 128 B), sender_rank=0, receiver_rank=1)``, sender cores
  ``(i, 0)``, receiver cores ``(i, 1)``, ``i < connections`` (<= fabric links = 2);
  the constructor is a rank-scoped rendezvous with a FIXED 10 s window, so both
  ranks call it right after ``distributed_context_barrier()`` (0.5 ms measured);
* direct mode: ``send_direct_async(t, sock)`` / ``recv_direct_async(t, sock)``;
  the two tensors must have the IDENTICAL TensorSpec + memory config (the sender
  addresses the receiver's pages with its own accessor), hence ``allocate`` below
  reproduces the model hook's staging recipe exactly (``from_torch`` zeros, TILE,
  DRAM interleaved, replicated to the (1,1) mesh);
* send-before-recv and recv-before-send both PARK the kernel on its CQ until the
  peer arrives (legal; bit-exact); only the ITEM ORDER on the channel must match;
* program cache key = (socket, tensor spec) with addresses re-patched per launch,
  so one warm send/recv per spec covers every buffer of that spec;
* ``FABRIC_2D`` must be set BEFORE the mesh opens (``set_fabric_config_for_pd``,
  called by the rank entry point's ``open_mesh_device`` wrapper; the plugin ignores
  ``fabric_config`` for 1-device meshes today) and the opened mesh is handed to the
  transport through ``register_mesh_device`` (same wrapper, before
  ``FabricSocketTransport.start()`` runs inside ``attach_runner``);
* ``ttnn.MeshSocket(mesh, cfg)`` runs INLINE on the engine thread: it allocates the
  socket's L1 on every worker core and its own handshake window is a fixed 10 s, so a
  helper thread would only race the caller's device teardown.  Only the world barrier
  (no native timeout) gets the ``_timed`` bound; on that timeout the caller must NOT
  deallocate device tensors (``RendezvousTimeout``): the helper may still be inside
  the MPI call and the process is about to exit anyway.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

try:  # vLLM's logger tree when the plugin is installed; plain logging otherwise
    from vllm_tt_plugin.logger import init_tt_logger

    logger = init_tt_logger(__name__)
except Exception:  # pragma: no cover
    logger = logging.getLogger(__name__)

SpecKey = tuple[tuple[int, ...], str, str]  # (shape, dtype name, layout name)

DEFAULT_PACKET_BYTES = 8704  # measured cross-process 58.8 GB/s (PHASE3_MICROBENCH 3)
DEFAULT_FIFO_BYTES = 128  # direct mode: the 64 B handshake page, one L1 page


# --- mesh registry ------------------------------------------------------------------
#
# ``TTKVConnector.attach_runner`` hands the transport no mesh device (tt_connector.py
# ``make_transport(...)`` takes config only).  The rank entry point registers the
# mesh it opened here; tests pass ``mesh_device=`` to the transport directly.

_REGISTERED_MESH: Any | None = None
_REGISTRY_LOCK = threading.Lock()


def register_mesh_device(mesh: Any) -> None:
    """Record the process's opened mesh for ``FabricSocketTransport.start()``."""
    global _REGISTERED_MESH
    with _REGISTRY_LOCK:
        _REGISTERED_MESH = mesh


def registered_mesh_device() -> Any | None:
    with _REGISTRY_LOCK:
        return _REGISTERED_MESH


def clear_registered_mesh_device() -> None:
    register_mesh_device(None)


# Started fabric transports of this process, so the rank entry point's
# ``close_mesh_device`` wrapper can shut them down BEFORE the mesh closes when
# vLLM's own teardown path did not (a MeshSocket / pool tensor released after its
# mesh is gone is a use-after-close on the device).  ``FabricSocketTransport.start()``
# registers itself, ``shutdown()`` unregisters.
_STARTED_TRANSPORTS: list[Any] = []


def register_transport(transport: Any) -> None:
    with _REGISTRY_LOCK:
        if transport not in _STARTED_TRANSPORTS:
            _STARTED_TRANSPORTS.append(transport)


def unregister_transport(transport: Any) -> None:
    with _REGISTRY_LOCK:
        if transport in _STARTED_TRANSPORTS:
            _STARTED_TRANSPORTS.remove(transport)


def registered_transports() -> list[Any]:
    with _REGISTRY_LOCK:
        return list(_STARTED_TRANSPORTS)


def shutdown_registered_transports() -> list[Any]:
    """``shutdown()`` every transport still registered (each unregisters itself);
    returns the ones that were still up.  Used by the mesh-close wrapper; a
    transport that raises is logged and dropped from the registry."""
    done: list[Any] = []
    for t in registered_transports():
        try:
            t.shutdown()
        except Exception:  # noqa: BLE001 - the mesh is closing regardless
            logger.exception("fabric transport %r shutdown raised at mesh close", t)
        unregister_transport(t)
        done.append(t)
    return done


# --- the seam -----------------------------------------------------------------------


class SocketLayer(Protocol):
    """Every ttnn / fabric / MPI call the fabric transport makes."""

    # distributed context (MPI world of the tt-run job)
    def is_distributed(self) -> bool: ...

    def rank(self) -> int: ...

    def size(self) -> int: ...

    def barrier(self, timeout_s: float | None = None) -> None:
        """World barrier; ``RuntimeError`` when the peer does not arrive in time."""

    # socket
    def create_socket(
        self,
        mesh: Any,
        *,
        connections: int,
        fifo_bytes: int,
        sender_rank: int,
        receiver_rank: int,
        timeout_s: float | None = None,
    ) -> Any:
        """``ttnn.MeshSocket(mesh, cfg)``; both ranks must call within the window."""

    def close_socket(self, sock: Any) -> None: ...

    # tensors
    def allocate(self, mesh: Any, spec: SpecKey) -> Any:
        """A zero device tensor of ``spec`` with the hook's staging recipe."""

    def deallocate(self, t: Any) -> None: ...

    def spec_of(self, t: Any) -> SpecKey: ...

    def relayout_copy(self, blk: Any, dst: Any) -> None:
        """Block-major ``[32,4,64,256]`` chunk -> head-major ``[1,4,2048,256]`` ``dst``
        (the hook's ``_relayout_into_staging`` twin; bfp8 bit-exact)."""

    def copy(self, src: Any, dst: Any) -> None:
        """Device copy between two tensors of one spec (``ttnn.copy``)."""

    def copy_host_to_device(self, host: Any, dst: Any) -> None: ...

    def send(self, t: Any, sock: Any) -> None:
        """``send_direct_async``: enqueue only (parks on the CQ until the recv)."""

    def recv(self, t: Any, sock: Any) -> None:
        """``recv_direct_async``: enqueue only (parks on the CQ until the send)."""

    def sync(self, mesh: Any) -> None:
        """``ttnn.synchronize_device`` (all CQs)."""

    def num_program_cache_entries(self, mesh: Any) -> int: ...


class RendezvousTimeout(RuntimeError):
    """A ``_timed`` call did not return: its helper thread is still inside the
    native call (abandoned, daemon).  The caller must not touch device state that
    the call may still be using; the process exits with the error."""


def _timed(fn: Callable[[], Any], timeout_s: float | None, what: str) -> Any:
    """Run ``fn`` with a wall-clock bound: ``timeout_s`` None/<= 0 runs it inline;
    otherwise in a helper thread that is abandoned (daemon) on timeout so the caller
    can raise ``RendezvousTimeout`` instead of hanging inside a rendezvous forever.
    Only used for the world barrier (no native timeout, touches no device tensor)."""
    if not timeout_s or timeout_s <= 0:
        return fn()
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["result"] = fn()
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller
            box["error"] = e

    th = threading.Thread(target=run, name=f"tt_pd_fabric_{what}", daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        raise RendezvousTimeout(
            f"fabric {what} did not complete within {timeout_s:.0f} s: the peer "
            "rank never reached the rendezvous (is the other engine alive and on "
            "the same tt-run job?)"
        )
    if "error" in box:
        raise box["error"]
    return box.get("result")


# --- real adapter --------------------------------------------------------------------


class TtnnSocketLayer:
    """``SocketLayer`` over ttnn; ``import ttnn`` happens here, lazily."""

    def __init__(self) -> None:
        import ttnn  # noqa: PLC0415 - lazy by design (no ttnn in the API server)

        self.ttnn = ttnn
        if not hasattr(ttnn.experimental, "send_direct_async") or not hasattr(
            ttnn.experimental, "recv_direct_async"
        ):
            raise RuntimeError(
                "this ttnn build has no experimental.send_direct_async / "
                "recv_direct_async: the fabric transport needs the MeshSocket ops "
                "(tt-metal branch qwen36-pd-disagg or newer)"
            )
        if not hasattr(ttnn, "MeshSocket"):
            raise RuntimeError("this ttnn build has no MeshSocket")

    # distributed context
    def is_distributed(self) -> bool:
        fn = getattr(self.ttnn, "distributed_context_is_initialized", None)
        return bool(fn()) if fn is not None else False

    def rank(self) -> int:
        return int(self.ttnn.distributed_context_get_rank())

    def size(self) -> int:
        return int(self.ttnn.distributed_context_get_size())

    def barrier(self, timeout_s: float | None = None) -> None:
        # elapsed is logged so the first pair bring-up can confirm the binding
        # releases the GIL (a barrier that cannot time out would show as a hang here)
        t0 = time.perf_counter()
        _timed(self.ttnn.distributed_context_barrier, timeout_s, "barrier")
        logger.info(
            "fabric barrier (rank %d) took %.1f ms",
            self.rank(),
            (time.perf_counter() - t0) * 1e3,
        )

    # socket
    def create_socket(
        self,
        mesh: Any,
        *,
        connections: int,
        fifo_bytes: int,
        sender_rank: int,
        receiver_rank: int,
        timeout_s: float | None = None,
    ) -> Any:
        ttnn = self.ttnn
        coord = ttnn.MeshCoordinate(0, 0)  # submesh-local; (1,1) mesh per rank
        conns = [
            ttnn.SocketConnection(
                ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(i, 0)),  # sender core
                ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(i, 1)),  # receiver core
            )
            for i in range(int(connections))
        ]
        cfg = ttnn.SocketConfig(
            conns,
            ttnn.SocketMemoryConfig(ttnn.BufferType.L1, int(fifo_bytes)),
            sender_rank=int(sender_rank),
            receiver_rank=int(receiver_rank),
        )
        # INLINE on the caller's thread: the constructor allocates L1 on every worker
        # core and has its own fixed 10 s handshake window (it raises on its own);
        # ``timeout_s`` is accepted for the seam but a helper thread would let the
        # caller tear the pool tensors down under a constructor still running.
        del timeout_s
        t0 = time.perf_counter()
        sock = ttnn.MeshSocket(mesh, cfg)
        logger.info(
            "fabric MeshSocket created (%d connections, fifo %d B) in %.1f ms",
            len(conns),
            int(fifo_bytes),
            (time.perf_counter() - t0) * 1e3,
        )
        return sock

    def close_socket(self, sock: Any) -> None:
        close = getattr(sock, "close", None)
        if callable(close):
            close()

    # tensors
    @staticmethod
    def _torch_dtype(dtype: str) -> Any:
        import torch  # noqa: PLC0415

        return torch.float32 if dtype == "float32" else torch.bfloat16

    def allocate(self, mesh: Any, spec: SpecKey) -> Any:
        import torch  # noqa: PLC0415

        ttnn = self.ttnn
        shape, dtype, layout = spec
        # EXACTLY the hook's staging recipe (kv_transfer.py warmup_kv_transfer):
        # direct mode needs the identical TensorSpec on both ends.
        return ttnn.from_torch(
            torch.zeros(*shape, dtype=self._torch_dtype(dtype)),
            dtype=getattr(ttnn.DataType, dtype.upper()),
            layout=getattr(ttnn.Layout, layout.upper()),
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def deallocate(self, t: Any) -> None:
        self.ttnn.deallocate(t)

    def spec_of(self, t: Any) -> SpecKey:
        return (
            tuple(int(x) for x in t.shape),
            t.dtype.name.lower(),
            t.layout.name.upper(),
        )

    def relayout_copy(self, blk: Any, dst: Any) -> None:
        ttnn = self.ttnn
        hm = ttnn.permute(blk, (1, 0, 2, 3))
        hm2 = ttnn.reshape(hm, tuple(int(d) for d in dst.shape))  # tile-aligned view
        ttnn.copy(hm2, dst)
        ttnn.deallocate(hm2)
        del hm

    def copy(self, src: Any, dst: Any) -> None:
        self.ttnn.copy(src, dst)

    def copy_host_to_device(self, host: Any, dst: Any) -> None:
        self.ttnn.copy_host_to_device_tensor(host, dst)

    def send(self, t: Any, sock: Any) -> None:
        self.ttnn.experimental.send_direct_async(t, sock)

    def recv(self, t: Any, sock: Any) -> None:
        self.ttnn.experimental.recv_direct_async(t, sock)

    def sync(self, mesh: Any) -> None:
        self.ttnn.synchronize_device(mesh)

    def num_program_cache_entries(self, mesh: Any) -> int:
        try:
            return int(mesh.num_program_cache_entries())
        except Exception:  # pragma: no cover - older bindings
            return -1


# --- fabric bring-up (rank entry point) --------------------------------------------


def set_fabric_config_for_pd(
    packet_bytes: int = DEFAULT_PACKET_BYTES,
    reliability: str = "STRICT_INIT",
    fabric_config: str = "FABRIC_2D",
) -> None:
    """The validated 7-arg ``ttnn.set_fabric_config`` call (p3_two_process_socket.py
    lines 65-67) that must run BEFORE ``open_mesh_device`` in each rank.  The plugin's
    ``set_fabric`` is a no-op for 1-device meshes, so the rank entry point's
    ``open_mesh_device`` wrapper (``pd_fabric_rank.install_mesh_hooks``) calls this."""
    import ttnn  # noqa: PLC0415

    rc = ttnn.FabricRouterConfig()
    rc.max_packet_payload_size_bytes = int(packet_bytes)
    ttnn.set_fabric_config(
        getattr(ttnn.FabricConfig, str(fabric_config)),
        getattr(ttnn.FabricReliabilityMode, str(reliability)),
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        rc,
    )
    logger.info(
        "fabric config set: %s %s packet %d B",
        fabric_config,
        reliability,
        int(packet_bytes),
    )


__all__ = [
    "DEFAULT_FIFO_BYTES",
    "DEFAULT_PACKET_BYTES",
    "RendezvousTimeout",
    "SocketLayer",
    "SpecKey",
    "TtnnSocketLayer",
    "clear_registered_mesh_device",
    "register_mesh_device",
    "register_transport",
    "registered_mesh_device",
    "registered_transports",
    "set_fabric_config_for_pd",
    "shutdown_registered_transports",
    "unregister_transport",
]
