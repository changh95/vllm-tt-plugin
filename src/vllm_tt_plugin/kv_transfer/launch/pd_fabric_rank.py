# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Rank entry point of the Phase 3 PD pair: a headless ``EngineCoreProc`` per MPI rank.

Launched by tt-run on BOTH ranks with the same argv (PHASE3_NOTES 2.4)::

    python -m ttnn.distributed.ttrun --bare --rank-binding <rendered yaml> -- \
        <venv python> -m vllm_tt_plugin.kv_transfer.launch.pd_fabric_rank \
        --pd-args <pd_node_args.json>

Per rank (F0 shape, stock vLLM 0.26.0, no fork after the mesh opens):

1. Role from the environment: ``PD_ROLE`` (env_overrides) must agree with the fixed
   table rank 0 = prefill (MeshSocket ``sender_rank``), rank 1 = decode
   (``receiver_rank``); rank/size from ``OMPI_COMM_WORLD_RANK/SIZE`` (fallback
   ``TT_RUN_RANK``); ``PD_RPC_PORT`` = the front-end's ``--data-parallel-rpc-port``.
2. Replay the role's ``vllm serve`` argv from ``pd_node_args.json`` through vLLM's
   own serve parser -> ``AsyncEngineArgs.from_cli_args(ns).create_engine_config(
   OPENAI_API_SERVER)`` (identical VllmConfig to the front-end's), then force the
   single-engine parallel fields (``data_parallel_size=1, size_local=1, rank=0,
   rank_local=0, distributed_executor_backend="uni"``).
3. Take the engine identity lock (``pd_launch_config.EngineLock`` on
   ``(segment_dir, engine_id)``): refuse to start while another live process holds
   the same shm segment dir + engine id (its producer sweep would destroy our
   segments and ours its), held until the rank exits.
4. Install the ``open_mesh_device`` / ``close_mesh_device`` wrappers on
   ``vllm_tt_plugin.worker``: the open wrapper sets ``FABRIC_2D`` + router packet
   size right before the mesh opens (``transport.fabric_socket.
   set_fabric_config_for_pd``; the stock plugin ignores ``fabric_config`` for
   1-device meshes; skipped for ``TRANSPORT=shm`` without ``SHM_FABRIC=1``, whose
   argv carries no ``tt.fabric_config``) and registers the opened mesh with the
   fabric transport's registry (``transport.fabric_socket.register_mesh_device``,
   read by ``FabricSocketTransport.start()`` inside ``attach_runner``, which passes
   no mesh to ``make_transport``); the close wrapper shuts down every fabric
   transport still registered BEFORE the mesh closes (``fabric_socket.
   shutdown_registered_transports``) and clears the mesh registry.  Both are hard
   dependencies: an import failure raises instead of falling back.
5. ``EngineCoreProc.run_engine_core(vllm_config, local_client=False,
   handshake_address="tcp://<dp_address>:<PD_RPC_PORT>", executor_class=UniProcExecutor,
   log_stats, dp_rank=0, local_dp_rank=0)``: HELLO (headless=True) to the front-end
   (5 min window -> start the front-ends FIRST), engine identity b"\\x00\\x00",
   ``TTWorker.init_device`` -> ``open_mesh_device`` (ControlPlane collectives across
   the two ranks, window TT_METAL_OPERATION_TIMEOUT_SECONDS), KV caches,
   ``attach_runner`` -> ``transport.start()`` (barrier + MeshSocket, 10 s window),
   warm-up, READY.

Shutdown order (SIGTERM/SIGINT from ``run_pd_pair_fabric.sh down``, or the busy loop
ending): vLLM's handler ends ``run_busy_loop`` with ``SystemExit`` and the
surrounding ``finally`` runs ``EngineCore.shutdown()`` -> ``TTWorker.shutdown()``:
``ensure_kv_transfer_shutdown`` (transport.shutdown: socket closed, pools freed,
rendezvous file unlinked) -> ``close_mesh_device`` (our wrapper: any transport still
registered is shut down first) -> ``cleanup_dist_env_and_memory``.  That last call
ends in ``torch.accelerator.empty_cache()``, which raises ``RuntimeError: Cannot
access accelerator device when none is available`` on the CPU-only torch of a TT
host and would turn the clean ``SystemExit`` into a non-zero exit (prterun then
aborts the WHOLE job, SIGKILLing the peer rank mid mesh-close).
``install_dist_cleanup_patch`` makes the rank's ``EngineCore.shutdown`` run that
cleanup with the accelerator cache release skipped when torch has no accelerator.
``run_headless_engine`` maps the ``SystemExit`` to exit code 0 (peer keeps closing
its mesh; the MPI job ends when both ranks have exited and MPI finalizes at
interpreter exit), releases the engine lock, and returns non-zero only for a real
failure.

``--dry-run`` builds and prints the resolved config without MPI, ttnn or a device.
Fallback if this shape fails on hardware: F2 (single engine, ``create_socket_pair`` on
two (1,1) submeshes) -- see profiles/pd/p3_device_validation_plan.md; nothing here
implements it because stock 0.26 supports F0 end-to-end (understanding stage).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from .pd_launch_config import (
    KV_ROLE_OF_ROLE,
    RANK_OF_ROLE,
    ROLE_OF_RANK,
    ROLES,
    WORLD_SIZE,
    EngineLock,
    engine_in_use,
    load_node_args,
    segment_dir_of,
    validate_argv_for_role,
)

log = logging.getLogger("vllm_tt_plugin.pd_fabric_rank")

_FABRIC_SET = False


# --------------------------------------------------------------------------- #
# environment -> rank context (pure)
# --------------------------------------------------------------------------- #
@dataclass
class RankContext:
    role: str
    rank: int
    world_size: int
    rpc_port: int
    mesh_id: int | None
    visible_devices: str | None
    metal_cache: str | None
    under_mpi: bool

    @property
    def kv_role(self) -> str:
        return KV_ROLE_OF_ROLE[self.role]


def resolve_rank_env(
    env: dict[str, str],
    *,
    role_override: str | None = None,
    rpc_override: int | None = None,
    allow_no_mpi: bool = False,
) -> RankContext:
    """Derive role/rank/port from the tt-run environment and cross-check them.

    Raises ``RuntimeError`` unless world size == 2 and (role, rank) is
    (prefill, 0) or (decode, 1).  Without MPI (``allow_no_mpi``, dry-run) the rank
    is taken from the role.
    """
    rank_s = env.get(
        "OMPI_COMM_WORLD_RANK", env.get("PMI_RANK", env.get("TT_RUN_RANK"))
    )
    size_s = env.get("OMPI_COMM_WORLD_SIZE", env.get("PMI_SIZE"))
    under_mpi = rank_s is not None and size_s is not None
    role = role_override or env.get("PD_ROLE")
    if not under_mpi:
        if not allow_no_mpi:
            raise RuntimeError(
                "pd_fabric_rank must run under tt-run/MPI "
                "(OMPI_COMM_WORLD_RANK/SIZE unset); "
                "use --dry-run to build the config without MPI"
            )
        if role is None:
            raise RuntimeError("PD_ROLE (or --role) is required without MPI")
        if role not in ROLES:
            raise RuntimeError(f"PD_ROLE must be one of {ROLES}, got {role!r}")
        rank = RANK_OF_ROLE[role]
        size = WORLD_SIZE
    else:
        rank = int(rank_s)  # type: ignore[arg-type]
        size = int(size_s)  # type: ignore[arg-type]
        if size != WORLD_SIZE:
            raise RuntimeError(
                f"PD pair needs exactly {WORLD_SIZE} MPI ranks, world size is {size}"
            )
        if rank not in ROLE_OF_RANK:
            raise RuntimeError(f"rank {rank} has no PD role (ranks 0/1 only)")
        if role is None:
            role = ROLE_OF_RANK[rank]
        if role not in ROLES:
            raise RuntimeError(f"PD_ROLE must be one of {ROLES}, got {role!r}")
        if ROLE_OF_RANK[rank] != role:
            raise RuntimeError(
                f"rank {rank} is bound to PD_ROLE={role!r} but the MeshSocket role "
                f"table requires rank {RANK_OF_ROLE[role]} for {role} "
                "(sender_rank=0 prefill, receiver_rank=1 decode)"
            )
    tt_run_rank = env.get("TT_RUN_RANK")
    if tt_run_rank is not None and under_mpi and int(tt_run_rank) != rank:
        raise RuntimeError(f"TT_RUN_RANK={tt_run_rank} disagrees with OMPI rank {rank}")
    mesh_id_s = env.get("TT_MESH_ID")
    mesh_id = int(mesh_id_s) if mesh_id_s is not None else None
    if mesh_id is not None and mesh_id != rank:
        raise RuntimeError(
            f"TT_MESH_ID={mesh_id} must equal the rank {rank} "
            "(mesh 0 = prefill, mesh 1 = decode)"
        )
    rpc_s = env.get("PD_RPC_PORT") if rpc_override is None else str(rpc_override)
    if rpc_s is None:
        raise RuntimeError("PD_RPC_PORT is required (rank binding env_overrides)")
    rpc_port = int(rpc_s)
    if not 1024 <= rpc_port <= 65535:
        raise RuntimeError(f"PD_RPC_PORT {rpc_port} out of range")
    return RankContext(
        role=role,
        rank=rank,
        world_size=size,
        rpc_port=rpc_port,
        mesh_id=mesh_id,
        visible_devices=env.get("TT_VISIBLE_DEVICES"),
        metal_cache=env.get("TT_METAL_CACHE"),
        under_mpi=under_mpi,
    )


def argv_for_role(node_args: dict[str, Any], role: str) -> list[str]:
    if role not in node_args:
        raise ValueError(f"pd_node_args has no entry for role {role!r}")
    return list(node_args[role])


# --------------------------------------------------------------------------- #
# vllm config (imports vllm)
# --------------------------------------------------------------------------- #
def build_vllm_config(argv: list[str]) -> tuple[Any, Any]:
    """Replay the serve argv as ``vllm serve`` does -> (VllmConfig, AsyncEngineArgs)."""
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.usage.usage_lib import UsageContext
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = make_arg_parser(
        FlexibleArgumentParser(description="pd_fabric_rank (vllm serve argv replay)")
    )
    ns = parser.parse_args(argv)
    # ServeSubcommand.cmd: the positional model_tag wins over --model
    if getattr(ns, "model_tag", None) is not None:
        ns.model = ns.model_tag
    validate_parsed_serve_args(ns)
    engine_args = AsyncEngineArgs.from_cli_args(ns)
    cfg = engine_args.create_engine_config(usage_context=UsageContext.OPENAI_API_SERVER)
    return cfg, engine_args


def force_single_engine(parallel_config: Any) -> Any:
    """Each node is its own dp_size=1 world with engine identity 0.

    The front-end's ParallelConfig has size_local=0 (remote engine); the rank's must
    say 1 local engine, rank 0 (run_engine_core re-forces the first three anyway,
    core.py:1296-1298); uniproc executor = nothing forks after the mesh opens.
    """
    pc = parallel_config
    pc.data_parallel_size = 1
    pc.data_parallel_size_local = 1
    pc.data_parallel_rank = 0
    pc.data_parallel_rank_local = 0
    backend = getattr(pc, "distributed_executor_backend", None)
    if backend not in (None, "uni"):
        raise RuntimeError(
            f"distributed_executor_backend must be 'uni' in the rank, got {backend!r}"
        )
    pc.distributed_executor_backend = "uni"
    return pc


def check_config_for_role(cfg: Any, ctx: RankContext) -> dict[str, Any]:
    """Cross-check the built VllmConfig against the rank context; return a summary."""
    ktc = cfg.kv_transfer_config
    if ktc is None:
        raise RuntimeError("kv_transfer_config missing from the replayed argv")
    kv_role = str(getattr(ktc.kv_role, "value", ktc.kv_role))
    if kv_role != ctx.kv_role:
        raise RuntimeError(
            f"kv_role {kv_role!r} != {ctx.kv_role!r} for PD_ROLE={ctx.role}"
        )
    extra = dict(getattr(ktc, "kv_connector_extra_config", None) or {})
    if extra.get("transport") not in ("fabric", "shm"):
        raise RuntimeError(
            "kv_connector_extra_config.transport "
            f"{extra.get('transport')!r} is neither 'fabric' (MeshSocket data plane) "
            "nor 'shm' (Phase 2 dumpfile data plane in the same headless-rank shape)"
        )
    if not segment_dir_of(extra):
        raise RuntimeError(
            f"transport {extra['transport']} has no segment dir in "
            "kv_connector_extra_config (shm_dir / fabric_control_dir)"
        )
    pc = cfg.parallel_config
    if int(pc.data_parallel_rpc_port) != ctx.rpc_port:
        raise RuntimeError(
            f"data_parallel_rpc_port {pc.data_parallel_rpc_port} != "
            f"PD_RPC_PORT {ctx.rpc_port}"
        )
    from vllm_tt_plugin.config import get_tt_config

    tt = get_tt_config(cfg)
    return {
        "role": ctx.role,
        "rank": ctx.rank,
        "engine_id": ktc.engine_id,
        "kv_role": kv_role,
        "transport": extra.get("transport"),
        "control_dir": extra.get(
            "fabric_control_dir", extra.get("control_dir", extra.get("shm_dir"))
        ),
        "segment_dir": segment_dir_of(extra),
        "handshake": handshake_address(cfg),
        "dp": {
            "size": pc.data_parallel_size,
            "size_local": pc.data_parallel_size_local,
            "rank": pc.data_parallel_rank,
            "rank_local": pc.data_parallel_rank_local,
            "backend": pc.distributed_executor_backend,
        },
        "model": cfg.model_config.model,
        "max_model_len": cfg.model_config.max_model_len,
        "max_num_seqs": cfg.scheduler_config.max_num_seqs,
        "block_size": cfg.cache_config.block_size,
        "tt": {
            k: tt.get(k)
            for k in (
                "fabric_config",
                "fabric_reliability_mode",
                "fabric_max_packet_payload_bytes",
                "trace_region_size",
                "l1_small_size",
            )
        },
        "env": {
            k: os.environ.get(k)
            for k in (
                "TT_MESH_ID",
                "TT_VISIBLE_DEVICES",
                "TT_METAL_CACHE",
                "TT_MESH_GRAPH_DESC_PATH",
                "MESH_DEVICE",
                "TT_CACHE_PATH",
                "QWEN36_MAX_TOKENS_ALL_USERS",
                "QWEN36_GDN_DECODE_FUSED",
                "TT_PD_FABRIC_DIR",
                "TT_PD_STRICT_SHAPES",
                # DEBUG knob: visible here so a leaked value shows in the boot
                # summary, not only in the transport's WARNING deep in the log
                "TT_PD_FABRIC_SKIP_WARMUP",
            )
        },
    }


def handshake_address(cfg: Any) -> str:
    from vllm.utils.network_utils import get_tcp_uri

    pc = cfg.parallel_config
    return get_tcp_uri(pc.data_parallel_master_ip, int(pc.data_parallel_rpc_port))


# --------------------------------------------------------------------------- #
# mesh hooks (ttnn is imported only inside fabric_socket's callables)
# --------------------------------------------------------------------------- #
def _default_set_fabric(
    fabric_config: str, reliability: str, packet_bytes: int
) -> None:
    """The validated 7-arg ``ttnn.set_fabric_config`` call, owned by the transport's
    device adapter (``fabric_socket.set_fabric_config_for_pd``)."""
    from vllm_tt_plugin.kv_transfer.transport import fabric_socket

    fabric_socket.set_fabric_config_for_pd(
        int(packet_bytes),
        reliability=str(reliability),
        fabric_config=str(fabric_config),
    )


def _default_register_mesh(mesh: Any) -> None:
    """Hand the opened mesh to ``FabricSocketTransport.start()`` through the ONE
    registry it reads (``fabric_socket.register_mesh_device``)."""
    from vllm_tt_plugin.kv_transfer.transport import fabric_socket

    fabric_socket.register_mesh_device(mesh)
    got = fabric_socket.registered_mesh_device()
    if got is not mesh:
        raise RuntimeError(
            f"fabric mesh registry did not take the opened mesh: registered {got!r}, "
            f"opened {mesh!r}"
        )


def _default_distributed_info() -> dict[str, Any]:
    import ttnn

    try:
        if not ttnn.distributed_context_is_initialized():
            return {"initialized": False}
        return {
            "initialized": True,
            "rank": int(ttnn.distributed_context_get_rank()),
            "size": int(ttnn.distributed_context_get_size()),
        }
    except Exception as exc:  # pragma: no cover - device only
        return {"initialized": None, "error": repr(exc)}


def _default_shutdown_transports() -> list[Any]:
    from vllm_tt_plugin.kv_transfer.transport import fabric_socket

    done = fabric_socket.shutdown_registered_transports()
    fabric_socket.clear_registered_mesh_device()
    return done


def install_mesh_hooks(
    *,
    fabric_config: str = "FABRIC_2D",
    reliability: str = "STRICT_INIT",
    packet_bytes: int = 8704,
    expected_rank: int | None = None,
    set_fabric: Callable[[str, str, int], None] | None = None,
    register_mesh: Callable[[Any], None] | None = None,
    distributed_info: Callable[[], dict[str, Any]] | None = None,
    shutdown_transports: Callable[[], list[Any]] | None = None,
    worker_module: Any = None,
    enable_fabric: bool = True,
) -> Callable[..., Any]:
    """Wrap ``vllm_tt_plugin.worker.open_mesh_device`` AND ``close_mesh_device``
    (both looked up as module globals by ``TTWorker``).

    Open: set the fabric config BEFORE the mesh opens, strip ``fabric_config`` from
    the tt_config handed to the original (so a future plugin that honours it at
    num_devices==1 does not set it twice), verify the ttnn distributed context sees
    this rank, and register the mesh with ``fabric_socket`` (``TTWorker.init_device``
    -> this wrapper -> KV caches -> ``attach_runner`` ->
    ``FabricSocketTransport.start()`` reads the registry).

    Close: shut down every fabric transport still registered (``shutdown_transports``,
    default ``fabric_socket.shutdown_registered_transports`` + clearing the mesh
    registry) and only then close the mesh.  On the orderly path
    ``TTWorker.shutdown`` already ran ``ensure_kv_transfer_shutdown`` (the transport
    unregistered itself, nothing to do); on any other path (``__del__``, a raise
    between ``attach_runner`` and the connector's shutdown) this keeps the socket
    and the pool tensors from outliving their mesh.  Returns the open wrapper.
    """
    if worker_module is None:
        import vllm_tt_plugin.worker as worker_module  # type: ignore[no-redef]
    set_fabric = set_fabric or _default_set_fabric
    register_mesh = register_mesh or _default_register_mesh
    distributed_info = distributed_info or _default_distributed_info
    shutdown_transports = shutdown_transports or _default_shutdown_transports
    orig = worker_module.open_mesh_device
    if getattr(orig, "_pd_fabric_wrapped", False):
        return orig
    orig_close = getattr(worker_module, "close_mesh_device", None)
    if orig_close is None:
        raise RuntimeError(
            "vllm_tt_plugin.worker has no close_mesh_device: TTWorker's mesh teardown "
            "moved; re-check pd_fabric_rank.install_mesh_hooks (transport shutdown "
            "must run before the mesh closes)"
        )

    def close_mesh_device_pd(mesh: Any, tt_config: Any) -> Any:
        t0 = time.perf_counter()
        still_up = shutdown_transports()
        if still_up:
            log.warning(
                "pd_fabric_rank: %d fabric transport(s) were still started at mesh "
                "close and have been shut down first (%.1f ms): %s",
                len(still_up),
                (time.perf_counter() - t0) * 1e3,
                [type(t).__name__ for t in still_up],
            )
        log.info("pd_fabric_rank: closing mesh (transports down, registry cleared)")
        return orig_close(mesh, tt_config)

    close_mesh_device_pd._pd_fabric_wrapped = True  # type: ignore[attr-defined]
    close_mesh_device_pd._pd_fabric_orig = orig_close  # type: ignore[attr-defined]
    worker_module.close_mesh_device = close_mesh_device_pd

    def open_mesh_device_pd(tt_config: Any, trace_mode: Any, local_dp_rank: int = 0):
        global _FABRIC_SET
        tt_config = dict(tt_config or {})
        fc = tt_config.pop("fabric_config", fabric_config) or fabric_config
        rel = tt_config.pop("fabric_reliability_mode", reliability) or reliability
        pkt = int(
            tt_config.pop("fabric_max_packet_payload_bytes", packet_bytes)
            or packet_bytes
        )
        if enable_fabric and not _FABRIC_SET:
            t0 = time.perf_counter()
            set_fabric(fc, rel, pkt)
            _FABRIC_SET = True
            log.info(
                "pd_fabric_rank: fabric %s %s pkt=%d set in %.1f ms (before mesh open)",
                fc,
                rel,
                pkt,
                (time.perf_counter() - t0) * 1e3,
            )
        t0 = time.perf_counter()
        mesh = orig(tt_config, trace_mode, local_dp_rank)
        info = distributed_info()
        log.info(
            "pd_fabric_rank: mesh opened in %.2f s; distributed context %s; "
            "TT_MESH_ID=%s TT_VISIBLE_DEVICES=%s",
            time.perf_counter() - t0,
            info,
            os.environ.get("TT_MESH_ID"),
            os.environ.get("TT_VISIBLE_DEVICES"),
        )
        if info.get("initialized") is False:
            raise RuntimeError(
                "ttnn distributed context not initialized after mesh open: "
                "not running under MPI?"
            )
        if expected_rank is not None and info.get("rank") not in (None, expected_rank):
            raise RuntimeError(
                f"ttnn distributed rank {info.get('rank')} != expected {expected_rank}"
            )
        if info.get("size") not in (None, WORLD_SIZE):
            raise RuntimeError(
                f"ttnn distributed world size {info.get('size')} != {WORLD_SIZE}"
            )
        register_mesh(mesh)
        return mesh

    open_mesh_device_pd._pd_fabric_wrapped = True  # type: ignore[attr-defined]
    open_mesh_device_pd._pd_fabric_orig = orig  # type: ignore[attr-defined]
    worker_module.open_mesh_device = open_mesh_device_pd
    return open_mesh_device_pd


# --------------------------------------------------------------------------- #
# shutdown path (see the module docstring, "Shutdown order")
# --------------------------------------------------------------------------- #
def install_dist_cleanup_patch(
    core_module: Any = None, torch_module: Any = None
) -> Callable[[], None]:
    """Replace ``vllm.v1.engine.core.cleanup_dist_env_and_memory`` (the reference
    ``EngineCore.shutdown`` calls) with a wrapper that runs the original with
    ``torch.accelerator.empty_cache`` swapped for a no-op while torch reports no
    accelerator (the CPU-only torch of a TT host; the call raises ``RuntimeError:
    Cannot access accelerator device when none is available`` and would replace the
    busy loop's clean ``SystemExit``).  Everything else the original does (model
    parallel / distributed environment destroy, gc) still runs.  Idempotent; returns
    the wrapper.  Raises if the vLLM module has no such attribute (a vLLM whose
    shutdown path moved needs this wrapper re-checked, not silently skipped)."""
    if core_module is None:
        import vllm.v1.engine.core as core_module  # type: ignore[no-redef]
    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]
    orig = getattr(core_module, "cleanup_dist_env_and_memory", None)
    if orig is None:
        raise RuntimeError(
            "vllm.v1.engine.core has no cleanup_dist_env_and_memory: the EngineCore "
            "shutdown path changed; re-check pd_fabric_rank.install_dist_cleanup_patch"
        )
    if getattr(orig, "_pd_fabric_wrapped", False):
        return orig

    def cleanup_dist_env_tt(*args: Any, **kwargs: Any) -> Any:
        acc = getattr(torch_module, "accelerator", None)
        is_available = getattr(acc, "is_available", None)
        if acc is None or (callable(is_available) and is_available()):
            return orig(*args, **kwargs)  # a real accelerator backend: stock path
        real_empty_cache = acc.empty_cache

        def empty_cache_noop() -> None:
            log.info(
                "pd_fabric_rank: torch has no accelerator on this host; skipping "
                "torch.accelerator.empty_cache() in cleanup_dist_env_and_memory"
            )

        acc.empty_cache = empty_cache_noop
        try:
            return orig(*args, **kwargs)
        finally:
            acc.empty_cache = real_empty_cache

    cleanup_dist_env_tt._pd_fabric_wrapped = True  # type: ignore[attr-defined]
    cleanup_dist_env_tt._pd_fabric_orig = orig  # type: ignore[attr-defined]
    core_module.cleanup_dist_env_and_memory = cleanup_dist_env_tt
    return cleanup_dist_env_tt


def run_headless_engine(
    run_engine_core: Callable[[], Any],
    *,
    lock: EngineLock | None = None,
    role: str = "?",
    rank: int = -1,
) -> int:
    """Run ``run_engine_core`` (``EngineCoreProc.run_engine_core`` bound to its
    arguments) and map its outcome to the rank's exit code:

    * ``SystemExit`` with code ``None`` / 0 (the busy loop ended after SIGTERM /
      SIGINT and ``EngineCore.shutdown`` completed) or a plain return -> 0;
    * ``SystemExit(code)`` with a non-zero code -> that code;
    * any other exception -> logged with traceback, 1.

    The engine lock is released in every case, after the engine (and with it the
    mesh) is gone.  A non-zero exit makes prterun abort the whole MPI job, so a
    clean SIGTERM shutdown MUST come out as 0 here or the peer rank is SIGKILLed
    while it is still closing its mesh."""
    code = 0
    try:
        run_engine_core()
        log.info("pd_fabric_rank: engine core returned (role=%s rank=%d)", role, rank)
    except SystemExit as exc:
        raw = exc.code
        if raw is None or raw == 0:
            log.info(
                "pd_fabric_rank: engine core exited cleanly (role=%s rank=%d)",
                role,
                rank,
            )
        else:
            code = raw if isinstance(raw, int) else 1
            log.error(
                "pd_fabric_rank: engine core exited with %r (role=%s rank=%d)",
                raw,
                role,
                rank,
            )
    except BaseException:  # noqa: BLE001 - the rank's outcome is the exit code
        code = 1
        log.exception(
            "pd_fabric_rank: engine core failed (role=%s rank=%d)", role, rank
        )
    finally:
        if lock is not None:
            lock.release()
    return code


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="PD fabric pair: headless EngineCoreProc rank"
    )
    ap.add_argument(
        "--pd-args",
        default=os.environ.get("PD_ARGS_JSON"),
        help="pd_node_args.json ({prefill: argv, decode: argv}); "
        "default env PD_ARGS_JSON",
    )
    ap.add_argument(
        "--role", choices=ROLES, default=None, help="override PD_ROLE (dry-run)"
    )
    ap.add_argument(
        "--rpc-port", type=int, default=None, help="override PD_RPC_PORT (dry-run)"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="build + print the config; no MPI, no ttnn, no device",
    )
    ap.add_argument(
        "--packet-bytes", type=int, default=int(os.environ.get("PD_FABRIC_PKT", "8704"))
    )
    ap.add_argument(
        "--reliability", default=os.environ.get("PD_FABRIC_RELIABILITY", "STRICT_INIT")
    )
    ap.add_argument(
        "--fabric-config", default=os.environ.get("PD_FABRIC_CONFIG", "FABRIC_2D")
    )
    ap.add_argument(
        "--no-fabric-wrapper",
        action="store_true",
        help="do not set the fabric config from the wrapper "
        "(the plugin honours tt.fabric_config itself)",
    )
    ap.add_argument("--log-level", default=os.environ.get("VLLM_LOGGING_LEVEL", "INFO"))
    return ap.parse_args(argv)


def prepare(
    args: argparse.Namespace, env: dict[str, str] | None = None
) -> tuple[RankContext, list[str], dict[str, Any]]:
    """Everything before vllm is imported: context, argv, pure argv checks."""
    env = dict(os.environ if env is None else env)
    if not args.pd_args:
        raise RuntimeError("--pd-args (or PD_ARGS_JSON) is required")
    ctx = resolve_rank_env(
        env,
        role_override=args.role,
        rpc_override=args.rpc_port,
        allow_no_mpi=args.dry_run,
    )
    node_args = load_node_args(args.pd_args)
    argv = argv_for_role(node_args, ctx.role)
    summary = validate_argv_for_role(argv, ctx.role, ctx.rpc_port)
    meta = node_args.get("meta") or {}
    fab = meta.get("fabric") or {}
    summary["tag"] = str(meta.get("tag") or "")
    # precedence: additional_config.tt (what both nodes' VllmConfigs carry) > node-args
    # meta (what the launch script rendered) > CLI/env defaults.  The wrapper reads the
    # tt keys again at mesh-open time; these values are its fallback.
    args.packet_bytes = int(
        summary.get("packet_bytes") or fab.get("packet_bytes") or args.packet_bytes
    )
    args.reliability = str(
        summary.get("reliability") or fab.get("reliability") or args.reliability
    )
    args.fabric_config = str(
        summary.get("fabric_config") or fab.get("config") or args.fabric_config
    )
    # TRANSPORT=shm without SHM_FABRIC=1: the argv carries no tt.fabric_config and
    # the mesh opens exactly as run_pd_pair.sh's does (no fabric config at all)
    if not summary.get("fabric_enabled"):
        args.no_fabric_wrapper = True
    return ctx, argv, summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=False,
    )
    ctx, serve_args, summary = prepare(args)
    log.info(
        "pd_fabric_rank: role=%s rank=%d/%d rpc=%d mesh_id=%s TT_VISIBLE_DEVICES=%s "
        "TT_METAL_CACHE=%s "
        "TT_MESH_GRAPH_DESC_PATH=%s argv_checks=%s",
        ctx.role,
        ctx.rank,
        ctx.world_size,
        ctx.rpc_port,
        ctx.mesh_id,
        ctx.visible_devices,
        ctx.metal_cache,
        os.environ.get("TT_MESH_GRAPH_DESC_PATH"),
        json.dumps(summary),
    )
    segment_dir, engine_id = summary["segment_dir"], summary["engine_id"]
    in_use = engine_in_use(segment_dir, engine_id)
    lock: EngineLock | None = None
    if not args.dry_run:
        # before vllm is imported: refuse fast when another live engine owns this
        # (segment_dir, engine_id) -- our producer sweep would destroy its segments
        lock = EngineLock(
            segment_dir, engine_id, role=ctx.role, tag=str(summary.get("tag") or "")
        ).acquire()
        log.info("pd_fabric_rank: engine lock %s taken (pid %d)", lock.path, lock.pid)

    try:
        return _main_locked(args, ctx, serve_args, summary, lock, in_use)
    except BaseException:
        # everything up to the engine start; run_headless_engine releases itself
        if lock is not None:
            lock.release()
        raise


def _main_locked(
    args: argparse.Namespace,
    ctx: RankContext,
    serve_args: list[str],
    summary: dict[str, Any],
    lock: EngineLock | None,
    in_use: str | None,
) -> int:
    t0 = time.perf_counter()
    cfg, engine_args = build_vllm_config(serve_args)
    force_single_engine(cfg.parallel_config)
    info = check_config_for_role(cfg, ctx)
    log.info(
        "pd_fabric_rank: VllmConfig built in %.1f s: %s",
        time.perf_counter() - t0,
        json.dumps(info, default=str),
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "context": asdict(ctx),
                    "engine_lock": {
                        "segment_dir": summary["segment_dir"],
                        "engine_id": summary["engine_id"],
                        "in_use": in_use,
                    },
                    "fabric_wrapper": not args.no_fabric_wrapper,
                    **info,
                },
                indent=1,
                default=str,
            )
        )
        return 0

    if not ctx.under_mpi:
        raise RuntimeError("refusing to open a device outside tt-run/MPI")
    install_mesh_hooks(
        fabric_config=args.fabric_config,
        reliability=args.reliability,
        packet_bytes=args.packet_bytes,
        expected_rank=ctx.rank,
        enable_fabric=not args.no_fabric_wrapper,
    )
    install_dist_cleanup_patch()

    from vllm.v1.engine.core import EngineCoreProc
    from vllm.v1.executor.abstract import UniProcExecutor

    address = handshake_address(cfg)
    log.info(
        "pd_fabric_rank: starting headless EngineCoreProc (identity 0) -> handshake %s "
        "(front-end must be up; HELLO window 5 min); fabric wrapper %s",
        address,
        "on" if not args.no_fabric_wrapper else "off (no tt.fabric_config)",
    )
    return run_headless_engine(
        lambda: EngineCoreProc.run_engine_core(
            vllm_config=cfg,
            local_client=False,
            handshake_address=address,
            executor_class=UniProcExecutor,
            log_stats=not getattr(engine_args, "disable_log_stats", False),
            dp_rank=0,
            local_dp_rank=0,
        ),
        lock=lock,
        role=ctx.role,
        rank=ctx.rank,
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
