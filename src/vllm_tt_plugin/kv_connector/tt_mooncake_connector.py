# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""TTMooncakeConnector: prefill/decode disaggregation for TT models over the Mooncake
Transfer Engine.  Pull model, host-staged. The prefill instance
(``kv_role=kv_producer``) prefills N-1 prompt tokens (the decoder recomputes the last
one from the transferred state), then stages the request's paged-KV blocks and its
model-internal per-slot state (the Qwen3.x GDN recurrent + conv state) into one
contiguous host buffer registered with Mooncake. The decode instance
(``kv_consumer``) learns the buffer's address over a ZMQ side channel, pulls it with
``transfer_sync_read`` (same host: TCP; cross-host: RDMA) -- or, when both run on one
host, maps the producer's /dev/shm-backed staging buffer read-only and imports straight
out of it (no copy; ``QWEN36_PD_SHM=0`` disables) -- writes the KV blocks into
its own paged cache, parks the state snapshot for the runner to write into the
request's decode slot when the request gets one, and runs the last prompt token as an
ordinary decode step.  vLLM plumbing follows the in-tree Mooncake/NIXL connectors
(scheduler side: mamba/GDN prompt truncation, ``kv_transfer_params`` round trip
through the proxy). The worker side is TT-specific: KV lives in ttnn tensors on a
mesh the model owns, TP is model-internal (vLLM TP=1), and vLLM sees one
FullAttentionSpec group; the model exposes ``export_kv_blocks`` /
``import_kv_blocks`` and parks GDN snapshots under ``pd_gdn_capture`` (tt-metal
``models/demos/blackhole/qwen36/tt/pd_transfer.py``).  Config (``--kv-transfer-
config`` JSON)::  {"kv_connector": "TTMooncakeConnector", "kv_connector_module_path":
"vllm_tt_plugin.kv_connector.tt_mooncake_connector", "kv_role": "kv_producer" |
"kv_consumer", "kv_connector_extra_config": {"side_channel_host": "127.0.0.1",
"side_channel_port": 18100, "mooncake_protocol": "tcp", "mooncake_device": ""}}  The
producer's ``side_channel_host``/``side_channel_port`` are what it advertises to
decoders in the returned ``kv_transfer_params``; a consumer needs no static port.  A
proxy may pick the ``transfer_id`` itself (``kv_transfer_params.transfer_id`` on the
producer request; the producer echoes it) and post to the consumer before the
producer has answered: the consumer derives ``num_tokens`` from its own tokenization
and its GET blocks on the side channel until the producer has staged."""

from __future__ import annotations

import contextlib
import json
import math
import mmap
import os
import queue
import socket
import threading
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import zmq
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

from vllm_tt_plugin.logger import init_tt_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_tt_logger(__name__)

_DEFAULT_SIDE_CHANNEL_PORT = 18100
# The producer holds a GET for a not-yet-staged transfer this long before answering
# "pending" (must stay below the consumer's 5 s REQ receive timeout); the consumer then
# re-polls every _GET_POLL_S until _GET_TIMEOUT_S.
_GET_WAIT_S = 4.0
_GET_POLL_S = 0.02
_GET_TIMEOUT_S = 600.0
_SHM_DIR = "/dev/shm"
_SHM_PREFIX = "qwen36-pd-"


def shm_enabled() -> bool:
    """Same-host zero-copy hand-off: the producer backs its staging buffers with
    /dev/shm files and a consumer on the same host maps them instead of pulling a
    copy. ``QWEN36_PD_SHM=0`` disables it on either side."""
    return os.environ.get("QWEN36_PD_SHM", "1") != "0" and os.path.isdir(_SHM_DIR)


def host_identity() -> str:
    """Identity a consumer compares with its own before trying a producer's shm
    segment: hostname plus the kernel boot id. Two containers on one machine share
    the boot id but normally neither the hostname nor /dev/shm, and the segment must
    also open, so a false match only costs a failed open (then the Mooncake pull runs
    as usual)."""
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            boot = f.read().strip()
    except OSError:
        boot = ""
    return f"{socket.gethostname()}:{boot}"


def shm_segment_name_is_valid(name: str) -> bool:
    return (
        isinstance(name, str)
        and name.startswith(_SHM_PREFIX)
        and "/" not in name
        and name not in (".", "..")
    )


def map_shm_segment(name: str) -> torch.Tensor | None:
    """Map a producer's staging segment read-only as one uint8 tensor (the whole file).
    Returns ``None`` when the segment cannot be opened (other host, producer gone, shm
    disabled there)."""
    if not shm_segment_name_is_valid(name):
        return None
    path = os.path.join(_SHM_DIR, name)
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            mm = mmap.mmap(
                fd,
                0,
                flags=mmap.MAP_SHARED | getattr(mmap, "MAP_POPULATE", 0),
                prot=mmap.PROT_READ,
            )
        finally:
            os.close(fd)
    except (OSError, ValueError):
        return None
    with warnings.catch_warnings():
        # torch warns that a read-only buffer must not be written; the consumer only
        # reads (views + borrowed uploads)
        warnings.simplefilter("ignore", UserWarning)
        return torch.frombuffer(mm, dtype=torch.uint8)


def unlink_stale_shm_segments(pids_alive=None) -> list[str]:
    """Remove ``/dev/shm/qwen36-pd-<pid>-*`` files whose producer process is gone (a
    producer killed with SIGKILL never ran ``shutdown``). Returns the removed names."""
    removed = []
    try:
        names = os.listdir(_SHM_DIR)
    except OSError:
        return removed
    for name in names:
        if not name.startswith(_SHM_PREFIX):
            continue
        try:
            pid = int(name[len(_SHM_PREFIX) :].split("-", 1)[0])
        except ValueError:
            continue
        alive = pid in pids_alive if pids_alive is not None else _pid_alive(pid)
        if alive:
            continue
        try:
            os.unlink(os.path.join(_SHM_DIR, name))
            removed.append(name)
        except OSError:
            pass
    return removed


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# --------------------------------------------------------------------------------------
# payload packing (host)
# --------------------------------------------------------------------------------------


def _dtype_name(t: torch.Tensor) -> str:
    return str(t.dtype).replace("torch.", "")


def _check_gdn_snapshot(rec_snap, conv_snap) -> None:
    """The GDN snapshot is the device-major pair the model's
    ``_snapshot_gdn_scratch_host`` produces: ``rec`` ``[n_dev, L, Nv, Dk, Dv]`` and
    ``taps`` ``[n_dev, L, K, C]``."""
    if not (isinstance(rec_snap, torch.Tensor) and isinstance(conv_snap, torch.Tensor)):
        raise TypeError(
            "GDN snapshot must be a (rec, taps) tensor pair; got "
            f"{type(rec_snap).__name__}, {type(conv_snap).__name__}"
        )
    if rec_snap.dim() != 5 or conv_snap.dim() != 4:
        shapes = f"{tuple(rec_snap.shape)} and {tuple(conv_snap.shape)}"
        raise ValueError(
            "GDN snapshot: expected rec [n_dev, L, Nv, Dk, Dv] and "
            f"taps [n_dev, L, K, C], got {shapes}"
        )
    if rec_snap.shape[:2] != conv_snap.shape[:2]:
        raise ValueError(
            "GDN snapshot: rec and taps disagree on [n_dev, L]: "
            f"{tuple(rec_snap.shape[:2])} vs {tuple(conv_snap.shape[:2])}"
        )


def pack_payload(
    kv,
    rec_snap,
    conv_snap,
    num_tokens: int,
    n_blocks: int,
    out: torch.Tensor | None = None,
):
    """Pack one request's KV pairs (per attention layer) and GDN snapshot into one uint8
    buffer.  Returns ``(buffer, header)``; ``header`` describes every tensor (name,
    dtype, shape, offset). With ``out`` (a pooled buffer of at least the payload
    size) the bytes are written into ``out`` and ``out`` is returned; use
    ``payload_nbytes`` to size it. The GDN snapshot travels as two tensors,
    ``gdn.rec`` ``[n_dev, L, Nv, Dk, Dv]`` and ``gdn.taps`` ``[n_dev, L, K, C]``
    (device-major: one memcpy each here, one borrowed upload each on the decoder)."""
    _check_gdn_snapshot(rec_snap, conv_snap)
    entries: list[dict[str, Any]] = []
    tensors: list[torch.Tensor] = []
    off = 0

    def add(name, t):
        nonlocal off
        t = t.contiguous()
        n = t.numel() * t.element_size()
        entries.append(
            {
                "name": name,
                "dtype": _dtype_name(t),
                "shape": list(t.shape),
                "offset": off,
                "nbytes": n,
            }
        )
        tensors.append(t)
        off += n

    for li, (k, v) in enumerate(kv):
        add(f"kv.{li}.k", k)
        add(f"kv.{li}.v", v)
    add("gdn.rec", rec_snap)
    add("gdn.taps", conv_snap)
    if out is not None:
        if out.numel() < off:
            raise ValueError(f"pooled buffer {out.numel()} B < payload {off} B")
        buf = out
    else:
        buf = torch.empty(max(off, 64), dtype=torch.uint8)
    for e, t in zip(entries, tensors):
        buf[e["offset"] : e["offset"] + e["nbytes"]].copy_(t.view(-1).view(torch.uint8))
    header = {
        "num_tokens": int(num_tokens),
        "n_blocks": int(n_blocks),
        "n_attn_layers": len(kv),
        "n_gdn_layers": int(rec_snap.shape[1]),
        "n_conv": int(conv_snap.shape[2]),
        "nbytes": int(off),
        "tensors": entries,
    }
    return buf, header


def payload_nbytes(kv, rec_snap, conv_snap) -> int:
    _check_gdn_snapshot(rec_snap, conv_snap)
    n = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in kv)
    n += rec_snap.numel() * rec_snap.element_size()
    n += conv_snap.numel() * conv_snap.element_size()
    return n


class _HostBufferPool:
    """Host uint8 buffers, page-faulted once and registered once with the Mooncake
    engine, reused across requests (per-request allocate + register + first-touch
    cost 20-30 ms and halved the TCP pull rate). With ``shm=True`` (producer, see
    ``shm_enabled``) every buffer is a MAP_SHARED mapping of a fresh
    ``/dev/shm/qwen36-pd-<pid>-<id>`` file, so a consumer on this host can map the same
    bytes instead of pulling them; the pointer registered with Mooncake is the same
    mapping, so a consumer elsewhere still pulls over TCP/RDMA."""

    _MIN = 256 << 20
    _STEP = 64 << 20

    def __init__(
        self, engine, engine_lock: threading.Lock, name: str, shm: bool = False
    ):
        self.engine, self.engine_lock, self.name = engine, engine_lock, name
        self.shm = shm
        self._free: list[torch.Tensor] = []
        self._lock = threading.Lock()
        self.total = 0
        self._shm_names: dict[int, str] = {}  # buffer data_ptr -> segment name
        self._shm_maps: list[mmap.mmap] = []

    def _alloc(self, cap: int) -> torch.Tensor:
        if not self.shm:
            b = torch.empty(cap, dtype=torch.uint8)
            b.fill_(0)  # fault the pages in now, not inside the transfer
            return b
        name = f"{_SHM_PREFIX}{os.getpid()}-{uuid.uuid4().hex[:8]}"
        path = os.path.join(_SHM_DIR, name)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, cap)
            mm = mmap.mmap(
                fd, cap, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE
            )
        finally:
            os.close(fd)
        b = torch.frombuffer(mm, dtype=torch.uint8)
        b.fill_(0)  # fault the tmpfs pages in now
        self._shm_maps.append(mm)
        self._shm_names[b.data_ptr()] = name
        return b

    def acquire(self, nbytes: int) -> torch.Tensor:
        with self._lock:
            fits = [i for i, b in enumerate(self._free) if b.numel() >= nbytes]
            if fits:
                i = min(fits, key=lambda j: self._free[j].numel())
                return self._free.pop(
                    i
                )  # by index: list.remove() would compare tensors element-wise
        cap = max(self._MIN, -(-nbytes // self._STEP) * self._STEP)
        b = self._alloc(cap)
        with self.engine_lock:
            rc = self.engine.register_memory(b.data_ptr(), cap)
        if rc != 0:
            raise RuntimeError(f"register_memory({cap}) failed ({rc})")
        self.total += cap
        logger.info(
            "[pd] %s buffer pool: +%.0f MiB (total %.1f GiB)%s",
            self.name,
            cap / 2**20,
            self.total / 2**30,
            f" shm {self._shm_names[b.data_ptr()]}" if self.shm else "",
        )
        return b

    def release(self, b: torch.Tensor) -> None:
        with self._lock:
            self._free.append(b)

    def shm_name(self, b: torch.Tensor) -> str | None:
        """The /dev/shm segment ``b`` is a mapping of (``None`` for a plain buffer)."""
        return self._shm_names.get(b.data_ptr())

    def close(self) -> None:
        """Unlink the pool's shm files (the mappings stay valid until dropped)."""
        for name in self._shm_names.values():
            with contextlib.suppress(OSError):
                os.unlink(os.path.join(_SHM_DIR, name))
        self._shm_names.clear()


def payload_digest(buf: torch.Tensor, nbytes: int) -> str:
    """Cheap content fingerprint: sha1 over every 4097th byte plus the length (for
    alone-vs-concurrent determinism checks of a request's staged state)."""
    import hashlib

    sample = buf[:nbytes:4097].numpy().tobytes()
    return hashlib.sha1(sample + nbytes.to_bytes(8, "little")).hexdigest()[:12]


def unpack_payload(buf: torch.Tensor, header: dict[str, Any]):
    """Inverse of ``pack_payload``: views into ``buf`` (no copies). Returns
    ``(kv, rec, taps)`` with the GDN pair in the device-major layout ``pack_payload``
    documents."""
    by_name = {}
    for e in header["tensors"]:
        dt = getattr(torch, e["dtype"])
        by_name[e["name"]] = (
            buf[e["offset"] : e["offset"] + e["nbytes"]].view(dt).view(*e["shape"])
        )
    kv = [
        (by_name[f"kv.{li}.k"], by_name[f"kv.{li}.v"])
        for li in range(header["n_attn_layers"])
    ]
    if "gdn.rec" not in by_name or "gdn.taps" not in by_name:
        names = sorted(n for n in by_name if n.startswith("gdn."))
        raise ValueError(
            "payload GDN snapshot is not the device-major (gdn.rec, gdn.taps) pair; "
            f"got {names[:4]}{'...' if len(names) > 4 else ''} (producer/consumer "
            "version mismatch?)"
        )
    return kv, by_name["gdn.rec"], by_name["gdn.taps"]


# --------------------------------------------------------------------------------------
# scheduler <-> worker metadata
# --------------------------------------------------------------------------------------


@dataclass
class StageReq:
    """Producer: one request whose state must be staged after this step's prefill."""

    req_id: str
    block_ids: list[int]
    num_tokens: int
    transfer_id: str = ""  # the side-channel key (see _SchedulerSide.transfer_id)


@dataclass
class RecvReq:
    """Consumer: one request whose state must be pulled from a producer."""

    req_id: str
    block_ids: list[int]
    remote_host: str
    remote_port: int
    transfer_id: str
    num_tokens: int


@dataclass
class TTMooncakeConnectorMetadata(KVConnectorMetadata):
    stage: list[StageReq] = field(default_factory=list)
    recv: list[RecvReq] = field(default_factory=list)
    # consumer requests aborted before their pull started: tell the producer to drop the
    # staging
    cancel: list[tuple[str, int, str]] = field(
        default_factory=list
    )  # (host, port, transfer_id)


# --------------------------------------------------------------------------------------
# connector
# --------------------------------------------------------------------------------------


class TTMooncakeConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        cfg = vllm_config.kv_transfer_config
        if cfg is None:
            raise ValueError("TTMooncakeConnector requires --kv-transfer-config")
        self._is_producer = cfg.kv_role == "kv_producer"
        self._is_consumer = cfg.kv_role == "kv_consumer"
        if not (self._is_producer or self._is_consumer):
            raise ValueError(
                "TTMooncakeConnector needs kv_role kv_producer or kv_consumer (no "
                "kv_both yet)"
            )
        extra = cfg.kv_connector_extra_config or {}
        self._side_host = str(extra.get("side_channel_host", "127.0.0.1"))
        self._side_port = int(
            extra.get("side_channel_port", _DEFAULT_SIDE_CHANNEL_PORT)
        )
        self._protocol = str(extra.get("mooncake_protocol", "tcp"))
        self._device_name = str(extra.get("mooncake_device", ""))
        self._block_size = vllm_config.cache_config.block_size
        self._scheduler: _SchedulerSide | None = None
        self._worker: _WorkerSide | None = None
        if role == KVConnectorRole.SCHEDULER:
            self._scheduler = _SchedulerSide(self)
        else:
            self._worker = _WorkerSide(self)

    # ---- TT-specific: the worker binds the runner once the model + KV caches exist
    # ----
    def _worker_side(self) -> _WorkerSide:
        if self._worker is None:
            raise RuntimeError(
                f"TTMooncakeConnector role {self.role} has no worker side"
            )
        return self._worker

    def _scheduler_side(self) -> _SchedulerSide:
        if self._scheduler is None:
            raise RuntimeError(
                f"TTMooncakeConnector role {self.role} has no scheduler side"
            )
        return self._scheduler

    def _meta(self) -> TTMooncakeConnectorMetadata:
        meta = self._get_connector_metadata()
        if not isinstance(meta, TTMooncakeConnectorMetadata):
            raise TypeError(f"unexpected connector metadata {type(meta).__name__}")
        return meta

    def bind_tt_runner(self, runner) -> None:
        self._worker_side().bind_runner(runner)

    def post_warmup(self) -> None:
        """Called by the TT worker after the model's own warmup/trace capture (consumer:
        pre-capture the per-slot GDN import traces so no request pays the compile +
        capture)."""
        self._worker_side().post_warmup()

    # ---- worker side ----
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        pass  # TT KV caches are ttnn tensors owned by the model; see bind_tt_runner

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        self._worker_side().start_step(self._meta())

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata, **kwargs: Any
    ) -> None:
        pass

    def wait_for_save(self):
        self._worker_side().stage_after_step(self._meta())

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        return self._worker_side().take_finished(finished_req_ids)

    def shutdown(self):
        if self._worker is not None:
            self._worker.shutdown()

    # ---- scheduler side ----
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        return self._scheduler_side().get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        self._scheduler_side().update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return self._scheduler_side().build_connector_meta(scheduler_output)

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        return self._scheduler_side().request_finished(request, block_ids)

    def update_connector_output(self, connector_output) -> None:
        self._scheduler_side().update_connector_output(connector_output)


# --------------------------------------------------------------------------------------
# scheduler side
# --------------------------------------------------------------------------------------


class _SchedulerSide:
    def __init__(self, c: TTMooncakeConnector):
        self.c = c
        self._to_stage: dict[str, StageReq] = {}
        self._to_recv: dict[str, RecvReq] = {}
        self._to_cancel: list[tuple[str, int, str]] = []
        self._staged_params: dict[
            str, dict[str, Any]
        ] = {}  # producer: req_id -> params handed to the proxy

    @staticmethod
    def _truncate_for_prefill(request: Request) -> None:
        """Producer: drop the last prompt token so the prefill computes h(N-1); the
        decoder recomputes token N-1."""
        params = request.kv_transfer_params
        if (
            params is None
            or params.get("_p_side_truncated")
            or request.num_prompt_tokens <= 1
        ):
            return
        if request.prompt_token_ids is not None:
            request.prompt_token_ids.pop()
        elif request.prompt_embeds is not None:
            request.prompt_embeds = request.prompt_embeds[:-1]
        else:
            return
        request._all_token_ids.pop()
        request.num_prompt_tokens -= 1
        request.max_tokens = 1
        params["_p_side_truncated"] = True

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        params = request.kv_transfer_params
        if not params:
            return 0, False
        if params.get("do_remote_prefill"):
            if self.c._is_producer:
                raise ValueError(
                    "a kv_producer instance received a do_remote_prefill request"
                )
            n = len(request.prompt_token_ids or [])
            count = (n - 1 if n > 1 else n) - num_computed_tokens
            if count > 0:
                return count, True
        if params.get("do_remote_decode") and self.c._is_producer:
            self._truncate_for_prefill(request)
        return 0, False

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        params = request.kv_transfer_params
        if not params:
            return
        if params.get("do_remote_prefill"):
            needed = ("remote_host", "remote_port", "transfer_id")
            if all(k in params for k in needed):
                block_ids = (
                    list(blocks.get_block_ids()[0]) if num_external_tokens > 0 else []
                )
                # num_tokens = the producer's truncated prompt length. A proxy that
                # posts to the consumer before the producer answered cannot send it;
                # it is the same tokenization minus the last token (the count
                # get_num_new_matched_tokens reported), and the worker checks it
                # against the payload header.
                if params.get("num_tokens") is not None:
                    num_tokens = int(params["num_tokens"])
                else:
                    n = len(request.prompt_token_ids or [])
                    num_tokens = n - 1 if n > 1 else n
                self._to_recv[request.request_id] = RecvReq(
                    req_id=request.request_id,
                    block_ids=block_ids,
                    remote_host=str(params["remote_host"]),
                    remote_port=int(params["remote_port"]),
                    transfer_id=str(params["transfer_id"]),
                    num_tokens=num_tokens,
                )
            else:
                logger.warning(
                    "TTMooncakeConnector: incomplete kv_transfer_params %s; no "
                    "transfer for %s",
                    params,
                    request.request_id,
                )
            params["do_remote_prefill"] = False  # one transfer per request
        elif params.get("do_remote_decode") and self.c._is_producer:
            self._to_stage[request.request_id] = StageReq(
                req_id=request.request_id,
                block_ids=list(blocks.get_block_ids()[0]),
                num_tokens=int(request.num_prompt_tokens),
                transfer_id=self.transfer_id(request),
            )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        # Pending entries are re-shipped every step until the worker reports them in
        # finished_sending / finished_recving (update_connector_output): the TT
        # scheduler may build a prefill-only SchedulerOutput, drop it when it scheduled
        # no tokens (a remote-KV request only transitions to WAITING_FOR_REMOTE_KVS) and
        # schedule decode-only instead -- metadata handed out once and cleared would be
        # lost with the discarded output. The worker de-duplicates.
        meta = TTMooncakeConnectorMetadata()
        meta.stage = list(self._to_stage.values())
        meta.recv = list(self._to_recv.values())
        meta.cancel = list(self._to_cancel)
        self._to_cancel.clear()
        return meta

    def update_connector_output(self, connector_output) -> None:
        for req_id in connector_output.finished_recving or ():
            self._to_recv.pop(req_id, None)
        for req_id in connector_output.finished_sending or ():
            self._to_stage.pop(req_id, None)

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        params = request.kv_transfer_params
        if not params:
            return False, None
        self._to_recv.pop(request.request_id, None)
        self._to_stage.pop(request.request_id, None)
        if params.get("do_remote_prefill"):
            # aborted before it was ever scheduled: nothing was allocated; tell the
            # producer to drop its staging
            if all(k in params for k in ("remote_host", "remote_port", "transfer_id")):
                self._to_cancel.append(
                    (
                        str(params["remote_host"]),
                        int(params["remote_port"]),
                        str(params["transfer_id"]),
                    )
                )
            params["do_remote_prefill"] = False
            return False, None
        if not params.get("do_remote_decode") or not self.c._is_producer:
            return False, None
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            # aborted / stopped early: the worker staged (or will stage) nothing useful;
            # free now
            return False, None
        # The worker stages synchronously in wait_for_save of the prefill step and
        # reports the request in finished_sending in that same step's output; the
        # scheduler frees the blocks from that report.
        return True, {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_host": self.c._side_host,
            "remote_port": self.c._side_port,
            "transfer_id": self.transfer_id(request),
            "num_tokens": int(request.num_prompt_tokens),
        }

    @staticmethod
    def transfer_id(request: Request) -> str:
        """The id the staging is filed under. A proxy that fans out to the consumer
        before this instance answered chooses it (``kv_transfer_params.transfer_id``)
        because it cannot predict ``request.request_id``: vLLM's InputProcessor
        appends ``-<8 hex>`` to the ``X-Request-Id``-derived id, differently on every
        instance. Without one, the engine request id (the serial proxy round trip)."""
        params = request.kv_transfer_params or {}
        tid = params.get("transfer_id")
        return str(tid) if tid else str(request.request_id)


# --------------------------------------------------------------------------------------
# worker side
# --------------------------------------------------------------------------------------


@dataclass
class _Staged:
    buf: torch.Tensor
    addr: int
    nbytes: int
    header: dict[str, Any]
    t_staged: float
    shm_name: str | None = None  # the /dev/shm segment ``buf`` maps (same-host path)


@dataclass
class _Fetched:
    """Consumer: one pulled (or mapped) payload waiting for the main thread."""

    rr: RecvReq
    buf: torch.Tensor  # exactly the payload bytes (a view for a mapped segment)
    header: dict[str, Any]
    release: Any  # callable: the consumer is done with ``buf``
    via: str  # "shm" | "pull"
    t_wait: float
    t_pull: float


class _PullAborted(Exception):
    """The consumer request finished (client hang-up) while its pull was in flight."""


class _WorkerSide:
    def __init__(self, c: TTMooncakeConnector):
        self.c = c
        self.runner = None
        self.model = None  # the inner Qwen36Model
        self.engine = None
        self.pool: _HostBufferPool | None = None
        self.local_host = "127.0.0.1"
        self.rpc_port = 0
        self._lock = threading.Lock()
        self._engine_lock = (
            threading.Lock()
        )  # Mooncake TransferEngine calls are serialized
        self._finished_sending: set[str] = set()
        self._finished_recving: set[str] = set()
        # de-dup sets for re-shipped metadata (pruned when the request finishes)
        self._stage_done: set[str] = set()
        self._recv_done: set[str] = set()
        # producer
        self._staged: dict[str, _Staged] = {}
        self._zmq_thread: threading.Thread | None = None
        self._stop = threading.Event()
        # consumer
        self._pool: ThreadPoolExecutor | None = None
        self._inflight: dict[str, RecvReq] = {}
        self._aborted: set[str] = set()  # in-flight pulls whose request finished
        self._fetched: queue.Queue[_Fetched] = queue.Queue()
        self._failed: queue.Queue[tuple[RecvReq, BaseException]] = queue.Queue()
        self._shm_ok = shm_enabled()
        self._host_id = host_identity()
        self._shm_segments: dict[str, torch.Tensor] = {}  # mapped producer segments
        self._shm_lock = threading.Lock()
        self.stats = {
            "staged": 0,
            "staged_bytes": 0,
            "stage_ms": 0.0,
            "pulled": 0,
            "pulled_bytes": 0,
            "pull_ms": 0.0,
            "import_ms": 0.0,
        }

    # ---- binding ----
    def bind_runner(self, runner) -> None:
        from mooncake.engine import TransferEngine

        self.runner = runner
        wrapper = runner.model
        if os.environ.get("QWEN36_PD_ALLOW_PTRACE") == "1":
            # Debug aid: let any process of this user attach a sampling profiler (py-
            # spy) to the engine core, which vLLM spawns detached from the operator's
            # shell (Yama ptrace_scope=1 only allows ancestors). PR_SET_PTRACER
            # (0x59616d61), PTRACER_ANY (-1).
            import ctypes

            ctypes.CDLL(None).prctl(0x59616D61, ctypes.c_long(-1), 0, 0, 0)
            logger.info(
                "[pd] QWEN36_PD_ALLOW_PTRACE=1: engine core is attachable by py-spy"
            )
        inner = getattr(wrapper, "model", None)
        self.model = inner[0] if isinstance(inner, (list, tuple)) else inner
        if not hasattr(self.model, "prefill_paged_slots"):
            raise TypeError(
                f"TTMooncakeConnector: model {type(self.model).__name__} has no "
                f"prefill_paged_slots (needs the qwen36 TP model)"
            )
        self.engine = TransferEngine()
        local_host = self.c._side_host if self.c._is_producer else "127.0.0.1"
        rc = self.engine.initialize(
            local_host, "P2PHANDSHAKE", self.c._protocol, self.c._device_name
        )
        if rc != 0:
            raise RuntimeError(f"Mooncake TransferEngine.initialize failed ({rc})")
        self.rpc_port = int(self.engine.get_rpc_port())
        self.local_host = local_host
        self.pool = _HostBufferPool(
            self.engine,
            self._engine_lock,
            "staging" if self.c._is_producer else "receive",
            shm=self.c._is_producer and self._shm_ok,
        )
        if self.c._is_producer:
            # park each prefilled request's GDN snapshot under its slot; never write
            # decode slots (P never decodes)
            self.model.pd_gdn_capture = {}
            self.model.pd_skip_gdn_slot_write = True
            self._zmq_thread = threading.Thread(
                target=self._serve_side_channel, name="tt-pd-side-channel", daemon=True
            )
            self._zmq_thread.start()
            logger.info(
                "TTMooncakeConnector producer: mooncake %s:%d, side channel %s:%d, "
                "staging %s",
                local_host,
                self.rpc_port,
                self.c._side_host,
                self.c._side_port,
                "/dev/shm (same-host consumers map it)" if self.pool.shm else "malloc",
            )
        else:
            self._pool = ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="tt-pd-pull"
            )
            runner.pd_pending_gdn = {}
            logger.info(
                "TTMooncakeConnector consumer: mooncake %s:%d, same-host shm %s",
                local_host,
                self.rpc_port,
                "on" if self._shm_ok else "off",
            )

    def post_warmup(self):
        if (
            self.c._is_producer
            and os.environ.get("QWEN36_PD_EXPORT_WARMUP", "1") == "1"
        ):
            from models.demos.blackhole.qwen36.tt import pd_transfer

            pd_transfer.export_warmup(
                self.model,
                max_bucket=int(os.environ.get("QWEN36_PD_EXPORT_WARMUP_MAX", "2048")),
            )
            return
        if not self.c._is_consumer:
            return
        from models.demos.blackhole.qwen36.tt import pd_transfer

        if os.environ.get("QWEN36_PD_IMPORT_WARMUP", "1") == "1":
            # compile the KV import programs of every block bucket now, not inside the
            # first request
            pd_transfer.import_warmup(
                self.model,
                max_bucket=int(os.environ.get("QWEN36_PD_IMPORT_WARMUP_MAX", "2048")),
            )
        if (
            os.environ.get("QWEN36_PD_GDN_IMPORT", "trace") != "trace"
            or os.environ.get("QWEN36_PD_GDN_PRECAPTURE", "1") != "1"
        ):
            return

        n_slots = int(getattr(self.runner, "tt_per_lane_max_num_seqs", 0) or 0)
        t0 = time.perf_counter()
        pd_transfer.get_traced_importer(self.model).precapture(range(n_slots))
        logger.info(
            "[pd] pre-captured %d GDN import traces in %.1f s",
            n_slots,
            time.perf_counter() - t0,
        )

    def shutdown(self):
        self._stop.set()
        if self._pool is not None:
            self._pool.shutdown(wait=False)
        if self.pool is not None:
            self.pool.close()

    # ---- producer ----
    def _get_reply(self, tid: str, host_id: str) -> dict[str, Any] | None:
        with self._lock:
            st = self._staged.get(tid)
        if st is None:
            return None
        rep = {
            "status": "ok",
            "segment": f"{self.local_host}:{self.rpc_port}",
            "addr": st.addr,
            "nbytes": st.nbytes,
            "header": st.header,
            "host": host_id,
            "shm": None,
        }
        if st.shm_name is not None:
            # ``buf`` is the pooled buffer itself, so the payload starts at 0; carry
            # the offset anyway so the consumer never assumes it
            rep["shm"] = {"name": st.shm_name, "offset": 0, "nbytes": st.nbytes}
        return rep

    @staticmethod
    def _router_send(sock, ident: bytes, rep: dict[str, Any]) -> None:
        # REQ clients expect [empty delimiter, body]; ROUTER prepends the identity
        sock.send_multipart([ident, b"", json.dumps(rep).encode()])

    def _serve_side_channel(self):
        """ROUTER loop: GET for a transfer that is not staged yet is parked (up to
        _GET_WAIT_S) and answered as soon as stage_after_step files it, so a consumer
        that was posted to concurrently with the producer does not poll; other
        clients are served meanwhile (a REP socket would block them)."""
        ctx = zmq.Context()
        sock = ctx.socket(zmq.ROUTER)
        sock.bind(f"tcp://{self.c._side_host}:{self.c._side_port}")
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        waiters: dict[str, list[tuple[bytes, float]]] = {}  # tid -> [(ident, deadline)]
        host_id = host_identity()
        while not self._stop.is_set():
            try:
                events = dict(poller.poll(1 if waiters else 500))
            except zmq.ZMQError as e:
                logger.warning("side channel poll failed: %s", e)
                continue
            if sock in events:
                try:
                    frames = sock.recv_multipart()
                    ident, msg = frames[0], json.loads(frames[-1])
                except Exception as e:  # noqa: BLE001
                    logger.warning("side channel recv failed: %s", e)
                    continue
                op, tid = msg.get("op"), str(msg.get("transfer_id", ""))
                if op == "GET":
                    rep = self._get_reply(tid, host_id)
                    if rep is None:
                        waiters.setdefault(tid, []).append(
                            (ident, time.monotonic() + _GET_WAIT_S)
                        )
                    else:
                        self._router_send(sock, ident, rep)
                elif op in ("DONE", "CANCEL"):
                    self._release(tid)
                    for w_ident, _ in waiters.pop(tid, []):
                        self._router_send(sock, w_ident, {"status": "cancelled"})
                    self._router_send(sock, ident, {"status": "ok"})
                else:
                    self._router_send(
                        sock, ident, {"status": "error", "msg": f"unknown op {op}"}
                    )
            if waiters:
                now = time.monotonic()
                for tid in list(waiters):
                    rep = self._get_reply(tid, host_id)
                    still = []
                    for ident, deadline in waiters[tid]:
                        if rep is not None:
                            self._router_send(sock, ident, rep)
                        elif now >= deadline:
                            self._router_send(sock, ident, {"status": "pending"})
                        else:
                            still.append((ident, deadline))
                    if still:
                        waiters[tid] = still
                    else:
                        del waiters[tid]
        sock.close(0)
        ctx.term()

    def _release(self, tid: str):
        with self._lock:
            st = self._staged.pop(tid, None)
        if st is not None:
            self.pool.release(st.buf)

    def stage_after_step(self, meta: TTMooncakeConnectorMetadata):
        if not self.c._is_producer or not meta.stage:
            return
        from models.demos.blackhole.qwen36.tt import pd_transfer

        for sr in meta.stage:
            if sr.req_id in self._stage_done:
                continue  # re-shipped until the scheduler sees finished_sending
            t0 = time.perf_counter()
            slot = self.runner._req_state_slot.get(sr.req_id)
            cap = (
                self.model.pd_gdn_capture.pop(slot, None) if slot is not None else None
            )
            if cap is None:
                logger.error(
                    "TTMooncakeConnector: no GDN snapshot for %s (slot %s); request "
                    "will not be transferable",
                    sr.req_id,
                    slot,
                )
                continue
            rec_snap, conv_snap = cap
            # the model reads snapshots into pooled host buffers; give them back once
            # the bytes are in the staging buffer (or the request cannot be staged)
            release_snapshot = getattr(self.model, "pd_gdn_snapshot_release", None)
            n_blocks = max(1, math.ceil(sr.num_tokens / self.c._block_size))
            block_ids = sr.block_ids[:n_blocks]
            if len(block_ids) < n_blocks:
                logger.error(
                    "TTMooncakeConnector: %s has %d blocks for %d tokens",
                    sr.req_id,
                    len(sr.block_ids),
                    sr.num_tokens,
                )
                if release_snapshot is not None:
                    release_snapshot(rec_snap, conv_snap)
                continue
            kv = pd_transfer.export_kv_blocks(self.model, block_ids)
            t1 = time.perf_counter()
            pooled = self.pool.acquire(payload_nbytes(kv, rec_snap, conv_snap))
            buf, header = pack_payload(
                kv, rec_snap, conv_snap, sr.num_tokens, n_blocks, out=pooled
            )
            if release_snapshot is not None:
                release_snapshot(rec_snap, conv_snap)
            addr, nbytes = buf.data_ptr(), int(header["nbytes"])
            tid = sr.transfer_id or sr.req_id
            shm_name = self.pool.shm_name(buf)
            with self._lock:
                self._staged[tid] = _Staged(
                    buf, addr, nbytes, header, time.time(), shm_name
                )
                self._finished_sending.add(sr.req_id)
                self._stage_done.add(sr.req_id)
            t2 = time.perf_counter()
            self.stats["staged"] += 1
            self.stats["staged_bytes"] += header["nbytes"]
            self.stats["stage_ms"] += 1e3 * (t2 - t0)
            logger.info(
                "[pd] staged %s: %d tokens, %d blocks, %.1f MiB (export %.1f ms, "
                "pack+register %.1f ms) digest %s transfer %s via %s",
                sr.req_id,
                sr.num_tokens,
                n_blocks,
                header["nbytes"] / 2**20,
                1e3 * (t1 - t0),
                1e3 * (t2 - t1),
                payload_digest(buf, header["nbytes"]),
                tid,
                f"shm {shm_name}" if shm_name else "mooncake",
            )
        # garbage-collect stagings nobody pulled (decoder died / aborted upstream)
        now = time.time()
        with self._lock:
            stale = [
                tid
                for tid, st in self._staged.items()
                if now - st.t_staged > _GET_TIMEOUT_S
            ]
        for tid in stale:
            logger.warning(
                "[pd] dropping staged %s: never pulled within %.0f s",
                tid,
                _GET_TIMEOUT_S,
            )
            self._release(tid)

    # ---- consumer ----
    def start_step(self, meta: TTMooncakeConnectorMetadata):
        if self.c._is_consumer:
            for rr in meta.recv:
                if rr.req_id in self._inflight or rr.req_id in self._recv_done:
                    continue  # re-shipped until the scheduler sees finished_recving
                self._inflight[rr.req_id] = rr
                self._pool.submit(self._pull, rr)
            for host, port, tid in meta.cancel:
                self._pool.submit(
                    self._side_channel_call,
                    host,
                    port,
                    {"op": "CANCEL", "transfer_id": tid},
                )
            self._drain_fetched()

    @staticmethod
    def _side_channel_call(
        host: str, port: int, msg: dict[str, Any], timeout_ms: int = 5000
    ):
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        try:
            sock.connect(f"tcp://{host}:{port}")
            sock.send_json(msg)
            return sock.recv_json()
        finally:
            sock.close(0)

    def _shm_segment(self, name: str) -> torch.Tensor | None:
        """The producer's staging segment ``name`` mapped read-only, cached per name
        (the producer reuses its pooled buffers, so a handful of mappings serve every
        request)."""
        with self._shm_lock:
            seg = self._shm_segments.get(name)
        if seg is not None:
            return seg
        t0 = time.perf_counter()
        seg = map_shm_segment(name)
        if seg is None:
            logger.info(
                "[pd] shm segment %s does not open here; pulling through Mooncake",
                name,
            )
            return None
        with self._shm_lock:
            self._shm_segments[name] = seg
        logger.info(
            "[pd] mapped producer shm segment %s (%.0f MiB) read-only in %.1f ms",
            name,
            seg.numel() / 2**20,
            1e3 * (time.perf_counter() - t0),
        )
        return seg

    def _send_done_later(self, rr: RecvReq) -> None:
        """DONE lets the producer recycle the staging; off the main thread."""
        msg = {"op": "DONE", "transfer_id": rr.transfer_id}
        with contextlib.suppress(RuntimeError):  # executor shut down
            self._pool.submit(
                self._side_channel_call, rr.remote_host, rr.remote_port, msg
            )

    def _is_aborted(self, req_id: str) -> bool:
        with self._lock:
            return req_id in self._aborted

    def _pull(self, rr: RecvReq):
        """Background: GET the staging descriptor, then either map the producer's shm
        segment (same host: no copy, DONE once the runner has imported) or pull the
        bytes into a registered local buffer (DONE right away)."""
        try:
            t0 = time.perf_counter()
            deadline = t0 + _GET_TIMEOUT_S
            while True:
                if self._is_aborted(rr.req_id):
                    raise _PullAborted(rr.req_id)
                rep = self._side_channel_call(
                    rr.remote_host,
                    rr.remote_port,
                    {"op": "GET", "transfer_id": rr.transfer_id},
                )
                if rep.get("status") == "ok":
                    break
                if rep.get("status") != "pending" or time.perf_counter() > deadline:
                    raise RuntimeError(
                        f"producer has no staging for {rr.transfer_id}: {rep}"
                    )
                time.sleep(_GET_POLL_S)
            t1 = time.perf_counter()
            nbytes = int(rep["nbytes"])
            shm = rep.get("shm")
            buf = None
            if shm and self._shm_ok and rep.get("host") == self._host_id:
                seg = self._shm_segment(str(shm["name"]))
                off = int(shm["offset"])
                if seg is not None and off + nbytes <= seg.numel():
                    buf = seg[off : off + nbytes]
                    # the producer keeps the bytes until DONE; send it when the runner
                    # has finished reading the mapping (release below)
                    release, via = (lambda: self._send_done_later(rr)), "shm"
            if buf is None:
                pooled = self.pool.acquire(nbytes)
                with self._engine_lock:
                    rc = self.engine.transfer_sync_read(
                        rep["segment"], pooled.data_ptr(), int(rep["addr"]), nbytes
                    )
                if rc != 0:
                    self.pool.release(pooled)
                    raise RuntimeError(f"transfer_sync_read failed ({rc})")
                self._side_channel_call(
                    rr.remote_host,
                    rr.remote_port,
                    {"op": "DONE", "transfer_id": rr.transfer_id},
                )
                buf = pooled[:nbytes]
                release, via = (lambda b=pooled: self.pool.release(b)), "pull"
            t2 = time.perf_counter()
            self._fetched.put(
                _Fetched(rr, buf, rep["header"], release, via, t1 - t0, t2 - t1)
            )
        except _PullAborted as e:
            self._failed.put((rr, e))
        except BaseException as e:  # noqa: BLE001
            logger.exception("[pd] pull failed for %s", rr.req_id)
            self._failed.put((rr, e))

    def _finish_recv(self, req_id: str) -> None:
        self._inflight.pop(req_id, None)
        with self._lock:
            self._aborted.discard(req_id)
            self._finished_recving.add(req_id)
            self._recv_done.add(req_id)

    def _drain_fetched(self):
        """Main thread: write pulled KV into the paged cache, park the GDN snapshot,
        report finished_recving."""
        from models.demos.blackhole.qwen36.tt import pd_transfer

        while True:
            try:
                f = self._fetched.get_nowait()
            except queue.Empty:
                break
            rr, buf, header = f.rr, f.buf, f.header
            t0 = time.perf_counter()
            n_blocks = int(header["n_blocks"])
            if self._is_aborted(rr.req_id):
                logger.info("[pd] %s: finished before its import; dropping", rr.req_id)
                f.release()
            elif int(header["num_tokens"]) != int(rr.num_tokens):
                # the producer prefilled a different number of tokens than this
                # instance derived from its own tokenization: the state does not fit
                # the request; the runner prefills it locally instead
                logger.error(
                    "[pd] %s: producer staged %d tokens, this instance expects %d "
                    "(tokenization mismatch?); skipping import",
                    rr.req_id,
                    int(header["num_tokens"]),
                    int(rr.num_tokens),
                )
                f.release()
            elif len(rr.block_ids) < n_blocks:
                logger.error(
                    "[pd] %s: %d local blocks for a %d-block payload; skipping import",
                    rr.req_id,
                    len(rr.block_ids),
                    n_blocks,
                )
                f.release()
            else:
                kv, rec, conv = unpack_payload(buf, header)
                pd_transfer.import_kv_blocks(self.model, rr.block_ids[:n_blocks], kv)
                # keep the snapshot alive (views into buf: the pooled receive buffer
                # or the producer's mapped segment) until the runner writes the decode
                # slot; the runner calls the release when done (for a mapped segment
                # that is what sends DONE to the producer)
                self.runner.pd_pending_gdn[rr.req_id] = (rec, conv, buf, f.release)
            t1 = time.perf_counter()
            self._finish_recv(rr.req_id)
            self.stats["pulled"] += 1
            self.stats["pulled_bytes"] += header["nbytes"]
            self.stats["pull_ms"] += 1e3 * f.t_pull
            self.stats["import_ms"] += 1e3 * (t1 - t0)
            if f.via == "shm":
                logger.info(
                    "[pd] pulled %s: %d tokens, %.1f MiB via shm (wait %.1f ms, map "
                    "%.1f ms, KV import %.1f ms) digest %s blocks %s",
                    rr.req_id,
                    header["num_tokens"],
                    header["nbytes"] / 2**20,
                    1e3 * f.t_wait,
                    1e3 * f.t_pull,
                    1e3 * (t1 - t0),
                    payload_digest(buf, header["nbytes"]),
                    rr.block_ids[:n_blocks],
                )
            else:
                logger.info(
                    "[pd] pulled %s: %d tokens, %.1f MiB (wait %.1f ms, pull %.1f ms = "
                    "%.2f GB/s, KV import %.1f ms) digest %s blocks %s",
                    rr.req_id,
                    header["num_tokens"],
                    header["nbytes"] / 2**20,
                    1e3 * f.t_wait,
                    1e3 * f.t_pull,
                    header["nbytes"] / max(f.t_pull, 1e-9) / 2**30,
                    1e3 * (t1 - t0),
                    payload_digest(buf, header["nbytes"]),
                    rr.block_ids[:n_blocks],
                )
        while True:
            try:
                rr, err = self._failed.get_nowait()
            except queue.Empty:
                break
            # Report it as received so the scheduler proceeds (an aborted request's
            # delayed block free needs it too); a live request's runner then prefills
            # locally (its state slot has no import parked), slow but correct.
            self._finish_recv(rr.req_id)
            if isinstance(err, _PullAborted):
                logger.info("[pd] %s: finished while its pull was pending", rr.req_id)
            else:
                logger.error(
                    "[pd] %s: transfer failed (%s); the decoder prefills it locally",
                    rr.req_id,
                    err,
                )

    def take_finished(
        self, finished_req_ids: set[str] | None = None
    ) -> tuple[set[str] | None, set[str] | None]:
        for req_id in finished_req_ids or ():
            self._stage_done.discard(req_id)
            self._recv_done.discard(req_id)
            if req_id in self._inflight:
                # client hung up while the pull is in flight: stop waiting on the
                # producer and never import (a payload already fetched is dropped by
                # the drain below); the drain reports it so the scheduler frees the
                # blocks it held back
                with self._lock:
                    self._aborted.add(req_id)
            pending = getattr(self.runner, "pd_pending_gdn", None)
            if pending and req_id in pending:
                # imported, parked, then aborted before it ever got a decode slot
                entry = pending.pop(req_id)
                if len(entry) > 3 and entry[3] is not None:
                    entry[3]()
        if self.c._is_consumer:
            self._drain_fetched()
        with self._lock:
            s, r = self._finished_sending, self._finished_recving
            self._finished_sending, self._finished_recving = set(), set()
        return (s or None), (r or None)
