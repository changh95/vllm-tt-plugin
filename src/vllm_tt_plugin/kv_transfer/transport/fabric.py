# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
"""FabricSocketTransport: KV handoff bytes over ONE TT-Fabric ``MeshSocket``
(PHASE2_DESIGN 4.3, PHASE3_NOTES 3, PHASE3_MICROBENCH).

HYBRID control plane
    The header / manifest / status / claim / lease / janitor machinery of the shm
    transport is reused verbatim (``_ControlSegments``, a thin ``ShmTransport``
    subclass in ``dumpfile`` mode under ``control_dir``): each xfer still has a
    ``{control_dir}/{producer}/{xfer_hex}`` segment with the READY -> claimed ->
    CONSUMED|LOAD_FAILED|RELEASED state machine, the host-side taps
    (``gdn.L{j}.taps.rows.pt``) and one extra sidecar ``fabric.json``.  Only the
    128 device parts (32 K/V x chunks, 48 recs) travel over the socket.  The two
    engines are MPI ranks of one tt-run job on one host, so the shm files are a
    shared-memory control channel, not a second data plane.

    Dependency on ``shm.py``: ``ShmTransport.open_put`` charges
    ``self._segment_charge(hdr, manifest)`` (default: the whole segment file,
    ``hdr.data_off + hdr.total_nbytes``) against ``budget_bytes`` and the tmpfs free
    space.  ``_ControlSegments`` overrides that hook to charge only what it stores
    (header + taps rows, ~4 MiB per xfer).  Without the hook every control segment
    would be charged its full K/V + rec size (216 MiB at 2k tokens, 1.2 GiB at 32k)
    against the 1 GiB control budget and be refused; guarded by
    ``test_control_plane_charges_only_stored_bytes``.  Commit '[feat] PD fabric: shm
    control-plane charge hook' carries the shm.py side.

Data path (direct mode, one FIFO channel, prefill rank 0 -> decode rank 1)
    producer  write_from_device(blk, chunk=c)  -> device relayout of the hook's
              block-major chunk into a transport-owned HEAD-major export buffer
              ([1,4,2048,256], cache dtype) -- the hook deallocates ``blk`` right
              after the call; rec -> device copy of the live ``rec_state`` into a
              rec buffer; taps -> shm rows file.
              finish_export(READY)  -> sidecar (item list, ``seq`` None) + the shm
              READY publish.  NOTHING is enqueued on the socket yet (claim-gated
              sends, below); the export buffers stay allocated.
    consumer  open_get  -> CLAIM (the shm rename ``{hx}`` -> ``{hx}.claimed-{D}``,
              only when a rec set is free), then None until the producer's marker
              says the sends are on the channel; the READY handle is issued once the
              xfer is at the channel head.  ``SourceChunk.read_into_device(
              hook_staging)`` = ``recv_direct_async`` posted right where the hook then
              runs ``paged_fill_cache`` on the same CQ (no host sync needed); after
              the LAST K/V item the 48 rec recvs are posted into a transport-owned
              per-xfer rec set (the hook installs recs at the JOIN step, possibly
              many steps later, and a single FIFO channel cannot hold them back
              behind the next xfer's K/V); ``install_gdn_state`` gets them by a
              device copy into the hook's ``_rec_staging``.
    producer  pump() (every ``TTKVWorker.end_step`` and the rank's idle tick, both on
              the engine thread) -> for every published export whose claim dir now
              exists: ALL ``send_direct_async`` enqueued in the canonical order
              (chunk-outer, manifest K/V part order, then recs in manifest order),
              ``seq`` assigned in SEND order, then the marker
              ``{control_dir}/.fabric_sent/{producer}/{hx}.json`` (epoch, seq, the
              items actually enqueued) written atomically.
    completion = the worker's existing ``synchronize_device`` before KV_DONE; the
              producer reclaims buffers when the segment has left the disk (the
              consumer only removes it after posting every recv; the in-order CQ
              makes buffer reuse safe).

Claim-gated sends (the D5 fix)
    Enqueuing the sends at READY parked the producer's CQ0 until the consumer posted
    recvs, and an export the consumer never admitted (bogus / foreign xfer id, lost
    D leg, client gone between the legs) parked it for good: 600 s device timeout,
    engine fatal (PHASE3_RESULTS v2 8.3).  Now the consumer's CLAIM is the trigger:

    * an ORPHAN (never claimed) never touches the channel; its READY segment is
      swept by the producer janitor at lease expiry like any shm segment and
      ``_reclaim`` frees the buffers -- nothing to drain, nothing parked;
    * the producer's sends park only between its enqueue and the consumer's recvs:
      the consumer claimed, so it polls the marker at every ``begin_step`` /
      ``end_step`` (``TTKVWorker``) and posts the recvs in the step it sees it --
      the park is bounded by one consumer step (plus ``claim_wait_s`` at most);
    * the consumer's recvs never wait for host-side work: the marker is written
      AFTER the sends were enqueued, so a producer that dies between the claim and
      the marker leaves the consumer idle (its claim expires with the lease), not
      parked;
    * a consumer that drops a claim BEFORE the marker (abort / lease expiry / a
      demotion) first FENCES it: ``{hx}.claimed-{D}`` -> ``{hx}.claimed-{D}.closing``
      (an atomic rename the producer never sends for; the janitor still treats it
      as a claim of a live pid, the producer still holds the buffers), then reads
      the marker: absent -> nothing is or will be on the channel, drop the claim;
      present -> the items are parked, drained at their channel turn.  The one
      residual window: the producer listed the claim, enqueued the sends and is
      about to write the marker while the consumer fences -> the marker lands
      after the consumer's check (or between that check and the drop), as an
      ORPHAN MARKER (no claim of ours).  The consumer never unlinks a marker it
      has not read (``_finalize`` unlinks only an ADOPTED marker; ``_drop_unsent``
      re-reads after the drop and reports a late one), so the orphan survives and
      the consumer's ``pump()`` (every step + idle tick) drains it at its turn:
      the producer's CQ is parked for at most one consumer step / idle tick, never
      a lease.
    * the common-path cost: the sends start at the producer's next ``pump`` after
      the claim -- at most the idle-tick period when P is idle.  When P is
      mid-prefill for another request the pump runs at that STEP's end, and the
      prefill rank runs a WHOLE prompt per step (``pd_launch_config`` emits
      ``--max-num-batched-tokens`` = ctx), so the claim waits one FULL prefill of
      the next queued request (run 2 step lengths: 6.4 s @8k, 26 s @32k tokens);
      the old protocol had the sends on the channel at READY and the next prefill
      merely queued behind them for one D poll.  This is the price of option (a)
      of PHASE3_RESULTS 12.
    * the claim lease: the consumer's wait for the marker is clocked from the
      CLAIM, not from READY (``claim_deadline``, asked by ``TTKVWorker._poll_ready``):
      a claimed xfer fails at claim + ``claim_lease_s`` (``TT_PD_FABRIC_CLAIM_LEASE_S``
      / ``fabric_claim_lease_s``, default 3 x ``lease_duration`` = 90 s: the HARD
      BOUND by which a dead or stuck producer is detected -- a producer mid-prefill
      of the next queued request is not dead); once the marker is seen the lease
      restarts from the marker (``lease_duration`` for the xfer to reach the channel
      head; an expiry there releases the claim = a drain at its turn, never a hang).
      Worst case claim -> failure = ``claim_lease_s`` + ``lease_duration`` (4 x
      lease).  With the READY lease alone, back-to-back prompts whose prefill
      exceeds the lease (~36k tokens at CTX 65536) expired deterministically while
      P was busy and D recomputed.  shm (no claim gating) keeps the READY lease.

Single-channel discipline
    * ``seq`` = send order per producer epoch, recorded in the marker; the sidecar
      of a published segment carries ``seq: None`` (nothing on the channel);
    * the consumer receives xfers in seq order (``open_get`` -> None while an older
      sent xfer is pending), enforces the item order (mismatch -> RuntimeError,
      nothing posted), and DRAINS instead of skipping: an aborted / failed /
      released xfer whose items are on the channel has them received into scratch
      at its turn (``pump`` / the next ``open_get``);
    * a ``send`` that raises part-way through the enqueue leaves the already-enqueued
      items parked on the channel: the marker keeps the ``seq`` and lists exactly
      them with ``status: FAILED``; the consumer's ``open_get`` returns a FAILED
      handle and drains them at their turn; nothing enqueued -> no ``seq`` consumed,
      marker ``seq: None`` (the consumer drops the claim, nothing to drain).

Device pools (allocated once in ``start()``, DRAM interleaved, cache dtype)
    producer  K/V export pool = ``fabric_export_budget_bytes`` / 2,228,224 B
              head-major buffers.  The default budget is DERIVED from the served
              context: ``fabric_export_slots`` (1) x ``max_export_kv_buffers``
              (``cdiv(cdiv(max_model_len, 64), 32)`` chunks x 32 K/V parts) x
              2,228,224 B = 1024 buffers = 2.125 GiB for ``max_model_len`` 65536;
              plus ``rec_sets`` x 48 x 3 MiB = 576 MiB of rec buffers (4 sets).
              ``start()`` refuses a budget that cannot hold ONE max-length export
              (the old fixed 2 GiB default held 963 < 1024 buffers, so every
              65k-token export was refused).  One slot means the NEXT export of a
              full-length prompt waits for the consumer to drain the previous one
              (``open_put`` -> None -> FAILED -> D recomputes); 2 slots (4.25 GiB)
              pipeline two full-length exports.  The control plane charges only
              the header + taps rows (~4 MiB per xfer) against its 1 GiB budget,
              so the export pool is the ONLY limiter on the producer.
    consumer  ``rec_sets`` x 48 rec buffers (576 MiB) + one K/V and one rec scratch
              buffer (5.1 MiB) for drains.
    P-side DRAM = weights + paged KV pool + this export pool + rec pool + trace
    region; p3_device_validation_plan.md V4 records the measured headroom
    (``ttnn.dump_device_memory_state`` after ``start()``).

``pump()`` (both roles) is the claim-gated protocol's clock: the producer sends for
new claims and reclaims finished exports, the consumer drains released xfers and
orphan markers at the channel head.  ``TTKVWorker.end_step`` calls it after every
step; the rank entry point wakes the idle engine thread for it
(``pd_fabric_rank.install_idle_ticker``, gated by ``wants_pump()``, a host-only
check any thread may make).  Device work happens on the engine thread only.

Nothing here imports ``ttnn`` or ``torch`` at module import; ``fabric_socket`` does,
lazily.  ``TT_PD_CHECKSUM=1`` and ``kv_both`` are unsupported over fabric (raised
at start / construction).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from .base import (
    BLOCK_TOKENS,
    BLOCKS_PER_CHUNK,
    HEAD_MAJOR_KV_SHAPE_HINT,
    KV_CHUNK_SHAPE,
    LAYOUT_VERSION,
    REC_SHAPE,
    TAPS_SHAPE,
    TILE_RECORD_BYTES,
    GetHandle,
    Manifest,
    PartSpec,
    PutHandle,
    Sink,
    Source,
    SourceChunk,
    TTKVTransport,
    cdiv,
    coerce_manifest,
)
from .fabric_socket import (
    DEFAULT_FIFO_BYTES,
    DEFAULT_PACKET_BYTES,
    RendezvousTimeout,
    SocketLayer,
    SpecKey,
    register_transport,
    registered_mesh_device,
    unregister_transport,
)
from .shm import (
    READY,
    RELEASED,
    ShmTransport,
    parse_xfer_id,
    read_header,
    write_status,
)

try:  # vLLM's logger tree when the plugin is installed; plain logging otherwise
    from vllm_tt_plugin.logger import init_tt_logger

    logger = init_tt_logger(__name__)
except Exception:  # pragma: no cover
    logger = logging.getLogger(__name__)

SIDECAR_NAME = "fabric.json"
RENDEZVOUS_DIR = ".fabric_rendezvous"
# producer-owned send markers: {control_dir}/.fabric_sent/{producer}/{hx}.json
SENT_DIR = ".fabric_sent"
# a consumer claim being dropped before the marker: {hx}.claimed-{D}.closing (the
# producer never sends for it; the shm janitor still sees a ".claimed-" dir)
CLOSING_SUFFIX = ".closing"
DEFAULT_CONTROL_DIR = "/dev/shm/tt_pd_fabric"

# One spec table for both ranks (design 4.3); the dtype placeholders are filled
# from FabricConfig (kv_dtype / rec_dtype).
STAGING_SPECS: dict[str, tuple[tuple[int, ...], str, str]] = {
    "kv_blocks": (HEAD_MAJOR_KV_SHAPE_HINT, "<cache dtype>", "TILE"),  # [1,4,2048,256]
    "gdn_rec": (REC_SHAPE, "float32", "TILE"),  # [1,48,128,128]
    "gdn_taps": (TAPS_SHAPE, "bfloat16", "ROW_MAJOR"),  # host rows via the segment
}

_NOT_STARTED = (
    "FabricSocketTransport is not started: start() runs inside "
    "TTKVConnector.attach_runner (EngineCore.__init__) and needs the opened mesh "
    "(register_mesh_device from the pd_fabric_rank entry point, or mesh_device=)"
)


def spec_nbytes(spec: SpecKey) -> int:
    """Packed bytes of a tensor of ``spec`` (TILE: tile records; ROW_MAJOR: elems)."""
    shape, dtype, layout = spec
    n = 1
    for s in shape:
        n *= int(s)
    if layout.upper() == "TILE":
        return (n // 1024) * TILE_RECORD_BYTES[dtype]
    return n * {"float32": 4, "bfloat16": 2, "bfloat8_b": 1}[dtype]


DEFAULT_MAX_MODEL_LEN = 65536  # the pair's --max-model-len (pd_launch_config CTX)
DEFAULT_KV_PARTS = 32  # 16 attention layers x (K, V)


def max_export_kv_buffers(
    max_model_len: int = DEFAULT_MAX_MODEL_LEN, kv_parts: int = DEFAULT_KV_PARTS
) -> int:
    """Head-major K/V buffers ONE export of the longest request needs:
    ``cdiv(cdiv(max_model_len, 64), 32)`` chunks x ``kv_parts`` (1024 at 65536)."""
    return cdiv(cdiv(int(max_model_len), BLOCK_TOKENS), BLOCKS_PER_CHUNK) * int(
        kv_parts
    )


def export_pool_bytes(
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    kv_parts: int = DEFAULT_KV_PARTS,
    kv_dtype: str = "bfloat8_b",
    slots: int = 1,
) -> int:
    """Producer export pool holding ``slots`` max-length exports (the derived
    ``fabric_export_budget_bytes``): 2,281,701,376 B = 2.125 GiB at 65536 / bfp8."""
    return (
        int(slots)
        * max_export_kv_buffers(max_model_len, kv_parts)
        * spec_nbytes((HEAD_MAJOR_KV_SHAPE_HINT, kv_dtype, "TILE"))
    )


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


@dataclass
class FabricConfig:
    """Knobs; explicit kwargs win over ``kv_connector_extra_config`` (the ``fabric_*``
    / ``socket_*`` keys below, exactly what ``pd_launch_config.kv_transfer_config``
    emits) which win over ``TT_PD_FABRIC_*`` env (tt-run passes ``TT_`` through)
    which win over the defaults."""

    control_dir: str = DEFAULT_CONTROL_DIR
    socket_connections: int = 2
    fifo_bytes: int = DEFAULT_FIFO_BYTES
    sender_rank: int = 0
    receiver_rank: int = 1
    packet_bytes: int = DEFAULT_PACKET_BYTES  # informational (set before mesh open)
    # producer K/V export pool; None / 0 = derived: export_slots x one max-length
    # export of max_model_len (see export_pool_bytes; 2.125 GiB at 65536)
    export_budget_bytes: int | None = None
    export_slots: int = 1  # max-length exports the derived pool holds at once
    max_model_len: int = DEFAULT_MAX_MODEL_LEN  # the pair's --max-model-len
    kv_parts: int = DEFAULT_KV_PARTS  # K/V parts per manifest (2 x attention layers)
    rec_sets: int = 4  # per-xfer rec buffer sets (both roles)
    rec_parts: int = 48  # rec buffers per set (GDN layers)
    kv_dtype: str = "bfloat8_b"
    rec_dtype: str = "float32"
    lease_duration: float = 30.0
    janitor_period: float = 0.1
    socket_timeout_s: float = 300.0  # barrier + handshake bound (0 = unbounded)
    # consumer: how long the FIRST open_get after the claim spins for the producer's
    # send marker (an idle producer answers within its idle tick, ~5 ms); after
    # that the worker re-polls at every step begin / end.  0 = never spin.
    claim_wait_s: float = 0.010
    # consumer: the lease of a CLAIM of ours, measured from the claim, until the
    # producer's send marker (the hard bound a dead / stuck producer is caught by;
    # a producer mid-prefill of the next queued request needs up to one full
    # prefill).  None / 0 = 3 x lease_duration.  After the marker the wait for the
    # channel head is clocked from the marker with lease_duration.
    claim_lease_s: float | None = None
    # DEBUG ONLY (p3 device validation): skip the warm-up send/recv in start() so a
    # pair can reach READY on a fabric link that does not pass payloads (the first
    # real export then compiles the programs and PARKS both CQs if the link is dead)
    skip_warmup: bool = False

    _EXTRA_KEYS = {
        "fabric_control_dir": "control_dir",
        "control_dir": "control_dir",  # alias (Phase 2 spelling)
        "socket_connections": "socket_connections",
        "fabric_fifo_bytes": "fifo_bytes",
        "socket_fifo_bytes": "fifo_bytes",  # alias
        "fabric_sender_rank": "sender_rank",
        "fabric_receiver_rank": "receiver_rank",
        "fabric_max_packet_payload_bytes": "packet_bytes",
        "fabric_export_budget_bytes": "export_budget_bytes",
        "fabric_export_slots": "export_slots",
        "fabric_max_model_len": "max_model_len",
        "fabric_kv_parts": "kv_parts",
        "fabric_rec_sets": "rec_sets",
        "fabric_rec_parts": "rec_parts",
        "fabric_kv_dtype": "kv_dtype",
        "fabric_rec_dtype": "rec_dtype",
        "kv_lease_duration": "lease_duration",
        "fabric_socket_timeout_s": "socket_timeout_s",
        "fabric_skip_warmup": "skip_warmup",
        "fabric_claim_wait_s": "claim_wait_s",
        "fabric_claim_lease_s": "claim_lease_s",
    }

    @classmethod
    def build(cls, extra: dict[str, Any] | None, **explicit: Any) -> FabricConfig:
        cfg = cls(
            control_dir=os.environ.get("TT_PD_FABRIC_DIR", DEFAULT_CONTROL_DIR),
            socket_connections=_env_int("TT_PD_FABRIC_CONNECTIONS", 2),
            packet_bytes=_env_int("TT_PD_FABRIC_PKT", DEFAULT_PACKET_BYTES),
            export_budget_bytes=_env_int("TT_PD_FABRIC_EXPORT_BUDGET", 0) or None,
            export_slots=_env_int("TT_PD_FABRIC_EXPORT_SLOTS", 1),
            max_model_len=_env_int("TT_PD_FABRIC_MAX_MODEL_LEN", DEFAULT_MAX_MODEL_LEN),
            rec_sets=_env_int("TT_PD_FABRIC_REC_SETS", 4),
            socket_timeout_s=_env_float("TT_PD_FABRIC_SOCKET_TIMEOUT_S", 300.0),
            claim_wait_s=_env_float("TT_PD_FABRIC_CLAIM_WAIT_S", 0.010),
            claim_lease_s=_env_float("TT_PD_FABRIC_CLAIM_LEASE_S", 0.0) or None,
            skip_warmup=os.environ.get("TT_PD_FABRIC_SKIP_WARMUP", "0") == "1",
        )
        for k, v in (extra or {}).items():
            attr = cls._EXTRA_KEYS.get(k)
            if attr is not None:
                setattr(cfg, attr, v)
        for k, v in explicit.items():
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)
        # normalise types
        cfg.socket_connections = int(cfg.socket_connections)
        cfg.fifo_bytes = int(cfg.fifo_bytes)
        cfg.sender_rank, cfg.receiver_rank = (
            int(cfg.sender_rank),
            int(cfg.receiver_rank),
        )
        cfg.packet_bytes = int(cfg.packet_bytes)
        cfg.export_slots = int(cfg.export_slots)
        cfg.max_model_len = int(cfg.max_model_len)
        cfg.kv_parts = int(cfg.kv_parts)
        cfg.rec_sets, cfg.rec_parts = int(cfg.rec_sets), int(cfg.rec_parts)
        cfg.lease_duration = float(cfg.lease_duration)
        cfg.janitor_period = float(cfg.janitor_period)
        cfg.socket_timeout_s = float(cfg.socket_timeout_s)
        cfg.claim_wait_s = float(cfg.claim_wait_s)
        if cfg.claim_wait_s < 0:
            raise ValueError(
                f"fabric_claim_wait_s must be >= 0 (got {cfg.claim_wait_s})"
            )
        if cfg.claim_lease_s in (None, "", "0", 0):
            cfg.claim_lease_s = 3.0 * cfg.lease_duration
        cfg.claim_lease_s = float(cfg.claim_lease_s)
        if cfg.claim_lease_s < cfg.lease_duration:
            raise ValueError(
                f"fabric_claim_lease_s ({cfg.claim_lease_s}) must be >= the lease "
                f"({cfg.lease_duration}): the claim lease covers the producer's "
                "prefill of the next queued request on top of the READY lease"
            )
        cfg.skip_warmup = str(cfg.skip_warmup).lower() in ("1", "true", "yes")
        if cfg.sender_rank == cfg.receiver_rank:
            raise ValueError("fabric sender_rank == receiver_rank")
        if cfg.socket_connections < 1:
            raise ValueError("socket_connections must be >= 1")
        if cfg.kv_dtype not in TILE_RECORD_BYTES:
            raise ValueError(f"unknown kv_dtype {cfg.kv_dtype!r}")
        if cfg.rec_dtype not in ("float32", "bfloat16"):
            raise ValueError(f"unknown rec_dtype {cfg.rec_dtype!r}")
        if cfg.export_slots < 1 or cfg.max_model_len < 1 or cfg.kv_parts < 1:
            raise ValueError(
                "fabric_export_slots, fabric_max_model_len and fabric_kv_parts must "
                f"be >= 1 (got {cfg.export_slots}, {cfg.max_model_len}, "
                f"{cfg.kv_parts})"
            )
        if not cfg.export_budget_bytes:  # None / 0: derived from the served context
            cfg.export_budget_bytes = export_pool_bytes(
                cfg.max_model_len, cfg.kv_parts, cfg.kv_dtype, cfg.export_slots
            )
        cfg.export_budget_bytes = int(cfg.export_budget_bytes)
        return cfg

    @property
    def max_export_kv_buffers(self) -> int:
        """K/V buffers one export of a ``max_model_len``-token request takes."""
        return max_export_kv_buffers(self.max_model_len, self.kv_parts)

    @property
    def export_pool_buffers(self) -> int:
        """K/V buffers the export pool holds (``export_budget_bytes`` / buffer)."""
        return max(1, int(self.export_budget_bytes or 0) // spec_nbytes(self.kv_spec))

    @property
    def kv_spec(self) -> SpecKey:
        return (HEAD_MAJOR_KV_SHAPE_HINT, self.kv_dtype, "TILE")

    @property
    def kv_block_major_spec(self) -> SpecKey:
        return (KV_CHUNK_SHAPE, self.kv_dtype, "TILE")

    @property
    def rec_spec(self) -> SpecKey:
        return (REC_SHAPE, self.rec_dtype, "TILE")

    def spec_table(self) -> dict[str, Any]:
        return {
            "layout_version": LAYOUT_VERSION,
            "kv_spec": [list(self.kv_spec[0]), self.kv_spec[1], self.kv_spec[2]],
            "rec_spec": [list(self.rec_spec[0]), self.rec_spec[1], self.rec_spec[2]],
            "socket_connections": self.socket_connections,
            "fifo_bytes": self.fifo_bytes,
            "sender_rank": self.sender_rank,
            "receiver_rank": self.receiver_rank,
            "rec_parts": self.rec_parts,
        }


# --- control segments: the shm state machine minus the two rules the channel forbids


class _ControlSegments(ShmTransport):
    """``ShmTransport`` in dumpfile mode, used for headers / manifests / taps / claims.

    Differences from the parent: ``open_put`` charges only what the segment really
    stores (header + taps rows, ~4 MiB per xfer) against ``budget_bytes`` and the
    tmpfs free space -- the K/V and rec bytes of the manifest travel over the
    socket, so a 65535-token manifest (2.27 GiB on the wire) costs the control plane
    nothing; the export pool is the producer's only limiter.  The janitor is the
    parent's: with claim-gated sends an UNCLAIMED segment (READY past its lease,
    RELEASED, FAILED) has nothing on the channel and is swept; a ``.claimed-*`` dir
    (also the ``.closing`` fence of a claim being dropped) is swept only when its
    consumer pid is dead.
    """

    KIND = "shm"  # segment files are ordinary dumpfile segments

    def __init__(self, **kw: Any) -> None:
        kw.setdefault("mode", "dumpfile")
        kw.setdefault("budget_bytes", 1 << 30)
        kw.setdefault("checksum", False)
        kw.setdefault("free_space_headroom", 64 << 20)
        super().__init__(**kw)

    def _segment_charge(self, hdr: Any, manifest: Manifest) -> int:
        """Header + manifest + the host-side taps rows: the only bytes on disk."""
        return int(hdr.data_off) + sum(
            p.nbytes for p in manifest.parts if p.kind == "gdn_taps"
        )

    def put_state(self, xfer_id: str) -> Any:
        with self._lock:
            return self._puts.get(xfer_id)

    def get_header(self, xfer_id: str) -> Any:
        with self._lock:
            gs = self._gets.get(xfer_id)
            return None if gs is None else gs.hdr

    def segment_dirs(self, engine: str, hx: str) -> tuple[str, str, str]:
        return (
            self._tmp_dir(engine, hx),
            self._pub_dir(engine, hx),
            self._claim_dir(engine, hx),
        )

    def any_dir_exists(
        self, engine: str, hx: str, names: set[str] | None = None
    ) -> bool:
        if names is None:
            names = self.list_engine_dir(engine)
        return (
            hx in names
            or f"{hx}.tmp" in names
            or any(n.startswith(f"{hx}.claimed-") for n in names)
        )

    def list_engine_dir(self, engine: str) -> set[str]:
        try:
            return set(os.listdir(self._engine_dir(engine)))
        except FileNotFoundError:
            return set()

    @staticmethod
    def claimed_hexes(names: set[str]) -> set[str]:
        """xfer hexes with an OPEN claim dir in ``names`` (a ``.closing`` fence is a
        claim being dropped: the producer must not send for it)."""
        return {
            n.split(".claimed-", 1)[0]
            for n in names
            if ".claimed-" in n and not n.endswith(CLOSING_SUFFIX)
        }

    def dead_claim_hexes(self, engine: str, names: set[str]) -> set[str]:
        """xfer hexes of the OPEN claim dirs in ``names`` whose consumer pid is dead
        (``consumer.pid`` + ``os.kill(pid, 0)``, the janitor's ``_claim_is_stale``
        rule; a legacy claim without a pid file counts as live until the janitor's
        age rule).  The producer must not send for them: nobody would post the
        recvs, the channel would park until the janitor sweeps the dir."""
        edir = self._engine_dir(engine)
        now = self._clock()
        dead: set[str] = set()
        for n in names:
            if ".claimed-" not in n or n.endswith(CLOSING_SUFFIX):
                continue
            if self._claim_is_stale(os.path.join(edir, n), now)[0]:
                dead.add(n.split(".claimed-", 1)[0])
        return dead

    def fence_claim(self, xfer_id: str) -> str | None:
        """Consumer: rename our open claim dir to its ``.closing`` fence (atomic) so
        the producer's next ``pump`` will not start sends for it; the claim state
        follows the rename (``release_remote`` / ``finish_import`` remove the fenced
        dir).  Returns the new path, None when we hold no claim / the rename failed
        (already fenced or gone)."""
        with self._lock:
            gs = self._gets.get(xfer_id)
            if gs is None or gs.claim_dir.endswith(CLOSING_SUFFIX):
                return None
            new = gs.claim_dir + CLOSING_SUFFIX
            try:
                os.rename(gs.claim_dir, new)
            except OSError:
                return None
            gs.header_path = os.path.join(new, os.path.basename(gs.header_path))
            gs.claim_dir = new
            return new

    def claim_status(self, xfer_id: str) -> int | None:
        """Consumer: the CURRENT header status of our claim (the producer flips it
        to RELEASED when it abandons a claimed, unsent export); None = no claim."""
        with self._lock:
            gs = self._gets.get(xfer_id)
        if gs is None:
            return None
        try:
            return read_header(gs.header_path, full=False).status
        except (OSError, ValueError):
            return None

    def mark_claims_released(self, hx: str) -> int:
        """Producer: write RELEASED into the header of every claim dir of ``hx``
        (the consumer reads it at its next ``open_get`` and drops the claim without
        waiting for a marker that will never come).  Returns the dirs touched."""
        edir = self._engine_dir()
        n = 0
        for name in self.list_engine_dir(self.engine_id):
            if name.startswith(f"{hx}.claimed-"):
                hp = os.path.join(edir, name, self._header_name())
                with contextlib.suppress(OSError):
                    write_status(hp, RELEASED)
                    n += 1
        return n


# --- device buffer pools --------------------------------------------------------------


@dataclass
class _Buf:
    tensor: Any
    spec: SpecKey
    idx: int


class _Pool:
    """Fixed set of same-spec device buffers allocated once in ``start()``."""

    def __init__(self, spec: SpecKey, bufs: list[Any]) -> None:
        self.spec = spec
        self._all = [_Buf(t, spec, i) for i, t in enumerate(bufs)]
        self._free: deque[_Buf] = deque(self._all)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._all)

    @property
    def free(self) -> int:
        with self._lock:
            return len(self._free)

    def take(self, n: int) -> list[_Buf] | None:
        with self._lock:
            if n > len(self._free):
                return None
            return [self._free.popleft() for _ in range(n)]

    def give(self, bufs: list[_Buf]) -> None:
        with self._lock:
            for b in bufs:
                self._free.append(b)

    def tensors(self) -> list[Any]:
        return [b.tensor for b in self._all]


# --- producer side --------------------------------------------------------------------


@dataclass
class _Export:
    xfer_id: str
    hx: str
    manifest: Manifest
    kv_bufs: dict[tuple[str, int], _Buf]  # (part, chunk) -> buffer
    rec_bufs: dict[str, _Buf]  # part -> buffer
    written: set[tuple[str, int]] = field(default_factory=set)
    published: bool = False
    status: str = ""
    seq: int | None = None  # assigned when the sends are enqueued (send order)
    nbytes: int = 0
    items: list[tuple[str, int]] = field(default_factory=list)  # canonical order
    n_kv: int = 0
    sent: bool = False  # sends enqueued (or attempted) and the marker written
    abandoned: bool = False  # abandon() after publish: never send
    ready_ts: float = 0.0  # perf_counter at READY publish (handoff evidence)
    dead_claim_warned: bool = False  # one log line per export for a dead consumer


class FabricSink(Sink):
    """Producer: relay the hook's chunk into its export buffer (device copy)."""

    supports_regions = False

    def __init__(
        self,
        spec: PartSpec,
        exp: _Export,
        layer: SocketLayer,
        cfg: FabricConfig,
        put_state: Any,
    ) -> None:
        self.spec, self._exp, self._layer, self._cfg = spec, exp, layer, cfg
        self._st = put_state

    def _buf(self, chunk: int) -> _Buf:
        if not 0 <= chunk < self.spec.nchunks:
            raise IndexError(f"chunk {chunk} of {self.spec.nchunks}")
        if self.spec.kind == "kv_blocks":
            return self._exp.kv_bufs[(self.spec.name, chunk)]
        return self._exp.rec_bufs[self.spec.name]

    def _note(self, chunk: int) -> None:
        key = (self.spec.name, chunk)
        if key in self._exp.written:
            raise ValueError(f"{self.spec.name} chunk {chunk} written twice")
        self._exp.written.add(key)
        if self._st is not None:
            self._st.note(self.spec.name, chunk)

    def write_from_device(
        self,
        device_tensor: Any,
        *,
        chunk: int,
        blocking: bool = True,
        cq_id: int | None = None,
    ) -> None:
        if self._exp.published:
            raise RuntimeError(f"{self._exp.xfer_id} already published")
        buf = self._buf(chunk)
        got = self._layer.spec_of(device_tensor)
        if self.spec.kind == "kv_blocks":
            if got == self._cfg.kv_block_major_spec:
                self._layer.relayout_copy(device_tensor, buf.tensor)  # hook's chunk
            elif got == self._cfg.kv_spec:
                self._layer.copy(device_tensor, buf.tensor)  # already head-major
            else:
                raise ValueError(
                    f"{self.spec.name}: tensor {got} is neither the block-major "
                    f"{self._cfg.kv_block_major_spec} nor the head-major "
                    f"{self._cfg.kv_spec} chunk"
                )
        else:
            if got != self._cfg.rec_spec:
                raise ValueError(
                    f"{self.spec.name}: tensor {got} != rec spec {self._cfg.rec_spec}"
                )
            self._layer.copy(device_tensor, buf.tensor)
        self._note(chunk)

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
            "fabric sinks have no byte regions (supports_regions)"
        )

    def write_rows(self, rows: Any) -> None:
        raise NotImplementedError("taps rows go through the control segment sink")

    def write_host(self, host_tensor: Any, *, chunk: int) -> None:
        if self._exp.published:
            raise RuntimeError(f"{self._exp.xfer_id} already published")
        buf = self._buf(chunk)
        got = self._layer.spec_of(host_tensor)
        want = (
            self._cfg.kv_spec if self.spec.kind == "kv_blocks" else self._cfg.rec_spec
        )
        if got != want:
            raise ValueError(f"{self.spec.name}: host tensor {got} != {want}")
        self._layer.copy_host_to_device(host_tensor, buf.tensor)
        self._note(chunk)


# --- consumer side --------------------------------------------------------------------


@dataclass
class _Xfer:
    """A claim of ours on the consumer: from the CLAIM (``seq`` None, items from the
    sidecar) through the producer's marker (``seq`` and the items actually sent)
    to the posted recvs (``posted``) or a deferred drain (``released``)."""

    xfer_id: str
    hx: str
    seq: int | None
    items: list[tuple[str, int]]  # canonical channel order: kv items then rec items
    n_kv: int
    rec_set: list[_Buf] | None  # one buffer per rec item, in item order
    cursor: int = 0
    rec_index: dict[str, int] = field(default_factory=dict)  # rec part -> set index
    kv_present: dict[str, int] = field(default_factory=dict)  # part -> chunks sent
    failed: str = ""
    posted: bool = False  # READY handle issued: the hook posts the recvs
    released: bool = False  # drop the claim; drain the sent items at their turn
    send_failed: bool = False  # the producer's enqueue raised part-way (marker FAILED)
    claimed_ts: float = 0.0  # perf_counter at the claim (handoff evidence)
    marker_seen: bool = False  # a marker of ours was READ (only then may we unlink it)
    claimed_at: float = 0.0  # transport clock at the claim (claim_deadline)
    marker_at: float = 0.0  # transport clock when the marker was adopted

    @property
    def nitems(self) -> int:
        return len(self.items)

    @property
    def complete(self) -> bool:
        return self.cursor >= self.nitems

    def index_items(self) -> None:
        self.rec_index = {
            name: k for k, (name, _c) in enumerate(self.items[self.n_kv :])
        }
        self.kv_present = {}
        for name, _c in self.items[: self.n_kv]:
            self.kv_present[name] = self.kv_present.get(name, 0) + 1


class FabricChunk(SourceChunk):
    is_head_major = True
    is_device_readable = True

    def __init__(
        self, spec: PartSpec, chunk: int, xf: _Xfer, transport: FabricSocketTransport
    ) -> None:
        self.spec, self._c, self._xf, self._t = spec, chunk, xf, transport

    def read_into_device(
        self, staging_tensor: Any, *, cq_id: int | None = None
    ) -> None:
        if self.spec.kind == "kv_blocks":
            self._t._recv_kv_item(self._xf, (self.spec.name, self._c), staging_tensor)
        else:
            self._t._copy_rec_item(self._xf, self.spec.name, staging_tensor)

    def read_device(self, mesh_device: Any) -> Any:
        raise NotImplementedError(
            "fabric chunks are received into the caller's staging tensor "
            "(read_into_device); a fresh tensor would not be device-readable in order"
        )


class FabricSource(Source):
    def __init__(
        self, spec: PartSpec, xf: _Xfer, transport: FabricSocketTransport, present: int
    ) -> None:
        self.spec, self._xf, self._t = spec, xf, transport
        self.nbytes_present = int(present) * spec.chunk_nbytes
        self.spec_crc = 0

    def crc32c(self) -> int:
        raise NotImplementedError(
            "checksums are unsupported over fabric (TT_PD_CHECKSUM)"
        )

    def chunk(self, c: int) -> SourceChunk:
        if not 0 <= c < self.spec.nchunks:
            raise IndexError(f"chunk {c} of {self.spec.nchunks}")
        return FabricChunk(self.spec, c, self._xf, self._t)

    def read_rows(self) -> Any:
        raise NotImplementedError("taps rows come from the control segment source")


@dataclass
class _Desc:
    xfer_id: str


# --- the transport --------------------------------------------------------------------


class FabricSocketTransport(TTKVTransport):
    KIND = "fabric"

    def __init__(
        self,
        *,
        engine_id: str,
        role: str | None = None,
        lease_duration: float | None = None,
        rank: int | None = None,
        control_endpoint: str | None = None,  # compatibility with the Phase 2 seam
        mesh_device: Any = None,
        socket_layer: SocketLayer | None = None,
        control_dir: str | None = None,
        extra_config: dict[str, Any] | None = None,
        clock: Any = time.time,
        janitor_period: float | None = None,
        checksum: bool = False,
        **cfg_kwargs: Any,
    ) -> None:
        self.engine_id, self.rank = engine_id, rank
        self.control_endpoint = control_endpoint
        if role is not None and role not in ("producer", "consumer", "both"):
            raise ValueError(f"unknown role {role!r}")
        if role == "both":
            raise ValueError(
                "FabricSocketTransport cannot serve kv_both: one MeshSocket carries "
                "one direction (prefill rank -> decode rank)"
            )
        self.role = role
        self.cfg = FabricConfig.build(
            extra_config,
            control_dir=control_dir,
            lease_duration=lease_duration,
            janitor_period=janitor_period,
            **{k: v for k, v in cfg_kwargs.items() if hasattr(FabricConfig, k)},
        )
        self.lease_duration = self.cfg.lease_duration
        self._checksum = bool(checksum)
        self._clock = clock
        self._layer: SocketLayer | None = socket_layer
        self._mesh = mesh_device
        self._ctrl: _ControlSegments | None = None
        self._sock: Any = None
        self._started = False
        self._shut = False
        self._lock = threading.RLock()
        self._epoch = f"{os.getpid()}-{int(time.time() * 1000)}"
        self._peer: dict[str, Any] | None = None
        # producer
        self._kv_pool: _Pool | None = None
        self._rec_pool: _Pool | None = None
        self._exports: dict[str, _Export] = {}
        self._next_seq = 0
        # consumer
        self._rec_sets: deque[list[_Buf]] = deque()
        self._rec_set_tensors: list[list[Any]] = []
        self._kv_scratch: Any = None
        self._rec_scratch: Any = None
        self._xfers: dict[str, _Xfer] = {}
        self._ctrl_sources: dict[str, dict[str, Source]] = {}  # taps (dumpfile rows)
        self._active: _Xfer | None = None  # posted-but-incomplete xfer (at most one)
        self._recv_epoch: str | None = None
        self._next_recv_seq = 0
        self._by_seq: dict[int, _Xfer] = {}  # sent claims of ours by channel seq
        self._markers: dict[str, dict[str, Any]] = {}  # hx -> marker (immutable)
        self.stats = {
            "sends": 0,
            "recvs": 0,
            "drains": 0,
            "drained_xfers": 0,
            "budget_refusals": 0,
            "order_violations": 0,
            "reclaimed": 0,
            "exports_ready": 0,
            "exports_sent": 0,
            "exports_failed": 0,
            "imports": 0,  # claims
            "orphan_markers": 0,
            "late_markers": 0,  # marker landed between our absent-check and the drop
            "dead_claims_skipped": 0,  # producer pumps skipping a dead consumer's claim
            "warm_programs": 0,
        }

    # -- properties -------------------------------------------------------------------
    @property
    def is_producer(self) -> bool:
        return self.role == "producer"

    @property
    def is_consumer(self) -> bool:
        return self.role == "consumer"

    @property
    def control(self) -> _ControlSegments:
        if self._ctrl is None:
            raise NotImplementedError(_NOT_STARTED)
        return self._ctrl

    @property
    def layer(self) -> SocketLayer:
        if self._layer is None:
            from .fabric_socket import TtnnSocketLayer  # noqa: PLC0415 - lazy ttnn

            self._layer = TtnnSocketLayer()
        return self._layer

    @property
    def peer_engine_id(self) -> str | None:
        return None if self._peer is None else str(self._peer.get("engine_id"))

    def descriptor(self) -> dict[str, Any]:
        return {"kind": self.KIND, "mode": "direct", "layout_version": LAYOUT_VERSION}

    def _require_started(self) -> None:
        if not self._started:
            raise NotImplementedError(_NOT_STARTED)

    # -- start / shutdown --------------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        cfg = self.cfg
        if self._checksum or os.environ.get("TT_PD_CHECKSUM", "0") == "1":
            raise RuntimeError(
                "TT_PD_CHECKSUM=1 is unsupported over fabric: the bytes never touch "
                "the host; unset it (or set kv_connector_extra_config.checksum=false)"
            )
        layer = self.layer
        mesh = self._mesh if self._mesh is not None else registered_mesh_device()
        if mesh is None:
            raise RuntimeError(
                "FabricSocketTransport.start(): no mesh device. Launch the engine "
                "through vllm_tt_plugin.kv_transfer.launch.pd_fabric_rank (it "
                "registers "
                "the opened mesh via fabric_socket.register_mesh_device) or pass "
                "mesh_device="
            )
        if not layer.is_distributed():
            raise RuntimeError(
                "FabricSocketTransport needs the two engines as MPI ranks of one "
                "tt-run job (ttnn.distributed_context is not initialised)"
            )
        size, rank = int(layer.size()), int(layer.rank())
        if size != 2:
            raise RuntimeError(f"fabric PD needs a world of 2 ranks, got {size}")
        if self.rank is not None and int(self.rank) != rank:
            raise RuntimeError(f"configured rank {self.rank} != MPI rank {rank}")
        self.rank = rank
        role_of_rank = {cfg.sender_rank: "producer", cfg.receiver_rank: "consumer"}
        if rank not in role_of_rank:
            raise RuntimeError(
                f"rank {rank} is neither sender_rank {cfg.sender_rank} nor "
                f"receiver_rank {cfg.receiver_rank}"
            )
        if self.role is None:
            self.role = role_of_rank[rank]
        elif self.role != role_of_rank[rank]:
            raise RuntimeError(
                f"role {self.role} on MPI rank {rank}: the prefill (kv_producer) "
                "engine "
                f"must be rank {cfg.sender_rank} and the decode (kv_consumer) engine "
                f"rank {cfg.receiver_rank} (check the rank binding yaml)"
            )
        self._mesh = mesh
        self._ctrl = _ControlSegments(
            engine_id=self.engine_id,
            shm_dir=cfg.control_dir,
            role=self.role,
            lease_duration=cfg.lease_duration,
            janitor_period=cfg.janitor_period,
            clock=self._clock,
        )
        self._ctrl.start()
        if self.is_producer:
            # a new epoch: markers of the previous producer process are void
            shutil.rmtree(self._marker_dir(self.engine_id), ignore_errors=True)
            os.makedirs(self._marker_dir(self.engine_id), exist_ok=True)
        t0 = time.perf_counter()
        try:
            self._allocate_pools()
            self._write_rendezvous()
            layer.barrier(cfg.socket_timeout_s)
            self._read_peer_rendezvous()
            self._sock = layer.create_socket(
                mesh,
                connections=cfg.socket_connections,
                fifo_bytes=cfg.fifo_bytes,
                sender_rank=cfg.sender_rank,
                receiver_rank=cfg.receiver_rank,
                timeout_s=cfg.socket_timeout_s,
            )
            if cfg.skip_warmup:
                logger.warning(
                    "FabricSocketTransport: TT_PD_FABRIC_SKIP_WARMUP set: no warm-up "
                    "transfer; the first export compiles the socket programs at "
                    "request time (debug knob for a fabric link under test)"
                )
            else:
                self._warm_programs()
            layer.sync(mesh)
            layer.barrier(cfg.socket_timeout_s)
        except RendezvousTimeout:
            # the abandoned helper thread may still be inside the native call: do
            # not deallocate device state under it; the process exits with the error
            self._teardown_device(release_tensors=False)
            with contextlib.suppress(Exception):
                self._ctrl.shutdown()
            self._ctrl = None
            raise
        except Exception:
            self._teardown_device()
            with contextlib.suppress(Exception):
                self._ctrl.shutdown()
            self._ctrl = None
            raise
        self.stats["warm_programs"] = int(layer.num_program_cache_entries(mesh))
        self._started = True
        # the rank's close_mesh_device wrapper shuts down what is still registered
        # BEFORE the mesh closes (socket + pool tensors must not outlive the mesh)
        register_transport(self)
        logger.info(
            "FabricSocketTransport(%s rank %d, %s): socket created (%d conns, "
            "fifo %d B, "
            "pkt %d B), pools kv=%d (%d per max-length export of %d tokens, "
            "%.2f GiB) rec=%d, %d programs cached, control %s, %.1f ms",
            self.role,
            rank,
            self.engine_id,
            cfg.socket_connections,
            cfg.fifo_bytes,
            cfg.packet_bytes,
            len(self._kv_pool) if self._kv_pool else 0,
            cfg.max_export_kv_buffers,
            cfg.max_model_len,
            (len(self._kv_pool) * spec_nbytes(cfg.kv_spec) if self._kv_pool else 0)
            / 2**30,
            len(self._rec_pool) if self._rec_pool else cfg.rec_sets * cfg.rec_parts,
            self.stats["warm_programs"],
            cfg.control_dir,
            (time.perf_counter() - t0) * 1e3,
        )

    def _allocate_pools(self) -> None:
        cfg, layer, mesh = self.cfg, self.layer, self._mesh
        if self.is_producer:
            n_kv = cfg.export_pool_buffers
            need = cfg.max_export_kv_buffers
            if n_kv < need:
                raise RuntimeError(
                    f"fabric export pool too small: fabric_export_budget_bytes "
                    f"{cfg.export_budget_bytes} holds {n_kv} K/V buffers of "
                    f"{spec_nbytes(cfg.kv_spec)} B but one export of a "
                    f"max_model_len={cfg.max_model_len} request needs {need} "
                    f"({cfg.kv_parts} parts x {need // cfg.kv_parts} chunks); raise "
                    f"the budget to >= {need * spec_nbytes(cfg.kv_spec)} B "
                    "(or leave it unset to derive it from fabric_max_model_len)"
                )
            self._kv_pool = _Pool(
                cfg.kv_spec, [layer.allocate(mesh, cfg.kv_spec) for _ in range(n_kv)]
            )
            self._rec_pool = _Pool(
                cfg.rec_spec,
                [
                    layer.allocate(mesh, cfg.rec_spec)
                    for _ in range(cfg.rec_sets * cfg.rec_parts)
                ],
            )
        else:
            for _ in range(cfg.rec_sets):
                ts = [layer.allocate(mesh, cfg.rec_spec) for _ in range(cfg.rec_parts)]
                self._rec_set_tensors.append(ts)
                self._rec_sets.append(
                    [_Buf(t, cfg.rec_spec, i) for i, t in enumerate(ts)]
                )
            self._kv_scratch = layer.allocate(mesh, cfg.kv_spec)
            self._rec_scratch = layer.allocate(mesh, cfg.rec_spec)

    def _rendezvous_path(self, rank: int) -> str:
        return os.path.join(self.cfg.control_dir, RENDEZVOUS_DIR, f"rank{rank}.json")

    def _write_rendezvous(self) -> None:
        p = self._rendezvous_path(int(self.rank))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        d = dict(
            self.cfg.spec_table(),
            engine_id=self.engine_id,
            role=self.role,
            epoch=self._epoch,
            pid=os.getpid(),
        )
        tmp = p + f".{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(d, f)
        os.replace(tmp, p)

    def _read_peer_rendezvous(self) -> None:
        peer_rank = self.cfg.receiver_rank if self.is_producer else self.cfg.sender_rank
        p = self._rendezvous_path(peer_rank)
        try:
            with open(p) as f:
                peer = json.load(f)
        except (OSError, ValueError) as e:
            raise RuntimeError(
                f"fabric rendezvous: peer rank {peer_rank} left no {p} ({e}); both "
                "engines must use the same fabric_control_dir on one host"
            ) from e
        mine = self.cfg.spec_table()
        theirs = {k: peer.get(k) for k in mine}
        if theirs != mine:
            raise RuntimeError(
                f"fabric spec table mismatch between ranks: mine {mine} vs peer "
                f"{theirs} "
                "(kv/rec dtype, connections, fifo, rec_parts must be identical)"
            )
        want_role = "consumer" if self.is_producer else "producer"
        if peer.get("role") != want_role:
            raise RuntimeError(
                f"peer rank {peer_rank} is a {peer.get('role')}, expected {want_role}"
            )
        self._peer = peer
        if self.is_consumer:
            self._recv_epoch = str(peer.get("epoch"))
            self._next_recv_seq = 0

    def _warm_programs(self) -> None:
        """Compile every program used inside a hook call, in lockstep with the peer."""
        cfg, layer, mesh, sock = self.cfg, self.layer, self._mesh, self._sock
        if self.is_producer:
            kv_pool, rec_pool = self._producer_pools()
            kv = kv_pool.take(1)
            rec = rec_pool.take(1)
            if kv is None or rec is None:
                raise RuntimeError(
                    f"fabric warm-up: pools too small (kv {len(kv_pool)}, rec "
                    f"{len(rec_pool)}); fabric_export_budget_bytes / fabric_rec_sets"
                )
            blk = layer.allocate(mesh, cfg.kv_block_major_spec)
            try:
                layer.relayout_copy(blk, kv[0].tensor)  # permute + reshape + copy
                # send kv + rec (the peer posts the matching recvs)
                layer.send(kv[0].tensor, sock)
                layer.send(rec[0].tensor, sock)
            finally:
                layer.deallocate(blk)
            # copy programs: kv head-major copy and rec copy (write_host / head-major)
            spare_kv = kv_pool.take(1)
            spare_rec = rec_pool.take(1)
            if spare_kv is not None:
                layer.copy(kv[0].tensor, spare_kv[0].tensor)
                kv_pool.give(spare_kv)
            if spare_rec is not None:
                layer.copy(rec[0].tensor, spare_rec[0].tensor)
                rec_pool.give(spare_rec)
            kv_pool.give(kv)
            rec_pool.give(rec)
        else:
            layer.recv(self._kv_scratch, sock)
            layer.recv(self._rec_scratch, sock)
            if self._rec_sets:
                layer.copy(self._rec_sets[0][0].tensor, self._rec_scratch)

    def _producer_pools(self) -> tuple[_Pool, _Pool]:
        if self._kv_pool is None or self._rec_pool is None:
            raise RuntimeError("fabric producer pools are not allocated (not started?)")
        return self._kv_pool, self._rec_pool

    def _teardown_device(self, release_tensors: bool = True) -> None:
        layer = self._layer
        if layer is None:
            return
        if self._sock is not None:
            with contextlib.suppress(Exception):
                layer.close_socket(self._sock)
            self._sock = None
        if not release_tensors:
            logger.error(
                "fabric start aborted inside a rendezvous: leaving %d kv / %d rec pool "
                "tensors allocated (a helper thread may still be in the native call)",
                len(self._kv_pool) if self._kv_pool else 0,
                len(self._rec_pool) if self._rec_pool else len(self._rec_set_tensors),
            )
            self._kv_pool = self._rec_pool = None
            self._rec_sets.clear()
            self._rec_set_tensors = []
            self._kv_scratch = self._rec_scratch = None
            return
        tensors: list[Any] = []
        if self._kv_pool is not None:
            tensors += self._kv_pool.tensors()
        if self._rec_pool is not None:
            tensors += self._rec_pool.tensors()
        for ts in self._rec_set_tensors:
            tensors += ts
        tensors += [t for t in (self._kv_scratch, self._rec_scratch) if t is not None]
        for t in tensors:
            with contextlib.suppress(Exception):
                layer.deallocate(t)
        self._kv_pool = self._rec_pool = None
        self._rec_sets.clear()
        self._rec_set_tensors = []
        self._kv_scratch = self._rec_scratch = None

    def shutdown(self) -> None:
        if self._shut:
            return
        self._shut = True
        unregister_transport(self)
        if self._ctrl is not None:
            with contextlib.suppress(Exception):
                self._ctrl.shutdown()
        if self._started:
            with contextlib.suppress(OSError):
                os.unlink(self._rendezvous_path(int(self.rank)))
        self._teardown_device()
        self._started = False

    # -- shared helpers ----------------------------------------------------------------
    @staticmethod
    def _sidecar_path(seg_dir: str) -> str:
        return os.path.join(seg_dir, SIDECAR_NAME)

    @staticmethod
    def _read_sidecar(seg_dir: str) -> dict[str, Any] | None:
        try:
            with open(os.path.join(seg_dir, SIDECAR_NAME)) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _marker_dir(self, engine: str) -> str:
        return os.path.join(self.cfg.control_dir, SENT_DIR, engine)

    def _marker_path(self, engine: str, hx: str) -> str:
        return os.path.join(self._marker_dir(engine), f"{hx}.json")

    def _read_marker_file(self, engine: str, hx: str) -> dict[str, Any] | None:
        try:
            with open(self._marker_path(engine, hx)) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None  # absent, or mid-rename (the producer writes tmp + rename)

    @staticmethod
    def _canonical_items(
        manifest: Manifest, written: set[tuple[str, int]]
    ) -> tuple[list[tuple[str, int]], int]:
        """Chunk-outer, manifest-part-inner K/V items, then recs in manifest order --
        the order BOTH import hooks read them in (kv_transfer.py / hooks.py)."""
        kv_parts = [p for p in manifest.parts if p.kind == "kv_blocks"]
        rec_parts = [p for p in manifest.parts if p.kind == "gdn_rec"]
        nch = max((p.nchunks for p in kv_parts), default=0)
        items: list[tuple[str, int]] = []
        for c in range(nch):
            for p in kv_parts:
                if (p.name, c) in written:
                    items.append((p.name, c))
        n_kv = len(items)
        for p in rec_parts:
            if (p.name, 0) in written:
                items.append((p.name, 0))
        return items, n_kv

    # -- producer ----------------------------------------------------------------------
    def open_put(self, xfer_id: str, manifest: Manifest) -> PutHandle | None:
        self._require_started()
        if not self.is_producer:
            raise RuntimeError("open_put on a producer-role fabric transport")
        ctrl, cfg = self.control, self.cfg
        kv_pool, rec_pool = self._producer_pools()
        self._send_claimed()
        self._reclaim()
        engine, hx = parse_xfer_id(xfer_id)
        if engine != self.engine_id:
            raise ValueError(
                f"xfer_id {xfer_id} does not belong to producer {self.engine_id}"
            )
        manifest = coerce_manifest(manifest)
        if manifest.layout_version != LAYOUT_VERSION:
            raise ValueError("manifest layout_version mismatch")
        kv_parts = [p for p in manifest.parts if p.kind == "kv_blocks"]
        rec_parts = [p for p in manifest.parts if p.kind == "gdn_rec"]
        for p in kv_parts:
            if p.spec_key() != cfg.kv_block_major_spec:
                raise ValueError(
                    f"{p.name} spec {p.spec_key()} != fabric kv chunk spec "
                    f"{cfg.kv_block_major_spec} (fabric_kv_dtype?)"
                )
        for p in rec_parts:
            if p.spec_key() != cfg.rec_spec:
                raise ValueError(
                    f"{p.name} spec {p.spec_key()} != fabric rec spec {cfg.rec_spec}"
                )
        if len(rec_parts) > cfg.rec_parts:
            raise ValueError(
                f"{len(rec_parts)} rec parts > fabric_rec_parts {cfg.rec_parts}"
            )
        with self._lock:
            if xfer_id in self._exports:
                logger.warning(
                    "PD fabric: %s is still in flight (published, not yet consumed); "
                    "refusing a second export",
                    xfer_id,
                )
                self.stats["budget_refusals"] += 1
                return None
            n_kv = sum(p.nchunks for p in kv_parts)
            kv = kv_pool.take(n_kv)
            if kv is None:
                self.stats["budget_refusals"] += 1
                logger.warning(
                    "PD fabric: export pool short (%d free of %d kv buffers, need %d): "
                    "refusing %s",
                    kv_pool.free,
                    len(kv_pool),
                    n_kv,
                    xfer_id,
                )
                return None
            rec = rec_pool.take(len(rec_parts))
            if rec is None:
                kv_pool.give(kv)
                self.stats["budget_refusals"] += 1
                logger.warning(
                    "PD fabric: rec pool short (%d free, need %d): refusing %s",
                    rec_pool.free,
                    len(rec_parts),
                    xfer_id,
                )
                return None
            h = ctrl.open_put(xfer_id, manifest)
            if h is None:
                kv_pool.give(kv)
                rec_pool.give(rec)
                self.stats["budget_refusals"] += 1
                return None
            kv_map: dict[tuple[str, int], _Buf] = {}
            i = 0
            for p in kv_parts:
                for c in range(p.nchunks):
                    kv_map[(p.name, c)] = kv[i]
                    i += 1
            rec_map = {p.name: rec[j] for j, p in enumerate(rec_parts)}
            exp = _Export(
                xfer_id, hx, manifest, kv_map, rec_map, nbytes=manifest.total_nbytes
            )
            st = ctrl.put_state(xfer_id)
            sinks: dict[str, Sink] = dict(h.sinks)  # taps keep the dumpfile sink
            for p in kv_parts + rec_parts:
                sinks[p.name] = FabricSink(p, exp, self.layer, cfg, st)
            self._exports[xfer_id] = exp
        return PutHandle(xfer_id, manifest, sinks, lease_expiry_ts=h.lease_expiry_ts)

    def finish_export(self, h: PutHandle, status: Literal["READY", "FAILED"]) -> None:
        """Publish the control segment.  READY enqueues NOTHING on the socket: the
        sends follow the consumer's claim (``_send_claimed`` from ``pump``)."""
        self._require_started()
        ctrl = self.control
        self._send_claimed()
        self._reclaim()
        with self._lock:
            exp = self._exports.get(h.xfer_id)
            if exp is None or exp.published:
                return  # abandoned / already published: idempotent
            st = ctrl.put_state(h.xfer_id)
            if st is None:  # control segment vanished (abandon raced): free buffers
                self._free_export(exp)
                self._exports.pop(h.xfer_id, None)
                return
            items, n_kv = self._canonical_items(exp.manifest, exp.written)
            side: dict[str, Any] = {
                "layout_version": LAYOUT_VERSION,
                "epoch": self._epoch,
                "seq": None,  # assigned in send order by _send_export (marker)
                "status": status,
                "claim_gated": True,
                "nitems": len(items) if status == "READY" else 0,
                "n_kv": n_kv if status == "READY" else 0,
                "items": [list(x) for x in items] if status == "READY" else [],
                "kv_spec": list(self.cfg.spec_table()["kv_spec"]),
                "rec_spec": list(self.cfg.spec_table()["rec_spec"]),
                "producer_pid": os.getpid(),
            }
            with open(self._sidecar_path(st.tmp_dir), "w") as f:
                json.dump(side, f)
            exp.status = status
            exp.items, exp.n_kv = (items, n_kv) if status == "READY" else ([], 0)
            ctrl.finish_export(h, status)  # part table, status LAST, atomic rename
            exp.published = True
            exp.ready_ts = time.perf_counter()
            if status == "READY" and items:
                self.stats["exports_ready"] += 1
                logger.info(
                    "PD fabric: export %s READY published: %d items (%.1f MiB) held "
                    "for the consumer's claim (nothing enqueued)",
                    h.xfer_id,
                    len(items),
                    exp.nbytes / 2**20,
                )
                return  # buffers stay until sent and consumed / drained
            # FAILED, or READY without device items: nothing will ever be on the
            # channel; buffers go back now, the segment is the control plane's
            if status == "FAILED":
                self.stats["exports_failed"] += 1
            self._free_export(exp)
            self._exports.pop(h.xfer_id, None)
            logger.info(
                "PD fabric: export %s %s published with no device items (%.1f MiB)",
                h.xfer_id,
                status,
                exp.nbytes / 2**20,
            )

    def _send_claimed(self) -> int:
        """Enqueue the sends of every published export the consumer has CLAIMED
        (open ``.claimed-*`` dir of a LIVE consumer pid, not a ``.closing`` fence),
        in ``open_put`` order (= publish order: the prefill rank runs
        ``max_num_seqs`` 1), one ``seq`` each in send order, then write the marker.
        A claim whose consumer pid is dead is left to the janitor: sending for it
        would park the channel with nobody to post the recvs (this narrows, not
        closes, the die-after-check window; the pair is one tt-run job, so a dead
        consumer means a restart anyway).  Engine thread only (device ops).
        Returns the number of exports sent."""
        if not self._exports:
            return 0
        ctrl = self.control
        with self._lock:
            pending = [
                e
                for e in self._exports.values()
                if e.published and not e.sent and not e.abandoned
            ]
            if not pending:
                return 0
            names = ctrl.list_engine_dir(self.engine_id)
            claimed = ctrl.claimed_hexes(names)
            dead = ctrl.dead_claim_hexes(self.engine_id, names) if claimed else set()
            n = 0
            for exp in pending:
                if exp.hx not in claimed:
                    continue
                if exp.hx in dead:
                    self.stats["dead_claims_skipped"] += 1
                    if not exp.dead_claim_warned:
                        exp.dead_claim_warned = True
                        logger.warning(
                            "PD fabric: %s is claimed by a dead consumer; not sending "
                            "(the janitor sweeps the claim, the buffers come back)",
                            exp.xfer_id,
                        )
                    continue
                self._send_export(exp)
                n += 1
            return n

    def _send_export(self, exp: _Export) -> None:
        seq = self._next_seq
        t0 = time.perf_counter()
        n_sent = 0
        status = "READY"
        try:
            for name, c in exp.items[: exp.n_kv]:
                self.layer.send(exp.kv_bufs[(name, c)].tensor, self._sock)
                n_sent += 1
            for name, _ in exp.items[exp.n_kv :]:
                self.layer.send(exp.rec_bufs[name].tensor, self._sock)
                n_sent += 1
        except Exception:
            # The n_sent items already enqueued keep their place in the FIFO
            # channel: the consumer must drain exactly them at this seq before
            # anything sent later, so the marker keeps the seq (consumed only if
            # something was enqueued) and lists exactly them, status FAILED.
            logger.exception(
                "PD fabric: send %d/%d of %s raised; marker FAILED with %d items "
                "parked on the channel",
                n_sent + 1,
                len(exp.items),
                exp.xfer_id,
                n_sent,
            )
            status = "FAILED"
        enqueue_ms = (time.perf_counter() - t0) * 1e3
        if n_sent > 0:
            self._next_seq = seq + 1
            exp.seq = seq
        exp.sent = True
        exp.status = status
        marker = {
            "layout_version": LAYOUT_VERSION,
            "epoch": self._epoch,
            "xfer_id": exp.xfer_id,
            "seq": exp.seq,
            "status": status,
            "nitems": n_sent,
            "n_kv": min(n_sent, exp.n_kv),
            "items": [list(x) for x in exp.items[:n_sent]],
            "kv_spec": list(self.cfg.spec_table()["kv_spec"]),
            "rec_spec": list(self.cfg.spec_table()["rec_spec"]),
            "producer_pid": os.getpid(),
        }
        self._write_marker(exp.hx, marker)
        self.stats["sends"] += n_sent
        if status == "READY":
            self.stats["exports_sent"] += 1
        else:
            self.stats["exports_failed"] += 1
        if n_sent == 0:
            # nothing on the channel: the consumer drops its claim on this marker;
            # the buffers are free now
            self._free_export(exp)
            self._exports.pop(exp.xfer_id, None)
        # else: buffers stay until the consumer received / drained the items
        logger.info(
            "PD fabric: export %s %s seq=%s items=%d/%d (%.1f MiB) enqueued in "
            "%.2f ms (%.1f ms after READY)",
            exp.xfer_id,
            status,
            exp.seq,
            n_sent,
            len(exp.items),
            exp.nbytes / 2**20,
            enqueue_ms,
            (time.perf_counter() - exp.ready_ts) * 1e3 if exp.ready_ts else 0.0,
        )

    def _write_marker(self, hx: str, marker: dict[str, Any]) -> None:
        """Atomic (tmp + rename) into the producer-owned marker dir.  A failure here
        would leave enqueued sends the consumer never learns about: raise."""
        path = self._marker_path(self.engine_id, hx)
        tmp = f"{path}.{os.getpid()}.tmp"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(tmp, "w") as f:
                json.dump(marker, f)
            os.replace(tmp, path)
        except OSError as e:
            raise RuntimeError(
                f"PD fabric: cannot write the send marker {path} ({e}); "
                f"{marker['nitems']} items of {marker['xfer_id']} are enqueued on the "
                "channel with no record the consumer can drain by"
            ) from e

    def _free_export(self, exp: _Export) -> None:
        if self._kv_pool is not None:
            self._kv_pool.give(list(exp.kv_bufs.values()))
        if self._rec_pool is not None:
            self._rec_pool.give(list(exp.rec_bufs.values()))
        exp.kv_bufs, exp.rec_bufs = {}, {}

    def abandon(self, xfer_id: str) -> None:
        if not self._started or not self.is_producer:
            return
        ctrl = self.control
        self._reclaim()  # never _send_claimed here: abandon starts no sends
        with self._lock:
            exp = self._exports.get(xfer_id)
            if exp is None:
                ctrl.abandon(xfer_id)
                return
            if not exp.published:
                self._exports.pop(xfer_id, None)
                self._free_export(exp)
                ctrl.abandon(xfer_id)
                return
            if exp.sent:
                # on the channel: the consumer that claimed it receives or drains
                # it; buffers come back through _reclaim
                logger.info(
                    "PD fabric: abandon(%s) after the sends: the consumer's claim "
                    "settles it",
                    xfer_id,
                )
                return
            # published, nothing sent: never send; unclaimed -> unlink the segment
            # (buffers back through _reclaim); claimed -> flip the claim's header
            # to RELEASED so the consumer drops it instead of waiting for a marker
            exp.abandoned = True
            ctrl.abandon(xfer_id)  # removes the pub dir if still unclaimed
            n = ctrl.mark_claims_released(exp.hx)  # a claim that raced the unlink
            logger.info(
                "PD fabric: abandon(%s) before any send: segment removed (%d claim "
                "dir(s) marked RELEASED)",
                xfer_id,
                n,
            )
        self._reclaim()

    def _reclaim(self) -> None:
        """Return the buffers of every published export whose segment left the disk
        (consumed, load-failed, released, drained or swept -- the consumer removes a
        claim dir only after posting every recv of the items on the channel)."""
        if not self._exports:
            return
        ctrl = self.control
        names = ctrl.list_engine_dir(self.engine_id)
        with self._lock:
            for xid, exp in list(self._exports.items()):
                if exp.published and not ctrl.any_dir_exists(
                    self.engine_id, exp.hx, names
                ):
                    self._free_export(exp)
                    del self._exports[xid]
                    self.stats["reclaimed"] += 1

    def export_complete(self, xfer_id: str) -> bool:
        """True once the consumer has taken (or drained) every item of ``xfer_id``."""
        if not self._started or not self.is_producer:
            return True
        self._reclaim()
        with self._lock:
            return xfer_id not in self._exports

    def outstanding_exports(self) -> int:
        with self._lock:
            return len(self._exports)

    def unsent_exports(self) -> int:
        """Published exports waiting for the consumer's claim (nothing enqueued)."""
        with self._lock:
            return sum(
                1
                for e in self._exports.values()
                if e.published and not e.sent and not e.abandoned
            )

    # -- consumer ----------------------------------------------------------------------
    def _adopt_epoch(self, epoch: str) -> None:
        if epoch != self._recv_epoch:
            logger.warning(
                "PD fabric: producer epoch %s -> %s; resetting receive sequence",
                self._recv_epoch,
                epoch,
            )
            self._recv_epoch = epoch
            self._next_recv_seq = 0
            self._by_seq.clear()
            self._markers.clear()

    def _marker(self, engine: str, hx: str) -> dict[str, Any] | None:
        """The producer's send marker of ``hx`` (cached once read: immutable)."""
        m = self._markers.get(hx)
        if m is None:
            m = self._read_marker_file(engine, hx)
            if m is not None:
                self._markers[hx] = m
        return m

    def _unlink_marker(self, engine: str, hx: str) -> None:
        self._markers.pop(hx, None)
        with contextlib.suppress(OSError):
            os.unlink(self._marker_path(engine, hx))

    def _list_markers(self, engine: str) -> list[str]:
        try:
            names = os.listdir(self._marker_dir(engine))
        except FileNotFoundError:
            return []
        return [n[: -len(".json")] for n in names if n.endswith(".json")]

    def _adopt_marker(self, xf: _Xfer, m: dict[str, Any]) -> None:
        """Marker seen for our claim: the items ACTUALLY enqueued and their seq."""
        epoch = str(m.get("epoch"))
        if epoch != self._recv_epoch:
            self._adopt_epoch(epoch)
        xf.marker_seen = True  # from here on _finalize may unlink it
        xf.marker_at = float(self._clock())
        xf.items = [tuple(x) for x in m.get("items", [])]
        xf.n_kv = int(m.get("n_kv", 0))
        xf.index_items()
        xf.send_failed = m.get("status") != "READY"
        seq = m.get("seq")
        xf.seq = None if seq is None else int(seq)
        if xf.seq is not None:
            self._by_seq[xf.seq] = xf
            if xf.send_failed:
                # parked items nobody can import: drained at their turn like a
                # released claim, whether or not the worker asks for it first
                xf.released = True

    def _marker_by_seq(self, engine: str, seq: int) -> dict[str, Any] | None:
        """Scan the marker dir for channel seq ``seq``: markers of our claims are
        adopted into their ``_Xfer`` on the way (``_by_seq`` fills), consumed or
        foreign-epoch markers are unlinked; returns an ORPHAN marker (no claim of
        ours) at ``seq``, else None."""
        orphan: dict[str, Any] | None = None
        for hx in self._list_markers(engine):
            m = self._marker(engine, hx)
            if m is None:
                continue  # torn (mid-rename): next scan
            xf = self._xfers.get(f"{engine}:{hx}")
            if xf is not None:
                if xf.seq is None and not xf.posted:
                    self._adopt_marker(xf, m)
                continue
            epoch = str(m.get("epoch"))
            if epoch != self._recv_epoch:
                self._adopt_epoch(epoch)  # a restarted producer: its markers rule
            mseq = m.get("seq")
            if mseq is None or int(mseq) < self._next_recv_seq:
                self._unlink_marker(engine, hx)  # nothing parked / already consumed
                continue
            if int(mseq) == seq:
                orphan = m
        return orphan

    def _settle_head(self, engine: str, upto_seq: int | None) -> bool:
        """Advance the receive sequence past drainable xfers at the head of the
        channel: released claims of ours and orphan markers (sends of a claim we
        fenced before the marker landed).  Returns True when the channel head is
        ``upto_seq`` (or, with None, when nothing drainable remains)."""
        act = self._active
        if act is not None and not act.complete and not act.released:
            return False  # an xfer is mid-receive: its remaining items come first
        while upto_seq is None or self._next_recv_seq < upto_seq:
            s = self._next_recv_seq
            xf = self._by_seq.get(s)
            if xf is None:
                m = self._marker_by_seq(engine, s)
                xf = self._by_seq.get(s)
                if xf is None:
                    if m is None:
                        return upto_seq is None  # head not sent yet
                    self._drain_orphan(engine, m)
                    continue
            if xf.released:
                self._drain_released(xf)
                continue
            return False  # a live claim owns the head: the hook posts its recvs
        return True

    def _drain_items(self, xf: _Xfer) -> None:
        layer, sock = self.layer, self._sock
        n = 0
        while xf.cursor < xf.nitems:
            if xf.cursor < xf.n_kv:
                layer.recv(self._kv_scratch, sock)
            else:
                layer.recv(self._rec_scratch, sock)
            xf.cursor += 1
            n += 1
        self.stats["drains"] += n
        self.stats["recvs"] += n
        if self._active is xf:
            self._active = None

    def _drain_orphan(self, engine: str, m: dict[str, Any]) -> None:
        """Sends the producer enqueued for a claim we had already fenced and
        dropped: receive them into scratch at their turn."""
        xfer_id = str(m.get("xfer_id"))
        hx = parse_xfer_id(xfer_id)[1]
        xf = _Xfer(
            xfer_id,
            hx,
            int(m["seq"]),
            [tuple(x) for x in m.get("items", [])],
            int(m.get("n_kv", 0)),
            None,
        )
        self._drain_items(xf)
        self._next_recv_seq = xf.seq + 1
        self._unlink_marker(engine, hx)
        self.stats["drained_xfers"] += 1
        self.stats["orphan_markers"] += 1
        logger.info(
            "PD fabric: drained %s (sent after our claim was dropped, %d items) into "
            "scratch",
            xfer_id,
            xf.nitems,
        )

    def _drain_released(self, xf: _Xfer) -> None:
        """A released claim of ours at the channel head: drain what is left, then
        drop the claim dir (the producer reclaims its buffers)."""
        n_left = xf.nitems - xf.cursor
        self._drain_items(xf)
        self._next_recv_seq = max(self._next_recv_seq, int(xf.seq) + 1)
        self._finalize(xf)
        self.control.release_remote(xf.xfer_id)  # RELEASED + claim dir removed
        self.stats["drained_xfers"] += 1
        logger.info(
            "PD fabric: drained %s (released, %d of %d items) into scratch",
            xf.xfer_id,
            n_left,
            xf.nitems,
        )

    def _finalize(self, xf: _Xfer) -> None:
        """Forget a claim of ours: rec set back, indexes cleared, and the marker
        gone -- but ONLY a marker we have read (``_adopt_marker``).  A marker we
        never read may be landing right now (the producer listed the claim before
        our fence and is finishing its enqueue): unlinking it blind would leave its
        sends parked on the channel with no record to drain by (the D5 shape).
        Such a marker is an ORPHAN the pump drains at its turn."""
        if xf.rec_set is not None:
            self._rec_sets.append(xf.rec_set)
            xf.rec_set = None
        if self._active is xf:
            self._active = None
        self._xfers.pop(xf.xfer_id, None)
        self._ctrl_sources.pop(xf.xfer_id, None)
        if xf.seq is not None:
            self._by_seq.pop(xf.seq, None)
        peer = self.peer_engine_id
        if peer is not None and xf.marker_seen:
            self._unlink_marker(peer, xf.hx)
        else:
            self._markers.pop(xf.hx, None)

    def _drop_unsent(self, xf: _Xfer, why: str) -> None:
        """Drop a claim nothing was (or will be) sent for: no drain, claim dir
        removed, the producer's buffers come back through its ``_reclaim``.  The
        caller fenced the claim and found no marker; the marker path is left alone
        (``_finalize`` unlinks only an adopted marker) and re-read once the drop is
        done: a marker that landed meanwhile is an orphan for ``pump`` to drain
        (``wants_pump`` lists it, so the idle ticker wakes the engine for it)."""
        engine = parse_xfer_id(xf.xfer_id)[0]
        self._finalize(xf)
        self.control.release_remote(xf.xfer_id)
        late = None if xf.marker_seen else self._read_marker_file(engine, xf.hx)
        if late is not None and late.get("seq") is not None:
            self.stats["late_markers"] += 1
            logger.info(
                "PD fabric: dropped claim %s (%s) but the producer's marker landed "
                "meanwhile (seq %s, %d items): orphan, drained by the pump at its turn",
                xf.xfer_id,
                why,
                late.get("seq"),
                int(late.get("nitems", 0)),
            )
            return
        logger.info(
            "PD fabric: dropped claim %s (%s; nothing on the channel)", xf.xfer_id, why
        )

    def open_get(self, desc: Any) -> GetHandle | None:
        self._require_started()
        if not self.is_consumer:
            raise RuntimeError("open_get on a producer-role fabric transport")
        ctrl, cfg = self.control, self.cfg
        xfer_id = str(getattr(desc, "xfer_id", desc))
        try:
            engine, hx = parse_xfer_id(xfer_id)
        except ValueError as e:
            return GetHandle(xfer_id, None, "FAILED", {}, str(e))
        tdesc = getattr(desc, "transport", None)
        if isinstance(tdesc, dict) and tdesc.get("kind", self.KIND) != self.KIND:
            return GetHandle(
                xfer_id,
                None,
                "FAILED",
                {},
                f"transport mismatch: {tdesc} vs {self.descriptor()}",
            )
        with self._lock:
            xf = self._xfers.get(xfer_id)
            if xf is not None:  # our claim: marker / head gate / re-open
                return self._reopen(engine, xf)
            _tmp, pub, mine = ctrl.segment_dirs(engine, hx)
            if not os.path.isdir(pub):
                # WRITING (None) / MISSING / dead-producer FAILED: the control plane
                # knows; nothing of ours is on the channel
                return ctrl.open_get(_Desc(xfer_id))
            hp = os.path.join(pub, "header")
            try:
                hdr = read_header(hp, full=False)
            except (OSError, ValueError):
                return None  # torn publish; retry
            if hdr.status != READY:
                # FAILED / RELEASED unclaimed: nothing on the channel (claim-gated);
                # the control plane claims and answers FAILED
                return ctrl.open_get(_Desc(xfer_id))
            side = self._read_sidecar(pub)
            if side is None:
                g = ctrl.open_get(_Desc(xfer_id))
                if g is not None and g.ready():
                    ctrl.finish_import(g, ok=False)
                return GetHandle(
                    xfer_id, None, "FAILED", {}, "segment has no fabric sidecar"
                )
            epoch = str(side.get("epoch"))
            if epoch != self._recv_epoch:
                self._adopt_epoch(epoch)
            items = [tuple(x) for x in side.get("items", [])]
            n_kv = int(side.get("n_kv", 0))
            if not items:
                # READY with nothing to send (no device parts): plain segment
                return ctrl.open_get(_Desc(xfer_id))
            mine_specs = cfg.spec_table()
            if (
                side.get("kv_spec") != mine_specs["kv_spec"]
                or side.get("rec_spec") != mine_specs["rec_spec"]
            ):
                raise RuntimeError(
                    f"PD fabric: {xfer_id} was exported with specs {side['kv_spec']}/"
                    f"{side['rec_spec']} but this consumer stages {cfg.spec_table()}: "
                    "direct-mode receive is impossible (fabric_kv_dtype mismatch)"
                )
            n_rec = len(items) - n_kv
            if n_rec > cfg.rec_parts:
                # never claim it: unclaimed, nothing is sent; the worker releases
                # it (RELEASED) and the producer janitor sweeps it
                return GetHandle(
                    xfer_id,
                    None,
                    "FAILED",
                    {},
                    f"{n_rec} rec items > rec_parts {cfg.rec_parts}",
                )
            if n_rec > 0 and not self._rec_sets:
                return None  # every rec set is held by a KV_DONE handle: retry
            g = ctrl.open_get(_Desc(xfer_id))  # CLAIM: the producer's send trigger
            if g is None or not g.ready():
                return g
            rec_set = self._rec_sets.popleft() if n_rec > 0 else None
            xf = _Xfer(
                xfer_id,
                hx,
                None,
                items,
                n_kv,
                rec_set,
                claimed_ts=time.perf_counter(),
                claimed_at=float(self._clock()),
            )
            xf.index_items()
            self._xfers[xfer_id] = xf
            self._ctrl_sources[xfer_id] = g.sources
            self.stats["imports"] += 1
            return self._reopen(engine, xf, first=True)

    def _reopen(
        self, engine: str, xf: _Xfer, *, first: bool = False
    ) -> GetHandle | None:
        """Our claim: None until the producer's marker is there and the xfer is at
        the channel head; then the READY handle (once: the hook posts the recvs)."""
        if xf.posted:
            return self._handle(xf)  # re-open of a live import
        if xf.released and not xf.send_failed:
            return GetHandle(xf.xfer_id, None, "MISSING", {}, "released")
        if xf.seq is None and not xf.send_failed:
            m = self._marker(engine, xf.hx)
            if m is None and first and self.cfg.claim_wait_s > 0:
                # an idle producer answers a claim within its idle tick: spin a
                # little so the recvs go out in THIS step (the common 1-user path)
                deadline = time.perf_counter() + self.cfg.claim_wait_s
                while m is None and time.perf_counter() < deadline:
                    time.sleep(0.0002)
                    m = self._marker(engine, xf.hx)
            if m is None:
                if self.control.claim_status(xf.xfer_id) == RELEASED:
                    # the producer abandoned the export before any send
                    self._drop_unsent(xf, "released by the producer before the send")
                    return GetHandle(
                        xf.xfer_id, None, "MISSING", {}, "released by the producer"
                    )
                return None  # sends not enqueued yet: the worker re-polls
            self._adopt_marker(xf, m)
        if xf.send_failed:
            n = xf.nitems
            if xf.seq is None:  # nothing was enqueued
                self._drop_unsent(xf, "producer send failed before any item")
            else:  # parked items (released by _adopt_marker): drain if at the head
                self._settle_head(engine, None)
            return GetHandle(
                xf.xfer_id,
                None,
                "FAILED",
                {},
                f"producer export failed after {n} items were sent",
            )
        if not self._settle_head(engine, int(xf.seq)):
            return None  # an older sent xfer is pending on the channel: retry
        xf.posted = True
        self._active = xf
        if xf.n_kv == 0:
            self._post_recs(xf)
        logger.info(
            "PD fabric: %s at the channel head seq=%d (%d items) %.1f ms after the "
            "claim; receiving",
            xf.xfer_id,
            xf.seq,
            xf.nitems,
            (time.perf_counter() - xf.claimed_ts) * 1e3,
        )
        return self._handle(xf)

    def _handle(self, xf: _Xfer) -> GetHandle:
        ctrl = self.control
        hdr = ctrl.get_header(xf.xfer_id)
        ctrl_sources = self._ctrl_sources.get(xf.xfer_id, {})
        if hdr is None:
            return GetHandle(xf.xfer_id, None, "FAILED", {}, "claim vanished")
        sources: dict[str, Source] = {}
        for rec in hdr.parts:
            p = rec.spec
            if p.kind == "kv_blocks":
                sources[p.name] = FabricSource(
                    p, xf, self, xf.kv_present.get(p.name, 0)
                )
            elif p.kind == "gdn_rec":
                sources[p.name] = FabricSource(
                    p, xf, self, 1 if p.name in xf.rec_index else 0
                )
            else:
                sources[p.name] = ctrl_sources.get(p.name)  # taps: dumpfile rows
        return GetHandle(xf.xfer_id, hdr.manifest, "READY", sources)

    def _recv_kv_item(self, xf: _Xfer, item: tuple[str, int], staging: Any) -> None:
        with self._lock:
            if xf.failed:
                raise RuntimeError(
                    f"PD fabric: {xf.xfer_id} already failed: {xf.failed}"
                )
            if not xf.posted:
                raise RuntimeError(
                    f"PD fabric: {xf.xfer_id} item {item} requested before the "
                    "producer's sends were on the channel (open_get was not READY)"
                )
            if xf.cursor >= xf.n_kv:
                raise RuntimeError(
                    f"PD fabric: {xf.xfer_id} item {item} requested after all "
                    f"{xf.n_kv} K/V items were received"
                )
            expected = xf.items[xf.cursor]
            if tuple(item) != tuple(expected):
                self.stats["order_violations"] += 1
                xf.failed = f"order violation: got {item}, channel head is {expected}"
                raise RuntimeError(f"PD fabric: {xf.xfer_id} {xf.failed}")
            got = self.layer.spec_of(staging)
            if got != self.cfg.kv_spec:
                xf.failed = f"staging spec {got} != {self.cfg.kv_spec}"
                raise ValueError(f"PD fabric: {xf.xfer_id} {xf.failed}")
            self.layer.recv(staging, self._sock)
            self.stats["recvs"] += 1
            xf.cursor += 1
            if xf.cursor == xf.n_kv:
                self._post_recs(xf)

    def _post_recs(self, xf: _Xfer) -> None:
        """Right after the last K/V item: receive every rec into the xfer's set."""
        n_rec = xf.nitems - xf.n_kv
        if n_rec == 0:
            if self._active is xf:
                self._active = None
            self._next_recv_seq = max(self._next_recv_seq, int(xf.seq) + 1)
            return
        if xf.rec_set is None:
            raise RuntimeError(
                f"PD fabric: {xf.xfer_id} has {n_rec} rec items but no rec set"
            )
        for k in range(n_rec):
            self.layer.recv(xf.rec_set[k].tensor, self._sock)
        self.stats["recvs"] += n_rec
        xf.cursor = xf.nitems
        if self._active is xf:
            self._active = None
        self._next_recv_seq = max(self._next_recv_seq, int(xf.seq) + 1)

    def _copy_rec_item(self, xf: _Xfer, part: str, rec_staging: Any) -> None:
        with self._lock:
            if xf.failed:
                raise RuntimeError(
                    f"PD fabric: {xf.xfer_id} already failed: {xf.failed}"
                )
            if not xf.complete:
                raise RuntimeError(
                    f"PD fabric: {xf.xfer_id} rec {part} requested before the K/V "
                    "items "
                    f"were all received ({xf.cursor}/{xf.nitems})"
                )
            k = xf.rec_index.get(part)
            if k is None or xf.rec_set is None:
                raise RuntimeError(f"PD fabric: {xf.xfer_id} has no rec item {part}")
            got = self.layer.spec_of(rec_staging)
            if got != self.cfg.rec_spec:
                raise ValueError(f"PD fabric: rec staging {got} != {self.cfg.rec_spec}")
            self.layer.copy(xf.rec_set[k].tensor, rec_staging)

    def finish_import(self, h: GetHandle, ok: bool) -> None:
        if not self._started:
            return
        ctrl = self.control
        with self._lock:
            xf = self._xfers.get(h.xfer_id)
            if xf is None:
                ctrl.finish_import(h, ok)  # CONSUMED | LOAD_FAILED + unlink
                return
            if not xf.posted:
                # a FAILED handle of ours (send failed) or a worker giving up
                # before READY: the release path settles it (drain at its turn)
                if not xf.released:
                    self._release_claimed(xf)
                return
            if not xf.complete:
                # posted and mid-receive = at the channel head: drain the rest now
                self._drain_items(xf)
                self._next_recv_seq = max(self._next_recv_seq, int(xf.seq) + 1)
            self._finalize(xf)
            ctrl.finish_import(h, ok)  # CONSUMED | LOAD_FAILED + unlink

    def release_remote(self, xfer_id: str) -> None:
        if not self._started or not self.is_consumer:
            return
        ctrl = self.control
        try:
            engine, hx = parse_xfer_id(xfer_id)
        except ValueError:
            return
        with self._lock:
            xf = self._xfers.get(xfer_id)
            if xf is not None:  # claimed by us
                self._release_claimed(xf)
                return
            # unclaimed (or missing / .tmp): nothing of it is on the channel; the
            # control plane marks a published segment RELEASED for the janitor
            ctrl.release_remote(xfer_id)

    def _release_claimed(self, xf: _Xfer) -> None:
        """Drop a claim of ours.  Unsent: FENCE first (the producer will not start
        sends for a ``.closing`` dir), then read the marker -- absent: nothing is or
        will be on the channel; present: its items are parked.  Sent: drain at the
        channel turn (now when at the head, else from ``pump`` / a later
        ``open_get``)."""
        if xf.released:
            return
        engine = parse_xfer_id(xf.xfer_id)[0]
        if xf.seq is None and not xf.send_failed:
            self.control.fence_claim(xf.xfer_id)
            m = self._marker(engine, xf.hx)
            if m is None:
                self._drop_unsent(xf, "released before the producer sent")
                return
            self._adopt_marker(xf, m)
        if xf.seq is None:  # marker says nothing was enqueued
            self._drop_unsent(xf, "producer sent nothing")
            return
        xf.released = True
        if xf.posted:
            # the READY handle was issued, so it is the channel head (or complete):
            # drain what is left now; the rec set comes back in CQ order behind
            # the recvs already posted into it
            self._drain_released(xf)
            return
        if xf.rec_set is not None:
            self._rec_sets.append(xf.rec_set)  # no recv targets it: back now
            xf.rec_set = None
        if not self._settle_head(engine, None):
            logger.info(
                "PD fabric: %s released with %d items on the channel behind seq %d; "
                "drained at its turn",
                xf.xfer_id,
                xf.nitems - xf.cursor,
                self._next_recv_seq,
            )

    # -- both roles --------------------------------------------------------------------
    def pump(self) -> None:
        """The protocol clock, ENGINE THREAD ONLY (device ops): the producer enqueues
        the sends of newly claimed exports and reclaims finished ones; the consumer
        drains released claims and orphan markers at the head of the channel.
        Called by ``TTKVWorker.end_step`` after every step and by the rank's idle
        ticker while the engine is idle."""
        if not self._started:
            return
        if self.is_producer:
            self._send_claimed()
            self._reclaim()
            return
        peer = self.peer_engine_id
        if peer is None:
            return
        with self._lock:
            self._settle_head(peer, None)

    def wants_pump(self) -> bool:
        """Host-only (any thread): would ``pump()`` have something to do soon?
        Producer: an export is published (a claim may appear / a segment may go).
        Consumer: a claim of ours is open, or a marker exists (a live import, or
        the sends of a claim we dropped, to drain)."""
        if not self._started:
            return False
        with self._lock:
            if self.is_producer:
                return bool(self._exports)
            if self._xfers:
                return True
            peer = self.peer_engine_id
            return peer is not None and bool(self._list_markers(peer))

    def pending_receive_seq(self) -> int:
        return self._next_recv_seq

    def claim_deadline(self, xfer_id: str) -> float | None:
        """Consumer, host-only (any thread): the deadline (transport ``clock``, wall
        time in production) of our pending CLAIM on ``xfer_id`` for the worker's
        lease check (``TTKVWorker._poll_ready``); None when we hold no pending claim
        (not claimed yet: the descriptor's READY lease applies; posted / released /
        finished: nothing left to time out).  Before the marker: claim +
        ``claim_lease_s`` (the hard bound; a producer mid-prefill of the next queued
        request is alive and needs up to one full prefill, a dead one is caught
        here).  After the marker: marker + ``lease_duration`` for the xfer to reach
        the channel head (an expiry there releases the claim = a drain at its turn,
        never a hang).  Worst case claim -> failure = ``claim_lease_s`` +
        ``lease_duration``."""
        with self._lock:
            xf = self._xfers.get(xfer_id)
            if xf is None or xf.posted or xf.released:
                return None
            if not xf.marker_seen:
                return xf.claimed_at + float(self.cfg.claim_lease_s)
            return xf.marker_at + float(self.cfg.lease_duration)


__all__ = [
    "CLOSING_SUFFIX",
    "DEFAULT_CONTROL_DIR",
    "DEFAULT_KV_PARTS",
    "DEFAULT_MAX_MODEL_LEN",
    "RENDEZVOUS_DIR",
    "SENT_DIR",
    "SIDECAR_NAME",
    "STAGING_SPECS",
    "FabricChunk",
    "FabricConfig",
    "FabricSink",
    "FabricSocketTransport",
    "FabricSource",
    "export_pool_bytes",
    "max_export_kv_buffers",
    "spec_nbytes",
]
