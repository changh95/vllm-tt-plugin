# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""``TTKVWorker``: the worker-role engine behind ``TTKVConnector``
(PHASE2_DESIGN.md 3.3).

Two step hooks, both on the ENGINE thread (I5):

* ``begin_step(meta, finished_req_ids, join_req_ids)`` — step-BEGIN, called from
  ``_kv_connector_step_begin`` before ``_prepare_model_inputs``: producer arms
  saves; consumer releases, claims slots, polls headers and installs the GDN
  row of exactly the requests whose FIRST decode is in this step's batch
  (``join_req_ids``, I11). No K/V device writes here.
* ``end_step(finished_req_ids=None)`` — step-END, called from
  ``_kv_connector_step_end`` after the forward returned (device idle): producer
  exports; consumer K/V block imports (request-private, may land in any step),
  then ``validate_gdn_parts`` and ``KV_DONE``. Jobs whose id finished this step
  (the set ``begin_step`` saw, plus anything passed here; critic NIT-5) are
  never imported: their blocks may already belong to another request.

LoadJob states: PENDING_SLOT -> PENDING_READY -> IMPORTING_KV -> KV_DONE
(``finished_recving`` reported; job stays, holding the claimed GetHandle) ->
INSTALLED (join step). FAILED at any point before KV_DONE reports the id and
the FULL local block list in the SAME step (I8).

``ttnn`` is imported lazily (only for ``synchronize_device`` when no
``sync_device`` callable is injected).
"""

from __future__ import annotations

import enum
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from vllm.utils.math_utils import cdiv

from vllm_tt_plugin.kv_transfer.metadata import (
    RecvMeta,
    SaveMeta,
    TTKVConnectorMetadata,
    TTKVWorkerMeta,
)
from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)


class LoadState(str, enum.Enum):
    PENDING_SLOT = "PENDING_SLOT"
    PENDING_READY = "PENDING_READY"
    IMPORTING_KV = "IMPORTING_KV"
    KV_DONE = "KV_DONE"
    INSTALLED = "INSTALLED"
    FAILED = "FAILED"


@dataclass
class LoadJob:
    state: LoadState
    meta: RecvMeta
    handle: Any = None  # GetHandle once the header was READY
    chunk_cursor: int = 0
    reported: bool = False
    step_admitted: int = 0
    kv_ms: float = 0.0
    # NOTE: no slot here; the slot is re-read from the runner at the install (I7).


@dataclass
class ExportState:
    xfer_id: str
    done: bool
    ok: bool


def _is_device_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "outofmemory" in type(exc).__name__.lower()


class TTKVWorker:
    """See the module docstring. ``connector`` may be ``None`` in tests when
    ``xfer_id_fn``/roles are given explicitly."""

    def __init__(
        self,
        connector: Any,
        runner: Any,
        mesh_device: Any,
        transport: Any,
        *,
        model: Any = None,
        is_producer: bool | None = None,
        is_consumer: bool | None = None,
        block_size: int = 64,
        max_import_chunks_per_step: int = 0,
        idle_sleep_s: float = 0.0005,
        sync_device: Callable[[], None] | None = None,
        xfer_id_fn: Callable[[str], str] | None = None,
        stats: Any = None,
    ):
        self.connector = connector
        self.runner = runner
        self.mesh_device = mesh_device
        self.transport = transport
        self.model = model if model is not None else getattr(runner, "model", None)
        self.is_producer = (
            bool(getattr(connector, "is_producer", False))
            if is_producer is None
            else is_producer
        )
        self.is_consumer = (
            bool(getattr(connector, "is_consumer", False))
            if is_consumer is None
            else is_consumer
        )
        self.block_size = int(block_size)
        self.max_import_chunks_per_step = int(max_import_chunks_per_step or 0)
        self.idle_sleep_s = float(idle_sleep_s)
        self._sync_device = sync_device
        self._xfer_id_fn = xfer_id_fn or getattr(connector, "_xfer_id", None)
        if stats is None:
            from vllm_tt_plugin.kv_transfer.tt_connector import TTKVConnectorStats

            stats = TTKVConnectorStats()
        self.stats = stats

        # producer
        self._pending_saves: dict[str, SaveMeta] = {}
        self._exports: dict[str, ExportState] = {}
        self._armed: dict[str, float] = {}
        # consumer
        self._loads: dict[str, LoadJob] = {}  # insertion order == FIFO
        self._invalid_block_ids: set[int] = set()
        self._finished_now: set[str] = set()
        self._events_this_step: bool = False
        self._last_emitted_free: int | None = (
            None  # None -> first step-end always emits
        )
        # per-step bookkeeping
        self._step: int = 0
        self._step_scheduled_tokens: int | None = None
        self._step_finished: set[str] = set()
        self._step_progress: bool = False
        self._step_device_ms: float = 0.0
        self._step_touched: set[str] = set()

    # ------------------------------------------------------------------ #
    # small helpers
    # ------------------------------------------------------------------ #
    def _sync(self) -> None:
        if self._sync_device is not None:
            self._sync_device()
            return
        if self.mesh_device is None:
            return
        import ttnn

        ttnn.synchronize_device(self.mesh_device)

    def _nchunks(self, m: RecvMeta) -> int:
        nblk = cdiv(m.num_tokens, self.block_size)
        bpc = max(1, m.xfer.chunk_tokens // self.block_size)
        return max(1, cdiv(nblk, bpc))

    def _xfer_id(self, req_id: str) -> str:
        if self._xfer_id_fn is None:
            raise RuntimeError("TTKVWorker needs a connector or an xfer_id_fn")
        return self._xfer_id_fn(req_id)

    def _release_quiet(self, xfer_id: str) -> None:
        try:
            self.transport.release_remote(xfer_id)
        except Exception:
            logger.exception("PD: release_remote(%s) raised", xfer_id)

    def _others_live(self) -> bool:
        reqs = getattr(self.runner, "requests", None) or {}
        return any(r not in self._step_touched for r in reqs)

    def _note_stall(self) -> None:
        if self._step_device_ms > 0 and self._others_live():
            self.stats.record_stall(self._step_device_ms)
        self._step_device_ms = 0.0
        self._step_touched = set()

    # ------------------------------------------------------------------ #
    # step-BEGIN
    # ------------------------------------------------------------------ #
    def begin_step(
        self,
        meta: TTKVConnectorMetadata | None,
        finished_req_ids: Iterable[str],
        join_req_ids: Iterable[str],
        *,
        num_scheduled_tokens: int | None = None,
    ) -> None:
        self._step += 1
        self._step_scheduled_tokens = num_scheduled_tokens
        self._step_progress = False
        self._step_device_ms = 0.0
        self._step_touched = set()
        finished = set(finished_req_ids or ())
        self._step_finished = finished
        join = set(join_req_ids or ())
        if meta is None:
            meta = TTKVConnectorMetadata()
        if self.is_producer:
            self._begin_producer(meta, finished)
        if self.is_consumer:
            self._begin_consumer(meta, finished, join)
        elif join:
            raise RuntimeError(f"PD: join ids {sorted(join)} on a producer-only node")

    def _begin_producer(self, meta: TTKVConnectorMetadata, finished: set[str]) -> None:
        self._pending_saves.update(meta.reqs_to_save)
        for r in meta.reqs_not_processed:
            s = self._pending_saves.pop(r, None)
            self._exports.pop(r, None)
            try:
                self.transport.abandon(s.xfer_id if s is not None else self._xfer_id(r))
            except Exception:
                logger.exception("PD: abandon(%s) raised", r)
        self._armed.update(meta.reqs_to_send)  # armed exactly once per id

    def _begin_consumer(
        self, meta: TTKVConnectorMetadata, finished: set[str], join: set[str]
    ) -> None:
        for x in meta.to_release:
            self._release_quiet(x)  # header -> RELEASED; idempotent
        for r, m in meta.reqs_to_recv.items():
            if r in join:
                raise RuntimeError(f"PD: {r} is both a fresh admission and a join id")
            if r in finished:
                # aborted before we saw it: report once, never claim a slot
                self._release_quiet(m.xfer.xfer_id)
                self._finished_now.add(r)
                self._events_this_step = True
                continue
            if r in self._loads:
                logger.warning("PD: duplicate RecvMeta for %s ignored", r)
                continue
            self._loads[r] = LoadJob(
                LoadState.PENDING_SLOT, m, step_admitted=self._step
            )

        # Rows whose FIRST decode is in THIS step's batch: install the GDN row now,
        # BEFORE _prepare_model_inputs' gather (I11). NIT-7: a join id is never a
        # fresh RecvMeta or a finished id in the same output.
        for r in sorted(join):
            job = self._loads.get(r)
            if job is None:
                raise RuntimeError(f"PD: join id {r} has no load job")
            if job.state != LoadState.KV_DONE:
                raise RuntimeError(
                    f"PD: join id {r} is {job.state.value}, expected KV_DONE"
                )
            if r in finished:
                raise RuntimeError(f"PD: join id {r} is also in finished_req_ids")
            slot = self.runner.remote_slot_of(
                r
            )  # CURRENT slot, settled after the last remap
            t0 = time.perf_counter()
            try:
                self.model.install_gdn_state(job.handle.sources, slot)
            except Exception as e:
                # after finished_recving: blocks cannot be invalidated -> FATAL
                raise RuntimeError(
                    f"PD: GDN install of {r} into slot {slot} failed after "
                    "finished_recving"
                ) from e
            ms = (time.perf_counter() - t0) * 1e3
            self._step_device_ms += ms
            self._step_touched.add(r)
            self.transport.finish_import(
                job.handle, ok=True
            )  # CONSUMED; segment unlinked
            job.state = LoadState.INSTALLED
            self._events_this_step = True
            self._step_progress = True
            self.stats.record_install(r, ms, job.kv_ms + ms)
            logger.info(
                "PD: %s installed into slot %d (kv %.1f ms, install %.1f ms, %d steps)",
                r,
                slot,
                job.kv_ms,
                ms,
                self._step - job.step_admitted,
            )

        # FIFO claim / poll; no device writes.
        for r, job in list(self._loads.items()):
            if r in finished:
                # _release_dead_state_slots already popped the slot;
                # get_finished cleans the job up this step
                continue
            if job.state == LoadState.PENDING_SLOT:
                slot = self.runner.claim_remote_state_slot(r)
                if slot is None:
                    continue  # retry next step
                job.state = LoadState.PENDING_READY
                self._step_progress = True
            if job.state == LoadState.PENDING_READY:
                expiry = job.meta.xfer.expiry
                if expiry is not None and time.time() > float(expiry):
                    # The producer's janitor sweeps an unclaimed segment past
                    # its lease; claiming it now would race that sweep (audit:
                    # janitor-vs-claim). The lease was valid at the offer
                    # (``_lease_ok``), it ran out while PENDING_SLOT: recompute.
                    self._fail(r, job, "lease expired before the claim", handle=None)
                    continue
                h = self.transport.open_get(job.meta.xfer)
                if h is None:
                    continue  # still WRITING
                status = getattr(h, "status", None)
                if status != "READY":
                    self._fail(r, job, f"segment header {status}", handle=None)
                    continue
                job.handle = h
                job.state = LoadState.IMPORTING_KV
                self._step_progress = True

    # ------------------------------------------------------------------ #
    # step-END
    # ------------------------------------------------------------------ #
    def end_step(self, finished_req_ids: Iterable[str] | None = None) -> None:
        if finished_req_ids is not None:
            # NIT-5: the caller's view of this step's finished ids, on top of
            # what ``begin_step`` recorded (robust to a step-end without a
            # matching step-begin, e.g. after a failed forward).
            self._step_finished |= set(finished_req_ids)
        if self.is_producer:
            self._run_exports()
        if self.is_consumer:
            self._run_kv_imports()
        self._note_stall()

    def _run_exports(self) -> None:
        for r, s in list(self._pending_saves.items()):
            ok = False
            nbytes = 0
            t0 = time.perf_counter()
            try:
                manifest = self.model.describe_request_state(s.num_tokens, s.block_ids)
                h = self.transport.open_put(s.xfer_id, manifest)
                if h is None:
                    logger.warning(
                        "PD: open_put(%s) refused (budget/tmpfs); export of %s FAILED",
                        s.xfer_id,
                        r,
                    )
                else:
                    nbytes = int(getattr(manifest, "total_nbytes", 0) or 0)
                    slot_map = getattr(self.runner, "_req_state_slot", None) or {}
                    slot = int(slot_map.get(r, 0))
                    try:
                        self.model.export_request_state(
                            s.block_ids, s.num_tokens, slot, h.sinks
                        )
                        ok = True
                    except Exception:
                        logger.exception("PD: export_request_state(%s) failed", r)
                    self.transport.finish_export(h, "READY" if ok else "FAILED")
            except Exception:
                logger.exception("PD: export of %s failed before the data plane", r)
            ms = (time.perf_counter() - t0) * 1e3
            self._step_device_ms += ms
            self._step_touched.add(r)
            # done even on failure -> finished_sending is reported (blocks freed)
            self._exports[r] = ExportState(s.xfer_id, done=True, ok=ok)
            self.stats.record_export(r, ms, nbytes, ok)
            logger.info(
                "PD: export %s %s (%d tokens, %d blocks, %.1f ms, %.1f MiB)",
                r,
                "READY" if ok else "FAILED",
                s.num_tokens,
                len(s.block_ids),
                ms,
                nbytes / 2**20,
            )
        self._pending_saves.clear()

    def _run_kv_imports(self) -> None:
        budget: int | None = self.max_import_chunks_per_step or None
        progressed = False
        for r, job in list(self._loads.items()):
            if job.state != LoadState.IMPORTING_KV or r in self._step_finished:
                # a finished id's blocks may already belong to another request:
                # never write them; get_finished cleans the job up this step
                continue
            if budget is not None and budget <= 0:
                break
            m = job.meta
            nch = self._nchunks(m)
            stop = nch if budget is None else min(nch, job.chunk_cursor + budget)
            t0 = time.perf_counter()
            try:
                n = self.model.import_kv_blocks(
                    job.handle.sources,
                    m.local_block_ids,
                    m.num_tokens,
                    chunk_range=slice(job.chunk_cursor, stop),
                )
                n = int(n or 0)
                if n <= 0 and stop > job.chunk_cursor:
                    raise RuntimeError(
                        "import_kv_blocks made no progress "
                        f"({job.chunk_cursor}/{nch} chunks)"
                    )
                job.chunk_cursor += n
                if budget is not None:
                    budget -= n
                progressed = progressed or n > 0
                if job.chunk_cursor >= nch:
                    # everything that can fail for a reason the SEGMENT is
                    # responsible for is caught here, BEFORE finished_recving
                    self.model.validate_gdn_parts(job.handle.sources)
                    self._sync()
                    job.kv_ms += (time.perf_counter() - t0) * 1e3
                    self.runner.mark_remote_ready(r)
                    job.state = LoadState.KV_DONE
                    self._events_this_step = True
                    nbytes = int(getattr(job.handle.manifest, "total_nbytes", 0) or 0)
                    self.stats.record_import_kv(r, job.kv_ms, nbytes, nch)
                else:
                    job.kv_ms += (time.perf_counter() - t0) * 1e3
            except Exception as e:
                job.kv_ms += (time.perf_counter() - t0) * 1e3
                self._fail(r, job, repr(e), handle=job.handle)
                if _is_device_oom(e):
                    # allocator state under parked traces is undefined (R17)
                    raise
            finally:
                self._step_device_ms += (time.perf_counter() - t0) * 1e3
                self._step_touched.add(r)
        self._step_progress = self._step_progress or progressed
        waiting = any(
            j.state
            in (LoadState.PENDING_SLOT, LoadState.PENDING_READY, LoadState.IMPORTING_KV)
            for j in self._loads.values()
        )
        idle_step = (
            self._step_scheduled_tokens == 0
            if self._step_scheduled_tokens is not None
            else not (getattr(self.runner, "requests", None) or {})
        )
        if (
            self.idle_sleep_s > 0
            and waiting
            and idle_step
            and not self._step_progress
            and not self._events_this_step
        ):
            time.sleep(self.idle_sleep_s)  # R13: D idles on zero-token steps

    def _fail(self, r: str, job: LoadJob, reason: str, *, handle: Any) -> None:
        self._invalid_block_ids |= set(job.meta.local_block_ids)  # ALWAYS the full list
        try:
            self.runner.release_remote_slot(r)
        except Exception:
            logger.exception("PD: release_remote_slot(%s) raised", r)
        try:
            if handle is not None:
                self.transport.finish_import(handle, ok=False)  # LOAD_FAILED
            else:
                self.transport.release_remote(job.meta.xfer.xfer_id)
        except Exception:
            logger.exception("PD: transport cleanup for %s raised", r)
        job.handle = None
        job.state = LoadState.FAILED
        self._events_this_step = True
        self.stats.record_failed_load(r)
        logger.warning("PD: load of %s FAILED: %s", r, reason)

    # ------------------------------------------------------------------ #
    # per-step reporting
    # ------------------------------------------------------------------ #
    def get_finished(
        self, finished_req_ids: Iterable[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        finished = set(finished_req_ids or ())
        sending: set[str] | None = None
        recving: set[str] | None = None
        if self.is_producer:
            sending = self._get_finished_producer(finished)
        if self.is_consumer:
            recving = self._get_finished_consumer(finished)
        return sending or None, recving or None

    def _get_finished_producer(self, finished: set[str]) -> set[str]:
        for r in finished:
            s = self._pending_saves.pop(r, None)
            if s is not None:  # aborted mid-step
                try:
                    self.transport.abandon(s.xfer_id)
                except Exception:
                    logger.exception("PD: abandon(%s) raised", r)
        done = {r for r in self._armed if r in self._exports and self._exports[r].done}
        never = {
            r
            for r in self._armed
            if r not in self._exports and r not in self._pending_saves
        }
        for r in never:
            logger.warning(
                "PD: %s armed for finished_sending but was never exported "
                "(execute_model raised?); freeing its blocks",
                r,
            )
        done |= never
        for r in done:
            self._exports.pop(r, None)
            self._armed.pop(r, None)
        return done

    def _get_finished_consumer(self, finished: set[str]) -> set[str]:
        report = set(self._finished_now)
        for r in [r for r in finished if r in self._loads]:
            job = self._loads.pop(r)  # aborted in ANY state, incl. PENDING_SLOT (N5)
            try:
                self.runner.release_remote_slot(r)
            except Exception:
                logger.exception("PD: release_remote_slot(%s) raised", r)
            try:
                if job.handle is not None and job.state != LoadState.INSTALLED:
                    self.transport.finish_import(job.handle, ok=False)
                elif job.state != LoadState.INSTALLED:
                    self.transport.release_remote(job.meta.xfer.xfer_id)
            except Exception:
                logger.exception("PD: transport cleanup for aborted %s raised", r)
            if not job.reported:
                report.add(r)  # KV_DONE/INSTALLED were reported already
            self._events_this_step = True
            if job.state != LoadState.INSTALLED:
                logger.info(
                    "PD: load of %s aborted while %s (%d steps after admission)",
                    r,
                    job.state.value,
                    self._step - job.step_admitted,
                )
        for r, j in self._loads.items():
            if j.state in (LoadState.KV_DONE, LoadState.FAILED) and not j.reported:
                j.reported = True
                report.add(r)
        for r in [
            r
            for r, j in self._loads.items()
            if j.state in (LoadState.INSTALLED, LoadState.FAILED)
        ]:
            self._loads.pop(r)  # KV_DONE stays until the join step
        self._finished_now.clear()
        return report

    def take_invalid_block_ids(self) -> set[int]:
        s, self._invalid_block_ids = self._invalid_block_ids, set()
        return s

    def free_state_slots(self) -> int:
        held = self.runner.held_state_slots(include_loading=True)
        return int(self.runner.tt_per_lane_max_num_seqs) - len(held)

    def build_worker_meta(self) -> TTKVWorkerMeta | None:
        """``TTKVWorkerMeta`` when the free-slot count changed since the last
        emission (the first step always emits, NIT-1) or a load reached a
        terminal event this step (KV_DONE, FAILED, INSTALLED, ABORTED in any
        state, N5); else ``None`` so idle steps stay ``EMPTY_MODEL_RUNNER_OUTPUT``."""
        if not self.is_consumer:
            return None
        n = self.free_state_slots()
        if (
            self._last_emitted_free is None
            or n != self._last_emitted_free
            or self._events_this_step
        ):
            self._last_emitted_free = n
            self._events_this_step = False
            return TTKVWorkerMeta(free_state_slots=n)
        return None

    def take_stats(self):
        if self.stats.is_empty():
            return None
        return self.stats.clone_and_reset()

    # ------------------------------------------------------------------ #
    def loads(self) -> dict[str, LoadJob]:
        return self._loads

    def shutdown(self) -> None:
        try:
            self.transport.shutdown()
        except Exception:
            logger.exception("PD: transport shutdown raised")
