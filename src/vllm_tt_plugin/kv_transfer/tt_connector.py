# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""``TTKVConnector``: vLLM 0.26 ``KVConnectorBase_V1`` for TT prefill/decode
disaggregation (PHASE2_DESIGN.md sections 3.1-3.3).

One class, role-switched: the SCHEDULER role runs the state machine of 3.2
(T-1 truncation on the producer, validate/demote/defer on the consumer,
metadata built once per step, ``delay_free_blocks`` on the producer); the
WORKER role delegates to ``worker.TTKVWorker`` (3.3).

This module is imported in the API-server process (``KVConnectorFactory.
supports_hma_config``, ``KVConnectorLogging``): NO ``ttnn`` import here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.request import RequestStatus

from vllm_tt_plugin.kv_transfer.metadata import (
    ENGINE_ID_RE,
    LAYOUT_VERSION,
    REQUIRED_PARAM_KEYS,
    RecvMeta,
    SaveMeta,
    TransferDescriptor,
    TTKVConnectorMetadata,
    TTKVWorkerMeta,
)
from vllm_tt_plugin.logger import init_tt_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

    from vllm_tt_plugin.kv_transfer.worker import TTKVWorker

logger = init_tt_logger(__name__)

# Environment knobs that change the exported bytes or how the consumer reads
# them. ``(name, default)``; the default is what ``M/model_config.py`` setdefaults.
# ``QWEN36_GDN_DECODE_FUSED`` is deliberately NOT here (deviation from 3.1/R16):
# the wire state (fp32 rec row + 4 bf16 tap rows) does not depend on the decode
# mode, and milestone M2 (AM3) runs the fused-conv decode on D only. Add names
# through ``kv_connector_extra_config["fingerprint_env"]`` when needed.
FINGERPRINT_ENV_VARS: tuple[tuple[str, str], ...] = (
    ("QWEN_SDPA_BF8", "1"),
    ("QWEN35_GDN_STATE_BF16", "0"),
)


# --------------------------------------------------------------------------- #
# Pure helpers (no vllm objects beyond the config)
# --------------------------------------------------------------------------- #
def _layer_types(vllm_config: Any) -> list[str]:
    mc = getattr(vllm_config, "model_config", None)
    if mc is None:
        return []
    for cfg in (getattr(mc, "hf_text_config", None), getattr(mc, "hf_config", None)):
        if cfg is None:
            continue
        lt = getattr(cfg, "layer_types", None)
        if lt is None:
            tc = getattr(cfg, "text_config", None)
            lt = getattr(tc, "layer_types", None) if tc is not None else None
        if lt:
            return [str(t) for t in lt]
    return []


def _resolve_hybrid_state(vllm_config: Any, setting: Any = "auto") -> bool:
    """``auto`` = any non-full-attention layer in ``hf_config.layer_types``."""
    if isinstance(setting, bool):
        return setting
    s = str(setting).strip().lower()
    if s in ("auto", ""):
        return any(t != "full_attention" for t in _layer_types(vllm_config))
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"hybrid_state must be auto/true/false, got {setting!r}")


def _fingerprint(vllm_config: Any, extra_env: Iterable[str] | None = None) -> str:
    """Model / layout signature both nodes must agree on. No ttnn."""
    mc = vllm_config.model_config
    hf = getattr(mc, "hf_text_config", None) or getattr(mc, "hf_config", None)
    envs = dict(FINGERPRINT_ENV_VARS)
    for name in extra_env or ():
        envs.setdefault(str(name), "")
    payload = {
        "model": os.path.basename(str(getattr(mc, "model", "")).rstrip("/")),
        "num_layers": getattr(hf, "num_hidden_layers", None),
        "layer_types": hashlib.blake2b(
            json.dumps(_layer_types(vllm_config)).encode(), digest_size=8
        ).hexdigest(),
        "block_size": int(vllm_config.cache_config.block_size),
        "env": {k: os.environ.get(k, d) for k, d in sorted(envs.items())},
        "layout_version": LAYOUT_VERSION,
    }
    return hashlib.blake2b(
        json.dumps(payload, sort_keys=True).encode(), digest_size=16
    ).hexdigest()


def _prompt_hash(token_ids: Iterable[int]) -> str:
    arr = np.asarray(list(token_ids), dtype=np.int32)
    return hashlib.blake2b(arr.tobytes(), digest_size=8).hexdigest()


def xfer_id_for(engine_id: str, request_id: str) -> str:
    """I10: ``{engine_id}:{blake2b(request_id)}``; never contains client bytes."""
    digest = hashlib.blake2b(request_id.encode(), digest_size=16).hexdigest()
    return f"{engine_id}:{digest}"


# --------------------------------------------------------------------------- #
# Stats (vLLM 0.26 contract: KV/metrics.py)
# --------------------------------------------------------------------------- #
_STAT_LISTS = (
    "export_ms",
    "import_h2d_ms",
    "import_install_ms",
    "import_ms",
    "bytes_export",
    "bytes_import",
    "chunks",
    "steps_to_first_token",
    "stall_ms_other_users",
    "num_failed_loads",
    "num_failed_exports",
    "num_demotions",
)


@dataclass
class TTKVConnectorStats(KVConnectorStats):
    """Everything lives in ``self.data`` (the scheduler ships ``.data``):
    per-request records under ``data["records"][req_id]`` plus flat lists for
    ``reduce``/Prometheus."""

    def __post_init__(self):
        if not self.data:
            self.reset()

    def reset(self):
        self.data = {k: [] for k in _STAT_LISTS}
        self.data["records"] = {}

    def _rec(self, req_id: str) -> dict[str, Any]:
        return self.data["records"].setdefault(req_id, {})

    def record_export(self, req_id: str, ms: float, nbytes: int, ok: bool):
        self._rec(req_id).update(export_ms=ms, bytes=nbytes, export_ok=ok)
        self.data["export_ms"].append(ms)
        self.data["bytes_export"].append(nbytes)
        if not ok:
            self.data["num_failed_exports"].append(1)

    def record_import_kv(self, req_id: str, ms: float, nbytes: int, chunks: int):
        self._rec(req_id).update(import_h2d_ms=ms, bytes=nbytes, chunks=chunks)
        self.data["import_h2d_ms"].append(ms)
        self.data["bytes_import"].append(nbytes)
        self.data["chunks"].append(chunks)

    def record_install(self, req_id: str, ms: float, total_ms: float):
        self._rec(req_id).update(import_install_ms=ms, import_ms=total_ms)
        self.data["import_install_ms"].append(ms)
        self.data["import_ms"].append(total_ms)

    def record_failed_load(self, req_id: str):
        self._rec(req_id)["load_failed"] = True
        self.data["num_failed_loads"].append(1)

    def record_first_token_steps(self, req_id: str, steps: int):
        self._rec(req_id)["steps_to_first_token"] = steps
        self.data["steps_to_first_token"].append(steps)

    def record_stall(self, ms: float):
        self.data["stall_ms_other_users"].append(ms)

    def record_demotion(self):
        self.data["num_demotions"].append(1)

    def is_empty(self) -> bool:
        return not any(self.data.get(k) for k in _STAT_LISTS) and not self.data.get(
            "records"
        )

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if other.is_empty():
            return self
        for k in _STAT_LISTS:
            self.data.setdefault(k, []).extend(other.data.get(k, []))
        recs = self.data.setdefault("records", {})
        for rid, rec in other.data.get("records", {}).items():
            recs.setdefault(rid, {}).update(rec)
        return self

    def clone_and_reset(self) -> TTKVConnectorStats:
        old = copy.copy(self)
        old.data = self.data
        self.reset()
        return old

    def reduce(self) -> dict[str, int | float]:
        def _avg(k):
            v = self.data.get(k, [])
            return round(float(np.mean(v)), 3) if v else 0

        def _p90(k):
            v = self.data.get(k, [])
            return round(float(np.percentile(v, 90)), 3) if v else 0

        return {
            "Exports": len(self.data.get("export_ms", [])),
            "Avg export (ms)": _avg("export_ms"),
            "P90 export (ms)": _p90("export_ms"),
            "Imports installed": len(self.data.get("import_ms", [])),
            "Avg import K/V (ms)": _avg("import_h2d_ms"),
            "Avg GDN install (ms)": _avg("import_install_ms"),
            "Avg import total (ms)": _avg("import_ms"),
            "Avg steps to first token": _avg("steps_to_first_token"),
            "Avg stall other users (ms)": _avg("stall_ms_other_users"),
            "MB exported": round(sum(self.data.get("bytes_export", [])) / 2**20, 3),
            "MB imported": round(sum(self.data.get("bytes_import", [])) / 2**20, 3),
            "Failed loads": len(self.data.get("num_failed_loads", [])),
            "Failed exports": len(self.data.get("num_failed_exports", [])),
            "Demotions": len(self.data.get("num_demotions", [])),
        }


class TTKVConnectorPromMetrics(KVConnectorPromMetrics):
    def __init__(self, vllm_config, metric_types, labelnames, per_engine_labelvalues):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        from vllm.v1.metrics.utils import create_metric_per_engine

        secs = [
            0.005,
            0.01,
            0.025,
            0.05,
            0.075,
            0.1,
            0.15,
            0.2,
            0.3,
            0.5,
            0.75,
            1.0,
            2.0,
            5.0,
        ]
        self.hist_export = create_metric_per_engine(
            self._histogram_cls(
                name="vllm:tt_pd_export_seconds",
                documentation="Producer export wall time per request (TT PD).",
                buckets=secs,
                labelnames=labelnames,
            ),
            per_engine_labelvalues,
        )
        self.hist_import = create_metric_per_engine(
            self._histogram_cls(
                name="vllm:tt_pd_import_seconds",
                documentation="Consumer import wall time per request, "
                "K/V + GDN install (TT PD).",
                buckets=secs,
                labelnames=labelnames,
            ),
            per_engine_labelvalues,
        )
        self.hist_steps = create_metric_per_engine(
            self._histogram_cls(
                name="vllm:tt_pd_steps_to_first_token",
                documentation="Engine steps from remote admission to promotion "
                "(TT PD).",
                buckets=[1, 2, 3, 4, 6, 8, 12, 16, 32, 64],
                labelnames=labelnames,
            ),
            per_engine_labelvalues,
        )
        bytes_total = self._counter_cls(
            name="vllm:tt_pd_bytes_total",
            documentation="Bytes moved by the TT PD connector, by direction.",
            labelnames=[*labelnames, "direction"],
        )
        self.counter_bytes = {
            idx: {d: bytes_total.labels(*lv, d) for d in ("export", "import")}
            for idx, lv in per_engine_labelvalues.items()
        }

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0):
        for ms in transfer_stats_data.get("export_ms", []):
            self.hist_export[engine_idx].observe(ms / 1e3)
        for ms in transfer_stats_data.get("import_ms", []):
            self.hist_import[engine_idx].observe(ms / 1e3)
        for n in transfer_stats_data.get("steps_to_first_token", []):
            self.hist_steps[engine_idx].observe(n)
        for direction, key in (("export", "bytes_export"), ("import", "bytes_import")):
            total = sum(transfer_stats_data.get(key, []))
            if total:
                self.counter_bytes[engine_idx][direction].inc(total)


# --------------------------------------------------------------------------- #
# The connector
# --------------------------------------------------------------------------- #
class TTKVConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        cfg = vllm_config.kv_transfer_config
        self.engine_id = str(cfg.engine_id)
        if not ENGINE_ID_RE.match(self.engine_id):
            raise ValueError(
                f"kv_transfer_config.engine_id {self.engine_id!r} must match "
                f"{ENGINE_ID_RE.pattern} (it is a segment directory component)"
            )
        self.is_producer = bool(cfg.is_kv_producer)
        self.is_consumer = bool(cfg.is_kv_consumer)
        groups = kv_cache_config.kv_cache_groups
        if len(groups) != 1:
            raise ValueError(
                f"TTKVConnector expects exactly one KV cache group, got {len(groups)}"
            )
        self.block_size = int(groups[0].kv_cache_spec.block_size)
        x = dict(cfg.kv_connector_extra_config or {})
        self.transport_kind = str(x.get("transport", "shm"))
        self.shm_mode = str(x.get("shm_mode", "dumpfile"))
        self.shm_dir = str(x.get("shm_dir", "/dev/shm/tt_pd"))
        self.shm_budget_bytes = int(x.get("shm_budget_bytes", 8 << 30))
        self.kv_lease_duration = float(x.get("kv_lease_duration", 30.0))
        self.max_inflight_loads = int(x.get("max_inflight_loads", 2))
        self.xfer_chunk_tokens = int(x.get("xfer_chunk_tokens", 2048))
        self.max_import_chunks_per_step = int(x.get("max_import_chunks_per_step", 0))
        self.hybrid_state = _resolve_hybrid_state(
            vllm_config, x.get("hybrid_state", "auto")
        )
        self.fingerprint = _fingerprint(vllm_config, x.get("fingerprint_env"))
        # NO paths: the consumer never reads a path from params (I10).
        self.transport_descriptor: dict[str, Any] = {
            "kind": self.transport_kind,
            "mode": self.shm_mode,
            "layout_version": LAYOUT_VERSION,
        }
        self.max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)

        # ---- scheduler-side state (3.2) ----
        self._reqs_need_save: dict[str, Request] = {}
        self._reqs_need_recv: dict[str, RecvMeta] = {}
        self._reqs_need_send: dict[
            str, float
        ] = {}  # armed ids awaiting finished_sending
        self._reqs_to_arm: dict[
            str, float
        ] = {}  # emitted once, then in _reqs_need_send
        self._reqs_not_processed: set[str] = set()
        self._to_release: set[str] = set()
        self._inflight: set[str] = set()
        self._free_slots_estimate: int = self.max_num_seqs
        self._first_token_steps: dict[str, int] = {}
        self._sched_stats = TTKVConnectorStats()

        # ---- worker-side state (3.3) ----
        self._w: TTKVWorker | None = None
        self._transport: Any = None
        self._runner: Any = None
        self._step_finished_ids: set[str] | None = None
        self._step_join_ids: set[str] | None = None
        self._step_num_scheduled_tokens: int | None = None

        logger.info(
            "TTKVConnector(%s) engine_id=%s producer=%s consumer=%s hybrid_state=%s "
            "transport=%s fingerprint=%s",
            role.name,
            self.engine_id,
            self.is_producer,
            self.is_consumer,
            self.hybrid_state,
            self.transport_descriptor,
            self.fingerprint,
        )

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _xfer_id(self, request_id: str) -> str:
        return xfer_id_for(self.engine_id, request_id)

    def _want(self, request: Request) -> int:
        return request.num_prompt_tokens - (1 if self.hybrid_state else 0)

    # ------------------------------------------------------------------ #
    # scheduler role: trivial hooks
    # ------------------------------------------------------------------ #
    def on_new_request(self, request: Request) -> None:
        return

    def bind_gpu_block_pool(self, gpu_block_pool) -> None:
        self._block_pool = gpu_block_pool

    def has_pending_push_work(self) -> bool:
        return False

    def get_finished_count(self) -> int | None:
        return None

    def take_events(self):
        return ()

    def request_finished_all_groups(self, request, block_ids):
        return self.request_finished(request, block_ids[0])

    def set_xfer_handshake_metadata_pp_aware(self, metadata) -> None:
        md = metadata.get((0, 0)) if metadata else None
        if md is not None:
            self.transport_descriptor = dict(md) if isinstance(md, dict) else md

    def set_xfer_handshake_metadata(self, metadata) -> None:
        md = metadata.get(0) if metadata else None
        if md is not None:
            self.transport_descriptor = dict(md) if isinstance(md, dict) else md

    # ------------------------------------------------------------------ #
    # scheduler role: admission
    # ------------------------------------------------------------------ #
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        params = request.kv_transfer_params
        if not params:
            return 0, False
        if self.is_producer and params.get("do_remote_decode"):
            if request.mm_features:
                # Batched TP prefill is text-only; D would schedule encoder
                # inputs for a request that never prefills. Ordinary local run.
                params["do_remote_decode"] = False
                logger.warning(
                    "PD: %s is multimodal; serving it locally", request.request_id
                )
                return 0, False
            if self.hybrid_state:
                self._truncate_for_prefill(request)
            # Recorded HERE (survives an allocate_slots failure that re-offers
            # the request next step).
            self._reqs_need_save[request.request_id] = request
            return 0, False
        if (
            self.is_consumer
            and params.get("_tt_recv_recorded")
            and not params.get("_tt_demoted")
            and num_computed_tokens == 0
            and request.status == RequestStatus.WAITING
        ):
            # FAILED-load re-admission (kv_load_failure_policy=recompute, B2):
            # _try_promote freed the blocks, zeroed num_computed_tokens and fell
            # through into the admission body of the SAME natural pass with
            # do_remote_prefill already False. Answering (0, False) would admit
            # a T-token local prefill NEXT TO the running decodes. Demote: one
            # deferred step, then the request is plain class. A PREEMPTED
            # resume has status PREEMPTED here and is not deferred.
            return self._demote(request, "remote load failed; local recompute")
        if self.is_consumer and params.get("do_remote_prefill"):
            want = self._want(request)
            if not params.get("_tt_validated"):
                ok, why = self._params_ok(params, request, want)
                if not ok:
                    return self._demote(request, why)
                params["_tt_validated"] = True
            if not self._lease_ok(params):
                return self._demote(request, "lease expired while waiting")
            if (
                self._free_slots_estimate <= 0
                or len(self._inflight) >= self.max_inflight_loads
            ):
                return None, False  # defer; retried next step (scheduler.py:777)
            if num_computed_tokens != 0:
                raise RuntimeError(
                    f"PD: consumer offer of {request.request_id} with "
                    f"num_computed_tokens={num_computed_tokens}; prefix caching must "
                    "be off on TT (6.4)"
                )
            self._first_token_steps[request.request_id] = 0
            return want, True
        return 0, False

    def _truncate_for_prefill(self, request: Request) -> None:
        """I1: P prefills exactly ``prompt[:T-1]`` (Nixl mechanism, KV/nixl/
        base_scheduler.py:341-364). Idempotent via ``_p_side_truncated``."""
        params = request.kv_transfer_params
        if (
            params is None
            or params.get("_p_side_truncated")
            or request.num_prompt_tokens <= 1
        ):
            return
        if request.prompt_token_ids is None:
            return
        request.prompt_token_ids.pop()
        request._all_token_ids.pop()
        request.num_prompt_tokens -= 1
        request.max_tokens = 1
        params["_p_side_truncated"] = True

    def _params_ok(
        self, params: dict[str, Any], request: Request, want: int
    ) -> tuple[bool, str]:
        if request.mm_features:
            return False, "multimodal request cannot use a remote prefill"
        if want <= 0:
            return (
                False,
                f"prompt too short for remote prefill (T={request.num_prompt_tokens})",
            )
        missing = [k for k in REQUIRED_PARAM_KEYS if k not in params]
        if missing:
            return False, f"kv_transfer_params missing {missing}"
        try:
            desc = TransferDescriptor.from_params(params)
        except (KeyError, TypeError, ValueError) as e:
            return False, f"malformed kv_transfer_params: {e}"
        if desc.layout_version != LAYOUT_VERSION:
            return False, f"tt_layout_version {desc.layout_version} != {LAYOUT_VERSION}"
        if desc.num_tokens != want:
            return False, f"remote_num_tokens {desc.num_tokens} != T-1 = {want}"
        if desc.fingerprint != self.fingerprint:
            return (
                False,
                "remote_fingerprint mismatch (model/dtype/layout drift between nodes)",
            )
        if desc.transport != self.transport_descriptor:
            return False, (
                f"remote_transport {desc.transport} != local "
                f"{self.transport_descriptor}"
            )
        prompt = request.prompt_token_ids or []
        if desc.prompt_hash != _prompt_hash(prompt[:want]):
            return False, "remote_prompt_hash mismatch (tokenizer/chat-template drift?)"
        return True, ""

    @staticmethod
    def _lease_ok(params: dict[str, Any]) -> bool:
        expiry = params.get("remote_blocks_expiry_time")
        if expiry is None:
            return True
        try:
            return time.time() < float(expiry)
        except (TypeError, ValueError):
            return False

    def _demote(self, request: Request, why: str) -> tuple[None, bool]:
        params = request.kv_transfer_params
        params["do_remote_prefill"] = False
        params["_tt_demoted"] = True
        x = params.get("xfer_id")
        if x:
            self._to_release.add(
                str(x)
            )  # idempotent: a failed load already released it
        # G4: the later update_state_after_alloc(..., 0) of the local prefill
        # must not count as a promotion.
        self._first_token_steps.pop(request.request_id, None)
        self._inflight.discard(request.request_id)
        self._sched_stats.record_demotion()
        logger.warning(
            "PD: %s -> local prefill on this node: %s", request.request_id, why
        )
        return None, False  # one deferred step; next offer is plain class

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        if num_external_tokens == 0:
            # producer calls and the post-promotion consumer call
            if request.request_id in self._first_token_steps:
                self._stats_note_promoted(request.request_id)
            return
        params = request.kv_transfer_params
        if params is None:
            raise RuntimeError(
                f"PD: {request.request_id} has external tokens but no params"
            )
        if num_external_tokens != self._want(request):
            raise RuntimeError(
                f"PD: {request.request_id} external tokens {num_external_tokens} != "
                f"T-1 = {self._want(request)} (I2)"
            )
        local = list(blocks.get_block_ids()[0])
        if len(local) != cdiv(num_external_tokens, self.block_size):
            raise RuntimeError(
                f"PD: {request.request_id}: {len(local)} blocks allocated for "
                f"{num_external_tokens} external tokens (block_size {self.block_size})"
            )
        self._reqs_need_recv[request.request_id] = RecvMeta(
            local_block_ids=local,
            num_tokens=num_external_tokens,
            xfer=TransferDescriptor.from_params(params),
        )
        params["do_remote_prefill"] = False  # only one transfer per request
        params["_tt_recv_recorded"] = True  # B2: marks "a load was issued"
        self._free_slots_estimate -= 1
        self._inflight.add(request.request_id)

    def _stats_note_promoted(self, req_id: str) -> None:
        steps = self._first_token_steps.pop(req_id)
        self._sched_stats.record_first_token_steps(req_id, steps)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> TTKVConnectorMetadata:
        """Called EXACTLY ONCE per engine step
        (``TTScheduler._finalize_scheduler_output``)."""
        meta = TTKVConnectorMetadata()
        meta.reqs_to_recv, self._reqs_need_recv = self._reqs_need_recv, {}
        for new_req in scheduler_output.scheduled_new_reqs:
            req = self._reqs_need_save.pop(new_req.req_id, None)
            if req is not None:
                meta.reqs_to_save[new_req.req_id] = SaveMeta(
                    block_ids=list(new_req.block_ids[0]),
                    num_tokens=req.num_prompt_tokens,  # already T-1
                    xfer_id=self._xfer_id(new_req.req_id),
                )
        meta.reqs_to_send, self._reqs_to_arm = self._reqs_to_arm, {}  # ARM ONCE
        meta.reqs_not_processed, self._reqs_not_processed = (
            self._reqs_not_processed,
            set(),
        )
        meta.to_release, self._to_release = self._to_release, set()
        for r in self._first_token_steps:
            self._first_token_steps[r] += 1
        return meta

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        params = request.kv_transfer_params
        if not params:
            return False, None
        rid = request.request_id
        if self.is_producer and params.get("do_remote_decode"):
            if request.status not in (
                RequestStatus.FINISHED_LENGTH_CAPPED,
                RequestStatus.FINISHED_STOPPED,
            ):
                # aborted P request: nothing exported, free now
                self._reqs_not_processed.add(rid)
                self._reqs_need_save.pop(rid, None)
                return False, None
            n = request.num_computed_tokens
            if not (n == request.num_prompt_tokens == len(request.prompt_token_ids)):
                raise RuntimeError(
                    f"PD producer {rid}: computed={n} "
                    f"prompt={request.num_prompt_tokens} "
                    f"ids={len(request.prompt_token_ids)} (chunked prefill must be off)"
                )
            xfer_id = self._xfer_id(rid)
            expiry = time.time() + self.kv_lease_duration
            delay = len(block_ids) > 0
            if delay:
                self._reqs_to_arm[rid] = expiry
                self._reqs_need_send[rid] = expiry
            return delay, dict(
                do_remote_prefill=True,
                do_remote_decode=False,
                remote_engine_id=self.engine_id,
                remote_request_id=rid,
                remote_block_ids=list(block_ids),
                remote_num_tokens=n,
                tp_size=1,
                remote_host=None,
                remote_port=None,
                remote_prompt_hash=_prompt_hash(request.prompt_token_ids[:n]),
                remote_fingerprint=self.fingerprint,
                remote_blocks_expiry_time=expiry,
                remote_transport=dict(self.transport_descriptor),  # no paths
                xfer_id=xfer_id,
                tt_layout_version=LAYOUT_VERSION,
                tt_chunk_tokens=self.xfer_chunk_tokens,
            )
        if self.is_consumer and params.get("xfer_id") is not None:
            if params.get("do_remote_prefill") and not params.get("_tt_demoted"):
                # aborted before the load was recorded, or the serving-layer
                # rejection path (notify_kv_transfer_request_rejected)
                self._to_release.add(str(params["xfer_id"]))
            self._inflight.discard(rid)
            self._first_token_steps.pop(rid, None)
            return False, None
        # incl. kv_both (R1): a P-leg multimodal request whose do_remote_decode
        # was cleared has xfer_id None -> here
        return False, None

    def update_connector_output(self, kv_connector_output: KVConnectorOutput):
        wm = kv_connector_output.kv_connector_worker_meta
        if wm is not None and isinstance(wm, TTKVWorkerMeta):
            self._free_slots_estimate = int(wm.free_state_slots)
        for r in kv_connector_output.finished_recving or ():
            self._inflight.discard(r)
        for r in kv_connector_output.finished_sending or ():
            self._reqs_need_send.pop(r, None)
        if self._reqs_need_send:
            now = time.time()
            stale = [
                r
                for r, exp in self._reqs_need_send.items()
                if now - (exp - self.kv_lease_duration) > 2 * self.kv_lease_duration
            ]
            for r in stale:
                logger.warning(
                    "PD: %s armed for finished_sending for > 2 leases (leak?)", r
                )
                self._reqs_need_send.pop(r, None)

    # ------------------------------------------------------------------ #
    # stats (both roles)
    # ------------------------------------------------------------------ #
    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats:
        return (
            TTKVConnectorStats(data=data) if data is not None else TTKVConnectorStats()
        )

    @classmethod
    def build_prom_metrics(
        cls, vllm_config, metric_types, labelnames, per_engine_labelvalues
    ):
        return TTKVConnectorPromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        if self._w is not None:
            return self._w.take_stats()
        if self._sched_stats.is_empty():
            return None
        return self._sched_stats.clone_and_reset()

    # ------------------------------------------------------------------ #
    # worker role (3.3): delegates to TTKVWorker
    # ------------------------------------------------------------------ #
    def attach_runner(
        self, runner: Any, *, transport: Any = None, model: Any = None
    ) -> None:
        """Create the transport and the ``TTKVWorker``. ``transport``/``model``
        override the defaults (tests, R1 loopback sharing one transport)."""
        from vllm_tt_plugin.kv_transfer.hooks import (
            DefaultKVTransferable,
            implements_kv_transfer,
        )
        from vllm_tt_plugin.kv_transfer.worker import TTKVWorker

        if transport is None:
            from vllm_tt_plugin.kv_transfer.transport import make_transport

            transport = make_transport(
                self.transport_kind,
                self.shm_mode,
                self._kv_transfer_config,
                self._kv_transfer_config.kv_role,
            )
        transport.start()
        if model is None:
            model = getattr(runner, "model", None)
            if not implements_kv_transfer(model):
                logger.warning(
                    "PD: %s does not implement TTKVTransferable; using the "
                    "paged-KV-only DefaultKVTransferable",
                    type(model).__name__,
                )
                model = DefaultKVTransferable(
                    runner.kv_caches,
                    mesh_device=getattr(runner, "mesh_device", None),
                    block_size=self.block_size,
                    chunk_tokens=self.xfer_chunk_tokens,
                    model_sig=self.fingerprint,
                )
        if self.is_consumer:
            self._check_fused_conv(model)
        self._runner = runner
        self._transport = transport
        self._w = TTKVWorker(
            self,
            runner,
            getattr(runner, "mesh_device", None),
            transport,
            model=model,
            is_producer=self.is_producer,
            is_consumer=self.is_consumer,
            block_size=self.block_size,
            max_import_chunks_per_step=self.max_import_chunks_per_step,
        )

    @staticmethod
    def _gdn_layers(model: Any) -> list[Any]:
        fn = getattr(model, "kv_transfer_gdn_layers", None)
        if callable(fn):
            try:
                return list(fn())
            except Exception:  # pragma: no cover - defensive
                return []
        out: list[Any] = []
        try:
            inner = getattr(model, "model", None) or model
            for layer in getattr(inner, "layers", None) or []:
                dn = getattr(layer, "attention", None)
                if dn is not None and hasattr(dn, "_decode_fused_conv"):
                    out.append(dn)
        except Exception:  # pragma: no cover - defensive
            return []
        return out

    def _check_fused_conv(self, model: Any) -> None:
        """R4 / AM3: the consumer refuses a fused-conv GDN decode unless
        ``TT_PD_ALLOW_FUSED_CONV=1`` (milestone M2)."""
        fused = [
            dn
            for dn in self._gdn_layers(model)
            if getattr(dn, "_decode_fused_conv", False)
        ]
        if fused and os.environ.get("TT_PD_ALLOW_FUSED_CONV", "0") != "1":
            raise RuntimeError(
                f"PD consumer: {len(fused)} GDN layers run the fused-conv decode "
                "(conv_hist_packed parity remap is not PD-safe, R4). Run the decode "
                "node with QWEN36_GDN_DECODE_FUSED=0 (M1) or set "
                "TT_PD_ALLOW_FUSED_CONV=1 (M2)."
            )

    def set_step_context(
        self,
        finished_req_ids: Iterable[str] | None,
        join_req_ids: Iterable[str] | None,
        num_scheduled_tokens: int | None = None,
    ) -> None:
        """Runner -> connector, once per step BEFORE ``start_load_kv`` (6.2):
        ``finished_req_ids = scheduler_output.finished_req_ids`` and
        ``join_req_ids = {r.req_id for r in scheduled_new_reqs} &
        runner._remote_ready``."""
        self._step_finished_ids = set(finished_req_ids or ())
        self._step_join_ids = set(join_req_ids or ())
        self._step_num_scheduled_tokens = num_scheduled_tokens

    def register_kv_caches(self, kv_caches) -> None:
        return  # TT hands the model object instead (attach_runner)

    def start_load_kv(self, forward_context=None, **kwargs) -> None:
        """Step-BEGIN hook: slot claims, header polls, GDN install of the rows
        joining THIS step's batch. No K/V device writes (I11)."""
        if self._w is None:
            raise RuntimeError("PD: attach_runner() must run before the first step")
        finished = kwargs.get("finished_req_ids")
        join = kwargs.get("join_req_ids")
        nsched = kwargs.get("num_scheduled_tokens")
        if finished is None:
            finished = self._step_finished_ids
            if finished is None:
                finished = getattr(self._runner, "_step_finished_ids", None)
        if join is None:
            join = self._step_join_ids
            if join is None:
                join = getattr(self._runner, "_step_join_ids", None)
        if nsched is None:
            nsched = self._step_num_scheduled_tokens
        self._step_finished_ids = self._step_join_ids = None
        self._step_num_scheduled_tokens = None
        self._w.begin_step(
            self._get_connector_metadata(),
            set(finished or ()),
            set(join or ()),
            num_scheduled_tokens=nsched,
        )

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        return

    def wait_for_save(self) -> None:
        """Step-END hook for BOTH roles: producer exports, consumer K/V imports."""
        if self._w is None:
            raise RuntimeError("PD: attach_runner() must run before the first step")
        self._w.end_step()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        if self._w is None:
            raise RuntimeError("PD: attach_runner() must run before the first step")
        return self._w.get_finished(finished_req_ids)

    def get_block_ids_with_load_errors(self) -> set[int]:
        if self._w is None:
            return set()
        return self._w.take_invalid_block_ids()

    def build_connector_worker_meta(self) -> TTKVWorkerMeta | None:
        if self._w is None:
            return None
        return self._w.build_worker_meta()

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata | None:
        return None  # v1: descriptor comes from config; v2 returns the fabric endpoint

    def handle_preemptions(self, kv_connector_metadata) -> None:
        return

    def shutdown(self) -> None:
        if self._w is not None:
            self._w.shutdown()
            self._w = None
