# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Pure-Python assembly of the Phase 3 PD pair launch inputs (no vllm, no ttnn; the
only plugin import is the transport's pool-sizing arithmetic).

One ``PairSettings`` (read from the same shell knobs ``run_pd_pair.sh`` uses) yields:

* ``serve_argv(settings, role)``: the ``vllm serve`` argument list of one node.  The
  API front-end is started with exactly this list plus nothing else, and the engine
  rank replays the SAME list through vLLM's serve parser, so both build the same
  ``VllmConfig`` (the engine re-runs ``VllmConfig.__post_init__`` after the handshake).
* ``node_args(settings)`` / ``write_node_args``: ``pd_node_args.json`` =
  ``{"prefill": argv, "decode": argv, "meta": {...}}`` read by ``pd_fabric_rank``.
* ``render_rank_binding(...)``: the tt-run rank-binding YAML (schema: ttrun.py
  ``TTRunConfig``) from a checked-in template (``pd_rank_binding_chips03.yaml`` /
  ``pd_rank_binding_chips12.yaml``) plus ``global_env`` (every VLLM_/HF_/QWEN36_/
  TT_PD_/... variable the engine needs: tt-run forwards only TT_/ARCH_/WH_/TTNN_/
  DEEPSEEK_/MESH_ prefixes) and per-rank ``env_overrides`` (chip, kernel cache, role,
  RPC port, per-node pool size).
* ``validate_argv_for_role(argv, role, rpc_port)``: the consistency checks the rank
  entry point runs before it touches vllm.

Role table (fixed by the MeshSocket config ``sender_rank=0, receiver_rank=1``):
rank 0 = prefill node = ``kv_producer`` on mesh 0, rank 1 = decode node =
``kv_consumer`` on mesh 1.

Per-pair isolation on one host (two pairs must never see each other's segments):
every path and id a pair owns is derived from ``TAG`` unless set explicitly --
``shm_dir=/dev/shm/tt_pd_{tag}`` (TRANSPORT=shm segments), ``ctrl_dir=
/dev/shm/tt_pd_fabric_{tag}`` (fabric control segments and, under it,
``.fabric_rendezvous``), engine ids ``p-{tag}`` / ``d-{tag}``.  A producer
``ShmTransport.start()`` runs ``startup_sweep`` over ``{segment_dir}/{engine_id}``
(unlinks every segment that is not a live consumer's claim), so two live engines
with the same ``(segment_dir, engine_id)`` destroy each other's READY segments.
``EngineLock`` (``{segment_dir}/.pd_engine-{engine_id}.lock``, a pid file) makes the
rank refuse to start over a live holder, and ``engine_in_use`` also recognises a
lock-less peer (``run_pd_pair.sh``) through the live ``producer_pid`` of its
segment headers / the live consumer pid of its claims.

CLI (used by ``profiles/pd/run_pd_pair_fabric.sh``)::

    python -m vllm_tt_plugin.kv_transfer.launch.pd_launch_config \
        --pair prod --out-dir /home/ttuser/experiments/qwen36_27b/profiles/pd --tag p3 \
        [--template <yaml>] [--print-argv0 prefill|decode] [--print-settings] \
        [--check-in-use]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from ..transport.fabric import (
    RENDEZVOUS_DIR,
    export_pool_bytes,
    max_export_kv_buffers,
)
from ..transport.shm import (
    CONSUMER_PID_FILE,
    parse_xfer_id,
    pid_alive,
    pid_wrote_at,
    read_header,
)

# --------------------------------------------------------------------------- #
# role table
# --------------------------------------------------------------------------- #
ROLES: tuple[str, str] = ("prefill", "decode")
RANK_OF_ROLE: dict[str, int] = {"prefill": 0, "decode": 1}
ROLE_OF_RANK: dict[int, str] = {0: "prefill", 1: "decode"}
KV_ROLE_OF_ROLE: dict[str, str] = {"prefill": "kv_producer", "decode": "kv_consumer"}
ROLE_OF_KV_ROLE: dict[str, str] = {v: k for k, v in KV_ROLE_OF_ROLE.items()}
WORLD_SIZE = 2

# chip pairs on the P300x2 box: inter-card Warp400 cable pairs only (PHASE3_NOTES 4);
# ``p150x2`` = two p150a cards linked by ethernet (the container product's board), one
# chip per card, so its (0, 1) is a cable pair, not a P300 on-package pair.
PAIRS: dict[str, tuple[int, int]] = {
    "prod": (0, 3),
    "test": (1, 2),
    "chips03": (0, 3),
    "chips12": (1, 2),
    "p150x2": (0, 1),
}
BOARDS: tuple[str, ...] = ("p300x2", "p150x2")
BOARD_OF_PAIR: dict[str, str] = {p: "p300x2" for p in PAIRS}
BOARD_OF_PAIR["p150x2"] = "p150x2"
TEMPLATE_OF_PAIR: dict[str, str] = {
    "prod": "pd_rank_binding_chips03.yaml",
    "chips03": "pd_rank_binding_chips03.yaml",
    "test": "pd_rank_binding_chips12.yaml",
    "chips12": "pd_rank_binding_chips12.yaml",
}

CONNECTOR_MODULE = "vllm_tt_plugin.kv_transfer.tt_connector"
CONNECTOR_CLASS = "TTKVConnector"
RANK_MODULE = "vllm_tt_plugin.kv_transfer.launch.pd_fabric_rank"

# env passed to the ranks through global_env (tt-run forwards nothing else).
# exact names first, then prefixes; values come from the parent environment.
GLOBAL_ENV_NAMES: tuple[str, ...] = (
    "ARCH_NAME",
    "MESH_DEVICE",
    "HF_MODEL",
    "HF_HUB_OFFLINE",
    "TT_QWEN35_TEXT_VER",
    "TT_CACHE_PATH",
    "VLLM_TARGET_DEVICE",
    "VLLM_RPC_TIMEOUT",
    "VLLM_CONFIGURE_LOGGING",
    "VLLM_LOGGING_LEVEL",
    "TORCHDYNAMO_DISABLE",
    "TT_PD_STRICT_SHAPES",
    "TT_PD_TIMING",
    "TT_PD_CHECKSUM",
    "TT_PD_FABRIC_DIR",
    "TRANSFORMERS_OFFLINE",
    "TOKENIZERS_PARALLELISM",
    "TT_HOST_SAMPLER_FAST",  # plugin host sampler knob (host_sampler.py; 0 = upstream)
)
GLOBAL_ENV_PREFIXES: tuple[str, ...] = (
    "VLLM_",
    "HF_",
    "QWEN36_",
    "QWEN35_",
    "QWEN_",
    "TT_PD_",
    "TRANSFORMERS_",
)
# never in global_env: tt-run manages them per rank or they must differ per rank.
GLOBAL_ENV_BLOCKLIST: frozenset[str] = frozenset(
    {
        "TT_MESH_ID",
        "TT_MESH_HOST_RANK",
        "TT_MESH_GRAPH_DESC_PATH",
        "TT_VISIBLE_DEVICES",
        "TT_METAL_CACHE",
        "TT_RUN_RANK",
        "TT_RUN_ORIGINAL_CWD",
        "QWEN36_MAX_TOKENS_ALL_USERS",  # per node (pool), env_overrides
        "QWEN36_GDN_DECODE_FUSED",  # per node, env_overrides
        "TT_PD_ALLOW_FUSED_CONV",  # decode only, env_overrides
        "VLLM_ENGINE_READY_TIMEOUT_S",  # front-end only
        "PD_ROLE",
        "PD_RPC_PORT",
    }
)
# mandatory in the rendered global_env (the rank cannot come up without them).
GLOBAL_ENV_REQUIRED: tuple[str, ...] = (
    "MESH_DEVICE",
    "VLLM_TARGET_DEVICE",
    "HF_HUB_OFFLINE",
    "TT_CACHE_PATH",
    "QWEN36_FORCE_TP_PATH",
    "PD_ARGS_JSON",
    "TT_METAL_OPERATION_TIMEOUT_SECONDS",
)


def _env_int(env: dict[str, str], key: str, default: int) -> int:
    return int(env.get(key, default))


def _env_float(env: dict[str, str], key: str, default: float) -> float:
    return float(env.get(key, default))


# The model's traced MASKED prefill buckets (tt-metal models/demos/blackhole/qwen36/
# tt/model.py ``Qwen36Model._PREFILL_MASK_BUCKETS``; the model is not importable on a
# host without ttnn, so the list is mirrored here).  ``QWEN36_PREFILL_BUCKET_TRACE``
# selects which are traced, parsed like ``masked_bucket_trace.parse_bucket_trace_gate``.
PREFILL_MASK_BUCKETS = (128, 256, 512, 1024, 2048)


def largest_traced_prefill_bucket(gate: str | None = "1") -> int:
    """Tokens the largest traced prefill bucket fills at warm-up (the KV-pool
    guard's quantity).  ``gate`` = ``QWEN36_PREFILL_BUCKET_TRACE``, spelled exactly
    as ``masked_bucket_trace.parse_bucket_trace_gate`` accepts it: "1" / "all" /
    "true" (every bucket) and unset / "" / "0" / "off" / "false" (the model traces
    nothing; the plugin's ``platform.py`` refuses KV transfer without the trace, so
    the guard stays conservative and assumes every bucket) -> the largest bucket; a
    comma list -> the largest listed (a value not in the model's table raises, as
    the model's parser does)."""
    if gate is None or gate.strip() in ("", "0", "off", "false", "1", "all", "true"):
        return max(PREFILL_MASK_BUCKETS)
    picked: list[int] = []
    for part in gate.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            b = int(part)
        except ValueError as e:
            raise ValueError(
                f"QWEN36_PREFILL_BUCKET_TRACE={gate!r}: {part!r} is not a bucket"
            ) from e
        if b not in PREFILL_MASK_BUCKETS:
            raise ValueError(
                f"QWEN36_PREFILL_BUCKET_TRACE={gate!r}: bucket {b} is not one of "
                f"{PREFILL_MASK_BUCKETS}"
            )
        picked.append(b)
    if not picked:
        return max(PREFILL_MASK_BUCKETS)
    return max(picked)


# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #
@dataclass
class PairSettings:
    """Every knob of ``run_pd_pair.sh`` plus the Phase 3 ones (same defaults)."""

    root: str = "/home/ttuser/experiments/qwen36_27b"
    weights: str = ""  # default: {root}/weights_qwen38
    served_model_name: str = "Qwen/Qwen3.8-27B"
    pair: str = "prod"
    # board the chip indices refer to: "p300x2" (dev box: (0,1)/(2,3) are on-package
    # TRACE pairs and refused) or "p150x2" (one chip per card, any two chips are a
    # cable pair); "" = the pair's board (BOARD_OF_PAIR)
    board: str = ""
    p_chip: int = -1  # default from pair
    d_chip: int = -1
    p_port: int = 8100
    d_port: int = 8200
    proxy_port: int = 8000
    p_rpc: int = 29551
    d_rpc: int = 29552
    dp_address: str = "127.0.0.1"
    ctx: int = 65536
    p_pool: int = 65536
    d_pool: int = 163840
    d_conc: int = 8
    d_fail_policy: str = "recompute"
    shm_mode: str = (
        "dumpfile"  # hook warm-up contract; the connector descriptor carries it
    )
    lease: float = 30.0
    # QWEN36_GDN_DECODE_FUSED per rank. The prefill rank never runs a decode step (T-1
    # protocol) and keeps the model default (0 = composite); the DECODE rank runs the
    # fused GDN decode + fused conv (2, Phase 2's "M2") with the connector's fused-conv
    # gate opened -- the validated p1d1 default (profiles/p1d1_opt/laneE_RESULTS.md:
    # TPOT 1.3-1.5x vs composite, identity = the fused single-die class, parity remap
    # soak clean). -1 = follow gdn_fused (the pre-lane-E behaviour).
    gdn_fused: int = 0
    d_gdn_fused: int = 2
    allow_fused_conv: int = 1
    strict: int = 1
    min_remote: int = 2
    trace_region: int = 536870912
    l1_small: int = 24576
    # fabric control segments + .fabric_rendezvous; "" = /dev/shm/tt_pd_fabric_{tag}
    ctrl_dir: str = ""
    max_inflight: int = 2
    chunk_tokens: int = 2048
    chunks_per_step: int = 0
    hybrid_state: str = "auto"
    # data plane: "fabric" (Phase 3 MeshSocket) or "shm" (the Phase 2 ShmTransport
    # inside the SAME F0 process shape; p3 device validation fallback when the
    # inter-card link does not pass payloads).  With "shm" the ranks open their mesh
    # WITHOUT a fabric config (like run_pd_pair.sh) unless shm_fabric=1, the
    # like-for-like comparison mode: FABRIC_2D set at mesh open exactly as the fabric
    # pair does, so the only variable versus it is the data plane.
    transport: str = "fabric"
    shm_fabric: int = 0
    shm_dir: str = ""  # transport=shm: segments; "" = /dev/shm/tt_pd_{tag}
    shm_budget: int = 8589934592  # transport=shm: shm_budget_bytes (8 GiB)
    # Phase 3 fabric knobs
    pkt: int = (
        8704  # measured cross-process 58.8 GB/s; 15232 = BH max, cross-process untested
    )
    conns: int = 2
    fifo_bytes: int = 128
    rec_sets: int = 4
    # producer export pool bytes; 0 = derived: export_slots x one max-length export of
    # ctx tokens (1024 head-major K/V buffers x 2,228,224 B = 2.125 GiB at 65536).
    # The transport refuses (at start) a budget below one such export.
    export_budget: int = 0
    export_slots: int = 1  # 2 pipelines two full-length exports (4.25 GiB of P DRAM)
    reliability: str = "STRICT_INIT"
    fabric_config: str = "FABRIC_2D"
    p_engine_id: str = ""  # "" = p-{tag}
    d_engine_id: str = ""  # "" = d-{tag}
    seed: int = 9472
    metal_cache_root: str = ""  # default: {root}/tt_cache_pd
    tag: str = "p3"
    p_extra_args: list[str] = field(default_factory=list)
    d_extra_args: list[str] = field(default_factory=list)
    # QWEN36_PREFILL_BUCKET_TRACE as the model parses it ("1" = every bucket; a
    # comma list = those); sizes the KV-pool guard below
    bucket_trace_gate: str = "1"
    # serve-argv knobs the container supervisor maps from the tool's ``vllm serve``
    # line (run_pd_pair_fabric.sh fixes them at these values)
    block_size: int = 64
    max_num_batched_tokens: int = 0  # 0 = ctx
    tool_call_parser: str = "qwen3_coder"
    reasoning_parser: str = "qwen3"
    enable_auto_tool_choice: int = 1
    # per-rank converted-weights cache: "" = one shared TT_CACHE_PATH from the parent
    # env (global_env); a root -> rank env_overrides TT_CACHE_PATH={root}/tp1-rank{N}
    # (two ranks converting into one dir on a cold boot is an unvalidated race)
    tensor_cache_root: str = ""
    # explicit two-mesh MGD for the rank binding; "" = {root}/profiles/pd/... (the
    # template's own path wins over both)
    mgd_path: str = ""

    def __post_init__(self) -> None:
        if not self.tag:
            raise ValueError("TAG must not be empty (it scopes every per-pair path)")
        # tag-scoped per-pair defaults (module docstring: two pairs on one host)
        if not self.shm_dir:
            self.shm_dir = f"/dev/shm/tt_pd_{self.tag}"
        if not self.ctrl_dir:
            self.ctrl_dir = f"/dev/shm/tt_pd_fabric_{self.tag}"
        if not self.p_engine_id:
            self.p_engine_id = f"p-{self.tag}"
        if not self.d_engine_id:
            self.d_engine_id = f"d-{self.tag}"
        for eid in (self.p_engine_id, self.d_engine_id):
            _check_engine_id(eid)
        if self.p_engine_id == self.d_engine_id:
            raise ValueError("P_ENGINE_ID and D_ENGINE_ID must differ")
        if self.pair not in PAIRS:
            raise ValueError(f"unknown PAIR {self.pair!r}; one of {sorted(PAIRS)}")
        if not self.board:
            self.board = BOARD_OF_PAIR[self.pair]
        if self.board not in BOARDS:
            raise ValueError(f"unknown BOARD {self.board!r}; one of {BOARDS}")
        pc, dc = PAIRS[self.pair]
        if self.p_chip < 0:
            self.p_chip = pc
        if self.d_chip < 0:
            self.d_chip = dc
        if self.p_chip == self.d_chip:
            raise ValueError("prefill and decode chips must differ")
        if self.board == "p300x2" and {self.p_chip, self.d_chip} in ({0, 1}, {2, 3}):
            raise ValueError(
                f"chips {self.p_chip},{self.d_chip} are an on-package TRACE pair on a "
                "p300x2, not a cable pair; use (0,3) or (1,2) (PHASE3_NOTES 4)"
            )
        if self.p_chip < 0 or self.d_chip < 0:
            raise ValueError("chip indices must be >= 0")
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.max_num_batched_tokens <= 0:
            self.max_num_batched_tokens = self.ctx
        if self.max_num_batched_tokens < self.ctx:
            raise ValueError(
                f"max_num_batched_tokens {self.max_num_batched_tokens} < ctx "
                f"{self.ctx}: "
                "the nodes prefill whole prompts (no chunked prefill)"
            )
        if not self.weights:
            self.weights = os.path.join(self.root, "weights_qwen38")
        if not self.metal_cache_root:
            self.metal_cache_root = os.path.join(self.root, "tt_cache_pd")
        if self.d_gdn_fused < 0:
            self.d_gdn_fused = self.gdn_fused
        if self.p_rpc == self.d_rpc:
            raise ValueError("P_RPC and D_RPC must differ (one handshake ROUTER each)")
        if len({self.p_port, self.d_port, self.proxy_port}) != 3:
            raise ValueError("P_PORT, D_PORT and PROXY_PORT must differ")
        if self.conns not in (1, 2):
            raise ValueError("CONNS must be 1 or 2 (fabric links between the chips)")
        if self.transport not in ("fabric", "shm"):
            raise ValueError(f"TRANSPORT must be fabric or shm, got {self.transport!r}")
        if self.shm_fabric not in (0, 1):
            raise ValueError(f"SHM_FABRIC must be 0 or 1, got {self.shm_fabric!r}")
        if self.shm_mode != "dumpfile":
            raise ValueError(
                "shm_mode must stay 'dumpfile' (model hook warm-up contract; the "
                "fabric transport ignores it, the shm transport's raw mode is blocked)"
            )
        if self.export_slots < 1:
            raise ValueError("EXPORT_SLOTS must be >= 1")
        # The model's prefill bucket traces (QWEN36_PREFILL_BUCKET_TRACE, which the
        # plugin requires ON for KV transfer) fill the WHOLE largest traced bucket
        # (2048 tokens = 32 KV blocks) through a fixed-width page table at warm-up,
        # independent of the transfer CHUNK_TOKENS; a KV pool with fewer blocks makes
        # that fill write past the pool in DRAM and the rank hangs in the next decode
        # warm-up (device stopped consuming commands; seen twice on the 2026-09-21
        # tiny pairs: P_POOL=1024 -> "KV cache 18 blocks" vs a 32-block bucket,
        # py-spy in SystemMemoryManager::fetch_queue_reserve_back).  One spare block
        # (the pad block the scheduler never hands out) on top.
        bucket = largest_traced_prefill_bucket(self.bucket_trace_gate)
        min_pool = bucket + 64
        for name, pool in (("P_POOL", self.p_pool), ("D_POOL", self.d_pool)):
            if pool < min_pool:
                raise ValueError(
                    f"{name}={pool} tokens is below the {bucket}-token prefill bucket "
                    f"trace + one pad block (>= {min_pool}); the warm-up fill would "
                    "write past the KV pool and hang the rank"
                )
        if self.export_budget <= 0:
            self.export_budget = export_pool_bytes(self.ctx, slots=self.export_slots)
        elif self.export_budget < export_pool_bytes(self.ctx):
            raise ValueError(
                f"EXPORT_BUDGET {self.export_budget} cannot hold one export of a "
                f"CTX={self.ctx} request ({max_export_kv_buffers(self.ctx)} K/V "
                f"buffers = {export_pool_bytes(self.ctx)} B); use 0 to derive it"
            )

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> PairSettings:
        """Read the ``run_pd_pair.sh`` knob names (P_CHIP, D_POOL, ...) from ``env``."""
        e = dict(os.environ if env is None else env)
        root = e.get("R", cls.root)
        s = cls(
            root=root,
            weights=e.get("WEIGHTS", ""),
            served_model_name=e.get("SERVED_MODEL_NAME", cls.served_model_name),
            pair=e.get("PAIR", cls.pair),
            board=e.get("BOARD", cls.board),
            p_chip=_env_int(e, "P_CHIP", -1),
            d_chip=_env_int(e, "D_CHIP", -1),
            p_port=_env_int(e, "P_PORT", cls.p_port),
            d_port=_env_int(e, "D_PORT", cls.d_port),
            proxy_port=_env_int(e, "PROXY_PORT", cls.proxy_port),
            p_rpc=_env_int(e, "P_RPC", cls.p_rpc),
            d_rpc=_env_int(e, "D_RPC", cls.d_rpc),
            dp_address=e.get("DP_ADDRESS", cls.dp_address),
            ctx=_env_int(e, "CTX", cls.ctx),
            p_pool=_env_int(e, "P_POOL", cls.p_pool),
            d_pool=_env_int(e, "D_POOL", cls.d_pool),
            d_conc=_env_int(e, "D_CONC", cls.d_conc),
            d_fail_policy=e.get("D_FAIL_POLICY", cls.d_fail_policy),
            shm_mode=e.get("SHM_MODE", cls.shm_mode),
            lease=_env_float(e, "LEASE", cls.lease),
            gdn_fused=_env_int(e, "GDN_FUSED", cls.gdn_fused),
            d_gdn_fused=_env_int(e, "D_GDN_FUSED", cls.d_gdn_fused),
            allow_fused_conv=_env_int(e, "ALLOW_FUSED_CONV", cls.allow_fused_conv),
            strict=_env_int(e, "STRICT", cls.strict),
            min_remote=_env_int(e, "MIN_REMOTE", cls.min_remote),
            trace_region=_env_int(e, "TRACE_REGION", cls.trace_region),
            l1_small=_env_int(e, "L1_SMALL", cls.l1_small),
            ctrl_dir=e.get("CTRL_DIR", cls.ctrl_dir),
            max_inflight=_env_int(e, "MAX_INFLIGHT", cls.max_inflight),
            chunk_tokens=_env_int(e, "CHUNK_TOKENS", cls.chunk_tokens),
            bucket_trace_gate=e.get(
                "QWEN36_PREFILL_BUCKET_TRACE", cls.bucket_trace_gate
            ),
            chunks_per_step=_env_int(e, "CHUNKS_PER_STEP", cls.chunks_per_step),
            hybrid_state=e.get("HYBRID_STATE", cls.hybrid_state),
            transport=e.get("TRANSPORT", cls.transport),
            shm_fabric=_env_int(e, "SHM_FABRIC", cls.shm_fabric),
            shm_dir=e.get("SHM_DIR", cls.shm_dir),
            shm_budget=_env_int(e, "SHM_BUDGET", cls.shm_budget),
            pkt=_env_int(e, "PKT", cls.pkt),
            conns=_env_int(e, "CONNS", cls.conns),
            fifo_bytes=_env_int(e, "FIFO_BYTES", cls.fifo_bytes),
            rec_sets=_env_int(e, "REC_SETS", cls.rec_sets),
            export_budget=_env_int(e, "EXPORT_BUDGET", cls.export_budget),
            export_slots=_env_int(e, "EXPORT_SLOTS", cls.export_slots),
            reliability=e.get("RELIABILITY", cls.reliability),
            fabric_config=e.get("FABRIC_CONFIG", cls.fabric_config),
            p_engine_id=e.get("P_ENGINE_ID", cls.p_engine_id),
            d_engine_id=e.get("D_ENGINE_ID", cls.d_engine_id),
            seed=_env_int(e, "SEED", cls.seed),
            metal_cache_root=e.get("METAL_CACHE_ROOT", ""),
            tag=e.get("TAG", cls.tag),
            p_extra_args=shlex.split(e.get("P_EXTRA_ARGS", "")),
            d_extra_args=shlex.split(e.get("D_EXTRA_ARGS", "")),
        )
        return s

    # --- per-role views ------------------------------------------------------- #
    def chip(self, role: str) -> int:
        return self.p_chip if role == "prefill" else self.d_chip

    def port(self, role: str) -> int:
        return self.p_port if role == "prefill" else self.d_port

    def rpc_port(self, role: str) -> int:
        return self.p_rpc if role == "prefill" else self.d_rpc

    def engine_id(self, role: str) -> str:
        return self.p_engine_id if role == "prefill" else self.d_engine_id

    def pool(self, role: str) -> int:
        return self.p_pool if role == "prefill" else self.d_pool

    def max_num_seqs(self, role: str) -> int:
        return 1 if role == "prefill" else self.d_conc

    def metal_cache(self, role: str) -> str:
        return os.path.join(self.metal_cache_root, f"metal_rank{RANK_OF_ROLE[role]}")

    def tensor_cache(self, role: str) -> str | None:
        """Per-rank ``TT_CACHE_PATH`` (``None`` = shared, from the parent env)."""
        if not self.tensor_cache_root:
            return None
        return os.path.join(self.tensor_cache_root, f"tp1-rank{RANK_OF_ROLE[role]}")

    @property
    def fabric_enabled(self) -> bool:
        """Whether the ranks set the fabric config at mesh open: always for the
        fabric data plane, for the shm data plane only in the like-for-like mode."""
        return self.transport == "fabric" or self.shm_fabric == 1

    @property
    def segment_dir(self) -> str:
        """The shm directory the pair's segments live under (the producer sweeps
        ``{segment_dir}/{p_engine_id}`` at start): ``shm_dir`` for the shm data
        plane, ``ctrl_dir`` (hybrid control plane) for fabric."""
        return self.shm_dir if self.transport == "shm" else self.ctrl_dir

    @property
    def rendezvous_dir(self) -> str:
        """Where the fabric transport's ``rank{0,1}.json`` spec tables go."""
        return os.path.join(self.ctrl_dir, RENDEZVOUS_DIR)

    def engine_locks(self) -> list[tuple[str, str]]:
        """The ``(segment_dir, engine_id)`` identities this pair's ranks hold."""
        return [(self.segment_dir, self.engine_id(role)) for role in ROLES]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check_role(role: str) -> str:
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    return role


def _check_engine_id(engine_id: str) -> str:
    """An engine id is a segment directory component: the transport's own
    ``xfer_id`` grammar (``[A-Za-z0-9_-]{1,32}``) decides what is allowed."""
    try:
        parse_xfer_id(f"{engine_id}:{'0' * 32}")
    except ValueError:
        raise ValueError(
            f"engine id {engine_id!r} is not a valid segment directory component "
            "([A-Za-z0-9_-]{1,32}); with the defaults it is derived from TAG, so use "
            "a TAG of letters, digits, '_' or '-'"
        ) from None
    return engine_id


# --------------------------------------------------------------------------- #
# argv / configs
# --------------------------------------------------------------------------- #
def additional_config(s: PairSettings) -> dict[str, Any]:
    """``--additional-config`` (identical on both nodes).

    The ``fabric_*`` keys are honoured by the rank entry point's ``open_mesh_device``
    wrapper (the stock plugin ignores ``fabric_config`` for 1-device meshes,
    PHASE3_NOTES 2.6.1).  They are emitted for the fabric data plane and for
    ``TRANSPORT=shm SHM_FABRIC=1`` (like-for-like comparison); plain ``TRANSPORT=shm``
    opens the mesh exactly as ``run_pd_pair.sh`` does (no fabric config).
    """
    tt: dict[str, Any] = {
        "trace_region_size": s.trace_region,
        "l1_small_size": s.l1_small,
    }
    if s.fabric_enabled:
        tt["fabric_config"] = s.fabric_config
        tt["fabric_reliability_mode"] = s.reliability
        tt["fabric_max_packet_payload_bytes"] = s.pkt
    return {"tt": tt}


def kv_transfer_config(s: PairSettings, role: str) -> dict[str, Any]:
    """``--kv-transfer-config`` for one node (Phase 2 JSON with transport 'fabric').

    The ``fabric_*`` / ``socket_connections`` keys are exactly the ones
    ``transport.fabric.FabricConfig._EXTRA_KEYS`` reads (round-trip test in
    tests/kv_transfer/test_pd_fabric_launch.py); ``shm_dir`` stays for the factory
    (popped for kind=fabric) and ``TT_PD_FABRIC_DIR`` in global_env carries the same
    control dir as the env fallback.
    """
    _check_role(role)
    extra: dict[str, Any]
    if s.transport == "shm":
        # exactly run_pd_pair.sh's Phase 2 shm config (dumpfile mode)
        extra = {
            "transport": "shm",
            "shm_mode": s.shm_mode,
            "shm_dir": s.shm_dir,
            "kv_lease_duration": s.lease,
            "hybrid_state": s.hybrid_state,
            "xfer_chunk_tokens": s.chunk_tokens,
        }
        if role == "prefill":
            extra["shm_budget_bytes"] = s.shm_budget
    else:
        extra = {
            "transport": "fabric",
            "shm_mode": s.shm_mode,
            "shm_dir": s.ctrl_dir,
            # the shm control segments of the hybrid control plane live here
            "fabric_control_dir": s.ctrl_dir,
            "kv_lease_duration": s.lease,
            "hybrid_state": s.hybrid_state,
            "xfer_chunk_tokens": s.chunk_tokens,
            "socket_connections": s.conns,
            "fabric_fifo_bytes": s.fifo_bytes,
            "fabric_rec_sets": s.rec_sets,
            "fabric_export_budget_bytes": s.export_budget,
            "fabric_export_slots": s.export_slots,
            "fabric_max_model_len": s.ctx,
            "fabric_max_packet_payload_bytes": s.pkt,
        }
    cfg: dict[str, Any] = {
        "kv_connector": CONNECTOR_CLASS,
        "kv_connector_module_path": CONNECTOR_MODULE,
        "kv_role": KV_ROLE_OF_ROLE[role],
        "engine_id": s.engine_id(role),
    }
    if role == "decode":
        cfg["kv_load_failure_policy"] = s.d_fail_policy
        extra["max_inflight_loads"] = s.max_inflight
        extra["max_import_chunks_per_step"] = s.chunks_per_step
    cfg["kv_connector_extra_config"] = extra
    return cfg


def serve_argv(s: PairSettings, role: str) -> list[str]:
    """The ``vllm serve`` arguments of one node (everything after ``vllm serve``)."""
    _check_role(role)
    argv = [
        s.weights,
        "--served-model-name",
        s.served_model_name,
        "--port",
        str(s.port(role)),
        "--block-size",
        str(s.block_size),
        "--max-model-len",
        str(s.ctx),
        "--max-num-batched-tokens",
        str(s.max_num_batched_tokens),
        "--max-num-seqs",
        str(s.max_num_seqs(role)),
        "--seed",
        str(s.seed),
        *(["--enable-auto-tool-choice"] if s.enable_auto_tool_choice else []),
        *(["--tool-call-parser", s.tool_call_parser] if s.tool_call_parser else []),
        *(["--reasoning-parser", s.reasoning_parser] if s.reasoning_parser else []),
        "--max-log-len",
        "32",
        # remote headless engine: the front-end binds the handshake ROUTER on
        # tcp://<dp_address>:<rpc> and starts no local engine (utils.py
        # launch_core_engines); the rank connects to the same address.
        "--data-parallel-size",
        "1",
        "--data-parallel-size-local",
        "0",
        "--data-parallel-address",
        s.dp_address,
        "--data-parallel-rpc-port",
        str(s.rpc_port(role)),
        "--additional-config",
        json.dumps(additional_config(s), separators=(",", ":")),
        "--kv-transfer-config",
        json.dumps(kv_transfer_config(s, role), separators=(",", ":")),
    ]
    argv += list(s.p_extra_args if role == "prefill" else s.d_extra_args)
    return argv


def node_args(s: PairSettings) -> dict[str, Any]:
    return {
        "prefill": serve_argv(s, "prefill"),
        "decode": serve_argv(s, "decode"),
        "meta": {
            "pair": s.pair,
            "chips": {"prefill": s.p_chip, "decode": s.d_chip},
            "rpc_ports": {"prefill": s.p_rpc, "decode": s.d_rpc},
            "ports": {"prefill": s.p_port, "decode": s.d_port, "proxy": s.proxy_port},
            "dp_address": s.dp_address,
            "transport": s.transport,
            "fabric_enabled": s.fabric_enabled,
            "engine_ids": {"prefill": s.p_engine_id, "decode": s.d_engine_id},
            "segment_dir": s.segment_dir,
            "shm_dir": s.shm_dir,
            "rendezvous_dir": s.rendezvous_dir,
            "fabric": {
                "config": s.fabric_config,
                "reliability": s.reliability,
                "packet_bytes": s.pkt,
                "connections": s.conns,
                "fifo_bytes": s.fifo_bytes,
                "export_budget_bytes": s.export_budget,
                "export_slots": s.export_slots,
                "max_export_kv_buffers": max_export_kv_buffers(s.ctx),
            },
            "ctrl_dir": s.ctrl_dir,
            "tag": s.tag,
        },
    }


def write_node_args(path: str, s: PairSettings) -> dict[str, Any]:
    d = node_args(s)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)
    return d


def load_node_args(path: str) -> dict[str, Any]:
    with open(path) as f:
        d = json.load(f)
    for role in ROLES:
        if role not in d or not isinstance(d[role], list) or not d[role]:
            raise ValueError(f"{path}: missing argv list for role {role!r}")
        if not all(isinstance(a, str) for a in d[role]):
            raise ValueError(f"{path}: argv of {role!r} must be a list of strings")
    return d


# --------------------------------------------------------------------------- #
# argv inspection (no vllm)
# --------------------------------------------------------------------------- #
def argv_value(argv: list[str], flag: str) -> str | None:
    """Last value of ``flag`` in ``argv`` (``--flag v`` or ``--flag=v``), else None."""
    val: str | None = None
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            val = argv[i + 1]
        elif a.startswith(flag + "="):
            val = a.split("=", 1)[1]
    return val


def argv_json(argv: list[str], flag: str) -> dict[str, Any]:
    v = argv_value(argv, flag)
    if v is None:
        raise ValueError(f"argv has no {flag}")
    d = json.loads(v)
    if not isinstance(d, dict):
        raise ValueError(f"{flag} must be a JSON object")
    return d


def segment_dir_of(extra: dict[str, Any]) -> str | None:
    """The shm directory an engine's segments live under, from its
    ``kv_connector_extra_config``: ``fabric_control_dir`` (fabric; ``shm_dir`` is set
    to the same value for the factory) or ``shm_dir`` (shm)."""
    if extra.get("transport") == "fabric":
        return extra.get("fabric_control_dir") or extra.get("shm_dir") or None
    return extra.get("shm_dir") or None


def validate_argv_for_role(argv: list[str], role: str, rpc_port: int) -> dict[str, Any]:
    """Checks the rank runs before importing vllm. Returns a summary dict.

    * ``--data-parallel-rpc-port`` == this rank's PD_RPC_PORT
    * ``--data-parallel-size 1`` / ``--data-parallel-size-local 0`` (headless remote
      engine shape) and no ``--data-parallel-rank`` (external-LB path is refused for
      non-MoE models)
    * ``kv_role`` matches the role, ``transport`` is ``fabric`` or ``shm``,
      ``shm_mode == "dumpfile"``
    * ``tt.fabric_config`` present (FABRIC_2D) when the transport is ``fabric``; with
      ``shm`` it is optional (present = SHM_FABRIC=1 like-for-like mode)
    """
    _check_role(role)
    problems: list[str] = []
    rpc = argv_value(argv, "--data-parallel-rpc-port")
    if rpc is None or int(rpc) != int(rpc_port):
        problems.append(f"--data-parallel-rpc-port {rpc} != PD_RPC_PORT {rpc_port}")
    if (argv_value(argv, "--data-parallel-size") or "1") != "1":
        problems.append("--data-parallel-size must be 1")
    if argv_value(argv, "--data-parallel-size-local") != "0":
        problems.append("--data-parallel-size-local must be 0 (remote headless engine)")
    if argv_value(argv, "--data-parallel-rank") is not None:
        problems.append("--data-parallel-rank must not be set (external LB path)")
    if "--headless" in argv:
        problems.append("--headless belongs to neither the front-end nor the rank")
    ktc: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    try:
        ktc = argv_json(argv, "--kv-transfer-config")
    except ValueError as exc:
        problems.append(str(exc))
    if ktc:
        want = KV_ROLE_OF_ROLE[role]
        if ktc.get("kv_role") != want:
            problems.append(
                f"kv_role {ktc.get('kv_role')!r} != {want!r} for role {role}"
            )
        extra = dict(ktc.get("kv_connector_extra_config") or {})
        if extra.get("transport") not in ("fabric", "shm"):
            problems.append(
                f"transport {extra.get('transport')!r} is neither 'fabric' nor 'shm'"
            )
        if extra.get("shm_mode", "dumpfile") != "dumpfile":
            problems.append(
                "shm_mode must stay 'dumpfile' (model hook warm-up contract; the "
                "fabric transport ignores it, the shm transport's raw mode is blocked)"
            )
        if ktc.get("kv_connector_module_path") != CONNECTOR_MODULE:
            problems.append("kv_connector_module_path is not the TT connector")
        if not ktc.get("engine_id"):
            problems.append("engine_id missing")
        if extra.get("transport") in ("fabric", "shm") and not segment_dir_of(extra):
            problems.append(
                f"transport {extra['transport']} needs a segment dir "
                "(kv_connector_extra_config.shm_dir / fabric_control_dir)"
            )
    tt: dict[str, Any] = {}
    try:
        tt = argv_json(argv, "--additional-config").get("tt") or {}
    except ValueError as exc:
        problems.append(str(exc))
    if extra.get("transport") == "fabric" and "fabric_config" not in tt:
        problems.append(
            "additional_config.tt.fabric_config missing (FABRIC_2D is required for "
            "the fabric data plane)"
        )
    if problems:
        raise ValueError(f"argv for role {role} rejected: " + "; ".join(problems))
    return {
        "role": role,
        "rpc_port": int(rpc),  # type: ignore[arg-type]
        "dp_address": argv_value(argv, "--data-parallel-address") or "127.0.0.1",
        "port": argv_value(argv, "--port"),
        "engine_id": ktc.get("engine_id"),
        "kv_role": ktc.get("kv_role"),
        "transport": extra.get("transport"),
        # the dir whose {engine_id} subdir this engine owns (EngineLock lives here)
        "segment_dir": segment_dir_of(extra),
        "fabric_enabled": "fabric_config" in tt,
        "fabric_config": tt.get("fabric_config"),
        "packet_bytes": tt.get("fabric_max_packet_payload_bytes"),
        "reliability": tt.get("fabric_reliability_mode", "STRICT_INIT"),
        "model": argv[0] if argv and not argv[0].startswith("-") else None,
    }


# --------------------------------------------------------------------------- #
# engine identity lock: one live engine per (segment_dir, engine_id) on the host
# --------------------------------------------------------------------------- #
LOCK_PREFIX = ".pd_engine-"


def engine_lock_path(segment_dir: str, engine_id: str) -> str:
    """``{segment_dir}/.pd_engine-{engine_id}.lock``: a FILE directly under the
    segment dir.  ``ShmTransport`` lists only ``{segment_dir}/{engine_id}`` and the
    segment dirs below it, so no sweep ever sees the lock."""
    return os.path.join(segment_dir, f"{LOCK_PREFIX}{_check_engine_id(engine_id)}.lock")


def read_engine_lock(path: str) -> dict[str, Any] | None:
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def engine_in_use(segment_dir: str, engine_id: str) -> str | None:
    """Why ``(segment_dir, engine_id)`` belongs to a LIVE process, else ``None``.

    1. its lock file names a live pid that started before the lock was written
       (an engine started by ``pd_fabric_rank``; a replayed/reused pid is stale);
    2. a segment under ``{segment_dir}/{engine_id}`` has a live header
       ``producer_pid`` (a lock-less producer: ``run_pd_pair.sh``'s ``vllm serve``);
    3. a ``.claimed-*`` dir there names a live ``consumer.pid`` (a new producer's
       startup sweep would keep it, but its consumer belongs to the other pair).
    An idle lock-less pair (no segments) is NOT detected: the tag-scoped defaults
    are what keeps two pairs apart, the lock is the guard for explicit overrides.
    """
    lock_path = engine_lock_path(segment_dir, engine_id)
    lock = read_engine_lock(lock_path)
    # (pid, created_ts): a pid is reused after a crash and REPLAYED by a restarted
    # container, so the holder must also have started before it wrote the lock
    if lock is not None and pid_wrote_at(
        int(lock.get("pid", 0) or 0), lock.get("created_ts")
    ):
        return (
            f"lock {lock_path} held by live pid {lock['pid']} "
            f"(role {lock.get('role')!r}, tag {lock.get('tag')!r})"
        )
    edir = os.path.join(segment_dir, engine_id)
    try:
        names = os.listdir(edir)
    except OSError:
        return None
    for n in sorted(names):
        seg = os.path.join(edir, n)
        if not os.path.isdir(seg):
            continue
        if ".claimed-" in n:
            try:
                with open(os.path.join(seg, CONSUMER_PID_FILE)) as f:
                    cpid = int(f.read().strip() or "0")
            except (OSError, ValueError):
                cpid = 0
            if pid_alive(cpid):
                return f"segment {seg} is claimed by live consumer pid {cpid}"
        for hname in ("data", "header"):  # ShmTransport._find_header's names
            hp = os.path.join(seg, hname)
            if not os.path.isfile(hp):
                continue
            try:
                hdr = read_header(hp, full=False)
            except (OSError, ValueError):
                break
            if pid_alive(int(hdr.producer_pid)):
                return (
                    f"segment {seg} was written by live producer pid {hdr.producer_pid}"
                )
            break
    return None


def pair_in_use(s: PairSettings) -> list[str]:
    """Every reason one of the pair's engine identities is held by a live process."""
    reasons = []
    for seg, eid in s.engine_locks():
        r = engine_in_use(seg, eid)
        if r:
            reasons.append(r)
    return reasons


@dataclass
class EngineLock:
    """Pid-file lock of one engine identity, held by the rank for its lifetime.

    ``acquire`` refuses (``RuntimeError``) while ``engine_in_use`` names a live
    holder and replaces a stale file otherwise; ``release`` removes only a file that
    still carries our pid (a successor that replaced a stale lock keeps its own).
    """

    segment_dir: str
    engine_id: str
    role: str = ""
    tag: str = ""
    pid: int = 0
    path: str = field(init=False)
    held: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.path = engine_lock_path(self.segment_dir, self.engine_id)
        if not self.pid:
            self.pid = os.getpid()

    def acquire(self) -> EngineLock:
        reason = engine_in_use(self.segment_dir, self.engine_id)
        if reason:
            raise RuntimeError(
                f"engine {self.engine_id!r} under {self.segment_dir} is in use: "
                f"{reason}. Another pair is live on this host with the same "
                "segment dir and engine id: bring it down first, or start this pair "
                "under a different TAG (or SHM_DIR / CTRL_DIR / *_ENGINE_ID)"
            )
        os.makedirs(self.segment_dir, exist_ok=True)
        body = {
            "pid": self.pid,
            "engine_id": self.engine_id,
            "role": self.role,
            "tag": self.tag,
            "created_ts": time.time(),
        }
        tmp = f"{self.path}.{self.pid}.tmp"
        with open(tmp, "w") as f:
            json.dump(body, f)
        os.replace(tmp, self.path)  # a stale lock (dead pid) is replaced atomically
        self.held = True
        return self

    def release(self) -> None:
        if not self.held:
            return
        self.held = False
        lock = read_engine_lock(self.path)
        if lock is not None and int(lock.get("pid", 0) or 0) == self.pid:
            with contextlib.suppress(OSError):
                os.unlink(self.path)


# --------------------------------------------------------------------------- #
# rank binding
# --------------------------------------------------------------------------- #
def rank_env_overrides(s: PairSettings, role: str) -> dict[str, str]:
    """Per-rank ``env_overrides`` (win over global_env)."""
    _check_role(role)
    ov = {
        "TT_VISIBLE_DEVICES": str(s.chip(role)),
        "TT_METAL_CACHE": s.metal_cache(role),
        "PD_ROLE": role,
        "PD_RPC_PORT": str(s.rpc_port(role)),
        "QWEN36_MAX_TOKENS_ALL_USERS": str(s.pool(role)),
        "QWEN36_GDN_DECODE_FUSED": str(
            s.gdn_fused if role == "prefill" else s.d_gdn_fused
        ),
        # Each rank owns an independent (1,1) mesh and its own tensor cache: ttnn's
        # default cache dump (DISTRIBUTED_GATHER) would all-gather across the MPI
        # world and let only world rank 0 write, so the decode rank's cache would
        # never be written and every converted tensor would be a cross-rank barrier
        # (2026-09-22: the container pair wedged at the transport rendezvous).
        "TTNN_TENSOR_CACHE_DUMP_MODE": "local",
    }
    if role == "decode":
        ov["TT_PD_ALLOW_FUSED_CONV"] = str(s.allow_fused_conv)
    tc = s.tensor_cache(role)
    if tc:
        ov["TT_CACHE_PATH"] = tc  # env_overrides win over global_env in tt-run
    return ov


def collect_global_env(
    parent_env: dict[str, str],
    s: PairSettings,
    node_args_path: str,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """global_env for the rank binding: pinned values + parent-env passthrough."""
    g: dict[str, str] = {
        "MESH_DEVICE": "P150",
        "ARCH_NAME": "blackhole",
        "TT_METAL_OPERATION_TIMEOUT_SECONDS": "600",
        "PYTHONUNBUFFERED": "1",
        "PD_ARGS_JSON": node_args_path,
        "PD_FABRIC_PKT": str(s.pkt),
        "PD_FABRIC_RELIABILITY": s.reliability,
        "TT_PD_FABRIC_DIR": s.ctrl_dir,
        "TT_PD_STRICT_SHAPES": str(s.strict),
    }
    for k, v in parent_env.items():
        if k in GLOBAL_ENV_BLOCKLIST:
            continue
        if k in GLOBAL_ENV_NAMES or k.startswith(GLOBAL_ENV_PREFIXES):
            g[k] = v
    if extra:
        for k, v in extra.items():
            if k in GLOBAL_ENV_BLOCKLIST:
                raise ValueError(f"{k} must not be set through global_env")
            g[k] = v
    return g


def render_rank_binding(
    template: dict[str, Any],
    s: PairSettings,
    node_args_path: str,
    parent_env: dict[str, str] | None = None,
    extra_global_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Template (checked-in YAML as a dict) -> final rank binding dict.

    The template fixes chips / caches / roles / ports per rank and the MGD path; the
    renderer overrides them from ``s`` (so knobs like P_RPC win), fills
    ``global_env`` and validates the result with ``check_rank_binding``.
    """
    env = dict(os.environ if parent_env is None else parent_env)
    out: dict[str, Any] = {}
    bindings: list[dict[str, Any]] = []
    tmpl_bindings = {int(b["rank"]): b for b in template.get("rank_bindings", [])}
    for rank in range(WORLD_SIZE):
        role = ROLE_OF_RANK[rank]
        tb = dict(tmpl_bindings.get(rank, {}))
        ov = dict(tb.get("env_overrides") or {})
        ov.update(rank_env_overrides(s, role))
        bindings.append(
            {
                "rank": rank,
                "mesh_id": int(tb.get("mesh_id", rank)),
                "env_overrides": {k: str(v) for k, v in ov.items()},
            }
        )
    out["rank_bindings"] = bindings
    g = dict(template.get("global_env") or {})
    g = {k: str(v) for k, v in g.items() if k not in GLOBAL_ENV_BLOCKLIST}
    g.update(collect_global_env(env, s, node_args_path, extra_global_env))
    out["global_env"] = g
    mgd = template.get("mesh_graph_desc_path") or s.mgd_path
    if not mgd:
        mgd = os.path.join(
            s.root, "profiles", "pd", "pd_two_p150_mesh_graph_descriptor.textproto"
        )
    out["mesh_graph_desc_path"] = str(mgd)
    check_rank_binding(out, s)
    return out


def check_rank_binding(d: dict[str, Any], s: PairSettings | None = None) -> None:
    """Structural checks of a rank binding dict (ttrun TTRunConfig + PD rules)."""
    bindings = d.get("rank_bindings")
    if not isinstance(bindings, list) or len(bindings) != WORLD_SIZE:
        raise ValueError(f"rank_bindings must have exactly {WORLD_SIZE} entries")
    ranks = sorted(int(b["rank"]) for b in bindings)
    if ranks != list(range(WORLD_SIZE)):
        raise ValueError(f"ranks must be {list(range(WORLD_SIZE))}, got {ranks}")
    mesh_ids = sorted(int(b["mesh_id"]) for b in bindings)
    if mesh_ids != [0, 1]:
        raise ValueError(f"mesh_ids must be [0, 1] (two-mesh MGD), got {mesh_ids}")
    chips: set[str] = set()
    rpcs: set[str] = set()
    caches: set[str] = set()
    for b in bindings:
        rank = int(b["rank"])
        ov = b.get("env_overrides") or {}
        for k in ("TT_VISIBLE_DEVICES", "TT_METAL_CACHE", "PD_ROLE", "PD_RPC_PORT"):
            if k not in ov:
                raise ValueError(f"rank {rank}: env_overrides missing {k}")
        if ov["PD_ROLE"] != ROLE_OF_RANK[rank]:
            raise ValueError(
                f"rank {rank} must be role {ROLE_OF_RANK[rank]!r} "
                "(MeshSocket sender_rank=0 = prefill), "
                f"got {ov['PD_ROLE']!r}"
            )
        if "," in ov["TT_VISIBLE_DEVICES"]:
            raise ValueError(
                f"rank {rank}: one chip per rank, got {ov['TT_VISIBLE_DEVICES']!r}"
            )
        chips.add(ov["TT_VISIBLE_DEVICES"])
        rpcs.add(ov["PD_RPC_PORT"])
        caches.add(ov["TT_METAL_CACHE"])
        if s is not None:
            role = ROLE_OF_RANK[rank]
            if int(ov["TT_VISIBLE_DEVICES"]) != s.chip(role):
                raise ValueError(
                    f"rank {rank}: chip {ov['TT_VISIBLE_DEVICES']} != "
                    f"settings {s.chip(role)}"
                )
            if int(ov["PD_RPC_PORT"]) != s.rpc_port(role):
                raise ValueError(
                    f"rank {rank}: PD_RPC_PORT {ov['PD_RPC_PORT']} != "
                    f"settings {s.rpc_port(role)}"
                )
    if len(chips) != WORLD_SIZE:
        raise ValueError("both ranks bind the same chip")
    if len(rpcs) != WORLD_SIZE:
        raise ValueError("both ranks use the same PD_RPC_PORT")
    if len(caches) != WORLD_SIZE:
        raise ValueError(
            "ranks must have distinct TT_METAL_CACHE dirs (PHASE3_MICROBENCH 8.4)"
        )
    g = d.get("global_env") or {}
    bad = sorted(k for k in g if k in GLOBAL_ENV_BLOCKLIST)
    if bad:
        raise ValueError(f"global_env carries per-rank/managed vars: {bad}")
    missing = [k for k in GLOBAL_ENV_REQUIRED if k not in g]
    if missing:
        raise ValueError(f"global_env missing {missing}")
    if not d.get("mesh_graph_desc_path"):
        raise ValueError("mesh_graph_desc_path missing")


def load_yaml(path: str) -> dict[str, Any]:
    import yaml  # PyYAML is a vllm dependency; imported lazily to keep the module light

    with open(path) as f:
        d = yaml.safe_load(f)
    if not isinstance(d, dict):
        raise ValueError(f"{path}: not a mapping")
    return d


def dump_yaml(d: dict[str, Any], path: str) -> None:
    import yaml

    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.write(
            "# RENDERED by vllm_tt_plugin.kv_transfer.launch.pd_launch_config;\n"
            "# do not edit: change the pd_rank_binding_chips*.yaml template or the\n"
            "# run_pd_pair_fabric.sh knobs instead.\n"
        )
        yaml.safe_dump(d, f, sort_keys=False, default_flow_style=False)
    os.replace(tmp, path)


def ttrun_command(
    python: str,
    rank_binding_path: str,
    node_args_path: str,
    extra_rank_args: list[str] | None = None,
) -> list[str]:
    """The tt-run command line (run it from TT_METAL_HOME with the venv active).

    The ``--`` before the program is required: tt-run's click parser would otherwise
    take the program's ``-m`` for its own ``-m/--mesh-graph-descriptor`` option.
    """
    return [
        python,
        "-m",
        "ttnn.distributed.ttrun",
        "--bare",
        "--rank-binding",
        rank_binding_path,
        "--",
        python,
        "-m",
        RANK_MODULE,
        "--pd-args",
        node_args_path,
        *(extra_rank_args or []),
    ]


def frontend_command(vllm_bin: str, s: PairSettings, role: str) -> list[str]:
    return [vllm_bin, "serve", *serve_argv(s, role)]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="render pd_node_args.json + rank binding for the PD fabric pair"
    )
    ap.add_argument(
        "--pair",
        default=None,
        help="prod (chips 0,3) | test (chips 1,2); default env PAIR or prod",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="where pd_node_args_<tag>.json and the rendered YAML go",
    )
    ap.add_argument("--tag", default=None)
    ap.add_argument(
        "--template", default=None, help="rank binding template YAML (default by pair)"
    )
    ap.add_argument(
        "--print-argv0",
        choices=ROLES,
        help="print the role's serve argv NUL-separated and exit",
    )
    ap.add_argument("--print-settings", action="store_true")
    ap.add_argument(
        "--print-ttrun", action="store_true", help="print the tt-run command line"
    )
    ap.add_argument(
        "--check-in-use",
        action="store_true",
        help="exit 3 when a live process holds one of this pair's engine identities "
        "(segment dir + engine id): the producer's startup sweep would destroy its "
        "segments",
    )
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args(argv)

    env = dict(os.environ)
    if a.pair:
        env["PAIR"] = a.pair
    if a.tag:
        env["TAG"] = a.tag
    s = PairSettings.from_env(env)
    out_dir = a.out_dir or os.path.join(s.root, "profiles", "pd")
    node_args_path = os.path.join(out_dir, f"pd_node_args_{s.tag}.json")
    rb_path = os.path.join(out_dir, f"pd_rank_binding_{s.tag}.rendered.yaml")

    if a.print_argv0:
        sys.stdout.write("\0".join(serve_argv(s, a.print_argv0)) + "\0")
        sys.stdout.flush()
        return 0
    if a.print_settings:
        print(json.dumps(s.to_dict(), indent=1))
        return 0
    if a.print_ttrun:
        print(shlex.join(ttrun_command(a.python, rb_path, node_args_path)))
        return 0
    if a.check_in_use:
        reasons = pair_in_use(s)
        for seg, eid in s.engine_locks():
            print(f"engine {eid} under {seg}: {engine_in_use(seg, eid) or 'free'}")
        if reasons:
            print(
                f"REFUSING: {len(reasons)} engine identit"
                f"{'y is' if len(reasons) == 1 else 'ies are'} held by a live process "
                "(the producer's startup sweep would destroy its segments); bring "
                "that pair down or use another TAG / SHM_DIR / CTRL_DIR"
            )
            return 3
        return 0

    os.makedirs(out_dir, exist_ok=True)
    write_node_args(node_args_path, s)
    template_path = a.template or os.path.join(out_dir, TEMPLATE_OF_PAIR[s.pair])
    template = load_yaml(template_path)
    rb = render_rank_binding(template, s, node_args_path, parent_env=env)
    dump_yaml(rb, rb_path)
    print(
        json.dumps(
            {
                "node_args": node_args_path,
                "rank_binding": rb_path,
                "template": template_path,
                "chips": rb["rank_bindings"][0]["env_overrides"]["TT_VISIBLE_DEVICES"]
                + ","
                + rb["rank_bindings"][1]["env_overrides"]["TT_VISIBLE_DEVICES"],
                "rpc": [s.p_rpc, s.d_rpc],
                "ports": [s.p_port, s.d_port, s.proxy_port],
                "transport": s.transport,
                "fabric_enabled": s.fabric_enabled,
                "segment_dir": s.segment_dir,
                "engine_ids": [s.p_engine_id, s.d_engine_id],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())


__all__ = [
    "BOARDS",
    "BOARD_OF_PAIR",
    "GLOBAL_ENV_BLOCKLIST",
    "GLOBAL_ENV_REQUIRED",
    "KV_ROLE_OF_ROLE",
    "LOCK_PREFIX",
    "PAIRS",
    "RANK_MODULE",
    "RANK_OF_ROLE",
    "ROLES",
    "ROLE_OF_RANK",
    "TEMPLATE_OF_PAIR",
    "WORLD_SIZE",
    "EngineLock",
    "PairSettings",
    "additional_config",
    "argv_json",
    "argv_value",
    "check_rank_binding",
    "collect_global_env",
    "dump_yaml",
    "engine_in_use",
    "engine_lock_path",
    "frontend_command",
    "kv_transfer_config",
    "load_node_args",
    "load_yaml",
    "node_args",
    "pair_in_use",
    "rank_env_overrides",
    "read_engine_lock",
    "render_rank_binding",
    "segment_dir_of",
    "serve_argv",
    "ttrun_command",
    "validate_argv_for_role",
    "write_node_args",
]
