# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Container supervisor of the PD pair: one ``vllm serve`` line -> prefill node +
decode node + proxy, with the tool's readiness contract.

Invoked (through ``dispatch.maybe_exec``, ``TT_SERVE_LAUNCHER=pd_container``) as::

    python -m vllm_tt_plugin.kv_transfer.launch.pd_container <argv after "vllm serve">

i.e. with exactly the argv the tt-model container tool composes for a ``vllm-plugin``
profile: ``<weights> --max-model-len N --max-num-seqs N --block-size N
--additional-config <json> --enable-auto-tool-choice --tool-call-parser X
--reasoning-parser Y <profile args> --port P``.  The argv is parsed with vLLM's own
serve parser and mapped onto the Phase 3 pair (``pd_launch_config.PairSettings``):

* ``<weights>`` -> both nodes' model, ``--served-model-name`` -> both nodes;
* ``--port`` -> the proxy's (published) port; P/D front-ends and handshake ROUTERs
  keep fixed ports inside the container network namespace (8100/8200, 29551/29552);
* ``--max-model-len`` -> ``ctx``, ``--max-num-seqs`` -> decode-node concurrency (the
  prefill node runs 1), ``--block-size``, ``--max-num-batched-tokens``;
* ``--additional-config`` ``tt.trace_region_size`` / ``tt.l1_small_size`` -> both
  nodes (``tt.fabric_*`` is refused: the ranks set FABRIC_2D themselves;
  ``tt.sample_on_device_mode`` is dropped with a warning: the TP=1 nodes sample on
  the host);
* tool / reasoning parser flags, ``--no-enable-prefix-caching``,
  ``--no-async-scheduling``, ``--seed`` -> both nodes.  Any other non-default flag
  is rejected (the tool's "unrecognized arguments" diagnosis).

Everything else comes from the environment (the profile's ``env``): ``TT_PD_CHIPS``
(``"0,1"``: prefill chip, decode chip), ``TT_PD_BOARD`` (``p150x2``; ``p300x2`` on the
dev box keeps the on-package-pair refusal), ``TT_PD_TAG`` (``ttm``: scopes the shm
control dir, engine ids and the launch dir), ``TT_PD_P_POOL`` / ``TT_PD_D_POOL``,
``TT_PD_CONNS``, ``TT_PD_PKT``, ``TT_PD_LEASE``, ``TT_PD_EXPORT_SLOTS``,
``TT_PD_MIN_REMOTE``, ``TT_PD_SSE_KEEPALIVE`` (proxy; default 0 = no keep-alive
comments: ``vllm bench serve`` 0.26 counts a stream whose first token arrives after a
keep-alive comment as FAILED, laneB_RESULTS 1.4), ``TT_PD_GDN_FUSED`` (the DECODE rank's
``QWEN36_GDN_DECODE_FUSED``; default 2 = fused GDN decode + fused conv, "M2"),
``TT_PD_ALLOW_FUSED_CONV`` (default 1: the connector accepts the fused-conv decode node,
whose parity-safe slot remap is validated), ``TT_PD_P_GDN_FUSED`` (the PREFILL rank's
value; default 0 = the model default, that rank never decodes), ``TT_PD_BOOT_TIMEOUT``
(P/D health, s), ``TT_METAL_CACHE`` (the tool's ``/cache`` mount: per-rank kernel caches
``<cache>/pd/metal_rank{0,1}`` and the launch dir ``<cache>/pd-launch/<tag>/`` with the
children's logs and rendered files), ``TT_CACHE_PATH`` (converted weights; per rank
under ``<TT_CACHE_PATH>/tp1-rank{0,1}`` unless ``TT_PD_SHARED_TENSOR_CACHE=1``),
``TT_PD_OMPI_BIN`` (prepended to PATH when it exists; the image's OpenMPI bin dir is not
on PATH), ``TT_PD_MGD`` (two-mesh MGD; default = the packaged
``data/pd_two_p150_mesh_graph_descriptor.textproto``).

Launch order = ``profiles/pd/run_pd_pair_fabric.sh up``: front-ends (``python -m
vllm.entrypoints.cli.main serve <node argv>``) -> both handshake ROUTERs listening ->
``tt-run --bare`` with the rendered rank binding -> ``/health`` of P then D -> the
proxy.  The proxy runs IN THIS PROCESS (uvicorn on ``0.0.0.0:<port>``) with a lifespan
whose startup performs those steps, so uvicorn prints ``Application startup complete.``
exactly once, when the pair is healthy -- the tool's READY line.  The children's
stdout/stderr go to files (they are vLLM API servers and print the same line); curated
boot-progress lines are forwarded with ``[P]`` / ``[D]`` / ``[api-p]`` prefixes.

SIGTERM (``docker stop``) -> uvicorn graceful shutdown -> lifespan exit -> front-ends
SIGTERM -> python ranks SIGTERM (graceful: transport shutdown, mesh close; <= 20 s) ->
prterun / tt-run -> SIGKILL leftovers -> orphans carrying this launch's
``PD_ARGS_JSON`` marker -> stale rendezvous files.  Only PIDs this supervisor started
(or that carry its marker) are ever signalled.  ``--dry-run`` renders everything
(node args, rank binding, commands, env summary) and launches nothing.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from .dispatch import ENV_VAR as LAUNCHER_ENV_VAR
from .pd_launch_config import (
    ROLE_OF_RANK,
    ROLES,
    PairSettings,
    dump_yaml,
    load_yaml,
    pair_in_use,
    render_rank_binding,
    serve_argv,
    ttrun_command,
    write_node_args,
)

log = logging.getLogger("vllm_tt_plugin.pd_container")

# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
DEFAULT_CACHE_ROOT = "/cache"  # the tool's TT_METAL_CACHE mount
DEFAULT_OMPI_BIN = "/opt/openmpi-v5.0.7-ulfm/bin"  # runtime image, not on PATH
DEFAULT_TAG = "ttm"
DEFAULT_CHIPS = "0,1"
DEFAULT_BOARD = "p150x2"
FRONTEND_MODULE = "vllm.entrypoints.cli.main"
PACKAGED_MGD = "pd_two_p150_mesh_graph_descriptor.textproto"
PACKAGED_TEMPLATE = "pd_rank_binding_p150x2.yaml"
# the tool's READY_LINE (container.py wait_ready); children print it too (they are
# vLLM API servers) and it must never reach the container log from them
READY_LINE = "Application startup complete"

# serve-parser attributes the supervisor maps; any other attribute that differs from
# the parser default is an unknown flag for the PD pair -> rejected
ALLOWED_SERVE_ATTRS: frozenset[str] = frozenset(
    {
        "model_tag",
        "model",
        "served_model_name",
        "port",
        "host",
        "max_model_len",
        "max_num_seqs",
        "block_size",
        "additional_config",
        "max_num_batched_tokens",
        "enable_auto_tool_choice",
        "tool_call_parser",
        "reasoning_parser",
        "enable_prefix_caching",
        "async_scheduling",
        "seed",
        "max_log_len",
        "uvicorn_log_level",
        "enable_log_requests",
        "disable_log_requests",
        "disable_log_stats",
    }
)
# tt.* keys of --additional-config
TT_KEYS_MAPPED: frozenset[str] = frozenset({"trace_region_size", "l1_small_size"})
TT_KEYS_DROPPED: frozenset[str] = frozenset({"sample_on_device_mode"})

# environment the children must not inherit: the launcher hook (children are plain
# vllm processes) and everything the rank binding sets per rank
CHILD_ENV_STRIP: frozenset[str] = frozenset(
    {
        LAUNCHER_ENV_VAR,
        "TT_VISIBLE_DEVICES",
        "TT_MESH_ID",
        "TT_MESH_HOST_RANK",
        "TT_MESH_GRAPH_DESC_PATH",
        "TT_RUN_RANK",
        "TT_RUN_ORIGINAL_CWD",
        "PD_ROLE",
        "PD_RPC_PORT",
        "PD_ARGS_JSON",
        "VLLM_ENGINE_READY_TIMEOUT_S",
    }
)
# defaults of the validated Phase 3 pair (run_pd_pair_fabric.sh common_env); an
# explicit value in the container env wins
CHILD_ENV_DEFAULTS: dict[str, str] = {
    "ARCH_NAME": "blackhole",
    "HF_HUB_OFFLINE": "1",
    "VLLM_TARGET_DEVICE": "tt",
    "VLLM_RPC_TIMEOUT": "900000",
    "VLLM_CONFIGURE_LOGGING": "1",
    "TORCHDYNAMO_DISABLE": "1",
    "TT_QWEN35_TEXT_VER": "qwen36_blackhole",
    "QWEN36_FORCE_TP_PATH": "1",
    "QWEN36_SKIP_VISION": "1",
    "QWEN36_PREFILL_BUCKET_TRACE": "1",  # required by the KV connector
    "QWEN36_MTP": "0",  # plain PD: no drafter head / MTP KV cache on either node
    "TT_METAL_OPERATION_TIMEOUT_SECONDS": "600",
    "TT_PD_FABRIC_PUMP": "1",
    "TT_PD_FABRIC_IDLE_TICK_S": "0.005",
    "TT_PD_FABRIC_CLAIM_WAIT_S": "0.010",
    "TT_PD_FABRIC_CLAIM_LEASE_S": "0",
    "VLLM_PROCESS_NAME_PREFIX": "PDFABRIC",
    "PYTHONUNBUFFERED": "1",
}
# speculative decoding has no mirror protocol on the PD decode node (merge review):
# refuse a container env that switches a drafter on
SPEC_ENV_FORBIDDEN: dict[str, tuple[str, ...]] = {
    "QWEN36_DRAFTER": ("", "none", "off", "0"),  # allowed values
    "QWEN36_MTP": ("", "0"),
    "QWEN36_SPEC": ("", "0"),
}

# boot-progress lines forwarded from the children's logs (tool boot_progress.py
# checklist + failure causes + the pair's own milestones)
FORWARD_RE = re.compile(
    r"Attempting to open mesh device|multidevice with .* is created|"
    r"KV cache size|init engine|engine lock .* taken|"
    r"VllmConfig built|fabric config set|mesh opened|socket created|"
    r"starting headless EngineCoreProc|"
    r"Traceback|Error|error:|RuntimeError|unrecognized arguments|"
    r"exited on signal|non-zero|engine core (exited|returned|failed)|"
    r"Started server process|Uvicorn running"
)
_TTRUN_PREFIX_RE = re.compile(r"^\[\d+,(\d+)\]<(stdout|stderr)>:\s?")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


# --------------------------------------------------------------------------- #
# serve argv -> ServeRequest
# --------------------------------------------------------------------------- #
@dataclass
class ServeRequest:
    """What the supervisor takes from the tool's ``vllm serve`` line."""

    model: str
    served_model_name: str = ""
    port: int = 8000
    max_model_len: int = 65536
    max_num_seqs: int = 8
    block_size: int = 64
    max_num_batched_tokens: int = 0  # 0 = max_model_len
    trace_region_size: int = 536870912
    l1_small_size: int = 24576
    tool_call_parser: str = ""
    reasoning_parser: str = ""
    enable_auto_tool_choice: bool = False
    seed: int | None = None
    extra_args: list[str] = field(default_factory=list)  # per-node passthrough flags
    dropped: dict[str, Any] = field(default_factory=dict)  # tt.* keys not applied


def _differing_attrs(ns: Any, defaults: Any) -> list[str]:
    out = []
    for k, v in vars(ns).items():
        if k.startswith("_"):
            continue
        if not hasattr(defaults, k) or getattr(defaults, k) != v:
            out.append(k)
    return sorted(out)


def serve_request_from_namespace(ns: Any, defaults: Any) -> ServeRequest:
    """Map a parsed serve namespace onto a ``ServeRequest``; ``defaults`` is the same
    parser's namespace for an empty argv (what "not given" looks like)."""
    given = _differing_attrs(ns, defaults)
    unknown = [k for k in given if k not in ALLOWED_SERVE_ATTRS]
    if unknown:
        raise ValueError(
            "vllm serve flags the PD pair does not take (unrecognized arguments for "
            f"TT_SERVE_LAUNCHER=pd_container): {unknown}; allowed: "
            f"{sorted(ALLOWED_SERVE_ATTRS)}"
        )
    model = getattr(ns, "model_tag", None) or getattr(ns, "model", None)
    if not model or "model_tag" not in given and "model" not in given:
        raise ValueError("vllm serve needs the model (weights repo id or path)")
    smn = getattr(ns, "served_model_name", None)
    if isinstance(smn, (list, tuple)):
        smn = smn[0] if smn else ""
    tt_cfg: dict[str, Any] = {}
    ac = getattr(ns, "additional_config", None) or {}
    if isinstance(ac, str):
        ac = json.loads(ac)
    if not isinstance(ac, dict):
        raise ValueError("--additional-config must be a JSON object")
    tt_cfg = dict(ac.get("tt") or {})
    other = sorted(k for k in ac if k != "tt")
    if other:
        raise ValueError(
            f"--additional-config keys outside 'tt' not supported: {other}"
        )
    dropped = {k: tt_cfg.pop(k) for k in list(tt_cfg) if k in TT_KEYS_DROPPED}
    bad = sorted(k for k in tt_cfg if k not in TT_KEYS_MAPPED)
    if bad:
        raise ValueError(
            f"--additional-config tt keys the PD pair sets itself or cannot take: "
            f"{bad} "
            f"(the ranks set fabric_config/FABRIC_2D; mapped: {sorted(TT_KEYS_MAPPED)})"
        )
    extra: list[str] = []
    if "enable_prefix_caching" in given:
        v = ns.enable_prefix_caching
        if v is not None:
            extra.append(
                "--enable-prefix-caching" if v else "--no-enable-prefix-caching"
            )
    if "async_scheduling" in given:
        v = ns.async_scheduling
        if v is not None:
            extra.append("--async-scheduling" if v else "--no-async-scheduling")

    def _int(name: str, dflt: int) -> int:
        v = getattr(ns, name, None)
        return dflt if v is None else int(v)

    return ServeRequest(
        model=str(model),
        served_model_name=str(smn or ""),
        port=_int("port", 8000),
        max_model_len=_int("max_model_len", 65536),
        max_num_seqs=_int("max_num_seqs", 8),
        block_size=_int("block_size", 64),
        max_num_batched_tokens=_int("max_num_batched_tokens", 0),
        trace_region_size=int(tt_cfg.get("trace_region_size", 536870912)),
        l1_small_size=int(tt_cfg.get("l1_small_size", 24576)),
        tool_call_parser=str(getattr(ns, "tool_call_parser", None) or ""),
        reasoning_parser=str(getattr(ns, "reasoning_parser", None) or ""),
        enable_auto_tool_choice=bool(getattr(ns, "enable_auto_tool_choice", False)),
        seed=(int(ns.seed) if "seed" in given and ns.seed is not None else None),
        extra_args=extra,
        dropped=dropped,
    )


def vllm_serve_parser() -> Callable[[list[str]], Any]:
    """vLLM's own ``vllm serve`` parser (imports vllm; the plugin's general-plugin
    hook runs during the import and is a no-op here: dispatch popped the variable)."""
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = make_arg_parser(FlexibleArgumentParser(prog="vllm serve"))
    return parser.parse_args


def parse_serve_argv(
    argv: list[str], parser: Callable[[list[str]], Any] | None = None
) -> ServeRequest:
    parse = parser or vllm_serve_parser()
    return serve_request_from_namespace(parse(list(argv)), parse([]))


# --------------------------------------------------------------------------- #
# environment -> settings / paths
# --------------------------------------------------------------------------- #
def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    v = env.get(key, "")
    return int(v) if v.strip() else default


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    v = env.get(key, "")
    return float(v) if v.strip() else default


def parse_chips(text: str) -> tuple[int, int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(
            f"TT_PD_CHIPS must be '<prefill chip>,<decode chip>', got {text!r}"
        )
    p, d = int(parts[0]), int(parts[1])
    return p, d


def packaged_data_path(name: str) -> str:
    import importlib.resources as r

    return str(r.files(__package__).joinpath("data", name))


@dataclass
class LaunchPaths:
    cache_root: str
    tag: str
    launch_dir: str = field(init=False)
    node_args: str = field(init=False)
    rank_binding: str = field(init=False)
    plan: str = field(init=False)
    ranks_log: str = field(init=False)
    frontend_logs: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        self.launch_dir = os.path.join(self.cache_root, "pd-launch", self.tag)
        self.node_args = os.path.join(self.launch_dir, "pd_node_args.json")
        self.rank_binding = os.path.join(
            self.launch_dir, "pd_rank_binding.rendered.yaml"
        )
        self.plan = os.path.join(self.launch_dir, "pd_container_plan.json")
        self.ranks_log = os.path.join(self.launch_dir, "ranks.log")
        self.frontend_logs = {
            role: os.path.join(self.launch_dir, f"{role}_frontend.log")
            for role in ROLES
        }


@dataclass
class ContainerKnobs:
    """Supervisor-only knobs (everything that is not a PairSettings field)."""

    boot_timeout: float = 1800.0  # P/D /health after the ranks start
    port_timeout: float = 600.0  # handshake ROUTERs listening
    ready_timeout: float = 3600.0  # VLLM_ENGINE_READY_TIMEOUT_S of the front-ends
    rank_grace: float = 20.0  # SIGTERM -> SIGKILL window of the python ranks
    frontend_grace: float = 10.0
    launcher_grace: float = 10.0
    min_remote: int = 2
    sse_keepalive: float = 0.0
    log_timing: bool = False
    ompi_bin: str = DEFAULT_OMPI_BIN
    tt_metal_home: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> ContainerKnobs:
        return cls(
            boot_timeout=_env_float(env, "TT_PD_BOOT_TIMEOUT", cls.boot_timeout),
            port_timeout=_env_float(env, "TT_PD_PORT_TIMEOUT", cls.port_timeout),
            ready_timeout=_env_float(env, "TT_PD_READY_TIMEOUT", cls.ready_timeout),
            rank_grace=_env_float(env, "TT_PD_RANK_GRACE_S", cls.rank_grace),
            min_remote=_env_int(env, "TT_PD_MIN_REMOTE", cls.min_remote),
            sse_keepalive=_env_float(env, "TT_PD_SSE_KEEPALIVE", cls.sse_keepalive),
            log_timing=env.get("TT_PD_PROXY_LOG_TIMING", "0") == "1",
            ompi_bin=env.get("TT_PD_OMPI_BIN", cls.ompi_bin),
            tt_metal_home=env.get("TT_METAL_HOME", ""),
        )


def build_pair_settings(
    req: ServeRequest, env: Mapping[str, str]
) -> tuple[PairSettings, LaunchPaths]:
    """Container-mode ``PairSettings`` from the serve request + environment."""
    p_chip, d_chip = parse_chips(env.get("TT_PD_CHIPS", DEFAULT_CHIPS))
    tag = env.get("TT_PD_TAG", DEFAULT_TAG) or DEFAULT_TAG
    cache_root = env.get("TT_METAL_CACHE", "") or DEFAULT_CACHE_ROOT
    paths = LaunchPaths(cache_root=cache_root, tag=tag)
    tensor_cache = env.get("TT_CACHE_PATH", "") or os.path.join(
        cache_root, "tensors-tp1"
    )
    shared_tc = env.get("TT_PD_SHARED_TENSOR_CACHE", "0") == "1"
    s = PairSettings(
        pair="p150x2",
        board=env.get("TT_PD_BOARD", DEFAULT_BOARD),
        p_chip=p_chip,
        d_chip=d_chip,
        weights=req.model,
        served_model_name=req.served_model_name or req.model,
        p_port=_env_int(env, "TT_PD_P_PORT", 8100),
        d_port=_env_int(env, "TT_PD_D_PORT", 8200),
        proxy_port=req.port,
        p_rpc=_env_int(env, "TT_PD_P_RPC", 29551),
        d_rpc=_env_int(env, "TT_PD_D_RPC", 29552),
        ctx=req.max_model_len,
        p_pool=_env_int(env, "TT_PD_P_POOL", 65536),
        d_pool=_env_int(env, "TT_PD_D_POOL", 163840),
        d_conc=req.max_num_seqs,
        lease=_env_float(env, "TT_PD_LEASE", 30.0),
        # decode rank: fused GDN decode + fused conv (2, M2) is the validated
        # default; the prefill rank never decodes and keeps the model default
        # (0). laneE_RESULTS.md 6.
        gdn_fused=_env_int(env, "TT_PD_P_GDN_FUSED", 0),
        d_gdn_fused=_env_int(env, "TT_PD_GDN_FUSED", 2),
        allow_fused_conv=_env_int(env, "TT_PD_ALLOW_FUSED_CONV", 1),
        min_remote=_env_int(env, "TT_PD_MIN_REMOTE", 2),
        trace_region=req.trace_region_size,
        l1_small=req.l1_small_size,
        max_inflight=_env_int(env, "TT_PD_MAX_INFLIGHT", 2),
        chunk_tokens=_env_int(env, "TT_PD_CHUNK_TOKENS", 2048),
        pkt=_env_int(env, "TT_PD_PKT", 8704),
        conns=_env_int(env, "TT_PD_CONNS", 2),
        export_slots=_env_int(env, "TT_PD_EXPORT_SLOTS", 1),
        seed=req.seed if req.seed is not None else 9472,
        metal_cache_root=os.path.join(cache_root, "pd"),
        tag=tag,
        p_extra_args=list(req.extra_args),
        d_extra_args=list(req.extra_args),
        bucket_trace_gate=env.get("QWEN36_PREFILL_BUCKET_TRACE", "1") or "1",
        block_size=req.block_size,
        max_num_batched_tokens=req.max_num_batched_tokens,
        tool_call_parser=req.tool_call_parser,
        reasoning_parser=req.reasoning_parser,
        enable_auto_tool_choice=int(req.enable_auto_tool_choice),
        tensor_cache_root="" if shared_tc else tensor_cache,
        mgd_path=env.get("TT_PD_MGD", "") or packaged_data_path(PACKAGED_MGD),
    )
    return s, paths


def check_spec_off(env: Mapping[str, str]) -> None:
    """The PD decode node imports KV without a speculative-decoding mirror: refuse an
    environment that turns a drafter on (merge review minor)."""
    bad = {
        k: env[k]
        for k, ok in SPEC_ENV_FORBIDDEN.items()
        if env.get(k, "").strip().lower() not in ok
    }
    if bad:
        raise ValueError(
            f"speculative decoding is not supported on the PD pair (no spec mirror on "
            f"the decode node): unset {bad}"
        )


def child_env(
    parent: Mapping[str, str],
    s: PairSettings,
    knobs: ContainerKnobs,
    *,
    isdir: Callable[[str], bool] = os.path.isdir,
) -> dict[str, str]:
    """Environment of the front-ends and of tt-run (whose ranks get global_env /
    env_overrides on top): parent minus the stripped keys, plus the pair defaults."""
    e = {k: v for k, v in parent.items() if k not in CHILD_ENV_STRIP}
    for k, v in CHILD_ENV_DEFAULTS.items():
        e.setdefault(k, v)
    e.setdefault("HF_MODEL", s.weights)
    # both nodes are (1,1) meshes; the tool's launcher sets the board's MESH_DEVICE
    e["MESH_DEVICE"] = "P150"
    e["TT_PD_FABRIC_DIR"] = s.ctrl_dir
    e["TT_PD_STRICT_SHAPES"] = str(s.strict)
    # shared converted-weights root (per-rank dirs come from env_overrides)
    e["TT_CACHE_PATH"] = s.tensor_cache_root or e.get("TT_CACHE_PATH", "")
    if not e["TT_CACHE_PATH"]:
        raise ValueError("TT_CACHE_PATH (converted weights cache) must be set")
    if knobs.ompi_bin and isdir(knobs.ompi_bin):
        path = e.get("PATH", "")
        if knobs.ompi_bin not in path.split(":"):
            e["PATH"] = f"{knobs.ompi_bin}:{path}" if path else knobs.ompi_bin
    check_spec_off(e)
    return e


def frontend_command(python: str, s: PairSettings, role: str) -> list[str]:
    """``python -m vllm.entrypoints.cli.main serve <node argv>`` (same ``main()`` as
    the ``vllm`` console script; the cmdline no longer reads "vllm serve", so a
    tenant's ``pkill -f "vllm serve"`` cannot take the front-ends down)."""
    return [python, "-m", FRONTEND_MODULE, "serve", *serve_argv(s, role)]


# --------------------------------------------------------------------------- #
# process table helpers (pure over a snapshot; /proc readers below)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProcInfo:
    pid: int
    ppid: int
    comm: str
    cmdline: str


def read_proc_table() -> list[ProcInfo]:
    out: list[ProcInfo] = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            with open(f"/proc/{pid}/stat") as f:
                stat = f.read()
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            continue
        # comm may contain spaces/parens: it is the text between the first "(" and
        # the last ")"
        lp, rp = stat.find("("), stat.rfind(")")
        comm = stat[lp + 1 : rp] if lp >= 0 and rp > lp else ""
        rest = stat[rp + 2 :].split()
        try:
            ppid = int(rest[1])
        except (IndexError, ValueError):
            ppid = 0
        out.append(ProcInfo(pid, ppid, comm, cmdline))
    return out


def descendants(table: Iterable[ProcInfo], root: int) -> list[ProcInfo]:
    """Every descendant of ``root`` (deepest first), from a table snapshot."""
    kids: dict[int, list[ProcInfo]] = {}
    for p in table:
        kids.setdefault(p.ppid, []).append(p)
    out: list[ProcInfo] = []

    def walk(pid: int) -> None:
        for c in kids.get(pid, []):
            walk(c.pid)
            out.append(c)

    walk(root)
    return out


_LAUNCHER_RE = re.compile(r"prterun|mpirun|mpiexec|prte\b")


def is_launcher(p: ProcInfo) -> bool:
    return bool(_LAUNCHER_RE.search(p.comm)) or bool(
        re.search(r"(^| )[^ ]*(prterun|mpirun|mpiexec)[^ ]*( |$)", p.cmdline)
    )


def is_ttrun(p: ProcInfo) -> bool:
    return "ttnn.distributed.ttrun" in p.cmdline


def is_rank(p: ProcInfo) -> bool:
    """A python rank: the ``pd_fabric_rank`` program (before vLLM retitles it) or a
    process retitled ``<prefix>::EngineCore``; never a launcher or tt-run."""
    if is_launcher(p) or is_ttrun(p):
        return False
    return (
        "::EngineCo" in p.comm
        or "::EngineCore" in p.cmdline
        or "pd_fabric_rank" in p.cmdline
    )


def select_rank_pids(table: Iterable[ProcInfo], ttrun_pid: int) -> list[int]:
    return [p.pid for p in descendants(table, ttrun_pid) if is_rank(p)]


def select_launcher_pids(table: Iterable[ProcInfo], ttrun_pid: int) -> list[int]:
    return [p.pid for p in descendants(table, ttrun_pid) if is_launcher(p)]


def pids_with_marker(marker: str, table: Iterable[ProcInfo] | None = None) -> list[int]:
    """PIDs whose environment carries ``PD_ARGS_JSON=<marker>`` (this launch's ranks,
    even after tt-run is gone); other users' processes are unreadable and skipped."""
    out = []
    want = f"PD_ARGS_JSON={marker}".encode()
    for p in read_proc_table() if table is None else table:
        if not ("EngineCore" in p.cmdline or "pd_fabric_rank" in p.cmdline):
            continue
        try:
            with open(f"/proc/{p.pid}/environ", "rb") as f:
                env = f.read()
        except OSError:
            continue
        if want in env.split(b"\0"):
            out.append(p.pid)
    return out


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def tcp_port_open(port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def http_health(port: int, host: str = "127.0.0.1", timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/health", timeout=timeout
        ) as r:
            return 200 <= r.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# log forwarding
# --------------------------------------------------------------------------- #
def emit_line(text: str) -> None:
    """Default progress sink: the container log is a pipe/file, so flush per line."""
    print(text, flush=True)


def forwardable(line: str, source: str) -> str | None:
    """The line to forward (prefixed), or None.  tt-run's ``[1,<rank>]<stream>: ``
    prefix
    becomes ``[P]``/``[D]``; a front-end's READY line is never forwarded."""
    text = _ANSI_RE.sub("", line.rstrip("\n"))
    if READY_LINE in text:
        return None
    prefix = f"[{source}]"
    m = _TTRUN_PREFIX_RE.match(text)
    if m:
        role = ROLE_OF_RANK.get(int(m.group(1)))
        prefix = "[P]" if role == "prefill" else "[D]" if role == "decode" else prefix
        text = text[m.end() :]
    if not FORWARD_RE.search(text):
        return None
    return f"{prefix} {text.strip()}"


def rotate_log(path: str) -> str | None:
    """Keep the previous boot's file as ``<path>.prev`` (one generation) and start
    ``path`` afresh; returns the rotated path or None when there was nothing."""
    if not os.path.exists(path):
        return None
    prev = f"{path}.prev"
    os.replace(path, prev)
    return prev


class LogForwarder:
    """Tails the children's log files and forwards curated lines to ``sink``."""

    def __init__(
        self,
        files: Mapping[str, str],
        sink: Callable[[str], None],
        interval: float = 0.5,
    ) -> None:
        self.files = dict(files)
        self.sink = sink
        self.interval = interval
        self._pos: dict[str, int] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def poll(self) -> int:
        n = 0
        for source, path in self.files.items():
            try:
                with open(path, "rb") as f:
                    f.seek(self._pos.get(source, 0))
                    data = f.read()
                    self._pos[source] = f.tell()
            except OSError:
                continue
            if not data:
                continue
            # keep an unterminated tail for the next poll
            if not data.endswith(b"\n"):
                cut = data.rfind(b"\n") + 1
                self._pos[source] -= len(data) - cut
                data = data[:cut]
            for raw in data.decode(errors="replace").splitlines():
                out = forwardable(raw, source)
                if out:
                    self.sink(out)
                    n += 1
        return n

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.poll()
        self.poll()

    def start(self) -> LogForwarder:
        self._thread = threading.Thread(
            target=self._run, name="pd-log-forward", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


# --------------------------------------------------------------------------- #
# supervisor
# --------------------------------------------------------------------------- #
class Supervisor:
    """Brings the pair up and down; every system interaction is injectable."""

    def __init__(
        self,
        s: PairSettings,
        paths: LaunchPaths,
        env: dict[str, str],
        knobs: ContainerKnobs,
        *,
        python: str = sys.executable,
        popen: Callable[..., Any] = subprocess.Popen,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        port_open: Callable[[int], bool] = tcp_port_open,
        health: Callable[[int], bool] = http_health,
        proc_table: Callable[[], list[ProcInfo]] = read_proc_table,
        marker_pids: Callable[[str], list[int]] = pids_with_marker,
        kill: Callable[[int, int], None] = os.kill,
        alive: Callable[[int], bool] = pid_alive,
        makedirs: Callable[[str], None] = lambda p: os.makedirs(p, exist_ok=True),
        in_use: Callable[[PairSettings], list[str]] = pair_in_use,
        should_stop: Callable[[], bool] = lambda: False,
        emit: Callable[[str], None] = emit_line,
        template: dict[str, Any] | None = None,
    ) -> None:
        self.s = s
        self.paths = paths
        self.env = env
        self.knobs = knobs
        self.python = python
        self._popen = popen
        self._clock = clock
        self._sleep = sleep
        self._port_open = port_open
        self._health = health
        self._proc_table = proc_table
        self._marker_pids = marker_pids
        self._kill = kill
        self._alive = alive
        self._makedirs = makedirs  # runtime dirs (shm control dir, caches)
        self._in_use = in_use  # live holders of this tag's engine identities
        self.should_stop = should_stop
        self.emit = emit
        self.template = template
        self.frontends: dict[str, Any] = {}
        self.ranks: Any | None = None  # the tt-run process
        self.events: list[str] = []  # ordered record of what happened (tests)
        self.forwarder: LogForwarder | None = None
        self.plan: dict[str, Any] = {}
        self._torn_down = False

    # --- rendering ---------------------------------------------------------- #
    def render(self) -> dict[str, Any]:
        os.makedirs(self.paths.launch_dir, exist_ok=True)
        write_node_args(self.paths.node_args, self.s)
        tmpl = self.template
        if tmpl is None:
            tmpl = load_yaml(packaged_data_path(PACKAGED_TEMPLATE))
        rb = render_rank_binding(
            tmpl, self.s, self.paths.node_args, parent_env=self.env
        )
        dump_yaml(rb, self.paths.rank_binding)
        ttrun = ttrun_command(
            self.python, self.paths.rank_binding, self.paths.node_args
        )
        self.plan = {
            "tag": self.s.tag,
            "chips": {"prefill": self.s.p_chip, "decode": self.s.d_chip},
            "board": self.s.board,
            "ports": {
                "proxy": self.s.proxy_port,
                "prefill": self.s.p_port,
                "decode": self.s.d_port,
                "rpc": [self.s.p_rpc, self.s.d_rpc],
            },
            "ctx": self.s.ctx,
            "pools": {"prefill": self.s.p_pool, "decode": self.s.d_pool},
            "max_num_seqs": {"prefill": 1, "decode": self.s.d_conc},
            "weights": self.s.weights,
            "served_model_name": self.s.served_model_name,
            "launch_dir": self.paths.launch_dir,
            "node_args": self.paths.node_args,
            "rank_binding": self.paths.rank_binding,
            "mesh_graph_desc_path": rb["mesh_graph_desc_path"],
            "metal_cache": {r: self.s.metal_cache(r) for r in ROLES},
            "tensor_cache": {
                r: self.s.tensor_cache(r) or self.env["TT_CACHE_PATH"] for r in ROLES
            },
            "ctrl_dir": self.s.ctrl_dir,
            "engine_ids": [self.s.p_engine_id, self.s.d_engine_id],
            "frontends": {
                r: shlex.join(frontend_command(self.python, self.s, r)) for r in ROLES
            },
            "ttrun": shlex.join(ttrun),
            "ttrun_cwd": self.knobs.tt_metal_home or os.getcwd(),
            "path": self.env.get("PATH", ""),
            "global_env_keys": sorted(rb["global_env"]),
            "logs": {**self.paths.frontend_logs, "ranks": self.paths.ranks_log},
            "proxy": {
                "min_remote_tokens": self.knobs.min_remote,
                "sse_keepalive": self.knobs.sse_keepalive,
            },
            "timeouts": {
                "ports": self.knobs.port_timeout,
                "boot": self.knobs.boot_timeout,
                "ready": self.knobs.ready_timeout,
                "rank_grace": self.knobs.rank_grace,
            },
        }
        with open(self.paths.plan, "w") as f:
            json.dump(self.plan, f, indent=1)
            f.write("\n")
        return self.plan

    # --- launch steps -------------------------------------------------------- #
    def _spawn(
        self, cmd: list[str], logfile: str, env: dict[str, str], cwd: str | None
    ):
        rotate_log(logfile)  # one boot per file: the forwarder must not replay the last
        f = open(logfile, "wb")  # noqa: SIM115 - handed to the child, closed below
        try:
            proc = self._popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                cwd=cwd,
                start_new_session=True,
            )
        finally:
            f.close()
        return proc

    def start_frontends(self) -> None:
        fe_env = dict(self.env)
        fe_env["VLLM_ENGINE_READY_TIMEOUT_S"] = str(int(self.knobs.ready_timeout))
        for role in ROLES:
            cmd = frontend_command(self.python, self.s, role)
            proc = self._spawn(cmd, self.paths.frontend_logs[role], fe_env, None)
            self.frontends[role] = proc
            self.events.append(f"start_frontend:{role}")
            self.emit(
                f"[pd_container] {role} front-end pid {proc.pid} port "
                f"{self.s.port(role)} "
                f"rpc {self.s.rpc_port(role)} log {self.paths.frontend_logs[role]}"
            )

    def _check_children(self) -> None:
        for role, proc in self.frontends.items():
            rc = proc.poll()
            if rc is not None:
                raise RuntimeError(
                    f"{role} front-end (pid {proc.pid}) exited with {rc}; see "
                    f"{self.paths.frontend_logs[role]}"
                )
        if self.ranks is not None:
            rc = self.ranks.poll()
            if rc is not None:
                raise RuntimeError(
                    f"tt-run job (pid {self.ranks.pid}) exited with {rc}; tail of "
                    f"{self.paths.ranks_log}:\n{self._tail(self.paths.ranks_log)}"
                )
        if self.should_stop():
            raise RuntimeError("stop requested during startup")

    @staticmethod
    def _tail(path: str, n: int = 30) -> str:
        try:
            with open(path, errors="replace") as f:
                lines = f.readlines()
        except OSError:
            return ""
        return "".join(_ANSI_RE.sub("", ln) for ln in lines[-n:])

    def wait_ports(self) -> None:
        t0 = self._clock()
        want = [self.s.p_rpc, self.s.d_rpc]
        while True:
            up = [p for p in want if self._port_open(p)]
            if len(up) == len(want):
                self.events.append("ports_listening")
                self.emit(
                    f"[pd_container] handshake ports {want} listening after "
                    f"{self._clock() - t0:.0f} s"
                )
                return
            self._check_children()
            if self._clock() - t0 > self.knobs.port_timeout:
                raise RuntimeError(
                    f"handshake ports {want} not listening after "
                    f"{self.knobs.port_timeout} s "
                    f"(have {up})"
                )
            self._sleep(2)

    def start_ranks(self) -> None:
        cmd = ttrun_command(self.python, self.paths.rank_binding, self.paths.node_args)
        cwd = self.knobs.tt_metal_home or None
        self.ranks = self._spawn(cmd, self.paths.ranks_log, self.env, cwd)
        self.events.append("start_ranks")
        self.emit(
            f"[pd_container] tt-run pid {self.ranks.pid} (ranks: prefill chip "
            f"{self.s.p_chip}, decode chip {self.s.d_chip}) log {self.paths.ranks_log}"
        )

    def wait_health(self, role: str) -> None:
        t0 = self._clock()
        port = self.s.port(role)
        while True:
            if self._health(port):
                self.events.append(f"healthy:{role}")
                self.emit(
                    f"[pd_container] {role} node healthy on {port} after "
                    f"{self._clock() - t0:.0f} s"
                )
                return
            self._check_children()
            if self._clock() - t0 > self.knobs.boot_timeout:
                raise RuntimeError(
                    f"{role} node not healthy after {self.knobs.boot_timeout} s; "
                    "tail of "
                    f"{self.paths.ranks_log}:\n{self._tail(self.paths.ranks_log)}"
                )
            self._sleep(5)

    def bring_up(self) -> None:
        """Front-ends -> handshake ports -> ranks -> P health -> D health.  Any failure
        tears down what started and re-raises (the lifespan startup then fails)."""
        try:
            self._makedirs(self.s.ctrl_dir)
            for role in ROLES:
                self._makedirs(self.s.metal_cache(role))
                tc = self.s.tensor_cache(role)
                if tc:
                    self._makedirs(tc)
            in_use = self._in_use(self.s)
            if in_use:
                raise RuntimeError(
                    "engine identity in use by a live process (another pair with the "
                    f"same TT_PD_TAG={self.s.tag!r}): {in_use}"
                )
            self.forwarder = LogForwarder(
                {
                    **{f"api-{r[0]}": p for r, p in self.paths.frontend_logs.items()},
                    "ranks": self.paths.ranks_log,
                },
                self.emit,
            ).start()
            self.start_frontends()
            self.wait_ports()
            self.start_ranks()
            for role in ROLES:
                self.wait_health(role)
            self.events.append("pair_ready")
            self.emit(
                f"[pd_container] pair ready: proxy 0.0.0.0:{self.s.proxy_port} -> "
                f"P 127.0.0.1:{self.s.p_port} / D 127.0.0.1:{self.s.d_port}"
            )
        except BaseException:
            self.emit("[pd_container] startup failed; tearing the pair down")
            self.teardown()
            raise

    # --- teardown ------------------------------------------------------------ #
    def _signal(self, pid: int, sig: int, what: str) -> None:
        try:
            self._kill(pid, sig)
            self.events.append(f"{signal.Signals(sig).name}:{what}:{pid}")
        except ProcessLookupError:
            pass

    def _wait_dead(self, pids: Iterable[int], timeout: float) -> list[int]:
        pids = list(pids)
        t0 = self._clock()
        while True:
            live = [p for p in pids if self._alive(p)]
            if not live or self._clock() - t0 >= timeout:
                return live
            self._sleep(1)

    def teardown(self) -> None:
        if self._torn_down:
            return
        self._torn_down = True
        self.events.append("teardown")
        # 1. front-ends first: their engine clients stop talking to the ranks
        fe = [(r, p) for r, p in self.frontends.items() if p.poll() is None]
        for role, proc in fe:
            self._signal(proc.pid, signal.SIGTERM, f"frontend:{role}")
        live = self._wait_dead([p.pid for _, p in fe], self.knobs.frontend_grace)
        for role, proc in fe:
            if proc.pid in live:
                self._signal(proc.pid, signal.SIGKILL, f"frontend:{role}")
        # 2. the python ranks ONLY (graceful: transport shutdown -> mesh close); never
        #    prterun (it would SIGKILL the ranks after a short grace) nor tt-run
        table = self._proc_table()
        tt_pid = self.ranks.pid if self.ranks is not None else None
        rank_pids: list[int] = []
        launcher_pids: list[int] = []
        if tt_pid is not None and self._alive(tt_pid):
            rank_pids = select_rank_pids(table, tt_pid)
            launcher_pids = select_launcher_pids(table, tt_pid)
        rank_pids = sorted(
            set(rank_pids) | set(self._marker_pids(self.paths.node_args))
        )
        for pid in rank_pids:
            self._signal(pid, signal.SIGTERM, "rank")
        live = self._wait_dead(rank_pids, self.knobs.rank_grace)
        if live:
            self.emit(
                f"[pd_container] ranks {live} still alive after "
                f"{self.knobs.rank_grace} s; "
                "SIGKILL (probe the chips before the next boot)"
            )
            for pid in live:
                self._signal(pid, signal.SIGKILL, "rank")
        # 3. the MPI job ends when its ranks exit; then leftovers, gentle first
        if tt_pid is not None:
            self._wait_dead([tt_pid], self.knobs.launcher_grace)
            for pid in [*launcher_pids, tt_pid]:
                if self._alive(pid):
                    self._signal(pid, signal.SIGTERM, "launcher")
            live = self._wait_dead([*launcher_pids, tt_pid], 5)
            for pid in live:
                self._signal(pid, signal.SIGKILL, "launcher")
            with contextlib.suppress(Exception):
                self.ranks.wait(timeout=1)
        # 4. orphans by marker (a rank whose tt-run is already gone)
        for pid in self._marker_pids(self.paths.node_args):
            if self._alive(pid):
                self._signal(pid, signal.SIGKILL, "orphan")
        self.sweep_stale_rendezvous()
        if self.forwarder is not None:
            self.forwarder.stop()
        self.events.append("teardown_done")
        self.emit("[pd_container] pair down")

    def sweep_stale_rendezvous(self) -> list[str]:
        removed: list[str] = []
        d = self.s.rendezvous_dir
        try:
            names = os.listdir(d)
        except OSError:
            return removed
        for n in names:
            if not (n.startswith("rank") and n.endswith(".json")):
                continue
            path = os.path.join(d, n)
            try:
                with open(path) as f:
                    pid = int((json.load(f) or {}).get("pid", 0) or 0)
            except (OSError, ValueError):
                pid = 0
            if not pid or not self._alive(pid):
                with contextlib.suppress(OSError):
                    os.unlink(path)
                removed.append(path)
        return removed


# --------------------------------------------------------------------------- #
# proxy app + uvicorn
# --------------------------------------------------------------------------- #
def proxy_config(s: PairSettings, knobs: ContainerKnobs) -> Any:
    from ..pd_proxy import PDProxyConfig

    return PDProxyConfig(
        prefill_url=f"http://127.0.0.1:{s.p_port}",
        decode_url=f"http://127.0.0.1:{s.d_port}",
        min_remote_tokens=knobs.min_remote,
        sse_keepalive=knobs.sse_keepalive,
        log_timing=knobs.log_timing,
    )


def build_app(sup: Supervisor, cfg: Any) -> Any:
    """The proxy app whose lifespan brings the pair up before serving (uvicorn's
    "Application startup complete." = pair healthy) and down at shutdown."""
    from ..pd_proxy import create_app

    async def startup() -> None:
        await asyncio.get_running_loop().run_in_executor(None, sup.bring_up)

    async def shutdown() -> None:
        await asyncio.get_running_loop().run_in_executor(None, sup.teardown)

    return create_app(cfg, startup=startup, shutdown=shutdown)


def serve_forever(sup: Supervisor, cfg: Any, host: str, port: int) -> int:
    """Run uvicorn in this process; SIGTERM/SIGINT -> graceful shutdown (lifespan
    exit tears the pair down).  Returns 0 after a clean shutdown, 3 when the startup
    failed (uvicorn's STARTUP_FAILURE)."""
    import uvicorn

    app = build_app(sup, cfg)
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        lifespan="on",
        timeout_graceful_shutdown=10,
    )
    server = uvicorn.Server(config)
    sup.should_stop = lambda: bool(server.should_exit)
    server.run()
    return 0 if server.started else 3


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def split_own_flags(argv: list[str]) -> tuple[list[str], bool]:
    """Take the supervisor's own ``--dry-run`` out of the serve argv."""
    dry = "--dry-run" in argv
    return [a for a in argv if a != "--dry-run"], dry


def main(
    argv: list[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    parser: Callable[[list[str]], Any] | None = None,
    run: Callable[[Supervisor, Any, str, int], int] = serve_forever,
    emit: Callable[[str], None] = emit_line,
    supervisor_kwargs: dict[str, Any] | None = None,
) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    argv = list(sys.argv[1:] if argv is None else argv)
    env = dict(os.environ if env is None else env)
    serve_args, dry_run = split_own_flags(argv)
    req = parse_serve_argv(serve_args, parser)
    for k, v in req.dropped.items():
        emit(
            f"[pd_container] dropping --additional-config tt.{k}={v!r}: the TP=1 "
            "PD nodes sample on the host"
        )
    s, paths = build_pair_settings(req, env)
    knobs = ContainerKnobs.from_env(env)
    cenv = child_env(env, s, knobs)
    sup = Supervisor(s, paths, cenv, knobs, emit=emit, **(supervisor_kwargs or {}))
    plan = sup.render()
    if dry_run:
        plan["dry_run"] = True
        plan["serve_request"] = asdict(req)
        emit(json.dumps(plan, indent=1))
        return 0
    emit(
        f"[pd_container] PD pair tag={s.tag} board={s.board} chips P={s.p_chip} "
        f"D={s.d_chip} ctx={s.ctx} d_conc={s.d_conc} pools P={s.p_pool} D={s.d_pool} "
        f"proxy 0.0.0.0:{s.proxy_port}; launch dir {paths.launch_dir}"
    )
    return run(sup, proxy_config(s, knobs), "0.0.0.0", s.proxy_port)


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ALLOWED_SERVE_ATTRS",
    "CHILD_ENV_DEFAULTS",
    "CHILD_ENV_STRIP",
    "READY_LINE",
    "ContainerKnobs",
    "LaunchPaths",
    "LogForwarder",
    "ProcInfo",
    "ServeRequest",
    "Supervisor",
    "build_app",
    "build_pair_settings",
    "check_spec_off",
    "child_env",
    "descendants",
    "emit_line",
    "forwardable",
    "frontend_command",
    "main",
    "packaged_data_path",
    "parse_chips",
    "parse_serve_argv",
    "proxy_config",
    "rotate_log",
    "select_launcher_pids",
    "select_rank_pids",
    "serve_forever",
    "serve_request_from_namespace",
    "split_own_flags",
]
