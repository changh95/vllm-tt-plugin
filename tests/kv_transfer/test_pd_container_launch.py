# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host tests of the container entry point: ``dispatch`` (TT_SERVE_LAUNCHER hook) and
``pd_container`` (serve argv -> pair settings, env assembly, dry-run rendering,
READY-line sequencing through the proxy lifespan, PID-scoped teardown order).

Everything system-facing is faked: no vllm parser (a small argparse stand-in with
vLLM's attribute names), no processes, no sockets, no /proc."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_tt_plugin.kv_transfer.launch import dispatch
from vllm_tt_plugin.kv_transfer.launch import pd_container as pc
from vllm_tt_plugin.kv_transfer.launch import pd_launch_config as lc

# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
TOOL_ARGV = [
    "Qwen/Qwen3.8-27B",
    "--max-model-len",
    "65536",
    "--max-num-seqs",
    "8",
    "--block-size",
    "64",
    "--additional-config",
    '{"tt": {"l1_small_size": 24576, "sample_on_device_mode": "decode_only", '
    '"trace_region_size": 536870912}}',
    "--enable-auto-tool-choice",
    "--tool-call-parser",
    "qwen3_coder",
    "--reasoning-parser",
    "qwen3",
    "--max-num-batched-tokens",
    "65536",
    "--no-enable-prefix-caching",
    "--no-async-scheduling",
    "--served-model-name",
    "Qwen/Qwen3.8-27B",
    "--port",
    "20000",
]


def fake_serve_parser(argv: list[str]) -> argparse.Namespace:
    """Stand-in for vLLM's ``make_arg_parser``: the attributes the supervisor reads
    plus a few it must reject when set."""
    ap = argparse.ArgumentParser(prog="vllm serve", exit_on_error=False)
    ap.add_argument("model_tag", nargs="?", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--served-model-name", nargs="+", default=None)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default=None)
    ap.add_argument("--max-model-len", type=int, default=None)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--block-size", type=int, default=None)
    ap.add_argument("--additional-config", type=json.loads, default=None)
    ap.add_argument("--max-num-batched-tokens", type=int, default=None)
    ap.add_argument("--enable-auto-tool-choice", action="store_true")
    ap.add_argument("--tool-call-parser", default=None)
    ap.add_argument("--reasoning-parser", default=None)
    ap.add_argument(
        "--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=None
    )
    ap.add_argument(
        "--async-scheduling", action=argparse.BooleanOptionalAction, default=None
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-log-len", type=int, default=None)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--kv-transfer-config", default=None)
    ap.add_argument("--dtype", default="auto")
    return ap.parse_args(argv)


def container_env(tmp_path, **over: str) -> dict[str, str]:
    tmp_path = Path(tmp_path)
    env = {
        "PATH": "/opt/tt-venv/bin:/usr/bin",
        "HOME": "/home/tt",
        "TT_SERVE_LAUNCHER": "pd_container",  # what the profile sets; stripped below
        "TT_METAL_CACHE": str(tmp_path / "cache"),
        "TT_CACHE_PATH": str(tmp_path / "tensors"),
        "TT_PD_CHIPS": "0,1",
        "TT_VISIBLE_DEVICES": "0,1",  # a stray value must not reach the children
        "TT_MESH_GRAPH_DESC_PATH": "/opt/tt-metal/x.textproto",
        "MESH_DEVICE": "P150x2",  # the tool's launcher value; per rank it is P150
        "HF_HUB_OFFLINE": "1",
        "TT_METAL_HOME": "/opt/tt-metal",
    }
    env.update(over)
    return env


class FakeProc:
    def __init__(self, pid: int, cmd: list[str], env: dict[str, str]) -> None:
        self.pid = pid
        self.cmd = cmd
        self.env = env
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int | None:
        return self.returncode


class FakePopen:
    def __init__(self, first_pid: int = 100) -> None:
        self.next_pid = first_pid
        self.procs: list[FakeProc] = []

    def __call__(self, cmd, **kw) -> FakeProc:
        p = FakeProc(self.next_pid, list(cmd), dict(kw.get("env") or {}))
        self.next_pid += 1
        self.procs.append(p)
        return p


class FakeSystem:
    """Ports/health flip to up after N polls; a fake /proc tree; kill recorder."""

    def __init__(self) -> None:
        self.t = 0.0
        self.ports_up = False
        self.health_up: set[int] = set()
        self.table: list[pc.ProcInfo] = []
        self.alive_pids: set[int] = set()
        self.kills: list[tuple[int, str]] = []
        self.marker: list[int] = []
        # pid -> signal name that kills it (others survive that signal)
        self.dies_on: dict[int, str] = {}

    def clock(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s

    def port_open(self, port: int) -> bool:
        return self.ports_up

    def health(self, port: int) -> bool:
        return port in self.health_up

    def proc_table(self) -> list[pc.ProcInfo]:
        return [p for p in self.table if p.pid in self.alive_pids]

    def marker_pids(self, marker: str) -> list[int]:
        return [p for p in self.marker if p in self.alive_pids]

    def kill(self, pid: int, sig: int) -> None:
        if pid not in self.alive_pids:
            raise ProcessLookupError(pid)
        name = signal.Signals(sig).name
        self.kills.append((pid, name))
        if name == "SIGKILL" or self.dies_on.get(pid) == name:
            self.alive_pids.discard(pid)
            # a dead python rank ends the MPI job: prterun and tt-run exit by
            # themselves once every rank is gone
            # job ranks = the ones under prterun (ppid != 1); a tenant's engine or
            # a marker orphan is not part of this MPI job
            ranks = {p.pid for p in self.table if pc.is_rank(p) and p.ppid != 1}
            if not ranks & self.alive_pids:
                for p in self.table:
                    if pc.is_launcher(p) or pc.is_ttrun(p):
                        self.alive_pids.discard(p.pid)

    def alive(self, pid: int) -> bool:
        return pid in self.alive_pids


def make_supervisor(tmp_path, env=None, sysm: FakeSystem | None = None, **kw):
    env = env or container_env(tmp_path)
    req = pc.parse_serve_argv(TOOL_ARGV, fake_serve_parser)
    s, paths = pc.build_pair_settings(req, env)
    knobs = pc.ContainerKnobs.from_env(env)
    cenv = pc.child_env(env, s, knobs, isdir=lambda p: False)
    sysm = sysm or FakeSystem()
    popen = FakePopen()
    emitted: list[str] = []
    sup = pc.Supervisor(
        s,
        paths,
        cenv,
        knobs,
        python="/v/python",
        popen=popen,
        clock=sysm.clock,
        sleep=sysm.sleep,
        port_open=sysm.port_open,
        health=sysm.health,
        proc_table=sysm.proc_table,
        marker_pids=sysm.marker_pids,
        kill=sysm.kill,
        alive=sysm.alive,
        makedirs=lambda p: None,  # never touch /dev/shm from a host test
        in_use=lambda s: [],  # nor read a live pair's locks there
        emit=emitted.append,
        **kw,
    )
    return sup, sysm, popen, emitted


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
def test_dispatch_is_a_noop_without_the_variable_or_outside_serve():
    assert dispatch.plan_exec(["/v/bin/vllm", "serve", "m"], {}) is None
    assert (
        dispatch.plan_exec(
            ["/v/bin/vllm", "bench", "serve"], {"TT_SERVE_LAUNCHER": "pd_container"}
        )
        is None
    )
    # children of the supervisor (vllm front-ends, ranks) never carry the variable
    calls: list = []
    env = {"OTHER": "1"}
    assert (
        dispatch.maybe_exec(
            ["vllm", "serve", "m"], env, execv=lambda *a: calls.append(a)
        )
        is None
    )
    assert calls == [] and env == {"OTHER": "1"}


def test_dispatch_execs_the_supervisor_with_the_argv_after_serve_and_pops_the_var():
    calls: list = []
    env = {"TT_SERVE_LAUNCHER": "pd_container", "KEEP": "1"}
    argv = ["/opt/tt-venv/bin/vllm", "serve", "Qwen/Q", "--port", "20000"]
    plan = dispatch.maybe_exec(argv, env, execv=lambda p, a: calls.append((p, a)))
    assert plan == [
        sys.executable,
        "-m",
        "vllm_tt_plugin.kv_transfer.launch.pd_container",
        "Qwen/Q",
        "--port",
        "20000",
    ]
    assert calls == [(plan[0], plan)]
    assert env == {"KEEP": "1"}  # the supervisor's children see a plain vllm env


def test_dispatch_rejects_an_unknown_launcher_name():
    with pytest.raises(RuntimeError, match="names no launcher"):
        dispatch.plan_exec(["vllm", "serve"], {"TT_SERVE_LAUNCHER": "nope"})


def test_dispatch_entry_point_is_declared_in_pyproject():
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    with open(os.path.join(root, "pyproject.toml")) as f:
        text = f.read()
    assert (
        'tt_serve_launcher = "vllm_tt_plugin.kv_transfer.launch.dispatch:maybe_exec"'
        in text
    )
    assert (
        '"vllm_tt_plugin.kv_transfer.launch" = ["data/*.textproto", "data/*.yaml"]'
        in (text)
    )
    for name in (pc.PACKAGED_MGD, pc.PACKAGED_TEMPLATE):
        assert os.path.isfile(pc.packaged_data_path(name)), name


# --------------------------------------------------------------------------- #
# serve argv -> ServeRequest -> PairSettings
# --------------------------------------------------------------------------- #
def test_serve_argv_maps_onto_the_pair():
    req = pc.parse_serve_argv(TOOL_ARGV, fake_serve_parser)
    assert (
        req.model == "Qwen/Qwen3.8-27B" and req.served_model_name == "Qwen/Qwen3.8-27B"
    )
    assert (req.port, req.max_model_len, req.max_num_seqs, req.block_size) == (
        20000,
        65536,
        8,
        64,
    )
    assert req.max_num_batched_tokens == 65536
    assert (req.trace_region_size, req.l1_small_size) == (536870912, 24576)
    assert (req.tool_call_parser, req.reasoning_parser) == ("qwen3_coder", "qwen3")
    assert req.enable_auto_tool_choice is True
    assert req.seed is None  # not given -> the pair's fixed seed
    assert req.extra_args == ["--no-enable-prefix-caching", "--no-async-scheduling"]
    assert req.dropped == {"sample_on_device_mode": "decode_only"}

    s, paths = pc.build_pair_settings(req, container_env("/tmp/x", TT_PD_CHIPS="0,1"))
    assert (s.p_chip, s.d_chip, s.board) == (0, 1, "p150x2")  # not an on-package pair
    assert s.proxy_port == 20000 and (s.p_port, s.d_port) == (8100, 8200)
    assert (s.ctx, s.d_conc, s.max_num_seqs("prefill")) == (65536, 8, 1)
    assert (s.p_pool, s.d_pool, s.conns) == (65536, 163840, 2)
    assert s.seed == 9472 and s.tag == "ttm"
    assert s.metal_cache("prefill") == "/tmp/x/cache/pd/metal_rank0"
    assert s.metal_cache("decode") == "/tmp/x/cache/pd/metal_rank1"
    assert s.tensor_cache("prefill") == "/tmp/x/tensors/tp1-rank0"
    assert s.tensor_cache("decode") == "/tmp/x/tensors/tp1-rank1"
    assert s.mgd_path.endswith("data/" + pc.PACKAGED_MGD)
    assert paths.launch_dir == "/tmp/x/cache/pd-launch/ttm"
    for role in lc.ROLES:
        argv = lc.serve_argv(s, role)
        assert argv[0] == "Qwen/Qwen3.8-27B"
        assert lc.argv_value(argv, "--served-model-name") == "Qwen/Qwen3.8-27B"
        assert lc.argv_value(argv, "--max-model-len") == "65536"
        assert lc.argv_value(argv, "--max-num-batched-tokens") == "65536"
        assert lc.argv_value(argv, "--block-size") == "64"
        assert lc.argv_value(argv, "--tool-call-parser") == "qwen3_coder"
        assert argv[-2:] == ["--no-enable-prefix-caching", "--no-async-scheduling"]
        tt = lc.argv_json(argv, "--additional-config")["tt"]
        assert tt["fabric_config"] == "FABRIC_2D" and "sample_on_device_mode" not in tt
        lc.validate_argv_for_role(argv, role, s.rpc_port(role))
    assert lc.argv_value(lc.serve_argv(s, "decode"), "--max-num-seqs") == "8"


def test_serve_argv_env_knobs_and_dev_box_pair():
    req = pc.parse_serve_argv(TOOL_ARGV, fake_serve_parser)
    env = container_env(
        "/tmp/y",
        TT_PD_CHIPS="0,3",
        TT_PD_BOARD="p300x2",
        TT_PD_TAG="lane_b",
        TT_PD_D_POOL="131072",
        TT_PD_CONNS="1",
        TT_PD_SHARED_TENSOR_CACHE="1",
        TT_PD_MGD="/cache/my_mgd.textproto",
    )
    s, paths = pc.build_pair_settings(req, env)
    assert (s.p_chip, s.d_chip, s.board) == (0, 3, "p300x2")
    assert s.d_pool == 131072 and s.conns == 1 and s.tag == "lane_b"
    assert s.tensor_cache("prefill") is None  # shared TT_CACHE_PATH
    assert s.mgd_path == "/cache/my_mgd.textproto"
    assert (s.p_engine_id, s.d_engine_id) == ("p-lane_b", "d-lane_b")
    assert paths.launch_dir == "/tmp/y/cache/pd-launch/lane_b"
    # the dev box's on-package pairs stay refused under board p300x2
    with pytest.raises(ValueError, match="on-package"):
        pc.build_pair_settings(
            req, container_env("/tmp/y", TT_PD_CHIPS="0,1", TT_PD_BOARD="p300x2")
        )
    with pytest.raises(ValueError, match="TT_PD_CHIPS"):
        pc.build_pair_settings(req, container_env("/tmp/y", TT_PD_CHIPS="0"))


@pytest.mark.parametrize(
    "argv, match",
    [
        (["Qwen/Q", "--tensor-parallel-size", "2"], "tensor_parallel_size"),
        (["Qwen/Q", "--kv-transfer-config", "{}"], "kv_transfer_config"),
        (["Qwen/Q", "--dtype", "bfloat16"], "dtype"),
        (["--port", "1"], "needs the model"),
        (
            ["Qwen/Q", "--additional-config", '{"tt": {"fabric_config": "FABRIC_1D"}}'],
            "fabric_config",
        ),
        (["Qwen/Q", "--additional-config", '{"other": 1}'], "outside 'tt'"),
    ],
)
def test_serve_argv_rejects_flags_the_pair_cannot_take(argv, match):
    with pytest.raises(ValueError, match=match):
        pc.parse_serve_argv(argv, fake_serve_parser)


def test_serve_argv_defaults_and_seed_passthrough():
    req = pc.parse_serve_argv(["/w/qwen38", "--seed", "7"], fake_serve_parser)
    assert req.model == "/w/qwen38" and req.served_model_name == ""
    assert (req.port, req.max_model_len, req.max_num_seqs, req.block_size) == (
        8000,
        65536,
        8,
        64,
    )
    assert req.seed == 7 and req.extra_args == [] and req.dropped == {}
    s, _ = pc.build_pair_settings(req, container_env("/tmp/z"))
    assert s.served_model_name == "/w/qwen38" and s.seed == 7
    assert s.max_num_batched_tokens == s.ctx
    assert s.tool_call_parser == "" and s.enable_auto_tool_choice == 0
    argv = lc.serve_argv(s, "prefill")
    assert "--enable-auto-tool-choice" not in argv and "--tool-call-parser" not in argv


def test_serve_request_from_namespace_flags_unknown_nondefault_attrs_only():
    dflt = SimpleNamespace(model_tag=None, model="x", port=8000, quantization=None)
    ok = SimpleNamespace(model_tag="m", model="x", port=9000, quantization=None)
    req = pc.serve_request_from_namespace(ok, dflt)
    assert req.model == "m" and req.port == 9000
    bad = SimpleNamespace(model_tag="m", model="x", port=8000, quantization="fp8")
    with pytest.raises(ValueError, match="quantization"):
        pc.serve_request_from_namespace(bad, dflt)


# --------------------------------------------------------------------------- #
# environment assembly
# --------------------------------------------------------------------------- #
def test_child_env_strips_hook_and_per_rank_vars_and_pins_pair_defaults(tmp_path):
    req = pc.parse_serve_argv(TOOL_ARGV, fake_serve_parser)
    env = container_env(tmp_path, TT_PD_OMPI_BIN="/opt/openmpi-v5.0.7-ulfm/bin")
    s, _ = pc.build_pair_settings(req, env)
    knobs = pc.ContainerKnobs.from_env(env)
    e = pc.child_env(env, s, knobs, isdir=lambda p: p == "/opt/openmpi-v5.0.7-ulfm/bin")
    for k in ("TT_SERVE_LAUNCHER", "TT_VISIBLE_DEVICES", "TT_MESH_GRAPH_DESC_PATH"):
        assert k not in e, k
    assert e["MESH_DEVICE"] == "P150"  # per-rank (1,1) mesh, not the board's P150x2
    assert e["QWEN36_MTP"] == "0" and e["QWEN36_PREFILL_BUCKET_TRACE"] == "1"
    assert e["QWEN36_FORCE_TP_PATH"] == "1" and e["VLLM_TARGET_DEVICE"] == "tt"
    assert e["HF_MODEL"] == "Qwen/Qwen3.8-27B" and e["HF_HUB_OFFLINE"] == "1"
    assert e["TT_PD_FABRIC_DIR"] == "/dev/shm/tt_pd_fabric_ttm"
    assert e["TT_PD_FABRIC_PUMP"] == "1" and e["VLLM_PROCESS_NAME_PREFIX"] == "PDFABRIC"
    assert e["TT_CACHE_PATH"] == str(tmp_path / "tensors")
    assert e["PATH"].split(":")[0] == "/opt/openmpi-v5.0.7-ulfm/bin"  # mpirun-ulfm
    # explicit values win over the defaults; a missing OMPI dir leaves PATH alone
    env2 = container_env(tmp_path, QWEN36_PREFILL_BUCKET_TRACE="128,2048")
    e2 = pc.child_env(env2, s, knobs, isdir=lambda p: False)
    assert e2["QWEN36_PREFILL_BUCKET_TRACE"] == "128,2048"
    assert e2["PATH"] == env2["PATH"]


@pytest.mark.parametrize(
    "over",
    [{"QWEN36_DRAFTER": "dflash2"}, {"QWEN36_MTP": "1"}, {"QWEN36_SPEC": "1"}],
)
def test_child_env_refuses_speculative_decoding_on_the_pair(tmp_path, over):
    req = pc.parse_serve_argv(TOOL_ARGV, fake_serve_parser)
    env = container_env(tmp_path, **over)
    s, _ = pc.build_pair_settings(req, env)
    with pytest.raises(ValueError, match="speculative decoding"):
        pc.child_env(env, s, pc.ContainerKnobs.from_env(env), isdir=lambda p: False)


def test_frontend_command_avoids_the_vllm_serve_cmdline():
    s = lc.PairSettings()
    cmd = pc.frontend_command("/v/python", s, "decode")
    assert cmd[:4] == ["/v/python", "-m", "vllm.entrypoints.cli.main", "serve"]
    assert cmd[4:] == lc.serve_argv(s, "decode")


# --------------------------------------------------------------------------- #
# dry run
# --------------------------------------------------------------------------- #
def test_dry_run_renders_everything_and_launches_nothing(tmp_path):
    env = container_env(tmp_path)
    out: list[str] = []
    called: list = []

    def never_run(*a):
        called.append(a)
        return 0

    rc = pc.main(
        [*TOOL_ARGV, "--dry-run"],
        env=env,
        parser=fake_serve_parser,
        run=never_run,
        emit=out.append,
        supervisor_kwargs={"python": "/v/python"},
    )
    assert rc == 0 and called == []
    assert any(
        "dropping --additional-config tt.sample_on_device_mode" in ln for ln in out
    )
    plan = json.loads("\n".join(ln for ln in out if not ln.startswith("[")))
    assert plan["dry_run"] is True
    assert plan["chips"] == {"prefill": 0, "decode": 1}
    assert plan["ports"]["proxy"] == 20000
    launch_dir = tmp_path / "cache" / "pd-launch" / "ttm"
    assert plan["launch_dir"] == str(launch_dir)
    for f in (
        "pd_node_args.json",
        "pd_rank_binding.rendered.yaml",
        "pd_container_plan.json",
    ):
        assert (launch_dir / f).is_file(), f
    na = lc.load_node_args(str(launch_dir / "pd_node_args.json"))
    assert lc.argv_value(na["prefill"], "--port") == "8100"
    assert lc.argv_value(na["decode"], "--port") == "8200"
    assert na["meta"]["ports"]["proxy"] == 20000
    rb = lc.load_yaml(str(launch_dir / "pd_rank_binding.rendered.yaml"))
    s = lc.PairSettings(pair="p150x2", metal_cache_root=str(tmp_path / "cache" / "pd"))
    lc.check_rank_binding(rb, s)
    ov = {b["rank"]: b["env_overrides"] for b in rb["rank_bindings"]}
    assert ov[0]["TT_VISIBLE_DEVICES"] == "0" and ov[1]["TT_VISIBLE_DEVICES"] == "1"
    assert ov[0]["TT_CACHE_PATH"] == str(tmp_path / "tensors" / "tp1-rank0")
    assert ov[1]["TT_CACHE_PATH"] == str(tmp_path / "tensors" / "tp1-rank1")
    assert ov[0]["TT_METAL_CACHE"] == str(tmp_path / "cache" / "pd" / "metal_rank0")
    g = rb["global_env"]
    assert g["PD_ARGS_JSON"] == str(launch_dir / "pd_node_args.json")
    assert g["QWEN36_MTP"] == "0" and g["MESH_DEVICE"] == "P150"
    assert "TT_SERVE_LAUNCHER" not in g and "TT_VISIBLE_DEVICES" not in g
    assert rb["mesh_graph_desc_path"].endswith("data/" + pc.PACKAGED_MGD)
    assert os.path.isfile(rb["mesh_graph_desc_path"])
    assert plan["ttrun"].startswith("/v/python -m ttnn.distributed.ttrun --bare")
    assert plan["frontends"]["prefill"].startswith(
        "/v/python -m vllm.entrypoints.cli.main serve Qwen/Qwen3.8-27B"
    )


def test_child_logs_are_rotated_per_boot_so_the_forwarder_never_replays(tmp_path):
    """The launch dir persists across boots (it is the tool's cache mount); a child
    log opened for append would make the forwarder re-emit the previous boot's lines
    (its READY-line filter aside, a stale Traceback would look current)."""
    sup, sysm, popen, out = make_supervisor(tmp_path)
    sup.render()
    log = sup.paths.frontend_logs["prefill"]
    with open(log, "w") as f:
        f.write("[api-p] old boot Traceback\n")
    sup.start_frontends()
    assert os.path.getsize(log) == 0  # fresh file for this boot
    with open(log + ".prev") as f:
        assert "old boot" in f.read()
    fw = pc.LogForwarder({"api-p": log}, out.append)
    assert fw.poll() == 0
    assert pc.rotate_log(str(tmp_path / "missing.log")) is None


def test_split_own_flags():
    assert pc.split_own_flags(["m", "--dry-run", "--port", "1"]) == (
        ["m", "--port", "1"],
        True,
    )
    assert pc.split_own_flags(["m"]) == (["m"], False)


# --------------------------------------------------------------------------- #
# launch order and READY-line sequencing
# --------------------------------------------------------------------------- #
def rank_tree(tt_pid: int) -> list[pc.ProcInfo]:
    return [
        pc.ProcInfo(tt_pid, 1, "python", "/v/python -m ttnn.distributed.ttrun --bare"),
        pc.ProcInfo(tt_pid + 1, tt_pid, "prterun", "/usr/bin/prterun --np 2 ..."),
        pc.ProcInfo(
            tt_pid + 2,
            tt_pid + 1,
            "python",
            "/v/python -m vllm_tt_plugin.kv_transfer.launch.pd_fabric_rank --pd-args x",
        ),
        pc.ProcInfo(
            tt_pid + 3, tt_pid + 1, "PDFABRIC::EngineCore", "PDFABRIC::EngineCore"
        ),
    ]


def arm_pair(sup: pc.Supervisor, sysm: FakeSystem, popen: FakePopen) -> None:
    """Make the fake world come up in the right order as the supervisor acts."""
    orig_start_ranks = sup.start_ranks

    def start_ranks():
        orig_start_ranks()
        tt = sup.ranks.pid
        sysm.table = rank_tree(tt)
        sysm.alive_pids |= {p.pid for p in sysm.table}
        sysm.dies_on.update({tt + 2: "SIGTERM", tt + 3: "SIGTERM"})
        sysm.health_up = {8100, 8200}

    sup.start_ranks = start_ranks  # type: ignore[method-assign]
    orig_start_fe = sup.start_frontends

    def start_frontends():
        orig_start_fe()
        for p in popen.procs:
            sysm.alive_pids.add(p.pid)
            sysm.dies_on[p.pid] = "SIGTERM"
        sysm.ports_up = True

    sup.start_frontends = start_frontends  # type: ignore[method-assign]


def test_bring_up_order_frontends_ports_ranks_health(tmp_path):
    sup, sysm, popen, out = make_supervisor(tmp_path)
    sup.render()
    arm_pair(sup, sysm, popen)
    sup.bring_up()
    assert sup.events == [
        "start_frontend:prefill",
        "start_frontend:decode",
        "ports_listening",
        "start_ranks",
        "healthy:prefill",
        "healthy:decode",
        "pair_ready",
    ]
    fe_p, fe_d, tt = popen.procs
    assert fe_p.cmd[:4] == ["/v/python", "-m", "vllm.entrypoints.cli.main", "serve"]
    assert lc.argv_value(fe_p.cmd, "--port") == "8100"
    assert lc.argv_value(fe_d.cmd, "--port") == "8200"
    assert fe_p.env["VLLM_ENGINE_READY_TIMEOUT_S"] == "3600"
    assert "TT_SERVE_LAUNCHER" not in fe_p.env
    assert tt.cmd[:4] == ["/v/python", "-m", "ttnn.distributed.ttrun", "--bare"]
    assert "VLLM_ENGINE_READY_TIMEOUT_S" not in tt.env
    # the supervisor's own progress lines never contain the tool's READY line
    assert not any(pc.READY_LINE in ln for ln in out)
    sup.teardown()


def test_bring_up_fails_fast_when_a_frontend_dies_and_tears_down(tmp_path):
    sup, sysm, popen, out = make_supervisor(tmp_path)
    sup.render()
    orig = sup.start_frontends

    def start_and_die():
        orig()
        popen.procs[1].returncode = 2  # decode front-end: unrecognized arguments

    sup.start_frontends = start_and_die  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="decode front-end .* exited with 2"):
        sup.bring_up()
    assert "teardown" in sup.events and sup.events[-1] == "teardown_done"
    assert "start_ranks" not in sup.events


def test_bring_up_honours_a_stop_request_and_a_boot_timeout(tmp_path):
    sup, sysm, popen, _ = make_supervisor(tmp_path, should_stop=lambda: True)
    sup.render()
    sysm.ports_up = True
    with pytest.raises(RuntimeError, match="stop requested"):
        sup.bring_up()
    sup2, sysm2, popen2, _ = make_supervisor(tmp_path)
    sup2.knobs.boot_timeout = 30
    sup2.render()
    sysm2.ports_up = True  # ranks never get healthy
    with pytest.raises(RuntimeError, match="prefill node not healthy after 30"):
        sup2.bring_up()
    assert sup2.events[-1] == "teardown_done"


def test_proxy_lifespan_brings_the_pair_up_before_serving_and_down_after(tmp_path):
    """The lifespan is what uvicorn wraps: startup complete (= the READY line) only
    after ``bring_up`` returned; teardown after the app stopped serving."""
    pytest.importorskip("fastapi")
    sup, sysm, popen, _ = make_supervisor(tmp_path)
    sup.render()
    arm_pair(sup, sysm, popen)
    app = pc.build_app(sup, pc.proxy_config(sup.s, sup.knobs))
    seen: list[list[str]] = []

    async def drive() -> None:
        async with app.router.lifespan_context(app):
            seen.append(list(sup.events))

    asyncio.run(drive())
    assert seen[0][-1] == "pair_ready" and "teardown" not in seen[0]
    assert sup.events.index("teardown") > sup.events.index("pair_ready")
    assert sup.events[-1] == "teardown_done"


def test_uvicorn_prints_startup_complete_once_after_the_pair_is_healthy(
    tmp_path, caplog
):
    uvicorn = pytest.importorskip("uvicorn")
    pytest.importorskip("fastapi")
    sup, sysm, popen, out = make_supervisor(tmp_path)
    sup.render()
    arm_pair(sup, sysm, popen)
    marks: list[tuple[str, float]] = []
    orig_bring_up = sup.bring_up

    def bring_up():
        time.sleep(0.2)  # the pair takes a while; READY must wait for it
        orig_bring_up()
        marks.append(("pair_ready", time.monotonic()))

    sup.bring_up = bring_up  # type: ignore[method-assign]
    app = pc.build_app(sup, pc.proxy_config(sup.s, sup.knobs))
    config = uvicorn.Config(
        app, host="127.0.0.1", port=0, log_level="info", lifespan="on", log_config=None
    )
    server = uvicorn.Server(config)
    sup.should_stop = lambda: bool(server.should_exit)

    class _Rec(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            marks.append((record.getMessage(), time.monotonic()))

    rec = _Rec()
    logging.getLogger("uvicorn.error").addHandler(rec)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    try:
        th = threading.Thread(target=server.run, daemon=True)
        th.start()
        t0 = time.monotonic()
        while not server.started and th.is_alive() and time.monotonic() - t0 < 30:
            time.sleep(0.05)
        assert server.started, marks
        server.should_exit = True
        th.join(timeout=30)
        assert not th.is_alive()
    finally:
        logging.getLogger("uvicorn.error").removeHandler(rec)
    ready = [t for m, t in marks if pc.READY_LINE in m]
    assert len(ready) == 1, marks
    pair_ready = [t for m, t in marks if m == "pair_ready"]
    assert pair_ready and pair_ready[0] <= ready[0]
    assert sup.events[-1] == "teardown_done"


# --------------------------------------------------------------------------- #
# teardown order (PID-scoped)
# --------------------------------------------------------------------------- #
def test_teardown_signals_frontends_then_ranks_then_launchers_and_nothing_else(
    tmp_path,
):
    sup, sysm, popen, _ = make_supervisor(tmp_path)
    sup.render()
    arm_pair(sup, sysm, popen)
    sup.bring_up()
    tt = sup.ranks.pid
    # rank tt+3 ignores SIGTERM (hung in a device call): SIGKILL after the grace
    sysm.dies_on[tt + 3] = "never"
    # an unrelated tenant's engine that must never be touched
    sysm.table.append(pc.ProcInfo(999, 1, "VLLM::EngineCore", "VLLM::EngineCore"))
    sysm.alive_pids.add(999)
    sup.teardown()
    names = [f"{sig}:{pid}" for pid, sig in sysm.kills]
    fe = [p.pid for p in popen.procs[:2]]
    # 1. front-ends TERM first
    assert names[:2] == [f"SIGTERM:{fe[0]}", f"SIGTERM:{fe[1]}"]
    # 2. python ranks TERM (never prterun / tt-run at this point)
    assert names[2:4] == [f"SIGTERM:{tt + 2}", f"SIGTERM:{tt + 3}"]
    # 3. the hung rank is SIGKILLed after the grace window, then the job ends by itself
    assert names[4] == f"SIGKILL:{tt + 3}"
    assert not any(pid in (tt, tt + 1) for pid, _ in sysm.kills)  # exited on their own
    assert 999 not in {pid for pid, _ in sysm.kills}
    assert sysm.alive_pids == {999}
    assert sup.events[-1] == "teardown_done"
    # idempotent
    n = len(sysm.kills)
    sup.teardown()
    assert len(sysm.kills) == n


def test_teardown_signals_launchers_only_after_the_ranks_and_reaps_marker_orphans(
    tmp_path,
):
    sup, sysm, popen, _ = make_supervisor(tmp_path)
    sup.render()
    arm_pair(sup, sysm, popen)
    sup.bring_up()
    tt = sup.ranks.pid
    # prterun/tt-run linger after the ranks are gone: TERM, then KILL
    orig_kill = sysm.kill
    gone: set[int] = set()

    def sticky_kill(pid: int, sig: int) -> None:
        orig_kill(pid, sig)
        if pid in (tt, tt + 1) and signal.Signals(sig).name == "SIGKILL":
            gone.add(pid)
        sysm.alive_pids |= {tt, tt + 1} - gone  # launchers linger until SIGKILLed

    sup._kill = sticky_kill  # type: ignore[attr-defined]
    sysm.marker = [4242]  # a rank of THIS launch whose tt-run is already gone
    sysm.alive_pids.add(4242)
    sysm.table.append(
        pc.ProcInfo(4242, 1, "PDFABRIC::EngineCore", "PDFABRIC::EngineCore")
    )
    sup.teardown()
    seq = [(pid, sig) for pid, sig in sysm.kills]
    first_launcher = min(i for i, (pid, _) in enumerate(seq) if pid in (tt, tt + 1))
    last_rank_term = max(
        i
        for i, (pid, sig) in enumerate(seq)
        if pid in (tt + 2, tt + 3) and sig == "SIGTERM"
    )
    assert first_launcher > last_rank_term
    launcher_sigs = [sig for pid, sig in seq if pid in (tt, tt + 1)]
    assert launcher_sigs[:2] == ["SIGTERM", "SIGTERM"]
    assert "SIGKILL" in launcher_sigs
    assert (4242, "SIGTERM") in seq  # the orphan is a rank: graceful first
    assert 4242 not in sysm.alive_pids


def test_proc_table_selection_is_pure():
    table = rank_tree(500)
    assert pc.select_rank_pids(table, 500) == [502, 503]
    assert pc.select_launcher_pids(table, 500) == [501]
    assert [p.pid for p in pc.descendants(table, 500)] == [502, 503, 501]
    assert pc.select_rank_pids(table, 999) == []


def test_sweep_stale_rendezvous_removes_dead_writers_only(tmp_path):
    from vllm_tt_plugin.kv_transfer.transport.fabric import RENDEZVOUS_DIR

    sup, sysm, _, _ = make_supervisor(tmp_path)
    sup.s.ctrl_dir = str(tmp_path)  # rendezvous_dir = ctrl_dir/.fabric_rendezvous
    d = tmp_path / RENDEZVOUS_DIR
    d.mkdir()
    (d / "rank0.json").write_text(json.dumps({"pid": 4}))
    (d / "rank1.json").write_text(json.dumps({"pid": 7}))
    (d / "other.txt").write_text("x")
    sysm.alive_pids = {7}
    removed = sup.sweep_stale_rendezvous()
    assert [os.path.basename(p) for p in removed] == ["rank0.json"]
    assert (tmp_path / RENDEZVOUS_DIR / "rank1.json").exists()
    assert (tmp_path / RENDEZVOUS_DIR / "other.txt").exists()


# --------------------------------------------------------------------------- #
# log forwarding
# --------------------------------------------------------------------------- #
def test_forwardable_curates_and_prefixes_and_never_leaks_the_ready_line(tmp_path):
    assert (
        pc.forwardable(
            "[1,0]<stdout>: (MainProcess pid=1) INFO [tt/worker.py:903] Attempting to "
            "open mesh device with grid shape (1, 1)",
            "ranks",
        )
        == "[P] (MainProcess pid=1) INFO [tt/worker.py:903] Attempting to open mesh "
        "device with grid shape (1, 1)"
    )
    assert pc.forwardable(
        "[1,1]<stdout>: INFO [kv_cache_utils.py] GPU KV cache size: 164,352 tokens", "r"
    ).startswith("[D] INFO")
    # per-bucket warm-up chatter and progress bars stay in the files
    assert (
        pc.forwardable("[1,0]<stderr>: Loading weights: 100%|##| 851/851", "r") is None
    )
    assert pc.forwardable("[1,1]<stderr>: ... Decode warmup completed", "r") is None
    assert pc.forwardable("Traceback (most recent call last):", "api-p") == (
        "[api-p] Traceback (most recent call last):"
    )
    # a front-end's own READY line would make the tool declare the boot ready early
    assert pc.forwardable("INFO:     Application startup complete.", "api-p") is None
    assert pc.forwardable("[1,0]<stdout>: some chatter", "ranks") is None
    # the tailer follows appends and keeps an unterminated tail for the next poll
    f = tmp_path / "ranks.log"
    f.write_text("")
    got: list[str] = []
    fw = pc.LogForwarder({"ranks": str(f)}, got.append)
    assert fw.poll() == 0
    with open(f, "a") as fh:
        fh.write("[1,0]<stdout>: KV cache size: 1\n[1,1]<stdout>: KV cache si")
    assert fw.poll() == 1 and got == ["[P] KV cache size: 1"]
    with open(f, "a") as fh:
        fh.write("ze: 2\n")
    assert fw.poll() == 1 and got[-1] == "[D] KV cache size: 2"


def test_launch_modules_stay_free_of_vllm_and_ttnn_at_import_time():
    import subprocess
    import sys

    code = (
        "import sys; import vllm_tt_plugin.kv_transfer.launch.dispatch, "
        "vllm_tt_plugin.kv_transfer.launch.pd_container; "
        "bad=[m for m in ('ttnn','vllm.engine.arg_utils','vllm.v1.engine.core',"
        "'uvicorn','fastapi') if m in sys.modules]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert r.returncode == 0, r.stdout + r.stderr
