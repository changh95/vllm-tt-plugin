# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host-only tests for the Phase 3 launch layer (no MPI, no ttnn, no device).

Covers ``launch.pd_launch_config`` (settings from shell knobs, per-role serve argv and
kv_transfer_config, its round trip into ``transport.fabric.FabricConfig``,
pd_node_args.json round trip, argv validation, rank binding rendering + structural
checks, tt-run command), the pure parts of ``launch.pd_fabric_rank`` (rank/role/port
resolution from the tt-run environment, single-engine forcing, the open_mesh_device
wrapper with injected fakes AND with the real ``fabric_socket`` registry hand-off that
``FabricSocketTransport.start()`` reads) and the pid selection of
``profiles/pd/run_pd_pair_fabric.sh`` (rank processes only, never prterun).

Run: .venv_vllm_server/bin/python -m pytest -q -p no:cacheprovider \
    tests/kv_transfer/test_pd_fabric_launch.py
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from vllm_tt_plugin.kv_transfer.launch import pd_fabric_rank as rank_mod
from vllm_tt_plugin.kv_transfer.launch import pd_launch_config as lc
from vllm_tt_plugin.kv_transfer.transport import fabric_socket, make_transport
from vllm_tt_plugin.kv_transfer.transport.fabric import (
    FabricConfig,
    FabricSocketTransport,
    export_pool_bytes,
    max_export_kv_buffers,
)

PD = "/home/ttuser/experiments/qwen36_27b/profiles/pd"


def base_env(**over: str) -> dict[str, str]:
    env = {
        "MESH_DEVICE": "P150",
        "ARCH_NAME": "blackhole",
        "HF_MODEL": "/w",
        "HF_HUB_OFFLINE": "1",
        "TT_CACHE_PATH": "/cache",
        "QWEN36_FORCE_TP_PATH": "1",
        "QWEN36_SKIP_VISION": "1",
        "QWEN_SDPA_BF8": "1",
        "VLLM_TARGET_DEVICE": "tt",
        "VLLM_RPC_TIMEOUT": "900000",
        "TORCHDYNAMO_DISABLE": "1",
        "TT_PD_STRICT_SHAPES": "1",
        # must NOT leak into global_env
        "TT_MESH_GRAPH_DESC_PATH": "/single/mesh.textproto",
        "TT_VISIBLE_DEVICES": "0",
        "QWEN36_MAX_TOKENS_ALL_USERS": "999",
        "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
        "PATH": "/usr/bin",
    }
    env.update(over)
    return env


def template(chips=(0, 3)) -> dict:
    return {
        "rank_bindings": [
            {
                "rank": 0,
                "mesh_id": 0,
                "env_overrides": {
                    "TT_VISIBLE_DEVICES": str(chips[0]),
                    "TT_METAL_CACHE": "/c/metal_rank0",
                    "PD_ROLE": "prefill",
                    "PD_RPC_PORT": "29551",
                },
            },
            {
                "rank": 1,
                "mesh_id": 1,
                "env_overrides": {
                    "TT_VISIBLE_DEVICES": str(chips[1]),
                    "TT_METAL_CACHE": "/c/metal_rank1",
                    "PD_ROLE": "decode",
                    "PD_RPC_PORT": "29552",
                },
            },
        ],
        "global_env": {
            "MESH_DEVICE": "P150",
            "TT_METAL_OPERATION_TIMEOUT_SECONDS": "600",
        },
        "mesh_graph_desc_path": f"{PD}/pd_two_p150_mesh_graph_descriptor.textproto",
    }


# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #
def test_settings_defaults_match_run_pd_pair_sh():
    s = lc.PairSettings()
    assert (s.p_chip, s.d_chip) == (0, 3)
    assert (s.p_port, s.d_port, s.proxy_port) == (8100, 8200, 8000)
    assert (s.p_rpc, s.d_rpc) == (29551, 29552)
    assert (s.ctx, s.p_pool, s.d_pool, s.d_conc) == (65536, 65536, 163840, 8)
    assert s.max_num_seqs("prefill") == 1 and s.max_num_seqs("decode") == 8
    assert s.weights.endswith("/weights_qwen38")
    assert s.metal_cache("decode").endswith("tt_cache_pd/metal_rank1")
    assert s.d_gdn_fused == s.gdn_fused == 0
    assert s.pkt == 8704 and s.conns == 2 and s.fifo_bytes == 128
    # export pool derived from CTX: one max-length export = 1024 buffers = 2.125 GiB
    assert s.export_slots == 1
    assert s.export_budget == export_pool_bytes(65536) == 2_281_701_376
    assert max_export_kv_buffers(s.ctx) == 1024
    assert lc.PairSettings(export_slots=2).export_budget == 2 * export_pool_bytes(65536)
    assert lc.PairSettings(ctx=8192).export_budget == export_pool_bytes(8192)
    # an explicit budget is kept when it holds one export, refused otherwise
    assert lc.PairSettings(export_budget=3 << 30).export_budget == 3 << 30
    with pytest.raises(ValueError, match="cannot hold one export"):
        lc.PairSettings(export_budget=2147483648)  # the old default: 963 < 1024
    assert lc.PairSettings.from_env({"EXPORT_SLOTS": "2"}).export_slots == 2


def test_settings_from_env_and_pairs():
    s = lc.PairSettings.from_env(
        {
            "PAIR": "test",
            "D_CONC": "4",
            "PKT": "15232",
            "GDN_FUSED": "2",
            "P_EXTRA_ARGS": "--foo 'a b'",
            "TAG": "x",
        }
    )
    assert (s.p_chip, s.d_chip) == (1, 2)
    assert s.d_conc == 4 and s.pkt == 15232 and s.d_gdn_fused == 2 and s.tag == "x"
    assert s.p_extra_args == ["--foo", "a b"]
    for name, chips in lc.PAIRS.items():
        ps = lc.PairSettings(pair=name)
        assert (ps.chip("prefill"), ps.chip("decode")) == chips


@pytest.mark.parametrize(
    "kw",
    [
        {"pair": "nope"},
        {"p_chip": 0, "d_chip": 1},  # on-package TRACE pair
        {"p_chip": 2, "d_chip": 3},
        {"p_chip": 3, "d_chip": 3},
        {"p_rpc": 1, "d_rpc": 1},
        {"d_port": 8100},
        {"conns": 3},
        {"shm_mode": "raw"},
        {"export_slots": 0},
    ],
)
def test_settings_rejects(kw):
    with pytest.raises(ValueError):
        lc.PairSettings(**kw)


# --------------------------------------------------------------------------- #
# argv / configs
# --------------------------------------------------------------------------- #
def test_serve_argv_per_role():
    s = lc.PairSettings(pair="test")
    p, d = lc.serve_argv(s, "prefill"), lc.serve_argv(s, "decode")
    assert p[0] == s.weights and d[0] == s.weights
    for argv, port, rpc, seqs in ((p, 8100, 29551, 1), (d, 8200, 29552, 8)):
        assert lc.argv_value(argv, "--port") == str(port)
        assert lc.argv_value(argv, "--data-parallel-rpc-port") == str(rpc)
        assert lc.argv_value(argv, "--data-parallel-size") == "1"
        assert lc.argv_value(argv, "--data-parallel-size-local") == "0"
        assert lc.argv_value(argv, "--data-parallel-address") == "127.0.0.1"
        assert lc.argv_value(argv, "--max-num-seqs") == str(seqs)
        assert lc.argv_value(argv, "--data-parallel-rank") is None
        assert "--headless" not in argv
        tt = lc.argv_json(argv, "--additional-config")["tt"]
        assert tt["fabric_config"] == "FABRIC_2D"
        assert tt["fabric_max_packet_payload_bytes"] == 8704
        assert (
            tt["trace_region_size"] == s.trace_region and tt["l1_small_size"] == 24576
        )
    # additional-config identical on both nodes (fingerprint / descriptor parity)
    assert lc.argv_value(p, "--additional-config") == lc.argv_value(
        d, "--additional-config"
    )
    kp, kd = (
        lc.argv_json(p, "--kv-transfer-config"),
        lc.argv_json(d, "--kv-transfer-config"),
    )
    assert (kp["kv_role"], kd["kv_role"]) == ("kv_producer", "kv_consumer")
    assert (kp["engine_id"], kd["engine_id"]) == ("p-p3", "d-p3")  # p-{tag}/d-{tag}
    assert (
        kp["kv_connector_module_path"]
        == kd["kv_connector_module_path"]
        == lc.CONNECTOR_MODULE
    )
    assert (
        "kv_load_failure_policy" not in kp
        and kd["kv_load_failure_policy"] == "recompute"
    )
    for k in (
        "transport",
        "shm_mode",
        "shm_dir",
        "fabric_control_dir",
        "fabric_fifo_bytes",
        "kv_lease_duration",
        "xfer_chunk_tokens",
        "socket_connections",
        "fabric_rec_sets",
        "fabric_export_budget_bytes",
        "fabric_export_slots",
        "fabric_max_model_len",
        "fabric_max_packet_payload_bytes",
        "hybrid_state",
    ):
        assert (
            kp["kv_connector_extra_config"][k] == kd["kv_connector_extra_config"][k]
        ), k
    assert kp["kv_connector_extra_config"]["transport"] == "fabric"
    assert kp["kv_connector_extra_config"]["shm_mode"] == "dumpfile"
    assert kp["kv_connector_extra_config"]["shm_dir"] == "/dev/shm/tt_pd_fabric_p3"
    assert kp["kv_connector_extra_config"]["fabric_control_dir"] == s.ctrl_dir
    assert kd["kv_connector_extra_config"]["max_inflight_loads"] == 2
    assert "max_inflight_loads" not in kp["kv_connector_extra_config"]
    # every argv token is a plain string usable by both `vllm serve` and the rank replay
    assert all(isinstance(a, str) and a for a in p + d)


def test_kv_transfer_config_round_trips_into_fabric_config(monkeypatch):
    """Every fabric knob the launch lane emits is a key FabricConfig reads, and the
    values land on the transport unchanged (serve_argv -> --kv-transfer-config JSON
    -> make_transport -> FabricConfig), with no env fallback doing the work."""
    for var in (
        "TT_PD_FABRIC_DIR",
        "TT_PD_FABRIC_CONNECTIONS",
        "TT_PD_FABRIC_PKT",
        "TT_PD_FABRIC_EXPORT_BUDGET",
        "TT_PD_FABRIC_EXPORT_SLOTS",
        "TT_PD_FABRIC_MAX_MODEL_LEN",
        "TT_PD_FABRIC_REC_SETS",
    ):
        monkeypatch.delenv(var, raising=False)
    s = lc.PairSettings(
        pair="test",
        ctrl_dir="/dev/shm/unit_ctrl",
        fifo_bytes=256,
        conns=1,
        rec_sets=3,
        pkt=15232,
        lease=7.5,
        ctx=32768,
        export_slots=2,
    )
    for role in lc.ROLES:
        ktc = lc.argv_json(lc.serve_argv(s, role), "--kv-transfer-config")
        extra = ktc["kv_connector_extra_config"]
        for k in extra:
            if k.startswith(("fabric_", "socket_")):
                assert k in FabricConfig._EXTRA_KEYS, f"{k} is not read by FabricConfig"
        cfg = SimpleNamespace(
            engine_id=ktc["engine_id"],
            kv_role=ktc["kv_role"],
            kv_connector_extra_config=extra,
        )
        t = make_transport("fabric", None, cfg, None)
        assert isinstance(t, FabricSocketTransport)
        assert t.role == ("producer" if role == "prefill" else "consumer")
        assert t.cfg.control_dir == "/dev/shm/unit_ctrl"
        assert (t.cfg.fifo_bytes, t.cfg.socket_connections) == (256, 1)
        assert (t.cfg.rec_sets, t.cfg.packet_bytes, t.lease_duration) == (
            3,
            15232,
            7.5,
        )
        assert (t.cfg.max_model_len, t.cfg.export_slots) == (32768, 2)
        assert (
            t.cfg.export_budget_bytes == s.export_budget == 2 * export_pool_bytes(32768)
        )
        assert t.cfg.max_export_kv_buffers == 512
        assert t.cfg.export_pool_buffers == 1024


def test_transport_shm_emits_phase2_config_and_builds_shm_transport(tmp_path):
    """TRANSPORT=shm: the SAME headless-rank launch shape carries run_pd_pair.sh's
    Phase 2 ShmTransport config (dumpfile, shm_dir, budget on P only).  By default
    the mesh opens as run_pd_pair.sh's does (NO tt.fabric_config, wrapper off);
    SHM_FABRIC=1 is the like-for-like mode that sets FABRIC_2D like the fabric pair."""
    s = lc.PairSettings.from_env({"TRANSPORT": "shm", "SHM_DIR": str(tmp_path / "shm")})
    assert s.transport == "shm" and not s.fabric_enabled
    assert s.segment_dir == str(tmp_path / "shm")
    for role in lc.ROLES:
        argv = lc.serve_argv(s, role)
        ktc = lc.argv_json(argv, "--kv-transfer-config")
        extra = ktc["kv_connector_extra_config"]
        assert extra["transport"] == "shm" and extra["shm_mode"] == "dumpfile"
        assert extra["shm_dir"] == str(tmp_path / "shm")
        fabric_keys = [k for k in extra if k.startswith(("fabric_", "socket_"))]
        assert not fabric_keys
        assert ("shm_budget_bytes" in extra) == (role == "prefill")
        if role == "prefill":
            assert extra["shm_budget_bytes"] == 8589934592
        tt = lc.argv_json(argv, "--additional-config")["tt"]
        assert "fabric_config" not in tt and tt["l1_small_size"] == 24576
        summ = lc.validate_argv_for_role(argv, role, s.rpc_port(role))
        assert summ["transport"] == "shm" and summ["fabric_enabled"] is False
        assert summ["segment_dir"] == str(tmp_path / "shm")
        cfg = SimpleNamespace(
            engine_id=ktc["engine_id"],
            kv_role=ktc["kv_role"],
            kv_connector_extra_config=extra,
        )
        t = make_transport(extra["transport"], extra["shm_mode"], cfg, ktc["kv_role"])
        assert t.KIND == "shm" and t.engine_id == ktc["engine_id"]
    meta = lc.node_args(s)["meta"]
    assert meta["transport"] == "shm" and meta["fabric_enabled"] is False
    assert meta["segment_dir"] == str(tmp_path / "shm")
    # like-for-like comparison mode: FABRIC_2D exactly as the fabric pair sets it
    lf = lc.PairSettings.from_env({"TRANSPORT": "shm", "SHM_FABRIC": "1"})
    assert lf.fabric_enabled
    tt_lf = lc.additional_config(lf)["tt"]
    tt_fab = lc.additional_config(lc.PairSettings())["tt"]
    assert tt_lf == tt_fab and tt_lf["fabric_config"] == "FABRIC_2D"
    summ = lc.validate_argv_for_role(lc.serve_argv(lf, "prefill"), "prefill", lf.p_rpc)
    assert summ["fabric_enabled"] is True and summ["fabric_config"] == "FABRIC_2D"
    with pytest.raises(ValueError, match="TRANSPORT"):
        lc.PairSettings(transport="tcp")
    with pytest.raises(ValueError, match="SHM_FABRIC"):
        lc.PairSettings(transport="shm", shm_fabric=2)
    assert lc.PairSettings().transport == "fabric"


def test_extra_args_appended_per_role():
    s = lc.PairSettings(p_extra_args=["--x", "1"], d_extra_args=["--y"])
    assert lc.serve_argv(s, "prefill")[-2:] == ["--x", "1"]
    assert lc.serve_argv(s, "decode")[-1:] == ["--y"]
    assert "--y" not in lc.serve_argv(s, "prefill")


def test_node_args_roundtrip(tmp_path):
    s = lc.PairSettings(pair="test", tag="t")
    path = str(tmp_path / "pd_node_args_t.json")
    d = lc.write_node_args(path, s)
    back = lc.load_node_args(path)
    assert back == d
    assert back["prefill"] == lc.serve_argv(s, "prefill")
    assert back["decode"] == lc.serve_argv(s, "decode")
    assert back["meta"]["chips"] == {"prefill": 1, "decode": 2}
    assert back["meta"]["rpc_ports"] == {"prefill": 29551, "decode": 29552}
    assert back["meta"]["fabric"]["packet_bytes"] == 8704
    assert not os.path.exists(path + ".tmp")
    (tmp_path / "bad.json").write_text(json.dumps({"prefill": ["a"]}))
    with pytest.raises(ValueError):
        lc.load_node_args(str(tmp_path / "bad.json"))


def test_validate_argv_for_role_accepts_own_argv():
    s = lc.PairSettings()
    for role in lc.ROLES:
        summ = lc.validate_argv_for_role(lc.serve_argv(s, role), role, s.rpc_port(role))
        assert summ["kv_role"] == lc.KV_ROLE_OF_ROLE[role]
        assert summ["fabric_config"] == "FABRIC_2D" and summ["packet_bytes"] == 8704
        assert summ["rpc_port"] == s.rpc_port(role)
        assert summ["model"] == s.weights


def _swap(argv, flag, value):
    out = list(argv)
    i = out.index(flag)
    out[i + 1] = value
    return out


def test_validate_argv_for_role_rejects():
    s = lc.PairSettings()
    p = lc.serve_argv(s, "prefill")
    with pytest.raises(ValueError, match="PD_RPC_PORT"):
        lc.validate_argv_for_role(p, "prefill", 29552)
    with pytest.raises(ValueError, match="kv_role"):
        lc.validate_argv_for_role(p, "decode", 29552)
    with pytest.raises(ValueError, match="size-local"):
        lc.validate_argv_for_role(
            _swap(p, "--data-parallel-size-local", "1"), "prefill", 29551
        )
    with pytest.raises(ValueError, match="data-parallel-rank"):
        lc.validate_argv_for_role(p + ["--data-parallel-rank", "0"], "prefill", 29551)
    ktc = lc.argv_json(p, "--kv-transfer-config")
    ktc["kv_connector_extra_config"]["transport"] = "tcp"
    with pytest.raises(ValueError, match="transport"):
        lc.validate_argv_for_role(
            _swap(p, "--kv-transfer-config", json.dumps(ktc)), "prefill", 29551
        )
    # the shm data plane in the same process shape is accepted (TRANSPORT=shm)
    ktc["kv_connector_extra_config"]["transport"] = "shm"
    summ = lc.validate_argv_for_role(
        _swap(p, "--kv-transfer-config", json.dumps(ktc)), "prefill", 29551
    )
    assert summ["transport"] == "shm"
    ac = lc.argv_json(p, "--additional-config")
    del ac["tt"]["fabric_config"]
    with pytest.raises(ValueError, match="fabric_config"):
        lc.validate_argv_for_role(
            _swap(p, "--additional-config", json.dumps(ac)), "prefill", 29551
        )
    with pytest.raises(ValueError, match="headless"):
        lc.validate_argv_for_role(p + ["--headless"], "prefill", 29551)


def test_argv_value_forms():
    assert lc.argv_value(["--a", "1", "--a=2"], "--a") == "2"
    assert lc.argv_value(["--ab", "1"], "--a") is None
    assert lc.argv_value([], "--a") is None


# --------------------------------------------------------------------------- #
# rank binding
# --------------------------------------------------------------------------- #
def test_render_rank_binding_prod_and_test():
    for pair, chips in (("prod", ("0", "3")), ("test", ("1", "2"))):
        s = lc.PairSettings(pair=pair, d_gdn_fused=2, allow_fused_conv=1)
        env = base_env(EXTRA_UNRELATED="x")
        rb = lc.render_rank_binding(
            template(tuple(int(c) for c in chips)),
            s,
            "/pd/pd_node_args_p3.json",
            parent_env=env,
        )
        b0, b1 = rb["rank_bindings"]
        assert (b0["rank"], b0["mesh_id"], b1["rank"], b1["mesh_id"]) == (0, 0, 1, 1)
        assert (
            b0["env_overrides"]["TT_VISIBLE_DEVICES"],
            b1["env_overrides"]["TT_VISIBLE_DEVICES"],
        ) == chips
        assert (b0["env_overrides"]["PD_ROLE"], b1["env_overrides"]["PD_ROLE"]) == (
            "prefill",
            "decode",
        )
        assert (
            b0["env_overrides"]["PD_RPC_PORT"],
            b1["env_overrides"]["PD_RPC_PORT"],
        ) == ("29551", "29552")
        assert (
            b0["env_overrides"]["TT_METAL_CACHE"]
            != b1["env_overrides"]["TT_METAL_CACHE"]
        )
        assert b0["env_overrides"]["QWEN36_MAX_TOKENS_ALL_USERS"] == "65536"
        assert b1["env_overrides"]["QWEN36_MAX_TOKENS_ALL_USERS"] == "163840"
        assert b0["env_overrides"]["QWEN36_GDN_DECODE_FUSED"] == "0"
        assert b1["env_overrides"]["QWEN36_GDN_DECODE_FUSED"] == "2"
        assert b1["env_overrides"]["TT_PD_ALLOW_FUSED_CONV"] == "1"
        assert "TT_PD_ALLOW_FUSED_CONV" not in b0["env_overrides"]
        g = rb["global_env"]
        assert g["PD_ARGS_JSON"] == "/pd/pd_node_args_p3.json"
        assert g["TT_METAL_OPERATION_TIMEOUT_SECONDS"] == "600"
        assert g["PYTHONUNBUFFERED"] == "1"
        assert g["TT_PD_FABRIC_DIR"] == s.ctrl_dir
        assert (
            g["PD_FABRIC_PKT"] == "8704" and g["PD_FABRIC_RELIABILITY"] == "STRICT_INIT"
        )
        for k in (
            "VLLM_TARGET_DEVICE",
            "HF_HUB_OFFLINE",
            "TT_CACHE_PATH",
            "QWEN36_FORCE_TP_PATH",
            "QWEN_SDPA_BF8",
            "TORCHDYNAMO_DISABLE",
            "VLLM_RPC_TIMEOUT",
            "TT_PD_STRICT_SHAPES",
            "ARCH_NAME",
            "MESH_DEVICE",
        ):
            assert k in g, k
        for k in (
            "TT_MESH_GRAPH_DESC_PATH",
            "TT_VISIBLE_DEVICES",
            "QWEN36_MAX_TOKENS_ALL_USERS",
            "VLLM_ENGINE_READY_TIMEOUT_S",
            "PATH",
            "EXTRA_UNRELATED",
            "TT_MESH_ID",
            "PD_ROLE",
        ):
            assert k not in g, k
        assert rb["mesh_graph_desc_path"].endswith(
            "pd_two_p150_mesh_graph_descriptor.textproto"
        )
        assert all(
            isinstance(v, str)
            for b in rb["rank_bindings"]
            for v in b["env_overrides"].values()
        )
        assert all(isinstance(v, str) for v in g.values())


def test_render_overrides_template_ports_and_chips():
    s = lc.PairSettings(pair="prod", p_rpc=31000, d_rpc=31001, p_chip=3, d_chip=0)
    rb = lc.render_rank_binding(
        template((0, 3)), s, "/pd/na.json", parent_env=base_env()
    )
    ov0, ov1 = (b["env_overrides"] for b in rb["rank_bindings"])
    assert (ov0["PD_RPC_PORT"], ov1["PD_RPC_PORT"]) == ("31000", "31001")
    assert (ov0["TT_VISIBLE_DEVICES"], ov1["TT_VISIBLE_DEVICES"]) == ("3", "0")


def test_render_rejects_blocklisted_extra_global_env():
    s = lc.PairSettings()
    with pytest.raises(ValueError):
        lc.render_rank_binding(
            template(),
            s,
            "/na.json",
            parent_env=base_env(),
            extra_global_env={"TT_VISIBLE_DEVICES": "1"},
        )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda d: d["rank_bindings"].pop(), "exactly 2"),
        (lambda d: d["rank_bindings"][1].__setitem__("mesh_id", 0), "mesh_ids"),
        (
            lambda d: d["rank_bindings"][0]["env_overrides"].__setitem__(
                "PD_ROLE", "decode"
            ),
            "role",
        ),
        (
            lambda d: d["rank_bindings"][0]["env_overrides"].__setitem__(
                "TT_VISIBLE_DEVICES", "3"
            ),
            "same chip",
        ),
        (
            lambda d: d["rank_bindings"][0]["env_overrides"].__setitem__(
                "TT_VISIBLE_DEVICES", "0,3"
            ),
            "one chip",
        ),
        (
            lambda d: d["rank_bindings"][0]["env_overrides"].__setitem__(
                "PD_RPC_PORT", "29552"
            ),
            "PD_RPC_PORT",
        ),
        (
            lambda d: d["rank_bindings"][0]["env_overrides"].__setitem__(
                "TT_METAL_CACHE",
                d["rank_bindings"][1]["env_overrides"]["TT_METAL_CACHE"],
            ),
            "TT_METAL_CACHE",
        ),
        (
            lambda d: d["global_env"].__setitem__("TT_MESH_GRAPH_DESC_PATH", "/x"),
            "managed",
        ),
        (lambda d: d["global_env"].pop("PD_ARGS_JSON"), "missing"),
        (lambda d: d.__setitem__("mesh_graph_desc_path", ""), "mesh_graph_desc_path"),
    ],
)
def test_check_rank_binding_rejects(mutate, match):
    s = lc.PairSettings()
    rb = lc.render_rank_binding(template(), s, "/na.json", parent_env=base_env())
    mutate(rb)
    with pytest.raises(ValueError, match=match):
        lc.check_rank_binding(rb)


def test_checked_in_templates_render(tmp_path):
    """The two committed templates render for their pair and dump as valid YAML."""
    for pair, fname in (
        ("prod", "pd_rank_binding_chips03.yaml"),
        ("test", "pd_rank_binding_chips12.yaml"),
    ):
        path = os.path.join(PD, fname)
        if not os.path.exists(path):
            pytest.skip(f"{path} missing")
        s = lc.PairSettings(pair=pair)
        tmpl = lc.load_yaml(path)
        rb = lc.render_rank_binding(tmpl, s, "/na.json", parent_env=base_env())
        chips = tuple(
            b["env_overrides"]["TT_VISIBLE_DEVICES"] for b in rb["rank_bindings"]
        )
        assert chips == tuple(str(c) for c in lc.PAIRS[pair])
        # the template's own overrides agree with the settings (the renderer would
        # silently override them otherwise)
        for b in tmpl["rank_bindings"]:
            role = lc.ROLE_OF_RANK[int(b["rank"])]
            assert int(b["env_overrides"]["TT_VISIBLE_DEVICES"]) == s.chip(role)
            assert int(b["env_overrides"]["PD_RPC_PORT"]) == s.rpc_port(role)
        assert os.path.exists(tmpl["mesh_graph_desc_path"])
        out = str(tmp_path / f"{pair}.yaml")
        lc.dump_yaml(rb, out)
        assert lc.load_yaml(out) == rb
        assert lc.TEMPLATE_OF_PAIR[pair] == fname


def test_ttrun_and_frontend_commands():
    cmd = lc.ttrun_command("/v/python", "/rb.yaml", "/na.json")
    assert cmd[:7] == [
        "/v/python",
        "-m",
        "ttnn.distributed.ttrun",
        "--bare",
        "--rank-binding",
        "/rb.yaml",
        "--",  # without it click takes the program's -m for --mesh-graph-descriptor
    ]
    assert cmd[7:] == ["/v/python", "-m", lc.RANK_MODULE, "--pd-args", "/na.json"]
    s = lc.PairSettings()
    fe = lc.frontend_command("/v/vllm", s, "decode")
    assert fe[:2] == ["/v/vllm", "serve"] and fe[2:] == lc.serve_argv(s, "decode")


def test_cli_print_argv0_and_render(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PAIR", "test")
    monkeypatch.setenv("TAG", "unit")
    for k, v in base_env().items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("R", "/home/ttuser/experiments/qwen36_27b")
    assert lc._main(["--print-argv0", "decode"]) == 0
    out = capsys.readouterr().out
    argv = out.split("\0")[:-1]
    assert argv == lc.serve_argv(lc.PairSettings.from_env(dict(os.environ)), "decode")
    tmpl = tmp_path / "tmpl.yaml"
    lc.dump_yaml(template((1, 2)), str(tmpl))
    assert lc._main(["--out-dir", str(tmp_path), "--template", str(tmpl)]) == 0
    res = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert res["chips"] == "1,2"
    assert os.path.exists(res["node_args"]) and os.path.exists(res["rank_binding"])
    rb = lc.load_yaml(res["rank_binding"])
    lc.check_rank_binding(rb, lc.PairSettings.from_env(dict(os.environ)))
    assert rb["global_env"]["PD_ARGS_JSON"] == res["node_args"]


# --------------------------------------------------------------------------- #
# pd_fabric_rank: pure parts
# --------------------------------------------------------------------------- #
def mpi_env(
    rank: int, role: str | None = None, size: int = 2, **over: str
) -> dict[str, str]:
    env = {
        "OMPI_COMM_WORLD_RANK": str(rank),
        "OMPI_COMM_WORLD_SIZE": str(size),
        "TT_RUN_RANK": str(rank),
        "TT_MESH_ID": str(rank),
        "TT_VISIBLE_DEVICES": "0" if rank == 0 else "3",
        "TT_METAL_CACHE": f"/c/metal_rank{rank}",
        "PD_RPC_PORT": "29551" if rank == 0 else "29552",
    }
    if role is not None:
        env["PD_ROLE"] = role
    env.update(over)
    return env


def test_resolve_rank_env_both_ranks():
    c0 = rank_mod.resolve_rank_env(mpi_env(0, "prefill"))
    c1 = rank_mod.resolve_rank_env(mpi_env(1, "decode"))
    assert (c0.role, c0.rank, c0.rpc_port, c0.mesh_id, c0.kv_role) == (
        "prefill",
        0,
        29551,
        0,
        "kv_producer",
    )
    assert (c1.role, c1.rank, c1.rpc_port, c1.mesh_id, c1.kv_role) == (
        "decode",
        1,
        29552,
        1,
        "kv_consumer",
    )
    assert c0.under_mpi and c1.under_mpi
    assert c1.visible_devices == "3" and c1.metal_cache == "/c/metal_rank1"
    # role defaults from the rank table when PD_ROLE is absent
    assert rank_mod.resolve_rank_env(mpi_env(1)).role == "decode"
    # TT_RUN_RANK alone (no OMPI vars) is not MPI evidence without a world size
    env = mpi_env(0, "prefill")
    del env["OMPI_COMM_WORLD_RANK"], env["OMPI_COMM_WORLD_SIZE"]
    with pytest.raises(RuntimeError, match="tt-run/MPI"):
        rank_mod.resolve_rank_env(env)


@pytest.mark.parametrize(
    "env,match",
    [
        (mpi_env(0, "decode"), "role table"),
        (mpi_env(1, "prefill"), "role table"),
        (mpi_env(0, "prefill", size=1), "exactly 2"),
        (mpi_env(0, "prefill", size=3), "exactly 2"),
        (mpi_env(2, size=3), "exactly 2"),
        (mpi_env(0, "prefill", TT_RUN_RANK="1"), "TT_RUN_RANK"),
        (mpi_env(0, "prefill", TT_MESH_ID="1"), "TT_MESH_ID"),
        (mpi_env(0, "both"), "PD_ROLE"),
        (
            {k: v for k, v in mpi_env(0, "prefill").items() if k != "PD_RPC_PORT"},
            "PD_RPC_PORT",
        ),
        (mpi_env(0, "prefill", PD_RPC_PORT="80"), "out of range"),
    ],
)
def test_resolve_rank_env_rejects(env, match):
    with pytest.raises(RuntimeError, match=match):
        rank_mod.resolve_rank_env(env)


def test_resolve_rank_env_dry_run_without_mpi():
    with pytest.raises(RuntimeError):
        rank_mod.resolve_rank_env({"PD_ROLE": "decode", "PD_RPC_PORT": "29552"})
    c = rank_mod.resolve_rank_env(
        {"PD_ROLE": "decode", "PD_RPC_PORT": "29552"}, allow_no_mpi=True
    )
    assert (c.role, c.rank, c.world_size, c.under_mpi) == ("decode", 1, 2, False)
    c = rank_mod.resolve_rank_env(
        {}, role_override="prefill", rpc_override=29551, allow_no_mpi=True
    )
    assert (c.role, c.rank, c.rpc_port) == ("prefill", 0, 29551)
    with pytest.raises(RuntimeError, match="PD_ROLE"):
        rank_mod.resolve_rank_env({"PD_RPC_PORT": "29551"}, allow_no_mpi=True)


def test_force_single_engine():
    pc = SimpleNamespace(
        data_parallel_size=1,
        data_parallel_size_local=0,
        data_parallel_rank=0,
        data_parallel_rank_local=None,
        distributed_executor_backend=None,
    )
    rank_mod.force_single_engine(pc)
    assert (
        pc.data_parallel_size,
        pc.data_parallel_size_local,
        pc.data_parallel_rank,
        pc.data_parallel_rank_local,
    ) == (1, 1, 0, 0)
    assert pc.distributed_executor_backend == "uni"
    pc.distributed_executor_backend = "mp"
    with pytest.raises(RuntimeError, match="uni"):
        rank_mod.force_single_engine(pc)


def test_prepare_reads_node_args_and_fabric_knobs(tmp_path, monkeypatch):
    s = lc.PairSettings(pair="test", pkt=15232, reliability="RELAXED_INIT")
    path = str(tmp_path / "na.json")
    lc.write_node_args(path, s)
    args = rank_mod.parse_args(
        ["--pd-args", path, "--dry-run", "--role", "decode", "--rpc-port", "29552"]
    )
    ctx, argv, summary = rank_mod.prepare(args, env={})
    assert ctx.role == "decode" and argv == lc.serve_argv(s, "decode")
    assert summary["engine_id"] == s.d_engine_id == "d-p3"
    assert summary["segment_dir"] == s.ctrl_dir and summary["fabric_enabled"]
    assert args.no_fabric_wrapper is False  # fabric data plane: wrapper stays on
    # tt config in the argv wins over CLI/env defaults
    assert (args.packet_bytes, args.reliability, args.fabric_config) == (
        15232,
        "RELAXED_INIT",
        "FABRIC_2D",
    )
    # wrong port for the role -> refused before vllm is imported
    args = rank_mod.parse_args(
        ["--pd-args", path, "--dry-run", "--role", "decode", "--rpc-port", "29551"]
    )
    with pytest.raises(ValueError, match="PD_RPC_PORT"):
        rank_mod.prepare(args, env={})
    # under fake MPI the role comes from the env
    args = rank_mod.parse_args(["--pd-args", path])
    ctx, argv, _ = rank_mod.prepare(args, env=mpi_env(0, "prefill"))
    assert (
        ctx.role == "prefill"
        and argv[lc.serve_argv(s, "prefill").index("--port") + 1] == "8100"
    )
    with pytest.raises(RuntimeError, match="pd-args"):
        rank_mod.prepare(
            rank_mod.parse_args(["--dry-run", "--role", "prefill"]),
            env={"PD_ARGS_JSON": ""},
        )


def test_install_mesh_hooks_sets_fabric_before_open_and_registers_mesh(monkeypatch):
    calls: list[tuple] = []
    fake_worker = SimpleNamespace()
    the_mesh = object()

    def orig_open(tt_config, trace_mode, local_dp_rank=0):
        calls.append(("open", dict(tt_config), trace_mode, local_dp_rank))
        return the_mesh

    fake_worker.open_mesh_device = orig_open
    fake_worker.close_mesh_device = lambda mesh, tt_config: calls.append(("close",))
    monkeypatch.setattr(rank_mod, "_FABRIC_SET", False)
    wrapped = rank_mod.install_mesh_hooks(
        fabric_config="FABRIC_2D",
        reliability="STRICT_INIT",
        packet_bytes=8704,
        expected_rank=1,
        set_fabric=lambda fc, rel, pkt: calls.append(("fabric", fc, rel, pkt)),
        register_mesh=lambda m: calls.append(("register", m)),
        distributed_info=lambda: {"initialized": True, "rank": 1, "size": 2},
        worker_module=fake_worker,
    )
    assert fake_worker.open_mesh_device is wrapped and wrapped._pd_fabric_wrapped
    tt_config = {
        "trace_region_size": 7,
        "fabric_config": "FABRIC_2D",
        "fabric_reliability_mode": "RELAXED_INIT",
        "fabric_max_packet_payload_bytes": 15232,
    }
    mesh = fake_worker.open_mesh_device(tt_config, "all", 0)
    assert mesh is the_mesh
    # fabric set BEFORE the original open, with the tt-config values, and the fabric
    # keys stripped from what the original sees (so a plugin honouring fabric_config
    # at num_devices==1 does not set it twice)
    assert calls[0] == ("fabric", "FABRIC_2D", "RELAXED_INIT", 15232)
    assert (
        calls[1][0] == "open"
        and calls[1][1] == {"trace_region_size": 7}
        and calls[1][2:] == ("all", 0)
    )
    assert calls[2] == ("register", the_mesh)
    # idempotent: a second install returns the same wrapper; fabric set once per process
    assert (
        rank_mod.install_mesh_hooks(
            worker_module=fake_worker, set_fabric=lambda *a: calls.append(("fabric2",))
        )
        is wrapped
    )
    fake_worker.open_mesh_device({}, "all")
    assert [c[0] for c in calls].count("fabric") == 1


def test_install_mesh_hooks_rejects_wrong_distributed_context(monkeypatch):
    fake_worker = SimpleNamespace(
        open_mesh_device=lambda tt_config, trace_mode, local_dp_rank=0: object(),
        close_mesh_device=lambda mesh, tt_config: None,
    )
    monkeypatch.setattr(rank_mod, "_FABRIC_SET", False)
    rank_mod.install_mesh_hooks(
        expected_rank=0,
        set_fabric=lambda *a: None,
        register_mesh=lambda m: None,
        distributed_info=lambda: {"initialized": True, "rank": 1, "size": 2},
        worker_module=fake_worker,
    )
    with pytest.raises(RuntimeError, match="rank"):
        fake_worker.open_mesh_device({}, "all")
    fake_worker2 = SimpleNamespace(
        open_mesh_device=lambda tt_config, trace_mode, local_dp_rank=0: object(),
        close_mesh_device=lambda mesh, tt_config: None,
    )
    monkeypatch.setattr(rank_mod, "_FABRIC_SET", False)
    rank_mod.install_mesh_hooks(
        expected_rank=0,
        set_fabric=lambda *a: None,
        register_mesh=lambda m: None,
        distributed_info=lambda: {"initialized": False},
        worker_module=fake_worker2,
    )
    with pytest.raises(RuntimeError, match="not initialized"):
        fake_worker2.open_mesh_device({}, "all")


def test_install_mesh_hooks_no_fabric_wrapper(monkeypatch):
    calls: list[str] = []
    fake_worker = SimpleNamespace(
        open_mesh_device=lambda tt_config, trace_mode, local_dp_rank=0: calls.append(
            "open"
        )
        or object(),
        close_mesh_device=lambda mesh, tt_config: None,
    )
    monkeypatch.setattr(rank_mod, "_FABRIC_SET", False)
    rank_mod.install_mesh_hooks(
        enable_fabric=False,
        set_fabric=lambda *a: calls.append("fabric"),
        register_mesh=lambda m: calls.append("register"),
        distributed_info=lambda: {"initialized": True, "rank": 0, "size": 2},
        worker_module=fake_worker,
    )
    fake_worker.open_mesh_device({"fabric_config": "FABRIC_2D"}, "all")
    assert calls == ["open", "register"]


class _NotUnderMpiLayer:
    """Minimal SocketLayer stub: start() passes the mesh check, then stops at the
    distributed-context check (so no barrier / socket is attempted)."""

    def is_distributed(self):
        return False


@pytest.fixture
def clean_mesh_registry():
    fabric_socket.clear_registered_mesh_device()
    yield
    fabric_socket.clear_registered_mesh_device()


def test_default_register_mesh_populates_fabric_socket_registry(clean_mesh_registry):
    """The rank's hand-off must land in the ONE registry FabricSocketTransport.start()
    reads: fabric_socket.registered_mesh_device()."""
    m = object()
    assert fabric_socket.registered_mesh_device() is None
    rank_mod._default_register_mesh(m)
    assert fabric_socket.registered_mesh_device() is m
    assert not hasattr(rank_mod, "MESH_DEVICE")  # the dead fallback is gone


def test_install_mesh_hooks_default_register_reaches_transport(
    monkeypatch, clean_mesh_registry, tmp_path
):
    """Wrapper with the DEFAULT register callable (no injection): opening the mesh
    populates the registry, and a transport built without mesh_device= gets past
    start()'s mesh check on it."""
    the_mesh = object()
    fake_worker = SimpleNamespace(
        open_mesh_device=lambda tt_config, trace_mode, local_dp_rank=0: the_mesh,
        close_mesh_device=lambda mesh, tt_config: None,
    )
    monkeypatch.setattr(rank_mod, "_FABRIC_SET", False)
    rank_mod.install_mesh_hooks(
        expected_rank=0,
        set_fabric=lambda *a: None,
        distributed_info=lambda: {"initialized": True, "rank": 0, "size": 2},
        worker_module=fake_worker,
    )
    t = FabricSocketTransport(
        engine_id="p0",
        role="producer",
        socket_layer=_NotUnderMpiLayer(),
        control_dir=str(tmp_path / "ctrl"),
        janitor_period=0,
    )
    with pytest.raises(RuntimeError, match="no mesh device"):
        t.start()  # before the wrapper ran: the registry is empty
    assert fake_worker.open_mesh_device({}, "all") is the_mesh
    assert fabric_socket.registered_mesh_device() is the_mesh
    with pytest.raises(RuntimeError, match="MPI ranks"):
        t.start()  # past the mesh check (fails at the next one: no MPI in a test)


def test_default_set_fabric_calls_fabric_socket(monkeypatch):
    calls = []
    monkeypatch.setattr(
        fabric_socket,
        "set_fabric_config_for_pd",
        lambda packet_bytes, reliability, fabric_config: calls.append(
            (packet_bytes, reliability, fabric_config)
        ),
    )
    rank_mod._default_set_fabric("FABRIC_2D", "RELAXED_INIT", 15232)
    assert calls == [(15232, "RELAXED_INIT", "FABRIC_2D")]


def test_fabric_socket_set_fabric_signature_matches_wrapper():
    """The real adapter takes exactly what the rank's wrapper passes."""
    params = inspect.signature(fabric_socket.set_fabric_config_for_pd).parameters
    assert list(params) == ["packet_bytes", "reliability", "fabric_config"]
    assert params["fabric_config"].default == "FABRIC_2D"
    assert params["reliability"].default == "STRICT_INIT"


# --------------------------------------------------------------------------- #
# run_pd_pair_fabric.sh: pid selection (rank processes only, never the launcher)
# --------------------------------------------------------------------------- #
SCRIPT = os.path.join(PD, "run_pd_pair_fabric.sh")

# "<pid>|<comm>|<cmdline>" as proc_table prints it: prterun -> two python ranks (one
# retitled VLLM::EngineCore by setproctitle, one still on its argv), a stray child
RANK_MOD = "vllm_tt_plugin.kv_transfer.launch.pd_fabric_rank"
PROC_TABLE = "\n".join(
    [
        "4103|VLLM::EngineCore|VLLM::EngineCore ",
        "4105|PDFABRIC::Engin|PDFABRIC::EngineCore ",  # comm is truncated to 15 chars
        f"4102|python|/v/bin/python -m {RANK_MOD} --pd-args /pd/pd_node_args_p3.json ",
        "4101|prterun|prterun --np 1 -x TT_MESH_ID=0 /v/bin/python -m "
        f"{RANK_MOD} --pd-args /pd/na.json : --np 1 ... ",
        "4100|mpirun|/opt/openmpi-v5.0.7-ulfm/bin/mpirun-ulfm --np 1 ... "
        "pd_fabric_rank ... ",
        "4104|sh|/bin/sh -c echo hello ",
    ]
)


def _sh(fn: str, stdin: str, tmp_path) -> list[str]:
    if not os.path.exists(SCRIPT):
        pytest.skip(f"{SCRIPT} missing")
    r = subprocess.run(
        ["bash", "-c", f"source {SCRIPT}; {fn}"],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "PD_FABRIC_LIB_ONLY": "1",
            "PIDDIR": str(tmp_path / "pids"),
            "TAG": "unit",
        },
    )
    assert r.returncode == 0, r.stderr
    return r.stdout.split()


def test_script_pid_selection_targets_ranks_only(tmp_path):
    ranks = _sh("select_rank_pids", PROC_TABLE, tmp_path)
    launchers = _sh("select_launcher_pids", PROC_TABLE, tmp_path)
    # the PDFABRIC-titled rank too; never prterun/mpirun, never the stray
    assert sorted(ranks) == ["4102", "4103", "4105"]
    assert sorted(launchers) == ["4100", "4101"]
    # library mode ran nothing: no pid files were created
    assert not os.listdir(tmp_path / "pids")


def test_script_kill_order_signals_ranks_before_launcher():
    """Structural: kill_ranks TERMs the rank pids, waits, and only then touches the
    launcher/ttrun pids (the MPI launcher SIGKILLs its ranks when TERMed itself)."""
    if not os.path.exists(SCRIPT):
        pytest.skip(f"{SCRIPT} missing")
    with open(SCRIPT) as f:
        body = f.read().split("kill_ranks() {", 1)[1].split("\n}\n", 1)[0]
    term_ranks = body.index("for p in $ranks; do kill -TERM $p")
    wait_ranks = body.index("alive $ranks || break")
    term_launchers = body.index(
        "for p in $launchers $tt; do kill -0 $p 2>/dev/null && kill -TERM"
    )
    assert term_ranks < wait_ranks < term_launchers
    assert "descendants $(cat $pidf)" not in body  # no blanket TERM of the whole tree


def test_launch_modules_do_not_import_ttnn_or_vllm_at_import_time():
    import subprocess
    import sys

    code = (
        "import sys; import vllm_tt_plugin.kv_transfer.launch.pd_launch_config, "
        "vllm_tt_plugin.kv_transfer.launch.pd_fabric_rank; "
        "bad=[m for m in ('ttnn','vllm.engine.arg_utils','vllm.v1.engine.core') "
        "if m in sys.modules]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert r.returncode == 0, r.stdout + r.stderr


# --------------------------------------------------------------------------- #
# per-pair isolation: tag-scoped paths / ids and the engine identity lock
# --------------------------------------------------------------------------- #
def test_pool_below_prefill_bucket_trace_is_refused():
    """P_POOL=1024 (18 KV blocks) under a 2048-token (32-block) bucket trace hung both
    tiny fabric pairs of 2026-09-21 in the decode warm-up (DRAM overrun by the fixed-
    width fill); the launch config refuses pools below chunk_tokens + one pad block."""
    with pytest.raises(ValueError, match="P_POOL=1024 .* would write past the KV pool"):
        lc.PairSettings(ctx=512, p_pool=1024, d_pool=4096)
    with pytest.raises(ValueError, match="D_POOL=2048 "):
        lc.PairSettings(ctx=512, p_pool=4096, d_pool=2048)
    assert lc.PairSettings(ctx=2048, p_pool=2112, d_pool=4096).p_pool == 2112
    assert lc.PairSettings(chunk_tokens=1024, p_pool=1088, d_pool=1088).p_pool == 1088


def test_tag_scopes_every_per_pair_path_and_id():
    """Two pairs on one host must never share a segment dir + engine id (a producer's
    startup_sweep unlinks the other pair's READY segments): every default is derived
    from TAG, explicit knobs still win."""
    a, b = lc.PairSettings(tag="alpha"), lc.PairSettings(tag="beta")
    for attr in ("shm_dir", "ctrl_dir", "p_engine_id", "d_engine_id"):
        assert getattr(a, attr) != getattr(b, attr), attr
        assert "alpha" in getattr(a, attr), attr
    assert a.shm_dir == "/dev/shm/tt_pd_alpha"
    assert a.ctrl_dir == "/dev/shm/tt_pd_fabric_alpha"
    assert a.rendezvous_dir == "/dev/shm/tt_pd_fabric_alpha/.fabric_rendezvous"
    assert (a.p_engine_id, a.d_engine_id) == ("p-alpha", "d-alpha")
    # the producer's swept dir per data plane
    assert a.segment_dir == a.ctrl_dir
    assert lc.PairSettings(tag="alpha", transport="shm").segment_dir == a.shm_dir
    assert a.engine_locks() == [(a.ctrl_dir, "p-alpha"), (a.ctrl_dir, "d-alpha")]
    # the kv_transfer_config carries the tag-scoped values on both nodes
    for role in lc.ROLES:
        extra = lc.kv_transfer_config(a, role)["kv_connector_extra_config"]
        assert extra["fabric_control_dir"] == a.ctrl_dir
        assert lc.kv_transfer_config(a, role)["engine_id"] == a.engine_id(role)
    # explicit overrides win (the shell knobs SHM_DIR / CTRL_DIR / *_ENGINE_ID)
    e = lc.PairSettings.from_env(
        {
            "TAG": "x",
            "SHM_DIR": "/dev/shm/custom",
            "CTRL_DIR": "/dev/shm/ctl",
            "P_ENGINE_ID": "p0",
            "D_ENGINE_ID": "d0",
        }
    )
    assert (e.shm_dir, e.ctrl_dir, e.p_engine_id, e.d_engine_id) == (
        "/dev/shm/custom",
        "/dev/shm/ctl",
        "p0",
        "d0",
    )
    assert (
        lc.render_rank_binding(template(), a, "/na.json", parent_env=base_env())[
            "global_env"
        ]["TT_PD_FABRIC_DIR"]
        == a.ctrl_dir
    )
    meta = lc.node_args(a)["meta"]
    assert meta["engine_ids"] == {"prefill": "p-alpha", "decode": "d-alpha"}
    assert meta["rendezvous_dir"] == a.rendezvous_dir
    assert meta["shm_dir"] == a.shm_dir and meta["segment_dir"] == a.ctrl_dir


@pytest.mark.parametrize(
    "kw", [{"tag": ""}, {"tag": "a.b"}, {"tag": "x" * 40}, {"p_engine_id": "d-p3"}]
)
def test_engine_ids_must_be_segment_dir_components(kw):
    """The derived ids obey the transport's xfer_id grammar (they are directory
    components under the segment dir) and P/D must differ."""
    with pytest.raises(ValueError):
        lc.PairSettings(**kw)
    assert lc.PairSettings(tag="ok_tag-1").p_engine_id == "p-ok_tag-1"


def test_engine_lock_refuses_live_holder_and_replaces_stale(tmp_path, monkeypatch):
    live = {os.getpid()}
    monkeypatch.setattr(lc, "pid_alive", lambda pid: pid in live)
    seg = str(tmp_path / "seg")
    a = lc.EngineLock(seg, "p-a", role="prefill", tag="a").acquire()
    assert os.path.isfile(a.path) and a.path == f"{seg}/.pd_engine-p-a.lock"
    assert lc.read_engine_lock(a.path)["pid"] == os.getpid()
    assert "held by live pid" in lc.engine_in_use(seg, "p-a")
    # a second engine with the same identity refuses while the holder is alive
    with pytest.raises(RuntimeError, match="in use"):
        lc.EngineLock(seg, "p-a", pid=424242).acquire()
    # another engine id under the same dir is independent
    lc.EngineLock(seg, "d-a", pid=424243).acquire()
    # release removes only our file; a stale lock (dead pid) is replaced, not refused
    a.release()
    assert not os.path.exists(a.path) and lc.engine_in_use(seg, "p-a") is None
    stale = lc.EngineLock(seg, "p-a", pid=777777).acquire()
    live.discard(777777)
    assert lc.engine_in_use(seg, "p-a") is None  # dead holder = free
    b = lc.EngineLock(seg, "p-a", pid=os.getpid()).acquire()
    assert lc.read_engine_lock(b.path)["pid"] == os.getpid()
    stale.release()  # not ours any more: must not remove b's file
    assert lc.read_engine_lock(b.path)["pid"] == os.getpid()
    b.release()
    assert not os.path.exists(b.path)


def test_engine_in_use_detects_lockless_live_producer_and_claims(tmp_path, monkeypatch):
    """run_pd_pair.sh's engines take no lock: a live producer shows through the
    producer_pid of its segment headers, a live consumer through its claim's
    consumer.pid.  Dead pids mean the dir is free (only stale files)."""
    from vllm_tt_plugin.kv_transfer.transport.base import build_manifest
    from vllm_tt_plugin.kv_transfer.transport.shm import (
        CONSUMER_PID_FILE,
        segment_layout,
    )

    live: set[int] = set()
    monkeypatch.setattr(lc, "pid_alive", lambda pid: pid in live)
    seg = str(tmp_path / "shm")
    edir = os.path.join(seg, "p0")
    hx = f"{1:032x}"
    os.makedirs(os.path.join(edir, hx))
    m = build_manifest(100, model_sig="s", prompt_hash="p")
    _, hdr = segment_layout(m, "dumpfile")
    hdr.producer_pid = 31337
    with open(os.path.join(edir, hx, "header"), "wb") as f:
        f.write(hdr.pack_fixed())
    assert lc.engine_in_use(seg, "p0") is None  # pid 31337 dead: leftovers only
    live.add(31337)
    assert "live producer pid 31337" in lc.engine_in_use(seg, "p0")
    live.clear()
    claim = os.path.join(edir, f"{2:032x}.claimed-d0")
    os.makedirs(claim)
    with open(os.path.join(claim, CONSUMER_PID_FILE), "w") as f:
        f.write("4242")
    assert lc.engine_in_use(seg, "p0") is None
    live.add(4242)
    assert "live consumer pid 4242" in lc.engine_in_use(seg, "p0")
    # the lock refuses on the same evidence, and an absent dir is simply free
    with pytest.raises(RuntimeError, match="live consumer"):
        lc.EngineLock(seg, "p0", pid=1).acquire()
    assert lc.engine_in_use(str(tmp_path / "nowhere"), "p0") is None
    s = lc.PairSettings(tag="t", transport="shm", shm_dir=seg, p_engine_id="p0")
    assert len(lc.pair_in_use(s)) == 1 and "p0" in lc.pair_in_use(s)[0]


def test_cli_check_in_use(tmp_path, monkeypatch, capsys):
    env = {"TAG": "cli", "CTRL_DIR": str(tmp_path / "ctl"), "TRANSPORT": "fabric"}
    monkeypatch.setattr(os, "environ", dict(os.environ, **env))
    assert lc._main(["--check-in-use"]) == 0
    assert "free" in capsys.readouterr().out
    lock = lc.EngineLock(str(tmp_path / "ctl"), "p-cli").acquire()  # our live pid
    assert lc._main(["--check-in-use"]) == 3
    out = capsys.readouterr().out
    assert "REFUSING" in out and "held by live pid" in out
    lock.release()
    assert lc._main(["--check-in-use"]) == 0


# --------------------------------------------------------------------------- #
# TRANSPORT=shm without fabric (default) vs SHM_FABRIC=1
# --------------------------------------------------------------------------- #
def test_validate_argv_fabric_requires_fabric_config_shm_does_not():
    fab = lc.PairSettings()
    p = lc.serve_argv(fab, "prefill")
    ac = lc.argv_json(p, "--additional-config")
    del ac["tt"]["fabric_config"]
    with pytest.raises(ValueError, match="required for the fabric data plane"):
        lc.validate_argv_for_role(
            _swap(p, "--additional-config", json.dumps(ac)), "prefill", 29551
        )
    shm = lc.PairSettings(transport="shm")
    ps = lc.serve_argv(shm, "prefill")
    assert "fabric_config" not in lc.argv_json(ps, "--additional-config")["tt"]
    summ = lc.validate_argv_for_role(ps, "prefill", 29551)
    assert summ["fabric_config"] is None and summ["fabric_enabled"] is False
    # the shm_mode message no longer claims transport=fabric
    ktc = lc.argv_json(ps, "--kv-transfer-config")
    ktc["kv_connector_extra_config"]["shm_mode"] = "raw"
    with pytest.raises(ValueError, match="warm-up contract") as ei:
        lc.validate_argv_for_role(
            _swap(ps, "--kv-transfer-config", json.dumps(ktc)), "prefill", 29551
        )
    assert "with transport=fabric" not in str(ei.value)
    # a transport without a segment dir is refused (the lock and the sweep need it)
    ktc = lc.argv_json(ps, "--kv-transfer-config")
    del ktc["kv_connector_extra_config"]["shm_dir"]
    with pytest.raises(ValueError, match="segment dir"):
        lc.validate_argv_for_role(
            _swap(ps, "--kv-transfer-config", json.dumps(ktc)), "prefill", 29551
        )
    assert lc.segment_dir_of({"transport": "fabric", "shm_dir": "/c"}) == "/c"
    assert lc.segment_dir_of({"transport": "shm"}) is None


def test_prepare_disables_fabric_wrapper_for_plain_shm(tmp_path):
    for shm_fabric, wrapper_on in ((0, False), (1, True)):
        s = lc.PairSettings(pair="test", transport="shm", shm_fabric=shm_fabric)
        path = str(tmp_path / f"na{shm_fabric}.json")
        lc.write_node_args(path, s)
        args = rank_mod.parse_args(
            ["--pd-args", path, "--dry-run", "--role", "prefill", "--rpc-port", "29551"]
        )
        ctx, argv, summary = rank_mod.prepare(args, env={})
        assert summary["transport"] == "shm" and summary["tag"] == "p3"
        assert summary["segment_dir"] == s.shm_dir
        assert args.no_fabric_wrapper is (not wrapper_on)
        assert summary["fabric_enabled"] is wrapper_on


def _fake_vllm_config(ktc: dict, rpc_port: int = 29551):
    extra = ktc["kv_connector_extra_config"]
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_role=ktc["kv_role"],
            engine_id=ktc["engine_id"],
            kv_connector_extra_config=extra,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_rpc_port=rpc_port,
            data_parallel_size=1,
            data_parallel_size_local=1,
            data_parallel_rank=0,
            data_parallel_rank_local=0,
            distributed_executor_backend="uni",
        ),
        model_config=SimpleNamespace(model="/w", max_model_len=65536),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        cache_config=SimpleNamespace(block_size=64),
    )


def test_check_config_for_role_accepts_fabric_and_shm_rejects_tcp(monkeypatch):
    import vllm_tt_plugin.config as tt_cfg_mod

    monkeypatch.setattr(tt_cfg_mod, "get_tt_config", lambda cfg: {"l1_small_size": 1})
    monkeypatch.setattr(rank_mod, "handshake_address", lambda cfg: "tcp://h:1")
    monkeypatch.setenv("TT_PD_FABRIC_SKIP_WARMUP", "1")
    ctx = rank_mod.resolve_rank_env(mpi_env(0, "prefill"))
    for transport, seg_attr in (("fabric", "ctrl_dir"), ("shm", "shm_dir")):
        s = lc.PairSettings(transport=transport, tag="ccr")
        ktc = lc.kv_transfer_config(s, "prefill")
        info = rank_mod.check_config_for_role(_fake_vllm_config(ktc), ctx)
        assert info["transport"] == transport
        assert info["segment_dir"] == getattr(s, seg_attr) == info["control_dir"]
        assert info["engine_id"] == "p-ccr" and info["kv_role"] == "kv_producer"
        # a leaked debug knob is visible in the boot summary
        assert info["env"]["TT_PD_FABRIC_SKIP_WARMUP"] == "1"
    ktc = lc.kv_transfer_config(lc.PairSettings(), "prefill")
    ktc["kv_connector_extra_config"]["transport"] = "tcp"
    with pytest.raises(RuntimeError, match="neither 'fabric'.*nor 'shm'"):
        rank_mod.check_config_for_role(_fake_vllm_config(ktc), ctx)
    ktc = lc.kv_transfer_config(lc.PairSettings(transport="shm"), "prefill")
    del ktc["kv_connector_extra_config"]["shm_dir"]
    with pytest.raises(RuntimeError, match="segment dir"):
        rank_mod.check_config_for_role(_fake_vllm_config(ktc), ctx)
    ktc = lc.kv_transfer_config(lc.PairSettings(), "decode")
    with pytest.raises(RuntimeError, match="kv_role"):
        rank_mod.check_config_for_role(_fake_vllm_config(ktc), ctx)
    with pytest.raises(RuntimeError, match="PD_RPC_PORT"):
        rank_mod.check_config_for_role(
            _fake_vllm_config(lc.kv_transfer_config(lc.PairSettings(), "prefill"), 1),
            ctx,
        )


# --------------------------------------------------------------------------- #
# shutdown path: transport -> mesh close -> dist cleanup -> engine exit -> lock
# --------------------------------------------------------------------------- #
class _FakeTransport:
    def __init__(self, log: list, name: str = "T"):
        self.log, self.name, self.shut = log, name, False

    def shutdown(self):
        self.shut = True
        self.log.append(f"transport.shutdown:{self.name}")


def test_install_mesh_hooks_close_shuts_down_transports_before_mesh_close(
    monkeypatch,
):
    log: list = []
    fake_worker = SimpleNamespace(
        open_mesh_device=lambda tt_config, trace_mode, local_dp_rank=0: object(),
        close_mesh_device=lambda mesh, tt_config: log.append(("close", mesh)),
    )
    still_up = [_FakeTransport(log)]

    def shutdown_transports():
        done = list(still_up)
        for t in done:
            t.shutdown()
        still_up.clear()
        return done

    monkeypatch.setattr(rank_mod, "_FABRIC_SET", False)
    rank_mod.install_mesh_hooks(
        set_fabric=lambda *a: None,
        register_mesh=lambda m: None,
        distributed_info=lambda: {"initialized": True, "rank": 0, "size": 2},
        shutdown_transports=shutdown_transports,
        worker_module=fake_worker,
    )
    assert fake_worker.close_mesh_device._pd_fabric_wrapped
    mesh = object()
    fake_worker.close_mesh_device(mesh, {"x": 1})
    assert log == ["transport.shutdown:T", ("close", mesh)]
    # nothing registered: the mesh closes right away
    fake_worker.close_mesh_device(mesh, {})
    assert log[-1] == ("close", mesh) and len(log) == 3
    # a worker module without close_mesh_device is a moved teardown: fail loudly
    with pytest.raises(RuntimeError, match="close_mesh_device"):
        rank_mod.install_mesh_hooks(
            set_fabric=lambda *a: None,
            worker_module=SimpleNamespace(open_mesh_device=lambda *a: None),
        )


def test_install_pump_hook_calls_transport_pump_after_end_step(monkeypatch):
    """I4 shim: off by default; on, end_step runs first and pump() follows even when
    end_step raises; a transport without pump (shm) is a no-op; idempotent."""
    log: list = []

    class Transport:
        def pump(self):
            log.append("pump")

    class Worker:
        def __init__(self, transport):
            self.transport = transport

        def end_step(self, finished_req_ids=None):
            log.append(("end_step", finished_req_ids))
            if finished_req_ids == "boom":
                raise RuntimeError("step failed")

    monkeypatch.delenv("TT_PD_FABRIC_PUMP", raising=False)
    assert rank_mod.install_pump_hook(worker_cls=Worker) is None  # default off
    Worker(Transport()).end_step(finished_req_ids={"a"})
    assert log == [("end_step", {"a"})]
    log.clear()
    monkeypatch.setenv("TT_PD_FABRIC_PUMP", "1")
    wrapped = rank_mod.install_pump_hook(worker_cls=Worker)
    assert wrapped is not None and wrapped._pd_fabric_wrapped
    assert rank_mod.install_pump_hook(worker_cls=Worker) is wrapped  # idempotent
    Worker(Transport()).end_step(finished_req_ids={"b"})
    assert log == [("end_step", {"b"}), "pump"]
    log.clear()
    with pytest.raises(RuntimeError, match="step failed"):
        Worker(Transport()).end_step("boom")
    assert log == [("end_step", "boom"), "pump"]  # pump still runs
    log.clear()
    Worker(object()).end_step()  # shm transport: no pump attribute -> no-op
    assert log == [("end_step", None)]

    class BadPump:
        def pump(self):
            raise ValueError("pump broke")

    Worker(BadPump()).end_step()  # a raising pump never takes the step down
    with pytest.raises(RuntimeError, match="end_step"):
        rank_mod.install_pump_hook(worker_cls=object, enabled=True)


def test_default_shutdown_transports_uses_fabric_socket_registry(
    clean_mesh_registry,
):
    log: list = []
    t1, t2 = _FakeTransport(log, "a"), _FakeTransport(log, "b")
    fabric_socket.register_transport(t1)
    fabric_socket.register_transport(t2)
    fabric_socket.register_transport(t1)  # idempotent
    fabric_socket.register_mesh_device(object())
    assert fabric_socket.registered_transports() == [t1, t2]
    done = rank_mod._default_shutdown_transports()
    assert done == [t1, t2] and t1.shut and t2.shut
    assert fabric_socket.registered_transports() == []
    assert fabric_socket.registered_mesh_device() is None
    assert rank_mod._default_shutdown_transports() == []


def _fake_torch(available: bool, log: list):
    acc = SimpleNamespace(
        is_available=lambda: available,
        empty_cache=lambda: log.append("empty_cache") or _raise_no_accel(available),
    )
    return SimpleNamespace(accelerator=acc)


def _raise_no_accel(available: bool):
    if not available:  # torch on a TT host: no accelerator backend
        raise RuntimeError("Cannot access accelerator device when none is available.")


def test_install_dist_cleanup_patch_skips_accelerator_cache_without_accelerator():
    log: list = []
    torch_mod = _fake_torch(False, log)

    def orig_cleanup(shutdown_ray=False):
        log.append(("cleanup", shutdown_ray))
        torch_mod.accelerator.empty_cache()  # what vLLM does last

    core = SimpleNamespace(cleanup_dist_env_and_memory=orig_cleanup)
    wrapped = rank_mod.install_dist_cleanup_patch(core, torch_mod)
    assert core.cleanup_dist_env_and_memory is wrapped and wrapped._pd_fabric_wrapped
    # idempotent
    assert rank_mod.install_dist_cleanup_patch(core, torch_mod) is wrapped
    core.cleanup_dist_env_and_memory()  # would raise RuntimeError unpatched
    assert log == [("cleanup", False)]  # the original ran, empty_cache was skipped
    assert torch_mod.accelerator.empty_cache is not None
    # the real empty_cache is restored afterwards (still raises without accelerator)
    with pytest.raises(RuntimeError, match="accelerator"):
        torch_mod.accelerator.empty_cache()
    # with a real accelerator the stock path runs untouched
    log2: list = []
    torch2 = _fake_torch(True, log2)
    core2 = SimpleNamespace(
        cleanup_dist_env_and_memory=lambda: torch2.accelerator.empty_cache()
    )
    rank_mod.install_dist_cleanup_patch(core2, torch2)
    core2.cleanup_dist_env_and_memory()
    assert log2 == ["empty_cache"]
    # a vLLM whose shutdown path moved is an error, not a silent skip
    with pytest.raises(RuntimeError, match="cleanup_dist_env_and_memory"):
        rank_mod.install_dist_cleanup_patch(SimpleNamespace(), torch_mod)


def test_install_idle_pump_hook_pumps_while_idle_and_handles_requests():
    """Idle half of the I4 shim: off by default; on, the blocking input-queue wait
    becomes a timeout poll that pumps every registered transport on each timeout,
    keeps the idle callbacks / aborts drain / request handling of the original, and
    falls through to the original when the engine is in the non-blocking shape."""
    import queue

    log: list = []

    class Proc:
        process_input_queue_block = True

        def __init__(self, work_after: int):
            self.input_queue = queue.Queue()
            self.aborts_queue = queue.Queue()
            self.aborts_queue.put("abort-1")
            self._work_after, self._polls = work_after, 0

        def has_work(self):
            return self._polls >= self._work_after

        def is_running(self):
            return True

        def _notify_idle_state_callbacks(self):
            self._polls += 1
            log.append("idle")

        def _handle_client_request(self, *req):
            log.append(("req", req))

        def _process_input_queue(self):
            log.append("orig")

    pumps: list = []
    assert rank_mod.install_idle_pump_hook(proc_cls=Proc) is None  # default off
    wrapped = rank_mod.install_idle_pump_hook(
        proc_cls=Proc, enabled=True, period_s=0.01, pump_all=lambda: pumps.append(1)
    )
    assert Proc._process_input_queue is wrapped and wrapped._pd_fabric_wrapped
    assert (
        rank_mod.install_idle_pump_hook(proc_cls=Proc, enabled=True) is wrapped
    )  # idempotent
    p = Proc(work_after=3)
    p._process_input_queue()
    assert len(pumps) >= 2 and log.count("idle") == 3  # pumped on every timeout
    assert p.aborts_queue.empty()  # aborts drained while idle (as the original)
    # a request arriving while idle is handled; the non-blocking tail drains the rest
    log.clear()
    pumps.clear()
    q = Proc(work_after=10)
    q.input_queue.put(("add", "r1"))
    q.input_queue.put(("add", "r2"))
    q.has_work = lambda: q._polls >= 2  # work appears after the first request
    q._process_input_queue()
    assert ("req", ("add", "r1")) in log and ("req", ("add", "r2")) in log
    # non-blocking shape: the original runs untouched
    log.clear()
    nb = Proc(work_after=0)
    nb.process_input_queue_block = False
    nb._process_input_queue()
    assert log == ["orig"]
    # a raising pump never takes the loop down
    log.clear()

    def boom():
        raise ValueError("pump broke")

    class Proc2(Proc):
        pass

    Proc2._process_input_queue = Proc._process_input_queue._pd_fabric_orig
    rank_mod.install_idle_pump_hook(
        proc_cls=Proc2, enabled=True, period_s=0.01, pump_all=boom
    )
    Proc2(work_after=2)._process_input_queue()
    assert log.count("idle") == 2
    # a vLLM whose idle loop moved is an error, not a silent skip
    with pytest.raises(RuntimeError, match="_process_input_queue"):
        rank_mod.install_idle_pump_hook(proc_cls=object, enabled=True)


def test_run_headless_engine_exit_codes_and_lock_release(tmp_path):
    seg = str(tmp_path / "seg")

    def run_with(outcome):
        lock = lc.EngineLock(seg, "p-x", role="prefill").acquire()

        def run():
            if isinstance(outcome, BaseException):
                raise outcome

        code = rank_mod.run_headless_engine(run, lock=lock, role="prefill", rank=0)
        assert not lock.held and not os.path.exists(lock.path)
        return code

    assert run_with(None) == 0
    assert run_with(SystemExit()) == 0  # vLLM's busy loop: `raise SystemExit`
    assert run_with(SystemExit(0)) == 0
    assert run_with(SystemExit(3)) == 3
    assert run_with(SystemExit("boom")) == 1
    assert run_with(RuntimeError("Cannot access accelerator device")) == 1
    assert run_with(KeyboardInterrupt()) == 1
    assert rank_mod.run_headless_engine(lambda: None) == 0  # no lock


def test_shutdown_sequence_transport_mesh_cleanup_exit_lock(
    tmp_path, monkeypatch, clean_mesh_registry
):
    """The whole rank teardown with fakes, in vLLM's real shape: SIGTERM -> busy loop
    raises SystemExit -> finally EngineCore.shutdown() = worker.shutdown (transport
    down, mesh close through our wrapper) then cleanup_dist_env_and_memory (patched)
    -> SystemExit propagates -> exit code 0, engine lock released last.  Also the
    path where the connector never shut the transport down: the mesh-close wrapper
    does it first, so the socket never outlives the mesh."""
    log: list = []
    torch_mod = _fake_torch(False, log)
    fake_worker = SimpleNamespace(
        open_mesh_device=lambda tt_config, trace_mode, local_dp_rank=0: object(),
        close_mesh_device=lambda mesh, tt_config: log.append("close_mesh"),
    )
    monkeypatch.setattr(rank_mod, "_FABRIC_SET", False)
    rank_mod.install_mesh_hooks(
        set_fabric=lambda *a: None,
        register_mesh=fabric_socket.register_mesh_device,
        distributed_info=lambda: {"initialized": True, "rank": 0, "size": 2},
        worker_module=fake_worker,
    )

    def orig_cleanup():
        log.append("cleanup_dist_env")
        torch_mod.accelerator.empty_cache()

    core = SimpleNamespace(cleanup_dist_env_and_memory=orig_cleanup)
    rank_mod.install_dist_cleanup_patch(core, torch_mod)

    class Lock(lc.EngineLock):
        def release(self):
            log.append("lock.release")
            super().release()

    def run_case(connector_shuts_transport: bool):
        del log[:]
        seg = str(tmp_path / "seg")
        lock = Lock(seg, "p-x", role="prefill").acquire()
        mesh = fake_worker.open_mesh_device({}, "all")
        transport = _FakeTransport(log)
        fabric_socket.register_transport(transport)

        def engine_core_shutdown():  # EngineCore.shutdown -> TTWorker.shutdown
            if connector_shuts_transport:  # ensure_kv_transfer_shutdown
                transport.shutdown()
                fabric_socket.unregister_transport(transport)
            fake_worker.close_mesh_device(mesh, {})
            core.cleanup_dist_env_and_memory()

        def run_engine_core():  # EngineCoreProc.run_engine_core after SIGTERM
            try:
                log.append("busy_loop")
                raise SystemExit
            finally:
                engine_core_shutdown()

        code = rank_mod.run_headless_engine(
            run_engine_core, lock=lock, role="prefill", rank=0
        )
        log.append(f"exit:{code}")
        assert not os.path.exists(lock.path)
        assert fabric_socket.registered_transports() == []
        assert fabric_socket.registered_mesh_device() is None
        return list(log)

    assert run_case(True) == [
        "busy_loop",
        "transport.shutdown:T",
        "close_mesh",
        "cleanup_dist_env",
        "lock.release",
        "exit:0",
    ]
    assert run_case(False) == [
        "busy_loop",
        "transport.shutdown:T",  # the close wrapper shut it down before the mesh
        "close_mesh",
        "cleanup_dist_env",
        "lock.release",
        "exit:0",
    ]


# --------------------------------------------------------------------------- #
# run_pd_pair_fabric.sh: tag scoping, in-use refusal, hang diagnostics
# --------------------------------------------------------------------------- #
def test_script_tag_scopes_dirs_and_ids(tmp_path):
    out = _sh('echo "$SHM_DIR $CTRL_DIR $P_ENGINE_ID $D_ENGINE_ID"', "", tmp_path)
    assert out == [
        "/dev/shm/tt_pd_unit",
        "/dev/shm/tt_pd_fabric_unit",
        "p-unit",
        "d-unit",
    ]
    s = lc.PairSettings(tag="unit")
    assert out == [s.shm_dir, s.ctrl_dir, s.p_engine_id, s.d_engine_id]
    # the functions the `up` / `down` flows rely on exist in library mode
    out = _sh(
        "type check_in_use hang_diagnostics pyspy_dump sweep_stale_rendezvous "
        ">/dev/null && echo ok",
        "",
        tmp_path,
    )
    assert out == ["ok"]


def test_script_refuses_in_use_and_dumps_stacks_before_kill():
    """Structural: `up` checks the engine identities after render and BEFORE any
    front-end starts; a health timeout writes hang diagnostics before `exit 1`;
    kill_ranks py-spy dumps the ranks still alive after the graceful window BEFORE
    the SIGKILL loop."""
    if not os.path.exists(SCRIPT):
        pytest.skip(f"{SCRIPT} missing")
    with open(SCRIPT) as f:
        body = f.read()
    up = body.split("\n  up)\n", 1)[1].split("\n    ;;\n", 1)[0]
    assert up.index("render || exit 1") < up.index("check_in_use || exit 1")
    assert up.index("check_in_use || exit 1") < up.index("launch_frontend prefill")
    assert "wait_health P $P_PORT $BOOT_TIMEOUT || { hang_diagnostics" in up
    assert "wait_health D $D_PORT $BOOT_TIMEOUT || { hang_diagnostics" in up
    kill = body.split("kill_ranks() {", 1)[1].split("\n}\n", 1)[0]
    term = kill.index("for p in $ranks; do kill -TERM $p")
    dump = kill.index("hang_diagnostics")
    sigkill = kill.index("kill -9 $p")
    assert term < dump < sigkill
    diag = body.split("hang_diagnostics() {", 1)[1].split("\n}\n", 1)[0]
    assert "pyspy_dump" in diag and "_summary.txt" in diag
    assert "sudo -n $PYSPY dump" in body  # ptrace_scope=1 fallback
    down = body.split("\n  down)\n", 1)[1].split("\n    ;;\n", 1)[0]
    assert down.index("kill_ranks") < down.index("sweep_stale_rendezvous")
    assert "SHM_DIR=${SHM_DIR:-/dev/shm/tt_pd_$TAG}" in body
    assert "CTRL_DIR=${CTRL_DIR:-/dev/shm/tt_pd_fabric_$TAG}" in body
